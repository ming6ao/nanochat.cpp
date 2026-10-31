// tests/sft_loop_parity_test.cc -- the R3 schedule gate of
// docs/sft-parity-plan.md sections 3 and 7 (phase E).
//
// The loop fixture records what the reference loop *computed* for one
// eight-step run: the 0-based counter, the rate progress, the epoch, the
// learning-rate multiplier, the Muon momentum, and the logged loss of every
// step, plus the final evaluation. This test replays that trace through the C++
// schedule and compares every number:
//
//   * `Scheduler::LrMultiplierForProgress(progress)` against
//     `loop/<k>/lr_multiplier` -- design decision D1, the progress schedule;
//   * `Scheduler::MuonMomentum(step)` against `loop/<k>/momentum` -- design
//     decision D2, the 1-based step and the 0-based reference counter;
//   * the epoch counter against the row-cursor wrap rule of section 5.1;
//   * the loss curve's shape, and the final `val/bpb/8` against the same trace:
//     the token-byte table times the shard's valid targets must reproduce the
//     recorded bits-per-byte at the recorded loss scale (design decision D6);
//   * `chatcore/8` as the sentinel it is, because the trace does not measure
//     ChatCORE.
//
// The schedule settings themselves (the warmup and warmdown ratios, the final
// fraction, and the four Muon momentum fields) are the pinned section 5.2 ones,
// which the step fixture records. Both fixtures come from the one pinned
// settings module (`tools/sft_fixture.py`), so reading the settings from the
// step fixture is what keeps the two files from drifting, not a hand copy.
//
// T0, CPU, no torch, no network: `tools/nanochat test
// //tests:sft_loop_parity_test`.

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nanochat/scheduler.h"
#include "src/sft_shard.h"
#include "tests/oracle_fixture.h"

namespace {

using nanochat::Scheduler;
using nanochat::SchedulerConfig;
using nanochat::oracle::Fixture;
using nanochat::oracle::Tensor;

// The schedule values are exact settings, so the only difference a correct file
// can carry is the fp32 round trip. The loss and the bits-per-byte are
// compared through their own relations below.
constexpr double kScheduleAtol = 1e-6;
constexpr double kScheduleRtol = 1e-6;
constexpr double kProgressAtol = 1e-6;
// The evaluation scores every row of the shard while `train/loss/<k>` logs the
// last micro-batch of its step, so the two agree to a fraction of a percent and
// never exactly. The band is wide enough for that difference and far too narrow
// for a different token-byte table or a different metric.
constexpr double kEvalLossRtol = 0.05;

int g_failures = 0;
double g_max_schedule_error = 0.0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

std::string Format(const char* fmt, ...) {
  char buffer[512];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  return std::string(buffer);
}

std::string LocateFile(int argc, char** argv, int index, const char* relative) {
  std::vector<std::string> candidates;
  if (argc > index && argv[index] != nullptr && argv[index][0] != '\0') {
    candidates.emplace_back(argv[index]);
  }
  const auto env = [](const char* name) -> std::string {
    const char* value = std::getenv(name);
    return value != nullptr ? std::string(value) : std::string();
  };
  const std::string src_dir = env("TEST_SRCDIR");
  if (!src_dir.empty()) {
    const std::string workspace = env("TEST_WORKSPACE");
    if (!workspace.empty()) {
      candidates.push_back(src_dir + "/" + workspace + "/" + relative);
    }
    candidates.push_back(src_dir + "/_main/" + relative);
  }
  candidates.emplace_back(relative);
  candidates.emplace_back(std::string("../") + relative);
  for (const std::string& candidate : candidates) {
    std::ifstream in(candidate, std::ios::binary);
    if (in.good()) return candidate;
  }
  throw std::runtime_error(std::string("cannot find ") + relative +
                           "; pass it as argument " + std::to_string(index));
}

double ScalarF32(const Fixture& fixture, const std::string& name) {
  if (!fixture.Has(name)) {
    throw std::runtime_error("the fixture is missing record '" + name + "'");
  }
  return fixture.Get(name).scalar_f32();
}

std::int64_t ScalarInt(const Fixture& fixture, const std::string& name) {
  if (!fixture.Has(name)) {
    throw std::runtime_error("the fixture is missing record '" + name + "'");
  }
  return fixture.Get(name).scalar_int();
}

void CheckSchedule(const std::string& what, double got, double want) {
  const double diff = std::fabs(got - want);
  g_max_schedule_error = std::max(g_max_schedule_error, diff);
  if (diff > kScheduleAtol + kScheduleRtol * std::fabs(want)) {
    Fail(Format("%s: got %.9g want %.9g (|diff|=%.3g)", what.c_str(), got, want,
                diff));
  }
}

// The pinned section 5.2 schedule, read from the step fixture.
SchedulerConfig SchedulerFromStepFixture(const Fixture& step_fixture,
                                         int num_iterations) {
  SchedulerConfig sched;
  sched.num_iterations = num_iterations;
  // An SFT run follows the progress curve, so the step-count warmup is off
  // (design decision D1); `src/sft_main.cc` sets the same value.
  sched.warmup_steps = 0;
  sched.warmup_ratio =
      step_fixture.Get("config/schedule/warmup_ratio").scalar_f32();
  sched.warmdown_ratio =
      step_fixture.Get("config/schedule/warmdown_ratio").scalar_f32();
  sched.final_lr_frac =
      step_fixture.Get("config/schedule/final_lr_frac").scalar_f32();
  sched.muon_momentum_warmup_steps = static_cast<float>(
      step_fixture.Get("config/schedule/muon_momentum_steps").scalar_int());
  sched.muon_momentum_start =
      step_fixture.Get("config/schedule/muon_momentum_start").scalar_f32();
  sched.muon_momentum_peak =
      step_fixture.Get("config/schedule/muon_momentum_peak").scalar_f32();
  sched.muon_momentum_final =
      step_fixture.Get("config/schedule/muon_momentum_final").scalar_f32();
  return sched;
}

void Run(const std::string& loop_path, const std::string& step_path,
         const std::string& shard_path) {
  const Fixture loop = Fixture::Load(loop_path);
  const Fixture step = Fixture::Load(step_path);
  if (ScalarInt(loop, "config/version") != 1 ||
      ScalarInt(step, "config/version") != 1) {
    throw std::runtime_error("unsupported SFT fixture version");
  }

  const int steps = static_cast<int>(ScalarInt(loop, "config/num_iterations"));
  const int rows = static_cast<int>(ScalarInt(loop, "config/rows"));
  const int batch =
      static_cast<int>(ScalarInt(loop, "config/device_batch_size"));
  const int accumulate = static_cast<int>(ScalarInt(loop, "config/grad_accum"));
  const int seq = static_cast<int>(ScalarInt(loop, "config/seq_len"));
  if (steps <= 0 || rows <= 0 || batch <= 0 || accumulate <= 0 || seq <= 0) {
    throw std::runtime_error("the fixture's run shape is not positive");
  }

  // The evaluation cadence of section 5.4. The trace evaluates once, at the
  // final step, which is the evaluation the reference always runs.
  if (ScalarInt(loop, "config/eval_every") != 200 ||
      ScalarInt(loop, "config/eval_tokens") != 40 * 524288 ||
      ScalarInt(loop, "config/chatcore_every") != 200 ||
      ScalarInt(loop, "config/chatcore_max_cat") != -1 ||
      ScalarInt(loop, "config/chatcore_max_sample") != 24) {
    Fail("the recorded evaluation cadence is not the pinned section 5.4 one");
  }
  if (ScalarInt(loop, "config/eval_at") != steps) {
    Fail("the recorded evaluation is not at the final step");
  }

  // The shard the loop consumed, and the widths the fixture claims for it.
  nanochat::SftShard shard;
  std::string shard_error;
  if (!nanochat::LoadSftShard(shard_path, &shard, &shard_error)) {
    throw std::runtime_error("cannot read the packed shard: " + shard_error);
  }
  if (shard.rows != rows) {
    Fail(Format("config/rows %d, the shard holds %d rows", rows, shard.rows));
  }
  if (shard.seq != seq) {
    Fail(Format("config/seq_len %d, the shard holds %d", seq, shard.seq));
  }

  const Scheduler scheduler(SchedulerFromStepFixture(step, steps));

  float previous_progress = -1.0f;
  int previous_epoch = 0;
  double previous_loss = 0.0;
  // The rows one step consumes, which is what advances the epoch counter of
  // section 5.1's stream.
  const int rows_per_step = accumulate * batch;
  // How many steps the progress curve and the step-count curve disagree on.
  // The count is the evidence for design decision D1: an SFT run follows the
  // progress, and a step total fixed before the run started describes a
  // different curve.
  int schedule_disagreements = 0;

  for (int k = 1; k <= steps; ++k) {
    const std::string prefix = "loop/" + std::to_string(k);
    // The reference loop counter is 0-based; the C++ counter is 1-based (D2).
    const int it = static_cast<int>(ScalarInt(loop, prefix + "/step"));
    if (it != k - 1) {
      Fail(Format("%s/step: got %d want %d", prefix.c_str(), it, k - 1));
    }

    // The progress stream: the reference advances progress once per step and
    // keeps the running maximum, so the recorded value is the step fraction of
    // the pinned iteration count.
    const float progress =
        static_cast<float>(ScalarF32(loop, prefix + "/progress"));
    const float expected_progress =
        static_cast<float>(k) / static_cast<float>(steps);
    if (std::fabs(progress - expected_progress) > kProgressAtol) {
      Fail(Format("%s/progress: got %.9g want %.9g", prefix.c_str(), progress,
                  expected_progress));
    }
    if (progress <= previous_progress) {
      Fail(prefix + "/progress is not increasing");
    }
    previous_progress = progress;

    // The epoch counter: every `rows` consumed rows wrap back to the start.
    const int consumed = k * rows_per_step;
    const int expected_epoch = (consumed - 1) / rows + 1;
    const int epoch = static_cast<int>(ScalarInt(loop, prefix + "/epoch"));
    if (epoch != expected_epoch) {
      Fail(Format("%s/epoch: got %d want %d", prefix.c_str(), epoch,
                  expected_epoch));
    }
    if (epoch < previous_epoch) Fail(prefix + "/epoch went backwards");
    previous_epoch = epoch;

    // The learning-rate multiplier, from the recorded progress (D1).
    const float multiplier = scheduler.LrMultiplierForProgress(progress);
    CheckSchedule(prefix + "/lr_multiplier", multiplier,
                  ScalarF32(loop, prefix + "/lr_multiplier"));
    // The pretraining curve, at the same step, for the D1 comparison above.
    if (std::fabs(scheduler.LrMultiplier(k) - multiplier) >
        kScheduleAtol + kScheduleRtol * std::fabs(multiplier)) {
      ++schedule_disagreements;
    }

    // The Muon momentum, from the 1-based step (D2). The reference evaluates it
    // at the recorded 0-based `it`, which is `step - 1`.
    CheckSchedule(prefix + "/momentum", scheduler.MuonMomentum(it + 1),
                  ScalarF32(loop, prefix + "/momentum"));

    // The logged loss of the step: the last micro-batch loss of the step, which
    // is what the reference logs and what `src/sft_main.cc` prints.
    const double loss = ScalarF32(loop, "train/loss/" + std::to_string(k));
    if (!std::isfinite(loss) || loss <= 0.0) {
      Fail(Format("train/loss/%d is not a positive finite loss", k));
    } else if (k > 1 && loss >= previous_loss) {
      Fail(Format("train/loss/%d %.9g does not improve on %.9g", k, loss,
                  previous_loss));
    }
    previous_loss = loss;
  }

  if (schedule_disagreements == 0) {
    Fail(
        "the progress curve and the step-count curve agree on every step, so "
        "the fixture does not pin design decision D1");
  }

  // The final evaluation (design decision D6): `val/bpb/8` is bits per byte
  // over the packed rows, so the token-byte table times the shard's valid
  // targets must reproduce it at the recorded loss scale.
  const double bpb = ScalarF32(loop, "val/bpb/8");
  if (!std::isfinite(bpb) || bpb <= 0.0 || bpb >= 8.0) {
    Fail(Format("val/bpb/8 %.9g is not a plausible bits-per-byte", bpb));
  } else {
    const Tensor& table = loop.Get("config/token_bytes");
    if (table.numel() < 256) {
      Fail("config/token_bytes: fewer entries than the vocabulary");
    } else {
      const std::int32_t* token_bytes = table.i32();
      std::int64_t valid_targets = 0;
      std::int64_t target_bytes = 0;
      for (std::size_t i = 0; i < shard.targets.size(); ++i) {
        const int target = shard.targets[i];
        if (target < 0) continue;
        if (target >= table.numel()) {
          Fail(Format("the shard target %d is outside the token-byte table",
                      target));
          break;
        }
        ++valid_targets;
        target_bytes += token_bytes[target];
      }
      if (valid_targets > 0 && target_bytes > 0) {
        const double total_nats =
            bpb * std::log(2.0) * static_cast<double>(target_bytes);
        const double mean_nats =
            total_nats / static_cast<double>(valid_targets);
        const double train_loss =
            ScalarF32(loop, "train/loss/" + std::to_string(steps));
        if (std::fabs(mean_nats - train_loss) >
            kEvalLossRtol * std::fabs(train_loss)) {
          Fail(Format(
              "val/bpb/8 implies a mean loss of %.9g nats, the recorded loss "
              "curve ends at %.9g",
              mean_nats, train_loss));
        }
      } else {
        Fail("the shard has no valid target to score");
      }
    }
  }

  // The ChatCORE record is the sentinel of an unmeasured run (section 5.4): the
  // trace records the cadence, not a value, so a test must accept -1 rather
  // than compare it against a measurement.
  if (ScalarInt(loop, "config/chatcore_measured") != 0) {
    Fail("config/chatcore_measured is 1, but the trace measures no ChatCORE");
  }
  if (ScalarF32(loop, "chatcore/8") != -1.0f) {
    Fail("chatcore/8 is not the -1 sentinel");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Run(LocateFile(argc, argv, 1, "tests/data/sft_loop_fixture.bin"),
        LocateFile(argc, argv, 2, "tests/data/sft_step_fixture.bin"),
        LocateFile(argc, argv, 3, "tests/data/sft_rows_fixture.nchsft"));
  } catch (const std::exception& error) {
    std::fprintf(stderr, "sft_loop_parity: %s\n", error.what());
    return 1;
  }
  if (g_failures != 0) {
    std::fprintf(stderr, "sft_loop_parity: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf(
      "sft_loop_parity: OK (progress, rate multiplier, momentum, epoch, loss "
      "curve, and the packed bits-per-byte; max schedule |diff| %.3g)\n",
      g_max_schedule_error);
  return 0;
}
