// tests/sft_parity_test.cc -- the R3 reader and driver gate of
// docs/sft-parity-plan.md sections 3 and 7 (phase F).
//
// Runs the real `src/sft_main` binary over the committed packed shard
// (`tests/data/sft_rows_fixture.nchsft`) from the committed base checkpoint
// (`tests/data/sft_base_fixture.ckpt`) with the pinned settings of sections 5.1
// to 5.4, then compares what the driver printed and what it wrote against the
// loop fixture (`tests/data/sft_loop_fixture.bin`):
//
//   * the eight `sft: step ...` lines: the counter, the epoch, the progress,
//   the
//     rate multiplier, the Muon momentum, and the logged loss;
//   * the `sft: eval ...` line: bits-per-byte over the packed validation rows
//     (design decision D6) and the valid-target and target-byte counts;
//   * the `sft: chatcore ... -1` line: the sentinel the driver reports at the
//     section 5.4 cadence, because it does not measure ChatCORE;
//   * the checkpoint it wrote: the stage, the stored step, the recorded recipe
//     (design decision D7), and the total parameter L2 against the final
//     parameters of the step fixture, which is the same reference run.
//
// Two things make the argument the driver needs from the committed bytes:
//
//   * the token-byte table of D6 lives inside the loop fixture
//     (`config/token_bytes`), and the driver takes it as a file, so the test
//     writes that record out as one little-endian `int32` per id;
//   * the base fixture records the container writer's *default* run facts in
//     its metadata block (`tests/data/sft_fixtures.txt` says so: the public
//     Python writer takes no metadata argument, so the tiny shape lives in the
//     tensors). `Checkpointer::LoadModel` compares the recorded run config with
//     the runtime model and refuses a mismatch, so a warm start from the
//     committed file as it stands is refused before a kernel reads a weight.
//     The test therefore restates the header with the config the tensors
//     actually have and writes that as the warm-start file in its temporary
//     directory: same parameter records, same (zero) moments, same step 0, and
//     no recipe block, so the driver applies the pinned section 5 settings.
//     No committed byte changes.
//
// The test is T2 work at a wave boundary, so it carries `gpu` and `manual`:
// `tools/nanochat test --gpu //tests:sft_parity_test` runs it, the T0 loop and
// the wildcard GPU sweep skip it. It is fp32-only and needs no torch, no
// network, and no reference checkout.

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/wait.h>

#include "nanochat/checkpoint_metadata.h"
#include "nanochat/data.h"
#include "nanochat/model.h"
#include "src/sft_shard.h"
#include "tests/oracle_fixture.h"
#include "tests/param_stage.h"

namespace {

using nanochat::Checkpoint;
using nanochat::Config;
using nanochat::TensorRecord;
using nanochat::oracle::Fixture;
using nanochat::oracle::Tensor;

// The driver prints six decimal places, so a scalar comparison is bounded from
// below by that rounding. The rest of the bound is the class
// `//tests:sft_step_parity_test` documents: the reference is an fp32 torch run
// and this recipe's `adam_eps = 1e-10` amplifies the last bits of a near-zero
// gradient, so the trajectory agrees to about a thousandth and not to `1e-5`.
constexpr double kLossAtol = 2e-3;
constexpr double kLossRtol = 2e-3;
constexpr double kScheduleAtol = 1e-5;
constexpr double kScheduleRtol = 1e-5;
constexpr double kBpbAtol = 1e-3;
constexpr double kBpbRtol = 1e-2;
constexpr double kParamL2Atol = 1e-3;
constexpr double kParamL2Rtol = 5e-3;

int g_failures = 0;
double g_max_loss_error = 0.0;
double g_max_schedule_error = 0.0;
double g_max_bpb_error = 0.0;
double g_max_param_l2_error = 0.0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

std::string Format(const char* fmt, ...) {
  char buffer[1024];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  return std::string(buffer);
}

bool Within(double got, double want, double atol, double rtol) {
  return std::fabs(got - want) <= atol + rtol * std::fabs(want);
}

std::string Getenv(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr ? std::string(value) : std::string();
}

std::string LocateFile(int argc, char** argv, int index, const char* relative) {
  std::vector<std::string> candidates;
  if (argc > index && argv[index] != nullptr && argv[index][0] != '\0') {
    candidates.emplace_back(argv[index]);
  }
  const std::string src_dir = Getenv("TEST_SRCDIR");
  if (!src_dir.empty()) {
    const std::string workspace = Getenv("TEST_WORKSPACE");
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

// The `//src:sft_main` binary, from the runfiles tree. The `//src:sft_main`
// data dependency is what puts it there.
std::string LocateSftMain() {
  std::vector<std::string> candidates;
  const std::string override = Getenv("NANOCHAT_SFT_MAIN");
  if (!override.empty()) candidates.push_back(override);
  const std::string src_dir = Getenv("TEST_SRCDIR");
  const std::string workspace = Getenv("TEST_WORKSPACE");
  if (!src_dir.empty()) {
    if (!workspace.empty()) {
      candidates.push_back(src_dir + "/" + workspace + "/src/sft_main");
    }
    candidates.push_back(src_dir + "/_main/src/sft_main");
  }
  candidates.emplace_back("bazel-bin/src/sft_main");
  candidates.emplace_back("../bazel-bin/src/sft_main");
  for (const std::string& candidate : candidates) {
    std::ifstream in(candidate);
    if (in.good()) return candidate;
  }
  throw std::runtime_error(
      "sft_main not found in the runfiles tree; add //src:sft_main to the test "
      "data or set NANOCHAT_SFT_MAIN");
}

std::string TempDir() {
  const std::string dir = Getenv("TEST_TMPDIR");
  return dir.empty() ? std::string("/tmp") : dir;
}

std::string ShellQuote(const std::string& text) {
  std::string quoted = "'";
  for (char c : text) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted += c;
    }
  }
  quoted += "'";
  return quoted;
}

std::int64_t ScalarInt(const Fixture& fixture, const std::string& name) {
  if (!fixture.Has(name)) {
    throw std::runtime_error("the fixture is missing record '" + name + "'");
  }
  return fixture.Get(name).scalar_int();
}

double ScalarF32(const Fixture& fixture, const std::string& name) {
  if (!fixture.Has(name)) {
    throw std::runtime_error("the fixture is missing record '" + name + "'");
  }
  return fixture.Get(name).scalar_f32();
}

// The tiny model of section 5.5, from the step fixture's `config/*` records. It
// is what the driver's `--layers 2 ...` flags produce, and what the warm-start
// header below has to record.
Config ConfigFromStepFixture(const Fixture& step) {
  Config config;
  config.num_layers = static_cast<int>(ScalarInt(step, "config/layers"));
  config.num_heads = static_cast<int>(ScalarInt(step, "config/heads"));
  config.num_kv_heads = static_cast<int>(ScalarInt(step, "config/kv_heads"));
  config.hidden_dim = static_cast<int>(ScalarInt(step, "config/embd"));
  config.seq_len = static_cast<int>(ScalarInt(step, "config/seq"));
  config.vocab_size = static_cast<int>(ScalarInt(step, "config/vocab"));
  config.padded_vocab_size =
      static_cast<int>(ScalarInt(step, "config/padded_vocab"));
  const Tensor& pattern = step.Get("config/window_pattern");
  config.window_pattern =
      std::string(pattern.data, pattern.data + pattern.numel());
  config.rope_base = 100000.0f;
  config.value_embedding = true;
  return config;
}

// Writes one little-endian int32 per token id, the format
// `sft_main --token-bytes` reads.
void WriteTokenBytes(const std::string& path, const Tensor& table) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write " + path);
  const std::int32_t* values = table.i32();
  for (std::int64_t i = 0; i < table.numel(); ++i) {
    const std::uint32_t bits = static_cast<std::uint32_t>(values[i]);
    const char bytes[4] = {static_cast<char>(bits & 0xffu),
                           static_cast<char>((bits >> 8) & 0xffu),
                           static_cast<char>((bits >> 16) & 0xffu),
                           static_cast<char>((bits >> 24) & 0xffu)};
    out.write(bytes, 4);
  }
}

// Writes the warm-start checkpoint: the committed parameter and optimizer-state
// records, under a metadata block that records the model those records are.
void WriteWarmStart(const std::string& source, const std::string& target,
                    const Config& config) {
  Checkpoint base;
  std::string error;
  if (!base.Load(source, &error)) {
    throw std::runtime_error("cannot read the base checkpoint: " + error);
  }
  Checkpoint warm;
  for (const TensorRecord& record : base.tensors()) warm.Add(record);
  for (const TensorRecord& record : base.optimizer_state()) {
    warm.AddOptimizerState(record);
  }
  nanochat::CheckpointMetadata metadata;
  metadata.stage = nanochat::Stage::kBase;
  metadata.step = 0;
  metadata.world_size = 1;
  metadata.config = config;
  // The recipe block stays at its defaults, which mean "not recorded", so the
  // driver applies the pinned section 5 settings exactly as it would for the
  // committed file (design decision D7).
  metadata.recipe.device_batch_size = 0;
  metadata.recipe.total_batch_size = 0;
  warm.set_metadata(metadata);
  if (!warm.Save(target)) {
    throw std::runtime_error("cannot write the warm-start checkpoint: " +
                             target);
  }
}

struct StepLine {
  int step = 0;
  int epoch = 0;
  double progress = 0.0;
  double lr_multiplier = 0.0;
  double momentum = 0.0;
  double loss = 0.0;
};

struct EvalLine {
  int step = 0;
  double bpb = 0.0;
  long long valid_targets = 0;
  long long target_bytes = 0;
};

// Parses the driver's progress log. The driver prints one `sft: step ...` line
// per step, one `sft: eval ...` line per evaluation, and one
// `sft: chatcore ... <value>` line per ChatCORE cadence.
void ParseLog(const std::string& path, std::vector<StepLine>* steps,
              std::vector<EvalLine>* evals, std::vector<double>* chatcore,
              int* written_step) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read the driver log: " + path);
  std::string line;
  while (std::getline(in, line)) {
    StepLine step;
    EvalLine eval;
    int at = 0;
    double chatcore_value = 0.0;
    char text[1024] = {0};
    if (std::sscanf(line.c_str(),
                    "sft: step %d epoch %d progress %lf lr_multiplier %lf "
                    "momentum %lf loss %lf",
                    &step.step, &step.epoch, &step.progress,
                    &step.lr_multiplier, &step.momentum, &step.loss) == 6) {
      steps->push_back(step);
    } else if (std::sscanf(line.c_str(),
                           "sft: eval step %d bpb %lf valid_targets %lld "
                           "target_bytes %lld",
                           &eval.step, &eval.bpb, &eval.valid_targets,
                           &eval.target_bytes) == 4) {
      evals->push_back(eval);
    } else if (std::sscanf(line.c_str(), "sft: chatcore step %d %lf", &at,
                           &chatcore_value) == 2) {
      chatcore->push_back(chatcore_value);
    } else if (std::sscanf(line.c_str(), "sft: wrote %1023s at step %d", text,
                           written_step) == 2) {
      // The path is informational; the step is not.
    }
  }
}

void Run(const std::string& loop_path, const std::string& step_path,
         const std::string& rows_path, const std::string& base_path) {
  const Fixture loop = Fixture::Load(loop_path);
  const Fixture step = Fixture::Load(step_path);
  if (ScalarInt(loop, "config/version") != 1 ||
      ScalarInt(step, "config/version") != 1) {
    throw std::runtime_error("unsupported SFT fixture version");
  }
  const int num_iterations =
      static_cast<int>(ScalarInt(loop, "config/num_iterations"));
  const int eval_every = static_cast<int>(ScalarInt(loop, "config/eval_every"));
  const int eval_tokens =
      static_cast<int>(ScalarInt(loop, "config/eval_tokens"));
  const int chatcore_every =
      static_cast<int>(ScalarInt(loop, "config/chatcore_every"));
  const int batch =
      static_cast<int>(ScalarInt(loop, "config/device_batch_size"));
  const int accumulate = static_cast<int>(ScalarInt(loop, "config/grad_accum"));
  const Config config = ConfigFromStepFixture(step);

  nanochat::SftShard shard;
  std::string shard_error;
  if (!nanochat::LoadSftShard(rows_path, &shard, &shard_error)) {
    throw std::runtime_error("cannot read the packed shard: " + shard_error);
  }
  if (shard.rows != static_cast<int>(ScalarInt(loop, "config/rows"))) {
    Fail(Format("the shard holds %d rows, the fixture records %lld", shard.rows,
                static_cast<long long>(ScalarInt(loop, "config/rows"))));
  }

  const std::string temp = TempDir();
  const std::string token_bytes_path = temp + "/sft_parity_token_bytes.bin";
  const std::string warm_path = temp + "/sft_parity_warm_start.ckpt";
  const std::string checkpoint_path = temp + "/sft_parity_final.ckpt";
  const std::string log_path = temp + "/sft_parity_driver.log";
  WriteTokenBytes(token_bytes_path, loop.Get("config/token_bytes"));
  WriteWarmStart(base_path, warm_path, config);

  const std::string binary = LocateSftMain();
  const std::string command =
      ShellQuote(binary) + " --rows " + ShellQuote(rows_path) +
      " --warm-start " + ShellQuote(warm_path) + " --val-rows " +
      ShellQuote(rows_path) + " --token-bytes " + ShellQuote(token_bytes_path) +
      " --num-iterations " + std::to_string(num_iterations) + " --batch " +
      std::to_string(batch) + " --grad-accum " + std::to_string(accumulate) +
      " --eval-every " + std::to_string(eval_every) + " --eval-tokens " +
      std::to_string(eval_tokens) + " --chatcore-every " +
      std::to_string(chatcore_every) + " --layers " +
      std::to_string(config.num_layers) + " --heads " +
      std::to_string(config.num_heads) + " --kv-heads " +
      std::to_string(config.num_kv_heads) + " --hidden " +
      std::to_string(config.hidden_dim) + " --seq " +
      std::to_string(config.seq_len) + " --vocab " +
      std::to_string(config.vocab_size) + " --padded-vocab " +
      std::to_string(config.padded_vocab_size) + " --window-pattern " +
      ShellQuote(config.window_pattern) + " --checkpoint " +
      ShellQuote(checkpoint_path) + " > " + ShellQuote(log_path) + " 2>&1";
  const int status = std::system(command.c_str());
  if (status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::ifstream in(log_path);
    std::string line;
    while (std::getline(in, line))
      std::fprintf(stderr, "driver: %s\n", line.c_str());
    Fail("sft_main did not run successfully (exit " + std::to_string(status) +
         ")");
    return;
  }

  std::vector<StepLine> lines;
  std::vector<EvalLine> evals;
  std::vector<double> chatcore;
  int written_step = 0;
  ParseLog(log_path, &lines, &evals, &chatcore, &written_step);

  // The loss curve, the counters, and the two schedules (sections 5.2 and 5.4).
  if (static_cast<int>(lines.size()) != num_iterations) {
    Fail(Format("the driver printed %d step lines, the fixture records %d",
                static_cast<int>(lines.size()), num_iterations));
  }
  for (const StepLine& line : lines) {
    const int k = line.step;
    if (k < 1 || k > num_iterations) {
      Fail(Format("the driver printed an out-of-range step %d", k));
      continue;
    }
    const std::string prefix = "loop/" + std::to_string(k);
    if (k - 1 != static_cast<int>(ScalarInt(loop, prefix + "/step"))) {
      Fail(Format("%s/step: the driver is at %d", prefix.c_str(), k - 1));
    }
    if (line.epoch != static_cast<int>(ScalarInt(loop, prefix + "/epoch"))) {
      Fail(Format("%s/epoch: got %d want %lld", prefix.c_str(), line.epoch,
                  static_cast<long long>(ScalarInt(loop, prefix + "/epoch"))));
    }
    const double progress = ScalarF32(loop, prefix + "/progress");
    const double lr_multiplier = ScalarF32(loop, prefix + "/lr_multiplier");
    const double momentum = ScalarF32(loop, prefix + "/momentum");
    g_max_schedule_error = std::max(
        g_max_schedule_error,
        std::max(std::fabs(line.progress - progress),
                 std::max(std::fabs(line.lr_multiplier - lr_multiplier),
                          std::fabs(line.momentum - momentum))));
    if (!Within(line.progress, progress, kScheduleAtol, kScheduleRtol)) {
      Fail(Format("%s/progress: got %.9g want %.9g", prefix.c_str(),
                  line.progress, progress));
    }
    if (!Within(line.lr_multiplier, lr_multiplier, kScheduleAtol,
                kScheduleRtol)) {
      Fail(Format("%s/lr_multiplier: got %.9g want %.9g", prefix.c_str(),
                  line.lr_multiplier, lr_multiplier));
    }
    if (!Within(line.momentum, momentum, kScheduleAtol, kScheduleRtol)) {
      Fail(Format("%s/momentum: got %.9g want %.9g", prefix.c_str(),
                  line.momentum, momentum));
    }
    const double want_loss = ScalarF32(loop, "train/loss/" + std::to_string(k));
    const double diff = std::fabs(line.loss - want_loss);
    g_max_loss_error = std::max(g_max_loss_error, diff);
    if (!Within(line.loss, want_loss, kLossAtol, kLossRtol)) {
      Fail(Format("train/loss/%d: got %.9g want %.9g (|diff|=%.3g)", k,
                  line.loss, want_loss, diff));
    }
  }

  // The evaluation (D6, section 5.4): bits-per-byte over the packed rows, one
  // pass, at the final step. The valid-target and target-byte counts come from
  // the shard and the fixture's token-byte table, so the metric's two inputs
  // are checked as well as its value.
  if (evals.empty()) {
    Fail("the driver reported no evaluation");
  } else {
    const EvalLine& eval = evals.back();
    if (eval.step != num_iterations) {
      Fail(Format("the evaluation ran at step %d, not the final step %d",
                  eval.step, num_iterations));
    }
    const double want_bpb = ScalarF32(loop, "val/bpb/8");
    const double diff = std::fabs(eval.bpb - want_bpb);
    g_max_bpb_error = std::max(g_max_bpb_error, diff);
    if (!Within(eval.bpb, want_bpb, kBpbAtol, kBpbRtol)) {
      Fail(Format("val/bpb/8: got %.9g want %.9g (|diff|=%.3g)", eval.bpb,
                  want_bpb, diff));
    }
    const Tensor& table = loop.Get("config/token_bytes");
    const std::int32_t* token_bytes = table.i32();
    long long valid_targets = 0;
    long long target_bytes = 0;
    for (std::size_t i = 0; i < shard.targets.size(); ++i) {
      const int target = shard.targets[i];
      if (target < 0 || target >= table.numel()) continue;
      ++valid_targets;
      target_bytes += token_bytes[target];
    }
    if (eval.valid_targets != valid_targets ||
        eval.target_bytes != target_bytes) {
      Fail(Format(
          "the evaluation scored %lld targets in %lld bytes, the shard "
          "holds %lld in %lld",
          eval.valid_targets, eval.target_bytes, valid_targets, target_bytes));
    }
  }

  // The ChatCORE cadence: the driver does not measure ChatCORE, so it reports
  // the same sentinel the fixture pins (D6, section 5.4).
  const double want_chatcore = ScalarF32(loop, "chatcore/8");
  if (chatcore.empty()) {
    Fail("the driver reported no ChatCORE cadence");
  } else {
    for (double value : chatcore) {
      if (value != want_chatcore) {
        Fail(Format("chatcore: got %.9g want %.9g", value, want_chatcore));
        break;
      }
    }
  }

  // The checkpoint the driver wrote: an SFT stage, the final step, the recipe
  // it ran (D7), and the reference's final parameters.
  Checkpoint written;
  std::string error;
  if (!written.Load(checkpoint_path, &error)) {
    Fail("the driver wrote no readable checkpoint: " + error);
    return;
  }
  const nanochat::CheckpointMetadata metadata = written.metadata();
  if (metadata.stage != nanochat::Stage::kSft) {
    Fail("the driver's checkpoint does not record the SFT stage");
  }
  if (metadata.step != static_cast<std::uint64_t>(num_iterations)) {
    Fail(Format("the driver's checkpoint records step %llu, want %d",
                static_cast<unsigned long long>(metadata.step),
                num_iterations));
  }
  if (written_step != num_iterations) {
    Fail(Format("the driver reported writing step %d, want %d", written_step,
                num_iterations));
  }
  // The recipe this run recorded is the shape it actually ran: the accumulation
  // of section 5.1 back-derives from it.
  const nanochat::Recipe& recipe = metadata.recipe;
  if (recipe.device_batch_size != batch) {
    Fail(Format("the recorded device batch is %d, want %d",
                recipe.device_batch_size, batch));
  }
  const int expected_total = batch * config.seq_len * accumulate;
  if (recipe.total_batch_size != expected_total) {
    Fail(Format("the recorded total batch is %d, want %d",
                recipe.total_batch_size, expected_total));
  }
  // The final parameters. As in `//tests:sft_step_parity_test`, the gate is the
  // total L2 norm of the whole parameter set, which is the same reference run's
  // `step/8/param_l2`; a per-element bound cannot hold on this recipe.
  double total = 0.0;
  for (const TensorRecord& record : written.tensors()) {
    const std::size_t count = record.data.size() / sizeof(float);
    const float* values = reinterpret_cast<const float*>(record.data.data());
    for (std::size_t i = 0; i < count; ++i) {
      total += static_cast<double>(values[i]) * static_cast<double>(values[i]);
    }
  }
  const double norm = std::sqrt(total);
  const double want_norm = ScalarF32(step, "step/8/param_l2");
  const double diff = std::fabs(norm - want_norm);
  g_max_param_l2_error = std::max(g_max_param_l2_error, diff);
  if (!Within(norm, want_norm, kParamL2Atol, kParamL2Rtol)) {
    Fail(Format("final parameter L2: got %.9g want %.9g (|diff|=%.3g)", norm,
                want_norm, diff));
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (nanochat::testutil::SkipFp32Golden("sft_parity")) return 0;
  try {
    Run(LocateFile(argc, argv, 1, "tests/data/sft_loop_fixture.bin"),
        LocateFile(argc, argv, 2, "tests/data/sft_step_fixture.bin"),
        LocateFile(argc, argv, 3, "tests/data/sft_rows_fixture.nchsft"),
        LocateFile(argc, argv, 4, "tests/data/sft_base_fixture.ckpt"));
  } catch (const std::exception& error) {
    std::fprintf(stderr, "sft_parity: %s\n", error.what());
    return 1;
  }
  std::printf(
      "sft_parity: max errors (loss %.3g, schedule %.3g, bpb %.3g, param_l2 "
      "%.3g)%s\n",
      g_max_loss_error, g_max_schedule_error, g_max_bpb_error,
      g_max_param_l2_error, g_failures == 0 ? "" : " [FAILED]");
  return g_failures == 0 ? 0 : 1;
}
