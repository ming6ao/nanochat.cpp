// tests/sft_fixture_load_test.cc -- the phase B gate of
// docs/sft-parity-plan.md section 7.
//
// The five committed SFT fixtures must be readable with no reference checkout,
// no torch, no tokenizer artifact, and no network. This test is the proof: it
// opens the two record fixtures through `tests/oracle_fixture.h`, reads the
// packed shard with `LoadSftShard`, loads the base checkpoint with
// `Checkpoint::Load`, and parses the pack fixture's `NANCHAT1` header and JSON
// with the reader `src/json.h` owns. Nothing here computes a model number; that
// is `//tests:sft_step_parity_test` (R2) and `//tests:sft_loop_parity_test`
// (R3).
//
// What it pins, per file (`tests/data/sft_fixtures.txt` is the durable record):
//
//   * sft_step_fixture.bin -- the `config/*` run facts, the effective rate
//     table, the eight recorded micro-batch pairs, and the 21 final
//     parameters;
//   * sft_loop_fixture.bin -- the iteration count, the evaluation cadence, the
//     eight `loop/<k>/*` counters, the loss curve, and the final evaluation
//     with its ChatCORE sentinel;
//   * sft_rows_fixture.nchsft -- four rows of 32 ids, and the wrap rule that
//     ties each recorded micro-batch to a row;
//   * sft_base_fixture.ckpt -- 21 parameter records and 20 optimizer-state
//     records, under the same names the step fixture uses;
//   * sft_pack_fixture.bin -- the `NANCHAT1` magic, the version word, and the
//     decoded JSON object the phase C.1 test reads.
//
// Run it as `tools/nanochat test //tests:sft_fixture_load_test`. It is T0: the
// five files together are 732 KiB (section 5.5 of the plan).

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "nanochat/data.h"
#include "src/json.h"
#include "src/sft_shard.h"
#include "tests/oracle_fixture.h"

namespace {

using nanochat::Checkpoint;
using nanochat::TensorRecord;
using nanochat::oracle::DType;
using nanochat::oracle::Fixture;
using nanochat::oracle::Tensor;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

void Check(bool condition, const std::string& message) {
  if (!condition) Fail(message);
}

std::string Format(const char* fmt, ...) {
  char buffer[1024];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  return std::string(buffer);
}

// The runfiles lookup of the other fixture tests: the Bazel runfiles tree
// first, then the tree relative to the working directory.
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

std::vector<std::uint8_t> ReadBytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
}

// One expected record of a `NANOORC1` fixture: the name, the dtype, and the
// shape. A mismatch names the record and both sides, so a renamed or reshaped
// record fails with the difference instead of a bare index.
struct ExpectedRecord {
  const char* name;
  DType dtype;
  std::vector<std::int64_t> shape;
};

void CheckRecord(const Fixture& fixture, const ExpectedRecord& expected) {
  if (!fixture.Has(expected.name)) {
    Fail(std::string("missing record '") + expected.name + "'");
    return;
  }
  const Tensor& got = fixture.Get(expected.name);
  if (got.dtype != expected.dtype) {
    Fail(std::string(expected.name) + ": dtype " +
         std::to_string(static_cast<int>(got.dtype)) + ", want " +
         std::to_string(static_cast<int>(expected.dtype)));
  }
  if (got.shape != expected.shape) {
    std::string got_shape;
    for (std::int64_t extent : got.shape) {
      got_shape += (got_shape.empty() ? "" : "x") + std::to_string(extent);
    }
    Fail(std::string(expected.name) + ": shape " + got_shape);
  }
}

void CheckInt(const Fixture& fixture, const char* name, std::int64_t want) {
  if (!fixture.Has(name)) {
    Fail(std::string("missing record '") + name + "'");
    return;
  }
  const std::int64_t got = fixture.Get(name).scalar_int();
  if (got != want) {
    Fail(Format("%s: got %lld want %lld", name, static_cast<long long>(got),
                static_cast<long long>(want)));
  }
}

void CheckFloat(const Fixture& fixture, const char* name, double want) {
  if (!fixture.Has(name)) {
    Fail(std::string("missing record '") + name + "'");
    return;
  }
  const double got = fixture.Get(name).scalar_f32();
  // Every value here is a pinned setting written from the same decimal, so the
  // only difference a correct file can carry is the fp32 round trip.
  if (std::fabs(got - want) > 1e-6 + 1e-6 * std::fabs(want)) {
    Fail(Format("%s: got %.9g want %.9g", name, got, want));
  }
}

// The 21 parameters both the step fixture and the base checkpoint carry, with
// the shapes of section 6.4. The names are the reference names, so the two
// files are comparable record by record.
const ExpectedRecord kParameters[] = {
    {"lm_head.weight", DType::kFp32, {256, 32}},
    {"transformer.wte.weight", DType::kFp32, {256, 32}},
    {"value_embeds.1.weight", DType::kFp32, {256, 32}},
    {"resid_lambdas", DType::kFp32, {2}},
    {"x0_lambdas", DType::kFp32, {2}},
    {"smear_gate.weight", DType::kFp32, {1, 24}},
    {"smear_lambda", DType::kFp32, {1}},
    {"backout_lambda", DType::kFp32, {1}},
    {"transformer.h.0.attn.c_q.weight", DType::kFp32, {32, 32}},
    {"transformer.h.0.attn.c_k.weight", DType::kFp32, {32, 32}},
    {"transformer.h.0.attn.c_v.weight", DType::kFp32, {32, 32}},
    {"transformer.h.0.attn.c_proj.weight", DType::kFp32, {32, 32}},
    {"transformer.h.0.mlp.c_fc.weight", DType::kFp32, {128, 32}},
    {"transformer.h.0.mlp.c_proj.weight", DType::kFp32, {32, 128}},
    {"transformer.h.1.attn.c_q.weight", DType::kFp32, {32, 32}},
    {"transformer.h.1.attn.c_k.weight", DType::kFp32, {32, 32}},
    {"transformer.h.1.attn.c_v.weight", DType::kFp32, {32, 32}},
    {"transformer.h.1.attn.c_proj.weight", DType::kFp32, {32, 32}},
    {"transformer.h.1.attn.ve_gate.weight", DType::kFp32, {2, 12}},
    {"transformer.h.1.mlp.c_fc.weight", DType::kFp32, {128, 32}},
    {"transformer.h.1.mlp.c_proj.weight", DType::kFp32, {32, 128}},
};

// The 20 optimizer-state records of the base checkpoint, in file order: the
// AdamW moments per group and the Muon buffers per shape (section 6.4).
const ExpectedRecord kOptimizerState[] = {
    {"adamw.lm_head.m", DType::kFp32, {8192}},
    {"adamw.lm_head.v", DType::kFp32, {8192}},
    {"adamw.embedding.m", DType::kFp32, {8192}},
    {"adamw.embedding.v", DType::kFp32, {8192}},
    {"adamw.value_embedding.m", DType::kFp32, {8192}},
    {"adamw.value_embedding.v", DType::kFp32, {8192}},
    {"adamw.resid.m", DType::kFp32, {2}},
    {"adamw.resid.v", DType::kFp32, {2}},
    {"adamw.x0.m", DType::kFp32, {2}},
    {"adamw.x0.v", DType::kFp32, {2}},
    {"adamw.smear.m", DType::kFp32, {26}},
    {"adamw.smear.v", DType::kFp32, {26}},
    {"muon.2x12.buf1", DType::kFp32, {24}},
    {"muon.2x12.buf2", DType::kFp32, {12}},
    {"muon.32x32.buf1", DType::kFp32, {8192}},
    {"muon.32x32.buf2", DType::kFp32, {256}},
    {"muon.32x128.buf1", DType::kFp32, {8192}},
    {"muon.32x128.buf2", DType::kFp32, {256}},
    {"muon.128x32.buf1", DType::kFp32, {8192}},
    {"muon.128x32.buf2", DType::kFp32, {256}},
};

void CheckNoDuplicateNames(const Fixture& fixture, const std::string& label) {
  std::set<std::string> seen;
  for (const std::string& name : fixture.names()) {
    if (!seen.insert(name).second) Fail(label + ": duplicate record " + name);
  }
  Check(static_cast<int>(seen.size()) == static_cast<int>(fixture.size()),
        label + ": the record order and the lookup disagree");
}

// The step fixture of section 6.2: the pinned run facts, the effective rate
// table, the eight recorded micro-batch pairs, and the final parameters.
void CheckStepFixture(const std::string& path) {
  const Fixture fixture = Fixture::Load(path);
  Check(fixture.magic() == "NANOORC1",
        "sft_step_fixture: magic " + fixture.magic());
  Check(fixture.format_version() == 1, "sft_step_fixture: format version");
  CheckNoDuplicateNames(fixture, "sft_step_fixture");
  Check(fixture.size() == 112, Format("sft_step_fixture: %d records, want 112",
                                      static_cast<int>(fixture.size())));

  CheckInt(fixture, "config/version", 1);
  CheckInt(fixture, "config/seed", 1234);
  CheckInt(fixture, "config/layers", 2);
  CheckInt(fixture, "config/heads", 2);
  CheckInt(fixture, "config/kv_heads", 2);
  CheckInt(fixture, "config/embd", 32);
  CheckInt(fixture, "config/seq", 32);
  CheckInt(fixture, "config/vocab", 256);
  CheckInt(fixture, "config/padded_vocab", 256);
  CheckInt(fixture, "config/device_batch_size", 1);
  CheckInt(fixture, "config/grad_accum", 2);
  CheckInt(fixture, "config/steps", 8);
  {
    const Tensor& pattern = fixture.Get("config/window_pattern");
    CheckRecord(fixture, {"config/window_pattern", DType::kUInt8, {1}});
    const std::string text(pattern.data, pattern.data + pattern.bytes);
    Check(text == "L", "config/window_pattern: " + text);
  }

  // The optimizer configuration of section 5.2.
  CheckFloat(fixture, "config/opt/unembedding_lr", 0.004);
  CheckFloat(fixture, "config/opt/embedding_lr", 0.3);
  CheckFloat(fixture, "config/opt/matrix_lr", 0.02);
  CheckFloat(fixture, "config/opt/scalar_lr", 0.5);
  CheckFloat(fixture, "config/opt/init_lr_frac", 0.8);
  CheckFloat(fixture, "config/opt/adam_eps", 1e-10);
  CheckFloat(fixture, "config/opt/clip", 0.0);
  // `(hidden_dim / 768) ** -0.5` for the tiny model.
  CheckFloat(fixture, "config/opt/lr_scale", std::sqrt(768.0 / 32.0));

  // The schedule: full warmdown from the midpoint, no warmup, and the Muon ramp
  // that has already reached its peak inside a 300-step window.
  CheckFloat(fixture, "config/schedule/warmup_ratio", 0.0);
  CheckFloat(fixture, "config/schedule/warmdown_ratio", 0.5);
  CheckFloat(fixture, "config/schedule/final_lr_frac", 0.0);
  CheckInt(fixture, "config/schedule/muon_momentum_steps", 300);
  CheckFloat(fixture, "config/schedule/muon_momentum_start", 0.85);
  CheckFloat(fixture, "config/schedule/muon_momentum_peak", 0.95);
  CheckFloat(fixture, "config/schedule/muon_momentum_final", 0.95);

  // The rate table after the `(hidden_dim / 768) ** -0.5` scale (section 6.2).
  CheckFloat(fixture, "config/opt/rate/lm_head",
             0.004 * std::sqrt(768.0 / 32.0) * 0.8);
  CheckFloat(fixture, "config/opt/rate/embedding",
             0.3 * std::sqrt(768.0 / 32.0) * 0.8);
  CheckFloat(fixture, "config/opt/rate/value_embedding",
             0.15 * std::sqrt(768.0 / 32.0) * 0.8);
  CheckFloat(fixture, "config/opt/rate/resid", 0.005 * 0.8);
  CheckFloat(fixture, "config/opt/rate/x0", 0.5 * 0.8);
  CheckFloat(fixture, "config/opt/rate/smear", 0.2 * 0.8);
  CheckFloat(fixture, "config/opt/rate/muon", 0.02 * 0.8);

  // The eight steps: two micro-batches of one row of 32 ids each, and the five
  // scalars the parity test compares.
  const int steps = 8;
  for (int step = 1; step <= steps; ++step) {
    const std::string prefix = "step/" + std::to_string(step);
    const std::string data = "data/step/" + std::to_string(step);
    CheckRecord(fixture,
                {(data + "/tokens").c_str(), DType::kInt32, {2, 1, 32}});
    CheckRecord(fixture,
                {(data + "/targets").c_str(), DType::kInt32, {2, 1, 32}});
    CheckRecord(fixture, {(prefix + "/loss").c_str(), DType::kFp32, {1}});
    CheckRecord(fixture, {(prefix + "/grad_norm").c_str(), DType::kFp32, {1}});
    CheckRecord(fixture, {(prefix + "/param_l2").c_str(), DType::kFp32, {1}});
    CheckRecord(fixture, {(prefix + "/lr").c_str(), DType::kFp32, {1}});
    CheckRecord(fixture, {(prefix + "/momentum").c_str(), DType::kFp32, {1}});
    if (fixture.Has(prefix + "/loss")) {
      const double loss = fixture.Get(prefix + "/loss").scalar_f32();
      Check(std::isfinite(loss) && loss > 0.0,
            prefix + "/loss: not a positive finite loss");
    }
    // Every target is a real piece or the ignore index.
    if (fixture.Has(data + "/targets")) {
      const Tensor& targets = fixture.Get(data + "/targets");
      const std::int32_t* values = targets.i32();
      for (std::int64_t i = 0; i < targets.numel(); ++i) {
        if (values[i] < -1 || values[i] >= 256) {
          Fail(Format("%s/targets[%lld] = %d", data.c_str(),
                      static_cast<long long>(i), values[i]));
          break;
        }
      }
    }
  }

  // The final parameters, in the 21 canonical names of section 6.4.
  for (const ExpectedRecord& expected : kParameters) {
    const std::string name = std::string("step/8/param/") + expected.name;
    CheckRecord(fixture, {name.c_str(), expected.dtype, expected.shape});
  }
  // The loss falls over the run: the fixture is a real trajectory, not a row of
  // repeated numbers. Step 8 steps at multiplier 0, so its parameter L2 equals
  // step 7's exactly (section 6.2).
  CheckFloat(fixture, "step/8/lr", 0.0);
  if (fixture.Has("step/7/param_l2") && fixture.Has("step/8/param_l2")) {
    Check(fixture.Get("step/7/param_l2").scalar_f32() ==
              fixture.Get("step/8/param_l2").scalar_f32(),
          "step/8/param_l2: the zero multiplier step changed the parameters");
  }
}

// The loop fixture of section 6.3: the cadence, the eight counters, the loss
// curve, and the final evaluation with its ChatCORE sentinel.
void CheckLoopFixture(const std::string& path) {
  const Fixture fixture = Fixture::Load(path);
  Check(fixture.magic() == "NANOORC1",
        "sft_loop_fixture: magic " + fixture.magic());
  Check(fixture.format_version() == 1, "sft_loop_fixture: format version");
  CheckNoDuplicateNames(fixture, "sft_loop_fixture");
  Check(fixture.size() == 64, Format("sft_loop_fixture: %d records, want 64",
                                     static_cast<int>(fixture.size())));

  CheckInt(fixture, "config/version", 1);
  CheckInt(fixture, "config/num_iterations", 8);
  CheckInt(fixture, "config/rows", 4);
  CheckInt(fixture, "config/eval_every", 200);
  // The evaluation the reference always runs: the final step (section 5.4).
  CheckInt(fixture, "config/eval_at", 8);
  CheckInt(fixture, "config/eval_tokens", 40 * 524288);
  CheckInt(fixture, "config/chatcore_every", 200);
  CheckInt(fixture, "config/chatcore_max_cat", -1);
  CheckInt(fixture, "config/chatcore_max_sample", 24);
  CheckInt(fixture, "config/device_batch_size", 1);
  CheckInt(fixture, "config/grad_accum", 2);
  CheckInt(fixture, "config/seq_len", 32);
  CheckInt(fixture, "config/chatcore_measured", 0);
  CheckRecord(fixture, {"config/token_bytes", DType::kInt32, {256}});
  {
    const Tensor& table = fixture.Get("config/token_bytes");
    const std::int32_t* bytes = table.i32();
    std::int64_t total = 0;
    for (std::int64_t i = 0; i < table.numel(); ++i) {
      // An id the fixture never emits can carry zero bytes; a negative count is
      // never valid, and a table that sums to zero would divide by zero in the
      // bits-per-byte metric (design decision D6).
      if (bytes[i] < 0) {
        Fail(Format("config/token_bytes[%lld] = %d", static_cast<long long>(i),
                    bytes[i]));
        break;
      }
      total += bytes[i];
    }
    Check(total > 0, "config/token_bytes: the table sums to zero bytes");
  }

  double previous_lr = 2.0;
  float previous_progress = -1.0f;
  int previous_epoch = 0;
  for (int k = 1; k <= 8; ++k) {
    const std::string prefix = "loop/" + std::to_string(k);
    CheckRecord(fixture, {(prefix + "/step").c_str(), DType::kInt32, {1}});
    CheckRecord(fixture, {(prefix + "/progress").c_str(), DType::kFp32, {1}});
    CheckRecord(fixture, {(prefix + "/epoch").c_str(), DType::kInt32, {1}});
    CheckRecord(fixture,
                {(prefix + "/lr_multiplier").c_str(), DType::kFp32, {1}});
    CheckRecord(fixture, {(prefix + "/momentum").c_str(), DType::kFp32, {1}});
    const std::string loss = "train/loss/" + std::to_string(k);
    CheckRecord(fixture, {loss.c_str(), DType::kFp32, {1}});
    if (!fixture.Has(prefix + "/step")) continue;
    // The reference loop counter is 0-based; the driver's is 1-based (D2).
    CheckInt(fixture, (prefix + "/step").c_str(), k - 1);
    const float progress = fixture.Get(prefix + "/progress").scalar_f32();
    Check(progress > previous_progress, prefix + "/progress is not increasing");
    previous_progress = progress;
    const int epoch =
        static_cast<int>(fixture.Get(prefix + "/epoch").scalar_int());
    Check(epoch >= previous_epoch, prefix + "/epoch went backwards");
    previous_epoch = epoch;
    // The rate multiplier is the progress curve: 1.0 until the midpoint of the
    // warmdown, then a linear decay to zero (section 5.2).
    const double lr = fixture.Get(prefix + "/lr_multiplier").scalar_f32();
    Check(lr <= previous_lr, prefix + "/lr_multiplier is not decreasing");
    previous_lr = lr;
    // The momentum ramps from 0.85 toward the 0.95 peak (D2, section 5.2).
    const double momentum = fixture.Get(prefix + "/momentum").scalar_f32();
    Check(momentum >= 0.85 && momentum <= 0.95,
          Format("%s/momentum %.9g is outside the ramp", prefix.c_str(),
                 momentum));
    const double train_loss = fixture.Get(loss).scalar_f32();
    Check(std::isfinite(train_loss) && train_loss > 0.0,
          loss + ": not a positive finite loss");
  }

  // The final evaluation: a bits-per-byte value over the packed rows, and the
  // ChatCORE sentinel, which the trace records because it does not measure it.
  CheckRecord(fixture, {"val/bpb/8", DType::kFp32, {1}});
  CheckRecord(fixture, {"chatcore/8", DType::kFp32, {1}});
  if (fixture.Has("val/bpb/8")) {
    const double bpb = fixture.Get("val/bpb/8").scalar_f32();
    // A uniform distribution over 256 ids is 8 bits per byte; a byte-level
    // model of this corpus sits well below that.
    Check(std::isfinite(bpb) && bpb > 0.0 && bpb < 8.0,
          Format("val/bpb/8 %.9g is not a plausible bits-per-byte", bpb));
  }
  if (fixture.Has("chatcore/8")) {
    Check(fixture.Get("chatcore/8").scalar_f32() == -1.0f,
          "chatcore/8: the sentinel of an unmeasured ChatCORE is -1");
  }
  // The loss curve falls over the eight steps.
  if (fixture.Has("train/loss/1") && fixture.Has("train/loss/8")) {
    Check(fixture.Get("train/loss/8").scalar_f32() <
              fixture.Get("train/loss/1").scalar_f32(),
          "the recorded loss curve does not fall");
  }
}

// The shard of section 6.4, and the wrap rule that ties it to the recorded
// micro-batches: `data/step/<k>/*` holds rows `2k - 2` and `2k - 1`, wrapping
// after the fourth.
void CheckShard(const std::string& path, const Fixture& step_fixture) {
  nanochat::SftShard shard;
  std::string error;
  if (!nanochat::LoadSftShard(path, &shard, &error)) {
    Fail("sft_rows_fixture: " + error);
    return;
  }
  Check(shard.rows == 4, Format("shard rows %d, want 4", shard.rows));
  Check(shard.seq == 32, Format("shard seq %d, want 32", shard.seq));
  Check(static_cast<int>(shard.tokens.size()) == shard.rows * shard.seq,
        "shard token count");
  Check(static_cast<int>(shard.targets.size()) == shard.rows * shard.seq,
        "shard target count");
  if (shard.rows != 4 || shard.seq != 32) return;

  for (int step = 1; step <= 8; ++step) {
    const std::string prefix = "data/step/" + std::to_string(step);
    if (!step_fixture.Has(prefix + "/tokens") ||
        !step_fixture.Has(prefix + "/targets")) {
      Fail("the step fixture is missing " + prefix);
      return;
    }
    const Tensor& tokens = step_fixture.Get(prefix + "/tokens");
    const Tensor& targets = step_fixture.Get(prefix + "/targets");
    const std::int32_t* token_values = tokens.i32();
    const std::int32_t* target_values = targets.i32();
    for (int micro = 0; micro < 2; ++micro) {
      const int row = (2 * step - 2 + micro) % shard.rows;
      for (int position = 0; position < shard.seq; ++position) {
        const std::size_t recorded =
            static_cast<std::size_t>(micro) * shard.seq + position;
        const std::size_t shard_index =
            static_cast<std::size_t>(row) * shard.seq + position;
        if (token_values[recorded] != shard.tokens[shard_index] ||
            target_values[recorded] != shard.targets[shard_index]) {
          Fail(Format("%s micro %d position %d is not shard row %d",
                      prefix.c_str(), micro, position, row));
          return;
        }
      }
    }
  }
}

// The base checkpoint of section 6.4: 21 parameters and 20 optimizer-state
// records at step 0, under the reference names the step fixture uses.
void CheckBaseCheckpoint(const std::string& path, const Fixture& step_fixture) {
  Checkpoint checkpoint;
  std::string error;
  if (!checkpoint.Load(path, &error)) {
    Fail("sft_base_fixture: " + error);
    return;
  }
  Check(static_cast<int>(checkpoint.tensors().size()) == 21,
        Format("base checkpoint: %d parameter records, want 21",
               static_cast<int>(checkpoint.tensors().size())));
  Check(static_cast<int>(checkpoint.optimizer_state().size()) == 20,
        Format("base checkpoint: %d optimizer-state records, want 20",
               static_cast<int>(checkpoint.optimizer_state().size())));

  for (const ExpectedRecord& expected : kParameters) {
    const TensorRecord* record = checkpoint.Find(expected.name);
    if (record == nullptr) {
      Fail(std::string("base checkpoint is missing '") + expected.name + "'");
      continue;
    }
    if (record->dtype != nanochat::DType::kFp32) {
      Fail(std::string(expected.name) + ": the payload is not float32");
    }
    if (record->shape != expected.shape) {
      Fail(std::string(expected.name) + ": shape differs from section 6.4");
    }
    // The step fixture records the same parameter under the same name, so the
    // two files describe one model.
    const std::string final_name = std::string("step/8/param/") + expected.name;
    if (!step_fixture.Has(final_name)) {
      Fail("the step fixture is missing " + final_name);
    }
  }
  for (const ExpectedRecord& expected : kOptimizerState) {
    const TensorRecord* record = checkpoint.FindOptimizerState(expected.name);
    if (record == nullptr) {
      Fail(std::string("base checkpoint is missing optimizer state '") +
           expected.name + "'");
      continue;
    }
    if (record->dtype != nanochat::DType::kFp32) {
      Fail(std::string(expected.name) + ": the payload is not float32");
    }
    if (record->shape != expected.shape) {
      Fail(std::string(expected.name) + ": shape differs from section 6.4");
    }
  }
  // The moments are the reference optimizer's state at step 0, which is a fresh
  // optimizer: every record is present and zero. A nonzero record would mean
  // the base checkpoint is not a step-0 file, and the step parity gate would
  // compare its first step against the wrong start.
  for (const TensorRecord& record : checkpoint.optimizer_state()) {
    for (std::size_t i = 0; i + sizeof(float) <= record.data.size();
         i += sizeof(float)) {
      float value = 0.0f;
      std::memcpy(&value, record.data.data() + i, sizeof(value));
      if (value != 0.0f) {
        Fail(record.name + ": the moments are not the step-0 state");
        return;
      }
    }
  }

  // The metadata block records the writer's default run facts at step 0 (the
  // note in `tests/data/sft_fixtures.txt` says so); the tiny shape lives in the
  // tensors.
  const nanochat::CheckpointMetadata metadata = checkpoint.metadata();
  Check(metadata.schema == 1,
        Format("base checkpoint: schema %u", metadata.schema));
  Check(metadata.stage == nanochat::Stage::kBase,
        "base checkpoint: the stage is not base");
  Check(metadata.step == 0,
        Format("base checkpoint: step %llu, want 0",
               static_cast<unsigned long long>(metadata.step)));
  Check(metadata.world_size == 1, "base checkpoint: world size is not 1");
}

// The pack fixture of section 6.1: the magic, the version word, and the JSON
// object the phase C.1 test reads.
void CheckPackFixture(const std::string& path) {
  const std::vector<std::uint8_t> raw = ReadBytes(path);
  if (raw.size() < 12) {
    Fail("sft_pack_fixture: the file is too short to carry a header");
    return;
  }
  Check(std::string(raw.begin(), raw.begin() + 8) == "NANCHAT1",
        "sft_pack_fixture: magic " + std::string(raw.begin(), raw.begin() + 8));
  const std::uint32_t version = static_cast<std::uint32_t>(raw[8]) |
                                (static_cast<std::uint32_t>(raw[9]) << 8) |
                                (static_cast<std::uint32_t>(raw[10]) << 16) |
                                (static_cast<std::uint32_t>(raw[11]) << 24);
  Check(version == 1, Format("sft_pack_fixture: version %u", version));

  nanochat::json::Value root;
  std::string error;
  const std::string text(raw.begin() + 12, raw.end());
  if (!nanochat::json::Parse(text, &root, &error)) {
    Fail("sft_pack_fixture: " + error);
    return;
  }
  if (!root.is_object()) {
    Fail("sft_pack_fixture: the payload is not a JSON object");
    return;
  }
  for (const char* key : {"version", "settings", "tokenizer", "conversations",
                          "render", "pack", "mixture", "validation_mixture"}) {
    Check(root.Has(key), std::string("sft_pack_fixture: missing key ") + key);
  }

  // The settings of section 5.5.
  const nanochat::json::Value* settings = root.Find("settings");
  if (settings != nullptr) {
    const auto number = [&](const char* key) {
      const nanochat::json::Value* value = settings->Find(key);
      int out = 0;
      if (value == nullptr || !value->Integer(&out)) {
        Fail(std::string("settings/") + key + " is missing or not an integer");
        return 0;
      }
      return out;
    };
    Check(number("buffer_size") == 100, "settings/buffer_size");
    Check(number("render_max_tokens") == 2048, "settings/render_max_tokens");
    Check(number("conversations") == 8, "settings/conversations");
    Check(number("rank") == 0, "settings/rank");
    Check(number("rows") == 4, "settings/rows");
    Check(number("seed") == 1234, "settings/seed");
    Check(number("seq_len") == 32, "settings/seq_len");
    Check(number("world_size") == 1, "settings/world_size");
    const nanochat::json::Value* stop = settings->Find("stop_rule");
    Check(stop != nullptr && stop->is_string() && stop->string() == "last_step",
          "settings/stop_rule is not \"last_step\"");
  }

  // Eight conversations, each with a name and a message list.
  const nanochat::json::Value* conversations = root.Find("conversations");
  if (conversations != nullptr) {
    Check(conversations->is_array() && conversations->array().size() == 8,
          "conversations: not eight entries");
    for (const nanochat::json::Value& entry : conversations->array()) {
      Check(entry.Has("name") && entry.Has("messages"),
            "a conversation has no name or no messages");
    }
  }

  // The render section: ids and mask of the same length, per conversation.
  const nanochat::json::Value* render = root.Find("render");
  if (render != nullptr) {
    Check(render->is_array() && render->array().size() == 8,
          "render: not eight entries");
    for (const nanochat::json::Value& entry : render->array()) {
      const nanochat::json::Value* ids = entry.Find("ids");
      const nanochat::json::Value* mask = entry.Find("mask");
      if (ids == nullptr || mask == nullptr || !ids->is_array() ||
          !mask->is_array()) {
        Fail("render: an entry has no ids or no mask");
        continue;
      }
      Check(ids->array().size() == mask->array().size(),
            "render: ids and mask differ in length");
    }
  }

  // The pack section: four rows of 32 inputs and 32 targets, with the inputs a
  // real piece (0 for BOS padding) and the targets `-1` where masked.
  const nanochat::json::Value* pack = root.Find("pack");
  if (pack != nullptr) {
    for (const char* key : {"inputs", "targets", "rows"}) {
      Check(pack->Has(key), std::string("pack: missing ") + key);
    }
    const nanochat::json::Value* inputs = pack->Find("inputs");
    const nanochat::json::Value* targets = pack->Find("targets");
    if (inputs != nullptr && targets != nullptr && inputs->is_array() &&
        targets->is_array()) {
      Check(inputs->array().size() == 4, "pack/inputs: not four rows");
      Check(targets->array().size() == 4, "pack/targets: not four rows");
      for (std::size_t row = 0; row < inputs->array().size(); ++row) {
        Check(inputs->array()[row].is_array() &&
                  inputs->array()[row].array().size() == 32,
              Format("pack/inputs row %zu is not 32 ids", row));
        Check(targets->array()[row].is_array() &&
                  targets->array()[row].array().size() == 32,
              Format("pack/targets row %zu is not 32 targets", row));
      }
    }
  }

  // The tokenizer of section 6.1: the nine chat specials, the piece table the
  // phase C.1 test builds its tokenizer from, and the byte table of D6.
  const nanochat::json::Value* tokenizer = root.Find("tokenizer");
  if (tokenizer != nullptr) {
    const nanochat::json::Value* specials = tokenizer->Find("special_tokens");
    Check(specials != nullptr && specials->is_object() &&
              specials->object().size() == 9,
          "tokenizer/special_tokens: not nine specials");
    const nanochat::json::Value* pieces = tokenizer->Find("pieces");
    Check(pieces != nullptr && pieces->is_object() && !pieces->object().empty(),
          "tokenizer/pieces: the piece table is missing or empty");
    const nanochat::json::Value* table = tokenizer->Find("token_bytes");
    Check(table != nullptr && table->is_array() && table->array().size() == 256,
          "tokenizer/token_bytes: not 256 entries");
    const nanochat::json::Value* name = tokenizer->Find("name");
    const nanochat::json::Value* digest = tokenizer->Find("sha256");
    Check(name != nullptr && name->is_string(), "tokenizer/name: missing");
    Check(digest != nullptr && digest->is_string() &&
              digest->string().size() == 64,
          "tokenizer/sha256: missing or not a hex digest");
  }

  // The mixture sections of D8: the synthetic task order, per epoch.
  for (const char* key : {"mixture", "validation_mixture"}) {
    const nanochat::json::Value* mixture = root.Find(key);
    if (mixture == nullptr) continue;
    const nanochat::json::Value* order = mixture->Find("order");
    Check(order != nullptr && order->is_array() && !order->array().empty(),
          std::string(key) + "/order: missing or empty");
    const nanochat::json::Value* tasks = mixture->Find("tasks");
    Check(tasks != nullptr && tasks->is_array() && !tasks->array().empty(),
          std::string(key) + "/tasks: missing or empty");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string step_path =
        LocateFile(argc, argv, 1, "tests/data/sft_step_fixture.bin");
    const std::string loop_path =
        LocateFile(argc, argv, 2, "tests/data/sft_loop_fixture.bin");
    const std::string shard_path =
        LocateFile(argc, argv, 3, "tests/data/sft_rows_fixture.nchsft");
    const std::string base_path =
        LocateFile(argc, argv, 4, "tests/data/sft_base_fixture.ckpt");
    const std::string pack_path =
        LocateFile(argc, argv, 5, "tests/data/sft_pack_fixture.bin");

    const Fixture step_fixture = Fixture::Load(step_path);
    CheckStepFixture(step_path);
    CheckLoopFixture(loop_path);
    CheckShard(shard_path, step_fixture);
    CheckBaseCheckpoint(base_path, step_fixture);
    CheckPackFixture(pack_path);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "sft_fixture_load: %s\n", error.what());
    return 1;
  }
  if (g_failures != 0) {
    std::fprintf(stderr, "sft_fixture_load: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf(
      "sft_fixture_load: OK (step, loop, shard, checkpoint, and pack fixtures "
      "read with no torch and no reference checkout)\n");
  return 0;
}
