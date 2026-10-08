#include "tests/numerics_trace.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/dataloader.h"
#include "nanochat/kernels.h"
#include "nanochat/model.h"
#include "nanochat/optim.h"
#include "nanochat/rand.h"
#include "nanochat/scheduler.h"
#include "nanochat/tokenizer.h"
#include "src/model_impl.h"

namespace nanochat {
namespace numerics {
namespace {

using oracle::DType;

// The reference batch of docs/numerics-integration.md section 6.2. `Config`
// carries the sequence length but not the batch, so the trace fixes it here.
constexpr int kTraceBatch = 4;

// Evaluation documents. The loader refills its buffer to `kDocumentBuffer`
// before every placement, so a source with more documents than the buffer makes
// the packing independent of producer timing.
constexpr int kDocumentBuffer = 64;
constexpr int kDocumentCount = 128;
constexpr int kDocumentLines = 40;
constexpr int kDocumentsPerFetch = 8;

// The evaluation weight rule (a mirrored conversation): the second half of each
// row is supervised, the first half is not, and every fifth target is ignored.
constexpr int kSftIgnoreEvery = 5;

// The RL reward rule: a weighted sum of the generated ids, normalized to
// [0, 1]. The rule depends on the whole sampled span, so four distinct rollouts
// give four distinct rewards.
constexpr char kRlRewardRule[] = "weighted-mean-id";

std::uint64_t HashFloats(const std::vector<float>& values) {
  return HashBytes(values.data(), values.size() * sizeof(float));
}

void AppendU16(std::vector<std::uint8_t>* out, std::uint16_t value) {
  out->push_back(static_cast<std::uint8_t>(value & 0xffu));
  out->push_back(static_cast<std::uint8_t>((value >> 8) & 0xffu));
}

void AppendU32(std::vector<std::uint8_t>* out, std::uint32_t value) {
  for (int i = 0; i < 4; ++i) {
    out->push_back(static_cast<std::uint8_t>((value >> (8 * i)) & 0xffu));
  }
}

void AppendU64(std::vector<std::uint8_t>* out, std::uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    out->push_back(static_cast<std::uint8_t>((value >> (8 * i)) & 0xffu));
  }
}

Record MakeRecord(const std::string& name, DType dtype,
                  std::vector<std::int64_t> shape,
                  std::vector<std::uint8_t> payload) {
  Record record;
  record.name = name;
  record.dtype = dtype;
  record.shape = std::move(shape);
  record.payload = std::move(payload);
  return record;
}

// Stages a (possibly device-resident) buffer to host floats.
std::vector<float> ReadFloats(const ComputeType* src, std::int64_t count) {
  std::vector<ComputeType> host(static_cast<std::size_t>(count));
  if (count > 0) {
    kernels::Memcpy(host.data(), src,
                    static_cast<std::size_t>(count) * sizeof(ComputeType),
                    CopyDir::kDeviceToHost);
  }
  std::vector<float> out(static_cast<std::size_t>(count));
  for (std::int64_t i = 0; i < count; ++i) {
    out[static_cast<std::size_t>(i)] = AsF(host[static_cast<std::size_t>(i)]);
  }
  return out;
}

// The L2 norm and the byte hash of one parameter or gradient buffer. `l2` is
// summed in double on the host in a fixed order, so it does not depend on the
// backend's own reduction order.
struct Digest {
  double l2 = 0.0;
  std::uint64_t hash = 0;
};

Digest DigestOf(const ComputeType* data, std::int64_t count) {
  const std::vector<float> values = ReadFloats(data, count);
  double sum = 0.0;
  for (float value : values) {
    sum += static_cast<double>(value) * static_cast<double>(value);
  }
  Digest digest;
  digest.l2 = std::sqrt(sum);
  digest.hash = HashFloats(values);
  return digest;
}

// The global L2 norm of every parameter gradient, summed in double on the host.
// The trace records this value instead of `Optimizer::GradNorm()` because the
// SFT and RL phases take no optimizer step, so one definition must cover every
// phase. `Model::params()` always provides a gradient buffer with a positive
// count.
double HostGradNorm(const std::vector<ParamView>& params) {
  double sum = 0.0;
  for (const ParamView& view : params) {
    const std::vector<float> values = ReadFloats(view.grad, view.count);
    for (float value : values) {
      sum += static_cast<double>(value) * static_cast<double>(value);
    }
  }
  return std::sqrt(sum);
}

// A deterministic pretraining-style batch: one row of `seq + 1` ids per row,
// where the row is the input and its one-position shift is the target. The
// trace records no batches, so this rule is the batch definition.
void FillSyntheticBatch(Rand* rand, int batch, int seq, int vocab,
                        std::vector<int>* tokens, std::vector<int>* targets) {
  const std::size_t rows =
      static_cast<std::size_t>(batch) * static_cast<std::size_t>(seq);
  tokens->assign(rows, 0);
  targets->assign(rows, 0);
  std::vector<int> row(static_cast<std::size_t>(seq) + 1, 0);
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i <= seq; ++i) {
      row[static_cast<std::size_t>(i)] = rand->NextInt(0, vocab - 1);
    }
    const std::size_t base =
        static_cast<std::size_t>(b) * static_cast<std::size_t>(seq);
    for (int i = 0; i < seq; ++i) {
      (*tokens)[base + static_cast<std::size_t>(i)] =
          row[static_cast<std::size_t>(i)];
      (*targets)[base + static_cast<std::size_t>(i)] =
          row[static_cast<std::size_t>(i) + 1];
    }
  }
}

std::vector<int> SyntheticPrompt(Rand* rand, int length, int vocab) {
  std::vector<int> prompt(static_cast<std::size_t>(length));
  for (int i = 0; i < length; ++i) {
    prompt[static_cast<std::size_t>(i)] = rand->NextInt(0, vocab - 1);
  }
  return prompt;
}

// A finite, fixed document stream for the evaluation phase.
class FixedDocumentSource final : public DocumentSource {
 public:
  bool Next(std::vector<std::string>* documents, std::string* error) override {
    (void)error;
    documents->clear();
    for (int i = 0; i < kDocumentsPerFetch && index_ < kDocumentCount; ++i) {
      documents->push_back(Document(index_));
      ++index_;
    }
    return !documents->empty();
  }

 private:
  static std::string Document(int index) {
    std::string text = "document " + std::to_string(index) + "\n";
    for (int line = 0; line < kDocumentLines; ++line) {
      text += "line ";
      text += std::to_string((index * 7 + line * 13) % 97);
      text += "\n";
    }
    return text;
  }

  int index_ = 0;
};

DocumentSourceFactory FixedFactory() {
  return [](std::string* error) -> std::unique_ptr<DocumentSource> {
    (void)error;
    return std::make_unique<FixedDocumentSource>();
  };
}

OptimizerConfig TraceOptimizerConfig() {
  // The reference recipe (tools/dump_api_fixture.py) with clipping disabled, so
  // the trace records unclipped gradients. `adam_eps = 1e-4` matches the
  // committed fixtures: several parameters have a true gradient of exactly zero
  // at this initialization (docs/testing.md).
  OptimizerConfig opt;
  opt.unembedding_lr = 0.004f;
  opt.embedding_lr = 0.2f;
  opt.matrix_lr = 0.02f;
  opt.scalar_lr = 0.5f;
  opt.weight_decay = 0.0f;
  opt.muon_ns_steps = 5;
  opt.muon_beta2 = 0.9f;
  opt.adam_eps = 1e-4f;
  opt.clip = 0.0f;
  return opt;
}

SchedulerConfig TraceSchedulerConfig() {
  // A flat schedule over the five steps: constant learning rate, constant Muon
  // momentum, and no weight decay, so the trace isolates the graph arithmetic
  // from the schedule.
  SchedulerConfig sched;
  sched.num_iterations = 5;
  sched.warmup_steps = 0;
  sched.warmdown_ratio = 0.0f;
  sched.final_lr_frac = 1.0f;
  sched.weight_decay_base = 0.0f;
  sched.muon_momentum_warmup_steps = 0.0f;
  sched.muon_momentum_start = 0.9f;
  sched.muon_momentum_peak = 0.9f;
  sched.muon_momentum_final = 0.9f;
  return sched;
}

// The top-1 minus top-2 gap of one logits row, and its argmax.
struct Margin {
  float value = 0.0f;
  int argmax = -1;
};

Margin MarginOf(const float* row, int vocab) {
  Margin margin;
  float best = row[0];
  float second = -std::numeric_limits<float>::infinity();
  int best_index = 0;
  for (int j = 1; j < vocab; ++j) {
    const float value = row[j];
    if (value > best) {
      second = best;
      best = value;
      best_index = j;
    } else if (value > second) {
      second = value;
    }
  }
  margin.value = best - second;
  margin.argmax = best_index;
  return margin;
}

// --- comparison ----------------------------------------------------------

struct GroupSpec {
  const char* name;
  double rtol;
  double atol;
  bool (*match)(const std::string& name);
};

bool MatchTrainLoss(const std::string& name) {
  return name.rfind("train/", 0) == 0 && name.size() > 6 &&
         name.compare(name.size() - 5, 5, "/loss") == 0;
}

bool MatchTrainGradNorm(const std::string& name) {
  return name.rfind("train/", 0) == 0 && name.size() > 10 &&
         name.compare(name.size() - 10, 10, "/grad_norm") == 0;
}

bool MatchParamL2(const std::string& name) {
  return name.find("/param_l2/") != std::string::npos;
}

bool MatchEvalBpb(const std::string& name) { return name == "eval/bpb"; }

bool MatchEvalLoss(const std::string& name) {
  return name.rfind("eval/loss/", 0) == 0;
}

bool MatchEvalMargin(const std::string& name) {
  return name == "eval/margin_min";
}

bool MatchGenMargin(const std::string& name) {
  return name == "gen/margin_min" || name.rfind("gen/margin/", 0) == 0;
}

bool MatchSftLoss(const std::string& name) { return name == "sft/loss"; }

bool MatchSftGradNorm(const std::string& name) {
  return name == "sft/grad_norm";
}

bool MatchRlLoss(const std::string& name) { return name == "rl/loss"; }

bool MatchRlGradNorm(const std::string& name) { return name == "rl/grad_norm"; }

bool MatchRlReward(const std::string& name) {
  return name.rfind("rl/reward/", 0) == 0 || name == "rl/mean_reward" ||
         name.rfind("rl/advantage/", 0) == 0;
}

bool MatchRlMargin(const std::string& name) { return name == "rl/margin_min"; }

// The last entry catches every remaining float record (the configuration and
// schedule scalars), which the legs compare within the scalar tolerance.
bool MatchAny(const std::string& name) {
  (void)name;
  return true;
}

const GroupSpec kGroups[] = {
    {"train_loss", kLossRtol, 1e-9, &MatchTrainLoss},
    {"train_grad_norm", kLossRtol, 1e-9, &MatchTrainGradNorm},
    {"param_l2", kParamL2Rtol, 1e-12, &MatchParamL2},
    {"eval_bpb", kLossRtol, 1e-9, &MatchEvalBpb},
    {"eval_loss", kLossRtol, 1e-9, &MatchEvalLoss},
    {"eval_margin", 0.0, kMarginAtol, &MatchEvalMargin},
    {"gen_margin", 0.0, kMarginAtol, &MatchGenMargin},
    {"sft_loss", kLossRtol, 1e-9, &MatchSftLoss},
    {"sft_grad_norm", kLossRtol, 1e-9, &MatchSftGradNorm},
    {"rl_loss", kLossRtol, 1e-9, &MatchRlLoss},
    {"rl_grad_norm", kLossRtol, 1e-9, &MatchRlGradNorm},
    {"rl_reward", kLossRtol, 1e-9, &MatchRlReward},
    {"rl_margin", 0.0, kMarginAtol, &MatchRlMargin},
    {"scalar", kLossRtol, 1e-9, &MatchAny},
};

constexpr int kGroupCount =
    static_cast<int>(sizeof(kGroups) / sizeof(kGroups[0]));

// The index of the comparison group of one float record. The last group matches
// every name, so the loop always returns.
int GroupOf(const std::string& name) {
  for (int i = 0; i < kGroupCount; ++i) {
    if (kGroups[i].match(name)) return i;
  }
  return kGroupCount - 1;
}

bool IsHashRecord(const std::string& name) {
  return name.find("/param_hash/") != std::string::npos ||
         name.find("/grad_hash/") != std::string::npos ||
         name == "eval/logits_hash";
}

bool IsHeaderString(const std::string& name) {
  return name.rfind("header/", 0) == 0;
}

std::int64_t PayloadCount(const Record& record) {
  std::int64_t count = 1;
  for (std::int64_t extent : record.shape) count *= extent;
  return count;
}

void Report(const std::string& label, const std::string& message) {
  std::fprintf(stderr, "FAIL: %s: %s\n", label.c_str(), message.c_str());
}

// Checks that the computed trace and the golden share one schema, so a
// name-driven comparison is meaningful. The header identity strings differ by
// design on the GPU leg ("cpu" against "cuda", and the build name with them),
// so a name-driven comparison checks only their dtype.
bool SameSchema(const std::vector<Record>& got, const oracle::Fixture& golden,
                const std::string& label, bool exact) {
  if (got.size() != golden.size()) {
    Report(label, "record count " + std::to_string(got.size()) +
                      " differs from the golden count " +
                      std::to_string(golden.size()));
    return false;
  }
  const std::vector<std::string>& names = golden.names();
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (got[i].name != names[i]) {
      Report(label, "record " + std::to_string(i) + " is '" + got[i].name +
                        "', the golden record is '" + names[i] + "'");
      return false;
    }
    const oracle::Tensor& want = golden.Get(names[i]);
    if (got[i].dtype != want.dtype) {
      Report(label, "record '" + names[i] + "' has a different dtype");
      return false;
    }
    if (!exact && IsHeaderString(names[i])) continue;
    if (got[i].shape != want.shape) {
      Report(label, "record '" + names[i] + "' has a different shape");
      return false;
    }
  }
  return true;
}

}  // namespace

std::string BackendName() {
#if defined(NANOCHAT_BACKEND_CUDA)
  return "cuda";
#else
  return "cpu";
#endif
}

std::string PrecisionName() {
#if defined(NANOCHAT_PRECISION_FP16)
  return "fp16";
#else
  return "fp32";
#endif
}

std::string BuildName() { return BackendName() + "-" + PrecisionName(); }

Record Fp32Record(const std::string& name, float value) {
  std::vector<std::uint8_t> payload(4);
  std::memcpy(payload.data(), &value, 4);
  return MakeRecord(name, DType::kFp32, {1}, std::move(payload));
}

Record Fp32VectorRecord(const std::string& name, const float* values,
                        std::int64_t count) {
  std::vector<std::uint8_t> payload(static_cast<std::size_t>(count) *
                                    sizeof(float));
  if (count > 0) {
    std::memcpy(payload.data(), values,
                static_cast<std::size_t>(count) * sizeof(float));
  }
  return MakeRecord(name, DType::kFp32, {count}, std::move(payload));
}

Record Int32Record(const std::string& name, std::int32_t value) {
  std::vector<std::uint8_t> payload(4);
  std::memcpy(payload.data(), &value, 4);
  return MakeRecord(name, DType::kInt32, {1}, std::move(payload));
}

Record Int32VectorRecord(const std::string& name, const int* values,
                         std::int64_t count) {
  std::vector<std::int32_t> narrow(static_cast<std::size_t>(count));
  for (std::int64_t i = 0; i < count; ++i) {
    narrow[static_cast<std::size_t>(i)] =
        static_cast<std::int32_t>(values[static_cast<std::size_t>(i)]);
  }
  std::vector<std::uint8_t> payload(static_cast<std::size_t>(count) *
                                    sizeof(std::int32_t));
  if (count > 0) {
    std::memcpy(payload.data(), narrow.data(),
                static_cast<std::size_t>(count) * sizeof(std::int32_t));
  }
  return MakeRecord(name, DType::kInt32, {count}, std::move(payload));
}

Record Int64Record(const std::string& name, std::int64_t value) {
  std::vector<std::uint8_t> payload(8);
  std::memcpy(payload.data(), &value, 8);
  return MakeRecord(name, DType::kInt64, {1}, std::move(payload));
}

Record BytesRecord(const std::string& name, const std::string& text) {
  std::vector<std::uint8_t> payload(text.begin(), text.end());
  const std::int64_t count = static_cast<std::int64_t>(payload.size());
  return MakeRecord(name, DType::kUInt8, {count}, std::move(payload));
}

std::uint64_t HashBytes(const void* data, std::size_t bytes) {
  // FNV-1a, 64 bit.
  const auto* bytes_in = static_cast<const std::uint8_t*>(data);
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  for (std::size_t i = 0; i < bytes; ++i) {
    hash ^= bytes_in[i];
    hash *= 0x100000001b3ULL;
  }
  return hash;
}

std::vector<std::uint8_t> SerializeTrace(const std::vector<Record>& records) {
  std::vector<std::uint8_t> out;
  const char* magic = kTraceMagic;
  for (int i = 0; i < 8; ++i) {
    out.push_back(static_cast<std::uint8_t>(magic[i]));
  }
  AppendU32(&out, 1);
  AppendU32(&out, static_cast<std::uint32_t>(records.size()));
  for (const Record& record : records) {
    AppendU16(&out, static_cast<std::uint16_t>(record.name.size()));
    out.insert(out.end(), record.name.begin(), record.name.end());
    out.push_back(static_cast<std::uint8_t>(record.dtype));
    out.push_back(static_cast<std::uint8_t>(record.shape.size()));
    for (std::int64_t extent : record.shape) {
      AppendU64(&out, static_cast<std::uint64_t>(extent));
    }
    out.insert(out.end(), record.payload.begin(), record.payload.end());
  }
  return out;
}

Config ReferenceConfig() {
  Config config;
  config.num_layers = 10;
  config.num_heads = 6;
  config.num_kv_heads = 3;
  config.hidden_dim = 384;
  config.seq_len = 256;
  config.vocab_size = 512;
  config.padded_vocab_size = 512;
  config.rope_base = 100000.0f;
  config.window_pattern = "SSSL";
  config.value_embedding = true;
  return config;
}

const TraceShape& DefaultShape() {
  static const TraceShape shape;
  return shape;
}

void PinSingleThreaded() {
  // The CPU backend reads `NANOCHAT_NUM_THREADS` once, before its first
  // parallel region, and its kernel reductions depend on the thread count. Pin
  // it to one so the golden trace does not depend on the sandbox budget.
  setenv("NANOCHAT_NUM_THREADS", "1", 1);
}

std::vector<Record> RunTrace(const Config& config, const TraceShape& shape,
                             const std::string& tokenizer_path,
                             TraceStats* stats) {
  std::vector<Record> records;
  const OptimizerConfig opt = TraceOptimizerConfig();
  const SchedulerConfig sched = TraceSchedulerConfig();
  const Scheduler scheduler(sched);
  const int batch = kTraceBatch;
  const int seq = config.seq_len;
  const int rows = batch * seq;
  const int vocab = config.vocab_size;
  const int padded_vocab = config.padded_vocab_size;

  std::unique_ptr<Model> model = Model::Create(config);
  model->InitWeights(shape.seed);
  std::unique_ptr<Optimizer> optimizer =
      CreateOptimizer(model.get(), opt, scheduler);
  const std::vector<ParamView> params = model->params();

  // --- header ------------------------------------------------------------
  records.push_back(Int32Record("header/version", 1));
  records.push_back(BytesRecord("header/magic", kTraceMagic));
  records.push_back(BytesRecord("header/backend", BackendName()));
  records.push_back(BytesRecord("header/precision", PrecisionName()));
  records.push_back(BytesRecord("header/build", BuildName()));
  records.push_back(
      Int64Record("header/seed", static_cast<std::int64_t>(shape.seed)));
  records.push_back(Int32Record("header/steps", shape.steps));
  records.push_back(Int32Record("config/layers", config.num_layers));
  records.push_back(Int32Record("config/heads", config.num_heads));
  records.push_back(Int32Record("config/kv_heads", config.num_kv_heads));
  records.push_back(Int32Record("config/hidden", config.hidden_dim));
  records.push_back(Int32Record("config/seq", config.seq_len));
  records.push_back(Int32Record("config/batch", batch));
  records.push_back(Int32Record("config/vocab", config.vocab_size));
  records.push_back(
      Int32Record("config/padded_vocab", config.padded_vocab_size));
  records.push_back(
      BytesRecord("config/window_pattern", config.window_pattern));
  records.push_back(
      Int32Record("config/value_embedding", config.value_embedding ? 1 : 0));
  records.push_back(Fp32Record("config/rope_base", config.rope_base));
  records.push_back(Int32Record("config/parameters",
                                static_cast<std::int32_t>(params.size())));
  records.push_back(Fp32Record("opt/unembedding_lr", opt.unembedding_lr));
  records.push_back(Fp32Record("opt/embedding_lr", opt.embedding_lr));
  records.push_back(Fp32Record("opt/matrix_lr", opt.matrix_lr));
  records.push_back(Fp32Record("opt/scalar_lr", opt.scalar_lr));
  records.push_back(Fp32Record("opt/weight_decay", opt.weight_decay));
  records.push_back(Fp32Record("opt/clip", opt.clip));
  records.push_back(Fp32Record("opt/adam_eps", opt.adam_eps));
  records.push_back(Int32Record("opt/muon_ns_steps", opt.muon_ns_steps));
  records.push_back(Fp32Record("opt/muon_beta2", opt.muon_beta2));
  records.push_back(Int32Record("sched/num_iterations", sched.num_iterations));
  records.push_back(Int32Record("sched/warmup_steps", sched.warmup_steps));
  records.push_back(Fp32Record("sched/warmdown_ratio", sched.warmdown_ratio));
  records.push_back(Fp32Record("sched/final_lr_frac", sched.final_lr_frac));
  records.push_back(
      Fp32Record("sched/weight_decay_base", sched.weight_decay_base));
  records.push_back(
      Fp32Record("sched/muon_momentum_start", sched.muon_momentum_start));
  records.push_back(
      Fp32Record("sched/muon_momentum_peak", sched.muon_momentum_peak));
  records.push_back(
      Fp32Record("sched/muon_momentum_final", sched.muon_momentum_final));
  records.push_back(Int32Record("train/batches", shape.batches));
  records.push_back(Int32Record("eval/steps", shape.eval_steps));
  records.push_back(Int32Record("gen/prompt_len", shape.gen_prompt));
  records.push_back(Int32Record("gen/tokens", shape.gen_tokens));
  records.push_back(Int32Record("gen/num_samples", 1));
  records.push_back(Int32Record("sft/ignore_every", kSftIgnoreEvery));
  records.push_back(Int32Record("rl/num_samples", shape.rl_samples));
  records.push_back(Int32Record("rl/prompt_len", shape.rl_prompt));
  records.push_back(Int32Record("rl/tokens", shape.rl_tokens));
  records.push_back(Int32Record("rl/num_passes", shape.rl_num_passes));
  records.push_back(
      Int32Record("rl/examples_per_rank", shape.rl_examples_per_rank));
  records.push_back(BytesRecord("rl/reward_rule", kRlRewardRule));

  TraceStats local;
  TraceStats* out = stats != nullptr ? stats : &local;

  // --- train phase -------------------------------------------------------
  // Step N records the loss of the model at that point on batch N, the global
  // gradient norm of the gradients batch N produced, the L2 norm and hash of
  // every parameter, and the hash of every gradient. Step 0 is the initial
  // state, taken before the first update; batches 1..5 each drive one update.
  Rand rand(shape.seed ^ 0x9e3779b97f4a7c15ULL);
  std::vector<std::vector<int>> tokens(static_cast<std::size_t>(shape.batches));
  std::vector<std::vector<int>> targets(
      static_cast<std::size_t>(shape.batches));
  for (int b = 0; b < shape.batches; ++b) {
    FillSyntheticBatch(&rand, batch, seq, vocab,
                       &tokens[static_cast<std::size_t>(b)],
                       &targets[static_cast<std::size_t>(b)]);
  }
  for (int step = 0; step <= shape.steps; ++step) {
    if (step > 0) {
      optimizer->Step(step);
    }
    const std::vector<int>& step_tokens =
        tokens[static_cast<std::size_t>(step)];
    const std::vector<int>& step_targets =
        targets[static_cast<std::size_t>(step)];
    model->ZeroGrad();
    const float loss =
        model->ForwardLoss(step_tokens.data(), step_targets.data(), batch, seq);
    model->BackwardAccumulate(1.0f);
    const std::string prefix = "train/step" + std::to_string(step);
    records.push_back(Fp32Record(prefix + "/loss", loss));
    records.push_back(Fp32Record(prefix + "/grad_norm",
                                 static_cast<float>(HostGradNorm(params))));
    out->train_loss.push_back(static_cast<double>(loss));
    for (const ParamView& view : params) {
      const Digest value = DigestOf(view.value, view.count);
      records.push_back(Fp32Record(prefix + "/param_l2/" + view.name,
                                   static_cast<float>(value.l2)));
      records.push_back(Int64Record(prefix + "/param_hash/" + view.name,
                                    static_cast<std::int64_t>(value.hash)));
      records.push_back(Int64Record(prefix + "/grad_hash/" + view.name,
                                    static_cast<std::int64_t>(HashFloats(
                                        ReadFloats(view.grad, view.count)))));
    }
  }

  // --- evaluation phase --------------------------------------------------
  std::unique_ptr<Tokenizer> tokenizer = LoadTokenizer(tokenizer_path);
  if (tokenizer == nullptr) {
    throw std::runtime_error("cannot load the tokenizer '" + tokenizer_path +
                             "'");
  }
  DataLoader loader(FixedFactory(), tokenizer.get(), batch, seq, shape.seed, 1,
                    kDocumentBuffer);
  std::vector<std::vector<int>> eval_tokens(
      static_cast<std::size_t>(shape.eval_steps));
  for (int b = 0; b < shape.eval_steps; ++b) {
    std::vector<int> tokens_host(static_cast<std::size_t>(rows));
    std::vector<int> targets_host(static_cast<std::size_t>(rows));
    if (!loader.Next(tokens_host.data(), targets_host.data())) {
      throw std::runtime_error("the evaluation loader ran out of documents");
    }
    eval_tokens[static_cast<std::size_t>(b)] = std::move(tokens_host);
    records.push_back(Int32VectorRecord(
        "eval/tokens/" + std::to_string(b),
        eval_tokens[static_cast<std::size_t>(b)].data(), rows));
  }
  loader.Reset();
  // `EvalBpb` installs its own `NoGradGuard`
  // (docs/numerics-integration.md section 7, phase 2).
  const float bpb = EvalBpb(model.get(), &loader, shape.eval_steps);
  if (!std::isfinite(bpb)) {
    throw std::runtime_error("the evaluation produced no bits-per-byte");
  }
  out->eval_bpb = static_cast<double>(bpb);
  records.push_back(Fp32Record("eval/bpb", bpb));

  double min_eval_margin = std::numeric_limits<double>::infinity();
  for (int b = 0; b < shape.eval_steps; ++b) {
    std::vector<ScoreResult> results;
    ScoreBatch(model.get(), eval_tokens[static_cast<std::size_t>(b)].data(),
               batch, seq, nullptr, nullptr, &results);
    if (static_cast<int>(results.size()) != batch) {
      throw std::runtime_error("ScoreBatch did not score the evaluation batch");
    }
    std::vector<int> argmax(static_cast<std::size_t>(rows), -1);
    double loss_sum = 0.0;
    int loss_count = 0;
    for (int r = 0; r < batch; ++r) {
      const ScoreResult& result = results[static_cast<std::size_t>(r)];
      for (int p = 0; p < seq; ++p) {
        argmax[static_cast<std::size_t>(r) * seq + p] =
            result.argmax[static_cast<std::size_t>(p)];
        if (result.argmax[static_cast<std::size_t>(p)] < 0) continue;
        loss_sum +=
            static_cast<double>(result.nll[static_cast<std::size_t>(p)]);
        ++loss_count;
      }
    }
    records.push_back(Fp32Record("eval/loss/" + std::to_string(b),
                                 static_cast<float>(loss_sum / loss_count)));
    records.push_back(Int32VectorRecord("eval/argmax/" + std::to_string(b),
                                        argmax.data(), rows));
    // The margins and the logits hash read the same buffer ScoreBatch scores,
    // so the recorded argmax and margin describe one forward. ScoreBatch scores
    // the training graph, so grad mode stays on for the call.
    auto* impl = static_cast<TrainModel*>(model.get());
    const std::int64_t logits_count =
        static_cast<std::int64_t>(rows) * padded_vocab;
    const std::vector<float> logits =
        ReadFloats(impl->raw_logits(), logits_count);
    for (int p = 0; p < rows; ++p) {
      if (argmax[static_cast<std::size_t>(p)] < 0) continue;
      const Margin margin = MarginOf(
          logits.data() + static_cast<std::size_t>(p) * padded_vocab, vocab);
      min_eval_margin =
          std::min(min_eval_margin, static_cast<double>(margin.value));
    }
    if (b == 0) {
      records.push_back(Int64Record(
          "eval/logits_hash", static_cast<std::int64_t>(HashFloats(logits))));
    }
  }
  out->min_eval_margin = min_eval_margin;
  records.push_back(
      Fp32Record("eval/margin_min", static_cast<float>(min_eval_margin)));

  // --- greedy generation phase -------------------------------------------
  Rand gen_rand(shape.seed ^ 0x2545f4914f6cdd1dULL);
  const std::vector<int> gen_prompt =
      SyntheticPrompt(&gen_rand, shape.gen_prompt, vocab);
  GenerateParams greedy;
  greedy.num_samples = 1;
  greedy.max_tokens = shape.gen_tokens;
  greedy.temperature = 0.0f;
  greedy.top_k = 0;
  greedy.seed = shape.seed;
  greedy.stop_id = -1;
  greedy.bos_id = -1;
  std::vector<GeneratedSequence> gen_rows;
  GenerateBatch(model.get(), gen_prompt.data(), shape.gen_prompt, greedy,
                &gen_rows);
  if (gen_rows.size() != 1 || static_cast<int>(gen_rows[0].tokens.size()) !=
                                  shape.gen_prompt + shape.gen_tokens) {
    throw std::runtime_error("GenerateBatch did not return one full row");
  }
  const std::vector<int> gen_ids(gen_rows[0].tokens.begin() + shape.gen_prompt,
                                 gen_rows[0].tokens.end());
  records.push_back(
      Int32VectorRecord("gen/prompt", gen_prompt.data(), shape.gen_prompt));
  records.push_back(
      Int32VectorRecord("gen/ids", gen_ids.data(), shape.gen_tokens));
  std::vector<int> gen_mask(gen_rows[0].mask.begin() + shape.gen_prompt,
                            gen_rows[0].mask.end());
  records.push_back(
      Int32VectorRecord("gen/mask", gen_mask.data(), shape.gen_tokens));
  // The margin of each step comes from the training graph over the same row:
  // the logits at the position before the sampled token. Greedy decoding on the
  // inference path and the full-row forward agree to within the margin.
  {
    const int length = shape.gen_prompt + shape.gen_tokens;
    std::vector<int> row_tokens = gen_prompt;
    row_tokens.insert(row_tokens.end(), gen_ids.begin(), gen_ids.end());
    std::vector<int> row_targets(row_tokens.begin() + 1, row_tokens.end());
    row_targets.push_back(0);
    model->ForwardLoss(row_tokens.data(), row_targets.data(), 1, length);
    auto* impl = static_cast<TrainModel*>(model.get());
    const std::vector<float> logits = ReadFloats(
        impl->raw_logits(), static_cast<std::int64_t>(length) * padded_vocab);
    double min_margin = std::numeric_limits<double>::infinity();
    for (int i = 0; i < shape.gen_tokens; ++i) {
      const std::int64_t position = shape.gen_prompt - 1 + i;
      const Margin margin = MarginOf(
          logits.data() + static_cast<std::size_t>(position) * padded_vocab,
          vocab);
      records.push_back(
          Fp32Record("gen/margin/" + std::to_string(i), margin.value));
      if (margin.argmax != gen_ids[static_cast<std::size_t>(i)]) {
        ++out->gen_path_mismatch;
      }
      min_margin = std::min(min_margin, static_cast<double>(margin.value));
    }
    out->min_gen_margin = min_margin;
    records.push_back(
        Fp32Record("gen/margin_min", static_cast<float>(min_margin)));
  }

  // --- SFT arithmetic phase ---------------------------------------------
  // The masked weighted cross-entropy of docs/post-training.md section 2: the
  // second half of each row carries weight 1, the first half carries weight 0,
  // and every fifth target is ignored.
  {
    const std::vector<int>& sft_tokens = eval_tokens[0];
    std::vector<int> sft_targets(static_cast<std::size_t>(rows), -1);
    std::vector<float> weights(static_cast<std::size_t>(rows), 0.0f);
    int valid = 0;
    int zero_weights = 0;
    for (int r = 0; r < batch; ++r) {
      for (int p = 0; p < seq; ++p) {
        const std::size_t index =
            static_cast<std::size_t>(r) * seq + static_cast<std::size_t>(p);
        if (p + 1 < seq && (p % kSftIgnoreEvery) != 0) {
          sft_targets[index] = sft_tokens[index + 1];
          ++valid;
        }
        if (p >= seq / 2) {
          weights[index] = 1.0f;
        } else {
          ++zero_weights;
        }
      }
    }
    model->ZeroGrad();
    model->ForwardLoss(sft_tokens.data(), sft_targets.data(), batch, seq);
    const std::vector<float> losses =
        ReadFloats(static_cast<TrainModel*>(model.get())->losses(), rows);
    double weighted = 0.0;
    for (std::size_t i = 0; i < losses.size(); ++i) {
      weighted +=
          static_cast<double>(weights[i]) * static_cast<double>(losses[i]);
    }
    const double masked_loss = weighted / static_cast<double>(valid);
    model->BackwardWeighted(weights.data(), 1.0f);
    records.push_back(
        Int32Record("sft/valid_targets", static_cast<std::int32_t>(valid)));
    records.push_back(Int32Record("sft/zero_weights",
                                  static_cast<std::int32_t>(zero_weights)));
    records.push_back(Fp32Record("sft/loss", static_cast<float>(masked_loss)));
    records.push_back(
        Fp32Record("sft/grad_norm", static_cast<float>(HostGradNorm(params))));
    out->sft_loss = masked_loss;
  }

  // --- RL arithmetic phase ----------------------------------------------
  // Four greedy rollouts from four distinct prompts, the advantage as the
  // reward minus the mean reward, and the DAPO-style normalizer
  // `num_valid * num_passes * examples_per_rank`
  // (docs/post-training.md section 5.2).
  {
    Rand rl_rand(shape.seed ^ 0x5bf03635ef1b0a0dULL);
    const int rl_len = shape.rl_prompt + shape.rl_tokens;
    std::vector<std::vector<int>> rollout(
        static_cast<std::size_t>(shape.rl_samples));
    for (int r = 0; r < shape.rl_samples; ++r) {
      const std::vector<int> prompt =
          SyntheticPrompt(&rl_rand, shape.rl_prompt, vocab);
      GenerateParams params = greedy;
      params.max_tokens = shape.rl_tokens;
      std::vector<GeneratedSequence> rows;
      GenerateBatch(model.get(), prompt.data(), shape.rl_prompt, params, &rows);
      if (rows.size() != 1 ||
          static_cast<int>(rows[0].tokens.size()) != rl_len) {
        throw std::runtime_error("the RL rollout returned a short row");
      }
      rollout[static_cast<std::size_t>(r)] = rows[0].tokens;
      records.push_back(Int32VectorRecord("rl/prompt/" + std::to_string(r),
                                          prompt.data(), shape.rl_prompt));
      records.push_back(Int32VectorRecord(
          "rl/ids/" + std::to_string(r),
          rollout[static_cast<std::size_t>(r)].data() + shape.rl_prompt,
          shape.rl_tokens));
    }
    std::vector<double> rewards(static_cast<std::size_t>(shape.rl_samples),
                                0.0);
    const double weight_sum = 0.5 * static_cast<double>(shape.rl_tokens) *
                              static_cast<double>(shape.rl_tokens + 1);
    for (int r = 0; r < shape.rl_samples; ++r) {
      const std::vector<int>& row = rollout[static_cast<std::size_t>(r)];
      double weighted = 0.0;
      for (int i = 0; i < shape.rl_tokens; ++i) {
        weighted += static_cast<double>(i + 1) *
                    static_cast<double>(
                        row[static_cast<std::size_t>(shape.rl_prompt + i)]);
      }
      rewards[static_cast<std::size_t>(r)] =
          weighted / (weight_sum * static_cast<double>(vocab - 1));
    }
    double reward_sum = 0.0;
    for (double reward : rewards) reward_sum += reward;
    const double mean_reward = reward_sum / shape.rl_samples;
    std::vector<float> advantages(static_cast<std::size_t>(shape.rl_samples));
    for (int r = 0; r < shape.rl_samples; ++r) {
      advantages[static_cast<std::size_t>(r)] = static_cast<float>(
          rewards[static_cast<std::size_t>(r)] - mean_reward);
      records.push_back(
          Fp32Record("rl/reward/" + std::to_string(r),
                     static_cast<float>(rewards[static_cast<std::size_t>(r)])));
      records.push_back(Fp32Record("rl/advantage/" + std::to_string(r),
                                   advantages[static_cast<std::size_t>(r)]));
    }
    records.push_back(
        Fp32Record("rl/mean_reward", static_cast<float>(mean_reward)));

    // Tokens, targets, and row weights of the rollout batch: weight 0 on the
    // prompt and on the final position, the advantage elsewhere.
    std::vector<int> tokens_flat(static_cast<std::size_t>(shape.rl_samples) *
                                 rl_len);
    std::vector<int> targets_flat(
        static_cast<std::size_t>(shape.rl_samples) * rl_len, -1);
    std::vector<float> weights(
        static_cast<std::size_t>(shape.rl_samples) * rl_len, 0.0f);
    int valid = 0;
    for (int r = 0; r < shape.rl_samples; ++r) {
      const std::vector<int>& row = rollout[static_cast<std::size_t>(r)];
      for (int p = 0; p < rl_len; ++p) {
        const std::size_t index =
            static_cast<std::size_t>(r) * rl_len + static_cast<std::size_t>(p);
        tokens_flat[index] = row[static_cast<std::size_t>(p)];
        if (p + 1 < rl_len) {
          targets_flat[index] = row[static_cast<std::size_t>(p) + 1];
          ++valid;
          if (p >= shape.rl_prompt) {
            weights[index] = advantages[static_cast<std::size_t>(r)];
          }
        }
      }
    }
    const double passes = static_cast<double>(shape.rl_num_passes) *
                          static_cast<double>(shape.rl_examples_per_rank);
    model->ZeroGrad();
    model->ForwardLoss(tokens_flat.data(), targets_flat.data(),
                       shape.rl_samples, rl_len);
    const std::vector<float> losses =
        ReadFloats(static_cast<TrainModel*>(model.get())->losses(),
                   static_cast<std::int64_t>(shape.rl_samples) * rl_len);
    double weighted = 0.0;
    for (std::size_t i = 0; i < losses.size(); ++i) {
      weighted +=
          static_cast<double>(weights[i]) * static_cast<double>(losses[i]);
    }
    const double objective = weighted / (static_cast<double>(valid) * passes);
    model->BackwardWeighted(weights.data(), static_cast<float>(1.0 / passes));
    records.push_back(
        Int32Record("rl/valid_targets", static_cast<std::int32_t>(valid)));
    records.push_back(Fp32Record("rl/loss", static_cast<float>(objective)));
    records.push_back(
        Fp32Record("rl/grad_norm", static_cast<float>(HostGradNorm(params))));
    out->rl_loss = objective;

    // The logit margin of every sampled step, from the full-row forward over
    // the rollout. The greedy identifiers are gated exactly, so the margin must
    // exceed the tie threshold.
    std::vector<int> row_targets(tokens_flat.begin() + 1, tokens_flat.end());
    row_targets.push_back(0);
    model->ForwardLoss(tokens_flat.data(), row_targets.data(), shape.rl_samples,
                       rl_len);
    auto* impl = static_cast<TrainModel*>(model.get());
    const std::vector<float> logits = ReadFloats(
        impl->raw_logits(),
        static_cast<std::int64_t>(shape.rl_samples) * rl_len * padded_vocab);
    double min_margin = std::numeric_limits<double>::infinity();
    for (int r = 0; r < shape.rl_samples; ++r) {
      const std::vector<int>& row = rollout[static_cast<std::size_t>(r)];
      for (int i = 0; i < shape.rl_tokens; ++i) {
        const std::int64_t position = shape.rl_prompt - 1 + i;
        const std::size_t base = (static_cast<std::size_t>(r) * rl_len +
                                  static_cast<std::size_t>(position)) *
                                 static_cast<std::size_t>(padded_vocab);
        const Margin margin = MarginOf(logits.data() + base, vocab);
        if (margin.argmax != row[static_cast<std::size_t>(position + 1)]) {
          ++out->rl_path_mismatch;
        }
        min_margin = std::min(min_margin, static_cast<double>(margin.value));
      }
    }
    out->min_rl_margin = min_margin;
    records.push_back(
        Fp32Record("rl/margin_min", static_cast<float>(min_margin)));
  }

  out->records = static_cast<int>(records.size());
  out->parameters = static_cast<int>(params.size());
  return records;
}

TraceComparison CompareTraces(const std::vector<Record>& got,
                              const oracle::Fixture& golden, bool exact,
                              const std::string& label) {
  TraceComparison comparison;
  comparison.groups.resize(static_cast<std::size_t>(kGroupCount));
  for (int i = 0; i < kGroupCount; ++i) {
    comparison.groups[static_cast<std::size_t>(i)].name = kGroups[i].name;
  }
  if (!SameSchema(got, golden, label, exact)) {
    comparison.failures = 1;
    return comparison;
  }
  const std::vector<std::string>& names = golden.names();
  // Compares one record against the golden byte for byte and advances the
  // counters. `prefix` names the record class in the failure message.
  auto compare_bytes = [&](const std::string& name, const Record& record,
                           const oracle::Tensor& want, const char* prefix) {
    if (record.payload.size() != want.bytes ||
        std::memcmp(record.payload.data(), want.data, want.bytes) != 0) {
      Report(label, std::string(prefix) + "record '" + name +
                        "' differs from the golden");
      ++comparison.failures;
    }
    ++comparison.compared;
  };
  for (std::size_t i = 0; i < got.size(); ++i) {
    const Record& record = got[i];
    const oracle::Tensor& want = golden.Get(names[i]);
    if (exact) {
      compare_bytes(names[i], record, want, "");
      continue;
    }
    if (record.dtype == DType::kUInt8) {
      if (IsHeaderString(names[i])) {
        // The producer identity must match the compiled backend. The build name
        // differs by design on the GPU leg, so it is skipped.
        if (names[i] == "header/backend" || names[i] == "header/precision") {
          const std::string produced(record.payload.begin(),
                                     record.payload.end());
          const std::string expected =
              names[i] == "header/backend" ? BackendName() : PrecisionName();
          if (produced != expected) {
            Report(label, "record '" + names[i] + "' is '" + produced +
                              "', the compiled backend is '" + expected + "'");
            ++comparison.failures;
          }
          ++comparison.compared;
          continue;
        }
        ++comparison.skipped;
        continue;
      }
      compare_bytes(names[i], record, want, "");
      continue;
    }
    if (record.dtype == DType::kInt32 || record.dtype == DType::kInt64) {
      if (IsHashRecord(names[i])) {
        // The bytes differ by design on the GPU leg; the hash is meaningless
        // there.
        ++comparison.skipped;
        continue;
      }
      compare_bytes(names[i], record, want, "integer ");
      continue;
    }
    const int group = GroupOf(names[i]);
    for (std::int64_t e = 0; e < PayloadCount(record); ++e) {
      const float value =
          reinterpret_cast<const float*>(record.payload.data())[e];
      const float expected = want.f32()[e];
      const double diff =
          std::fabs(static_cast<double>(value) - static_cast<double>(expected));
      GroupStat& stat = comparison.groups[static_cast<std::size_t>(group)];
      stat.max_abs_diff = std::max(stat.max_abs_diff, diff);
      const double scale =
          std::max(std::fabs(static_cast<double>(expected)), 1e-12);
      stat.max_rel_diff = std::max(stat.max_rel_diff, diff / scale);
      const double limit = kGroups[group].atol + kGroups[group].rtol * scale;
      if (diff > limit) {
        Report(label, "record '" + names[i] + "'[" + std::to_string(e) +
                          "] is " + std::to_string(value) + ", the golden is " +
                          std::to_string(expected));
        ++comparison.failures;
      }
      ++stat.count;
    }
    ++comparison.compared;
  }
  return comparison;
}

}  // namespace numerics
}  // namespace nanochat
