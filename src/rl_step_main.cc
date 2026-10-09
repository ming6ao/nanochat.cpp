// `rl_step` -- the file-driven reinforcement-learning step.
//
// It is the RL analog of the training-parity harness (docs/rl-notebook.md
// section 4, docs/training-seam.md section 6.4, docs/post-training.md section
// 12). It reads a fixture that holds the model configuration, the optimizer and
// schedule configuration, the initial parameters, and, per step, a recorded
// rollout (tokens and targets with `-1` at every ignored position) and the
// per-position advantage weights. It runs the *shared* RL step
// (`nanochat::RlStep`, the same code the C ABI and the persistent worker call),
// and reports the loss, the global gradient norm, and the L2 norm of every
// parameter after every step. It writes those as a result fixture and prints a
// summary.
//
// The step is shared, not copied: the divisor `num_valid * num_passes *
// examples_per_rank` is computed in C++ by `RlStep` (docs/training-seam.md
// section 5.7), so this binary never computes it.
//
// The fixture wire format is the little-endian `NANOORC1` container of
// `tests/oracle_fixture.h`; the fixture is produced by
// `tools/dump_rl_fixture.py`. The result file carries only `result/` records,
// so a reader can parse it on its own.
//
//   tools/nanochat run t0-cpu -- <rl_step> --fixture in.bin --out out.bin

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/model.h"
#include "nanochat/optim.h"
#include "nanochat/sandbox.h"
#include "nanochat/scheduler.h"
#include "src/rl.h"

namespace {

// --- Fixture container (mirrors `src/score_main.cc`) -----------------------

constexpr char kMagic[8] = {'N', 'A', 'N', 'O', 'O', 'R', 'C', '1'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint8_t kFp32 = 0;
constexpr std::uint8_t kInt32 = 1;
constexpr std::uint8_t kFp64 = 2;
constexpr std::uint8_t kInt64 = 3;
constexpr std::uint8_t kUInt8 = 4;

std::size_t ElemSize(std::uint8_t dtype) {
  switch (dtype) {
    case kFp32:
    case kInt32:
      return 4;
    case kFp64:
    case kInt64:
      return 8;
    case kUInt8:
      return 1;
    default:
      return 0;
  }
}

struct Record {
  std::string name;
  std::uint8_t dtype = kFp32;
  std::vector<std::uint64_t> shape;
  std::vector<std::uint8_t> data;

  std::int64_t numel() const {
    std::int64_t total = 1;
    for (std::uint64_t extent : shape) {
      total *= static_cast<std::int64_t>(extent);
    }
    return total;
  }
};

// Little-endian byte cursor over an in-memory file image.
struct Reader {
  const std::vector<std::uint8_t>* bytes = nullptr;
  std::size_t offset = 0;
  bool ok = true;

  bool Need(std::size_t count) {
    if (!ok) return false;
    if (offset + count > bytes->size()) {
      ok = false;
      return false;
    }
    return true;
  }

  template <typename T>
  T Read() {
    T value{};
    if (!Need(sizeof(T))) return value;
    std::memcpy(&value, bytes->data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
  }
};

void AppendU16(std::vector<std::uint8_t>* out, std::uint16_t value) {
  for (int shift = 0; shift < 16; shift += 8) {
    out->push_back(static_cast<std::uint8_t>((value >> shift) & 0xff));
  }
}

void AppendU32(std::vector<std::uint8_t>* out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    out->push_back(static_cast<std::uint8_t>((value >> shift) & 0xff));
  }
}

void AppendU64(std::vector<std::uint8_t>* out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) {
    out->push_back(static_cast<std::uint8_t>((value >> shift) & 0xff));
  }
}

bool LoadRecords(const std::string& path, std::vector<Record>* records,
                 std::string* error) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    *error = "cannot open fixture " + path;
    return false;
  }
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(file)),
                                  std::istreambuf_iterator<char>());
  if (bytes.size() < 16 ||
      std::memcmp(bytes.data(), kMagic, sizeof(kMagic)) != 0) {
    *error = "bad fixture magic in " + path;
    return false;
  }
  Reader reader;
  reader.bytes = &bytes;
  reader.offset = 8;
  const std::uint32_t version = reader.Read<std::uint32_t>();
  const std::uint32_t count = reader.Read<std::uint32_t>();
  if (version != kVersion) {
    *error = "unsupported fixture version";
    return false;
  }
  records->clear();
  for (std::uint32_t i = 0; i < count && reader.ok; ++i) {
    Record record;
    const std::uint16_t name_len = reader.Read<std::uint16_t>();
    if (!reader.Need(name_len)) break;
    record.name.assign(
        reinterpret_cast<const char*>(bytes.data() + reader.offset), name_len);
    reader.offset += name_len;
    record.dtype = reader.Read<std::uint8_t>();
    const std::uint8_t ndim = reader.Read<std::uint8_t>();
    const std::size_t elem = ElemSize(record.dtype);
    if (elem == 0 || ndim > 16) {
      *error = "malformed record " + record.name;
      return false;
    }
    record.shape.reserve(ndim);
    for (std::uint8_t d = 0; d < ndim; ++d) {
      record.shape.push_back(reader.Read<std::uint64_t>());
    }
    const std::int64_t numel = record.numel();
    if (numel < 0) {
      *error = "negative record extent";
      return false;
    }
    const std::size_t payload = static_cast<std::size_t>(numel) * elem;
    if (!reader.Need(payload)) break;
    record.data.assign(bytes.begin() + reader.offset,
                       bytes.begin() + reader.offset + payload);
    reader.offset += payload;
    records->push_back(std::move(record));
  }
  if (!reader.ok) {
    *error = "truncated fixture " + path;
    return false;
  }
  return true;
}

bool SaveRecords(const std::string& path, const std::vector<Record>& records,
                 std::string* error) {
  std::vector<std::uint8_t> bytes;
  bytes.insert(bytes.end(), kMagic, kMagic + sizeof(kMagic));
  AppendU32(&bytes, kVersion);
  AppendU32(&bytes, static_cast<std::uint32_t>(records.size()));
  for (const Record& record : records) {
    if (record.name.size() > 0xffff || record.shape.size() > 16) {
      *error = "record too large to serialise";
      return false;
    }
    AppendU16(&bytes, static_cast<std::uint16_t>(record.name.size()));
    bytes.insert(bytes.end(), record.name.begin(), record.name.end());
    bytes.push_back(record.dtype);
    bytes.push_back(static_cast<std::uint8_t>(record.shape.size()));
    for (std::uint64_t extent : record.shape) AppendU64(&bytes, extent);
    bytes.insert(bytes.end(), record.data.begin(), record.data.end());
  }
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) {
    *error = "cannot write fixture " + path;
    return false;
  }
  file.write(reinterpret_cast<const char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!file) {
    *error = "write failed for " + path;
    return false;
  }
  return true;
}

Record MakeRecord(const std::string& name, std::uint8_t dtype,
                  std::vector<std::uint64_t> shape,
                  const std::vector<std::uint8_t>& payload) {
  Record record;
  record.name = name;
  record.dtype = dtype;
  record.shape = std::move(shape);
  record.data = payload;
  return record;
}

Record MakeInt32(const std::string& name, std::vector<std::uint64_t> shape,
                 std::vector<std::int32_t> values) {
  std::vector<std::uint8_t> payload;
  payload.reserve(values.size() * sizeof(std::int32_t));
  for (std::int32_t value : values) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value);
    for (int shift = 0; shift < 32; shift += 8) {
      payload.push_back(static_cast<std::uint8_t>((bits >> shift) & 0xff));
    }
  }
  return MakeRecord(name, kInt32, std::move(shape), payload);
}

Record MakeFloat32(const std::string& name, std::vector<std::uint64_t> shape,
                   const std::vector<float>& values) {
  std::vector<std::uint8_t> payload;
  payload.reserve(values.size() * sizeof(float));
  for (float value : values) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    for (int shift = 0; shift < 32; shift += 8) {
      payload.push_back(static_cast<std::uint8_t>((bits >> shift) & 0xff));
    }
  }
  return MakeRecord(name, kFp32, std::move(shape), payload);
}

const Record* FindRecord(const std::vector<Record>& records,
                         const std::string& name) {
  for (const Record& record : records) {
    if (record.name == name) return &record;
  }
  return nullptr;
}

std::int32_t IntAt(const Record& record, std::size_t index) {
  std::int32_t value = 0;
  std::memcpy(&value, record.data.data() + index * sizeof(std::int32_t),
              sizeof(std::int32_t));
  return value;
}

float FloatAt(const Record& record, std::size_t index) {
  float value = 0.0f;
  std::memcpy(&value, record.data.data() + index * sizeof(float),
              sizeof(float));
  return value;
}

// Reads a required scalar.
bool ScalarInt(const std::vector<Record>& records, const std::string& name,
               std::int32_t* out, std::string* error) {
  const Record* record = FindRecord(records, name);
  if (record == nullptr || record->dtype != kInt32 || record->numel() != 1) {
    *error = "missing or malformed int record '" + name + "'";
    return false;
  }
  *out = IntAt(*record, 0);
  return true;
}

bool ScalarFloat(const std::vector<Record>& records, const std::string& name,
                 float* out, std::string* error) {
  const Record* record = FindRecord(records, name);
  if (record == nullptr || record->dtype != kFp32 || record->numel() != 1) {
    *error = "missing or malformed float record '" + name + "'";
    return false;
  }
  *out = FloatAt(*record, 0);
  return true;
}

bool ReadInt32Vector(const std::vector<Record>& records,
                     const std::string& name, std::int64_t expected,
                     std::vector<std::int32_t>* out, std::string* error) {
  const Record* record = FindRecord(records, name);
  if (record == nullptr || record->dtype != kInt32 ||
      record->numel() != expected) {
    *error = "record '" + name + "' has the wrong dtype or shape";
    return false;
  }
  out->resize(static_cast<std::size_t>(expected));
  for (std::int64_t i = 0; i < expected; ++i) {
    (*out)[static_cast<std::size_t>(i)] =
        IntAt(*record, static_cast<std::size_t>(i));
  }
  return true;
}

bool ReadFloat32Vector(const std::vector<Record>& records,
                       const std::string& name, std::int64_t expected,
                       std::vector<float>* out, std::string* error) {
  const Record* record = FindRecord(records, name);
  if (record == nullptr || record->dtype != kFp32 ||
      record->numel() != expected) {
    *error = "record '" + name + "' has the wrong dtype or shape";
    return false;
  }
  out->resize(static_cast<std::size_t>(expected));
  for (std::int64_t i = 0; i < expected; ++i) {
    (*out)[static_cast<std::size_t>(i)] =
        FloatAt(*record, static_cast<std::size_t>(i));
  }
  return true;
}

// --- Model plumbing --------------------------------------------------------

using nanochat::ComputeType;
using nanochat::Config;
using nanochat::CopyDir;
using nanochat::Model;
using nanochat::Optimizer;
using nanochat::OptimizerConfig;
using nanochat::ParamView;
using nanochat::Scheduler;
using nanochat::SchedulerConfig;

template <typename T = ComputeType>
float AsFloat32(T value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else {
    return nanochat::Fp16ToFloat(value);
  }
}

template <typename T = ComputeType>
void StoreFloat(T* dst, float value) {
  if constexpr (std::is_same_v<T, float>) {
    *dst = value;
  } else {
    *dst = nanochat::Fp16FromFloat(value);
  }
}

// Copies a (possibly device-resident) buffer to host floats.
std::vector<float> ReadCompute(const ComputeType* src, std::int64_t count) {
  std::vector<ComputeType> host(static_cast<std::size_t>(count));
  if (count > 0) {
    nanochat::kernels::Memcpy(
        host.data(), src, static_cast<std::size_t>(count) * sizeof(ComputeType),
        CopyDir::kDeviceToHost);
  }
  std::vector<float> out(static_cast<std::size_t>(count));
  for (std::int64_t i = 0; i < count; ++i) {
    out[static_cast<std::size_t>(i)] =
        AsFloat32(host[static_cast<std::size_t>(i)]);
  }
  return out;
}

double L2Norm(const ComputeType* data, std::int64_t count) {
  const std::vector<float> host = ReadCompute(data, count);
  double sum = 0.0;
  for (float value : host) sum += static_cast<double>(value) * value;
  return std::sqrt(sum);
}

Config ConfigFromFixture(const std::vector<Record>& records,
                         std::string* error) {
  Config config;
  std::int32_t value = 0;
  auto int_field = [&](const char* name, int* out) -> bool {
    if (!ScalarInt(records, name, &value, error)) return false;
    *out = static_cast<int>(value);
    return true;
  };
  if (!int_field("config/layers", &config.num_layers) ||
      !int_field("config/heads", &config.num_heads) ||
      !int_field("config/kv_heads", &config.num_kv_heads) ||
      !int_field("config/embd", &config.hidden_dim) ||
      !int_field("config/seq", &config.seq_len) ||
      !int_field("config/vocab", &config.vocab_size) ||
      !int_field("config/padded_vocab", &config.padded_vocab_size)) {
    return config;
  }
  config.rope_base = 100000.0f;
  const Record* pattern = FindRecord(records, "config/window_pattern");
  if (pattern != nullptr) {
    config.window_pattern =
        std::string(pattern->data.begin(), pattern->data.end());
  }
  return config;
}

OptimizerConfig OptimizerFromFixture(const std::vector<Record>& records,
                                     std::string* error) {
  OptimizerConfig opt;
  if (!ScalarFloat(records, "config/opt/unembedding_lr", &opt.unembedding_lr,
                   error) ||
      !ScalarFloat(records, "config/opt/embedding_lr", &opt.embedding_lr,
                   error) ||
      !ScalarFloat(records, "config/opt/matrix_lr", &opt.matrix_lr, error) ||
      !ScalarFloat(records, "config/opt/scalar_lr", &opt.scalar_lr, error) ||
      !ScalarFloat(records, "config/opt/weight_decay", &opt.weight_decay,
                   error) ||
      !ScalarFloat(records, "config/opt/clip", &opt.clip, error) ||
      !ScalarFloat(records, "config/opt/adam_eps", &opt.adam_eps, error) ||
      !ScalarFloat(records, "config/opt/muon_beta2", &opt.muon_beta2, error)) {
    return opt;
  }
  std::int32_t ns_steps = 0;
  if (!ScalarInt(records, "config/opt/muon_ns_steps", &ns_steps, error)) {
    return opt;
  }
  opt.muon_ns_steps = static_cast<int>(ns_steps);
  return opt;
}

SchedulerConfig SchedulerFromFixture(const std::vector<Record>& records,
                                     std::string* error) {
  SchedulerConfig sched;
  std::int32_t int_value = 0;
  if (!ScalarInt(records, "config/sched/num_iterations", &int_value, error)) {
    return sched;
  }
  sched.num_iterations = static_cast<int>(int_value);
  if (!ScalarInt(records, "config/sched/warmup_steps", &int_value, error)) {
    return sched;
  }
  sched.warmup_steps = static_cast<int>(int_value);
  if (!ScalarFloat(records, "config/sched/warmdown_ratio",
                   &sched.warmdown_ratio, error) ||
      !ScalarFloat(records, "config/sched/final_lr_frac", &sched.final_lr_frac,
                   error) ||
      !ScalarFloat(records, "config/sched/weight_decay_base",
                   &sched.weight_decay_base, error) ||
      !ScalarFloat(records, "config/sched/muon_momentum_warmup_steps",
                   &sched.muon_momentum_warmup_steps, error) ||
      !ScalarFloat(records, "config/sched/muon_momentum_start",
                   &sched.muon_momentum_start, error) ||
      !ScalarFloat(records, "config/sched/muon_momentum_peak",
                   &sched.muon_momentum_peak, error) ||
      !ScalarFloat(records, "config/sched/muon_momentum_final",
                   &sched.muon_momentum_final, error)) {
    return sched;
  }
  return sched;
}

// Loads `param/<name>` into every model parameter. The names match PyTorch's
// `named_parameters()` (proven by the oracle test), so the mapping is direct.
bool LoadParameters(const std::vector<Record>& records, Model* model,
                    std::string* error) {
  for (const ParamView& view : model->params()) {
    const std::string name = std::string("param/") + view.name;
    const Record* source = FindRecord(records, name);
    if (source == nullptr || source->dtype != kFp32 ||
        source->numel() != view.count) {
      *error = "missing or malformed parameter record '" + name + "'";
      return false;
    }
    std::vector<ComputeType> host(static_cast<std::size_t>(view.count));
    for (std::int64_t i = 0; i < view.count; ++i) {
      StoreFloat(host.data() + i,
                 FloatAt(*source, static_cast<std::size_t>(i)));
    }
    nanochat::kernels::Memcpy(
        view.value, host.data(),
        static_cast<std::size_t>(view.count) * sizeof(ComputeType),
        CopyDir::kHostToDevice);
  }
  return true;
}

void Usage() {
  std::fprintf(
      stderr,
      "usage: rl_step --fixture PATH --out PATH\n"
      "  --fixture PATH   input RL fixture with parameters and "
      "rollouts\n"
      "  --out PATH       output fixture with the reported trajectory\n");
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("rl_step");

  std::string fixture_path;
  std::string out_path;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "-h" || flag == "--help") {
      Usage();
      return 0;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "rl_step: %s needs a value\n", flag.c_str());
      return 2;
    }
    const char* value = argv[++i];
    if (flag == "--fixture" || flag == "--in" || flag == "--input") {
      fixture_path = value;
    } else if (flag == "--out" || flag == "--output") {
      out_path = value;
    } else {
      std::fprintf(stderr, "rl_step: unknown flag %s\n", flag.c_str());
      Usage();
      return 2;
    }
  }
  if (fixture_path.empty() || out_path.empty()) {
    std::fprintf(stderr, "rl_step: --fixture and --out are required\n");
    Usage();
    return 2;
  }

  std::vector<Record> records;
  std::string error;
  if (!LoadRecords(fixture_path, &records, &error)) {
    std::fprintf(stderr, "rl_step: %s\n", error.c_str());
    return 1;
  }

  std::int32_t version = 0;
  if (!ScalarInt(records, "config/version", &version, &error) || version != 1) {
    std::fprintf(stderr, "rl_step: unsupported fixture version\n");
    return 1;
  }

  const Config config = ConfigFromFixture(records, &error);
  if (!error.empty()) {
    std::fprintf(stderr, "rl_step: %s\n", error.c_str());
    return 1;
  }
  const OptimizerConfig opt = OptimizerFromFixture(records, &error);
  if (!error.empty()) {
    std::fprintf(stderr, "rl_step: %s\n", error.c_str());
    return 1;
  }
  const SchedulerConfig sched = SchedulerFromFixture(records, &error);
  if (!error.empty()) {
    std::fprintf(stderr, "rl_step: %s\n", error.c_str());
    return 1;
  }
  const Scheduler scheduler(sched);

  std::int32_t batch = 0;
  std::int32_t seq = 0;
  std::int32_t steps = 0;
  std::int32_t num_passes = 0;
  std::int32_t examples_per_rank = 0;
  if (!ScalarInt(records, "config/batch", &batch, &error) ||
      !ScalarInt(records, "config/seq", &seq, &error) ||
      !ScalarInt(records, "config/steps", &steps, &error) ||
      !ScalarInt(records, "config/num_passes", &num_passes, &error) ||
      !ScalarInt(records, "config/examples_per_rank", &examples_per_rank,
                 &error)) {
    std::fprintf(stderr, "rl_step: %s\n", error.c_str());
    return 1;
  }
  if (batch <= 0 || seq <= 0 || steps <= 0 || num_passes <= 0 ||
      examples_per_rank <= 0) {
    std::fprintf(stderr, "rl_step: the fixture has a nonpositive dimension\n");
    return 1;
  }

  std::unique_ptr<Model> model = Model::Create(config);
  if (model == nullptr) {
    std::fprintf(stderr, "rl_step: cannot create the model\n");
    return 1;
  }
  if (!LoadParameters(records, model.get(), &error)) {
    std::fprintf(stderr, "rl_step: %s\n", error.c_str());
    return 1;
  }
  std::unique_ptr<Optimizer> optimizer =
      nanochat::CreateOptimizer(model.get(), opt, scheduler);
  if (optimizer == nullptr) {
    std::fprintf(stderr, "rl_step: cannot create the optimizer\n");
    return 1;
  }

  const std::int64_t rows =
      static_cast<std::int64_t>(batch) * static_cast<std::int64_t>(seq);
  std::vector<Record> output;
  output.push_back(
      MakeInt32("result/version", {1}, {static_cast<std::int32_t>(kVersion)}));
  output.push_back(MakeInt32("result/batch", {1}, {batch}));
  output.push_back(MakeInt32("result/seq", {1}, {seq}));
  output.push_back(MakeInt32("result/steps", {1}, {steps}));

  for (int step = 1; step <= steps; ++step) {
    const std::string prefix = "rollout/" + std::to_string(step - 1) + "/";
    std::vector<std::int32_t> tokens_i32;
    std::vector<std::int32_t> targets_i32;
    std::vector<float> advantages;
    if (!ReadInt32Vector(records, prefix + "tokens", rows, &tokens_i32,
                         &error) ||
        !ReadInt32Vector(records, prefix + "targets", rows, &targets_i32,
                         &error) ||
        !ReadFloat32Vector(records, prefix + "advantages", rows, &advantages,
                           &error)) {
      std::fprintf(stderr, "rl_step: %s\n", error.c_str());
      return 1;
    }
    const std::vector<int> tokens(tokens_i32.begin(), tokens_i32.end());
    const std::vector<int> targets(targets_i32.begin(), targets_i32.end());

    const float loss = nanochat::RlStep(
        model.get(), optimizer.get(), tokens.data(), targets.data(),
        advantages.data(), batch, seq, num_passes, examples_per_rank, step);

    std::int32_t valid = 0;
    for (int index = 0; index < static_cast<int>(rows); ++index) {
      if (targets[static_cast<std::size_t>(index)] != -1) ++valid;
    }
    const std::string step_prefix = "result/step/" + std::to_string(step);
    output.push_back(MakeFloat32(step_prefix + "/loss", {1}, {loss}));
    output.push_back(
        MakeFloat32(step_prefix + "/grad_norm", {1}, {optimizer->GradNorm()}));
    output.push_back(MakeInt32(step_prefix + "/valid_targets", {1}, {valid}));
    for (const ParamView& view : model->params()) {
      const double l2 = L2Norm(view.value, view.count);
      output.push_back(MakeFloat32(step_prefix + "/param_l2/" + view.name, {1},
                                   {static_cast<float>(l2)}));
    }
    std::printf("rl_step: step %d loss %.6g grad_norm %.6g valid %d\n", step,
                static_cast<double>(loss),
                static_cast<double>(optimizer->GradNorm()), valid);
  }

  if (!SaveRecords(out_path, output, &error)) {
    std::fprintf(stderr, "rl_step: %s\n", error.c_str());
    return 1;
  }
  std::printf("rl_step: ran %d step(s) over %d x %d, wrote %s\n", steps, batch,
              seq, out_path.c_str());
  return 0;
}
