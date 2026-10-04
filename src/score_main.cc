// `score_main` -- stateless scorer over an evaluation fixture.
//
// It reads a `NANOEVL1` fixture of padded token sequences (plus per-row valid
// lengths and an optional focus set), runs one `ScoreBatch`, and writes a
// result fixture with the per-position negative log-likelihood, the
// per-position argmax, and the focused logits. It performs no reduction:
// whether a candidate is correct is decided by the Python bridge, next to the
// reference task logic (docs/eval.md section 3.2).
//
// The fixture wire format is owned by `python/nanochat_cpp/eval_fixture.py` and
// mirrored here and by the test dumper. The C++ side is tokenizer-agnostic: ids
// in, ids and logits out. The result file carries only `config/` and
// `result/` records, so a reader can parse it on its own.
//
//   tools/nanochat run t0-cpu -- <score_main> --fixture in.bin --out out.bin
//       --model model.ckpt

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/model.h"
#include "nanochat/sandbox.h"
#include "src/cli.h"
#include "src/train.h"

namespace {

// --- Fixture container -----------------------------------------------------

constexpr char kMagic[8] = {'N', 'A', 'N', 'O', 'E', 'V', 'L', '1'};
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
                 const std::vector<std::int32_t>& values) {
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

bool ReadInt32Vector(const Record* record, std::int64_t expected,
                     std::vector<std::int32_t>* out, std::string* error) {
  if (record == nullptr) {
    *error = "missing required record";
    return false;
  }
  if (record->dtype != kInt32 || record->numel() != expected) {
    *error = "record " + record->name + " has the wrong dtype or shape";
    return false;
  }
  out->resize(static_cast<std::size_t>(expected));
  for (std::int64_t i = 0; i < expected; ++i) {
    (*out)[static_cast<std::size_t>(i)] =
        IntAt(*record, static_cast<std::size_t>(i));
  }
  return true;
}

void Usage() {
  std::fprintf(
      stderr,
      "usage: score_main [options]\n"
      "  --fixture PATH         input eval fixture with input cases\n"
      "  --out PATH             output eval fixture with result records\n"
      "  --model PATH           checkpoint to load\n"
      "  --seed N               weight init seed when no checkpoint\n"
      "  [model flags: --layers --heads --kv-heads --hidden --seq\n"
      "   --vocab --padded-vocab --window-pattern --rope-base]\n");
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("score");

  nanochat::Config model_config;
  std::string fixture_path;
  std::string out_path;
  std::string model_path;
  std::uint64_t seed = 42;

  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "-h" || flag == "--help") {
      Usage();
      return 0;
    }
    if (nanochat::cli::IsModelFlag(flag)) {
      if (i + 1 >= argc ||
          !nanochat::cli::ApplyModelFlag(flag, argv[i + 1], &model_config)) {
        std::fprintf(stderr, "score_main: bad value for %s\n", flag.c_str());
        return 2;
      }
      ++i;
      continue;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "score_main: %s needs a value\n", flag.c_str());
      return 2;
    }
    const char* value = argv[++i];
    std::uint64_t parsed_u64 = 0;
    if (flag == "--fixture" || flag == "--in" || flag == "--input") {
      fixture_path = value;
    } else if (flag == "--out" || flag == "--output") {
      out_path = value;
    } else if (flag == "--model") {
      model_path = value;
    } else if (flag == "--seed") {
      seed = nanochat::cli::ParseU64(value, &parsed_u64) ? parsed_u64 : 0;
    } else {
      std::fprintf(stderr, "score_main: unknown flag %s\n", flag.c_str());
      Usage();
      return 2;
    }
  }

  if (fixture_path.empty() || out_path.empty()) {
    std::fprintf(stderr, "score_main: --fixture and --out are required\n");
    Usage();
    return 2;
  }

  std::vector<Record> records;
  std::string error;
  if (!LoadRecords(fixture_path, &records, &error)) {
    std::fprintf(stderr, "score_main: %s\n", error.c_str());
    return 1;
  }

  const int batch = [&]() {
    const Record* r = FindRecord(records, "config/batch");
    return r != nullptr && r->dtype == kInt32 && r->numel() == 1
               ? static_cast<int>(IntAt(*r, 0))
               : 0;
  }();
  const int seq = [&]() {
    const Record* r = FindRecord(records, "config/seq");
    return r != nullptr && r->dtype == kInt32 && r->numel() == 1
               ? static_cast<int>(IntAt(*r, 0))
               : 0;
  }();
  const int pad_id = [&]() {
    const Record* r = FindRecord(records, "config/pad_id");
    return r != nullptr && r->dtype == kInt32 && r->numel() == 1
               ? static_cast<int>(IntAt(*r, 0))
               : 0;
  }();
  if (batch <= 0 || seq <= 0) {
    std::fprintf(stderr,
                 "score_main: fixture has no usable config/batch or "
                 "config/seq\n");
    return 1;
  }

  const std::int64_t rows =
      static_cast<std::int64_t>(batch) * static_cast<std::int64_t>(seq);

  std::vector<std::int32_t> tokens_i32;
  if (!ReadInt32Vector(FindRecord(records, "input/tokens"), rows, &tokens_i32,
                       &error)) {
    std::fprintf(stderr, "score_main: input/tokens: %s\n", error.c_str());
    return 1;
  }
  std::vector<std::int32_t> lengths_i32;
  if (!ReadInt32Vector(FindRecord(records, "input/lengths"), batch,
                       &lengths_i32, &error)) {
    std::fprintf(stderr, "score_main: input/lengths: %s\n", error.c_str());
    return 1;
  }

  std::vector<int> tokens(tokens_i32.begin(), tokens_i32.end());
  std::vector<int> lengths(lengths_i32.begin(), lengths_i32.end());
  for (int b = 0; b < batch; ++b) {
    if (lengths[static_cast<std::size_t>(b)] < 0 ||
        lengths[static_cast<std::size_t>(b)] > seq) {
      std::fprintf(stderr, "score_main: input/lengths[%d] is outside [0, %d]\n",
                   b, seq);
      return 1;
    }
  }

  // Focus is optional. When present, each row's ids are a slice of the flat
  // focus id list; rows without focus carry position -1.
  std::vector<int> focus_positions(static_cast<std::size_t>(batch), -1);
  std::vector<int> focus_offsets(static_cast<std::size_t>(batch) + 1, 0);
  std::vector<int> focus_ids;
  bool has_focus = false;
  const Record* positions_record = FindRecord(records, "input/focus_positions");
  if (positions_record != nullptr) {
    std::vector<std::int32_t> positions;
    std::vector<std::int32_t> offsets;
    std::vector<std::int32_t> ids;
    if (!ReadInt32Vector(positions_record, batch, &positions, &error) ||
        !ReadInt32Vector(FindRecord(records, "input/focus_offsets"), batch + 1,
                         &offsets, &error) ||
        !ReadInt32Vector(FindRecord(records, "input/focus_ids"),
                         offsets.empty() ? 0 : offsets.back(), &ids, &error)) {
      std::fprintf(stderr, "score_main: focus records: %s\n", error.c_str());
      return 1;
    }
    if (offsets.front() != 0 ||
        offsets.back() != static_cast<std::int32_t>(ids.size())) {
      std::fprintf(stderr, "score_main: focus offsets do not span focus ids\n");
      return 1;
    }
    for (int b = 0; b < batch; ++b) {
      if (offsets[static_cast<std::size_t>(b)] >
          offsets[static_cast<std::size_t>(b) + 1]) {
        std::fprintf(stderr, "score_main: focus offsets are not monotonic\n");
        return 1;
      }
      if (positions[static_cast<std::size_t>(b)] >= 0) has_focus = true;
    }
    focus_positions.assign(positions.begin(), positions.end());
    focus_offsets.assign(offsets.begin(), offsets.end());
    focus_ids.assign(ids.begin(), ids.end());
  }

  std::unique_ptr<nanochat::Model> model =
      nanochat::Model::Create(model_config);
  if (model == nullptr) {
    std::fprintf(stderr, "score_main: cannot create the model\n");
    return 1;
  }
  if (!model_path.empty()) {
    if (!nanochat::Checkpointer::LoadModel(model.get(), model_path)) {
      std::fprintf(stderr, "score_main: cannot load checkpoint %s\n",
                   model_path.c_str());
      return 1;
    }
  } else {
    model->InitWeights(seed);
  }

  std::vector<nanochat::ScoreFocus> focus(static_cast<std::size_t>(batch));
  for (int b = 0; b < batch; ++b) {
    const int start = focus_offsets[static_cast<std::size_t>(b)];
    const int end = focus_offsets[static_cast<std::size_t>(b) + 1];
    focus[static_cast<std::size_t>(b)].position =
        focus_positions[static_cast<std::size_t>(b)];
    focus[static_cast<std::size_t>(b)].ids =
        focus_ids.empty() ? nullptr : focus_ids.data() + start;
    focus[static_cast<std::size_t>(b)].count = end - start;
  }

  std::vector<nanochat::ScoreResult> results;
  nanochat::ScoreBatch(model.get(), tokens.data(), batch, seq, lengths.data(),
                       has_focus ? focus.data() : nullptr, &results);
  if (results.size() != static_cast<std::size_t>(batch)) {
    std::fprintf(stderr, "score_main: ScoreBatch did not score the batch\n");
    return 1;
  }

  std::vector<std::int32_t> argmax_out;
  std::vector<float> nll_values;
  std::vector<std::int32_t> focus_offsets_out(
      static_cast<std::size_t>(batch) + 1, 0);
  std::vector<float> focus_logits_out;
  nll_values.reserve(static_cast<std::size_t>(rows));
  argmax_out.reserve(static_cast<std::size_t>(rows));
  for (int b = 0; b < batch; ++b) {
    const nanochat::ScoreResult& result = results[static_cast<std::size_t>(b)];
    for (int p = 0; p < seq; ++p) {
      nll_values.push_back(result.nll[static_cast<std::size_t>(p)]);
      argmax_out.push_back(result.argmax[static_cast<std::size_t>(p)]);
    }
    for (float value : result.focus_logits) focus_logits_out.push_back(value);
    focus_offsets_out[static_cast<std::size_t>(b) + 1] =
        static_cast<std::int32_t>(focus_logits_out.size());
  }

  std::vector<Record> output;
  output.push_back(
      MakeInt32("config/version", {1}, {static_cast<std::int32_t>(kVersion)}));
  output.push_back(MakeInt32("config/batch", {1}, {batch}));
  output.push_back(MakeInt32("config/seq", {1}, {seq}));
  output.push_back(MakeInt32("config/pad_id", {1}, {pad_id}));
  output.push_back(MakeFloat32(
      "result/nll",
      {static_cast<std::uint64_t>(batch), static_cast<std::uint64_t>(seq)},
      nll_values));
  output.push_back(MakeInt32(
      "result/argmax",
      {static_cast<std::uint64_t>(batch), static_cast<std::uint64_t>(seq)},
      argmax_out));
  output.push_back(MakeInt32("result/focus_offsets",
                             {static_cast<std::uint64_t>(batch) + 1},
                             focus_offsets_out));
  output.push_back(MakeFloat32(
      "result/focus_logits",
      {static_cast<std::uint64_t>(focus_logits_out.size())}, focus_logits_out));

  if (!SaveRecords(out_path, output, &error)) {
    std::fprintf(stderr, "score_main: %s\n", error.c_str());
    return 1;
  }
  std::printf("score_main: scored %d x %d, wrote %s\n", batch, seq,
              out_path.c_str());
  return 0;
}
