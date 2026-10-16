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
#include "src/record_file.h"
#include "src/train.h"

namespace {

using nanochat::FindRecord;
using nanochat::LoadRecordFile;
using nanochat::MakeFloat32Record;
using nanochat::MakeInt32Record;
using nanochat::Record;
using nanochat::RecordDType;
using nanochat::RecordIntAt;
using nanochat::SaveRecordFile;

// The `NANOEVL1` eval fixture (docs/eval.md). The layout lives in
// `src/record_file.h`, so this binary shares the reader with the oracle tests.
constexpr std::uint32_t kVersion = nanochat::kRecordFileVersion;

// Reads a required int32 vector from a record the caller already looked up.
bool ReadRecordInt32(const Record* record, std::int64_t expected,
                     std::vector<std::int32_t>* out, std::string* error) {
  if (record == nullptr) {
    *error = "missing required record";
    return false;
  }
  if (record->dtype != RecordDType::kInt32 || record->numel() != expected) {
    *error = "record " + record->name + " has the wrong dtype or shape";
    return false;
  }
  out->resize(static_cast<std::size_t>(expected));
  for (std::int64_t i = 0; i < expected; ++i) {
    (*out)[static_cast<std::size_t>(i)] =
        RecordIntAt(*record, static_cast<std::size_t>(i));
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
  if (!LoadRecordFile(fixture_path, {"NANOEVL1"}, &records, &error)) {
    std::fprintf(stderr, "score_main: %s\n", error.c_str());
    return 1;
  }

  const int batch = [&]() {
    const Record* r = FindRecord(records, "config/batch");
    return r != nullptr && r->dtype == RecordDType::kInt32 && r->numel() == 1
               ? static_cast<int>(RecordIntAt(*r, 0))
               : 0;
  }();
  const int seq = [&]() {
    const Record* r = FindRecord(records, "config/seq");
    return r != nullptr && r->dtype == RecordDType::kInt32 && r->numel() == 1
               ? static_cast<int>(RecordIntAt(*r, 0))
               : 0;
  }();
  const int pad_id = [&]() {
    const Record* r = FindRecord(records, "config/pad_id");
    return r != nullptr && r->dtype == RecordDType::kInt32 && r->numel() == 1
               ? static_cast<int>(RecordIntAt(*r, 0))
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
  if (!ReadRecordInt32(FindRecord(records, "input/tokens"), rows, &tokens_i32,
                       &error)) {
    std::fprintf(stderr, "score_main: input/tokens: %s\n", error.c_str());
    return 1;
  }
  std::vector<std::int32_t> lengths_i32;
  if (!ReadRecordInt32(FindRecord(records, "input/lengths"), batch,
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
    if (!ReadRecordInt32(positions_record, batch, &positions, &error) ||
        !ReadRecordInt32(FindRecord(records, "input/focus_offsets"), batch + 1,
                         &offsets, &error) ||
        !ReadRecordInt32(FindRecord(records, "input/focus_ids"),
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
  output.push_back(MakeInt32Record("config/version", {1},
                                   {static_cast<std::int32_t>(kVersion)}));
  output.push_back(MakeInt32Record("config/batch", {1}, {batch}));
  output.push_back(MakeInt32Record("config/seq", {1}, {seq}));
  output.push_back(MakeInt32Record("config/pad_id", {1}, {pad_id}));
  output.push_back(MakeFloat32Record(
      "result/nll",
      {static_cast<std::uint64_t>(batch), static_cast<std::uint64_t>(seq)},
      nll_values));
  output.push_back(MakeInt32Record(
      "result/argmax",
      {static_cast<std::uint64_t>(batch), static_cast<std::uint64_t>(seq)},
      argmax_out));
  output.push_back(MakeInt32Record("result/focus_offsets",
                                   {static_cast<std::uint64_t>(batch) + 1},
                                   focus_offsets_out));
  output.push_back(MakeFloat32Record(
      "result/focus_logits",
      {static_cast<std::uint64_t>(focus_logits_out.size())}, focus_logits_out));

  if (!SaveRecordFile(out_path, "NANOEVL1", output, &error)) {
    std::fprintf(stderr, "score_main: %s\n", error.c_str());
    return 1;
  }
  std::printf("score_main: scored %d x %d, wrote %s\n", batch, seq,
              out_path.c_str());
  return 0;
}
