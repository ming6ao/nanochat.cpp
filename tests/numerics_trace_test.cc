// tests/numerics_trace_test.cc -- the staged numeric-trace replay gate of
// docs/testing.md.
//
// The test computes the whole trace (training, evaluation, greedy generation,
// the SFT arithmetic, and the RL arithmetic) in process and compares it against
// one section of the committed fixture `tests/data/numerics_golden_10l.bin`:
//
//   * the fixture holds the exact class only -- configuration, inputs, token
//     ids, masks, greedy identifiers, argmax, and valid-target counts -- so the
//     comparison is byte for byte on every backend and every precision;
//   * the fixture carries one section per precision (`fp32/`, `bf16/`, `fp8/`),
//     and this build compares the section that names its own precision;
//   * the floating-point contract is the per-stage parity tests of
//     docs/testing.md, not this one. This gate never applies a tolerance.
//
// The same source builds as `//tests:numerics_trace_test` (CPU),
// `//tests:numerics_trace_cuda_test` (GPU), and the simulator leg runs the CPU
// target under `--config=sim`. See `tools/dump_fixtures.py numerics` for the
// generator and `tests/numerics_trace.h` for the record format.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nanochat/tensor.h"
#include "tests/numerics_trace.h"

namespace {

using nanochat::Config;
using nanochat::numerics::BackendName;
using nanochat::numerics::BuildName;
using nanochat::numerics::CompareTraces;
using nanochat::numerics::DefaultShape;
using nanochat::numerics::kMinMargin;
using nanochat::numerics::PrecisionName;
using nanochat::numerics::Record;
using nanochat::numerics::ReferenceConfig;
using nanochat::numerics::RunTrace;
using nanochat::numerics::TraceShape;
using nanochat::numerics::TraceStats;
using nanochat::oracle::Fixture;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

std::string LocateFile(int argc, char** argv, int index, const char* relative) {
  std::vector<std::string> candidates;
  if (argc > index && argv[index] != nullptr && argv[index][0] != '\0') {
    candidates.emplace_back(argv[index]);
  }
  if (const char* src_dir = std::getenv("TEST_SRCDIR")) {
    if (const char* workspace = std::getenv("TEST_WORKSPACE")) {
      candidates.emplace_back(std::string(src_dir) + "/" + workspace + "/" +
                              relative);
    }
    candidates.emplace_back(std::string(src_dir) + "/_main/" + relative);
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

// The tie rule of docs/testing.md Part 2, section 6.4: one unit of
// roundoff must not be able to flip a greedy identifier, so every recorded
// logit margin must clear a threshold.
//
// The margin is a diagnostic, not a fixture record. The rule guards the fp32
// leg. A bf16 or fp8 build resolves the logits at 8 mantissa bits, so the top
// two logits can tie, and the margin is a multiple of the bf16 unit at the
// logit magnitude. The test still prints the margin, and it still compares the
// identifiers exactly against the section of its own build.
void CheckMargin(const std::string& what, double margin) {
  const bool peaked = nanochat::kComputeIsBf16 || margin > kMinMargin;
  if (!peaked) {
    Fail(what + ": the logit margin " + std::to_string(margin) +
         " does not clear the tie threshold " + std::to_string(kMinMargin));
    return;
  }
  std::printf("  %-24s min logit margin %.6g (threshold %.1g)\n", what.c_str(),
              margin, kMinMargin);
}

// Compares two runs of the same build, record by record. The device kernels use
// atomic scatter-adds, so this is the device determinism clause: the discrete
// records must not move between two runs of one binary.
int CompareRuns(const std::vector<Record>& first,
                const std::vector<Record>& second, const std::string& label) {
  if (first.size() != second.size()) {
    Fail(label + ": record count " + std::to_string(first.size()) +
         " differs from " + std::to_string(second.size()));
    return 1;
  }
  int failures = 0;
  for (std::size_t i = 0; i < first.size(); ++i) {
    const Record& a = first[i];
    const Record& b = second[i];
    if (a.name != b.name || a.dtype != b.dtype || a.shape != b.shape ||
        a.payload != b.payload) {
      Fail(label + ": record '" + a.name + "' differs between two runs");
      ++failures;
    }
  }
  return failures;
}

int Run(int argc, char** argv) {
  const std::string golden_path =
      LocateFile(argc, argv, 1, "tests/data/numerics_golden_10l.bin");
  const std::string tokenizer_path =
      LocateFile(argc, argv, 2, "tests/data/loader_tokenizer.nctoken");

  const Fixture golden = Fixture::Load(golden_path);
  const Config config = ReferenceConfig();
  const TraceShape& shape = DefaultShape();
  std::printf("numerics_trace: backend %s precision %s build %s fixture %s\n",
              BackendName().c_str(), PrecisionName().c_str(),
              BuildName().c_str(), golden_path.c_str());

  TraceStats stats;
  const std::vector<Record> records =
      RunTrace(config, shape, tokenizer_path, &stats);
  std::printf("numerics_trace: %d records, %d parameters, %d steps\n",
              stats.records, stats.parameters, shape.steps);

  const nanochat::numerics::TraceComparison comparison =
      CompareTraces(records, golden, "numerics_trace");
  g_failures += comparison.failures;
  std::printf("numerics_trace: compared %d records in section %s\n",
              comparison.compared, PrecisionName().c_str());

  if (stats.gen_path_mismatch != 0 || stats.rl_path_mismatch != 0) {
    // A diagnostic, not a gate: the greedy decode path and the full-row
    // training path agree to within the margin.
    std::printf("numerics_trace: greedy path disagreements (gen %d, rl %d)\n",
                stats.gen_path_mismatch, stats.rl_path_mismatch);
  }
  CheckMargin("gen/margin_min", stats.min_gen_margin);
  CheckMargin("eval/margin_min", stats.min_eval_margin);
  CheckMargin("rl/margin_min", stats.min_rl_margin);

  if (BackendName() != "cpu") {
    // The device determinism clause: two runs of one binary must produce the
    // same discrete records, because the CUDA kernels use atomic scatter-adds.
    const std::vector<Record> repeat =
        RunTrace(config, shape, tokenizer_path, nullptr);
    g_failures += CompareRuns(records, repeat, "numerics_trace/repeat");
    std::printf("numerics_trace: repeat: compared %d records\n",
                static_cast<int>(repeat.size()));
  }

  if (g_failures == 0) {
    std::printf("numerics_trace: OK (bpb %.6f, sft loss %.6f, rl loss %.6f)\n",
                stats.eval_bpb, stats.sft_loss, stats.rl_loss);
    return 0;
  }
  std::fprintf(stderr, "numerics_trace: %d failure(s)\n", g_failures);
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::numerics::PinSingleThreaded();
  try {
    return Run(argc, argv);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "numerics_trace: %s\n", error.what());
    return 1;
  }
}
