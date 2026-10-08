// tests/numerics_trace_test.cc -- the staged numeric-trace replay gate of
// docs/numerics-integration.md.
//
// The test computes the whole trace (training, evaluation, greedy generation,
// the SFT arithmetic, and the RL arithmetic) in process and compares it against
// the committed golden file `tests/data/numerics_golden_10l.bin`:
//
//   * the CPU reference backend and the `--config=sim` reference engine must
//     reproduce the golden byte for byte (one source, two build configs); and
//   * the CUDA backend must reproduce every scalar within the tolerance table
//     of section 4 and every greedy identifier exactly. The parameter and
//     gradient hashes are skipped there, because the device kernels use atomic
//     scatter-adds and the bytes differ by design.
//
// The same source builds as `//tests:numerics_trace_test` (CPU),
// `//tests:numerics_trace_cuda_test` (GPU), and the simulator leg runs the CPU
// target under `--config=sim`. See `tools/dump_numerics_golden.py` for the
// generator and `tests/numerics_trace.h` for the record format.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

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
using nanochat::numerics::SerializeTrace;
using nanochat::numerics::TraceShape;
using nanochat::numerics::TraceStats;
using nanochat::oracle::Fixture;

// The comparison mode. The CPU target and the simulator leg must reproduce the
// golden byte for byte; the device target compares within the tolerance table
// and skips the hashes. The intent belongs to the target definition in
// `tests/BUILD.bazel` and not to a backend define, because one source builds
// both targets.
#if defined(NANOCHAT_TRACE_EXACT)
constexpr bool kExactTrace = true;
#else
constexpr bool kExactTrace = false;
#endif

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

// The tie rule of docs/numerics-integration.md section 6.4: one unit of
// roundoff must not be able to flip a greedy identifier, so every recorded
// logit margin must clear a threshold.
void CheckMargin(const std::string& what, double margin) {
  if (!(margin > kMinMargin)) {
    Fail(what + ": the logit margin " + std::to_string(margin) +
         " does not clear the tie threshold " + std::to_string(kMinMargin));
    return;
  }
  std::printf("  %-24s min logit margin %.6g (threshold %.1g)\n", what.c_str(),
              margin, kMinMargin);
}

void ReportGroups(const nanochat::numerics::TraceComparison& comparison) {
  for (const nanochat::numerics::GroupStat& group : comparison.groups) {
    if (group.count == 0) continue;
    std::printf("  %-16s n=%-6d max |diff| %.3g  max rel %.3g\n",
                group.name.c_str(), group.count, group.max_abs_diff,
                group.max_rel_diff);
  }
}

int Run(int argc, char** argv) {
  const std::string golden_path =
      LocateFile(argc, argv, 1, "tests/data/numerics_golden_10l.bin");
  const std::string tokenizer_path =
      LocateFile(argc, argv, 2, "tests/data/loader_tokenizer.nctoken");

  const Config config = ReferenceConfig();
  const TraceShape& shape = DefaultShape();
  const bool exact = kExactTrace;
  std::printf("numerics_trace: backend %s precision %s build %s %s golden %s\n",
              BackendName().c_str(), PrecisionName().c_str(),
              BuildName().c_str(), exact ? "(exact)" : "(tolerance)",
              golden_path.c_str());
  if (PrecisionName() != "fp32") {
    // The trace covers fp32 only (docs/numerics-integration.md section 6.2).
    // Skip rather than fail, because `tools/nanochat simulate --precision fp16`
    // runs this target as part of the `correctness` suite.
    std::printf(
        "numerics_trace: skipped, the trace covers fp32 only "
        "(docs/numerics-integration.md section 6.2)\n");
    return 0;
  }

  TraceStats stats;
  const std::vector<Record> records =
      RunTrace(config, shape, tokenizer_path, &stats);
  const Fixture golden = Fixture::Load(golden_path);
  std::printf("numerics_trace: %d records, %d parameters, %d steps\n",
              stats.records, stats.parameters, shape.steps);

  const nanochat::numerics::TraceComparison comparison =
      CompareTraces(records, golden, exact, "numerics_trace");
  g_failures += comparison.failures;
  std::printf("numerics_trace: compared %d records, skipped %d\n",
              comparison.compared, comparison.skipped);
  ReportGroups(comparison);

  if (stats.gen_path_mismatch != 0 || stats.rl_path_mismatch != 0) {
    // A diagnostic, not a gate: the greedy decode path and the full-row
    // training path agree to within the margin.
    std::printf("numerics_trace: greedy path disagreements (gen %d, rl %d)\n",
                stats.gen_path_mismatch, stats.rl_path_mismatch);
  }
  CheckMargin("gen/margin_min", stats.min_gen_margin);
  CheckMargin("eval/margin_min", stats.min_eval_margin);
  CheckMargin("rl/margin_min", stats.min_rl_margin);

  if (!exact) {
    // Clause 3 of the contract: the device trace must also reproduce itself
    // within the same tolerances across two runs, because the CUDA kernels use
    // atomic scatter-adds. The first run is the reference of the comparison.
    const std::vector<Record> repeat =
        RunTrace(config, shape, tokenizer_path, nullptr);
    const Fixture first = Fixture::Parse(SerializeTrace(records));
    const auto again =
        CompareTraces(repeat, first, /*exact=*/false, "numerics_trace/repeat");
    g_failures += again.failures;
    double worst_rel = 0.0;
    double worst_abs = 0.0;
    std::string worst_name;
    for (const nanochat::numerics::GroupStat& group : again.groups) {
      if (group.count == 0) continue;
      // The margin groups set the relative tolerance to zero, so a relative
      // difference carries no meaning for them.
      if (group.name.find("margin") == std::string::npos &&
          group.max_rel_diff > worst_rel) {
        worst_rel = group.max_rel_diff;
      }
      if (group.max_abs_diff > worst_abs) {
        worst_abs = group.max_abs_diff;
        worst_name = group.name;
      }
    }
    std::printf(
        "numerics_trace: repeat: compared %d records, skipped %d, worst "
        "relative %.3g, worst absolute %.3g (%s)\n",
        again.compared, again.skipped, worst_rel, worst_abs,
        worst_name.empty() ? "none" : worst_name.c_str());
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
