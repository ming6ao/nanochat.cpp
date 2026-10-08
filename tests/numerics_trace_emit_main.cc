// tests/numerics_trace_emit_main.cc -- the golden-trace emitter.
//
// Runs the whole staged trace once on the CPU reference backend and writes the
// committed golden file, in the container that `tests/oracle_fixture.h` reads.
// `tools/dump_numerics_golden.py` builds this binary, runs it through
// `tools/nanochat`, and records the summary. Run it from the repository root:
//
//   tools/nanochat build //tests:numerics_trace_emit
//   tools/nanochat run t0-cpu -- ./bazel-bin/tests/numerics_trace_emit
//       --out tests/data/numerics_golden_10l.bin
//       --tokenizer tests/data/loader_tokenizer.nctoken
//
// The emitter pins the host thread count to one, so the file does not depend on
// the sandbox thread budget. The trace is the reference configuration of
// docs/numerics-integration.md section 6.2; the emitter writes the CPU
// reference build only.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nanochat/sandbox.h"
#include "tests/numerics_trace.h"

namespace {

using nanochat::Config;
using nanochat::numerics::BackendName;
using nanochat::numerics::BuildName;
using nanochat::numerics::DefaultShape;
using nanochat::numerics::HashBytes;
using nanochat::numerics::PrecisionName;
using nanochat::numerics::Record;
using nanochat::numerics::ReferenceConfig;
using nanochat::numerics::RunTrace;
using nanochat::numerics::SerializeTrace;
using nanochat::numerics::TraceShape;
using nanochat::numerics::TraceStats;

std::string ValueOf(int argc, char** argv, const char* flag) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == flag) return argv[i + 1];
  }
  throw std::runtime_error(std::string("missing ") + flag);
}

void WriteFile(const std::string& path,
               const std::vector<std::uint8_t>& bytes) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot open '" + path + "' for writing");
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  if (!out) throw std::runtime_error("cannot write '" + path + "'");
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("emit");
  nanochat::numerics::PinSingleThreaded();
  try {
    const std::string out_path = ValueOf(argc, argv, "--out");
    const std::string tokenizer_path = ValueOf(argc, argv, "--tokenizer");
    if (BackendName() != "cpu" || PrecisionName() != "fp32") {
      throw std::runtime_error(
          "the golden trace must come from the fp32 CPU reference build");
    }
    const Config config = ReferenceConfig();
    const TraceShape& shape = DefaultShape();
    TraceStats stats;
    const std::vector<Record> records =
        RunTrace(config, shape, tokenizer_path, &stats);
    const std::vector<std::uint8_t> bytes = SerializeTrace(records);
    WriteFile(out_path, bytes);

    double first_loss = stats.train_loss.empty() ? 0.0 : stats.train_loss[0];
    double last_loss = stats.train_loss.empty() ? 0.0 : stats.train_loss.back();
    std::printf("numerics_trace_emit: %s\n", out_path.c_str());
    std::printf("numerics_trace_emit: backend %s precision %s build %s\n",
                BackendName().c_str(), PrecisionName().c_str(),
                BuildName().c_str());
    std::printf(
        "numerics_trace_emit: records %d, parameters %d, bytes %zu, "
        "fnv1a %016llx\n",
        stats.records, stats.parameters, bytes.size(),
        static_cast<unsigned long long>(HashBytes(bytes.data(), bytes.size())));
    std::printf("numerics_trace_emit: steps %d, loss %.6f -> %.6f\n",
                shape.steps, first_loss, last_loss);
    std::printf("numerics_trace_emit: eval bpb %.6f\n", stats.eval_bpb);
    std::printf(
        "numerics_trace_emit: min logit margins gen %.6g eval %.6g "
        "rl %.6g\n",
        stats.min_gen_margin, stats.min_eval_margin, stats.min_rl_margin);
    std::printf("numerics_trace_emit: sft loss %.6f, rl loss %.6f\n",
                stats.sft_loss, stats.rl_loss);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "numerics_trace_emit: %s\n", error.what());
    return 1;
  }
}
