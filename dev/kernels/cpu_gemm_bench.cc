// dev/kernels/cpu_gemm_bench.cc -- CPU reference GEMM micro-benchmark.
//
// Times the four dominant GEMM shapes from docs/cpu-performance.md Part 2,
// section 2, plus the model forward layout:
//   * gemm_forward -- language-model head forward, M=1024 N=32768 K=256.
//   * gemm_forward_tb1 -- the same head forward with transpose_b, the layout
//                     that src/ops.cc stores the weights in.
//   * gemm_wgrad   -- language-model head weight gradient, M=32768 N=256 K=1024
//                     with a transposed A (the saved activation).
//   * gemm_dgrad   -- language-model head input gradient, M=1024 N=256 K=32768.
//   * mlp_gemm     -- MLP forward up-projection, M=1024 N=1024 K=256.
//   * mlp_gemm_tb1 -- the same MLP forward with transpose_b.
//
// The body calls kernels::Gemm directly. It warms up, then reports the best of
// N rounds. The report uses the `nanochat.bench.v1` schema from
// docs/performance.md. Phase 1 makes the CPU Gemm reordered and threaded with
// OpenMP, so the benchmark reports the active OpenMP thread count.
//
// Run it through the project entry point:
//
//   tools/nanochat build //dev/kernels:cpu_gemm_bench
//   tools/nanochat bench -- ./bazel-bin/dev/kernels/cpu_gemm_bench
//
// See dev/kernels/README.md and docs/cpu-performance.md.

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"

namespace {

using nanochat::ComputeType;
using nanochat::GemmMode;
using nanochat::GemmParams;

// Build the arithmetic type for this precision configuration. The default and
// only Phase 0 configuration is fp32, but the fp16 build must still compile.
ComputeType ToCompute(float value) {
#if defined(NANOCHAT_PRECISION_FP16)
  ComputeType out;
  out.bits = nanochat::FloatToHalfBits(value);
  return out;
#else
  return value;
#endif
}

void Fill(std::vector<ComputeType>& buffer, float value) {
  for (ComputeType& element : buffer) element = ToCompute(value);
}

// One measured GEMM. `mode` is the semantic mode the host graph supplies; the
// shipped Gemm treats it as advisory and derives the operand layout from
// `transpose_a` and `transpose_b`.
struct Shape {
  const char* name;
  GemmMode mode;
  int m;
  int n;
  int k;
  bool transpose_a;
  bool transpose_b;
};

// The four dominant shapes from docs/cpu-performance.md Part 2, section 2, plus
// the model forward layout (`transpose_b = true`). The model config is depth 4,
// hidden 256, MLP 1024, vocab 32768, and 1024 token rows.
const Shape kShapes[] = {
    {"gemm_forward", GemmMode::kForward, 1024, 32768, 256, false, false},
    {"gemm_forward_tb1", GemmMode::kForward, 1024, 32768, 256, false, true},
    {"gemm_wgrad", GemmMode::kWgrad, 32768, 256, 1024, true, false},
    {"gemm_dgrad", GemmMode::kDgrad, 1024, 256, 32768, false, false},
    {"mlp_gemm", GemmMode::kForward, 1024, 1024, 256, false, false},
    {"mlp_gemm_tb1", GemmMode::kForward, 1024, 1024, 256, false, true},
};

struct Options {
  int warmup = 1;
  int rounds = 3;
  bool json = false;
  bool quiet = false;
  std::string out;
};

struct Result {
  double ms = 0.0;
  double gflops = 0.0;
};

// The number of threads the GEMM uses. This target compiles with `-fopenmp`
// (`dev/kernels/BUILD.bazel`), so `_OPENMP` is defined and this reports the
// OpenMP pool. The pool reads `NANOCHAT_NUM_THREADS` or `OMP_NUM_THREADS` from
// the sandbox profile (docs/cpu-performance.md Part 2, section 4.2).
int ActiveThreadCount() {
#if defined(_OPENMP)
  return omp_get_max_threads();
#else
  return 1;
#endif
}

// Warm up, then time `rounds` single calls and keep the fastest. Each of these
// shapes takes hundreds of milliseconds, so a per-call timer resolves well.
Result TimeShape(const Shape& shape, const Options& options) {
  const int rows_a = shape.transpose_a ? shape.k : shape.m;
  const int cols_a = shape.transpose_a ? shape.m : shape.k;
  const int rows_b = shape.transpose_b ? shape.n : shape.k;
  const int cols_b = shape.transpose_b ? shape.k : shape.n;
  std::vector<ComputeType> a(static_cast<std::size_t>(rows_a) * cols_a);
  std::vector<ComputeType> b(static_cast<std::size_t>(rows_b) * cols_b);
  std::vector<ComputeType> c(static_cast<std::size_t>(shape.m) * shape.n);
  Fill(a, 0.5f);
  Fill(b, 0.5f);
  Fill(c, 0.0f);

  GemmParams params;
  params.m = shape.m;
  params.n = shape.n;
  params.k = shape.k;
  params.transpose_a = shape.transpose_a;
  params.transpose_b = shape.transpose_b;

  const auto run = [&] {
    nanochat::kernels::Gemm(shape.mode, params, a.data(), b.data(), c.data());
  };
  for (int i = 0; i < options.warmup; ++i) run();

  double best_ms = 0.0;
  for (int round = 0; round < options.rounds; ++round) {
    const auto start = std::chrono::steady_clock::now();
    run();
    const auto finish = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(finish - start).count();
    if (round == 0 || ms < best_ms) best_ms = ms;
  }

  // `2.0` first keeps the product in double and avoids a 32-bit overflow.
  const double flops = 2.0 * shape.m * shape.n * shape.k;
  Result result;
  result.ms = best_ms;
  result.gflops = best_ms > 0.0 ? flops / (best_ms / 1000.0) / 1e9 : 0.0;
  return result;
}

void PrintTable(const Shape& shape, const Result& result) {
  std::fprintf(stderr, "  %-16s M=%-6d N=%-6d K=%-6d ta=%d tb=%d", shape.name,
               shape.m, shape.n, shape.k, shape.transpose_a ? 1 : 0,
               shape.transpose_b ? 1 : 0);
  std::fprintf(stderr, "  %10.3f ms %8.3f GFLOP/s\n", result.ms, result.gflops);
}

void PrintJson(const Shape* shapes, const Result* results, std::size_t count,
               int threads, const std::string& out) {
  std::string json = "{\"schema\":\"nanochat.bench.v1\",\"threads\":";
  json += std::to_string(threads);
  json += ",\"rows\":[";
  for (std::size_t i = 0; i < count; ++i) {
    char row[320];
    std::snprintf(row, sizeof(row),
                  "%s{\"name\":\"%s\",\"shape\":\"M=%d N=%d K=%d ta=%d tb=%d\","
                  "\"ms\":%.6f,\"gflops\":%.6f,\"work_ratio\":1.0}",
                  i == 0 ? "" : ",", shapes[i].name, shapes[i].m, shapes[i].n,
                  shapes[i].k, shapes[i].transpose_a ? 1 : 0,
                  shapes[i].transpose_b ? 1 : 0, results[i].ms,
                  results[i].gflops);
    json += row;
  }
  json += "]}";
  std::printf("%s\n", json.c_str());
  if (!out.empty()) {
    std::FILE* file = std::fopen(out.c_str(), "w");
    if (file == nullptr) {
      std::fprintf(stderr, "cpu_gemm_bench: cannot write %s\n", out.c_str());
      return;
    }
    std::fprintf(file, "%s\n", json.c_str());
    std::fclose(file);
  }
}

void Usage() {
  std::fprintf(stderr,
               "usage: cpu_gemm_bench [options]\n"
               "  --warmup N   untimed warm-up calls per shape (default 1)\n"
               "  --rounds N   timed best-of-N rounds per shape (default 3)\n"
               "  --json       emit the nanochat.bench.v1 JSON report\n"
               "  --out PATH   also write the JSON report to PATH\n"
               "  --quiet      suppress the human-readable table\n");
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("bench");

  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "-h" || flag == "--help") {
      Usage();
      return 0;
    }
    if (flag == "--quiet") {
      options.quiet = true;
      continue;
    }
    if (flag == "--json") {
      options.json = true;
      continue;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "cpu_gemm_bench: %s needs a value\n", flag.c_str());
      return 2;
    }
    const char* value = argv[++i];
    if (flag == "--warmup") {
      options.warmup = std::atoi(value);
    } else if (flag == "--rounds") {
      options.rounds = std::atoi(value);
    } else if (flag == "--out") {
      options.out = value;
    } else {
      std::fprintf(stderr, "cpu_gemm_bench: unknown flag %s\n", flag.c_str());
      Usage();
      return 2;
    }
  }
  if (options.warmup < 0 || options.rounds < 1) {
    std::fprintf(stderr, "cpu_gemm_bench: bad warmup/rounds\n");
    return 2;
  }

  const int threads = ActiveThreadCount();
  const std::size_t count = sizeof(kShapes) / sizeof(kShapes[0]);
  std::vector<Result> results(count);
  if (!options.quiet) {
    std::fprintf(stderr, "cpu_gemm_bench: threads=%d rounds=%d warmup=%d\n",
                 threads, options.rounds, options.warmup);
  }
  for (std::size_t i = 0; i < count; ++i) {
    results[i] = TimeShape(kShapes[i], options);
    if (!options.quiet && !options.json) PrintTable(kShapes[i], results[i]);
  }
  PrintJson(kShapes, results.data(), count, threads, options.out);
  return 0;
}
