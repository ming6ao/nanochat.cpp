// dev/kernels/cpu_linear_layout_probe.cc -- CPU GEMM operand-layout probe.
//
// The model's `ops::LinearForward` stores weights as `[out, in]` and sets
// `transpose_b = true` (src/ops.cc). The first Phase 1 run read `b`
// contiguously only when `transpose_b` is false, so the model forward did not
// get the Phase 1 speedup. The model-layout fix makes the `transpose_b = true`
// branch read `b` contiguously too.
//
// This probe times the same logical shapes under both layouts. It confirms
// that the two layouts now agree:
//
//   * `lm_head_fwd_tb1` -- the model layout, `transpose_b = true`.
//   * `lm_head_fwd_tb0` -- the micro-benchmark layout, `transpose_b = false`.
//   * `lm_head_wgrad`   -- `transpose_a = true`, `transpose_b = false`.
//   * `lm_head_dgrad`   -- both false.
//   * `mlp_fwd_tb1`     -- the model MLP forward, `transpose_b = true`.
//   * `mlp_fwd_tb0`     -- the micro-benchmark MLP forward.
//
// Run it through the project entry point:
//
//   tools/nanochat build //dev/kernels:cpu_linear_layout_probe
//   tools/nanochat bench -- ./bazel-bin/dev/kernels/cpu_linear_layout_probe
//
// See docs/cpu-performance.md.

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

// One logical GEMM with an explicit operand layout. `mode` stays advisory;
// `transpose_a` and `transpose_b` select the stored operand shapes.
struct Shape {
  const char* name;
  GemmMode mode;
  int m;
  int n;
  int k;
  bool transpose_a;
  bool transpose_b;
};

// The four model linears from src/ops.cc plus their benchmark counterparts.
const Shape kShapes[] = {
    {"lm_head_fwd_tb1", GemmMode::kForward, 1024, 32768, 256, false, true},
    {"lm_head_fwd_tb0", GemmMode::kForward, 1024, 32768, 256, false, false},
    {"lm_head_wgrad", GemmMode::kWgrad, 32768, 256, 1024, true, false},
    {"lm_head_dgrad", GemmMode::kDgrad, 1024, 256, 32768, false, false},
    {"mlp_fwd_tb1", GemmMode::kForward, 1024, 1024, 256, false, true},
    {"mlp_fwd_tb0", GemmMode::kForward, 1024, 1024, 256, false, false},
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

int ActiveThreadCount() {
#if defined(_OPENMP)
  return omp_get_max_threads();
#else
  return 1;
#endif
}

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

  const double flops = 2.0 * shape.m * shape.n * shape.k;
  Result result;
  result.ms = best_ms;
  result.gflops = best_ms > 0.0 ? flops / (best_ms / 1000.0) / 1e9 : 0.0;
  return result;
}

void PrintTable(const Shape& shape, const Result& result) {
  std::fprintf(stderr, "  %-18s M=%-6d N=%-6d K=%-6d ta=%d tb=%d", shape.name,
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
      std::fprintf(stderr, "cpu_linear_layout_probe: cannot write %s\n",
                   out.c_str());
      return;
    }
    std::fprintf(file, "%s\n", json.c_str());
    std::fclose(file);
  }
}

void Usage() {
  std::fprintf(stderr,
               "usage: cpu_linear_layout_probe [options]\n"
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
      std::fprintf(stderr, "cpu_linear_layout_probe: %s needs a value\n",
                   flag.c_str());
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
      std::fprintf(stderr, "cpu_linear_layout_probe: unknown flag %s\n",
                   flag.c_str());
      Usage();
      return 2;
    }
  }
  if (options.warmup < 0 || options.rounds < 1) {
    std::fprintf(stderr, "cpu_linear_layout_probe: bad warmup/rounds\n");
    return 2;
  }

  const int threads = ActiveThreadCount();
  const std::size_t count = sizeof(kShapes) / sizeof(kShapes[0]);
  std::vector<Result> results(count);
  if (!options.quiet) {
    std::fprintf(stderr,
                 "cpu_linear_layout_probe: threads=%d rounds=%d warmup=%d\n",
                 threads, options.rounds, options.warmup);
  }
  for (std::size_t i = 0; i < count; ++i) {
    results[i] = TimeShape(kShapes[i], options);
    if (!options.quiet && !options.json) PrintTable(kShapes[i], results[i]);
  }
  PrintJson(kShapes, results.data(), count, threads, options.out);
  return 0;
}
