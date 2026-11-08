// The rate gate of the hand tensor-core GEMM
// (docs/tensor-core-gemm-plan.md sections B.3 and B.4).
//
// One row per shape of the closed list, measured twice: once with the tile and
// once with cuBLAS, through the shipped `kernels::Gemm` and the
// `NANOCHAT_GEMM_LIB` switch. A ratio of the two is the fraction the promotion
// gate asks for, 0.70 to promote a row and 0.85 for a shape replacement.
//
// `--schedule all` sweeps the five schedules of `gemm_plan.h` over every row.
// Without it the benchmark measures the schedule each row owns, which is the
// one a promotion would run.
//
// The benchmark calls the seam and not the launcher, so it measures the wiring
// a training step takes and not only the kernel.
//
// A run needs a bf16 build, because the tile's operands are bf16. The build
// compiles this source only then; an fp32 build runs the stand-in `main`.
//
// See docs/performance.md for the measurement protocol.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "backends/cuda/gemm/dispatch.h"
#include "backends/cuda/gemm/gemm_shape.h"
#include "backends/cuda/kernels/testing/bench_utils.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"

namespace {

using nanochat::ComputeType;
using nanochat::CopyDir;
using nanochat::GemmMode;
using nanochat::GemmParams;
using nanochat::GradientType;
using nanochat::kernels::Alloc;
using nanochat::kernels::CutlassRefusal;
using nanochat::kernels::CutlassRefusalName;
using nanochat::kernels::Free;
using nanochat::kernels::Gemm;
using nanochat::kernels::GemmScheduleCount;
using nanochat::kernels::GemmScheduleName;
using nanochat::kernels::GemmTileShapeSchedule;
using nanochat::kernels::JudgeCutlass;
using nanochat::kernels::Memcpy;
using nanochat::kernels::ResolvedGemm;
using nanochat::kernels::Synchronize;

// The report aliases the tree's benchmark header gives every benchmark.
using nanochat::dev::BenchOptions;
using nanochat::dev::BenchReport;
using nanochat::dev::EventTimer;
using nanochat::dev::UseDevicePeak;

struct Options {
  BenchOptions bench;
  std::string row;  // empty runs every row
  // Which libraries to time: "all" runs cuBLAS, the Hopper mainloop, and the
  // hand tile; "cublas", "cutlass", and "hand" each run one.
  std::string hand = "all";
  // Empty measures the row's own schedule, which is what a shipped promotion
  // would run; "all" sweeps the whole list, and a name measures one schedule.
  std::string schedule;
};

// The operands a row needs, on the device and on the host.
struct Buffers {
  void* a = nullptr;
  void* b = nullptr;
  void* c = nullptr;
  std::vector<ComputeType> host_a;
  std::vector<ComputeType> host_b;
};

bool Fill(const ResolvedGemm& gemm, Buffers* buffers) {
  const std::size_t rows_a = gemm.transpose_a ? gemm.k : gemm.m;
  const std::size_t rows_b = gemm.transpose_b ? gemm.n : gemm.k;
  const std::size_t count_a = static_cast<std::size_t>(gemm.batch_count) *
                              rows_a * static_cast<std::size_t>(gemm.lda);
  const std::size_t count_b = static_cast<std::size_t>(gemm.batch_count) *
                              rows_b * static_cast<std::size_t>(gemm.ldb);
  const std::size_t count_c = static_cast<std::size_t>(gemm.batch_count) *
                              static_cast<std::size_t>(gemm.m) *
                              static_cast<std::size_t>(gemm.ldc);
  buffers->host_a.assign(count_a, ComputeType(0.25f));
  buffers->host_b.assign(count_b, ComputeType(-0.125f));
  buffers->a = Alloc(count_a * sizeof(ComputeType));
  buffers->b = Alloc(count_b * sizeof(ComputeType));
  buffers->c = Alloc(count_c * sizeof(GradientType));
  if (buffers->a == nullptr || buffers->b == nullptr || buffers->c == nullptr) {
    return false;
  }
  Memcpy(buffers->a, buffers->host_a.data(), count_a * sizeof(ComputeType),
         CopyDir::kHostToDevice);
  Memcpy(buffers->b, buffers->host_b.data(), count_b * sizeof(ComputeType),
         CopyDir::kHostToDevice);
  return true;
}

void Release(Buffers* buffers) {
  Free(buffers->a);
  Free(buffers->b);
  Free(buffers->c);
  *buffers = Buffers();
}

// One timing of the seam under one value of the switch.
double TimeGemm(const ResolvedGemm& gemm, const Buffers& buffers,
                const char* library, const BenchOptions& options) {
  setenv("NANOCHAT_GEMM_LIB", library, 1);
  GemmParams params;
  params.m = gemm.m;
  params.n = gemm.n;
  params.k = gemm.k;
  params.batch_count = gemm.batch_count;
  params.lda = gemm.lda;
  params.ldb = gemm.ldb;
  params.ldc = gemm.ldc;
  params.stride_a = gemm.stride_a;
  params.stride_b = gemm.stride_b;
  params.stride_c = gemm.stride_c;
  params.transpose_a = gemm.transpose_a;
  params.transpose_b = gemm.transpose_b;
  // The weight-gradient form stores fp32; the other form stores the compute
  // type. One row measures one store.
  const ComputeType* a = static_cast<const ComputeType*>(buffers.a);
  const ComputeType* b = static_cast<const ComputeType*>(buffers.b);
  EventTimer timer;
  if (gemm.fp32_c) {
    return timer.Time(
        [&] {
          nanochat::kernels::GemmWgrad(params, a, b,
                                       static_cast<GradientType*>(buffers.c));
        },
        options);
  }
  return timer.Time(
      [&] {
        Gemm(GemmMode::kForward, params, a, b,
             static_cast<ComputeType*>(buffers.c));
      },
      options);
}

void Usage(const char* program) {
  std::fprintf(stderr,
               "usage: %s [--row NAME] [--all|cutlass|hand|cublas] "
               "[--schedule NAME|all] [--iters N] [--rounds N] [--warmup N] "
               "[--json] [--out PATH]\n",
               program);
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("bench");

  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next_int = [&](int* out) {
      if (i + 1 >= argc) {
        Usage(argv[0]);
        std::exit(2);
      }
      *out = std::atoi(argv[++i]);
    };
    if (arg == "--row" && i + 1 < argc) {
      options.row = argv[++i];
    } else if (arg == "--hand") {
      options.hand = "hand";
    } else if (arg == "--cublas") {
      options.hand = "cublas";
    } else if (arg == "--cutlass") {
      options.hand = "cutlass";
    } else if (arg == "--all" || arg == "--both") {
      // `--both` is the older spelling of the two-side measurement, and it
      // still means every side.
      options.hand = "all";
    } else if (arg == "--schedule" && i + 1 < argc) {
      options.schedule = argv[++i];
    } else if (arg == "--iters") {
      next_int(&options.bench.iters);
    } else if (arg == "--rounds") {
      next_int(&options.bench.rounds);
    } else if (arg == "--warmup") {
      next_int(&options.bench.warmup);
    } else if (arg == "--json") {
      options.bench.json = true;
    } else if (arg == "--out" && i + 1 < argc) {
      options.bench.out = argv[++i];
    } else {
      Usage(argv[0]);
      return 2;
    }
  }

  const bool run_all = options.hand == "all";
  const bool run_hand = run_all || options.hand == "hand";
  const bool run_cublas = run_all || options.hand == "cublas";
  const bool run_cutlass = run_all || options.hand == "cutlass";
  const int row_count = nanochat::kernels::GemmTileShapeCount();

  BenchReport report;
  UseDevicePeak(&report);
  report.SetRowIdentity("gemm", "forward");
  report.SetExtraInt("shape_rows", row_count);
  report.SetExtraText("mode", options.hand.c_str());

  int measured = 0;
  for (int index = 0; index < row_count; ++index) {
    const std::string name = nanochat::kernels::GemmTileShapeName(index);
    if (!options.row.empty() && options.row != name) continue;
    const ResolvedGemm gemm = nanochat::kernels::GemmTileShapeGemm(index);

    Buffers buffers;
    if (!Fill(gemm, &buffers)) {
      std::fprintf(stderr,
                   "gemm benchmark: cannot allocate the operands of %s\n",
                   name.c_str());
      Release(&buffers);
      continue;
    }

    // 2 FLOPs per element of the product, over every batch.
    const double flops =
        2.0 * static_cast<double>(gemm.m) * gemm.n * gemm.k * gemm.batch_count;
    const std::string shape =
        std::to_string(gemm.m) + "x" + std::to_string(gemm.n) + "x" +
        std::to_string(gemm.k) + " batch " + std::to_string(gemm.batch_count) +
        (gemm.fp32_c ? " fp32 out" : "");

    // cuBLAS does not depend on the schedule, so one timing serves them all.
    double cublas_ms = 0.0;
    if (run_cublas) {
      unsetenv("NANOCHAT_GEMM_SCHEDULE");
      cublas_ms = TimeGemm(gemm, buffers, "cublas", options.bench);
      report.Add("gemm:" + name + ":cublas", shape, cublas_ms,
                 1e-9 * flops / (cublas_ms * 1e-3));
    }
    if (run_cutlass) {
      // The seam answers a refused call with another path, so a timing of the
      // switch alone would report the fallback's rate under this column's name.
      // The rule is asked first, and the reason is printed when it refuses.
      unsetenv("NANOCHAT_GEMM_SCHEDULE");
      const CutlassRefusal refusal =
          JudgeCutlass(gemm, buffers.a, buffers.b, buffers.c);
      if (refusal == CutlassRefusal::kNone) {
        const double cutlass_ms =
            TimeGemm(gemm, buffers, "cutlass", options.bench);
        report.Add("gemm:" + name + ":cutlass", shape, cutlass_ms,
                   1e-9 * flops / (cutlass_ms * 1e-3));
        if (run_cublas && cublas_ms > 0.0 && cutlass_ms > 0.0) {
          std::fprintf(stderr,
                       "gemm benchmark: %-22s %-10s fraction of cuBLAS %.3f\n",
                       name.c_str(), "cutlass", cublas_ms / cutlass_ms);
        }
      } else {
        std::fprintf(stderr, "gemm benchmark: %-22s cutlass refuses it: %s\n",
                     name.c_str(), CutlassRefusalName(refusal));
      }
    }
    if (run_hand) {
      // An empty request measures the schedule the row owns, which is the one a
      // promotion would run. "all" sweeps the list, and a name measures one.
      const int first =
          options.schedule.empty() ? GemmTileShapeSchedule(index) : 0;
      const int last =
          options.schedule.empty() ? first + 1 : GemmScheduleCount();
      for (int schedule = first; schedule < last; ++schedule) {
        const char* schedule_name = GemmScheduleName(schedule);
        if (options.schedule != "all" && !options.schedule.empty() &&
            options.schedule != schedule_name) {
          continue;
        }
        setenv("NANOCHAT_GEMM_SCHEDULE", schedule_name, 1);
        const double hand_ms = TimeGemm(gemm, buffers, "hand", options.bench);
        const std::string label =
            std::string("gemm:") + name + ":" + schedule_name + ":hand";
        report.Add(label, shape, hand_ms, 1e-9 * flops / (hand_ms * 1e-3));
        if (run_cublas && cublas_ms > 0.0 && hand_ms > 0.0) {
          std::fprintf(stderr,
                       "gemm benchmark: %-22s %-10s fraction of cuBLAS %.3f\n",
                       name.c_str(), schedule_name, cublas_ms / hand_ms);
        }
      }
      unsetenv("NANOCHAT_GEMM_SCHEDULE");
    }

    Synchronize();
    Release(&buffers);
    ++measured;
  }

  if (measured == 0) {
    std::fprintf(stderr, "gemm benchmark: no row matched\n");
    return 2;
  }
  report.Print(options.bench);
  return 0;
}
