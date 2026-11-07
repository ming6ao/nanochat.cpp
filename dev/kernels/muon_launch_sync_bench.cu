// Phase 0 of the Muon fusion campaign, second half: the sync term of the
// section 3 inequality (docs/kernels.md) measured on this card.
//
// The fused prototype would replace the five dependent launches of one Polar
// Express iteration (three GEMMs and the two elementwise passes) with one
// launch and an in-kernel grid barrier between the stages. The trade is
//
//   five dependent launches of a full batched grid
//       against
//   one launch of a resident persistent grid plus three cooperative grid
//   barriers,
//
// so both sides need a measured value on the card, not an assumed one. This
// binary measures them:
//
//   launch:empty:<blocks>   per-launch stream cost of an empty kernel at that
//                           grid, which is the gap one dependent kernel adds
//   launch:coop:<blocks>    the same for a cooperative launch of that grid, so
//                           the cooperative entry point's own cost is visible
//   sync:barrier:<blocks>   one `grid.sync()` inside a cooperative kernel, from
//                           the difference between a launch with N barriers and
//                           one with none
//
// Rows are milliseconds (the `nanochat.bench.v3` unit) and the same values are
// also carried in microseconds as top-level fields, because a launch gap is a
// few microseconds and a reader of the table wants them that way.
//
// Run it through the broker:
//
//   tools/nanochat bench -- bazel-bin/dev/kernels/muon_launch_sync_bench --json
//
// See dev/kernels/README.md.

#include <cstdio>
#include <cstdlib>
#include <string>

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/testing/bench_utils.h"
#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "dev/kernels/muon_fused_gemm.h"
#include "nanochat/sandbox.h"

namespace cg = cooperative_groups;

namespace {

using nanochat::BenchOptions;
using nanochat::BenchReport;
using nanochat::dev::BenchDeviceName;
using nanochat::dev::DevBuf;
using nanochat::dev::EventTimer;

constexpr int kThreads = 256;
// The barriers one cooperative launch runs. Large enough that the launch cost
// is a small part of the number the difference divides out.
constexpr int kBarriersPerLaunch = 64;

// The empty kernel stands in for a real launch: one block, 256 threads, one
// dependent dispatch. It writes nothing reachable but the compiler cannot tell.
__global__ void EmptyKernel(float* sink) {
  if (blockIdx.x == 0xffffffffu && threadIdx.x == 0) sink[0] = 1.0f;
}

// `iterations` grid-wide barriers and no arithmetic, so the timing difference
// against `iterations == 0` is the barrier itself.
__global__ void BarrierKernel(int iterations, float* sink) {
  const cg::grid_group grid = cg::this_grid();
  for (int i = 0; i < iterations; ++i) grid.sync();
  if (iterations < 0) sink[0] = 1.0f;
}

// Per-launch cost in microseconds of an empty kernel at `blocks` blocks.
double EmptyLaunchUs(int blocks, int iters, float* sink) {
  BenchOptions opt;
  opt.warmup = 1;
  opt.iters = iters;
  opt.rounds = 3;
  EventTimer timer;
  return 1e3 * timer.Time(
                   [&] {
                     nanochat::cuda_backend::Launch(EmptyKernel, dim3(blocks),
                                                    dim3(kThreads), 0, sink);
                   },
                   opt);
}

// Per-launch cost in microseconds of one cooperative launch of `blocks` blocks
// running `iterations` barriers.
double CoopLaunchUs(int blocks, int iterations, int iters, float* sink) {
  BenchOptions opt;
  opt.warmup = 1;
  opt.iters = iters;
  opt.rounds = 3;
  EventTimer timer;
  return 1e3 *
         timer.Time(
             [&] {
               int count = iterations;
               void* args[] = {&count, &sink};
               const cudaError_t status = cudaLaunchCooperativeKernel(
                   reinterpret_cast<void*>(BarrierKernel), dim3(blocks),
                   dim3(kThreads), args, 0, nullptr);
               (void)status;
               nanochat::cuda_backend::CheckLastError("cooperative launch");
             },
             opt);
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("bench");

  bool verbose = true;
  BenchOptions opt;
  std::string out;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--json") {
      opt.json = true;
      verbose = false;
    } else if (arg == "--out" && i + 1 < argc) {
      out = argv[++i];
    } else {
      std::fprintf(stderr, "usage: %s [--json] [--out PATH]\n", argv[0]);
      return 2;
    }
  }
  opt.out = out;

  int device = 0;
  nanochat::cuda_backend::CheckCuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp props{};
  nanochat::cuda_backend::CheckCuda(cudaGetDeviceProperties(&props, device),
                                    "cudaGetDeviceProperties");
  const int sm_count = props.multiProcessorCount;

  // The cooperative limit: the resident grid, which is the whole grid a
  // persistent fused kernel may launch. The prototype's own limit is the
  // 128-thread-block occupancy of a 12 KB, 256-thread block, three per SM.
  int coop_supported = 0;
  nanochat::cuda_backend::CheckCuda(
      cudaDeviceGetAttribute(&coop_supported, cudaDevAttrCooperativeLaunch,
                             device),
      "cudaDevAttrCooperativeLaunch");
  int resident = 0;
  nanochat::cuda_backend::CheckCuda(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, BarrierKernel,
                                                    kThreads, 0),
      "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
  const int coop_blocks = resident * sm_count;
  // The prototype's resident grid: the section 4 `__launch_bounds__(256, 3)`.
  const int gemm_resident = nanochat::dev::kMuonGemmBlocksPerSm * sm_count;

  DevBuf<float> sink(64);
  BenchReport report;
  report.UseDevicePeak(BenchDeviceName());
  report.SetRowIdentity("muon_orthogonalisation", "sync");

  auto add_row = [&](const std::string& name, const std::string& shape,
                     double micros) {
    report.Add(name, shape, micros / 1e3);
    report.SetExtra("us:" + name, micros);
  };

  // Launch gap at one block, at the prototype's resident grid, and at the
  // thousands of blocks a full batched launch of one GEMM needs.
  const int wide_blocks = 4000;
  const double empty_one = EmptyLaunchUs(1, 200, sink.ptr);
  const double empty_resident = EmptyLaunchUs(gemm_resident, 200, sink.ptr);
  const double empty_wide = EmptyLaunchUs(wide_blocks, 200, sink.ptr);
  add_row("launch:empty:1", "1 block", empty_one);
  add_row("launch:empty:" + std::to_string(gemm_resident),
          std::to_string(gemm_resident) + " blocks", empty_resident);
  add_row("launch:empty:" + std::to_string(wide_blocks),
          std::to_string(wide_blocks) + " blocks", empty_wide);

  double coop_zero_resident = 0.0;
  double barrier_us = 0.0;
  if (coop_supported != 0) {
    coop_zero_resident = CoopLaunchUs(gemm_resident, 0, 3, sink.ptr);
    const double coop_many =
        CoopLaunchUs(gemm_resident, kBarriersPerLaunch, 3, sink.ptr);
    barrier_us = (coop_many - coop_zero_resident) / kBarriersPerLaunch;
    add_row("launch:coop:" + std::to_string(gemm_resident),
            std::to_string(gemm_resident) + " blocks, 0 barriers",
            coop_zero_resident);
    add_row("sync:barrier:" + std::to_string(gemm_resident),
            std::to_string(gemm_resident) + " blocks x " +
                std::to_string(kBarriersPerLaunch) + " barriers",
            barrier_us);
    // The same barrier at the largest grid a cooperative launch may use, which
    // is the alternative residency a persistent kernel would pick.
    if (coop_blocks > gemm_resident) {
      const double zero = CoopLaunchUs(coop_blocks, 0, 3, sink.ptr);
      const double many =
          CoopLaunchUs(coop_blocks, kBarriersPerLaunch, 3, sink.ptr);
      add_row("sync:barrier:" + std::to_string(coop_blocks),
              std::to_string(coop_blocks) + " blocks x " +
                  std::to_string(kBarriersPerLaunch) + " barriers",
              (many - zero) / kBarriersPerLaunch);
    }
  }

  // The section 3 trade of one Polar Express iteration. Today the optimizer
  // launches five full batched grids (three GEMMs and the two elementwise
  // passes) and each pays the launch gap of a thousands-of-blocks grid. A
  // fused launch is one persistent grid (resident blocks only) with three stage
  // barriers, so both sides change:
  //
  //   five launches at the full grid   against   one launch + three barriers
  const double five_launch_us = 5.0 * empty_wide;
  const double fused_launch_us = empty_resident + 3.0 * barrier_us;
  report.SetExtra("us:iteration:five_launch", five_launch_us);
  report.SetExtra("us:iteration:fused_launch", fused_launch_us);
  report.SetExtra("us:iteration:sync_gain", five_launch_us - fused_launch_us);
  report.SetExtraInt("resident_blocks", gemm_resident);

  if (verbose) {
    std::printf(
        "muon launch/sync probe: %s, %d SMs (sm_%d%d), %d blocks resident for "
        "a %d-thread 12 KB block, cooperative launch %s\n",
        BenchDeviceName().c_str(), sm_count, props.major, props.minor,
        gemm_resident, kThreads, coop_supported != 0 ? "supported" : "ABSENT");
    std::printf("  launch gap, empty kernel, 1 block      : %7.3f us\n",
                empty_one);
    std::printf("  launch gap, empty kernel, %-4d blocks   : %7.3f us\n",
                gemm_resident, empty_resident);
    std::printf("  launch gap, empty kernel, %-4d blocks   : %7.3f us\n",
                wide_blocks, empty_wide);
    if (coop_supported != 0) {
      std::printf("  cooperative launch, %d blocks, no barrier: %7.3f us\n",
                  gemm_resident, coop_zero_resident);
      std::printf("  grid barrier, %d blocks                 : %7.3f us\n",
                  gemm_resident, barrier_us);
      std::printf(
          "  per iteration: five launches %.3f us against one launch plus "
          "three barriers %.3f us, net %.3f us\n\n",
          five_launch_us, fused_launch_us, five_launch_us - fused_launch_us);
    } else {
      std::printf("  no cooperative launch on this card: no barrier value\n\n");
    }
  }
  report.Print(opt);
  return 0;
}
