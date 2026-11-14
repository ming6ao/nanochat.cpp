// The device gate of the hand tensor-core GEMM
// (docs/tensor-core-gemm-plan.md section B.8, gate G2).
//
// `gemm_emu_test` checks the tile against a host reference off device. This one
// checks it against cuBLAS on a card, over the same contract: the four operand
// layouts, a ragged shape with a k tail, a batch above one, the alpha and beta
// terms, the fp32 store of the weight-gradient form, and every schedule.
//
// Two candidates run the same comparison. The hand tile takes the tiny shapes
// and the whole sweep, and the Hopper mainloop of `cutlass_bf16.cu` takes its
// own table, at shapes its alignment rule accepts. The runner is a template
// parameter, so the two cannot drift apart in how they are measured.
//
// cuBLAS is the reference and not a host loop, because both sides accumulate
// bf16 products in fp32. The only difference is the accumulation order, so the
// tolerance is the tightest that order allows.
//
// The tile is written and selected directly. The shape table is inert, so the
// seam would answer with cuBLAS and the test would compare cuBLAS with itself.
//
// A run needs a bf16 build, because the tile's operands are bf16. The build
// compiles this source only then; an fp32 build runs the stand-in `main`.

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <type_traits>
#include <vector>

#include "backends/cuda/gemm/cutlass_launch.h"
#include "backends/cuda/gemm/gemm_launch.h"
#include "backends/cuda/gemm/gemm_plan.h"
#include "backends/cuda/gemm/gemm_shape.h"
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
using nanochat::kernels::Free;
using nanochat::kernels::Gemm;
using nanochat::kernels::GemmWgrad;
using nanochat::kernels::IsCutlassCallable;
using nanochat::kernels::LaunchGemmCutlass;
using nanochat::kernels::LaunchGemmCutlassToFp32;
using nanochat::kernels::Memcpy;
using nanochat::kernels::ResolvedGemm;
using nanochat::kernels::ResolveGemm;
using nanochat::kernels::Synchronize;

int g_failures = 0;

// A bf16 store rounds once, and the two sides round the same way from slightly
// different accumulators, so one unit in the last place is the whole budget. An
// fp32 store keeps the accumulator, so only the order of the sum differs.
constexpr double kBf16Relative = 1e-2;
constexpr double kFp32Relative = 1e-5;

// One operand value, small enough that the sum stays inside a bf16 step of the
// case magnitudes.
float OperandValue(int index, int modulus, int shift) {
  return 0.25f * static_cast<float>((index * 5 + shift) % modulus - 3);
}

struct Case {
  const char* name;
  int m;
  int n;
  int k;
  int batch;
  bool transpose_a;
  bool transpose_b;
  float alpha;
  float beta;
  // True for the weight-gradient form, which stores fp32.
  bool fp32_out;
  // The schedule index, or -1 for every schedule in the sweep.
  int schedule;
};

// How one candidate runs. `Count` says how many runs make, and `Run` makes one.
// A runner that refuses a case returns false, which fails the test: a case in
// these tables is a case the runner claims to serve.
struct TileRunner {
  static int Count() { return nanochat::kernels::gemm_tile::kScheduleCount; }
  static const char* Name() { return "tile"; }
  template <typename COut>
  static bool Run(const GemmParams& params, const ComputeType* a,
                  const ComputeType* b, COut* c, int index) {
    if constexpr (std::is_same_v<COut, float>) {
      return nanochat::kernels::LaunchGemmTileToFp32(
          params, a, b, static_cast<GradientType*>(c), index);
    } else {
      return nanochat::kernels::LaunchGemmTile(params, a, b, c, index);
    }
  }
};

// The Hopper mainloop. It has no schedule to sweep, because its tile and its
// stage count are compile-time facts of `cutlass_bf16.cu`, so one run covers
// it. The case field `schedule` is ignored here and the run follows the sweep
// above.
struct HopperRunner {
  static int Count() { return 1; }
  static const char* Name() { return "cutlass"; }
  template <typename COut>
  static bool Run(const GemmParams& params, const ComputeType* a,
                  const ComputeType* b, COut* c, int index) {
    (void)index;
    const ResolvedGemm gemm = ResolveGemm(params, std::is_same_v<COut, float>);
    if (!IsCutlassCallable(gemm, a, b, c)) return false;
    if constexpr (std::is_same_v<COut, float>) {
      return LaunchGemmCutlassToFp32(gemm, a, b, static_cast<GradientType*>(c));
    } else {
      return LaunchGemmCutlass(gemm, a, b, c);
    }
  }
};

// Runs one case: cuBLAS into one buffer, the candidate into another, and every
// element of the two compared.
template <typename COut, class Runner>
void RunCase(const Case& c) {
  const int rows_a = c.transpose_a ? c.k : c.m;
  const int rows_b = c.transpose_b ? c.n : c.k;
  const int lda = c.transpose_a ? c.m : c.k;
  const int ldb = c.transpose_b ? c.k : c.n;
  const int ldc = c.n;
  const long long stride_a = static_cast<long long>(rows_a) * lda;
  const long long stride_b = static_cast<long long>(rows_b) * ldb;
  const long long stride_c = static_cast<long long>(c.m) * ldc;

  std::vector<ComputeType> host_a(static_cast<std::size_t>(c.batch) * stride_a);
  std::vector<ComputeType> host_b(static_cast<std::size_t>(c.batch) * stride_b);
  std::vector<COut> host_init(static_cast<std::size_t>(c.batch) * stride_c);
  for (std::size_t i = 0; i < host_a.size(); ++i) {
    host_a[i] = ComputeType(OperandValue(static_cast<int>(i), 9, 0));
  }
  for (std::size_t i = 0; i < host_b.size(); ++i) {
    host_b[i] = ComputeType(OperandValue(static_cast<int>(i), 7, 2));
  }
  for (std::size_t i = 0; i < host_init.size(); ++i) {
    host_init[i] =
        static_cast<COut>(0.125f * static_cast<float>(static_cast<int>(i % 5)));
  }

  void* dev_a = Alloc(host_a.size() * sizeof(ComputeType));
  void* dev_b = Alloc(host_b.size() * sizeof(ComputeType));
  void* dev_ref = Alloc(host_init.size() * sizeof(COut));
  void* dev_got = Alloc(host_init.size() * sizeof(COut));
  Memcpy(dev_a, host_a.data(), host_a.size() * sizeof(ComputeType),
         CopyDir::kHostToDevice);
  Memcpy(dev_b, host_b.data(), host_b.size() * sizeof(ComputeType),
         CopyDir::kHostToDevice);

  GemmParams params;
  params.m = c.m;
  params.n = c.n;
  params.k = c.k;
  params.batch_count = c.batch;
  params.alpha = c.alpha;
  params.beta = c.beta;
  params.transpose_a = c.transpose_a;
  params.transpose_b = c.transpose_b;

  const ComputeType* a = static_cast<const ComputeType*>(dev_a);
  const ComputeType* b = static_cast<const ComputeType*>(dev_b);

  // cuBLAS first, into `dev_ref`.
  Memcpy(dev_ref, host_init.data(), host_init.size() * sizeof(COut),
         CopyDir::kHostToDevice);
  if constexpr (std::is_same_v<COut, float>) {
    GemmWgrad(params, a, b, static_cast<GradientType*>(dev_ref));
  } else {
    Gemm(GemmMode::kForward, params, a, b, static_cast<ComputeType*>(dev_ref));
  }
  Synchronize();

  // `schedule` names one run of the sweep, or -1 for all of them. A runner with
  // no sweep answers to one run, so the field selects the whole of it.
  const int first = c.schedule < 0 || Runner::Count() == 1 ? 0 : c.schedule;
  const int last =
      c.schedule < 0 || Runner::Count() == 1 ? Runner::Count() : c.schedule + 1;
  std::vector<COut> got(host_init.size());
  for (int schedule = first; schedule < last; ++schedule) {
    Memcpy(dev_got, host_init.data(), host_init.size() * sizeof(COut),
           CopyDir::kHostToDevice);
    const bool took =
        Runner::Run(params, a, b, static_cast<COut*>(dev_got), schedule);
    if (!took) {
      std::fprintf(stderr,
                   "FAIL: %s: the %s refused a shape it claims to serve\n",
                   c.name, Runner::Name());
      ++g_failures;
      break;
    }
    Synchronize();
    Memcpy(got.data(), dev_got, got.size() * sizeof(COut),
           CopyDir::kDeviceToHost);

    std::vector<COut> want(host_init.size());
    Memcpy(want.data(), dev_ref, want.size() * sizeof(COut),
           CopyDir::kDeviceToHost);

    const double relative =
        std::is_same_v<COut, float> ? kFp32Relative : kBf16Relative;
    double worst = 0.0;
    std::size_t worst_index = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      const double expected = static_cast<double>(want[i]);
      const double error = std::fabs(static_cast<double>(got[i]) - expected) /
                           (relative * (1.0 + std::fabs(expected)));
      if (error > worst) {
        worst = error;
        worst_index = i;
      }
    }
    if (worst > 1.0) {
      std::fprintf(stderr,
                   "FAIL: %s (%s %d) at element %zu: candidate %f, cuBLAS %f, "
                   "%.2f times the bound\n",
                   c.name, Runner::Name(), schedule, worst_index,
                   static_cast<double>(got[worst_index]),
                   static_cast<double>(want[worst_index]), worst);
      ++g_failures;
    }
  }

  Free(dev_a);
  Free(dev_b);
  Free(dev_ref);
  Free(dev_got);
}

template <typename COut, class Runner>
void RunCases(const Case* cases, int count) {
  for (int i = 0; i < count; ++i) RunCase<COut, Runner>(cases[i]);
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");

  const nanochat::Caps caps = nanochat::kernels::GetCaps();
  if (!caps.is_device || !caps.has_cublas) {
    std::fprintf(stderr, "gemm gpu test: no device or cuBLAS available\n");
    return 1;
  }
  if (!caps.supports_bf16) {
    std::fprintf(
        stderr,
        "gemm gpu test: %s has no bf16 tensor core, so the tile cannot "
        "run there\n",
        caps.device_name);
    return 1;
  }
  std::printf("gemm gpu test: device %s, compute capability %d.%d\n",
              caps.device_name, caps.compute_major, caps.compute_minor);

  // The contract, at the tiny shapes the runtime gate uses: the four operand
  // layouts, a batch, and both stores.
  const Case kContract[] = {
      {"plain 4x3x5", 4, 3, 5, 1, false, false, 1.0f, 0.0f, false, -1},
      {"transpose B 3x4x5", 3, 4, 5, 1, false, true, 0.5f, 0.25f, false, -1},
      {"transpose A 3x4x5", 3, 4, 5, 1, true, false, 1.0f, 0.0f, false, -1},
      {"both 4x3x5", 4, 3, 5, 1, true, true, 1.0f, 0.5f, false, -1},
      {"batch 2x3x4 of 3", 2, 3, 4, 3, false, false, 1.0f, 0.0f, false, -1},
      {"wgrad 2x4x3", 2, 4, 3, 1, false, true, 1.0f, 0.0f, true, -1},
      // A ragged shape with a k tail, and a whole tile pair.
      {"ragged 45x23x50", 45, 23, 50, 1, false, false, 0.5f, 0.25f, false, -1},
      {"tile 128x64x32", 128, 64, 32, 1, false, false, 1.0f, 0.0f, false, -1},
      // Past one tile on both extents and past one k slice.
      {"multi 200x130x96", 200, 130, 96, 1, false, true, 1.0f, 0.0f, false, -1},
  };
  RunCases<ComputeType, TileRunner>(kContract,
                                    sizeof(kContract) / sizeof(kContract[0]));

  // The weight-gradient form at a shape that covers more than one tile.
  const Case kWgrad[] = {
      {"wgrad multi 96x64x40", 96, 64, 40, 2, false, true, 1.0f, 0.0f, true,
       -1},
  };
  RunCases<float, TileRunner>(kWgrad, sizeof(kWgrad) / sizeof(kWgrad[0]));

  // One real shape on each schedule, so the sweep is checked on a card and not
  // only off it. The loop follows the sweep, so a schedule added to
  // `gemm_plan.h` is covered here without a second edit.
  for (int schedule = 0;
       schedule < nanochat::kernels::gemm_tile::kScheduleCount; ++schedule) {
    const Case one{"sweep 640x640x96",
                   640,
                   640,
                   96,
                   2,
                   false,
                   false,
                   1.0f,
                   0.0f,
                   false,
                   schedule};
    RunCase<ComputeType, TileRunner>(one);
  }

  // The Hopper mainloop over the same contract, at shapes its own rule accepts:
  // every operand axis a multiple of eight, so every stride shares the
  // sixteen-byte boundary the instruction reads through, and an output large
  // enough to clear the crossover.
  const Case kHopper[] = {
      {"hopper plain 256x256x64", 256, 256, 64, 1, false, false, 1.0f, 0.0f,
       false, -1},
      {"hopper transpose B 256x192x64", 256, 192, 64, 1, false, true, 0.5f,
       0.25f, false, -1},
      {"hopper transpose A 192x256x64", 192, 256, 64, 1, true, false, 1.0f,
       0.0f, false, -1},
      {"hopper both 128x256x64", 128, 256, 64, 1, true, true, 1.0f, 0.5f, false,
       -1},
      {"hopper batch 256x128x64 of 3", 256, 128, 64, 3, false, false, 1.0f,
       0.0f, false, -1},
      // Past one tile on both extents and past one k slice, with a k tail.
      {"hopper multi 300x400x200", 300, 400, 200, 1, false, true, 1.0f, 0.25f,
       false, -1},
  };
  RunCases<ComputeType, HopperRunner>(kHopper,
                                      sizeof(kHopper) / sizeof(kHopper[0]));

  const Case kHopperWgrad[] = {
      {"hopper wgrad 256x256x64", 256, 256, 64, 1, false, true, 1.0f, 0.0f,
       true, -1},
      {"hopper wgrad batch 192x256x64 of 2", 192, 256, 64, 2, true, false, 1.0f,
       0.5f, true, -1},
  };
  RunCases<float, HopperRunner>(kHopperWgrad,
                                sizeof(kHopperWgrad) / sizeof(kHopperWgrad[0]));

  if (g_failures != 0) {
    std::fprintf(stderr, "gemm gpu test: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("gemm gpu test: ok\n");
  return 0;
}
