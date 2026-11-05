// Benchmark for the Phase 1 prototype: the device-side batched fp32 GEMM of the
// Muon Polar Express iteration measured against the cuBLAS call the optimizer
// issues for the same product (`kernels::Gemm`, backends/cuda/kernels/muon.cu).
//
// One row per (path, stage): `hand:` is the prototype and `cublas:` is the seam
// GEMM. The three stages are the three GEMMs of one Polar Express iteration at
// one group's shapes:
//
//   A     the reduction product: X^T X on the tall branch, X X^T on the wide
//   one A2    the square product A A prod  the application: X B on the tall
//   branch, B X on the wide one
//
// `iter` times the three in sequence, which is the loop body the fusion would
// carry in one launch. The report also states the share of the cuBLAS rate
// every stage and the iteration reach, as an extra field, because a share is
// not a row of milliseconds. The section 4 gate is `share >= 0.85`.
//
// The default groups are the two the gate names, from the `big` profile of
// `muon_benchmark` (docs/post-training.md Part 3, section 5): the 640x1280
// key/value group (batch 40, the group the fusion targets) and the largest
// group 1280x5120x1280 (batch 20). `--rows`, `--cols`, and `--batch` measure
// another group. Run it through the broker:
//
//   tools/nanochat bench -- bazel-bin/dev/kernels/muon_fused_gemm_bench --json
//
// The prototype is fp32-only, so a bf16 or fp8 build reports the schedule and
// skips the measurement. See dev/kernels/muon_fused_gemm.h and
// dev/kernels/README.md.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <string>
#include <vector>

#include "backends/cuda/kernels/testing/bench_utils.h"
#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "dev/kernels/muon_fused_gemm.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"

namespace {

using nanochat::BenchOptions;
using nanochat::BenchReport;
using nanochat::dev::BenchDeviceName;
using nanochat::dev::DevBuf;
using nanochat::dev::EventTimer;
using nanochat::dev::MuonFusedGemm;
using nanochat::dev::MuonGemmForm;
using nanochat::dev::MuonGemmFormName;
using nanochat::dev::MuonGemmOp;
using nanochat::dev::MuonGemmSeamOp;
using nanochat::dev::MuonPolarGemm;
using nanochat::dev::MuonSquareSide;

// The share of the cuBLAS rate the gate requires on the target group.
constexpr double kGateShare = 0.85;

// One stacked shape group: `batch` matrices of `rows` by `cols`.
struct Group {
  const char* label;
  int rows;
  int cols;
  int batch;
};

constexpr Group kGroups[] = {
    // The 640x1280 key/value group of the `big` profile: the group the fusion
    // targets, 40 matrices. `rows < cols`, so it takes the wide branch.
    {"640x1280", 640, 1280, 40},
    // The largest group of the `big` profile: 1280 by 5120 with a reduction of
    // 1280 on the application product, also the wide branch.
    {"1280x5120", 1280, 5120, 20},
};

#if NANOCHAT_PRECISION == 0

// The three products of one Polar Express iteration at one group's shapes.
struct Stages {
  MuonGemmOp a;
  MuonGemmOp a2;
  MuonGemmOp prod;
};

Stages StagesFor(const Group& g, const float* x, float* square, float* square2,
                 float* prod) {
  const bool tall = g.rows > g.cols;
  Stages s;
  s.a = MuonPolarGemm(tall ? MuonGemmForm::kXtX : MuonGemmForm::kXXt,
                      MuonSquareSide::kRight, g.rows, g.cols, g.batch, x,
                      square, square);
  s.a2 = MuonPolarGemm(MuonGemmForm::kAA, MuonSquareSide::kRight, g.rows,
                       g.cols, g.batch, x, square, square2);
  // The wide branch names the side the square operand sits on, which is the
  // selector `transpose_a`/`transpose_b` cannot express.
  s.prod = MuonPolarGemm(tall ? MuonGemmForm::kXB : MuonGemmForm::kBX,
                         tall ? MuonSquareSide::kRight : MuonSquareSide::kLeft,
                         g.rows, g.cols, g.batch, x, square2, prod);
  return s;
}

void RunGroup(const Group& g, const BenchOptions& opt, BenchReport* report,
              bool verbose) {
  const int min_dim = std::min(g.rows, g.cols);
  const int mat = g.rows * g.cols;
  const int min_sq = min_dim * min_dim;

  nanochat::dev::Rng rng;
  rng.s ^= static_cast<std::uint64_t>(mat);
  DevBuf<float> x(nanochat::dev::RandomVec(g.batch * mat, &rng));
  DevBuf<float> square(nanochat::dev::RandomVec(g.batch * min_sq, &rng));
  DevBuf<float> square2(g.batch * min_sq);
  DevBuf<float> prod(g.batch * mat);

  const Stages stages = StagesFor(g, x.ptr, square.ptr, square2.ptr, prod.ptr);

  // The seam GEMM is the incumbent: `kernels::Gemm` is the cuBLAS call
  // `MuonUpdate` issues for the same product.
  const MuonGemmOp* all[3] = {&stages.a, &stages.a2, &stages.prod};
  const char* stage_name[3] = {"A", "A2", "prod"};
  double hand_ms[3] = {0.0, 0.0, 0.0};
  double cublas_ms[3] = {0.0, 0.0, 0.0};

  for (int i = 0; i < 3; ++i) {
    const MuonGemmOp& op = *all[i];
    const nanochat::GemmParams gp = MuonGemmSeamOp(op);
    const double flops = nanochat::dev::MuonGemmFlops(op);
    EventTimer hand_timer;
    hand_ms[i] = hand_timer.Time([&] { MuonFusedGemm(op); }, opt);
    EventTimer cublas_timer;
    cublas_ms[i] = cublas_timer.Time(
        [&] {
          nanochat::kernels::Gemm(nanochat::GemmMode::kForward, gp, op.a, op.b,
                                  op.c);
        },
        opt);
    const std::string shape =
        std::to_string(op.m) + "x" + std::to_string(op.n) + "x" +
        std::to_string(op.k) + " b=" + std::to_string(op.batch);
    report->Add("hand:" + std::string(g.label) + ":" + stage_name[i], shape,
                hand_ms[i], 1e-9 * flops / (hand_ms[i] * 1e-3));
    report->Add("cublas:" + std::string(g.label) + ":" + stage_name[i], shape,
                cublas_ms[i], 1e-9 * flops / (cublas_ms[i] * 1e-3));
  }

  // The loop body: the three products back to back, as one iteration issues
  // them. It is the unit a fused launch would carry.
  const double flops_iter = nanochat::dev::MuonGemmFlops(stages.a) +
                            nanochat::dev::MuonGemmFlops(stages.a2) +
                            nanochat::dev::MuonGemmFlops(stages.prod);
  EventTimer hand_timer;
  const double hand_iter = hand_timer.Time(
      [&] {
        MuonFusedGemm(stages.a);
        MuonFusedGemm(stages.a2);
        MuonFusedGemm(stages.prod);
      },
      opt);
  EventTimer cublas_timer;
  const double cublas_iter = cublas_timer.Time(
      [&] {
        for (int i = 0; i < 3; ++i) {
          const MuonGemmOp& op = *all[i];
          const nanochat::GemmParams gp = MuonGemmSeamOp(op);
          nanochat::kernels::Gemm(nanochat::GemmMode::kForward, gp, op.a, op.b,
                                  op.c);
        }
      },
      opt);
  const std::string iter_shape = std::to_string(g.rows) + "x" +
                                 std::to_string(g.cols) +
                                 " b=" + std::to_string(g.batch);
  report->Add("hand:" + std::string(g.label) + ":iter", iter_shape, hand_iter,
              1e-9 * flops_iter / (hand_iter * 1e-3));
  report->Add("cublas:" + std::string(g.label) + ":iter", iter_shape,
              cublas_iter, 1e-9 * flops_iter / (cublas_iter * 1e-3));

  // The share of the cuBLAS rate the prototype reaches: the prototype's rate
  // over cuBLAS's, which is the reciprocal of the ratio of the times.
  const std::string prefix = std::string(g.label) + ":";
  auto share_of = [](double hand, double cublas) {
    return hand > 0.0 ? cublas / hand : 0.0;
  };
  for (int i = 0; i < 3; ++i) {
    report->SetExtra("share:" + prefix + stage_name[i],
                     share_of(hand_ms[i], cublas_ms[i]));
  }
  const double iter_share = share_of(hand_iter, cublas_iter);
  report->SetExtra("share:" + prefix + "iter", iter_share);
  report->SetExtra("gate", kGateShare);

  if (verbose) {
    const bool tall = g.rows > g.cols;
    std::printf("%s group %dx%d batch %d (%s branch)\n", g.label, g.rows,
                g.cols, g.batch, tall ? "tall" : "wide");
    for (int i = 0; i < 3; ++i) {
      std::printf("  %-5s hand %9.4f ms   cuBLAS %9.4f ms   share %.3f\n",
                  stage_name[i], hand_ms[i], cublas_ms[i],
                  share_of(hand_ms[i], cublas_ms[i]));
    }
    std::printf("  %-5s hand %9.4f ms   cuBLAS %9.4f ms   share %.3f  %s\n\n",
                "iter", hand_iter, cublas_iter, iter_share,
                iter_share >= kGateShare ? "PASS" : "BELOW GATE");
  }
}

#endif  // NANOCHAT_PRECISION == 0

// The section 4 schedule the run used, on standard error so a `--json` run
// leaves standard output to the report.
void PrintSchedule() {
  std::fprintf(
      stderr,
      "muon fused gemm prototype: %d threads, launch_bounds(%d, %d), tile "
      "%dx%d, micro %dx%d, %d-deep k slices, %d KB shared\n",
      nanochat::dev::kMuonGemmBlockThreads,
      nanochat::dev::kMuonGemmBlockThreads, nanochat::dev::kMuonGemmBlocksPerSm,
      nanochat::dev::kMuonGemmBlockM, nanochat::dev::kMuonGemmBlockN,
      nanochat::dev::kMuonGemmMicroM, nanochat::dev::kMuonGemmMicroN,
      nanochat::dev::kMuonGemmSliceK, nanochat::dev::kMuonGemmSmemBytes / 1024);
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("bench");

  BenchOptions opt;
  opt.warmup = 1;
  opt.iters = 3;
  opt.rounds = 3;
  bool verbose = true;
  std::vector<Group> groups(std::begin(kGroups), std::end(kGroups));
  Group custom{"custom", 0, 0, 0};
  int group_count = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto next = [&](int* out) {
      if (i + 1 < argc) *out = std::atoi(argv[++i]);
    };
    if (arg == "--rows") {
      next(&custom.rows);
      ++group_count;
    } else if (arg == "--cols") {
      next(&custom.cols);
      ++group_count;
    } else if (arg == "--batch") {
      next(&custom.batch);
      ++group_count;
    } else if (arg == "--iters") {
      next(&opt.iters);
    } else if (arg == "--rounds") {
      next(&opt.rounds);
    } else if (arg == "--warmup") {
      next(&opt.warmup);
    } else if (arg == "--json") {
      opt.json = true;
      verbose = false;
    } else if (arg == "--out" && i + 1 < argc) {
      opt.out = argv[++i];
    } else {
      std::fprintf(stderr,
                   "usage: %s [--rows N --cols N --batch N] [--iters N] "
                   "[--rounds N] [--warmup N] [--json] [--out PATH]\n",
                   argv[0]);
      return 2;
    }
  }
  if (group_count > 0) {
    if (custom.rows <= 0 || custom.cols <= 0 || custom.batch <= 0) {
      std::fprintf(stderr,
                   "muon_fused_gemm_bench: --rows, --cols, and --batch go "
                   "together\n");
      return 2;
    }
    groups = {custom};
  }

  PrintSchedule();
  BenchReport report;
  report.UseDevicePeak(BenchDeviceName());
  // The same join identity `muon_benchmark` states, so a prototype row and the
  // step row it belongs to compare directly.
  report.SetRowIdentity("muon_orthogonalisation", "optimizer");

#if NANOCHAT_PRECISION == 0
  for (const Group& g : groups) RunGroup(g, opt, &report, verbose);
#else
  std::fprintf(stderr, "  skipped: the prototype is fp32-only\n");
#endif
  report.Print(opt);
  return 0;
}
