// Standalone micro-benchmark for the Muon optimizer family. Three kinds of row:
//
//   polar:<shape>:<rows>x<cols>
//       the five Polar Express iterations of one stacked shape group: three
//       batched GEMMs each, with the shapes and the flags `MuonUpdate` issues.
//   polar:<shape>:step
//       every group in the optimizer's order, so the row is the step's
//       orthogonalisation cost alone.
//   muon:<shape>:step
//       one `MuonUpdate` per group over the same buffers `src/optim.cc`
//       passes, so the row is the whole optimizer phase of one step, at zero,
//       one, and five Polar Express iterations. The zero row is the rest of
//       the arithmetic, so the difference against five is the loop's own
//       cost.
//
// The shapes default to the two profiles the plan measures
// (docs/pipeline-scaleup-plan.md section 5): `h100` (8 layers, hidden 512,
// eight heads, four key/value heads) and `big` (20 layers, hidden 1280, ten
// heads, five key/value heads). `--profile` selects one of the two, and
// `--layers`, `--hidden`, `--heads`, and `--kv-heads` build another.
//
// Every group is the grouped shape `src/optim.cc` builds: the query and output
// projections of one shape, the key and value projections of another, and the
// two MLP matrices of two more, plus the value gate. The arithmetic count is
// the necessary count of the three GEMMs of each iteration, so the rate is
// what the plan divides by.
//
// Run it through the broker, never directly:
//
//   tools/nanochat bench -- bazel-bin/backends/cuda/kernels/muon_benchmark
//   tools/nanochat bench -- .../muon_benchmark --profile big --json
//
// See backends/cuda/kernels/README.md and docs/performance.md section 3.3.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backends/cuda/kernels/testing/bench_utils.h"
#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::ComputeType;
using nanochat::GemmMode;
using nanochat::GemmParams;
using nanochat::MuonParams;
using nanochat::dev::BenchOptions;
using nanochat::dev::BenchReport;
using nanochat::dev::DevBuf;
using nanochat::dev::EventTimer;

// The Polar Express iteration count of the reference recipe.
constexpr int kPolarIters = 5;

// One stacked shape group: `num_params` matrices of `rows` by `cols`, which is
// the layout `MuonUpdate` reads.
struct Group {
  int num_params = 0;
  int rows = 0;
  int cols = 0;
};

// One model shape, named by the job profile it comes from.
struct Profile {
  const char* name;
  int layers;
  int hidden;
  int heads;
  int kv_heads;
};

constexpr Profile kProfiles[] = {
    {"h100", 8, 512, 8, 4},
    {"big", 20, 1280, 10, 5},
};

// The Muon groups of one model shape, in the ascending shape order
// `NanochatOptimizer::Build` uses. `src/model.cc` creates two matrices of
// `query_dim` by `hidden` per layer, two of `kv_dim` by `hidden`, one of
// `hidden` by `hidden`, and the two MLP matrices; `has_value_embedding` puts
// the value gate on half of the layers.
std::vector<Group> GroupsFor(const Profile& profile) {
  const int head_dim = profile.hidden / profile.heads;
  const int query_dim = profile.heads * head_dim;
  const int kv_dim = profile.kv_heads * head_dim;
  const int mlp_dim = 4 * profile.hidden;
  const int value_gate_layers = profile.layers / 2;
  // `kVeGateChannels` of src/ops.h, the value gate's column extent.
  constexpr int kVeGateChannels = 12;

  std::map<std::pair<int, int>, int> counts;
  counts[std::make_pair(query_dim, profile.hidden)] += profile.layers;
  counts[std::make_pair(kv_dim, profile.hidden)] += 2 * profile.layers;
  counts[std::make_pair(profile.hidden, profile.hidden)] += profile.layers;
  counts[std::make_pair(mlp_dim, profile.hidden)] += profile.layers;
  counts[std::make_pair(profile.hidden, mlp_dim)] += profile.layers;
  counts[std::make_pair(profile.kv_heads, kVeGateChannels)] +=
      value_gate_layers;

  std::vector<Group> groups;
  for (const auto& entry : counts) {
    if (entry.second <= 0) continue;
    groups.push_back(
        Group{entry.second, entry.first.first, entry.first.second});
  }
  return groups;
}

std::int64_t Mat(const Group& group) {
  return static_cast<std::int64_t>(group.rows) * group.cols;
}

std::int64_t Total(const Group& group) {
  return static_cast<std::int64_t>(group.num_params) * Mat(group);
}

// The Polar Express working set of one group: the reduction extent squared,
// stacked over the group.
std::int64_t MinTotal(const Group& group) {
  const std::int64_t min_dim = std::min(group.rows, group.cols);
  return static_cast<std::int64_t>(group.num_params) * min_dim * min_dim;
}

// One Polar Express iteration's necessary arithmetic: `A = X^T X` (or
// `X X^T`), `A2 = A A`, and `prod = X B` (or `B X`), each `2 * m * n * k` per
// matrix.
double IterationFlops(const Group& group) {
  const double rows = group.rows;
  const double cols = group.cols;
  const double min_dim = std::min(rows, cols);
  const double max_dim = std::max(rows, cols);
  const double a_mat = 2.0 * min_dim * min_dim * max_dim;
  const double a2_mat = 2.0 * min_dim * min_dim * min_dim;
  const double prod = 2.0 * rows * cols * min_dim;
  return static_cast<double>(group.num_params) * (a_mat + a2_mat + prod);
}

// One Polar Express iteration over one group, with the shapes and the flags of
// `muon.cu`. The working set is `[x | prod | a | a2 | b]` from one
// allocation, and `total` and `min_total` are that group's, so the caller sizes
// the allocation for the widest group and reuses it.
void PolarIteration(const Group& group, ComputeType* x, ComputeType* prod,
                    ComputeType* a, ComputeType* a2, ComputeType* b) {
  const std::int64_t mat = Mat(group);
  const std::int64_t min_sq = std::min(group.rows, group.cols);
  const bool tall = group.rows > group.cols;
  for (int it = 0; it < kPolarIters; ++it) {
    if (tall) {
      GemmParams ga;
      ga.m = group.cols;
      ga.n = group.cols;
      ga.k = group.rows;
      ga.batch_count = group.num_params;
      ga.transpose_a = true;
      ga.stride_a = mat;
      ga.stride_b = mat;
      ga.stride_c = min_sq * min_sq;
      nanochat::kernels::Gemm(GemmMode::kForward, ga, x, x, a);

      GemmParams ga2;
      ga2.m = group.cols;
      ga2.n = group.cols;
      ga2.k = group.cols;
      ga2.batch_count = group.num_params;
      ga2.stride_a = min_sq * min_sq;
      ga2.stride_b = min_sq * min_sq;
      ga2.stride_c = min_sq * min_sq;
      nanochat::kernels::Gemm(GemmMode::kForward, ga2, a, a, a2);

      GemmParams gp;
      gp.m = group.rows;
      gp.n = group.cols;
      gp.k = group.cols;
      gp.batch_count = group.num_params;
      gp.stride_a = mat;
      gp.stride_b = min_sq * min_sq;
      gp.stride_c = mat;
      nanochat::kernels::Gemm(GemmMode::kForward, gp, x, b, prod);
    } else {
      GemmParams ga;
      ga.m = group.rows;
      ga.n = group.rows;
      ga.k = group.cols;
      ga.batch_count = group.num_params;
      ga.transpose_b = true;
      ga.stride_a = mat;
      ga.stride_b = mat;
      ga.stride_c = min_sq * min_sq;
      nanochat::kernels::Gemm(GemmMode::kForward, ga, x, x, a);

      GemmParams ga2;
      ga2.m = group.rows;
      ga2.n = group.rows;
      ga2.k = group.rows;
      ga2.batch_count = group.num_params;
      ga2.stride_a = min_sq * min_sq;
      ga2.stride_b = min_sq * min_sq;
      ga2.stride_c = min_sq * min_sq;
      nanochat::kernels::Gemm(GemmMode::kForward, ga2, a, a, a2);

      GemmParams gp;
      gp.m = group.rows;
      gp.n = group.cols;
      gp.k = group.rows;
      gp.batch_count = group.num_params;
      gp.stride_a = min_sq * min_sq;
      gp.stride_b = mat;
      gp.stride_c = mat;
      nanochat::kernels::Gemm(GemmMode::kForward, gp, b, x, prod);
    }
  }
}

// One group's parameter and optimizer-state buffers, sized like the ones
// `NanochatOptimizer::Build` allocates.
struct GroupState {
  explicit GroupState(const Group& group)
      : values(static_cast<int>(Total(group))),
        master(static_cast<int>(Total(group))),
        grads(static_cast<int>(Total(group))),
        buf1(static_cast<int>(Total(group))),
        buf2(static_cast<int>(
            static_cast<std::int64_t>(group.num_params) *
            (group.rows >= group.cols ? group.rows : group.cols))) {}

  DevBuf<ComputeType> values;
  DevBuf<float> master;
  DevBuf<float> grads;
  DevBuf<float> buf1;
  DevBuf<float> buf2;
};

std::string ShapeLabel(const Profile& profile) { return profile.name; }

std::string GroupLabel(const Group& group) {
  return std::to_string(group.rows) + "x" + std::to_string(group.cols);
}

// One model shape's benchmark: the groups, the shared Polar Express workspace,
// and the per-group state.
// One `MuonUpdate` call, with the scalars `NanochatOptimizer::Step` resolves
// for these tests.
void RunMuon(const Group& group, GroupState& state, int ns_steps) {
  MuonParams p;
  p.num_params = group.num_params;
  p.rows = group.rows;
  p.cols = group.cols;
  p.lr = 0.02f;
  p.momentum = 0.95f;
  p.beta2 = 0.9f;
  p.weight_decay = 0.01f;
  p.ns_steps = ns_steps;
  p.red_dim = group.rows >= group.cols ? -1 : -2;
  p.nesterov = true;
  nanochat::kernels::MuonUpdate(p, state.grads.ptr, state.master.ptr,
                                state.values.ptr, state.buf1.ptr,
                                state.buf2.ptr);
}

struct Bench {
  Profile profile;
  std::vector<Group> groups;
  std::vector<std::unique_ptr<GroupState>> states;
  DevBuf<ComputeType> workspace;  // [x | prod | a | a2 | b], widest group

  Bench(const Profile& p, std::vector<Group> g, std::size_t workspace_len)
      : profile(p),
        groups(std::move(g)),
        workspace(static_cast<int>(workspace_len)) {
    for (const Group& group : groups) {
      states.push_back(std::make_unique<GroupState>(group));
    }
  }

  // The workspace slices of one group, at that group's offsets.
  void Slice(const Group& group, ComputeType** x, ComputeType** prod,
             ComputeType** a, ComputeType** a2, ComputeType** b) const {
    const std::int64_t total = Total(group);
    const std::int64_t min_total = MinTotal(group);
    *x = workspace.ptr;
    *prod = *x + total;
    *a = *prod + total;
    *a2 = *a + min_total;
    *b = *a2 + min_total;
  }

  double PolarStepMs(const BenchOptions& opt) {
    EventTimer timer;
    return timer.Time(
        [&] {
          for (const Group& group : groups) {
            ComputeType *x, *prod, *a, *a2, *b;
            Slice(group, &x, &prod, &a, &a2, &b);
            PolarIteration(group, x, prod, a, a2, b);
          }
        },
        opt);
  }

  // One group's optimizer phase at `ns_steps` iterations.
  double MuonGroupMs(const BenchOptions& opt, std::size_t index, int ns_steps) {
    EventTimer timer;
    return timer.Time(
        [&] {
          const Group& group = groups[index];
          GroupState& state = *states[index];
          RunMuon(group, state, ns_steps);
        },
        opt);
  }

  // The whole optimizer phase of one step, over every group, at `ns_steps`
  // Polar Express iterations. Zero iterations leaves the momentum, the
  // equilibration, the renormalisation, the variance reduction, and the
  // update, so the difference against five is the loop's own cost.
  double MuonStepMs(const BenchOptions& opt, int ns_steps) {
    EventTimer timer;
    return timer.Time(
        [&] {
          for (std::size_t i = 0; i < groups.size(); ++i) {
            RunMuon(groups[i], *states[i], ns_steps);
          }
        },
        opt);
  }
};

std::int64_t WorkspaceLen(const std::vector<Group>& groups) {
  std::int64_t len = 0;
  for (const Group& group : groups) {
    const std::int64_t needed = 2 * Total(group) + 3 * MinTotal(group);
    len = std::max(len, needed);
  }
  return len;
}

double StepFlops(const std::vector<Group>& groups) {
  double flops = 0.0;
  for (const Group& group : groups) flops += IterationFlops(group);
  return flops * kPolarIters;
}

const Profile* FindProfile(const std::string& name) {
  for (const Profile& profile : kProfiles) {
    if (name == profile.name) return &profile;
  }
  return nullptr;
}

void PrintGroups(const Bench& bench) {
  std::printf("%s: %d layers, hidden %d, %d heads, %d key/value heads\n",
              bench.profile.name, bench.profile.layers, bench.profile.hidden,
              bench.profile.heads, bench.profile.kv_heads);
  for (const Group& group : bench.groups) {
    std::printf("  group %-10s batch %-3d %8.3f MFLOP/step\n",
                GroupLabel(group).c_str(), group.num_params,
                IterationFlops(group) * kPolarIters / 1e6);
  }
  std::printf("  orthogonalisation %8.3f MFLOP/step\n\n",
              StepFlops(bench.groups) / 1e6);
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("bench");

  BenchOptions opt;
  opt.warmup = 1;
  opt.iters = 1;
  opt.rounds = 3;
  std::vector<std::string> wanted = {"h100", "big"};
  Profile custom = {"custom", 0, 0, 0, 0};
  bool muon = false;
  bool verbose = true;

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto next = [&](int* out) {
      if (i + 1 < argc) *out = std::atoi(argv[++i]);
    };
    if (arg == "--profile" && i + 1 < argc) {
      wanted = {argv[++i]};
    } else if (arg == "--layers") {
      next(&custom.layers);
    } else if (arg == "--hidden") {
      next(&custom.hidden);
    } else if (arg == "--heads") {
      next(&custom.heads);
    } else if (arg == "--kv-heads") {
      next(&custom.kv_heads);
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
    } else if (arg == "--muon") {
      muon = true;
    } else {
      std::fprintf(stderr,
                   "usage: %s [--profile h100|big] [--layers N "
                   "--hidden N --heads N --kv-heads N] [--iters N] "
                   "[--rounds N] [--warmup N] [--muon] [--json] "
                   "[--out PATH]\n",
                   argv[0]);
      return 2;
    }
  }
  if (custom.layers > 0) {
    custom.name = "custom";
    wanted = {"custom"};
  }

  std::printf(
      "muon benchmark: compute dtype %s, %d Polar Express iterations\n\n",
      nanochat::kComputeIsBf16 ? "bf16" : "fp32", kPolarIters);

  for (const std::string& name : wanted) {
    const Profile* profile = name == "custom" ? &custom : FindProfile(name);
    if (profile == nullptr) {
      std::fprintf(stderr, "muon benchmark: unknown profile %s\n",
                   name.c_str());
      return 2;
    }
    if (profile->layers <= 0 || profile->hidden <= 0 || profile->heads <= 0 ||
        profile->kv_heads <= 0) {
      std::fprintf(stderr,
                   "muon benchmark: profile %s needs --layers, --hidden, "
                   "--heads, and --kv-heads\n",
                   profile->name);
      return 2;
    }

    Bench bench(*profile, GroupsFor(*profile),
                static_cast<std::size_t>(WorkspaceLen(GroupsFor(*profile))));
    if (verbose) PrintGroups(bench);

    BenchReport report;
    const std::string shape = ShapeLabel(bench.profile);
    for (std::size_t i = 0; i < bench.groups.size(); ++i) {
      const Group& group = bench.groups[i];
      const double flops = IterationFlops(group) * kPolarIters;
      EventTimer timer;
      const double ms = timer.Time(
          [&] {
            ComputeType *x, *prod, *a, *a2, *b;
            bench.Slice(group, &x, &prod, &a, &a2, &b);
            PolarIteration(group, x, prod, a, a2, b);
          },
          opt);
      report.Add("polar:" + shape + ":" + GroupLabel(group),
                 "batch=" + std::to_string(group.num_params), ms,
                 1e-9 * flops / (ms * 1e-3));
    }

    const double step_flops = StepFlops(bench.groups);
    const double polar_ms = bench.PolarStepMs(opt);
    report.Add("polar:" + shape + ":step",
               std::to_string(bench.groups.size()) + " groups", polar_ms,
               1e-9 * step_flops / (polar_ms * 1e-3));
    if (muon) {
      for (std::size_t i = 0; i < bench.groups.size(); ++i) {
        const Group& group = bench.groups[i];
        const double one_flops = IterationFlops(group) * kPolarIters;
        const double ms = bench.MuonGroupMs(opt, i, kPolarIters);
        report.Add("muon:" + shape + ":" + GroupLabel(group),
                   "batch=" + std::to_string(group.num_params), ms,
                   1e-9 * one_flops / (ms * 1e-3));
      }
      for (int ns_steps : {0, 1, kPolarIters}) {
        const double muon_ms = bench.MuonStepMs(opt, ns_steps);
        report.Add(
            "muon:" + shape + ":step",
            std::to_string(bench.groups.size()) +
                " groups, ns=" + std::to_string(ns_steps),
            muon_ms,
            1e-9 * step_flops * ns_steps / kPolarIters / (muon_ms * 1e-3));
      }
    }
    report.Print(opt);
    if (verbose) std::printf("\n");
  }
  return 0;
}
