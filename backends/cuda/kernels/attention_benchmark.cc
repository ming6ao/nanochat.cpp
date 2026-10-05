// Standalone attention-family benchmark. It measures the shipped
// `AttentionForward` and `AttentionBackward` with device events at the
// `d8_s512` training shape, a sliding-window variant, a grouped-query variant,
// the production `SSSL` mixed-window pattern, and a head-dimension sweep.
//
// Every row carries an "issue/need" column: the work the shipped path issues
// over the work the visible (causal or windowed) query-key pairs need. The
// shipped path runs dense cuBLAS GEMMs and masks afterwards, so a windowed
// layer costs the same as a full layer and the ratio shows the mask waste. See
// docs/attention.md for the frozen Phase 0 numbers.
//
// The head-dimension sweep is the diagnostic that matters for the known
// redundancy in `OnlineSoftmaxTile` (backends/cuda/kernels/device_utils.cuh):
// the forward tile recomputes the query-key dot product once per owned output
// dimension, so the issued work scales about quadratically with `head_dim`
// while the necessary work scales linearly. Fitting the sweep exponent
// confirms or rejects that root cause before any fix.
//
// Run it through the project entry point (which takes the GPU broker):
//
//   tools/nanochat profile --json --out /tmp/attention.json
//
// See docs/performance.md.

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

#include "backends/cuda/kernels/testing/bench_utils.h"
#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::AttentionParams;
using nanochat::ComputeType;
using nanochat::dev::BenchOptions;
using nanochat::dev::BenchReport;
using nanochat::dev::DevBuf;
using nanochat::dev::EventTimer;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;

// Visible query-key pairs summed over query rows for the causal and optional
// sliding-window mask. This is independent of the head dimension, so it
// isolates the per-pair arithmetic when the sweep varies `head_dim`.
long long VisiblePairs(const AttentionParams& p) {
  const long long kv_len = p.kv_len > 0 ? p.kv_len : p.seq;
  long long pairs = 0;
  for (long long t = 0; t < p.seq; ++t) {
    const long long qpos = kv_len - p.seq + t;
    for (long long j = 0; j < kv_len; ++j) {
      if (p.causal && j > qpos) continue;
      if (p.window_left >= 0 && qpos - j > p.window_left) continue;
      if (p.window_right >= 0 && j - qpos > p.window_right) continue;
      ++pairs;
    }
  }
  return pairs;
}

// Measured cost of one attention call plus the work accounting.
struct ShapeResult {
  double fwd_ms = 0.0;
  double bwd_ms = 0.0;
  double fwd_gflop = 0.0;  // total necessary forward GFLOP
  double bwd_gflop = 0.0;  // total necessary backward GFLOP
  double issued_pairs = 0.0;
  double necessary_pairs = 0.0;
};

// Time one attention shape. `issued_pairs` counts the dense query-key pairs
// the shipped GEMMs touch; `necessary_pairs` counts only the visible pairs.
// Their ratio is the mask waste the fused kernel must remove.
ShapeResult MeasureShape(Rng* rng, const BenchOptions& opt,
                         const AttentionParams& p) {
  const int dim = p.head_dim;
  const int heads = p.num_heads;
  const int kv_heads = p.num_kv_heads > 0 ? p.num_kv_heads : heads;
  const int kv_len = p.kv_len > 0 ? p.kv_len : p.seq;
  const int q_count = p.batch * p.seq * heads * dim;
  const int kv_count = p.batch * kv_len * kv_heads * dim;
  const int stats_count = nanochat::AttentionStatsCount(p);

  DevBuf<ComputeType> q(ToStorage(RandomVec(q_count, rng)));
  DevBuf<ComputeType> k(ToStorage(RandomVec(kv_count, rng)));
  DevBuf<ComputeType> v(ToStorage(RandomVec(kv_count, rng)));
  DevBuf<ComputeType> out(q_count);
  DevBuf<float> stats(stats_count);
  DevBuf<ComputeType> dout(ToStorage(RandomVec(q_count, rng)));
  DevBuf<ComputeType> dq(q_count);
  DevBuf<ComputeType> dk(kv_count);
  DevBuf<ComputeType> dv(kv_count);

  EventTimer timer;
  ShapeResult r;
  r.fwd_ms = timer.Time(
      [&] {
        nanochat::kernels::AttentionForward(p, q.ptr, k.ptr, v.ptr, out.ptr,
                                            stats.ptr);
      },
      opt);
  r.bwd_ms = timer.Time(
      [&] {
        nanochat::kernels::AttentionBackward(p, q.ptr, k.ptr, v.ptr, stats.ptr,
                                             dout.ptr, dq.ptr, dk.ptr, dv.ptr);
      },
      opt);

  r.necessary_pairs = static_cast<double>(p.batch) * heads *
                      static_cast<double>(VisiblePairs(p));
  r.issued_pairs = static_cast<double>(p.batch) * heads *
                   static_cast<double>(p.seq) * static_cast<double>(kv_len);
  // Necessary floating-point operations per query-key pair: one dot product
  // (2*dim) plus one value accumulation (2*dim) = 4*dim for the forward. The
  // backward adds about twice that, matching the reference 12 = 4 + 8
  // accounting. Reporting efficiency against the necessary count makes the
  // redundancy visible: a correct tile should approach the device ceiling.
  r.fwd_gflop = 4.0 * r.necessary_pairs * dim / 1e9;
  r.bwd_gflop = 8.0 * r.necessary_pairs * dim / 1e9;
  return r;
}

void BenchShape(BenchReport* report, Rng* rng, const BenchOptions& opt,
                const AttentionParams& p, const char* label) {
  const ShapeResult r = MeasureShape(rng, opt, p);
  const double ratio =
      r.necessary_pairs > 0.0 ? r.issued_pairs / r.necessary_pairs : 0.0;
  const int kv_heads = p.num_kv_heads > 0 ? p.num_kv_heads : p.num_heads;

  char shape[128];
  std::snprintf(shape, sizeof(shape), "B=%d T=%d H=%d KV=%d D=%d wl=%d",
                p.batch, p.seq, p.num_heads, kv_heads, p.head_dim,
                p.window_left);
  char name[96];
  std::snprintf(name, sizeof(name), "attention_fwd:%s", label);
  report->Add(name, shape, r.fwd_ms, r.fwd_gflop / (r.fwd_ms / 1e3), ratio);
  std::snprintf(name, sizeof(name), "attention_bwd:%s", label);
  report->Add(name, shape, r.bwd_ms, r.bwd_gflop / (r.bwd_ms / 1e3), ratio);
}

// Average per-layer cost over a mixed window pattern such as the production
// `SSSL`. The shipped path is dense, so every layer costs the same; the pattern
// changes only the necessary work. The row reports the mean over one pattern
// cycle, which is the per-layer cost the model pays. `S` maps to the short
// window (a quarter of the context, rounded up to a tile) and every other
// letter maps to full context; the final entry is always full, matching
// `Config::window_left`.
void BenchPattern(BenchReport* report, Rng* rng, const BenchOptions& opt,
                  const AttentionParams& base, const std::string& pattern,
                  const char* label) {
  const int short_window = ((base.seq / 4 + 127) / 128) * 128;
  std::map<int, int> counts;
  for (std::size_t i = 0; i < pattern.size(); ++i) {
    const bool full = (i + 1 == pattern.size()) || pattern[i] != 'S';
    counts[full ? -1 : short_window] += 1;
  }

  const double n = static_cast<double>(pattern.size());
  double fwd_ms = 0.0;
  double bwd_ms = 0.0;
  double fwd_gflop = 0.0;
  double bwd_gflop = 0.0;
  double issued = 0.0;
  double necessary = 0.0;
  for (const auto& entry : counts) {
    AttentionParams p = base;
    p.window_left = entry.first;
    const ShapeResult r = MeasureShape(rng, opt, p);
    const double weight = static_cast<double>(entry.second) / n;
    fwd_ms += weight * r.fwd_ms;
    bwd_ms += weight * r.bwd_ms;
    fwd_gflop += weight * r.fwd_gflop;
    bwd_gflop += weight * r.bwd_gflop;
    issued += weight * r.issued_pairs;
    necessary += weight * r.necessary_pairs;
  }

  const double ratio = necessary > 0.0 ? issued / necessary : 0.0;
  const int kv_heads =
      base.num_kv_heads > 0 ? base.num_kv_heads : base.num_heads;
  char shape[160];
  std::snprintf(shape, sizeof(shape), "B=%d T=%d H=%d KV=%d D=%d pattern=%s",
                base.batch, base.seq, base.num_heads, kv_heads, base.head_dim,
                pattern.c_str());
  char name[96];
  std::snprintf(name, sizeof(name), "attention_fwd:%s", label);
  report->Add(name, shape, fwd_ms, fwd_gflop / (fwd_ms / 1e3), ratio);
  std::snprintf(name, sizeof(name), "attention_bwd:%s", label);
  report->Add(name, shape, bwd_ms, bwd_gflop / (bwd_ms / 1e3), ratio);
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("bench");

  BenchOptions opt;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--json") {
      opt.json = true;
    } else if (arg == "--out" && i + 1 < argc) {
      opt.out = argv[++i];
    } else if (arg == "--warmup" && i + 1 < argc) {
      opt.warmup = std::atoi(argv[++i]);
    } else if (arg == "--iters" && i + 1 < argc) {
      opt.iters = std::atoi(argv[++i]);
    } else if (arg == "--rounds" && i + 1 < argc) {
      opt.rounds = std::atoi(argv[++i]);
    } else {
      std::fprintf(stderr,
                   "usage: attention_bench [--json] [--out PATH] "
                   "[--warmup N] [--iters N] [--rounds N]\n");
      return 2;
    }
  }

  if (!opt.json) {
    std::printf("attention benchmark: device events, best of %d rounds\n",
                opt.rounds);
  }

  Rng rng;
  BenchReport report;

  // The production `d8_s512` training shape: 8 layers, batch 8, sequence 512,
  // 4 query heads, full context.
  AttentionParams base;
  base.batch = 8;
  base.seq = 512;
  base.num_heads = 4;
  base.num_kv_heads = 4;
  base.head_dim = 128;
  base.causal = true;
  base.window_left = -1;
  base.window_right = 0;
  base.kv_len = 0;
  BenchShape(&report, &rng, opt, base, "d8_s512");

  // Sliding window over a quarter of the context.
  AttentionParams windowed = base;
  windowed.window_left = 128;
  BenchShape(&report, &rng, opt, windowed, "d8_s512_win128");

  // Grouped-query attention: 8 query heads sharing 2 key/value heads.
  AttentionParams gqa = base;
  gqa.batch = 2;
  gqa.num_heads = 8;
  gqa.num_kv_heads = 2;
  BenchShape(&report, &rng, opt, gqa, "gqa_b2_t512_h8_kv2");

  // Production mixed pattern: three sliding-window layers to one full layer.
  // The shipped path is dense, so the time matches a single layer while the
  // necessary work drops; the row is the mean over one `SSSL` cycle.
  BenchPattern(&report, &rng, opt, base, "SSSL", "d8_s512_sssl");

  // Head-dimension sweep. Small batch and sequence length keep the sweep
  // quick; the point is the exponent, not the absolute time.
  for (int dim : {32, 64, 128, 256}) {
    AttentionParams sweep;
    sweep.batch = 2;
    sweep.seq = 128;
    sweep.num_heads = 4;
    sweep.num_kv_heads = 4;
    sweep.head_dim = dim;
    sweep.causal = true;
    sweep.window_left = -1;
    sweep.window_right = 0;
    sweep.kv_len = 0;
    char label[32];
    std::snprintf(label, sizeof(label), "dim_sweep_d%d", dim);
    BenchShape(&report, &rng, opt, sweep, label);
  }

  report.Print(opt);
  return 0;
}
