// Standalone benchmark for the attention-tile-v3 prototype. It measures the
// fused fp32 tiled forward with device events at the production `d8_s512`
// training shape, then searches the tile space.
//
// Sections:
//   * the required `attention_tile_fwd:d8_s512` and
//     `attention_tile_fwd:d8_s512_win128` rows plus a grouped-query row;
//   * the tile-space sweep: Br, Bc, block size, and `__launch_bounds__`, one
//     axis at a time around the default tile;
//   * the head-dimension sweep, which confirms the tile has no per-owned-
//     dimension redundancy (the defect that made the older OnlineSoftmaxTile
//     scale about quadratically with `head_dim`).
//
// Each row carries an "issue/need" column: the dense query-key pairs the
// shipped path issues over the visible pairs the mask needs. The tile skips
// fully masked tiles, so its issue/need stays near one and the window win is
// visible in the time.
//
// A zeroed input keeps the benchmark deterministic and free of triangular
// denominators, so it measures the tile's launch, staging, and loop cost only.
//
// Run it through the project entry point (which takes the GPU broker):
//
//   tools/nanochat bench -- ./bazel-bin/dev/kernels/attention_tile_v3_bench
//   tools/nanochat bench -- ./bazel-bin/dev/kernels/attention_tile_v3_bench
//       --json --out /tmp/attention_tile_v3.json
//
// See docs/performance.md.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "attention_tile_v3.h"
#include "bench_utils.h"
#include "gpu_test_utils.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::AttentionParams;
using nanochat::ComputeType;
using nanochat::dev::AttentionTileV3Config;
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

// One measured tile configuration.
struct TileResult {
  double ms = 0.0;
  double gflops = 0.0;  // achieved GFLOP/s using the necessary operation count
  double work_ratio = 0.0;
  int status = 0;
};

// Time one tile configuration at one attention shape. `status` is nonzero when
// the configuration is not compiled or does not fit at this `head_dim`.
TileResult MeasureTile(Rng* rng, const BenchOptions& opt,
                       const AttentionParams& p,
                       const AttentionTileV3Config& config) {
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

  TileResult r;
  // The first call doubles as the status probe and the first warm-up.
  r.status = nanochat::dev::AttentionTileV3Forward(p, q.ptr, k.ptr, v.ptr,
                                                   out.ptr, stats.ptr, config);
  if (r.status != 0) return r;

  EventTimer timer;
  r.ms = timer.Time(
      [&] {
        nanochat::dev::AttentionTileV3Forward(p, q.ptr, k.ptr, v.ptr, out.ptr,
                                              stats.ptr, config);
      },
      opt);

  const double necessary_pairs =
      static_cast<double>(p.batch) * heads *
      static_cast<double>(VisiblePairs(p));
  const double issued_pairs = static_cast<double>(p.batch) * heads *
                              static_cast<double>(p.seq) *
                              static_cast<double>(kv_len);
  const double necessary_gflop = 4.0 * necessary_pairs * dim / 1e9;
  r.gflops = necessary_gflop / (r.ms / 1e3);
  r.work_ratio = necessary_pairs > 0.0 ? issued_pairs / necessary_pairs : 0.0;
  return r;
}

// Appends one tile row. An unsupported configuration is reported on stderr and
// skipped so the sweep can name every axis without a guard at each call site.
void BenchTile(BenchReport* report, Rng* rng, const BenchOptions& opt,
               const AttentionParams& p, const AttentionTileV3Config& config,
               const char* label) {
  const TileResult r = MeasureTile(rng, opt, p, config);
  if (r.status != 0) {
    std::fprintf(stderr, "attention_tile_fwd:%s unsupported (status %d)\n",
                 label, r.status);
    return;
  }
  const int kv_heads = p.num_kv_heads > 0 ? p.num_kv_heads : p.num_heads;
  char shape[224];
  std::snprintf(shape, sizeof(shape),
                "B=%d T=%d H=%d KV=%d D=%d br=%d bc=%d thr=%d mb=%d wl=%d",
                p.batch, p.seq, p.num_heads, kv_heads, p.head_dim, config.br,
                config.bc, config.threads, config.min_blocks, p.window_left);
  char name[128];
  std::snprintf(name, sizeof(name), "attention_tile_fwd:%s", label);
  report->Add(name, shape, r.ms, r.gflops, r.work_ratio);
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
                   "usage: attention_tile_v3_bench [--json] [--out PATH] "
                   "[--warmup N] [--iters N] [--rounds N]\n");
      return 2;
    }
  }

  if (!opt.json) {
    std::printf(
        "attention_tile_v3 benchmark: device events, best of %d rounds\n",
        opt.rounds);
  }

  Rng rng;
  BenchReport report;
  const AttentionTileV3Config baseline{};

  // Production `d8_s512` training shape: batch 8, sequence 512, 4 query
  // heads, 4 key/value heads, head_dim 128, full causal context.
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
  BenchTile(&report, &rng, opt, base, baseline, "d8_s512");

  // Sliding window over a quarter of the context. The gate needs this row to
  // be at least 1.6x faster than the full-causal row above.
  AttentionParams windowed = base;
  windowed.window_left = 128;
  BenchTile(&report, &rng, opt, windowed, baseline, "d8_s512_win128");

  // Grouped-query attention: 8 query heads sharing 2 key/value heads.
  AttentionParams gqa = base;
  gqa.batch = 2;
  gqa.num_heads = 8;
  gqa.num_kv_heads = 2;
  BenchTile(&report, &rng, opt, gqa, baseline, "gqa_b2_t512_h8_kv2");

  // Tile-space sweep. The sweep shape keeps head_dim at 64 so every candidate
  // Bc fits under the Pascal 48 KB shared-memory cap; the d8_s512 rows above
  // report the production shape at the default tile.
  AttentionParams tile_shape;
  tile_shape.batch = 2;
  tile_shape.seq = 256;
  tile_shape.num_heads = 4;
  tile_shape.num_kv_heads = 4;
  tile_shape.head_dim = 64;
  tile_shape.causal = true;
  tile_shape.window_left = -1;
  tile_shape.window_right = 0;
  tile_shape.kv_len = 0;

  // Br sweep: query rows per block.
  for (int br : {16, 32, 64}) {
    AttentionTileV3Config config = baseline;
    config.br = br;
    char label[48];
    std::snprintf(label, sizeof(label), "tile_br%d", br);
    BenchTile(&report, &rng, opt, tile_shape, config, label);
  }

  // Bc sweep: key/value rows staged per shared-memory tile.
  for (int bc : {32, 64}) {
    AttentionTileV3Config config = baseline;
    config.bc = bc;
    char label[48];
    std::snprintf(label, sizeof(label), "tile_bc%d", bc);
    BenchTile(&report, &rng, opt, tile_shape, config, label);
  }

  // Block-size sweep: threads per block.
  for (int threads : {128, 256, 512}) {
    AttentionTileV3Config config = baseline;
    config.threads = threads;
    char label[48];
    std::snprintf(label, sizeof(label), "tile_thr%d", threads);
    BenchTile(&report, &rng, opt, tile_shape, config, label);
  }

  // Launch-bounds sweep: the `__launch_bounds__` residency hint.
  for (int min_blocks : {1, 2, 3}) {
    AttentionTileV3Config config = baseline;
    config.min_blocks = min_blocks;
    char label[48];
    std::snprintf(label, sizeof(label), "tile_mb%d", min_blocks);
    BenchTile(&report, &rng, opt, tile_shape, config, label);
  }

  // Head-dimension sweep. Small batch and sequence keep it quick; the point is
  // the shape of the curve, not the absolute time. The tile caps head_dim at
  // 128 by construction.
  for (int dim : {32, 64, 128}) {
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
    char label[48];
    std::snprintf(label, sizeof(label), "dim_sweep_d%d", dim);
    BenchTile(&report, &rng, opt, sweep, baseline, label);
  }

  report.Print(opt);
  return 0;
}
