// The rate report of the Hopper attention forward tile of
// `docs/attention.md` Part 8, over the production rows of the family.
//
// The rows are the forward half of `attention_benchmark.cc`, at the same shapes
// and the same accounting, so a row here and a row there are one number twice:
// `gflops` is the necessary work over the elapsed time, `work_ratio` is the
// dense issued work over the necessary work, and `mfu` is the rate over the
// card's declared peak. The two programs differ in one thing: this one calls
// `LaunchHopperAttention` directly, because its target states
// `archs = ["sm_90a"]` and no other build compiles the tile.
//
// So the comparison the tile exists for is a run of this program beside a run
// of `attention_bench`, on the same card at the same precision. The tile covers
// `head_dim` 128 only, so every row here is a width of 128.
//
//   modal run tools/modal/app.py::card_job --name attention_hopper_bench
//   (the job takes the benchmark's own arguments after --argv)
//
// The measurement is the one of
// `backends/cuda/kernels/testing/bench_protocol.h`'s sibling `EventTimer`: a
// window of `--iters` calls, `--rounds` windows, the minimum of the windows,
// and a warm-up of `--warmup` calls before the first. One call is one launch,
// and no row takes a shortcut through a shape.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "backends/cuda/kernels/attention_dispatch.h"
#include "backends/cuda/kernels/attention_hopper.h"
#include "backends/cuda/kernels/testing/bench_utils.h"
#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::AttentionParams;
using nanochat::AttentionStatsCount;
using nanochat::ComputeType;
using nanochat::dev::BenchOptions;
using nanochat::dev::BenchReport;
using nanochat::dev::DevBuf;
using nanochat::dev::EventTimer;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;

// The visible query-key pairs of one head: the necessary work of the row. It is
// the count `attention_benchmark` states, so the two reports join.
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

// Times one shape and states its row: the necessary work of the forward over
// the elapsed time, and the dense issued pairs over the necessary ones.
void BenchShape(BenchReport* report, Rng* rng, const BenchOptions& opt,
                const AttentionParams& p, const char* label) {
  const int dim = p.head_dim;
  const int heads = p.num_heads;
  const int kv_heads = p.num_kv_heads > 0 ? p.num_kv_heads : heads;
  const int kv_len = p.kv_len > 0 ? p.kv_len : p.seq;
  const int q_count = p.batch * p.seq * heads * dim;
  const int kv_count = p.batch * kv_len * kv_heads * dim;
  const int stats_count = AttentionStatsCount(p);

  DevBuf<ComputeType> q(ToStorage(RandomVec(q_count, rng)));
  DevBuf<ComputeType> k(ToStorage(RandomVec(kv_count, rng)));
  DevBuf<ComputeType> v(ToStorage(RandomVec(kv_count, rng)));
  DevBuf<ComputeType> out(q_count);
  DevBuf<float> stats(stats_count);

  auto call = [&] {
    nanochat::kernels::LaunchHopperAttention(p, q.ptr, k.ptr, v.ptr, out.ptr,
                                             stats.ptr);
  };

  EventTimer timer;
  const double fwd_ms = timer.Time(call, opt);

  // The host side of the same call: the four tensor maps and the launch, which
  // the event timer cannot separate from device work. A launch that the host
  // cannot feed makes the device wait inside the timed span, so a row whose
  // host time meets its device time is a row where the two are one number.
  double host_ms = 1e300;
  for (int round = 0; round < opt.rounds; ++round) {
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < opt.iters; ++i) call();
    const auto end = std::chrono::steady_clock::now();
    const double per_call =
        std::chrono::duration<double, std::milli>(end - begin).count() /
        opt.iters;
    host_ms = std::min(host_ms, per_call);
  }
  nanochat::kernels::Synchronize();
  if (!opt.json) {
    std::printf("  %-34s device %.4f ms  host %.4f ms\n", label, fwd_ms,
                host_ms);
  }

  const double necessary = static_cast<double>(p.batch) * heads *
                           static_cast<double>(VisiblePairs(p));
  const double issued = static_cast<double>(p.batch) * heads *
                        static_cast<double>(p.seq) *
                        static_cast<double>(kv_len);
  // One query-key pair costs one dot product and one value accumulation, so the
  // forward is four products of the head width a pair.
  const double fwd_gflop = 4.0 * necessary * dim / 1e9;

  char shape[128];
  std::snprintf(shape, sizeof(shape), "B=%d T=%d H=%d KV=%d D=%d wl=%d",
                p.batch, p.seq, p.num_heads, kv_heads, p.head_dim,
                p.window_left);
  char name[96];
  std::snprintf(name, sizeof(name), "attention_hopper_fwd:%s", label);
  report->SetRowIdentity("attention", "forward");
  report->Add(name, shape, fwd_ms, fwd_gflop / (fwd_ms / 1e3),
              necessary > 0.0 ? issued / necessary : 0.0);
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("bench");
  // A measurement names its arm, so this report is of the tile whatever the
  // promotion policy answers (`docs/attention.md` Part 8, section 7).
  setenv("NANOCHAT_ATTENTION_LIB", "hopper", 1);

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
                   "usage: attention_hopper_bench [--json] [--out PATH] "
                   "[--warmup N] [--iters N] [--rounds N]\n");
      return 2;
    }
  }

  if (!opt.json) {
    std::printf(
        "hopper attention benchmark: device events, best of %d rounds\n",
        opt.rounds);
  }

  Rng rng;
  BenchReport report;
  report.UseDevicePeak(nanochat::dev::BenchDeviceName());

  // Whether the tile answered for this card at all. A card that refuses it
  // states the reason and no rate, because a row of the stand-in would be a
  // launch that did nothing.
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
  if (!nanochat::kernels::UseHopperAttention(base)) {
    std::fprintf(stderr,
                 "attention_hopper_bench: this build or this card does not "
                 "take the Hopper tile; the rows of the family are the ones "
                 "attention_bench states\n");
    return 3;
  }

  // The production `d8_s512` training shape and its sliding-window twin over a
  // quarter of the context.
  BenchShape(&report, &rng, opt, base, "d8_s512");
  AttentionParams windowed = base;
  windowed.window_left = 128;
  BenchShape(&report, &rng, opt, windowed, "d8_s512_win128");

  // The grouped-query shape: eight query heads over two key/value heads.
  AttentionParams gqa = base;
  gqa.batch = 2;
  gqa.num_heads = 8;
  gqa.num_kv_heads = 2;
  BenchShape(&report, &rng, opt, gqa, "gqa_b2_t512_h8_kv2");

  // The long shape the tile is argued at, and the short window of the
  // production mixed pattern.
  AttentionParams long_shape = base;
  long_shape.batch = 4;
  long_shape.seq = 2048;
  long_shape.num_heads = 8;
  long_shape.num_kv_heads = 8;
  long_shape.window_left = 512;
  BenchShape(&report, &rng, opt, long_shape, "b4_s2048_h8_win512");
  AttentionParams short_window = base;
  short_window.window_left = 256;
  BenchShape(&report, &rng, opt, short_window, "d8_s512_short");

  // The key-count ladder at one grid size: each row holds 128 blocks, which is
  // one wave on the card, and the keys of a block rise from 512 to 2048, so the
  // key tiles a block walks rise from four to sixteen. A row that does not move
  // with that count states a cost that is not the products, and the shape is
  // what separates a floor of the launch from a floor of the block.
  AttentionParams ladder = base;
  ladder.batch = 8;
  ladder.seq = 512;
  ladder.kv_len = ladder.seq;
  BenchShape(&report, &rng, opt, ladder, "ladder_t512_4tiles");
  ladder.batch = 4;
  ladder.seq = 1024;
  ladder.kv_len = ladder.seq;
  BenchShape(&report, &rng, opt, ladder, "ladder_t1024_8tiles");
  ladder.batch = 2;
  ladder.seq = 2048;
  ladder.kv_len = ladder.seq;
  BenchShape(&report, &rng, opt, ladder, "ladder_t2048_16tiles");

  // The floor ladder at one wave: 128 blocks whose longest block walks one,
  // two, or four key tiles. A row that reads the same at one tile as at four
  // states a floor of the block and not of the products.
  AttentionParams floor = base;
  floor.num_heads = 4;
  floor.num_kv_heads = 4;
  floor.batch = 32;
  floor.seq = 128;
  floor.kv_len = floor.seq;
  BenchShape(&report, &rng, opt, floor, "floor_1tile");
  floor.batch = 16;
  floor.seq = 256;
  floor.kv_len = floor.seq;
  BenchShape(&report, &rng, opt, floor, "floor_2tile_causal");
  floor.window_left = 128;
  BenchShape(&report, &rng, opt, floor, "floor_2tile_window");

  // The grid ladder: one shape and one longest block, and a block count that
  // rises from one wave to four. A time that rises with the waves is a cost of
  // a block, and a time that does not is a cost of the launch.
  AttentionParams grid = base;
  grid.seq = 512;
  grid.kv_len = grid.seq;
  grid.batch = 8;
  BenchShape(&report, &rng, opt, grid, "grid_1wave");
  grid.batch = 16;
  BenchShape(&report, &rng, opt, grid, "grid_2waves");
  grid.batch = 32;
  BenchShape(&report, &rng, opt, grid, "grid_4waves");

  // The head-major query: the same numbers in the layout the fused `qk_prep`
  // writes, which is the layout the model runs.
  AttentionParams head_major = windowed;
  head_major.q_head_major = true;
  BenchShape(&report, &rng, opt, head_major, "d8_s512_win128_head_major");

  report.Print(opt);
  return 0;
}
