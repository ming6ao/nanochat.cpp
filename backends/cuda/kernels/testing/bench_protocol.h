#ifndef NANOCHAT_DEV_KERNELS_BENCH_PROTOCOL_H_
#define NANOCHAT_DEV_KERNELS_BENCH_PROTOCOL_H_

// The measurement protocol two benchmarks share: one event pair around a window
// of calls, the minimum and the median of the rounds, the second-level cache
// flush and the subtraction of its cost, and the one rate the report states.
//
// The protocol lives here because a row of one binary must be comparable with a
// row of another. Two binaries that state two statistics cannot compare a row
// with a row: `backends/cuda/gemm/gemm_benchmark.cc` timed a window of calls
// and `dev/kernels/wgmma_bench.cu` timed one launch, and the two numbers were
// the minimum of three rounds and a single sample (docs/kernels.md section
// 11.4, docs/performance.md section 2).
//
// **The rate.** The rate is the necessary work over the elapsed time, in
// GFLOP/s. `BenchReport` turns it into the share of the card's declared peak,
// so `mfu` is the one share a reader compares. A benchmark therefore states a
// FLOP count and a call, and it divides by no peak by hand.
//
// **The rows.** `AddTiming` writes the two rows one measurement needs: the
// minimum under the arm's own name, and the median beside it. The primary row
// states the minimum, which is the statistic the rate gates of docs/kernels.md
// Part 3 read; the companion row states the median, which is the statistic a
// decision reads.
//
// This header uses the CUDA runtime and the seam, so only the host pass of a
// `cuda_binary` includes it.

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "backends/cuda/kernels/testing/bench_utils.h"
#include "nanochat/kernels.h"

namespace nanochat {
namespace dev {

// --- The timing protocol ----------------------------------------------------

// One row's samples: the fastest window, the middle window, and the mean.
struct Timing {
  double min_ms = 0.0;
  double median_ms = 0.0;
  double mean_ms = 0.0;
  int iters = 0;
};

// One event pair around one window of calls. A window holds the calls of one
// round, so no host gap between two calls is charged to a measurement.
class WindowTimer {
 public:
  WindowTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&end_);
  }
  ~WindowTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(end_);
  }
  WindowTimer(const WindowTimer&) = delete;
  WindowTimer& operator=(const WindowTimer&) = delete;

  // One sample for each round: the device time of one call, in milliseconds.
  template <typename Fn>
  std::vector<double> Samples(Fn&& fn, int rounds, int iters) {
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(rounds));
    for (int round = 0; round < rounds; ++round) {
      cudaEventRecord(start_, nullptr);
      for (int i = 0; i < iters; ++i) fn();
      cudaEventRecord(end_, nullptr);
      cudaEventSynchronize(end_);
      float elapsed_ms = 0.0f;
      cudaEventElapsedTime(&elapsed_ms, start_, end_);
      samples.push_back(static_cast<double>(elapsed_ms) / iters);
    }
    return samples;
  }

 private:
  cudaEvent_t start_{};
  cudaEvent_t end_{};
};

// The minimum, the median, and the mean of one sample set.
inline Timing Summarize(std::vector<double> samples, int iters) {
  Timing timing;
  timing.iters = iters;
  if (samples.empty()) return timing;
  std::sort(samples.begin(), samples.end());
  timing.min_ms = samples.front();
  timing.mean_ms = 0.0;
  for (double sample : samples) timing.mean_ms += sample;
  timing.mean_ms /= static_cast<double>(samples.size());
  timing.median_ms = samples[samples.size() / 2];
  if (samples.size() % 2 == 0) {
    timing.median_ms =
        0.5 * (samples[samples.size() / 2 - 1] + samples[samples.size() / 2]);
  }
  return timing;
}

// A buffer larger than the second-level cache, written before every timed call.
// The write is outside the timed window, and its cost is measured separately so
// that the window can be corrected for it.
class Flusher {
 public:
  Flusher() = default;
  ~Flusher() { Release(); }
  Flusher(const Flusher&) = delete;
  Flusher& operator=(const Flusher&) = delete;

  bool Init() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    int l2_bytes = 0;
    cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device);
    std::size_t bytes = l2_bytes > 0 ? static_cast<std::size_t>(l2_bytes) * 2
                                     : std::size_t{32} << 20;
    buffer_ = kernels::Alloc(bytes);
    if (buffer_ == nullptr) return false;
    bytes_ = bytes;
    return true;
  }

  void Flush() const {
    if (buffer_ != nullptr) kernels::Memset(buffer_, 0, bytes_);
  }

  std::size_t bytes() const { return bytes_; }

  void Release() {
    if (buffer_ == nullptr) return;
    kernels::Free(buffer_);
    buffer_ = nullptr;
    bytes_ = 0;
  }

 private:
  void* buffer_ = nullptr;
  std::size_t bytes_ = 0;
};

// One arm over one row. The flush runs before every call, inside the window, so
// the call never reads a warm cache. Its own cost is measured once and removed,
// which leaves the call's device time.
template <typename Fn>
Timing TimeWithFlush(WindowTimer* timer, Fn&& call, int iters, int rounds,
                     int warmup, const Flusher* flusher, double flush_ms) {
  for (int i = 0; i < warmup; ++i) {
    if (flusher != nullptr) flusher->Flush();
    call();
  }
  kernels::Synchronize();
  std::vector<double> samples = timer->Samples(
      [&] {
        if (flusher != nullptr) flusher->Flush();
        call();
      },
      rounds, iters);
  for (double& sample : samples) sample = std::max(0.0, sample - flush_ms);
  return Summarize(std::move(samples), iters);
}

// The cost of one flush, measured on its own loop so the window above can state
// a call's time and not the flush's.
inline double TimeFlush(WindowTimer* timer, const Flusher& flusher,
                        int rounds) {
  constexpr int kCalibrationIters = 8;
  for (int i = 0; i < kCalibrationIters; ++i) flusher.Flush();
  kernels::Synchronize();
  const std::vector<double> samples = timer->Samples(
      [&flusher] { flusher.Flush(); }, rounds, kCalibrationIters);
  return Summarize(samples, kCalibrationIters).min_ms;
}

// The rounds of the one calibration loop, which is a constant because the cost
// it measures is the same for every benchmark.
inline constexpr int kFlushCalibrationRounds = 5;

// The protocol as one object: the timer, the flush buffer, the measured cost of
// one flush, and the controls the caller set. A benchmark states its call and
// reads a `Timing` back, so no benchmark carries the six controls itself.
class Harness {
 public:
  explicit Harness(const BenchOptions& options, bool flush = true)
      : options_(options), flush_wanted_(flush) {}

  // Allocates the flush buffer and measures the cost of one flush. Returns
  // false when the caller asked for no flush or the buffer does not fit, and
  // the protocol then times a call with no flush.
  bool Init() {
    if (!flush_wanted_) return false;
    if (!flusher_.Init()) return false;
    flush_ms_ = TimeFlush(&timer_, flusher_, kFlushCalibrationRounds);
    flushing_ = true;
    return true;
  }

  // One measurement: `options_.iters` calls in each of `options_.rounds`
  // windows, after `options_.warmup` calls that are not timed.
  template <typename Fn>
  Timing Time(Fn&& call) {
    return TimeWithFlush(&timer_, call, std::max(1, options_.iters),
                         std::max(1, options_.rounds),
                         std::max(0, options_.warmup),
                         flushing_ ? &flusher_ : nullptr, flush_ms_);
  }

  std::size_t flush_bytes() const { return flusher_.bytes(); }
  double flush_ms() const { return flush_ms_; }

 private:
  BenchOptions options_;
  bool flush_wanted_ = true;
  WindowTimer timer_;
  Flusher flusher_;
  // True only after `Init` measured the cost of one flush.
  bool flushing_ = false;
  double flush_ms_ = 0.0;
};

// --- The rate ---------------------------------------------------------------

// The rate of `flops` over `ms`, in GFLOP/s. The count is the necessary work
// and not the issued work, so two arms of one row compare directly.
inline double Gflops(double flops, double ms) {
  if (!(ms > 0.0) || !(flops > 0.0)) return 0.0;
  return 1e-9 * flops / (ms * 1e-3);
}

// One measurement as two rows: the minimum under the row's own name, and the
// median beside it. The caller states the row in full, and that is the only
// form, so identity, shape, and the call's own facts travel with both rows and
// a reader takes the shape of the call from the report and not from the console
// (docs/kernels.md section 11.7). `BenchReport::Add` inherits the precision and
// not the family, so a benchmark of one family states that family here.
inline void AddTiming(BenchReport* report, const BenchRow& base,
                      const Timing& timing, double flops) {
  BenchRow row = base;
  row.ms = timing.min_ms;
  row.gflops = Gflops(flops, timing.min_ms);
  row.iterations = timing.iters;
  report->Add(row);
  row.name = base.name + ":median";
  row.ms = timing.median_ms;
  row.gflops = Gflops(flops, timing.median_ms);
  report->Add(row);
}

}  // namespace dev
}  // namespace nanochat

#endif  // NANOCHAT_DEV_KERNELS_BENCH_PROTOCOL_H_
