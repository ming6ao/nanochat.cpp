#ifndef NANOCHAT_DEV_KERNELS_BENCH_UTILS_H_
#define NANOCHAT_DEV_KERNELS_BENCH_UTILS_H_

// Host-side benchmarking helpers for the kernel benchmarks: CUDA-event timing
// with warm-up and best-of-rounds, plus the `nanochat.bench.v3` report.
//
// The report itself lives in `nanochat/bench.h`, so the CUDA benchmarks and the
// host-only harness benchmarks share one writer for one schema
// (docs/performance.md section 3.4). This header adds the two things that need
// the CUDA runtime:
//
//   * `EventTimer`, which records on the legacy default stream, the same stream
//     the backend kernels use (`backends/cuda/device.h`). Ordering is therefore
//     guaranteed without an explicit stream handle, so timing measures device
//     time and not host dispatch time.
//   * `BenchDeviceName`, which reads the card from the seam, and the
//     `UseDevicePeak` overload that names it. A benchmark then states its
//     denominator without carrying a device name of its own.
//
// This header uses the CUDA runtime, so only a target built as `cuda_binary`
// (the host pass of nvcc) may include it. See docs/performance.md and the
// `performance-investigation` skill.

#include <cuda_runtime.h>

#include <algorithm>
#include <string>

#include "nanochat/bench.h"
#include "nanochat/kernels.h"

namespace nanochat {
namespace dev {

// The names the report, its rows, and its controls have in this header. They
// are the public names, so a benchmark reads one vocabulary whether it includes
// the CUDA helpers or the host-only header.
using BenchOptions = nanochat::BenchOptions;
using BenchRow = nanochat::BenchRow;
using BenchReport = nanochat::BenchReport;

// The name of the card the process runs on, as the seam reports it. An empty
// string means the backend named no device.
inline std::string BenchDeviceName() {
  const Caps caps = kernels::GetCaps();
  return caps.device_name != nullptr ? std::string(caps.device_name)
                                     : std::string();
}

// States the denominator from the card the process runs on. It is the one-arg
// form of `BenchReport::UseDevicePeak`, and it is what a benchmark calls when
// it does not name a device itself.
inline double UseDevicePeak(BenchReport* report,
                            BuildPrecision precision = kBuildPrecision) {
  return report->UseDevicePeak(BenchDeviceName(), precision);
}

// Device-event timer. Non-copyable because it owns two CUDA events.
class EventTimer {
 public:
  EventTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&end_);
  }
  ~EventTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(end_);
  }
  EventTimer(const EventTimer&) = delete;
  EventTimer& operator=(const EventTimer&) = delete;

  // Time one callable and return the best mean milliseconds per call.
  template <typename Fn>
  double Time(Fn&& fn, const BenchOptions& opt) {
    for (int i = 0; i < opt.warmup; ++i) fn();
    kernels::Synchronize();
    double best = 1e300;
    for (int round = 0; round < opt.rounds; ++round) {
      cudaEventRecord(start_, nullptr);
      for (int i = 0; i < opt.iters; ++i) fn();
      cudaEventRecord(end_, nullptr);
      cudaEventSynchronize(end_);
      float elapsed_ms = 0.0f;
      cudaEventElapsedTime(&elapsed_ms, start_, end_);
      best = std::min(best, static_cast<double>(elapsed_ms) / opt.iters);
    }
    return best;
  }

 private:
  cudaEvent_t start_{};
  cudaEvent_t end_{};
};

}  // namespace dev
}  // namespace nanochat

#endif  // NANOCHAT_DEV_KERNELS_BENCH_UTILS_H_
