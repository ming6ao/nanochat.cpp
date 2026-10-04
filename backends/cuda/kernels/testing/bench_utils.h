#ifndef NANOCHAT_DEV_KERNELS_BENCH_UTILS_H_
#define NANOCHAT_DEV_KERNELS_BENCH_UTILS_H_

// Host-side benchmarking helpers for the kernel benchmarks. CUDA-event timing with
// warm-up and best-of-rounds, plus a small JSON report that agents and scripts
// can consume. This header uses the CUDA runtime, so it is only included by
// targets built as `cuda_binary` (the host pass of nvcc). See
// docs/performance.md and the `performance-investigation` skill.
//
// The events are recorded on the legacy default stream, the same stream the
// backend kernels use (backends/cuda/device.h), so ordering is guaranteed
// without an explicit stream handle. Timing therefore measures device time,
// not host dispatch time.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "nanochat/kernels.h"

namespace nanochat {
namespace dev {

// Benchmark controls. `iters` calls are timed per round; the reported value is
// the best (minimum) mean per call across `rounds` rounds. Warm-up calls are
// not timed. Defaults are tuned for millisecond-scale attention kernels: few
// calls, because each call is expensive and event resolution is already fine.
struct BenchOptions {
  int warmup = 2;
  int iters = 3;
  int rounds = 3;
  bool json = false;
  std::string out;  // optional path to also write the JSON report
};

// One measured feature.
struct BenchRow {
  std::string name;
  std::string shape;
  double ms = 0.0;
  double gflops = 0.0;  // achieved GFLOP/s using the necessary operation count
  // Issued work over necessary work, when the benchmark accounts for both.
  // Zero means the benchmark does not report the ratio.
  double work_ratio = 0.0;
};

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

// Collects rows and emits a human-readable table, a JSON document, or both.
// The JSON schema is `nanochat.bench.v1`.
class BenchReport {
 public:
  void Add(const std::string& name, const std::string& shape, double ms,
           double gflops = 0.0, double work_ratio = 0.0) {
    rows_.push_back(BenchRow{name, shape, ms, gflops, work_ratio});
  }

  // Print to stdout, and write the JSON report to `opt.out` when set.
  void Print(const BenchOptions& opt) const {
    if (!opt.out.empty()) WriteFile(opt.out, Json());
    if (opt.json) {
      std::fputs(Json().c_str(), stdout);
      std::fputc('\n', stdout);
      return;
    }
    std::size_t name_width = 4;
    std::size_t shape_width = 5;
    for (const BenchRow& row : rows_) {
      name_width = std::max(name_width, row.name.size());
      shape_width = std::max(shape_width, row.shape.size());
    }
    std::printf("%-*s  %-*s  %10s  %10s  %10s\n", static_cast<int>(name_width),
                "name", static_cast<int>(shape_width), "shape", "ms",
                "GFLOP/s", "issue/need");
    for (const BenchRow& row : rows_) {
      char ratio[16];
      if (row.work_ratio > 0.0) {
        std::snprintf(ratio, sizeof(ratio), "%.3f", row.work_ratio);
      } else {
        std::snprintf(ratio, sizeof(ratio), "-");
      }
      std::printf("%-*s  %-*s  %10.4f  %10.1f  %10s\n",
                  static_cast<int>(name_width), row.name.c_str(),
                  static_cast<int>(shape_width), row.shape.c_str(), row.ms,
                  row.gflops, ratio);
    }
  }

 private:
  std::string Json() const {
    std::string text = "{\"schema\":\"nanochat.bench.v1\",\"rows\":[";
    for (std::size_t i = 0; i < rows_.size(); ++i) {
      const BenchRow& row = rows_[i];
      char buffer[640];
      std::snprintf(buffer, sizeof(buffer),
                    "%s{\"name\":\"%s\",\"shape\":\"%s\",\"ms\":%.6f,"
                    "\"gflops\":%.6f,\"work_ratio\":%.6f}",
                    i == 0 ? "" : ",", Escape(row.name).c_str(),
                    Escape(row.shape).c_str(), row.ms, row.gflops,
                    row.work_ratio);
      text += buffer;
    }
    text += "]}";
    return text;
  }

  // Minimal JSON string escaping for the identifiers used in this report.
  static std::string Escape(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (char c : value) {
      if (c == '"' || c == '\\') out.push_back('\\');
      out.push_back(c);
    }
    return out;
  }

  static void WriteFile(const std::string& path, const std::string& text) {
    std::FILE* file = std::fopen(path.c_str(), "w");
    if (file == nullptr) {
      std::fprintf(stderr, "bench: cannot write %s\n", path.c_str());
      return;
    }
    std::fputs(text.c_str(), file);
    std::fputc('\n', file);
    std::fclose(file);
  }

  std::vector<BenchRow> rows_;
};

}  // namespace dev
}  // namespace nanochat

#endif  // NANOCHAT_DEV_KERNELS_BENCH_UTILS_H_
