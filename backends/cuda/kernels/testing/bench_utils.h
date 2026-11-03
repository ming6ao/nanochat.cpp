#ifndef NANOCHAT_DEV_KERNELS_BENCH_UTILS_H_
#define NANOCHAT_DEV_KERNELS_BENCH_UTILS_H_

// Host-side benchmarking helpers for the kernel benchmarks. CUDA-event timing
// with warm-up and best-of-rounds, plus a small JSON report that agents and
// scripts can consume. This header uses the CUDA runtime, so it is only
// included by targets built as `cuda_binary` (the host pass of nvcc). See
// docs/performance.md and the `performance-investigation` skill.
//
// The events are recorded on the legacy default stream, the same stream the
// backend kernels use (backends/cuda/device.h), so ordering is guaranteed
// without an explicit stream handle. Timing therefore measures device time,
// not host dispatch time.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/device_profile.h"
#include "nanochat/kernels.h"
#include "nanochat/tensor.h"

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

// ---------------------------------------------------------------------------
// Model FLOPs utilization
//
// One report row may state its share of the peak dense rate of the card that
// ran (`docs/precision.md`). Three helpers supply it, and no benchmark carries
// a device name of its own: the card is read from `GetCaps()` and its rates
// from the entry the card declares in `backends/<backend>/caps.json`.
// ---------------------------------------------------------------------------

// The precision a benchmark's arithmetic uses. A benchmark names the one it
// issues, because the peak of one precision has nothing to say about the
// arithmetic of another.
enum class BenchPrecision { kFp32, kBf16, kFp8 };

// The build's compute precision, from `kBuildPrecision` in
// `nanochat/tensor.h`.
inline BenchPrecision BuildBenchPrecision() {
  switch (kBuildPrecision) {
    case BuildPrecision::kBf16:
      return BenchPrecision::kBf16;
    case BuildPrecision::kFp8:
      return BenchPrecision::kFp8;
    case BuildPrecision::kFp32:
    default:
      return BenchPrecision::kFp32;
  }
}

// Peak dense FLOP/s of one card at one precision, from that card's declaration
// so every share divides by the rate of the card that ran. Zero means the
// table has no row for the card, or the card declares no rate at `precision`.
inline double PeakFlopsForCard(const std::string& device_name,
                               BenchPrecision precision) {
  const DeviceProfile* profile = FindDeviceProfile(device_name);
  if (profile == nullptr) return 0.0;
  switch (precision) {
    case BenchPrecision::kBf16:
      return profile->peak_bf16_flops;
    case BenchPrecision::kFp8:
      return profile->peak_fp8_flops;
    case BenchPrecision::kFp32:
    default:
      return profile->peak_fp32_flops;
  }
}

// The name of the card the process runs on, as the seam reports it. An empty
// string means the backend named no device.
inline std::string BenchDeviceName() {
  const Caps caps = kernels::GetCaps();
  return caps.device_name != nullptr ? std::string(caps.device_name)
                                     : std::string();
}

// Achieved rate over `peak_flops`, clamped to [0, 1] as `ComputeMfu` does. A
// row with no rate, or a report with no peak, states no share.
inline double MfuOf(double gflops, double peak_flops) {
  if (!(peak_flops > 0.0) || !(gflops > 0.0)) return 0.0;
  const double mfu = gflops * 1e9 / peak_flops;
  if (!std::isfinite(mfu) || mfu < 0.0) return 0.0;
  return mfu > 1.0 ? 1.0 : mfu;
}

// One measured feature.
struct BenchRow {
  std::string name;
  std::string shape;
  double ms = 0.0;
  double gflops = 0.0;  // achieved GFLOP/s using the necessary operation count
  // Issued work over necessary work, when the benchmark accounts for both.
  // Zero means the benchmark does not report the ratio.
  double work_ratio = 0.0;
  // Model FLOPs utilization: the achieved rate over the peak dense rate of the
  // running card. Zero means the report states no peak, or that the card
  // declares no rate at the precision the benchmark named.
  double mfu = 0.0;
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
//
// A report that states a peak rate gives every row its share of that rate. One
// call does it, so a benchmark reports model FLOPs utilization without naming
// a device, a precision, or a table:
//
//   BenchReport report;
//   report.UseDevicePeak();          // the running card, the build precision
//   report.Add(...);                 // each row carries its share
//   report.Print(opt);               // the table states the denominator
class BenchReport {
 public:
  void Add(const std::string& name, const std::string& shape, double ms,
           double gflops = 0.0, double work_ratio = 0.0) {
    BenchRow row{name, shape, ms, gflops, work_ratio, 0.0};
    row.mfu = MfuOf(row.gflops, peak_flops_);
    rows_.push_back(std::move(row));
  }

  // States the denominator from the card the process runs on: its declared
  // rate at `precision`, read through `nanochat/device_profile.h`. Returns the
  // rate, or zero when the table holds no row for the card or the card
  // declares no rate at that precision.
  double UseDevicePeak(BenchPrecision precision = BuildBenchPrecision()) {
    device_ = BenchDeviceName();
    peak_flops_ = PeakFlopsForCard(device_, precision);
    stated_ = true;
    FillShare();
    return peak_flops_;
  }

  // States the denominator by hand, in FLOP/s, for a card the table does not
  // hold or for a rate a caller fixes. Zero leaves every share out.
  void SetPeakFlops(double peak_flops) {
    peak_flops_ = peak_flops;
    stated_ = true;
    FillShare();
  }

  double peak_flops() const { return peak_flops_; }
  const std::string& device() const { return device_; }

  // One line naming the denominator, or the reason there is none.
  std::string PeakCaption() const {
    const char* device = device_.empty() ? "this device" : device_.c_str();
    char buffer[192];
    if (!(peak_flops_ > 0.0)) {
      std::snprintf(buffer, sizeof(buffer),
                    "no peak rate for %s, MFU is not reported", device);
    } else {
      std::snprintf(buffer, sizeof(buffer), "MFU peak %.1f GFLOP/s on %s",
                    peak_flops_ * 1e-9, device);
    }
    return std::string(buffer);
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
    bool show_mfu = false;
    for (const BenchRow& row : rows_) {
      name_width = std::max(name_width, row.name.size());
      shape_width = std::max(shape_width, row.shape.size());
      show_mfu = show_mfu || row.mfu > 0.0;
    }
    // The denominator stands above the table it explains, and only a report
    // that states one prints it. A report that states none keeps the earlier
    // three-column shape.
    if (stated_) std::printf("# %s\n", PeakCaption().c_str());
    std::printf("%-*s  %-*s  %10s  %10s  %10s", static_cast<int>(name_width),
                "name", static_cast<int>(shape_width), "shape", "ms", "GFLOP/s",
                "issue/need");
    // The column appears only for a benchmark that reports MFU, so a report
    // that does not stays byte-identical to the earlier one.
    if (show_mfu) std::printf("  %6s", "MFU");
    std::printf("\n");
    for (const BenchRow& row : rows_) {
      char ratio[16];
      if (row.work_ratio > 0.0) {
        std::snprintf(ratio, sizeof(ratio), "%.3f", row.work_ratio);
      } else {
        std::snprintf(ratio, sizeof(ratio), "-");
      }
      std::printf("%-*s  %-*s  %10.4f  %10.1f  %10s",
                  static_cast<int>(name_width), row.name.c_str(),
                  static_cast<int>(shape_width), row.shape.c_str(), row.ms,
                  row.gflops, ratio);
      if (show_mfu) {
        char mfu[16];
        if (row.mfu > 0.0) {
          std::snprintf(mfu, sizeof(mfu), "%.1f%%", 100.0 * row.mfu);
        } else {
          std::snprintf(mfu, sizeof(mfu), "-");
        }
        std::printf("  %6s", mfu);
      }
      std::printf("\n");
    }
  }

 private:
  // Recomputes the share of every row from its own rate and the current
  // denominator, so the order of `Add` and `UseDevicePeak` never matters.
  void FillShare() {
    for (BenchRow& row : rows_) row.mfu = MfuOf(row.gflops, peak_flops_);
  }

  std::string Json() const {
    std::string text = "{\"schema\":\"nanochat.bench.v1\"";
    if (!device_.empty()) {
      text += ",\"device\":\"" + Escape(device_) + "\"";
    }
    if (peak_flops_ > 0.0) {
      char buffer[64];
      std::snprintf(buffer, sizeof(buffer), ",\"peak_flops\":%.6f",
                    peak_flops_);
      text += buffer;
    }
    text += ",\"rows\":[";
    for (std::size_t i = 0; i < rows_.size(); ++i) {
      const BenchRow& row = rows_[i];
      char buffer[640];
      std::snprintf(buffer, sizeof(buffer),
                    "%s{\"name\":\"%s\",\"shape\":\"%s\",\"ms\":%.6f,"
                    "\"gflops\":%.6f,\"work_ratio\":%.6f,\"mfu\":%.6f}",
                    i == 0 ? "" : ",", Escape(row.name).c_str(),
                    Escape(row.shape).c_str(), row.ms, row.gflops,
                    row.work_ratio, row.mfu);
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
  std::string device_;
  double peak_flops_ = 0.0;
  // True once a caller stated a denominator, so `Print` knows whether to
  // explain it. A report that never asked prints no caption.
  bool stated_ = false;
};

}  // namespace dev
}  // namespace nanochat

#endif  // NANOCHAT_DEV_KERNELS_BENCH_UTILS_H_
