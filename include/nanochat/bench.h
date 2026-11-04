#ifndef NANOCHAT_BENCH_H_
#define NANOCHAT_BENCH_H_

// The benchmark report: one row per measured feature, one JSON document, and
// the denominator that turns a rate into a share of a device's peak.
//
// This header is the *only* writer of the `nanochat.bench.v3` document. A
// benchmark binary in `backends/` and the harness benchmarks in `src/` both
// reach the writer here, so one schema has one producer and cannot drift
// between two hand-formatted strings (docs/performance.md section 3.4).
//
// It holds no vendor header and no CUDA event. The device-event timer stays in
// `backends/cuda/kernels/testing/bench_utils.h`, which includes this header and
// adds `EventTimer`; the host-only `src/eval_bench.cc` includes this header and
// times with a steady clock. A host-only target therefore reports the same
// document without linking the CUDA runtime.
//
// The row identity is what makes a measured row and a predicted row join
// (docs/performance-model.md section 7.4). A row carries `family`, `phase`,
// `precision`, and `shape`:
//
//   * `family` is the operation the counter table names, with no phase and no
//     shape in it (`attention`, `gemm`, `muon_orthogonalisation`).
//   * `phase` is the phase of the optimizer step the row belongs to, spelled as
//     the roofline report spells it (`forward`, `backward`, `optimizer`,
//     `sync`). It is empty when the benchmark does not separate phases.
//   * `precision` is the precision the arithmetic ran at, spelled `fp32`,
//     `bf16`, or `fp8`. It is empty when the benchmark does not state one.
//   * `shape` is the workload, in the `B= T= H= KV= D=` spelling the attention
//     benchmarks already use.
//
// The four fields are compared as strings. A model row and a measured row of
// the same family, phase, precision, and shape describe the same work, so their
// `ms` and `work_ratio` are comparable without a conversion step.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/device_profile.h"
#include "nanochat/tensor.h"

namespace nanochat {

// Benchmark controls. `iters` calls are timed per round; the reported value is
// the best (minimum) mean per call across `rounds` rounds. Warm-up calls are
// not timed. Defaults are tuned for millisecond-scale device kernels: few
// calls, because each call is expensive and event resolution is already fine.
struct BenchOptions {
  int warmup = 2;
  int iters = 3;
  int rounds = 3;
  bool json = false;
  std::string out;  // optional path to also write the JSON report
};

// The short name of `precision`, as a report spells it.
inline const char* BenchPrecisionName(BuildPrecision precision) {
  switch (precision) {
    case BuildPrecision::kBf16:
      return "bf16";
    case BuildPrecision::kFp8:
      return "fp8";
    case BuildPrecision::kFp32:
    default:
      return "fp32";
  }
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
  // The join identity. Empty means the row does not state that field, and the
  // row is reported but takes part in no join.
  std::string family;
  std::string phase;
  std::string precision;
};

// Collects rows and emits a human-readable table, a JSON document, or both.
// The JSON schema is `nanochat.bench.v3`.
//
// A report that states a peak rate gives every row its share of that rate. One
// call does it, so a benchmark reports Model FLOPs Utilization without naming a
// device, a precision, or a table:
//
//   BenchReport report;
//   report.UseDevicePeak(DeviceName(), BuildPrecision::kBf16);
//   report.SetRowIdentity("attention", "forward");
//   report.Add("attention_fwd:d8_s512", "B=8 T=512 H=4 KV=4 D=128", 12.3, 45.6,
//              ratio);
//   report.Print(opt);
class BenchReport {
 public:
  // Appends one row. The identity set by `SetRowIdentity` and the report's
  // precision are inherited, so a loop over rows of one family states them once.
  void Add(const std::string& name, const std::string& shape, double ms,
           double gflops = 0.0, double work_ratio = 0.0) {
    BenchRow row;
    row.name = name;
    row.shape = shape;
    row.ms = ms;
    row.gflops = gflops;
    row.work_ratio = work_ratio;
    row.family = family_;
    row.phase = phase_;
    row.precision = precision_;
    row.mfu = MfuOf(row.gflops, peak_flops_);
    rows_.push_back(std::move(row));
  }

  // Appends a row whose fields the caller states in full. It is the form a
  // benchmark uses when its rows belong to more than one family.
  void Add(const BenchRow& row) {
    BenchRow copy = row;
    if (copy.precision.empty()) copy.precision = precision_;
    copy.mfu = MfuOf(copy.gflops, peak_flops_);
    rows_.push_back(std::move(copy));
  }

  // The family and the phase every later `Add` inherits.
  void SetRowIdentity(const std::string& family,
                      const std::string& phase = std::string()) {
    family_ = family;
    phase_ = phase;
  }

  // The precision every later row inherits, and the one the report states. A row
  // that already carries a precision keeps it, because the row stated it.
  void SetPrecision(BuildPrecision precision) {
    precision_ = BenchPrecisionName(precision);
  }

  // States the denominator from a named card: its declared rate at
  // `precision`, read through `nanochat/device_profile.h`. Returns the rate, or
  // zero when no declaration names the card or it declares no rate at that
  // precision. The device name the report carries is the name it was given, so
  // a reader compares it with the name the roofline report carries.
  double UseDevicePeak(const std::string& device_name,
                       BuildPrecision precision = kBuildPrecision) {
    device_ = device_name;
    precision_ = BenchPrecisionName(precision);
    peak_flops_ = PeakFlopsFor(device_name, precision);
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

  // One extra top-level field, for a benchmark whose report carries a number
  // that is not a row (the evaluation arena's bytes, the token rate of a
  // forward, and the baseline it is compared with). The writer emits it beside
  // the rows, so such a benchmark still uses one writer for its rows.
  void SetExtra(const std::string& key, double value) {
    extras_.emplace_back(key, JsonDouble(value));
  }
  // The same field as an exact integer, for a count a reader may parse as one
  // (a byte count, a batch size).
  void SetExtraInt(const std::string& key, long long value) {
    extras_.emplace_back(key, std::to_string(value));
  }
  void SetExtraText(const std::string& key, const std::string& value) {
    extras_.emplace_back(key, JsonString(value));
  }

  double peak_flops() const { return peak_flops_; }
  const std::string& device() const { return device_; }
  const std::string& precision() const { return precision_; }
  const std::vector<BenchRow>& rows() const { return rows_; }

  // One line naming the denominator, or the reason there is none.
  std::string PeakCaption() const {
    const char* device = device_.empty() ? "this device" : device_.c_str();
    char buffer[224];
    if (!(peak_flops_ > 0.0)) {
      std::snprintf(buffer, sizeof(buffer),
                    "no declared peak rate for %s, MFU is not reported",
                    device);
    } else {
      std::snprintf(buffer, sizeof(buffer), "MFU peak %.1f GFLOP/s on %s (%s)",
                    peak_flops_ * 1e-9, device,
                    precision_.empty() ? "build precision"
                                       : precision_.c_str());
    }
    return std::string(buffer);
  }

  // The JSON document. The row fields come first and the extras follow, so a
  // reader that knows the schema reads the same keys in the same order.
  std::string Json() const {
    std::string text = "{\"schema\":\"nanochat.bench.v3\"";
    if (!device_.empty()) text += ",\"device\":" + JsonString(device_);
    if (!precision_.empty()) {
      text += ",\"precision\":" + JsonString(precision_);
    }
    if (peak_flops_ > 0.0) {
      text += ",\"peak_flops\":" + JsonDouble(peak_flops_);
    }
    text += ",\"rows\":[";
    for (std::size_t i = 0; i < rows_.size(); ++i) {
      if (i != 0) text += ",";
      const BenchRow& row = rows_[i];
      text += "{\"name\":" + JsonString(row.name);
      if (!row.family.empty()) {
        text += ",\"family\":" + JsonString(row.family);
      }
      if (!row.phase.empty()) text += ",\"phase\":" + JsonString(row.phase);
      if (!row.precision.empty()) {
        text += ",\"precision\":" + JsonString(row.precision);
      }
      text += ",\"shape\":" + JsonString(row.shape) +
              ",\"ms\":" + JsonDouble(row.ms) +
              ",\"gflops\":" + JsonDouble(row.gflops) +
              ",\"work_ratio\":" + JsonDouble(row.work_ratio) +
              ",\"mfu\":" + JsonDouble(row.mfu) + "}";
    }
    text += "]";
    for (const auto& extra : extras_) {
      text += "," + JsonString(extra.first) + ":" + extra.second;
    }
    text += "}";
    return text;
  }

  // Print to stdout, and write the JSON report to `opt.out` when set. A report
  // whose `opt.json` is set prints only the document, so a caller can pipe it.
  void Print(const BenchOptions& opt) const {
    if (!opt.out.empty()) {
      std::FILE* file = std::fopen(opt.out.c_str(), "wb");
      if (file != nullptr) {
        const std::string text = Json();
        std::fwrite(text.data(), 1, text.size(), file);
        std::fputc('\n', file);
        std::fclose(file);
      }
    }
    if (opt.json) {
      const std::string text = Json();
      std::fwrite(text.data(), 1, text.size(), stdout);
      std::fputc('\n', stdout);
      return;
    }
    PrintTable();
  }

 private:
  // Recomputes the share of every row from its own rate and the current
  // denominator, so the order of `Add` and `UseDevicePeak` never matters.
  void FillShare() {
    for (BenchRow& row : rows_) row.mfu = MfuOf(row.gflops, peak_flops_);
  }

  void PrintTable() const {
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
          std::snprintf(mfu, sizeof(mfu), "%.1f%%", row.mfu * 100.0);
        } else {
          std::snprintf(mfu, sizeof(mfu), "-");
        }
        std::printf("  %6s", mfu);
      }
      std::printf("\n");
    }
  }

  static std::string JsonString(const std::string& value) {
    std::string out = "\"";
    for (char c : value) {
      if (c == '"' || c == '\\') {
        out.push_back('\\');
        out.push_back(c);
      } else if (c == '\n') {
        out += "\\n";
      } else {
        out.push_back(c);
      }
    }
    out.push_back('"');
    return out;
  }

  static std::string JsonDouble(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.6g", value);
    return std::string(buffer);
  }

  std::vector<BenchRow> rows_;
  std::vector<std::pair<std::string, std::string>> extras_;
  std::string device_;
  std::string precision_;
  std::string family_;
  std::string phase_;
  double peak_flops_ = 0.0;
  bool stated_ = false;
};

}  // namespace nanochat

#endif  // NANOCHAT_BENCH_H_
