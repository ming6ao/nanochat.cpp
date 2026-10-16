// The roofline estimator of docs/performance-model.md section 7.
//
// The estimator uses the device peaks directly. It holds no fitted value and
// no measured correction: every term comes from the operation counters, the
// precision assignment, and the device limit table. A measured step below the
// bound is normal, and the gap is the finding.
//
// The unit is one optimizer step. A run of `total_tokens` multiplies the step
// bound by the step count. Every device runs the same work in parallel, so the
// aggregate rate is the world size times the one-device peak.
//
// This file is deliberately host-only: no vendor header, no GPU, and no
// broker. It links the public headers and the device profile table.

#include "nanochat/mfu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "nanochat/config.h"
#include "nanochat/device_profile.h"
#include "src/ops.h"

namespace nanochat {
namespace {

// The value size in bytes of one element at `precision`.
double PrecisionBytes(ComputePrecision precision) {
  switch (precision) {
    case ComputePrecision::kFp8:
      return 1.0;
    case ComputePrecision::kBf16:
      return 2.0;
    case ComputePrecision::kFp32:
      return 4.0;
  }
  return 4.0;
}

// One matrix family that shares a precision assignment. `params` is the sum of
// N * K over the group, `sum_n` and `sum_k` the operand widths, and `count` the
// number of matrices.
struct MatrixGroup {
  ComputePrecision precision = ComputePrecision::kFp32;
  double params = 0.0;
  double sum_n = 0.0;
  double sum_k = 0.0;
  int count = 0;

  void AddMatrix(int n, int k) {
    params += static_cast<double>(n) * k;
    sum_n += n;
    sum_k += k;
    ++count;
  }
};

void AddGrouped(std::vector<MatrixGroup>* groups, int n, int k,
                ComputePrecision precision) {
  for (MatrixGroup& group : *groups) {
    if (group.precision == precision) {
      group.AddMatrix(n, k);
      return;
    }
  }
  MatrixGroup group;
  group.precision = precision;
  group.AddMatrix(n, k);
  groups->push_back(group);
}

// The matrix multiplies of one layer and the global head, grouped by their
// precision assignment. The value gate and the smear gate stay at the build
// precision because their smaller dimension is below the fp8 threshold.
void CollectMatrixGroups(const Config& config, ComputePrecision build,
                         std::vector<MatrixGroup>* groups) {
  const int hidden = config.hidden_dim;
  for (int layer = 0; layer < config.num_layers; ++layer) {
    const int pairs[][2] = {
        {config.query_dim(), hidden}, {config.kv_dim(), hidden},
        {config.kv_dim(), hidden},    {hidden, hidden},
        {config.mlp_dim(), hidden},   {hidden, config.mlp_dim()},
    };
    for (const auto& pair : pairs) {
      AddGrouped(groups, pair[0], pair[1],
                 AssignMatrixPrecision(pair[0], pair[1], build));
    }
    if (config.has_value_embedding(layer)) {
      AddGrouped(
          groups, config.num_kv_heads, kVeGateChannels,
          AssignMatrixPrecision(config.num_kv_heads, kVeGateChannels, build));
    }
  }
  AddGrouped(groups, config.padded_vocab_size, hidden,
             AssignMatrixPrecision(config.padded_vocab_size, hidden, build));
  AddGrouped(groups, 1, kSmearChannels,
             AssignMatrixPrecision(1, kSmearChannels, build));
}

// The binding limit of one row. `compute_seconds` and `memory_seconds` are the
// two roofline terms; the launch cost is additive on top of the larger one.
const char* BindingLimit(double compute_seconds, double memory_seconds,
                         double launch_seconds) {
  if (launch_seconds >= compute_seconds && launch_seconds >= memory_seconds) {
    return "launch";
  }
  return compute_seconds >= memory_seconds ? "compute" : "memory";
}

RooflineRow MakeRow(const std::string& name, ComputePrecision precision,
                    double flops, double bytes, int launches,
                    const PerformanceLimits& limits, int world_size) {
  RooflineRow row;
  row.name = name;
  row.precision = precision;
  row.flops = flops;
  row.bytes = bytes;
  row.launches = launches;
  const double peak = limits.PeakFlops(precision);
  const double rate = peak > 0.0 ? peak * world_size : 0.0;
  row.compute_seconds = rate > 0.0 ? flops / rate : 0.0;
  row.memory_seconds =
      limits.bandwidth_bytes_per_second > 0.0
          ? bytes / (limits.bandwidth_bytes_per_second * world_size)
          : 0.0;
  row.launch_seconds = launches * limits.launch_us * 1e-6;
  row.bound_seconds =
      std::max(row.compute_seconds, row.memory_seconds) + row.launch_seconds;
  row.limit =
      BindingLimit(row.compute_seconds, row.memory_seconds, row.launch_seconds);
  return row;
}

std::string FormatSeconds(double seconds) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.2f", seconds);
  return std::string(buffer);
}

std::string JsonString(const std::string& value) {
  std::string out = "\"";
  for (char c : value) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
      out.push_back(c);
    } else {
      out.push_back(c);
    }
  }
  out.push_back('"');
  return out;
}

std::string JsonDouble(double value) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.6g", value);
  return std::string(buffer);
}

}  // namespace

double FlopCounts::total() const {
  return gemm_forward + gemm_backward + attention_forward + attention_backward +
         norm_and_pointwise + classifier + embedding;
}

const char* PrecisionName(ComputePrecision precision) {
  switch (precision) {
    case ComputePrecision::kFp32:
      return "fp32";
    case ComputePrecision::kBf16:
      return "bf16";
    case ComputePrecision::kFp8:
      return "fp8";
  }
  return "unknown";
}

ComputePrecision BuildComputePrecision() { return ComputePrecision::kFp32; }

ComputePrecision AssignMatrixPrecision(int rows, int cols,
                                       ComputePrecision build) {
  // The fp8 path only exists in an fp8 build. A matrix that does not convert
  // keeps the reduced compute type: bf16 beside an fp8 path, and the build
  // precision otherwise (docs/performance-model.md section 5.2).
  const ComputePrecision fallback =
      build == ComputePrecision::kFp8 ? ComputePrecision::kBf16 : build;
  if (build != ComputePrecision::kFp8) return build;
  if (rows <= 0 || cols <= 0) return fallback;
  if (rows % 16 != 0 || cols % 16 != 0) return fallback;
  if (std::min(rows, cols) < 128) return fallback;
  return ComputePrecision::kFp8;
}

double PerformanceLimits::PeakFlops(ComputePrecision precision) const {
  switch (precision) {
    case ComputePrecision::kFp32:
      return fp32_flops;
    case ComputePrecision::kBf16:
      return bf16_flops;
    case ComputePrecision::kFp8:
      return fp8_flops;
  }
  return 0.0;
}

PerformanceLimits LimitsForDevice(const std::string& device_name) {
  PerformanceLimits limits;
  const DeviceProfile* profile = FindDeviceProfile(device_name);
  if (profile == nullptr) return limits;
  limits.fp32_flops = profile->peak_fp32_flops;
  limits.bf16_flops = profile->peak_bf16_flops;
  limits.fp8_flops = profile->peak_fp8_flops;
  limits.bandwidth_bytes_per_second = profile->peak_bandwidth_bytes_per_second;
  limits.launch_us = profile->launch_us;
  return limits;
}

bool IsComputeBound(double flops, double bytes, const PerformanceLimits& limits,
                    ComputePrecision precision) {
  const double peak = limits.PeakFlops(precision);
  if (!(peak > 0.0) || !(limits.bandwidth_bytes_per_second > 0.0)) {
    return peak > 0.0;
  }
  const double compute_seconds = flops / peak;
  const double memory_seconds = bytes / limits.bandwidth_bytes_per_second;
  return compute_seconds >= memory_seconds;
}

PerformanceEstimate EstimatePerformance(const Config& config,
                                        const PerformanceLimits& limits,
                                        const ModelPlan& plan) {
  PerformanceEstimate estimate;
  const int world = std::max(1, plan.world_size);
  const int batch = std::max(1, plan.batch);
  const int accum = std::max(1, plan.grad_accum);
  const double device_tokens = static_cast<double>(batch) * config.seq_len;
  const double tokens = device_tokens * accum * world;

  const ComputePrecision build = plan.build;
  // The attention engine, the norms, and the embeddings run at bf16 in a
  // reduced-precision build and at the build precision otherwise
  // (docs/performance-model.md section 5.2).
  const ComputePrecision row_precision =
      (build == ComputePrecision::kFp8 || build == ComputePrecision::kBf16)
          ? ComputePrecision::kBf16
          : build;

  std::vector<MatrixGroup> groups;
  CollectMatrixGroups(config, build, &groups);

  // The matrix multiplies. The forward writes one output and reads two
  // operands; the backward reads and writes twice over.
  for (const MatrixGroup& group : groups) {
    const double elem = PrecisionBytes(group.precision);
    const double forward_bytes =
        elem * (tokens * group.sum_k + group.params + tokens * group.sum_n);
    const int launches = group.count * accum;
    estimate.rows.push_back(
        MakeRow("gemm_forward:" + std::string(PrecisionName(group.precision)),
                group.precision, 2.0 * group.params * tokens, forward_bytes,
                launches, limits, world));
    estimate.rows.push_back(
        MakeRow("gemm_backward:" + std::string(PrecisionName(group.precision)),
                group.precision, 4.0 * group.params * tokens,
                2.0 * forward_bytes, 2 * launches, limits, world));
  }

  // The quantization cast: one fused kernel per operand, five bytes of traffic
  // per element (measure, then read and write). It only exists in the fp8
  // build, and it covers the fp8 operands.
  double cast_elements = 0.0;
  int cast_launches = 0;
  for (const MatrixGroup& group : groups) {
    if (group.precision != ComputePrecision::kFp8) continue;
    cast_elements += tokens * (group.sum_n + group.sum_k);
    cast_launches += 2 * group.count * accum;
  }
  if (cast_elements > 0.0) {
    estimate.rows.push_back(MakeRow("quantization_cast", ComputePrecision::kFp8,
                                    0.0, 5.0 * cast_elements, cast_launches,
                                    limits, world));
  }

  // The layout copies. The target stores the weight in the layout the fp8 GEMM
  // wants, once, so this row is zero. The reference copies per backward call,
  // which section 7.2 prices at about five minutes.
  if (!plan.transposed_weights) {
    double weight_elements = 0.0;
    int copy_launches = 0;
    for (const MatrixGroup& group : groups) {
      if (group.precision != ComputePrecision::kFp8) continue;
      weight_elements += group.params;
      copy_launches += 3 * group.count * accum;
    }
    if (weight_elements > 0.0) {
      estimate.rows.push_back(
          MakeRow("fp8_layout_copies", ComputePrecision::kFp8, 0.0,
                  2.0 * weight_elements, copy_launches, limits, world));
    }
  }

  // The attention. `window` is the effective left extent of the layer.
  const double row_elem = PrecisionBytes(row_precision);
  double attention_forward = 0.0;
  double attention_backward = 0.0;
  for (int layer = 0; layer < config.num_layers; ++layer) {
    const int left = config.window_left(layer);
    const int window =
        left < 0 ? config.seq_len : std::min(left, config.seq_len);
    const double per_token =
        static_cast<double>(config.num_heads) * config.head_dim() * window;
    attention_forward += 4.0 * per_token * tokens;
    attention_backward += 8.0 * per_token * tokens;
  }
  // Flash attention keeps the score and probability matrices in the tile, so
  // only the query, key, value, and output operands touch HBM. Four head rows
  // per token, once per layer.
  const double attention_bytes = row_elem * tokens * 4.0 * config.num_heads *
                                 config.head_dim() * config.num_layers;
  const int attention_launches = config.num_layers * accum;
  estimate.rows.push_back(MakeRow("attention_forward", row_precision,
                                  attention_forward, attention_bytes,
                                  attention_launches, limits, world));
  estimate.rows.push_back(MakeRow("attention_backward", row_precision,
                                  attention_backward, 2.0 * attention_bytes,
                                  2 * attention_launches, limits, world));

  // The row kernels, the classifier, the embedding, and the optimizer. They
  // carry no counted arithmetic; their bound is the memory term.
  const double row_width =
      config.num_layers *
      (6.0 * config.hidden_dim + 2.0 * config.query_dim() +
       4.0 * config.kv_dim() + 2.0 * config.mlp_dim() + config.num_kv_heads);
  estimate.rows.push_back(MakeRow("norms_and_pointwise", row_precision, 0.0,
                                  row_elem * tokens * row_width,
                                  config.num_layers * accum, limits, world));
  estimate.rows.push_back(MakeRow("classifier", ComputePrecision::kFp32, 0.0,
                                  2.0 * 4.0 * tokens * config.padded_vocab_size,
                                  accum, limits, world));
  estimate.rows.push_back(MakeRow("embedding", row_precision, 0.0,
                                  2.0 * row_elem * tokens * config.hidden_dim,
                                  2 * accum, limits, world));

  const ParamBreakdown params = CountParams(config);
  estimate.rows.push_back(MakeRow("optimizer", ComputePrecision::kFp32, 0.0,
                                  4.0 * 4.0 * static_cast<double>(params.total),
                                  1, limits, world));

  // The gradient reduction, once per optimizer step (docs/performance-model.md
  // section 7). The payload is the fp32 gradient of every parameter.
  if (world > 1 && plan.link_bandwidth_bytes_per_second > 0.0) {
    const double payload = 4.0 * static_cast<double>(params.total);
    const double moved = 2.0 * (world - 1) / world * payload;
    const double full = moved / plan.link_bandwidth_bytes_per_second +
                        plan.link_latency_seconds;
    estimate.sync_seconds =
        (1.0 - std::min(1.0, std::max(0.0, plan.sync_overlap))) * full;
  }

  for (const RooflineRow& row : estimate.rows) {
    estimate.step_seconds += row.bound_seconds;
  }
  estimate.step_seconds += estimate.sync_seconds;

  const double tokens_per_step = tokens > 0.0 ? tokens : 1.0;
  if (plan.total_tokens > 0) {
    estimate.steps = static_cast<std::int64_t>(
        std::ceil(static_cast<double>(plan.total_tokens) / tokens_per_step));
    if (estimate.steps < 1) estimate.steps = 1;
  } else {
    estimate.steps = 1;
  }
  estimate.run_seconds = estimate.step_seconds * estimate.steps;
  return estimate;
}

std::string FormatPerformanceReport(const PerformanceEstimate& estimate) {
  std::string out;
  char line[256];
  std::snprintf(line, sizeof(line), "%-26s %6s %14s %14s %12s %10s\n",
                "operation", "prec", "compute (ms)", "memory (ms)",
                "bound (ms)", "limit");
  out += line;
  for (const RooflineRow& row : estimate.rows) {
    std::snprintf(line, sizeof(line), "%-26s %6s %14.3f %14.3f %12.3f %10s\n",
                  row.name.c_str(), PrecisionName(row.precision),
                  row.compute_seconds * 1000.0, row.memory_seconds * 1000.0,
                  row.bound_seconds * 1000.0, row.limit.c_str());
    out += line;
  }
  std::snprintf(line, sizeof(line), "%-26s %6s %14s %14s %12.3f %10s\n",
                "gradient_sync", "-", "-", "-", estimate.sync_seconds * 1000.0,
                "sync");
  out += line;
  std::snprintf(line, sizeof(line),
                "step bound %.3f ms  (%s s over %lld steps)\n",
                estimate.step_seconds * 1000.0,
                FormatSeconds(estimate.run_seconds).c_str(),
                static_cast<long long>(estimate.steps));
  out += line;
  return out;
}

std::string PerformanceReportJson(const PerformanceEstimate& estimate) {
  std::string out = "{\"schema\":\"nanochat.roofline.v1\",\"rows\":[";
  for (std::size_t i = 0; i < estimate.rows.size(); ++i) {
    const RooflineRow& row = estimate.rows[i];
    if (i != 0) out += ",";
    out += "{\"name\":" + JsonString(row.name) +
           ",\"precision\":" + JsonString(PrecisionName(row.precision)) +
           ",\"flops\":" + JsonDouble(row.flops) +
           ",\"bytes\":" + JsonDouble(row.bytes) +
           ",\"launches\":" + std::to_string(row.launches) +
           ",\"compute_seconds\":" + JsonDouble(row.compute_seconds) +
           ",\"memory_seconds\":" + JsonDouble(row.memory_seconds) +
           ",\"bound_seconds\":" + JsonDouble(row.bound_seconds) +
           ",\"limit\":" + JsonString(row.limit) + "}";
  }
  out += "],\"sync_seconds\":" + JsonDouble(estimate.sync_seconds) +
         ",\"step_seconds\":" + JsonDouble(estimate.step_seconds) +
         ",\"steps\":" + std::to_string(estimate.steps) +
         ",\"run_seconds\":" + JsonDouble(estimate.run_seconds) + "}\n";
  return out;
}

}  // namespace nanochat
