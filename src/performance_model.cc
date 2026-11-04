// The performance model of docs/performance-model.md: the operation counters
// of section 4 and the roofline estimator of section 7.
//
// The counters are the operation table the estimator reads, and `mfu.h` is
// their only declaration: `CountParams`, `CountFlopsPerToken`,
// `EstimateFlopsPerToken`, `EstimateMuonFlopsPerStep`, `EstimatePrefillFlops`,
// and `EstimateDecodeFlops` live here rather than in the harness, so the
// accounting of a step has one home.
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
#include "src/ops.h"  // `kVeGateChannels` and `kSmearChannels`,
                      // the two gate widths the counters add.

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

RooflineRow MakeRow(const std::string& name, StepPhase phase,
                    ComputePrecision precision, double flops, double bytes,
                    int launches, const PerformanceLimits& limits, int world,
                    bool replicated) {
  RooflineRow row;
  row.name = name;
  row.phase = phase;
  row.precision = precision;
  row.flops = flops;
  row.bytes = bytes;
  row.launches = launches;
  // A row that scales with the token count carries a world-wide count, so the
  // division by the world size leaves the per-device time. A replicated row
  // does not: every device reads and writes the whole payload, so its rate and
  // its bandwidth stay at one device.
  const double scale = replicated ? 1.0 : static_cast<double>(world);
  const double peak = limits.PeakFlops(precision);
  const double rate = peak > 0.0 ? peak * scale : 0.0;
  row.compute_seconds = rate > 0.0 ? flops / rate : 0.0;
  row.memory_seconds = limits.bandwidth_bytes_per_second > 0.0
                           ? bytes / (limits.bandwidth_bytes_per_second * scale)
                           : 0.0;
  row.launch_seconds = launches * limits.launch_us * 1e-6;
  row.bound_seconds =
      std::max(row.compute_seconds, row.memory_seconds) + row.launch_seconds;
  row.limit =
      BindingLimit(row.compute_seconds, row.memory_seconds, row.launch_seconds);
  // The per-device rate the bound implies, and its share of the row's own peak.
  // It uses the same convention as a benchmark row, so the two join
  // (docs/performance-model.md section 7.4).
  row.flops_per_second =
      row.bound_seconds > 0.0 ? flops / (row.bound_seconds * scale) : 0.0;
  if (peak > 0.0) {
    const double mfu = row.flops_per_second / peak;
    row.mfu = mfu > 1.0 ? 1.0 : mfu;
  }
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

// Number of parameters that participate in matmuls with the token stream. The
// token embedding and value embeddings are lookups and are not counted; every
// `Linear` (lm_head, the attention/MLP projections, the value gate, and the
// smear gate) is. This mirrors `GPT.num_matmul_params` structurally.
double NumMatmulParams(const Config& config) {
  double total =
      static_cast<double>(config.padded_vocab_size) * config.hidden_dim;
  for (int i = 0; i < config.num_layers; ++i) {
    total += static_cast<double>(config.query_dim()) * config.hidden_dim;
    total += static_cast<double>(config.kv_dim()) * config.hidden_dim;
    total += static_cast<double>(config.kv_dim()) * config.hidden_dim;
    total += static_cast<double>(config.hidden_dim) * config.hidden_dim;
    if (config.has_value_embedding(i)) {
      total += static_cast<double>(config.num_kv_heads) * kVeGateChannels;
    }
    total += static_cast<double>(config.mlp_dim()) * config.hidden_dim;
    total += static_cast<double>(config.hidden_dim) * config.mlp_dim();
  }
  total += kSmearChannels;
  return total;
}

}  // namespace

// ---------------------------------------------------------------------------
// The operation counters (docs/performance-model.md section 4)
// ---------------------------------------------------------------------------

ParamBreakdown CountParams(const Config& config) {
  const int64_t hidden = config.hidden_dim;
  const int64_t padded_vocab = config.padded_vocab_size;
  const int64_t query_dim = config.query_dim();
  const int64_t kv_dim = config.kv_dim();
  const int64_t mlp_dim = config.mlp_dim();

  int64_t matrices = 0;
  // transformer.wte.weight is a lookup, so it belongs to embeddings.
  int64_t embeddings = padded_vocab * hidden;
  for (int i = 0; i < config.num_layers; ++i) {
    matrices += query_dim * hidden;  // attn.c_q.weight
    matrices += kv_dim * hidden;     // attn.c_k.weight
    matrices += kv_dim * hidden;     // attn.c_v.weight
    matrices += hidden * hidden;     // attn.c_proj.weight
    if (config.has_value_embedding(i)) {
      // attn.ve_gate.weight sits inside transformer.h, so it counts as a
      // matrix, not as a scalar.
      matrices += static_cast<int64_t>(config.num_kv_heads) * kVeGateChannels;
      embeddings += padded_vocab * kv_dim;  // value_embeds.i.weight
    }
    matrices += mlp_dim * hidden;  // mlp.c_fc.weight
    matrices += hidden * mlp_dim;  // mlp.c_proj.weight
  }

  const int64_t lm_head = padded_vocab * hidden;
  // resid_lambdas and x0_lambdas hold one value per layer. The top-level
  // smear_gate is a Linear, but `num_scaling_params` groups it with the
  // scalars, so CountParams does too.
  const int64_t scalars =
      2 * static_cast<int64_t>(config.num_layers) + kSmearChannels + 1 + 1;

  ParamBreakdown counts{};
  counts.transformer_matrices = matrices;
  counts.lm_head = lm_head;
  counts.embeddings = embeddings;
  counts.scalars = scalars;
  counts.total = matrices + lm_head + embeddings + scalars;
  return counts;
}

double EstimateFlopsPerToken(const Config& config) {
  return CountFlopsPerToken(config).total();
}

double EstimateMuonFlopsPerStep(const Config& config, int ns_steps) {
  if (ns_steps <= 0) return 0.0;
  // One matrix's three GEMMs of one iteration: `A = X'X` (or `X X'`), `A2 = A
  // A`, and `prod = X B` (or `B X`). These are the shapes `MuonUpdate` issues,
  // so the count is the necessary count of the executed arithmetic.
  const auto one = [ns_steps](int rows, int cols) {
    const double r = rows;
    const double c = cols;
    const double min_dim = std::min(r, c);
    const double max_dim = std::max(r, c);
    const double iteration = 2.0 * min_dim * min_dim * max_dim +
                             2.0 * min_dim * min_dim * min_dim +
                             2.0 * r * c * min_dim;
    return static_cast<double>(ns_steps) * iteration;
  };
  const int query_dim = config.query_dim();
  const int kv_dim = config.kv_dim();
  const int mlp_dim = config.mlp_dim();
  const double per_layer =
      one(query_dim, config.hidden_dim) + one(kv_dim, config.hidden_dim) +
      one(kv_dim, config.hidden_dim) +
      one(config.hidden_dim, config.hidden_dim) +
      one(mlp_dim, config.hidden_dim) + one(config.hidden_dim, mlp_dim);
  return per_layer * static_cast<double>(config.num_layers);
}

FlopCounts CountFlopsPerToken(const Config& config) {
  const int heads = config.num_heads;
  const int head_dim = config.head_dim();
  const int seq = config.seq_len;
  double attention_forward = 0.0;
  double attention_backward = 0.0;
  for (int i = 0; i < config.num_layers; ++i) {
    const int left = config.window_left(i);
    const int effective = left < 0 ? seq : std::min(left, seq);
    // The scores and the output GEMMs are the forward pass; the backward pass
    // is twice the forward work. The two rows sum to the 12 FLOPs per head,
    // head dimension, and window that `EstimateFlopsPerToken` counts.
    attention_forward += 4.0 * heads * head_dim * effective;
    attention_backward += 8.0 * heads * head_dim * effective;
  }
  const double matmul = NumMatmulParams(config);
  FlopCounts counts;
  counts.gemm_forward = 2.0 * matmul;
  counts.gemm_backward = 4.0 * matmul;
  counts.attention_forward = attention_forward;
  counts.attention_backward = attention_backward;
  return counts;
}

double EstimatePrefillFlops(const Config& config, int num_tokens) {
  if (num_tokens <= 0) return 0.0;
  const int heads = config.num_heads;
  const int head_dim = config.head_dim();
  double attention = 0.0;
  for (int i = 0; i < config.num_layers; ++i) {
    const int left = config.window_left(i);
    const int window = left < 0 ? num_tokens : std::min(left, num_tokens);
    if (window <= 0) continue;
    // Ramp up to the window, then attend a flat `window` per new token.
    const double attended = static_cast<double>(window) * (window + 1) / 2.0 +
                            static_cast<double>(num_tokens - window) * window;
    attention += 4.0 * heads * head_dim * attended;
  }
  return 2.0 * NumMatmulParams(config) * num_tokens + attention;
}

double EstimateDecodeFlops(const Config& config, int context_len) {
  if (context_len <= 0) return 0.0;
  const int heads = config.num_heads;
  const int head_dim = config.head_dim();
  double attention = 0.0;
  for (int i = 0; i < config.num_layers; ++i) {
    const int left = config.window_left(i);
    const int window = left < 0 ? context_len : std::min(left, context_len);
    attention += 4.0 * heads * head_dim * window;
  }
  return 2.0 * NumMatmulParams(config) + attention;
}

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

const char* PhaseName(StepPhase phase) {
  switch (phase) {
    case StepPhase::kForward:
      return "forward";
    case StepPhase::kBackward:
      return "backward";
    case StepPhase::kOptimizer:
      return "optimizer";
  }
  return "unknown";
}

ComputePrecision BuildComputePrecision() {
  switch (kBuildPrecision) {
    case BuildPrecision::kBf16:
      return ComputePrecision::kBf16;
    case BuildPrecision::kFp8:
      return ComputePrecision::kFp8;
    case BuildPrecision::kFp32:
    default:
      return ComputePrecision::kFp32;
  }
}

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
    estimate.rows.push_back(MakeRow(
        "gemm_forward:" + std::string(PrecisionName(group.precision)),
        StepPhase::kForward, group.precision, 2.0 * group.params * tokens,
        forward_bytes, launches, limits, world, false));
    estimate.rows.push_back(MakeRow(
        "gemm_backward:" + std::string(PrecisionName(group.precision)),
        StepPhase::kBackward, group.precision, 4.0 * group.params * tokens,
        2.0 * forward_bytes, 2 * launches, limits, world, false));
  }

  // The quantization cast: one fused kernel per operand, five bytes of traffic
  // per element (measure, then read and write). It only exists in the fp8
  // build. The forward casts the activation and the backward casts the output
  // gradient, so the traffic splits by operand role and the total equals the
  // one-row count of section 4.
  double cast_forward_elements = 0.0;
  double cast_backward_elements = 0.0;
  int cast_launches = 0;
  for (const MatrixGroup& group : groups) {
    if (group.precision != ComputePrecision::kFp8) continue;
    cast_forward_elements += tokens * group.sum_k;
    cast_backward_elements += tokens * group.sum_n;
    cast_launches += group.count * accum;
  }
  if (cast_forward_elements > 0.0) {
    estimate.rows.push_back(MakeRow("quantization_cast:forward",
                                    StepPhase::kForward, ComputePrecision::kFp8,
                                    0.0, 5.0 * cast_forward_elements,
                                    cast_launches, limits, world, false));
  }
  if (cast_backward_elements > 0.0) {
    estimate.rows.push_back(
        MakeRow("quantization_cast:backward", StepPhase::kBackward,
                ComputePrecision::kFp8, 0.0, 5.0 * cast_backward_elements,
                cast_launches, limits, world, false));
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
      estimate.rows.push_back(MakeRow(
          "fp8_layout_copies", StepPhase::kBackward, ComputePrecision::kFp8,
          0.0, 2.0 * weight_elements, copy_launches, limits, world, false));
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
  estimate.rows.push_back(MakeRow("attention_forward", StepPhase::kForward,
                                  row_precision, attention_forward,
                                  attention_bytes, attention_launches, limits,
                                  world, false));
  estimate.rows.push_back(MakeRow("attention_backward", StepPhase::kBackward,
                                  row_precision, attention_backward,
                                  2.0 * attention_bytes, 2 * attention_launches,
                                  limits, world, false));

  // The row kernels, the classifier, the embedding, and the optimizer. They
  // carry no counted arithmetic; their bound is the memory term. A row kernel
  // runs once in the forward pass and once in the backward pass, so each of
  // these rows becomes a forward row and a backward row of equal traffic. The
  // total is unchanged: a row with no counted arithmetic splits linearly
  // through the `max` of the estimator.
  const double row_width =
      config.num_layers *
      (6.0 * config.hidden_dim + 2.0 * config.query_dim() +
       4.0 * config.kv_dim() + 2.0 * config.mlp_dim() + config.num_kv_heads);
  const double pointwise_bytes = row_elem * tokens * row_width;
  const int pointwise_launches = config.num_layers * accum;
  estimate.rows.push_back(MakeRow(
      "norms_and_pointwise:forward", StepPhase::kForward, row_precision, 0.0,
      pointwise_bytes / 2.0, pointwise_launches / 2, limits, world, false));
  estimate.rows.push_back(MakeRow(
      "norms_and_pointwise:backward", StepPhase::kBackward, row_precision, 0.0,
      pointwise_bytes / 2.0, pointwise_launches - pointwise_launches / 2,
      limits, world, false));

  const double classifier_bytes = 2.0 * 4.0 * tokens * config.padded_vocab_size;
  estimate.rows.push_back(MakeRow(
      "classifier:forward", StepPhase::kForward, ComputePrecision::kFp32, 0.0,
      classifier_bytes / 2.0, accum / 2, limits, world, false));
  estimate.rows.push_back(MakeRow(
      "classifier:backward", StepPhase::kBackward, ComputePrecision::kFp32, 0.0,
      classifier_bytes / 2.0, accum - accum / 2, limits, world, false));

  const double embedding_bytes = 2.0 * row_elem * tokens * config.hidden_dim;
  estimate.rows.push_back(MakeRow("embedding:forward", StepPhase::kForward,
                                  row_precision, 0.0, embedding_bytes / 2.0, 1,
                                  limits, world, false));
  estimate.rows.push_back(MakeRow("embedding:backward", StepPhase::kBackward,
                                  row_precision, 0.0, embedding_bytes / 2.0, 1,
                                  limits, world, false));

  const ParamBreakdown params = CountParams(config);
  // The parameters are replicated on every device, so the optimizer state
  // travels in full on each rank and the row does not divide by the world
  // size.
  estimate.rows.push_back(MakeRow(
      "optimizer", StepPhase::kOptimizer, ComputePrecision::kFp32, 0.0,
      4.0 * 4.0 * static_cast<double>(params.total), 1, limits, world, true));
  // The Muon orthogonalisation. Its GEMMs run at the compute type, and its
  // state stays in the fp32 traffic of the row above. It gets its own row, so
  // the model count and the optimizer count stay apart (`mfu.h`).
  const double muon_flops =
      EstimateMuonFlopsPerStep(config, plan.muon_ns_steps);
  if (muon_flops > 0.0) {
    estimate.optimizer_flops = muon_flops;
    estimate.rows.push_back(MakeRow("muon_orthogonalisation",
                                    StepPhase::kOptimizer, row_precision,
                                    muon_flops, 0.0, 1, limits, world, true));
  }

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
    switch (row.phase) {
      case StepPhase::kForward:
        estimate.forward_seconds += row.bound_seconds;
        break;
      case StepPhase::kBackward:
        estimate.backward_seconds += row.bound_seconds;
        break;
      case StepPhase::kOptimizer:
        estimate.optimizer_seconds += row.bound_seconds;
        break;
    }
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

  // The numbers a benchmark reports, evaluated at the bound
  // (docs/performance-model.md section 7.4). The tokens per second is the
  // step's own token count over the step bound.
  estimate.tokens_per_step = static_cast<std::int64_t>(tokens + 0.5);
  if (estimate.step_seconds > 0.0) {
    estimate.tokens_per_second = tokens / estimate.step_seconds;
  }

  // The precision-weighted peak of the model rows: the harmonic mean of the
  // row peaks, weighted by the row arithmetic. A step that runs every model
  // row at its own peak reads 1.0, so a measured MFU and `model_mfu` share one
  // denominator. The optimizer rows are excluded, because `mfu.h` reports the
  // Muon count beside the model count and never adds them.
  double model_flops = 0.0;
  double model_peak_seconds = 0.0;
  for (const RooflineRow& row : estimate.rows) {
    const double peak = limits.PeakFlops(row.precision);
    if (row.phase == StepPhase::kOptimizer) continue;
    if (!(row.flops > 0.0) || !(peak > 0.0)) continue;
    model_flops += row.flops;
    model_peak_seconds += row.flops / peak;
  }
  if (model_peak_seconds > 0.0) {
    estimate.weighted_peak_flops = model_flops / model_peak_seconds;
    if (estimate.step_seconds > 0.0) {
      estimate.model_mfu =
          (model_flops / static_cast<double>(world)) /
          (estimate.step_seconds * estimate.weighted_peak_flops);
    }
  }
  if (estimate.optimizer_flops > 0.0 && estimate.step_seconds > 0.0) {
    const double optim_peak = limits.PeakFlops(row_precision);
    if (optim_peak > 0.0) {
      estimate.optim_mfu =
          estimate.optimizer_flops / (estimate.step_seconds * optim_peak);
    }
  }
  return estimate;
}

std::string FormatPerformanceReport(const PerformanceEstimate& estimate) {
  std::string out;
  char line[320];
  std::snprintf(line, sizeof(line), "%-30s %-9s %6s %14s %14s %12s %10s\n",
                "operation", "phase", "prec", "compute (ms)", "memory (ms)",
                "bound (ms)", "limit");
  out += line;
  for (const RooflineRow& row : estimate.rows) {
    std::snprintf(
        line, sizeof(line), "%-30s %-9s %6s %14.3f %14.3f %12.3f %10s\n",
        row.name.c_str(), PhaseName(row.phase), PrecisionName(row.precision),
        row.compute_seconds * 1000.0, row.memory_seconds * 1000.0,
        row.bound_seconds * 1000.0, row.limit.c_str());
    out += line;
  }
  std::snprintf(line, sizeof(line), "%-30s %-9s %6s %14s %14s %12.3f %10s\n",
                "gradient_sync", "sync", "-", "-", "-",
                estimate.sync_seconds * 1000.0, "sync");
  out += line;
  std::snprintf(
      line, sizeof(line),
      "phases: forward %.3f | backward %.3f | optimizer %.3f | "
      "sync %.3f (ms)\n",
      estimate.forward_seconds * 1000.0, estimate.backward_seconds * 1000.0,
      estimate.optimizer_seconds * 1000.0, estimate.sync_seconds * 1000.0);
  out += line;
  std::snprintf(line, sizeof(line),
                "step bound %.3f ms  (%s s over %lld steps)\n",
                estimate.step_seconds * 1000.0,
                FormatSeconds(estimate.run_seconds).c_str(),
                static_cast<long long>(estimate.steps));
  out += line;
  std::snprintf(line, sizeof(line),
                "at the bound: %.0f tokens/s | model_mfu %.1f%% | "
                "optim_mfu %.1f%% | weighted peak %.1f TFLOP/s\n",
                estimate.tokens_per_second, estimate.model_mfu * 100.0,
                estimate.optim_mfu * 100.0,
                estimate.weighted_peak_flops / 1e12);
  out += line;
  return out;
}

std::string PerformanceReportJson(const PerformanceEstimate& estimate) {
  std::string out = "{\"schema\":\"nanochat.roofline.v2\",\"rows\":[";
  for (std::size_t i = 0; i < estimate.rows.size(); ++i) {
    const RooflineRow& row = estimate.rows[i];
    if (i != 0) out += ",";
    out += "{\"name\":" + JsonString(row.name) +
           ",\"phase\":" + JsonString(PhaseName(row.phase)) +
           ",\"precision\":" + JsonString(PrecisionName(row.precision)) +
           ",\"flops\":" + JsonDouble(row.flops) +
           ",\"bytes\":" + JsonDouble(row.bytes) +
           ",\"launches\":" + std::to_string(row.launches) +
           ",\"compute_seconds\":" + JsonDouble(row.compute_seconds) +
           ",\"memory_seconds\":" + JsonDouble(row.memory_seconds) +
           ",\"bound_seconds\":" + JsonDouble(row.bound_seconds) +
           ",\"limit\":" + JsonString(row.limit) +
           ",\"gflops\":" + JsonDouble(row.flops_per_second / 1e9) +
           ",\"mfu\":" + JsonDouble(row.mfu) + "}";
  }
  out +=
      "],\"forward_seconds\":" + JsonDouble(estimate.forward_seconds) +
      ",\"backward_seconds\":" + JsonDouble(estimate.backward_seconds) +
      ",\"sync_seconds\":" + JsonDouble(estimate.sync_seconds) +
      ",\"optimizer_seconds\":" + JsonDouble(estimate.optimizer_seconds) +
      ",\"step_seconds\":" + JsonDouble(estimate.step_seconds) +
      ",\"steps\":" + std::to_string(estimate.steps) +
      ",\"run_seconds\":" + JsonDouble(estimate.run_seconds) +
      ",\"tokens_per_step\":" + std::to_string(estimate.tokens_per_step) +
      ",\"tokens_per_second\":" + JsonDouble(estimate.tokens_per_second) +
      ",\"weighted_peak_flops\":" + JsonDouble(estimate.weighted_peak_flops) +
      ",\"model_mfu\":" + JsonDouble(estimate.model_mfu) +
      ",\"optim_mfu\":" + JsonDouble(estimate.optim_mfu) +
      ",\"optimizer_flops\":" + JsonDouble(estimate.optimizer_flops) + "}\n";
  return out;
}

}  // namespace nanochat
