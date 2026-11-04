#ifndef NANOCHAT_MFU_H_
#define NANOCHAT_MFU_H_

#include <cstdint>
#include <string>
#include <vector>

#include "nanochat/config.h"

// Model FLOPs utilization (docs/model.md). The accounting follows nanochat:
// every matmul parameter contributes 6 FLOPs/token for a training step
// (2 forward + 4 backward), plus 12 * heads * head_dim * effective_seq per
// attention layer.

namespace nanochat {

// Structural parameter counts for the scaling-law plan (docs/python.md
// section 4.1). Each field mirrors one group of `GPT.num_scaling_params` from
// the reference. `total` counts every allocated parameter and equals
// embeddings + transformer_matrices + lm_head + scalars.
struct ParamBreakdown {
  int64_t total;
  int64_t transformer_matrices;
  int64_t lm_head;
  int64_t embeddings;
  int64_t scalars;
};

// Counts the parameters of `config` by group. This is a structural count of
// `Config`, not a walk of runtime parameter names. It lands beside
// `NumMatmulParams` in `src/performance_model.cc` and mirrors
// `GPT.num_scaling_params()`.
// `NumMatmulParams` mirrors `GPT.num_matmul_params()`; the two differ, so use
// the right one.
ParamBreakdown CountParams(const Config& config);

// Forward + backward FLOPs for one training token.
double EstimateFlopsPerToken(const Config& config);

// The Muon optimizer's arithmetic for one training step. Every matrix the
// optimizer owns runs `ns_steps` Polar Express iterations of three GEMMs:
// `A = X'X` (or `X X'`),
// `A2 = A A`, and `prod = X B` (or `B X`). The arithmetic is a property of the
// shape and the iteration count, not of the batch, so it is counted per step.
// The scalars and the value gate are excluded: their matrices are one row or
// one column, so their share is zero at this precision.
//
// The model's count and this one are reported side by side and never added:
// the model's MFU divides the model's count by the peak, and a reader who wants
// the step's total arithmetic adds the two numbers themselves.
double EstimateMuonFlopsPerStep(const Config& config, int ns_steps);

// Forward FLOPs to prefill `num_tokens` (causal, windowed).
double EstimatePrefillFlops(const Config& config, int num_tokens);

// Forward FLOPs to decode one token against `context_len` cached tokens.
double EstimateDecodeFlops(const Config& config, int context_len);

// Peak device FLOPs for a known device name, or 0 when unknown. The table is
// best-effort and mirrors nanochat's `get_peak_flops`.
double PeakFlopsForDevice(const std::string& device_name);

// Peak dense FLOPs for `device_name` at the build's compute precision
// (docs/precision.md). fp32 is the only compute type today, so the MFU
// denominator is the CUDA-core fp32 rate. A future bf16 build reads the
// tensor-core rate. `PeakFlopsForDevice` keeps the reference tensor-core rate
// for compatibility, so the two differ on a device whose reference recipe uses
// tensor cores. Returns 0 for an unknown device.
double PeakFlopsForCompute(const std::string& device_name);

// tokens_per_second * flops_per_token / peak_flops, clamped to [0, 1].
double ComputeMfu(const Config& config, double tokens_per_second,
                  double peak_flops);

// ---------------------------------------------------------------------------
// The performance model (docs/performance-model.md)
// ---------------------------------------------------------------------------

// The precision one counter row uses. The model selects a rate per operation
// from the assignment in docs/performance-model.md section 5; it never derives
// one rate from the build precision alone.
enum class ComputePrecision { kFp32, kBf16, kFp8 };

// The short name of `precision`, for a report row.
const char* PrecisionName(ComputePrecision precision);

// The build's compute precision, from `kBuildPrecision` in
// `nanochat/tensor.h`. It is the fallback assignment for an operation that has
// no fp8 path in the current tree.
ComputePrecision BuildComputePrecision();

// Peak dense FLOPs for `device_name` at an explicit `precision`. It is the
// explicit-precision form of the `PeakFlopsForCompute` overload above, so the
// model selects the rate from the precision assignment of section 5 rather
// than from the build precision alone. Returns 0 for an unknown device or an
// unsupported precision.
double PeakFlopsForCompute(const std::string& device_name,
                           ComputePrecision precision);

// The arithmetic of one training token, split by family. The fields sum to
// `EstimateFlopsPerToken`, so the reference MFU line and the model share one
// source of truth (docs/performance-model.md section 4).
struct FlopCounts {
  double gemm_forward = 0.0;   // 2 * params * tokens
  double gemm_backward = 0.0;  // 4 * params * tokens
  double attention_forward = 0.0;
  double attention_backward = 0.0;
  double norm_and_pointwise = 0.0;
  double classifier = 0.0;
  double embedding = 0.0;

  double total() const;
};

// Fills `FlopCounts` from the same terms `EstimateFlopsPerToken` uses.
FlopCounts CountFlopsPerToken(const Config& config);

// The precision a matrix multiply uses. The rule follows the reference fp8
// filter (docs/performance-model.md section 5.1): the matrix converts when the
// build has an fp8 path, both dimensions are divisible by 16, and the smaller
// dimension is at least 128. Otherwise the matrix keeps `build`.
ComputePrecision AssignMatrixPrecision(int rows, int cols,
                                       ComputePrecision build);

// The device limits the estimator reads (docs/performance-model.md section 6).
// The rates are dense (non-sparse) and in FLOP/s; the bandwidth is in bytes per
// second.
struct PerformanceLimits {
  double fp32_flops = 0.0;
  double bf16_flops = 0.0;
  double fp8_flops = 0.0;
  double bandwidth_bytes_per_second = 0.0;
  double launch_us = 0.0;

  // The rate for `precision`, or 0 when the device has no path.
  double PeakFlops(ComputePrecision precision) const;
};

// Reads the device profile `device_name`. An unknown device yields all-zero
// limits, and the estimator reports a zero bound. The limit table is a rate
// table: it does not depend on the build or on the precision assignment.
PerformanceLimits LimitsForDevice(const std::string& device_name);

// The plan the estimator needs beyond `Config`: the batch, the topology, and
// the run length. `total_tokens` 0 models one optimizer step.
struct ModelPlan {
  int batch = 1;                  // micro-batch per device
  int grad_accum = 1;             // micro-batches per optimizer step
  int world_size = 1;             // data-parallel device count
  std::int64_t total_tokens = 0;  // the run length; 0 keeps one step
  double link_bandwidth_bytes_per_second = 0.0;
  double link_latency_seconds = 0.0;
  double sync_overlap = 0.0;  // the fraction of the sync hidden by the backward
  // The build's compute precision, which sets the fp8 eligibility of a matrix.
  ComputePrecision build = BuildComputePrecision();
  // True stores the weight in the layout the fp8 GEMM wants, once. False
  // models the reference, which copies the weight per backward call.
  bool transposed_weights = true;
  // The Muon Newton-Schulz iteration count. Zero keeps the optimizer row
  // memory bound, and so reproduces a report that does not count the
  // orthogonalisation (docs/performance-model.md section 4).
  int muon_ns_steps = 0;
};

// The phase of one optimizer step that a counter row belongs to
// (docs/performance-model.md section 7.4). The names match the phase split of
// the training log, so a bound and a measurement join phase by phase.
enum class StepPhase { kForward, kBackward, kSync, kOptimizer };

// The short name of `phase`, for a report row.
const char* PhaseName(StepPhase phase);

// One operation of the counter table, with its roofline result.
struct RooflineRow {
  std::string name;
  StepPhase phase = StepPhase::kForward;
  ComputePrecision precision = ComputePrecision::kFp32;
  double flops = 0.0;
  double bytes = 0.0;
  int launches = 0;
  double compute_seconds = 0.0;
  double memory_seconds = 0.0;
  double launch_seconds = 0.0;
  double bound_seconds = 0.0;
  // The binding limit: "compute", "memory", or "launch".
  std::string limit;
  // The rate the row's bound implies, in FLOP/s over the whole world. It is the
  // predicted counterpart of a benchmark row's `gflops`
  // (docs/performance.md section 3.1), and `mfu` is that rate over the peak the
  // row's precision declares. A row with no counted arithmetic reads 0.
  double flops_per_second = 0.0;
  double mfu = 0.0;
};

// The full roofline result. `step_seconds` is the sum of the row bounds plus
// the gradient sync. `run_seconds` multiplies it by the optimizer steps.
//
// The phase fields sum the row bounds by phase, so `forward_seconds` plus
// `backward_seconds` plus `optimizer_seconds` plus `sync_seconds` equals
// `step_seconds`. They carry no new physics; they are the attribution the
// training log reports, so a reader compares (`LogRecord` in
// `include/nanochat/logger.h`).
struct PerformanceEstimate {
  std::vector<RooflineRow> rows;
  double forward_seconds = 0.0;
  double backward_seconds = 0.0;
  double sync_seconds = 0.0;
  double optimizer_seconds = 0.0;
  double step_seconds = 0.0;
  std::int64_t steps = 1;
  double run_seconds = 0.0;

  // The numbers a benchmark reports, evaluated at the bound
  // (docs/performance-model.md section 7.4).
  std::int64_t tokens_per_step = 0;
  double tokens_per_second = 0.0;
  // The precision-weighted peak of the model rows, per device: the harmonic
  // mean of the row peaks, weighted by the row arithmetic. It reads 1.0 when
  // every model row runs at its own peak, so a measured MFU and `model_mfu`
  // share one denominator. Zero when the model rows carry no arithmetic.
  double weighted_peak_flops = 0.0;
  // The model arithmetic over the weighted peak, and the Muon arithmetic over
  // the compute-type peak. The two are never added: `model_mfu` divides the
  // count that `mfu.h` reports beside the optimizer's, exactly as the training
  // log does.
  double model_mfu = 0.0;
  double optim_mfu = 0.0;
  // The Muon orthogonalisation arithmetic for one step, over the whole world.
  double optimizer_flops = 0.0;
};

// The roofline estimator of docs/performance-model.md section 7. It uses the
// device peaks directly and holds no fitted value.
PerformanceEstimate EstimatePerformance(const Config& config,
                                        const PerformanceLimits& limits,
                                        const ModelPlan& plan);

// True when `flops` and `bytes` at `limits` are compute bound, and false when
// they are memory bound. It is the roofline of section 7.1 without a device.
bool IsComputeBound(double flops, double bytes, const PerformanceLimits& limits,
                    ComputePrecision precision);

// A human-readable table of `estimate`, one row per operation, plus the bound.
std::string FormatPerformanceReport(const PerformanceEstimate& estimate);

// The same table as the `nanochat.roofline.v2` JSON object. It carries the row
// fields, the phase split, the tokens per second, and the two utilization
// numbers, so a benchmark report and a bound join field by field.
std::string PerformanceReportJson(const PerformanceEstimate& estimate);

}  // namespace nanochat

#endif  // NANOCHAT_MFU_H_
