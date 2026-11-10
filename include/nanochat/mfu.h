#ifndef NANOCHAT_MFU_H_
#define NANOCHAT_MFU_H_

#include <cstdint>
#include <string>
#include <vector>

#include "nanochat/config.h"
#include "nanochat/tensor.h"

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

// Peak device FLOPs for a known device name, or 0 when unknown. The rate is the
// `reference_peak_flops` the device's declaration carries
// (`backends/<name>/caps.json`), so a card's reference rate lives beside its
// other rates and this function holds no table of its own. It is the
// denominator of the reference Model FLOPs Utilization line, which counts at
// one precision per card.
double PeakFlopsForDevice(const std::string& device_name);

// Peak dense FLOPs for `device_name` at the build's compute precision
// (docs/precision.md). fp32 is the only compute type today, so the MFU
// denominator is the CUDA-core fp32 rate. A future bf16 build reads the
// tensor-core rate. `PeakFlopsForDevice` reads the reference rate, which is the
// tensor-core rate on a device whose reference recipe uses tensor cores, so the
// two differ on such a device by design. Both read the same declaration.
// Returns 0 for an unknown device.
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
//
// It is `BuildPrecision` from `nanochat/tensor.h` rather than an enumeration of
// its own: the build precision, the benchmark's precision, and a report row's
// precision are one vocabulary, so no caller maps between two enums that hold
// the same three values (docs/performance-model.md section 6).
using ComputePrecision = BuildPrecision;

// The short name of `precision`, for a report row.
const char* PrecisionName(ComputePrecision precision);

// The build's compute precision, from `kBuildPrecision` in
// `nanochat/tensor.h`. It is the fallback assignment for an operation that has
// no fp8 path in the current tree. The two names are one type, so this is the
// build precision itself.
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
// limits, so a caller must resolve the name first: `FindDeviceProfile` in
// `nanochat/device_profile.h` returns null for a name no declaration owns, and
// that header's `DeviceNameError` renders the message a command prints before
// it exits. The limit table is a rate table: it does not depend on the build or
// on the precision assignment.
PerformanceLimits LimitsForDevice(const std::string& device_name);

// The plan the estimator needs beyond `Config`: the batch, the topology, and
// the run length. `total_tokens` 0 models one optimizer step.
struct ModelPlan {
  // The device the limit table is read for, and the name the report carries.
  // Empty means the caller passes the limits itself and the report names no
  // device.
  std::string device_name;
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

// The throughput shape of the reference: 20 layers of width 1280, ten query
// heads over five key-value heads, 1024 tokens of context, a 32768 word
// vocabulary, and the `SSL` window. It is the `big` profile of the chain and
// the shape `profiles/pgo/big` measures, so a bound and a measurement state one
// shape (docs/performance-model.md section 7.3).
//
// The estimator's fields it does not set keep the `Config` default, exactly as
// a caller that passes the shape as flags leaves them. That is what makes the
// report match a run: a flag overrides one field, and every other field stays.
Config ReferenceThroughputConfig();

// The plan that pairs with `ReferenceThroughputConfig`: one micro-batch of
// sixteen sequences, which is 16384 tokens per optimizer step, and the
// reference token budget of 5.84 billion at ratio 8
// (docs/host-portability.md Part 1), with the five Polar Express iterations the
// shipped optimizer runs, read against the `h100` rate table. A run length is
// in the plan because a step bound alone says nothing about a run.
ModelPlan ReferenceThroughputPlan();

// The phase of one optimizer step that a counter row belongs to
// (docs/performance-model.md section 7.4). The names match the phase split of
// the training log, so a bound and a measurement join phase by phase.
enum class StepPhase { kForward, kBackward, kSync, kOptimizer };

// The short name of `phase`, for a report row.
const char* PhaseName(StepPhase phase);

// One operation of the counter table, with its roofline result.
//
// A row carries a join identity so a predicted row and a measured row compare
// field by field (docs/performance-model.md section 7.4): `family` is the
// operation, `phase` is the phase of the step it belongs to, `precision` is the
// rate its arithmetic is charged at, and the run's `shape` is on the estimate.
// The four are strings on both sides of the join, so a comparison needs no
// conversion step.
struct RooflineRow {
  std::string name;    // the display spelling; equal to `family`
  std::string family;  // the join key: the operation, with no phase in it
  StepPhase phase = StepPhase::kForward;
  ComputePrecision precision = ComputePrecision::kFp32;
  double flops = 0.0;
  // The work the operation issues, against the `flops` it must do. The counter
  // table counts necessary work, so this starts equal to `flops` and the
  // predicted ratio reads 1.0 wherever the model knows of no redundancy
  // (docs/performance-model.md section 8).
  double issued_flops = 0.0;
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
  // The identity of the run the bound describes: the device the limit table was
  // read for, the build precision, and the workload shape. They are the header
  // a report states, and the shape is the workload key a measured row carries
  // (docs/performance-model.md section 7.4).
  std::string device;
  std::string precision;
  std::string shape;
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

// The same estimator, with the limits resolved from `plan.device_name` through
// `LimitsForDevice`. It is the form a command uses: the device the report names
// and the rate table the bound divides by then come from one lookup, so a
// report cannot carry one device's name beside another device's rates.
PerformanceEstimate EstimatePerformance(const Config& config,
                                        const ModelPlan& plan);

// True when `flops` and `bytes` at `limits` are compute bound, and false when
// they are memory bound. It is the roofline of section 7.1 without a device.
bool IsComputeBound(double flops, double bytes, const PerformanceLimits& limits,
                    ComputePrecision precision);

// A human-readable table of `estimate`, one row per operation, plus the bound.
std::string FormatPerformanceReport(const PerformanceEstimate& estimate);

// The same table as the `nanochat.roofline.v3` JSON object. It carries the row
// fields, the identity of the run, the phase split, the tokens per second, and
// the two utilization numbers. A row states its duration as `ms` and its
// predicted work ratio, which are the two keys a `nanochat.bench.v3` row also
// states, so a bound and a measurement join on `family`, `phase`, `precision`,
// and `shape` (docs/performance.md section 3.4).
std::string PerformanceReportJson(const PerformanceEstimate& estimate);

}  // namespace nanochat

#endif  // NANOCHAT_MFU_H_
