// tests/performance_model_test.cc -- the performance-model gate
// (docs/performance-model.md section 12).
//
// The estimator must follow the counter table, the precision assignment, and
// the device limit table. The checks are objective and need no GPU:
//
//   1. The counter struct sums to `EstimateFlopsPerToken` for every committed
//      configuration.
//   2. The precision assignment matches the fp8 rule in section 5.1.
//   3. The roofline labels the dense attention and the cast as memory bound,
//      and the matrix multiplies as compute bound.
//   4. The estimator reproduces the bound in section 7.2 for the depth-24
//      shape.
//
// It is tier S0: a plain T0 CPU test, no fixture and no broker.

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "nanochat/config.h"
#include "nanochat/mfu.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::ComputePrecision;
using nanochat::Config;
using nanochat::ModelPlan;
using nanochat::PerformanceEstimate;
using nanochat::PerformanceLimits;
using nanochat::RooflineRow;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

std::string Format(const char* fmt, ...) {
  char buffer[512];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  return std::string(buffer);
}

void ExpectTrue(bool condition, const std::string& what) {
  if (!condition) Fail(what);
}

void ExpectNear(double actual, double expected, double tolerance,
                const std::string& what) {
  if (!(std::fabs(actual - expected) <= tolerance)) {
    Fail(Format("%s: got %.6g, want %.6g", what.c_str(), actual, expected));
  }
}

void ExpectEqual(ComputePrecision actual, ComputePrecision expected,
                 const std::string& what) {
  if (actual != expected) {
    Fail(Format("%s: got %s, want %s", what.c_str(),
                nanochat::PrecisionName(actual),
                nanochat::PrecisionName(expected)));
  }
}

Config TinyConfig() {
  Config config;
  config.num_layers = 2;
  config.num_heads = 8;
  config.num_kv_heads = 2;
  config.hidden_dim = 32;
  config.seq_len = 8;
  config.vocab_size = 64;
  config.padded_vocab_size = 64;
  config.window_pattern = "SL";
  return config;
}

Config D12Config() {
  Config config;
  config.num_layers = 12;
  config.num_heads = 6;
  config.num_kv_heads = 6;
  config.hidden_dim = 768;
  config.seq_len = 1024;
  config.padded_vocab_size = 32768;
  config.window_pattern = "SSSL";
  return config;
}

// The depth-24 run shape of docs/node-h100.md.
Config D24Config() {
  Config config;
  config.num_layers = 24;
  config.num_heads = 12;
  config.num_kv_heads = 12;
  config.hidden_dim = 1536;
  config.seq_len = 2048;
  config.padded_vocab_size = 32768;
  config.window_pattern = "SSSL";
  return config;
}

void TestCounterSum() {
  const std::vector<Config> configs = {TinyConfig(), D12Config(), D24Config()};
  for (std::size_t i = 0; i < configs.size(); ++i) {
    const Config& config = configs[i];
    const nanochat::FlopCounts counts = nanochat::CountFlopsPerToken(config);
    ExpectNear(
        counts.total(), nanochat::EstimateFlopsPerToken(config), 1e-6,
        Format("counter sum == EstimateFlopsPerToken for config %zu", i));
    const double fields[] = {
        counts.gemm_forward,       counts.gemm_backward,
        counts.attention_forward,  counts.attention_backward,
        counts.norm_and_pointwise, counts.classifier,
        counts.embedding};
    for (double field : fields) {
      ExpectTrue(field >= 0.0, "every counter field is non-negative");
    }
  }
  // The depth-24 arithmetic matches the value recorded in docs/node-h100.md.
  ExpectNear(nanochat::EstimateFlopsPerToken(D24Config()), 4.775e9, 5e6,
             "depth-24 FLOP/token");
}

void TestPrecisionAssignment() {
  using nanochat::AssignMatrixPrecision;
  // Every matrix of the depth-24 model converts, because both dimensions are
  // divisible by 16 and the smaller dimension is at least 128.
  ExpectEqual(AssignMatrixPrecision(1536, 1536, ComputePrecision::kFp8),
              ComputePrecision::kFp8, "c_q converts");
  ExpectEqual(AssignMatrixPrecision(6144, 1536, ComputePrecision::kFp8),
              ComputePrecision::kFp8, "mlp.c_fc converts");
  ExpectEqual(AssignMatrixPrecision(32768, 1536, ComputePrecision::kFp8),
              ComputePrecision::kFp8, "lm_head converts");
  // The two gates stay bf16: the smaller dimension is below 128.
  ExpectEqual(AssignMatrixPrecision(12, 12, ComputePrecision::kFp8),
              ComputePrecision::kBf16, "ve_gate stays bf16");
  ExpectEqual(AssignMatrixPrecision(1, 24, ComputePrecision::kFp8),
              ComputePrecision::kBf16, "smear_gate stays bf16");
  // A dimension that is not divisible by 16 stays bf16.
  ExpectEqual(AssignMatrixPrecision(120, 120, ComputePrecision::kFp8),
              ComputePrecision::kBf16, "120 is not divisible by 16");
  // The boundary: 128 converts, 64 does not.
  ExpectEqual(AssignMatrixPrecision(128, 128, ComputePrecision::kFp8),
              ComputePrecision::kFp8, "128 converts");
  ExpectEqual(AssignMatrixPrecision(64, 1536, ComputePrecision::kFp8),
              ComputePrecision::kBf16, "64 stays bf16");
  // Without an fp8 build, the matrix keeps the build precision.
  ExpectEqual(AssignMatrixPrecision(1536, 1536, ComputePrecision::kFp32),
              ComputePrecision::kFp32, "fp32 build stays fp32");
  ExpectEqual(AssignMatrixPrecision(1536, 1536, ComputePrecision::kBf16),
              ComputePrecision::kBf16, "bf16 build stays bf16");
}

void TestRooflineLabels() {
  const PerformanceLimits limits = nanochat::LimitsForDevice("h100");
  // The c_q GEMM at the depth-24 shape is compute bound (intensity 750).
  const double gemm_flops = 2.0 * 32768.0 * 1536.0 * 1536.0;
  const double gemm_bytes =
      2.0 * (32768.0 * 1536.0 + 1536.0 * 1536.0 + 32768.0 * 1536.0);
  ExpectTrue(nanochat::IsComputeBound(gemm_flops, gemm_bytes, limits,
                                      ComputePrecision::kBf16),
             "c_q is compute bound");
  // The dense attention scores (2048 x 2048 x 128) are memory bound
  // (intensity 114).
  const double score_flops = 2.0 * 2048.0 * 2048.0 * 128.0;
  const double score_bytes =
      2.0 * (2048.0 * 128.0 + 128.0 * 2048.0 + 2048.0 * 2048.0);
  ExpectTrue(!nanochat::IsComputeBound(score_flops, score_bytes, limits,
                                       ComputePrecision::kBf16),
             "the attention scores are memory bound");
  // The cast issues no arithmetic and is memory bound.
  ExpectTrue(
      !nanochat::IsComputeBound(0.0, 1e9, limits, ComputePrecision::kFp8),
      "the quantization cast is memory bound");

  // The report rows carry the same labels.
  Config config = D24Config();
  ModelPlan plan;
  plan.batch = 16;
  plan.grad_accum = 4;
  plan.world_size = 8;
  plan.build = ComputePrecision::kFp8;
  const PerformanceEstimate estimate =
      nanochat::EstimatePerformance(config, limits, plan);
  bool saw_compute = false;
  bool saw_memory = false;
  for (const RooflineRow& row : estimate.rows) {
    if (row.name == "gemm_forward:fp8") {
      ExpectTrue(row.limit == "compute", "the matrix row is compute bound");
      saw_compute = true;
    }
    if (row.name == "quantization_cast") {
      ExpectTrue(row.limit == "memory", "the cast row is memory bound");
      saw_memory = true;
    }
  }
  ExpectTrue(saw_compute, "the report has a matrix row");
  ExpectTrue(saw_memory, "the report has a cast row");
}

double RowSeconds(const PerformanceEstimate& estimate,
                  const std::string& prefix) {
  double total = 0.0;
  for (const RooflineRow& row : estimate.rows) {
    if (row.name.compare(0, prefix.size(), prefix) == 0) {
      total += row.bound_seconds;
    }
  }
  return total;
}

void TestDepth24Bound() {
  Config config = D24Config();
  const PerformanceLimits limits = nanochat::LimitsForDevice("h100");
  ModelPlan plan;
  plan.batch = 16;
  plan.grad_accum = 4;
  plan.world_size = 8;
  plan.total_tokens = 5840000000LL;
  plan.link_bandwidth_bytes_per_second = 600e9;
  plan.build = ComputePrecision::kFp8;
  const PerformanceEstimate estimate =
      nanochat::EstimatePerformance(config, limits, plan);

  ExpectTrue(estimate.steps == 5570, "the depth-24 run has 5570 steps");
  // The rows are per optimizer step; the run bound multiplies by the
  // step count.
  const double steps = static_cast<double>(estimate.steps);
  const double matmul_min = RowSeconds(estimate, "gemm_") * steps / 60.0;
  const double attention_min =
      RowSeconds(estimate, "attention_") * steps / 60.0;
  const double cast_min =
      RowSeconds(estimate, "quantization_cast") * steps / 60.0;
  const double sync_min = estimate.sync_seconds * steps / 60.0;
  // The table in docs/performance-model.md section 7.2, fp8 column.
  ExpectNear(matmul_min, 27.0, 1.5, "depth-24 fp8 matrix bound");
  ExpectNear(cast_min, 13.0, 1.5, "depth-24 cast bound");
  ExpectNear(attention_min, 5.0, 0.8, "depth-24 attention bound");
  ExpectNear(sync_min, 1.5, 0.8, "depth-24 sync bound");
  ExpectNear(matmul_min + cast_min + attention_min + sync_min, 47.0, 3.0,
             "depth-24 fp8 subtotal");

  // The bf16-everywhere column is 61 minutes.
  ModelPlan bf16_plan = plan;
  bf16_plan.build = ComputePrecision::kBf16;
  const PerformanceEstimate bf16 =
      nanochat::EstimatePerformance(config, limits, bf16_plan);
  const double bf16_min = (RowSeconds(bf16, "gemm_") +
                           RowSeconds(bf16, "attention_") + bf16.sync_seconds) *
                          steps / 60.0;
  ExpectNear(bf16_min, 61.0, 3.0, "depth-24 bf16 subtotal");
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  TestCounterSum();
  TestPrecisionAssignment();
  TestRooflineLabels();
  TestDepth24Bound();
  if (g_failures != 0) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("performance model: all checks passed\n");
  return 0;
}
