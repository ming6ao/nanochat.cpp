// `roofline_main` -- the performance-model report
// (docs/performance-model.md).
//
// It prints the operation counter table, the roofline result, and the step and
// run bound for one configuration and one device. Every term comes from the
// device limit table and the counter table; the tool holds no fitted value and
// no measurement.
//
// The tool is CPU-only. It reads no fixture, allocates no device memory, and
// needs no GPU broker. Run it through the entry point:
//
//   tools/nanochat roofline --devices 8 --device h100 --precision fp8
//       --layers 24 --heads 12 --kv-heads 12 --hidden 1536 --seq 2048
//       --batch 16 --grad-accum 4 --tokens 5.84e9 --link-gbps 600
//
// The `tools/nanochat roofline` command builds this binary and runs it under
// the `t0-cpu` sandbox profile.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "nanochat/mfu.h"
#include "nanochat/sandbox.h"
#include "src/cli.h"

namespace {

void Usage() {
  std::fprintf(
      stderr,
      "usage: roofline_main [options]\n"
      "  --device NAME          device profile (default h100)\n"
      "  --precision P          fp32|bf16|fp8, or auto (default auto)\n"
      "  --batch N              micro-batch per device (default 1)\n"
      "  --grad-accum N         micro-batches per step (default 1)\n"
      "  --devices N            data-parallel world size (default 1)\n"
      "  --tokens N             total run tokens; 0 for one step\n"
      "  --link-gbps N          per-rank link bandwidth (default 0)\n"
      "  --latency-us N         link latency in microseconds (default 5)\n"
      "  --overlap F            fraction of the sync hidden (default 0)\n"
      "  --reference-copies     model the reference per-step weight copies\n"
      "  --json                 write nanochat.roofline.v1 JSON\n"
      "  --help                 this message\n"
      "  [model flags: --layers --heads --kv-heads --hidden --seq\n"
      "   --vocab --padded-vocab --window-pattern --rope-base]\n");
}

bool ParseDouble(const char* text, double* out) {
  if (text == nullptr || *text == '\0') return false;
  char* end = nullptr;
  const double value = std::strtod(text, &end);
  if (end == text || *end != '\0') return false;
  *out = value;
  return true;
}

bool ParsePrecision(const std::string& name, nanochat::ComputePrecision* out) {
  using nanochat::ComputePrecision;
  if (name == "fp32") {
    *out = ComputePrecision::kFp32;
  } else if (name == "bf16") {
    *out = ComputePrecision::kBf16;
  } else if (name == "fp8") {
    *out = ComputePrecision::kFp8;
  } else if (name == "auto") {
    *out = nanochat::BuildComputePrecision();
  } else {
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("roofline");

  nanochat::Config config;
  nanochat::ModelPlan plan;
  std::string device = "h100";
  std::string precision_name = "auto";
  bool json = false;

  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    const auto value = [&](const char** out) -> bool {
      if (i + 1 >= argc) return false;
      *out = argv[++i];
      return true;
    };
    const char* text = nullptr;
    if (flag == "--help" || flag == "-h") {
      Usage();
      return 0;
    } else if (flag == "--json") {
      json = true;
    } else if (flag == "--reference-copies") {
      plan.transposed_weights = false;
    } else if (nanochat::cli::IsModelFlag(flag)) {
      if (!value(&text) ||
          !nanochat::cli::ApplyModelFlag(flag, text, &config)) {
        std::fprintf(stderr, "roofline: bad value for %s\n", flag.c_str());
        return 2;
      }
    } else if (flag == "--device") {
      if (!value(&text)) return 2;
      device = text;
    } else if (flag == "--precision") {
      if (!value(&text)) return 2;
      precision_name = text;
    } else if (flag == "--batch") {
      if (!value(&text) || !nanochat::cli::ParseInt(text, &plan.batch)) {
        return 2;
      }
    } else if (flag == "--grad-accum") {
      if (!value(&text) || !nanochat::cli::ParseInt(text, &plan.grad_accum)) {
        return 2;
      }
    } else if (flag == "--devices") {
      if (!value(&text) || !nanochat::cli::ParseInt(text, &plan.world_size)) {
        return 2;
      }
    } else if (flag == "--tokens") {
      double tokens = 0.0;
      if (!value(&text) || !ParseDouble(text, &tokens) || tokens < 0.0) {
        return 2;
      }
      plan.total_tokens = static_cast<std::int64_t>(tokens);
    } else if (flag == "--link-gbps") {
      double gbps = 0.0;
      if (!value(&text) || !ParseDouble(text, &gbps) || gbps < 0.0) {
        return 2;
      }
      plan.link_bandwidth_bytes_per_second = gbps * 1e9;
    } else if (flag == "--latency-us") {
      double micros = 0.0;
      if (!value(&text) || !ParseDouble(text, &micros) || micros < 0.0) {
        return 2;
      }
      plan.link_latency_seconds = micros * 1e-6;
    } else if (flag == "--overlap") {
      if (!value(&text) || !ParseDouble(text, &plan.sync_overlap)) return 2;
    } else {
      std::fprintf(stderr, "roofline: unknown option %s\n", flag.c_str());
      Usage();
      return 2;
    }
  }

  if (!ParsePrecision(precision_name, &plan.build)) {
    std::fprintf(stderr, "roofline: unknown precision %s\n",
                 precision_name.c_str());
    return 2;
  }

  const nanochat::PerformanceLimits limits = nanochat::LimitsForDevice(device);
  const nanochat::PerformanceEstimate estimate =
      nanochat::EstimatePerformance(config, limits, plan);

  if (json) {
    std::fputs(nanochat::PerformanceReportJson(estimate).c_str(), stdout);
    return 0;
  }

  const nanochat::FlopCounts counts = nanochat::CountFlopsPerToken(config);
  std::printf("performance model: %s, build %s\n", device.c_str(),
              nanochat::PrecisionName(plan.build));
  std::printf(
      "model: layers %d heads %d kv-heads %d hidden %d seq %d "
      "vocab %d window %s\n",
      config.num_layers, config.num_heads, config.num_kv_heads,
      config.hidden_dim, config.seq_len, config.padded_vocab_size,
      config.window_pattern.c_str());
  std::printf("plan: batch %d grad-accum %d devices %d tokens %lld\n",
              plan.batch, plan.grad_accum, plan.world_size,
              static_cast<long long>(plan.total_tokens));
  std::printf(
      "arithmetic: %.6g FLOP/token "
      "(gemm_fwd %.4g, gemm_bwd %.4g, attn_fwd %.4g, attn_bwd %.4g)\n",
      counts.total(), counts.gemm_forward, counts.gemm_backward,
      counts.attention_forward, counts.attention_backward);
  std::printf("\n%s", nanochat::FormatPerformanceReport(estimate).c_str());
  return 0;
}
