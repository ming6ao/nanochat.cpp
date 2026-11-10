// `roofline_main` -- the performance-model report
// (docs/performance-model.md).
//
// It prints the operation counter table, the roofline result, the phase split,
// and the step and run bound for one configuration and one device. Every term
// comes from the device limit table and the counter table; the tool holds no
// fitted value and no measurement. The phase split and the two utilization
// numbers are the predicted counterparts of the training log's phase fields
// (docs/performance-model.md section 7.4).
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

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

#include "nanochat/device_profile.h"
#include "nanochat/mfu.h"
#include "src/cli.h"
#include "src/command.h"

namespace nanochat {
namespace {

constexpr char kUsage[] =
    "usage: nanochat_cli roofline [options]   (alias: roofline [options])\n"
    "  --device NAME          device profile (default h100)\n"
    "  --list-devices         print the declared device names and exit\n"
    "  --precision P          fp32|bf16|fp8, or auto (default auto)\n"
    "  --batch N              micro-batch per device (default 16)\n"
    "  --grad-accum N         micro-batches per step (default 1)\n"
    "  --devices N            data-parallel world size (default 1)\n"
    "  --tokens N             total run tokens; 0 for one step\n"
    "                         (default 5.84e9, the reference budget)\n"
    "  --link-gbps N          per-rank link bandwidth (default 0)\n"
    "  --latency-us N         link latency in microseconds (default 5)\n"
    "  --overlap F            fraction of the sync hidden (default 0)\n"
    "  --reference-copies     model the reference per-step weight copies\n"
    "  --muon-ns-steps N      Muon iteration count; 0 keeps it memory bound\n"
    "                         (default 5, the shipped stage)\n"
    "  --json                 write nanochat.roofline.v3 JSON\n"
    "  --help                 this message\n"
    "  [model flags: --layers --heads --kv-heads --hidden --seq\n"
    "   --vocab --padded-vocab --window-pattern --rope-base\n"
    "   --value-embedding <0|1>]\n"
    "  [model default: the reference throughput shape, 20 layers of 1280\n"
    "   wide at 1024 tokens (--layers 20 --heads 10 --kv-heads 5\n"
    "   --hidden 1280 --seq 1024 --vocab 32768 --window-pattern SSL)]\n";

// Parses a whole token. `std::from_chars` needs no NUL terminator and ignores
// the locale, which `std::strtod` does not.
bool ParseDouble(std::string_view text, double* out) {
  if (text.empty()) return false;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), *out);
  return result.ec == std::errc() && result.ptr == text.data() + text.size();
}

bool ParsePrecision(std::string_view name, nanochat::ComputePrecision* out) {
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

constexpr char kProgram[] = "roofline";

int RunRoofline(int argc, char** argv) {
  // The default describes the model the card trains, not the smallest model the
  // tree can hold: a report at 8 layers of 768 wide describes a step no run
  // takes. One helper serves the command and the test, so the shape a report
  // states and the shape a test pins cannot drift.
  nanochat::Config config = nanochat::ReferenceThroughputConfig();
  nanochat::ModelPlan plan = nanochat::ReferenceThroughputPlan();
  std::string device = plan.device_name;
  std::string precision_name = "auto";
  bool json = false;

  // `--list-devices` prints the table and stops, so it is read after the parse.
  bool list_devices = false;
  cli::FlagParser flags(kProgram);
  flags.SetUsage(kUsage);
  flags.AddSwitch("--list-devices", &list_devices);
  flags.AddSwitch("--json", &json);
  flags.AddSwitchOff("--reference-copies", &plan.transposed_weights);
  flags.AddString("--device", &device);
  flags.AddString("--precision", &precision_name);
  flags.AddInt("--batch", &plan.batch);
  flags.AddInt("--grad-accum", &plan.grad_accum);
  flags.AddValue("--muon-ns-steps", [&plan](std::string_view value) {
    return cli::ParseInt(value, &plan.muon_ns_steps) && plan.muon_ns_steps >= 0;
  });
  flags.AddInt("--devices", &plan.world_size);
  flags.AddValue("--tokens", [&plan](std::string_view value) {
    double tokens = 0.0;
    if (!ParseDouble(value, &tokens) || tokens < 0.0) return false;
    plan.total_tokens = static_cast<std::int64_t>(tokens);
    return true;
  });
  flags.AddValue("--link-gbps", [&plan](std::string_view value) {
    double gbps = 0.0;
    if (!ParseDouble(value, &gbps) || gbps < 0.0) return false;
    plan.link_bandwidth_bytes_per_second = gbps * 1e9;
    return true;
  });
  flags.AddValue("--latency-us", [&plan](std::string_view value) {
    double micros = 0.0;
    if (!ParseDouble(value, &micros) || micros < 0.0) return false;
    plan.link_latency_seconds = micros * 1e-6;
    return true;
  });
  flags.AddValue("--overlap", [&plan](std::string_view value) {
    return ParseDouble(value, &plan.sync_overlap);
  });
  flags.AddModelFlags(&config);

  const int parse_status = flags.Parse(argc, argv);
  if (parse_status != -1) return parse_status;

  if (list_devices) {
    std::printf("%s\n", DeviceNameList().c_str());
    return 0;
  }

  if (!ParsePrecision(precision_name, &plan.build)) {
    std::fprintf(stderr, "roofline: unknown precision %s\n",
                 precision_name.c_str());
    return 2;
  }

  // An unknown device is refused, with the declared list in the message. A
  // zero limit table would produce a report of zeros that reads like a very
  // fast step, and the simulator refuses the same way with the same text
  // (docs/performance-model.md section 6).
  if (nanochat::FindDeviceProfile(device) == nullptr) {
    std::fprintf(stderr, "roofline: %s\n",
                 nanochat::DeviceNameError(device).c_str());
    return 2;
  }

  // The name the report carries and the rate table the bound divides by come
  // from one lookup, so the two cannot disagree.
  plan.device_name = device;
  const nanochat::PerformanceEstimate estimate =
      nanochat::EstimatePerformance(config, plan);

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
  std::printf(
      "plan: batch %d grad-accum %d devices %d tokens %lld muon-ns %d\n",
      plan.batch, plan.grad_accum, plan.world_size,
      static_cast<long long>(plan.total_tokens), plan.muon_ns_steps);
  std::printf(
      "arithmetic: %.6g FLOP/token "
      "(gemm_fwd %.4g, gemm_bwd %.4g, attn_fwd %.4g, attn_bwd %.4g)\n",
      counts.total(), counts.gemm_forward, counts.gemm_backward,
      counts.attention_forward, counts.attention_backward);
  std::printf("\n%s", nanochat::FormatPerformanceReport(estimate).c_str());
  return 0;
}
}  // namespace nanochat
