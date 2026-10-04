// Host utilities and the training driver (docs/model.md, harness workstream):
//
//   * `Logger` -- one structured `LogRecord` per step to stdout and an optional
//     run file;
//   * the `mfu.h` FLOP accounting -- the same `estimate_flops` /
//     `get_peak_flops` contract as nanochat;
//   * `Checkpointer` -- save/load through the self-describing `Checkpoint`
//     container;
//   * `TrainLoop` -- the `ForwardLoss` -> `Backward` -> `Optimizer::Step` loop
//     with evaluation and checkpointing.
//
// This translation unit is CPU-safe: no vendor headers, no CUDA calls. The
// compute itself stays behind `Model`/`Optimizer`.

#include "src/train.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/data.h"
#include "nanochat/dataloader.h"
#include "nanochat/kernels.h"
#include "nanochat/logger.h"
#include "nanochat/mfu.h"
#include "nanochat/model.h"
#include "nanochat/optim.h"
#include "nanochat/scheduler.h"
#include "nanochat/tensor.h"
#include "src/ops.h"

namespace nanochat {

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------

struct Logger::Impl {
  std::string path;
  std::ofstream file;
  bool to_file = false;
};

Logger::Logger() : impl_(std::make_unique<Impl>()) {}

Logger::Logger(const std::string& path) : impl_(std::make_unique<Impl>()) {
  impl_->path = path;
  if (!path.empty()) {
    impl_->file.open(path, std::ios::out | std::ios::app);
    impl_->to_file = impl_->file.is_open();
  }
}

Logger::~Logger() = default;

void Logger::Log(const LogRecord& record) {
  char buffer[256];
  std::snprintf(buffer, sizeof(buffer),
                "step %06d | loss %.6f | lr %.6g | grad_norm %.6g | "
                "tok/s %.3f | mfu %.2f%%",
                record.step, static_cast<double>(record.loss),
                static_cast<double>(record.lr),
                static_cast<double>(record.grad_norm),
                static_cast<double>(record.tokens_per_second),
                static_cast<double>(record.mfu) * 100.0);
  const std::string line(buffer);
  std::fputs(line.c_str(), stdout);
  std::fputc('\n', stdout);
  if (impl_->to_file) {
    impl_->file << line << '\n';
    impl_->file.flush();
  }
}

void Logger::Info(const std::string& message) {
  const std::string line = "[nanochat] " + message;
  std::fputs(line.c_str(), stdout);
  std::fputc('\n', stdout);
  if (impl_->to_file) {
    impl_->file << line << '\n';
    impl_->file.flush();
  }
}

void Logger::Close() {
  if (impl_->to_file) {
    impl_->file.close();
    impl_->to_file = false;
  }
}

// ---------------------------------------------------------------------------
// MFU / FLOP accounting (mirrors nanochat's `estimate_flops` and
// `get_peak_flops`)
// ---------------------------------------------------------------------------

namespace {

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

std::string ToLower(const std::string& value) {
  std::string result = value;
  for (char& c : result) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return result;
}

}  // namespace

double EstimateFlopsPerToken(const Config& config) {
  const int heads = config.num_heads;
  const int head_dim = config.head_dim();
  const int seq = config.seq_len;
  double attention = 0.0;
  for (int i = 0; i < config.num_layers; ++i) {
    const int left = config.window_left(i);
    const int effective = left < 0 ? seq : std::min(left, seq);
    attention += 12.0 * heads * head_dim * effective;
  }
  return 6.0 * NumMatmulParams(config) + attention;
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

double PeakFlopsForDevice(const std::string& device_name) {
  struct Entry {
    std::vector<const char*> patterns;
    double flops;
  };
  // Order matters: more specific patterns first. Mirrors nanochat's
  // `get_peak_flops`; unknown devices return 0 so MFU reads 0.
  static const Entry kTable[] = {
      // NVIDIA Blackwell
      {{"gb200"}, 2.5e15},
      {{"grace blackwell"}, 2.5e15},
      {{"b200"}, 2.25e15},
      {{"b100"}, 1.8e15},
      // NVIDIA Hopper
      {{"h200", "nvl"}, 836e12},
      {{"h200", "pcie"}, 836e12},
      {{"h200"}, 989e12},
      {{"h100", "nvl"}, 835e12},
      {{"h100", "pcie"}, 756e12},
      {{"h100"}, 989e12},
      {{"h800", "nvl"}, 989e12},
      {{"h800"}, 756e12},
      // NVIDIA Ampere data center
      {{"a100"}, 312e12},
      {{"a800"}, 312e12},
      {{"a40"}, 149.7e12},
      {{"a30"}, 165e12},
      // NVIDIA Ada data center
      {{"l40s"}, 362e12},
      {{"l40-s"}, 362e12},
      {{"l40 s"}, 362e12},
      {{"l4"}, 121e12},
      // AMD CDNA accelerators
      {{"mi355"}, 2.5e15},
      {{"mi325"}, 1.3074e15},
      {{"mi300x"}, 1.3074e15},
      {{"mi300a"}, 980.6e12},
      {{"mi250x"}, 383e12},
      {{"mi250"}, 362.1e12},
      // Consumer RTX
      {{"5090"}, 209.5e12},
      {{"4090"}, 165.2e12},
      {{"3090"}, 71e12},
      // Pascal (FP32 CUDA-core peak; no tensor cores)
      {{"1080 ti"}, 11.34e12},
  };

  const std::string name = ToLower(device_name);
  if (name.empty()) return 0.0;
  for (const Entry& entry : kTable) {
    bool match = true;
    for (const char* pattern : entry.patterns) {
      if (name.find(pattern) == std::string::npos) {
        match = false;
        break;
      }
    }
    if (match) return entry.flops;
  }
  return 0.0;
}

double ComputeMfu(const Config& config, double tokens_per_second,
                  double peak_flops) {
  if (!(peak_flops > 0.0) || !std::isfinite(peak_flops)) return 0.0;
  if (!(tokens_per_second > 0.0) || !std::isfinite(tokens_per_second)) {
    return 0.0;
  }
  const double mfu =
      tokens_per_second * EstimateFlopsPerToken(config) / peak_flops;
  if (mfu < 0.0) return 0.0;
  if (mfu > 1.0) return 1.0;
  return mfu;
}

// ---------------------------------------------------------------------------
// Checkpointer
// ---------------------------------------------------------------------------

bool Checkpointer::SaveModel(const Model& model, const std::string& path) {
  Checkpoint checkpoint;
  for (const ParamView& view : model.params()) {
    if (view.value == nullptr || view.count <= 0) continue;
    TensorRecord record;
    record.name = view.name != nullptr ? view.name : "";
    record.dtype = kComputeDType;
    // `assign` rather than an initializer-list assignment: GCC's -O2
    // -Warray-bounds analysis mis-handles the `std::vector` initializer-list
    // path for a one-element `std::int64_t` shape.
    record.shape.assign(1, view.count);
    const std::size_t bytes =
        static_cast<std::size_t>(view.count) * kComputeTypeSize;
    record.data.resize(bytes);
    kernels::Memcpy(record.data.data(), view.value, bytes,
                    CopyDir::kDeviceToHost);
    checkpoint.Add(std::move(record));
  }
  return checkpoint.Save(path);
}

bool Checkpointer::LoadModel(Model* model, const std::string& path) {
  if (model == nullptr) return false;
  Checkpoint checkpoint;
  std::string error;
  if (!checkpoint.Load(path, &error)) return false;
  for (const ParamView& view : model->params()) {
    const TensorRecord* record =
        checkpoint.Find(view.name != nullptr ? view.name : "");
    if (record == nullptr || record->dtype != kComputeDType) continue;
    const std::size_t wanted =
        static_cast<std::size_t>(view.count) * kComputeTypeSize;
    const std::size_t bytes = std::min(wanted, record->data.size());
    if (bytes == 0) continue;
    kernels::Memcpy(view.value, record->data.data(), bytes,
                    CopyDir::kHostToDevice);
  }
  return true;
}

// ---------------------------------------------------------------------------
// TrainLoop
// ---------------------------------------------------------------------------

TrainLoop::TrainLoop(TrainConfig config) : config_(std::move(config)) {
  if (config_.seq > 0) {
    config_.model.seq_len = config_.seq;
  } else {
    config_.seq = config_.model.seq_len;
  }
  const int seq = config_.effective_seq();
  if (config_.batch <= 0 || seq <= 0) {
    config_.batch = std::max(1, config_.batch);
    config_.seq = std::max(1, seq);
  }

  model_ = Model::Create(config_.model);

  bool resumed = false;
  if (!config_.resume_path.empty()) {
    std::ifstream probe(config_.resume_path, std::ios::binary);
    if (probe.good()) {
      resumed = Checkpointer::LoadModel(model_.get(), config_.resume_path);
    }
  }
  if (!resumed) model_->InitWeights(config_.seed);

  SchedulerConfig scheduler_config = config_.scheduler;
  if (scheduler_config.num_iterations <= 0) {
    scheduler_config.num_iterations = config_.num_iterations;
  }
  config_.scheduler = scheduler_config;
  scheduler_ = std::make_unique<Scheduler>(scheduler_config);
  optimizer_ = CreateOptimizer(model_.get(), config_.optimizer, *scheduler_);
  logger_ = std::make_unique<Logger>(config_.log_path);

  train_loader_ = std::make_unique<DataLoader>(
      config_.train_shards, config_.batch, config_.effective_seq(),
      config_.seed, config_.shuffle);
  if (!config_.val_shards.empty()) {
    val_loader_ = std::make_unique<DataLoader>(
        config_.val_shards, config_.batch, config_.effective_seq(),
        config_.seed + 1, false);
  }

  tokens_.assign(static_cast<std::size_t>(config_.batch) *
                     static_cast<std::size_t>(config_.effective_seq()),
                 0);
  targets_.assign(tokens_.size(), 0);
  peak_flops_ = PeakFlopsForDevice(config_.device_name);
}

TrainLoop::~TrainLoop() = default;

void TrainLoop::Save(int step) {
  if (config_.checkpoint_path.empty()) return;
  if (Checkpointer::SaveModel(*model_, config_.checkpoint_path)) {
    logger_->Info("saved checkpoint at step " + std::to_string(step));
  } else {
    logger_->Info("failed to save checkpoint at step " + std::to_string(step));
  }
}

float TrainLoop::Run() {
  const int seq = config_.effective_seq();
  const int batch = config_.batch;
  const std::int64_t tokens_per_step = static_cast<std::int64_t>(batch) * seq;
  logger_->Info("training for " + std::to_string(config_.num_iterations) +
                " steps, batch " + std::to_string(batch) + " x " +
                std::to_string(seq));

  for (int step = 1; step <= config_.num_iterations; ++step) {
    const auto start = std::chrono::steady_clock::now();

    // Gradient accumulation: sum `grad_accum` micro-batch gradients (each
    // scaled by 1/grad_accum) before one optimizer step. Mirrors nanochat's
    // `loss = loss / grad_accum_steps` before each backward.
    const int accum = config_.grad_accum > 0 ? config_.grad_accum : 1;
    const float inv_accum = 1.0f / static_cast<float>(accum);
    optimizer_->ZeroGrad();
    float loss_sum = 0.0f;
    for (int micro = 0; micro < accum; ++micro) {
      if (!train_loader_->Next(tokens_.data(), targets_.data())) {
        train_loader_->Reset();
        if (!train_loader_->Next(tokens_.data(), targets_.data())) break;
      }
      loss_sum +=
          model_->ForwardLoss(tokens_.data(), targets_.data(), batch, seq);
      model_->BackwardAccumulate(inv_accum);
    }
    optimizer_->Step(step);
    const float loss = loss_sum * inv_accum;

    const auto finish = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double>(finish - start).count();
    const double tokens_per_second =
        elapsed > 0.0 ? static_cast<double>(tokens_per_step) / elapsed : 0.0;

    last_loss_ = loss;
    last_step_ = step;

    if (config_.log_every > 0 && step % config_.log_every == 0) {
      LogRecord record;
      record.step = step;
      record.loss = loss;
      record.lr = config_.optimizer.matrix_lr * scheduler_->LrMultiplier(step);
      record.grad_norm = optimizer_->GradNorm();
      record.tokens_per_second = static_cast<float>(tokens_per_second);
      record.mfu = static_cast<float>(
          ComputeMfu(config_.model, tokens_per_second, peak_flops_));
      logger_->Log(record);
    }

    if (val_loader_ != nullptr && config_.eval_every > 0 &&
        step % config_.eval_every == 0) {
      const float bpb =
          EvalBpb(model_.get(), val_loader_.get(), config_.eval_steps);
      logger_->Info("step " + std::to_string(step) + " val bpb " +
                    std::to_string(bpb));
    }

    if (config_.save_every > 0 && step % config_.save_every == 0) {
      Save(step);
    }
  }

  if (!config_.checkpoint_path.empty()) Save(last_step_);
  logger_->Info("training complete at step " + std::to_string(last_step_) +
                ", last loss " + std::to_string(last_loss_));
  return last_loss_;
}

}  // namespace nanochat
