// src/mesh_train.cc -- the mesh training driver
// (docs/client-runtime-design.md section 10.1 and docs/training-seam.md
// section 6.6).
//
// The driver owns the step loop, the metric record, the summary, the
// checkpoint interval, and the resume. It drives one `MeshRuntime` and one
// `DataLoader`. The JSON serializer owns the job file, so Python does not own
// a second schema.
//
// This translation unit is CPU-safe: no vendor header and no CUDA call.

#include "src/mesh_train.h"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/dataloader.h"
#include "nanochat/mesh.h"
#include "nanochat/mfu.h"
#include "nanochat/tokenizer.h"
#include "src/mesh_runtime.h"
#include "src/parquet/reader.h"
#include "src/train.h"

namespace nanochat {
namespace {

// The interrupt flag. The handler sets it and the loop stops at the next step
// boundary, so the driver saves a checkpoint before it returns
// (docs/client-runtime-design.md section 10.1).
volatile std::sig_atomic_t g_interrupted = 0;

void OnInterrupt(int) { g_interrupted = 1; }

// The parquet factory. Each call opens the dataset again, so
// `DataLoader::Reset` can cycle epochs (docs/parquet-native.md).
DocumentSourceFactory MakeParquetFactory(std::vector<std::string> files,
                                         std::string column) {
  return [files = std::move(files), column = std::move(column)](
             std::string* error) -> std::unique_ptr<DocumentSource> {
    return OpenParquetSource(files, column, 256, error);
  };
}

void EnsureDirectory(const std::string& path) {
  if (path.empty()) return;
  std::error_code code;
  std::filesystem::create_directories(path, code);
}

double ElapsedSeconds(const std::chrono::steady_clock::time_point& from,
                      const std::chrono::steady_clock::time_point& to) {
  return std::chrono::duration<double>(to - from).count();
}

// The model FLOPs utilization, clamped to `[0, 1]`. A nonpositive peak returns
// zero (docs/client-runtime-design.md section 10.1).
double MfuOf(double tokens_per_sec, double flops_per_token, double peak_flops) {
  if (peak_flops <= 0.0) return 0.0;
  const double value = tokens_per_sec * flops_per_token / peak_flops;
  return std::min(std::max(value, 0.0), 1.0);
}

// ---------------------------------------------------------------------------
// A minimal JSON writer. The key order is sorted, so the file is deterministic.
// ---------------------------------------------------------------------------

std::string JsonString(const std::string& value) {
  std::string out = "\"";
  for (const char raw : value) {
    const unsigned char byte = static_cast<unsigned char>(raw);
    switch (raw) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (byte < 0x20) {
          char buffer[8];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x", byte);
          out += buffer;
        } else {
          out += raw;
        }
        break;
    }
  }
  out += "\"";
  return out;
}

std::string JsonDouble(double value) {
  char buffer[40];
  std::snprintf(buffer, sizeof(buffer), "%.17g", value);
  return buffer;
}

std::string JsonFloatArray(const std::vector<float>& values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) out += ",";
    out += JsonDouble(static_cast<double>(values[i]));
  }
  out += "]";
  return out;
}

// One metric record as one JSON line. The keys are sorted.
std::string RecordJson(const MeshStepRecord& record) {
  std::string out = "{";
  out += "\"device_loss\":" + JsonFloatArray(record.device_loss) + ",";
  out += "\"grad_norm\":" + JsonDouble(record.grad_norm) + ",";
  out += "\"loss\":" + JsonDouble(record.loss) + ",";
  out += "\"mfu\":" + JsonDouble(record.mfu) + ",";
  out += "\"seconds\":" + JsonDouble(record.seconds) + ",";
  out += "\"step\":" + std::to_string(record.step) + ",";
  out += "\"step_seconds\":" + JsonDouble(record.step_seconds) + ",";
  out += "\"tokens_per_sec\":" + JsonDouble(record.tokens_per_sec) + ",";
  out += "\"version\":" + std::to_string(kMeshTrainSchemaVersion);
  out += "}";
  return out;
}

std::string HexDigest(std::uint64_t value) {
  char buffer[24];
  std::snprintf(buffer, sizeof(buffer), "%016llx",
                static_cast<unsigned long long>(value));
  return buffer;
}

std::string SummaryJson(const MeshTrainSummary& summary) {
  std::string digests = "[";
  for (std::size_t i = 0; i < summary.replica_digests.size(); ++i) {
    if (i > 0) digests += ",";
    digests += JsonString(HexDigest(summary.replica_digests[i]));
  }
  digests += "]";
  std::string out = "{";
  out += "\"checkpoint\":" + JsonString(summary.checkpoint) + ",";
  out += "\"mean_loss\":" + JsonDouble(summary.mean_loss) + ",";
  out += "\"replica_digests\":" + digests + ",";
  out += "\"steady_mfu\":" + JsonDouble(summary.steady_mfu) + ",";
  out += "\"steps\":" + std::to_string(summary.steps) + ",";
  out += "\"version\":" + std::to_string(kMeshTrainSchemaVersion);
  out += "}";
  return out;
}

}  // namespace

std::string MeshTrainConfig::checkpoint() const {
  return checkpoint_path.empty() ? out_dir + "/checkpoint.ckpt"
                                 : checkpoint_path;
}

std::string MeshTrainConfig::metrics() const {
  return metrics_path.empty() ? out_dir + "/metrics.jsonl" : metrics_path;
}

std::string MeshTrainConfig::summary() const {
  return summary_path.empty() ? out_dir + "/summary.json" : summary_path;
}

MeshTrainDriver::MeshTrainDriver(MeshTrainConfig config)
    : config_(std::move(config)) {
  std::string error;
  if (!config_.mesh.Validate(&error)) {
    throw std::invalid_argument(error);
  }
  if (config_.mesh.size() < 1) {
    throw std::invalid_argument("the mesh must have at least one device");
  }

  model_ = config_.model;
  if (config_.seq > 0) {
    model_.seq_len = config_.seq;
  } else {
    config_.seq = model_.seq_len;
  }
  if (config_.batch <= 0 || config_.seq <= 0) {
    throw std::invalid_argument("the batch and the sequence must be positive");
  }

  // The preset owns the optimizer and the schedule. The run length is
  // absolute, so a resume continues the schedule (docs/training-seam.md
  // section 10).
  OptimizerConfig optimizer;
  SchedulerConfig scheduler;
  if (config_.preset.empty() || config_.preset == "track3") {
    const TrainConfig baseline = TrackThreeBaseline();
    optimizer = baseline.optimizer;
    scheduler = baseline.scheduler;
  }
  scheduler.num_iterations = config_.num_iterations;

  MeshRuntimeConfig runtime_config;
  runtime_config.model = model_;
  runtime_config.optimizer = optimizer;
  runtime_config.scheduler = scheduler;
  runtime_config.batch = config_.batch;
  runtime_config.grad_accum = config_.grad_accum;
  runtime_config.seq = config_.seq;
  runtime_config.seed = config_.seed;
  runtime_ =
      std::make_unique<MeshRuntime>(config_.mesh, std::move(runtime_config));

  tokenizer_ = LoadTokenizer(config_.tokenizer_path);
  if (tokenizer_ == nullptr) {
    throw std::runtime_error("cannot load tokenizer: " +
                             config_.tokenizer_path);
  }
  DocumentSourceFactory factory =
      config_.train_source
          ? config_.train_source
          : MakeParquetFactory(config_.train_parquet, config_.text_column);
  loader_ = std::make_unique<DataLoader>(
      std::move(factory), tokenizer_.get(), config_.batch,
      config_.effective_seq(), config_.data_seed, config_.tokenizer_threads,
      static_cast<std::size_t>(config_.document_buffer));

  // The MFU inputs. A zero override selects the model estimate and the active
  // device table, so the client does not pass a device name.
  std::string device_name = config_.device_name;
  if (device_name.empty()) {
    const std::vector<DeviceInfo> devices = EnumerateDevices();
    if (!devices.empty() && devices.front().name != nullptr) {
      device_name = devices.front().name;
    }
  }
  flops_per_token_ = config_.flops_per_token > 0.0
                         ? config_.flops_per_token
                         : EstimateFlopsPerToken(model_);
  peak_flops_ = config_.peak_flops > 0.0 ? config_.peak_flops
                                         : PeakFlopsForCompute(device_name);

  tokens_.assign(static_cast<std::size_t>(config_.batch) *
                     static_cast<std::size_t>(config_.effective_seq()),
                 0);
  targets_.assign(tokens_.size(), 0);

  // A resume loads the parameters, the optimizer state, and the step, then
  // replays the consumed batches, so the data stream also continues
  // (docs/client-runtime-design.md section 10.1).
  if (config_.resume) {
    std::ifstream probe(config_.checkpoint(), std::ios::binary);
    if (probe.good()) {
      bool loaded = true;
      try {
        runtime_->Load(config_.checkpoint());
      } catch (const std::exception&) {
        loaded = false;
      }
      if (loaded) {
        start_step_ = runtime_->step() + 1;
        for (int step = 1; step < start_step_; ++step) {
          if (!LoadBatch(&tokens_, &targets_)) break;
        }
      }
    }
  }
}

MeshTrainDriver::~MeshTrainDriver() = default;

bool MeshTrainDriver::LoadBatch(std::vector<int>* tokens,
                                std::vector<int>* targets) {
  if (loader_ == nullptr) return false;
  if (loader_->Next(tokens->data(), targets->data())) return true;
  loader_->Reset();
  return loader_->Next(tokens->data(), targets->data());
}

void MeshTrainDriver::Save(int step) {
  if (step <= 0) return;
  try {
    runtime_->Save(config_.checkpoint());
  } catch (const std::exception& error) {
    std::fprintf(stderr, "mesh_train: save failed: %s\n", error.what());
  }
}

void MeshTrainDriver::WriteSummary() {
  double loss_sum = 0.0;
  for (const MeshStepRecord& record : records_) loss_sum += record.loss;
  summary_.steps = static_cast<int>(records_.size());
  summary_.mean_loss =
      records_.empty() ? 0.0 : loss_sum / static_cast<double>(records_.size());

  const std::size_t half = records_.size() / 2;
  double tail_sum = 0.0;
  for (std::size_t i = half; i < records_.size(); ++i) {
    tail_sum += records_[i].mfu;
  }
  summary_.steady_mfu =
      records_.size() > half
          ? tail_sum / static_cast<double>(records_.size() - half)
          : 0.0;

  summary_.replica_digests.clear();
  if (runtime_ != nullptr) {
    for (int rank = 0; rank < runtime_->world_size(); ++rank) {
      try {
        summary_.replica_digests.push_back(runtime_->ParamDigest(rank));
      } catch (const std::exception&) {
        break;
      }
    }
  }
  summary_.checkpoint = config_.checkpoint();

  EnsureDirectory(config_.out_dir);
  std::ofstream out(config_.summary(), std::ios::out | std::ios::trunc);
  if (!out) return;
  out << SummaryJson(summary_) << "\n";
}

float MeshTrainDriver::Run() {
  EnsureDirectory(config_.out_dir);
  records_.clear();
  last_loss_ = 0.0f;
  last_step_ = 0;

  const int steps = config_.num_iterations;
  const std::int64_t tokens_per_step =
      static_cast<std::int64_t>(config_.batch) * config_.effective_seq();
  const auto start = std::chrono::steady_clock::now();
  auto previous = start;
  g_interrupted = 0;

  const std::string metrics_path = config_.metrics();
  std::ofstream sink(
      metrics_path,
      std::ios::out | (start_step_ <= 1 ? std::ios::trunc : std::ios::app));
  if (!sink) {
    throw std::runtime_error("cannot open metrics: " + metrics_path);
  }

  using InterruptHandler = void (*)(int);
  const InterruptHandler previous_handler = std::signal(SIGINT, OnInterrupt);
  for (int step = start_step_; step <= steps; ++step) {
    if (g_interrupted) break;
    if (!LoadBatch(&tokens_, &targets_)) {
      std::fprintf(stderr, "mesh_train: the loader returned no batch\n");
      break;
    }
    const float loss = runtime_->TrainStep(tokens_.data(), targets_.data());
    const auto now = std::chrono::steady_clock::now();
    double step_seconds = ElapsedSeconds(previous, now);
    previous = now;
    if (step_seconds < 1e-9) step_seconds = 1e-9;

    const MeshMetrics& metrics = runtime_->metrics();
    MeshStepRecord record;
    record.step = step;
    record.loss = loss;
    record.grad_norm = metrics.grad_norm;
    record.tokens_per_sec = static_cast<double>(tokens_per_step) / step_seconds;
    record.mfu = MfuOf(record.tokens_per_sec, flops_per_token_, peak_flops_);
    record.step_seconds = step_seconds;
    record.compute_seconds = static_cast<double>(metrics.step_seconds);
    record.seconds = ElapsedSeconds(start, now);
    record.device_loss = metrics.loss;
    records_.push_back(record);
    sink << RecordJson(record) << "\n";
    sink.flush();

    last_loss_ = loss;
    last_step_ = step;

    if (config_.save_every > 0 && step % config_.save_every == 0) Save(step);
    const bool show =
        step == start_step_ || step == steps ||
        (config_.log_every > 0 && step % std::max(1, config_.log_every) == 0);
    if (show) {
      std::printf(
          "[%5d/%d] loss %.4f  grad_norm %.4g  step %.2fs (compute %.2fs)  "
          "%.0f tok/s  mfu %.2f%%\n",
          step, steps, static_cast<double>(record.loss),
          static_cast<double>(record.grad_norm), record.step_seconds,
          record.compute_seconds, record.tokens_per_sec, record.mfu * 100.0);
      std::fflush(stdout);
    }
  }
  std::signal(SIGINT, previous_handler);

  if (last_step_ > 0) Save(last_step_);
  summary_.interrupted = g_interrupted != 0;
  WriteSummary();
  return last_loss_;
}

}  // namespace nanochat
