#ifndef NANOCHAT_SRC_TRAIN_H_
#define NANOCHAT_SRC_TRAIN_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "nanochat/config.h"
#include "nanochat/dataloader.h"
#include "nanochat/optim.h"
#include "nanochat/scheduler.h"
#include "src/distributed.h"

// The training driver's small internal interface (docs/model.md, harness
// workstream). Plain C++ with no vendor headers, so `*_main.cc` and the tests
// can include it; all compute stays behind `Model` and `Optimizer`.
//
// `TrainLoop` owns the model, optimizer, scheduler, logger, and data iterators,
// and runs `ForwardLoss`/`Backward`/`Optimizer::Step` (through
// `Model::TrainStep`) with evaluation and checkpointing at the configured
// intervals.

namespace nanochat {

class DataLoader;
class Logger;
class Model;
class Optimizer;
class Tokenizer;

struct TrainConfig {
  Config model;
  OptimizerConfig optimizer;
  SchedulerConfig scheduler;

  int batch = 8;
  int grad_accum = 1;  // micro-batches summed before each optimizer step
  int seq = 0;         // 0 selects `model.seq_len`
  int num_iterations = 100;
  int log_every = 1;
  int eval_every = 0;  // 0 disables evaluation
  int save_every = 0;  // 0 disables periodic checkpoints
  int eval_steps = 8;
  std::uint64_t seed = 42;
  std::string device_name;      // fed to `PeakFlopsForDevice` for MFU
  std::string checkpoint_path;  // final checkpoint written here
  std::string log_path;         // empty logs to stdout only
  std::string resume_path;      // optional checkpoint to resume from

  // Data (docs/parquet-native.md): read parquet and tokenize during the run.
  std::vector<std::string> train_parquet;
  std::vector<std::string> val_parquet;
  std::string tokenizer_path;  // the NCTOKEN1 artifact
  std::string text_column = "text";
  int tokenizer_threads = 4;
  int document_buffer = 1000;

  // Distributed (docs/distributed-design.md): data parallel rank and group.
  // `world_size` 1 keeps the single-process run unchanged.
  int rank = 0;
  int world_size = 1;
  std::string master = "127.0.0.1";
  int port = 29500;

  // Optional in-memory sources. A test sets them to bypass the parquet reader;
  // `train_main` leaves them empty and reads the `train_parquet` globs.
  DocumentSourceFactory train_source;
  DocumentSourceFactory val_source;

  int effective_seq() const { return seq > 0 ? seq : model.seq_len; }
};

// The modded-nanogpt Track 3 baseline (docs/distributed-design.md sections 8
// and 9): dense (`L`) attention, no value embeddings, and the AdamW/Muon rates
// and schedule of the reference recipe. It is a preset, not a new set of flags:
// the caller fills in the data paths, the batch, and the step count.
//
// The preset sets only fields that `OptimizerConfig` exposes. The reference
// AdamW betas (0.8, 0.95) and AdamW weight decay (0.001) are not representable:
// `src/optim.cc` hardcodes the per-group betas and decays, and
// `OptimizerConfig::weight_decay` is unused. Muon's decay maps through the
// weight-decay schedule.
TrainConfig TrackThreeBaseline();

// The scale applied to one micro-batch backward pass
// (docs/distributed-design.md section 6). The model's loss is a mean, so the
// factor is 1 / (grad_accum * world_size). The micro-batch sum and the
// all-reduce sum then give the mean over the global batch, which equals the
// one-rank update. A `world_size` of 1 leaves the accumulation scale unchanged.
float DistributedBackwardScale(int grad_accum, int world_size);

// Saves and loads a model's parameter set through the self-describing
// `Checkpoint` container (docs/model.md). This is the harness's checkpoint
// format; `Model::Save`/`Load` remain available with their own simpler format.
class Checkpointer {
 public:
  // Collects `model.params()` into a `Checkpoint` and writes it to `path`.
  static bool SaveModel(const Model& model, const std::string& path);

  // Also writes the optimizer's AdamW moments and Muon buffers beside the
  // parameters (docs/post-training.md section 7). The parameter-only file
  // stays loadable.
  static bool SaveModel(const Model& model, const Optimizer& optimizer,
                        const std::string& path);

  // Loads `path` and copies every matching record into the model parameters.
  // Returns false when the file is missing or malformed.
  static bool LoadModel(Model* model, const std::string& path);

  // Also restores the optimizer state when the file carries it. A
  // parameter-only file leaves the optimizer state untouched and still
  // returns true when the parameters loaded.
  static bool LoadModel(Model* model, Optimizer* optimizer,
                        const std::string& path);
};

class TrainLoop {
 public:
  explicit TrainLoop(TrainConfig config);
  ~TrainLoop();

  TrainLoop(const TrainLoop&) = delete;
  TrainLoop& operator=(const TrainLoop&) = delete;

  // Runs `num_iterations` steps, evaluating and checkpointing as configured.
  // Returns the last training loss in nats.
  float Run();

  Model* model() { return model_.get(); }
  Optimizer* optimizer() { return optimizer_.get(); }
  const Scheduler& scheduler() const { return *scheduler_; }
  float last_loss() const { return last_loss_; }
  int last_step() const { return last_step_; }

 private:
  void Save(int step);

  TrainConfig config_;
  std::unique_ptr<Model> model_;
  std::unique_ptr<Optimizer> optimizer_;
  std::unique_ptr<Scheduler> scheduler_;
  std::unique_ptr<Logger> logger_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<GradientSync> sync_;
  std::unique_ptr<DataLoader> train_loader_;
  std::unique_ptr<DataLoader> val_loader_;
  std::vector<int> tokens_;
  std::vector<int> targets_;
  double peak_flops_ = 0.0;
  float last_loss_ = 0.0f;
  int last_step_ = 0;
};

}  // namespace nanochat

#endif  // NANOCHAT_SRC_TRAIN_H_
