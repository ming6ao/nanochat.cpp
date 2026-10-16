#ifndef NANOCHAT_SRC_MESH_TRAIN_H_
#define NANOCHAT_SRC_MESH_TRAIN_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "nanochat/config.h"
#include "nanochat/dataloader.h"
#include "nanochat/mesh.h"
#include "nanochat/optim.h"
#include "nanochat/scheduler.h"
#include "src/sft_shard.h"

// The mesh training driver (docs/client-runtime-design.md section 10.1 and
// docs/training-seam.md section 6.6). The driver owns the step loop, the
// metric record, the summary, the checkpoint interval, and the resume. It runs
// a `MeshRuntime` one global batch at a time.
//
// The client calls one driver function per run. It does not call one function
// per step. The driver reads parquet through the native reader, or an injected
// `DocumentSourceFactory` in a test.
//
// The config is a plain value. Python owns the job file for the child process,
// so this layer carries no serializer and no parser.

namespace nanochat {

class MeshRuntime;
class Tokenizer;

// The schema version of the job, the metric record, and the summary.
inline constexpr int kMeshTrainSchemaVersion = 1;

// One metric record, written as one JSON line in `metrics.jsonl`
// (docs/client-runtime-design.md section 10.1). The field set follows
// `LogRecord` and adds one loss entry per device.
struct MeshStepRecord {
  int step = 0;
  float loss = 0.0f;
  float grad_norm = 0.0f;
  double tokens_per_sec = 0.0;
  double mfu = 0.0;
  double step_seconds = 0.0;
  double compute_seconds = 0.0;
  double seconds = 0.0;
  std::vector<float> device_loss;
};

// The run summary, written as `summary.json` (docs/client-runtime-design.md
// section 10.1).
struct MeshTrainSummary {
  std::string checkpoint;
  int steps = 0;
  double mean_loss = 0.0;
  double steady_mfu = 0.0;
  std::vector<std::uint64_t> replica_digests;
  bool interrupted = false;
};

// The driver inputs. The field shape mirrors `TrainConfig` and
// `MeshRuntimeConfig` (docs/training-seam.md section 6.6). `preset` selects
// the optimizer and the schedule; `track3` is the reference recipe.
struct MeshTrainConfig {
  MeshSpec mesh;
  Config model;
  std::string preset = "track3";
  // The explicit optimizer and schedule. `use_preset` false selects these and
  // ignores `preset`. The binding sets them from the C `nanochat_optim_config`.
  OptimizerConfig optimizer;
  SchedulerConfig scheduler;
  bool use_preset = true;
  // A weights-only warm start. The driver loads the parameter records and
  // keeps the fresh optimizer state (docs/checkpoint-format.md section 9.1).
  std::string warm_start_path;

  int batch = 1;       // the global batch, in sequences
  int grad_accum = 1;  // micro-batches per replica per step
  int seq = 0;         // 0 selects the shard or `model.seq_len`
  std::uint64_t seed = 42;
  std::uint64_t data_seed = 42;  // the loader seed, separate from the weights
  int num_iterations = 100;
  int log_every = 1;
  int save_every = 0;  // 0 disables the interval; the final save always runs
  bool resume = false;

  // The MFU inputs (docs/client-runtime-design.md section 20.5). A zero
  // `flops_per_token` selects `EstimateFlopsPerToken`; a zero `peak_flops`
  // selects `PeakFlopsForCompute(device_name)`.
  double flops_per_token = 0.0;
  double peak_flops = 0.0;
  std::string device_name;

  std::string out_dir;
  std::string checkpoint_path;  // empty selects `out_dir/checkpoint.ckpt`
  std::string metrics_path;     // empty selects `out_dir/metrics.jsonl`
  std::string summary_path;     // empty selects `out_dir/summary.json`

  // The data (docs/parquet-native.md). A nonempty `packed_rows_path` selects
  // the packed `NCSFT001` SFT shard and ignores the parquet fields
  // (docs/training-seam.md section 6.2). `train_source` is empty in production,
  // so the driver builds the parquet factory. A test injects a source.
  std::vector<std::string> train_parquet;
  std::string tokenizer_path;
  std::string packed_rows_path;
  std::string text_column = "text";
  int tokenizer_threads = 4;
  int document_buffer = 1000;
  DocumentSourceFactory train_source;

  int effective_seq() const { return seq > 0 ? seq : model.seq_len; }
  std::string checkpoint() const;
  std::string metrics() const;
  std::string summary() const;
};

// The driver. The constructor builds the runtime, the tokenizer, and the
// loader. `Run` runs the loop and writes the record, the summary, and the
// checkpoint.
class MeshTrainDriver {
 public:
  explicit MeshTrainDriver(MeshTrainConfig config);
  ~MeshTrainDriver();

  MeshTrainDriver(const MeshTrainDriver&) = delete;
  MeshTrainDriver& operator=(const MeshTrainDriver&) = delete;

  // Runs steps `resume + 1` to `num_iterations` and returns the last loss. On
  // an interrupt the driver saves a checkpoint before it returns
  // (docs/client-runtime-design.md section 10.1).
  float Run();

  const std::vector<MeshStepRecord>& records() const { return records_; }
  const MeshTrainSummary& summary() const { return summary_; }
  int last_step() const { return last_step_; }
  float last_loss() const { return last_loss_; }

 private:
  bool LoadBatch(std::vector<int>* tokens, std::vector<int>* targets);
  void Save(int step);
  void WriteSummary();

  MeshTrainConfig config_;
  Config model_;
  std::unique_ptr<MeshRuntime> runtime_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<DataLoader> loader_;
  std::vector<int> tokens_;
  std::vector<int> targets_;
  SftShard shard_;
  int row_count_ = 0;
  int row_index_ = 0;
  std::vector<MeshStepRecord> records_;
  MeshTrainSummary summary_;
  double flops_per_token_ = 0.0;
  double peak_flops_ = 0.0;
  int start_step_ = 1;
  int last_step_ = 0;
  float last_loss_ = 0.0f;
};

}  // namespace nanochat

#endif  // NANOCHAT_SRC_MESH_TRAIN_H_
