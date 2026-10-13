#ifndef NANOCHAT_SRC_MESH_RUNTIME_H_
#define NANOCHAT_SRC_MESH_RUNTIME_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "nanochat/config.h"
#include "nanochat/mesh.h"
#include "nanochat/model.h"
#include "nanochat/optim.h"
#include "nanochat/scheduler.h"
#include "src/mesh_collective.h"

// The single-client multi-device training runtime
// (docs/client-runtime-design.md). One process owns `world_size` model
// replicas, one per device thread. The controller reads one global batch,
// splits it into contiguous slices, and dispatches one slice per replica. The
// in-process collective reduces the gradients, so every replica applies the
// same update.
//
// The runtime is the C++ core of the client. The Python `MeshRuntime` and the
// C application binary interface call into it; `//src:mesh_runtime_test`
// covers the CPU path and `//src:runtime_gpu_test` covers two device threads on
// one card.

namespace nanochat {

// The per-replica configuration. One replica builds one `Model` and one
// `Optimizer` from these fields, with the shared `seed`, so the replicas start
// equal.
struct MeshRuntimeConfig {
  Config model;
  OptimizerConfig optimizer;
  SchedulerConfig scheduler;
  int batch = 1;       // the global batch, in sequences
  int grad_accum = 1;  // micro-batches per replica per step
  int seq = 0;         // 0 selects `model.seq_len`
  std::uint64_t seed = 42;
};

// The report of the last training step (docs/client-runtime-design.md section
// 20.4). `loss` holds one entry per replica, so a client can show the
// per-device split. `grad_norm` is the reduced gradient norm; every replica
// reports the same value. `step_seconds` is the wall-clock time of the last
// `TrainStep`.
struct MeshMetrics {
  std::vector<float> loss;
  float grad_norm = 0.0f;
  float step_seconds = 0.0f;
};

class MeshRuntime {
 public:
  MeshRuntime(const MeshSpec& mesh, MeshRuntimeConfig config);
  ~MeshRuntime();

  MeshRuntime(const MeshRuntime&) = delete;
  MeshRuntime& operator=(const MeshRuntime&) = delete;

  int world_size() const { return static_cast<int>(replicas_.size()); }
  int effective_seq() const { return seq_; }
  int global_batch() const { return config_.batch; }
  int slice_batch() const { return slice_rows_; }

  // Runs one step over `global_batch * effective_seq()` row-major ids. The
  // controller splits the batch into one contiguous slice per replica, and each
  // replica runs `grad_accum` micro-batches. Returns the global mean loss.
  float TrainStep(const int* tokens, const int* targets);

  // Test and client access. `model` returns null outside `[0, world_size())`.
  Model* model(int replica);
  const Model* model(int replica) const;
  int param_count(int replica) const;

  // The metrics of the last `TrainStep`, or zero values before the first step.
  const MeshMetrics& metrics() const { return metrics_; }

  // A 64-bit FNV-1a digest of every parameter of `replica`. The two replicas
  // return the same digest after a step, so a client can check the gradient
  // reduce without copying the whole model to the host. Throws on a bad
  // replica index.
  std::uint64_t ParamDigest(int replica) const;

  // Writes the canonical checkpoint from replica 0. Every replica is equal,
  // so one file is the whole state.
  void Save(const std::string& path);

  // Loads one checkpoint into every replica, so the replicas stay equal.
  void Load(const std::string& path);

 private:
  struct Replica;

  void ReplicaLoop(Replica& replica);
  float RunReplicaStep(Replica& replica, const int* tokens, const int* targets,
                       int step);

  MeshSpec mesh_;
  MeshRuntimeConfig config_;
  int seq_ = 0;
  int slice_rows_ = 0;
  int micro_rows_ = 0;
  int step_ = 0;
  std::unique_ptr<InProcessCollective> collective_;
  std::vector<std::unique_ptr<Replica>> replicas_;
  MeshMetrics metrics_;
};

}  // namespace nanochat

#endif  // NANOCHAT_SRC_MESH_RUNTIME_H_
