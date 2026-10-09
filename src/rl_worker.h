#ifndef NANOCHAT_SRC_RL_WORKER_H_
#define NANOCHAT_SRC_RL_WORKER_H_

// The persistent reinforcement-learning worker (docs/rl-notebook.md section 4,
// docs/training-seam.md section 6.4, docs/post-training.md section 8).
//
// One long-lived process holds the `Model` and the `Optimizer` across steps, so
// the optimizer moments and the step counter stay alive and no step crosses the
// C ABI. It speaks a line protocol with the Python bridge (see below) and
// reuses the one shared step `nanochat::RlStep`, so it cannot drift from the
// file-driven `rl_step` binary.
//
// ---------------------------------------------------------------------------
// The pipe protocol
// ---------------------------------------------------------------------------
//
// Every message is one line of ASCII text terminated by `\n`. The first token
// is the message type; the remaining tokens are `key=value` fields separated by
// single spaces. A list value is comma-separated with no spaces. Unknown fields
// are ignored, so the protocol can grow additively. Floats use `%.9g`, which
// round-trips through a Python `float`.
//
// The bridge -> worker messages:
//
//   `rollout prompt_len=<int> num_prompts=<int> prompts=<ids>
//            [num_samples=<int>] [max_tokens=<int>] [temperature=<float>]
//            [top_k=<int>] [seed=<uint64>] [stop_id=<int>] [bos_id=<int>]
//            [stop_ids=<ids>]`
//     `prompts` is `num_prompts * prompt_len` row-major prompt ids (so every
//     prompt is the same length and the caller pads). The worker generates
//     `num_samples` rows per prompt in one `GenerateMultiPrompt` call, so a
//     rollout does not cross the ABI once per prompt
//     (docs/rl-notebook.md section 5 phase 5).
//
//   `advantage batch=<int> seq=<int> num_passes=<int>
//              examples_per_rank=<int> tokens=<ids> targets=<ids>
//              advantages=<floats>`
//     Each list holds `batch * seq` entries. `targets` carries `-1` at every
//     ignored position: the one mask channel. The bridge owns the chat
//     template and the loss mask (docs/post-training.md section 9), so it sends
//     the prompt positions as `-1` and the sampled positions as the token id.
//
//   `save path=<path>`   -- write an NCHKPT01 checkpoint of the live model,
//                           optimizer moments, and step counter.
//   `ping`               -- liveness probe.
//   `quit`               -- flush and close.
//
// The worker -> bridge messages:
//
//   `rollout ok=1 num_prompts=<> num_samples=<> rows=<int>
//            lengths=<ids> masks=<ids> tokens=<ids>`
//     `rows` is `num_prompts * num_samples`. `lengths[r]` is the effective
//     length of row `r`; `tokens` is the rows concatenated in order, and
//     `masks` is aligned with `tokens` and holds 1 for a sampled id and 0 for a
//     prompt (or prepended) id.
//
//   `advantage ok=1 step=<> loss=<> grad_norm=<> valid_targets=<>`
//   `pong ok=1`
//   `save ok=1`
//   `quit ok=1`
//   `<type> ok=0 error=<message>`  -- `message` has its whitespace replaced by
//                                     `_` so the line stays one frame.
//
// A failing request leaves the worker alive and does not advance the step
// counter.

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include "nanochat/config.h"
#include "nanochat/model.h"
#include "nanochat/optim.h"

namespace nanochat {

// The result of one `advantage` message.
struct RlWorkerResult {
  float loss = 0.0f;
  float grad_norm = 0.0f;
  int valid_targets = 0;
  int step = 0;
};

class RlWorker {
 public:
  RlWorker(const Config& model_config, const OptimizerConfig& optimizer_config,
           const SchedulerConfig& scheduler_config);
  ~RlWorker();
  RlWorker(const RlWorker&) = delete;
  RlWorker& operator=(const RlWorker&) = delete;

  Model* model() { return model_.get(); }
  Optimizer* optimizer() { return optimizer_.get(); }

  // The next 1-based optimizer step. Restored from a checkpoint or reset by
  // the fixture smoke run.
  int step() const { return step_; }
  void set_step(int step) { step_ = step; }

  // Runs one RL step over a full token/target/advantage batch through the
  // shared `RlStep`, advances the step counter, and reports the loss, the
  // gradient norm, and the valid-target count. `tokens`, `targets`, and
  // `advantages` each hold `batch * seq` entries.
  RlWorkerResult Step(const int* tokens, const int* targets,
                      const float* advantages, int batch, int seq,
                      int num_passes, int examples_per_rank);

  // Generates `params.num_samples` rows for each of the `num_prompts` prompts.
  // `prompts` holds `num_prompts * prompt_len` row-major ids. `out` is resized
  // to `num_prompts * params.num_samples` rows in prompt-major order.
  void Generate(const std::vector<int>& prompts, int num_prompts,
                int prompt_len, const GenerateParams& params,
                std::vector<GeneratedSequence>* out);

  // Writes the live model, the optimizer moments, and the step counter as an
  // NCHKPT01 checkpoint (docs/post-training.md section 7). Returns false with a
  // message in `*error` on a write failure.
  bool SaveCheckpoint(const std::string& path, std::string* error);

  // Serves the pipe protocol on `in`/`out` until end of input or a `quit`.
  // Returns 0 on a clean shutdown and a nonzero process code on an I/O error.
  int Serve(std::istream& in, std::ostream& out);

 private:
  // The model configuration the worker was built with; kept so the model can
  // be recreated and so a checkpoint can be described by its architecture.
  Config config_;
  std::unique_ptr<Model> model_;
  std::unique_ptr<Optimizer> optimizer_;
  int step_ = 0;
};

}  // namespace nanochat

#endif  // NANOCHAT_SRC_RL_WORKER_H_
