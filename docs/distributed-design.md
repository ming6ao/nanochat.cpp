# Distributed training design

Data-parallel training on two devices. This document describes the gradient-sync
seam, the data sharding, the model configuration, and the launch.

Status: P0, P1, and P2 are complete for architecture support. `Config` carries
`value_embedding`. `src/distributed.h` is the seam. `src/distributed.cc` is the
backend-free host reference. `TrainLoop` averages every gradient once per step.
Documents shard by a stride.

`train_main` accepts `--preset track3`. The CPU gates `//src:distributed_test`
and `//src:train_parallel_test` pass. P3 and P4 stay open. Scope: architecture
support only.

## 1. Goal

Run the modded-nanogpt Track 3 model on two T4 cards in a Kaggle Notebook. The
two cards train one model with data parallel. The project supports this in C++
with no Python framework.

This document covers the gradient synchronization seam and the launch. It does
not cover the FineWeb10B data stream.

## 2. Scope

In scope:

- Data parallel training on two devices.
- Document sharding by rank.
- Gradient all-reduce, averaged across ranks.
- A two-process launch on Kaggle.
- The Track 3 model configuration.

Out of scope:

- Exact reproduction of the published 3.28 loss. That work needs FineWeb10B and
  the GPT-2 tokenizer.
- FP8 and FlashAttention-3.
- Distributed evaluation.
- Tensor parallel and pipeline parallel.
- More than one node.

## 3. The reference and the mean

The PyTorch reference splits one global token batch into contiguous rank slices.
Each rank computes a loss sum over its slice. The ranks sum the gradients with
`all_reduce(SUM)`. Every rank then applies the same optimizer step. The global
batch stays constant, so the number of ranks changes only the work per rank.

nanochat.cpp computes a mean loss, not a sum. So the equivalent operation is a
mean over the ranks. The harness scales each local backward pass by
`1 / world_size`, and the all-reduce sums the result. The global gradient is
then the mean over the global batch.

## 4. The seam

The model exposes every gradient through `ParamView`. The `grad` pointer and the
`count` field are all the synchronization needs. So the seam is small and it
needs no change to a frozen header.

New header `src/distributed.h`:

```cpp
namespace nanochat {

struct DistributedConfig {
  int rank = 0;
  int world_size = 1;
  std::string master = "127.0.0.1";
  int port = 29500;
};

class GradientSync {
 public:
  virtual ~GradientSync() = default;
  virtual int rank() const = 0;
  virtual int world_size() const = 0;
  virtual void AllReduceSum(ComputeType* buffer, std::int64_t count) = 0;
};

std::unique_ptr<GradientSync> CreateGradientSync(const DistributedConfig& config);

}  // namespace nanochat
```

Rules:

- `src/distributed.h` includes only `nanochat/tensor.h` and the standard
  library.
- The host reference lives in `src/distributed.cc`. The device-specific NCCL
  factory lives in the backend, and the build selects it at link time in P3.
- The `src` layer never includes a vendor header.
- The seam method is `AllReduceSum`. The harness supplies the `1 / world_size`
  factor through the backward scale, so the seam stays a standard sum.
- A `world_size` of 1 returns a no-op object.

## 5. Implementations

Two implementations.

| Implementation | File | Use |
|---|---|---|
| Host reference | `src/distributed.cc` | CPU tests and a portable fallback |
| NCCL | `backends/cuda/nccl_sync.cu` | The two T4 cards |

The host reference uses a TCP exchange. Rank 0 receives each rank's buffer, sums
the buffers, and broadcasts the result. This parameter server is enough for a
small rank count. The code includes POSIX sockets and the standard library
only. `DESIGN.md` section 2.1 permits host POSIX headers in `src`; the rule
forbids vendor headers.

The NCCL implementation calls `ncclAllReduce` with `ncclSum`. It selects the
device from the rank. This file is the only place that includes `nccl.h`.

## 6. Training loop integration

`TrainLoop::Run` scales the local backward pass and then reduces the gradients
once per step.

```cpp
const float backward_scale = DistributedBackwardScale(accum, world_size);
// ... for each micro-batch:
model_->BackwardAccumulate(backward_scale);
// ... after the accumulation:
for (const ParamView& p : model_->params()) {
  sync_->AllReduceSum(p.grad, p.count);
}
optimizer_->Step(step);
```

The helper is one function:

```cpp
float DistributedBackwardScale(int grad_accum, int world_size) {
  const int accum = grad_accum > 0 ? grad_accum : 1;
  const int world = world_size > 0 ? world_size : 1;
  return 1.0f / static_cast<float>(accum * world);
}
```

Properties:

- The backward scale is `1 / (grad_accum * world_size)`.
- The model's loss is a mean, so the sum over micro-batches and the all-reduce
  sum give the mean over the global batch.
- The reduction runs before the gradient norm. The optimizer clips the global
  norm after the sum. So the clip sees the global mean gradient.
- A `world_size` of 1 leaves the accumulation scale unchanged.
- If `CreateGradientSync` returns null at a `world_size` above 1, the
  constructor stops the process. The run never continues without reduction.

## 7. Data sharding

`TrainConfig` gains `rank` and `world_size`. The document source factory uses
both.

nanochat.cpp shards documents by a stride: rank `r` takes document `i` when
`i % world_size == r`. This rule is deterministic and reproducible.

The stride is not the reference rule. The reference splits one global token
batch into contiguous slices. The stride re-partitions the same document pool.
So a two-rank run and a one-rank run do not produce the same tokens in one step.
The CPU gate pins the mean scale and the equality of the two ranks. The
validation-loss equality on the fixed validation set is the later gate.

The global batch stays 524288 tokens for Track 3. Each rank owns
`524288 / world_size` tokens. `TrainConfig.batch` is the per-rank batch in
sequences. `grad_accum` splits it into microbatches.

Example for two ranks and sequence 1024:

- Global batch: 512 sequences.
- Per-rank batch: 256 sequences.
- `grad_accum` of 4: four microbatches of 64 sequences.

## 8. Model configuration for Track 3

The `Config` maps to the Track 3 baseline as follows.

| Field | Value |
|---|---|
| `num_layers` | 12 |
| `num_heads` | 6 |
| `num_kv_heads` | 6 |
| `hidden_dim` | 768 |
| `seq_len` | 1024 |
| `vocab_size` | 32768 (the native tokenizer) |
| `padded_vocab_size` | 32768 |
| `window_pattern` | `"L"` |
| `value_embedding` | `false` |

The `"L"` pattern gives full context, so the attention is dense. The current
`window_pattern` field already supports this.

The `Config` field is additive:

```cpp
bool value_embedding = true;
```

The default keeps the current model. The Track 3 baseline sets the field to
false. `Config::has_value_embedding` reads the new field. The architect owns
this change.

The native tokenizer has 32768 entries, so the loss target is not 3.28. This
work supports the architecture, not the published number.

The C ABI mirror `nanochat_config` and the Python `Config` do not expose the
field yet. `--preset track3` and the C++ API do. A later change adds the field
to the C ABI and to the Python surface.

## 9. Optimizer and schedule mapping

Track 3 uses AdamW plus Muon. `OptimizerConfig` exposes the learning rates,
`adam_eps`, and the weight-decay schedule.

| Group | Rate |
|---|---|
| Embedding | 0.7 |
| Unembedding | 0.004 |
| Scalars | 0.015 |
| Matrices | 0.025 |

AdamW uses betas (0.8, 0.95), epsilon 1e-10, and a weight decay of 0.001. Muon
uses a weight decay of 0.05.

Two values are not representable. `src/optim.cc` hardcodes the AdamW betas and
the AdamW weight decay, and `OptimizerConfig::weight_decay` has no effect. So
the preset sets the learning rates, `adam_eps`, and the schedule, and records
the gap in a comment. A follow-up must express the betas and the AdamW decay, or
the architect must confirm that the difference is acceptable.

The schedule is stable first, then a linear decay over the last 70 percent. Set
`warmdown_ratio` to 0.7 and `final_lr_frac` to 0.0. The `Scheduler` supports
this shape.

Initialization is an open item. The reference zeroes the projection weights and
uses a different normal distribution. nanochat.cpp uses its own xorshift
generator. This difference changes the loss curve.

## 10. Launch

`train_main` gains five options: `--preset track3`, `--rank`, `--world-size`,
`--master`, and `--port`.

The preset sets the Track 3 model, optimizer, and scheduler fields. Pass
`--preset track3` before the other flags, so a later flag can override one
field.

A two-process run on Kaggle starts one process per card:

```bash
CUDA_VISIBLE_DEVICES=0 train_main --preset track3 --rank 0 --world-size 2 --master 127.0.0.1 &
CUDA_VISIBLE_DEVICES=1 train_main --preset track3 --rank 1 --world-size 2 --master 127.0.0.1 &
wait
```

The notebook starts the two processes and waits. The Kaggle sandbox is off, so
no cgroup step is necessary.

## 11. Determinism and resume

The two ranks produce the same parameter update, because the gradient sum is the
same. The optimizer state stays equal on both ranks.

`Checkpointer` saves the model and the optimizer state. A resumed run reads the
same file on both ranks.

The data position is an open item. Each rank shards documents by a stride. A
resumed run restarts the epoch, because the checkpoint does not store the
document index. P4 adds the document index to the checkpoint.

## 12. Tests

| Tier | Test | Gate |
|---|---|---|
| T0 CPU | `//src:distributed_test` | A two-process sum returns the correct buffer |
| T0 CPU | `//src:train_parallel_test` | The mean scale and two equal ranks |
| T1 GPU | `//backends/cuda:nccl_sync_test` | The NCCL sum matches the host reference |
| T2 GPU | A two-rank run versus one rank | The validation loss matches one rank |

The host reference uses two processes on the CPU backend. This test needs no
GPU. The GPU test runs on Kaggle, because the WSL2 host has one card.

The two T0 CPU tests are complete. `//src:distributed_test` runs two processes
over loopback TCP and checks the sum and the sharding.
`//src:train_parallel_test` checks the backward scale and the equality of two
ranks. The two GPU rows stay in P3.

The per-step parameter equality of a two-rank run and a one-rank run needs the
data split of section 7. Until then, the T2 gate is the validation loss on the
fixed validation set.

## 13. Kaggle notebook

Changes to `notebooks/nanochat-cpp-on-t4x2-gpus.ipynb`:

1. Build the `t4` config.
2. Stage the ClimbMix shards and the tokenizer, as today.
3. Start the two-process run.
4. Save the checkpoints under `/kaggle/working`.
5. Save the notebook version to keep the output.

The two cards do not have a peer link. NCCL falls back to PCIe or shared memory.
If NCCL fails, the run uses the host reference.

## 14. Phasing

| Phase | Work | Gate | Status |
|---|---|---|---|
| P0 | Freeze the seam and the `Config` field | The architect approves this document | Done |
| P1 | Track 3 configuration preset | `--preset track3` expresses the baseline | Done for the expressible fields; the betas and the AdamW decay stay open |
| P2 | Host reference and the CPU two-rank test | The backward scale and the rank equality | Done |
| P3 | NCCL and the Kaggle notebook | A two-card run completes a short run | Open |
| P4 | Long run and resume | A resumed run reaches the step target | Open |

P1 removes the GPU from the critical path, as `DESIGN.md` section 4 requires.

## 15. Risks

| Risk | Control |
|---|---|
| The Kaggle image lacks NCCL | The host reference is the fallback |
| The two T4 cards lack a peer link | NCCL uses PCIe or shared memory |
| The host reference is slow | It carries about 0.5 GB per step; measure it in P3 |
| The stride sharding differs from the reference | `parity.md` D6 records the difference |
| A resumed run repeats data | Store the document index in the checkpoint |

## 16. Open questions

1. Does the Kaggle image carry NCCL?
2. Does the WSL2 host allow a two-process test on one card? If not, P3 runs on
   Kaggle.
3. Does the team want reference-exact initialization? This choice affects the
   loss curve.
4. Should the data loader split one global batch into contiguous rank slices, as
   the reference does? That change would make the two-rank and one-rank runs
   equivalent per step.
