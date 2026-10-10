# Distributed training design

See [distributed-plan.md](distributed-plan.md) for the execution plan.

Data-parallel training on two devices. This document describes the gradient-sync
seam, the data sharding, the data cursor, the model configuration, and the
launch.

Status: the project implements the data-parallel architecture. `Config` carries
`value_embedding`. `src/distributed.h` is the seam. The CPU backend links the
host reference in `src/distributed.cc`. The CUDA backend links NCCL in
`backends/cuda/nccl_sync.cu`.

`TrainLoop` averages every gradient once per step. It overlaps the bucket
reductions with the pack copies on a side stream. Documents shard by a stride.
A sidecar beside the checkpoint stores the data cursor for an exact resume.

`train_main` accepts `--preset track3`. The CPU gates `//src:distributed_test`,
`//src:train_parallel_test`, and `//src:data_test` pass. The NCCL gate and the
two-rank parity gate run on Kaggle (see section 12).

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

Two implementations. The build selects one, and the selection follows the
backend (docs/distributed-plan.md phase 0). There is no host-reference fallback
on the CUDA path.

| Implementation | File | Use |
|---|---|---|
| Host reference | `src/distributed.cc` | The CPU backend and the CPU tests |
| NCCL | `backends/cuda/nccl_sync.cu` | The CUDA backend and the two T4 cards |

The host reference uses a TCP exchange. Rank 0 receives each rank's buffer, sums
the buffers, and broadcasts the result. This parameter server is enough for a
small rank count. The code includes POSIX sockets and the standard library
only. `DESIGN.md` section 2.1 permits host POSIX headers in `src`; the rule
forbids vendor headers.

The NCCL implementation calls `ncclAllReduce` with `ncclSum`. Rank 0 generates
the `ncclUniqueId` and broadcasts it to the peers over a one-shot TCP exchange;
every rank then calls `ncclCommInitRank`. The launch sets
`CUDA_VISIBLE_DEVICES` to the rank, so each process selects device 0. This file
is the only place that includes `nccl.h`.

The seam also has an asynchronous form, `AllReduceSumAsync` and `Wait`. The host
reference treats it as synchronous, so the CPU behavior does not change. NCCL
runs the reduction on a side stream. It synchronizes at `Wait`, so the pack copy
of the next bucket overlaps the reduction of the current one (section 6).

## 6. Training loop integration

`TrainLoop::Run` scales the local backward pass and then reduces the gradients
once per step.

```cpp
const float backward_scale = DistributedBackwardScale(accum, world_size);
// ... for each micro-batch:
model_->BackwardAccumulate(backward_scale);
// ... after the accumulation, pack each bucket and issue the async reduce:
sync_->AllReduceSumAsync(bucket, bucket_elems);
// ... after every bucket:
sync_->Wait();
optimizer_->Step(step);
```

The pack copy of one bucket runs on the main stream while the reduction of the
previous bucket runs on the backend's side stream. The host reference runs the
asynchronous call synchronously, so a CPU run does not change.

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
`--master`, and `--port`. The preset sets the Track 3 model, optimizer, and
scheduler fields. Pass `--preset track3` before the other flags, so a later
flag can override one field.

One process owns one rank. `tools/nanochat dist` starts the group. The
environment carries the rank values, in the style of `torchrun`. The `dist`
command takes one broker lock and one sandbox scope for the whole group.
[distributed-launch.md](distributed-launch.md) is the authority for the
launch.

This design removes the per-rank launch in `notebooks/parallel.py`. That code
took the broker lock once per rank, so a multi-card host could deadlock.
Kaggle hid the fault, because the sandbox and the broker are off there.

### 10.1 The best fit and the measured run

The model-fit skill (`.agents/skills/model-fit/`) computes the shape and the
memory. The best fit for the two T4 cards is `L16 C1024 H8 KV4 T4096 b1`: about
252M parameters, about 9.1 GiB of card memory, and an estimated 5.1k tokens/s
global.

The measured 500-step run on two T4 cards used the host reference sync and
`--grad-accum 8`. The global batch was 65536 tokens, not the reference 524288.

| Property | Value |
|---|---|
| Exit status | 0 on both ranks |
| First loss | 10.40 |
| Final loss | 3.59 (rank 0), 3.75 (rank 1) |
| Validation bpb | 1.88 at step 50, 1.13 at step 450 |
| Throughput | 1214 tokens/s per rank, 2427 global |
| MFU | 0.249 (fp32 peak, 8.1 TFLOP/s) |
| Phase split | forward 26.4%, backward 63.2%, sync 4.4%, optimizer 5.9%, data 0.1% |
| Checkpoints | byte-identical on the two ranks |

The backward dominates the step. The sync is below the 0.1 target. Phase D1
(recompute) and D2 (the attention tile) carry the next throughput gain; see
[distributed-plan.md](distributed-plan.md).

## 11. Determinism and resume

The two ranks produce the same parameter update, because the gradient sum is the
same. The optimizer state stays equal on both ranks.

`Checkpointer` saves the model and the optimizer state. A resumed run reads the
same file on both ranks.

The data cursor rides in a sidecar file beside the checkpoint. The sidecar holds
the epoch, the source position, and the encoded, not-yet-packed buffer. The
buffer is part of the cursor. The reference best-fit packing takes the largest
document that fits, so it reorders documents inside the buffer. A document
index alone cannot reproduce it. A snapshot takes the producer gate, so it
never records a read that has not reached the buffer.

A resumed run applies the stored source position on the stored epoch's source
creation. The loader skips earlier epochs without a full replay. The sidecar
keeps the main checkpoint byte-identical across ranks, so the checkpoint hash
check stays valid, and `//src:train_parallel_test` pins the exact resume.

## 12. Tests

| Tier | Test | Gate |
|---|---|---|
| T0 CPU | `//src:distributed_test` | A two-process sum returns the correct buffer |
| T0 CPU | `//src:train_parallel_test` | The mean scale, two equal ranks, and an exact resume |
| T0 CPU | `//src:data_test` | The sharded start and the data cursor round-trip |
| Simulator | `//tests:collective_sim_test` | The NCCL call sequence and the deadlock check |
| T1 GPU | `//backends/cuda:nccl_sync_test` | The NCCL sum matches the host reference |
| T2 GPU | A two-rank run versus one rank | The validation loss matches one rank |

The host reference uses two processes on the CPU backend. This test needs no
GPU. The simulator drives the NCCL mock on the CPU. The T1 and T2 gates run on
Kaggle, because the WSL2 host has no NCCL and one card (section 15).

The per-step parameter equality of a two-rank run and a one-rank run needs the
data split of section 7. Until then, the T2 gate is the validation loss on the
fixed validation set.

## 13. Kaggle notebook

The two-card notebook is `notebooks/nanochat-cpp-on-2x-t4.ipynb`. The plan
lives in [distributed-plan.md](distributed-plan.md).

The notebook runs these steps:

1. Locate NCCL and set `NANOCHAT_NCCL_PATH`.
2. Build the `t4` shared library and `//src:train_main`.
3. Stage one ClimbMix shard, the validation shard, and the tokenizer.
4. Run the best-fit model with one process on each card.
5. Compare the two checkpoint hashes.

The two cards do not have a peer link, so NCCL uses PCIe or shared memory. The
CUDA build requires NCCL (section 5); there is no host-reference fallback. If
NCCL is absent, the CUDA build fails at compile time.

## 14. Phasing

The architecture phases P0 to P4 are complete. They cover the seam, the Track 3
preset, the host reference, the CPU two-rank test, NCCL, the overlap seam, and
the data cursor.

The execution plan for the remaining throughput work is
[distributed-plan.md](distributed-plan.md). The NCCL gate and the two-rank
parity gate run on Kaggle (section 15).

The CPU path keeps the GPU off the correctness path, as `DESIGN.md` section 4
requires.

## 15. Risks

| Risk | Control |
|---|---|
| The Kaggle image lacks NCCL | Install NCCL, or point `NANOCHAT_NCCL_PATH` at it; a CUDA build needs it |
| The two T4 cards lack a peer link | NCCL uses PCIe or shared memory |
| The host reference is slow | It is CPU-only now; it is not on the GPU path |
| The stride sharding differs from the reference | `parity.md` D6 records the difference |
| A resumed run repeats or skips data | The data cursor restores the source position and the buffer |

## 16. Open questions

1. Which NCCL does the Kaggle image carry, and what is `NANOCHAT_NCCL_PATH`?
2. The WSL2 host has no NCCL and one card, so the T1 and T2 gates run on Kaggle.
3. Does the team want reference-exact initialization? This choice affects the
   loss curve.
4. Should the data loader split one global batch into contiguous rank slices, as
   the reference does? That change would make the two-rank and one-rank runs
   equivalent per step.
5. Should the backward expose a per-parameter gradient-ready hook? It would
   overlap the reduction with the backward itself. It needs a frozen-header
   change and a GPU measurement.
