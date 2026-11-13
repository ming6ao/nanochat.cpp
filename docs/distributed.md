# Distributed and multi-device training

Two or more devices train one model with data parallelism. The model has one
seam for it, and the seam is small: every gradient is reachable through
`ParamView`, so a collective needs the `grad` pointer and the `count` field and
no change to a frozen header.

This page states the seam, both implementations, the training-loop integration,
the data sharding, the resume rule, and the two ways to run a group: one process
per rank, or one process with a device mesh.

```bash
# One process per rank.
tools/nanochat dist --nproc-per-node 2 -- nanochat_cli train --preset track3

# One process, two cards, one device thread per card.
tools/nanochat client --profile train --mesh 2 -- nanochat_cli train --layers 12
```

Sections 1 to 13 cover the model, the loop, and the data. Part 1 is the seam
between Python and C++, Part 2 is the native collective, and Part 3 is the
single-client runtime.

## 1. Goal

Run the modded-nanogpt Track 3 model on two T4 cards in a Kaggle notebook, and
the same model on eight H100 cards. The cards train one model with data
parallel, in C++ with no Python framework.

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
- FlashAttention-3.
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
- The host reference lives in `src/distributed.cc`. The device-specific
  factory lives in the backend, and the build selects it at link time in P3.
- The `src` layer never includes a vendor header.
- The seam method is `AllReduceSum`. The harness supplies the `1 / world_size`
  factor through the backward scale, so the seam stays a standard sum.
- A `world_size` of 1 returns a no-op object.

## 5. Implementations

Two implementations ship. The build selects one, and the selection follows the
backend. There is no host-reference fallback on the CUDA path.

| Implementation | File | Use |
|---|---|---|
| Host reference | `src/distributed.cc` | The CPU backend and the CPU tests |
| Native ring | `backends/cuda/collective/native_sync.cu` | The CUDA backend, and the only collective |

The native ring runs a ring reduce-scatter + all-gather over
a completion-based `Transport`. The selection is per ring edge
(section 3.2 of the native plan): NVLink when `cudaDeviceCanAccessPeer` passes,
RDMA when an adapter reaches the peer, and host staging otherwise. The 2x T4
pair has no peer access, so it exercises the host-staged path; the fast path is
the 8x H100 node.

The host reference uses a TCP exchange. Rank 0 receives each rank's buffer, sums
the buffers, and broadcasts the result. This parameter server is enough for a
small rank count. The code includes POSIX sockets and the standard library
only. The [dependency
invariants](build.md#dependency-invariants) permit
host POSIX headers in `src`; the rule forbids vendor headers.

The seam also has an asynchronous form, `AllReduceSumAsync` and `Wait`. The host
reference treats it as synchronous, so the CPU behavior does not change. The
native ring runs the reduction on a side stream. It synchronizes at `Wait`, so the pack copy
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

The C ABI mirror `nanochat_config` and the Python `Config` expose the field.
`--preset track3`, the C++ API, and `nanochat_cpp.Config(value_embedding=False)`
all select the Track 3 shape.

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

`nanochat_cli train` owns one rank, and the launcher owns the process group.
`tools/nanochat dist` starts the group, applies one sandbox scope, and submits
one queue job, so a multi-card host never serializes the ranks.

### 10.1 The launch command

```text
tools/nanochat dist [--nproc-per-node N] [--device LIST] [--profile P]
                    [--master ADDR] [--port PORT] [--log-dir DIR] [--json]
                    -- <command...>
```

The command does these steps:

1. Select the device set: `0..N-1`, or the first `N` entries of the worker's
   pool when `NANOCHAT_GPU_DEVICES` is set.
2. Submit one queue job for the whole group, and wait for it.
3. Apply one sandbox profile to the group. The `dist` profile covers the group;
   the `train` profile covers one rank and is too small.
4. Set `CUDA_VISIBLE_DEVICES` and the four rank variables for each rank.
5. Start rank 0 first, so the bind wins the port race, and redirect each rank to
   `<log-dir>/rank<i>.stdio.log`.
6. Wait for every rank, kill them all on a signal, and return the first nonzero
   status. `--json` prints one summary object.

The command skips the queue when `NANOCHAT_GPU_QUEUED` is set, because a queue
job already runs it, and when the sandbox backend is `none`, as on Kaggle.

`nanochat_cli train` gains five options: `--preset track3`, `--rank`,
`--world-size`, `--master`, and `--port`. The preset sets the Track 3 model,
optimizer, and scheduler fields, so pass it before the other flags.

### 10.2 The environment contract

`nanochat_cli train` reads five variables. The resolution order is the flag,
then the `NANOCHAT_*` variable, then the default.

| Variable | Meaning | Flag |
|---|---|---|
| `NANOCHAT_WORLD_SIZE` | the group size | `--world-size` |
| `NANOCHAT_RANK` | the global rank | `--rank` |
| `NANOCHAT_LOCAL_RANK` | the rank on this node | none |
| `NANOCHAT_MASTER_ADDR` | the rank 0 host | `--master` |
| `NANOCHAT_MASTER_PORT` | the rank 0 port | `--port` |

The names follow `torchrun`, so `nanochat_cli train` also runs under `torchrun`,
`mpirun`, or a shell loop. On one node `NANOCHAT_LOCAL_RANK` equals
`NANOCHAT_RANK`; the variable exists for a later multi-node change.
`src/cli.h` holds the two helpers `EnvInt` and `EnvString`.

A C++ launcher would move host tooling into L6 and would run before the sandbox,
so it could not apply the sandbox itself. The project does not add one
([sandbox.md](sandbox.md) § Entry point).

### 10.3 The Python wrapper

`python/nanochat_cpp/distributed.py` is an optional notebook wrapper. It never
starts a process: it calls `tools/nanochat dist --json` and parses the summary.

```python
@dataclasses.dataclass(frozen=True)
class DistributedPlan:
    argv: tuple[str, ...]
    nproc_per_node: int = 2
    device: str | None = None
    master: str = "127.0.0.1"
    port: int = 29500
    profile: str = "dist"
    log_dir: str | None = None

@dataclasses.dataclass(frozen=True)
class DistributedResult:
    ranks: tuple[RankResult, ...]

def launch(plan: DistributedPlan) -> DistributedResult: ...
def parse_log(path) -> list[dict]: ...
def loss_rows(result) -> list[dict]: ...
```

The summary carries the per-rank return code, the log path, the stdio path, and
the seconds. `Logger::Log` in `src/train.cc` writes the log line, and one
fixture test pins the format.

### 10.4 When to use the mesh instead

The two-card notebook uses the single-client multi-device runtime, not this
launch: one process, one `MeshRuntime`, and the in-process collective (Part 3).
The mesh removes the rank from the user program, so the notebook opens no port
and sets no rank variable.

The launch stays for the process-isolation case: some drivers forbid two CUDA
contexts in one process, and one bad device then stops one worker only.

### 10.5 The best fit and the measured run

The model-fit skill (`.agents/skills/model-fit/`) computes the shape and the
memory. The best fit for the two T4 cards is `L16 C1024 H8 KV4 T4096 b1`: about
252M parameters, about 9.1 GiB of card memory, and an estimated 5.1k tokens/s
global.

The measured 500-step run on two T4 cards uses the native ring collective
(`--config=cuda`), the windowed attention backward, and pinned host staging.
It runs `--grad-accum 8`. The global batch is 65536 tokens, not the reference
524288.

| Property | Value |
|---|---|
| Exit status | 0 on both ranks |
| First loss | 10.40 |
| Final loss | 3.59 (rank 0), 3.75 (rank 1) |
| Validation bpb | 1.88 at step 50, 1.13 at step 450 |
| Throughput | about 1470 tokens/s per rank, about 2940 global |
| MFU | 0.30 steady state (fp32 peak, 8.1 TFLOP/s) |
| Phase split | forward 30.5%, backward 59.6%, sync 3.0%, optimizer 6.7%, data 0.1% |
| Checkpoints | byte-identical on the two ranks |

The backward still dominates the step. Two changes raise the MFU above the
30% target. The windowed attention backward
(`backends/cuda/kernels/attention.cu`) runs the banded query-block schedule, so
the `SSSL` sliding-window layers no longer pay for a dense `seq x seq` backward;
that cut the backward by about a quarter. The host-staged ring uses pinned host
memory, which cut the sync from about 2.5 s to about 0.7 s per step. The 70 W
T4 power cap holds the sustained SM clock near 900 MHz, so the achievable MFU
stays below the boost-clock peak.

The remaining headroom is the overlap of the reduction with the backward and a
fused windowed attention forward on the backward path.

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

A resumed run applies the stored source position on the source the loader's
constructor creates, and the constructor primes the epoch and the buffer from
the sidecar. The packing then continues where it stopped, so the documents that
were packed but not consumed are not skipped. The epochs before the stored one
are never opened, so a resume costs no replay. A `Reset` after that opens the
next epoch, and it starts at position 0. Both drivers resume this way: the
in-process `MeshTrainDriver` and the multi-process `TrainLoop`. The sidecar
keeps the main checkpoint byte-identical across ranks, so the checkpoint hash
check stays valid, and `//src:train_parallel_test` pins the exact resume.

## 12. Tests

| Tier | Test | Gate |
|---|---|---|
| T0 CPU | `//src:distributed_test` | A two-process sum returns the correct buffer |
| T0 CPU | `//src:train_parallel_test` | The mean scale, two equal ranks, and an exact resume |
| T0 CPU | `//src:data_test` | The sharded start and the data cursor round-trip |
| T0 CPU | `//backends/cuda/collective:schedule_test` | The ring at world sizes 1, 2, 4, and 8 |
| T1 GPU | `//backends/cuda/collective:nvlink_test` | The native ring sum matches the host reference |
| T2 GPU | A two-rank run versus one rank | The validation loss matches one rank |

The host reference uses two processes on the CPU backend. This test needs no
GPU. The ring schedule runs on the CPU with the memory transport. The T1 and T2
gates run on Kaggle, because the WSL2 host has one card (section 13).

The per-step parameter equality of a two-rank run and a one-rank run needs the
data split of section 7. Until then, the T2 gate is the validation loss on the
fixed validation set.

## 13. Kaggle notebook

The two-card notebook is `notebooks/nanochat-cpp-on-2x-t4.ipynb`. It uses the
single-client multi-device runtime, so one process owns both cards
(Part 3). The multi-process plan is section 10.

The notebook runs these steps:

1. Build the Turing shared library, `--config=cuda --@rules_cuda//cuda:archs=sm_75`.
   The library already holds the mesh runtime.
2. Stage one ClimbMix shard, the validation shard, and the tokenizer.
3. Run the best-fit model with one `MeshRuntime` over the two cards.
4. Compare the two replica parameter digests, and keep the canonical
   checkpoint.
5. Load the checkpoint in process. Run the base evaluation, the supervised
   fine-tuning, and the chat evaluation.

The runtime stages the gradients through the kernel seam, so the CUDA build
needs no vendor collective. The T4 cards do not have a peer link, so the
peer-to-peer fast path waits (section 10.1).

## Part 1. The training seam between Python and C++

_Absorbed from the former `training-seam.md`. Its section numbers are
unchanged, so a reference to a section of that document resolves to the same
number here._

This part states the boundary between the Python surface and the C++ training
driver. Section 6.6 extends the boundary to the mesh training driver.

`post-training.md` stays the design of record for the post-training
features. This document is the authority for the Python and C++ boundary. It
revises section 4.1 of [post-training.md](post-training.md) on the owner of the
packer.

The companion documents are:

- [python](python.md) for the Python surface.
- [post-training](post-training.md) for supervised fine-tuning and
  reinforcement learning.
- [eval](eval.md) for the evaluation seam.
- [model](model.md) for the model interface.

### 1. Purpose

[build.md](build.md#layers-and-the-one-seam) puts the training driver at level
L5. Level L5
holds `TrainLoop`, `Scheduler`, `Checkpointer`, `Logger`, and the MFU
accounting. The driver is C++ by design.

The Python surface must not rebuild that driver. Today it does.

The compute layer in `python/nanochat_cpp/api.py` exposes `forward_loss`,
`backward`, and `optimizer.step`. A caller therefore writes a training loop in
Python. The `Trainer` class covers pretraining only. A user who wants to
fine-tune writes the loop by hand, as the Kaggle notebook does. That loop
duplicates `TrainLoop`.

The mesh client repeated the same defect. `notebooks/mesh_train.py` owned the
step loop, the metric record, the MFU, the summary, and the job serializer. The
C++ mesh runtime provided one step and no loop. Section 5.8 records the defect,
and section 6.6 gives the target.

This document defines the correct split. It then proposes the smallest interface
that removes the duplication.

### 2. The seam rule

The repository already states the rule for evaluation. [eval.md](eval.md)
section 2 says "metric in Python, compute in C++." Python owns the datasets, the
prompt rendering, the answer extraction, the code execution, and the metric
reduction. C++ owns the model compute behind a few primitives.

This document proposes the same shape for training. The proposed rule is:

- C++ owns mechanism. Mechanism is the per-step arithmetic and the state that
  the arithmetic needs.
- Python owns policy and data. Policy is the dataset choice, the chat template,
  the reward, and the task metric.

Mechanism and policy meet at one interface. The interface carries data. It does
not carry the per-step control flow, with one exception for reinforcement
learning (section 6.4).

### 3. What C++ owns

The step arithmetic already lives in C++ and is correct.

| Item | Location |
|---|---|
| Forward loss with a valid-token mean | `Model::ForwardLoss` (`include/nanochat/model.h`) |
| Hand-written backward | `Model::Backward` |
| Gradient accumulation | `Model::BackwardAccumulate(scale)` |
| Weighted policy-gradient backward | `Model::BackwardWeighted(row_weights, scale)` |
| Forward plus backward plus one optimizer step | `Model::TrainStep` |
| Valid-token normalization | `CountValidTargets` (`src/model.cc`) |
| Batched, tokenizer-agnostic sampling | `GenerateBatch` |
| AdamW and Muon | `src/optim.cc` |
| Learning-rate schedule | `Scheduler` (`include/nanochat/scheduler.h`) |
| Checkpoint container and optimizer state | `Checkpointer`, `Checkpoint` |
| Step loop, logging, evaluation, checkpoint interval | `TrainLoop` (`src/train.cc`) |

The gaps are in the driver, not in the arithmetic. `TrainLoop` reads parquet
documents through `DataLoader` and uses the unweighted path only. The mesh
driver gained a packed `NCSFT001` shard path and a weights-only warm start
(section 6.2). The `TrainLoop` packed-row source and the weighted step remain
open.

### 4. What Python owns

Python owns four items. None of them needs tensor math.

1. Data acquisition and mixture. The reference mixes SmolTalk, MMLU, and GSM8K.
2. The chat template and the loss mask. Python turns a conversation into token
   ids and a target list with `-1` at an ignored position.
3. The reward and the answer extraction. GSM8K and HumanEval need text logic.
   HumanEval also runs generated code in a guarded executor.
4. The task metrics. [eval.md](eval.md) already grants this to Python.

The planning layer in `python/nanochat_cpp/plan.py` also stays in Python. It
needs two numbers from C and no tensors.

The split of item 2 is precise. Python owns the renderer and the mask. C++ owns
the row packing, the padding, and the batching.

### 5. The seam as it stands

The C ABI carries one call per stage, so a Python driver never reimplements the
arithmetic: `nanochat_train_step`, `nanochat_backward_accumulate`, and the
public `render_conversation` in `python/nanochat_cpp/chat.py`. `TrainLoop`
persists the step counter, so a driver owns the schedule and `Optimizer::Step`
stays in C++.

Two gaps remain. `Trainer` still owns gradient accumulation and the epoch reset
in Python, and only a per-step loss fixture gates it against `src/train.cc`. The
supervised and reinforcement-learning stages have no `SftTrainer` and no
`RlTrainer`; [post-training.md](post-training.md) section 5.5 lists what those need.

#### 5.7 The reinforcement-learning divisor

`Model::BackwardWeighted(row_weights, scale)` takes raw per-row weights, and
[post-training.md](post-training.md) gives the divisor as the product of the
valid count, the pass count, and the example count per rank. Python would
compute that divisor today, so the normalization would not be the single source
of truth. The divisor moves into C++ with the reinforcement-learning step.
### 6. Target design

#### 6.1 Extend the C interface

Add the calls below to `include/nanochat/capi.h`. Each call maps to one existing
C++ method or one new driver method.

| Call | Maps to |
|---|---|
| `nanochat_train_step` | `Model::TrainStep`; returns the loss |
| `nanochat_backward_accumulate` | `Model::BackwardAccumulate` |
| `nanochat_rl_step` | a new file-driven reinforcement-learning step |

No new batch type is necessary. A target of `-1` already encodes the loss mask
for supervised fine-tuning (`include/nanochat/model.h:56`). The weighted
backward already carries the per-row weight for reinforcement learning
(`include/nanochat/model.h:76`). A separate mask field would create two mask
channels with no precedence rule.

#### 6.2 The packed-row data path (partly done)

`DocumentSource` yields raw text (`include/nanochat/dataloader.h:22`). The
`DataLoader` tokenizes and packs that text. A supervised fine-tuning source
cannot use `DocumentSource`, because Python already rendered the text into token
ids and targets.

The mesh driver now reads one packed `NCSFT001` shard. Python writes the shard
with `nanochat_cpp.sft.pack`. The shard is a typed-record file
(`src/record_file.h`) with two `int32` records, `tokens` and `targets`, each of
shape `[rows, seq]`. A target of `-1` marks an ignored position. The file is
self-describing and versioned, so `src/sft_shard.h` holds only the typed view,
and the `nanochat_cli sft` stage and the mesh driver share one loader. The
driver loads the shard and cycles the rows, so it needs no reader thread.

`TrainLoop` still reads parquet only. A packed-row source beside
`DocumentSource` in `include/nanochat/dataloader.h` remains open, so a
parquet-free `TrainLoop` can run. The header is harness-owned, so that change
needs no frozen-header edit.

#### 6.3 Explicit rates and the warm start (partly done)

The mesh driver takes an explicit optimizer and schedule, so a fine-tuning run
passes its own rates and does not use the `track3` preset. It also takes a
weights-only warm start: the driver loads the parameter records of a base
checkpoint and keeps the fresh optimizer state
(`docs/checkpoint-format.md` section 9.1).

`TrainLoop` (`src/train.h`) still lacks the packed-row source and these items
from [post-training.md](post-training.md) section 12:

1. A progress-based learning-rate schedule.
2. A flat or ramped Muon momentum option.
3. Hyperparameter inheritance from a base checkpoint.
4. An optimizer-state warm start from a base checkpoint.

The driver owns the step counter. The caller must not pass a step number for
the schedule. `Checkpointer` must also restore the counter, or the resume will
restart the schedule.

#### 6.4 The reinforcement-learning path is a message loop

Reinforcement learning differs from supervised fine-tuning. The reward lives in
Python, and Python must compute it for each rollout. Python is therefore in the
control loop for every step. This is the one exception to the data-only rule.

The design follows [post-training.md](post-training.md) section 8:

- A file-driven `rl_step` binary consumes rollouts and advantages from a
  fixture. It is the correctness gate for the objective, and it uses
  `nanochat_rl_step`.
- A persistent worker holds the model and the optimizer. It exchanges rollout
  and advantage messages with the Python bridge over pipes.

The divisor from section 5.7 lives in the C++ step, not in Python.

#### 6.5 Keep the Python surface thin

Python keeps four jobs:

1. Render a batch of conversations to tokens and targets.
2. Download and mix the datasets.
3. Compute the reward and extract the answer.
4. Report the task metrics.

For supervised fine-tuning, Python then writes a packed shard and launches the
C++ driver. Python never steps the optimizer.

Proposed Python shape:

```python
# Python renders the conversations and writes a packed shard.
shard = nc.sft.pack("chatsft.nchsft", conversations, tokenizer, seq_len=512)
# The facade launches the C++ driver through tools/nanochat, so the
# sandbox and the GPU queue apply.
nc.sft.run(shard, "chatsft.nchkpt01", num_iterations=1000)
```

`nc.sft.run` calls the toolchain, which runs
`tools/nanochat train -- nanochat_cli sft
...`. It does not run an in-process loop.

A workstation run therefore obeys the entry-point rule in `AGENTS.md` section
8. A Kaggle notebook uses the `none` sandbox backend, as [python](python.md)
section 3.1 describes.

`render_conversation` becomes a public function in `nanochat_cpp.chat`. The
underscore helpers stay internal.

The reference mixture and the packer live in `nanochat_cpp.sft_data`. The
mixture is SmolTalk, MMLU, and GSM8K. `sft_data.pack_rows` implements the
BOS-aligned best-fit packer with BOS padding and a `-1` mask. It mirrors
`sft_data_generator_bos_bestfit` in `scripts/chat_sft.py`.

#### 6.6 The mesh path uses one C++ driver

The mesh path follows the same split. The C++ `src/mesh_train.cc` driver owns
the loop, the metric record, the summary, the checkpoint interval, and the
resume. It owns the parquet loader too, and it accepts an injected document
source, so a test supplies documents without parquet.

A coarse C interface in `capi.h` carries the driver. The client fills one
caller-owned config struct and calls one run function. The run call takes the
struct, not a file. Python owns the job file for the child process, so the C
interface carries no serializer and no parser.

`notebooks/mesh_train.py` keeps four jobs. It builds the driver config from the
notebook arguments, writes the job file, launches the child through
`tools/nanochat client`, and polls `metrics.jsonl` and `worker.log`. It does not
step the optimizer, compute the MFU, or write the summary.

The notebook builds the Python driver config directly, so no second config
class exists. The phase removes `iterate()`, `mfu()`, `format_row()`, `job()`,
and `from_job()`. `read_metrics()`, `read_summary()`, and `MeshLiveRun` stay,
because they read and display a file that C++ wrote. A file reader is not
mechanism.

The C interface carries data only. No callback from C++ into Python exists, as
section 7 states.

### 7. Non-goals

- A distributed supervised fine-tuning run. The C++ workflow has no distributed
  mode ([host-portability.md](host-portability.md)).
- A preference-optimization method such as DPO.
- A native chat template in the C++ model layer. The model stays
  tokenizer-agnostic ([post-training.md](post-training.md) section 9).
- A callback from C++ into Python. The C interface carries data only. The
  reinforcement-learning worker uses a message protocol instead.
- Autograd. The graph stays fixed ([model.md](model.md)
  § No autograd engine).

### 9. Invariants

- One step implementation. The C++ driver is the only place that advances the
  optimizer for a training run. The mesh driver is the mesh form of the same
  rule.
- One metric record. The C++ driver writes `metrics.jsonl` and `summary.json`.
  Python reads the files and displays the rows. Python does not compute a
  record.
- Valid-token normalization is the single source of truth. The
  reinforcement-learning divisor moves into C++.
- The model layer stays tokenizer-agnostic. The driver layer may use the
  tokenizer, and it already does.
- Data crosses the interface. Control flow does not, except for the
  reinforcement-learning reward, which Python computes for each rollout.
- Python calls one driver function per supervised fine-tuning run. It does not
  call one function per step.
- One mask channel. A target of `-1` marks an ignored position. No second mask
  field exists in the interface.
- The model does not replace its weights between steps. A persistent worker
  holds the model and the optimizer.

## Part 2. Native collectives over NVLink and RDMA

_Absorbed from the former `distributed-native-plan.md`. Its section numbers are
unchanged, so a reference to a section of that document resolves to the same
number here._

This document is the execution plan for a native all-reduce backend. The backend
is the one collective on the CUDA path. this page
holds the current seam design.

The ring schedule, the host-staged transport, the transfer-window pool, and
`NativeGradientSync` are in `backends/cuda/collective/`. The 2x T4 pair has no
peer access, so the ring stages through pinned host memory; the NVLink and RDMA
transports await the H100 node. Phases 3 to 5 remain.

Phase 6 removed the transition machinery: `--define=collective=native` is the
only value, `backends/cuda/nccl_sync.cu`, `third_party/nccl.bzl`, and the
`@nccl` repository are gone, and the NCCL mock and `//tests:collective_sim_test`
went with them.

### 1. Goal and scope

**Goal.** Remove NCCL as a hard dependency of the CUDA build. Provide a native
all-reduce behind the existing `GradientSync` seam. The training loop calls
`AllReduceSumAsync` and later `Wait` on that asynchronous seam. Use NVLink
inside one node. Use remote direct memory access (RDMA) as the second transport
to reach 8 ranks on one H100 node.

**In scope.**

- A backend-private collective under `backends/cuda/collective/`.
- An asynchronous, completion-based transport interface.
- Two transports: NVLink peer copy and RDMA.
- A ring schedule with full-duplex send and receive.
- A single-precision reduction only.
- Support for world sizes 1, 2, 4, and 8.

**Out of scope.**

- Tensor parallel and pipeline parallel.
- Distributed evaluation.
- Elastic rendezvous.
- A replacement for the reserved collectives: `Broadcast`, `AllGather`,
  `ReduceScatter`, `Send`, and `Recv`.

**Assumptions.**

- The 8 H100 cards sit in one node with NVLink 4.0 and NVSwitch. This node is the
  fast path.
- RDMA is the second transport. It serves a peer that NVLink cannot reach, or a
  future second node. One schedule runs on both transports.
- The node has at least one RDMA adapter with GPUDirect RDMA. Section 11 records
  how to confirm this.
- The reduction operand is single precision. The collective does not use fp16.

### 2. Why the transport is completion-based

A CUDA stream does not describe an RDMA transfer. RDMA posts a work request and
polls a completion queue. So the transport must not take a `cudaStream_t` in its
main interface. The interface exposes a completion token instead. The NVLink
transport implements the token with stream events. The RDMA transport implements
the token with a work completion.

This is the most important choice in the plan. A stream-shaped interface would
force the RDMA path into a poor design.

### 3. The private interface

New directory `backends/cuda/collective/`. The interface header is backend
private. It is not part of the frozen seam, so it may include CUDA headers.

The transport is pairwise. One `Transport` connects this rank to one peer. A
group with mixed links holds one instance for each ring neighbor. The next
subsection defines the selection. `Rendezvous` is the control plane. A
`TimePoint` is `std::chrono::steady_clock::time_point`.

```cpp
namespace nanochat {
namespace cuda_backend {

// A completion handle. The transport that issued the token defines the
// encoding. A token names one posted operation: one direction, one slot, and
// one stream or queue pair. The value 0 is invalid. The token is single-use
// and bound to the issuing transport.
struct Token {
  std::uint64_t value = 0;
};

enum class WaitResult { kCompleted, kTimeout, kError };

enum class TransportKind { kNone, kNvlink, kRdma, kHost };

// Moves chunks between this rank and one peer. One implementation per kind.
class Transport {
 public:
  virtual ~Transport() = default;
  virtual int self_rank() const = 0;
  virtual int peer_rank() const = 0;
  // Binds the control plane and opens the data plane to the one peer. Returns
  // false on failure.
  virtual bool Connect(const DistributedConfig& config,
                       Rendezvous* rendezvous) = 0;
  // Publishes the local transfer window to the peer and imports the peer's
  // window. `window` is owned by the adapter. Call once, and again after a
  // growth (section 5.4).
  virtual bool RegisterWindow(void* window, std::size_t bytes) = 0;
  // The number of receive slots for this direction. A ring needs two.
  virtual int SlotCount() const = 0;
  // Posts a send of `bytes` from `local` into the peer's receive slot `slot`.
  // The call returns at once.
  virtual Token Send(int slot, const void* local, std::size_t bytes) = 0;
  // Posts a receive of `bytes` into `local`, the local receive slot `slot`.
  virtual Token Recv(int slot, void* local, std::size_t bytes) = 0;
  // Returns true when the transfer completed. Never blocks.
  virtual bool Done(Token token) = 0;
  // Waits for the transfer until `deadline`. It returns kCompleted, kTimeout,
  // or kError. It never hangs past the deadline.
  virtual WaitResult Wait(Token token, TimePoint deadline) = 0;
  // Frees the receive slot. The transfer must be complete. The call
  // invalidates the token.
  virtual void Release(Token token) = 0;
};

}  // namespace cuda_backend
}  // namespace nanochat
```

The schedule uses only this interface. A memory transport backs the central
processing unit test. The memory transport can reuse the POSIX shared-memory
rendezvous from `tools/cuda_sim/collective/`.

#### 3.1 Token lifetime and slot ownership

A slot is a receive-side resource. `Recv(slot, ...)` posts into the local
receive slot. `Send(slot, ...)` targets the peer's receive slot with the same
index. One index pairs a send with its matching receive.

Each direction owns its own slots. The NVLink transport keeps two receive slots
for each stream. The RDMA transport keeps two receive slots for each queue pair.
The send side has no slot. The caller owns the send buffer until completion.

A token has three states.

| State | Allowed operations |
|---|---|
| Issued | `Done`, `Wait` |
| Completed | `Done`, `Wait` again, `Release` |
| Released | none; the token is invalid |

Rules:

- `Done` and `Wait` do not consume the token. The schedule may call either
  more than once.
- `Release` requires completion. It returns the receive slot to the free list.
- A released token is invalid. A later use is a programming error. The debug
  build asserts; the release build returns a failure.
- A token is bound to the transport that issued it. Do not pass a token to
  another transport.
- The schedule must release every issued token exactly once. A leaked receive
  slot stalls the ring.

#### 3.2 Pairwise transports and the selection rule

The ring needs two peers. The next rank is the send direction. The previous
rank is the receive direction. The group holds one transport for each neighbor.

```cpp
// Holds the two ring neighbors. It builds the neighbor table at Connect.
class RingTransports {
 public:
  bool Connect(const DistributedConfig& config, Rendezvous* rendezvous);
  Transport* Next();  // the send direction
  Transport* Prev();  // the receive direction
};
```

`RingTransports::Connect` selects the kind for each neighbor, in this order.

1. `cudaDeviceCanAccessPeer` is true for the pair: NVLink.
2. An RDMA adapter reaches the peer: RDMA.
3. Otherwise: host staging.

The selection is per neighbor. So one group can use NVLink on one edge and RDMA
on the other. A mixed group is the normal case at a node boundary. Section 11.2
records the rule.

### 4. The control plane

The socket rendezvous becomes a small, vendor-free helper in
`src/rendezvous.{h,cc}`. It exposes one class.

```cpp
// The control plane. One instance per process. Every call is a collective: all
// ranks call it with the same length and in the same order.
class Rendezvous {
 public:
  static std::unique_ptr<Rendezvous> Create(const DistributedConfig& config);
  // Exchanges `bytes` of opaque data across every rank. Rank 0 gathers and
  // broadcasts. Returns false on failure.
  bool ExchangeBytes(void* data, std::size_t bytes);
};
```

The helper carries both payloads.

- The NVLink payload: the CUDA IPC memory handles and the device identifiers.
- The RDMA payload: the queue pair number, the global identifier or local
  identifier, the remote key, and the buffer addresses.

The helper includes only POSIX sockets. The `src` layer stays vendor-free. The
backend links the helper. The host reference in `src/distributed.cc` uses the
same helper.

### 5. The schedule

Use a ring reduce-scatter, then an all-gather. It is bandwidth-optimal. Each
rank moves about `2 * (N - 1) / N` of the buffer. For 8 ranks that is 1.75 times
the buffer.

The schedule is pure host logic. It sends on the `Next` transport and receives
on the `Prev` transport (section 3.2). It calls `Send`, `Recv`, `Done`, `Wait`,
`Release`, and one device add. A memory transport exercises it exactly.

#### 5.1 Double buffering

Split the bucket into `N` chunks. Each direction keeps two receive slots, so
`SlotCount()` is 2. Slot reuse gives the pipeline. A send has no slot. Its
buffer stays alive until the token completes.

For each ring step `s`:

1. Receive the incoming chunk into local slot `s % 2`. The receive waits until
   the add of step `s - 2` released that slot.
2. Add the received chunk into the local chunk.
3. Send the local chunk to the next rank, into the peer's receive slot
   `s % 2`. The send waits until the add for that chunk completed.

#### 5.2 Two streams

The NVLink transport uses two streams.

- `recv_stream`: every incoming copy and every add.
- `send_stream`: every outgoing copy.

A receive slot belongs to one direction and one stream, as section 3.1 states.

The dependency rules:

- Before `Recv(s)` into slot `s % 2`, `recv_stream` waits on the event recorded
  after the add of step `s - 2`.
- The add of step `s` runs on `recv_stream` after `Recv(s)`.
- After the add of step `s`, record an event on `recv_stream`.
- Before `Send(chunk c)`, `send_stream` waits on the event for chunk `c`.

The two streams let the outgoing copy of one chunk overlap the incoming copy of
the next. One stream would serialize them and halve the link use.

The RDMA transport keeps the same schedule. Each direction uses its own queue
pair, so the work requests overlap without stream events. `Done` and `Wait` poll
the completion queue.

#### 5.3 Small messages

The ring latency is `2 * (N - 1)` steps. For a small bucket, use a direct
exchange or a recursive-doubling schedule. Add this only after the measurement
shows a need. Keep the ring as the default.

#### 5.4 The registered transfer window

The NVLink transport moves bytes between two processes. A device pointer from
one process is not valid in another process. The transport must export the
memory that a peer writes into and import the peer's memory. This rule applies
to NVLink and to RDMA (`ibv_reg_mr`).

The seam passes an arbitrary bucket pointer to `AllReduceSumAsync` on every
step. The training loop allocates the buckets each step and frees them after.
Per-step registration is too expensive. A per-step `cudaIpcOpenMemHandle` also
leaks a mapping.

So `NativeGradientSync` owns a pool of registered transfer windows.

- Allocate the pool with `kernels::Alloc`. The pool size is the
  `DistributedConfig.max_inflight_reductions` field, default 8.
- Export each window once with `cudaIpcGetMemHandle`.
- Exchange the handle and the device identifier with every rank through
  `Rendezvous::ExchangeBytes`. The transport keeps the entry for its neighbor.
- Import the peer handles once with `cudaIpcOpenMemHandle`, and keep the
  mappings.
- Register the same windows with `ibv_reg_mr` on the RDMA path.
- Pass a pointer inside a local window to `Send` and `Recv`. The transport maps
  it to the same offset in the peer window.

The adapter calls `Transport::RegisterWindow` with each window at connect time
and after a growth.

The pool size matches the harness bucket count. `train.cc` uses four to eight
buckets, so eight windows let every bucket be in flight. A reduction in excess
of the pool drains one completion first. That case is rare.

The window holds one bucket. The adapter copies the caller's bucket into a free
window, runs the ring, and copies the sum back. Both copies are device-to-device
on the local card. They cost local bandwidth only.

A window grows when a reduction is larger than the window.

- The growth condition is `count` alone. Every rank sees the same `count` in the
  same order, because the model and the bucket plan are equal. So every rank
  grows on the same call and the handle exchange is symmetric.
- Grow every window in the pool to the next power of two of the new size. So
  the exchange is rare.
- The host-staging transport has no window. It stages through host memory.

#### 5.5 The GradientSync adapter

`NativeGradientSync` implements `GradientSync` and drives the ring schedule.
The adapter is the bridge between the contiguous seam and the chunked token
schedule.

- `AllReduceSumAsync(buffer, count)`:
  1. Record an event on the main stream. The event fires after the caller
     produced `buffer`.
  2. Take a free window from the pool. If the pool is empty, drain one
     completion first.
  3. Make the adapter stream wait on the event.
  4. Copy `buffer` into the window.
  5. Enqueue the ring reduce-scatter and all-gather for the window.
  6. Append the request to a FIFO queue and return at once.
- `Wait()`:
  1. Drain the FIFO in order. For each request, poll `Done`, then call `Wait`
     with the watchdog deadline (section 5.6).
  2. After a ring completes, copy its window back into the caller buffer on the
     adapter stream. Return the window to the pool.
  3. Synchronize the adapter streams.

The ring runs on the adapter's streams, so the main stream is free. The pack
copy of the next bucket overlaps the reduction of the current bucket. This is
the overlap the asynchronous seam promises (docs/distributed.md
section 6).

The adapter also implements `AllReduceSum` for a synchronous caller. The call
issues `AllReduceSumAsync` and then `Wait`.

#### 5.6 The watchdog

A rank failure must not hang the group. The seam `Wait()` takes no deadline, so
the adapter enforces one.

- `DistributedConfig` gains a `timeout_ms` field. The default is 600000
  milliseconds. That value gives one bucket ample time.
- `Wait()` computes a deadline as `now + timeout_ms` before it drains the FIFO.
- On `kTimeout`, the adapter prints the rank, the step, the slot, and the peer,
  then calls `Die`. A missing peer is a fatal group error, not a retry.
- The memory transport returns `kTimeout` on demand, so the CPU test covers the
  path. Section 13 adds the test.

### 6. Single precision only

The collective reduces single precision only. `ComputeType` is `float`. The
fp16 dtype branch is gone from the collective ([precision.md](precision.md)).

[precision.md](precision.md) section 6 defines the policy. Stage 1 removed fp16
from the tree, so every reduction is fp32. The collective needs fp32 only.

A rank group reduces in fp32 on every configuration, because the policy keeps
the persistent state single precision ([precision.md](precision.md)). The device
target for this work is `--config=hopper`, which computes in bf16 and adds the
native ring. A simulated run of the same build adds `sim`.

### 9. Broker and launch

The local GPU queue serves one card. It is not the broker for an 8x H100
node. Treat the node as one resource.

- One `tools/nanochat dist` invocation holds one lock for the whole rank group.
  this page already defines this. Keep that
  rule. Do not take one lock per rank.
- The lock covers all 8 cards. The `nvidia-smi` gate must see all 8 cards as
  idle before the group starts.
- Add a node-class sandbox profile to `tools/sandbox/profiles.json`, for example
  `node-train`. Size the memory, the central processing unit quota, and the
  process count for eight ranks on one node. Do not reuse the `train` profile,
  which serves the WSL2 host.
- One rank per card. Set `CUDA_VISIBLE_DEVICES` to the rank, as the ring
  already does.
- The T1 and T2 gates run on the node, under the broker. Keep the GPU tiers
  serialized, as `AGENTS.md` section 5.1 requires.
- Deploy the broker on the node. Do not point the node at the WSL2 broker
  configuration.

If a scheduler already serializes jobs on the node, do not add a second lock.
Integrate the rank group as one scheduler job. A second lock would deadlock
against the scheduler.

If the node has no broker and no scheduler, run the T1 and T2 gates through a
managed job on the node. Record the result with the commit hash.

### 10. The schedule trace

The two-stream event logic is the largest correctness risk. Build a trace, and
check it on the central processing unit.

The interposer already has the pattern: `tools/cuda_sim/cuda_sim_log.h` exposes
the JSONL record writer and reader the emulator suites use.

Extend the pattern. The schedule emits one record per send, receive, and add.
Each record carries the rank, the direction, the transport kind, the step, the
chunk, the slot, and the event dependencies. A central processing unit test
with the memory transport replays the records and checks four rules.

1. The schedule uses every slot once at a time.
2. The dependency graph has no cycle.
3. Every wait has a matching completion.
4. A peer that never completes returns `kTimeout`, and the adapter stops.

The simulator agent owns the trace convention. The oracle agent owns the test.

### 13. Test matrix

| Tier | Target | What it checks |
|---|---|---|
| T0 CPU | `//backends/cuda/collective:schedule_test` | The ring at world sizes 1, 2, 4, 8; the trace; the missing-peer timeout |
| T0 CPU | `//backends/cuda/collective:transport_select_test` | The per-neighbor selection, including a mixed group |
| T0 CPU | `//src:distributed_test` | The host reference, unchanged |
| T0 CPU | `//src:train_parallel_test` | The mean scale and the resume, unchanged |
| T1 GPU | `//backends/cuda/collective:nvlink_test` | A two-rank NVLink sum against the host reference |
| T1 GPU | `//backends/cuda/collective:rdma_test` | A two-rank RDMA sum against the host reference |
| T2 GPU | An 8-rank loss curve against 1 rank | The full training parity |

The schedule test is the primary new gate. It carries the algorithm. The
transport tests carry only data movement.

## Part 3. The single-client multi-device runtime

_Absorbed from the former `distributed.md`. Its section numbers are
unchanged, so a reference to section K of that document resolves to section K
here._

The tree implements and CPU-tests the C++ mesh runtime, the in-process
collective, the C application binary interface, the mesh training driver, and the
Python `mesh`/`runtime` client. The CUDA thread-local change, the loader
integration, and the compatibility process
runtime are later phases. The kernel seam and the public model API do not
change.

The mesh training driver is C++ by decision (section 20.5). The driver owns the
step loop, the metric record, the summary, and the resume. Python calls one
driver function per run. Phase 6 delivers the driver, so
`notebooks/mesh_train.py` is a thin launcher over it.

Related: this page,
this page, [python.md](python.md),
[build.md](build.md#layers-and-the-one-seam), and
[model-and-device-bringup.md](model-and-device-bringup.md) §18.

### 1. Purpose

This document gives a design for one client program that controls every local
accelerator. The client sees all devices as one system. The user starts one
program. The user needs no `torchrun`, no rank variable, and no rendezvous
port.

The design keeps two properties:

- **Single point of control.** One client program is the controller.
- **Unified view.** The client treats every local device as one machine.

The reference model is JAX on one host. The API shape follows
`jax.devices()`, `Mesh`, `NamedSharding`, and `PartitionSpec`.

### 2. Scope

In scope:

- One host.
- Two to eight local accelerators.
- One client process.
- Data-parallel training.
- Replicated inference and evaluation.
- The current fixed model graph.

Out of scope:

- More than one host.
- Tensor parallel and pipeline parallel.
- A general parallel array library.
- A sharding compiler.
- An autograd engine.
- Dynamic shapes and runtime dtype dispatch.

### 3. The current model and its cost

The project uses a single program, multiple data model today.
`tools/nanochat dist` starts one process per device. The launcher sets
`CUDA_VISIBLE_DEVICES` and the `NANOCHAT_*` variables for each process.
`src/train_command.cc` reads those variables. The gradient-sync seam reduces the
gradients.

The user pays a cost for this model:

- The rank is visible in the shape of the run.
- The user must start the group.
- The user must pick a master address and a port.
- The model exists one time per process.
- A checkpoint exists one time per rank.

The client design removes the rank from the user program. The runtime owns the
rank.

### 4. Decision

Two runtimes can give a single control point.

#### 4.1 Runtime A: one process, one device thread

One process owns every device. One host thread binds to one device. The thread
calls `cudaSetDevice` one time. The CUDA runtime then uses that device for every
later call on that thread. Each thread builds one model replica on its device.
The collective runs between threads in one address space.

#### 4.2 Runtime B: one process, one device process

One client process owns every device. The client starts one worker process per
device. The workers use the current runtime. The client is a supervisor. The
client sets the device variables for each worker.

#### 4.3 The choice

The project selects Runtime A as the target. Runtime A gives a true unified
address space. It removes the rendezvous and the rank variables.

Runtime B stays as a compatibility backend behind the same client interface.
Use Runtime B when a driver forbids more than one CUDA context in one process.

The rest of this document describes Runtime A. Section 15 describes Runtime B.

### 5. Programming model

The client exposes four concepts. They match JAX.

#### 5.1 Device list

`nanochat.devices()` returns every local accelerator. The call does not create a
CUDA context. The runtime reads the device count and the device properties.

```python
devices = nanochat.devices()
```

#### 5.2 Mesh

`Mesh(devices, axis_names)` binds a named axis to the device list.

```python
mesh = nanochat.Mesh(devices, ("data",))
```

The runtime accepts a rank-1 mesh only. The one axis must have the name
`"data"`. The `MeshSpec` type carries a shape vector and a name vector, so a
rank-N mesh is a later additive change. The runtime rejects a rank-N mesh and
an unknown axis name at construction time. The error names the supported axis.

The project puts tensor parallel and pipeline parallel out of scope (section
2). A second axis has no executor, because the kernel seam has no device
argument. Section 20.1 records this decision.

#### 5.3 Placement

`PartitionSpec` names the sharded axes of one tensor. A value of `None` means
replicated. A name means sharded along that mesh axis.

```python
from nanochat import PartitionSpec as P

params = nanochat.NamedSharding(mesh, P())        # replicated
batch = nanochat.NamedSharding(mesh, P("data"))   # sharded on 'data'
```

#### 5.4 Supported placements

The project has a fixed graph. It has no sharding compiler. So the runtime
supports a small placement set.

| Object | Placement | Meaning |
|---|---|---|
| Model parameters | `P()` | One full replica per device |
| Optimizer state | `P()` | One full state per device |
| Global batch | `P("data")` | Contiguous slices, one per device |
| Activations | derived | Follow the batch placement |
| Gradients | derived | Follow the parameter placement |

The runtime rejects any other placement at construction time. The error names
the supported set. Parameter sharding is tensor parallel. Section 2 puts that
work out of scope.

#### 5.5 The program

A training program mentions no rank.

```python
import nanochat

devices = nanochat.devices()
model = nanochat.Config(num_layers=12, hidden_dim=768)

train = nanochat.MeshTrainConfig(
    model, preset="track3", world_size=len(devices),
    tokenizer="tokenizer.nctoken", train_parquet="shard-*.parquet",
    per_rank_batch=1, grad_accum=4, steps=5000, out_dir="run")
result = train.run()
print(result.mean_loss)
```

The `MeshTrainConfig.run` call returns when the run stops. It writes
`metrics.jsonl`, `summary.json`, and the canonical checkpoint in `out_dir`. The
client never names a rank and never steps the optimizer. Section 20.5 records
this decision.

The lower-level `MeshRuntime` stays. `MeshRuntime.train_step()` splits one
global batch across the mesh, reduces the gradients in process, and returns one
loss. A test and a custom loop use it. `MeshRuntime.metrics()` returns the
per-device breakdown and the timing fields. Section 20.4 records that decision.

### 6. What the model is not

The client is not a general parallel array library.

- There is no `jnp`.
- There is no `jit` over user code. The graph is the code.
- There is no `vmap` or `pmap` over user functions.
- There is no `psum` or `pmean` in user code.

The runtime reduces the gradients itself. The parallel entry points are the
fixed operations: `forward_loss`, `step`, `generate`, and `evaluate`.

### 7. Architecture

The runtime adds one host layer. The layer sits above the per-replica driver. It
never touches the kernel seam.

```text
App / CLI        train.py   (one process, one program)
Runtime (new)    DeviceMesh  DeviceContext  ReplicaGroup  Collective
Train driver     TrainLoop   Checkpointer  Logger      (one per replica)
Optim            AdamW  Muon  Scheduler
Model / graphs   Model  Prefill  Decode  KvCache
Ops
KERNEL API       <-- unchanged seam
Backend          backends/cpu/   backends/cuda/
```

New files:

| File | Purpose | Owner |
|---|---|---|
| `include/nanochat/mesh.h` | `DeviceInfo`, `EnumerateDevices`, `MeshSpec` | Architect |
| `src/mesh.cc` | The mesh validation shared by the runtime and the client | Architect |
| `src/mesh_collective.h`, `src/mesh_collective.cc` | The in-process all-reduce | Collective |
| `src/mesh_runtime.h`, `src/mesh_runtime.cc` | Device threads, replicas, step dispatch | Harness |
| `src/mesh_train.h`, `src/mesh_train.cc` | The mesh training driver: the step loop, the metric record, the summary, and the resume | Harness |
| `capi.h` additions | Device enumeration, the mesh handle, the runtime handle, the driver config struct, and the driver run call | Architect |
| `python/nanochat_cpp/mesh.py` | `devices`, `Mesh`, `NamedSharding`, `PartitionSpec` | Python surface |
| `python/nanochat_cpp/runtime.py` | `MeshRuntime`, and the coarse driver client | Python surface |

The frozen header `kernels.h` does not change. Section 8 explains why.

### 8. Thread-affine device selection

The seam has no device argument. Every kernel call carries a pointer and a
shape. The runtime does not change the seam. The runtime uses one rule instead:
one device thread owns one device.

- The thread calls `cudaSetDevice(index)` one time.
- The CUDA runtime uses that device for every later call on the thread.
- `kernels::Alloc`, `kernels::Gemm`, and the other calls use the calling
  thread's device.
- Each thread builds its own `Model`. The buffers land on the right device.
- Each thread destroys its own model and its own workspace.

This rule keeps the seam frozen.

#### 8.1 Backend state that is thread-local

The CUDA backend holds process-global state in the first draft. Each item is
thread-local now:

| Item | File |
|---|---|
| cuBLAS handle | `backends/cuda/gemm.cu` |
| Tensor-core flag | `backends/cuda/gemm.cu` |
| Device name buffer | `backends/cuda/device.cu` |
| Muon scratch | `backends/cuda/kernels/muon.cu` |
| Attention scratch | `backends/cuda/kernels/attention.cu` |

The device globals in `backends/cuda/kernels/grad_norm.cu` stay device
globals. Each device has its own copy of the module state. One thread drives one
device, so the order is safe.

#### 8.2 The seam stays frozen

A thread-local change is invisible above the seam. The `src/**` layer sees no
change. The `kernels.h` contract is unchanged. The evolution rules in
[model-and-device-bringup.md](model-and-device-bringup.md) §18 permit this class
of change.

### 9. The in-process collective

The in-process collective replaces the TCP host reference on one host.

#### 9.1 Interface

The collective satisfies `GradientSync` in `src/distributed.h`. The seam does
not change.

#### 9.2 Structure

One `CollectiveGroup` holds every device thread.

- A barrier synchronizes the group.
- One pinned host buffer holds one slot per device.
- Each device copies its gradient bucket to its slot.
- After the barrier, each device sums every slot in rank order.
- Each device copies the sum back to its bucket.

The sum order is the rank order. Every replica sees the same bytes. So every
optimizer step is identical.

#### 9.3 Memory

One bucket holds a fixed element count. The staging buffer holds `world_size`
slots. A large gradient needs several buckets. The current harness already
buckets the gradients. The runtime reuses that code.

#### 9.4 The fast path

When a device pair has peer access, the runtime can copy between devices with
`cudaMemcpyPeerAsync`. The ring schedule in
`backends/cuda/collective/ring_schedule.cc` already exists. A later phase can
reuse it. The host path is correct first.

### 10. One training step

1. The controller reads one global micro-batch from the loader.
2. The controller splits the batch into `world_size` contiguous slices.
3. The controller puts one slice on each device thread queue.
4. Each device thread runs `forward_loss` and `backward_accumulate` with the
   scale `1 / (accum * world_size)`.
5. Each device thread packs its gradient buckets.
6. The collective sums every bucket. Section 9.2 gives the order.
7. Each device thread runs one optimizer step with the same reduced gradient.
8. The controller collects the per-device losses and returns the global mean.

The global batch stays constant on any device count. Only the work per device
changes. This matches `docs/distributed.md` section 3.

#### 10.1 The training driver owns the loop

`MeshRuntime::TrainStep` is one step. It is the mechanism. A separate C++
driver owns the run around it. The driver is `src/mesh_train.cc`.

The driver owns five items:

1. The step loop from the resume step to the last step.
2. The metric record of each step, and the write to `metrics.jsonl`.
3. The run summary, and the write to `summary.json`.
4. The checkpoint interval and the final checkpoint.
5. The interrupt path: the driver saves a checkpoint before the error leaves.

The driver owns the loader. It accepts an injected `DocumentSourceFactory`, so
a test supplies documents without parquet. Section 11 gives the loader rule.

The driver writes one record shape. The shape follows the `LogRecord` fields in
`src/train.cc` and adds one loss entry per device:

```json
{"version": 2, "step": 1, "loss": 4.1, "grad_norm": 0.9,
 "tokens_per_sec": 12000.0, "mfu": 0.31, "step_seconds": 0.2,
 "data_seconds": 0.004, "compute_seconds": 0.19, "seconds": 0.2,
 "device_loss": [4.1, 4.1]}
```

Schema 2 adds `data_seconds` and writes `compute_seconds`, which schema 1
measured and printed only to the human log. The three timing fields split one
step: `step_seconds` is the wall clock between two step starts, `data_seconds`
is the batch load, and `compute_seconds` is the training call. The report reads
them instead of dividing the stage wall clock by the step count, because that
quotient charges the build and the corpus copy to every step.

The summary holds `version`, `checkpoint`, `steps`, `mean_loss`,
`steady_mfu`, and `replica_digests`. The driver owns both files. Python reads
the files and displays the rows. Python does not compute a record.

The `version` field is one integer. A consumer ignores an unknown field and
rejects an unknown version. Section 20.5 records this decision.

### 11. Data and the loader

The loader yields a global batch. The runtime owns the split. The user does not
see a rank.

In phase 2 the controller reads one global batch and slices it. This path
matches the reference rule in `docs/distributed.md` section 7: one global
token batch splits into contiguous slices. The CPU mesh test needs no overlap,
so the simple path is enough. The host batch is small. A 524288-token batch
needs about 4 MiB for the tokens and the targets.

The controller path needs a small interface change. The per-replica `TrainLoop`
gains an entry point that accepts an injected batch. The current `Run()` reads
its own data, so it cannot serve the mesh. Section 17 phase 2 covers this
change.

In phase 5 the read moves behind the runtime. Each device thread reads its own
shard through a pinned double buffer. That change helps the overlap of the data
read with the step. It is not necessary for correctness. Section 20.2 records
this decision.

### 12. Checkpoints

One checkpoint is canonical. The runtime writes the checkpoint from replica 0.
The runtime loads a checkpoint into every replica. The replicas stay
byte-identical. The current design already gives this property. The runtime
keeps the sidecar data cursor for an exact resume.

### 13. Inference and evaluation

Replicated inference: every device holds a full copy of the model. The runtime
puts one prompt on each device and fans out the rows. `GenerateBatch` and
`GenerateMultiPrompt` already batch rows.

Evaluation: the runtime splits the validation set across the devices and
reduces the byte-per-byte totals.

These cases need no gradient reduction. They use the barrier and a small scalar
reduction only.

### 14. The entry point and the sandbox

The client is the entry point. The runtime keeps the sandbox rule.

- The client process runs inside one sandbox scope. It takes one queue job for
  the device set.
- `tools/nanochat run train -- python train.py` is the strict form.
- The console script `python -m nanochat_cpp.run train.py` re-enters
  `tools/nanochat client`. The user sees one command.

The `nanochat_init()` sandbox check stays. A run outside the entry point still
fails fast.

The worker owns the full device set, for example `0,1,2,3`. One job covers the
group. A second job on the same set waits.

### 15. The compatibility runtime

Runtime B starts one worker process per device behind the same client
interface.

- `Mesh` and `PartitionSpec` behave the same.
- The client sets `CUDA_VISIBLE_DEVICES` and the rank variables per worker.
- The client uses `tools/nanochat dist` today.
- The reduction uses the native ring.

The user program does not change. The runtime backend is a construction option.

```python
runtime = nanochat.Runtime(mesh, backend="inproc")     # default
runtime = nanochat.Runtime(mesh, backend="process")    # compatibility
```

Runtime B stays a supported backend. It is not a second API. Three cases need
it:

- Some drivers forbid more than one CUDA context in one process.
- Kaggle runs two T4 cards, and a notebook can prefer process isolation.
- One bad device stops one worker, not the whole client.

Runtime B shares the mesh code and the placement code. So the extra cost is
small. Section 20.3 records this decision.

### 16. Determinism and parity

- The reduction sums the buckets in rank order.
- The gradient is the mean over the global batch.
- Every replica applies the same update. So the replicas stay equal.
- The validation loss of a multi-device run equals the validation loss of a
  one-device run on the fixed set.

Section 18 pins these properties.

### 18. Tests

| Tier | Test | Gate |
|---|---|---|
| T0 CPU | `//src:mesh_test` | Device enumeration and mesh construction |
| T0 CPU | `//src:runtime_test` | Two CPU devices equal one CPU device |
| T0 CPU | `//src:mesh_train_test` | The driver record, the summary, the resume, the interrupt save, and the data-cursor resume |
| T0 CPU | `//backends/host/collective:inproc_sync_test` | The in-process sum matches the host reference |
| T0 CPU | `//python:mesh_test` | The mesh, the placement, and the rejection of a bad placement |
| T0 CPU | `//python:runtime_test` | The mesh runtime client over the C application binary interface |
| T0 CPU | `//python:mesh_train_test` | The driver client writes the same records as the C++ driver |
| T1 GPU | `//src:runtime_gpu_test` | Two device threads on one card |
| T2 GPU | A two-device run versus one device | The validation loss matches one device |

The CPU tests carry the mesh logic. The development host has one card, so the
CPU path is the main gate. This matches [kernels.md](kernels.md) § Compute
strategy and `AGENTS.md` section 5.1.

### 20. Resolved decisions

This section answers the four questions that the first draft left open. Each
answer gives the decision and the reason.

#### 20.1 The mesh has one axis

Decision: the runtime accepts a rank-1 mesh. The one axis is named `"data"`. A
rank-N mesh is rejected at construction time.

Reasons:

- Tensor parallel and pipeline parallel stay out of scope (section 2).
- The kernel seam has no device argument, so a second axis has no executor.
- The current collective reduces one gradient group along one axis.

`MeshSpec` already carries vectors for the shape and the names. So a later
second axis is an additive change. The first axis keeps the name `"data"`.

#### 20.2 The controller reads the global batch in phase 2

Decision: in phase 2 the controller reads one global batch and slices it. Phase
5 moves the read behind the runtime.

Reasons:

- The contiguous global slice matches the reference rule in
  `docs/distributed.md` section 7.
- The CPU mesh test needs no overlap, and phase 2 must stay small.
- The host batch is small. A 524288-token batch needs about 4 MiB.

Phase 5 adds a pinned double buffer per device and reads each shard on its
device thread. That change is an optimization. It does not change the result.

#### 20.3 The compatibility runtime stays

Decision: Runtime B stays a supported backend. It is a construction option, not
a second API. The project does not remove it at phase 5.

Reasons:

- Some drivers forbid more than one CUDA context in one process.
- Kaggle runs two T4 cards, and a notebook can prefer process isolation.
- One bad device stops one worker, not the whole client.

Runtime B shares the mesh code and the placement code. So the cost is small. A
later removal needs a design review.

#### 20.4 One loss, with a per-device breakdown

Decision: `MeshRuntime.train_step()` returns one global mean loss. A later
`metrics()` accessor returns the per-device split and the timing fields.

Reasons:

- The unified view wants one result for one step.
- The JAX reference returns one value from a jitted function.
- Debugging still needs the per-device split. The harness already collects it.

The runtime computes the global loss with a scalar reduction over the valid
tokens. The delivered client returns one loss, and `MeshRuntime.param_read`
exposes one replica at a time. A later `metrics()` follows the `LogRecord`
fields in `src/train.cc` and adds one loss entry per device.

#### 20.5 The C++ driver owns the mesh loop

Decision: a C++ driver owns the mesh step loop, the metric record, the summary,
and the resume. Python calls one driver function per run. Python keeps the data
choice, the tokenizer path, the MFU inputs, and the display.

Reasons:

- `docs/distributed.md` Part 2, section 2 already fixes the split. C++ owns mechanism.
  Python owns policy and data. Section 9 of that document states the invariant.
- `notebooks/mesh_train.py` rebuilds the harness layer. It repeats the step
  loop, the MFU, the log format, and the summary.
- The two log formats already differ. `Logger::Log` and `format_row` print
  different fields for the same step.
- `notebooks/mesh_train.py` carried a second config class beside the driver
  config. One field change needed edits in both places.

The driver config mirrors the existing `TrainConfig` and `MeshRuntimeConfig`
fields. Python owns the job file for the child process, so the C interface
carries no serializer. The notebook keeps the child-process launch and the file
polling, because those are process policy.

This decision does not change the seam, the frozen headers, or the runtime
step arithmetic. It moves the loop and the record only.
