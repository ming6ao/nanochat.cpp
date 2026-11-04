# Distributed training design

Data-parallel training on two devices. This document describes the gradient-sync
seam, the data sharding, the data cursor, the model configuration, and the
launch.

Status: the project implements the data-parallel architecture. `Config` carries
`value_embedding`. `src/distributed.h` is the seam. The CPU backend links the
host reference in `src/distributed.cc`. The CUDA backend links NCCL in
`backends/cuda/nccl_sync.cu` by default. The native collective
(`backends/cuda/collective/`, selected with `--config=native` or
`--define=collective=native`) is the NCCL replacement
([distributed-design.md](distributed-design.md)). It is complete for
the ring schedule, the CPU memory transport, and the host-staged CUDA
transport; the NVLink and RDMA transports await the H100 node.

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

Two implementations ship today, and a third is selectable. The build selects
one, and the selection follows the backend. There is no host-reference fallback
on the CUDA path.

| Implementation | File | Use |
|---|---|---|
| Host reference | `src/distributed.cc` | The CPU backend and the CPU tests |
| NCCL | `backends/cuda/nccl_sync.cu` | The CUDA backend and the two T4 cards (default) |
| Native ring | `backends/cuda/collective/native_sync.cu` | The CUDA backend with `collective=native` |

The native ring ([distributed-design.md](distributed-design.md))
replaces NCCL on the CUDA path. It runs a ring reduce-scatter + all-gather over
a completion-based `Transport`. The selection is per ring edge
(section 3.2 of the native plan): NVLink when `cudaDeviceCanAccessPeer` passes,
RDMA when an adapter reaches the peer, and host staging otherwise. The 2x T4
pair has no peer access, so it exercises the host-staged path; the fast path is
the 8x H100 node.

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

`train_main` gains five options: `--preset track3`, `--rank`, `--world-size`,
`--master`, and `--port`. The preset sets the Track 3 model, optimizer, and
scheduler fields. Pass `--preset track3` before the other flags, so a later
flag can override one field.

One process owns one rank. `tools/nanochat dist` starts the group. The
environment carries the rank values, in the style of `torchrun`. The `dist`
command submits one queue job and applies one sandbox scope for the whole group.
[distributed-design.md](distributed-design.md) is the authority for the
launch.

This design removes the per-rank launch in `notebooks/parallel.py`. That code
took a lock once per rank, so a multi-card host could deadlock. Kaggle hid the
fault, because the sandbox and the queue are off there.

### 10.1 The best fit and the measured run

The model-fit skill (`.agents/skills/model-fit/`) computes the shape and the
memory. The best fit for the two T4 cards is `L16 C1024 H8 KV4 T4096 b1`: about
252M parameters, about 9.1 GiB of card memory, and an estimated 5.1k tokens/s
global.

The measured 500-step run on two T4 cards uses the native ring collective
(`--config=native`), the windowed attention backward, and pinned host staging.
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

The two-card notebook is `notebooks/nanochat-cpp-on-2x-t4.ipynb`. It uses the
single-client multi-device runtime, so one process owns both cards
([client-runtime-design.md](client-runtime-design.md)). The multi-process plan
lives in [distributed-design.md](distributed-design.md) and
[distributed-design.md](distributed-design.md).

The notebook runs these steps:

1. Build the `t4` shared library. The library already holds the mesh runtime.
2. Stage one ClimbMix shard, the validation shard, and the tokenizer.
3. Run the best-fit model with one `MeshRuntime` over the two cards.
4. Compare the two replica parameter digests, and keep the canonical
   checkpoint.
5. Load the checkpoint in process. Run the base evaluation, the supervised
   fine-tuning, and the chat evaluation.

The runtime stages the gradients through the kernel seam, so the CUDA build
needs no NCCL. The T4 cards do not have a peer link, so the peer-to-peer fast
path waits (section 10.1).

## 14. Phasing

The architecture phases P0 to P4 are complete. They cover the seam, the Track 3
preset, the host reference, the CPU two-rank test, NCCL, the overlap seam, and
the data cursor.

The NCCL gate and the two-rank parity gate run on Kaggle (section 15).

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

---

## Part 1. The launch of a data-parallel run

_Absorbed from the former `distributed-launch.md`. Its section numbers are
unchanged, so a reference to a section of that document resolves to the same
number here. The execution record is in [campaigns.md](campaigns.md)._

Status: delivered. `tools/nanochat dist` starts the rank group, and
`python/nanochat_cpp/distributed.py` wraps it. The two-card notebook now uses
the single-client multi-device runtime instead
(`docs/client-runtime-design.md`), so this launch serves the process-isolation
case. The superseded per-rank helper `notebooks/parallel.py` is gone (phase 4).
This document is the authority for the launch.
[distributed-design.md](distributed-design.md) section 10 holds a summary.

### 1. Problem

`notebooks/parallel.py` starts one rank per card. It calls `tools/nanochat
train` for each rank. Two faults follow.

The first fault is the layer. The file is process orchestration.
`DESIGN.md` section 2.2 puts the process entry point outside the model layers.
The notebooks directory ([../notebooks](../notebooks)) holds
presentation, not orchestration.

The second fault is the GPU queue. `tools/nanochat train` submits one job for
one rank. Two ranks submit twice, and the queue runs them one after the other.
A multi-card host then serializes the ranks, or the run blocks. Kaggle hides
the fault, because the sandbox and the queue are off there.

### 2. Goals and non-goals

Goals:

- One process per rank.
- One queue job for the rank group.
- One sandbox scope for the group.
- A rank contract in the environment, as `torchrun` uses.
- One shell entry point, `tools/nanochat dist`.
- No change to the gradient seam or to the model.

Non-goals:

- More than one node.
- Tensor parallel and pipeline parallel.
- Elastic rendezvous.
- Distributed evaluation.
- A launcher in C++.

### 3. Position in the architecture

`train_main` owns one rank. The rank values configure training, so C++
owns their defaults. The launch, the queue, and the sandbox are host tooling.
`DESIGN.md` section 2.2 and section 6 put them there.

| Task | Owner | File |
|---|---|---|
| Rank and group values | C++ at L6 | `src/train_main.cc` |
| Gradient sum | C++ | `src/distributed.cc`, `backends/cuda/nccl_sync.cu` |
| Process start | Host tooling | `tools/nanochat` |
| GPU queue job | Host tooling | `python/nanochat_cpp/gpu_queue.py` |
| Sandbox scope | Host tooling | `python/nanochat_cpp/sandbox.py` |
| Notebook helper | Python | `python/nanochat_cpp/distributed.py` |

A launcher in C++ would move host tooling into L6. It would also run before
the sandbox, so it could not apply the sandbox itself. The project does not
add it.

### 4. The environment contract

`train_main` reads five variables. The launcher sets them. A flag overrides
the variable.

| Variable | Meaning | Flag |
|---|---|---|
| `NANOCHAT_WORLD_SIZE` | The group size | `--world-size` |
| `NANOCHAT_RANK` | The global rank | `--rank` |
| `NANOCHAT_LOCAL_RANK` | The rank on this node | none |
| `NANOCHAT_MASTER_ADDR` | The rank 0 host | `--master` |
| `NANOCHAT_MASTER_PORT` | The rank 0 port | `--port` |

The resolution order is the flag, then the `NANOCHAT_*` variable, then the
default. The prefix `NANOCHAT_` prevents a clash with another tool. The names
follow `torchrun`, so `train_main` can also run under `torchrun`, `mpirun`, or
a shell loop.

On one node, `NANOCHAT_LOCAL_RANK` equals `NANOCHAT_RANK`. The variable exists
for a later multi-node change.

The C++ side adds two small helpers to `src/cli.h`: `EnvInt` and `EnvString`.
The helpers read the variable and parse the value.

### 5. The launcher

`tools/nanochat dist` starts the group. The command line is:

```text
tools/nanochat dist [--nproc-per-node N] [--device LIST] [--profile P]
                    [--master ADDR] [--port PORT] [--log-dir DIR] [--json]
                    -- <command...>
```

The `dist` command does these steps:

1. Select the device set. The default is `0..N-1`, or the first `N` entries
   of the worker's pool when `NANOCHAT_GPU_DEVICES` is set.
2. Submit one queue job for the whole group, and wait for it.
3. Apply one sandbox profile to the group.
4. Set `CUDA_VISIBLE_DEVICES`, `NANOCHAT_RANK`, `NANOCHAT_LOCAL_RANK`, and
   `NANOCHAT_WORLD_SIZE` for each rank.
5. Start rank 0 first, so the bind wins the port race.
6. Redirect each rank to `<log-dir>/rank<i>.stdio.log`.
7. Wait for every rank. Return the first nonzero status.
8. Print a summary. With `--json`, print one JSON object.

The `dist` command skips the queue in two cases:

- `NANOCHAT_GPU_QUEUED` has a value, so a queue job already runs it.
- The sandbox backend is `none`, as on Kaggle.

The `dist` command must kill every rank on a signal. It must also wait for
every child, so no orphan rank holds a card.

The `dist` profile in `tools/sandbox/profiles.json` covers the group. The
`train` profile covers one rank, so it is too small.

The summary gives the per-rank return code, the log path, the stdio path, and
the seconds. The format follows `tools/nanochat doctor --json`.

### 6. The Python surface

`python/nanochat_cpp/distributed.py` is an optional wrapper for a notebook.
It does not start a process. It calls `tools/nanochat dist --json` and parses
the summary.

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

The command table `_entry.py` gains one `dist` row. The `Toolchain` class
gains one `dist` method. `parse_log` and `loss_rows` move from
`notebooks/parallel.py`.

`Logger::Log` in `src/train.cc` writes the log line. The format is a
contract. One fixture test pins it.

### 7. The notebook

The two-card notebook no longer uses this launch. It uses the single-client
multi-device runtime: one process, one `MeshRuntime`, and the in-process
collective (`docs/client-runtime-design.md`). The mesh removes the rank from the
user program, so the notebook opens no port and sets no rank variable. The
launch stays for the process-isolation case: some drivers forbid two CUDA
contexts in one process, and one bad device then stops one worker only.

### 8. Plan

#### Phase 0 — Design and ownership

1. Register this document in `docs/README.md`.
2. Change [distributed-design.md](distributed-design.md) section 10 to a
   summary.
3. Assign the work: architect, harness, Python surface, and notebooks
   (`AGENTS.md` section 1).

Gate: the architect approves the variable names and the `dist` command line.

#### Phase 1 — The environment contract

1. Add `EnvInt` and `EnvString` to `src/cli.h`.
2. Set the five defaults in `src/train_main.cc`.

Gate: a CPU test covers the flag-over-variable order.

#### Phase 2 — The launcher

1. Add the `dist` case to `tools/nanochat`.
2. Add the `dist` profile to `tools/sandbox/profiles.json`.
3. Add the signal handler and the wait.
4. Add the `--json` summary.

Gate: a CPU test with a fake command. The test checks the first nonzero
status, the rank logs, the signal kill, and one queue job for the group.

#### Phase 3 — The Python surface

1. Add `python/nanochat_cpp/distributed.py`.
2. Add the `dist` row to `_entry.py` and the method to `toolchain.py`.

Gate: a Python test with a fake `tools/nanochat`.

#### Phase 4 — The notebook and the removal

1. Change the notebook to `nanochat_cpp.distributed`.
2. Delete `notebooks/parallel.py`, `notebooks/parallel_test.py`, and the two
   `notebooks/BUILD.bazel` targets.
3. Remove `SMOKE_FLAGS` (it lived only in `notebooks/parallel.py`).

Gate: one short two-rank Kaggle run passes the checkpoint hash check.

### 9. Definition of Done

1. `tools/nanochat dist --nproc-per-node 2 -- <command>` starts two ranks.
2. One queue job covers the group. A second job on the same device set waits.
3. The `dist` command exits with the first nonzero rank status.
4. A signal kills every rank.
5. `--json` prints one object with the per-rank fields.
6. A flag overrides its `NANOCHAT_*` variable.
7. `notebooks/parallel.py` and `notebooks/parallel_test.py` are gone.
8. `tools/nanochat lint` passes. The CPU tests pass under `t0-cpu`.

### 10. Risks and controls

| Risk | Control |
|---|---|
| A generic variable name clashes with another tool | Use the `NANOCHAT_` prefix. A flag wins |
| The completed NCCL plan gains a new phase | This document holds the launch plan. The NCCL plan stays closed |
| The group exceeds the `train` profile | Add a `dist` profile sized for the group |
| A signal leaves an orphan rank | Trap the signal and kill the group. Test it |
| The log format drifts from the parser | One fixture test pins the format |
| The move crosses a workstream boundary | Phase 0 assigns the owners first |
| Kaggle has no queue | Skip the queue when the backend is `none` |

### 11. Ownership

| Phase | Owner | Files |
|---|---|---|
| 0 | Architect | `docs/**` |
| 1 | Harness | `src/train_main.cc`, `src/cli.h` |
| 2 | Architect | `tools/nanochat`, `tools/sandbox/profiles.json` |
| 3 | Python surface | `python/nanochat_cpp/**` |
| 4 | Notebooks | `notebooks/**` |

### 12. Open questions

1. Does the `dist` profile replace the `train` profile, or sit beside it?
2. Should `dist` default the log directory, or require it?
3. Should `--device 0` allow two ranks on one card, for a smoke test?


---

## Part 2. The training seam between Python and C++

_Absorbed from the former `training-seam.md`. Its section numbers are
unchanged, so a reference to a section of that document resolves to the same
number here. The execution record is in [campaigns.md](campaigns.md)._

Status: partly delivered. The project completed the first increment of section
8, and section 5 marks the fixed defects. This document specifies the boundary
between the Python surface and the C++ training driver. It also records the
defects in the current boundary. Section 6.6 extends the boundary to the mesh
training driver, and phase 6 of section 8 delivers it.

`post-training.md` section 11 stays the status of record for the post-training
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

`DESIGN.md` section 2 puts the training driver at level L5. Level L5 holds
`TrainLoop`, `Scheduler`, `Checkpointer`, `Logger`, and the MFU accounting. The
driver is C++ by design.

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

### 5. Defects in the current seam

The first increment fixed 5.1, 5.2, and 5.6, and partly fixed 5.3. The mesh
driver phase fixed 5.8. The other items remain.

#### 5.1 The C++ interface is richer than the C interface (fixed)

`Model::TrainStep` implements forward, backward, and one optimizer step
(`src/model.cc:951`). The first increment added `nanochat_train_step`, and
`Model.train_step` calls it (`python/nanochat_cpp/api.py`). The `Trainer` keeps
the lower-level calls, because it supports gradient accumulation.

#### 5.2 The C interface has no accumulation call (fixed)

`Model::BackwardAccumulate(scale)` exists and the tests cover it. The first
increment added `nanochat_backward_accumulate`, and `Trainer.__next__` calls
`Model.backward_accumulate` (`python/nanochat_cpp/api.py`). The host float array
of `batch * seq` entries is gone.

#### 5.3 The schedule leaks to the caller (partly fixed)

`Optimizer::Step(int step)` takes a 1-based step from the caller
(`include/nanochat/optim.h:56`). `TrainLoop` now persists the step and resumes
from it, so a driver owns the counter. A custom Python loop still passes the
step. Recommendation 4 in section 10 keeps `Optimizer::Step` in C++.

#### 5.4 The Python trainer duplicates the C++ loop

`Trainer` reimplements gradient accumulation, the epoch reset, and `zero_grad`
(`python/nanochat_cpp/api.py:800-833`). `src/train.cc` is the source of truth
for the same logic. The Python tests do gate the loss curve against a C++
fixture (`python/tests/api_test.py:143`). The gate compares per-step losses only.
The loop structure can still drift.

#### 5.5 There is no supervised fine-tuning or reinforcement-learning driver
(partly fixed)

`Trainer` takes `TokenData`, which reads parquet. The mesh driver now reads a
packed `NCSFT001` shard and does a weights-only warm start, so `posttrain.ipynb`
runs through the C++ driver (section 6.2). There is still no `SftTrainer` or
`sft_main` binary, and no `RlTrainer`. The user writes mechanism for the
reinforcement-learning path. Section 11 of
[post-training.md](post-training.md) lists the missing parts.

#### 5.6 The renderer is private and incomplete (fixed)

The first increment added the public `render_conversation` in
`python/nanochat_cpp/chat.py`. `ChatEvaluator` and `python/tests/chat_test.py:68`
use the private `_PromptRenderer`, which stays private. The `_specials` helper
stays internal.

#### 5.7 Forward-looking risk: the reinforcement-learning divisor

Reinforcement learning does not exist yet, so no code has this defect today.
The risk is for the future.

`Model::BackwardWeighted(row_weights, scale)` takes raw per-row weights.
[post-training](post-training.md) gives the divisor as the product of the valid
count, the pass count, and the example count per rank. Python would compute
that divisor today. The normalization is therefore not the single source of
truth. The divisor must move into C++ with the reinforcement-learning step.

#### 5.8 The mesh client rebuilt the driver (fixed)

`notebooks/mesh_train.py` drove the mesh runtime one batch at a time. It owned
the loop, the record, and the summary. Each item in the first four rows
duplicated a C++ source of truth. The last two rows had no C++ writer:

| Python item | C++ source of truth |
|---|---|
| `iterate()` | `TrainLoop::Run` (`src/train.cc`) |
| `mfu()`, `steady_mfu()` | `ComputeMfu` (`include/nanochat/mfu.h`) |
| `format_row()` | `Logger::Log` (`src/train.cc:70`) |
| `job()`, `from_job()` | `TrainConfig` (`src/train.h:32`) |
| the `metrics.jsonl` writer | none |
| the `summary.json` writer | none |

The two log formats differed. `Logger::Log` prints `lr`, the timing split, and
the evaluation time. `format_row` printed the gradient norm, the step time,
the compute time, the throughput, and the MFU. So one record had two shapes.

`job()` and `from_job()` were a hand-written serializer of twenty fields. No
version marked the payload, so a stale `job.json` lost a field in silence.

`src/mesh_runtime.cc` provides `TrainStep` and no loop. So the client could not
delegate the loop. [client-runtime-design](client-runtime-design.md)
section 10.1 and section 20.5 record the decision. `src/mesh_train.h` and
`src/mesh_train.cc` now own the loop, the record, and the summary. Python owns
the job file, and the record carries a version.

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
and `sft_main` and the mesh driver share one loader. The driver loads the shard
and cycles the rows, so it needs no reader thread.

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

`nc.sft.run` calls the toolchain, which runs `tools/nanochat train -- sft_main
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
- Autograd. The graph stays fixed ([DESIGN.md](../DESIGN.md) section 5).

### 8. Migration plan

Each phase has an owner and a gate. The owners follow `AGENTS.md` section 1.
The phases extend [post-training.md](post-training.md) section 12 with the
interface work.

| Phase | Owner | Gate |
|---|---|---|
| 1. Expose the existing step. Add `nanochat_train_step` and `nanochat_backward_accumulate`, use `backward_accumulate` in `Trainer`, and persist the optimizer step. (Done.) | Architect for `capi.h`; Python surface for `bindings/` and `api.py`; Harness for `src/train.cc` | The Python tests and `//bindings:nanochat_capi_test` |
| 2. Add the packed-row source and the reader. Add `render_conversation`. (Partly done: the renderer, the Python packer, and the mesh driver shard reader exist. The `TrainLoop` packed-row source remains.) | Harness for the loader; Python surface for the renderer | A packer test and a mask-shift test |
| 3. Add the supervised fine-tuning driver and `sft_main`. (Partly done: the mesh driver runs the SFT loop from a shard. The standalone `sft_main` remains.) | Harness for `src/`; Python surface for the facade | A short run at tier T2 against the reference loss curve |
| 4. Add `nanochat_rl_step` and the parity fixture. | Harness for the step; Oracle for the fixture and the test | `//tests:rl_parity_test` at tier T2 |
| 5. Add the reinforcement-learning worker. | Harness for the worker; Python surface for the bridge | An RL smoke run at tier T2 |
| 6. Add the mesh training driver. Add `src/mesh_train.cc` and the driver calls in `capi.h`. Use the Python driver config directly. (Done.) | Harness for `src/**`; Architect for `capi.h`; Python surface for the client; Notebooks for the launcher | The CPU driver loss sequence equals the old Python loop. The resume and the interrupt save work |

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

### 10. Open questions and recommendations

1. **Should the step counter survive a resume?** Yes.
   `Scheduler::LrMultiplier(step)` needs the step, and `TrainLoop::Run` starts
   at step 1 today. A resume therefore replays the warmup and breaks the curve.
   Store one more record beside the optimizer state in `Checkpointer::SaveModel`.
   On load, the driver continues to the absolute `num_iterations`. The first
   increment restores the schedule but not the data position, so a resumed run
   reads the stream from the first batch.
2. **Is `capi.h` part of the public API?** No.
   Section 1 of `AGENTS.md` freezes `capi.h`, and the architect owns it. The C
   interface is an internal seam for the Python binding, not a public API.
   Consumers use the Python package. Section 7 of the design document lists the
   public C++ headers, and it stays unchanged.
3. **Should the chat template stay in Python?** Yes, for now.
   [the evaluation document](eval.md) section 2 already gives prompt rendering
   to Python. The reference `nanochat` package owns the template, and parity is
   the goal. The C++ tokenizer has no render method
   (`include/nanochat/tokenizer.h`), so a C++ copy would be a second
   implementation that can drift. Promote `render_conversation` to a public
   Python function. Move the template to C++ only when a native command needs
   it.
4. **Should `Optimizer.step(step)` stay public?** Yes, in C++.
   The mechanism needs the step number, because the scheduler computes the
   multiplier from it. Remove the call from the primary Python path. The driver
   owns the counter for a normal run, and the Python `Optimizer` stays for a
   custom loop and for tests. One owner for the counter removes the drift
   between `TrainLoop::Run` and `Model::optimizer_step_`.
5. **Does the supervised fine-tuning driver need a thread for the reader?**
   Yes. `DataLoader` already owns a producer thread and a bounded buffer
   (`src/data.cc:274`). The packed-row loader is I/O-bound and fits the same
   design. One thread is enough, because Python did the packing.

### 11. References

- [python.md](python.md): the single Python surface.
- [post-training.md](post-training.md): supervised fine-tuning and
  reinforcement learning, and the status of record.
- [eval.md](eval.md): the evaluation seam and the "metric in Python, compute in
  C++" rule.
- [model.md](model.md): the model interface and the graphs.
- [../DESIGN.md](../DESIGN.md): section 2 for the layers, section 5 for the
  autograd policy, and section 7 for the evolution rules.


---

## Part 3. Native collectives over NVLink and RDMA

_Absorbed from the former `distributed-native-plan.md`. Its section numbers are
unchanged, so a reference to a section of that document resolves to the same
number here. The execution record is in [campaigns.md](campaigns.md)._

This document is the execution plan for a native all-reduce backend. The backend
replaces NCCL on the CUDA path. [distributed-design.md](distributed-design.md)
holds the current seam design.

Status: delivered through phase 2 on the local T4 host. The ring schedule, the
host-staged transport, the transfer-window pool, and `NativeGradientSync` are
in `backends/cuda/collective/`. The 2x T4 pair has no peer access, so the ring
stages through pinned host memory; the NVLink and RDMA transports await the
H100 node. Phases 3 to 6 remain.

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
the overlap the asynchronous seam promises (docs/distributed-design.md
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

The device target for this work is `--config=h100`, which is fp32. Stage 1 moves
the `h200` target to fp32 and Stage 2 moves it to bf16 (precision.md section 4).
The `h200` config is outside the collective path.

### 7. Phases

#### Phase 0 — Decision and scaffolding

Owner: architect, with the collective agent.

- Record the decision to drop NCCL as a hard dependency. Update
  [distributed-design.md](distributed-design.md) and
  [distributed-design.md](distributed-design.md).
- Add a build define, for example `--define=collective=native`. Keep `nccl` as
  the default until parity.
- Move the rendezvous into `src/rendezvous.{h,cc}`. Update `src/distributed.cc`
  to use it.
- Add the `backends/cuda/collective/` directory and its `BUILD.bazel`.
- Add the RDMA repository rule `third_party/rdma.bzl` for libibverbs, in the
  style of `third_party/nccl.bzl`.
- Add the collective workstream to `AGENTS.md` section 1 (section 8 below).

Gate: the tree builds under both `--define=collective=native` and
`--define=collective=nccl`.

#### Phase 1 — The schedule and the memory transport

Owner: collective agent, with the oracle agent for tests. This phase is CPU
only.

- Implement `RingAllReduceSum` against the `Transport` interface.
- Implement the memory transport. Reuse the shared-memory rendezvous from the
  NCCL mock.
- Add `//backends/cuda/collective:schedule_test`. It runs world sizes 1, 2, 4,
  and 8 against a reference sum.
- Add `//backends/cuda/collective:transport_select_test`. It checks the
  per-neighbor selection, including a mixed group.
- Use a relative tolerance. A ring adds in a different order than a single sum.
- Check that every rank ends with equal bytes.
- Add the schedule trace (section 10).

Gate: `tools/nanochat test --config=cpu` passes. The test needs no GPU.

#### Phase 2 — The NVLink transport

Owner: collective agent.

- Add the registered transfer window and the CUDA IPC handle exchange
  (section 5.4).
- Add `cudaDeviceCanAccessPeer` and `cudaDeviceEnablePeerAccess`.
- Add the two streams, the event dependencies, and the add kernel.
- Add the host-staged fallback for a failed peer access.
- Implement `NativeGradientSync`, the seam adapter, and `CreateGradientSync` in
  `backends/cuda/collective/native_sync.cu` (section 5.5).
- Add `//backends/cuda/collective:nvlink_test` with the `gpu` and `manual`
  tags. Two processes, two cards, tiny shapes.

Gate: the two-rank test passes and matches the host reference within tolerance.

#### Phase 3 — The RDMA transport

Owner: collective agent.

- Add the RDMA connection setup over the control plane.
- Register the transfer window for GPUDirect RDMA (section 5.4).
- Post one-sided RDMA writes with immediate, or two-sided send and receive. Poll
  the completion queue.
- Add `//backends/cuda/collective:rdma_test` with the `gpu` and `manual` tags.
  Two ranks over the adapter.
- Keep the NVLink transport as the first choice. Use RDMA when a peer is not
  reachable by NVLink.

Gate: the RDMA transport matches the host reference within tolerance. The
transport selection is correct for a mixed group.

#### Phase 4 — Overlap and performance

Owner: collective agent, with the harness agent.

- Tune the chunk count and the slot count per bucket size.
- Measure the bus bandwidth and the sync fraction. Use `tools/nanochat doctor`
  first, then the `t3-bench` profile.
- Add the small-message schedule only if the measurement asks.
- Keep the bucket plan in `src/train.cc` unchanged unless the measurement asks.

Gate: the sync fraction is below 0.05. Record the bus bandwidth with the commit
hash.

#### Phase 5 — Eight ranks on H100

Owner: collective agent, with the architect for the build.

- Build with `--config=h100`. Check the compile under `--config=sim-gate`.
- Run an 8-rank GPU test on the 8x H100 node.
- Run the T2 parity gate: an 8-rank loss curve against a 1-rank loss curve,
  within tolerance.

Gate: the 8-rank test and the T2 parity test pass.

#### Phase 6 — Make native the default and clean up

Owner: architect.

- Make `collective=native` the default for the `h100` config.
- Keep `nccl` selectable for the `t4` config during the transition.
- Remove `@nccl//:nccl` from the CUDA config after the transition.
- Update the design documents.
- Decide the fate of the NCCL mock in `tools/cuda_sim/collective/`.

Gate: the merge gate in `AGENTS.md` section 5.5 passes with NCCL absent from the
default CUDA build.

### 8. Ownership

The current map in `AGENTS.md` section 1 has no row for the collective. The
kernel agents own `backends/cuda/kernels/<family>.cu`. The map does not list an
owner for `device.cu` and `gemm.cu`. This work needs a clear owner.

Add one workstream to the map.

| Workstream | Owns | Depends on |
|---|---|---|
| **Collective** | `backends/cuda/collective/**`, `dev/collective/**` | frozen `kernels.h`, `src/rendezvous.h` |

Suggested file split.

| File | Owner |
|---|---|
| `backends/cuda/collective/transport.h` | Collective |
| `backends/cuda/collective/nvlink_transport.cu` | Collective |
| `backends/cuda/collective/rdma_transport.cc` | Collective |
| `backends/cuda/collective/ring_schedule.{h,cc}` | Collective |
| `backends/cuda/collective/native_sync.cu` | Collective |
| `backends/cuda/collective/BUILD.bazel` | Collective |
| `src/rendezvous.{h,cc}` | Workflow |
| `third_party/rdma.bzl` | Architect |
| `.bazelrc`, root `BUILD`, `MODULE.bazel` | Architect |
| `docs/**`, `AGENTS.md` | Architect |

`include/nanochat/kernels.h` and `include/nanochat/model.h` do not change. The
asynchronous seam already exists. Keep the add kernel backend-private.

The RDMA work needs a device and a driver that the WSL2 host does not have. The
collective agent develops the schedule on the central processing unit and the
transports on the H100 node.

### 9. Broker and launch

The local GPU queue serves one GTX 1080 Ti. It is not the
broker for an 8x H100 node. Treat the node as one resource.

- One `tools/nanochat dist` invocation holds one lock for the whole rank group.
  [distributed-design.md](distributed-design.md) already defines this. Keep that
  rule. Do not take one lock per rank.
- The lock covers all 8 cards. The `nvidia-smi` gate must see all 8 cards as
  idle before the group starts.
- Add a node-class sandbox profile to `tools/sandbox/profiles.json`, for example
  `node-train`. Size the memory, the central processing unit quota, and the
  process count for eight ranks on one node. Do not reuse the `train` profile,
  which serves the WSL2 host.
- One rank per card. Set `CUDA_VISIBLE_DEVICES` to the rank, as the NCCL path
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

The NCCL mock already has the pattern. `tools/cuda_sim/collective/nccl_mock.h`
exposes `AppendCollectiveRecord`, `ReadCollectiveLog`, and
`JudgeCollectiveRecords`. The deadlock case is already a test in
`tests/collective_sim_test.cc`.

Extend the pattern. The schedule emits one record per send, receive, and add.
Each record carries the rank, the direction, the transport kind, the step, the
chunk, the slot, and the event dependencies. A central processing unit test
with the memory transport replays the records and checks four rules.

1. The schedule uses every slot once at a time.
2. The dependency graph has no cycle.
3. Every wait has a matching completion.
4. A peer that never completes returns `kTimeout`, and the adapter stops.

The simulator agent owns the trace convention. The oracle agent owns the test.

### 11. Open questions and how to close them

Each question is either a discovery task or a decision.

- A discovery task has a command or a measurement. Run it, record the output,
  and close the question.
- A decision needs an owner and a deadline. Write a short record, then close the
  question.

#### 11.1 The RDMA adapter and GPUDirect RDMA

Type: discovery. Run the probes on the 8x H100 node.

- `ibv_devinfo -v` — the adapter list, the active speed, and the active width.
- `ibv_devices` and `rdma link show` — the device names and the link state.
- `nvidia-smi topo -m` — the GPU-to-adapter affinity. A `PIX` or `NODE` entry is
  good. A `SYS` entry crosses the central processing unit and is slow.
- `nvidia-smi nvlink -s` — the NVLink status and the per-link speed.
- `lspci | grep -i -E 'mellanox|infiniband|ethernet'` — the adapter vendor and
  model.
- `lsmod | grep -E 'nvidia_peermem|nv_peer_mem'` — the peer-memory module.

A list of adapters is not proof of GPUDirect RDMA. Write a small test that calls
`cudaMalloc`, then `ibv_reg_mr` on the device pointer, then frees the region. A
failure at `ibv_reg_mr` means no device registration. Put this test in the
collective directory and tag it `manual`.

Extend `tools/nanochat doctor` with an RDMA section. It already prints the GPU,
the driver, the sandbox, and the lock state. Add the adapter list, the link
speed, and the peer-memory module. Keep the `--json` shape.

Record the result in a new `docs/node-h100.md`, or extend
[host-portability.md](host-portability.md).

#### 11.2 RDMA as a fallback or as a required transport

Type: decision with a reversible design.

Make transport selection a runtime rule.

| Condition | Transport |
|---|---|
| Every peer pair passes `cudaDeviceCanAccessPeer` | NVLink |
| A peer is on another node, or peer access fails | RDMA |
| No RDMA adapter and no peer access | Host staging |

The rule is per edge, not per group. `RingTransports` applies it to each of the
two ring neighbors, so a mixed group works (section 3.2).

With this rule, the answer changes only the default preference and the tests.
The schedule does not change.

Ask the roadmap owner one question in writing: will a second node join within
the current plan horizon? If the answer is no, RDMA is a fallback and its
performance work can wait. If the answer is yes, RDMA is a first-class path and
every phase tests it.

Record the rule in [distributed-design.md](distributed-design.md) beside the
current section 5 table.

#### 11.3 Retire fp16, or only remove it from the collective

Type: decision, mostly closed.

[precision.md](precision.md) section 6 already accepts Stage 1. Stage 1 removes
fp16 from the tree, so every reduction is fp32, and the collective needs fp32
only.

Scope the collective to fp32. Stage 1 moves the `h200` target to fp32 and Stage
2 moves it to bf16 (precision.md section 4). Neither step blocks the collective.

#### 11.4 Node ownership and the broker

Type: discovery plus an access contract.

Ask the node owner four questions, in writing.

1. Is the node exclusive to this project, or shared?
2. Does a scheduler already serialize jobs on it, for example Slurm or
   Kubernetes?
3. What are the driver and toolkit versions?
4. Is a systemd user manager available for the sandbox?

Probe the node. Extend `tools/nanochat doctor` to report the card count, the
driver, the toolkit, the interconnect, and whether a scheduler is present.

Then pick one broker case, as section 9 describes. Record the access contract in
`docs/node-h100.md`, in the style of the Kaggle section in
[host-portability.md](host-portability.md).

#### 11.5 The schedule trace

Type: recommendation, accepted.

Build the trace in phase 1, as section 10 describes. It is cheap, and it removes
the largest correctness risk in the two-stream design.

### 12. Summary of open questions

| Question | Type | How to close | Owner | Blocks |
|---|---|---|---|---|
| RDMA adapter and GPUDirect | Discovery | `ibv_devinfo`, `nvidia-smi topo -m`, and an `ibv_reg_mr` probe | Collective | Phase 3 |
| RDMA fallback or required | Decision | One written roadmap answer, plus a reversible selection rule | Architect | Phase 3 default |
| Retire fp16 | Closed | precision.md section 6 Stage 1 removes fp16 from the tree; the `h200` target is fp32 until Stage 2 | Architect | Nothing |
| Node ownership and broker | Discovery and contract | Four written answers; a `doctor` extension; `docs/node-h100.md` | Architect | Phase 5 gates |
| Schedule trace | Recommendation | Reuse the mock's JSONL and coordinator pattern | Collective and Oracle | Phase 1 |

### 13. Test matrix

| Tier | Target | What it checks |
|---|---|---|
| T0 CPU | `//backends/cuda/collective:schedule_test` | The ring at world sizes 1, 2, 4, 8; the trace; the missing-peer timeout |
| T0 CPU | `//backends/cuda/collective:transport_select_test` | The per-neighbor selection, including a mixed group |
| T0 CPU | `//src:distributed_test` | The host reference, unchanged |
| T0 CPU | `//src:train_parallel_test` | The mean scale and the resume, unchanged |
| S0 | `//tests:collective_sim_test` | The NCCL mock, only while NCCL stays selectable |
| T1 GPU | `//backends/cuda/collective:nvlink_test` | A two-rank NVLink sum against the host reference |
| T1 GPU | `//backends/cuda/collective:rdma_test` | A two-rank RDMA sum against the host reference |
| T2 GPU | An 8-rank loss curve against 1 rank | The full training parity |

The schedule test is the primary new gate. It carries the algorithm. The
transport tests carry only data movement.

### 14. Risks and controls

| Risk | Effect | Control |
|---|---|---|
| A ring adds in a different order | The result differs from the reference by a few ulps | Use a relative tolerance in every test |
| Peer access is unavailable | The NVLink copy fails | Detect with `cudaDeviceCanAccessPeer`, then choose RDMA or host staging |
| GPUDirect RDMA needs a kernel module | The RDMA path registers no device memory | Check the peer-memory module at setup. Fail with a clear message |
| A rank fails during a collective | The job hangs | Bound every wait with the watchdog deadline (section 5.6). Test the timeout path |
| Per-step buffer registration | The NVLink path is slow or leaks mappings | Register a fixed window pool once (section 5.4). Grow only when `count` grows |
| Two streams deadlock each other | The pipeline stalls | Document the two event rules. Test with the schedule trace |
| The H100 node is not available locally | The T1 and T2 gates cannot run here | Use `--config=sim-gate` for the compile, and the node for the gates |
| The RDMA adapter topology changes | The tuned chunk count is wrong | Derive the chunk count from the measured bandwidth |
| The node broker is misconfigured | Concurrent jobs collide | One lock per group. Deploy the node broker separately |

### 15. Definition of Done

1. `tools/nanochat test --config=cpu` passes with the new schedule test.
2. The T1 NVLink test passes with two ranks.
3. The T1 RDMA test passes with two ranks.
4. The 8-rank T2 parity test passes on the 8x H100 node.
5. The sync fraction is below 0.05 for the target model.
6. The default CUDA build needs no NCCL.
7. The `h100` compile gate passes under `--config=sim-gate`.
8. `tools/nanochat lint` passes.
9. The design documents match the code.
