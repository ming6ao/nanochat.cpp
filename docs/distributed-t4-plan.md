# Two-T4 Kaggle training plan

One plan for training the best-fit nanochat model on the two T4 cards of a
Kaggle session. The precision is fp32 on every device. The gradient sync is the
seam in `src/distributed.h`.

See [distributed-design.md](distributed-design.md) for the full design. See
[precision.md](precision.md) for the precision policy. See
[performance.md](performance.md) for the measurement method.

## 1. Goal

Train one model on two T4 cards in one Kaggle session. Each card owns one
process. The processes average their gradients at each step. The run must be
correct, must fit the 16 GiB card, and must scale.

## 2. The design

The design lives in [distributed-design.md](distributed-design.md). This
section summarizes the parts the plan changes.

### 2.1 The seam

`src/distributed.h` defines `GradientSync`. One method does the work:
`AllReduceSum(buffer, count)`. The model exposes every gradient through
`ParamView`. The seam needs only the `grad` pointer and the `count` field.

`src/distributed.cc` holds the host reference. Rank 0 binds a port. The other
ranks connect. Rank 0 receives each buffer, sums the buffers, and broadcasts the
result. A world size of 1 returns a no-op object.

### 2.2 The device staging

The CUDA backend keeps the model in device memory. The host reference reads and
writes host memory. So the sync stages each gradient through host memory:

1. Copy the gradient to a host buffer with `kernels::Memcpy(kDeviceToHost)`.
2. Reduce the host buffer over loopback TCP.
3. Copy the sum back with `kernels::Memcpy(kHostToDevice)`.

`DistributedConfig::device_buffers` selects the staged path. `src/train.cc` sets
it from `kernels::GetCaps().is_device`. The CPU backend reports false and pays
nothing. This staging is what makes a correct two-card run possible before NCCL
exists.

### 2.3 Document sharding

The loader shards documents by a stride. Rank `r` takes document `i` when
`i % world_size == r`. The rule is deterministic. It differs from the reference
contiguous split, so a two-rank step is not token-identical to a one-rank step.

### 2.4 The launch

`train_main` accepts `--preset track3`, `--rank`, `--world-size`, `--master`,
and `--port`. One process owns each card. `CUDA_VISIBLE_DEVICES` selects the
card, because the backend has no `cudaSetDevice` call.

### 2.5 Current state

| Piece | State |
|---|---|
| The gradient-sync seam | Done |
| The host reference | Done |
| The device staging | Done |
| `TrainLoop` reduction and sharding | Done |
| The launch flags | Done |
| The CPU two-rank gates | Done |
| The launch helper (`notebooks/parallel.py`) | Done |
| The model-fit skill (`.agents/skills/model-fit/`) | Done |
| The Kaggle notebook | Run on two cards (smoke and best-fit) |
| A phase timer in `TrainLoop` | Done (B1) |
| Gradient bucketing | Done (C1) |
| NCCL | Not started |
| Recompute and the fused classifier | Proposed |

A correct two-card run is possible today. The run is slow, because the host
reference moves each gradient over loopback TCP.

## 3. The fit

### 3.1 The method

The memory model is exact. It mirrors the source:

- The training arena comes from `BuildTrainWorkspace` in `src/model.cc`.
- The parameter counts come from `CountParams` in `src/train.cc`.
- The optimizer state comes from `NanochatOptimizer::Build` in `src/optim.cc`.

The arena needs about 1.1 MB for each token row in fp32. The tool reproduces
the committed 36 GB arena at `d12_s1024` batch 32 ([eval.md](eval.md)
section 2). The skill test `scripts/model_fit_test.py` pins that value.

The throughput model is an estimate. The T4 runs fp32 at 8.1 TFLOP/s
([precision.md](precision.md) section 3). The attention kernel uses no tensor
cores. The measurement phase replaces the estimates with data.

The tool is `.agents/skills/model-fit/scripts/model_fit.py`. One card holds
16 GiB. The tool keeps 14.5 GiB for the model.

The same tool answers the question for other hardware and topologies. Run it
with ``--hardware``, ``--devices``, and ``--interconnect``. See the skill at
`.agents/skills/model-fit/SKILL.md`.

### 3.2 The corners

The three goals conflict. Each corner is a separate best.

| Goal | Config | Parameters | Card | Est. tokens/s | Efficiency |
|---|---|---:|---:|---:|---:|
| Biggest | L12 C2560 H20 KV5 T1024 b1 | 994M | 14.3 GiB | 1.7k | 0.79 |
| Longest | L16 C1024 H8 KV2 T8192 b1 | 243M | 14.4 GiB | 3.6k | 0.99 |
| Fastest | L12 C768 H6 KV1 T1024 b13 | 123M | 13.6 GiB | 17.0k | 0.98 |
| Balanced | L16 C1024 H8 KV4 T4096 b1 | 252M | 9.1 GiB | 5.1k | 0.96 |

The tokens per second are estimates for the two cards together. They include
the gradient reduction over PCIe. The efficiency is the group rate over twice
the one-card rate.

### 3.3 The recommendation

**Best fit: `L16 C1024 H8 KV4 T4096 b1`.**

| Property | Value |
|---|---|
| Layers | 16 |
| Hidden | 1024 |
| Heads | 8, head dimension 128 |
| KV heads | 4, grouped-query ratio 2 to 1 |
| Sequence | 4096 |
| Micro-batch | 1 |
| Value embeddings | off |
| Precision | fp32 |
| Parameters | about 252M |
| Card memory | about 9.1 GiB |
| Estimated throughput | about 5.1k tokens/s |
| Scaling efficiency | about 0.96 |

This point has the best estimated throughput at 250M parameters or more and a
sequence of 4096 or more. It keeps about 5 GiB of headroom for the recompute
work and for a larger micro-batch.

Trade-offs:

- More parameters: `L12 C1536 H12 KV6 T4096 b1`. About 412M parameters, 11.8 GiB,
  3.6k tokens/s.
- Longer sequence: `L12 C1024 H8 KV8 T8192 b1`. About 218M parameters, 13.2 GiB,
  4.6k tokens/s.
- More throughput: `L12 C1024 H8 KV4 T2048 b4`. About 206M parameters,
  12.3 GiB, 8.7k tokens/s.

### 3.4 The `train_main` flags

The `track3` preset disables the value embeddings and sets the Track 3 optimizer
and schedule. The shape flags come after the preset, so they override the
baseline shape.

```text
--preset track3
--layers 16 --heads 8 --kv-heads 4 --hidden 1024 --seq 4096
--window-pattern SSSL
--batch 1 --grad-accum 64
--device t4
```

The `SSSL` window gives three of every four layers a quarter-length window. It
cuts the attention arithmetic at long sequences. Use `L` for full context and
the exact Track 3 recipe.

### 3.5 The measured run

The 500-step best fit on two T4 cards, one process per card, with the host
reference sync. The run used `--grad-accum 8` instead of 64, so the global batch
is 65536 tokens instead of the reference 524288; the architecture, the preset,
and the sequence are unchanged. A larger accumulation only amortizes the fixed
sync and optimizer cost further, so the per-step throughput barely moves.

| Property | Value |
|---|---|
| Steps | 500 |
| Exit status | 0 on both ranks |
| First loss | 10.40 (ln 32768) |
| Final loss | 3.59 (rank 0), 3.75 (rank 1) |
| Minimum loss | 3.54 |
| Validation bpb | 1.88 at step 50, 1.13 at step 450 |
| Throughput | 1214 tokens/s per rank, 2427 global |
| MFU | 0.249 (fp32 peak, 8.1 TFLOP/s) |
| Phase split | forward 26.4%, backward 63.2%, sync 4.4%, optimizer 5.9%, data 0.1% |
| Checkpoints | byte-identical on the two ranks |

The throughput is about 48 percent of the section 3.3 estimate (5.1k global).
The estimate assumes 80 percent GEMM efficiency and 25 percent attention
efficiency. A forward-only measurement at the same shape reaches 5411 tokens/s
per rank (3.0 TFLOP/s), so the gap is the fp32 GEMM and attention efficiency,
not the reduction. The sync is below the 0.1 target at every step. The
backward dominates the step, which is why phase D1 (recompute) and D2 (the
attention tile) carry the next throughput gain.

## 4. The notebook

`notebooks/nanochat-cpp-on-2x-t4.ipynb` is the target. It runs these steps:

1. Clone the repository and prepare the Kaggle host.
2. Build `//src:train_main` for sm_75 with `--config=t4`.
3. Fetch one ClimbMix training shard, the validation shard, and the tokenizer.
4. Run a small two-card smoke test at a four-layer model.
5. Compare the two checkpoint hashes. They must match.
6. Run the best-fit configuration.
7. Plot the loss and save the checkpoints.

The helper `notebooks/parallel.py` starts one process per card. It sets
`CUDA_VISIBLE_DEVICES` to the rank, writes one log for each rank, and returns
the parsed metrics. The model-fit skill computes the shape and the memory.

The target cell:

```python
import model_fit, parallel

plan = parallel.ParallelPlan(
    train_parquet=..., val_parquet=..., tokenizer=...,
    out_dir=..., batch=1, grad_accum=64,
    preset="track3",                        # value embeddings off
    flags=model_fit.shape_flags(model_fit.BEST_FIT_T4),
    log_every=1, eval_every=10, save_every=10)
result = parallel.launch(plan, root=REPO)
```

## 5. The execution plan

Each milestone delivers a result. Each phase has a gate.

### Milestone A — a correct two-card run

| Phase | Work | Files | Gate | Status |
|---|---|---|---|---|
| A0 | Stage a device gradient through host memory | `src/distributed.*`, `src/train.cc`, `src/distributed_test.cc` | `//src:distributed_test` | Done |
| A1 | The launch helper and the model-fit skill | `notebooks/parallel.py`, `.agents/skills/model-fit/` | `//notebooks:all` and the skill test | Done |
| A2 | A short two-card run on Kaggle | the notebook | The smoke run exits 0 and the checkpoints match | Done |

### Milestone B — the best-fit run

| Phase | Work | Files | Gate | Status |
|---|---|---|---|---|
| B1 | A phase timer and the baselines | `src/train.cc` | The two-card phase split | Done |
| B2 | The best fit on two cards | the notebook | The run completes at the target tokens/s | Done |

### Milestone C — throughput

| Phase | Work | Files | Gate | Status |
|---|---|---|---|---|
| C1 | Bucket the gradients | `src/train.cc`, `src/model.cc` | The round trips drop | Done |
| C2 | Overlap the sync with the backward | `src/model.cc`, `src/train.cc` | The sync fraction is below 0.1 | Open |
| C3 | NCCL | `backends/cuda/nccl_sync.cu`, BUILD | `//backends/cuda:nccl_sync_test` | Open |
| C4 | The data path | `src/data.cc`, the notebook | The data phase is below five percent | Open |

### Milestone D — scale

| Phase | Work | Files | Gate | Status |
|---|---|---|---|---|
| D1 | Recompute and the fused classifier | `src/model.cc` | `workspace_bytes()` fits and parity passes | Open |
| D2 | The attention tile | `backends/cuda/kernels/` | The head-dimension sweep flattens | Open |
| D3 | The larger or longer configuration | the notebook | The run completes | Open |

Run D1 and C1 in parallel. They touch different files.

### Phase detail

**A2 — the smoke run.** Use a four-layer model at 512 tokens. The two ranks
must write equal checkpoints. The hashes are the check. A mismatch points to a
broken reduce.

**B1 — the phase timer.** Time the data fetch, the forward, the backward, the
sync, the optimizer, and the evaluation. Emit the milliseconds in the step
record. Correct `tok/s` and `mfu` for the micro-batch count. The current record
reports one micro-batch.

**B2 — the best-fit run.** Use the configuration from section 3.3. Record the
revision, the build, and the device with each number. Compare against the
one-card baseline.

**C1 and C2 — bucketing and overlap.** Copy the gradients into four to eight
large buckets. Start a bucket reduction when the last gradient of the bucket is
ready. Run the reduction on a second stream. The optimizer waits on one event.

**C3 — NCCL.** Write `backends/cuda/nccl_sync.cu`. It selects the device from
the rank and calls `ncclAllReduce`. The two cards have no NVLink, so measure the
transport.

**D1 — recompute and the classifier.** Add the compile-time recompute level from
[optimizations.md](optimizations.md) part 1. Add the fused classifier from
part 2. The two changes raise the micro-batch and the launch efficiency.

## 6. Risks

| Risk | Control |
|---|---|
| The Kaggle session has one T4 | The notebook asserts two cards |
| The session ends mid-run | Save often. Keep the notebook version |
| The TCP sync dominates | Phase B1 measures the fraction. C1 to C3 fix it |
| NCCL is slow on Kaggle | Measure the transport at C3. Keep C1 and C2 |
| The micro-batch does not fit | The model-fit skill computes it. D1 raises the ceiling |
| The two checkpoints differ | The smoke run compares the hashes |
| A resumed run repeats data | The checkpoint stores no document index |
| The reference rates do not fit a 300M model | The rates scale with the hidden size. The horizon does not |

## 7. Open questions

1. Does Kaggle enable peer access between the two T4 cards?
2. Which phase pays first, memory or the sync? The B1 split answers this.
3. Does the reference recipe allow a different global batch?
4. Does the new model need a new schedule? The Track 3 rates scale with the
   hidden size, but the horizon does not.

## 8. Definition of done

1. The smoke run completes and the two checkpoint hashes match.
2. The phase timer reports the split.
3. The best-fit model runs at the target tokens per second.
4. The sync fraction is below 0.1.
5. NCCL passes the T1 gate.
6. The larger or longer configuration runs.

## 9. Reproduction

```bash
python3 .agents/skills/model-fit/scripts/model_fit.py \
    --hardware t4 --devices 2 --interconnect pcie3 --precision fp32
cd .agents/skills/model-fit/scripts
python3 -m unittest discover -s . -p 'model_fit_test.py'
```
