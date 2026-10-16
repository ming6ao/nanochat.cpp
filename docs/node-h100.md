# The H100 node: the d24 run book

Status: proposed. No H100 run has happened. This document is the run book that
`docs/distributed-native-plan.md` section 11 asks for, plus the acceptance
criteria for the depth-24 pretraining run.

This is a run book, not a design. The designs are in
[fp8-training.md](fp8-training.md), [attention-tensor-core.md](attention-tensor-core.md),
[performance-model.md](performance-model.md), and
[simulator-precision.md](simulator-precision.md).

## 1. Goal

Train one depth-24 model on eight H100 cards, and compare the result and the
step time against the reference.

The reference is the nanochat leaderboard entry for commit `a825e63`: a depth-24
model with fp8 training, a token-to-parameter ratio of 8, and an average of
1.65 hours over five runs.

## 2. The node

| Item | Value |
|---|---|
| Accelerator | 8 H100 SXM cards, 80 GiB each |
| Interconnect | NVLink 4.0 with NVSwitch, one node |
| Host toolkit | CUDA 12.0 or newer; record the exact version |
| Collective | NCCL, or the native ring with the NVLink transport |
| Node count | One |
| Second transport | An RDMA adapter with GPUDirect RDMA, if the native ring is used |

Prerequisites:

1. Phases 3 to 6 of [distributed-native-plan.md](distributed-native-plan.md),
   or a verified NCCL path.
2. The bf16 compute path from `docs/precision.md` Stage 2a.
3. The fp8 path from [fp8-training.md](fp8-training.md), if fp8 is used.
4. The acceptance criteria in section 8. Agree them before the run starts.

Record the `nvlink` view of the node with `nvidia-smi topo -m`. The
`docs/distributed-native-plan.md` section 11 discovery step owns the probe list.

## 3. The memory plan

Every number below is for depth 24, sequence 2048, batch 16 per card, and the
value embeddings on. The source is the model-fit engine, which mirrors
`BuildTrainWorkspace` in `src/model.cc` and `NanochatOptimizer::Build` in
`src/optim.cc`.

| Group | bf16 activations | fp8 activations |
|---|---:|---:|
| Parameters, gradients, and the fp32 master | 12.9 GiB | 11.6 GiB |
| The optimizer state | 7.8 GiB | 7.8 GiB |
| The training arena | 52.2 GiB | 26.1 GiB |
| The dense attention scratch | 4.5 GiB | 2.2 GiB |
| **Total** | **77.4 GiB** | **47.8 GiB** |

Two points are load bearing:

1. **The dense attention scratch is not in the arena.** The cuBLAS attention
   forward allocates a cached `[batch, heads, seq, kv_len]` buffer through
   `Alloc`. At this shape the request is three matrices, which is 4.5 GiB in
   bf16. The arena model does not count it. Size the card from the total row.
2. **The bf16 total leaves about 1 GiB of margin.** The card budget is 80 GiB
   minus a reserve of about 1.5 GiB for the CUDA context and the library
   workspaces. A bf16 run at batch 16 fits, and only just. Measure the free
   memory after the first step, before a long run.

The fp8 activation path removes the tight margin. It also allows batch 32, at a
total of 76.1 GiB. Prefer batch 32 with fp8 activations over batch 16 with bf16
activations, if the step time allows it.

The Track 3 preset turns the value embeddings off. That drops the parameter and
optimizer rows to 7.3 GiB and 3.3 GiB. The arena is unchanged. Use the preset
only if the model change is acceptable, because the value embeddings help the
reference result.

## 4. The batch and step plan

| Quantity | Value |
|---|---|
| Sequence length | 2048 |
| Device batch per card | 16 |
| Cards | 8 |
| Tokens per forward | 262,144 |
| Total batch | 1,048,576 |
| Gradient accumulation | 4 |

The total batch follows the reference formula in `python/nanochat_cpp/plan.py`:
`B_REF * (D / D_REF) ** 0.383`, rounded to a power of two. At depth 24 the
result is 1,048,576 for both horizons below.

| Token-to-parameter ratio | Tokens | Optimizer steps | Total arithmetic |
|---|---:|---:|---:|
| 8:1 | 5.84 billion | 5,570 | 2.79 times 10^19 FLOPs |
| 12:1 | 8.76 billion | 8,356 | 4.18 times 10^19 FLOPs |

The arithmetic count comes from `EstimateFlopsPerToken` in `src/train.cc`, which
returns 4.775 billion FLOPs per token for this configuration.

Use the 8:1 ratio for the GPT-2 comparison. The reference uses ratio 8 for the
same target, and a 12:1 run needs about 50 percent more time.

## 5. The data plan

The run needs 5.84 billion tokens at ratio 8. The reference uses NVIDIA ClimbMix
shards, and it downloads about 150 shards for a GPT-2 capability run.

1. Download the shards with the dataset helper.
2. Train the tokenizer on about 2 billion characters, and write one `NCTOKEN1`
   artifact. The vocabulary size is 32768.
3. Hold one validation shard for the bits-per-byte loop.
4. Measure the token count per shard, and record it. Do not assume the reference
   count.
5. Confirm the sustained input rate.

The input rate is the hard requirement. The reference trains at about 983
thousand tokens per second on the same node. The loader and the tokenizer must
sustain at least that rate across eight reading ranks, or the GPUs idle.

The native parquet reader matches `pyarrow` on real ClimbMix shards. See
[parquet-native.md](parquet-native.md). Measure the rate before the long run.

## 6. The launch

One process per rank, one broker lock for the group:

```bash
tools/nanochat dist --nproc-per-node 8 --profile train \
    --device 0,1,2,3,4,5,6,7 -- <train_main> \
    --layers 24 --heads 12 --kv-heads 12 --hidden 1536 --seq 2048 \
    --vocab 32768 --window-pattern SSSL \
    --batch 16 --grad-accum 4 --num-iterations 5570 \
    --train-parquet 'data/*.parquet' --val-parquet 'data/val/*.parquet' \
    --tokenizer tokenizer.nctoken \
    --device 'NVIDIA H100 80GB HBM3' \
    --save-every 1000 --eval-every 500 --log-every 10
```

`tools/nanochat dist` starts one process per rank. It takes one broker lock
for the group, applies the `train` sandbox profile, and sets the five rank
variables that `train_main` reads:

| Variable | Meaning |
|---|---|
| `NANOCHAT_WORLD_SIZE` | The group size |
| `NANOCHAT_RANK` | The global rank |
| `NANOCHAT_LOCAL_RANK` | The rank on this node |
| `NANOCHAT_MASTER_ADDR` | The rank 0 host |
| `NANOCHAT_MASTER_PORT` | The rank 0 port |

The contract is in [distributed-launch.md](distributed-launch.md). A flag
overrides a variable.

Run one step first, then check the free memory, then start the full run.

## 7. The broker and the sandbox on a native host

The sandbox profiles in `tools/sandbox/profiles.conf` are host-local. They were
written for a 12 vCPU and 7.7 GiB WSL2 host. The H100 node has a different CPU
and memory budget.

Before the run:

1. Re-derive the `train` profile for the node. The profile bounds the CPU, the
   host memory, the process count, and the wall clock. It does not bound the
   device memory.
2. Confirm the aggregate ceiling in `tools/sandbox/nanochat.slice`.
3. Treat the eight cards as one resource. Take one broker lock for the whole
   group, as `tools/nanochat dist` does.
4. Confirm that `tools/nanochat doctor` reports the device, the sandbox, and the
   lock state.

Do not run the training outside the entry point. Every executable calls
`nanochat::RequireSandboxOrDie` at startup, and it exits with a corrective
command.

## 8. The acceptance run

The acceptance run has four criteria. Agree them before the run.

| # | Criterion | Measure |
|---|---|---|
| 1 | The loss curve matches the reference shape | The bits-per-byte curve against the reference curve |
| 2 | The step time reaches the bound | The measured step against the bound in [performance-model.md](performance-model.md) |
| 3 | The capability reaches the target | The CORE score against the GPT-2 threshold of 0.256525 |
| 4 | The run is reproducible | Two short runs at the same seed give the same loss |

Criteria 1 and 2 need no capability evaluation, so check them early. Stop the
run if criterion 2 is far above the bound, because a long run at the wrong
rate wastes the node.

Run the pipeline in this order:

1. One step, to check the memory and the launch.
2. Fifty steps, to check the loss trajectory and the step time.
3. The full run.
4. The evaluation in section 9.

## 9. The evaluation

| Metric | Tool | Side |
|---|---|---|
| Bits per byte | The validation loop, on the validation shard | The C++ harness |
| CORE | `score_main` plus the Python reduction | Split |
| Samples | The generate path | The C++ harness |

The C++ side computes the logits. The Python side owns the task data and the
reduction. See [eval.md](eval.md).

Compare against the reference:

| Metric | Reference value |
|---|---|
| CORE, GPT-2 threshold | 0.256525 |
| CORE, nanochat commit `a825e63` | 0.2626 |
| Validation bits per byte, commit `a825e63` | 0.71800 |

The bits-per-byte value is comparable only on the same dataset and the same
tokenizer. Record both.

## 10. The run record

Record one row per run. The broker log and the sandbox log hold the rest.

| Field | Value |
|---|---|
| Date and revision | |
| Node and driver | |
| Toolkit version | |
| Build configuration | |
| Precision | |
| Token-to-parameter ratio | |
| Tokens and steps | |
| Wall-clock time and step time | |
| Peak device memory | |
| Validation bits per byte | |
| CORE score | |
| Prediction against measurement | |
| Deviations from this book | |

Store the record beside the checkpoint, or in this document. The record is the
only durable evidence of the run.

## 11. Risks

- **The bf16 memory margin is about 1 GiB.** Measure it at step one.
- **The data rate may be the limit.** A loader that delivers less than 983
  thousand tokens per second makes the GPUs wait.
- **The collective may not be ready.** Phases 3 to 6 of the native plan remain.
  NCCL is the fallback.
- **The fp8 accuracy may need a correction.** Keep a bf16 comparison run.
- **The sandbox profiles are host-local.** Re-derive them before the run.
- **The node is one resource.** A second job on the same node doubles the
  memory pressure and invalidates the timings.

## 12. Open questions

1. Which collective is the default for the first H100 run, NCCL or the native
   ring?
2. Does the run use the Track 3 preset, or keep the value embeddings?
3. Does the first run use ratio 8 or ratio 12?
4. Who owns the node during the run, and how is the broker coordinated?
5. Is one H100 node the target, or a second node later?

## See also

- [distributed-launch.md](distributed-launch.md) — the launch contract.
- [distributed-native-plan.md](distributed-native-plan.md) — the native ring.
- [fp8-training.md](fp8-training.md) — the fp8 path.
- [performance-model.md](performance-model.md) — the step-time bound.
- [eval.md](eval.md) — the evaluation bridge.
- [sandbox.md](sandbox.md) — the resource governance.
