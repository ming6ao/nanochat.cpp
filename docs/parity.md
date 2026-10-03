# Reference parity: known differences from PyTorch nanochat

`nanochat.cpp` reimplements the architecture and training recipe of the PyTorch
reference (`scripts/base_train.py`, `nanochat/gpt.py`, `nanochat/optim.py`,
`nanochat/dataloader.py`). This document tracks every known difference between
the two so that a divergence in a loss curve, a metric, or a parameter can be
attributed quickly. It is the companion to
[testing.md](testing.md) (how parity is gated) and
[python-bridge.md](python-bridge.md) (how the reference is driven).

Status values:

| Status | Meaning |
|---|---|
| `closed` | fixed; no remaining difference |
| `equivalent` | a different implementation of the same mathematics, up to floating-point order |
| `open` | a real behavioral difference that can change a loss curve or a metric |
| `out-of-scope` | intentionally not implemented on this host |

## Summary

| ID | Status | Area | Difference |
|---|---|---|---|
| D1 | `open` | data pipeline | contiguous random window instead of BOS-aligned best-fit packing |
| D2 | `equivalent` | attention | fused RMSNorm→RoPE order vs RoPE→RMSNorm |
| D3 | `equivalent` | attention | cuBLAS GEMM + softmax kernels vs SDPA math backend |
| D4 | `equivalent` | initialization | custom xorshift RNG vs the PyTorch RNG |
| D5 | `equivalent` | optimizer | gradient norm computed for logging with `--clip 0` |
| D6 | `out-of-scope` | scaling | no distributed data parallel training |
| D7 | `out-of-scope` | runtime | no `torch.compile`; disabled on Pascal anyway |
| D8 | `equivalent` | harness | no batch prefetch overlap during backward |

## How parity is verified

Two data fixtures pin the model and optimizer against the reference:

- `tests/debug_state.bin` (forward, backward, and a few optimizer steps) via
  `tools/dump_oracle.py`.
- `tests/data/train_parity.bin` (a full training trajectory: initial
  parameters, the exact token/target batch at every step, and the per-step
  loss, gradient norm, and parameter norms) via `tools/dump_train_fixture.py`.

Both fixtures **share the initial parameters and the exact batches** with the
C++ side, so they isolate the model, backward pass, optimizer, and schedules
from the data pipeline. That is deliberate: it means a failure points at the
graph, and it also means D1 is *not* covered by these gates.

```bash
tools/nanochat test //tests:oracle_test //tests:train_parity_test
tools/nanochat test --gpu //tests:oracle_cuda_test //tests:train_parity_cuda_test
```

## Open differences

### D1 — Data loader packing and ordering

**Status:** `open`. This is the only difference that changes the training data
distribution, so it is the reason a bridge loss curve does not match
`scripts.base_train` step for step.

**Reference** (`nanochat/dataloader.py`,
`tokenizing_distributed_data_loader_with_state_bos_bestfit`):

- BOS-aligned best-fit packing: every row starts with BOS; documents are placed
  largest-first; when nothing fits, the shortest buffered document is cropped
  to fill the row exactly. Row capacity is `T + 1`, so `inputs = row[:-1]` and
  `targets = row[1:]` are independent per row.
- Documents are iterated **sequentially** over parquet row groups, sharded
  across ranks and cycling epochs.
- Roughly 35% of tokens are discarded to cropping at `T = 2048` (higher at
  shorter `T`), which means more unique documents are consumed per step.

**nanochat.cpp** (`src/data.cc`, `DataLoader::Next`;
`python/nanochat_cpp/data.py` builds the stream):

- The tokenizer writes a flat, BOS-separated document stream.
- `DataLoader::Next` reads a **random contiguous window** of `B*T + 1` tokens
  (`start = rng % span`) and forms `tokens = buffer[0..N)` and
  `targets = buffer[1..N]`. There is no BOS alignment, no document packing, and
  no cropping.
- The flat shift couples adjacent rows: the last target of row `r` is the first
  token of row `r + 1`, whereas the reference packs each row independently.
- Rows are sampled with replacement over `total / rows` windows and the loader
  resets when exhausted, rather than scanning documents in order.

**Impact:** different token grouping and coverage; rows can begin mid-document;
BOS appears at arbitrary positions rather than at every row start. The model
and optimizer math are unaffected, but the loss curve and any metric computed
through the same loader (for example `EvalBpb`) are not directly comparable.

**Evidence:** `docs/python-bridge.md` already records the mismatch as a "Known
difference". The parity harness (above) shares batches, so it does not exercise
the loader.

**Fix direction:** port the best-fit packer into `src/data.cc` over the existing
BOS-separated stream. `python/nanochat_cpp/data.py` already emits documents in
order with a leading BOS, so only the row packer is missing; the tokenizer side
does not need to change. The header comment in
`include/nanochat/dataloader.h` used to claim best-fit packing while the
implementation did a contiguous window; it now states the actual behavior, and
this entry is the tracker for closing the gap.

## Equivalent differences

### D2 — QK-norm and RoPE order

The reference applies RoPE, then RMSNorm over `head_dim`, then a `1.2` scale
(`CausalSelfAttention.forward`). nanochat.cpp fuses the three as
RMSNorm→RoPE→scale in `QkPrepForward` (`src/ops.cc`,
`backends/cuda/kernels/qk_prep.cu`). The two orders are mathematically
identical: RoPE is an orthogonal rotation within `head_dim` and RMSNorm is a
scalar rescale, so `RMSNorm(R x) = R · RMSNorm(x)`. Only floating-point
rounding differs. See [kernels.md](kernels.md).

### D3 — Attention backend

The reference uses the SDPA math backend on Pascal (`nanochat/flash_attention.py`
falls back from Flash Attention 3, which needs SM 80+). nanochat.cpp computes
the scores, the probability-value product, and every backward product as
batched cuBLAS GEMMs, with the softmax and its gradient as custom row kernels
(`backends/cuda/kernels/attention.cu`). Both compute the same causal, optionally
sliding-window, grouped-query softmax attention with the same mask and the same
statistics contract; only the summation order differs. See
[performance.md](performance.md).

### D4 — Initialization RNG

The reference initializes with PyTorch's global RNG; nanochat.cpp uses a
xorshift64\* source in `src/model.cc` (`InitWeights`). The **distributions** are
identical (verified against `GPT.init_weights`): `wte` normal std 0.8, `lm_head`
normal std 0.001, attention projections uniform with standard deviation
`1/sqrt(hidden)`, `c_proj` zero, `mlp.c_fc` at 0.4× that scale, and the scalar,
smear-gate, value-embedding, and value-gate initializations all match. The
**concrete values** differ, so the same seed does not reproduce the reference
weights; the parity fixtures sidestep this by sharing the initial parameters.

### D5 — Gradient norm during the optimizer step

The reference does no gradient clipping in `scripts/base_train.py` and does not
compute a norm. nanochat.cpp's optimizer computes a global norm every step
(`kernels::GlobalNorm` with `clip = 0`) so the logger can report it. This is
numerically the identity (the clip scale is 1) and does not change the update;
it does add one device-to-host synchronization per step. It is not a
training-math difference.

### D8 — Batch prefetch overlap

The reference prefetches the next batch while the GPU runs the backward pass
(`x, y, ... = next(train_loader)` inside the accumulation loop). nanochat.cpp
loads the next batch at the top of the next step. This is a throughput
difference only; the sequence of batches presented to the model is the same.

## Out of scope

### D6 — Distributed data parallel training

The reference supports DDP (document sharding, gradient all-reduce, and
rank-agreeing `inf`/`nan` handling). nanochat.cpp is single-process. Adding DDP
is a scaling workstream, not a correctness gap for the single-GPU target.

### D7 — `torch.compile`

The reference compiles the model and the fused optimizer steps. On Pascal this
is disabled (`TORCH_COMPILE_DISABLE=1`; Triton needs SM 70+), so it has no
effect on the current host and is not a parity item here. On newer hardware it
would fuse elementwise work and change host dispatch cost, not the mathematics.

## Adding a difference

When you find a divergence, add a row to the summary table and a section with:
the reference location, the nanochat.cpp location, the observed impact, the
evidence (which gate does or does not cover it), and the fix direction. Prefer
`equivalent` only when the two forms are provably the same function; otherwise
mark it `open` until measured.
