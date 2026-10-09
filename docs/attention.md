# Attention: baseline, Pascal feasibility, and the fused window

Attention is the largest CUDA kernel family. This document records the shipped
baseline, the Pascal feasibility study, the fused windowed-forward promotion,
and the measured results. Full-causal attention and every backward stay on
cuBLAS. The fused windowed forward lives in
`backends/cuda/kernels/attention.cu` behind `UseFusedAttention`.

Read [performance.md](performance.md) for the measurement protocol and
[kernels.md](kernels.md) for the `Attention` family.

This document replaces the former `attention-baseline.md`,
`flash-attention-pascal.md`, `flash-attention-window.md`, and
`attention-window-results.md`. The frozen reports stay at
[attention-baseline.json](attention-baseline.json) and
[attention-baseline-e2e.json](attention-baseline-e2e.json).

## Part 1. Phase 0 baseline

This document freezes the Phase 0 baseline for the shipped attention path. Use
it as the reference for the gates in Part 2.

The baseline has three files:

| File | Content |
|---|---|
| [attention-baseline.json](attention-baseline.json) | The `nanochat.bench.v1` report of `attention_benchmark`. |
| [attention-baseline-e2e.json](attention-baseline-e2e.json) | The forward-only eval report at `d8_s512` with the `SSSL` pattern. |
| Part 1 of this document | This note. |

### Measurement

- Device: GTX 1080 Ti, `sm_61`.
- Precision: fp32.
- Command: `tools/nanochat profile --json --out docs/attention-baseline.json`.
- Source: the measured table in
  Part 2, section 4.

### Issued work against necessary work

The shipped path runs dense cuBLAS GEMMs and masks the result afterwards. Thus
every query-key pair costs time, even a masked pair. "Issued work" counts all
dense pairs. "Necessary work" counts only the visible pairs. The ratio of the
two shows the mask waste.

| Case | Issued pairs | Necessary pairs | Issued / necessary |
|---|---|---|---|
| `d8_s512` full causal | 8,388,608 | 4,202,496 | 1.996 |
| `d8_s512` window 128 | 8,388,608 | 1,849,344 | 4.536 |
| GQA `B=2 T=512 H=8 KV=2` | 4,194,304 | 2,101,248 | 1.996 |
| `d8_s512` `SSSL` mean | 8,388,608 | 2,437,632 | 3.441 |
| head-dim sweep `d32` to `d256` | 131,072 | 66,048 | 1.984 |

Two readings:

- The window 128 layer needs 44% of the full-causal work, but it issues 4.54
  times its necessary work. The measured time is the same as the full layer
  (1.571 ms against 1.586 ms). This is the clearest defect.
- The `SSSL` pattern issues 3.44 times its necessary work. The pattern is three
  window layers to one full layer. The mean necessary work is 0.29 of the dense
  work.

### Forward overhead and the GEMM floor

At `d8_s512` full causal, the forward takes **1.586 ms** and the backward takes
**2.857 ms**.

The forward issues 4.29 GFLOP of dense GEMM work: 2.147 GFLOP for the query-key
product and 2.147 GFLOP for the probability-value product.

The measured product rates in commit `d253ef3` are 6,991 GFLOP/s for query-key
and 5,843 GFLOP/s for probability-value. Thus the two GEMMs take about 0.674 ms.
The remaining **0.91 ms** is transposes, softmax, and launches. The overhead is
57% of the forward.

The necessary forward work is 2.15 GFLOP.

At 4 TFLOP/s that is 0.54 ms. At 2.5 TFLOP/s that is 0.86 ms.

Both values beat 1.586 ms. The device has headroom.

### End-to-end baseline

`attention-baseline-e2e.json` records the forward-only eval at `d8_s512` with
the `SSSL` pattern. The shape is `B=8 T=512 L=8 H=4 KV=4 C=512 V=32768 W=SSSL`.

| Metric | Value |
|---|---|
| Forward time | 74.47 ms |
| Tokens per second | 55,000 |
| Eval arena | 755,269,888 bytes |

The Phase 3 gate in Part 2 compares the new eval
rate against the `tokens_per_second` value.

**Note.** The end-to-end value is an estimate. It scales the documented d12 fp32
eval rate in [performance.md](performance.md) section 6 to the `d8_s512` shape.
The arena size is exact. Replace the estimate with a real
`tools/nanochat bench` run at the first opportunity.

### Reproduction

Run the attention benchmark through the project entry point:

```bash
tools/nanochat profile --json --out /tmp/attention_p0.json
```

Run the end-to-end eval benchmark:

```bash
tools/nanochat build --config=cuda //src:eval_bench
tools/nanochat bench -- ./bazel-bin/src/eval_bench --batch 8 --seq 512 \
  --layers 8 --heads 4 --kv-heads 4 --hidden 512 \
  --window-pattern SSSL --json
```

### Limits

- The numbers are from one host and one precision. Do not compare across
  devices.
- The baseline is a snapshot. Update it only with the revision and the
  configuration that produced the new numbers.

## Part 2. Pascal feasibility and plan

This document answers a single question: **can a flash-attention-style kernel be
implemented in `nanochat.cpp` for the Pascal host?** **Would it be faster than
the current attention path?** It records the investigation, the measured
baseline, the feasibility judgment, and a staged plan with go/no-go gates.

Status: delivered for the windowed forward. The fused windowed forward runs in
`backends/cuda/kernels/attention.cu` behind `UseFusedAttention`. Full-causal
attention and every backward stay on cuBLAS. The project removed the Phase 1
prototype after promotion.

Related: [performance.md](performance.md) (measurement protocol, the attention
benchmark), [kernels.md](kernels.md) (the `Attention` family), [DESIGN.md](../DESIGN.md)
section 4 (Pascal constraints, no megakernels), [parity.md](parity.md) (D3,
attention parity).

---

### 1. Summary

1. **The upstream `flash-attention` kernels cannot run on Pascal.** Every
   compiled kernel in the checkout is `*_sm80.cu`; `setup.py` only emits
   `compute_80` and newer. They are fp16/bf16 tensor-core kernels that use
   `mma.sync`, `ldmatrix`, and `cp.async`. Pascal sm_61 has none of these, and
   GeForce GP102 runs fp16 arithmetic at roughly 1/64 of fp32. There is no
   fallback path to port; FA1 was also tensor-core based.

2. **What is worth porting is the algorithm, not the kernels.** The tiled
   online softmax, the causal/window tile skip, the recompute backward, and the
   GQA handling are all architecture-neutral. On Pascal they must use fp32 against
   CUDA cores and shared memory.

3. **The current path leaves two large, measurable wins on the table.** It runs
   the query-key and probability-value products as *dense* cuBLAS GEMMs and
   masks afterwards. So a sliding-window layer costs exactly the same as a full
   causal layer: measured 1.571 ms versus 1.586 ms forward, even though the
   window needs only 44% of the work.

   About 0.9 ms of the 1.586 ms forward is
   transpose, softmax, and launch overhead around only ~0.67 ms of GEMM. A
   properly tiled kernel can skip masked tiles and fold the softmax and the
   layout conversion into the GEMM.

4. **The win is plausible but not guaranteed, so the plan must gate it.** A prior
   experiment in this tree (the stale `attention_variant_bench` artifact)
   implemented a one-block-per-query-row tile ("score-caching v2"). It was
   ~7x *slower* than today's batched cuBLAS path (11.57 ms versus 1.59 ms
   forward). A fused kernel only wins if it is a **register-blocked tiled GEMM**
   that reuses each key/value tile across many query rows. The plan therefore
   makes a forward prototype and a backward prototype explicit kill gates.

5. **Recommendation.** Pursue it, with the gates in section 6. Expected
   outcome if the gates pass: roughly **2-3x on attention** at full context and
   roughly **2.3x on the windowed layers** that make up three quarters of the
   production `SSSL` model. At the current `d8_s512` shape attention is about
   11% of the step. That is an estimated **6-8% end-to-end step time** there, and
   it grows with sequence length as attention's share grows. If the forward
   prototype cannot beat cuBLAS at `d8_s512`, stop and keep the current path.

---

### 2. Why the upstream flash-attention cannot run on Pascal

The reference Python `nanochat` (`/home/egrader/repos/nanochat`) imports
`nanochat.flash_attention`, which tries to load FlashAttention-3 and otherwise
falls back to `torch.nn.functional.scaled_dot_product_attention`. The FA3
loader requires a device major version of 8 or 9. On the GTX 1080 Ti it is not
loaded, and SDPA falls back to its math backend, which materializes the full
attention matrix. So even the Python reference does not use a fused attention
kernel on Pascal.

The `flash-attention` checkout is the current upstream master (FA2/FA3/FA4).
The concrete blockers:

| Upstream requirement | Pascal sm_61 reality |
|---|---|
| `mma.sync.aligned.m16n8k16` (sm_80) | No tensor cores, no `mma` on sm_61. |
| `ldmatrix` (sm_75+) | Not available. |
| `cp.async` global→shared (sm_80) | Not available. |
| fp16/bf16 storage and math | GP102 fp16 is ~1/64 of fp32; unusable. |
| `setup.py` arch list `80;90;100;110;120` | No sm_61 or sm_75 code at all. |
| All source files are `*_sm80.cu` | Nothing to compile for sm_61. |

Full-fused, IO-aware attention is not a tensor-core-only idea, however. The
kernel just needs enough shared memory, registers, and fp32 FMA throughput.
Pascal has 48 KB of shared memory per block (96 KB per streaming
multiprocessor), 64K 32-bit registers per streaming multiprocessor, and 11.34
TFLOP/s of fp32.

That is enough to hold a tiled online softmax. What it does not have is the
async copy pipeline that hides global latency. So the kernel has to be
compute-bound (or rely on occupancy) to win.

---

### 3. What the current CUDA attention actually does

`backends/cuda/kernels/attention.cu`, at commit `d253ef3`.

Forward, multi-head path:

1. transpose `q`, `k`, `v` from `[batch, seq, heads, dim]` into contiguous
   `[batch, heads, seq, dim]` scratch;
2. one batched cuBLAS GEMM `scores = Q Kᵀ * scale` over all heads;
3. a row-per-block softmax kernel that writes the normalized probabilities and
   the `(max, sum_exp)` statistics over the *full* `seq x seq` matrix;
4. one batched cuBLAS GEMM `out = P V`;
5. transpose the output back.

Backward is the same decomposition with four batched GEMMs (`dP = dO Vᵀ`,
`dQ = dS K`, `dK = dSᵀ Q`, `dV = Pᵀ dO`), a softmax-gradient kernel, and
transposes. Grouped-query attention falls back to one strided GEMM per
`(batch, head)`.

The path is a faithful "GEMM plus a row kernel" decomposition, the same shape
as the PyTorch math backend. Its costs are structural:

- the softmax kernels materialize the `seq x seq` score, probability, and
  gradient matrices in HBM and stream them several times;
- the kernels compute every tile of the matrix, then mask it, so causality and
  the sliding window save nothing;
- the four layout transposes and the separate softmax kernel are pure overhead
  around the GEMMs.

---

### 4. Measured baseline and roofline

Reproduced with `tools/nanochat profile --json` (the shipped
`attention_benchmark`), GTX 1080 Ti, fp32, sm_61, `-O2`, at `d8_s512`
(`batch 8, seq 512, heads 4, kv-heads 4, head_dim 128`). "Necessary" is the
operation count over only the visible key/value pairs, which is the honest
denominator for a fused kernel.

| Case | Forward | Backward |
|---|---|---|
| `d8_s512` full causal | 1.586 ms / 1,357 GFLOP/s | 2.857 ms / 1,506 GFLOP/s |
| `d8_s512` window 128 | 1.571 ms / 603 GFLOP/s | 2.806 ms / 675 GFLOP/s |
| GQA `B=2 T=512 H=8 KV=2 D=128` | 1.312 ms / 820 GFLOP/s | 3.155 ms / 682 GFLOP/s |
| head-dim sweep `d32/64/128/256` (fwd) | 0.068 / 0.092 / 0.112 / 0.164 ms | — |

Key readings:

- **The sliding window is free only in theory.** Window 128 needs 44% of the
  full-causal work (57,792 versus 131,328 visible pairs) but takes the same
  1.57 ms, because the GEMMs are dense. This is the single clearest defect and
  the one a fused kernel fixes outright.
- **The path runs at ~2.7 TFLOP/s forward and ~3.0 TFLOP/s backward** against
  dense operation counts, versus 11.34 TFLOP/s peak and ~6-7 TFLOP/s for a
  well-shaped batched cuBLAS GEMM on this device.
- **Overhead dominates the GEMM at this shape.** From the measured per-product
  rates in commit `d253ef3` (batched QK 6,991 GFLOP/s, PV 5,843 GFLOP/s), the
  forward GEMMs take about 2.147/6.991 + 2.147/5.843 = **0.674 ms**.

  The remaining **~0.91 ms** (57%) is transposes, softmax, and launches. The
  backward has the same shape of overhead around four GEMMs.
- **The device has headroom.** Forward necessary work is 2.15 GFLOP. At 4
  TFLOP/s that is 0.54 ms; at 2.5 TFLOP/s it is 0.86 ms. Both beat 1.586 ms;
  the question is only whether a hand-written fp32 tile can reach that.

Why the model shape matters: the production `Config` default is
`window_pattern = "SSSL"`, so the model windows three of every four layers (the
parity harness pins `L`). The average fraction of necessary attention work
across the `SSSL` pattern is `(3 * 0.44 + 1 * 0.50) / 4 = 0.455` of the dense
work. The fraction is roughly constant in sequence length (`w = seq/4`). The
opportunity does not shrink as context grows; it grows in absolute terms
because attention itself grows quadratically.

---

### 5. What a Pascal-native flash attention would look like

#### 5.1 The algorithm to port

This is the FlashAttention-2 algorithm with the MMA replaced by fp32 CUDA-core
outer products:

- **Forward.** One thread block owns a tile of `Br` query rows for one
  `(batch, head)`. It loops over key/value tiles of `Bc` rows. For each tile it
  computes `S = Q Kᵀ` and applies the causal/window mask. It updates a running
  row maximum and running denominator in the online-softmax style, and it
  accumulates `O` in registers. The kernel writes `(row_max, sum_exp)` once per
  query row, exactly as today. The kernel skips fully masked key tiles before
  the dot product.
- **Tile skipping.** For a query tile `[t0, t0+Br)` and a key tile
  `[j0, j0+Bc)`, the tile is entirely masked when
  `j0 > qpos_max` (fully causal) or when
  `j0 + Bc - 1 < qpos_min - window_left` (window) or the symmetric
  `window_right` condition. This is the source of the window win.
- **Backward.** The FlashAttention-2 backward: recompute `S` and
  `P = exp(S - row_max)` from the saved statistics, form `dP = dO Vᵀ`, apply
  `dS = P ∘ (dP - rowsum(P ∘ dP))`, then `dQ = dS K`, `dK = dSᵀ Q`, and
  `dV = Pᵀ dO`. The key/value gradients accumulate across the query tiles that
  share a key/value head (GQA). The kernel needs either atomic adds or a group
  reduction, exactly as the current grouped-query fallback already does.
- **Recompute beats storing `P` here.** Avoiding a 4-byte store and a 4-byte
  load of `P` saves 8 bytes per element; recomputing `S` costs `2 * head_dim`
  flops, so 256 flops per 8 bytes = 32 flop/byte. The device ridge point is
  11.34 TFLOP/s / 484 GB/s ≈ 23 flop/byte, so recomputation is on the right
  side of the roofline. This is the same trade the current path does *not*
  make.

#### 5.2 Pascal constraints that shape the tile

- No `cp.async`: global→shared copies cannot overlap the math without
  double-buffering, which doubles shared memory. The kernel should be
  compute-bound and use `__ldg`/`__ldcs` where a streaming hint helps.
- Shared memory: keep the block at or under 48 KB so two blocks can be resident
  per streaming multiprocessor (96 KB total) without opt-in. A candidate for
  `head_dim = 128` (the only shape the model uses) is
  `Br = 32` queries, `Bc = 32` keys, and block 256 threads. The tile uses
  `Q` 16 KB + `K` 16 KB + `V` 16 KB = 48 KB. The kernel holds `S` and `O` in
  registers (`O` is `32 x 128 / 256 = 16` registers per thread). Search the tile
  sizes; do not assume them.
- No tensor cores and fp16 at 1/64 rate: stay in fp32. `expf` matches the CPU
  reference; revisit `__expf` only if the oracle tolerance proves it safe.
- No `ldmatrix`: register blocking must use plain shared-memory loads, so the
  microkernel is the low-level risk.

#### 5.3 Why the naive version fails

The stale `attention_variant_bench` binary in the Bazel output base is a
reminder. It benchmarked the shipped block-per-query-row tile
(`OnlineSoftmaxTile`, one block per query row, all keys scanned per row) against
the then-shipped GEMM path:

| Variant (`d8_s512`) | Forward | Backward |
|---|---|---|
| shipped at the time (per-head GEMM) | 12.88 ms | 30.26 ms |
| "score-caching v2" tile | 11.57 ms | 30.09 ms |

Neither is competitive with today's batched path (1.59 ms / 2.86 ms). The tile
has no key/value reuse across query rows, so its traffic is `O(seq^2 * head_dim)`
and it is bandwidth-bound. This is exactly the failure mode the plan must avoid.
It is also the reason the existing `OnlineSoftmaxTile` is appropriate for
batch-1 decode and not for training.

---

### 6. Plan

Work proceeds in `dev/kernels` (the `llm.c` `dev/cuda` convention) and is only
promoted into `backends/cuda/kernels/attention.cu` after each gate passes.
The frozen seam (`include/nanochat/kernels.h`) does **not** need to change:
`AttentionForward` and `AttentionBackward` keep their signatures, and the
backend may select a fused kernel internally by shape and capability.

Every phase runs through `tools/nanochat`; benchmarks under the broker, tests
under the appropriate tier. See [performance.md](performance.md) and
[testing.md](testing.md).

#### Phase 0 — bound the opportunity (0.5 session)

- Extend `attention_benchmark` with a "necessary versus issued work" column so the
  mask waste is visible in the report, and add the `SSSL` mixed-pattern shape.
- Record a committed baseline JSON of the current numbers.
- Deliverable: the baseline file plus a one-page roofline note.
- **Gate:** none. This is measurement.

#### Phase 1 — fused fp32 forward prototype (1-2 sessions, the critical gate)

- A `dev/kernels` prototype implementing the tiled
  forward: causal, left/right window, MHA and GQA, empty-window contract.
- Correctness first: compare against `backends/cuda/kernels/testing/sequence_ref.h` at
  `B=2, T=8, head_dim=4` (tiny) and against the shipped path at `d8_s512`;
  finite-difference is not needed for the forward.
- Then a tile-size sweep (`Br`, `Bc`, block size, `__launch_bounds__`) using
  `bench_utils.h`, and the head-dimension sweep to confirm no per-dimension
  redundancy.
- **Gate:** at `d8_s512` full causal, forward must beat the shipped 1.586 ms,
  and window 128 must be at least 1.6x faster than full causal. If the best
  tile cannot beat cuBLAS, stop: report the negative result and keep the
  shipped path. This is the kill criterion for the whole effort.

**Phase 1 result.** The prototype passed the window condition and failed the
full-causal condition. The project removed the prototype after it promoted the
windowed forward.

| Case | Tile (ms) | cuBLAS (ms) | Speedup |
|---|---|---|---|
| full causal | 1.980 | 1.586 | 0.80x |
| window 128 | 1.078 | 1.571 | 1.46x |
| grouped query | 1.059 | 1.312 | 1.24x |

At `head_dim = 128` the 48 KB shared-memory cap admits only `Bc = 32`. A sweep
found no axis that closes the gap. The project kept the default tile and did
not promote the full-causal case.

#### Phase 2 — fused fp32 backward prototype (2-2.5 sessions)

- The prototype backward using the recompute scheme, with atomic
  key/value gradient accumulation for GQA (the existing `AtomicAddDev` handles
  the sm_61 fp16 case; fp32 uses the native atomic).
- Correctness: the existing `attention_test.cc` shapes plus finite differences
  at fp32, and a direct comparison against `sequence_ref`.
- **Gate:** at `d8_s512` full causal, backward must beat the shipped 2.857 ms,
  across MHA, GQA, sliding window, and the KV-cache offset case. If only the
  forward wins, consider promoting the forward alone (a partial win) before
  deciding on the backward.

#### Phase 3 — promotion, dispatch, and end-to-end (1-2 sessions)

- Move the validated kernels into `backends/cuda/kernels/attention.cu` behind a
  `UseFusedAttention(params)` predicate. The kernel keeps the cuBLAS path for
  shapes the fused kernel does not cover. Those shapes include a `head_dim`
  other than 128 and very short sequences, where the batched GEMM is already
  launch-bound.
- Keep the seam byte-for-byte; no header edit.
- **Gates (all must pass):**
  - `tools/nanochat test --gpu //backends/cuda/kernels:attention_gpu_test`
  - `tools/nanochat test --gpu //tests:oracle_cuda_test //tests:train_parity_cuda_test`
    (strict: trajectory loss 1e-5, parameters 1e-4)
  - `tools/nanochat test` (full CPU suite) and the default CUDA build
  - `tools/nanochat profile --json` beats the Phase 0 baseline
  - a training run at `d8_s512` (and, if available, a longer-context run)
    shows an end-to-end tokens/s improvement, not just a kernel improvement
- **Gate:** the end-to-end step must improve; a kernel win that does not move
  tokens/s is not a win.

#### Phase 4 — optional, separate (Turing)

- On sm_75 a real fp16 tensor-core flash attention (adapted from the upstream
  sm_75 forks) is possible, but there is no Turing device on this host, so it is
  compile-only and the project cannot benchmark it here. Keep it out of this effort; the
  DESIGN already defers the decode/MMA path to Turing.

---

### 7. Risks and mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| Hand-written fp32 GEMM is slower than cuBLAS | cuBLAS is near-peak and per-shape tuned; a naive tile is 7x slower (section 5.3) | Register-blocked outer-product microkernel; tile sweep; Phase 1 kill gate. |
| Shared-memory pressure at `head_dim = 128` | 48 KB/block caps the tile and occupancy | Hold `S`/`O` in registers; keep `Q`,`K`,`V` in shared; search tile sizes; fall back to splitting `head_dim`. |
| Numerics drift from the online-softmax rescaling | The task has a strict parity gate | Use `expf`; promote only after `train_parity_cuda_test` passes; keep the cuBLAS path as the default until then. |
| Window/GQA corner cases | Empty windows, `window_right`, KV-cache offset | Reuse `sequence_ref.h` and the existing `attention_test.cc` cases; add the `SSSL` shape. |
| Attention is only ~11% of the step | A 2x attention win is ~6% end-to-end at `d8_s512` | Set expectations by the tokens/s gate; the payoff grows with context length, so measure at the longest available shape. |
| Build/GPU contention | One small GPU shared by agents | All runs go through `tools/nanochat`; prototypes stay tiny; benchmarks under the broker. |

---

### 8. Alternatives considered and rejected

- **Use the upstream kernels unchanged.** Impossible on sm_61 (section 2).
- **Port an FA1-era sm_75 path.** Also tensor-core based; there is no CUDA-core
  flash path upstream to port.
- **Keep cuBLAS and add a mask-skipping wrapper.** cuBLAS has no causal or
  banded mask, and per-tile GEMMs would be launch-bound. Rejected.
- **Fuse only the softmax.** The scores are still computed densely, so the
  window win is lost; the overhead win is small. Rejected as a primary path.
- **Use the existing `OnlineSoftmaxTile` for training.** It is the
  one-block-per-row schedule that measured 7x slower (section 5.3). It stays a
  decode-path helper.
- **Enable `torch.compile` or another framework for the reference.** Pascal is
  unsupported by Triton, and this project has no framework dependency
  (DESIGN.md section 6).

---

### 9. References

- Upstream implementation: `/home/egrader/repos/flash-attention`
  (`csrc/flash_attn/src/*_sm80.cu`, `setup.py` arch list).
- Reference Python attention and its SDPA fallback:
  `/home/egrader/repos/nanochat/nanochat/flash_attention.py`.
- FP32 tiled attention reference (correct but slow):
  `/home/egrader/repos/llm.c/dev/cuda/attention_forward.cu`
  (`attention_forward_kernel2`).
- Current attention kernel:
  `backends/cuda/kernels/attention.cu`; shared helpers:
  `backends/cuda/kernels/device_utils.cuh`.
- Benchmarks and references:
  `backends/cuda/kernels/attention_benchmark.cc`, `backends/cuda/kernels/attention_test.cc`,
  `backends/cuda/kernels/testing/sequence_ref.h`.
- Measurement protocol: [performance.md](performance.md).

## Part 3. Windowed-forward promotion

Status: promoted. Owner: architect/integrator.

Promotion results: Part 4.

### 1. Why reopen

The Phase 1 kill gate failed for the full-causal forward. The fused tile
measured 1.98 ms at `d8_s512` full causal against the shipped 1.586 ms. The
tile space cannot close the gap. At `head_dim` 128, the 48 KB shared-memory
cap admits only `Bc=32`.

The same prototype won on the windowed shape. At window 128 it measured
1.078 ms against the shipped 1.571 ms. That is a 1.46x win.

The production model uses the `SSSL` pattern. Three of every four layers use a
window. The window path is worth promoting on its own.

This part scopes the reopened effort. The technical background, the
baseline, and the prototype are in Part 1 and Part 2.

### 2. Goal

Promote the fused fp32 tiled forward for windowed attention shapes. Keep the
cuBLAS path for full-causal attention and for every shape the fused kernel
does not cover. Show an end-to-end tokens per second win on the `SSSL` model.

### 3. Scope

In scope:

- Promote the fused forward into `backends/cuda/kernels/attention.cu`.
- Add a `UseFusedAttention(params)` predicate. Select the fused kernel only
  for `window_left >= 0` and `head_dim == 128`.
- Keep the kernel seam byte for byte. Do not edit `include/nanochat/kernels.h`.
- Keep the cuBLAS forward for full-causal attention and for all other shapes.
- Measure the production `SSSL` shape, not only window 128.

Out of scope:

- The fused backward. The prototype has no backward. The cuBLAS backward
  stays.
- A full-causal improvement. That needs a different tile, for example a split
  `head_dim`. It is a separate effort.

### 4. Phases

#### Phase A — promotion and dispatch

Move the validated forward from the `dev/kernels` prototype into
`backends/cuda/kernels/attention.cu`. Add the dispatch predicate. Keep the
cuBLAS path for the uncovered shapes. Update
`backends/cuda/kernels/BUILD.bazel` only when the dependency set changes.

Gate: the CUDA build succeeds and the existing attention tests stay green.

#### Phase B — correctness and parity

Run the oracle and the training trajectory on the promoted path.

Gate: `tests:oracle_cuda_test` and `tests:train_parity_cuda_test` pass with
the strict tolerances. The CPU parity tests stay green.

#### Phase C — window performance

Re-run the shipped attention benchmark at the production `SSSL` shape and at
window 128.

Gate: the windowed forward row beats the Phase 0 baseline. The full-causal
row stays on the cuBLAS path and must not regress.

#### Phase D — end-to-end

Measure the end-to-end effect on the `SSSL` eval benchmark.

Gate: the eval tokens per second beats `docs/attention-baseline-e2e.json`.

### 5. Risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| The window win shrinks at the production window | The window-128 win may not hold at window 512 | Measure the production shape in Phase C |
| Attention is a small part of the step | A window win moves end-to-end tokens per second little | Set the gate on tokens per second, not the kernel |
| A wrong dispatch predicate | The fused kernel can be selected for a losing shape | Keep the predicate narrow and test the full-causal path |
| Numerics drift | The task has a strict parity gate | Promote only after the parity gate passes |

## Part 4. Promotion results

This note records the promotion verdict for the fused windowed attention
forward. The campaign is `flash-attn-window`.

- Design: Part 3
- Frozen reference: [attention-baseline.json](attention-baseline.json)
- Frozen end-to-end reference: [attention-baseline-e2e.json](attention-baseline-e2e.json)

### Verdict

The windowed forward passes every gate. The campaign promotes it. Full-causal
attention stays on cuBLAS. Every backward stays on cuBLAS.

### Dispatch predicate

The predicate lives in `backends/cuda/kernels/attention.cu`:

```c
bool UseFusedAttention(const AttentionParams& p) {
  return p.window_left >= 0 && p.head_dim == 128;
}
```

The fused tile covers two conditions. The first condition is
`window_left >= 0`, which is a sliding-window shape. The second condition is
`head_dim == 128`. Full-causal attention has `window_left < 0`, so it stays on
cuBLAS. Every other head dimension stays on cuBLAS. Every backward call stays
on cuBLAS.

The kernel seam does not change. `include/nanochat/kernels.h` stays byte for
byte.

### Phase B — parity

The strict parity gates pass. Table 1 gives the measured error against the
recorded fixture.

| Test | Metric | Measured | Tolerance | Verdict |
|---|---|---|---|---|
| `tests:oracle_cuda_test` | Maximum forward error | 3.73e-09 | 1e-5 | pass |
| `tests:oracle_cuda_test` | Maximum backward error | 5.96e-08 | 1e-5 | pass |
| `tests:oracle_cuda_test` | Maximum optimizer step 1 error | 2.38e-07 | 1e-5 | pass |
| `tests:oracle_cuda_test` | Maximum trajectory loss error | 2.38e-07 | 1e-5 | pass |
| `tests:oracle_cuda_test` | Maximum trajectory parameter error | 4.84e-05 | 1e-4 | pass |

**Table 1.** Strict oracle parity on the promoted path.

The training-trajectory test also passes. Its measured errors are: loss
`4.77e-07`, parameter L2 `1.39e-04`, gradient L2 `5.39e-07`, and gradient norm
`1.19e-07`. The full CPU suite passes 16 of 16 tests.

The `SSSL` dispatch check confirms the coverage. Three of the four layers take
the fused branch. The last layer stays on cuBLAS. A `head_dim` 128 windowed
case compares the fused forward against the cuBLAS forward.

### Phase C — window performance

The measurement command is
`tools/nanochat profile --json --out /tmp/attention-window.json`. The command
runs the shipped `attention_benchmark`. Table 2 compares the promoted path against
the frozen `docs/attention-baseline.json`.

| Row | Frozen baseline (ms) | Promoted path (ms) | Speedup | Verdict |
|---|---|---|---|---|
| `attention_fwd:d8_s512_win128` | 1.571 | 1.087488 | 1.44x | pass |
| `attention_fwd:d8_s512_sssl` | 1.57475 | 1.208891 | 1.30x | pass |
| `attention_fwd:d8_s512` | 1.586 | 1.577643 | 1.01x | pass, cuBLAS |
| `attention_bwd:d8_s512_win128` | 2.806 | 2.808224 | 1.00x | pass, cuBLAS |
| `attention_bwd:d8_s512_sssl` | 2.81875 | 2.815197 | 1.00x | pass, cuBLAS |
| `attention_bwd:d8_s512` | 2.857 | 2.836267 | 1.01x | pass, cuBLAS |

**Table 2.** Windowed, mixed-pattern, and full-causal rows.

The windowed forward row beats the frozen `1.571 ms`. The mixed-pattern forward
row beats the frozen `1.57475 ms`. The full-causal forward row stays on cuBLAS
and does not regress. The backward rows stay on cuBLAS and do not regress.

### Phase D — end-to-end

The measurement command is:

```bash
tools/nanochat bench -- ./bazel-bin/src/eval_bench --batch 8 --seq 512 \
  --layers 8 --heads 4 --kv-heads 4 --hidden 512 --window-pattern SSSL --json
```

The report goes to `/tmp/eval-window.json`. Table 3 compares the result against
the frozen `docs/attention-baseline-e2e.json`.

| Metric | Frozen baseline | Promoted path | Change |
|---|---|---|---|
| Tokens per second | 55000 | 57908.3 | +5.3% |
| Forward time (ms) | 74.472727 | 70.732475 | -5.0% |

**Table 3.** End-to-end `SSSL` evaluation at `B=8 T=512 L=8`.

The tokens per second value beats the frozen baseline. The end-to-end win is
small, because attention is a small part of the step.

### Limits

- The frozen end-to-end baseline is an estimate, not a measured run. The
  reported margin of 5.3% is therefore not exact.
- The fused tile covers only `head_dim == 128` and `window_left >= 0`.
- The full-causal fused tile lost the Phase 1 kill gate. Do not widen the
  dispatch predicate without a new measurement.
- The numbers come from one host and one precision. Do not compare them across
  devices.

### Reproduction

Run the two measurements through the project entry point:

```bash
tools/nanochat profile --json --out /tmp/attention-window.json
tools/nanochat bench -- ./bazel-bin/src/eval_bench --batch 8 --seq 512 \
  --layers 8 --heads 4 --kv-heads 4 --hidden 512 --window-pattern SSSL --json \
  > /tmp/eval-window.json
```
