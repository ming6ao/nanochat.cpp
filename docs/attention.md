# Attention: the shipped paths, the measurements, and the fused tiles

Attention has three forward paths, and one backward. Two of them are fused
tiles that keep the online-softmax state in registers and skip fully masked key
tiles. The third is the batched cuBLAS decomposition, which is also the only
backward.

| Path | Covers | Engine | Part |
|---|---|---|---|
| fp32 tile | a windowed shape at head width 128, fp32 build | CUDA cores | 2, 4 |
| bf16 tile | a windowed shape at head width 128, bf16 or fp8 build, a device with tensor cores | Tensor cores | 6 |
| cuBLAS | every other shape, and every backward | cuBLAS | 2 |

The two predicates live in `backends/cuda/kernels/attention_dispatch.h`, the
tiles in `backends/cuda/kernels/attention.cu` and `attention_bf16.cuh`, and the
measured rows in the `attention`, `attention_e2e`, and `attention_bf16` entries
of [baselines.json](baselines.json).

Read [performance.md](performance.md) for the measurement protocol and
[kernels.md](kernels.md) for the family inventory.

## Part 1. The window saving

The cuBLAS path runs dense GEMMs and masks the result afterwards, so every
query-key pair costs time, even a masked pair. "Issued work" counts all dense
pairs. "Necessary work" counts only the visible ones. Their ratio is the mask
waste a tile removes.

| Case | Issued pairs | Necessary pairs | Issued / necessary |
|---|---|---|---|
| `d8_s512` full causal | 8,388,608 | 4,202,496 | 1.996 |
| `d8_s512` window 128 | 8,388,608 | 1,849,344 | 4.536 |
| GQA `B=2 T=512 H=8 KV=2` | 4,194,304 | 2,101,248 | 1.996 |
| `d8_s512` `SSSL` mean | 8,388,608 | 2,437,632 | 3.441 |
| head-dim sweep `d32` to `d256` | 131,072 | 66,048 | 1.984 |

The window 128 layer needs 44 percent of the full-causal work, and it issues
4.54 times its necessary work. The measured time is the same as the full layer:
1.571 ms against 1.586 ms. That is the defect a fused tile fixes. The `SSSL`
pattern is three window layers to one full layer, so three quarters of the
production layers pay less in theory and the same in practice.

## Part 2. The Pascal baseline and the feasibility study

### 1. What the cuBLAS path does

`backends/cuda/kernels/attention.cu`, forward:

1. transpose `q`, `k`, `v` from `[batch, seq, heads, dim]` into contiguous
   `[batch, heads, seq, dim]` scratch;
2. one batched cuBLAS GEMM `scores = Q Kᵀ * scale` over all heads;
3. a row-per-block softmax kernel that writes the normalized probabilities and
   the `(max, sum_exp)` statistics over the whole `seq x seq` matrix;
4. one batched cuBLAS GEMM `out = P V`;
5. transpose the output back.

The backward is the same decomposition with four batched GEMMs, a
softmax-gradient kernel, and transposes. Grouped-query attention widens its key
and value heads first, so one batched call still covers the batch. The backward
of a windowed shape takes the banded query-block loop instead, which never
builds the dense matrix, and keeps cuBLAS for its four products.

The costs are structural. The softmax kernels materialize three `seq x seq`
matrices in device memory and stream them several times. Every dense tile is
computed and then masked, so causality and the window save nothing. The four
layout transposes and the separate softmax kernel are pure overhead around the
GEMMs.

### 2. Why the upstream kernels cannot run on Pascal

The reference Python `nanochat` loads FlashAttention-3 and otherwise falls back
to `torch.nn.functional.scaled_dot_product_attention`. The loader requires a
device major version of 8 or 9, so the GTX 1080 Ti falls back to the math
backend, which materializes the full matrix.

| Upstream requirement | Pascal `sm_61` |
|---|---|
| `mma.sync.aligned.m16n8k16` | No tensor cores, no `mma` |
| `ldmatrix` | Not available |
| `cp.async` global to shared | Not available |
| fp16 or bf16 storage and math | GP102 fp16 is about 1/64 of fp32 |
| `setup.py` arch list `80;90;100;110;120` | No `sm_61` code at all |

Full-fused, input-aware attention is not a tensor-core-only idea. The kernel
needs enough shared memory, registers, and fp32 FMA throughput. Pascal has 48 KB
of shared memory per block, 64K 32-bit registers per streaming multiprocessor,
and 11.34 TFLOP/s of fp32. That is enough to hold a tiled online softmax. It has
no async copy pipeline, so the kernel must be compute-bound or rely on
occupancy.

### 3. The measured baseline

Measured with `tools/nanochat profile --json` on the GTX 1080 Ti at `sm_61`,
fp32, `-O2`, at `d8_s512`.

| Case | Forward | Backward |
|---|---|---|
| `d8_s512` full causal | 1.586 ms / 1,357 GFLOP/s | 2.857 ms / 1,506 GFLOP/s |
| `d8_s512` window 128 | 1.571 ms / 603 GFLOP/s | 2.806 ms / 675 GFLOP/s |
| GQA `B=2 T=512 H=8 KV=2 D=128` | 1.312 ms / 820 GFLOP/s | 3.155 ms / 682 GFLOP/s |
| head-dim sweep `d32/64/128/256` (fwd) | 0.068 / 0.092 / 0.112 / 0.164 ms | — |

The path runs at about 2.7 TFLOP/s forward against a dense operation count. The
device peak is 11.34 TFLOP/s, and a well-shaped batched GEMM reaches about 6 to
7 TFLOP/s. From the measured per-product rates, the two forward GEMMs take about
0.674 ms of the 1.586 ms. The other **0.91 ms** is transposes, softmax, and
launches. Necessary forward work is 2.15 GFLOP, which is 0.54 ms at 4 TFLOP/s
and 0.86 ms at 2.5 TFLOP/s. Both beat 1.586 ms, so the device has headroom.

### 4. What a Pascal-native fused tile does

The algorithm is FlashAttention-2 with the tensor-core MMA replaced by fp32
CUDA-core outer products.

- **Forward.** One thread block owns a tile of `Br` query rows for one
  `(batch, head)`. It loops over key tiles of `Bc` rows. For each tile it
  computes `S = Q Kᵀ`, applies the causal or window mask, updates a running row
  maximum and denominator in the online-softmax style, and accumulates `O` in
  registers. It writes `(row_max, sum_exp)` once per query row.
- **Tile skipping.** A key tile is skipped before the dot product when it is
  provably invisible to every query row in the block. This is the source of the
  window win.
- **Recompute, do not store.** A store and a load of `P` cost eight bytes per
  element. Recomputing `S` instead costs `2 * head_dim` flops, which is 32 flop
  per byte at head width 128. The device ridge point is 23 flop per byte, so
  recomputation is on the right side.

Three constraints shape the tile. There is no `cp.async`, so keep the block at
or under 48 KB and rely on occupancy. There are no tensor cores, so stay in fp32
and use `expf`. There is no `ldmatrix`, so the microkernel must use plain
shared-memory loads. That last one is the low-level risk.

The naive version fails. A block-per-query-row tile has no key or value reuse
across query rows, so it is bandwidth bound. The retired `OnlineSoftmaxTile`
variant measured 11.57 ms forward against the batched path's 1.59 ms. A fused
kernel only wins as a register-blocked tiled GEMM that reuses each key and value
tile across many query rows. `OnlineSoftmaxTile` stays a decode-path helper.

Rejected alternatives. Porting the upstream kernels is impossible on `sm_61`
(section 2). An FA1-era `sm_75` path is also tensor-core based. A mask-skipping
wrapper around cuBLAS has no mask, and per-tile GEMMs are launch-bound. Fusing
only the softmax leaves the scores dense, so the window win is lost.

References: `/home/egrader/repos/flash-attention`,
`/home/egrader/repos/nanochat/nanochat/flash_attention.py`, and
`/home/egrader/repos/llm.c/dev/cuda/attention_forward.cu`.

## Part 3. The windowed-forward promotion

The fused tile passed the window gate and failed the full-causal gate, so the
project promoted the windowed forward on its own.

| Case | Tile (ms) | cuBLAS (ms) | Speedup |
|---|---|---|---|
| full causal | 1.980 | 1.586 | 0.80x |
| window 128 | 1.078 | 1.571 | 1.46x |
| grouped query | 1.059 | 1.312 | 1.24x |

At `head_dim` 128 the 48 KB shared-memory cap admits only `Bc = 32`, and no
swept axis closed the full-causal gap. The `SSSL` pattern keeps the window win,
because three of every four layers use a window.

Scope, and it still holds: forward only, the cuBLAS backward stays, the
predicate stays narrow (a window and head width 128), and the kernel seam stays
byte for byte. A full-causal tile needs a different design and is a separate
effort.

## Part 4. Promotion results

The windowed forward passed every gate and was promoted.

### Verdict

The windowed forward is promoted. Full-causal attention stays on cuBLAS. Every
backward stays on cuBLAS.

### Parity

| Test | Metric | Measured | Tolerance | Verdict |
|---|---|---|---|---|
| `tests:oracle_cuda_test` | Maximum forward error | 3.73e-09 | 1e-5 | pass |
| `tests:oracle_cuda_test` | Maximum backward error | 5.96e-08 | 1e-5 | pass |
| `tests:oracle_cuda_test` | Maximum optimizer step 1 error | 2.38e-07 | 1e-5 | pass |
| `tests:oracle_cuda_test` | Maximum trajectory loss error | 2.38e-07 | 1e-5 | pass |
| `tests:oracle_cuda_test` | Maximum trajectory parameter error | 4.84e-05 | 1e-4 | pass |

The training-trajectory test also passes: loss `4.77e-07`, parameter L2
`1.39e-04`, gradient L2 `5.39e-07`, and gradient norm `1.19e-07`. The `SSSL`
dispatch check confirms that three of four layers take the fused branch.

### Window performance

`tools/nanochat profile --json --out /tmp/attention-window.json` runs the
shipped `attention_benchmark`.

| Row | Frozen baseline (ms) | Promoted path (ms) | Speedup |
|---|---|---|---|
| `attention_fwd:d8_s512_win128` | 1.571 | 1.087488 | 1.44x |
| `attention_fwd:d8_s512_sssl` | 1.57475 | 1.208891 | 1.30x |
| `attention_fwd:d8_s512` | 1.586 | 1.577643 | 1.01x, cuBLAS |
| `attention_bwd:d8_s512_win128` | 2.806 | 2.808224 | 1.00x, cuBLAS |
| `attention_bwd:d8_s512_sssl` | 2.81875 | 2.815197 | 1.00x, cuBLAS |
| `attention_bwd:d8_s512` | 2.857 | 2.836267 | 1.01x, cuBLAS |

### End to end

`d8_s512` `SSSL`, `B=8 T=512 L=8 H=4 KV=4 C=512`:

| Metric | Frozen baseline | Promoted path | Change |
|---|---|---|---|
| Tokens per second | 55000 | 57908.3 | +5.3% |
| Forward time (ms) | 74.472727 | 70.732475 | -5.0% |

The end-to-end win is small, because attention is a small part of the step.

### Limits

- The frozen end-to-end baseline is an estimate, not a measured run, so the
  5.3 percent margin is not exact.
- The fp32 tile covers only `head_dim == 128` and `window_left >= 0`.
- The full-causal tile lost its gate. Do not widen the predicate without a new
  measurement.
- The numbers come from one host and one precision. Do not compare across
  devices.

## Part 5. The tensor-core decision

The promoted fp32 tile runs on the CUDA cores. A bf16 build moves every other
matrix multiply to the tensor cores, so the question is what happens to the tile
there.

### 1. Why the fp32 tile loses in a bf16 build

Take the depth-24 shape: 24 layers, 12 heads, head width 128, sequence 2048,
`SSSL`.

| Path | Attention arithmetic per token | Engine rate | Time per token |
|---|---:|---:|---:|
| Fused windowed tile, fp32 | 0.396 GFLOP | 20 to 40 TFLOP/s | 9.9 to 19.8 microseconds |
| Dense cuBLAS, bf16 | 0.906 GFLOP | 791 TFLOP/s | 1.15 microseconds |
| Dense cuBLAS, fp8 | 0.906 GFLOP | 1,187 TFLOP/s | 0.76 microseconds |

The windowed tile does 2.29 times less arithmetic and still loses by a large
factor, because the engine rate decides it. The dense path pays for its speed
with memory traffic. At this shape one micro-batch of one layer moves about
6 GiB through the score and probability matrices, which is about 17 percent of
the step. Part 6 removes that traffic instead of racing it.

### 2. The three options

| Option | What it does | Off-GPU verifiable |
|---|---|---|
| A. Tensor-core tile | Rewrite the tile with `mma` or `wgmma` | No, without a new emulator feature |
| B. Retire the tile | Send the windowed forward to cuBLAS in the non-fp32 builds | Yes |
| C. Keep the fp32 tile | No change | Yes |

The decision was **option B first, then option A behind a measurement gate**.
Option B removes code and needs no new kernel: a non-fp32 build sends the
windowed forward to the cuBLAS path, which already exists and already runs at
the bf16 rate. Option C stays in the fp32 build, where the promoted tile wins.
Option A is Part 6, and its gate passed.

### 3. The fp8 operands

The fp8 build needs no separate attention kernel: the two products become
scaled fp8 GEMMs through the seam from [fp8-training.md](fp8-training.md).

| Tensor | Build | Reason |
|---|---|---|
| `q`, `k`, `v` operands | E4M3 | Input to the products |
| Scores | fp32 | The softmax needs the range |
| Probability matrix | E4M3, or the compute type | Input to the value product |
| Output accumulator | fp32 | The accumulation spans every key tile |
| Row maximum, row sum, statistics | fp32 | A reduction with an exponential |

Use one scale per tensor for a saved probability matrix. Do not reuse the score
scale, because the exponential changes the range.

### 4. The invariants

1. The row maximum, the row sum, and the saved statistics are fp32.
2. The output accumulator is fp32.
3. The causal and window mask is exact. A skipped tile must be provably
   invisible to every query row in the block.
4. The dispatch predicate stays narrow and tested.
5. The kernel seam holds no vendor type.
6. The fp32 build keeps the promoted tile, because it wins there.
7. Every new tile has a kill gate with a measured number.

### 5. Status

| Phase | Work | Status |
|---|---|---|
| A0 | Measure the fp32 tile and the cuBLAS path on the `SSSL` shape in the bf16 build | Delivered |
| A1 | Gate the tile on the fp32 build, and send the other builds to cuBLAS | Delivered: `FusedAttentionFp32Covers` |
| A2 | Add fp8 operands to the attention GEMMs | Delivered |
| A3 | Build the tensor-core tile | Delivered as Part 6 |
| A4 | Promote, or delete the prototype | Delivered: promotion is the gate of Part 6, section 7 |

The backward stays on cuBLAS by decision. It already runs at the bf16 rate
there, and the banded windowed loop already avoids the dense matrix. Revisit
that only if a measurement shows the backward is the limit, in a new document.

## Part 6. The bf16 tensor-core tile

Status: delivered. The bf16 tile runs the windowed forward on a card with
tensor cores, and the dense cuBLAS forward keeps every other shape.

### 1. Goal and scope

**Goal.** Make a bf16 build keep the sliding-window saving, which needs a fused
tile, because the dense path computes every query-key pair and masks it after.

**In scope.** The windowed forward at head width 128, in the bf16 build, on a
device with tensor cores.

**Out of scope.** The backward, which keeps the banded cuBLAS loop of Part 2.
The full-causal forward, which keeps cuBLAS for the reason in Part 4. The fp32
build, which keeps the tile of Part 2.

### 2. Why the dense path loses on Hopper

On Hopper the dense bf16 forward is memory bound, not compute bound. At the
depth-24 shape, batch 16, and sequence 2048:

| Value | Dense bf16 cuBLAS | Fused bf16 tile, window 512 |
|---|---:|---:|
| Arithmetic per token per layer | 12.6 MFLOP | 5.7 MFLOP |
| Traffic per token per layer | about 196 KB | about 55 KB |
| Arithmetic intensity | about 64 flop/byte | about 104 flop/byte |

The Hopper ridge point is 989 TFLOP/s over 3.35 TB/s, which is 295 flop/byte.
The dense path sits below the ridge, so the device rate does not help it. The
tile sits above the dense path's intensity, so it converts more of the rate into
time.

Two further wins follow. The dense forward asks `CachedBuffer(0, 3 * matrix)`
for its score, its probability, and its gradient scratch, which at this shape is
4.50 GiB in bf16. The tile asks for none, and
[node-h100.md](node-h100.md) records a bf16 margin of about 1 GiB at batch 16,
so the removal is a batch-size win as well as a time win. Second, the tile reads
the model layout, so three transposes leave the path.

### 3. Precision: H100 already runs bf16

| Place | Fact |
|---|---|
| `.bazelrc` `build:hopper` | `sm_90`, `precision=bf16`, `backend=cuda` |
| `tools/modal/app.py`, the `H100` leg | card `H100`, architecture `sm_90`, precision `bf16` |
| [precision.md](precision.md) section 4 | Ampere and newer compute in bf16 |
| [node-h100.md](node-h100.md) | the H100 node runs the demo chain at bf16 |

The tile adds no precision work. It adds one consumer of the bf16 build, and two
rules keep that honest:

1. The tile's source is compiled only by the CUDA compiler in a bf16 or fp8
   build, which is the pair `__CUDACC__` and `NANOCHAT_PRECISION != 0` state. An
   fp32 build holds no tile at all, so it cannot stage a four-byte element into
   a two-byte tile. The fp32 tile of Part 2 is the only fp32 path.
2. The dispatch reads `Caps::has_tensor_cores` at run time, so a bf16 build on a
   card without tensor cores keeps cuBLAS.

### 4. The tile

#### 4.1 Schedule

| Item | Value | Reason |
|---|---|---|
| Query rows per block, `Br` | 64 | Four warps, one warp per 16 rows |
| Key rows per tile, `Bc` | 64 | Four 16-key sub-tiles |
| Threads | 128 | One warp per 16 query rows |
| Grid | `(ceil(seq / Br), heads, batch)` | The index scheme of the fp32 tile |
| Resident blocks | 2 per streaming multiprocessor | 225,280 bytes of the 232,448 available |

#### 4.2 Microkernel

The microkernel uses the `wmma` interface with 16 by 16 by 16 bf16 fragments and
fp32 accumulators. It is the first tensor-core code in the tree. The fragment
layout of `wmma` is not a contract the kernel writes against, so a layout
mistake cannot reach the mask arithmetic. Neither product needs a shared
transpose: the key tile is staged as `[key][dim]`, which is the column-major
form of a `[dim][key]` operand. The value tile is staged the same way and reads
as a row-major operand.

A later revision can replace `wmma` with hand-written `mma.sync` fragments. That
revision removes the score scratch tile of section 4.4, because the documented
fragment layout then gives each element its row.

#### 4.3 Shared memory

At `head_dim` 128 and `Br = Bc = 64`: the row stride is 136 elements, which is
the head width plus eight for bank padding.

| Buffer | Bytes | Lifetime |
|---|---:|---|
| Query tile, 64 by 136 | 17,408 | Whole block |
| Key tile, 2 stages of 64 by 136 | 34,816 | Double buffered |
| Value tile, 2 stages of 64 by 136 | 34,816 | Double buffered |
| Probability tile, 4 warps of 16 by 72 | 9,216 | One key tile |
| Score scratch, 4 warps of 16 by 64 fp32 | 16,384 | One key tile |
| Total | 112,640 | |

The request exceeds the 48 KB default, so the launcher calls
`cudaFuncSetAttribute` once for the opt-in limit. `//tests:cuda_sim_test` fails a
launch above 48 KB without that call, so the emulator catches the omission.

#### 4.4 The score scratch tile

The `wmma` accumulator layout is not part of its interface, so a kernel cannot
read the row of one accumulator element. The tile stores its four score
fragments into a per-warp scratch tile of 16 rows by 64 columns in fp32, then
processes one row per two lanes. The shared traffic is about 4.6 KB per token
per layer, which is 1.9 nanoseconds of the aggregate shared bandwidth of the
device.

The measurement of section 10 puts the tile at 22.7 TFLOP/s, which is 2.3
percent of the bf16 peak. The tile is therefore at neither the memory bound nor
the tensor-core roof. Three likely limiters are four warps per block, one
dependent accumulation chain per key sub-tile, and the scratch round trip. The
`mma.sync` revision of section 4.2 removes the first two, and it is the recorded
next step.

#### 4.5 The key tile loop

For each visible key tile, in this order:

1. **Stage.** Copy the next key and value tile into the free pipeline stage
   before waiting for the current one, so the copy overlaps the arithmetic.
2. **Skip.** The visible key tiles are one contiguous range: the left window
   bounds it from below, and causality or the right window bounds it from above.
   The block drops a run of leading tiles and a run of trailing tiles, and it
   tests nothing inside the range.
3. **Score.** `S = scale * Q Kᵀ` as four sub-tiles of 16 keys, each over eight
   steps of the head dimension.
4. **Mask.** Apply `KeyAllowedDev` to every stored score and write `-inf` to a
   hidden pair. The mask runs on the scratch tile, so it sees the true row and
   column of every element.
5. **Online softmax.** Per row, in fp32: the tile maximum, then
   `alpha = exp(previous - updated)`, then `p = exp(s - updated)`, then the
   running sum.
6. **Rescale.** Multiply the output accumulator by `alpha`. The alpha of each
   row is published into a 16 by 16 tile and loaded as an accumulator fragment,
   so the element-wise product needs no layout knowledge.
7. **Round.** Store `p` as bf16 into the probability tile. This is the one new
   rounding step against the fp32 tile, and it matches the shipped cuBLAS path,
   whose scratch buffer already holds the compute type.
8. **Value product.** `O += P V` as eight sub-tiles of 16 head elements per
   sixteen keys.

#### 4.6 Epilogue

The output accumulator of each warp is its own 16 query rows. The epilogue
writes a zero row and the pair `(0, 0)` when the row sum is zero, which is the
empty-window contract of the reference. Otherwise it writes `O / row_sum` as
bf16, and writes `(row_max, row_sum)` as fp32 to the statistics. The layout
stays `[batch, heads, seq, 2]` from `AttentionStatsCount`, because the backward
reads it.

#### 4.7 Numerics

| Value | Type |
|---|---|
| `q`, `k`, `v`, probability tile | bf16 |
| Score and output accumulators | fp32 |
| Row maximum, row sum, `alpha` | fp32 |
| Statistics | fp32 |
| The exponential | `expf` |

The accuracy target is the bf16 target of [testing.md](testing.md), which is
1e-2 relative. The strict 1e-5 tolerances of Part 4 belong to the fp32 build and
do not move.

### 5. Dispatch

`attention_dispatch.h` holds the two shape predicates, and `attention.cu` holds
the branch order.

```c
bool FusedAttentionFp32Covers(const AttentionParams& p);
bool FusedAttentionBf16Covers(const AttentionParams& p, bool tensor_cores);
```

The fp32 predicate is the one of Part 4. The bf16 predicate adds the bf16 build,
a device with tensor cores, the window, and head width 128. Both keep
`!p.q_head_major`, because the tiles read the model stream. The header is
host-only, so `attention_test.cc` includes it and checks the same conditions
without a card.

`AttentionForward` selects the fp32 tile, then the bf16 tile, then cuBLAS. The
seam keeps its two entries, and `include/nanochat/kernels.h` stays byte for
byte.

### 6. Files and verification

| Path | What |
|---|---|
| `backends/cuda/kernels/attention_bf16.cuh` | The tile: schedule, microkernel, mask, epilogue |
| `backends/cuda/kernels/attention_dispatch.h` | The two shape predicates, shared with the test |
| `backends/cuda/kernels/attention.cu` | The branch order and the opt-in shared-memory request |
| `backends/cuda/kernels/attention_test.cc` | The head-width-128 cases and the dispatch check |
| `backends/cuda/kernels/attention_benchmark.cc` | The rows of section 8 |
| `tools/modal/app.py` | The H100 jobs of section 9 |

| Gate | Tier | Check |
|---|---|---|
| V1 | T0, no card | The fp32 tile and the cuBLAS path are unchanged: `attention_gpu_test`, `precision_gpu_test`, and the numeric trace |
| V2 | T0, no card | `--config=hopper --config=sim-gate` builds the tile with warnings as errors and reports 174 registers, no spill, and 112,640 bytes of shared memory |
| V3 | T1, card | The head-width-128 windowed forward matches the fp32 host reference at tiny shapes, in one bf16 build |
| V4 | T1, card | The dispatch check covers each predicate condition, and only one tile covers a windowed head-width-128 shape |
| V5 | T3, card | The benchmark rows of section 8, in a bf16 build |

Gate V3 compares against the fp32 host reference, not against the fp32 build,
because the two round differently by design. The emulator cannot run the tile:
[simulator.md](simulator.md) section 9.2 records that it has no tensor-core
primitive. Gate V3 therefore needs a card.

### 7. The gate

The tile replaces the dense cuBLAS forward on the windowed layers when it wins
by ten percent on the `SSSL` row at the production sequence length. If it does
not win, delete it and keep option B of Part 5.

### 8. The measurement on Modal

The H100 leg of `tools/modal/app.py` builds the tree at `sm_90` and bf16, the
same build the chain runs. Two jobs run in that image:

```bash
# The family test, including the head-width-128 cases of gate V3.
modal run tools/modal/app.py::attention_check

# The benchmark rows of section 9.
modal run tools/modal/app.py::attention_bench --argv '["--json"]'
```

Both jobs run the binary from the image's own build, so every number belongs to
the library that image built. The image sets `NANOCHAT_SANDBOX_BACKEND=none`, so
the T1 profile is a marker and the binary runs in place.

The benchmark also reports the dense path on the same shape: a head-major query
keeps the batched path, so one run measures both branches at one shape in one
build. That twin is marginally favoured, because it skips the query transpose.
The rows land in the `attention_bf16` entry of [baselines.json](baselines.json).

### 9. Results

One run of one binary on one card: NVIDIA H100 80GB HBM3, `sm_90`, bf16,
`bazel-bin/backends/cuda/kernels/attention_benchmark`. The tile row and the
dense row of a pair therefore differ only by the forward branch.

| Row | Shape | Tile | Dense cuBLAS | Ratio |
|---|---|---:|---:|---:|
| `attention_fwd:d8_s512_win128` | B=8 T=512 H=4 D=128 wl=128 | 0.0646 ms | 0.0993 ms | **1.54x** |
| `attention_fwd:d8_s512_sssl` | same, `SSSL` | 0.0747 ms | 0.0974 ms | **1.31x** |
| `attention_fwd:b4_s2048_h8_win512` | B=4 T=2048 H=8 D=128 wl=512 | 0.6639 ms | 0.8435 ms | **1.27x** |
| `attention_fwd:d8_s512` | full causal | — | 0.1048 ms | cuBLAS |

| Row | Value |
|---|---:|
| Forward rate, `b4_s2048_h8_win512` | 22.7 TFLOP/s |
| Forward rate, `d8_s512_win128` | 14.7 TFLOP/s |
| Dense forward rate, `b4_s2048_h8_win512` | 17.9 TFLOP/s |
| Fraction of the bf16 peak, production row | 2.3 percent |

Two readings. The tile wins on every windowed row, and the win is smaller as the
shape gets smaller, because a fixed launch and staging cost weighs more there.
Both paths are far below the tensor-core roof: the dense path is bound by the
score matrix it streams, and the tile is bound by its own tile efficiency. That
is why the tile wins by 1.27 to 1.54 rather than by the 2.3 factor of arithmetic
it saves.

The backward rows move the other way, and by design.
`attention_bwd:d8_s512_sssl` measures 0.2301 ms against 0.2042 ms for the dense
twin, and `attention_bwd:b4_s2048_h8_win512` measures 0.9729 ms against
0.9103 ms. Both backwards run the same banded cuBLAS loop, so the difference is
the query transpose that the head-major twin does not pay.

### 10. Verdict

The gate of section 7 passes. The tile beats the dense bf16 cuBLAS forward on
the `SSSL` shape by 31 percent at `d8_s512` and by 27 percent at the production
sequence length, against a 10 percent requirement. The tile stays.

Gate V3 passes:

| Case | Forward output | Statistics |
|---|---:|---:|
| `window-d128` | 3.8e-3 | 5.2e-3 |
| `window-d128-near-full` | 3.8e-3 | 1.5e-2 |
| `gqa-window-d128` | 4.4e-3 | 5.6e-3 |
| `kv-cache-window-d128` | 2.8e-3 | 5.3e-3 |
| `bidirectional-window-d128` | 3.1e-3 | 8.3e-3 |
| `window-d128-tile-tail` | 3.0e-3 | 7.5e-3 |

The values are the maximum absolute error against the fp32 host reference, and
they sit inside the bf16 tolerance of 3e-2.

One change came with the gate. `attention_test.cc` compared every case against
the fp32 host reference at 1e-4, which is the fp32 tolerance, so a bf16 build
failed on the rounding alone. The file now carries one build-aware tolerance,
and it differentiates only in an fp32 build, because a central difference over a
bf16 loss measures the rounding rather than the gradient.
[testing.md](testing.md) gives both rules.

### 11. Risks and non-goals

- **The first tensor-core code in the tree.** The `wmma` interface keeps the
  fragment layout out of the kernel, and gate V3 compares against a second bf16
  path.
- **The scratch-tile round trip.** Section 4.4 bounds it. If the measured rate
  shows it binds, the `mma.sync` revision of section 4.2 removes it.
- **A wide predicate.** The predicate stays narrow, and gate V4 tests each of
  its conditions.
- **Not a goal:** a backward rewrite, a full-causal tile, a split head
  dimension, and any change to the seam, the arena, or the statistics layout.

## See also

- [performance.md](performance.md) — the measurement protocol.
- [performance-model.md](performance-model.md) — the bound that judges a tile.
- [precision.md](precision.md) — the fp32 and bf16 policy.
- [fp8-training.md](fp8-training.md) — the fp8 operand rules.
- [node-h100.md](node-h100.md) — the memory margin and the node plan.
- [simulator.md](simulator.md) — the emulation engine limits.
