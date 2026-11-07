# Attention

Attention has three forward paths and one backward. Two paths are fused tiles
that keep the online-softmax state in registers and skip fully masked key tiles.
The third is the batched cuBLAS decomposition, which is also the only backward.

| Path | Covers | Engine |
|---|---|---|
| fp32 tile | a windowed shape at head width 128, fp32 build | CUDA cores |
| bf16 tile | a windowed shape at head width 128, bf16 or fp8 build, a device with tensor cores | tensor cores |
| cuBLAS | every other shape, and every backward | cuBLAS |

The two predicates live in `backends/cuda/kernels/attention_dispatch.h`, the
tiles in `backends/cuda/kernels/attention.cu` and `attention_bf16.cuh`, and the
measured rows on a supported card in the `attention_bf16` entry of
[baselines.json](baselines.json).

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
4.54 times its necessary work. Its measured time was the same as the full
layer's. That is the defect a fused tile fixes. The `SSSL`
pattern is three window layers to one full layer, so three quarters of the
production layers pay less in theory and the same in practice.

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
scaled fp8 GEMMs through the seam from [precision.md](precision.md).

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
7. Every new tile has a gate with a measured number.

## Part 6. The bf16 tensor-core tile

The bf16 tile runs the windowed forward on a card with
tensor cores, and the dense cuBLAS forward keeps every other shape.

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
[host-portability.md](host-portability.md) Part 1 records a bf16 margin of about 1 GiB at batch 16,
so the removal is a batch-size win as well as a time win. Second, the tile reads
the model layout, so three transposes leave the path.

### 3. Precision: H100 already runs bf16

| Place | Fact |
|---|---|
| `.bazelrc` `build:hopper` | `sm_90`, `precision=bf16`, `backend=cuda` |
| `tools/modal/app.py`, the `H100` leg | card `H100`, architecture `sm_90`, precision `bf16` |
| [precision.md](precision.md) section 4 | Ampere and newer compute in bf16 |
| [host-portability.md](host-portability.md) Part 1 | the H100 node runs the demo chain at bf16 |

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

The measured run puts the tile at 22.7 TFLOP/s, which is 2.3
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

# The family test, including the head-width-128 cases of gate V3.
modal run tools/modal/app.py::card_job --name attention_check

# The benchmark rows of section 9.
modal run tools/modal/app.py::card_job --name attention_bench --argv '["--json"]'
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

The backward rows move the other way, and by design. Both backwards run the
same banded cuBLAS loop, so the only difference is the query transpose that the
head-major twin does not pay, and the dense twin is marginally faster.

## See also

- [performance.md](performance.md) — the measurement protocol.
- [performance-model.md](performance-model.md) — the bound that judges a tile.
- [precision.md](precision.md) — the fp32 and bf16 policy.
- [precision.md](precision.md) — the fp8 operand rules.
- [host-portability.md](host-portability.md) Part 1 — the memory margin and the node plan.
- [simulator.md](simulator.md) — the emulation engine limits.
