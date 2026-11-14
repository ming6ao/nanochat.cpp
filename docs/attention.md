# Attention

Attention has four forward paths and two backward paths. The flash family is
fused, keeps the online-softmax state in registers, skips every key tile that no
query row in the block can see, and reads either query layout. Two earlier fused
tiles remain. The last path is the batched cuBLAS decomposition, which covers
every shape the flash family leaves out and the whole backward of an fp32 build.

| Path | Covers | Engine |
|---|---|---|
| flash family | a bf16 or fp8 build, a tensor-core device, head width 64 or 128, a one-sided window | tensor cores |
| fp32 tile | a windowed shape at head width 128, fp32 build | CUDA cores |
| bf16 tile | the Part 6 shapes, on a tensor-core device whose opt-in shared memory is below 139,264 bytes | tensor cores |
| cuBLAS | every other shape, a decode step, and every other backward | cuBLAS |

The three predicates live in `backends/cuda/kernels/attention_dispatch.h`, the
tiles in `backends/cuda/kernels/attention.cu`, `attention_bf16.cu`, and
`attention_flash.cu`, and the measured rows on a supported card in the
`attention_flash` and `attention_bf16` entries of
[baselines.json](baselines.json).

Part 7 designs the flash family, and its section 7 records the measurement: the
correctness gate passed and both rate gates missed. The family is therefore
selected in a bf16 build and the promotion is an open decision. The table above
states what the tree runs today.

The part numbers are historical. Parts 2 to 4 described the Pascal study, the
fp32 tile, the windowed-forward promotion, and the promotion's results. This
document retired them on 2026-11-06 and kept the numbers, so a reference from
an earlier commit points at text that has left. The fp32 tile and the banded
backward are described in `backends/cuda/kernels/attention.cu`, and Parts 6 and
7 name their replacement.

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
   build, because the build selects it
   (`backends/cuda/kernels/BUILD.bazel`). An fp32 build links the stand-in, which
   answers false and launches nothing, so it cannot stage a four-byte element
   into a two-byte tile. The fp32 tile in `attention.cu` is the only fp32 path.
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
1e-2 relative. The strict 1e-5 tolerance belongs to the fp32 build and does not
move.

### 5. Dispatch

`attention_dispatch.h` holds the two shape predicates, and `attention.cu` holds
the branch order.

```c
bool FusedAttentionFp32Covers(const AttentionParams& p);
bool FusedAttentionBf16Covers(const AttentionParams& p, bool tensor_cores);
```

The fp32 predicate is the one that `attention_dispatch.h` states for the fp32
tile. The bf16 predicate adds the bf16 build,
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

## Part 7. The flash kernel

Part 6 fuses the windowed forward. The tree still spends at least 52
milliseconds of attention in one step of the production shape. Two causes
remain. The bf16 tile covers neither the backward nor the head-major query that
the fused `qk_prep` writes, and the tile reaches 22.7 TFLOP/s, which is 2.3
percent of the peak. This part designs one family for both passes, in the
layouts that the model already holds.

### 1. The measured defect

One traced base step of the `big` profile, from
`profiles/pgo-big/base-h100/iter-02/before/trace/`: 20 layers, hidden 1280, ten
query heads over five key-value heads, head width 128, sequence 1024, the `SSL`
window, batch 16, and 16,384 tokens per step.

| Work | Launches per step | Milliseconds per step |
|---|---:|---:|
| Softmax, forward | 20 | 10.84 |
| Softmax, backward | 59 | 12.28 |
| Head transposes, both passes | 220 | 9.93 |
| `qk_prep`, both passes | 80 | 11.65 |
| The four products, dense or banded | inside the GEMM row | not split |
| **The bound for the whole family** | — | **3.18** |

The softmax forward runs once per layer, so no layer takes the fused bf16 tile.
`QueryIsHeadMajor` in `src/ops.cc` sets `q_head_major` when the fused `qk_prep`
is registered, and both dispatch predicates require `!q_head_major`. The two
fused families are therefore exclusive. The tree keeps the cheaper one by
accident: `qk_prep` saves 11.65 ms, and the dense attention pays 33.1 ms of
softmax and transpose passes that a fused tile would not issue.

The dense forward issues 1.72 TFLOP per step at this shape. That is 1.95 times
the arithmetic the counter table counts, because a masked pair costs the same
as a visible one. The backward has no fused path: the dense forward and the
banded backward stage or write eleven head-major buffers on every layer, which
is the 220 transpose launches.

### 2. The decisions

| # | Decision | Reason |
|---|---|---|
| F1 | One family for both passes | The softmax passes and the transposes serve only the cuBLAS decomposition |
| F2 | Accept the head-major query, and keep the stream layout for k, v, out, dk and dv | The fused `qk_prep` already writes the query head-major, and one head's query rows are contiguous in that layout |
| F3 | `mma.sync` m16n8k16 with fp32 accumulators, not `wmma` | The `wmma` fragment layout hides the row of a score element, which forces the fp32 scratch tile of Part 6, section 4.4, and 22.7 TFLOP/s |
| F4 | Keep the probability operand in registers between the two products | Two adjacent 16 by 8 accumulators are the 16 by 16 A operand of the value product, so the scratch round trip leaves |
| F5 | Two backward kernels, and no atomics | dQ is written once per query row, and dK with dV once per key row, so two runs agree bit for bit |
| F6 | Recompute the score and the probability from the saved statistics | Two extra products per pair, against the score and probability matrices that the banded loop reads and writes |
| F7 | Scores, statistics and accumulators in fp32; the probability operand in the compute type | The invariants of Part 5, section 4 do not move |
| F8 | No seam change before phase F3 | `AttentionParams` already carries the scale, the window, `kv_len`, `stats_row_stride` and `q_head_major` |

### 3. The forward tile

#### 3.1 Schedule

| Item | Value | Reason |
|---|---|---|
| Query rows, `Br` | 128 | Eight warps, one 16-row fragment set per warp |
| Key rows, `Bc` | 64 | Four 16-key sub-tiles per product |
| Threads | 256 | Eight warps |
| Grid | `(ceil(seq / Br), heads, batch)` | Each block owns its own output rows |
| Blocks per streaming multiprocessor | 1 | 120 KiB of shared memory, and about 120 registers per thread |
| Key and value stages | 2 | The copy of the next tile overlaps the arithmetic of the current one |

#### 3.2 Shared memory

The row stride is the head width plus eight elements, which is 272 bytes and a
multiple of sixteen.

| Buffer | Bytes | Lifetime |
|---|---:|---|
| Query tile, 128 by 136 | 34,816 | Whole block |
| Key tile, 2 stages of 64 by 136 | 34,816 | Double buffered |
| Value tile, 2 stages of 64 by 136 | 34,816 | Double buffered |
| Total | 104,448 | |

Decision F4 keeps the probability operand in registers, so the probability
free. At head width 64 the same three buffers read 55,296 bytes.

The request exceeds the 48 KiB default, so the launcher calls
`cudaFuncSetAttribute` once for the opt-in limit. `//tests:cuda_sim_test` fails
a launch above 48 KiB without that call.

#### 3.3 The microkernel

One instruction serves both products: `mma.sync.aligned.m16n8k16` with bf16
operands and an fp32 accumulator. Each warp owns 16 query rows and the whole
128-column output accumulator of those rows.

The score accumulator of a warp is a 16 by 64 tile in eight `m16n8` fragments.
A row of that tile spans four lanes of a quad, so the row maximum, the row sum
and the rescale factor need a shuffle over four lanes and no shared memory.
That is the whole difference against Part 6, section 4.4: the tile reads and
writes the row of a score in registers, so the fp32 scratch tile of 16,384
bytes leaves the kernel.

The probability operand of the value product is the same eight fragments,
rounded to bf16 in registers. Two adjacent 16 by 8 fragments hold the 16 by 16 A
operand, and the B operand is the staged value tile.

Both products read their operands with `ldmatrix`, as the reference kernel does
(`SmemCopyAtom` of its `kernel_traits.h`). One `ldmatrix.x4` fills the four
registers of the query operand, and one fills two key fragments or two
value fragments. The value product needs the transpose form, because its `k`
dimension is the key index and the staged value row is not: `ldmatrix.x4.trans`
transposes the eight-row segments it reads, so the value tile stays in the
`[key][dim]` layout of the staging copy and no second staging pass exists. The
padding of eight elements in a staged row is the only layout requirement: it
keeps every segment aligned to sixteen bytes, and it puts the eight rows of one
matrix on distinct banks, so no swizzle is needed.

#### 3.4 The key tile loop

The loop is the one of Part 6, section 4.5, with three changes.

1. **Layout.** The query tile reads `[batch, heads, seq, dim]` when
   `q_head_major` is set, and `[batch, seq, heads, dim]` otherwise. One index
   function serves both, so the predicate no longer rejects a caller.
2. **Mask.** `KeyAllowedDev` runs on the accumulator fragment, so a tile that
   is wholly visible skips the mask. A window of `-1` is the full causal case,
   which now takes the same path.
3. **Reduce.** The row maximum and the row sum are warp quads, not lanes of a
   shared tile. The statistics are written once, by the lane that holds the
   row, in the layout of `AttentionStatsCount`.

A block issues `(window + Br) / window` of the necessary arithmetic at the
production shape, which is 1.25 for a window of 512.

#### 3.5 The fp8 operand

The fp8 build keeps the bf16 probability operand of a bf16 build, because the
family rounds `P` in registers to the compute type and the compute type of an
fp8 build is bf16. The scores stay fp32 and the accumulator stays fp32. The
per-row E4M3 scale of Part 5, section 3 is not landed, and it is a later change
to the one packing step.

### 4. The backward

Two kernels, one per direction, so no element is written twice.

#### 4.1 The query kernel

One block per `(query block, head, batch)`. It loops over the visible key tiles
and recomputes the score, because the forward keeps no probability matrix. The
saved statistics give the row maximum and the row sum, so `P` is exact in one
step.

`row_dot` is the one value a single pass cannot produce, because it sums over
every key of the row. The kernel therefore makes two passes over the same
visible tiles, and each pass stages its tiles again:

1. **Pass 0.** `S = scale * Q K^T`, masked by `KeyAllowedDev`, then
   `P = exp(S - row_max) / row_sum`, then `dP = dO V^T`, then
   `row_dot = sum(P * dP)` per query row, a quad reduction. The lane that owns
the row writes `row_dot` where the key and value kernel reads it.
2. **Pass 1.** The same score, probability and output-gradient products, then
   `dS = P * (dP - row_dot)` in registers, then `dQ += scale * dS K`, in fp32
   across the key tiles and written once.

Two properties hold the cost of the second pass down. Pass 0 writes no
gradient, so it needs no accumulator beyond `row_dot`. And the identity
`row_dot = dO . O` would remove the pass completely, but the seam carries no
forward output, so the pass stays until phase F3 (decision F8).

#### 4.2 The key and value kernel

One block per `(key block, key-value head, batch)`. It owns 128 key rows, one
16-row fragment set per warp, and it loops over the query tiles whose window
contains the key block, staging 32 query rows at a time. The block recomputes
the score again.

1. **Score and probability.** The same two steps, for each query tile in the
   window.
2. **Output gradient product.** `dP = dO V^T`, for the same tile.
3. **Score gradient.** `dS = P * (dP - row_dot)`, per query row.
4. **Key gradient.** `dK += scale * dS^T Q`, in fp32 across the query tiles.
5. **Value gradient.** `dV += P^T dO`, in fp32 across the query tiles.

Grouped-query attention needs no fold kernel. The block owns one key-value head
and loops over the query heads that read it, which is two at the production
shape. The two heads accumulate into the same fp32 registers.

#### 4.3 Shared memory

The query kernel, at head width 128:

| Buffer | Bytes |
|---|---:|
| Query tile, 128 by 136 | 34,816 |
| Output gradient tile, 128 by 136 | 34,816 |
| Key tile, 2 stages | 34,816 |
| Value tile, 2 stages | 34,816 |
| Total | 139,264 |

The key and value kernel, at the same width:

| Buffer | Bytes |
|---|---:|
| Key tile, 128 by 136 | 34,816 |
| Value tile, 128 by 136 | 34,816 |
| Query tile, 32 by 136 | 8,704 |
| Output gradient tile, 32 by 136 | 8,704 |
| Row maximum, inverse sum, row dot, 3 by 32 fp32 | 384 |
| Total | 87,424 |

The score gradient needs no tile. It lives in the accumulator fragments of the
step that produces it, and the row dot reads it there. The query kernel's
139,264 bytes is the family's high-water mark, and it is the figure that the
device gate compares with the opt-in limit, so a card whose limit is below it
keeps cuBLAS.

#### 4.4 The arithmetic cost

The family issues nine products per query and key pair: five in the query
kernel, which is two in pass 0 and three in pass 1, and four in the key and
value kernel. The banded loop issues five, and the counter table counts four.
The backward row therefore gains an issued-work ratio of 2.25 in the report of
[performance-model.md](performance-model.md) section 8, and the design states
that number rather than hiding it.

One product per pair is 0.433 TFLOP of this step, so the four products the
family issues over the shipped banded loop are 1.7 TFLOP, which is 1.7 ms at
the declared peak. It buys the removal of the score and probability traffic,
which is 0.94 GiB of scratch at this shape, and the 9.93 ms of transposes. The
two passes of section 4.1 are half of that extra work, and the seam change they
wait on is the first thing phase F3 should retire.

### 5. Dispatch

One predicate replaces the two of `attention_dispatch.h` for a bf16 build.

```c
bool FusedAttentionFlashCovers(const AttentionParams& p, bool tensor_cores) {
  const int width = p.head_dim;
  return kComputeIsBf16 && tensor_cores && (width == 64 || width == 128) &&
         p.window_right == 0 && p.kv_len <= p.seq;
}
```

Three conditions change against Part 6, section 5.

1. The head-major query is accepted, because F2 reads it.
2. A full-causal layer is accepted, because F1 keeps the window in the mask.
3. A decode step keeps cuBLAS, because `kv_len > seq` needs the cache path and
   a different tile split.

`attention.cu` selects the flash family first, then the fp32 tile, then
cuBLAS. The seam keeps its two entries and its version.

The family reads both query layouts, so a bf16 build sends every windowed and
full-causal head-width-64 or head-width-128 shape to it. The `*_cublas` rows of
`attention_benchmark` therefore measure the head-major layout of the family and
not the dense path in such a build. The dense comparison is the `attention_bf16`
entry of [baselines.json](baselines.json), and section 7 reads both entries.

### 6. Phases and gates

| Phase | Change | Gate |
|---|---|---|
| F1 | The forward tile, both query layouts, and the full-causal case | `//backends/cuda/kernels:attention_gpu_test` at the bf16 tolerance of [testing.md](testing.md); the forward row of `attention_benchmark` at 200 TFLOP/s or more; no softmax forward launch in a trace |
| F2 | The two backward kernels | The family test at the same tolerance; the backward row at 150 TFLOP/s or more; no softmax backward and no attention transpose launch in a trace; the step at the production shape at 182 ms or less |
| F3 | Fold `qk_prep` into the tile, which removes 11.65 ms | Seam version 4 with `cos`, `sin` and `rms_eps` in `AttentionParams`; the loss trace inside the measured noise floor of 9.1e-4 over 200 steps |

The step projection is linear in the family time. At 25 ms for the family the
step reads 182 ms, which is a Model FLOPs Utilization of 23.4 percent against
the 20.5 percent of the trace. Phase F3 removes a further 11.65 ms and reads
170 ms, which is 25.1 percent.

Each phase lands behind its own gate, and a phase that misses its rate gate is
reported and not promoted. The measurement is one base stage of the `big`
profile through `tools/modal/app.py::measure_stage`, at the protocol of
[performance.md](performance.md) section 7, and the row lands in
`docs/baselines.json`.

### 7. Results

One run of one binary: NVIDIA H100 80GB HBM3, `sm_90`, bf16, through the two
card jobs `attention_check` and `attention_bench`. Every measured row is in the
`attention_flash` entry of [baselines.json](baselines.json).

**The correctness gate passed.** The family test holds the bf16 tolerance of
[testing.md](testing.md), which is 3e-2 relative, on both passes. The case list
adds a two-block windowed case, a full-causal case, a head-major case, and head
width 64. The largest error of the two-block case is 0.0031 on the output,
0.0013 on the query gradient, and 0.0060 on the value gradient.

**The step gate of phase F2 was not measured.** It needs one base stage of the
`big` profile through `measure_stage`, and this campaign measured the family
alone.

**Both rate gates missed.** The production row is `b4_s2048_h8_win512`.

| Row | Family | The Part 6 path | Ratio | Gate | Verdict |
|---|---:|---:|---:|---:|---|
| Forward, production | 47.3 TFLOP/s | 22.7 TFLOP/s | 2.08 | 200 | miss |
| Backward, production | 27.3 TFLOP/s | 31.0 TFLOP/s | 0.88 | 150 | miss |
| Forward, `d8_s512_win128` | 22.9 TFLOP/s | 14.7 TFLOP/s | 1.56 | — | — |
| Backward, `d8_s512_win128` | 14.1 TFLOP/s | 7.8 TFLOP/s | 1.82 | — | — |
| Forward, `d8_s512` full causal | 0.0684 ms | 0.1048 ms | 1.53 | — | — |
| Backward, `d8_s512` full causal | 0.2275 ms | 0.1865 ms | 0.82 | — | — |

The forward wins on every row, and by 1.53 to 2.66 times. The backward wins on
every windowed row, by 1.32 to 1.82 times, and loses on the two full-causal
rows, where the banded loop issued five products to the family's nine. Grouped
query is the weakest backward row, at 4.24 TFLOP/s.

**The tile is bound by its own efficiency.** The forward issues 1.25 times the
necessary arithmetic at a window of 512, so 47.3 TFLOP/s of necessary work is
59 TFLOP/s of issued work, which is 6 percent of the bf16 peak. A block issues
9,216 `mma` instructions, so the instruction roof of one block is about
4.7 microseconds and the measurement is about four times that. The three
candidates are the softmax exponential, which is one `expf` per score element,
the scalar shared-memory loaders, which issue two to four loads per `mma`, and
the one resident block per multiprocessor. A profile decides between them, and
none of the three is a change to the family's structure.

**The promotion is open.** Section 6 states that a phase that misses its rate
gate is reported and not promoted, and both phases missed. The family is
selected in a bf16 build today, so the tree runs it. Two figures argue for that
selection: the forward is faster on every row, and the family removes the
softmax passes and the 220 transpose launches of the traced step. Two argue
against it: the forward reaches 4.8 percent of the peak, and the full-causal
backward is slower than the banded loop it replaces. The next step is a profile
of the tile, and `Br` of 64 with two resident blocks is the first configuration
to try.

### 8. Risks

- **The fragment map is a hardware layout.** F4 relies on the `m16n8`
  accumulator of one instruction being the `m16k16` A operand of another. That
  holds on `sm_80` and newer and is a documented layout, but a card below
  `sm_80` keeps cuBLAS, and the predicate says so.
- **One block per streaming multiprocessor is a bet.** The register budget
  decides it. A miss at 120 registers per thread costs the launch, so `Br` of
  64 is the fallback and it needs no other change.
- **The extra products are not free on every shape.** At a short sequence the
  dense path is not memory bound, and a 2.25 work ratio loses. The predicate
  keeps a sequence floor, which phase F1 sets from the first measurement.
- **Two kernels read the forward statistics.** A wrong `stats_row_stride` or a
  wrong query offset is silent, and it scales one row by the wrong sum. The
  family test covers a windowed case at a block boundary for that reason.
- **A shared-memory region that overlaps its neighbour is silent.** The three
  buffers of the key and value kernel are sized by one template each, and an
  offset that names the wrong one moves a staged tile onto its neighbour
  without a compile error. The family test of section 7 is what caught an
  overlap of that kind, so a new buffer joins the size expression and the
  offset expression in one change.
- **This design does not touch the GEMM, the optimizer, or the launch count.**
  Those are the larger terms of the step, and
  [performance-model.md](performance-model.md) section 10.1 states the share of
  each.

### 9. Alignment with the reference kernel

The reference is the Ampere path of flash-attention (`csrc/flash_attn/src`):
`mma.sync` fragments, `cp.async` staging, and the online softmax in base two.
The family takes three things from it.

1. **The operand load.** `ldmatrix.x4` and `ldmatrix.x4.trans` replace the
   scalar loaders of section 3.3, which issued two to four loads and a merge
   per fragment.
2. **The base-two exponential.** The softmax scale carries `log2(e)`, so one
   fused multiply-add and one `exp2` serve each score. The mask writes the
   sentinel into the score, so the probability pass holds no branch and no
   row-empty guard, and the saved row maximum is converted back to natural
   units at the epilogue.
3. **The mask split.** The three kernels keep the tile-level visibility test,
   which is the reference's peeled boundary loop in a runtime form.

Four differences against the reference remain, and each is a separate change.

| Difference | The reference | Why it is not landed here |
|---|---|---|
| Warps and query rows | Four warps, 32 query rows each, at head width 128 | It needs two fragment sets per warp in all three kernels, and the reference measures 8 warps as 28 percent slower |
| Key tile at head width 64 | 128 keys | It doubles the score registers, and the reference pairs it with four warps |
| The row dot product | One element-wise kernel computes `sum(dO * O)` before the backward, so the query kernel runs one pass | The kernel needs `out`, which `AttentionBackward` does not take; that is a frozen header |
| The output epilogue | Staged in shared memory, written in 16-byte stores | Small; the epilogue is one write per row |

The shared tile keeps two barriers per key tile, which is the reference's
schedule as well, so section 6 does not count it as a defect.

**The build state.** The three kernels compile at `sm_90` in a bf16 build with
no spill: 164 and 132 registers for the forward at head width 128 and 64, 244
and 216 for the query kernel, and 240 and 182 for the key and value kernel.

The numeric gate and the rate rows were measured on the H100 leg of
`tools/modal/app.py` after this revision, and the rows are the
`attention_flash_ldmatrix` entry of [baselines.json](baselines.json).

| Row | Before, ms | After, ms | Gain | TFLOP/s | MFU |
|---|---:|---:|---:|---:|---:|
| `attention_fwd:d8_s512` | 0.06845 | 0.04563 | 1.50x | 47.2 | 4.8 percent |
| `attention_bwd:d8_s512` | 0.22752 | 0.17963 | 1.27x | 24.0 | 2.4 percent |
| `attention_fwd:d8_s512_win128` | 0.04142 | 0.03131 | 1.32x | 30.2 | 3.1 percent |
| `attention_bwd:d8_s512_win128` | 0.13423 | 0.11114 | 1.21x | 17.0 | 1.7 percent |
| `attention_fwd:b4_s2048_h8_win512` | 0.31858 | 0.20952 | 1.52x | 71.9 | 7.3 percent |
| `attention_bwd:b4_s2048_h8_win512` | 1.10463 | 0.88387 | 1.25x | 34.1 | 3.4 percent |

The rate is the necessary work over the time, so it understates the work the
family issues. The tile quantum of section 3.4 puts the production forward at
1.25 times that count, which is 89.7 TFLOP/s and 9.1 percent of the peak, and
the backward issues nine products a pair where the accounting counts eight, so
its issued rate is 47.8 TFLOP/s and 4.8 percent.

Both rate gates of section 6 still miss: the production forward reads 71.9
TFLOP/s against the 200 of phase F1, and the backward 34.1 against the 150 of
phase F2. The two remaining structural differences of the table above are the
next candidates, and the products of the query kernel are the larger of them:
its second pass is two of the nine. Part 8 answers both: the warpgroup path
replaces the warp geometry, and its decision H6 adds the preprocess kernel that
removes the second pass.

**The numeric gate.** `//backends/cuda/kernels:attention_gpu_test` passed in the
same image at the bf16 tolerance of [testing.md](testing.md), on every case of
section 7. The largest figures the test prints are 0.0038 on a forward output,
0.0094 on a value gradient, and 0.0887 on the statistics of the block-boundary
case. The block-boundary case is the one section 7 records for the revision
before this one, at 0.0031 on the output, 0.0013 on the query gradient, and
0.0060 on the value gradient, and this run reproduces all three to the digits
the test prints. The two revisions therefore agree where the test can see.


## Part 8. The Hopper path

This part is a plan with its gates. Part 7's family stays as the Ampere path,
and it stays as the fallback for every shape and card this part does not reach.

### 1. The measured reason

The ceiling probe of [kernels.md](kernels.md) Part 3 runs
`wgmma.mma_async.m64n128k16` from a resident shared-memory tile on the H100 leg
and reads 972.1 TFLOP/s on 264 blocks, which is 98.3 percent of the 989
TFLOP/s that `backends/cuda/caps.json` declares. The warp-level tile of Part 7
reads 89.7 TFLOP/s on the issued work of the production shape. No warp-level
`mma.sync` kernel reaches the declared rate on this card, so the gap is the
instruction and the schedule around it, not the arithmetic and not the library.

### 2. The decisions

| Decision | Choice |
|---|---|
| H1. Instruction | `wgmma.mma_async.m64nNk16`, with the query operand in shared memory and the probability operand in registers. It is architecture conditional, so the target states `archs = ["sm_90a"]` and no `.bazelrc` line moves ([build.md](build.md)) |
| H2. Load path | The Tensor Memory Accelerator for the query, key, value, and output tiles, with one `mbarrier` per stage. This path holds no `cp.async` and no `ldmatrix`: the warpgroup instruction reads shared memory itself |
| H3. Warp specialization | One producer warp issues the copies and never computes; the consumer warpgroups issue the products and never wait on a global load. `setmaxnreg` moves registers from the producer to the consumers |
| H4. Overlap | Two consumer warpgroups interleave, so one runs the softmax of a key tile while the other issues the score product of the next tile |
| H5. The pair | A cluster of two CTAs. One multicast copy fills the key and value stage of both, and each CTA owns an adjacent 128-row query tile. The pair therefore reads half the key and value bytes of two independent blocks |
| H6. The row dot | A preprocess kernel writes `sum(dO * O)` per row, as the reference does, so the query-gradient kernel keeps one pass and issues seven products a pair instead of nine. Section 4 states the seam change it needs |
| H7. Coverage | The `mma.sync` family of Part 7 stays for a card below `sm_90` and as the fallback of section 7. The host reference that Part 7 section 7 measures against stays the one oracle for both paths |

### 3. The geometry

| Quantity | Value |
|---|---|
| Query rows per CTA | 128, in two tiles of 64, one per consumer warpgroup |
| Key rows per tile | 128 |
| Threads | 128 producer threads and 256 consumer threads, 384 in total; the first phase runs 32 producer threads and 128 consumer threads |
| Stages | Two of the key tile and two of the value tile |
| Registers | The score accumulator is 64 by 128 fp32 per warpgroup, which is 64 registers a thread; the output accumulator is another 64; the probability operand of the second product is another 64 |
| Shared memory at width 128 | Query 32 KiB resident, key 2 by 32 KiB, value 2 by 32 KiB, and the barriers: 160 KiB |
| Shared memory at width 64 | The same three buffers at half the width: 80 KiB |

At width 128 the request leaves one block per multiprocessor and a pair spans
two of them. The production shape has 512 blocks, which is two waves of the
card.

### 4. The seam change

Decision H6 needs the forward output in the backward. `AttentionBackward`
therefore takes `const ComputeType* out` after `dout`, in
`include/nanochat/kernels.h`, which is a frozen header and an architect change.
Four definition sites move with it: `backends/cuda/kernels/attention.cu`,
`backends/cpu/kernels.cc`, the caller in `src/ops.cc`, which holds the forward
output, and the two test helpers of the family and the numeric trace.

A refusal of that change leaves the two-pass query kernel of Part 7 section 4.1,
which costs two of the nine products a pair. The phase gate of section 6 states
which one it measures.

### 5. The forward tile

One key tile, in order:

1. The producer issues the copy of key tile `t + 1` and value tile `t + 1` into
the stage that tile `t - 1` released, and arrives on that stage's barrier.
2. The consumer waits the barrier of stage `t`, then issues the score product
from the resident query tile and the staged key tile.
3. It waits the product, applies the mask of Part 7 section 3.4, computes the
running maximum in base two, and writes the probability operand to registers.
4. It issues the value product with that operand, rescales the output
accumulator, and updates the row sum. The row reductions stay in the warpgroup:
a row of the 64-row accumulator lies in four lanes of one warp, so the shuffle
is over four lanes and then over the four warps through shared memory.
5. The output leaves through shared memory and one copy at the end of the block.

The pair of section 2 differs in one step only: the copies of step 1 name both
CTAs of the cluster, so one instruction fills the stage of each.

**The operand form.** The instruction reads each operand as a run of core
matrices, and a core matrix is eight rows of eight elements with the rows
sixteen bytes apart. A staged tile must therefore be **interleaved**: for each
group of eight rows the two halves of the `k` axis sit adjacent, and the next
group of eight rows follows. The descriptor states that form with a leading
offset of 128 bytes and a stride offset of 256.

A plain row-major tile is not readable. A sum-based check cannot see the
difference, because the sum of a product is invariant under the row and column
permutation that the wrong form induces, so a probe of this family must check
one element at a time or compare against a reference built in the staged
layout. The first version of the copy probe did neither, and a one-hot probe
was what caught it: of fourteen elements, only the five whose row and column
are both multiples of eight were read at all.

The value product's second operand is the staged value tile, which has the
opposite orientation. The transpose bit is not what fixes it: the interleaved
form is a property of which axis the staged rows hold, so the value tile is
staged **transposed**, as `dim` rows by `kAtomN` keys, and both products then
read the operand form this section describes with the transpose bits clear. The
staged transposition is one thread reading eight keys of one head-width row,
which is coalesced across the warp, and writing the sixteen bytes the
interleaved form keeps adjacent.

That is measured, not assumed. The probe tries both stagings, both transpose
bits, and three atom strides, and six one-hot triples have to land on their
expected `(row, dim)`: the transposed staging with both bits clear and a stride
of 256 lands six of six, and no other combination lands more than four. The
staged value tile therefore has the same descriptor parameters as the staged
key tile, and the tile of section 3 needs no second staged form.

**The accumulator map.** The map is measured two ways and both agree. Fifteen
one-hot elements each land in exactly one register, and a decode of the whole
table is a bijection: 8,192 slots, none missing and none repeated. With `lane`
the thread's index in the warpgroup and `index` the register:

| Field | Value |
|---|---|
| row | `16 * (lane / 32) + (lane % 32) / 4 + 8 * ((index / 2) % 2)` |
| column | `8 * (index / 4) + 2 * (lane % 4) + (index % 2)` |

So one warp holds sixteen rows, a lane holds two of them eight apart, a row
lives in the four lanes of one quad, and a register holds two adjacent columns
of an eight-column group. That is the map of the `mma.sync` accumulator of Part
7, repeated sixteen times along the key axis, so the row reductions of the
softmax are a shuffle over four lanes and never cross a warp.

The formula is now checked and not only stated: the probe decodes all 8,192
slots and compares each against the two lines above, and no slot disagrees.

**The register operand.** The value product takes the probability tile as its
first operand in registers, and that operand has no descriptor. One element at a
time gives it, with `reg` the register and `half` the two-byte half of it:

| Field | Value |
|---|---|
| row | `16 * (lane / 32) + (lane % 32) / 4 + 8 * (reg % 2)` |
| key | `8 * (reg / 2) + 2 * (lane % 4) + half` |

So the register operand is the accumulator map with the two fields exchanged,
which is the fact the value product needs: a score element and its probability
share the same lane, the same `2 * (lane % 4)` term, and one eight-element block
index. The probability operand therefore leaves the score accumulator of the
lane that holds it with **no shuffle**: an accumulator slot at `index` becomes
the register `2 * (index / 4) + ((index / 2) % 2)` and the half `index % 2`. The
decode of that table is a bijection over 1,024 slots, none missing and none
repeated.

The consequence is checked and not only argued. All 8,192 slots of a 64 by 128
score tile have one accumulator slot and one register slot, no slot is repeated
on either side, and the two name the same lane for every one of them. So the one
cost of the probability operand is the bf16 rounding and the register move, and
the shared round trip of Part 6, section 4.4 stays gone.

### 6. Phases and gates

| Phase | Change | Gate |
|---|---|---|
| H1 | The prototype of the tile in `dev/kernels/`, as the probe of section 1 is: a resident query tile, copies of key and value, and the products with no softmax | The prototype reads 500 TFLOP/s or more with two query tiles per staged key tile, and the copy-only row states the bytes a second the copies sustain. **Measured: passed** at 755.4 TFLOP/s and 6.03 TB/s, from a binary that is not in the tree; the committed probe measures the same rows by sixteen-byte stores, at 295.4 TFLOP/s and 2.36 TB/s, and the Tensor Memory Accelerator path of H2b is what restores the recorded rate. The shared-ring rows below take the committed source to 643.1 TFLOP/s, at six query tiles behind one staged key tile |
| H2 | The forward tile of Part 7's arithmetic: mask, base-two softmax, both products, one consumer warpgroup, staged by a 16-byte copy into the interleaved form | The family test passes at the bf16 tolerance of [testing.md](testing.md) on every case of Part 7 section 7; the forward row of `attention_benchmark` reads 200 TFLOP/s or more |
| H2b | The copy path of decision H2 by the Tensor Memory Accelerator, which needs the 128-byte swizzle and descriptor layout type 1 rather than the interleaved form | A one-hot map and a laid-out reference pass on the staged path, and the rate matches the copy-free path within 15 percent at the intensity of two query tiles |
| H3 | Warp specialization and the overlap of decision H4 | The forward row reads 350 TFLOP/s or more; a trace shows the tensor pipe busy above 60 percent |
| H4 | The pair of decision H5 | The forward row reads 500 TFLOP/s or more; the key and value bytes of the pair are half of two independent blocks |
| H5 | The query gradient with the preprocess kernel of decision H6 | The family test passes; the backward row reads 250 TFLOP/s or more |
| H6 | The key and value gradient | The family test passes; the backward row reads 300 TFLOP/s or more; no atomics and no softmax backward launch |
| H7 | The dispatch, the stand-in, the documentation, and the baselines entry | `tools/nanochat lint`, the CPU tests, the family test on the card, and a baselines entry under a new name |

A phase is reported and not promoted when it misses its gate, as in Part 7
section 6.

**The H1 result.** `//dev/kernels:attention_wgmma_probe` measured these rows on
the H100 leg, over a 128-key tile of 32 KiB against a resident query tile, with
a four-stage window and the release of section 5. The staged rows name the copy
path, because the two paths do not agree.

| Row | Path | Rate | Share of the peak |
|---|---|---:|---:|
| Products from a resident key tile | none | 963.9 to 979.7 TFLOP/s | 97.5 to 99.1 percent |
| Products from a staged tile, one query tile | Tensor Memory Accelerator | 375.6 TFLOP/s | 38.0 percent |
| Products from a staged tile, two query tiles | Tensor Memory Accelerator | 755.4 TFLOP/s | 76.4 percent |
| The copies with no products | Tensor Memory Accelerator | 6.03 TB/s | none |
| Products from a staged tile, one query tile | sixteen-byte stores | 145.4 to 151.6 TFLOP/s | 14.7 to 15.3 percent |
| Products from a staged tile, two query tiles | sixteen-byte stores | 284.7 to 295.4 TFLOP/s | 28.8 to 29.9 percent |
| The copies with no products | sixteen-byte stores | 2.29 to 2.43 TB/s | none |

The two staged halves of the table are the same geometry over two copy paths.
The Tensor Memory Accelerator rows come from the binary that first measured this
phase and are not reproducible from the tree; the store rows are, and both
staged rows are exactly the copy cap of their path multiplied by the arithmetic
one staged byte buys: 6.03 TB/s at 128 FLOP per byte is 772, and 2.36 TB/s is
302. Measured, that is 755.4 of 772 and 289.0 to 295.4 of 302, which is 98 and 96
to 98 percent of what each path allows. The staged rows are the spread of five
runs; the card's clock and its neighbours move the resident row by 1.6 percent
and the copies by 4 percent. The shared-ring rows below carry that arithmetic
further, over six geometries and three runs that agree to within 0.7 percent.

The copy rows are the ones the staged rows read, because a copy row fixes the
arithmetic that one staged byte must buy:

| Tile | Arithmetic per staged byte | Cap at 6.03 TB/s | Cap at 2.36 TB/s |
|---|---:|---:|---:|
| One query tile, score product only | 64 FLOP | 386 TFLOP/s | 151 TFLOP/s |
| Two query tiles, score product only | 128 FLOP | 772 TFLOP/s | 302 TFLOP/s |
| Four query tiles, score product only | 256 FLOP | 1,543 TFLOP/s | 604 TFLOP/s |
| Six query tiles, score product only | 384 FLOP | 2,315 TFLOP/s | 906 TFLOP/s |
| The tile of section 3: two query tiles, both products, key and value staged | 262 FLOP | 1,580 TFLOP/s | 618 TFLOP/s |

**The shared ring, and what it buys.** A key tile costs 32 KiB of copies and
serves every query tile that reads it, so the arithmetic of one staged byte is
the count of query rows behind that tile. Two warpgroups of one block share one
ring, which doubles that count for the same copies and needs no new instruction.
The committed source now carries both counts as template parameters, and
`dev/kernels/wgmma_probe_report.md` section 10 is the record of three runs:

| Warpgroups | Query tiles of each | Arithmetic per staged byte | Cap | Measured | Share of the peak |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 64 FLOP | 156 TFLOP/s | 151.7 | 15.3 percent |
| 1 | 2 | 128 FLOP | 302 TFLOP/s | 293.8 | 29.7 percent |
| 1 | 3 | 192 FLOP | 453 TFLOP/s | 424.9 | 43.0 percent |
| 2 | 1 | 128 FLOP | 312 TFLOP/s | 308.5 | 31.2 percent |
| 2 | 2 | 256 FLOP | 604 TFLOP/s | 535.4 | 54.1 percent |
| 2 | 3 | 384 FLOP | 906 TFLOP/s | 643.1 | 65.0 percent |

So the copies stop being the constraint above 256 FLOP a staged byte. The
six-query-tile row reads 643.1 TFLOP/s of a 906 cap, and its copies reach only
1.67 of the 2.36 TB/s the path supplies, so the limit of that row is the issue
path the copies share with the products and not the copies. That is what H3
changes and what H2b does not. The tile of section 3 keeps 262 FLOP a staged
byte, because the value tile doubles the staged bytes along with the products,
so the store path still binds the forward tile at 604 TFLOP/s and H2b keeps its
place before H3.

The H1 gate stays where it is and is still unmet by the store path. The gate
reads two query tiles behind one staged key tile, which is 293.8 TFLOP/s on
sixteen-byte stores against 755.4 on the Tensor Memory Accelerator. The ladder
passes 500 TFLOP/s only at four query tiles behind one staged key tile, which is
a different geometry from the one the gate names.

Three further rows bound the store path. The depth of the ring does not matter
once it is deep enough to cover one tile of copies: two stages read 2.14 TB/s,
four read 2.29 to 2.36, and five read 2.30. The order of the stores does not matter
either: varying the `k` block first reads 2.29 to 2.37 TB/s and varying the row
of the core matrix first reads 2.34 to 2.43, so the interleaved form costs no
bank conflict this path can see. The per-block rate is 17.4 to 18.0 gigabytes a
second, against the 27.4 one block reaches alone, so the limit is the device and
not one scheduler.

At the production shape the issued arithmetic is 18.8 GFLOP, which is 71.8 MB of
staged tile at that intensity: 21.4 microseconds of high-bandwidth memory at
3.35 TB/s, 11.9 of the second-level cache at the measured 6.03 TB/s, and 19.0 of
compute at the peak. The tile therefore sits at the balance point, which is what
decision H5 exists to move: the pair's multicast halves those bytes.

**What the rows do not say.** Every row above is measured over operands in the
interleaved form, and two checks now stand behind that. One compares a full
product with a host reference one element at a time, which is 1.9e-07 relative
to the largest element of the reference; a row-major stage fails it, and a
sum-based check cannot. The other compares the staged product with the resident
product, and both stagings agree exactly, so the ring, its barriers, and its
stages are right. Neither check existed when the recorded rows were measured,
and the withdrawal of the old correctness row is why.

Two checks cover the second product and the register operand instead of a rate.
The value product lands six one-hot triples of six only on the transposed
staging with both transpose bits clear and an atom stride of 256; the direct
staging reaches four of six at best, over both bits and three strides. The
register operand decodes as a bijection over 1,024 slots, and its row and key
formulas are the accumulator's two formulas exchanged, so the probability of a
score tile reaches the value product from the same lane with no shuffle.

The staged rows time a score product only, so they say nothing about the second
product's operand traffic, about the cost of the transposed staging pass, about
the exponential, or about the epilogue. Those are the costs section 3 lists and
the gates of section 6 allow for.

**The H2 prototype.** The `attention-flash:` rows of the same binary price the
whole tile instead of the score product alone: the score product, the base-two
softmax in the registers, the value product from the probability operand the
score lane already holds, and the running maximum, sum, and rescale. One
`FlashStep` is the only written form of the step, so a resident row and a staged
row cannot state two chains. No phase gate is stated against them, because they
are a prototype ceiling and not the production tile.

The check runs first and it passes, for both masks. One element at a time
against the host reference of the family reads 2.09e-03 of a reference scale of
0.184 with no bad element of 8,192 for the dense step, and 1.00e-03 of a scale of
1.000 with none for the causal boundary tile. The row sums read 2.11e-07 and
1.34e-07, which is the bf16 rounding of the probability operand. The dense check
is the first end-to-end test of the register operand, whose element order the H1
rows measured with one-hot elements and never carried a full product through.
Both stagings then agree with the resident step of the same mask exactly, in all
eight geometries.

The boundary tile is the mask the forward keeps: the query block and the key tile
start at the same offset, so a row sees the keys up to its own index. It issues a
full 128-key tile for every query row and needs the triangle of it, so the row
states `work_ratio` as 256 over the query rows plus one, which is 3.94 at 64 rows
and 1.98 at 128.

| Row | Path | Rate | Share |
|---|---|---:|---:|
| The chain with no traffic, one warpgroup, one block | none | 3.37 TFLOP/s | 45.0 percent of the per-SM share |
| The chain with no traffic, one warpgroup, every SM | none | 448.2 TFLOP/s | 45.3 percent of the peak |
| The same chain, causal boundary tile | none | 446.8 TFLOP/s | 45.2 percent of the peak |
| The key ring, two stages / three stages, one warpgroup | copy engine | 2.26 / 2.43 TB/s | none |
| The key ring, two stages / three stages, two warpgroups | copy engine | 2.22 / 2.49 TB/s | none |
| The key and value rings, either depth, one warpgroup | plain stores | 0.949 / 0.943 TB/s | none |
| The key and value rings, either depth, two warpgroups | plain stores | 1.495 / 1.492 TB/s | none |
| The whole tile, 64 query rows, dense / causal | both rings | 52.84 / 53.05 TFLOP/s | 5.3 percent |
| The whole tile, 128 query rows, dense / causal | both rings | 146.1 / 137.2 TFLOP/s | 13.9 to 14.8 percent |

**The chain costs half the tensor rate, with no traffic at all.** The score
product alone reads 978.7 TFLOP/s on the same card. The whole step reads 448.2.
The instruction is not the gap; the chain is. One block reads 45.0 percent of the
per-SM share and the full grid reads 45.3 percent of the peak, so no occupancy is
lost and the serialized step is the whole of it. That is what decision H4 is for,
and the rows now price it.

**The mask is free.** The boundary tile reads 446.8 TFLOP/s against 448.2 dense,
which is 0.3 percent, and a staged row moves by 0.0 to 6.1 percent, which is
inside or beside the spread. The tile is therefore bound by the chain and not by
the arithmetic: the select per score and the exponentials it forces to zero cost
nothing measurable, so the half of the work a causal mask removes is a gain in
necessary work and not a loss in rate.

**The value ring is the new limit.** A transposition is not a contiguous
sixteen-byte move, so no copy engine carries one and the value ring is plain
strided loads and interleaved stores. It costs the mixed path 0.949 TB/s at one
warpgroup and 1.495 at two, against 2.26 to 2.49 TB/s for the key ring alone.
Doubling the staged bytes costs 5.2 times the time at one warpgroup. Two
warpgroups recover part of it, because more threads cover the strided reads.
This is the reason decision H2b must reach the value tile and not only the key
tile.

**Ring depth belongs to the key ring only.** The key ring gains 7.7 percent at one
warpgroup from a third stage, 2.255 to 2.429 TB/s, and 12 percent at two, 2.224 to
2.493. The mixed ring does not move with depth at all. So the third stage is what
the copy engine needs and the value staging ignores, which is the same fact as the
paragraph above from the other side.

**The staged tile reaches 72 to 88 percent of its cap.** At one warpgroup the cap
is 0.949 TB/s times 64 FLOP a byte, which is 60.7 TFLOP/s, and the dense row reads
52.84, which is 87.0 percent. At two warpgroups the cap is 1.495 times 128, which
is 191.4, and the dense row reads 146.1, which is 76.3 percent. The score-only
staged rows reach 97 to 99 percent of their own cap, so the missing share is the
softmax and the value product. Against those rows at the same reuse, the whole
tile is a half at 128 query rows, 295.4 against 146.1, and a third at 64, 150.1
against 52.84.

Runs of this source before the mask axis agree with every row within 1.3 percent,
which is the spread of the card and not of the code. The checks are identical
between runs, and the operands are fixed, so the step is deterministic.

### 7. Dispatch

The new path takes a shape when the Part 7 predicate holds **and** the card
reports a compute capability of 9.0 or more **and** the driver answers for a
cluster of two. `attention.cu` selects it before the family, so a card that
cannot launch a cluster keeps the family and no other path.

The two backward predicates stay one: a shape that the forward takes, the
backward takes. That is why section 8 keeps the statistics layout unchanged.

### 8. Risks

- **The instruction is architecture conditional.** The file compiles at
  `sm_90a` only, so the default `//...` build and the emulation engine never
  compile or run it. The pattern of `attention_flash_absent.cc` keeps the other
  builds linking, and the gate is the card test of the H100 leg.
- **Two kernels produce one statistic.** The family and the new path write the
  same two numbers per row, and the backward predicates cannot tell them apart.
  The layout of `AttentionStatsCount` therefore stays, and the two paths must
  agree in units and in scale.
- **A cluster can refuse to form.** A card that reports `sm_90` may answer false
  for a cluster of two, as a partitioned instance does. The dispatch asks the
  driver and falls back, rather than failing a launch.
- **The register budget is the whole budget.** Two 64-row accumulators and the
  probability operand leave no room, and one more stage of the key tile spills.
  `setmaxnreg` is what makes the two roles fit in one block.
- **The probe is an upper bound and not a forecast.** It issues products from a
  resident tile with no softmax and no global traffic. The softmax of a 128-key
tile is one vector pass over 16,384 values against two products of the same
size, so the tile cannot reach the probe's rate.
- **The seam change is a frozen header.** Decision H6 moves four definition
  sites, one of which is another workstream's.
- **The simulator has no leg.** The emulation engine maps no warpgroup
  instruction, as [simulator.md](simulator.md) records, so this path's
  correctness rests on the card test and the host reference.

## See also

- [performance.md](performance.md) — the measurement protocol.
- [performance-model.md](performance-model.md) — the bound that judges a tile.
- [precision.md](precision.md) — the fp32 and bf16 policy.
- [precision.md](precision.md) — the fp8 operand rules.
- [host-portability.md](host-portability.md) Part 1 — the memory margin and the node plan.
- [simulator.md](simulator.md) — the emulation engine limits.
