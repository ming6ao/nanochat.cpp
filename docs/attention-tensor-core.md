# Tensor-core attention

Status: proposed. This document is Part 5 of the attention work. Parts 1 to 4
are in [attention.md](attention.md): the Pascal baseline, the feasibility
study, the windowed-forward promotion, and the promotion results.

The subject of this document is the next step. The promoted fused forward is an
fp32 tile. It runs on the CUDA cores. The bf16 and fp8 builds move every other
matrix multiply to the tensor cores. This document decides what happens to the
fused tile in those builds.

- [attention.md](attention.md) — the baseline and the promoted tile.
- [fp8-training.md](fp8-training.md) — the fp8 operand rules.
- [performance-model.md](performance-model.md) — the bound that judges the
  options.
- [simulator-precision.md](simulator-precision.md) — the emulator limits.

## 1. Goal and scope

**Goal.** Make attention run at the tensor-core rate in the bf16 and fp8 builds,
and keep the sliding-window saving where it pays.

**In scope.**

- The decision on the fused fp32 tile in a non-fp32 build.
- The design of a tensor-core fused tile, if the decision keeps one.
- The fp8 operands for the scores and the probability-value product.
- The gates, including the ones that run without a GPU.

**Out of scope.**

- The backward pass. It stays on the cuBLAS decomposition.
- A split `head_dim` design for full-causal attention. Part 2 rejected it.
- Attention variants that are not in the model.
- A cycle-accurate model of the tile.

## 2. What the tree has today

The forward path has two branches, selected by
`UseFusedAttention` in `backends/cuda/kernels/attention.cu`:

```c
bool UseFusedAttention(const AttentionParams& p) {
  return p.window_left >= 0 && p.head_dim == 128;
}
```

| Branch | Shape | Engine |
|---|---|---|
| Fused windowed forward | `window_left >= 0` and `head_dim == 128` | An fp32 FMA tile on the CUDA cores |
| cuBLAS forward | full-causal, and every other shape | Batched cuBLAS GEMMs plus a row softmax |
| Backward, windowed | `window_left > 0` and a causal mask | A banded query-block loop, on the CUDA cores |
| Backward, other | every other shape | Four batched cuBLAS GEMMs |

The fused tile uses `kFusedThreads = 256`, `kFusedBr = 32`, `kFusedBc = 32`, and
two resident blocks per streaming multiprocessor. Its shared memory at
`head_dim == 128` is 37,120 bytes in fp32.

The model uses the `SSSL` pattern. At depth 24, 18 of the 24 layers are windowed
and 6 are full-causal. Three quarters of the layers therefore take the fused
tile.

The cuBLAS forward materializes a `[batch, heads, seq, kv_len]` score matrix.
It uses a cached device scratch, `CachedBuffer(0, 3 * matrix)`, which sits
outside the training arena. At depth 24, a batch of 16, and a sequence of 2048,
that request is 4.50 GiB in bf16 and 9.00 GiB in fp32.

Part 4 measured the promotion in fp32 on the T4. The windowed
forward beat the fp32 cuBLAS path by 1.30x on the `SSSL` shape. That comparison
held the rate constant, because both paths ran on the CUDA cores.

## 3. Why the fp32 tile changes value

The fp32 result does not carry over to a bf16 or fp8 build. The tile stays on
the CUDA cores. The cuBLAS path moves to the tensor cores. The rate ratio
between the two paths becomes large.

Take the depth-24 shape: 24 layers, 12 heads, head width 128, sequence 2048, and
`SSSL`.

| Path | Attention arithmetic per token | Engine rate | Time per token |
|---|---:|---:|---:|
| Fused windowed tile, fp32 | 0.396 GFLOP | 20 to 40 TFLOP/s | 9.9 to 19.8 microseconds |
| Dense cuBLAS, bf16 | 0.906 GFLOP | 791 TFLOP/s | 1.15 microseconds |
| Dense cuBLAS, fp8 | 0.906 GFLOP | 1,187 TFLOP/s | 0.76 microseconds |

The windowed tile does 2.29 times less arithmetic, and still loses by a large
factor. The cause is the engine, not the count.

The dense path pays a cost for that speed: memory traffic over the score matrix.
At depth 24, one micro-batch of one layer moves about 6 GiB through the score
and probability matrices in bf16. Over 24 layers that is about 46 milliseconds
per micro-batch, or about 185 milliseconds per optimizer step at a gradient
accumulation of 4. The reference step is about 1,067 milliseconds, so the
traffic is about 17 percent of the step.

The conclusion is narrow and clear: **do not carry the fp32 tile into a bf16 or
fp8 build without a new measurement.**

## 4. The three options

| Option | What it does | Arithmetic | Off-GPU verifiable |
|---|---|---|---|
| A. Tensor-core tile | Rewrite the tile with `mma` or `wgmma` instructions | Lowest | No, without a new emulator feature |
| B. Retire the tile | Send the windowed forward to the cuBLAS path in the non-fp32 builds | Highest, at the tensor-core rate | Yes |
| C. Keep the fp32 tile | No change | Lowest count, at the CUDA-core rate | Yes |

The recommendation is **option B first, then A behind a measurement gate**.

Why option B first:

1. It removes code rather than adding it. The dispatch predicate becomes a
   build condition or an empty function.
2. It needs no new kernel. The cuBLAS path already exists, and the reference
   engine already verifies it.
3. The fp8 and bf16 operands work through the existing GEMM decomposition. No
   new tile arithmetic is necessary.
4. The emulation engine cannot check an `mma` tile today. Section 5.2 of
   [simulator.md](simulator.md) lists `extern __shared__` as unsupported, and
   the emulator has no tensor-core primitive.

Why option A stays open: the windowed tile avoids 2.29 times the arithmetic and
most of the memory traffic. If a tensor-core tile reaches even 40 percent of the
bf16 peak, the windowed layers cost 0.21 microseconds per token instead of
0.29. That is a real gain at a long sequence. Promote it only after a
measurement.

There is no window in the fp32 build for the change. Keep option C in the fp32
build, because the promoted tile already wins there.

## 5. The option A design

This section describes a tensor-core tile. The design is a proposal. Build it in
`dev/kernels/` first, as `docs/performance.md` section 3.2 requires.

### 5.1 The forward tile

One thread block owns a block of query rows for one `(batch, head)` pair.

| Item | Value | Reason |
|---|---|---|
| Query rows per block | 64 or 128 | The tensor-core tile wants a taller query block than the fp32 tile |
| Key rows per tile | 64 | Two `mma` steps of 16 rows, or one `wgmma` step |
| Resident blocks | 2 | Occupancy, and the shared-memory budget |
| Score accumulator | fp32 registers | The online-softmax state needs fp32 |
| Output accumulator | fp32 registers | The accumulation spans every key tile |

The three steps of the online softmax stay in fp32: the row maximum, the
exponential, and the running sum. Only the two matrix multiplies change engine.

The probability matrix needs a reduced store type for the second multiply. Use
the compute type: bf16 in the bf16 build, E4M3 in the fp8 build. Store the
shifted exponential in that type after the fp32 reduction.

### 5.2 The backward pass

Keep the backward on cuBLAS. Two reasons:

1. The backward already uses cuBLAS for every shape. In a bf16 or fp8 build it
   therefore already runs at the tensor-core rate. No rewrite is necessary to
   get the rate.
2. The banded windowed backward already avoids the dense score matrix. The
   saving already exists.

If a later measurement shows the backward is the limit, revisit this decision in
a new document. Do not couple the backward rewrite to the forward decision.

### 5.3 The transposes

The cuBLAS forward transposes `q`, `k`, and `v` from the model layout
`[batch, seq, heads, dim]` into `[batch, heads, seq, dim]`, and transposes the
output back. Those copies are not free.

A fused tile can read the model layout directly. A lane reads one row segment,
so the load coalesces without a transpose. Account for this in the
performance model as a saved pass, not as arithmetic.

### 5.4 The window handling

Keep the masked-tile skip. A tile whose keys are all left of the window start is
invisible to every query row in the block. Skip it before the load. This is the
saving that makes the windowed tile worth having.

Keep the predicate narrow. It must require a window and the supported head
width, as the current predicate does.

## 6. The fp8 coupling

The fp8 operands need no new attention kernel in option B. The two matrix
multiplies become scaled fp8 GEMMs through the seam from
[fp8-training.md](fp8-training.md).

| Tensor | Build | Reason |
|---|---|---|
| `q`, `k` operands | E4M3 | Input to the score multiply |
| Scores | fp32 | The softmax needs the range |
| Probability matrix | E4M3 or the compute type | Input to the value multiply |
| `v` operand | E4M3 | Input to the value multiply |
| Output accumulator | fp32 | The accumulation spans every key tile |
| Row maximum, row sum, statistics | fp32 | A reduction with an exponential |

Use one scale per tensor for the saved probability matrix. Do not reuse the
score scale, because the exponential changes the range.

If option A is later built, the fp8 tile needs the same three rules. The
probability store needs a scale, and the row statistics stay fp32.

## 7. Invariants

1. The row maximum, the row sum, and the saved statistics are fp32.
2. The output accumulator is fp32.
3. The causal and window mask is exact. A skipped tile must be provably
   invisible to every query row in the block.
4. The dispatch predicate stays narrow and tested.
5. The kernel seam holds no vendor type.
6. The fp32 build keeps the promoted tile, because it wins there.
7. Every new tile has a kill gate with a measured number.

## 8. Deliverables and owners

| Deliverable | Path | Owner |
|---|---|---|
| The build condition for the fp32 tile | `backends/cuda/kernels/attention.cu` | Kernel agents |
| The tensor-core tile prototype | `dev/kernels/` | Kernel agents |
| The scaled fp8 operands | `backends/cuda/kernels/attention.cu` | Kernel agents |
| The attention benchmark rows | `backends/cuda/kernels/attention_benchmark.cc` | Kernel agents |
| The dispatch test | `backends/cuda/kernels/attention_test.cc` | Oracle |
| The emulator support for the tile | `tools/cuda_sim/emu/` | Simulator |
| The bounds and the measured rows | `docs/performance-model.md` | Architect |
| This document | `docs/attention-tensor-core.md` | Architect |

## 9. Validation

1. The parity gates from Part 4 pass on the changed path.
2. The dispatch test confirms which layers take which branch at depth 24.
3. The attention benchmark reports the `SSSL` row and the full-causal row.
4. The emulated tile matches the sequence reference, when the emulator supports
   the tile.
5. The end-to-end evaluation shows a tokens-per-second win on the `SSSL` shape.
6. The measured row goes into the attainment record in
   [performance-model.md](performance-model.md) with its fraction of the bound.

Gate 4 is the hard one. Without an emulator feature for `mma` and
`extern __shared__`, gate 4 needs a real card. Plan for one.

## 10. Phases

| Phase | Work | Gate | GPU |
|---|---|---|---|
| A0 | Measure the fp32 tile and the cuBLAS path on the `SSSL` shape in the bf16 build | The two rows exist in the benchmark output | no |
| A1 | Gate the tile on the fp32 build, and send the other builds to cuBLAS | The parity gates pass; the benchmark shows no fp32 regression | no |
| A2 | Add fp8 operands to the attention GEMMs | The emulated GEMM matches the CPU reference | no |
| A3 | Build the tensor-core tile in `dev/kernels/` | It beats the cuBLAS row by the phase gate margin | yes |
| A4 | Promote, or delete the prototype | The record is in the benchmark and in this document | yes |

The kill gate for A3: the tile must beat the dense cuBLAS row on the `SSSL`
shape by 10 percent at the production sequence length. If it does not, keep
option B and delete the prototype.

## 11. Status by section

| Section | Status |
|---|---|
| 3 The value change | Proposed, with an estimate |
| 4 The options | Proposed |
| 5 The option A design | Proposed |
| 6 The fp8 coupling | Proposed |
| 10 Phases | Not started |

## 12. Risks and non-goals

- **The bound may be wrong.** Measure A0 before A1. The bound uses the device
  peaks from the profile table, not a measurement.
- **The emulator cannot check an `mma` tile.** Budget one hardware leg.
- **The window saving may vanish.** The arithmetic saving is real. The time
  saving depends on the tile rate.
- **A wide dispatch predicate is dangerous.** Past experience in Part 2 shows a
  tile that wins on one shape and loses on another.
- **Not a goal:** a backward rewrite, a split `head_dim` tile, and a
  cycle-accurate emulator.

## 13. Open questions

1. Does the full-causal forward also move to a tile, or stay on cuBLAS?
2. Does the probability matrix use E4M3, or the compute type?
3. Is 64 or 128 the better query block at `head_dim` 128 with `wgmma`?
4. Does the emulator gain an `mma` and `extern __shared__` feature, or does the
   tile wait for hardware?

## See also

- [attention.md](attention.md) — the baseline, the study, and the promotion.
- [fp8-training.md](fp8-training.md) — the fp8 operand rules.
- [performance-model.md](performance-model.md) — the bound.
- [simulator.md](simulator.md) — the emulation engine limits.
