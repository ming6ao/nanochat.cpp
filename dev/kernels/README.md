# dev/kernels — unpromoted prototypes

This directory holds work that is not part of the shipped backend. The
per-family correctness tests and the micro-benchmarks moved next to the kernels
in [backends/cuda/kernels](../../backends/cuda/kernels/README.md). This directory
keeps the combined QkPrep fusion prototype and the decode-path CUDA-graph
baseline.

The canonical tiers, tagging, and resource rules are in
[docs/testing.md](../../docs/testing.md) and [AGENTS.md](../../AGENTS.md) §5.
Run everything through the project entry point:

```bash
tools/nanochat test                       # T0 CPU tests (excludes tag `gpu`)
tools/nanochat test --gpu //...           # T1 GPU tests (tag `gpu`)
tools/nanochat bench -- <benchmark>       # T3-style micro-benchmark under the broker
```

## Layout

| File | What |
|---|---|
| `qk_prep_fused.h`, `qk_prep_fused.cu` | Combined fused QkPrep prototype (one launch over q and k) plus the explicit decomposed baseline. |
| `qk_prep_fused_test.cc` | Fusion prototype correctness against the host `row_ref.h` and the seam `QkPrep`. |
| `qk_prep_bench.cc` | Decomposed vs seam-fused vs combined-fused QkPrep, tiny and medium shapes. |
| `decode_kernels.h`, `decode_kernels.cu` | Capture-safe (stream-taking) mirrors of the decode-path seam ops. |
| `decode_fused_bench.cc` | Batch-1 decode baseline: eager seam path vs `cuBLAS + CUDA Graphs`. |
| `build_config_test.cc` | CPU smoke test for the precision config and the sandbox guard. |
| `cpu_gemm_bench.cc` | CPU reference GEMM micro-benchmark over the four dominant training shapes plus the model-layout forward shapes; emits `nanochat.bench.v3`. |
| `cpu_linear_layout_probe.cc` | CPU GEMM probe over the model operand layouts (`transpose_b` true and false); emits `nanochat.bench.v3`. |
| `muon_fused_gemm.h`, `muon_fused_gemm.cuh`, `muon_fused_gemm.cu` | Muon fusion prototype (Phase 1): the device-side batched fp32 GEMM of one Polar Express iteration, all five operand forms. |
| `muon_fused_gemm_test.cc` | Five-form correctness of the prototype against the seam GEMM (cuBLAS), including the wide-branch side selector. |
| `muon_fused_gemm_bench.cc` | Hand GEMM versus cuBLAS at the groups the fusion targets, with the section 4 gate share. |
| `muon_launch_sync_bench.cu` | Phase 0 sync term: launch gap versus cooperative grid-barrier cost on this card. |
| `wgmma_bench.cu` | Phase 0 of [docs/kernels.md](../../docs/kernels.md) Part 3 and Phase H1 of [docs/attention.md](../../docs/attention.md) Part 8, in one binary: the `wgmma:` ceiling rows of the warpgroup rate on a resident tile, the accumulator, operand, and register maps of `wgmma.mma_async.m64n128k16` checked one element at a time, and the `attention-wgmma:` rows of a staged key tile. The staged rows sweep the arithmetic that one staged byte buys: the warpgroups of one block and the query tiles of each, over two copy orders and two source windows, beside a copy-only row and the depth of the ring. It also carries the `attention-flash:` rows of the resident flash tile, which check their product against the sequence reference and state the cost of the serialized score-softmax-value chain. `--rows phase0|h1|flash|all` selects a row set. Needs `sm_90a`. |

## Conventions

The shared conventions live in
[backends/cuda/kernels/README.md](../../backends/cuda/kernels/README.md): tiny
shapes, a host reference, finite differences, fixed seeds, the `gpu` tag, and
the benchmark report.

These prototypes are dev-only. The CUDA backend does not depend on this
directory.

## CPU GEMM micro-benchmark (Phase 0)

`cpu_gemm_bench.cc` times the four dominant GEMM shapes from
Part 2, section 2 of [docs/performance.md](../../docs/performance.md). The rows are
`gemm_forward`, `gemm_wgrad`, `gemm_dgrad`, and `mlp_gemm`. The benchmark warms
up, then keeps the best of N rounds. It calls `kernels::Gemm` directly and
prints the `nanochat.bench.v3` schema on standard output. The shipped CPU GEMM
is scalar, so the run is single-threaded until Phase 1 adds OpenMP.

```bash
tools/nanochat build //dev/kernels:cpu_gemm_bench
tools/nanochat bench -- ./bazel-bin/dev/kernels/cpu_gemm_bench
tools/nanochat bench -- ./bazel-bin/dev/kernels/cpu_gemm_bench \
  --json --out /tmp/cpu-gemm.json
```

The `--json` flag suppresses the human-readable table and emits the report on
standard output. The `--out PATH` flag also writes the report to `PATH`.

The `gemm_dgrad` shape is the slowest at roughly 70 seconds per call, so a
small `--rounds` keeps a full run short. The benchmark reports the active GEMM
thread count in the `threads` field and on standard error. Phase 0 is scalar,
so the value is 1.

## CPU operand-layout probe (Phase 1)

`cpu_linear_layout_probe.cc` times the same logical GEMM under
`transpose_b = false` and `transpose_b = true`. The model uses
`transpose_b = true`: `ops::LinearForward` stores weights as `[out, in]`.

Phase 1 first read `b` contiguously only when `transpose_b` is false. That run
failed the end-to-end gate at 44.1 seconds per step. The model-layout fix
rewrites the `transpose_b = true` branch as a dot product, so both reads are
contiguous.

The probe now confirms that the two layouts agree. The model `lm_head` forward
reaches 19.35 GFLOP/s and the benchmark layout reaches 19.55 GFLOP/s (12
threads). The Phase 1 gate passes at 22.3 seconds per step. See
[docs/performance.md](../../docs/performance.md).

```bash
tools/nanochat build //dev/kernels:cpu_linear_layout_probe
tools/nanochat bench -- ./bazel-bin/dev/kernels/cpu_linear_layout_probe \
  --rounds 5 --warmup 2 --json --out /tmp/cpu-gap-p1-layout.json
```

## Fusion results (Wave 8, commit `62706c0`)

Measured with `tools/nanochat bench -- ./bazel-bin/dev/kernels/qk_prep_bench`.
Tiny is `B=2, T=8, heads=2, kv_heads=2, d=4` (200 iterations); medium is
`B=8, T=512, heads=8, kv_heads=8, d=64` (100 iterations). Each timing is the best
of three rounds. "decomposed" is `RmsNormForward` + a standalone RoPE/scale
kernel (4 launches for q and k); "seam fused" is the shipped
`kernels::QkPrepForward`/`Backward` (2 launches); "combined fused" is the
dev/kernels prototype (1 launch over q and k).

The rows were measured on a card the tree no longer supports, so they are
removed. The conclusions they carried are:

- The combined fusion buys most on the launch-bound tiny shape, in both
  directions.
- On the medium shape it buys on the forward, where it avoids the HBM round trip
  of the normalized row.
- The medium backward is memory-bound, and the two paths tie there.

**§3 verdict — all five conditions hold.** (1) *Decomposition match*: the
RMSNorm reduction is over `head_dim` and is contained in one tile row; the
combined kernel only extends the grid to cover q then k, with the same
per-row ownership. (2) *Residency*: the normalized row stays in registers
across RoPE, so the decomposed `q_normed`/`k_normed` (and backward
`dq_normed`/`dk_normed`) HBM traffic disappears. (3) *No library
displacement*: neither RMSNorm nor RoPE is a cuBLAS/cuDNN op. (4) *Occupancy
survives*: the fused kernel uses the same block reduction and shared-memory
budget as `RmsNormForward`, and it matches or beats the decomposed path.
(5) *Derivable backward*: derived as a unit and finite-differenced in
`qk_prep_fused_test.cc` (max abs error within tolerance).

Decision: legal and worth promoting. The per-tensor fusion already ships in
`backends/cuda/kernels/qk_prep.cu`; the *combined* single-launch variant is the
extra win on launch-bound shapes and needs a `backends/cuda` change to promote
(out of scope here — follow-up).

## Batch-1 decode baseline (Wave 8, commit `62706c0`)

Measured with `tools/nanochat bench -- ./bazel-bin/dev/kernels/decode_fused_bench`.
Config: `layers=6, d_model=384, heads=6, head_dim=64, kv_heads=6, d_ff=1536,
vocab=4096, seq_len=128`. The step is a full batch-1 decode token: embedding,
per-layer RMSNorm/QKV GEMM/QkPrep/KV-cache append/attention/output GEMM/residual,
MLP GEMM + relu^2 + residual, final norm, and the language-model head.

The benchmark compares the eager seam path on the legacy default stream against
the same step captured on a private stream as `cuBLAS + CUDA Graphs`. The rows
were measured on a card the tree no longer supports, so they are removed. The
conclusion they carried is a speedup of about a tenth for the graph path, with
the graph path numerically identical to the eager seam path (max abs logit
error `0`).

CUDA forbids capturing the legacy default stream, and the CUDA backend hardcodes
it (`backends/cuda/device.cu`), so the graph is built with direct cuBLAS on a
private stream plus the stream-taking mirrors in `decode_kernels.cu`; capturing
the seam itself would need a backend stream change (follow-up). The whole-model
persistent megakernel is a Turing-and-newer (`sm_75` and up) experiment, so only
the baseline is reported here.

## Attention window promotion results (Phase C)

The campaign promotes the windowed forward only. The dispatch predicate lives
in `backends/cuda/kernels/attention.cu`:

```c
bool UseFusedAttention(const AttentionParams& p) {
  return !kComputeIsBf16 && p.window_left >= 0 && p.head_dim == 128;
}
```

The fused tiled forward serves a sliding-window shape when the head dimension
is 128. Full-causal attention (`window_left < 0`) and every other head
dimension stay on the cuBLAS path. A bf16 build stays on the cuBLAS path too,
because the scalar FMA tile loses to the tensor cores there
([../../docs/attention.md](../../docs/attention.md)). Every
backward call stays on the cuBLAS path.

The measurement command is:

```bash
tools/nanochat profile --json --out /tmp/attention-window.json
```

The command runs the shipped `attention_benchmark`. The rows were measured on a
card the tree no longer supports, so they are removed, and a rerun on a supported
card is required before a promotion claim.

**Gate verdict: PASS at the time of the promotion.** The windowed forward beat
the frozen dense baseline on the same shape. The mixed-pattern forward beat the
frozen mixed-pattern baseline. The full-causal forward stays on cuBLAS and does
not regress. The backward rows stay on cuBLAS and do not regress.

The full-causal fused tile lost the Phase 1 gate. Do not widen the dispatch
predicate to full causal without a new measurement.

## Muon fusion prototype (Phase 0 and Phase 1)

`docs/kernels.md` asks whether one Polar Express iteration of
`MuonUpdate` can run as a single device-side launch. Today the iteration is
five dependent launches: `A = X^T X` (or `X X^T`), `A2 = A A`, the elementwise
`B = cb*A + cc*A2`, `prod = X B` (or `B X`), and the elementwise
`X = ca*X + prod` (backends/cuda/kernels/muon.cu). A fused launch would carry
the three products and fold both elementwise passes into their neighbours, with
a grid barrier between the stages.

The rows below were measured on a card the tree no longer supports, so the
numbers are removed. The commands and the method are still correct, and a rerun
on a supported card is required before a claim.

### Phase 0: the launch and elementwise share of the step

From the existing `muon_benchmark` rows at the two profiles the plan measures
(`docs/post-training.md` Part 3, section 5), with `--muon`:

```bash
tools/nanochat bench -- bazel-bin/backends/cuda/kernels/muon_benchmark \
  --profile h100 --muon --json
tools/nanochat bench -- bazel-bin/backends/cuda/kernels/muon_benchmark \
  --profile big --muon --json
```

Three rows give the share without any new kernel. `polar:<shape>:step` is the
loop's fifteen batched GEMMs alone, `muon:<shape>:step` at `ns=0` is the rest of
the optimizer phase, and `muon:<shape>:step` at `ns=5` is the whole step, so

```text
loop                 = muon:step(ns=5) - muon:step(ns=0)
launch + elementwise = loop - polar:step
share of the step    = (launch + elementwise) / muon:step(ns=5)
```

The removed rows put every value above the 2 percent floor of the section 3
inequality. Phase 1 was therefore written instead of stopping on a negative
result.

### Phase 0: the sync term

The launch gap and the cooperative grid barrier are measured with
`muon_launch_sync_bench`:

```bash
tools/nanochat bench -- bazel-bin/dev/kernels/muon_launch_sync_bench --json
```

The removed rows show that a dependent launch of a full batched grid costs
about one order of magnitude more than a grid barrier at the resident grid. One
fused iteration therefore replaces a launch gap with one launch plus three
barriers. The gain is less than one percent of an iteration at either
profile. The launch gaps the fusion removes
are a rounding error against the elementwise passes it also removes: the Phase 0
share above is elementwise HBM traffic, not dispatch.

### Phase 1: the device-side batched GEMM and its gate

`muon_fused_gemm.cuh` is the section 4 schedule: 256 threads under
`__launch_bounds__(256, 3)`, block tile 128 by 64, micro-tile 8 by 4 per thread
(32 fp32 accumulators, 16 by 16 thread groups), two double-buffered 8-deep k
slices in 12288 bytes of shared memory, and one flat `(matrix, tile_i, tile_j)`
index per block. `ptxas` reports 80 registers per thread and no local-memory
traffic in the inner loop (`#pragma unroll 2`; unrolling the whole slice spills
at this occupancy).

The five operand forms are resolved by `MuonPolarGemm`:
`XtX` (`transpose_a`), `XXt` (`transpose_b`), `AA` (plain), `XB` (plain, the
square operand on the right), and the wide branch `BX`, which is also plain by
plain and is therefore named by the `MuonSquareSide` selector rather than by a
transpose flag. `muon_fused_gemm_test.cc` checks all five against the seam
`kernels::Gemm`, which is the cuBLAS call `MuonUpdate` issues, on six groups
(tall, wide, whole 128 by 64 tiles so the `float4` loader path runs, ragged
tiles so the scalar gather runs, and a square group), at the family tolerance.
The benchmark:

```bash
tools/nanochat build //dev/kernels:muon_fused_gemm_bench
tools/nanochat test --gpu //dev/kernels:muon_fused_gemm_gpu_test
tools/nanochat bench -- bazel-bin/dev/kernels/muon_fused_gemm_bench --json
```

`hand` is the prototype and `cublas` is the seam GEMM; `A`, `A2`, and `prod` are
the three products of one iteration at the group's shapes, and `iter` runs the
three in sequence.

**Gate verdict: FAIL on the target group.** The section 4 gate asks for at least
85 percent of the cuBLAS rate on a group the fusion targets. The 640x1280
key/value group stayed well below that fraction, and the prototype was the
slower path on every stage of it. The largest group, 1280x5120x1280, came close
to the gate, but only because that product is bandwidth-bound and cuBLAS is weak
on it too.

**Decision: do not promote the fused GEMM.** The section 3 inequality is
negative on the measured card. The fusion could win at most the launch gaps it
removes, plus the elementwise passes folded into the neighbouring GEMM
epilogues. The hand GEMM gives back more than that on every large group, so the
loss is an order of magnitude larger than the win.

The prototype also shows *why*: 80 registers per thread, 12 KB of shared
memory, and three blocks per SM leave 768 of 2048 threads resident, the 8 by 4
micro-tile reads 12 shared floats per 32 FMAs, and an 8-deep k slice exposes one
global round trip per ~1500 cycles of arithmetic per SM. A schedule that changes
those three numbers (8 by 8 micro-tile, 16 to 32-deep slices, a 256 by 64 tile)
is a different prototype and a different measurement; nothing here says the
fusion is impossible, only that this schedule is well off cuBLAS and the launch
gaps it removes are worth a fraction of a percent.

## The WGMMA probe (Part 3 phase 0 and Part 8 phase H1)

[docs/kernels.md](../../docs/kernels.md) Part 3 asks whether a hand-written
Hopper mainloop can hold the cuBLAS rate on the six rows that matter. Phase 0
before that work asks one narrower question: how fast can one warpgroup issue
the warpgroup matrix multiply-accumulate from a resident shared-memory tile? If
one warpgroup cannot hold the per-SM rate, no mainloop built on it can hold the
card rate, and the campaign stops without writing one.
[docs/attention.md](../../docs/attention.md) Part 8 then asks what the attention
tile needs from the same instruction: the operand form it reads, the map of its
accumulator and register operands, and the rate it holds when the key tile is
staged by a copy path.

One file answers both: `wgmma_bench.cu` absorbed the Phase 0 ceiling
probe, so there is one build and one card job pair instead of two. It uses three
hand-written pieces and no library:

- the shared-memory matrix descriptor, encoded from the sixteen-byte fields the
  instruction reads;
- the interleaved K-major staging layout, which is the descriptor form whose
  address arithmetic is short enough to check by hand;
- the instruction itself, `wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16`,
  with its sixty-four fp32 accumulators.

The probe checks the product one element at a time against a host reference
before it reports a rate, so a descriptor that reads the wrong element fails the
check instead of producing a number. It then reports four resident rows. Eight
atoms per pass, which is the head width, on one block and on one block per SM,
are the H1 rows of Part 8. Four atoms per pass on one block, and on two per SM,
are the Phase 0 rows of Part 3. The two prefixes keep the two card jobs apart.

**The Phase 0 gate is a rate for each SM, not a rate for the card.** One
warpgroup cannot exceed its SM's share of the peak, which is 989 TFLOP/s over
132 SMs, or 7.49 TFLOP/s. The gate reads 85 percent of that share on the
`wgmma:1block` row. The `wgmma:grid` row cannot answer it, because two
warpgroups fill an SM whether or not either one could fill it alone.

WGMMA is architecture conditional and the tree's device configurations emit
`sm_90`, so the one target states `archs = ["sm_90a"]` for itself and no build
command carries a flag. It carries the `gpu` and `manual` tags, and the
emulation engine maps no WGMMA, so there is no off-device leg.

**The timing is the shared protocol.** `wgmma_bench` times through
`backends/cuda/kernels/testing/bench_protocol.h`, as
`backends/cuda/gemm/gemm_benchmark.cc` does, so a row here and a row of the
sweep state the same statistic. One call is one launch. `--iters` calls fill one
window, and `--rounds` windows give the samples.

- `--warmup N` states the calls of the warm-up, which no window times.
- `--no-flush` removes the flush of the second-level cache.

The flush is on by default, because a staged window of 512 key tiles is 16 MiB
and stays warm across calls. Its own cost is measured once and removed from the
window. Each row states the minimum under its own name and the median in a
`:median` companion row. The console prints both numbers on one line, and the
JSON document holds the pair as two rows.

```bash
tools/nanochat build //dev/kernels:wgmma_bench
```

On a card the host cannot reach, the one Modal card job builds and runs it. The
job states a `nanochat.bench.v3` document, and `--rows` selects the row set:

```bash
modal run tools/modal/app.py::card_job --name wgmma_bench \
  --argv '["--rows","phase0","--json"]'
modal run tools/modal/app.py::card_job --name wgmma_bench \
  --argv '["--rows","h1","--json"]'
```

## Adding a family

See [backends/cuda/kernels/README.md](../../backends/cuda/kernels/README.md). The
per-family tests and benchmarks live next to the kernel, not here.
