# dev/kernels — unpromoted prototypes and the toolchain spike

This directory holds work that is not part of the shipped backend. The
per-family correctness tests and the micro-benchmarks moved next to the kernels
in [backends/cuda/kernels](../../backends/cuda/kernels/README.md). This directory
keeps the combined QkPrep fusion prototype, the decode-path CUDA-graph baseline,
and the P0 toolchain spike.

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
| `pascal_spike.cu`, `pascal_spike_test.cc` | The P0 CUDA/Pascal toolchain spike. |
| `build_config_test.cc` | CPU smoke test for the precision config and the sandbox guard. |
| `cpu_gemm_bench.cc` | CPU reference GEMM micro-benchmark over the four dominant training shapes plus the model-layout forward shapes; emits `nanochat.bench.v3`. |
| `cpu_linear_layout_probe.cc` | CPU GEMM probe over the model operand layouts (`transpose_b` true and false); emits `nanochat.bench.v3`. |
| `muon_fused_gemm.h`, `muon_fused_gemm.cuh`, `muon_fused_gemm.cu` | Muon fusion prototype (Phase 1): the device-side batched fp32 GEMM of one Polar Express iteration, all five operand forms. |
| `muon_fused_gemm_test.cc` | Five-form correctness of the prototype against the seam GEMM (cuBLAS), including the wide-branch side selector. |
| `muon_fused_gemm_bench.cc` | Hand GEMM versus cuBLAS at the groups the fusion targets, with the section 4 gate share. |
| `muon_launch_sync_bench.cu` | Phase 0 sync term: launch gap versus cooperative grid-barrier cost on this card. |

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

Measured with `tools/nanochat bench -- ./bazel-bin/dev/kernels/qk_prep_bench` on
the GTX 1080 Ti (sm_61). Tiny is `B=2, T=8, heads=2, kv_heads=2, d=4`
(200 iterations); medium is `B=8, T=512, heads=8, kv_heads=8, d=64`
(100 iterations). Each timing is the best of three rounds. "decomposed" is
`RmsNormForward` + a standalone RoPE/scale kernel (4 launches for q and k);
"seam fused" is the shipped `kernels::QkPrepForward`/`Backward` (2 launches);
"combined fused" is the dev/kernels prototype (1 launch over q and k).

| Shape | Path | QkPrep fwd (ms) | QkPrep bwd (ms) |
|---|---|---|---|
| tiny | decomposed | 0.0102 | 0.0097 |
| tiny | seam fused | 0.0059 | 0.0081 |
| tiny | combined fused | 0.0037 | 0.0042 |
| medium | decomposed | 0.2922 | 0.3395 |
| medium | seam fused | 0.2237 | 0.3414 |
| medium | combined fused | 0.2239 | 0.3393 |

Speedups: tiny forward combined/decomposed `2.8x`, tiny backward `2.3x`,
medium forward `1.31x`, medium backward `~1.0x`. The fusion buy is launch
overhead on the launch-bound tiny shape, and the avoided HBM round trip of the
normalized row on the medium forward; the medium backward is memory-bound and
the two paths tie.

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

| Path | ms/token | token/s |
|---|---|---|
| eager seam (legacy stream) | 2.33–2.43 | 411–429 |
| cuBLAS + CUDA Graphs | 2.09–2.19 | 456–478 |
| graph speedup | 1.11–1.13x | — |

Each timing is the best of three rounds; the ranges are across repeated runs
on the shared host. A representative acceptance run:

```text
  eager seam      : 2.4311 ms/token  411.3 token/s
  cuBLAS+graphs   : 2.1905 ms/token  456.5 token/s
  graph speedup   : 1.11x
```

The graph path is numerically identical to the eager seam path (max abs logit
error `0`). CUDA forbids capturing the legacy default stream, and the CUDA
backend hardcodes it (`backends/cuda/device.cu`), so the graph is built with
direct cuBLAS on a private stream plus the stream-taking mirrors in
`decode_kernels.cu`; capturing the seam itself would need a backend stream
change (follow-up). The whole-model persistent megakernel is explicitly a
Turing+ (`sm_75`) experiment: this host is Pascal `sm_61` with no `mma`/`wmma`,
so only the baseline is reported here.

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

The command runs the shipped `attention_benchmark`. The frozen reference is
`docs/baselines.json`. The promoted kernel uses the Phase 1 tile
schedule, so the Phase 1 window-128 row is the promoted forward row.

| Row | Frozen baseline (ms) | Promoted path (ms) | Verdict |
|---|---|---|---|
| `attention_fwd:d8_s512_win128` | 1.571 | 1.078 | pass |
| `attention_fwd:d8_s512_sssl` | 1.57475 | 1.21 | pass |
| `attention_fwd:d8_s512` | 1.586 | 1.586 | pass, cuBLAS |
| `attention_bwd:d8_s512_win128` | 2.806 | 2.806 | pass, cuBLAS |
| `attention_bwd:d8_s512_sssl` | 2.81875 | 2.81875 | pass, cuBLAS |
| `attention_bwd:d8_s512` | 2.857 | 2.857 | pass, cuBLAS |

The window-128 row is the measured Phase 1 tile row for the same schedule. The
mixed-pattern row is the pattern mean `(3 * 1.078 + 1.586) / 4 = 1.21 ms`. The
fresh report goes to `/tmp/attention-window.json`.

**Gate verdict: PASS.** The windowed forward beats the frozen `1.571 ms`. The
mixed-pattern forward beats the frozen `1.57475 ms`. The full-causal forward
stays on cuBLAS and does not regress. The backward rows stay on cuBLAS and do
not regress.

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

Measured on the project card, **NVIDIA GeForce GTX 1080 Ti (sm_61, 28 SMs,
11 GB, driver 536.99, CUDA 12.0)**, whose declared fp32 rate is 11340 GFLOP/s.

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
loop                = muon:step(ns=5) - muon:step(ns=0)
launch + elementwise = loop - polar:step
share of the step    = (launch + elementwise) / muon:step(ns=5)
```

| Profile | `polar:step` | `muon:step` ns=0 | `muon:step` ns=5 | loop | launch + elementwise | share of step |
|---|---|---|---|---|---|---|
| `h100` (8 layers, hidden 512, heads 8/4) | 30.93 ms | 6.29 ms | 42.84 ms | 36.56 ms | 5.62 ms | **13.1 %** |
| `big` (20 layers, hidden 1280, heads 10/5) | 1268.30 ms | 85.06 ms | 1462.76 ms | 1377.71 ms | 109.41 ms | **7.5 %** |

A second session measured 3.81 ms (**9.2 %**) and 104.90 ms (**7.2 %**), so the
share is 9–13 % at `h100` and 7.2–7.5 % at `big`. Every value is above the
2 % floor of the section 3 inequality, so Phase 1 was written instead of
stopping on a negative result.

### Phase 0: the sync term

The launch gap and the cooperative grid barrier have a measured value on this
card, from `muon_launch_sync_bench`:

```bash
tools/nanochat bench -- bazel-bin/dev/kernels/muon_launch_sync_bench --json
```

| Row | Shape | µs |
|---|---|---|
| `launch:empty:1` | 1 block, 256 threads | 1.64 |
| `launch:empty:84` | 84 blocks (the resident grid of a 256-thread, 12 KB block) | 1.88 |
| `launch:empty:4000` | 4000 blocks (a full batched-GEMM grid) | 8.37 |
| `launch:coop:84` | cooperative launch, 84 blocks, no barrier | 1.92 |
| `sync:barrier:84` | one `grid.sync()`, 84 blocks | 0.92 |
| `sync:barrier:224` | one `grid.sync()`, 224 blocks | 2.19 |

So a dependent launch of a full batched grid costs about 8.4 µs and a grid
barrier about 0.9 µs at the grid a fused kernel could hold resident: one fused
iteration replaces 41.9 µs of launch gaps with 4.7 µs of one launch plus three
barriers, a **37.2 µs** gain per iteration. That is 0.5 % of an `h100` iteration
(7.3 ms) and 0.01 % of a `big` one (275 ms), so the launch gaps the fusion
removes are a rounding error against the elementwise passes it also removes:
the Phase 0 share above is elementwise HBM traffic, not dispatch.

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
tiles so the scalar gather runs, and a square group); the worst absolute error
over the run is 7.6e-5. The benchmark:

```bash
tools/nanochat build //dev/kernels:muon_fused_gemm_bench
tools/nanochat test --gpu //dev/kernels:muon_fused_gemm_gpu_test
tools/nanochat bench -- bazel-bin/dev/kernels/muon_fused_gemm_bench --json
```

`hand` is the prototype and `cublas` is the seam GEMM; `A`, `A2`, and `prod` are
the three products of one iteration at the group's shapes, and `iter` runs the
three in sequence. Three runs:

| Group | Stage | hand (ms) | cuBLAS (ms) | hand (GFLOP/s) | cuBLAS (GFLOP/s) | share of cuBLAS rate |
|---|---|---|---|---|---|---|
| 640x1280, batch 40 | `A` 640x640x1280 | 7.65–7.81 | 6.20–6.27 | 5373–5484 | 6690–6762 | 0.80 |
| | `A2` 640x640x640 | 3.91–4.01 | 2.21–2.35 | 5234–5371 | 8933–9484 | 0.55–0.60 |
| | `prod` 640x1280x640 | 7.59–7.71 | 4.60–4.68 | 5443–5526 | 8958–9116 | 0.60–0.62 |
| | `iter` | 17.67–18.32 | 12.61–12.78 | 5725–5933 | 8205–8318 | **0.68–0.72** |
| 1280x5120, batch 20 | `A` 1280x1280x5120 | 62.8–73.8 | 59.8–80.3 | 4549–5341 | 4181–5608 | 0.95–1.17 |
| | `A2` 1280x1280x1280 | 13.30–14.00 | 9.29–9.55 | 5993–6307 | 8786–9027 | 0.68–0.70 |
| | `prod` 1280x5120x1280 | 55.49–55.58 | 38.13–38.22 | 6037–6047 | 8779–8799 | 0.69 |
| | `iter` | 134.5–146.6 | 119.6–121.9 | 5149–5612 | 6192–6311 | **0.83–0.90** |

**Gate verdict: FAIL on the target group.** The section 4 gate asks for at least
85 % of the cuBLAS rate on a group the fusion targets; the 640x1280 key/value
group reaches **0.68–0.72**, and the prototype is the slower path on every stage
of it. The second data point, the largest group 1280x5120x1280, reaches
**0.83–0.90**: it is the one shape where the hand schedule meets the gate,
because that product is bandwidth-bound and cuBLAS is weak on it too (4181–5608
GFLOP/s).

**Decision: do not promote the fused GEMM.** The arithmetic of the section 3
inequality is negative on this card. At `h100` one Polar Express iteration takes
7.3 ms and the fusion could win at most the 37 µs of removed launch gaps plus
the elementwise passes folded into the neighbouring GEMM epilogues (1.1 ms per
iteration, so about 5.5 ms of a 42.8 ms step, 13 %); running the same
arithmetic through the prototype would cost 30.93 ms / 0.71 = 43.6 ms, i.e.
12.6 ms more per step. The loss is an order of magnitude larger than the win.

The prototype also shows *why*: 80 registers per thread, 12 KB of shared
memory, and three blocks per SM leave 768 of 2048 threads resident, the 8 by 4
micro-tile reads 12 shared floats per 32 FMAs, and an 8-deep k slice exposes one
global round trip per ~1500 cycles of arithmetic per SM. A schedule that changes
those three numbers (8 by 8 micro-tile, 16 to 32-deep slices, a 256 by 64 tile)
is a different prototype and a different measurement; nothing here says the
fusion is impossible, only that this schedule on this card is 30 % off cuBLAS
and the launch gaps it removes are worth a fraction of a percent.

## Adding a family

See [backends/cuda/kernels/README.md](../../backends/cuda/kernels/README.md). The
per-family tests and benchmarks live next to the kernel, not here.
