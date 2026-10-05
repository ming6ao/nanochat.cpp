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
| `cpu_gemm_bench.cc` | CPU reference GEMM micro-benchmark over the four dominant training shapes plus the model-layout forward shapes; emits `nanochat.bench.v1`. |
| `cpu_linear_layout_probe.cc` | CPU GEMM probe over the model operand layouts (`transpose_b` true and false); emits `nanochat.bench.v1`. |

## Conventions

The shared conventions live in
[backends/cuda/kernels/README.md](../../backends/cuda/kernels/README.md): tiny
shapes, a host reference, finite differences, fixed seeds, the `gpu` tag, and
the benchmark report.

These prototypes are dev-only. The CUDA backend does not depend on this
directory.

## CPU GEMM micro-benchmark (Phase 0)

`cpu_gemm_bench.cc` times the four dominant GEMM shapes from
Part 2, section 2 of [docs/cpu-performance.md](../../docs/cpu-performance.md). The rows are
`gemm_forward`, `gemm_wgrad`, `gemm_dgrad`, and `mlp_gemm`. The benchmark warms
up, then keeps the best of N rounds. It calls `kernels::Gemm` directly and
prints the `nanochat.bench.v1` schema on standard output. The shipped CPU GEMM
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
[docs/cpu-performance.md](../../docs/cpu-performance.md).

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
  return p.window_left >= 0 && p.head_dim == 128;
}
```

The fused tiled forward serves a sliding-window shape when the head dimension
is 128. Full-causal attention (`window_left < 0`) and every other head
dimension stay on the cuBLAS path. Every backward call stays on the cuBLAS
path.

The measurement command is:

```bash
tools/nanochat profile --json --out /tmp/attention-window.json
```

The command runs the shipped `attention_benchmark`. The frozen reference is
`docs/attention-baseline.json`. The promoted kernel uses the Phase 1 tile
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

## Adding a family

See [backends/cuda/kernels/README.md](../../backends/cuda/kernels/README.md). The
per-family tests and benchmarks live next to the kernel, not here.
