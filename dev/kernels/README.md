# dev/kernels — standalone per-kernel tests and benchmarks

This directory follows the `llm.c` `dev/cuda` convention: each kernel family
gets its own small, self-contained test and (optionally) a benchmark, separate
from the full-graph tests in `//tests`. The point is to make a single family
fast to reason about and cheap to gate.

The canonical tiers, tagging, and resource rules are in
[docs/testing.md](../../docs/testing.md) and [AGENTS.md](../../AGENTS.md) §5.
Run everything through the project entry point:

```bash
tools/nanochat test                      # T0 CPU tests (excludes tag `gpu`)
tools/nanochat test --gpu //...          # T1 GPU tests (tag `gpu`)
tools/nanochat bench -- <row_bench>      # T3-style micro-benchmark under the broker
```

## Layout

| File | What |
|---|---|
| `row_ref.h` | Host reference for the row/elementwise families, ported line-for-line from `backends/cpu/kernels.cc`. |
| `sequence_ref.h` | Host reference for the Attention/Classifier/Embedding families, ported line-for-line from `backends/cpu/kernels.cc`. |
| `gpu_test_utils.h` | Storage conversion, deterministic RNG, `DevBuf`, tolerance checks, central finite-difference helper. |
| `bench_utils.h` | CUDA-event timer and the `nanochat.bench.v1` JSON report used by benchmarks. |
| `rms_norm_test.cc` | RmsNorm forward/backward correctness and finite differences, plus the fused residual variant. |
| `qk_prep_test.cc` | QkPrep forward/backward correctness and finite differences, including a grouped-query case. |
| `pointwise_test.cc` | All five pointwise op codes, forward/backward, plus finite differences. |
| `global_norm_test.cc` | GlobalNorm norm/clip correctness, including a multi-block reduction. |
| `attention_test.cc` | Attention forward/backward correctness and finite differences: MHA, GQA, sliding window, non-causal, and a KV-cache offset. |
| `classifier_test.cc` | Classifier softcap cross-entropy forward/backward, padded tail, and ignore index. |
| `embedding_test.cc` | Embedding gather forward and scatter-add backward over a persistent buffer, including duplicate ids. |
| `sequence_oracle_test.cc` | Classifier and Embedding against `tests/data/debug_state.bin`, plus a config-driven Attention check. |
| `row_bench.cc` | Micro-benchmark for every family on tiny and medium shapes. |
| `attention_bench.cc` | Attention forward/backward device-event timing at the `d8_s512` shape, with a head-dimension sweep that exposes the per-owned-dimension redundancy. |
| `qk_prep_fused.h`, `qk_prep_fused.cu` | Combined fused QkPrep prototype (one launch over q and k) plus the explicit decomposed baseline (`RmsNormForward` + standalone RoPE/scale). |
| `qk_prep_fused_test.cc` | Fusion prototype correctness against `row_ref.h`, cross-check against the seam `QkPrep`, and finite differences for the fused backward. |
| `qk_prep_bench.cc` | Decomposed vs seam-fused vs combined-fused QkPrep, tiny and medium shapes. |
| `decode_kernels.h`, `decode_kernels.cu` | Capture-safe (stream-taking) mirrors of the decode-path seam ops, used to build the CUDA-graph baseline. |
| `decode_fused_bench.cc` | Batch-1 decode baseline: eager seam path vs `cuBLAS + CUDA Graphs`; the acceptance binary. |
| `pascal_spike.cu`, `pascal_spike_test.cc` | The P0 toolchain spike. |

## Conventions

- **Tiny shapes by default.** The attention-shaped kernels use `B=2, T=8`,
  `head_dim=4`, `num_heads=2`. They cost microseconds, so they are proper
  gates rather than iteration tools.
- **Compare against a host reference.** `row_ref.h` is the device-independent
  specification; a correctness test compares device output to it. This keeps
  the oracle torch-free and the test deterministic.
- **Finite-difference every backward.** Analytic gradients are checked against
  central differences of the reference forward (`h = 1e-2`, relative tolerance
  `2e-3`). Finite differences run in fp32 only; fp16 storage cannot resolve
  the perturbation.
- **Fixed seeds.** `Rng` is a deterministic xorshift, so failures reproduce.
- **Benchmarks report machine-readable output.** Use `bench_utils.h`:
  `EventTimer` for device-event timing and `BenchReport` for the table and the
  `nanochat.bench.v1` JSON. Benchmarks are `cuda_binary` targets and are never
  added to the `all` test suite. Run them with `tools/nanochat profile` or
  `tools/nanochat bench`; see [docs/performance.md](../../docs/performance.md).
- **Tag GPU tests `gpu`.** `tools/nanochat test` filters them out with
  `--test_tag_filters=-gpu`; `tools/nanochat test --gpu //...` runs only them.
  Do **not** add the `manual` tag to a test that the `--gpu //...` acceptance
  command must discover, because Bazel excludes `manual` targets from wildcard
  expansion.

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

## Adding a family

1. Add the CUDA translation unit under `backends/cuda/kernels/` and its
   `cuda_library` target (see `backends/cuda/kernels/BUILD.bazel`).
2. Port the reference math into `row_ref.h` (or a sibling reference header).
3. Add a `<family>_gpu_test.cc` here with a correctness check and, when the
   family has a backward, a finite-difference check. Add it to the `all`
   test suite.
4. Optionally add the family to `row_bench.cc`.
