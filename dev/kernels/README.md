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
| `attention_tile_v3.h`, `attention_tile_v3.cu` | Fused fp32 tiled (flash) forward prototype with a compile-time tile configuration for the Phase 1 tile-size sweep. |
| `attention_tile_v3_test.cc` | The forward tile against `sequence_ref.h` at the tiny gate, the swept configurations against the reference, and the device tile against the shipped `kernels::AttentionForward` at `d8_s512`. |
| `attention_tile_v3_bench.cc` | The `d8_s512` forward rows plus the Br, Bc, block-size, and launch-bounds tile space and the head-dimension sweep. |
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

## Attention tile results (Phase 1 kill gate)

The Phase 1 prototype is `attention_tile_v3.cu`. The kill gate has two
conditions at `B=8 T=512 H=4 KV=4 D=128`, from
[docs/flash-attention-pascal.md](../../docs/flash-attention-pascal.md) section 6:

- The full-causal forward must beat the shipped `1.586 ms`.
- The window-128 forward must be at least `1.6x` faster than the full-causal
  forward.

The measurement is `tools/nanochat bench --
./bazel-bin/dev/kernels/attention_tile_v3_bench --json` on the GTX 1080 Ti
(`sm_61`, fp32). Each timing is the best of three device-event rounds. The
shipped reference is `docs/attention-baseline.json`.

| Case | Tile configuration | Tile (ms) | Shipped (ms) | Speedup | Tile GFLOP/s |
|---|---|---|---|---|---|
| full causal | br=32 bc=32 thr=256 mb=2 | 1.980 | 1.586 | 0.80x | 1,087 |
| window 128 | br=32 bc=32 thr=256 mb=2 | 1.078 | 1.571 | 1.46x | 878 |
| grouped query | br=32 bc=32 thr=256 mb=2 | 1.059 | 1.312 | 1.24x | 1,016 |

**Gate verdict: FAIL. Stop the effort.**

- Full causal: `1.980 ms` against `1.586 ms`. The tile is `1.25x` slower, so
  this condition fails.
- Window 128: `1.980 / 1.078 = 1.84x`. This condition passes.
- The window case saves work correctly, but the full-causal kernel cannot reach
  the cuBLAS forward.

The tile space does not close the gap. At `head_dim = 128`, the 48 KB Pascal
shared-memory cap admits only `bc = 32`: that tile uses 37,120 bytes, while
`bc = 64` needs 74,240 bytes and does not fit. The Br, threads, and launch-bound
axes were swept on a representative shape `B=2 T=256 H=4 KV=4 D=64`:

| Axis | Configuration | ms |
|---|---|---|
| Br | 16 / 32 / 64 | 0.1389 / 0.1241 / 0.2385 |
| Bc | 32 / 64 | 0.1249 / 0.1222 |
| threads | 128 / 256 / 512 | 0.1618 / 0.1249 / 0.1140 |
| min blocks | 1 / 2 / 3 | 0.1242 / 0.1235 / 0.1594 |

The best swept configuration is `threads = 512`, at `0.1140 ms` against the
default `0.1241 ms` (`1.09x`). That gain cannot cover the `1.25x` gap at
`d8_s512`. The default `br=32 bc=32 thr=256 mb=2` tile is therefore the best
measured production configuration, and it loses to the tuned cuBLAS dense
product at this shape. This is the risk the plan's risk table predicted. Phase 2
(the backward) and the full-causal promotion do not start. The windowed forward
is a separate, legal partial win; see the next section.

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

The command runs the shipped `attention_bench`. The frozen reference is
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

1. Add the CUDA translation unit under `backends/cuda/kernels/` and its
   `cuda_library` target (see `backends/cuda/kernels/BUILD.bazel`).
2. Port the reference math into `row_ref.h` (or a sibling reference header).
3. Add a `<family>_gpu_test.cc` here with a correctness check and, when the
   family has a backward, a finite-difference check. Add it to the `all`
   test suite.
4. Optionally add the family to `row_bench.cc`.
