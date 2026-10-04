# Attention window promotion results

This note records the promotion verdict for the fused windowed attention
forward. The campaign is `flash-attn-window`.

- Design: [flash-attention-window.md](flash-attention-window.md)
- Frozen reference: [attention-baseline.json](attention-baseline.json)
- Frozen end-to-end reference: [attention-baseline-e2e.json](attention-baseline-e2e.json)

## Verdict

The windowed forward passes every gate. The campaign promotes it. Full-causal
attention stays on cuBLAS. Every backward stays on cuBLAS.

## Dispatch predicate

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

## Phase B — parity

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

## Phase C — window performance

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

## Phase D — end-to-end

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

## Limits

- The frozen end-to-end baseline is an estimate, not a measured run. The
  reported margin of 5.3% is therefore not exact.
- The fused tile covers only `head_dim == 128` and `window_left >= 0`.
- The full-causal fused tile lost the Phase 1 kill gate. Do not widen the
  dispatch predicate without a new measurement.
- The numbers come from one host and one precision. Do not compare them across
  devices.

## Reproduction

Run the two measurements through the project entry point:

```bash
tools/nanochat profile --json --out /tmp/attention-window.json
tools/nanochat bench -- ./bazel-bin/src/eval_bench --batch 8 --seq 512 \
  --layers 8 --heads 4 --kv-heads 4 --hidden 512 --window-pattern SSSL --json \
  > /tmp/eval-window.json
```
