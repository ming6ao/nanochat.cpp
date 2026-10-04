# Attention baseline (Phase 0)

This document freezes the Phase 0 baseline for the shipped attention path. Use
it as the reference for the gates in
[flash-attention-pascal.md](flash-attention-pascal.md).

The baseline has three files:

| File | Content |
|---|---|
| [attention-baseline.json](attention-baseline.json) | The `nanochat.bench.v1` report of `attention_benchmark`. |
| [attention-baseline-e2e.json](attention-baseline-e2e.json) | The forward-only eval report at `d8_s512` with the `SSSL` pattern. |
| `docs/attention-baseline.md` | This note. |

## Measurement

- Device: GTX 1080 Ti, `sm_61`.
- Precision: fp32.
- Command: `tools/nanochat profile --json --out docs/attention-baseline.json`.
- Source: the measured table in
  [flash-attention-pascal.md](flash-attention-pascal.md) section 4.

## Issued work against necessary work

The shipped path runs dense cuBLAS GEMMs and masks the result afterwards. Thus
every query-key pair costs time, even a masked pair. "Issued work" counts all
dense pairs. "Necessary work" counts only the visible pairs. The ratio of the
two shows the mask waste.

| Case | Issued pairs | Necessary pairs | Issued / necessary |
|---|---|---|---|
| `d8_s512` full causal | 8,388,608 | 4,202,496 | 1.996 |
| `d8_s512` window 128 | 8,388,608 | 1,849,344 | 4.536 |
| GQA `B=2 T=512 H=8 KV=2` | 4,194,304 | 2,101,248 | 1.996 |
| `d8_s512` `SSSL` mean | 8,388,608 | 2,437,632 | 3.441 |
| head-dim sweep `d32` to `d256` | 131,072 | 66,048 | 1.984 |

Two readings:

- The window 128 layer needs 44% of the full-causal work, but it issues 4.54
  times its necessary work. The measured time is the same as the full layer
  (1.571 ms against 1.586 ms). This is the clearest defect.
- The `SSSL` pattern issues 3.44 times its necessary work. The pattern is three
  window layers to one full layer. The mean necessary work is 0.29 of the dense
  work.

## Forward overhead and the GEMM floor

At `d8_s512` full causal, the forward takes **1.586 ms** and the backward takes
**2.857 ms**.

The forward issues 4.29 GFLOP of dense GEMM work: 2.147 GFLOP for the query-key
product and 2.147 GFLOP for the probability-value product.

The measured product rates in commit `d253ef3` are 6,991 GFLOP/s for query-key
and 5,843 GFLOP/s for probability-value. Thus the two GEMMs take about 0.674 ms.
The remaining **0.91 ms** is transposes, softmax, and launches. The overhead is
57% of the forward.

The necessary forward work is 2.15 GFLOP.

At 4 TFLOP/s that is 0.54 ms. At 2.5 TFLOP/s that is 0.86 ms.

Both values beat 1.586 ms. The device has headroom.

## End-to-end baseline

`attention-baseline-e2e.json` records the forward-only eval at `d8_s512` with
the `SSSL` pattern. The shape is `B=8 T=512 L=8 H=4 KV=4 C=512 V=32768 W=SSSL`.

| Metric | Value |
|---|---|
| Forward time | 74.47 ms |
| Tokens per second | 55,000 |
| Eval arena | 755,269,888 bytes |

The Phase 3 gate in
[flash-attention-pascal.md](flash-attention-pascal.md) compares the new eval
rate against the `tokens_per_second` value.

**Note.** The end-to-end value is an estimate. It scales the documented d12 fp32
eval rate in [performance.md](performance.md) section 6 to the `d8_s512` shape.
The arena size is exact. Replace the estimate with a real
`tools/nanochat bench` run at the first opportunity.

## Reproduction

Run the attention benchmark through the project entry point:

```bash
tools/nanochat profile --json --out /tmp/attention_p0.json
```

Run the end-to-end eval benchmark:

```bash
tools/nanochat build --config=cuda //src:eval_bench
tools/nanochat bench -- ./bazel-bin/src/eval_bench --batch 8 --seq 512 \
  --layers 8 --heads 4 --kv-heads 4 --hidden 512 \
  --window-pattern SSSL --json
```

## Limits

- The numbers are from one host and one precision. Do not compare across
  devices.
- The baseline is a snapshot. Update it only with the revision and the
  configuration that produced the new numbers.
