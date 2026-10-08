# backends/cpu

The reference implementation of every symbol in `nanochat/kernels.h`. It is the
correctness baseline, the oracle-on-CPU path, and the backend that lets the
workflow, optimizer, and model run before any GPU kernel exists.

- `kernels.cc` — all eleven compute families, `Gemm`, and the device utilities.
  `--config=fp32` (default) / `--config=fp16` select `ComputeType`.
- `kernels_test.cc` — analytic checks plus finite-difference gradient checks
  for every family with a backward. Run with `tools/nanochat test
  //backends/cpu:kernels_test`.

The library target is `//backends/cpu:cpu`.

## Contracts worth knowing

- **Layouts.** Q/K/V are `[batch, seq, heads, head_dim]` row-major (the layout
  nanochat's SDPA fallback uses). Attention stats are
  `[batch, num_heads, seq, 2]` holding `(max, sum_exp)`.
- **GQA head mapping.** Query head `h` uses key/value head
  `h / (num_heads / num_kv_heads)`, matching `enable_gqa`.
- **`QkPrep` backward.** The `q`/`k` buffers passed to `QkPrepBackward` are the
  *saved pre-norm projection outputs*; they are overwritten with the gradient
  with respect to those projection outputs. The upstream `dq`/`dk` are taken
  with respect to the final (RMSNorm -> RoPE -> scale) activation. Saving the
  pre-norm rows is what makes the RMSNorm statistic recoverable. The forward
  writes the final activation into whatever buffer it is given, so a model that
  needs both must keep a copy of the pre-norm rows (for example
  `q_proj` for backward and a `q_prep` copy for attention).
- **Write-not-accumulate.** `RmsNorm*` and `Pointwise*` write their outputs.
  `Classifier*` writes its outputs. `EmbeddingBackward` scatter-adds onto the
  dense gradient buffer, so the caller zeroes that buffer once with `ZeroGrad`
  and repeated backward calls sum. `AttentionBackward` zeroes `dq`/`dk`/`dv`
  before accumulating.
- **Loss scaling.** `ClassifierForward` writes per-row cross-entropy and
  `ClassifierBackward` writes the gradient of the *sum* of those rows
  (`softmax - onehot` through the softcap). The caller divides by the number of
  non-ignored rows to get a mean-loss gradient.
- **Gemm.** Universal row-major `C = alpha * op(A) * op(B) + beta * C`; the
  `GemmMode` enum is advisory and the transpose flags decide the operands.
