# Windowed fused attention: promotion of the Phase 1 prototype

Status: promoted. Owner: architect/integrator.

Promotion results: [attention-window-results.md](attention-window-results.md).

## 1. Why reopen

The Phase 1 kill gate failed for the full-causal forward. The fused tile
measured 1.98 ms at `d8_s512` full causal against the shipped 1.586 ms. The
tile space cannot close the gap. At `head_dim` 128, the 48 KB shared-memory
cap admits only `Bc=32`.

The same prototype won on the windowed shape. At window 128 it measured
1.078 ms against the shipped 1.571 ms. That is a 1.46x win.

The production model uses the `SSSL` pattern. Three of every four layers use a
window. The window path is worth promoting on its own.

This document scopes the reopened effort. The technical background, the
baseline, and the prototype are in
[flash-attention-pascal.md](flash-attention-pascal.md).

## 2. Goal

Promote the fused fp32 tiled forward for windowed attention shapes. Keep the
cuBLAS path for full-causal attention and for every shape the fused kernel
does not cover. Show an end-to-end tokens per second win on the `SSSL` model.

## 3. Scope

In scope:

- Promote the fused forward into `backends/cuda/kernels/attention.cu`.
- Add a `UseFusedAttention(params)` predicate. Select the fused kernel only
  for `window_left >= 0` and `head_dim == 128`.
- Keep the kernel seam byte for byte. Do not edit `include/nanochat/kernels.h`.
- Keep the cuBLAS forward for full-causal attention and for all other shapes.
- Measure the production `SSSL` shape, not only window 128.

Out of scope:

- The fused backward. The prototype has no backward. The cuBLAS backward
  stays.
- A full-causal improvement. That needs a different tile, for example a split
  `head_dim`. It is a separate effort.

## 4. Phases

### Phase A — promotion and dispatch

Move the validated forward from the `dev/kernels` prototype into
`backends/cuda/kernels/attention.cu`. Add the dispatch predicate. Keep the
cuBLAS path for the uncovered shapes. Update
`backends/cuda/kernels/BUILD.bazel` only when the dependency set changes.

Gate: the CUDA build succeeds and the existing attention tests stay green.

### Phase B — correctness and parity

Run the oracle and the training trajectory on the promoted path.

Gate: `tests:oracle_cuda_test` and `tests:train_parity_cuda_test` pass with
the strict tolerances. The CPU parity tests stay green.

### Phase C — window performance

Re-run the shipped attention benchmark at the production `SSSL` shape and at
window 128.

Gate: the windowed forward row beats the Phase 0 baseline. The full-causal
row stays on the cuBLAS path and must not regress.

### Phase D — end-to-end

Measure the end-to-end effect on the `SSSL` eval benchmark.

Gate: the eval tokens per second beats `docs/attention-baseline-e2e.json`.

## 5. Risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| The window win shrinks at the production window | The window-128 win may not hold at window 512 | Measure the production shape in Phase C |
| Attention is a small part of the step | A window win moves end-to-end tokens per second little | Set the gate on tokens per second, not the kernel |
| A wrong dispatch predicate | The fused kernel can be selected for a losing shape | Keep the predicate narrow and test the full-causal path |
| Numerics drift | The task has a strict parity gate | Promote only after the parity gate passes |
