# Planned training optimizations

This document collects the training-memory and kernel-bandwidth proposals that
are not implemented in the tree. Each part is a design with a plan and gates.
The result sections stay empty until a phase lands.

- [performance.md](performance.md) — the measurement protocol.
- [kernels.md](kernels.md) — the kernel seam.
- [model.md](model.md) — the workspace.

This document replaces the former `recompute.md`, `fused-classifier.md`, and
`vectorized-row-kernels.md`. The implemented forward-only evaluation path is in
[eval.md](eval.md).

## Part 1. Recompute of the row activations

Status: proposal. The analysis has code evidence and the memory figures come
from [eval.md](eval.md). The savings predictions do not have
measurements.

The training workspace holds every activation that the backward pass reads. The
arena does not fit a production batch on the 11 GB card. This document specifies
a compile-time recompute level that re-derives the cheap row activations in the
backward pass.

Related documents:

- [eval.md](eval.md): the training and evaluation arena sizes.
- [model.md](model.md): the graph inventory and the workspace design.
- [kernels.md](kernels.md): the kernel inventory.
- [../DESIGN.md](../DESIGN.md) section 5 and section 6.
- [../AGENTS.md](../AGENTS.md): the ownership map.

---

### 1. Purpose

The training arena is about 36 GB at `d12_s1024`, batch 32, fp32. The card holds
11 GB. [eval.md](eval.md) section 11 records the numbers. The
evaluation path solves its own problem with a small arena and a row-chunked
classifier. The training path still needs the full arena.

The backward pass reads a small group of row activations. The forward pass can
build each of them again from data that the workspace keeps. The rebuild costs
one extra pass over a row. The arena then drops by the size of the rebuilt
buffers.

[model.md](model.md) already states the intent: `recompute` is a compile-time
flag. [../DESIGN.md](../DESIGN.md) section 6 excludes a runtime recompute
branch. This document makes the intent concrete.

---

### 2. The measured problem

[eval.md](eval.md) section 2 gives the training arena:

| Group | Buffers | Size |
|---|---|---|
| Layer activations | `lacts_[0..L-1]` | `L * 20 * R * hidden` |
| Classifier | `raw_logits_`, `dlogits_` | `2 * R * padded_vocab` |
| Backward and global scratch | `dx_*`, `g_a_`, `g_b_`, `block_scratch_` | `33 * R * hidden` |

Here `R` is `batch * seq`. Use `R = 32768`, `hidden = 768`, `L = 12`, and
`padded_vocab = 32768`. The layer activations are about 24 GB.

The classifier buffers are about 8.6 GB. The scratch is about 3.3 GB.

The layer term is the largest single term that a recompute can shrink. The
classifier term needs a different change, described in Part 2.

---

### 3. The cause

`BuildTrainWorkspace` in `src/model.cc` gives every layer its own
`ops::BlockActivations` set. `RunForward` keeps all of them, because
`Backward()` reads them in reverse order. `docs/model.md` states the rule:
"the loop writes to `lacts_[i]` for each layer."

The per-layer set has 20 units of `R * hidden` size, plus small buffers:

| Slot | Size in `R * hidden` units |
|---|---:|
| `xr` | 1 |
| `h` | 1 |
| `q_pre` | 1 |
| `k_pre` | 1 |
| `q_final` | 1 |
| `k_final` | 1 |
| `ve_values` | 1 |
| `v_final` | 1 |
| `attn_out` | 1 |
| `x_mid` | 1 |
| `h2` | 1 |
| `pre_act` | 4 |
| `act` | 4 |
| `x_out` | 1 |
| Total | 20 |

`rstd1`, `rstd2`, `ve_gate`, and `attn_stats` are small and stay.

---

### 4. What llm.c does

`llm.c` gives the training run a `recompute` level. The option is `-r` and the
value is 0, 1, or 2.

- Level 0: save the GeLU activation and the LayerNorm activations.
- Level 1: rebuild the GeLU activation in the backward pass.
- Level 2: rebuild the GeLU activation and both LayerNorm activations.

The saved LayerNorm output for every layer becomes one shared buffer. The
forward pass writes the shared buffer and uses the value at once. The backward
pass rebuilds the value into the same buffer. `llm.c` calls this buffer
`acts.lnf`.

The trade is memory against compute. Level 1 removes the `4 * C` GeLU buffer
from every layer. Level 2 removes two `C` buffers more.

---

### 5. Design

#### 5.1 The compile-time level

The level is a compile-time constant. The build sets
`NANOCHAT_RECOMPUTE_LEVEL` to 0, 1, 2, or 3. The default is 0, which keeps the
current behavior.

A runtime branch would add a decision to every block call. The graph topology
does not change, so the build can fix the level. This rule matches
[../DESIGN.md](../DESIGN.md) section 6 and [model.md](model.md).

#### 5.2 Level 1 — the relu^2 activation

`BlockBackward` needs `act` for the `c_proj` weight gradient only. The
`relu^2` derivative comes from `pre_act`, which stays saved.

`MlpBackward` rebuilds `act` from `pre_act` at its start:

```cpp
kernels::PointwiseForward(PointwiseOp::kReluSquare, n, pre_act, nullptr,
                          1.0f, 0.0f, act);
```

The `act` slot becomes shared across the layers. The layer arena loses 4 units
of 20, or 20%.

#### 5.3 Level 2 — the RMSNorm outputs

`BlockBackward` needs `h` for the `c_q`, `c_k`, and `c_v` weight gradients and
for the value residual. `BlockBackward` needs `h2` for the MLP. Both values come
from a saved input and a saved statistic.

- `h = rmsnorm(xr, rstd1)`
- `h2 = rmsnorm(x_mid, rstd2)`

The `rstd1` and `rstd2` buffers are small and stay saved. The `h` and `h2` slots
become shared.

Level 2 removes 2 units on top of level 1. The total is 6 of 20, or 30%.

#### 5.4 Level 3 — the residual blend and the value gather

Two more slots come from a cheap rebuild.

- `xr = resid * x_in + x0_lambda * x0`. `BlockBackward` already receives `x_in`
  and `x0`, and the scalars `resid` and `x0_lambda`.
- `ve_values` is a gather from `weights.value_embeds` by the saved tokens.
  `EmbeddingForward` rebuilds it.

Level 3 removes 8 of 20, or 40%.

The `xr` rebuild needs `x_in`. The model passes the input of each block to
`BlockBackward`, so the API does not change.

#### 5.5 The shared buffers

The recomputed slots become one buffer for the whole model, not one for each
layer. `BlockForward` writes the shared buffer and uses the value at once.
`BlockBackward` rebuilds the value into the same buffer at the start of the
layer.

The layer order does not cause a conflict. The forward pass does not read a slot
after the block returns. The backward pass processes one layer at a time.

#### 5.6 What stays saved

The following slots stay per layer, because a rebuild needs a full matrix
product:

- `q_pre` and `k_pre`, because a rebuild repeats the `c_q` and `c_k` products.
- `q_final` and `k_final`, because `QkPrepBackward` overwrites them in place.
- `v_final`, because a rebuild repeats the `c_v` product.
- `attn_out` and `x_mid`, because a rebuild repeats the attention.
- `pre_act`, because the `relu^2` derivative needs it.
- `x_out`, because it is the input to the next layer.

#### 5.7 The arena figures

At level 2 the layer term drops by 30%. The layer term falls from about 24 GB to
about 16.8 GB. The total falls from about 36 GB to about 28.8 GB. The classifier
term does not change.

The training arena still does not fit 11 GB at this batch. The win is the growth
of the largest batch that fits, together with a smaller peak. The classifier
term and the logits chunk are the next step.

---

### 6. Ownership and interfaces

The frozen seam does not change. The change needs no new kernel. Every rebuild
uses an existing op:

| Rebuild | Op |
|---|---|
| `act` | `kernels::PointwiseForward(kReluSquare)` |
| `h`, `h2` | `ops::RmsNormForward` |
| `xr` | `kernels::PointwiseForward(kScaleAdd)` |
| `ve_values` | `kernels::EmbeddingForward` |

| Item | Owner |
|---|---|
| `src/model.cc`, `src/model_impl.h`, `src/ops.cc`, `src/ops.h` | Workflow |
| The build definition for `NANOCHAT_RECOMPUTE_LEVEL` | Build |
| This document | Architect |

---

### 7. Plan and gates

Run every step through `tools/nanochat`.

#### Phase 0 — the level and the counter (0.5 session)

- Add the `NANOCHAT_RECOMPUTE_LEVEL` build definition with the default 0.
- Extend `workspace_bytes()` reporting to print the level.
- Gate: the level-0 build reports the same arena size as today.

#### Phase 1 — level 1 (1 session)

- Share the `act` slot and rebuild it in `MlpBackward`.
- Gate: `workspace_bytes()` drops by about 20% of the layer term.
- Gate: the training-parity harness stays green at `1e-5` loss.
- Gate: the T1 GPU block tests stay green.

#### Phase 2 — level 2 (1 session)

- Share the `h` and `h2` slots and rebuild them in `BlockBackward`.
- Gate: the arena drops by about 30% of the layer term.
- Gate: the training-parity harness stays green.
- Gate: a T3 timing shows the step cost. Report the number.

#### Phase 3 — level 3 (1 session, optional)

- Rebuild `xr` and `ve_values`.
- Gate: the harness stays green and the timing stays inside the budget.

Stop a phase when its gate fails. Report the negative result.

---

### 8. Tests

- `tests/train_parity_test.cc`: the loss, gradient, and parameter trajectory.
  This is the main gate. The rebuild reproduces the same value from the same
  input, so the trajectory should not move.
- `src/model_gradient_test.cc`: the finite-difference check.
- `src/grad_mode_test.cc`: the arena comparison. Extend it with the levels.
- `backends/cuda/kernels/pointwise_test.cc`: the `kReluSquare` op.
- `backends/cuda/kernels/rms_norm_test.cc`: the norm rebuild.

---

### 9. Result

To fill after the phases. Record the revision and the configuration.

| Level | Layer term | Total arena | Step cost |
|---|---:|---:|---:|
| 0 | about 24 GB | about 36 GB | |
| 1 | | | |
| 2 | | | |
| 3 | | | |

---

### 10. Reproduction

```bash
tools/nanochat build --copt=-DNANOCHAT_RECOMPUTE_LEVEL=2
tools/nanochat test
tools/nanochat gpu --profile t2-parity -- \
  ./bazel-bin/tests/train_parity /tmp/train_parity_d8_s512_50.bin
```

## Part 2. Fused classifier

Status: proposal. The analysis has code evidence. The traffic figures are
derived from the buffer sizes. The timing gain does not have a measurement.

The classifier touches the largest tensor in the graph. The forward pass and the
backward pass both read the full logits buffer. The backward pass builds the
softmax statistics again. A separate pass scales the logits gradient. This
document specifies a statistics hand-off and a fused scale. Together they remove
most of the repeated traffic.

Related documents:

- [kernels.md](kernels.md): the kernel inventory and the classifier API.
- [model.md](model.md): the graph inventory and the workspace.
- [eval.md](eval.md): the classifier row chunking for evaluation.
- [performance.md](performance.md): the measurement protocol.
- [../DESIGN.md](../DESIGN.md) section 3: the fusion boundary rule.

---

### 1. Purpose

The classifier holds `raw_logits_` and `dlogits_`, each `rows * padded_vocab`.
At `R = 32768` and `padded_vocab = 32768` in fp32, each buffer is about 4.3 GB.
The two buffers are about 8.6 GB of the 36 GB training arena.

The classifier also moves the logits buffer several times for each step. The
arithmetic for each byte is small. The traffic dominates the kernel cost. This
document removes the avoidable passes.

---

### 2. The measured problem

`ClassifierForwardKernel` in `backends/cuda/kernels/classifier.cu` calls
`RowLogSumExp`. That helper reads the row twice:

1. one pass for `row_max`;
2. one pass for `sum_exp`.

The kernel then reads the target logit and writes the loss.

`ClassifierBackwardKernel` calls `RowLogSumExp` again. It reads the row twice
more for the same statistics. It then reads the row a third time to build the
gradient. The model then runs a separate scale pass:

```cpp
kernels::ClassifierBackward(classifier, raw_logits_, targets_.data(),
                            dlogits_);
kernels::PointwiseForward(PointwiseOp::kScale, rows * padded_vocab, dlogits_,
                          nullptr, scale / rows, 0.0f, dlogits_);
```

The scale pass reads and writes the full `dlogits_` buffer.

The traffic for each step, in full-buffer passes:

| Pass | Count |
|---|---:|
| Forward logits reads | 2 |
| Backward logits reads | 3 |
| Backward dlogits writes | 1 |
| Scale dlogits reads | 1 |
| Scale dlogits writes | 1 |

The backward pass builds the same statistics that the forward pass already
built. The input does not change, so the value is the same.

---

### 3. What llm.c does

`llm.c` fuses the classifier into one kernel, `fused_classifier_kernel5`. The
kernel computes the softmax statistics, the loss, and the logit gradients in one
call. It reads the logits once and overwrites the buffer with the gradients.
The header comment states the reason: "saves bandwidth from probs during
training".

`llm.c` can do this because its loss computation lives in the backward call. It
passes `dloss` into the kernel and never needs the raw logits again.

`nanochat.cpp` computes the loss in `ForwardLoss` and returns it. The backward
pass runs later. A single kernel cannot cover both calls without a change to the
`Model` API. This document keeps the API and fuses what it can.

---

### 4. Design

#### 4.1 Option A — the statistics hand-off

The forward pass already computes `(row_max, sum_exp)` for each row. Save the
pair. The backward pass reads the pair instead of building it again.

The statistics are `2 * rows` floats, or about 256 KB at `R = 32768`. The cost
is negligible.

The saved value equals the rebuilt value bit for bit. The forward and the
backward use the same input, the same reduction, and the same order. The parity
result does not change.

This option removes two logits reads from the backward pass. It also removes one
`RowLogSumExp` call, which is two `tanh` passes over the row.

#### 4.2 Option B — the fused scale

Pass the loss scale into the backward pass. The classifier writes the scaled
gradient in one pass:

```cpp
dlogits[j] = (p - onehot) * sech2 * dlogit_scale;
```

The model passes `dlogit_scale = scale / rows` from `BackwardAccumulate`. The
separate `PointwiseForward(kScale)` call disappears.

The arithmetic keeps the same order. The current code forms `(p - onehot) *
sech2` and then multiplies by the scale. The fused form does the same. The
result is bit-identical.

This option removes one read and one write of the full `dlogits_` buffer.

#### 4.3 The combined result

Apply both options. The traffic becomes:

| Pass | Before | After |
|---|---:|---:|
| Forward logits reads | 2 | 2 |
| Backward logits reads | 3 | 1 |
| Backward dlogits writes | 1 | 1 |
| Scale dlogits passes | 2 | 0 |

At `R = 32768` and `padded_vocab = 32768` in fp32, the change removes about
17 GB of traffic for each step.

#### 4.4 The rejected option — the in-place gradient

`llm.c` overwrites the logits buffer with the gradients. That removes the
`dlogits_` buffer and 4.3 GB of arena.

`nanochat.cpp` cannot do this without a correctness loss. `BackwardAccumulate`
can run more than once for one forward pass.
`src/harness_test.cc` calls it twice with a half scale. The second call reads
the raw logits again. An in-place overwrite destroys the raw logits, so the
second call reads gradients instead. The test catches this.

The separate `dlogits_` buffer stays. The statistics hand-off and the fused
scale recover most of the benefit without the risk.

---

### 5. Seam change

The frozen header `include/nanochat/kernels.h` changes. The change is additive
at the parameter level. The architect owns the header.

```cpp
// Forward also writes the per-row softmax statistics when `stats` is non-null:
// stats[2*r] = row_max, stats[2*r+1] = sum_exp. The evaluation path passes null.
void ClassifierForward(const ClassifierParams& params,
                       const ComputeType* logits, const int* targets,
                       ComputeType* losses, float* stats = nullptr);

// Backward consumes the statistics from the matching forward when `stats` is
// non-null; otherwise it rebuilds them (the current behavior). `dlogit_scale`
// folds the loss normalization into the gradient write.
void ClassifierBackward(const ClassifierParams& params,
                        const ComputeType* logits, const int* targets,
                        const float* stats, float dlogit_scale,
                        ComputeType* dlogits);
```

The default argument keeps the old forward call sites valid at the source level.
Both backends must still define the new signature.

The null `stats` path keeps the current behavior. The per-kernel tests use it to
compare the two paths.

---

### 6. Ownership and interfaces

| Item | Owner |
|---|---|
| `include/nanochat/kernels.h` | Architect |
| `backends/cpu/kernels.cc` | Runtime |
| `backends/cuda/kernels/classifier.cu` | Kernel agents |
| `src/model.cc`, `src/model_impl.h` | Workflow |
| This document | Architect |

Coordinate before the header change. See [../AGENTS.md](../AGENTS.md) section 1.

---

### 7. Plan and gates

Run every step through `tools/nanochat`.

#### Phase 0 — the seam and the statistics (1 session)

- Add the `stats` output to `ClassifierForward` and the `stats` input to
  `ClassifierBackward`.
- Implement both paths in the CPU backend and the CUDA backend.
- Gate: the T1 classifier test passes for the null path and the stats path.
- Gate: the two paths agree to the last bit.

#### Phase 1 — the model wiring (0.5 session)

- Allocate `classifier_stats_` in the training workspace.
- Pass the statistics from `RunForward` to `BackwardAccumulate`.
- Pass null on the evaluation path.
- Gate: the training-parity harness stays green.

#### Phase 2 — the fused scale (0.5 session)

- Add `dlogit_scale` to `ClassifierBackward`.
- Remove the `PointwiseForward(kScale)` call from `BackwardAccumulate`.
- Gate: `harness_test.cc` stays green, including the double accumulation case.
- Gate: the training-parity harness stays green.

#### Phase 3 — measurement (0.5 session)

- Time the classifier before and after with `eval_bench` and a T3 profile.
- Record the numbers in section 9.

Stop a phase when its gate fails. Report the negative result.

---

### 8. Tests

- `backends/cuda/kernels/classifier_test.cc`: the correctness and finite
  difference gate. Add a case for the statistics path and the scale.
- `backends/cuda/kernels/sequence_oracle_test.cc`: the classifier against
  `tests/data/debug_state.bin`.
- `src/harness_test.cc`: the double `BackwardAccumulate` case. This is the gate
  for the rejected in-place option.
- `tests/train_parity_test.cc`: the end-to-end trajectory.
- `src/eval_bench.cc`: the forward-only classifier timing.

---

### 9. Result

To fill after the phases. Record the revision and the configuration.

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Classifier forward, ms | | | |
| Classifier backward, ms | | | |
| Full step, ms | | | |
| Training arena, GB | | | |

---

### 10. Reproduction

```bash
tools/nanochat build
tools/nanochat test
tools/nanochat test --gpu //backends/cuda/kernels:classifier_gpu_test
tools/nanochat bench -- ./bazel-bin/src/eval_bench --batch 16 --seq 1024 \
  --layers 12 --heads 6 --kv-heads 6 --hidden 768 \
  --vocab 32768 --padded-vocab 32768
```

## Part 3. Vectorized row kernels

Status: proposal. The analysis has code evidence. The performance predictions do
not have measurements.

The row and elementwise families move large tensors through the device. They do
little arithmetic for each byte. A wider memory access raises the rate. This
document specifies a shared packed-access helper and its use in the affected
families.

Related documents:

- [performance.md](performance.md): the measurement protocol and the benchmarks.
- [kernels.md](kernels.md): the kernel inventory and the shared device helpers.
- [build.md](build.md): the backend layout and ownership.
- [../AGENTS.md](../AGENTS.md): the ownership map.
- [../DESIGN.md](../DESIGN.md) section 3: the fusion boundary rule.

---

### 1. Purpose

The row kernels are bandwidth-bound. `RmsNormForward` reads one tensor and
writes one tensor. `PointwiseForward` reads two and writes one. The classifier
reads the largest tensor in the graph. A packed access lets the compiler emit a
128-bit `LDG`/`STS` pair. The streaming hint keeps the reused data in cache.

`llm.c` proves the idea on the same architecture class. `nanochat.cpp` does not
use it. This document closes that difference.

---

### 2. The measured problem

The CUDA backend uses scalar strided access in every row kernel. A search of
`backends/cuda` finds no use of `__ldcs`, `__stcs`, `__ldg`, `int4`, or
`float4`. The loops step one element at a time:

```cpp
for (int d = threadIdx.x; d < params.dim; d += blockDim.x) {
  orow[d] = ToComputeDev(AsFloatDev(xr[d]) * r);
}
```

The same shape appears in `rms_norm.cu`, `pointwise.cu`, `embedding.cu`,
`classifier.cu`, and `device_utils.cuh`.

`row_benchmark.cc` times these families. It reports milliseconds only. It does
not report the achieved byte rate or the fraction of the device peak. The first
task is to add that figure.

The GTX 1080 Ti moves about 484 GB/s at the peak. The fp32 engine runs at about
11.34 TFLOP/s. The ridge point is about 23 flop for each byte. The row kernels
sit far below that point, so they lose time on memory traffic.

A scalar 32-bit load asks the memory system for one value. A 128-bit load asks
for four values in one transaction. Coalesced scalar access also reaches one
transaction per warp, so the win is not free. The win comes from fewer
instructions, more work in flight, and the streaming hint. The measurement
decides the size of the win.

---

### 3. What llm.c does

`llmc/cuda_utils.cuh` defines the pattern. The comment states the reason: the
compiler must emit `LDG.128` and `STS.128`.

```cpp
template<class ElementType>
struct alignas(16) Packed128 {
  ElementType payload[sizeof(int4) / sizeof(ElementType)];
};
template<class ElementType>
__device__ Packed128<ElementType> load128(const ElementType* address) {
  return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
}
template<class ElementType>
__device__ Packed128<ElementType> load128cs(const ElementType* address) {
  return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4*>(address))};
}
```

`store128`, `store128cs`, and `store128cg` complete the set. The `cs` suffix
streams the data through the cache. The `cg` suffix caches in L2 only.

The kernels use the helper everywhere. `layernorm_forward_kernel6` keeps the
weights in shared memory and streams the activations. `fused_residual_forward_kernel5`
does the same for the residual add. `fused_classifier` streams the logits.

---

### 4. Design

#### 4.1 The packed helper

Add one helper to `backends/cuda/kernels/device_utils.cuh`. The helper is
header-only, so it inlines into every caller. It does not merge kernel
boundaries.

```cpp
template <typename T>
struct alignas(16) Packed {
  static constexpr int kCount = static_cast<int>(16 / sizeof(T));
  T elem[kCount];
};

template <typename T>
__device__ __forceinline__ Packed<T> LoadPacked(const T* p);
template <typename T>
__device__ __forceinline__ Packed<T> LoadPackedStreaming(const T* p);
template <typename T>
__device__ __forceinline__ void StorePacked(T* p, Packed<T> v);
template <typename T>
__device__ __forceinline__ void StorePackedStreaming(T* p, Packed<T> v);
```

The fp32 build packs four elements. The fp16 build packs eight elements. The
same helper serves both. The load and the store use a `reinterpret_cast` to
`int4`, as `llm.c` does.

#### 4.2 The streaming hint

Use `__ldcs` and `__stcs` for the streamed operand. Use the plain load for the
operand that sees reuse. The choice is per kernel:

- `RmsNormForward`: stream `x` and `out`.
- `RmsNormBackward`: stream `x`, `dy`, and `dx`.
- `PointwiseForward` and `PointwiseBackward`: stream every operand.
- `ClassifierForward` and `ClassifierBackward`: stream `logits`.
- `EmbeddingForward`: stream the output. Keep the table gather scalar, because
  the kernel indexes the rows.
- `GlobalNorm`: stream `grads`.
- The attention softmax row kernels: stream `scores`, `probs`, and `dprobs`.

#### 4.3 Alignment and the tail

A 128-bit access needs a 16-byte aligned pointer. The row width `dim` and the
token count may not divide by the packed count. The seam allows any positive
value.

Three conditions define the fast path:

1. The pointer is 16-byte aligned.
2. The row width divides by `Packed<T>::kCount`.
3. The element count of the row divides by the block stride.

If all three hold, the kernel uses the packed loop. If any fails, the kernel
uses the scalar loop. The two loops produce the same arithmetic. The host does
not need a branch, because the kernel checks the conditions once at the start.

The workspace aligns each slot to `alignof(T)`. That is four bytes in the fp32
build. The doc proposes a 16-byte alignment for the slots that the packed
kernels read. `Workspace::Reserve` aligns the arena to 64 bytes, so the change
stays inside `src`.

#### 4.4 Numerics

A packed loop changes the order of the per-thread accumulation. The reduction
across the block stays the same. The last bits of a sum can move. The fp32
oracle tolerance is `1e-5`. The finite-difference tests and the oracle fixture
decide the result.

The reduction must stay exact where the reference demands it. The classifier
softmax is the sensitive case. A packed reduction can change `row_max` by one
unit in the last place. The design keeps the scalar reduction for the
classifier until the parity run proves the packed one safe.

---

### 5. Affected kernels

| Kernel | Operand | Change | Risk |
|---|---|---|---|
| `rms_norm.cu` | `x`, `out` | packed load and store | Low |
| `pointwise.cu` | `a`, `b`, `out` | packed load and store | Low |
| `embedding.cu` | `out`, `dtable` | packed store, scalar gather | Low |
| `global_norm.cu` | `grads` | packed load and store | Low |
| `attention.cu` | softmax rows | packed load and store | Medium |
| `classifier.cu` | `logits` | packed load, scalar reduction | Medium |

---

### 6. Ownership and interfaces

The frozen seam does not change. `include/nanochat/kernels.h` keeps every
signature.

| Item | Owner |
|---|---|
| `device_utils.cuh` helper and the row kernels | Kernel agents (`backends/cuda/kernels/`) |
| The arena alignment in `src` | Workflow (`src/`) |
| The benchmark bandwidth column | Kernel agents (`backends/cuda/kernels/`) |
| This document | Architect (`docs/`) |

One writer for each file. See [../AGENTS.md](../AGENTS.md) section 1.

---

### 7. Plan and gates

Run every step through `tools/nanochat`. Use the `t3-bench` profile for the
timings. Use the `t0-cpu` profile for the CPU tests.

#### Phase 0 — measurement (0.5 session)

- Add a byte-rate column to `row_benchmark.cc`.
- Record the shipped numbers as `docs/vectorized-row-baseline.json`.
- Gate: the benchmark reports the byte rate and the peak fraction.

#### Phase 1 — the helper and the simple families (1 session)

- Add the packed helper to `device_utils.cuh`.
- Convert `rms_norm.cu`, `pointwise.cu`, `embedding.cu`, and `global_norm.cu`.
- Gate: each family gains at least 10% in byte rate.
- Gate: the finite-difference test and the oracle fixture stay green.

#### Phase 2 — the classifier and the softmax rows (1 session)

- Convert `classifier.cu` and the attention softmax kernels.
- Keep the scalar reduction where the parity run fails.
- Gate: the classifier and attention benchmarks improve.
- Gate: the training-parity harness stays green.

If a phase fails its gate, report the negative result and stop the phase.

---

### 8. Tests

The existing tests already compare the device result to a host reference. They
do not need a change if the arithmetic is the same.

- `backends/cuda/kernels/rms_norm_test.cc` and `pointwise_test.cc`: T1.
- `backends/cuda/kernels/row_kernels_test.cc`: the smoke test over the seam.
- `backends/cuda/kernels/precision_test.cc`: both precisions.
- `backends/cuda/kernels/classifier_test.cc`: the classifier and the padded tail.
- `tests/oracle_test.cc` and `tests/train_parity_test.cc`: the end-to-end gate.

---

### 9. Result

To fill after the phases. Record the revision and the configuration with each
number.

| Family | Before | After | Change |
|---|---:|---:|---:|
| rmsnorm | | | |
| pointwise | | | |
| classifier | | | |
| attention softmax | | | |

---

### 10. Reproduction

```bash
tools/nanochat build
tools/nanochat test
tools/nanochat bench -- ./bazel-bin/backends/cuda/kernels/row_benchmark
tools/nanochat profile --json --out /tmp/attention_p1.json
```
