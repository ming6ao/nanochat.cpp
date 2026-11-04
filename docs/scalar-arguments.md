# Scalar arguments: a device-resident scalar in the seam

Status: rows S.1 to S.7 have landed, and section 14 records the measurement.
`ScalarArg` is in `include/nanochat/kernels.h`, a frozen header, so the
architect should review that edit. The result is a precision change and not a
bit-identical refactor: section 9 records why, and it corrects the claim this
document carried first.

This document proposes one type, `ScalarArg`, and the five call sites that use
it. The result is that a scalar the optimizer owns never reaches the host on a
training step.

Read [performance.md](performance.md) for the measurement protocol and
[performance.md](performance.md) for the loop
that found the cost.

## 1. The rule this proposal makes explicit

| Scalar kind | Examples | Form |
|---|---|---|
| A constant or a schedule value | `eps`, softcap, `kQkScale`, `lr`, `beta1`, warmup | host `float`, by value. Correct today, and stays |
| **Model state** | `resid_lambdas`, `x0_lambdas`, `smear_lambda`, `backout_lambda`, the clip scale | a device pointer the kernel loads |

The seam already has the second form once: `AdamWParams::grad_scale_dev` and
`MuonParams::grad_scale_dev`, added by row `R2-D5` so the clip scale stays on
the device. That row proved the idea and spelled the field pair out twice. This
proposal replaces the pair with one type and moves the optimizer site onto it,
because a second and a third consumer need the same thing.

The pointwise family never got the form at all, and it is where two model
parameters enter the arithmetic.

### 1.1 The second rule: resolve once

A kernel reads a scalar argument **once, before the loop**, and keeps it in a
register. `AdamWUpdateKernel` is the model:

```cpp
const float scale = scale_on_device ? *grad_scale_dev : grad_scale;
for (long long i = ...; i < n; i += stride) { ... }
```

Every new arm follows that shape. A read inside the element loop would turn one
L2 hit into `n` loads.

## 2. What it costs today

`src/ops.cc` calls `Pw(PointwiseOp::kScaleAdd, ..., resid, x0_lambda)` once per
layer. `PointwiseParams` carries `float alpha` and `float beta` and no device
form, so the value must reach the host before the launch. `src/model.cc` read it
back with `ReadHost`, which is a synchronous `cudaMemcpy`:

| Read | Where | Count per step |
|---|---|---|
| `resid_lambdas[i]` | forward and backward, once per layer | 16 |
| `x0_lambdas[i]` | forward and backward, once per layer | 16 |
| `smear_lambda` | `SmearForward`, `SmearBackward` | 2 |
| `backout_lambda` | `BackoutForward`, `BackoutBackward` | 2 |

`ReadHostScalars` already replaced the first two rows with two copies per pass,
so four array copies and four single copies remain. The trace of the tree before
that change put the `blocking_dtoh_scalars` signature at **0.3197** of a step, and
the three largest stalls followed a **0-byte** copy. The bytes are free; the queue
drain is not.

## 3. Why not copy `c10::Scalar` or `at::Tensor`

The source settles both questions.

`c10::Scalar` is a tagged union: a `double`, `int64_t`, `uint64_t`,
`complex<double>`, or a pointer, behind an eight-way `Tag`. It is trivially
copyable and about 24 bytes. **No tag selects the pointer member**, so a
`Scalar` carries a value: it is host-only by construction. When PyTorch needs a
scalar on the device, it uses a 0-dim `Tensor`.

`at::Tensor` is one member, `c10::intrusive_ptr<TensorImpl>`, so 8 bytes. Its
`TensorImpl` is about 208 bytes, with an 88-byte `SizesAndStrides`, a device
tag, a dtype, a dispatch key set, a version counter, and an autograd slot. The
handle is cheap. The policy around it is not.

| Option | Removes the host read | Cost here |
|---|---|---|
| a copy of `c10::Scalar` | no | an eight-way tag, complex and symint payloads, implicit conversions |
| a `Tensor` object | yes, through an impl allocation | refcounting, per-use dtype and device validation, a dispatch layer; [../README.md](../README.md) § What this is not forbids a tensor engine |
| **`ScalarArg`** | yes | two fields, one branch per kernel |

The loop issues about 16100 driver calls per step at 2 to 3 microseconds each.
Per-call operand validation is the wrong thing to add to that loop.

## 4. The type

```cpp
// One scalar argument to a kernel family.
//
// `value` is a host copy. When `dev` is not null the kernel loads the value
// there instead, once, before its loop, and ignores `value`. A null pointer,
// the default, keeps the host value, so every existing call site is unchanged
// and bit-identical.
//
// Use it for a scalar the optimizer owns. A constant or a schedule value stays
// a plain `float`: there is nothing to read.
struct ScalarArg {
  ScalarArg() = default;
  ScalarArg(float host) : value(host) {}  // NOLINT: implicit on purpose

  float value = 0.0f;
  const float* dev = nullptr;
};
```

`ScalarArg(float)` is implicit so that `params.alpha = 2.0f` and an aggregate
initialiser both keep compiling. The kernel body changes, and that change is the
point: `p.alpha * av` becomes `alpha * av`, with `alpha` resolved in the
prologue.

## 5. The seam diff

```diff
--- a/include/nanochat/kernels.h
+++ b/include/nanochat/kernels.h
@@
+// One scalar argument to a kernel family. See docs/scalar-arguments.md.
+struct ScalarArg {
+  ScalarArg() = default;
+  ScalarArg(float host) : value(host) {}  // NOLINT: implicit on purpose
+
+  float value = 0.0f;
+  const float* dev = nullptr;
+};
+
 struct PointwiseParams {
   PointwiseOp op = PointwiseOp::kScale;
   int count = 0;      // elements in `a`, in `out`, and in `dy` on the backward
   int row_width = 0;  // width of the row a broadcast reads; 0 => count
-  float alpha = 1.0f;
-  float beta = 0.0f;
+  ScalarArg alpha{1.0f};
+  ScalarArg beta{0.0f};
 };
@@ struct AdamWParams {
-  // The gradient clip scale, folded into the gradient read. One when the clip
-  // is disabled or inactive, so an unscaled read is bit-identical.
-  float grad_scale = 1.0f;
-  // The same scale as a device scalar. When it is not null the kernel reads it
-  // and ignores `grad_scale`, so the caller keeps the norm on the device and
-  // the step needs no host read of the clip. A null pointer, the default,
-  // keeps the host value.
-  const float* grad_scale_dev = nullptr;
+  // The gradient clip scale, folded into the gradient read.
+  ScalarArg grad_scale{1.0f};
 };
@@ struct MuonParams {
-  float grad_scale = 1.0f;
-  const float* grad_scale_dev = nullptr;
+  ScalarArg grad_scale{1.0f};
 };
```

The change is additive in behaviour and no enum grows a member, so the closed
`PointwiseOp` list stays closed and no backend gains a missing arm.

## 6. The backend arms

### 6.1 The optimizer pair, the first migration

`AdamWParams::grad_scale` and `MuonParams::grad_scale` migrate first. They are
the one site that already keeps a device scalar, so the type proves itself on
code that works today. The change is mechanical. The field pair becomes one
field, and the kernels read it in the prologue they already have.

The two structs lose the `_dev` field and the comment that explains the pair:

```diff
@@ struct AdamWParams
-  // The gradient clip scale, folded into the gradient read. One when the clip
-  // is disabled or inactive, so an unscaled read is bit-identical.
-  float grad_scale = 1.0f;
-  // The same scale as a device scalar. When it is not null the kernel reads it
-  // and ignores `grad_scale`, so the caller keeps the norm on the device and
-  // the step needs no host read of the clip. A null pointer, the default,
-  // keeps the host value.
-  const float* grad_scale_dev = nullptr;
+  // The gradient clip scale, folded into the gradient read. A caller that
+  // keeps the norm on the device sets `dev` and never reads the value back.
+  ScalarArg grad_scale{1.0f};
```

`backends/cuda/kernels/adamw.cu` drops the `bool scale_on_device` argument,
because the struct now says whether the value is on the device:

```diff
 __global__ void AdamWUpdateKernel(
     long long n, float lr, float beta1, float beta2, float eps,
-    float weight_decay, float step_size, float bias2, float grad_scale,
-    bool scale_on_device, const float* __restrict__ grad_scale_dev,
+    float weight_decay, float step_size, float bias2, ScalarArg grad_scale,
     float* __restrict__ master, ComputeType* __restrict__ value,
     const float* __restrict__ g, float* __restrict__ m, float* __restrict__ v) {
   const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
-  const float scale = scale_on_device ? *grad_scale_dev : grad_scale;
+  const float scale =
+      grad_scale.dev != nullptr ? *grad_scale.dev : grad_scale.value;
@@ void AdamWUpdate(int n, const AdamWParams& params, ...)
-  const bool scale_on_device = params.grad_scale_dev != nullptr;
   cuda_backend::Launch(AdamWUpdateKernel, dim3(grid), dim3(kThreads), 0,
                        static_cast<long long>(n), params.lr, params.beta1,
                        params.beta2, params.eps, params.weight_decay, step_size,
-                       bias2, params.grad_scale, scale_on_device,
-                       params.grad_scale_dev, master, value, g, m, v);
+                       bias2, params.grad_scale, master, value, g, m, v);
```

`backends/cuda/kernels/muon.cu` takes the same three edits at
`MuonMomentumKernel` and at the launch in `MuonUpdate`, which passes the same
pair today.

`backends/cpu/kernels.cc` loses two helpers and gains one:

```diff
-inline float GradScale(const AdamWParams& params) {
-  return params.grad_scale_dev != nullptr ? *params.grad_scale_dev
-                                          : params.grad_scale;
-}
-
-inline float GradScale(const MuonParams& params) {
-  return params.grad_scale_dev != nullptr ? *params.grad_scale_dev
-                                          : params.grad_scale;
-}
+inline float Coefficient(const ScalarArg& arg) {
+  return arg.dev != nullptr ? *arg.dev : arg.value;
+}
```

`src/optim.cc` sets one field instead of two at both group loops, and keeps its
local pointer for the one host read of the norm that closes the step:

```diff
       params.step = step;
-      params.grad_scale = 1.0f;
-      params.grad_scale_dev = grad_scale_dev;
+      params.grad_scale.dev = grad_scale_dev;
@@
-      params.grad_scale = 1.0f;
-      params.grad_scale_dev = grad_scale_dev;
+      params.grad_scale.dev = grad_scale_dev;
```

The local name `grad_scale_dev` stays, because it is the address the norm
reduction publishes and the closing host read uses it. Only the two parameter
structs change shape.

`src/optim_test.cc` keeps its own `float grad_scale` argument at lines 417, 428,
and 441. That is the test's reference arithmetic, not the seam, and it does not
change.

### 6.2 The pointwise arms

`backends/cuda/kernels/pointwise.cu` resolves the two coefficients in the
prologue of each kernel that uses them, in the shape the optimizer kernels now
share. The forward kernel uses `alpha` at the scale, scale-add, and softcap arms;
the backward kernel uses it at two places. Each becomes a local.

```cpp
const float alpha = p.alpha.dev != nullptr ? *p.alpha.dev : p.alpha.value;
const float beta = p.beta.dev != nullptr ? *p.beta.dev : p.beta.value;
```

### 6.3 The simulator needs no arm

It interposes CUDA calls, so it sees the launch and not the parameter struct.
`ScalarArg` reaches a kernel as a kernel argument either way, and the simulator
records the argument list as it already does.

## 7. The model and the ops layer

`src/ops.h` is not a frozen header, so `BlockForward` and `BlockBackward` take
device pointers:

```diff
 void BlockForward(const BlockShape& shape, const BlockWeights& weights,
                   const BlockActivations& acts, const int* tokens,
                   const float* cos, const float* sin, const ComputeType* x,
-                  const ComputeType* x0, float resid, float x0_lambda);
+                  const ComputeType* x0, const float* resid_dev,
+                  const float* x0_lambda_dev);
```

Inside, the residual becomes:

```cpp
PointwiseParams params;
params.op = PointwiseOp::kScaleAdd;
params.count = static_cast<int>(rows * hidden);
params.alpha.dev = resid_dev;
params.beta.dev = x0_lambda_dev;
kernels::PointwiseForward(params, x, x0, acts.xr);
```

`src/model.cc` passes the **fp32 master copy**, which is the authority the
optimizer updates. The change needs no new buffer and no conversion
kernel:

```cpp
// The fp32 view of one parameter, for a kernel that reads a device scalar. In
// a bf16 build this is the master copy, which the optimizer keeps current. In
// an fp32 build the compute buffer is already fp32.
const float* FloatView(const Param& p) {
  if (p.master != nullptr) return p.master;
  if constexpr (std::is_same_v<ComputeType, float>) return p.value;
  return nullptr;
}
```

The forward and backward loops pass `FloatView(resid) + i` and
`FloatView(x0_lambda) + i`. `ReadHostScalars` then has no caller and leaves the
tree.

## 8. What this removes, and what it does not

| Read | After this change |
|---|---|
| `resid_lambdas[i]`, `x0_lambdas[i]` | **gone**, 32 synchronous copies per step to zero |
| `smear_lambda` | stays. `SmearForward` and `SmearBackward` are host implementations that stage the whole activation, so they need a host value until Smear moves to the device |
| `backout_lambda` | stays, until `BackoutForward` stops staging |
| the gradient clip scale | **no traffic change.** `R2-D5` already keeps it on the device; rows S.1 to S.4 change its form and delete two duplicated helpers |

The Smear move is its own row, and it needs a second seam decision: a
pointwise code with a row-shifted operand, or a dedicated family. Section 9 of
[performance.md](performance.md) records it as
the next candidate with a ceiling of 0.2726.

## 9. The numerics: a precision change, and the measurement that found it

This document first claimed that the change is bit-identical. The measurement
disproved it, and the correction matters more than the claim did.

The old path read `resid_` and `x0_lambda_`, which are `ComputeType`. A bf16
build therefore staged the **bf16-rounded** scalar and widened it. The new path
reads `p.master`, which is fp32. The two differ in the bits that the bf16 copy
drops, so the residual blend consumes a more precise coefficient than before.

The reference run, 200 steps, the ScalarArg tree against the tree it replaced:

| Fact | Value |
|---|---|
| Worst relative loss difference | 4.85e-3 |
| Worst relative `grad_norm` difference | 2.5e-1 |
| First step that differs | step 1 |
| The unchanged tree against itself | 9.1e-4 loss, 3.8e-2 `grad_norm`, first difference at step 29 |

A difference at step 1 is not noise, and it is not the plan cache: it is the
parameter value itself. The change is therefore a precision change, and it
belongs in a decision record, not in a claim of identical bytes.

Two ways forward, and the tree currently takes the first:

1. **Accept it.** The optimizer owns the fp32 master, and the model derives the
   bf16 copy from it. A scalar that feeds float arithmetic should carry its fp32
   value. The cost is a new baseline for every parity fixture that pins the
   trajectory.
2. **Restore bit-identity.** Keep an fp32 mirror derived from `ComputeType`, so
   the coefficient is the bf16-rounded value widened again. That needs a device
   cast, and the seam has no general one: only `CastKernel` in `fp8.cu`, which
   writes fp8. Row S.9 below carries it.

The CPU backend and every fp32 build take path 1 for free, because their
`ComputeType` is `float` and `ScalarSource` returns the compute buffer itself.

## 10. Tests and the measurement

| Check | Why |
|---|---|
| `//backends/cpu:kernels_test` and `//backends/cuda/kernels:..._test` | the helper and the prologue read the same value as the host field |
| `//src:model_gradient_test` | the residual path is unchanged |
| `//tests:oracle_cuda_test` | a whole step still matches the fixture |
| `tools/nanochat lint` | the frozen header edit is formatted and ASCII |
| one PGO iteration on `base`/`h100` | the untraced step time and MFU, against the numeric floor of the gate |

The loop measurement is the acceptance for the performance claim. The
`blocking_dtoh_scalars` signature should read zero for the two arrays, and the
step should fall by what those copies cost, which the same trace bounds.

## 11. Work rows

| # | Work | Gate | Owner |
|---|---|---|---|
| S.1 | `ScalarArg` in `kernels.h`, and the three structs expressed as it: `PointwiseParams.alpha/beta`, `AdamWParams.grad_scale`, `MuonParams.grad_scale` | the header compiles for every backend, no enum grows a member, and no `_dev` field remains beside a host twin | architect |
| S.2 | the optimizer kernels: `adamw.cu` and `muon.cu` take a `ScalarArg`, drop `scale_on_device`, and resolve the scale in the prologue | `//src:optim_test` and `//src:optimizer_state_test` pass, and a step is bit-identical | kernel families |
| S.3 | `src/optim.cc`: one field at both group loops, and the closing host read unchanged | the norm and the clip still agree with the reference, and `grad_norm` is unchanged | Optimizer |
| S.4 | `backends/cpu/kernels.cc`: one `Coefficient` helper, and the two `GradScale` helpers deleted | `//backends/cpu:kernels_test` passes | Runtime |
| S.5 | the pointwise arms: the forward and backward kernels resolve the coefficients in the prologue | `//backends/cuda/kernels` tests pass, and a step is bit-identical | kernel families |
| S.6 | `src/ops.h` and `src/ops.cc`: `BlockForward` and `BlockBackward` take `const float*`, and the residual passes `alpha.dev` and `beta.dev` | `//src:model_gradient_test` and `//tests:oracle_cuda_test` pass | Harness |
| S.7 | `src/model.cc`: `FloatView`, the two call sites, and the removal of `ReadHostScalars` | no `ReadHost` remains in the forward or backward path of `src/model.cc` | Harness |
| S.8 | the loop iteration, and the report | step time and MFU, with the gate of `iter-01`'s floor | the operator |
| S.9 | optional: an fp32 mirror derived from `ComputeType`, and a device cast entry for it | the loss trace matches the pre-ScalarArg tree bit for bit | architect, Harness |

Rows S.1 to S.4 are the optimizer migration, and they landed first because that
site already keeps a device scalar. Rows S.5 to S.7 are the model scalars, and
they landed with them. Row S.9 is optional and it is the only one that would
restore bit-identity.

## 12. Non-goals

- A `Tensor` or `TensorRef` type in the seam. If the composition layer later
  wants a 24-byte borrow, `{data, count, dtype, device}`, it goes in `src/ops.*`
  or the C ABI and not in `kernels.h`.
- A dtype tag on `ScalarArg`. A build has one compute type, and the master copy
  is fp32 by construction.
- Migrating constants. `eps`, the softcap, `kQkScale`, and the schedule values
  stay host floats.
- Removing the `smear_lambda` and `backout_lambda` reads. That needs the Smear
  move first.

## 13. The question for the architect

Two forms are possible for the pointwise coefficient, and section 4 takes the
first:

1. **`ScalarArg` as a struct**, which costs one type in the frozen header and
   keeps every parameter struct to one field per scalar.
2. A pair of fields, `float alpha` and `const float* alpha_dev`, which adds no
   type but repeats the pair in every struct that migrates.

The proposal takes the first because `AdamWParams` and `MuonParams` already carry
the pair, and four more scalars are candidates. The second form is acceptable if
the architect prefers no new type in a frozen header: the work rows are the same
with the pair spelled out, and `ScalarArg` disappears.


## 14. What landed, and what it measured

Every row from S.1 to S.7 landed in the tree, and the whole tree builds for
fp32 and for bf16.

| Change | Where |
|---|---|
| `ScalarArg`, with the `host()` accessor and the implicit `float` constructor | `include/nanochat/kernels.h` |
| the three parameter structs on the new type, and the `_dev` twins deleted | `include/nanochat/kernels.h` |
| the prologue read, and the `bool scale_on_device` argument deleted | `backends/cuda/kernels/adamw.cu`, `muon.cu` |
| one `Coefficient` helper instead of two `GradScale` overloads | `backends/cpu/kernels.cc` |
| one field at both group loops | `src/optim.cc` |
| the coefficients resolved once per thread | `backends/cuda/kernels/pointwise.cu` |
| `ScalarArg` in the ops signatures, and `ops::DevScalar` | `src/ops.h`, `src/ops.cc` |
| `ScalarSource`, the two fp32 views, and `ReadHostScalars` deleted | `src/model.cc`, `src/model_impl.h` |

Two deviations from the sections above, both found by building:

- The accessor takes the name `host()`, because four consumers need the value
  and only the kernels want the raw pointer. It carries
  `#if defined(__CUDACC__)`, because `tensor.h` undefines its own macro.
- The fp32 fallback of `ScalarSource` is an **overload**, not an `if constexpr`.
  A discarded `if constexpr` branch still type-checks in a non-template member,
  so `return p.value` broke only the bf16 build, and the local fp32 build could
  not see it. The first container build failed on exactly that line.

Measured on `base` on the H100, two untraced runs of 200 steps:

| Point | Step time | Throughput | MFU | Two runs |
|---|---|---|---|---|
| Before, the iteration 1 tree | 42.90 ms | 95488 tokens/s | 2.3512 percent | 42.90, 43.71 ms |
| After, both scalar rows | 41.98 ms | 97672 tokens/s | 2.3834 percent | 41.98, 43.35 ms |

The step fell 2.1 percent and the MFU rose 1.4 percent. The two points overlap,
so the loop recorded the pair as reverted, and the 2.1 percent is inside the
noise of this card. Four of the eight synchronous scalar copies left the step:
`smear_lambda` and `backout_lambda` remain, as section 8 predicted.

The operator kept the change anyway, and the numbers do not express why. It is
the seam that the Smear work needs. It deletes two duplicated fields and two
duplicated helpers. And the precision change is the more correct behaviour.

`profiles/pgo/base-h100/iter-02/DECISION.md` records that decision beside the
measurement that argues against it.
