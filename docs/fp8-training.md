# The fp8 training path

Status: proposed. The tree holds no fp8 arithmetic today. This document is the
design and the plan for the Hopper target. It depends on Stage 2a of
[precision.md](precision.md), the bf16 compute path, which is also not
implemented.

The tree does hold the pieces this design builds on: the device profile table,
the kernel seam, the workspace, and the simulator. The document states which
piece each phase changes.

- [precision.md](precision.md) — the precision policy and the storage rule.
- [kernels.md](kernels.md) — the backend seam.
- [simulator-precision.md](simulator-precision.md) — the fp8 simulator engines.
- [performance-model.md](performance-model.md) — the bound.
- [attention-tensor-core.md](attention-tensor-core.md) — the attention coupling.

## 1. Goal and scope

**Goal.** Train the nanochat model at 8-bit floating point on Hopper, with the
optimizer state and the loss in fp32.

**In scope.**

- Two fp8 storage tags, E4M3 for the forward and E5M2 for the backward.
- A per-tensor scale with a delayed amax history.
- A scaled GEMM entry point in the kernel seam.
- The fp8 layer set, and the rule that keeps the other layers in fp32.
- The fp8 backward pass for the input gradient and the weight gradient.
- The correctness gates that run without a GPU.

**Out of scope.**

- A runtime precision switch. `DESIGN.md` section 6 forbids one. The precision
  stays a build decision.
- Loss scaling. `docs/precision.md` section 1 rejects it for fp32 and bf16. fp8
  uses a scale, but the scale is not a loss scale.
- fp8 in the optimizer, the norms, the softmax, or the loss.
- fp8 for the value gate and the smear gate.
- Block-scaled formats such as MXFP8 or NVFP4.

## 2. What the tree has today

| Item | Today | Path |
|---|---|---|
| Compute type | One compile-time type: fp32, or fp16 for the simulator | `include/nanochat/tensor.h:141-149`, `:224-226` |
| Storage tags | fp32, bf16, int8, fp16. No fp8 | `include/nanochat/tensor.h:177-182` |
| Capability flags | `supports_fp32`, `supports_fp16`. No bf16, no fp8 | `include/nanochat/tensor.h:235-257` |
| Device rates | fp32, fp16, bf16. No fp8 column | `include/nanochat/device_profile.h`, `src/device_profile.cc` |
| GEMM seam | `Gemm(mode, params, const ComputeType* a, const ComputeType* b, ComputeType* c)` | `include/nanochat/kernels.h:197` |
| GEMM engine | `cublasSgemm` and `cublasGemmEx` | `backends/cuda/gemm.cu` |
| Peak for the MFU line | `PeakFlopsForCompute` reads the fp32 rate, or the fp16 rate in the fp16 build | `src/train.cc` |
| fp8 test gates | None | — |

`docs/precision.md` section 6 lists the bf16 work as Stage 2. That work is the
prerequisite. Section 3.3 explains why.

The estimator already counts the arithmetic correctly for the new path.
`EstimateFlopsPerToken` in `src/train.cc` counts six operations per matmul
parameter and the attention terms. fp8 changes the rate, not the count.

## 3. The precision tier

### 3.1 The storage tags

Add two members to `DType` in `include/nanochat/tensor.h`:

- `kFp8E4M3` — one byte. Use it for the forward operands and the activations.
- `kFp8E5M2` — one byte. Use it for the backward operands.

The existing tags keep their values. `kFp32` stays 0, `kBf16` stays 1, `kInt8`
stays 2, and `kFp16` stays 3. The two new tags take the next two codes. The
value of `kComputeDType` does not change, so the on-disk contract does not
change in this work. The checkpoint container stays fp32.

The two formats differ in the exponent width. E4M3 has four exponent bits and
three mantissa bits. E4M3 saturates at 448. E5M2 has five exponent bits and two
mantissa bits. E5M2 carries a larger range and less precision.

| Format | Exponent bits | Mantissa bits | Largest value | Use |
|---|---|---|---|---|
| E4M3 | 4 | 3 | 448 | Forward operands |
| E5M2 | 5 | 2 | 57344 | Backward operands |

### 3.2 The rule that keeps the rest in fp32

One rule governs every buffer. Keep the following in fp32 at all times:

- the master weights;
- the parameter gradients;
- the optimizer moments;
- the norm statistics and the attention statistics;
- the loss, the softmax, and the row reductions.

Use fp8 only for the GEMM operands and the activations that feed them.
`docs/precision.md` section 5 already states this rule for bf16. fp8 inherits
it, because fp8 has less precision than bf16 in the mantissa.

An fp8 value carries a scale. The scale is a separate fp32 scalar or a separate
fp32 row vector. Never fold the scale into the loss.

### 3.3 The prerequisite: the bf16 path

The fp8 path needs the mixed-precision seam from Stage 2a. Two reasons:

1. The memory plan. The fp32 d24 training arena is 104 GiB at a device batch of
   16. The arena must shrink before the model fits an 80 GiB card.
2. The type seam. `ComputeType` is one type for the whole graph. fp8 must be an
   operand type, not the compute type.

Do Stage 2a first. Then add fp8 as a second operand type beside bf16.

## 4. Scaling

### 4.1 The scale rule

fp8 has few mantissa bits, so a raw cast loses most of the range. Apply one
scale to each operand instead.

```text
scaled = quantize(value / scale)
value  = dequantize(scaled) * scale
```

Use one scale per tensor. This is the tensorwise mode that the reference
nanochat uses. A finer scale mode, such as one scale per row, is later work.

The GEMM receives the scale as a pointer, not as a pre-multiplied operand. The
tensor core applies the scale during the multiply. So the host does not pay for
a second pass over the operand.

### 4.2 The amax history

Compute the scale from the largest absolute value of the tensor, the amax. A
single step is a bad sample, so keep a history.

1. Track a running amax per tensor with a decay.
2. Convert the running amax to a scale at the start of the step.
3. Use the scale for the whole step.
4. Update the history after the step, from the observed amax.

This is the delayed-scaling scheme. It costs one reduction per tensor per step
and no per-element work.

### 4.3 Where the scales live

Store the scales in the workspace beside the tensors they describe. The
workspace slot list is static, and `docs/model.md` owns it. Add one fp32 slot
per quantized tensor and one slot per amax history entry.

Record the scales in the checkpoint. A resumed run must reproduce the same
quantization. `docs/checkpoint-format.md` owns the container. Add the scales as
a separate optional record group, so an older reader still loads the fp32
weights.

## 5. The GEMM seam change

### 5.1 The new entry point

The current entry point takes `ComputeType` pointers:

```cpp
void Gemm(GemmMode mode, const GemmParams& params, const ComputeType* a,
          const ComputeType* b, ComputeType* c);  // kernels.h:197
```

Add a second entry point for scaled fp8. Keep the existing point unchanged, so
the Pascal, Turing, and CPU backends keep one code path.

```cpp
struct ScaledGemmParams {
  int m, n, k, batch_count;
  float alpha, beta;
  bool transpose_a, transpose_b;
  const float* scale_a;  // one value, or null for a scale of one
  const float* scale_b;
  const float* scale_c;  // optional, fp32 accumulate
};

void GemmScaledFp8(GemmMode mode, const ScaledGemmParams& params,
                   const std::uint8_t* a, const std::uint8_t* b,
                   float* c);
```

Two design points matter:

1. The accumulator and the output stay fp32. `C` is `float*`.
2. The operand type is `std::uint8_t*`, not `ComputeType*`. The byte carries an
   E4M3 or an E5M2 value. Keep the vendor type out of the header, as
   `DESIGN.md` section 2.1 rule 3 requires.

`include/nanochat/kernels.h` is frozen and architect-owned. This change is
additive, so it obeys the evolution rule in `DESIGN.md` section 7.

### 5.2 The cuBLASLt call

`backends/cuda/gemm.cu` uses `cublasSgemm` and `cublasGemmEx` today. The fp8
path needs `cublasLtMatmul`, because only that entry point accepts the scale
pointers.

The call sequence is:

1. Create a matrix layout for A, for B, and for C.
2. Create a matrix multiply descriptor.
3. Set the scale pointers on the descriptor.
4. Ask for a heuristic with `cublasLtMatmulAlgoGetHeuristic`.
5. Call `cublasLtMatmul`.

Set these values:

| Field | Value |
|---|---|
| A type, B type | `CUDA_R_8F_E4M3` in the forward, `CUDA_R_8F_E5M2` in the backward |
| C type | `CUDA_R_32F` |
| Compute type | `CUBLAS_COMPUTE_32F` |
| Scale type | `CUDA_R_32F` |

`cublasLt` ships with the CUDA toolkit. It is not a new third-party
dependency, so `DESIGN.md` section 6 stays satisfied. The backend build gains
one link edge, `@cuda//:cublasLt`, which the build agent owns.

Keep the workspace. `cublasLtMatmul` needs a workspace for some algorithms. Use
the existing workspace allocator, and size the request from the heuristic.

### 5.3 The capability gate

Require compute capability 8.9 or 9.0 for this path. Hopper reports 9.0 and Ada
reports 8.9. Ampere reports 8.0, and its tensor cores have no fp8 mode.

Add the gate in three places:

1. `Caps::supports_fp8` in `include/nanochat/tensor.h`.
2. `DeviceProfile::supports_fp8` in `include/nanochat/device_profile.h`.
3. The startup check in `src/train.cc`, beside the existing precision check.

A build with fp8 on a device without support must fail at startup. Do not fall
back at runtime, because `DESIGN.md` section 6 forbids a runtime precision
branch.

The host toolkit must be CUDA 11.8 or newer. The pinned host toolkit is CUDA
12.0, so the pinned host is enough. The H100 node records its own version
in [node-h100.md](node-h100.md).

## 6. The layer set

Apply fp8 to the matrix multiplies. Keep every other layer in bf16 or fp32.

### 6.1 The rule from the reference

The reference nanochat applies `--fp8` through a module filter. A matrix
multiply converts when all three conditions hold:

1. it is a matrix multiply;
2. both dimensions are divisible by 16;
3. the smaller dimension is at least 128.

The first condition excludes the gather, the row kernels, and the optimizer.
The second condition is a hardware requirement of the fp8 tensor core. The
third condition excludes the small gate matrices, which lose more than they gain.

Use the same rule. It is objective, and it needs no per-layer judgment.

### 6.2 The assignment at depth 24

| Operation | Precision | Reason |
|---|---|---|
| `c_q`, `c_k`, `c_v`, `c_proj` | fp8 | A matrix, and both dimensions are 1536 |
| `mlp.c_fc`, `mlp.c_proj` | fp8 | A matrix, 1536 by 6144 |
| the language-model head | fp8 | A matrix, 1536 by 32768 |
| `ve_gate` | bf16 | The output width is 12, which is below 128 |
| `smear_gate` | bf16 | The output width is 24, which is below 128 |
| the embedding | bf16 | A gather, not a matrix multiply |
| `rms_norm`, `qk_prep` | bf16 | A row kernel, and the statistics stay fp32 |
| the attention | bf16 | The attention engine runs the compute type |
| the attention softmax | fp32 | A row reduction with an exponential |
| the value residual and the backout | bf16 | A row blend |
| the classifier and the loss | fp32 | The loss must stay fp32 |
| AdamW and Muon | fp32 | The optimizer state stays fp32 |
| the gradient all-reduce | fp32 | The reduction operand stays fp32 |

The reference nanochat uses the same split. It converts every eligible matrix
multiply and keeps the optimizer in fp32.

A weight has three forms: the fp32 master, the fp8 operand, and the fp8 scale.
Keep the fp8 operand and the scale as workspace slots, and rebuild them once
per step.

### 6.3 The two costs that come with fp8

The fp8 path adds two costs. Both decide whether the change is worth it. See
the bound table in [performance-model.md](performance-model.md) section 7.2.

1. **The cast.** One kernel per operand measures the amax, computes the scale,
   and writes the fp8 value. A fused kernel moves about five bytes per element.
   An unfused sequence of calls can move six times that, and it makes fp8 slower
   than bf16. Fuse the amax, the scale, and the cast into one kernel.
2. **The layout.** The fp8 GEMM engine requires one operand row-major and the
   other column-major. Store each weight in the layout the GEMM wants, once at
   load time. Do not copy it every step.

## 7. The backward pass

The backward pass has two matrix multiplies per weight: the input gradient and
the weight gradient. Both need fp8 operands for the rate, and both need an fp32
accumulator.

| Call | Operand A | Operand B | Accumulator |
|---|---|---|---|
| Forward | the activation, E4M3 | the weight, E4M3 | fp32 |
| Input gradient (dgrad) | the output gradient, E5M2 | the weight, E4M3 | fp32 activation |
| Weight gradient (wgrad) | the output gradient, E5M2 | the activation, E4M3 | fp32 parameter gradient |

The reference uses E4M3 for the activation and the weight, and E5M2 for the
output gradient. E5M2 has a wider range, and a gradient has a wider spread than
an activation. Use the same pair.

Three hazards follow:

1. **The layout.** `cublasLtMatmul` accepts a limited set of fp8 layouts. A
   transpose may need a physical copy. Prefer a layout that the library takes
   directly, and fold any transpose into the existing pack step.
2. **The scale direction.** The wgrad operand is an activation that the forward
   pass already quantized. Reuse its scale. Do not compute a second scale for
   the same tensor in one step.

3. **The cast count.** An operand that the forward pass already quantized must
   not be quantized again. Reuse the stored fp8 tensor and its scale.

Keep the gradient in fp32 from the moment the accumulator writes it. Do not
quantize the parameter gradient.

### 7.1 The mixed choice

A safe first implementation uses fp8 in the forward only, and bf16 in the
backward. The backward pass carries the smaller arithmetic share in the first
weeks, and the bf16 path already exists. Add fp8 to the backward after the
forward gates pass.

Measure both. `docs/performance-model.md` holds the bound for each option.

## 8. The optimizer and the checkpoint

The optimizer needs no change. The master weight, the moments, and the update
stay fp32. `docs/precision.md` section 5 states the rule, and `src/optim.cc`
already obeys it for fp32.

The checkpoint needs one addition: the amax history and the scales. Treat them
as derived state with an exact-reproduce requirement. A resume that drops them
changes the first steps after the resume.

Keep the checkpoint records in fp32. The record dtype is separate from the
build type, so an fp8 build writes an fp32 checkpoint. `src/train.cc` already
converts a non-fp32 compute type to fp32 on save.

## 9. Invariants

1. The loss, the optimizer state, the norms, and the statistics are fp32.
2. No runtime precision switch exists. The precision is a build decision.
3. The fp8 GEMM writes a `float` accumulator.
4. Every fp8 operand has a scale, and no scale is null in a call.
5. The scale update happens once per step, before the first GEMM of the step.
6. The kernel seam holds no vendor type.
7. The simulator validates every fp8 call shape without a GPU.

## 10. Deliverables and owners

| Deliverable | Path | Owner |
|---|---|---|
| The storage tags and the capability flag | `include/nanochat/tensor.h` | Architect |
| The fp8 rate in the device table | `include/nanochat/device_profile.h`, `src/device_profile.cc` | Architect |
| The seam entry point | `include/nanochat/kernels.h` | Architect |
| The `cublasLt` GEMM | `backends/cuda/gemm.cu` | Kernel agents |
| The scale and amax kernels | `backends/cuda/kernels/` | Kernel agents |
| The workspace slots | `src/model.cc`, `include/nanochat/model.h` | Workflow |
| The layer wiring | `src/ops.cc`, `src/model.cc` | Workflow |
| The startup gate | `src/train.cc` | Harness |
| The checkpoint records | `src/train.cc`, `docs/checkpoint-format.md` | Harness |
| The CPU fp8 emulation | `backends/cpu/kernels.cc` | Runtime |
| The simulator coverage | `tools/cuda_sim/**` | Simulator |
| The test gates | `tests/`, `backends/cuda/kernels/` | Oracle |

## 11. Validation

Run every check below. The first five need no GPU.

1. The CPU fp8 emulation matches the CPU fp32 reference within the tolerance
   from [numerics-integration.md](numerics-integration.md).
2. The emulated fp8 GEMM matches the emulated CPU reference at the shapes in
   the model. Use `//tests:kernel_emu_test`.
3. The finite-difference check passes for the fp8 attention and MLP backward.
4. The API interposer accepts the `cublasLt` call shape and rejects a null
   scale pointer.
5. The staged numeric trace passes in the fp8 build on the `h100` profile.
6. The T1 fp8 kernel test passes on a real Ada or Hopper card.
7. One short training run matches the bf16 run within the recorded loss
   tolerance at the same step count.

Gate 7 is the only check that needs a Hopper-class card. Section 6 of
[numerics-integration.md](numerics-integration.md) owns the trace.

## 12. Phases

| Phase | Work | Gate | GPU |
|---|---|---|---|
| F0 | The tags, the capability flag, the rate column, and the seam entry point | `//tests:device_profile_test` passes; the tree builds for `sm_90` | no |
| F1 | The amax history and the scale update | The scale reaches the expected value on a synthetic tensor | no |
| F2 | The scaled GEMM through `cublasLt` | The emulated GEMM matches the CPU reference | no |
| F3 | The forward layer set | The fp8 forward matches the bf16 forward on the trace | no |
| F4 | The backward pass, in the mixed mode first | The finite-difference check passes | no |
| F5 | The full fp8 backward, and the scale checkpoint records | The one-step loss matches bf16 within tolerance | no |

## 13. Status by section

| Section | Status |
|---|---|
| 3 Precision tier | Proposed |
| 4 Scaling | Proposed |
| 5 Seam change | Proposed |
| 6 Layer set | Proposed |
| 7 Backward pass | Proposed |
| 8 Optimizer and checkpoint | Proposed |
| 12 Phases | Not started |

## 14. Risks and non-goals

- **The rate may disappoint.** fp8 doubles the peak only where the tensor core
  is the limit. The attention softmax and the transposes do not speed up. See
  the bound in [performance-model.md](performance-model.md).
- **The layout may need a copy.** A required transpose adds a pass over the
  operand and removes part of the gain.
- **The emulator cannot prove the hardware conversion.** Only a real card
  proves the fp8 rounding. Keep one hardware leg.
- **The accuracy may drop.** fp8 is lossy. A short comparison run is mandatory
  before a long one.
- **Not a goal:** a runtime precision switch, block-scaled formats, fp8 in the
  optimizer, and fp8 for the gates.

## 15. Open questions

1. Does the weight gradient use E5M2, or the same E4M3 as the forward?
2. Does the scale live per tensor or per row for the attention operands?
3. Does the checkpoint store the amax history, or only the current scales?
4. Which stage of the pipeline owns the fp8 conversion of the embedding table?

## See also

- [precision.md](precision.md) — the precision policy.
- [simulator-precision.md](simulator-precision.md) — the fp8 simulator engines.
- [performance-model.md](performance-model.md) — the bound.
- [attention-tensor-core.md](attention-tensor-core.md) — the attention coupling.
- [numerics-integration.md](numerics-integration.md) — the tolerance table.
