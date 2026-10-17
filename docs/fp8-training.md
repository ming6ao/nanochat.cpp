# The fp8 training path

Status: proposed, after a review and a decision pass. The tree holds no fp8
arithmetic. The prerequisite is met: the bf16 compute path from Stage 2 of
[precision.md](precision.md) is delivered (commit `387a88a`).

This document is the design and the plan for the Hopper target. It states which
piece each phase changes.

The reference is the PyTorch tree. The file `nanochat/fp8.py` holds the fp8
implementation, and `scripts/base_train.py` holds the module filter and the
evaluation exclusion. This repository is at `../nanochat`.

- [precision.md](precision.md) — the precision policy and the storage rule.
- [kernels.md](kernels.md) — the backend seam.
- [simulator-precision.md](simulator-precision.md) — the fp8 simulator engines.
- [performance-model.md](performance-model.md) — the bound and the precision
  rule.
- [attention-tensor-core.md](attention-tensor-core.md) — the attention coupling.
- [numerics-integration.md](numerics-integration.md) — the tolerance table.

## 1. Goal and scope

**Goal.** Train the nanochat model at 8-bit floating point on Hopper, with the
optimizer state and the loss in fp32.

**In scope.**

- Two fp8 storage tags: E4M3 for the forward and E5M2 for the backward.
- A per-tensor scale, derived from the current operand.
- A scaled GEMM entry point in the kernel seam.
- The fp8 layer set, and the rule that keeps the other layers in bf16 or fp32.
- The fp8 backward pass for the input gradient and the weight gradient.
- The correctness gates that run without a GPU.

**Out of scope.**

- A runtime precision switch. `DESIGN.md` section 6 forbids one. The precision
  stays a build decision.
- Loss scaling. `docs/precision.md` section 1 rejects it for fp32 and bf16.
  fp8 uses a scale, but the scale is not a loss scale.
- fp8 in the optimizer, the norms, the softmax, or the loss.
- fp8 for the value gate and the smear gate.
- Block-scaled formats such as MXFP8 or NVFP4.
- A delayed amax history and a rowwise scale. Section 15 records the reasons.

## 2. What the tree has today

| Item | Today | Path |
|---|---|---|
| Compute type | One compile-time type: fp32, or bf16 | `include/nanochat/tensor.h:118-125` |
| Storage tags | fp32, bf16, int8. The code 3 is retired | `include/nanochat/tensor.h:149-153` |
| Capability flags | `supports_fp32`, `supports_bf16`. No `supports_fp8` | `include/nanochat/tensor.h:199-219` |
| Device rates | fp32, bf16, and fp8. The `h100` row carries 1979 TFLOP/s | `src/device_profile.cc:24` |
| GEMM seam | `Gemm` and `GemmWgrad` take `ComputeType` pointers | `include/nanochat/kernels.h:197`, `:202` |
| GEMM engine | `cublasSgemm` and `cublasGemmEx` | `backends/cuda/gemm.cu` |
| Precision assignment | `AssignMatrixPrecision` applies the fp8 rule | `src/performance_model.cc:185-197` |
| Peak for the MFU line | `PeakFlopsForCompute` knows the fp8 rate | `src/train.cc:325-340` |
| Roofline | The fp8 build and the fp8 rates are reportable | `src/roofline_main.cc` |
| fp8 arithmetic | None | — |
| fp8 build define | None | `.bazelrc` |
| fp8 test gates | None | `tests/` |

The estimator already counts the arithmetic correctly for the new path.
`EstimateFlopsPerToken` in `src/train.cc` counts six operations per matmul
parameter and the attention terms. fp8 changes the rate, not the count.

**The revision rule.** `docs/performance-model.md` section 5 owns the precision
rule, and `src/performance_model.cc` implements it. This review aligned this
document with that code and with the reference. A later change to the rule
changes both documents in one commit.

## 3. The precision tier

### 3.1 The storage tags

Add two members to `DType` in `include/nanochat/tensor.h`:

- `kFp8E4M3` — one byte. Use it for the forward operands and the activations.
- `kFp8E5M2` — one byte. Use it for the backward operands.

The existing tags keep their values. `kFp32` stays 0, `kBf16` stays 1, and
`kInt8` stays 2. The code 3 belonged to `kFp16`, and it stays retired. The two
new tags take the next two codes. The value of `kComputeDType` does not change,
so the on-disk contract does not change in this work. The checkpoint container
stays fp32.

The two formats differ in the exponent width. E4M3 has four exponent bits and
three mantissa bits. E5M2 has five exponent bits and two mantissa bits. E5M2
carries a larger range and less precision.

| Format | Exponent bits | Mantissa bits | Largest value | Use |
|---|---|---|---|---|
| E4M3 | 4 | 3 | 448 | Forward operands |
| E5M2 | 5 | 2 | 57344 | Backward operands |

Use the finite encoding. The reference calls the E4M3 form
`torch.float8_e4m3fn`, and cuBLAS names it `CUDA_R_8F_E4M3`. This form has no
infinity. The `E4M3FNUZ` form is a different encoding with a different bias and
no negative zero. Do not mix the two.

### 3.2 The rule that keeps the rest in fp32

One rule governs every buffer. Keep the following in fp32 at all times:

- the master weights;
- the parameter gradients;
- the optimizer moments;
- the norm statistics and the attention statistics;
- the loss, the softmax, and the row reductions.

Use fp8 only for the GEMM operands and the activations that feed them.
`docs/precision.md` section 5 already states this rule for bf16. fp8 inherits
it, because fp8 has fewer mantissa bits than bf16.

An fp8 value carries a scale. The scale is a separate fp32 device scalar.
Never fold the scale into the loss.

The logits tensor takes the compute type. The softcap, the softmax, the
cross-entropy, and every reduction stay fp32.

### 3.3 The prerequisite: the bf16 path (delivered)

Status: delivered. The bf16 path landed in commit `387a88a` and in
`docs/precision.md` Stage 2. The fp8 work no longer waits on it.

Two results of that stage still shape this work.

1. **The type seam.** `ComputeType` is one type for the whole graph. fp8 is an
   operand type, not the compute type. An fp8 operand is a byte buffer with a
   separate scale, and `ComputeType` stays bf16 beside it.
2. **The memory plan.** The fp32 depth-24 training arena is 104 GiB at a device
   batch of 16. The bf16 arena is 52.2 GiB, and the bf16 total is 77.4 GiB
   against an 80 GiB card. The margin is about 1 GiB. The fp8 arena is 26.1 GiB
   and the fp8 total is 47.8 GiB. Batch 32 fits at 76.1 GiB
   ([node-h100.md](node-h100.md) section 3).

## 4. Scaling

### 4.1 The scale rule

fp8 has few mantissa bits, so a raw cast loses most of the range. Apply one
scale to each operand instead. The reference computes the largest absolute
value of the tensor, the amax, and derives the scale from it.

```text
fp8_max = 448        for E4M3
fp8_max = 57344      for E5M2

scale   = fp8_max / max(amax, 1e-12)
scaled  = clamp(value * scale, -fp8_max, fp8_max)
fp8     = cast(scaled)
value  ~= fp8 / scale
```

Three points matter.

1. **The clamp is required.** A cast without the clamp does not saturate. In
the `E4M3FN` encoding a large value overflows to NaN, and the reference clamps
to `fp8_max` to avoid it.
2. **The `1e-12` floor is required.** An all-zero tensor has an amax of zero,
   and the scale would be infinite.
3. **Two names exist for the same number.** The reference names the multiplier
   `scale` and passes its reciprocal to `torch._scaled_mm`. The `cublasLt`
   scale pointer is the dequantization value, so it takes `1 / scale`. Write
   the multiplier in the kernel and convert at the seam.

Use one scale per tensor. This is the tensorwise mode, and it is the reference
default. It is also the only recipe the reference implements; a rowwise scale
needs the CUTLASS kernel. A rowwise scale is later work.

The GEMM receives the scale as a pointer. The tensor core applies the scale
during the multiply, so the host does not pay for a second pass over the
operand.

Round to nearest, with ties to even. The reference uses the default PyTorch
cast, and the fp8 conversion instructions use the same rule. Write the rule
once, in the shared function of section 10, so the central processing unit,
the emulator, and the device agree.

### 4.2 The amax source: dynamic, not delayed

The scale comes from the amax of the operand in the same step. The reference
computes it that way: `_to_fp8` calls `x.abs().max()` on every quantization.
This is dynamic scaling. The tree keeps no amax history and no decay.

1. Measure the amax of the tensor on the device.
2. Derive the scale on the device, with the floor from section 4.1.
3. Cast the tensor with that scale, in the same fused kernel.
4. Keep the fp8 tensor and its scale for the step. Reuse them for every GEMM
   that reads the tensor.

Three reasons support this choice.

1. **Parity.** Dynamic scaling is what the reference does. A decayed history
   makes the fp8 loss diverge from the reference by an unknown amount, and it
   needs its own parity record.
2. **The delivered estimator already prices it.** `src/performance_model.cc:273-287`
   prices the cast at "five bytes of traffic per element (measure, then read
   and write)". That is the dynamic scheme.
3. **No persistent state.** There is nothing to checkpoint. A resume
   reproduces exactly, because the scale is a function of the tensor.

The scale never leaves the device. The cast kernel writes it to a device
scalar, and the GEMM reads it. The host does not read the amax, so no device
synchronization occurs.

A delayed amax history with a decay is the NVIDIA Transformer Engine scheme.
It saves one read of the operand. Section 15 records it as a later option, and
it needs its own parity entry.

### 4.3 Where the scales live

Store the fp8 operand and its scale in the workspace beside the tensor they
describe. The workspace slot list is static, and `docs/model.md` owns it. Add
one byte slot per quantized tensor and one fp32 slot per scale.

The checkpoint stores no scale state. The scale is derived within the step
(section 4.2), and a resume reproduces it. `docs/checkpoint-format.md` does not
change in this work.

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
  bool fast_accum;       // reduced-precision reduction; forward only
  const float* scale_a;  // the dequantization value, 1 / scale from 4.1
  const float* scale_b;
};

void GemmScaledFp8(GemmMode mode, const ScaledGemmParams& params,
                   const std::uint8_t* a, const std::uint8_t* b, float* c);
```

Four design points matter:

1. The accumulator and the output stay fp32. `C` is `float*`.
2. The operand type is `std::uint8_t*`, not `ComputeType*`. The byte carries an
   E4M3 or an E5M2 value. Keep the vendor type out of the header, as
   `DESIGN.md` section 2.1 rule 3 requires.
3. There is no C scale. The caller passes `beta = 0` for a pure overwrite. The
   reference has no C scale, and the design does not add one.
4. The `transpose_a` and `transpose_b` flags describe the logical layout. The
   backend maps them to the `cublasLt` stride triplets.

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
| Scale attribute | `CUBLASLT_MATMUL_DESC_A_SCALE_POINTER`, `..._B_SCALE_POINTER` |
| Fast accumulation | `CUBLASLT_MATMUL_DESC_FAST_ACCUM`: on in the forward, off in the backward |

The fast accumulation setting changes the numerics. The reference sets
`use_fast_accum=True` in the forward and `False` in the two backward GEMMs
(`nanochat/fp8.py:152`, `:173`, `:189`). Match it, and record the setting in
the tolerance table.
A reduced-precision reduction in the forward adds a small error that the
backward does not have.

`cublasLt` ships with the CUDA toolkit. It is not a new third-party dependency,
so `DESIGN.md` section 6 stays satisfied. The backend build gains one link
edge, `@cuda//:cublasLt`, which the build agent owns.

Keep the workspace. `cublasLtMatmul` needs a workspace for some algorithms. Use
the existing workspace allocator, and size the request from the heuristic.

### 5.3 The capability gate

Require compute capability 8.9 or 9.0 for this path. Hopper reports 9.0 and Ada
reports 8.9. Ampere reports 8.0, and its tensor cores have no fp8 mode.

Add the gate in three places:

1. `Caps::supports_fp8` in `include/nanochat/tensor.h`.
2. `DeviceProfile::supports_fp8` in `include/nanochat/device_profile.h`.
3. The model-creation check in `src/model.cc` (`Model::Create`), beside the
   existing compute-dtype check at `src/model.cc:338`. That check is the
   existing fail-fast pattern, so fp8 follows it.

A build with fp8 on a device without support must fail when the model is
created. Do not fall back at runtime, because `DESIGN.md` section 6 forbids a
runtime precision branch.

The host toolkit must be CUDA 11.8 or newer. The pinned host toolkit is CUDA
12.0, so the pinned host is enough. The H100 node records its own version
in [node-h100.md](node-h100.md).

## 6. The layer set

Apply fp8 to the matrix multiplies. Keep every other layer in bf16 or fp32.

### 6.1 The rule from the reference

The reference applies `--fp8` through a module filter. A matrix multiply
converts when all three conditions hold:

1. it is a matrix multiply;
2. both dimensions are divisible by 16;
3. the smaller dimension is at least 128.

The first condition excludes the gather, the row kernels, and the optimizer.
The second condition is a hardware requirement of the fp8 tensor core. The
third condition excludes the small gate matrices, which lose more than they
gain.

Use the same rule. It is objective, and it needs no per-layer judgment.
`src/performance_model.cc` already implements it as `AssignMatrixPrecision`.

The reference filter accepts a `torch.nn.Linear` and nothing else. The token
embedding and the value embeddings are `torch.nn.Embedding`, so they never
convert. The reference casts them to the compute type for the forward
(`nanochat/gpt.py:262-266`), and the fp32 master stays.

### 6.2 The assignment at depth 24

| Operation | Precision | Reason |
|---|---|---|
| `c_q`, `c_k`, `c_v`, `c_proj` | fp8 | A matrix, and both dimensions are 1536 |
| `mlp.c_fc`, `mlp.c_proj` | fp8 | A matrix, 1536 by 6144 |
| the language-model head | fp8 | A matrix, 1536 by 32768 |
| the logits tensor | The compute type (bf16) | The GEMM writes the compute type |
| the softcap and the softmax | The compute type; the statistics fp32 | A row kernel, and the reduction stays fp32 |
| the loss | fp32 | The loss must stay fp32 |
| `ve_gate` | bf16 | The output width is 12, which is below 128 |
| `smear_gate` | bf16 | The output width is 24, which is below 128 |
| the embedding | bf16 | A gather, not a matrix multiply |
| `rms_norm`, rope, `qk_prep` | bf16 | A row kernel |
| the attention | bf16 | The reference attention engine runs the compute type |
| the value residual and the backout | bf16 | A row blend |
| AdamW and Muon | fp32 | The optimizer state stays fp32 |
| the gradient all-reduce | fp32 | The reduction operand stays fp32 |

The reference uses the same split. It converts every eligible matrix multiply
and keeps the optimizer in fp32.

A weight has three forms: the fp32 master, the fp8 operand, and the fp8 scale.
Keep the fp8 operand and the scale as workspace slots, and rebuild them once
per step. Do not build a separate bf16 copy of an eligible weight.

### 6.3 The two costs that come with fp8

The fp8 path adds two costs. Both decide whether the change is worth it. See
the bound table in [performance-model.md](performance-model.md) section 7.2.

1. **The cast.** One kernel per operand measures the amax, computes the scale,
   and writes the fp8 value. A fused kernel moves about five bytes per element.
   An unfused sequence of calls moves six times that, and it makes fp8 slower
   than bf16. Fuse the amax, the scale, and the cast into one kernel.
2. **The layout.** The fp8 GEMM engine requires operand A row-major and operand
   B column-major. Store each weight in the layout the GEMM wants, once at load
   time. Do not copy it every step.

### 6.4 The evaluation precision

Evaluation and generation run in the compute type, not in fp8. The reference
disables fp8 around every evaluation call (`with disable_fp8(model)` in
`scripts/base_train.py`). The reason is accuracy: an fp8 forward changes the
sampled logits.

Match the reference. Do not build the fp8 operands on the evaluation path. An
`eval` path that runs fp8 needs a new decision and a new parity record.

## 7. The backward pass

The backward pass has two matrix multiplies per weight: the input gradient and
the weight gradient. Both need fp8 operands for the rate, and both need an fp32
accumulator.

| Call | Operand A | Operand B | Accumulator | Output |
|---|---|---|---|---|
| Forward | the activation, E4M3 | the weight, E4M3 | fp32 | The compute type |
| Input gradient (dgrad) | the output gradient, E5M2 | the weight, E4M3 | fp32 | The compute type |
| Weight gradient (wgrad) | the output gradient, E5M2 | the activation, E4M3 | fp32 | fp32 |

The reference uses E4M3 for the activation and the weight, and E5M2 for the
output gradient. E5M2 has a wider range, and a gradient has a wider spread than
an activation. Use the same pair. Both gradient GEMMs share the output-gradient
operand and its scale.

The output types differ on purpose. The input gradient is an activation for the
next layer, so it takes the compute type. The weight gradient is a persistent
parameter gradient, so it stays fp32 (section 3.2).

Four hazards follow:

1. **The layout.** `cublasLtMatmul` accepts a limited set of fp8 layouts. Name
   each tensor, because the three GEMMs disagree.

   | GEMM | Operand A | Operand B |
   |---|---|---|
   | Forward | the activation, row-major | the weight transpose, column-major |
   | Input gradient | the output gradient, row-major | the weight, column-major |
   | Weight gradient | the output gradient transpose, row-major | the activation, column-major |

   A weight stored `[N, K]` row-major gives its `[K, N]` column-major transpose
   as operand B with no copy. The same buffer serves the forward and the input
   gradient. That removes the weight copy. The output-gradient transpose and
   the column-major activation still need a rearrangement in the reference. The
   target removes both: the fused cast writes the activation in both layouts,
   and the backward reads the output gradient with a column-major stride. See
   the `fp8_layout_copies` row in `src/performance_model.cc:296-305`.
2. **The scale direction.** The wgrad operand is an activation that the forward
   pass already quantized. Reuse its scale. Do not compute a second scale for
   the same tensor in one step.
3. **The cast count.** An operand that the forward pass already quantized must
   not be quantized again. Reuse the stored fp8 tensor and its scale.
4. **The shared output gradient.** One output gradient feeds the input-gradient
   GEMM and the weight-gradient GEMM of the same layer. Quantize it once per
   layer, and pass the same tensor and scale to both.

Set the fast accumulation off in both backward GEMMs, as the reference does.

Keep the parameter gradient in fp32 from the moment the fp32 accumulator
writes it. Do not quantize the parameter gradient.

### 7.1 The mixed choice

A safe first implementation uses fp8 in the forward only, and bf16 in the
backward. The backward pass carries the smaller arithmetic share in the first
weeks, and the bf16 path already exists. Add fp8 to the backward after the
forward gates pass.

Measure both. `docs/performance-model.md` holds the bound for each option.

## 8. The optimizer and the checkpoint

The optimizer needs no change. The master weight, the moments, and the update
stay fp32. `docs/precision.md` section 5 states the rule, and `src/optim.cc`
already obeys it. The fp8 operand is a derived buffer, and the optimizer reads
the master. The reference shares the parameter the same way
(`Float8Linear.from_float`).

The checkpoint needs no addition. The tree keeps no scale state (section 4.2),
so a resume reproduces every scale exactly. Keep the checkpoint records in
fp32. The record dtype is separate from the build type, so an fp8 build writes
an fp32 checkpoint. `src/train.cc` already converts a non-fp32 compute type to
fp32 on save.

## 9. Invariants

1. The loss, the optimizer state, the norms, and the statistics are fp32.
2. No runtime precision switch exists. The precision is a build decision.
3. The fp8 GEMM writes a `float` accumulator.
4. Every fp8 operand has a scale, and no scale is null in a call.
5. The scale is derived from the current operand in the same step, before the
   first GEMM that reads it. The tree keeps no decayed history.
6. The kernel seam holds no vendor type.
7. The simulator validates every fp8 call shape without a GPU.
8. The scale is a device scalar. The host does not read the amax in the step.
9. The fp8 cast saturates. A value above `fp8_max` clamps to `fp8_max`.
10. Evaluation and generation run in the compute type (section 6.4).

## 10. Deliverables and owners

| Deliverable | Path | Owner |
|---|---|---|
| The storage tags and the capability flag | `include/nanochat/tensor.h` | Architect |
| The fp8 rate in the device table | delivered | Architect |
| The seam entry point | `include/nanochat/kernels.h` | Architect |
| The `cublasLt` GEMM and the fast-accumulation attribute | `backends/cuda/gemm.cu` | Kernel agents |
| The fused amax, scale, and saturating cast kernel | `backends/cuda/kernels/` | Kernel agents |
| The workspace slots | `src/model.cc`, `include/nanochat/model.h` | Workflow |
| The layer wiring and the evaluation exclusion | `src/ops.cc`, `src/model.cc` | Workflow |
| The creation-time gate | `src/model.cc` (`Model::Create`) | Workflow |
| The `--config=fp8` build define and the `h100` fp8 config | `.bazelrc`, `nanochat.bzl` | Architect and Build |
| The shared fp8 rounding function and the CPU emulation | `include/nanochat/fp8.h`, `backends/cpu/kernels.cc` | Runtime |
| The simulator rounding and the `cublasLt` coverage | `tools/cuda_sim/**` | Simulator |
| The fixtures, the dump command, and the test gates | `tests/`, `tests/data/`, `tools/dump_training_fixtures.py` | Oracle |
| The checkpoint records | none needed (section 8) | — |

## 11. Validation

Run every check below. The first six need no GPU.

1. The shared fp8 rounding rule, which covers the saturation, the floor, and
   the round-to-nearest-even rule. The CPU emulation and the simulator share
   one function, so one fixture proves both. Fixture
   `tests/data/fp8_round_fixture.bin`, gate `//tests:fp8_round_test`.
2. The emulated scaled GEMM against the CPU fp32 reference, at the shapes in
   the model. Fixture `tests/data/fp8_gemm_fixture.bin`, gate
   `//tests:fp8_gemm_test`, plus the fp8 leg of `//tests:kernel_emu_test`.
3. The finite-difference check for the fp8 attention and MLP backward.
4. One fp8 forward and one training step against the reference trajectory.
   Fixture `tests/data/fp8_step_fixture.bin`, gate `//tests:fp8_step_test`.
5. The API interposer accepts the `cublasLtMatmul` shape with both scale
   pointers, and it rejects a null scale pointer. Gate:
   `tools/nanochat simulate --suite api`, and `//tests:cuda_sim_preload_test`.
6. The staged numeric trace passes in the fp8 build on the `h100` profile.
   Gate: `tools/nanochat test //tests:numerics_trace_test --config=fp8`, and
   `tools/nanochat simulate --device h100`.
7. The T1 fp8 kernel test passes on a real Ada or Hopper card. Gate:
   `//backends/cuda/kernels:fp8_gemm_gpu_test`.
8. One short training run matches the bf16 run within the recorded loss
   tolerance at the same step count.

Two prerequisites gate the list.

1. **The tolerance table.** `docs/numerics-integration.md` section 4 needs fp8
   rows before the trace can be required: the fp8 loss, the fp8 gradient norm,
   and the fp8 parameter L2. The fast accumulation changes the forward
   numerics, so record it with the rows.
2. **The bf16 trace leg.** The fp8 trace needs a stored golden from its own
   precision family. The bf16 trace machinery is the first instance, and it is
   still open (`docs/precision.md` section 6, "What the bf16 leg does not cover
   yet"). Do not schedule the fp8 trace before the bf16 trace.

Gate 8 is the only check that needs a Hopper-class card. Section 6 of
[numerics-integration.md](numerics-integration.md) owns the trace.

## 12. Phases

| Phase | Work | Gate | GPU |
|---|---|---|---|
| F0 | The storage tags, the capability flag, the seam entry point, the build define, and the shared rounding function | `//tests:device_profile_test` passes; the tree builds for `sm_90` and for `--config=fp8` | no |
| F1 | The fused amax, scale, and saturating cast | The round fixture and the scale test pass | no |
| F2 | The scaled GEMM through `cublasLt`, with the fast-accumulation attribute | The emulated GEMM matches the CPU reference | no |
| F3 | The forward layer set, and the evaluation exclusion | The fp8 forward matches the bf16 forward on the trace | no |
| F4 | The backward pass, in the mixed mode first | The finite-difference check passes | no |
| F5 | The full fp8 backward, with the type and output table | The one-step loss matches bf16 within tolerance | no |
| F6 | The `cublasLt` interposer coverage and the trace leg | The simulator suites pass | no |
| F7 | The hardware confirmation | The fp8 trace and the comparison run pass on a real card | yes |

The Oracle owns the fixtures. Commit the F1 and F2 fixtures before F3 starts,
so the forward work has a gate.

## 13. Status by section

| Section | Status |
|---|---|
| 3 Precision tier | Proposed; the tags are not in the tree |
| 4 Scaling | Decided: dynamic tensorwise scaling (section 4.2) |
| 5 Seam change | Proposed |
| 6 Layer set | Decided; aligned with `src/performance_model.cc` |
| 7 Backward pass | Decided; the format table matches `src/performance_model.cc` |
| 8 Optimizer and checkpoint | No work needed (section 8) |
| 12 Phases | Not started |

## 14. Risks and non-goals

- **The rate may disappoint.** fp8 doubles the peak only where the tensor core
  is the limit. The attention softmax and the transposes do not speed up. See
  the bound in [performance-model.md](performance-model.md).
- **The layout may need a copy.** A required transpose adds a pass over the
  operand and removes part of the gain. Section 7 names the removal.
- **The emulator cannot prove the hardware conversion.** Only a real card
  proves the fp8 rounding and the layout. Keep one hardware leg.
- **The fast accumulation changes the forward numerics.** The comparison
  tolerance must cover it. Do not compare an fp8 forward against a bf16 forward
  with the fp32 tolerance.
- **The two encodings differ.** `CUDA_R_8F_E4M3` is the finite form. A library
  that expects `E4M3FNUZ` gives different bytes for the same input.
- **The accuracy may drop.** fp8 is lossy. A short comparison run is mandatory
  before a long one.
- **The evaluation path is not covered.** Evaluation runs in the compute type,
  so an fp8 build does not prove an fp8 evaluation path. Do not add one without
  a new decision (section 6.4).
- **The tests may diverge from the reference.** The comparison run is the only
  check of the fp8 accuracy against the reference. Record it in the run record
  of [node-h100.md](node-h100.md).
- **Not a goal:** a runtime precision switch, block-scaled formats, fp8 in the
  optimizer, fp8 for the gates, a delayed amax history, a rowwise scale, and an
  fp8 evaluation path.

## 15. Decisions on the former open questions

Each decision cites its basis. The first four are the former open questions.

1. **The weight-gradient operand format.** Both gradient GEMMs use E5M2 for the
   output gradient and E4M3 for the other operand. The input gradient uses the
   weight as the second operand. The weight gradient uses the activation as the
   second operand. The weight gradient itself is fp32.
   Basis: `nanochat/fp8.py:158-192` reuses `go_fp8` (E5M2) for both gradient
   GEMMs. `src/performance_model.cc` implements the same table.
2. **The scale granularity.** One scale per tensor (tensorwise). The former
   question asked about "the attention operands", and that premise is void: the
   reference converts only `torch.nn.Linear`, so the attention engine stays in
   the compute type. The query, key, value, and output projections use the same
   tensorwise scale as every other matrix.
   Basis: `nanochat/fp8.py:91`; `scripts/base_train.py:179-190`;
   `nanochat/gpt.py:110`. A rowwise scale needs the CUTLASS kernel and is out
   of scope.
3. **The checkpoint content for the scale state.** Neither the amax history nor
   the current scales. The tree keeps no scale state, so the checkpoint does
   not change.
   Basis: `nanochat/fp8.py:93-95` measures the amax on every call. Section 4.2
   adopts the same scheme.
4. **The owner of the embedding conversion.** No stage converts the embedding
   table to fp8. The gather reads the compute copy. The workflow layer owns
   that copy (`src/model.cc`), and the loader keeps the fp32 master.
   Basis: `scripts/base_train.py:180-186` accepts `nn.Linear` only;
   `nanochat/gpt.py:174` and `:192` are `nn.Embedding`.
5. **The amax source.** Dynamic, from the current operand.
   Basis: the reference, and the delivered cast row at
   `src/performance_model.cc:273-287`.
6. **The fast accumulation.** On in the forward, off in the two backward GEMMs.
   Basis: `nanochat/fp8.py:152`, `:173`, `:189`.
7. **The evaluation precision.** The compute type (bf16). Evaluation and
   generation never build the fp8 operands.
   Basis: `scripts/base_train.py:426`, `:445`, `:472`.
8. **The fixtures, the producer, and the gate names.** The Oracle owns them.
   Add the `fp8` command to `tools/dump_training_fixtures.py`, and add the
   three fixtures of section 11.
9. **The layout removal.** The fused cast writes the fp8 activation in both
   layouts, and the backward reads the output gradient with a column-major
   stride. The `fp8_layout_copies` row stays zero.
   Basis: `docs/performance-model.md` section 5.3, and
   `nanochat/fp8.py:181-182`.

One item stays a later option, and it is not a decision today.

- **The delayed amax history.** It saves one read of the operand. It changes
  the loss against the reference. It needs a new parity entry in
  `docs/parity.md` and a new tolerance row before it lands.

## See also

- [precision.md](precision.md) — the precision policy.
- [simulator-precision.md](simulator-precision.md) — the fp8 simulator engines.
- [performance-model.md](performance-model.md) — the bound and the rule.
- [attention-tensor-core.md](attention-tensor-core.md) — the attention coupling.
- [numerics-integration.md](numerics-integration.md) — the tolerance table.
- [node-h100.md](node-h100.md) — the H100 run book and the memory plan.
