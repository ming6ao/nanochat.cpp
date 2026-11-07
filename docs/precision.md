# Precision policy

One build has one compute type. The tree uses fp32 and bf16 only, holds no fp16
code, and keeps every persistent buffer in fp32. The fp8 tier is opt-in on
Hopper and Ada, and it computes in bf16.

Select the tier with a build configuration:

```bash
tools/nanochat build                     # fp32 on Pascal, Turing, and the CPU
tools/nanochat build --config=bf16       # bf16 compute, fp32 state
tools/nanochat build --config=hopper     # sm_90, bf16 compute
tools/nanochat test --config=bf16
tools/nanochat roofline --device h100 --precision bf16 --layers 20
```

## 1. The rule

Use one rule for every device:

1. Use fp32 for the compute on Pascal, Turing, and the central processing unit.
2. Use bf16 for the compute on Ampere and newer.
3. Keep the persistent state in fp32 on every device.
4. Do not use loss scaling. Neither fp32 nor bf16 needs it.

## 2. Why fp16 is gone

fp16 is the only format with a small exponent range.

- bf16 has the fp32 exponent range. A tiny gradient does not underflow, so bf16
  needs no loss scaling.
- fp16 has five exponent bits. A tiny gradient underflows, so fp16 needs loss
  scaling and fp32 gradient storage.
- On Pascal, fp16 runs at 0.177 TFLOP/s against 11.34 TFLOP/s in fp32, which is
  64 times slower.
- On Turing, fp16 is the only accelerated format and there is no bf16. The fp16
  peak is 65 TFLOP/s against 8.1 TFLOP/s in fp32.

One card gains from fp16: the Turing T4. The gain costs a second compute type, a
second storage tag, a loss-scaling project, an emulator header, and a second
test leg. The bf16 path replaces that build and needs no loss scaling. The
PyTorch reference makes the same choice and uses fp32 on pre-Ampere hardware
(`nanochat/common.py:17-32`).

## 3. Device capabilities

The table is from `src/device_profile.cc`. The rates are dense.

| Device | Major.Minor | Tensor cores | fp32 | bf16 | fp8 |
|---|---|---|---|---|---|
| t4 | 7.5 | yes | 8.1 TFLOP/s | none | none |
| h100 | 9.0 | yes | 66.9 TFLOP/s | 989 TFLOP/s | 1979 TFLOP/s |

The device table carries no fp16 field, and the estimator has no fp16 mode.

## 4. Policy per target

| Target | Arch | Compute | Persistent state | Config |
|---|---|---|---|---|
| Workstation | sm_61 | fp32 | fp32 | default |
| Central processing unit and continuous integration | none | fp32 | fp32 | default |
| Kaggle T4 | sm_75 | fp32 | fp32 | `--config=cuda --@rules_cuda//cuda:archs=sm_75` |
| H100 | sm_90 | bf16 | fp32 | `--config=hopper` |

The `h100` and `h200` targets are sm_90 with bf16 and the simulator.
`--config=bf16` selects the bf16 compute type on any architecture.

## 5. The storage rule

This rule applies to fp32, bf16, and fp8.

Keep these buffers in fp32:

- the master weights;
- the parameter gradients;
- the optimizer moments;
- the norm statistics and the attention statistics;
- the loss, the softmax, and the row reductions.

Use the compute dtype for the activations, the GEMM operands, and the key/value
cache.

bf16 has seven mantissa bits, and a parameter update needs more precision than
that. bf16 has the fp32 exponent range, so no scale is necessary.

The checkpoint container is a storage format. It stores fp32 or bf16 records,
and the record dtype is separate from the build compute type. Store the
canonical checkpoint in fp32. A bf16 container is a derived compact artifact.
The `DType` codes 0, 1, and 2 stay; code 3 belonged to `kFp16` and stays
reserved.

This rule is stricter than the reference. The reference keeps matrix parameters
in fp32 but casts the embeddings to the compute dtype (`gpt.py:262-266`), so its
embedding gradients are bf16. This tree keeps every parameter gradient in fp32.

A bf16 build needs a mixed GEMM. The weight-gradient GEMM reads bf16 operands
and writes an fp32 result, through `cublasGemmEx` with `Atype = Btype =
CUDA_R_16BF`, `Ctype = CUDA_R_32F`, and `CUBLAS_COMPUTE_32F`.

## 6. The bf16 compute path

The tree has the `Bf16` type, the `DType::kBf16` tag, device-profile bf16
support, and bf16 peak rates. The compute path follows this contract:

- Keep master weights, parameter gradients, optimizer state, loss values, and
  reduction statistics in fp32.
- Store parameter compute copies, activations, activation gradients, GEMM
  operands, and the key/value cache in bf16.
- Accumulate GEMM products and reductions in fp32. Store activation outputs as
  bf16, and store weight-gradient outputs as fp32.
- Accumulate embedding gradients directly into fp32 parameter-gradient buffers.
- Round each stored bf16 value to nearest-even. Do not use loss scaling.

The mechanism:

- `ComputeType` is chosen at build time. `include/nanochat/tensor.h` maps
  `NANOCHAT_PRECISION` to `BuildPrecision`: `float` for fp32, and `Bf16` for
  bf16 and fp8.
- `GradientType` is fp32 in every build, so a parameter gradient is never bf16.
- `Caps::supports_bf16` comes from the device profile, and the CUDA backend sets
  it from `major >= 8`. `Model::Create` fails fast when the build compute type
  is not supported by the device.
- The CPU reference stores bf16 where the compute type is bf16 and accumulates
  in fp32 or double, so the CPU records the same rounding the device does.
- `Gemm` uses `cublasGemmEx` with `CUDA_R_16BF` operands and
  `CUBLAS_COMPUTE_32F`. `GemmWgrad` writes an fp32 result.
- `AtomicAddDev` gains a bf16 compare-and-swap form, because a half-word write
  cannot use the native atomic add.
- `AdamWUpdate` and `MuonUpdate` read fp32 gradients, keep fp32 moments, and
  write the bf16 compute copy.
- `CollectModelRecords` reads the master, a load writes the master and rebuilds
  the compute copy, and a load accepts an fp32 or a bf16 record in any build.
- `Model.dtype` and `TensorView.dtype` report the compute type, and the
  parameter view reads and writes Python floats in both builds.

## 7. Testing a bf16 build

A bf16 build cannot reproduce a committed fp32 golden, so a test that compares
against one prints a skip. Those tests keep full strength in the fp32 build. The
skipped legs are the numeric trace, the oracle, the training parity, the scoring
parity, the model oracle, the model gradient, and the two Python fixtures.

A bf16 build still runs every one of those code paths through the tests that do
not compare against a golden. The in-build two-path checks run in both builds
with a precision-aware bound.

The numeric trace has its own bf16 leg. The fixture carries an fp32, a bf16, and
an fp8 section, and a bf16 build compares its own section byte for byte through
`tools/dump_fixtures.py numerics`.

The integration leg is the chain:
`python/nanochat_cpp/pipeline.py` runs base, SFT, RL, and eval on a bf16 build,
and the Modal H100 leg of `tools/modal/app.py` runs that chain on real Hopper
silicon at `sm_90`. The T4 leg is the fp32 control: the same job at fp32 on
`sm_75`, so the two reports differ in precision and in nothing else the job
sets.

## 8. Limits

- **The T4 is the Kaggle target, and Turing has no bf16.** It runs fp32 at 8.1
  TFLOP/s. Do not restore fp16 and do not start a loss-scaling project; reduce
  the model size instead.
- **The attention kernel does not use tensor cores today.** It runs at the fp32
  rate in either build, so the realized T4 loss is below 8 times.
- **The fp32 build stays the correctness gate.** Every fp32 test keeps its
  recorded tolerance, and a bf16 golden is a skip and not a pass.
- **The precision is a build decision.** There is no runtime precision switch
  ([../README.md](../README.md) § What this is not).

## Part 1. The fp8 tier

The fp8 tier is `--config=fp8`. It computes in bf16 with fp8 GEMM operands, and
it requires compute capability 8.9 or 9.0. Hopper reports 9.0 and Ada reports
8.9. Ampere reports 8.0, and its tensor cores have no fp8 mode.

### 1. Goal and scope

Apply fp8 to the matrix multiplies, keep every other layer in bf16 or fp32, and
keep every persistent value in fp32. The parity target is the reference
`--fp8` path in `nanochat/fp8.py`.

### 2. What the tree holds

`backends/cuda/kernels/fp8.cu` holds the quantize side and the shared rounding
function. `GemmScaledFp8` runs through `cublasLt` in `backends/cuda/gemm.cu`.
`src/fp8_policy.h` decides which operation is eligible. The simulator rounds fp8
values without a GPU ([simulator.md](simulator.md) Part 7).

### 3. The precision tier

#### 3.1 The storage tags

`DType` gains `kFp8E4M3` and `kFp8E5M2`, one byte each. The existing tags keep
their values, and `kComputeDType` does not change, so the on-disk contract does
not change. The checkpoint container stays fp32.

| Format | Exponent bits | Mantissa bits | Largest value | Use |
|---|---|---|---|---|
| E4M3 | 4 | 3 | 448 | Forward operands |
| E5M2 | 5 | 2 | 57344 | Backward operands |

Use the finite encoding. The reference calls it `torch.float8_e4m3fn`, and
cuBLAS names it `CUDA_R_8F_E4M3`. This form has no infinity. `E4M3FNUZ` is a
different encoding with a different bias and no negative zero, and the two must
never mix.

#### 3.2 The rule that keeps the rest in fp32

The rule of section 5 applies unchanged: master weights, parameter gradients,
optimizer moments, and every statistic stay fp32. Use fp8 only for the GEMM
operands and the activations that feed them.

An fp8 value carries a scale, and the scale is a separate fp32 device scalar.
Never fold the scale into the loss. The logits tensor takes the compute type,
and the softcap, the softmax, the cross-entropy, and every reduction stay fp32.

### 4. Scaling

#### 4.1 The scale rule

fp8 has few mantissa bits, so a raw cast loses most of the range. Apply one
scale to each operand, derived from the largest absolute value of the tensor.

```text
fp8_max = 448        for E4M3
fp8_max = 57344      for E5M2

scale   = fp8_max / max(amax, 1e-12)
scaled  = clamp(value * scale, -fp8_max, fp8_max)
fp8     = cast(scaled)
value  ~= fp8 / scale
```

Three points matter.

1. **The clamp is required.** A cast without the clamp does not saturate. In the
   `E4M3FN` encoding a large value overflows to NaN, so the reference clamps to
   `fp8_max`.
2. **The `1e-12` floor is required.** An all-zero tensor has an amax of zero,
   and the scale would be infinite.
3. **Two names exist for the same number.** The kernel writes the multiplier.
   The `cublasLt` scale pointer is the dequantization value, so it takes
   `1 / scale`.

Use one scale per tensor. This is the tensorwise mode, and it is the reference
default and the only recipe the reference implements. A rowwise scale needs a
CUTLASS kernel and is later work.

The GEMM receives the scale as a pointer, so the tensor core applies it during
the multiply and the host pays for no second pass. Round to nearest with ties to
even, once, in the shared function of `include/nanochat/fp8.h`, so the central
processing unit, the emulator, and the device agree.

#### 4.2 The amax source: dynamic

The scale comes from the amax of the operand in the same step. The tree keeps no
amax history and no decay:

1. Measure the amax of the tensor on the device.
2. Derive the scale on the device, with the floor above.
3. Cast the tensor with that scale, in the same fused kernel.
4. Keep the fp8 tensor and its scale for the step, and reuse them for every GEMM
   that reads the tensor.

Dynamic scaling is what the reference does, so parity is direct. The estimator
prices the cast at five bytes of traffic per element: measure, then read and
write. There is no persistent state, so a resume reproduces exactly.

The host never reads the amax, so no device synchronization occurs. A delayed
amax history with a decay is the NVIDIA Transformer Engine scheme. It saves one
read of the operand and needs its own parity record.

#### 4.3 Where the scales live

Store the fp8 operand and its scale in the workspace beside the tensor they
describe. The slot list stays static, with one byte slot per quantized tensor
and one fp32 slot per scale. The checkpoint stores no scale state, because the
scale is derived within the step.

### 5. The GEMM seam

#### 5.1 The new entry point

The existing entry point takes `ComputeType` pointers. A second entry point
carries scaled fp8, so the Pascal, Turing, and CPU backends keep one code path:

```cpp
struct ScaledGemmParams {
  int m, n, k, batch_count;
  float alpha, beta;
  bool transpose_a, transpose_b;
  bool fast_accum;       // reduced-precision reduction; forward only
  const float* scale_a;  // the dequantization value, 1 / scale
  const float* scale_b;
};

void GemmScaledFp8(GemmMode mode, const ScaledGemmParams& params,
                   const std::uint8_t* a, const std::uint8_t* b, float* c);
```

Four design points matter:

1. The accumulator and the output stay fp32. `C` is `float*`.
2. The operand type is `std::uint8_t*`, not `ComputeType*`, so the header holds
   no vendor type ([build.md](build.md#dependency-invariants) rule 3).
3. There is no C scale. The caller passes `beta = 0` for a pure overwrite.
4. `transpose_a` and `transpose_b` describe the logical layout. The backend maps
   them to the `cublasLt` stride triplets.

`include/nanochat/kernels.h` is frozen and architect-owned. This change is
additive, so it follows the evolution rules of
[model-and-device-bringup.md](model-and-device-bringup.md) §18.

#### 5.2 The cuBLASLt call

Only `cublasLtMatmul` accepts the scale pointers:

1. Create a matrix layout for A, for B, and for C.
2. Create a matrix multiply descriptor.
3. Set the scale pointers on the descriptor.
4. Ask for a heuristic with `cublasLtMatmulAlgoGetHeuristic`.
5. Call `cublasLtMatmul`.

| Field | Value |
|---|---|
| A type, B type | `CUDA_R_8F_E4M3` in the forward, `CUDA_R_8F_E5M2` in the backward |
| C type | `CUDA_R_32F` |
| Compute type | `CUBLAS_COMPUTE_32F` |
| Scale type | `CUDA_R_32F` |
| Scale attribute | `CUBLASLT_MATMUL_DESC_A_SCALE_POINTER`, `..._B_SCALE_POINTER` |
| Fast accumulation | `CUBLASLT_MATMUL_DESC_FAST_ACCUM`: on in the forward, off in the backward |

Fast accumulation changes the numerics. The reference sets `use_fast_accum=True`
in the forward and `False` in the two backward GEMMs, and the fp8 parity bounds
record the setting. `cublasLt` ships with the CUDA toolkit, so it is not a new
third-party dependency. The backend gains one link edge, `@cuda//:cublasLt`.

`cublasLtMatmul` needs a workspace for some algorithms. Use the existing
workspace allocator, and size the request from the heuristic.

#### 5.3 The capability gate

Add the gate in three places: `Caps::supports_fp8`, `DeviceProfile::supports_fp8`,
and the model-creation check in `Model::Create`, beside the compute-dtype check.

A build with fp8 on a device without support must fail when the model is
created. Do not fall back at runtime, because
[../README.md](../README.md) § What this is not forbids a runtime precision
branch. The host toolkit must be CUDA 11.8 or newer, and the pinned toolkit is
CUDA 12.0.

### 6. The layer set

#### 6.1 The rule from the reference

The reference applies `--fp8` through a module filter. A matrix multiply
converts when all three conditions hold:

1. it is a matrix multiply;
2. both dimensions are divisible by 16;
3. the smaller dimension is at least 128.

The first condition excludes the gather, the row kernels, and the optimizer. The
second is a hardware requirement of the fp8 tensor core. The third excludes the
small gate matrices, which lose more than they gain.

The token embedding and the value embeddings are gathers, so they never convert.
The reference casts them to the compute type for the forward and keeps the fp32
master. `src/fp8_policy.h` implements the rule as the one assignment.

#### 6.2 The assignment at depth 24

| Operation | Precision | Reason |
|---|---|---|
| `c_q`, `c_k`, `c_v`, `c_proj` | fp8 | A matrix, and both dimensions are 1536 |
| `mlp.c_fc`, `mlp.c_proj` | fp8 | A matrix, 1536 by 6144 |
| the language-model head | fp8 | A matrix, 1536 by 32768 |
| the logits tensor | the compute type (bf16) | the GEMM writes the compute type |
| the softcap and the softmax | the compute type; the statistics fp32 | a row kernel, and the reduction stays fp32 |
| the loss | fp32 | the loss must stay fp32 |
| `ve_gate` | bf16 | the output width is 12, below 128 |
| `smear_gate` | bf16 | the output width is 24, below 128 |
| the embedding | bf16 | a gather, not a matrix multiply |
| `rms_norm`, rope, `qk_prep` | bf16 | a row kernel |
| the attention | bf16 | the reference attention engine runs the compute type |
| the value residual and the backout | bf16 | a row blend |
| AdamW and Muon | fp32 | the optimizer state stays fp32 |
| the gradient all-reduce | fp32 | the reduction operand stays fp32 |

A weight has three forms: the fp32 master, the fp8 operand, and the fp8 scale.
Keep the fp8 operand and the scale as workspace slots, and rebuild them once per
step. Do not build a separate bf16 copy of an eligible weight.

#### 6.3 The two costs that come with fp8

1. **The cast.** One kernel per operand measures the amax, computes the scale,
   and writes the fp8 value. A fused kernel moves about five bytes per element.
   An unfused sequence of calls moves six times that, and it makes fp8 slower
   than bf16. Fuse the amax, the scale, and the cast.
2. **The layout.** The fp8 GEMM engine requires operand A row-major and operand
   B column-major. Store each weight in the layout the GEMM wants, once at load
   time, and never copy it every step.

#### 6.4 The evaluation precision

Evaluation and generation run in the compute type, not in fp8. The reference
disables fp8 around every evaluation call, because an fp8 forward changes the
sampled logits. Match it: do not build fp8 operands on the evaluation path. An
`eval` path that runs fp8 needs a new decision and a new parity record.

### 7. The backward pass

The backward pass has two matrix multiplies per weight. Both need fp8 operands
for the rate, and both need an fp32 accumulator.

| Call | Operand A | Operand B | Accumulator | Output |
|---|---|---|---|---|
| Forward | the activation, E4M3 | the weight, E4M3 | fp32 | the compute type |
| Input gradient (dgrad) | the output gradient, E5M2 | the weight, E4M3 | fp32 | the compute type |
| Weight gradient (wgrad) | the output gradient, E5M2 | the activation, E4M3 | fp32 | fp32 |

E5M2 has a wider range, and a gradient has a wider spread than an activation,
so the gradient operand takes E5M2. Both gradient GEMMs share the
output-gradient operand and its scale. The input gradient is an activation for
the next layer, so it takes the compute type; the weight gradient is a
persistent parameter gradient, so it stays fp32.

Four hazards follow:

1. **The layout.** `cublasLtMatmul` accepts a limited set of fp8 layouts, and the
   three GEMMs disagree.

   | GEMM | Operand A | Operand B |
   |---|---|---|
   | Forward | the activation, row-major | the weight transpose, column-major |
   | Input gradient | the output gradient, row-major | the weight, column-major |
   | Weight gradient | the output gradient transpose, row-major | the activation, column-major |

   A weight stored `[N, K]` row-major gives its `[K, N]` column-major transpose
   as operand B with no copy, so one buffer serves the forward and the input
   gradient. The fused cast writes the activation in both layouts, and the
   backward reads the output gradient with a column-major stride, which removes
   both rearrangements the reference needs.
2. **The scale direction.** The wgrad operand is an activation that the forward
   pass already quantized. Reuse its scale, and do not compute a second scale
   for the same tensor in one step.
3. **The cast count.** An operand that the forward pass already quantized must
   not be quantized again. Reuse the stored fp8 tensor and its scale.
4. **The shared output gradient.** One output gradient feeds both gradient
   GEMMs of a layer. Quantize it once per layer, and pass the same tensor and
   scale to both.

Set fast accumulation off in both backward GEMMs. Keep the parameter gradient in
fp32 from the moment the fp32 accumulator writes it, and never quantize it.

#### 7.1 A staged choice

A safe first implementation uses fp8 in the forward only and bf16 in the
backward, because the backward carries the smaller arithmetic share and the bf16
path already exists. Add fp8 to the backward after the forward gates pass, and
measure both. [performance-model.md](performance-model.md) holds the bound for
each option.

### 8. The optimizer and the checkpoint

The optimizer needs no change. The master weight, the moments, and the update
stay fp32, and the fp8 operand is a derived buffer the optimizer never reads.
The reference shares the parameter the same way through
`Float8Linear.from_float`.

The checkpoint needs no addition, because the tree keeps no scale state: a
resume reproduces every scale exactly. An fp8 build writes an fp32 checkpoint,
and `src/train.cc` already converts a non-fp32 compute type to fp32 on save.

### 9. Invariants

1. The loss, the optimizer state, the norms, and the statistics are fp32.
2. No runtime precision switch exists. The precision is a build decision.
3. The fp8 GEMM writes a `float` accumulator.
4. Every fp8 operand has a scale, and no scale is null in a call.
5. The scale is derived from the current operand in the same step, before the
   first GEMM that reads it. The tree keeps no decayed history.
6. The kernel seam holds no vendor type.
7. The simulator validates every fp8 call shape without a GPU.
8. The scale is a device scalar, and the host does not read the amax in a step.
9. The fp8 cast saturates: a value above `fp8_max` clamps to `fp8_max`.
10. Evaluation and generation run in the compute type.

### 10. Validation

The first six checks need no GPU.

1. The shared fp8 rounding rule, which covers the saturation, the floor, and
   round-to-nearest-even. The CPU emulation and the simulator share one
   function, so one test proves both. Gate `//tests:fp8_rounding_test`.
2. The emulated scaled GEMM against the CPU fp32 reference, at the shapes in the
   model. Gate `//tests:fp8_gemm_test`, plus the fp8 leg of
   `//tests:kernel_emu_test`.
3. The finite-difference check for the fp8 attention and MLP backward.
4. One fp8 forward and one training step against the reference trajectory,
   through the fp8 section of `tests/data/numerics_golden_10l.bin`.
5. The graph check, `//tests:fp8_graph_test`, which runs the layer set, the
   operand formats, and the fp32 state on the simulator.
6. `tools/nanochat simulate --device h100 --suite all`, which validates every
   call shape and the `cublasLt` symbols without a card.
7. One H100 run of `--config=fp8` on the depth-24 shape, with the step time and
   the loss in [baselines.json](baselines.json).

### 11. Non-goals

- A rowwise scale, which needs a CUTLASS kernel.
- A delayed amax history with a decay.
- fp8 evaluation and generation.
- fp8 for the embeddings, the row kernels, the optimizer, or the gradient
  all-reduce.
- fp8 on Ampere, whose tensor cores have no fp8 mode.

## See also

- [build.md](build.md) — backends and the build configuration.
- [kernels.md](kernels.md) — the backend seam and the GEMM entry points.
- [performance-model.md](performance-model.md) — the precision assignment in the
  analytic model.
- [host-portability.md](host-portability.md) — the host and the Kaggle
  procedure.
