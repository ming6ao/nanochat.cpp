# Precision policy

Status: accepted for Stage 1. Stage 2 comes later. This document defines the
precision policy for every host. It replaces the fp16 Turing path with a
two-tier policy. The fp32 arithmetic and the oracle fixtures do not change.

## 1. Decision

Use one rule for every device:

1. Use fp32 for the compute on Pascal, Turing, and the CPU.
2. Use bf16 for the compute on Ampere and newer.
3. Keep the persistent state in fp32 on every device.
4. Do not use loss scaling. Neither fp32 nor bf16 needs it.

Remove fp16 from the production path. Keep the fp16 CPU build for the simulator
only. Section 6 gives the two stages.

This reverses the earlier decision to ship an fp16 production build. Commit
`02c1dd6` kept the fp32 master weight beside an fp16 compute copy. Option A
removes the fp16 compute copy from the production path. The gradient-underflow
evidence justifies the reversal.

## 2. Reason

fp16 is the odd format. It is the only format with a small exponent range.

- bf16 has the fp32 exponent range. A tiny gradient does not underflow. bf16
  needs no loss scaling.
- fp16 has five exponent bits. A tiny gradient underflows. fp16 therefore needs
  loss scaling and fp32 gradient storage.
- On Pascal, fp16 runs at 0.177 TFLOP/s. The fp32 rate is 11.34 TFLOP/s. The
  fp16 rate is 64 times slower.
- On Turing, fp16 is the only accelerated format, and Turing has no bf16.

So fp16 helps on one card only. It costs a large amount of code. The PyTorch
reference makes the same choice. It uses fp32 on pre-Ampere hardware
(`nanochat/common.py:17-32`).

## 3. Device capabilities

The table is from `src/device_profile.cc`. The rates are dense (non-sparse).

| Device | Major.Minor | Tensor cores | fp32 | fp16 | bf16 |
|---|---|---|---|---|---|
| gtx1080ti | 6.1 | no | 11.34 TFLOP/s | 0.177 TFLOP/s | none |
| t4 | 7.5 | yes | 8.1 TFLOP/s | 65 TFLOP/s | none |
| a100 | 8.0 | yes | 19.5 TFLOP/s | 312 TFLOP/s | 312 TFLOP/s |
| h100 | 9.0 | yes | 66.9 TFLOP/s | 989 TFLOP/s | 989 TFLOP/s |

The table shows the decision. bf16 starts at Ampere. fp16 is useful on Turing
only.

## 4. Policy per target

| Target | Arch | Compute | Persistent state | Config |
|---|---|---|---|---|
| Workstation | sm_61 | fp32 | fp32 | default |
| CPU and CI | none | fp32 | fp32 | `--config=cpu` |
| Kaggle T4 | sm_75 | fp32 | fp32 | `--config=t4` |
| Ampere and newer | sm_80+ | bf16 | fp32 | `--config=bf16` |

The Ampere row is Stage 2. It comes later (section 6).

## 5. Storage rule

This rule applies to bf16 and to fp32. It also applies to a later format.

Keep these buffers in fp32:

- the master weights;
- the parameter gradients;
- the optimizer moments;
- the norm statistics and the attention statistics.

Use the compute dtype for the activations, the GEMM operands, and the key/value
cache.

bf16 has seven mantissa bits. A parameter update needs more precision than
that. bf16 has the fp32 exponent range, so no scale is necessary.

This rule is stricter than the reference. The reference keeps the matrix
parameters in fp32, but it casts the embeddings to the compute dtype
(`gpt.py:262-266`). Its embedding gradients are therefore bf16. This rule keeps
every parameter gradient in fp32. That is a deliberate divergence.

A bf16 build needs a mixed GEMM. The weight-gradient GEMM reads bf16 operands
and writes an fp32 result. Use `cublasGemmEx` with `Atype = Btype =
CUDA_R_16BF`, `Ctype = CUDA_R_32F`, and `CUBLAS_COMPUTE_32F`.

## 6. Staged rollout

### Stage 1: remove the fp16 production path (accepted)

1. Change `--config=t4` to fp32 in `.bazelrc`.
2. Keep `--config=fp16` for the simulator only. Mark it as a test tool.
3. Remove `--config=t4-fp32`. It becomes the same as `--config=t4`.
4. Update the `t4-fp32` rows in `docs/build.md` and
   `docs/host-portability.md`.
5. Change the Python default in `python/nanochat_cpp/_build.py:261`. The
   `precision == "fp16"` branch maps to `--config=t4` today.
6. Change the notebook default in `notebooks/kaggle_setup.py:125` and `:154`.
7. Update the example in `docs/python.md:116`.

Stage 1 needs no kernel change. In the fp32 build, `ComputeType` is `float`. The
parameter gradients are already fp32. The master weight is already an alias of
the parameter.

### Stage 2: add bf16 for Ampere and newer (pending)

1. Add `DType::kBF16` to `include/nanochat/tensor.h`.
2. Add `supports_bf16` to `Caps` and to `Caps::Supports`.
3. Wire `supports_bf16` from `src/device_profile.cc` into `Caps`.
4. Add a storage type for bf16.
5. Add a bf16 GEMM. Use `cublasGemmEx` with `CUDA_R_16BF` and
   `CUBLAS_COMPUTE_32F`.
6. Add `--config=bf16` and the Ampere targets.
7. Keep the persistent state in fp32 (section 5).

Stage 2 needs the fp32 gradient seam change. It does not need loss scaling.

## 7. What this avoids

Option A removes the loss-scaling project:

- the dynamic scale, the growth, and the backoff;
- the inf and nan detection and the step skip;
- the half to fp32 gradient unscale.

Option A removes the fp16 master-weight mirror. The fp32 build needs no mirror,
because the master weight aliases the parameter.

The checkpoint format is precision-independent. The model saves an fp32 master
with no dtype tag (`src/model.cc:1040`). No migration is necessary.

## 8. Risks

- The T4 is the Kaggle target (`docs/build.md:205`). Option A removes the
  tensor-core path from that card permanently, because Turing has no bf16.
- The T4 loses the tensor-core rate. The peak fp16 rate is 65 TFLOP/s. The fp32
  rate is 8.1 TFLOP/s. Measure the real training step before you accept the
  change.
- The attention kernel does not use tensor cores today (`docs/build.md`).
  Attention runs at the fp32 rate in either case. The realized loss is below 8x.
- The bf16 path is new code. Keep the fp32 build as the correctness gate.
- The CUDA `precision_gpu_test` target keeps a test-only fp16 leg on sm_61.
  The T4 gate runs fp32 (`docs/build.md`). Keep the fp16 CPU emulation for the
  simulator.

## 9. Open questions

1. Does the T4 training speed matter for the Kaggle target? Stage 1 does not
   wait for the answer. If the speed regresses, keep fp16 for the T4 and run
   the loss-scaling project instead.
2. Do we keep the fp16 CPU emulation, or move the simulator to bf16?
3. Which Ampere target do we add first?
4. The `.bazelrc:47` reference to `docs/backends.md`, the `.bazelrc:52`
   reference to `docs/python-api.md`, and the `tools/sandbox/profiles.conf:26`
   reference to `docs/kaggle.md` stay dangling. Record them for a later cleanup.

## See also

- [build.md](build.md) — backends and the build configuration.
- [host-portability.md](host-portability.md) — the host and the Kaggle procedure.
- [kernels.md](kernels.md) — the backend seam.
