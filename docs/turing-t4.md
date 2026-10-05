# Turing (T4, sm_75)

Design for the Turing target. The T4 card is the second GPU family that
`nanochat.cpp` supports. Read [precision.md](precision.md) and
[backends.md](backends.md) for the selection rules.

Status: partially implemented. The build configs exist. The correctness gates
on Turing are not recorded yet.

## 1. Goal

T4 is a first-class target. A Turing port uses the existing backend seam. No
new backend directory is necessary.

## 2. Capability difference

| Feature | Pascal sm_61 | Turing sm_75 |
|---|---|---|
| fp32 arithmetic | yes | yes |
| fp16 storage | software | native `__half` |
| fp16 tensor cores | no | yes |
| `__half` atomic add | fallback loop | native, sm_70 and newer |
| cuBLAS fp16 path | HGEMM, 16F compute | tensor-op, 32F compute |
| `mma` instructions | no | yes |

`GetCaps()` reports the differences. `caps.has_tensor_cores` is true for
`major >= 7`. `caps.supports_fp16` is true for sm_53 and newer.

## 3. Build configuration

| Config | Arch | Precision | Use |
|---|---|---|---|
| `sm_75` | sm_75 | fp32 | fp32 correctness gate |
| `t4` | sm_75 | fp16 | default Kaggle build |
| `t4-fp32` | sm_75 | fp32 | explicit fp32 GPU build |

The `t4` config includes `--config=cuda`. The half-precision build uses the
tensor-op GEMM path on this card.

The T4 supports both precisions. The fp32 build is the correctness gate. The
fp16 build is the performance build.

## 4. GEMM behavior

`backends/cuda/gemm.cu` selects the compute type from the device capability.
On sm_75 the fp16 build uses `CUBLAS_GEMM_DEFAULT_TENSOR_OP` with
`CUBLAS_COMPUTE_32F`. The accumulation stays in fp32. The fp32 build uses the
classic `cublasSgemm` path for a single GEMM.

The host graph and the optimizer state stay fp32 wherever the seam requires
it. See [precision.md](precision.md).

## 5. Attention

The current attention kernel uses fp32 statistics and an online softmax. It
does not use tensor cores. A `mma` variant is later work under
[DESIGN.md §3](../DESIGN.md). A candidate belongs in `dev/kernels` first, and
it must beat the current kernel at the training shape.

## 6. Correctness gates

| Tier | Target | Precision |
|---|---|---|
| T1 | `//backends/cuda/kernels:precision_gpu_test` | fp32, fp16 |
| T1 | `//backends/cuda:cuda_runtime_gpu_test` | fp32 |
| T2 | `//tests:oracle_cuda_test` | fp32 |
| T2 | `//tests:train_parity_cuda_test` | fp32 |
| T3 | `//backends/cuda/kernels:attention_benchmark` | fp16 |

The first fp16 run may exceed the recorded Pascal tolerance. Re-measure the
worst error and record the new value in `precision_test.cc`. Do not widen the
tolerance without a measurement.

Extend `//backends/cuda:cuda_runtime_gpu_test` to assert three things on
sm_75: the device name, `has_tensor_cores`, and `supports_fp16`.

## 7. Two cards

A Kaggle Notebook gives two T4 cards. See
[host-portability.md](host-portability.md) for device selection and
[kaggle.md §8](kaggle.md) for the two options. The short form:

- Option A: two independent jobs, one per card.
- Option B: data-parallel training. Later work.

## 8. Definition of done

1. `tools/nanochat build --config=t4` succeeds.
2. Every T1 GPU test passes on the T4 in fp32 and fp16.
3. The oracle and train-parity gates pass on sm_75.
4. The recorded fp16 tolerance matches a fresh measurement.
5. `cuda_runtime_gpu_test` asserts the Turing capabilities.
6. [performance.md](performance.md) holds one T4 row at the training shape.
