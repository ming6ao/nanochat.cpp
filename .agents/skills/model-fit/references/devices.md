# Devices

The device table in `scripts/model_fit.py` mirrors `src/device_profile.cc` in
nanochat.cpp. The rates are dense (non-sparse) peaks.

| Device | Memory | fp32 | fp16 | bf16 | Tensor cores |
|---|---:|---:|---:|---:|---|
| h200 | 141 GiB | 67.0 TFLOP/s | 989 TFLOP/s | 989 TFLOP/s | yes |
| h100 | 80 GiB | 66.9 TFLOP/s | 989 TFLOP/s | 989 TFLOP/s | yes |
| a100 | 80 GiB | 19.5 TFLOP/s | 312 TFLOP/s | 312 TFLOP/s | yes |
| t4 | 16 GiB | 8.1 TFLOP/s | 65 TFLOP/s | none | yes |
| gtx1080ti | 11 GiB | 11.34 TFLOP/s | 0.177 TFLOP/s | none | no |

The source is `include/nanochat/device_profile.h` and its table in
`src/device_profile.cc`. The same rates appear in `docs/precision.md`
section 3.

## The model budget

The engine keeps a reserve for the CUDA context and the library workspaces.
The default reserve is 1.5 GiB. The model sees the difference:

```text
budget_bytes = memory_bytes - reserve_bytes
```

For a T4, the budget is 14.5 GiB.

## The efficiency factors

The engine maps a peak rate to an estimated achieved rate:

```text
gemm_rate      = peak_rate * gemm_efficiency        # default 0.8
attention_rate = fp32_rate * attention_efficiency   # default 0.25
```

The attention kernel uses no tensor cores in nanochat.cpp. That is why its rate
follows the fp32 rate at every precision. See `docs/precision.md` section 8.

A caller can override the factors with the `Hardware` fields
`gemm_efficiency`, `attention_efficiency`, and `attention_flops`.

## The precision layout

The memory model needs three sizes: the value, the gradient, and the fp32
master weight.

| Precision | Value | Gradient | Master |
|---|---:|---:|---:|
| fp32 | 4 | 4 | 0 |
| fp16 | 2 | 2 | 4 |
| bf16 | 2 | 4 | 4 |

The fp16 row follows the nanochat.cpp fp16 build: the parameter and the
gradient are half, and the fp32 master weight is separate. The fp32 master
aliases the value in the fp32 build, so it adds nothing.

The bf16 row follows the target policy in `docs/precision.md` section 5: the
gradient and the master stay fp32. That path is not implemented.

The optimizer state stays fp32 at every precision.

## Add a device

Add one row to `DEVICES` in `scripts/model_fit.py`:

```python
DEVICES["rtx4090"] = Hardware(
    "rtx4090", 24 * GIB, 82.6e12, 330e12, 330e12, tensor_cores=True)
```

A device without a public peak rate can use the `custom` hardware path on the
command line:

```bash
python3 model_fit.py --hardware custom --memory-gib 24 \
    --fp32-tflops 82.6 --fp16-tflops 330 --devices 1
```
