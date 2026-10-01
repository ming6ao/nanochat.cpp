# Precision

`-DNANOCHAT_PRECISION=FP32|FP16` selects `using ComputeType = ...;` at build
time. One precision per build; no runtime dtype dispatch.

- Pascal (sm_61): fp32.
- Turing (sm_75): fp16 permitted, with loss scaling.
- The backend reports supported dtypes via `GetCaps()`; an invalid combination
  fails fast at startup.

Precision is chosen at build time alongside the backend; see
[build.md](build.md) and [backends.md](backends.md).
