# Backends and selection

Selection is a build/link choice:

```
--backend=cpu|cuda        link the backend library
--precision=fp32|fp16     select ComputeType
--arch=sm_61|sm_75        CUDA --generate-code
```

| Backend | Implements | Notes |
|---|---|---|
| `backends/cpu` | all of `kernels.h` with naive loops | reference, CI, oracle baseline; `-lm` (+ OpenMP) |
| `backends/cuda` | all of `kernels.h`; cuBLAS for GEMM | Pascal fp32, Turing fp16; cuDNN/NCCL optional |

The shared workflow (`ops.cc`, `model.cc`, `generate.cc`, `optim.cc`,
`train.cc`) compiles once and links against either backend. This is the one
improvement over llm.c, which duplicates its top-level files per backend.

## Adding a backend

Adding hardware = add `backends/<x>/` implementing `kernels.h` plus a config
value. `src/` and `include/` normally need no changes; a new kernel family
(below the replacement boundary) does touch the frozen headers, under
[DESIGN.md §7](../DESIGN.md).

The interface to implement is [kernels.md](kernels.md). The dependency rules a
backend must respect are in [DESIGN.md §2](../DESIGN.md): `backends/**` includes
only `nanochat/kernels.h` and `nanochat/tensor.h`, and a kernel knows nothing
about GPT.
