# Contributing

This is a from-scratch, small-surface codebase. The goal is that the mainline
stays readable enough to audit in one sitting, so changes should keep it that
way. For how multiple agents share the single host, see [AGENTS.md](AGENTS.md);
this file covers code conventions and the definition of done for a human
contributor.

## Style

- Project: `nanochat.cpp`. Namespace: `nanochat`.
- C++20. Host code in `.cc`/`.h`; CUDA in `.cu`; device helpers in `.cuh`.
- **Google C++ Style Guide**, enforced with `clang-format --style=Google`
  (80 columns, 2-space indent). CUDA follows the same conventions.

| Element | Convention | Example |
|---|---|---|
| Files | `lower_snake.cc/.h` | `rms_norm.cu`, `model.h` |
| Namespace | `lower_snake` | `nanochat`, `nanochat::kernels`, `nanochat::cuda` |
| Types | `PascalCase` | `class Model`, `struct Config`, `enum class DType` |
| Functions | `PascalCase` | `TrainStep()`, `RmsNormForward()` |
| Accessors | `snake_case` | `seq_len()`, `set_lr()` |
| Variables | `snake_case` | `batch_size`, `num_layers` |
| Members | `snake_case_` | `num_layers_`, `workspace_` |
| Constants | `kPascalCase` | `kSoftcap`, `kRopeBase` |
| Enum values | `kPascalCase` | `DType::kFp32` |
| Device kernels | `PascalCaseKernel` | `__global__ void RmsNormKernel(...)` |
| Macros | `NANOCHAT_*` | `NANOCHAT_CUDA_CHECK(x)`, `NANOCHAT_CHECK(cond)` |
| Guards | `NANOCHAT_PATH_FILE_H_` (or `#pragma once`) | `NANOCHAT_KERNELS_H_` |
| Errors | no exceptions; fail fast | `NANOCHAT_CHECK(cond) << "msg"` |

CUDA builds use `-fno-exceptions`. Prefer `std::span`, `constexpr`,
`[[nodiscard]]`, `std::unique_ptr`. No third-party header-only libraries unless
they ship with the CUDA toolkit.

Run `tools/nanochat lint` before a commit. It runs `clang-format` in check mode
plus the text checks: no using-directive, project-relative includes, no C-style
cast, and ASCII only. The project skill `.agents/skills/google-cpp-style/`
carries the same rules.

## Definition of done

Run `tools/nanochat check` (build plus CPU tests), `tools/nanochat lint` (the
style gate), and `tools/nanochat test --gpu <target>` for small-shape GPU
correctness. The full checklist, including finite-difference checks, oracle
parity, and formatting, is in
[docs/testing.md](docs/testing.md). Never call `bazel test` or a test binary
directly; the entry point is what applies the sandbox. See
[docs/sandbox.md](docs/sandbox.md).

## Where to change what

- The backend seam is [docs/kernels.md](docs/kernels.md). Adding hardware means
  a new `backends/<x>/`; see [docs/build.md](docs/build.md).
- The public model API and the four graphs are [docs/model.md](docs/model.md).
- Design decisions and the invariants that constrain them live in
  [DESIGN.md](DESIGN.md). Keep frozen headers `nanochat/{kernels,tensor,config,
  model}.h` architect-owned, and prefer additive changes.
