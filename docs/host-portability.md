# Host portability: toolchain, sandbox, and devices

This document specifies how `nanochat.cpp` runs on more than one host. The
development host is a WSL2 machine with one GTX 1080 Ti. The second host is a
Kaggle Notebook with two T4 cards. See [kaggle.md](kaggle.md) for the Kaggle
procedure, and [turing-t4.md](turing-t4.md) for the Turing target.

Status: implemented in the tree. The WSL2 host file is `.bazelrc.wsl`.

## 1. Goals

- Keep one source tree for every host.
- Keep the compute code free of host concerns.
- Let each host choose its compiler, its sandbox, and its devices.
- Fail loudly when a binary runs outside the entry point.

## 2. Host differences

| Concern | Workstation | Kaggle Notebook | Mechanism |
|---|---|---|---|
| Host compiler | gcc-12 | image gcc | `.bazelrc.local` |
| CUDA arch | sm_61 | sm_75 | `--config=t4` |
| Precision | fp32 | fp16 and fp32 | `--config=fp16` |
| Sandbox | systemd cgroups | none | `NANOCHAT_SANDBOX_BACKEND` |
| GPU count | one | two | `NANOCHAT_GPU_DEVICES` |
| Build jobs | four | two | `.bazelrc.local` |

## 3. Toolchain

### 3.1 The current defect

The tracked `.bazelrc` pins `CC` to `/usr/bin/gcc-12`. That path is correct for
the WSL2 host. It is wrong for every other host, and it stops the build there.

### 3.2 Decision: keep gcc, do not switch to clang

Do not switch the host compiler to clang. The reasons follow.

- `nvcc` calls the host compiler through `-ccbin`. NVIDIA tests this path with gcc.
- The WSL2 host has no clang. A switch adds a dependency to that host.
- Kaggle already has a gcc that the image CUDA supports.
- The defect is the hard-coded path, not the compiler family.

A clang migration is later work. It pays off for compile speed and for
`clangd`. It does not change device code correctness. Do it only when a
measurement shows a clear benefit.

Clang can compile CUDA source directly, without `nvcc`. That path needs a
different `rules_cuda` toolchain. It is a separate project with its own risk.

### 3.3 Fix

- Remove `CC` and `CXX` from the tracked `.bazelrc`.
- Add `try-import %workspace%/.bazelrc.local` at the end of `.bazelrc`.
- Keep the gcc-12 pins in a host file, `.bazelrc.wsl`.
- Let each host write its own `.bazelrc.local`.
- Set `CUDA_HOME` in the host file when autodetection fails.

### 3.4 Named configs

| Config | Arch | Precision | Backend |
|---|---|---|---|
| default | sm_61 | fp32 | cpu |
| `sm_75` | sm_75 | fp32 | cpu |
| `fp16` | sm_61 | fp16 | cpu |
| `cuda` | sm_61 | fp32 | cuda |
| `t4` | sm_75 | fp16 | cuda |

The `t4` config selects the Turing half-precision CUDA build. Add a `t4-fp32`
config for the fp32 gate on the same card.

The Kaggle host file sets the arch to sm_75 and the resource budget. It does
not set the backend or the precision. The backend and the precision stay
per-command, so the CPU test loop keeps the CPU backend and fp32.

## 4. Optional sandbox

### 4.1 Principle

The sandbox is an execution concern. It never appears in `kernels.h`,
`tensor.h`, `config.h`, or `model.h`. A host without cgroup v2 must still
build, test, and train.

### 4.2 Backend selection

`NANOCHAT_SANDBOX_BACKEND` selects the sandbox backend:

| Value | Behavior |
|---|---|
| `systemd` | Use `systemd-run --user --scope` and cgroup v2 |
| `none` | Do not sandbox. Run the command directly |
| `auto` | Probe `systemd-run`. Use it when the probe succeeds |

The entry point chooses in this order:

1. Use `NANOCHAT_SANDBOX_BACKEND` when the environment sets it.
2. Otherwise use `auto`.
3. In `auto`, run `systemd-run --user --scope true`.
4. Use `systemd` when the probe succeeds.
5. Otherwise use `none`.

### 4.3 The `none` backend

The `none` backend does this:

- Set `NANOCHAT_SANDBOX=off`.
- Do not start a scope.
- Do not apply `taskset` or `prlimit`.
- Do not install the aggregate slice.
- Log one line to `/tmp/nanochat-sandbox.log`.
- Exec the command.

The `none` backend keeps the entry-point guarantee. A binary that runs outside
`tools/nanochat` still fails, because the sentinel is absent.

### 4.4 The entry-point sentinel

`SandboxProfile()` returns the profile name. The value `off` is a valid profile
for the guard. The guard checks for a marker. The marker means "started by the
entry point". It does not mean "the host applied a cgroup".

This is a small behavior change. A reader must not treat `off` as an error.

### 4.5 The verify step

`verify.sh` reads the limits back from the process cgroup. The `none` backend
skips this step. A host without cgroups cannot verify them. On Kaggle the
container is the boundary.

### 4.6 Forward the selection to test actions

Bazel gives a test action a small environment. Add the following lines to
`.bazelrc` so a test sees the same selection as the caller:

```
test --test_env=NANOCHAT_SANDBOX_BACKEND
test --test_env=NANOCHAT_GPU_DEVICES
test --test_env=LD_LIBRARY_PATH
```

The last line forwards the CUDA driver directory. Bazel vendors the toolkit's
`libcudart`/`libcublas` into the runfiles but not the driver's `libcuda.so.1`,
which lives under the driver mount (for example `/usr/local/nvidia/lib64`).
The runfiles are put on the test's `LD_LIBRARY_PATH`; without the client value
`libcuda` is unresolvable and `cudaMalloc` fails with
`CUDA_ERROR_INSUFFICIENT_DRIVER`.

A test action does not inherit the broker's shell, so `CUDA_VISIBLE_DEVICES`
never reaches it on its own. The test wrapper (`tools/sandbox.sh`) therefore
sets `CUDA_VISIBLE_DEVICES` from the forwarded `NANOCHAT_GPU_DEVICES`. Without
that step a GPU suite runs on the default device even when the broker locked a
different one.

## 5. GPU broker and device selection

The broker serializes heavy GPU jobs. Keep the flock lock and the `nvidia-smi`
gate on every host. Add device selection:

- `tools/gpu.sh --device N` sets `CUDA_VISIBLE_DEVICES=N`.
- Keep one lock per device, at `/tmp/nanochat-gpu-<N>.lock`.
- `NANOCHAT_GPU_DEVICES` sets the default device set. The test wrapper turns
  it into `CUDA_VISIBLE_DEVICES` for the test action (section 4.6).
- Two independent jobs can use two devices at the same time.

The `none` backend does not enforce the aggregate RAM and CPU budget. The host
quota is the only ceiling. Do not start two heavy jobs on one card.

## 6. Build resources

The tracked `.bazelrc` sizes the Bazel server for the 12 vCPU workstation. A
host with fewer cores must lower `--jobs` and `--local_resources`.

| Setting | Workstation | Kaggle |
|---|---|---|
| `--jobs` | 4 | 2 |
| `--local_resources=memory` | 3072 | 2048 |
| server heap | 1500m | 1024m |

Put the host values in `.bazelrc.local`.

## 7. Definition of done

1. `tools/nanochat build` works on both hosts.
2. `tools/nanochat test` works with the `systemd` backend.
3. `tools/nanochat test` works with the `none` backend.
4. A binary outside `tools/nanochat` still fails.
5. `tools/nanochat doctor` reports the selected sandbox backend.
6. The tracked `.bazelrc` contains no absolute compiler path.
