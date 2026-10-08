# Host portability: toolchain, sandbox, and devices

This document specifies how `nanochat.cpp` runs on more than one host. The
development host is a WSL2 machine with one GTX 1080 Ti. The second host is a
Kaggle Notebook with two T4 cards. Section 8 gives the Kaggle procedure. See
[build.md](build.md) for the Turing target.

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
Bazel puts the runfiles on the test's `LD_LIBRARY_PATH`. Without the client
value, `libcuda` is unresolvable and `cudaMalloc` fails with
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
7. `tools/nanochat simulate --device h100` passes with the `none` backend. The
   S0 simulator is CPU-only by construction. It needs no toolkit, no driver, no
   GPU, and no broker. It is the portable way to check Hopper-class behavior on
   a host that has no Hopper device ([simulator.md](simulator.md)).

## 8. Kaggle Notebooks

Design and procedure for `nanochat.cpp` and `pi` on a Kaggle Notebook. Sections
1 to 7 define the toolchain, the optional sandbox, and the device selection.

Status: implemented. `tools/kaggle/bootstrap.sh` and
`tools/kaggle/github-setup.sh` are in the tree.

### 1. Host description

| Property | Value |
|---|---|
| Environment | container, no systemd |
| GPU | two T4 cards, sm_75 |
| GPU memory | 16 GiB per card |
| CUDA | the image toolkit, version 12.x |
| CPU and RAM | the notebook plan quota |
| Working directory | `/kaggle/working`, writable |
| Input directory | `/kaggle/input`, read only |
| Output | `/kaggle/working`, within the output quota |
| Session | the notebook session limit |

The notebook uses one container. The container has no systemd and no cgroup v2
user manager. The resource sandbox in [sandbox.md](sandbox.md) cannot run here.

### 2. Sandbox decision

Do not use the sandbox on Kaggle. Set `NANOCHAT_SANDBOX_BACKEND=none`. The
container is the resource boundary, and the notebook quota is the ceiling.

The `none` backend still keeps the entry-point guarantee. A binary outside
`tools/nanochat` still fails.

Do not set `NANOCHAT_ALLOW_UNSANDBOXED`. The entry point handles the
configuration. The override is only for a manual one-off run.

### 3. Setup

The bootstrap script `tools/kaggle/bootstrap.sh` prepares one session. It does
this:

1. Install `git`, `build-essential`, `curl`, and `ripgrep` when absent.
2. Install Bazelisk into `~/.local/bin`.
3. Install Node.js 22.19 or newer.
4. Install `pi` with the pi installer.
5. Export `CUDA_HOME=/usr/local/cuda` and add `nvcc` to `PATH`.
6. Write `.bazelrc.local` for the Kaggle host.
7. Write `~/.nanochat.env` with the host contract.
8. Run `tools/nanochat doctor`.

`~/.nanochat.env` holds the host contract: `NANOCHAT_SANDBOX_BACKEND`,
`NANOCHAT_CPP_BACKEND`, `NANOCHAT_CPP_PRECISION`, `NANOCHAT_CUDA_ARCH`,
`NANOCHAT_CPP_CACHE`, and `CUDA_HOME`. The notebook reads the file after
bootstrap. The terminal sources it with `source ~/.nanochat.env`.

The Kaggle host file sets only the arch and the resource budget:

```
build --@rules_cuda//cuda:archs=sm_75
build --jobs=2
build --local_resources=memory=2048
startup --host_jvm_args=-Xmx1024m
```

The backend and the precision stay per-command. The CPU test loop keeps the CPU
backend and fp32. A GPU command adds `--config=cuda`.

#### 3.1 The setup module

The notebook loader cell clones the repository and adds `python/` and
`notebooks/` to `sys.path`. The module `notebooks/kaggle_setup.py` does the
rest:

1. `ensure_repository` fetches and pulls.
2. `run_bootstrap` runs `tools/kaggle/bootstrap.sh`.
3. `load_host_env` reads `~/.nanochat.env`.
4. `apply_host_path` prepends the tool directories to `PATH`.
5. `build_library` builds the T4 library and pins `NANOCHAT_CPP_LIB`.

The notebook calls `kaggle_setup.setup()` once. A change to a setup step takes
effect on the next run, so you do not import the notebook again.

Set the notebook to GPU T4 x2 and Internet on. The committed notebook metadata
requests both. Confirm them in the notebook settings.

### 4. Build and test

Run every command through `tools/nanochat`, as on the workstation.

| Command | What it does |
|---|---|
| `tools/nanochat build` | Build every target for sm_75, CPU backend |
| `tools/nanochat test` | CPU tests under the `none` sandbox |
| `tools/nanochat test --gpu //...` | T1 GPU tests, fp32 |
| `tools/nanochat test --gpu --config=fp16 //...` | T1 GPU tests, fp16 |
| `tools/nanochat test --gpu //tests:oracle_cuda_test` | T2 oracle parity |

The first session must prove the toolchain. Do this in order:

1. `tools/nanochat doctor`
2. `tools/nanochat build`
3. `tools/nanochat test`
4. `tools/nanochat test --gpu //backends/cuda/kernels:precision_gpu_test`
5. `tools/nanochat test --gpu //tests:oracle_cuda_test`
6. `tools/nanochat test --gpu //tests:train_parity_cuda_test`

The entry point gains a `--config NAME` option for `build` and `test`. It
forwards the named config to Bazel. This is how the fp16 gate runs on the T4.

### 5. Data and persistence

The repository lives under `/kaggle/working/nanochat.cpp`. The directory is
writable. It does not persist between sessions unless the notebook saves it.

- Put the parquet dataset and the checkpoints under `/kaggle/working`.
- Put read-only data in a Kaggle dataset, and mount it at `/kaggle/input`.
- The repository holds the oracle fixtures in `tests/data/`. They need no download.
- Commit the notebook to save the output. The session stops when the time
  limit expires, and the running processes stop with it.

### 6. Running pi

`pi` needs Node.js 22.19 or newer. The bootstrap installs it. Then run pi in
the repository directory.

- The terminal interface needs a terminal. Use the notebook terminal when it
  exists.
- Print mode needs no terminal. Use `pi --print "..."`.
- Start the terminal interface under `tmux` for a long session.
- Set the provider key with an environment variable. Do not write the key to
  the repository.

Example print-mode call:

```bash
pi --print "Run tools/nanochat doctor and summarize the device state."
```

### 7. Clone, commit, and merge pull requests

#### 7.1 Credentials

Use one fine-grained personal access token for the `nanochat.cpp` repository.
Give the token these permissions:

- Contents: read and write
- Pull requests: read and write
- Metadata: read

Store the token in a Kaggle Notebook secret with the label `GH_TOKEN`. Kaggle
secrets belong to the notebook and stay out of the repository.

A deploy key can push a branch. A deploy key cannot call the pull-request API.
Use the token for both push and merge.

#### 7.2 Session setup

A Python cell reads the secret and exports it. Child shells and `pi` then
inherit the value.

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["GH_TOKEN"] = UserSecretsClient().get_secret("GH_TOKEN")
os.environ["NANOCHAT_REPO"] = "owner/nanochat.cpp"
os.environ["GIT_AUTHOR_NAME"] = "Kaggle Agent"
os.environ["GIT_AUTHOR_EMAIL"] = "kaggle@example.com"
```

A shell cell then prepares the host:

```bash
tools/kaggle/bootstrap.sh
tools/kaggle/github-setup.sh
```

#### 7.3 The GitHub helper

`tools/kaggle/github-setup.sh` does this:

1. Install `gh` into `~/.local/bin` when absent.
2. Stop when `NANOCHAT_REPO` or `GH_TOKEN` is absent.
3. Run `gh auth setup-git`. Git then uses the token.
4. Set `user.name` and `user.email` from `GIT_AUTHOR_NAME` and `GIT_AUTHOR_EMAIL`.
5. Clone `nanochat.cpp` into `/kaggle/working/nanochat.cpp`.

The key line is `gh auth setup-git`. A remote URL with the token leaks the
token to `git remote -v` and to the model context.

```bash
##!/usr/bin/env bash
set -euo pipefail

: "${GH_TOKEN:?set GH_TOKEN from a Kaggle secret}"
: "${NANOCHAT_REPO:?set NANOCHAT_REPO to owner/nanochat.cpp}"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v gh >/dev/null 2>&1; then
  ver=2.63.2
  tmp=$(mktemp -d)
  curl -fsSL "https://github.com/cli/cli/releases/download/v${ver}/gh_${ver}_linux_amd64.tar.gz" \
    | tar -xz -C "$tmp"
  install -m 0755 "$tmp/gh_${ver}_linux_amd64/bin/gh" "$HOME/.local/bin/gh"
fi

gh auth setup-git
git config --global user.name  "${GIT_AUTHOR_NAME:-Kaggle Agent}"
git config --global user.email "${GIT_AUTHOR_EMAIL:-kaggle@example.com}"

dest=/kaggle/working/nanochat.cpp
clone_url="https://github.com/$NANOCHAT_REPO"
[[ -d $dest ]] || git clone "$clone_url" "$dest"
git -C "$dest" fetch --all --prune
```

Set `NANOCHAT_REPO` to the account and the repository, for example
`owner/nanochat.cpp`.

#### 7.4 Branch, commit, push, and pull request

```bash
cd /kaggle/working/nanochat.cpp
git switch -c agent/t4-kaggle
git add -A
git commit -m "feat: add the T4 build config"
git push -u origin agent/t4-kaggle
gh pr create --fill --base main
```

#### 7.5 Merge

```bash
gh pr merge --squash --delete-branch
gh pr merge --auto --squash     # wait for the checks
```

GitHub does not let a user approve their own pull request. When `main`
requires an approving review, the token cannot supply it. Use a second account,
a machine user, or `--admin`. When only status checks gate `main`,
`gh pr merge --auto --squash` waits for the checks.

#### 7.6 What pi may do

- The project `AGENTS.md` stops the main agent from a commit without approval.
- A prompt can grant the approval for the session. Example: "Commit to a
  branch and open a pull request. Do not merge."
- A sliceme worker may commit in its own worktree. That exception already
  exists.
- Keep the merge as a human step. The agent opens the pull request.

#### 7.7 Session lifetime

A Kaggle session ends at the time limit. The container and its processes stop.

- Push every branch before the session ends. The session loses a local commit
  otherwise.
- `/kaggle/working` does not survive a new session on its own. Clone the
  repository again, or keep it in a Kaggle dataset.
- The `pi` session directory is ephemeral. Put durable notes in the repository.

#### 7.8 Security

- Use a fine-grained token for one repository, with a short expiry.
- Do not put the token in a remote URL. Use `gh auth setup-git`.
- Do not print the environment. The token enters the model context.
- Rotate the token after a shared session.

### 8. Two T4 cards

The model is single-device. One job uses one card. Two cards give two options.

Option A: two independent jobs. Run one job on device 0 and one on device 1.
Each job takes its own broker lock with `tools/gpu.sh --device N`. This option
needs only the device selection from section 5.

Option B: data-parallel training. One process owns each card. The processes
all-reduce their gradients with NCCL, then each process steps its optimizer.
The design is in [python.md](python.md) section 10. Start it only after
the single-card path is fast.

### 9. Limits and risks

| Risk | Effect | Response |
|---|---|---|
| No cgroup sandbox | A runaway job can exhaust the container | Rely on the notebook quota |
| Session time limit | Processes stop without warning | Save often. Commit the notebook |
| Output quota | Large checkpoints fail to save | Keep few checkpoints |
| No internet | Bazelisk, npm, and pi fail to install | Preload a Kaggle dataset |
| GPU quota | A weekly GPU-hour limit | Keep the GPU for gates |
| Driver mismatch | The image CUDA and the driver disagree | Check with `tools/nanochat doctor` |
| `rules_cuda` autodetect | The toolkit path is not found | Set `CUDA_HOME` |
| No CUDA toolkit | The CUDA build cannot compile | Use the S0 simulator: `tools/nanochat simulate --device h100` needs no toolkit |

### 10. Definition of done

1. `tools/kaggle/bootstrap.sh` prepares a fresh session.
2. `tools/nanochat build` succeeds with the `none` sandbox.
3. `tools/nanochat test` passes.
4. The T1 GPU tests pass in fp32 and fp16.
5. The T2 oracle and train-parity gates pass on sm_75.
6. `pi --print` runs a `tools/nanochat` command.
7. `gh pr create` opens a pull request from the session.
8. `gh pr merge` merges the pull request on the origin.
