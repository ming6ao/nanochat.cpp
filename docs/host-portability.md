# Host portability: toolchain, sandbox, and devices

`nanochat.cpp` runs on more than one host. The development host is a WSL2
machine. The second host is a Kaggle notebook with two T4 cards. The target node
is eight H100 cards on Modal or on native Linux. Section 8 gives the Kaggle
procedure, section 9 the pipeline chain on three hosts, and Part 1 the H100 run
book.

The one rule: the compute code holds no host concern. Each host chooses its
compiler, its sandbox, and its devices in a host file, and a binary that runs
outside the entry point fails loudly.

| Host | Host file | Arch | Precision | Sandbox |
|---|---|---|---|---|
| WSL2 workstation | `.bazelrc.wsl` | none | fp32 | systemd cgroups |
| Kaggle T4 | notebook setup | sm_75 | fp32 | none |
| H100 node | `.bazelrc.local` | sm_90 | bf16 or fp8 | systemd cgroups |

The workstation card is below `sm_75`, so the tree no longer supports it. The
host still builds and runs the CPU backend and the simulator. A device gate
needs a Turing card or newer, on this host or on another one.

## 1. Goals

- Keep one source tree for every host.
- Keep the compute code free of host concerns.
- Let each host choose its compiler, its sandbox, and its devices.
- Fail loudly when a binary runs outside the entry point.

## 2. Host differences

| Concern | Workstation | Kaggle Notebook | Mechanism |
|---|---|---|---|
| Host compiler | gcc-12 | image gcc | `.bazelrc.local` |
| CUDA arch | none | sm_75 | `--@rules_cuda//cuda:archs=sm_75` |
| Precision | fp32 | fp32 | build default |
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
| default | sm_75 | fp32 | cpu |
| `cuda` | sm_75 | fp32 | cuda |
| `hopper` | sm_90 | bf16 | cuda |

A Turing build passes `--config=cuda --@rules_cuda//cuda:archs=sm_75`, because
Turing has no bf16 and the T4 runs fp32 ([precision.md](precision.md)). The tree
keeps no named Turing config: the build helper emits the flags.

The `hopper` config selects the bf16 CUDA build for a Hopper card, `sm_90`. It
is the hand-run form of the same build the chain's H100 leg runs, and it is not
the `h100` config: `h100` is the *simulator* profile, which sets
`--define=simulator=1` and answers from the device profile table instead of the
card ([build.md](build.md), the Precision section). It is a hand-run shorthand
only: `_build._build_command` still emits its explicit flag triple,
`--config=cuda --define=precision=<p> --@rules_cuda//cuda:archs=<a>`, and never
a named config, so the H100 leg and a hand run are the same build.

The Kaggle host file sets the arch to sm_75 and the resource budget. It does
not set the backend. The backend stays per-command, so the CPU test loop keeps
the CPU backend and fp32.

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
- Do not apply `taskset`.
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

the verifier reads the limits back from the process cgroup. The `none` backend
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

A test action does not inherit the worker's shell, so `CUDA_VISIBLE_DEVICES`
never reaches it on its own. The worker exports `NANOCHAT_GPU_DEVICES` for the
job, and the test wrapper (`python/nanochat_cpp/sandbox.py`) sets `CUDA_VISIBLE_DEVICES` from
it. Without that step a GPU job runs on the default device even when the worker
owned a different one.

## 5. GPU queue and device selection

The GPU job queue serializes the heavy GPU jobs
([sandbox.md](sandbox.md)). The worker owns the devices and the
`nvidia-smi` gate. The client does not name a device.

- The worker reads the pool from `nvidia-smi`, or from `NANOCHAT_GPU_DEVICES`
  when it is set.
- At claim, the worker assigns the pool to the job and exports
  `NANOCHAT_GPU_DEVICES`.
- The test wrapper turns that value into `CUDA_VISIBLE_DEVICES` for the job
  (section 4.6).
- Stage 1 runs one job at a time and assigns the whole pool. A later stage can
  run one job per free device subset.

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

## 8. Kaggle Notebooks

Design and procedure for `nanochat.cpp` and `pi` on a Kaggle Notebook. Sections
1 to 7 define the toolchain, the optional sandbox, and the device selection.

`tools/kaggle/bootstrap.sh` prepares the host and
`tools/kaggle/github-setup.sh` prepares the clone.

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

Do not set `NANOCHAT_SANDBOX` by hand. The entry point handles the
configuration. The sentinel is only for a manual one-off run.

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
| `tools/nanochat test --gpu //tests:oracle_cuda_test` | T2 oracle parity |

The first session must prove the toolchain. Do this in order:

1. `tools/nanochat doctor`
2. `tools/nanochat build`
3. `tools/nanochat test`
4. `tools/nanochat test --gpu //backends/cuda/kernels:precision_gpu_test`
5. `tools/nanochat test --gpu //tests:oracle_cuda_test`
6. `tools/nanochat test --gpu //tests:train_parity_cuda_test`

The entry point gains a `--config NAME` option for `build` and `test`. It
forwards the named config to Bazel.

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
The queue runs one job at a time, so this option needs stage Q6 of
[sandbox.md](sandbox.md): one job per free device subset.

Option B: data-parallel training. One process owns each card. The processes
all-reduce their gradients with the native ring, then each process steps its
optimizer.
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
4. The T1 GPU tests pass in fp32.
5. The T2 oracle and train-parity gates pass on sm_75.
6. `pi --print` runs a `tools/nanochat` command.
7. `gh pr create` opens a pull request from the session.
8. `gh pr merge` merges the pull request on the origin.

## 9. The pipeline chain on three hosts

One job file runs base training, then SFT, then RL, and each stage reads the
checkpoint of the stage before it. `docs/post-training.md` Part 1, section 9 gives the
chain itself; this section gives what each host contributes and what the run
records about it. The entry is always the tool (design decision D10 of
`docs/post-training.md`), because every stage subcommand calls
`RequireSandboxOrDie`, and a notebook's `NANOCHAT_SANDBOX_BACKEND=none`
satisfies the library only.

### 9.1 The host block of a run

`pipeline._host()` fills one block of `deliver/pipeline_report.json` from the
run, not from the host file: `backend` is `_build.backend()`, `arch` is
`_build.architecture()`, `precision` is `_build.precision()`, and `revision` is
`NANOCHAT_SOURCE_REVISION` when it is set, else the short revision
`git rev-parse HEAD` reports, else the `git_revision` the stages recorded. A
container that copies the tree without `.git` sets the variable, so a report
still says which build ran, and the block on two legs is the one field a reader
compares.

| Leg | `backend` | `arch` | `precision` | Sandbox | Build |
|---|---|---|---|---|---|
| Local CPU | `cpu` | empty | `fp32` | `train` scope | Bazel |
| Local GPU | `cuda` | the card's capability | `fp32` | `t1-gpu` queue job | Bazel |
| Colab T4 | `cuda` | `sm_75` | `fp32` | `none` | source build |
| Modal T4 leg | `cuda` | `sm_75` | `fp32` | `none` | image build |
| Modal H100 leg | `cuda` | `sm_90` | `bf16` | `none` | image build, bf16 |
| Modal CPU | `cpu` | empty | `fp32` | `none` | image build, CPU config |

The two card legs are `sm_75` because the image compiles for `sm_75` and a free
Colab session hands out a T4. A local card uses its own architecture through
`NANOCHAT_CUDA_ARCH`, and `tools/nanochat doctor` reports the name, the driver,
the memory, the use, and the utilisation but not the capability, so the value
comes from `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`.

The last row is the verification-only leg of section 9.4. It needs the second
image, which the tree does not hold today, so it records no gate and no number.

The two Modal rows are one run, not two independent ones: the T4 leg runs first
at fp32, the H100 leg second at bf16, each in its own image, and the host block
of each report is what tells a reader which one produced it. Section 9.3 has
the two functions, the run ids, and the gates that compare the pair.

### 9.2 Colab with a T4: the gate is owed

Row 9 of `docs/post-training.md` names a manual Colab run as its gate, and
section 6.2 of that plan gives the four cells. This campaign wrote
`notebooks/colab_pipeline.ipynb` and `notebooks/colab_setup.py` and ran neither
on Colab, so the gate stays owed until a session runs the four cells. No leg of
the table below is a measurement: each is a fact the runtime fixes, and a
measurement appears here only after a run produces one.

| Property | Value | Where it comes from |
|---|---|---|
| Backend | `cuda` | cell 2 sets `NANOCHAT_CPP_BACKEND` |
| Arch | `sm_75` | `probe()` maps `nvidia-smi --query-gpu=compute_cap` to `sm_<major><minor>`, and a free session gives a T4 |
| Precision | `fp32` | the build default |
| Sandbox | `none` | Colab has no systemd user manager, so no cgroup scope exists |
| Build | from source | decision D8: the install root carries no `tools/nanochat` and no stage binary |
| Entry | `tools/nanochat train -- python -m nanochat_cpp.pipeline` | decision D10 |

A paid session can give a V100, `sm_70`, or an A100, `sm_80`. An `sm_75` build
then fails inside `nanochat_init` with a message about the architecture, not
about the job, so cell 2 sets `NANOCHAT_CUDA_ARCH` from `probe()` and the setup
build asks for the same value.

The procedure, which is the record until a run exists:

1. **Setup.** Clone the tree, install Bazelisk with `tools/kaggle/bootstrap.sh`,
   then `colab_setup.setup(backend="cuda", arch=probe(),
   prefer_installed=False)`, then `colab_setup.stages(...)`. Two calls, not one:
   `setup` resolves the library only. `prefer_installed=False` is required, not
   a preference, because `setup` prefers an installed distribution by default
   and that root has no `tools/nanochat` and no stage binary, which contradicts
   D8 and breaks D10.
2. **Environment.** Five values: `NANOCHAT_BASE_DIR=/content/cache`,
   `NANOCHAT_RUNS_DIR=/content/runs`, `NANOCHAT_CPP_BACKEND=cuda`,
   `NANOCHAT_CUDA_ARCH=probe()`, and `NANOCHAT_SANDBOX_BACKEND=none`. The cell
   also puts the tree's `python/` on `PYTHONPATH`, because the tool replaces
   itself with the command it wraps and the child starts in the notebook's
   directory. It never sets `NANOCHAT_SANDBOX` by hand: section 4 and
   `docs/sandbox.md` forbid it, and the container quota is the only bound here.
3. **Run.** `tools/nanochat train -- python -m nanochat_cpp.pipeline
   /content/runs/job.json`. Cell 3 must wrap the module, because the backend
   value satisfies `nanochat_init` only: a bare module starts the chain and then
   fails inside the first stage executable, which accepts `NANOCHAT_SANDBOX` or
   the tool's sentinel and not the backend value.
4. **Collect.** Print the report and offer one download of `deliver/`.

What a passing run must show, as a checklist and not as a result:

- The four stages with the step counts the job asked for, and the eval record
  with both `bpb` values finite. The committed example job asks for `base`@200,
  `sft`@200, `rl`@8, and one evaluator batch.
- `deliver/chatrl.ckpt` and `deliver/pipeline_report.json` under
  `/content/runs/<run_id>/`.
- `host.backend` equal to `cuda` and `host.arch` equal to `sm_75`.
- A base loss trace that falls from its first step to its last.

The manual gate is recorded as owed: a later run replaces this status line with
the date of the session and adds the report excerpt it printed.

### 9.3 Modal: two legs, two images, two precisions

`tools/modal/app.py` holds four functions over two images, and one entry point
runs both legs of one job. Each leg's host block is a row of the table in
section 9.1, and the legs themselves are rows of this table:

| Leg | Card | Arch | Precision | Image | Functions | Run id |
|---|---|---|---|---|---|---|
| `t4` | `T4` | `sm_75` | `fp32` | fp32, `sm_75` | `train`, `pipeline` | `<run_id>-t4` |
| `h100` | `H100` | `sm_90` | `bf16` | bf16, `sm_90` | `train_h100`, `pipeline_h100` | `<run_id>-h100` |

The two images are the point. One Bazel output base holds one build of
`libnanochat_shared.so`, and a container compiles for exactly one architecture
and one precision, so the T4 leg and the H100 leg cannot share a library. The
frozen `Leg` record carries the name, the card type, the card count, the
architecture, the precision, and the build configuration; `image_for(leg)`
returns that leg's image, and the stage executables, `NANOCHAT_CPP_PRECISION`,
`NANOCHAT_CUDA_ARCH`, and the build command all read the leg. The function a
leg deploys is one shape:

```
@app.function(image=image_for(leg), gpu=leg.gpu, cpu=CPU_CORES,
              volumes={"/runs": runs}, secrets=_secrets(),
              timeout=TIMEOUT_SECONDS, max_containers=1)
def pipeline(job_json, run_id, stage_json="", publish_json="") -> str
def pipeline_h100(job_json, run_id, stage_json="", publish_json="") -> str
```

What the two legs share, so they cannot drift:

- `pipeline` and `pipeline_h100` are thin faces of one `_pipeline_on(leg, ...)`.
- The image builds the library and `//src:nanochat_cli` in one call, so a
  container needs no second build. It names the one binary through the
  `NANOCHAT_SFT_MAIN` and `NANOCHAT_RL_WORKER` overrides, because the installed
  distribution's `repo_root()` is not the image tree (B6).
- `_check_cards(leg, ...)` stops a container whose card does not match the
  leg's type, count, and architecture, and `max_containers=1` keeps one chain at
  a time.
- The app rewrites `out_root` to `/runs/<run_id>`, because the app owns the
  Volume path and one job runs four stages. The chain nests the job's own
  `run_id` under it, so `deliver/` lands at `/runs/<run_id>/<run_id>/deliver/`.
- The work starts as `tools/nanochat client --profile train -- python -m
  nanochat_cpp.pipeline <job.json>`, which is the container entry of D10, and
  `worker.log` holds its output. The commit keeper makes the run directory
  visible to a poll of the `nanochat-runs` Volume.
- An empty `publish_json` stops after the chain, which has already delivered
  `deliver/`. A non-empty value reuses `_publish` for the Hub push of the final
  stage, so the chain gains no second uploader.
- `stage_json` is accepted and ignored: the chain warm starts each stage from
  the checkpoint of the stage before it, so a Modal stage cache has nothing to
  point at.
- Both functions pass the same `cpu=` value. Modal sizes the CPU by memory when
  `cpu=` is absent, so two legs with different memory would get different host
  budgets, and gate G2 of [post-training.md](post-training.md) Part 3,
  section 8 compares the two base step times: an equal `cpu=` is what makes
  that comparison mean something.

What only the entry point does:

- `pipeline_demo` runs the T4 leg first and the H100 leg second. The T4 report is
  on the Volume before the H100 container starts, so a reviewer reads the cheap
  leg while the expensive one runs, and an H100 failure leaves the T4 report
  readable.
- The two run ids are `<run_id>-t4` and `<run_id>-h100`, so both reports stay on
  the `nanochat-runs` Volume side by side.
- It prints one document with three keys: `legs` (each leg's host block, base
  rates, RL checkpoint facts, and roofline record), `comparison` (the cross-leg
  step-time ratio, the speed ratio, and the MFU), and `gates` (G1 to G3, each
  pass or fail with the numbers it read). It exits non-zero when a gate fails.
- Gate G1 reads the H100 report's `host.precision`, `host.arch`, and the RL
  checkpoint metadata. G2 bounds the H100 base step time at 1.25 times the T4
  one. G3 divides the measured H100 step by the roofline prediction the H100
  container recorded and holds the ratio inside `[1, 200]` until row 9 of
  `docs/post-training.md` narrows it. No gate reads the MFU or the speed
  ratio.

The T4 leg has a recorded run. `modal run tools/modal/app.py::pipeline_demo`
ran on 2026-10-31, before the second leg existed, on a T4 and at the demo job of
that day. This is the host block and the outcome of the report it delivered:

| Item | Value |
|---|---|
| `host.backend` | `cuda` |
| `host.arch` | `sm_75` |
| `host.precision` | `fp32` |
| run directory | `/runs/pipeline-demo/demo` |
| delivered | `deliver/chatrl.ckpt` and `deliver/pipeline_report.json` |
| stages | `base` at 100, `sft` at 100, `rl` at 105 |
| base loss | 6.1874 to 0.0049 |
| `total_seconds` | 4.46 |

The two-leg run is row 9 of `docs/post-training.md`. It ran on 2026-10-31
at the `standard` job and recorded both legs:

| Item | T4 leg | H100 leg |
|---|---|---|
| `host.arch` | `sm_75` | `sm_90` |
| `host.precision` | `fp32` | `bf16` |
| base `step_ms` | 128.74 | 124.39 |
| MFU | 0.93 % | 0.008 % |
| `total_seconds` | 60.18 | 59.94 |
| delivered | `deliver/chatrl.ckpt` | `deliver/chatrl.ckpt` |
| gates G1 to G3 | pass | pass |

`startup_check` reads `not_run` in both reports: the chain's mesh trainer does
not call the hardware correctness gate. A hand-wired gate failed on the H100 in
the scaled fp8 GEMM (`cublasLtMatmulAlgoGetHeuristic`, status 15), and the fp8
fast-accumulate attribute is an Ada feature that Hopper rejects. Both Hopper
defects are guarded in `backends/cuda/gemm.cu`, `src/hw_gate.cc`, and
`src/ops.cc`.

The image work is not what it looks like. The layer chain takes 49 seconds on
a cached base image: copy 2.1 s, install 6.6 s, environment 1.6 s, and the
Bazel CUDA build 36.8 s. The first build cost about 25 minutes, almost all of
it pulling the 4.11 GB CUDA devel base and installing `build-essential`. The
second leg doubles both numbers once: the H100 image is a second build and a
second cached image, and later deploys of either use the cache.

Two app defects surfaced only when the gate ran, and both are fixed in the
follow-up pull request. The module derived the repository from its own path,
which `modal run` mounts at `/root/app.py`, so `Path(__file__).parents[2]`
raised `IndexError`. And the chain wrote `deliver/` one directory above the run
folder, because `run()` passed the run root to `deliver()`, which names that
argument the run directory. The Colab gate of section 9.2 is still owed.

### 9.4 The Modal CPU leg is a second image

The image above pins one compiled backend: `NANOCHAT_CPP_LIB` names the CUDA
library and the build passes `--config=cuda`. A container with no card therefore
fails inside `nanochat_init`, and `_check_cards` calls `nvidia-smi`, which does
not exist there.

So a CPU verification leg is a second image whose build states
`--define=backend=cpu` and whose function omits `gpu=`, not a CPU branch inside
`tools/modal/app.py`.
One Bazel output base holds one build of `libnanochat_shared.so`, so the two
images must stay apart. The tree holds the CUDA image today; the CPU image is
the verification-only leg of the table in section 9.1, and it runs no gate.

## Part 1. The H100 node run book

This part is a run book for one depth-24 pretraining run on eight H100 cards. It
states the memory plan, the batch plan, the data plan, the launch, the
acceptance criteria, and the record to keep. The designs it depends on are
[precision.md](precision.md), [attention.md](attention.md),
[performance-model.md](performance-model.md), and
[simulator.md](simulator.md).

### 1. Goal

Train one depth-24 model on eight H100 cards, and compare the result and the
step time against the reference.

The reference is the nanochat leaderboard entry for commit `a825e63`: a depth-24
model with fp8 training, a token-to-parameter ratio of 8, and an average of
1.65 hours over five runs.

### 2. The node

| Item | Value |
|---|---|
| Accelerator | 8 H100 SXM cards, 80 GiB each |
| Interconnect | NVLink 4.0 with NVSwitch, one node |
| Host toolkit | CUDA 12.0 or newer; record the exact version |
| Collective | The native ring with the NVLink transport |
| Node count | One |
| Second transport | An RDMA adapter with GPUDirect RDMA, if the native ring is used |

Prerequisites:

   ring has its NVLink transport.
2. The bf16 compute path from `docs/precision.md` Stage 2a.
3. The fp8 path from [precision.md](precision.md), if fp8 is used.
4. The acceptance criteria in section 8. Agree them before the run starts.

Record the `nvlink` view of the node with `nvidia-smi topo -m`. The
`docs/distributed.md` Part 3 discovery step owns the probe list.

### 3. The memory plan

Every number below is for depth 24, sequence 2048, batch 16 per card, and the
value embeddings on. The source is the model-fit engine, which mirrors
`BuildTrainWorkspace` in `src/model.cc` and `NanochatOptimizer::Build` in
`src/optim.cc`.

| Group | bf16 activations | fp8 activations |
|---|---:|---:|
| Parameters, gradients, and the fp32 master | 12.9 GiB | 11.6 GiB |
| The optimizer state | 7.8 GiB | 7.8 GiB |
| The training arena | 52.2 GiB | 26.1 GiB |
| The dense attention scratch | 4.5 GiB | 2.2 GiB |
| **Total** | **77.4 GiB** | **47.8 GiB** |

Two points are load bearing:

1. **The dense attention scratch is not in the arena.** The cuBLAS attention
   forward allocates a cached `[batch, heads, seq, kv_len]` buffer through
   `Alloc`. At this shape the request is three matrices, which is 4.5 GiB in
   bf16. The arena model does not count it. Size the card from the total row.
2. **The bf16 total leaves about 1 GiB of margin.** The card budget is 80 GiB
   minus a reserve of about 1.5 GiB for the CUDA context and the library
   workspaces. A bf16 run at batch 16 fits, and only just. Measure the free
   memory after the first step, before a long run.

The fp8 activation path removes the tight margin. It also allows batch 32, at a
total of 76.1 GiB. Prefer batch 32 with fp8 activations over batch 16 with bf16
activations, if the step time allows it.

The Track 3 preset turns the value embeddings off. That drops the parameter and
optimizer rows to 7.3 GiB and 3.3 GiB. The arena is unchanged. Use the preset
only if the model change is acceptable, because the value embeddings help the
reference result.

### 4. The batch and step plan

| Quantity | Value |
|---|---|
| Sequence length | 2048 |
| Device batch per card | 16 |
| Cards | 8 |
| Tokens per forward | 262,144 |
| Total batch | 1,048,576 |
| Gradient accumulation | 4 |

The total batch follows the reference formula in `python/nanochat_cpp/plan.py`:
`B_REF * (D / D_REF) ** 0.383`, rounded to a power of two. At depth 24 the
result is 1,048,576 for both horizons below.

| Token-to-parameter ratio | Tokens | Optimizer steps | Total arithmetic |
|---|---:|---:|---:|
| 8:1 | 5.84 billion | 5,570 | 2.79 times 10^19 FLOPs |
| 12:1 | 8.76 billion | 8,356 | 4.18 times 10^19 FLOPs |

The arithmetic count comes from `EstimateFlopsPerToken` in
`src/performance_model.cc`, which returns 4.775 billion FLOPs per token for this
configuration.

Use the 8:1 ratio for the GPT-2 comparison. The reference uses ratio 8 for the
same target, and a 12:1 run needs about 50 percent more time.

### 5. The data plan

The run needs 5.84 billion tokens at ratio 8. The reference uses NVIDIA ClimbMix
shards, and it downloads about 150 shards for a GPT-2 capability run.

1. Download the shards with the dataset helper.
2. Train the tokenizer on about 2 billion characters, and write one `NCTOKEN1`
   artifact. The vocabulary size is 32768.
3. Hold one validation shard for the bits-per-byte loop.
4. Measure the token count per shard, and record it. Do not assume the reference
   count.
5. Confirm the sustained input rate.

The input rate is the hard requirement. The reference trains at about 983
thousand tokens per second on the same node. The loader and the tokenizer must
sustain at least that rate across eight reading ranks, or the GPUs idle.

The native parquet reader matches `pyarrow` on real ClimbMix shards. See
[data.md](data.md). Measure the rate before the long run.

### 6. The launch

One process per rank, one queue job for the group:

```bash
tools/nanochat dist --nproc-per-node 8 --profile train \
    --device 0,1,2,3,4,5,6,7 -- <nanochat_cli> train \
    --layers 24 --heads 12 --kv-heads 12 --hidden 1536 --seq 2048 \
    --vocab 32768 --window-pattern SSSL \
    --batch 16 --grad-accum 4 --num-iterations 5570 \
    --train-parquet 'data/*.parquet' --val-parquet 'data/val/*.parquet' \
    --tokenizer tokenizer.nctoken \
    --device 'NVIDIA H100 80GB HBM3' \
    --save-every 1000 --eval-every 500 --log-every 10
```

`tools/nanochat dist` starts one process per rank. It submits one queue job
for the group, applies the `train` sandbox profile, and sets the five rank
variables that `nanochat_cli train` reads:

| Variable | Meaning |
|---|---|
| `NANOCHAT_WORLD_SIZE` | The group size |
| `NANOCHAT_RANK` | The global rank |
| `NANOCHAT_LOCAL_RANK` | The rank on this node |
| `NANOCHAT_MASTER_ADDR` | The rank 0 host |
| `NANOCHAT_MASTER_PORT` | The rank 0 port |

overrides a variable.

Run one step first, then check the free memory, then start the full run.

### 7. The broker and the sandbox on a native host

The sandbox profiles in `tools/sandbox/profiles.json` are host-local. They were
written for a 12 vCPU and 7.7 GiB WSL2 host. The H100 node has a different CPU
and memory budget.

Before the run:

1. Re-derive the `train` profile for the node. The profile bounds the CPU, the
   host memory, the process count, and the wall clock. It does not bound the
   device memory.
2. Confirm the aggregate ceiling in `tools/sandbox/nanochat.slice`.
3. Treat the eight cards as one resource. Submit one queue job for the whole
   group, as `tools/nanochat dist` does.
4. Confirm that `tools/nanochat doctor` reports the device, the sandbox, and the
   lock state.

Do not run the training outside the entry point. Every executable calls
`nanochat::RequireSandboxOrDie` at startup, and it exits with a corrective
command.

### 8. The acceptance run

The acceptance run has four criteria. Agree them before the run.

| # | Criterion | Measure |
|---|---|---|
| 1 | The loss curve matches the reference shape | The bits-per-byte curve against the reference curve |
| 2 | The step time reaches the bound | The measured step against the bound in [performance-model.md](performance-model.md) |
| 3 | The capability reaches the target | The CORE score against the GPT-2 threshold of 0.256525 |
| 4 | The run is reproducible | Two short runs at the same seed give the same loss |

Criteria 1 and 2 need no capability evaluation, so check them early. Stop the
run if criterion 2 is far above the bound, because a long run at the wrong
rate wastes the node.

Run the pipeline in this order:

1. One step, to check the memory and the launch.
2. Fifty steps, to check the loss trajectory and the step time.
3. The full run.
4. The evaluation in section 9.

### 9. The evaluation

| Metric | Tool | Side |
|---|---|---|
| Bits per byte | The validation loop, on the validation shard | The C++ harness |
| CORE | `nanochat_cli score` plus the Python reduction | Split |
| Samples | The generate path | The C++ harness |

The C++ side computes the logits. The Python side owns the task data and the
reduction. See [eval.md](eval.md).

Compare against the reference:

| Metric | Reference value |
|---|---|
| CORE, GPT-2 threshold | 0.256525 |
| CORE, nanochat commit `a825e63` | 0.2626 |
| Validation bits per byte, commit `a825e63` | 0.71800 |

The bits-per-byte value is comparable only on the same dataset and the same
tokenizer. Record both.

### 10. The run record

Record one row per run. The broker log and the sandbox log hold the rest.

| Field | Value |
|---|---|
| Date and revision | |
| Node and driver | |
| Toolkit version | |
| Build configuration | |
| Precision | |
| Token-to-parameter ratio | |
| Tokens and steps | |
| Wall-clock time and step time | |
| Peak device memory | |
| Validation bits per byte | |
| CORE score | |
| Prediction against measurement | |
| Deviations from this book | |

Store the record beside the checkpoint, or in this document. The record is the
only durable evidence of the run.
