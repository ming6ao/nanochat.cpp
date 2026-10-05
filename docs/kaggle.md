# Kaggle Notebooks

Design and procedure for `nanochat.cpp` and `pi` on a Kaggle Notebook. Read
[host-portability.md](host-portability.md) first. That document defines the
toolchain, the optional sandbox, and the device selection.

Status: implemented. `tools/kaggle/bootstrap.sh` and
`tools/kaggle/github-setup.sh` are in the tree.

## 1. Host description

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

## 2. Sandbox decision

Do not use the sandbox on Kaggle. Set `NANOCHAT_SANDBOX_BACKEND=none`. The
container is the resource boundary, and the notebook quota is the ceiling.

The `none` backend still keeps the entry-point guarantee. A binary outside
`tools/nanochat` still fails.

Do not set `NANOCHAT_ALLOW_UNSANDBOXED`. The entry point handles the
configuration. The override is only for a manual one-off run.

## 3. Setup

The bootstrap script `tools/kaggle/bootstrap.sh` prepares one session. It does
this:

1. Install `git`, `build-essential`, `curl`, and `ripgrep` when absent.
2. Install Bazelisk into `~/.local/bin`.
3. Install Node.js 22.19 or newer.
4. Install `pi` with the pi installer.
5. Export `CUDA_HOME=/usr/local/cuda` and add `nvcc` to `PATH`.
6. Write `.bazelrc.local` for the Kaggle host.
7. Export `NANOCHAT_SANDBOX_BACKEND=none`.
8. Run `tools/nanochat doctor`.

The Kaggle host file sets only the arch and the resource budget:

```
build --@rules_cuda//cuda:archs=sm_75
build --jobs=2
build --local_resources=memory=2048
startup --host_jvm_args=-Xmx1024m
```

The backend and the precision stay per-command. The CPU test loop keeps the CPU
backend and fp32. A GPU command adds `--config=cuda`.

## 4. Build and test

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

## 5. Data and persistence

The repository lives under `/kaggle/working/nanochat.cpp`. The directory is
writable. It does not persist between sessions unless the notebook saves it.

- Put the token shards and the checkpoints under `/kaggle/working`.
- Put read-only data in a Kaggle dataset, and mount it at `/kaggle/input`.
- The repository holds the oracle fixtures in `tests/data/`. They need no download.
- Commit the notebook to save the output. The session stops when the time
  limit expires, and the running processes stop with it.

## 6. Running pi

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

## 7. Clone, commit, and merge pull requests

### 7.1 Credentials

Use one fine-grained personal access token for the `nanochat.cpp` repository.
Give the token these permissions:

- Contents: read and write
- Pull requests: read and write
- Metadata: read

Store the token in a Kaggle Notebook secret with the label `GH_TOKEN`. Kaggle
secrets belong to the notebook and stay out of the repository.

A deploy key can push a branch. A deploy key cannot call the pull-request API.
Use the token for both push and merge.

### 7.2 Session setup

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

### 7.3 The GitHub helper

`tools/kaggle/github-setup.sh` does this:

1. Install `gh` into `~/.local/bin` when absent.
2. Stop when `NANOCHAT_REPO` or `GH_TOKEN` is absent.
3. Run `gh auth setup-git`. Git then uses the token.
4. Set `user.name` and `user.email` from `GIT_AUTHOR_NAME` and `GIT_AUTHOR_EMAIL`.
5. Clone `nanochat.cpp` into `/kaggle/working/nanochat.cpp`.

The key line is `gh auth setup-git`. A remote URL with the token leaks the
token to `git remote -v` and to the model context.

```bash
#!/usr/bin/env bash
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

### 7.4 Branch, commit, push, and pull request

```bash
cd /kaggle/working/nanochat.cpp
git switch -c agent/t4-kaggle
git add -A
git commit -m "feat: add the T4 build config"
git push -u origin agent/t4-kaggle
gh pr create --fill --base main
```

### 7.5 Merge

```bash
gh pr merge --squash --delete-branch
gh pr merge --auto --squash     # wait for the checks
```

GitHub does not let a user approve their own pull request. When `main`
requires an approving review, the token cannot supply it. Use a second account,
a machine user, or `--admin`. When only status checks gate `main`,
`gh pr merge --auto --squash` waits for the checks.

### 7.6 What pi may do

- The project `AGENTS.md` stops the main agent from a commit without approval.
- A prompt can grant the approval for the session. Example: "Commit to a
  branch and open a pull request. Do not merge."
- A sliceme worker may commit in its own worktree. That exception already
  exists.
- Keep the merge as a human step. The agent opens the pull request.

### 7.7 Session lifetime

A Kaggle session ends at the time limit. The container and its processes stop.

- Push every branch before the session ends. The session loses a local commit
  otherwise.
- `/kaggle/working` does not survive a new session on its own. Clone the
  repository again, or keep it in a Kaggle dataset.
- The `pi` session directory is ephemeral. Put durable notes in the repository.

### 7.8 Security

- Use a fine-grained token for one repository, with a short expiry.
- Do not put the token in a remote URL. Use `gh auth setup-git`.
- Do not print the environment. The token enters the model context.
- Rotate the token after a shared session.

## 8. Two T4 cards

The model is single-device. One job uses one card. Two cards give two options.

Option A: two independent jobs. Run one job on device 0 and one on device 1.
Each job takes its own broker lock with `tools/gpu.sh --device N`. This option
needs only the device selection from [host-portability.md](host-portability.md).

Option B: data-parallel training. Two processes exchange gradients with a
collective. This option needs NCCL, a rank argument, and a new parity test.
Treat it as a separate project. Start it only after the single-card path is
fast.

## 9. Limits and risks

| Risk | Effect | Response |
|---|---|---|
| No cgroup sandbox | A runaway job can exhaust the container | Rely on the notebook quota |
| Session time limit | Processes stop without warning | Save often. Commit the notebook |
| Output quota | Large checkpoints fail to save | Keep few checkpoints |
| No internet | Bazelisk, npm, and pi fail to install | Preload a Kaggle dataset |
| GPU quota | A weekly GPU-hour limit | Keep the GPU for gates |
| Driver mismatch | The image CUDA and the driver disagree | Check with `tools/nanochat doctor` |
| `rules_cuda` autodetect | The toolkit path is not found | Set `CUDA_HOME` |

## 10. Definition of done

1. `tools/kaggle/bootstrap.sh` prepares a fresh session.
2. `tools/nanochat build` succeeds with the `none` sandbox.
3. `tools/nanochat test` passes.
4. The T1 GPU tests pass in fp32 and fp16.
5. The T2 oracle and train-parity gates pass on sm_75.
6. `pi --print` runs a `tools/nanochat` command.
7. `gh pr create` opens a pull request from the session.
8. `gh pr merge` merges the pull request on the origin.
