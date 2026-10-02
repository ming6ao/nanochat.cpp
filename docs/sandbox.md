# Execution sandbox and resource governance

Every long-running or agent-launched process — training, eval, GPU tests,
dev-kernel tests, benchmarks — runs through `tools/sandbox.sh`, which applies a
named resource **profile** using cgroup v2. This is an execution concern, not a
compute concern: it never appears in `kernels.h`, `tensor.h`, `config.h`, or
`model.h`, and nothing in `src/` links against it.

## Mechanism (WSL2, unprivileged)

- **Primary:** `systemd-run --user --scope -p <props> --`.
  Gives `MemoryHigh`/`MemoryMax`/`MemorySwapMax`, `CPUQuota`, `TasksMax`, and
  `RuntimeMaxSec`, with correct exit-code and signal propagation. No root, no
  container runtime, no new dependency.
- **CPU pinning:** `taskset` (`sched_setaffinity`). The `cpuset` controller is
  **not** delegated to the user manager on this WSL image, so `AllowedCPUs=` is
  silently ignored and must not be relied on.
- **Hard backstops:** `prlimit --core=0` and an optional `RuntimeMaxSec`.
  `RLIMIT_NPROC` is deliberately left alone — it is per-UID host-wide and would
  break unrelated agent processes; `TasksMax` is the per-cgroup fork-bomb
  guard instead.
- **Optional isolation:** `unshare --user --map-root-user --mount --pid --net
  --mount-proc` for untrusted runs. Verified working unprivileged. GPU profiles
  that use it must bind `/dev/nvidia*` into the mount namespace; GPUs need no
  network, so `--net` is safe there.

## Budget model

The host is 12 vCPU / 7.7 GiB under WSL2. The sandbox budget reserves headroom
for WSL itself, systemd, the Bazel server, and the agent shells: **≤ 8 CPUs and
≤ 6 GiB for all nanochat work combined.**

Two tiers of limit:

1. **Per-job profile** — one cgroup per launch, from `tools/sandbox/profiles.conf`.
2. **Aggregate slice** — `~/.config/systemd/user/nanochat.slice` with its own
   `MemoryMax`/`CPUQuota`, so several concurrent T0 jobs cannot add up past the
   host budget even if each is individually small.

`MemoryHigh` (reclaim) sits below `MemoryMax` (OOM), and `MemorySwapMax=0` keeps
OOM deterministic instead of a slow swap death.

## Install

```bash
mkdir -p ~/.config/systemd/user
cp tools/sandbox/nanochat.slice ~/.config/systemd/user/
systemctl --user daemon-reload
```

If the slice is not installed, launches still work but only get per-profile
limits, with no aggregate ceiling.

## Profiles

| Profile | CPUQuota | Mem high/max | pids | affinity | notes |
|---|---|---|---|---|---|
| `t0-cpu` | 200% | 1.5G / 2G | 128 | 0-1 | default CPU tests, oracle-on-CPU |
| `t1-gpu` | 200% | 2G / 3G | 128 | 0-1 | tiny kernel tests, `B=2,T=8` |
| `t2-parity` | 800% | 5G / 6G | 256 | 0-7 | exclusive, GPU broker held |
| `t3-bench` | 1000% | 5G / 6G | 256 | 0-11 | exclusive, native Linux for Nsight |
| `train` | 800% | 5G / 6G | 512 | 0-7 | `RuntimeMaxSec` optional |
| `eval` | 400% | 3G / 4G | 128 | 0-3 | forward only |

The profile also exports thread-count env (`OMP_NUM_THREADS`,
`NANOCHAT_NUM_THREADS`) matching `CPUQuota`, because cuBLAS and the OpenMP CPU
backend otherwise spawn 12 workers and thrash. The cgroup is the ceiling; the
env is the polite hint. The defaults assume the 12 vCPU / 7.7 GiB host; edit
`tools/sandbox/profiles.conf` for other machines.

## Entry point

Use `tools/nanochat` for every run. It chooses the profile, applies the
sandbox, acquires the GPU broker, and wires the Bazel test wrapper.

```bash
tools/nanochat build                  # bazel build //...
tools/nanochat test                   # all non-GPU tests, each sandboxed
tools/nanochat test --gpu <target>    # GPU tests, broker held for the suite
tools/nanochat check                  # build, then CPU tests
tools/nanochat train -- <binary> ...  # profile=train
tools/nanochat eval  -- <binary> ...  # profile=eval
tools/nanochat bench -- <binary> ...  # profile=t3-bench
tools/nanochat verify -- <binary> ... # profile=t2-parity
tools/nanochat run <profile> -- <cmd> # any profile
```

`tools/nanochat help` lists the commands.

### Lower-level scripts

These are the pieces the entry point uses; call them directly only when you
need to. `tools/gpu.sh` takes the exclusive GPU lock and runs the `nvidia-smi`
gate; `tools/sandbox.sh` applies the cgroup budget. The broker's lock serializes
all heavy jobs, not only GPU ones, because RAM and CPU are also single-host
resources.

```bash
# A single binary under a profile.
tools/sandbox.sh --profile=train -- ./bazel-bin/train/train_main --config d8

# GPU binary: broker acquires the lock, then applies a GPU profile.
tools/gpu.sh --profile=t1-gpu -- ./bazel-bin/.../rms_norm_dev

# Escape the aggregate slice (benchmarks that want the whole host).
tools/sandbox.sh --no-slice --profile=t3-bench -- ./bench
```

## Runtime guard

`tools/sandbox.sh` exports `NANOCHAT_SANDBOX=<profile>`. Executables call
`nanochat::RequireSandboxOrDie(<task>)` at startup; if the sentinel is absent
and `NANOCHAT_ALLOW_UNSANDBOXED` is not set, they print the corrective command
and exit non-zero. This is the guarantee: a binary launched outside the entry
point fails instead of running unbounded. The guard is declared in
`include/nanochat/sandbox.h`.

## Verification

Before running the command, `tools/sandbox/verify.sh` reads `memory.max`,
`cpu.max`, `pids.max`, and the CPU affinity back from the process cgroup and
exits non-zero if they do not match the requested profile. A budget that was
silently not applied is treated as a failure, not as a sandboxed run. Each
launch is logged to `/tmp/nanochat-sandbox.log`.

## Bazel

Test processes are children of the long-lived Bazel server, so wrapping the
`bazel` client does not cap them. `tools/nanochat test` handles this by passing
`--run_under='tools/sandbox.sh --profile=… --'`, which sandboxes each test
action, together with `--spawn_strategy=local`, which is required so the
wrapper can reach the systemd user manager and `/sys/fs/cgroup`.

```bash
tools/nanochat test                 # --test_tag_filters=-gpu, per-test sandbox
tools/nanochat test --gpu <target>  # broker lock held, per-test sandbox
```

Bazel's own filesystem sandbox is disabled for these runs, so the resource
sandbox and the test's own hermeticity are what bound a run.

### Builds

Builds themselves are deliberately not sandboxed: the long-lived server does
not belong in a per-invocation cgroup, and each worktree runs its own server.
They are instead bounded from inside and serialized from outside:

- `.bazelrc` caps the server heap (`--host_jvm_args=-Xmx1500m`), lets an idle
  server exit (`--max_idle_secs=600`), lets the kernel stop it under memory
  pressure (`--shutdown_on_low_sys_mem`), and bounds the local action pool
  (`--jobs=4`, `--local_ram_resources=3072`).
- `tools/nanochat build` and the build phase of `tools/nanochat test` take an
  exclusive lock, so concurrent worktrees queue instead of each running a full
  action pool. The lock descriptor is closed on the Bazel client so the
  detached server cannot inherit it and hold the lock forever.
- `tools/nanochat shutdown` reaps the current workspace's server immediately.

Together these hold the Bazel footprint to roughly one active action pool plus
idle-server heaps, which fits the headroom the budget leaves.

## WSL2 caveats

- `systemd-run` as **root** fails ("Interactive authentication required"); only
  `--user` works. Do not design for system-level cgroups here.
- **cpuset is not delegated** to the user manager in this WSL image, so
  `AllowedCPUs=` is silently ignored. Use `taskset`.
- `MemorySwapMax=0` matters: the 2 GiB swap otherwise turns an OOM into a long
  stall.
- `RLIMIT_NPROC` is intentionally not set: it is per-UID host-wide and would
  break unrelated agent processes. `TasksMax` is the per-cgroup guard.
- The outermost guard is `.wslconfig` (`[wsl2] memory=8GB`, `processors=10`),
  which caps the whole distro independent of any script.
- GPU VRAM is not a cgroup resource. A CUDA OOM still takes the card; VRAM
  budgeting stays with the broker's single-job rule and tiny-shape convention.
- Profile on native Linux: CUPTI is unavailable under WSL2.

## Deliberately not used

No Docker/Podman/bwrap/firejail. The kernel already provides cgroup v2 and user
namespaces, and a container runtime would be a new third-party dependency and
an image to maintain for no isolation we actually need.
