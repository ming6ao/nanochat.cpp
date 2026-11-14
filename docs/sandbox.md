# Execution sandbox, session isolation, and resource governance

Every long-running or agent-launched process — build, test, training, eval,
GPU tests, dev-kernel tests, benchmarks — runs through
`python/nanochat_cpp/sandbox.py`, which applies a named resource **profile**
using cgroup v2. This is an execution concern, not a compute concern: it never
appears in `kernels.h`, `tensor.h`, `config.h`, or `model.h`, and nothing in
`src/` links against it.

The sandbox has two jobs. The first is resource governance: one process tree
cannot starve the host. The second is blast radius: one agent session cannot
reach another session's files, the network, or another session's build. The
first job is in the tree. The second is the **session scope** below, and Part 2.

The sandbox is optional. Set `NANOCHAT_SANDBOX_BACKEND=none` on a host without
cgroup v2, such as a Kaggle Notebook. The entry point then runs the command
directly and keeps the entry-point guard. See
[host-portability.md](host-portability.md) and [host-portability.md](host-portability.md).

## Mechanism (WSL2, unprivileged)

- **Primary:** `systemd-run --user --scope -p <props> --`.
  Gives `MemoryHigh`/`MemoryMax`/`MemorySwapMax`, `CPUQuota`, `TasksMax`, and
  `RuntimeMaxSec`, with correct exit-code and signal propagation. No root, no
  container runtime, no new dependency.
- **CPU pinning:** `taskset` (`sched_setaffinity`). The `cpuset` controller is
  **not** delegated to the user manager on this WSL image, so `AllowedCPUs=` is
  silently ignored and must not be relied on.
- **Hard backstops:** the verifier clears the core limit
  (`resource.RLIMIT_CORE`), and every launch sets `RuntimeMaxSec` from the
  profile wall clock.
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

1. **Per-job profile** — one cgroup per launch, from `tools/sandbox/profiles.json`.
2. **Aggregate slice** — `~/.config/systemd/user/nanochat.slice` with its own
   `MemoryMax`/`CPUQuota`, so several concurrent T0 jobs cannot add up past the
   host budget even if each is individually small.

`MemoryHigh` (reclaim) sits below `MemoryMax` (OOM), and `MemorySwapMax=0` keeps
OOM deterministic instead of a slow swap death.

## Session scope

One **session scope** per agent session is the isolation unit. The record is a
file under `~/.local/state/nanochat/sessions/`, and its slice is a systemd user
slice named `nanochat-agent-<id>.slice`. The slice sits under `nanochat.slice`
by the systemd name hierarchy, so the aggregate ceiling stays in force.

Every command the session starts goes into that slice: the build, the tests,
the GPU job, and the shell. The build goes in because the Bazel server leaves
the client behind and keeps the cgroup of its parent (see `### Builds`). A
session that exhausts memory then loses its own work, and not the host.

A session also owns the network, and one long-lived **supervisor** holds those
namespaces. The supervisor is a child process of nothing and the parent of
everything the session runs, so every command inherits one network namespace.
That namespace is why a denied build is cheap: the Bazel server stays reachable
between commands, and a repeat build costs 0.5 s instead of a server start.

Two limits hold one session. The slice carries the session memory ceiling, so
several commands in one session share it. Each scope inside the slice carries
one profile budget. The verifier writes both and reads both back.

The slice has no wall clock. A Bazel server outlives the client that started it,
so a wall clock would stop the server in the middle of a later build.

Stages, in order. Each stage is useful alone, and each one is a separate change
to `sandbox.py`.

| Stage | Adds | Risk | State |
|---|---|---|---|
| S1 | per-session slice, build read-back | low | in the tree |
| S2 | private network namespace, deny by default | low | in the tree |
| S3 | session supervisor, one namespace, orphan teardown | medium | in the tree |
| S4 | minimal writable set, read-only frozen headers | high | in the tree |

The interface is `tools/nanochat session start|status|stop`. The record, the
slice, the placement check, and the supervisor control live in
`python/nanochat_cpp/session.py`. The supervisor is
`python/nanochat_cpp/supervisor.py`. Part 2 gives the design, the filesystem
contract, and the limits.

## Install

```bash
mkdir -p ~/.config/systemd/user
cp tools/sandbox/nanochat.slice ~/.config/systemd/user/
systemctl --user daemon-reload
```

If the slice is not installed, launches still work but only get per-profile
limits, with no aggregate ceiling.

Install [Bazelisk](https://github.com/bazelbuild/bazelisk) so the Bazel version
pinned in the tree's `.bazelversion` is honored. `tools/nanochat` prefers it and
falls back to a system `bazel`.

## Profiles

| Profile | CPUQuota | Mem high/max | pids | affinity | Wall clock | notes |
|---|---|---|---|---|---|---|
| `t0-cpu` | 200% | 1.5G / 2G | 512 | 0-1 | 3600 s | default CPU tests, oracle-on-CPU |
| `t1-gpu` | 200% | 2G / 3G | 512 | 0-1 | 900 s | tiny kernel tests, `B=2,T=8` |
| `build` | 400% | 4G / 5G | 512 | 0-3 | none | one build inside a session scope |
| `t2-parity` | 800% | 5G / 6G | 256 | 0-7 | 7200 s | exclusive, GPU queue held |
| `t3-bench` | 1000% | 5G / 6G | 256 | 0-11 | 1800 s | exclusive, native Linux for Nsight |
| `train` | 800% | 5G / 6G | 512 | 0-7 | 43200 s | training runs |
| `eval` | 400% | 3G / 4G | 512 | 0-3 | 7200 s | forward only |

`build` is the one profile with no wall clock. A Bazel server outlives the
client that started it, so the server would keep the scope alive past any
backstop, and the next build would die at that moment. Memory bounds that scope
instead. Use it through the session, not by hand; see `## Session scope`.

A profile that hosts a Bazel server needs a large `pids` value. On a cold
output base the Bazel server starts about one hundred `skyframe` threads before
it loads the first target. A `TasksMax` of 128 kills that start with
`pthread_create failed (EAGAIN)`, and Bazel reports it as an out-of-memory
crash. The `t0-cpu`, `t1-gpu`, `build`, and `eval` profiles therefore carry 512
tasks. The aggregate ceiling is the `nanochat.slice` value of 1024.

The profile also exports thread-count env (`OMP_NUM_THREADS`,
`NANOCHAT_NUM_THREADS`) matching `CPUQuota`, because cuBLAS and the OpenMP CPU
backend otherwise spawn 12 workers and thrash. The cgroup is the ceiling; the
env is the polite hint. The defaults assume the 12 vCPU / 7.7 GiB host; edit
`tools/sandbox/profiles.json` for other machines.

Every profile carries a wall clock. The launcher passes `RuntimeMaxSec` at the
profile value plus 60 seconds. `gpuq submit` uses the profile value as the
default job timeout, so one number bounds both layers. The worker kills a long
job at the profile runtime and writes a note. The scope stops only a job that
escaped the worker, so the record wins. `verify` cannot read the wall clock
back, because no cgroup file exposes it. Set the value as a backstop, not as a
policy: a cold tree build plus a suite must still fit inside it.

## Entry point

Use `tools/nanochat` for every run. It chooses the profile, applies the
sandbox, submits the GPU job to the queue, and wires the Bazel test wrapper.

```bash
tools/nanochat build                  # in a session: slice + no network
tools/nanochat build --net host ...   # one build that may fetch
tools/nanochat test                   # all non-GPU tests, each sandboxed
tools/nanochat test --gpu <target>    # GPU tests, one queue job per action
tools/nanochat check                  # build, then CPU tests
tools/nanochat train -- <binary> ...  # profile=train
tools/nanochat eval  -- <binary> ...  # profile=eval
tools/nanochat bench -- <binary> ...  # profile=t3-bench
tools/nanochat verify -- <binary> ... # profile=t2-parity
tools/nanochat run <profile> -- <cmd> # any profile
tools/nanochat gpuq status            # the queue depth and the worker
tools/nanochat session start          # one record and one slice per session
tools/nanochat session status         # slice, memory, scopes, build server
tools/nanochat session stop           # stop the slice and reap the record
```

`tools/nanochat help` lists the commands. The GPU job queue is described in
[this document](sandbox.md) Part 1.

The entry point sits outside the model layers L0 to L6 in
[build.md](build.md#layers-and-the-one-seam), and it changes neither the kernel
seam nor the
public API. The simulator's S0 tier rides the same entry point:
`tools/nanochat test --profile` runs the CPU-only capability, API, collective, and
emulation checks with no GPU broker ([simulator.md](simulator.md)). The one
layering exception that the S0 tier needs is in
[build.md](build.md#dependency-invariants) rule 2.

### Lower-level scripts

These are the pieces the entry point uses; call them directly only when you
need to. `python/nanochat_cpp/sandbox.py` applies the cgroup budget. The GPU worker owns the
device and the jobs, so no separate lock file exists. The queue serializes all
heavy jobs, not only GPU ones, because RAM and CPU are also single-host
resources.

```bash
# A single binary under a profile.
python/nanochat_cpp/sandbox.py --profile=train -- \
    ./bazel-bin/src/nanochat_cli train --config d8

# GPU binary: submit one job to the queue and wait for it.
tools/nanochat gpu --profile=t1-gpu -- ./bazel-bin/.../rms_norm_dev

# Escape the aggregate slice (benchmarks that want the whole host).
python/nanochat_cpp/sandbox.py --no-slice --profile=t3-bench -- ./bench
```

## Runtime guard

`python/nanochat_cpp/sandbox.py` exports `NANOCHAT_SANDBOX=<profile>`, or
`NANOCHAT_SANDBOX=off` on the `none` backend. Executables call
`nanochat::RequireSandboxOrDie(<task>)` at startup; if the sentinel is absent
they print the corrective command and exit non-zero. This is the guarantee: a
binary launched outside the entry point fails instead of running unbounded. The
guard is declared in `include/nanochat/sandbox.h`.

## Verification

Before running the command, `python/nanochat_cpp/sandbox.py verify` reads `memory.max`,
`cpu.max`, `pids.max`, and the CPU affinity back from the process cgroup and
exits non-zero if they do not match the requested profile. A budget that was
silently not applied is treated as a failure, not as a sandboxed run. Each
launch is logged to `/tmp/nanochat-sandbox.log`.

A session launch checks two more claims, and both writes come from the verifier
because systemd cannot set them from outside:

- `memory.oom.group=1` in the scope, so one kill takes the whole compile tree.
- The session slice `memory.max`, so every command in one session shares one
  ceiling.

Both are written and then read back. The verifier also reads the slice
name back from the parent cgroup path, so a scope that landed outside its slice
is a failure.

## Bazel

Test processes are children of the long-lived Bazel server, so wrapping the
`bazel` client does not cap them. `tools/nanochat test` handles this by passing
`--run_under='python/nanochat_cpp/sandbox.py --profile=… --'`, which sandboxes each test
action, together with `--spawn_strategy=local`, which is required so the
wrapper can reach the systemd user manager and `/sys/fs/cgroup`. A GPU test
action uses `--run_under='tools/nanochat gpuq submit --profile=t1-gpu --'`
instead: the worker sandboxes the test and owns the device.

```bash
tools/nanochat test                 # --test_tag_filters=-gpu, per-test sandbox
tools/nanochat test --gpu <target>  # one queue job per test action
```

Bazel's own filesystem sandbox is disabled for these runs, so the resource
sandbox and the test's own hermeticity are what bound a run.

### Builds

A build runs inside the session scope, and the build verifies where the Bazel
server landed. Both parts matter, because the crash on 2026-10-01 came from
the gap between them.

The Bazel client starts a long-lived server and then exits. The server keeps
compiling after the client returns, and no cgroup held that server. Its heap
and its four compile jobs were bounded by flags only, so two worktrees could
exceed the host together. The kernel then killed a process of its own choosing.
It often chose one outside the build.

Three rules close the gap:

- **The scope comes from the session.** A per-invocation scope cannot hold the
  server. Bazel reuses a running server for an output base, so every later
  build compiles inside the cgroup of the first invocation, under that
  invocation's limits and its wall clock.
- **The build verifies the server cgroup.** After a build, `tools/nanochat
  build` reads `/proc/<pid>/cgroup` for the process that `bazel info
  server_pid` names, and fails if the path is not the session slice. An
  unapplied budget is a failure, as for every other launch (`## Verification`).
- **The server is stopped, not left idle.** `systemctl --user stop` on the
  slice stops every command inside it, and `tools/nanochat session stop` runs
  that stop and then reaps the record.

The build command needs no new step. `tools/nanochat build` reads the session
record of its worktree and starts Bazel in the slice, so one `session start`
covers every later build in that worktree:

```bash
tools/nanochat session start           # record, slice, and no command
tools/nanochat build //src:nanochat_cli
```

The build prints where the server landed, and a build that runs anywhere else
fails:

```text
session: nanochat-cpp: bazel server in
  /user.slice/.../nanochat.slice/nanochat-agent-nanochat-cpp.slice/...scope
```

Inside the session the build gets five controls:

| Control | Setting | Effect |
|---|---|---|
| slice `memory.max` | session budget | two commands in one session share one ceiling |
| scope `MemoryMax` | `build` profile | a runaway compile kills this session only |
| `memory.oom.group` | `1` | one kill takes the compile tree, not one action |
| `CPUQuota` | `build` profile | `--jobs=4` gets the session share and no more |
| `TasksMax` | 512 | the server's thread start needs it (`## Profiles`) |

The verifier writes the slice ceiling and the OOM group, then reads both back,
because systemd cannot set a property on a transient slice. A write that does
not take is a failure, like every other unapplied budget.

`.bazelrc` keeps the limits from inside. It caps the server heap
(`--host_jvm_args=-Xmx1500m`), lets an idle server exit
(`--max_idle_secs=600`), lets the kernel stop it under memory pressure
(`--shutdown_on_low_sys_mem`), and bounds the action pool (`--jobs=4`,
`--local_resources=memory=3072`). Those flags are the polite hint. The cgroup
is the ceiling, and the OOM kill now has a small and correct victim.

The build phase of `tools/nanochat test` rides the same boundary, because a
test run compiles before it runs anything. Its test actions keep their own
profile scope through `--run_under`, so a run is bounded twice.

The build lock stays, and it stays outside the session. `tools/nanochat build`
and the build phase of `tools/nanochat test` take an exclusive lock, so
concurrent sessions queue instead of each running a full action pool. The lock
descriptor is closed on the Bazel client, so the detached server cannot inherit
it and hold the lock forever. The lock is the queue for the CPU. The session
slice is the RAM ceiling.

Two changes belong to S4, and Part 2 section 18 gives them. The first is a
per-session `--output_user_root`, so the enclave mounts one directory and a
prune deletes one session. The second is a shared read-only
`--repository_cache`, so a build needs no network.

### Reaping Bazel output bases

Bazel names each output base `md5(workspace_path)` under
`~/.cache/bazel/_bazel_$USER`. Every worktree is a distinct workspace, so it
gets its own output base (roughly 150-310 MB), and that base survives the
worktree. One node accumulates dozens of dead trees and several GB. All of it
streams through the page cache and inflates the WSL2 `Vmmem` reading.
`tools/nanochat
shutdown` stops a server but does not delete its output base.

The coordinator reaps them with one command. It is a dry run by default and only
removes bases whose recorded workspace (`execroot/DO_NOT_BUILD_HERE`) no longer
exists; a base a live server holds is never removed.

```bash
# After a node is landed or closed and its worktree directory is removed:
tools/nanochat prune --worktree <removed-worktree-root> --apply

# After each wave, or whenever memory looks high:
tools/nanochat prune            # review
# tools/nanochat prune --apply  # reap
```

Steady state is an empty `tools/nanochat prune`: if it lists anything, a worktree
was removed without reaping its base. Use `--worktree DIR` to restrict the reap
to bases at or under a just-removed directory, `--sizes` to report bytes, and
`--json` for machine-readable output. The mechanism is
`python/nanochat_cpp/prune.py`; the command wraps it.

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
- **Order matters:** start the cgroup first and the namespaces second.
  `systemd-run --user` reaches the user manager over the session bus, so a
  mount or network namespace hides the bus and the launch fails.
- A user namespace gives no `systemd-run --scope` of its own. Create the scope
  outside, then enter the namespaces inside the scope.
- GPU VRAM is not a cgroup resource. A CUDA OOM still takes the card; VRAM
  budgeting stays with the broker's single-job rule and tiny-shape convention.
- Profile on native Linux: CUPTI is unavailable under WSL2.

### The agent process and the crash dumps

The coding agent starts inside a session scope. Its node heap limit is 2 GiB,
and it reached 1.7 GiB resident on 2026-11-10. The session scope covers the
agent, its shell, and every command that the shell starts:

```bash
tools/nanochat session start --agent pi     # slice, then the agent
```

Without a session, start the agent in a hand scope, so the agent tree still
carries a budget:

```bash
systemd-run --user --scope --quiet --collect \
  -p MemoryHigh=3G -p MemoryMax=4G --description "pi agent" pi
```

A hand scope covers the agent and nothing else. A sandboxed job makes its own
scope under `nanochat.slice`, so the two budgets add. Keep the sum of the high
watermarks at the 8 GiB guest ceiling: 3 GiB for the agent and 5 GiB for
`nanochat.slice`.

The shell startup file defines this as a `pi` function, so a new terminal gets
the bound with no extra step.

WSL writes a crash dump to Windows when a guest process dies from a fatal
signal. The folder is `%LOCALAPPDATA%\Temp\wsl-crashes`. On 2026-11-10 a
`qmd embed` worker aborted eight times, and each abort wrote about 1.8 GB. Eight
dumps filled 14 GB of the host disk. A dump also holds process memory, which can
hold a credential token.

WSL has no switch that turns the capture off. Two settings control it.
`maxCrashDumpCount` sets how many dumps WSL keeps, and the default is 10.
`crashDumpFolder` sets the directory. The `.wslconfig` file therefore sets
`maxCrashDumpCount=1`.

An earlier version of this document used `crashDumpEnabled`. That key does not
exist. WSL 3.0.5 prints `Unknown key 'wsl2.crashDumpEnabled'` and ignores the
line, so the capture stayed at the default retention of 10 dumps.

Capture is on for one fault only, so the next fatal guest signal names the
driver behind the `0x3B` bugcheck. The change takes effect at the next guest
start, because `.wslconfig` is read when the virtual machine boots. Reprocess
one fault, then set `maxCrashDumpCount=0` and delete the folder, because WSL
never reaps it:

```bash
rm -f /mnt/c/Users/*/AppData/Local/Temp/wsl-crashes/*.dmp
```

## Deliberately not used

No Docker, Podman, bubblewrap, or firejail. The kernel already provides
cgroup v2 and user namespaces, and a container runtime would be a new
third-party dependency and an image to maintain for no isolation we need.

A container per session is also the wrong size for this host. One image plus
one Bazel server per session exceeds 7.8 GiB before a compile starts. The card
is below `sm_75`, so no session can own part of the device in any case.

The tree uses the kernel primitives directly: `systemd-run` for the cgroup,
`unshare` for the namespaces, and `taskset` for the affinity. The command
`unshare --user --map-root-user --mount --pid --net --mount-proc` is verified
working unprivileged on this host.

---

## Part 1. The GPU job queue

_Absorbed from the former `gpu-queue-plan.md`. Its section numbers are
unchanged, so a reference to a section of that document resolves to the same
number here._

The queue replaces the GPU lock in `tools/gpu.sh`.

Stages Q1 to Q5 are in the tree. `python/nanochat_cpp/gpu_queue.py` holds the
store, the worker, and the client. `tools/nanochat gpuq` is the client
interface. `tools/gpu.sh` is gone. Stage Q6 stays optional, and the systemd
unit in section 8 is not shipped: the on-demand detached worker is the path
that works on WSL2 and on a Kaggle Notebook.

This plan removes the GPU lock. It adds one process that owns the GPU and a
queue for the jobs. An agent submits a job to the queue and reads the exit code
and the logs. One worker runs the jobs in order.

### 1. Why the lock goes away

`tools/gpu.sh` takes an exclusive `flock` on
`/tmp/nanochat-gpu-<devices>.lock`. The broker shell holds the lock for the run.
A second job blocks on the lock with no status output.

The lock has four problems.

1. The lock is advisory. A process that does not call the entry point ignores
   it.
2. An agent cannot see the order or the wait time. The only trace is
   `/tmp/nanochat-gpu.log`.
3. A short smoke test waits behind a long training run with no explanation.
4. The lock file says nothing about the job. A crash leaves no record.

The queue fixes all four. The worker is the only runner, so it serializes by
construction. No shared lock file is necessary.

### 2. Design summary

- One long-lived worker process owns the GPU.
- A SQLite database holds the job records and the state machine. Plain files
  hold the logs.
- A client command submits a job, starts the worker when necessary, waits, and
  returns the job exit code.
- The worker runs each job through the existing sandbox path, so every resource
  limit stays in force.
- The worker owns the devices and assigns them. The client does not name a
  device.
- The order is first-in, first-out. One job runs at a time.

The lock disappears. The `nvidia-smi` gate stays as a pre-flight check. The
gate is a check, not a lock, so it does not return.

### 3. Queue store

Recommendation: use SQLite for the job records and the state machine, and use
plain files for the logs.

Five reasons support this choice.

1. `sqlite3` is in the Python standard library. It is not a third-party
   dependency, so the non-goal in section 15 stays true.
2. The claim is one transaction. Two workers cannot claim the same job.
3. Recovery after a crash is one query. A torn write cannot corrupt the state.
4. Status is one query. Many clients can read while the worker writes, because
   of write-ahead logging.
5. The store is one file. A backup is one file copy.

Do not store the logs in the database. A log is large, and it streams. A file
supports `tail`, and it avoids bloat. The log paths come from the identifier, so
the database does not store them. The paths are `<base>/logs/<id>.stdout` and
`<base>/logs/<id>.stderr`.

Location: `$XDG_STATE_HOME/nanochat/gpu-queue/`, default
`~/.local/state/nanochat/gpu-queue/`. Use a persistent path, not `/tmp`. The
state then survives a logout. Create the directory with mode `0700`.

```text
<base>/
  queue.db          # the job records and the state machine
  logs/
    <job-id>.stdout
    <job-id>.stderr
  worker.log        # the worker's own log
```

Set these pragmas at open: `journal_mode=WAL`, `busy_timeout=5000`,
`synchronous=NORMAL`, `foreign_keys=ON`.

#### 3.1 Schema

```sql
CREATE TABLE jobs (
  id        TEXT PRIMARY KEY,      -- also the log file name
  state     TEXT NOT NULL,         -- see the state machine
  command   TEXT NOT NULL,         -- JSON array: the command and its arguments
  directory TEXT NOT NULL,         -- the working directory
  profile   TEXT NOT NULL,         -- the sandbox profile
  env       TEXT NOT NULL,         -- JSON object: the selected variables
  timeout   INTEGER NOT NULL DEFAULT 0,   -- seconds; 0 means no cap
                                          -- (submit defaults to the wall clock)
  submitter TEXT,
  submitted INTEGER NOT NULL,      -- nanoseconds; the queue order
  started   INTEGER,
  ended     INTEGER,
  devices   TEXT,                  -- assigned by the worker
  worker    TEXT,                  -- the worker that ran the job
  exit_code INTEGER,
  note      TEXT
);

CREATE INDEX jobs_queue_order ON jobs(state, submitted);

CREATE TABLE workers (
  id        TEXT PRIMARY KEY,      -- the worker identity
  pid       INTEGER NOT NULL,
  devices   TEXT NOT NULL,         -- the pool the worker owns
  started   INTEGER NOT NULL,
  heartbeat INTEGER NOT NULL
);
```

Fifteen columns hold the whole queue. Each one earns its place:

| Column | Why it is necessary |
|---|---|
| `id` | the key, and the log file name |
| `state` | the state machine |
| `command` | the command to run |
| `directory` | the worktree to run in |
| `profile` | the sandbox budget |
| `env` | the test environment of a Bazel action |
| `timeout` | the only wall-clock cap |
| `submitter` | shows who queued the job |
| `submitted` | the queue order |
| `started`, `ended` | the duration in `gpuq status` |
| `devices` | the device assignment |
| `worker` | the crash reconciliation |
| `exit_code` | the job result |
| `note` | the failure reason |

Store only the selected environment variables in `env`. Do not copy the whole
environment. Record `directory`, so the job runs in the worktree of the
submitter.

#### 3.2 Job state machine

```text
queued ──► running ──► done
   │           ├────► failed
   │           └────► cancelled
   └──────────────► cancelled
```

The transitions move forward only. A terminal state never changes. The worker
owns `running`, `done`, and `failed`. A client may cancel a queued job, and it
may request the cancellation of a running job.

#### 3.3 Submit

The client inserts one row in one transaction. It sets `submitted` to the
current time in nanoseconds. The row starts in `state='queued'`. The worker
never sees a half-written job, because the transaction is atomic.

#### 3.4 One sandbox for every job

Every job runs under the sandbox. The `command` column holds a leaf command,
and the worker always runs it as:

```text
python/nanochat_cpp/sandbox.py --profile <profile> -- <command...>
```

There is no second mode. This removes the `wrapper` column, and the worker has
one launch path.

Some current commands are not leaf commands. `tools/nanochat test --gpu` starts
a Bazel client, and the Bazel server is a detached daemon. Wrap that client in
a sandbox scope, and the server inherits the scope. The server then holds the
scope open, and the memory cap applies to the server itself.

So do not submit the whole Bazel invocation. Submit each GPU test action. Set
`--run_under` to the queue submit command:

```text
--run_under=tools/nanochat gpuq submit --profile=t1-gpu --
```

Bazel then appends the test binary and its arguments. Each action becomes one
leaf job, and the worker sandboxes that leaf. The Bazel client and the server
stay outside the sandbox, and the GPU is granted for the test alone.

This changes two decisions in `docs/python.md`: decision 3
(`python/nanochat_cpp/sandbox.py` as the only `--run_under` program) and decision 4 (the
frozen `--run_under` string). Update both documents and the contract test
together.

`dist` and `client` are not Bazel actions. Submit the group launcher as one
leaf job. The worker sandboxes the launcher, and the ranks are children of it.

### 4. Worker state management

The job state is durable. The worker state is a lease.

The worker is a Python module, for example `python/nanochat_cpp/gpu_queue.py`.
Python owns orchestration, and it reaches the sandbox through the one allowed
form `python/nanochat_cpp/sandbox.py --profile <name> -- <cmd...>`. The module imports only
the standard library.

#### 4.1 The lease

The `workers` table holds one row per live worker. The worker writes a
heartbeat every 5 seconds. A worker is alive when the heartbeat is newer than
15 seconds and the process at `pid` is alive.

The plan does not need a boot identifier. An earlier draft used
`/proc/sys/kernel/random/boot_id`, the random value the kernel makes at each
startup. The heartbeat age already covers a reboot, because a reboot takes
longer than 15 seconds. A client that sees a live row but receives no heartbeat
within the timeout also starts a new worker. Drop the column.

The worker holds its phase in memory: `starting`, `idle`, `running`,
`draining`, `stopped`. The table needs only the lease fields. The worker
process needs no durable state.

The lease is also a step-down signal. A worker that cannot write its heartbeat,
or whose row another worker removed, stops after the current job. Without that
rule an orphan polls beside its replacement, and `gpuq status` reports no worker
while two of them run.

#### 4.2 Device ownership

The worker is the resource manager. It owns the devices, and the client does
not name one.

1. At start, the worker reads the local device list. Use
   `nvidia-smi --query-gpu=index --format=csv,noheader`, or
   `NANOCHAT_GPU_DEVICES` when it is set.
2. The worker stores the pool in `workers.devices`.
3. At claim, the worker assigns the devices to the job and writes
   `jobs.devices`.
4. The worker exports `NANOCHAT_GPU_DEVICES` for the job.
   `python/nanochat_cpp/sandbox.py` turns that value into `CUDA_VISIBLE_DEVICES`.

Stage 1 runs one job at a time and assigns the whole pool. A later stage can
run one job per free device subset. The per-device lock files disappear,
because the pool is the allocator.

#### 4.3 Worker start

1. Reconcile. Every job in `state='running'` whose worker is not alive becomes
   `failed`, with the note `worker died`.
2. Delete the worker rows that are not alive.
3. Read the device pool.
4. Insert a new worker row with a new identifier.
5. Start the heartbeat loop.

Do not re-queue a recovered job. A training run is not idempotent. The
submitter resubmits when the job is safe to repeat.

#### 4.4 Claim

The claim is one transaction. `BEGIN IMMEDIATE` serializes the writers.

```sql
BEGIN IMMEDIATE;
UPDATE jobs
   SET state='running', started=?, worker=?, devices=?
 WHERE id = (SELECT id FROM jobs
              WHERE state='queued'
              ORDER BY submitted
              LIMIT 1)
   AND state='queued';
COMMIT;
```

The change count must be `1`. A count of `0` means another worker claimed the
job first. `ORDER BY submitted` gives the first-in, first-out order.

#### 4.5 Run

1. Read the claimed row.
2. Append one line to `/tmp/nanochat-gpu.log`.
3. Run the `nvidia-smi` gate when `nvidia-smi` is present. Refuse the job when a
   foreign process owns the assigned device, unless `--force` is set.
4. Build the launch command:
   `python/nanochat_cpp/sandbox.py --profile <profile> -- <command...>`.
5. Start it with `setsid`, so the job owns a process group. Set the working
   directory to `directory`. Redirect standard output and standard error to
   `<base>/logs/<id>.stdout` and `.stderr`.
6. Wait for the child. Poll the cancellation file and the timeout in the same
   loop.
7. Kill the process group on a cancel request or a timeout.

#### 4.6 Finish

Write `state`, `exit_code`, `ended`, and `note` in one transaction. Encode a
signal as a negative `exit_code`, the value Python returns. Then read the next
queued job.

#### 4.7 Cancel and stop

Cancellation is a request, not job state, so it is not a column.

- `gpuq cancel <id>` creates `<base>/logs/<id>.cancel` for a running job, or
  sets `state='cancelled'` for a queued job.
- The worker polls for the cancel file, kills the process group, and writes
  `state='cancelled'`. It removes the file when the job ends.
- `gpuq stop` sends `SIGTERM` to the worker process from the `workers` row. The
  worker finishes the current job and exits.
- `gpuq stop --now` also cancels the current job.
- An idle worker exits after `idle_sec` (default 1800) and deletes its row.

The worker never takes a GPU lock. It is the only runner.

### 5. Singleton guard without a lock

Two workers must not run at the same time. Use the lease and one atomic
directory create. `mkdir` is atomic on a local filesystem.

1. Read the `workers` table. When a live row exists, reuse that worker.
2. Stop a worker whose process lives but whose lease went stale, then delete
   its row. Such an orphan would poll beside the new worker, so the client
   repairs a split instead of doubling it.
3. Otherwise `mkdir <base>/worker.starting`. The call fails when another client
   is already starting a worker.
4. Start the worker and wait for the first heartbeat.
5. Remove `<base>/worker.starting`.

This guard is not a GPU lock. It protects one process identity. The optional
systemd unit in section 8 enforces the same rule.

### 6. Client interface

Add a subcommand group: `tools/nanochat gpuq <action>`.

| Action | Behavior |
|---|---|
| `submit [opts] -- <cmd...>` | enqueue a job; wait by default |
| `list` | list the queued, running, and finished jobs |
| `status [job]` | show the queue depth and one job |
| `wait <job>` | block until the job finishes; return its exit code |
| `logs <job> [--follow]` | print the job logs |
| `cancel <job>` | request cancellation |
| `start` | start the worker |
| `stop` | stop the worker after the current job |
| `prune [--keep-days N]` | reap job logs older than the retention window |

`submit` options: `--profile NAME`, `--timeout SECONDS`, `--directory DIR`,
`--submitter NAME`, `--no-wait`, `--json`. There is no `--device` option. The
worker owns the devices and assigns them. `--timeout` defaults to the profile
wall clock; `--timeout 0` asks for no cap.

`submit` waits by default. It prints the queue position, follows the logs, and
exits with the job exit code. The agent experience matches a direct call. With
`--no-wait`, it prints the job identifier and exits `0`.

`--json` makes the output machine readable. This option supports a later
orchestration layer.

### 7. Entry point changes

Each existing command keeps its name and its arguments. The implementation
submits a job instead of calling the broker.

| Current command | New behavior |
|---|---|
| `gpu --profile P -- <cmd>` | `gpuq submit --profile P -- <cmd>` |
| `test --gpu <target>` | build under the build lock; submit each action through `--run_under` |
| `dist ... -- <cmd>` | submit the group launcher as one job |
| `client ... -- <cmd>` | submit the client as one job |
| `train`, `rl`, `eval`, `verify` | submit a job with the matching profile |
| `bench -- <cmd>` | submit a job with profile `t3-bench` |
| `profile` | build, then submit the benchmark as one job |
| `run <profile> -- <cmd>` | submit for a GPU profile; run directly for `t0-cpu` |

`run t0-cpu` stays direct. It needs no GPU. `test --profile` stays direct, and
its first step, the device compile, holds the build lock and no queue: the
simulator is CPU-only by design.

The worker sets `NANOCHAT_GPU_QUEUED=1` for every job. `dist` and `client`
check that one variable instead of the two guards `NANOCHAT_DIST_BROKERED` and
`NANOCHAT_CLIENT_BROKERED`. This removes the two-name decision.

### 8. Worker lifetime

The client starts the worker on demand. The primary path is a detached process:
`setsid` plus a redirect of the worker log. This path works on WSL2 and on a
Kaggle Notebook, where systemd is absent.

The workstation also gets an optional systemd user unit,
`nanochat-gpu-queue.service`, with `Restart=on-failure`. The unit is not
required. It gives automatic restart and clean logging. The unit does not take
a lock, because systemd runs one instance.

The worker exits after an idle period, for example 30 minutes with an empty
queue. A later submit starts a new worker. An idle exit keeps the process count
low.

### 9. Observability

- `gpuq list` shows the order, the state, and the submitter of each job.
- `gpuq status` shows the worker identifier, the heartbeat age, the owned
  device pool, the queue depth, and the running job.
- `tools/nanochat doctor --json` replaces the `gpu_lock` key with a `gpu_queue`
  object: `worker_id`, `alive`, `devices`, `depth`, `running_job`, `queued`,
  `log_files`, and `log_bytes`.
- `/tmp/nanochat-gpu.log` keeps its path. Each line gains the job identifier.
  Keep the existing start line, and add `queued`, `started`, and `finished`
  lines. Update the contract test in the same commit.

### 10. Limits and safety

- Every job runs under `python/nanochat_cpp/sandbox.py`, so no mode is exempt.
- The worker enforces `timeout` and kills the process group. A job with no
  explicit timeout inherits the profile wall clock.
- The worker reaps job logs older than 14 days at start, and keeps every job
  record. `gpuq prune --keep-days N` applies the same rule by hand.
- The queue directory has mode `0700`. The database and the logs are readable
  only by the user.
- The worker refuses a job when a foreign process owns the device.
- The queue has a maximum length, for example 64 jobs. A full queue refuses a
  new job with a clear message.

### 14. Decisions

1. Store: SQLite for the records and files for the logs. The fallback is a
   filesystem spool with atomic renames and a reconcile pass.
2. Schema: fifteen columns for `jobs`, five for `workers`. No priority, no
   wrapper, no boot identifier, no duplicate timestamps, no stored log paths.
3. Every job runs under `python/nanochat_cpp/sandbox.py`.
4. The worker owns the devices and assigns them. The client does not name one.
5. Submit mode: blocking by default. `gpuq submit` waits and returns the job
   exit code. `--no-wait` prints the job identifier and exits `0`.
6. Scope: the GPU profiles plus `train`, `eval`, `dist`, and `client`.
   `run t0-cpu` and `test --profile` stay direct.
7. `/tmp/nanochat-gpu.log`: keep the current line and add fields. Add the job
   identifier and the `queued`, `started`, and `finished` lines. Update the
   contract test in the same commit.
8. Wall clock: every profile carries a `runtime`. The scope receives it plus
   60 seconds, and `gpuq submit` uses it as the default job timeout. The value
   is a backstop, so a cold build plus a suite must still fit inside it.
9. Retention: the worker reaps job logs older than 14 days at start, and
   `gpuq prune --keep-days N` applies the same rule by hand. A job record is
   never reaped, because a row is a few hundred bytes and it is the history.
10. The lease is a step-down signal. A worker whose lease write fails, or
    whose row another worker removed, stops after the current job. A client
    retires a stale-lease orphan before it starts a replacement.

### 15. Non-goals

- A change to the sandbox mechanism or to `profiles.json`.
- A change to the fixture format, the C++ guard, or a frozen header.
- A new third-party dependency. SQLite is in the standard library, so it does
  not count.
- A distributed queue across hosts. One host owns one queue.
- A fair-share scheduler. The order is first-in, first-out.

---

## Part 2. The session enclave

_A design record. Stages S1 and S2 are in the tree: the record, the slice, the
build routing, the private network namespace, and the two read-backs. Stages S3
and S4 are planned. Part 1 stays as it is: the GPU queue is unchanged and moves
nothing into a session._

### 16. The problem

An agent session runs as the same user as every other session. It can read,
write, and delete any file that the user owns. It can write to `main`, to
another session's worktree, and to a frozen header. It has the whole network.
Its children live on after the step that started them, in no cgroup at all.

That shape caused three incidents:

| Date | Symptom | Cause |
|---|---|---|
| 2026-10-01 | host memory exhausted | Bazel servers outside every cgroup |
| 2026-11-10 | agent reached 1.7 GiB | no scope on the agent tree |
| 2026-11-10 | eight crash dumps, 14 GB | a fatal signal in an unsupervised child |

The ownership map in `AGENTS.md` section 1 is a rule about files. Nothing
enforces it. A session scope turns three of those rules into enforced rules:
the worktree, the network, and the process lifetime.

### 17. The boundary

The supervisor is the **parent** of everything the session runs. Containment
comes from process inheritance, so no command creates a namespace and no
command is wrapped. "Every command is sandboxed" is a fact of process
parentage.

This is the difference from Claude Code and from Codex. Both start one sandbox
per command, under a policy. That shape needs a human approval step, because a
command that finds a path around the wrapper is unconstrained. A session with
no human in the loop needs the enclosing shape.

The nesting order is fixed, and it is what the supervisor itself starts in:

1. `systemd-run --user --unit=nanochat-supervisor-<id> --slice=<session slice>`
   with the profile budget.
2. `unshare --user --map-root-user --mount --net`.
3. The supervisor process, then its commands as children.

The cgroup is outermost because `systemd-run --user` needs the session bus, and
a network or mount namespace hides it. A user namespace gives no scope of its
own, so the scope cannot come from inside.

**There is no process namespace.** That was the plan and the measurement
rejected it. A process namespace makes `systemd-run --user --scope` fail with
`Failed to connect to bus`, and every test action needs that scope for its own
cgroup. Measured on 2026-11-11: a user and mount namespace keeps the bus
working, and adding a process namespace breaks it.

| Namespaces | `systemd-run --user --scope` |
|---|---|
| user + mount | works |
| user + net | works |
| user + mount + net | works |
| user + pid | **fails** |

Orphan teardown comes from the slice instead, and it already did the job. A
process cannot leave a cgroup without privilege, so a `setsid` child is still
in the slice, and `systemctl --user stop <slice>` ends it. That was measured:
a detached child survived a command, then died with the slice.

### Why the supervisor holds one namespace

A network namespace created inside a user namespace cannot be joined from
outside: `nsenter --net` answers `Operation not permitted`, because the caller
is not root in the namespace that owns it. So a namespace per command cannot
share its loopback, and Bazel's client and server need one loopback.

The supervisor therefore holds the namespace for the life of the session and
runs each command as a child. The Bazel server started by the first build stays
reachable by the second, which is what makes a denied build cheap. Measured:

| Run | Wall clock |
|---|---|
| deny, namespace per command (S2) | 21 s, server restart every time |
| deny, supervisor namespace (S3) | **0.5 s** for a repeat build |

### 18. The writable set (S4)

The supervisor gets a mount namespace, so it decides what the session can
change. One fact decides the plan: **the user namespace already protects the
system.** Root inside it is the host user, and the host user does not own
`/usr` or `/etc`, so a session cannot write there with or without a mount. A
probe agreed: `touch /usr/local/x` inside the namespace fails with
`Permission denied`. Remounting `/usr` read-only would add a claim and no
protection, and on this host the kernel refuses that bind anyway.

What the session can reach is the home directory. The plan protects that:

| Path | Mode | Why |
|---|---|---|
| the worktree's parent | `ro` | a sibling worktree is one accidental `rm` away |
| the worktree | `rw` | the one place a session may change |
| `include/nanochat` | `ro` | the frozen headers, unless `--architect` |
| `~/.ssh`, `~/.aws`, `~/.config/gh`, `~/.netrc` | hidden | a build never needs a credential |

A bind mount inherits the read-only flag of its source mount, so the worktree
needs `mount --bind` and then `-o remount,bind,rw`. Without the second step the
worktree stays read-only and every build fails.

Every claim is read back before the supervisor serves its first command. The
test tries the operation itself. It creates a file in each read-only path and
expects `EROFS`, lists each hidden directory and expects nothing, and writes a
probe into the worktree and expects success. A claim that does not hold stops
`session start` and names the path. The ready file carries the claims, so
`session status` prints them.

`--architect` keeps `include/nanochat` writable, because that workstream owns
those files. The reachable set is the same otherwise.

### 19. The network (S2)

Deny by default. A build in a session runs in a private network namespace with
one interface, `lo`, and no route. A fetch at build time is a hidden input, and
the oracle fixtures are committed data.

The mechanism is three facts, and each one was measured on this host:

| Fact | Consequence |
|---|---|
| `unshare --user --map-root-user --net` needs no privilege | the namespace is available on this host |
| a new namespace has `lo` **down**, so loopback fails | `netprep` runs `ip link set lo up` first |
| the client and server talk over loopback TCP | they must share one namespace |

The third fact set the shape. Bazel needs loopback, so the user namespace stays
and `lo` comes up. A `unshare --user --map-user=<uid>` form keeps the user
identifier but loses `CAP_NET_ADMIN`, so `ip link` fails; `--map-root-user` is
the form that works. Bazel then runs as root inside its own user namespace.
That is invisible to Bazel, and file ownership still maps to the same host user.

`netprep` refuses to continue when any interface besides `lo` exists. That is
the read-back: a namespace with a route must not run a build that claims to be
hermetic. `session status` reads the same fact from the server's
`/proc/<pid>/ns/net`, so a server that kept the host network fails the check.

A denied build runs with no network and still succeeds in about 4.5 seconds on
a warm output base, because every external repository is already in it. The
cost is one Bazel server start per denied run: the next launch enters a
different namespace, so the previous server is unreachable and Bazel replaces
it. `tools/nanochat build` stops it first, so the replacement is not a race.

That cost was measured on the full CPU suite on 2026-11-11. The suite is 121
targets, and `dev/cutlass` is left out because it is a non-test probe that
`bazel test //...` still builds.

| Run | Wall clock | Server |
|---|---|---|
| no session, server alive | 0.6 s to 2.4 s | reused |
| no session, server restarted | 40 s to 48 s | cold |
| `test` in a session, deny | 21 s to 22 s | restarted every run |
| `test` in a session, host network | 2.8 s | reused |

So a namespace per command costs about 20 s per invocation, and the reason is
the server start, not the namespace. That measurement is why S3 exists. With a
supervisor the namespace outlives the command, and the same denied run costs
what a normal run costs:

| Run | Wall clock |
|---|---|
| `test`, no session | 0.6 s to 2.4 s |
| `test`, session, supervisor, deny | **2.8 s** |
| `test`, session, no supervisor, deny | 21 s |

Both `build` and `test` deny by default in a session, and the deny costs
nothing. A session without a supervisor falls back to the host network and says
so, because a namespace that lives for one command pays that 21 s every time.

Deny has exactly one meaning: the supervisor's namespace. A `--net deny`
invocation with no supervisor is no longer a path, so there is nothing to keep
in step. `--net host` means "run this command outside the supervisor, on the
host network", and it costs one server start, because the server is then in the
wrong namespace.

The measured peak of the whole CPU suite is 3.22 GiB against the 4 GiB `build`
profile, so the ceiling holds for the suite and not for a CUDA suite. A build
with `--config=cuda` or a CUTLASS target needs a session with a larger profile,
or a hand scope.

Two things need the network, and each has one explicit path:

- **The first fetch on a fresh tree.** `tools/nanochat build --net host ...`,
  or `session start --net host` for a session that always fetches. A build that
  fails without the network prints that line.
- **The Python surface and Hugging Face.** The same one flag, and the sandbox
  log records the mode of every launch.

The GPU queue needs no network. It needs the device.

A proxy with an allowlist is the step after S2. The namespace is what makes it
an enforcement boundary instead of advice.

### 20. The build goes first

`### Builds` gives the mechanism. This section gives the reason the build is
stage S1.

The build is the only step that can take the whole host down. It is the
largest memory consumer, it runs for minutes, and it leaves a detached process
behind. The GPU job is the second largest, and the queue serializes it already.
So the build is the first thing to put in a boundary.

Three rules hold:

- The build lock stays outside the session, because the CPU is shared.
- The session slice caps one session, and `nanochat.slice` caps all sessions.
- `--jobs=4` stays the per-server hint. Do not raise it for a session.

The read-back is the gate. `tools/nanochat build` fails when the server is not
in the session slice, because a build that ran somewhere else is the exact
failure of 2026-10-01. That failure is now a stopped build with a message
instead of a stopped host.

The measured cost of S1 is one cold rebuild per worktree, because the server
starts in a new cgroup. After that the tree is warm and nothing changes.

### 21. The GPU stays brokered

The session must not hold `/dev/nvidia*`, and it must not own the device. It
submits a job to `gpuq`. The worker runs outside every session, and it runs the
job in a profile scope.

This is the one place where a session stays less privileged than the host, and
it is the main design error to avoid. Two namespaces cannot share one device
safely here, and the card is below `sm_75` in any case. One device, one owner,
one record in `/tmp/nanochat-gpu.log`.

### 22. Lifecycle and observability

`tools/nanochat session` is the whole interface:

```bash
tools/nanochat session start --agent pi     # record, slice, supervisor
tools/nanochat session start --net host     # record only, no namespace
tools/nanochat session start --architect    # keep the frozen headers writable
tools/nanochat session status [--json]      # slice, supervisor, claims, server
tools/nanochat session stop                 # supervisor, slice, record
```

- The record is the session. `--id NAME` names it, and the default is the
  worktree directory name, so `nanochat.cpp` becomes `nanochat-cpp`.
- `session start` with `net=deny` starts the supervisor as a transient service
  in the session slice, then waits for its ready file. The ready file holds the
  process identifier and the user, mount, and network namespace identifiers.
  `status` reads them back, so "the supervisor is up" is a fact and not an
  assumption.
- `status` reports `state=idle` when no command runs in the slice. It fails only
  when no record exists, and then it names the command that creates one.
- `status` prints the slice ceiling, one line per command scope, the supervisor
  and its network namespace, one line per writable-set claim, and the build
  server with the namespace it is in.
- `stop` stops the supervisor unit first, then the slice, then reaps the record
  and the socket and ready files. The slice stop is the teardown: a process in
  it dies even when it called `setsid`.
- `/tmp/nanochat-sandbox.log` gains one line per session launch:
  `sandbox: session=<id> slice=<slice> cgroup=<path>`, and one line per denied
  namespace: `sandbox: net=deny netns=<inode> interfaces=lo`.

The supervisor speaks newline-delimited JSON over a Unix socket in the session
state directory. A request names `argv`, `cwd`, and `env`. A reply is a
sequence of `{"stream": "out"|"err", "data": ...}` frames and one
`{"exit": N}` frame. Output streams while the command runs, so a long build
shows progress. One command at a time, because the build lock already
serializes Bazel.

### 23. Migration

1. Done: `tools/nanochat shutdown`, then `tools/nanochat session start`. The
   first build after that reports the server inside the slice.
2. Done: S1, the record, the slice, and the placement read-back.
3. Done: S2, the private network namespace, with `netprep` as its read-back.
4. Done: S3, the supervisor that holds one namespace and runs commands as
   children, with orphan teardown by the slice.
5. Done: S4, the writable set, the hidden credentials, and the read-only
   frozen headers, with every claim read back before the first command.
6. Add `session stop` beside `shutdown` and `prune` in the worktree retirement
   step.

A host without cgroup v2 keeps `NANOCHAT_SANDBOX_BACKEND=none`, and the session
is then off: `session start` refuses, and a command skips the routing. A test
that checks the routing must set `NANOCHAT_SESSION_DIR` to a temporary
directory, so a real session in the worktree cannot reach it.

A tree whose output base holds no external repository needs one run with
`--net host` first. After that the deny rule costs nothing.

### 24. Decisions

1. The session is the isolation unit: one session, one record, one slice, one
   supervisor, one log line.
2. The record is the session, not the slice. A transient slice disappears when
   it is empty, so the record on disk is what a later command reads.
3. The supervisor is the parent of every command. Do not wrap single commands.
4. The cgroup is outermost, and the namespaces are inside it.
5. **No process namespace.** It breaks the systemd bus, and the test tier needs
   that bus for its per-action scope. Orphan teardown comes from the slice.
6. One namespace per session, not one per command, because a namespace created
   inside a user namespace cannot be joined from outside.
7. The supervisor is a transient service in the session slice, so its memory is
   inside the session ceiling and the slice stop kills it.
8. The supervisor's ready file carries the namespace identifiers, and `status`
   reads them back. "The supervisor is up" is a measurement.
9. The build is the first step in the boundary, because it is the largest
   consumer and it detaches.
10. The build verifies the server cgroup and the server namespace. An unapplied
    budget is a failure.
11. The verifier writes the slice ceiling and the OOM group, because systemd
    cannot set a property on a transient slice. It reads each one back.
12. The `build` profile has no wall clock, because the server outlives the
    client. Memory bounds that scope instead.
13. The GPU is brokered, and it stays outside every session.
14. The network is denied by default, and `--net host` is one visible opt-in.
15. The server identifier comes from `<output_base>/server/server.pid.txt`. A
    `bazel info` call from the host would start a host server, which is the
    state the read-back must detect.
16. A server that cannot serve this command is stopped before the command, so
    the read-back confirms a placement instead of rescuing one.
17. The user namespace is the system protection. `/usr` and `/etc` need no
    mount claim, because root inside the namespace is the host user.
18. The writable set is relative to the worktree, so a missing worktree is an
    error and not an empty plan.
19. A credential directory is hidden, because a build never needs one.
20. `--architect` is the only way to keep the frozen headers writable.
21. Kernel primitives only. No container runtime and no bubblewrap.
22. `unshare`, `systemd-run`, and the supervisor stay wrapped by
    `python/nanochat_cpp/`. `tools/nanochat` gains subcommands, not scripts.

### 25. Non-goals

- A change to the GPU queue, the existing profile budgets, or the fixture
  format. S1 adds one profile and changes no other.
- A change to a frozen header, to the C++ guard, or to `kernels.h`.
- A container image or an environment per session.
- A proxy with a domain allowlist. That comes after S2 has run for a while.
- A virtual machine per session. The host cannot hold them.
- A process namespace. The measurement in section 17 rejects it.
- A new third-party dependency.

### 26. Limits

- The kernel is shared. A namespace is not a defence against a kernel defect.
- A bind mount shares storage. A write inside the worktree is a write on the
  host disk.
- A user namespace limits no CPU time. Only the cgroup does, so keep both.
- The private network namespace has loopback up, by design, because Bazel needs
  it. A denied build can reach a service that listens on the host's loopback
  only through a shared namespace, and it does not share one.
- The deny rule costs one server start per run, about 20 s. That is why `test`
  uses the host network by default, and why a session that runs many small
  builds is faster with `session start --net host`.
- The whole CPU suite peaks at 3.22 GiB. The `build` profile carries 4 GiB high
  and 5 GiB max, which holds one 3.0 GiB device compile beside the Bazel server
  and stays under the 6 GiB aggregate. Two such compiles do not fit at once, so a
  CUTLASS target builds with `--jobs=1` (`dev/cutlass/README.md` section 7).
- The read-back proves placement, not enforcement. `verify` still reads the
  limits back from the cgroup.
- The writable set does not stop a session from reading a file that stays
  readable. It protects writes, and it hides the credential paths it names; a
  token in another file is still readable.
- The set covers the home directory. The rest of the system is protected by
  ownership, not by a mount, so a future writable system path needs a new
  claim.
