# GPU job queue plan

Status: implemented. Owner: Architect. Replaces: the GPU lock in `tools/gpu.sh`.

Stages Q1 to Q5 are in the tree. `python/nanochat_cpp/gpu_queue.py` holds the
store, the worker, and the client. `tools/nanochat gpuq` is the client
interface. `tools/gpu.sh` is gone. Stage Q6 stays optional, and the systemd
unit in section 8 is not shipped: the on-demand detached worker is the path
that works on WSL2 and on a Kaggle Notebook.

This plan removes the GPU lock. It adds one process that owns the GPU and a
queue for the jobs. An agent submits a job to the queue and reads the exit code
and the logs. One worker runs the jobs in order.

## 1. Why the lock goes away

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

## 2. Design summary

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

## 3. Queue store

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

### 3.1 Schema

```sql
CREATE TABLE jobs (
  id        TEXT PRIMARY KEY,      -- also the log file name
  state     TEXT NOT NULL,         -- see the state machine
  command   TEXT NOT NULL,         -- JSON array: the command and its arguments
  directory TEXT NOT NULL,         -- the working directory
  profile   TEXT NOT NULL,         -- the sandbox profile
  env       TEXT NOT NULL,         -- JSON object: the selected variables
  timeout   INTEGER NOT NULL DEFAULT 0,   -- seconds; 0 means no cap
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

### 3.2 Job state machine

```text
queued ──► running ──► done
   │           ├────► failed
   │           └────► cancelled
   └──────────────► cancelled
```

The transitions move forward only. A terminal state never changes. The worker
owns `running`, `done`, and `failed`. A client may cancel a queued job, and it
may request the cancellation of a running job.

### 3.3 Submit

The client inserts one row in one transaction. It sets `submitted` to the
current time in nanoseconds. The row starts in `state='queued'`. The worker
never sees a half-written job, because the transaction is atomic.

### 3.4 One sandbox for every job

Every job runs under the sandbox. The `command` column holds a leaf command,
and the worker always runs it as:

```text
tools/sandbox.sh --profile <profile> -- <command...>
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

This changes two decisions in `docs/script-consolidation-plan.md`: decision 3
(`tools/sandbox.sh` as the only `--run_under` program) and decision 4 (the
frozen `--run_under` string). Update both documents and the contract test
together.

`dist` and `client` are not Bazel actions. Submit the group launcher as one
leaf job. The worker sandboxes the launcher, and the ranks are children of it.

## 4. Worker state management

The job state is durable. The worker state is a lease.

The worker is a Python module, for example `python/nanochat_cpp/gpu_queue.py`.
Python owns orchestration, and it reaches the sandbox through the one allowed
form `tools/sandbox.sh --profile <name> -- <cmd...>`. The module imports only
the standard library.

### 4.1 The lease

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

### 4.2 Device ownership

The worker is the resource manager. It owns the devices, and the client does
not name one.

1. At start, the worker reads the local device list. Use
   `nvidia-smi --query-gpu=index --format=csv,noheader`, or
   `NANOCHAT_GPU_DEVICES` when it is set.
2. The worker stores the pool in `workers.devices`.
3. At claim, the worker assigns the devices to the job and writes
   `jobs.devices`.
4. The worker exports `NANOCHAT_GPU_DEVICES` for the job.
   `tools/sandbox.sh` turns that value into `CUDA_VISIBLE_DEVICES`.

Stage 1 runs one job at a time and assigns the whole pool. A later stage can
run one job per free device subset. The per-device lock files disappear,
because the pool is the allocator.

### 4.3 Worker start

1. Reconcile. Every job in `state='running'` whose worker is not alive becomes
   `failed`, with the note `worker died`.
2. Delete the worker rows that are not alive.
3. Read the device pool.
4. Insert a new worker row with a new identifier.
5. Start the heartbeat loop.

Do not re-queue a recovered job. A training run is not idempotent. The
submitter resubmits when the job is safe to repeat.

### 4.4 Claim

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

### 4.5 Run

1. Read the claimed row.
2. Append one line to `/tmp/nanochat-gpu.log`.
3. Run the `nvidia-smi` gate when `nvidia-smi` is present. Refuse the job when a
   foreign process owns the assigned device, unless `--force` is set.
4. Build the launch command:
   `tools/sandbox.sh --profile <profile> -- <command...>`.
5. Start it with `setsid`, so the job owns a process group. Set the working
   directory to `directory`. Redirect standard output and standard error to
   `<base>/logs/<id>.stdout` and `.stderr`.
6. Wait for the child. Poll the cancellation file and the timeout in the same
   loop.
7. Kill the process group on a cancel request or a timeout.

### 4.6 Finish

Write `state`, `exit_code`, `ended`, and `note` in one transaction. Encode a
signal as a negative `exit_code`, the value Python returns. Then read the next
queued job.

### 4.7 Cancel and stop

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

## 5. Singleton guard without a lock

Two workers must not run at the same time. Use the lease and one atomic
directory create. `mkdir` is atomic on a local filesystem.

1. Read the `workers` table. When a live row exists, reuse that worker.
2. Otherwise `mkdir <base>/worker.starting`. The call fails when another client
   is already starting a worker.
3. Start the worker and wait for the first heartbeat.
4. Remove `<base>/worker.starting`.

This guard is not a GPU lock. It protects one process identity. The optional
systemd unit in section 8 enforces the same rule.

## 6. Client interface

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

`submit` options: `--profile NAME`, `--timeout SECONDS`, `--directory DIR`,
`--submitter NAME`, `--no-wait`, `--json`. There is no `--device` option. The
worker owns the devices and assigns them.

`submit` waits by default. It prints the queue position, follows the logs, and
exits with the job exit code. The agent experience matches a direct call. With
`--no-wait`, it prints the job identifier and exits `0`.

`--json` makes the output machine readable. This option supports a later
orchestration layer.

## 7. Entry point changes

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

`run t0-cpu` stays direct. It needs no GPU. `simulate` stays direct. It is
CPU-only by design.

The worker sets `NANOCHAT_GPU_QUEUED=1` for every job. `dist` and `client`
check that one variable instead of the two guards `NANOCHAT_DIST_BROKERED` and
`NANOCHAT_CLIENT_BROKERED`. This removes the two-name decision.

## 8. Worker lifetime

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

## 9. Observability

- `gpuq list` shows the order, the state, and the submitter of each job.
- `gpuq status` shows the worker identifier, the heartbeat age, the owned
  device pool, the queue depth, and the running job.
- `tools/nanochat doctor --json` replaces the `gpu_lock` key with a `gpu_queue`
  object: `worker_id`, `alive`, `devices`, `depth`, `running_job`, and `queued`.
- `/tmp/nanochat-gpu.log` keeps its path. Each line gains the job identifier.
  Keep the existing start line, and add `queued`, `started`, and `finished`
  lines. Update the contract test in the same commit.

## 10. Limits and safety

- Every job runs under `tools/sandbox.sh`, so no mode is exempt.
- The worker enforces `timeout` and kills the process group.
- The queue directory has mode `0700`. The database and the logs are readable
  only by the user.
- The worker refuses a job when a foreign process owns the device.
- The queue has a maximum length, for example 64 jobs. A full queue refuses a
  new job with a clear message.

## 11. Migration stages

Land one stage per commit. Each stage must pass on its own.

| Stage | Change | Risk | Depends on |
|---|---|---|---|
| Q1 | Add the store, the worker, and `gpuq start/stop/status`. No routing change. | Low | — |
| Q2 | Add `gpuq submit`. Route `gpu`. Keep `tools/gpu.sh` for the other commands. | Low | Q1 |
| Q3 | Route `dist`, `client`, `train`, `rl`, `eval`, `verify`, `bench`, `profile`, and `test --gpu`. | Medium | Q2 |
| Q4 | Delete `tools/gpu.sh` and `--lock-only`. Remove the `gpu_lock` key. Add `gpu_queue`. | Low | Q3 |
| Q5 | Update the documents, the contract test, and `AGENTS.md`. | None | Q4 |
| Q6 | Optional: run one job per free device subset on a multi-card host. | Medium | Q5 |

Stage Q1 is additive. Stage Q2 makes the queue the default for one command.
Stage Q3 moves the remaining commands. Stage Q4 removes the old path.

## 12. Acceptance criteria

Run these after each stage.

1. `tools/nanochat lint`
2. `tools/nanochat test`
3. `tools/nanochat test //tools:prune_test //tools:kaggle_scripts_test`
4. One T1 GPU smoke test through the queue.
5. A queue test with two concurrent submits. The jobs must run in order, one at
   a time. Both exit codes must match.
6. A timeout test. A job with `--timeout 1` and a sleeping command must stop.
7. A cancel test. `gpuq cancel` must stop a running job.
8. A worker restart test. Kill the worker during a job. The next start must mark
   the job `failed` and must not block the queue.

A stage is not done until the default loop passes and the queue tests pass.

## 13. Risks and controls

| Risk | Control |
|---|---|
| The worker dies during a job | The lease goes stale; the next start marks the job `failed` |
| A job hangs forever | Per-job `timeout`, `RuntimeMaxSec`, and `gpuq cancel` |
| Two workers start at once | The `workers` lease plus an atomic directory create, or the systemd unit |
| A stale job blocks the queue | Reconciliation at worker start moves a dead `running` job to `failed` |
| Two jobs fight for the device | The worker owns the pool and assigns devices; one job at a time |
| The database is locked | `busy_timeout=5000`; short transactions; one writer at a time |
| The state is lost at logout | Use `$XDG_STATE_HOME` for the database, not `/tmp` |
| The queue is a single point of failure | On-demand start; the client starts the worker |
| An agent waits with no output | `gpuq status` prints the position and the running job |
| A job builds and then runs | The client takes the build lock before submit |
| The old broker stays in use | Stage Q4 deletes `tools/gpu.sh` |
| The log format changes by accident | Update the contract test in the same commit |

## 14. Decisions

1. Store: SQLite for the records and files for the logs. The fallback is a
   filesystem spool with atomic renames and a reconcile pass.
2. Schema: fifteen columns for `jobs`, five for `workers`. No priority, no
   wrapper, no boot identifier, no duplicate timestamps, no stored log paths.
3. Every job runs under `tools/sandbox.sh`.
4. The worker owns the devices and assigns them. The client does not name one.
5. Submit mode: blocking by default. `gpuq submit` waits and returns the job
   exit code. `--no-wait` prints the job identifier and exits `0`.
6. Scope: the GPU profiles plus `train`, `eval`, `dist`, and `client`.
   `run t0-cpu` and `simulate` stay direct.
7. `/tmp/nanochat-gpu.log`: keep the current line and add fields. Add the job
   identifier and the `queued`, `started`, and `finished` lines. Update the
   contract test in the same commit.

## 15. Boundary with sliceme

The queue belongs to this project. Do not move it into `sliceme`.

Three facts decide the boundary.

1. Sliceme removed GPU management on purpose. `sliceme/ownership.py` and
   `sliceme/sandbox.py` reject a `gpu` field with the message "Sliceme does not
   manage GPUs". `docs/directory-slices-plan.md` section 18 records the removal.
2. The Sliceme database keeps no jobs table and no lease. `docs/database.md`
   records that child execution state belongs to pi-subagents. A queue in that
   database reverses the decision.
3. The Sliceme sandbox is isolation (`bwrap`, `unshare`, a project command). It
   is not a resource budget. The isolation digest feeds the verification
   fingerprint. A cgroup budget does not belong in that fingerprint.

The two mechanisms meet at the project sandbox manifest. Sliceme already lets
the project own test isolation through `sliceme.sandbox.json`. The `command`
field is a prefix, and it receives `/bin/sh -lc "<command>"` as the final
arguments.

Ship this manifest in this project:

```jsonc
{
  "version": 1,
  "command": ["tools/nanochat", "gpuq", "submit", "--profile=t0-cpu", "--"]
}
```

Sliceme then runs its combined-tree checks through the queue, and it never
learns what a GPU is. A CPU-only campaign can point the command at
`tools/sandbox.sh` instead.

The Sliceme worker does not submit jobs. The worker contract tells it not to run
any test. The engine runs the checks. The interactive agent and the ad-hoc
subagents call `tools/nanochat gpuq submit` directly. One mechanism serves all
callers.

## 16. Non-goals

- A change to the sandbox mechanism or to `profiles.conf`.
- A change to the fixture format, the C++ guard, or a frozen header.
- A new third-party dependency. SQLite is in the standard library, so it does
  not count.
- A distributed queue across hosts. One host owns one queue.
- A fair-share scheduler. The order is first-in, first-out.
