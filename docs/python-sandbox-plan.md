# Python sandbox plan

Status: implemented. Owner: Architect for the documents, Python surface for the
new modules.

This plan moves the resource launcher, the style gate, and the symbol gate from
shell to Python. It absorbs `tools/sandbox.sh`, `tools/sandbox/verify.sh`,
`tools/lint.sh`, and `tools/cuda_sim/check_symbols.sh`. It supersedes decisions
2, 3, 5, 7, and 9 of
[script-consolidation-plan.md](script-consolidation-plan.md), and it removes the
shell trust boundary from the non-goals of that plan.

## 1. Goal

One language for the host tooling. The trust path stays small and auditable.

Five results are the target.

1. One Python program launches, verifies the limits, and execs a job.
2. The style gate is `python/nanochat_cpp/lint.py`.
3. The symbol gate is `tools/cuda_sim/check_symbols.py`, and it reads only
   `global:` blocks.
4. The profile values live in one data file.
5. The frozen contract does not change.

## 2. Decisions

The operator locked three decisions before the work started.

1. **The profile data is JSON.** `tools/sandbox/profiles.conf` becomes
   `tools/sandbox/profiles.json`.
2. **No shell shim keeps a deleted path.** The paths below disappear, and every
   document that names them changes.
3. **The symbol gate reads only `global:` blocks.** A strict parse and the old
   rule both return 57 names for `tools/cuda_sim/cuda_sim.map`, so the gate
   result does not change. The `local:` blocks hold only `*;`.

One path stays. `tools/nanochat` keeps its name, because
`include/nanochat/sandbox.h` prints `run: tools/nanochat help` and about 530
references name it. The file becomes a Python program with a shebang.

## 3. What does not change

These items are frozen by decision 4 of the old plan and by
`python/tests/contract_test.py`. The port preserves every one.

| Item | Value |
|---|---|
| Sentinel, sandboxed run | `NANOCHAT_SANDBOX=<profile>` |
| Sentinel, `none` backend | `NANOCHAT_SANDBOX=off`, plus `NANOCHAT_SANDBOX_PROFILE=<profile>` |
| Backend name | `NANOCHAT_SANDBOX_BACKEND=systemd` or `none` |
| Sandbox log | `/tmp/nanochat-sandbox.log` |
| Verified line | `sandbox: profile=%s mem.max=%s cpu.max=%s pids.max=%s cpus=%s` |
| Unsandboxed line | `sandbox: backend=none profile=%s (not sandboxed)` |
| Launcher interface | `[--profile NAME\|--profile=NAME] [--backend auto\|systemd\|none] [--no-slice] [--] <cmd...>` |
| Quiet switch | `NANOCHAT_SANDBOX_QUIET=1` hides the stderr line, not the log line |
| Device translation | `NANOCHAT_GPU_DEVICES` becomes `CUDA_VISIBLE_DEVICES` when unset |
| GPU test wrapper | `tools/nanochat gpuq submit --profile=t1-gpu --` |
| Symbol gate output | `check_symbols: OK (N required symbols)` and the failure lines |
| Symbol gate exit codes | `0` pass, `1` missing symbol, `2` bad input |
| Style gate output | `lint: OK (N files)`, `lint: FAILED`, exit `1` |

One string changes. The CPU test wrapper becomes
`python/nanochat_cpp/sandbox.py --profile=t0-cpu --`.

## 4. Layout

| Path | Role |
|---|---|
| `python/nanochat_cpp/sandbox.py` | The launcher and the limit verifier. Standard library only. |
| `tools/sandbox/profiles.json` | The profile table. The single source. |
| `tools/sandbox/nanochat.slice` | Unchanged. The aggregate ceiling. |
| `python/nanochat_cpp/lint.py` | The style gate. |
| `tools/cuda_sim/check_symbols.py` | The symbol gate, beside the map and the mock. |
| `tools/nanochat` | A Python program with a shebang. The entry point. |

### 4.1 The launcher keeps two processes

The verifier must stay a separate program. It reads the cgroup of its own
process. A merged program reads the parent cgroup, which sits outside the scope.
Decision 7 of the old plan records this reason, and this port keeps it.

The chain keeps its shape.

```
launch -> systemd-run --user --scope -> taskset -> verify -> exec cmd
```

`taskset` stays external: it sets the affinity mask, and the verifier reads that
mask back, so the verifier must not set it. The core limit uses
`resource.setrlimit` inside the verifier, because no read-back check covers it.
Python builds the argument list for `systemd-run` and `taskset`.

The inner step uses `sys.executable` and the absolute path of `sandbox.py`. The
outer step already checked that interpreter.

### 4.2 The profile data

`tools/sandbox/profiles.json` keeps every value as the exact string from the
earlier `profiles.conf`. Strings remove formatting risk from the cgroup
comparison and from the log line.

```json
{
  "version": 1,
  "slice": "nanochat.slice",
  "profiles": {
    "t0-cpu": {
      "quota": "200%", "mem_high": "1536M", "mem_max": "2G",
      "pids": "128", "cpus": "0-1", "runtime": null
    }
  }
}
```

The file carries eight profiles: `t0-cpu`, `t1-gpu`, `t2-parity`, `t3-bench`,
`train`, `eval`, `kaggle-cpu`, and `kaggle-gpu`.

An unknown name prints `sandbox: unknown profile '<name>'` and exits `1`, the
same as the shell version.

## 5. Stages

### P0 - Unlock the old decisions

Amend [script-consolidation-plan.md](script-consolidation-plan.md). Record the
new decisions and the reason. Mark decisions 2, 3, 5, 7, and 9 as superseded.

### P1 - Profile data and launcher

Add the JSON data and `python/nanochat_cpp/sandbox.py`. Keep the shell launcher
in the tree for one differential check. Add `python/tests/sandbox_test.py`.

### P2 - Switch the call sites

Change `SANDBOX` in `cli.py`, `resolve_sandbox` in `gpu_queue.py`, the two test
suites, and `tools/BUILD.bazel`. Delete the shell launcher and the verifier.

The launcher stops the queue worker first, because a worker from the old code
resolves the deleted path.

### P3 - Style gate

Add `python/nanochat_cpp/lint.py`. Call it in process from `cmd_lint`. Delete
`tools/lint.sh`. Add `python/tests/lint_test.py`.

### P4 - Symbol gate

Add `tools/cuda_sim/check_symbols.py` with the `global:` fix. Delete the shell
version. Add `python/tests/cuda_sim_symbols_test.py`.

### P5 - Entry point

Rewrite `tools/nanochat` as a Python program with a shebang. Delete the Bash
guard. The path and the message it prints stay, except that an interpreter
older than 3.10 now fails with a syntax error.

### P6 - Sweep and record

Update every remaining document. Add the new rows to the section 12 table of the
old plan. Run the final gate.

### P7 - Core limit in Python

The verifier sets `RLIMIT_CORE` with `resource.setrlimit`. The `prlimit` binary
leaves the chain, so one less process runs per test action. The affinity stays
with `taskset`, because the verifier reads that mask back and must not write it.

## 6. Test map

| Test | Tier | What it pins |
|---|---|---|
| `python/tests/sandbox_test.py` | T0 | Argument forms, backend `none` variables, log lines, profile table, limit comparison |
| `python/tests/lint_test.py` | T0 | The four text checks and the exit codes |
| `python/tests/cuda_sim_symbols_test.py` | T0 | The required set, the `global:` rule, the exit codes |
| `python/tests/contract_test.py` | T0 | The frozen strings, the planned layers, the executable bit |
| `python/tests/gpu_queue_test.py` | T0 | The worker resolves the new launcher |
| `tools/nanochat simulate --suite api` | T0 | The real mock, the real gate, and the negative self-test |
| `tools/nanochat test --gpu ...` | T1 | One kernel family through the queue and the real scope |

## 7. Risks and controls

| Risk | Control |
|---|---|
| The launcher is the wrapper for every CPU test, so an error fails the suite | The P1 differential check, then the full T0 run in P2 |
| The frozen log line changes | The contract test compares the exact bytes |
| The `none` sentinel is `off`, not the profile name | The contract test asserts `NANOCHAT_SANDBOX == "off"` |
| Python start-up cost runs once per test action | Measure the delta and record it. Keep imports inside functions |
| The profile data drifts from the slice ceiling | The contract test pins the eight profiles |
| A file loses its executable bit | The contract test asserts the execute permission |
| A stale worker resolves the deleted path | Stop the worker before P2 |
| The `global:` fix changes the required set | Both rules return 57 names, and a unit test pins a `local:` name |

## 8. Definition of Done

For each stage the same checklist applies, from `docs/testing.md`.

1. `tools/nanochat build` passes.
2. `tools/nanochat test` passes.
3. `tools/nanochat lint` passes.
4. The stage gate passes.
5. No stale path remains in the documents.

## 9. What landed

The whole change is in one working tree on `agent/script-consolidation`.

| Stage | Artifact |
|---|---|
| P0 | Section 0 of `docs/script-consolidation-plan.md`. |
| P1 | `tools/sandbox/profiles.json`, `python/nanochat_cpp/sandbox.py`, and `//python:sandbox_test`. |
| P2 | The call sites name the launcher. `tools/sandbox.sh`, `tools/sandbox/verify.sh`, and `tools/sandbox/profiles.conf` are gone. `//tools:sandbox_data`. |
| P3 | `python/nanochat_cpp/lint.py` and `//python:lint_test`. `tools/lint.sh` is gone. |
| P4 | `tools/cuda_sim/check_symbols.py` and `//python:cuda_sim_symbols_test`. The shell gate is gone. |
| P5 | `tools/nanochat` is a Python program with a shebang. |
| P6 | The documents name the new paths. |
| P7 | The verifier sets the core limit with `resource.setrlimit`. `prlimit` leaves the chain. |

The recorded measurements.

| Check | Result |
|---|---|
| Launcher start-up, shell against Python | About 0.03 s each, so no regression |
| The `none` backend, both launchers | Identical environment, stderr line, and log line |
| The error paths, both launchers | Identical text and exit codes |
| The symbol gate, both gates | 57 required symbols, and the same output |
| `tools/nanochat test` | 77 tests pass |
| `tools/nanochat lint` | `lint: OK (213 files)` |
| `tools/nanochat simulate --suite api` | Compile gate and symbol gate pass, including the negative self-test |
| `tools/nanochat test --gpu //backends/cuda/kernels:rms_norm_gpu_test` | Passes through the queue and the new launcher |

One deviation from the plan text is worth a record. Stage P1 did not commit a
differential test case. It ran the differential check once, by hand, against
the shell launcher. The committed suite pins the behavior instead, and the
contract suite runs the two real modes.
