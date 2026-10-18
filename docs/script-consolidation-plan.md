# Host script consolidation plan

Status: proposed. Owner: Architect.

This plan changes the host tooling under `tools/**` and one Python test area.
Land one stage per commit. Run the full test loop after each stage. Each stage
must pass on its own, so a later stage can wait.

## 1. Goal

Give each host task one home. Keep the trust boundary in shell. Move the
orchestration to Python. Delete the scripts that no longer earn their place.

Three results are the target:

1. One command table, not two.
2. One fixture generator, not four.
3. One worker process that owns the GPU, not a separate lock script
   ([gpu-queue-plan.md](gpu-queue-plan.md)).

## 2. Current inventory

| File | Lines | Role |
|---|---|---|
| `tools/nanochat` | 757 | Command dispatch, argument assembly, Bazel wiring |
| `tools/sandbox.sh` | 170 | Resource launcher through `systemd-run` |
| `tools/sandbox/verify.sh` | 63 | Limit read-back inside the scope |
| `python/nanochat_cpp/gpu_queue.py` | 700 | The GPU job queue: the store, the worker, and the client |
| `tools/lint.sh` | 71 | Style gate |
| `tools/prune_bazel_output_bases` | 210 | Output base reaper |
| `tools/prune_test.sh` | 115 | Hermetic reaper test |
| `tools/cuda_sim/check_symbols.sh` | 136 | Symbol gate for the API mock |
| `tools/kaggle/bootstrap.sh` | 163 | Kaggle host setup |
| `tools/kaggle/github-setup.sh` | 62 | Kaggle git setup |
| `tools/kaggle/scripts_test.sh` | 37 | Syntax test for the Kaggle scripts |
| `tools/dump_data_fixtures.py` | 781 | Fixture generator: tokenizer, loader, chat |
| `tools/dump_model_fixtures.py` | 939 | Fixture generator: generate, eval, plan |
| `tools/dump_runtime_fixtures.py` | 666 | Fixture generator: api, numerics |
| `tools/dump_training_fixtures.py` | 1067 | Fixture generator: oracle, train, rl |
| `tools/fixture_format.py` | 104 | Record writer |
| `tools/gen_unicode_tables.py` | 267 | Unicode table generator |
| `tools/check_tokenizer_parity.py` | 129 | Tokenizer parity runner |
| `tools/convert_checkpoint.py` | 59 | Forwarder to the Python package |
| `python/nanochat_cpp/_entry.py` | 47 | Python copy of the command table |
| `python/nanochat_cpp/toolchain.py` | 438 | Python copy of the command arguments |

## 3. Duplication in the current tree

- The four fixture generators repeat a preamble and a module-level dispatch.
  The helpers `build_config`, `_as_f32`, `_as_i32`, and `_scalar` appear in
  more than one file with the same body.
- The command list lives in `tools/nanochat` and in `_entry.py`.
- The argument list for each command lives in `tools/nanochat` and in
  `toolchain.py`.
- `tools/sandbox.sh` and `tools/sandbox/verify.sh` are one mechanism in two
  files.
- The lock discipline appears in `tools/nanochat` and in `tools/gpu.sh`.
  [gpu-queue-plan.md](gpu-queue-plan.md) removes both and adds one worker.
- Three documents point at `tools/nanochat_cpp_selftest.sh`,
  `tools/dump_sft_fixture.py`, and `tools/dump_sft_loop_fixture.py`. None of
  these three files exists.

## 4. Locked decisions

Lock these decisions before any code moves.

1. **Python owns the orchestration.** Python holds the command table, the
   dispatch, the argument assembly, the lock, the reaper, the fixture
   generators, `doctor`, and `simulate`.
2. **Shell owns the resource launcher.** `tools/sandbox.sh` keeps the
   `systemd-run` call, the CPU affinity, the rlimits, and the limit read-back.
3. **`tools/sandbox.sh` stays the `--run_under` program for a CPU test.** Bazel
   invokes it once per test action. A GPU test action uses
   `tools/nanochat gpuq submit --profile=<p> --` instead, so the queue worker
   sandboxes the action and owns the device
   ([gpu-queue-plan.md](gpu-queue-plan.md)). Do not put a GPU lock in
   `tools/sandbox.sh`.
4. **Freeze the observable contract.** Keep the environment variable names,
   the two log paths and their line formats, the exit codes, the two
   `--run_under` strings, the `NANOCHAT_SANDBOX` sentinel, the `doctor --json`
   keys, and the fixture binary layouts.
5. **Keep `profiles.conf` as the only profile source.** Do not copy the
   numbers into Python. Let `tools/sandbox.sh` validate the profile name.
6. **Land one stage per commit.** Do not mix a code move with a rewrite.

## 5. The GPU queue decision

The GPU lock is gone. [gpu-queue-plan.md](gpu-queue-plan.md) replaces
`tools/gpu.sh` with one worker process and a queue. The worker is the only
runner, so it serializes the GPU jobs by construction. The worker owns the
devices and assigns them. The `nvidia-smi` gate stays as a pre-flight check.

This supersedes the earlier option to move the lock into the host command
process. The queue gives the wait order and the job record that a lock file
cannot.

The build lock stays, and it keeps the required pattern:

```bash
with_build_lock() { exec 9>"$build_lock"; flock 9; "$@" 9>&-; }
```

The queue replaces `NANOCHAT_DIST_BROKERED` and `NANOCHAT_CLIENT_BROKERED`
with one reentrancy guard, `NANOCHAT_GPU_QUEUED`. The worker sets it for every
job.

## 6. Constraints: do not merge these

- Do not put the GPU queue in `tools/sandbox.sh`. Bazel invokes that script once
  per test action. The queue belongs to the worker.
- Do not drop the `nvidia-smi` gate while another job uses the card. It is a
  check, not a lock, so it stays.
- Do not convert `tools/sandbox.sh` to Python. `systemd-run`, `taskset`,
  `prlimit`, and the `--run_under` string are shell work.
- Do not convert the Kaggle scripts. They install packages and edit `PATH`.
- Keep the `Makefile`, which is the non-Bazel fallback.

## 7. Preparatory changes

Do these before the large moves. Each one lowers the risk of a later stage.

### C1. Add a contract test suite

Pin the current behavior before any code moves. Test these items:

- The command list from `tools/nanochat help` equals `_entry.names()`.
- The key set of `tools/nanochat doctor --json`.
- The `--run_under` string and the exported variables for each profile.
- The line format of `/tmp/nanochat-sandbox.log` and `/tmp/nanochat-gpu.log`.

This is a CPU-only `py_test`. It becomes the net for every later stage.

### C2. Add `--dry-run` to `tools/nanochat`

For `test`, `build`, `simulate`, and `run`, print the exact argument list and
exit 0. This flag gives three results:

- `toolchain.py` can compare its own list against the printed list.
- The language move becomes verifiable.
- Debugging improves on its own.

### C3. Delete `tools/gpu.sh`

Done by stage Q4 of [gpu-queue-plan.md](gpu-queue-plan.md). The queue worker
replaces the lock, the `nvidia-smi` gate, and the `CUDA_VISIBLE_DEVICES`
export. `--lock-only` is gone from the surface. The documents and `AGENTS.md`
are updated in stage Q5.

### C4. Use extract, reroute, delete

Do not move and rewrite in one step. Use three reversible steps:

1. Extract the shared fixtures helpers into `tools/fixture_common.py`.
2. Reroute `tools/dump_fixtures.py` to the shared helpers. Keep the four old
   scripts as thin forwarders.
3. Delete the four forwarders and their file groups.

The same pattern applies to `tools/prune_bazel_output_bases` and to
`tools/cuda_sim/check_symbols.sh`.

### C5. Make the languages meet at one seam

The orchestrator must reach the sandbox through one command form only:

```text
tools/sandbox.sh --profile <name> -- <cmd...>
```

Add a test for this form. Then the orchestrator can change language without a
change to the sandbox. The same rule applies in reverse. The Python layer
reaches the host tool only through `tools/nanochat <subcommand>`.

### C6. Remove the ghost references

Remove the references to `tools/nanochat_cpp_selftest.sh`,
`tools/dump_sft_fixture.py`, and `tools/dump_sft_loop_fixture.py`. A refactor
must not carry dead paths, because they hide the real dependency graph.

### C7. Pin the Python version

The host tool already runs `python3` for `doctor --json`. If Python becomes the
dispatch language, add one version check at the start of `tools/nanochat`.
Record the minimum version in `docs/build.md`.

## 8. Stage order

| Stage | Change | Owner | Language | Risk | Depends on |
|---|---|---|---|---|---|
| A1 | Remove the ghost references | Architect | Docs | None | — |
| A2 | Delete `gpu.sh`, add the queue | Architect | Python + Bash | Low | Section 5 |
| A3 | Add `--dry-run` | Architect | Bash | Low | — |
| B1 | Add the contract test suite | Architect | Python test | None | A3 |
| B2 | Extract and merge the four generators | Oracle | Python | Low | B1 |
| B3 | Move the reaper to Python with a `py_test` | Architect | Python | Low | B1 |
| C1 | Fold `verify.sh` into `sandbox.sh` | Architect | Bash | Low | — |
| D1 | Move the dispatch to Python | Architect | Both | Medium | A3, B1 |
| D2 | Remove the argument copy in `toolchain.py` | Python surface | Python | Medium | D1 |

Stages A and B give most of the cleanup. Stage D is the strategic move, and it
is much safer after A3 and B1.

## 9. Acceptance criteria

Run these after each stage:

1. `tools/nanochat lint`
2. `tools/nanochat test`
3. `tools/nanochat test //tools:prune_test //tools:kaggle_scripts_test`
4. `tools/nanochat simulate --device h100 --suite api` after stage C1 or B3
5. One T1 GPU smoke test after stage D2

A stage is not done until the default test loop passes and no stale path
remains in the documents.

## 10. Risks and controls

| Risk | Control |
|---|---|
| A language move changes the output by accident | C1 contract tests and decision 4 |
| The sandbox takes the queue per test action | Decision 3 |
| The detached Bazel server inherits the lock descriptor | The queue worker replaces the descriptor lock |
| The Python layer and the shell disagree on the arguments | C2 `--dry-run` |
| A large move cannot be rolled back | C4 extract, reroute, delete |
| The profile numbers drift between two files | Decision 5 |
| The host lacks the needed Python | C7 version check |
| The tests fake the entry script and break on the move | C5 single seam, then rewrite those tests inside D1 |
| The fixture generator loses a per-command docstring | Keep each docstring on its command function, and print it from the shared help |
| A heavy import runs at startup | Keep the `torch` and `numpy` imports inside the command function |

## 11. Non-goals

- A single language for every host script. The shell trust boundary stays.
- A change to the C++ guard, the kernel seam, or any frozen header.
- A change to the fixture binary format.
- A new third-party dependency.
