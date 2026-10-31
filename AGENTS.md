# AGENTS.md — Multi-agent coordination

How several coding agents work on `nanochat.cpp` in parallel without conflicting
edits, and how they share the single local GPU for tests.

See `DESIGN.md` for the architecture. This file is about process.

> **Run everything through `tools/nanochat`.** Do not call `bazel test`,
> `train_main`, `eval_main`, or any `*_test`/`*_dev` binary directly. The entry
> point applies the resource sandbox, the GPU job queue, and the Bazel test
> wrapper for you. Run `tools/nanochat help`; see [docs/sandbox.md](docs/sandbox.md)
> and [docs/testing.md](docs/testing.md). Executables refuse to start outside it.

The host: one **GTX 1080 Ti (sm_61, 11 GB)** under **WSL2**. The driver is
536.99, and CUDA is 12.0. The project pins Bazel 9.2.0 in `.bazelversion` and
runs it through Bazelisk. The host has no cmake, ninja, or clang. CUPTI is **not**
available under
WSL2, so Nsight/perf profiling must happen on native Linux.

---

## 1. Ownership map: one directory, one owner

Each agent owns a directory and everything under it. **No two agents edit the
same file.** Ownership is the conflict-avoidance mechanism; reviews and merges
happen through the integrator.

| Workstream | Owns | Depends on |
|---|---|---|
| **Architect / Integrator** | `include/nanochat/*.h`, `MODULE.bazel`, `.bazelrc`, top-level `BUILD` files, `tools/nanochat`, `python/nanochat_cpp/cli.py`, `python/nanochat_cpp/_entry.py`, `README.md`, `DESIGN.md`, `AGENTS.md`, `CONTRIBUTING.md`, `docs/**` | — |
| **Contract** (the architect) | `spec/**`, the frozen headers below, and this document | — |
| **Model owners** (one per model) | `models/<name>/**` | `spec/**`, the frozen headers |
| **Runtime** | `include/nanochat/tensor.h`, `src/tensor.cc`, `backends/cpu/` (`caps.json` is the device's single declaration; no header row is added per device) | frozen `kernels.h` |
| **Oracle** | `tools/dump_*.py`, `tests/`, `tests/data/*.bin` | frozen `tensor.h` |
| **Kernel agents** (one per family) | `backends/cuda/kernels/<family>.cu` + `<family>_test.cc` + `<family>_benchmark.cc` in the same directory | frozen `kernels.h` |
| **Kernel agents** (shared) | `backends/cuda/kernels/testing/**` (host test scaffolding), `dev/kernels/**` (unpromoted prototypes) | — |
| **Collective** | `backends/cuda/collective/**`, `dev/collective/**` | frozen `kernels.h`, `src/rendezvous.h` |
| **Workflow** | `src/ops.cc`, `src/model.cc`, `src/generate.cc` (the rollout seed mix, B4 of `docs/pipeline-plan.md`) | `kernels.h`, CPU backend |
| **Optimizer** | `src/optim.cc` | `kernels.h` |
| **Harness** | `src/cli.h`, `src/train.cc`, `src/data.cc`, `src/eval.cc`, `src/*_main.cc` (`src/train_main.cc` among them), `src/device_profile.cc`, `include/nanochat/{data,scheduler,logger,mfu}.h` | Model API |
| **Python surface** | `python/**` except `cli.py` and `_entry.py` (`pipeline.py` and `tests/pipeline_test.py` are here), `bindings/**`, `tools/modal/app.py`, `tools/modal/modal_app_test.py` | C ABI `capi.h`, Model API |
| **Data pipeline** | `src/tokenizer/**`, `src/parquet/**`, `include/nanochat/{tokenizer,bpe_trainer}.h` | frozen `data.h` |
| **Notebooks** | `notebooks/**`, `tools/kaggle/**` | `docs/host-portability.md` |
| **Examples** | `examples/**`, one package per directory: `examples/hello_world/**` (inference) and `examples/checkpoint_roundtrip/**` (the container) | public `include/nanochat/**` and `//src:model`; no vendor header, no `backends/` include (`docs/examples.md`) |
| **Simulator** | `tools/cuda_sim/**` (the API interposer, the emulation engine, the collective mock, `cuda_sim.map`, `check_symbols.py`) | `docs/simulator.md`, frozen `tensor.h` |
| **Build** | `MODULE.bazel`, `.bazelrc`, `BUILD.bazel` (setup only) | Architect |

The one-job chain of [docs/pipeline-plan.md](docs/pipeline-plan.md) adds no
workstream, because decision D9 gives each of its files an owner that already
exists. `python/nanochat_cpp/pipeline.py` and `python/tests/pipeline_test.py`
are the Python surface's, as are `tools/modal/app.py` and
`tools/modal/modal_app_test.py`. `src/cli.h` is the Harness's, beside
`src/train_main.cc`, so one workstream owns the flag and the binary that reads
it. `src/generate.cc` is the Workflow's. A per-directory `BUILD.bazel` still
belongs to the directory's owner (`AGENTS.md` §6), so `python/BUILD.bazel` is
the Python surface's and `src/BUILD.bazel` is the owning workstream's; the
two stage binaries are declared there and the chain's two tests in
`python/BUILD.bazel`, so the root `BUILD.bazel` gains nothing.

**Frozen interfaces** (contract-owned: the architect; coordinate before
editing):

- `include/nanochat/kernels.h` — the backend seam.
- `include/nanochat/device_profile.h` — the simulator's capability table
  (additive; it stays beside `PeakFlopsForDevice`).
- `include/nanochat/tensor.h` — `Tensor`, `ComputeType`, `DType`, `Caps`.
- `include/nanochat/config.h` — `Config`.
- `include/nanochat/model.h` — the public Model API.
- `include/nanochat/capi.h` — the C ABI for the in-process Python API.
- `include/nanochat/plan.h` — the pure `BuildPlan` and its `PlanRequest`
  (contract-owned).
- `include/nanochat/model_spec.h` — the closed `ModelSpec` description
  (contract-owned).

A frozen header may still gain symbols during Wave 0. After the freeze, edits
stay architect-owned, and additive changes are preferred.

A bring-up may not edit a frozen header, a shared build file, or another
package without a contract extension first; that is the rejection list in §13
of [docs/model-and-device-bringup.md](docs/model-and-device-bringup.md).

---

## 2. Branch and worktree model

- Each agent works in its **own git worktree and branch**: `agent/<workstream>`.
  No agent commits to `main`.
- `main` is integrated by the architect only, from the merge queue.
- Rebase on `main` before requesting a merge; the integrator resolves the rest.

```bash
git worktree add ../nanochat.cpp.<ws> -b agent/<ws>
```

---

## 3. Sequencing: freeze first, then fan out

Parallelism is gated on interface readiness, not on agent availability.

**Wave 0 (serial, blocking) — do not parallelize before this**

1. `MODULE.bazel` + `.bazelrc` + Bazel/CUDA spike (`sm_61`, one kernel, one
   `gpu` test). Fall back to a Makefile if `rules_cuda` blocks.
2. Freeze `kernels.h`, `tensor.h`, `config.h`, `model.h`.
3. CPU reference backend: every kernel as a naive loop.
4. `tools/dump_fixtures.py oracle` + `tests/data/debug_state.bin` fixtures.
5. `Model` skeleton on the CPU backend (forward + backward end to end).

**Wave 1 (parallel).** Kernel agents each own one family: the implementation,
its finite-difference test in the same directory, and the oracle fixture test.

**Wave 2 (parallel).** Workflow, optimizer, harness — unblocked because the CPU
backend from Wave 0 already runs the full graph.

**Wave 3 (serial).** Oracle parity (fwd -> bwd -> loss curve), Turing port,
fusion/tuning.

The CPU backend is what makes Wave 2 parallel with Wave 1. Never let the GPU be
on the critical path for correctness iteration.

---

## 4. Definition of Done (per workstream)

The canonical checklist — build, CPU tests, small-shape GPU correctness,
finite-difference checks, oracle parity, and formatting — lives in
[docs/testing.md](docs/testing.md). Scheduling of the GPU tiers is in §5.

---

## 5. Coordinating the single local GPU via the sandbox gateway

There is **one GPU** and **one host**. Two failure modes are worth avoiding.
The first is several agents launching training jobs or benchmarks at once. That
thrashes the 11 GB card and invalidates timings. The second is a runaway test
that exhausts host RAM or CPU.

The process-launch boundary handles both, through `python/nanochat_cpp/sandbox.py` (limits) and
the GPU job queue (one worker owns the GPU). See
[docs/gpu-queue-plan.md](docs/gpu-queue-plan.md) and
[docs/sandbox.md](docs/sandbox.md).

### 5.1 Tier the tests

Test tiers are defined in [docs/testing.md](docs/testing.md). The scheduling
rule here is: T0 runs in parallel under the `t0-cpu` profile; T1/T2/T3 are
serialized through the queue and exclusive; T3 runs on native Linux.

Rule: **the inner development loop is T0.** GPU tiers are gates, not iteration
tools. If an agent finds itself waiting on the GPU to test logic, it should have
been testing on CPU.

### 5.2 The sandbox gateway

The entry point is **`tools/nanochat`** (`tools/nanochat help`): it picks the
profile, applies the sandbox, submits the GPU job, and builds the correct Bazel
test command. Use it for every run:

```bash
tools/nanochat build                 # bazel build //...
tools/nanochat test                  # all non-GPU tests, each sandboxed
tools/nanochat test --gpu <target>   # GPU tests, one queue job per action
tools/nanochat lint                  # Google C++ Style gate (clang-format + checks)
tools/nanochat train -- <binary> ...  # profile=train
tools/nanochat eval  -- <binary> ...  # profile=eval
tools/nanochat verify -- <binary> ... # profile=t2-parity
tools/nanochat gpuq status            # the queue depth and the worker
```

Below it, the GPU queue owns the device. One worker process runs the jobs in
first-in, first-out order, and it assigns the devices. The worker runs each job
through `python/nanochat_cpp/sandbox.py`. That program applies a cgroup v2 budget (RAM, CPU,
pids, wall clock) through the **unprivileged systemd user manager**, then
verifies the limits. It needs no root and no container runtime. Full sources:
`tools/nanochat`, `python/nanochat_cpp/gpu_queue.py`, `python/nanochat_cpp/sandbox.py`,
`tools/sandbox/profiles.json`.

- GPU suites: `tools/nanochat test --gpu <target>` submits one job per test
  action; a bare GPU binary runs as `tools/nanochat gpu -- <binary>`.
- The queue: `tools/nanochat gpuq list`, `gpuq status`, `gpuq logs <id>`,
  `gpuq cancel <id>`, `gpuq stop`.
- A bare CPU test or binary: `tools/nanochat run t0-cpu -- <binary>`.
- Keep a log (`/tmp/nanochat-gpu.log`) so agents can see who used the GPU, and
  `/tmp/nanochat-sandbox.log` for the per-launch limits.

**Profiles** (`tools/sandbox/profiles.json`; host budget 12 CPU / 7.7 GiB):

| Profile | CPUQuota | Mem high/max | Use |
|---|---|---|---|
| `t0-cpu` | 200% | 1.5G / 2G | CPU tests, oracle-on-CPU |
| `t1-gpu` | 200% | 2G / 3G | tiny-shape kernel tests |
| `t2-parity` | 800% | 5G / 6G | full oracle, small training |
| `t3-bench` | 1000% | 5G / 6G | exclusive benchmark/profile |
| `train` | 800% | 5G / 6G | training runs |
| `eval` | 400% | 3G / 4G | forward-only eval |

The aggregate ceiling is installed automatically on the first launch; the
drop-in is `tools/sandbox/nanochat.slice`. Every launch is placed in
`nanochat.slice`, so concurrent T0 jobs cannot add up past the host budget even
if each is small.

### 5.3 Test tagging in Bazel

```python
cc_test(
    name = "rms_norm_gpu_test",
    srcs = ["rms_norm_test.cu"],
    tags = ["gpu", "manual"],   # excluded from default runs
    ...
)
```

- `-gpu` filter for the fast default test loop.
- `--local_test_jobs=1` for GPU runs.
- `manual` keeps GPU targets out of `bazel test //...` churn.

### 5.4 Discipline

- **Tiny shapes by default.** All T1 tests use `B=2, T=8`, `head_dim=4`,
  `num_heads=2`. Microseconds, not seconds.
- **Fixed seeds and committed fixtures.** The oracle is data, so GPU tests never
  need torch at runtime.
- **One job at a time on the GPU.** No concurrent training, no concurrent
  benchmarks, no profiler alongside a test.
- **Sandbox everything that runs longer than a few seconds.** Training, eval,
  GPU tests, and benchmarks go through `tools/nanochat`; the aggregate
  `nanochat.slice` is the only thing preventing cross-agent out-of-memory.
- **Builds are bounded and serialized.** Each worktree starts its own Bazel
  server; `.bazelrc` caps its heap and idle lifetime and `tools/nanochat` holds
  one build lock, so concurrent worktrees queue instead of multiplying the
  budget. Run `tools/nanochat shutdown` when a worktree is finished, then
  `tools/nanochat prune --worktree <root> --apply`: the output base outlives the
  worktree and piles up (see [docs/sandbox.md](docs/sandbox.md)).
- **Never bypass the entry point.** Executables call
  `nanochat::RequireSandboxOrDie` at startup and exit with the corrective
  command when launched outside `tools/nanochat`.
- **Benchmarks are separate from correctness.** A benchmark that runs anytime
  will eventually collide with a test; run benchmarks through the queue and
  record the commit hash.
- **`CUDA_LAUNCH_BLOCKING=1`** only for debugging, never for timing.
- **Profile on native Linux.** Under WSL2, CUPTI is unavailable; do not try
  Nsight there.
- **Detect a stuck GPU.** If a job leaves the device in a bad state, the
  worker's `nvidia-smi` gate refuses the next job until it clears; the
  integrator can `nvidia-smi` / reset.

### 5.5 Merge-gate GPU budget

The integrator runs, per merge:

- `tools/nanochat lint` (the Google C++ Style gate).
- `bazel test //... --test_tag_filters=-gpu` (all CPU tests, under `t0-cpu`).
  The S0 simulator suites ride this loop: they are CPU-only and need no queue.
  Two targets carry `manual` and stay out of it. `//tests:cuda_sim_preload_test`
  needs `tools/nanochat simulate --suite api`. `//tests:numerics_trace_test`
  replays a 10-layer trace, which is too slow for the inner loop; run it
  through the simulator suite below.
- `//python:pipeline_test` rides that loop too. It is the gate of the whole
  post-training chain ([docs/pipeline-plan.md](docs/pipeline-plan.md) section
  7): one job file, base then SFT then RL at the `demo` profile, over the
  committed fixtures and the real stage executables, about 5 s. Its CUDA twin
  is `//python:pipeline_gpu_test`, the same source with the `gpu` and `manual`
  tags, so the `-gpu` filter keeps it out of the default loop and it runs only
  through `tools/nanochat test --gpu //python:pipeline_gpu_test`.
- `tools/nanochat simulate --device h100` when the change touches a kernel, the
  backend seam, or the device profile (`docs/simulator.md`). It is still CPU
  only. Its `correctness` suite carries the CPU and simulator legs of the staged
  numeric trace (`docs/numerics-integration.md`), so that run also gates Path A.
- A single T1 GPU smoke test covering the changed family (`t1-gpu`).
- T2 parity only at Wave boundaries, not per commit (`t2-parity`).
  `tools/nanochat test --gpu //tests:numerics_trace_cuda_test` is the numeric
  trace at that tier: about ten seconds in the queue, and it needs its own
  `--config=sim` CPU run to be meaningful.

This keeps GPU time bounded and predictable.

---

## 6. Conflict avoidance checklist

- One directory, one owner; frozen headers touched only by the architect.
- Bazel dependency edges enforce §2.1 of `DESIGN.md`; an agent literally cannot
  reach across a layer.
- Stubs exist from Wave 0, so the tree always builds and links.
- New files go under the agent's own directory; never append to another agent's
  file.
- `BUILD.bazel` files are owned per directory; only the build agent edits the
  root ones.
- The two example packages aggregate through `test_suite(name = "all_tests")`
  in the root `BUILD.bazel`. An example edits its own directory and never the
  root file: the build agent adds the label, so the two examples cannot fight
  each other or the build agent over one shared file. The include rule for an
  example is in §1 and in `docs/examples.md`.
- Shared docs (`README.md`, `DESIGN.md`, `AGENTS.md`, `CONTRIBUTING.md`,
  `docs/**`) are architect-owned.

---

## 7. Communication artifacts

- Per-workstream brief: the frozen header(s), the owned directory, the
  Definition of Done, the exact test command. Agents should not need to read the whole repo.
- `backends/cuda/kernels/README.md`: the convention for the per-kernel tests
  and benchmarks that sit next to each kernel.
- `dev/kernels/README.md`: the convention for the unpromoted prototypes and the
  toolchain spike.
- `docs/performance.md`: the performance measurement protocol, the debugging
  interface (`tools/nanochat doctor` / `profile`), and the benchmark battery.
  Read it before investigating a slowdown. `tools/nanochat doctor` is the
  first command to run when a timing looks wrong.

---

## 8. Anti-patterns

- Editing a frozen header to make a kernel compile.
- Running training or long benchmarks on the GPU outside the queue.
- Running training, eval, a GPU test, or a benchmark without `tools/nanochat`.
- Calling `bazel test`, `train_main`, `eval_main`, or a `*_test`/`*_dev` binary
  directly instead of going through `tools/nanochat`.
- Using the GPU as the first place to test logic instead of the CPU reference.
- Two agents editing the same file "because it was small".
- Adding a third-party runtime dependency without architect sign-off.
- Building a compiler/IR/autograd engine — see `DESIGN.md` §6.
