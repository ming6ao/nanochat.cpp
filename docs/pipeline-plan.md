# The pipeline plan: one chain, three hosts

Status: draft. Written for architect review, before any code changes. Nothing in
this document is implemented.

The oracle agent reviewed sections 3 to 9 against the tree. Corrections C1 to C9
of section 12 are applied in place, so this document reads as the corrected
version. The two decisions the review asked for are recorded as D10 and D11.

This plan extends [rl-modal-plan.md](rl-modal-plan.md). That document covers the
RL phase on Modal. This one covers the whole chain, the defects that block it,
and the three hosts that must run it: a local CPU, Colab with a T4, and Modal
with a T4.

## 1. Goal

One job file runs base training, then supervised fine-tuning (SFT), then
reinforcement learning (RL), on any of three hosts. Each stage reads the
previous stage's weights. The run delivers one final checkpoint and one report.

The chain must run with a small model, so a person can watch it end to end in
seconds on the CPU and in under a minute on a card.

## 2. What runs today

I ran the three stages by hand at 2 layers, hidden 32, 2 heads, 2 key/value
heads, sequence 32, and vocabulary 487 with 512 padded. The results:

| Stage | Entry | Steps | Wall clock | Recorded |
|---|---|---|---|---|
| Base | `runtime.MeshTrainConfig` over `src/parquet/testdata/text.parquet` | 100 | 0.49 s | `base`, step 100 |
| SFT | `sft_main --warm-start base.ckpt` | 100 | 0.51 s | `sft`, step 100 |
| RL | `nc.rl.run` warm started from SFT | 5 | 1.05 s | `rl`, step 105 |

The parameter L2 norm moved at every stage: 487.8, then 473.5, then 470.8. The
chain works. Six defects block a host-neutral, automated run of it.

## 3. The six blockers

| # | Blocker | Evidence |
|---|---|---|
| B1 | The SFT stage cannot set the model shape | `sft_main: cannot warm-start ...: the recorded run config differs` |
| B2 | No model flag sets `value_embedding` | The same message, after every flag matched except that field |
| B3 | The RL stage overwrites the checkpoint it warm starts from | `rl.py` loads and saves `checkpoint` |
| B4 | Every rollout in a group is identical | All four rewards of a group were equal; the graded and constant runs gave the same parameters |
| B5 | A nested `tools/nanochat` starts a nested sandbox scope | `Toolchain.train` has no in-sandbox branch, but `Toolchain.gpu` does |
| B6 | The two stage executables exist only in a Bazel tree | `sft_main` and `rl_worker` have no installed home |
| B7 | The startup gate probes a fast-accumulate fp8 GEMM on every card | `cuBLAS error ... set fast accum: code 7` on a T4 and on a 1080 Ti |
| B8 | `--hw-check` rejects every value | `src/train_main.cc:155` compares a `const char*` with a literal |
| B11 | The two-replica path mismatches a parameter about once in twenty runs | `runtime_gpu_test: FAIL: replica parameter mismatch: lm_head.weight` |

### B1. The SFT stage cannot set the model shape

`nc.sft.run` forwards no model flag (`python/nanochat_cpp/sft.py`, `run`). The
driver therefore builds the 12-layer default, and `Checkpointer::LoadModel`
refuses the warm start.

Change: add `config: Config | None = None` to `nc.sft.run`. Factor the flag
list out of `rl.worker_argv` (`python/nanochat_cpp/rl.py`) into one helper,
`api.model_flags(config) -> list[str]`. Both facades call it, so the two stages
cannot drift. When `config` and `seq_len` disagree, raise, because the shard
owns the sequence length.

Test: a new `//python:sft_test` pins the argument list with a fake toolchain, in
the style of `python/tests/toolchain_test.py`. The facade has no test today.

### B2. No model flag sets `value_embedding`

`nanochat::cli::IsModelFlag` and `ApplyModelFlag` (`src/cli.h`) cover nine
fields. `Config::value_embedding` is not one of them, so a base run with value
embeddings off cannot be reproduced by any stage command line.

Change: add `--value-embedding <0|1>`, default true, and return false for a bad
value. Add the flag to `api.model_flags`. Add the flag to the usage line of
every binary that prints the model flags.

`src/cli.h` is not a frozen header, so the architect does not need to approve a
header change. Check the usage strings with a grep before editing, because
seven binaries print them.

Test: extend `TestModelFlagParsing` in `src/harness_test.cc:388`, which already
covers `ApplyModelFlag` and which `//src:harness_test` runs in the default
suite. Add the new flag, then loop over every flag in `IsModelFlag` and assert
that `ApplyModelFlag` accepts it. A flag that one function knows and the other
does not is then a failure. Do not add a second test target: it would give the
flag table two homes.

### B3. The RL stage overwrites the checkpoint it warm starts from

`nc.rl.run` derives the load path from `checkpoint` when that file exists, and
it saves to the same path (`python/nanochat_cpp/rl.py`, `run`). A warm start
therefore destroys its parent.

Change: add `warm_start: str | os.PathLike | None = None`. It becomes the
`--checkpoint` load argument. Keep `checkpoint` as the save target only. When
`warm_start` is given, never derive a load from `checkpoint`.

The legacy derivation stays for one release, with a note in `docs/python.md`,
because a notebook in the field may depend on it.

Test: `python/tests/rl_test.py` gains a fake-worker case. `warm_start` must
reach the worker as `--checkpoint`, and the `save` message must name
`checkpoint`.

### B4. Every rollout in a group is identical

`SampleToken` builds a fresh generator from `params.seed` on every call
(`src/generate.cc`, `Rand local(params.seed)`). `GenerateBatch` copies one
`SampleParams` for every row and keeps one seed for every decode step. So row
`r` and row `r + 1` draw the same quantile at the same step, forever.

Consequence: the four samples of a group hold the same text, every advantage is
zero, and the RL stage trains on nothing. I confirmed it: a graded reward and a
constant reward gave the same final parameters to the last bit.

Change: mix the row index and the decode step into the seed before each
`Decode` call in `GenerateBatch`:

```
sample.seed = MixSeed(params.seed, r, step)
```

`MixSeed` is a file-local helper in `src/generate.cc`, so no frozen header
moves. `include/nanochat/model.h` and `include/nanochat/sampler.h` stay as they
are. The greedy path is unchanged, because it draws no random value.

`GenerateMultiPrompt` copies `params` per prompt and keeps the same seed
(`src/generate.cc:559`), so row `r` of prompt 0 and row `r` of prompt 1 draw the
same quantile. Without a fix the rollout set stays diverse only because the four
prompts differ, which is too weak a foundation. Mix the prompt index in the same
way, `per_prompt.seed = MixSeed(params.seed, prompt, 0)`; a single-prompt call
is then unchanged.

Three properties must hold, and all become tests:

- Row independence. Row `r` of a batch equals row `r` of a one-row run at the
  same seed. A one-row run uses row 0, so the property holds.
- Reproducibility. One seed, one row, one step gives one draw.
- Prompt independence. Row `r` of prompt 0 and row `r` of prompt 1 differ, in the
  same way that two rows of one prompt differ.

Test: `//src:generate_test` gains three cases. Four rows at one seed give four
distinct streams. Row 0 of a one-row batch equals row 0 of a four-row batch. Two
prompts at one seed give different first rows.

`//tests:generate_parity_test` stays green, because its fixture is greedy
(`temperature = 0.0f`, `tests/generate_parity_test.cc:203`). The committed
`tests/data/generate_parity.bin` records greedy output only, and `rl_parity.bin`
replays recorded rows, so no fixture changes.

The blast radius is wider than the RL rollout path. Parity item E1
(`docs/parity.md`) lists the base-evaluation sample mode and the generative
`ChatEvaluator` tasks, so every recorded sampled evaluation number moves once.
`docs/eval.md` §4 says only bits-per-byte ships today, so in practice
`ChatEvaluator` is the one consumer. The E1 edit is two sentences: the stream is
derived per row, per prompt, and per step, and sampled evaluation numbers moved
once at this change.

### B5. A nested `tools/nanochat` starts a nested sandbox scope

`Toolchain.gpu` runs the command directly when `NANOCHAT_SANDBOX` or
`NANOCHAT_GPU_QUEUED` is set. `train`, `rl`, `eval`, `bench`, and `verify` do
not, so a stage launched from a sandboxed test action re-enters `sandbox.py` and
starts a scope inside a scope.

Change: one private helper in `python/nanochat_cpp/toolchain.py` holds the
in-sandbox branch, and the six methods call it. The helper keeps the documented
rule from `Toolchain.gpu`'s docstring.

Test: `python/tests/toolchain_test.py` gains one case per method, beside
`test_gpu_inside_sandbox_skips_the_queue`.

This change makes one sandbox profile the real ceiling for the whole chain. It
is not what makes the chain run: without it each stage child starts a scope
inside the scope, which systemd allows and which the inner profile hides from
the outer budget. B5 removes that doubled budget. The order of section 8 puts it
first because the later rows add stage launches a person will run by hand.

### B6. The two stage executables exist only in a Bazel tree

`nc.sft.run` defaults its binary to `bazel-bin/src/sft_main`, and `nc.rl.run`
defaults to `bazel-bin/src/rl_worker`. A wheel install has neither. The Modal
image keeps the tree, so both paths exist there; a Colab clone must build them.

Change: one resolver, `_build.stage_binary(name) -> Path`. It follows the
implemented order of `_lib.load`, not the five-entry list in `docs/python.md`
§3.2: that list omits the runfiles tree and the wheel, and `_lib.py:6` cites
"sections 5.1 and 8.3", which do not exist. Row 11 fixes both. The resolver
order is:

1. the `NANOCHAT_SFT_MAIN` and `NANOCHAT_RL_WORKER` overrides;
2. the Bazel runfiles tree;
3. `bazel-bin/src/<name>`;
4. `shutil.which("nanochat-" + name)`, for a future wheel;
5. an error that names the build command.

Both facades call it. Add a sibling builder, `_build.build_stages(options)`, which
builds the library and the two binaries in one `tools/nanochat build` call, so
`_build_command` keeps one home for the command line. Name it `build_stages`
here, in section 6.3, and in row 6 of section 8, because one name is enough. The
Modal image extends its existing `.run_commands` build call
(`tools/modal/app.py`) to name `//src:sft_main` and `//src:rl_worker`. The Colab
setup runs the same call.

Test: a unit test for `stage_binary` with a fake tree, covering the override,
the runfiles miss, and the error text.

### B7. The startup gate probes fast-accumulate fp8 on every card

`CheckScaledGemmCases` runs one scaled fp8 GEMM with `fast_accum = true`
(`src/hw_gate.cc:311`). The CUDA backend then sets
`CUBLASLT_MATMUL_DESC_FAST_ACCUM` (`backends/cuda/gemm.cu:190`). That attribute
needs a TF32-class card. On a T4 (7.5) and on a GeForce 1080 Ti (6.1) the call
returns code 7, and `CheckCublas` prints a fatal message and calls `std::exit(1)`
from a device thread (`backends/cuda/gemm.cu:33`).

Measured on 2026-10-31:

- `train_main` on the 1080 Ti exits 1 with
  `fatal: cuBLAS error in cublasLt set fast accum: code 7`.
- `//src:runtime_gpu_test` on the 1080 Ti dies with `Segmentation fault`.
- The base stage through `MeshTrainConfig` dies with `Segmentation fault` on the
  T4 and on the 1080 Ti, at the first step.

So every CUDA host below compute capability 8.9 cannot start. That includes the
deployed Modal card, which is the plan's pinned triple (D3).

Change: gate the fp8 probe, and the fp8 path, on compute capability 8.9, which
`docs/fp8-training.md` §305 already requires. Report an unsupported probe as a
skip, not a fatal error. Do not call `std::exit` from a device thread: return
the status to the caller, so the driver can stop cleanly. The segfault in the
mesh path is the second half of this defect, and it disappears when the exit is
not taken mid-flight.

Test: `//src:runtime_gpu_test` passes on a T4. `train_main` at `--hw-check auto`
starts two steps on a T4. The chain of section 13 reaches its first base step.

Status on 2026-10-31: fixed and verified. The gate reports
`hw gate: skipped 6 fp8 cases: the fp8 scaled GEMM needs compute capability 8.9
or 9.0, and this device reports 7.5` on a T4 and `6.1` on a 1080 Ti, and the
cuBLAS abort is gone. `//src:runtime_gpu_test` and the chain of section 13 pass
on both cards.

### B8. `--hw-check` rejects every value

`config.hw_check` is set from three `if` branches that compare a `const char*`
with a string literal (`src/train_main.cc:155`). The comparison is of pointers,
so it is false for every argument, and the parser exits 2 with
`--hw-check must be auto, on, or off`. The gate therefore cannot be turned off
from the command line, which is the one workaround B7 needs.

Change: compare `std::string(value)` with `"auto"`, `"on"`, and `"off"`.

Test: a parser case for each value, beside the model-flag cases of
`src/harness_test.cc`.

### B9. The host rope leg read the device table

`ops::QkPrepForward` and `ops::QkPrepBackward` run their rotation on the host,
but the tables they were handed come from `TrainModel::cos_table()`, which
returns the **device** copy (`src/model_impl.h:89`) because the kernels read it
from device memory. A host loop over a device pointer segfaults. The CPU backend
hides it, because there `Alloc` returns host memory.

Measured: `train_main` on the 1080 Ti died in `ops::QkPrepForward` on the first
forward pass, and the T4 chain died at the first base step.

Fixed: `src/ops.cc` stages the two tables to host once per call, in both legs. On
the CPU backend the staging copy is a memcpy, and `//tests:sft_step_parity_test`
stays green.

### B10. The CUDA train step has host/device staging defects

`kernels::SpanReduce` reads its span list from device memory
(`SpanReduceParams::counts` is documented as a device array, and the two pointer
arguments as a device-resident list). Three callers in `src/ops.cc` built that
list in stack storage: `BackoutBackward` and the two embedding-backward calls.
`RaggedSumOfSquares` is not one of them: its only caller, the optimizer, already
keeps `norm_grads_` and `norm_counts_` as device tables (`src/optim.cc:504`).
The defect is invisible on the CPU backend, where `Alloc` returns host memory.

Measured after B7, B8, and B9:

- `train_main` on the 1080 Ti: `fatal: CUDA error in cudaMemcpy: an illegal
  memory access`. With `CUDA_LAUNCH_BLOCKING=1` the stack names
  `ops::BackoutBackward` -> `kernels::SpanReduce` ->
  `SpanReducePartialKernel`.
- `//src:runtime_gpu_test`, the tree's own CUDA gate, fails the same way. So the
  tree has no working CUDA train step on this host, and the defect is older than
  this plan.
- The T4 chain fails identically, so it is not a capability issue.

Fixed: `src/ops.cc` gains a `SpanList` type that stages the span pointers, the
second operand pointers, and the counts to device memory for the life of one
call, the same idiom the optimizer uses. On the CPU backend the staging copy is a
memcpy, and `//tests:sft_step_parity_test` and `//tests:sft_loop_parity_test`
stay green.

Verified on 2026-10-31: `//src:runtime_gpu_test` passes on the 1080 Ti,
`train_main --hw-check auto` runs two steps on the card (loss 6.1874 to
5.675526), and the chain of section 13 finishes on a T4 in 8.6 s.

### B11. The two-replica runtime path is not reliable

`//src:runtime_gpu_test` builds two replicas on one card and compares their
parameters after a step. It passed 17 of 18 runs on the 1080 Ti and failed once
with `replica parameter mismatch: lm_head.weight`. The A/B is unambiguous: with
the B10 fix removed the test segfaults in 8 of 8 runs, so the flake only became
observable once the earlier faults were gone.

`kernels::Alloc` is `cudaMalloc`, which is thread-safe, so the per-call staging
of B9 and B10 is not the cause. The mismatch is a nondeterministic parameter
difference between two replicas that share one card, so either the in-process
collective or a per-thread buffer is racy.

This does not block the pipeline: the chain runs `world_size=1`. It blocks any
distributed claim, and it belongs to the Runtime owner. The test is the
reproducer: run it in a loop and keep the failing output.

## 4. Decisions

**D1. One orchestration module, three hosts.** `python/nanochat_cpp/pipeline.py`
holds the chain. Each stage calls an existing facade. The module holds no
training logic, no model flag list of its own, and no host name. A host supplies
the environment and the artifact root, and nothing else.

**D2. Every host builds the two stage binaries.** The chain needs a process for
the SFT driver and a process for the RL worker. A tree build is the only route
until D6 of the dependency plan publishes a binary wheel. The resolver of B6
makes the route swappable.

**D3. The architecture triple is T4, one card, `sm_75`, fp32 on Colab and
Modal.** That is the deployed Modal image (D7 of `rl-modal-plan.md`) and the
free Colab card. A local card uses its own architecture through
`NANOCHAT_CUDA_ARCH`.

**D4. The default corpus is the committed fixture.** The base stage reads
`src/parquet/testdata/text.parquet` with `tests/data/loader_tokenizer.nctoken`.
The SFT stage reads one shard that `nc.sft.pack` writes from a built-in
conversation set. The RL stage reads four of those conversations. No stage reads
the Hub, so one job runs on every host with no download.

**D5. The demo model has 2 layers.** Two layers, hidden 64, 4 heads, 2 key/value
heads, sequence 64, vocabulary 487 with 512 padded. That shape costs about 4 s
of CPU for the whole chain, and less than a second on a card. A larger profile
is a timing exercise, not a gate.

**D6. `deliver/` is the artifact contract.** The run root holds one directory
per stage and one `deliver/` directory:

```
<run_root>/<run_id>/base/checkpoint.ckpt
<run_root>/<run_id>/sft/checkpoint.ckpt
<run_root>/<run_id>/rl/checkpoint.ckpt
<run_root>/<run_id>/deliver/chatrl.ckpt
<run_root>/<run_id>/deliver/pipeline_report.json
```

The report is the review artifact. It holds the stage order, the step counts,
the loss traces, the mean rewards, the checkpoint digests, the metadata blocks,
the seconds, the seeds, the model shape, the tokenizer digest, and the source
revision.

**D7. No frozen header changes.** B2 lands in `src/cli.h`. B4 stays inside
`src/generate.cc`. B1, B3, B5, and B6 are Python. `include/nanochat/model.h`,
`include/nanochat/sampler.h`, and the other frozen headers do not move.

**D8. Colab uses the source tree, not the wheel.** The chain needs
`tools/nanochat` and the two binaries. A wheel path lands when a binary wheel
exists, and B6 makes that a one-line change.

**D9. One owner per file.** The architect assigns the rows this plan adds:

| File | Owner |
|---|---|
| `python/nanochat_cpp/pipeline.py`, `python/tests/pipeline_test.py` | Python surface |
| `python/nanochat_cpp/sft.py`, `rl.py`, `api.py`, `toolchain.py`, `_build.py` | Python surface |
| `src/cli.h` | Unassigned, exactly as `src/rl*.cc` is (`rl-modal-plan.md` §6.1). The architect assigns it before row 2 of section 8 |
| `src/generate.cc` | Workflow |
| `tools/modal/app.py`, `tools/modal/modal_app_test.py` | Architect assigns; `rl-modal-plan.md` §6.1 records the gap |
| `notebooks/colab_pipeline.ipynb`, `notebooks/colab_setup.py` | Notebooks |
| `docs/**`, `AGENTS.md`, and the root `BUILD.bazel` files | Architect. A per-directory `BUILD.bazel` belongs to that directory's owner (`AGENTS.md` §6), so `python/BUILD.bazel` is the Python surface's and `src/BUILD.bazel` is the owning workstream's |

**D10. The pipeline process entry is always wrapped by `tools/nanochat`.** The
root of the chain runs as `tools/nanochat train -- python -m nanochat_cpp.pipeline <job>`
on a host, and as `tools/nanochat client --profile train -- python -m nanochat_cpp.pipeline <job>`
in a container. Two child executables check the sentinel with
`RequireSandboxOrDie`, and no host may satisfy that check with the override. The
wrapper sets the sentinel, so one rule covers the local card, Colab, and Modal.

**D11. A6 compares two runs, and no warm start changes.** The optimizer state
stays part of what a warm start carries, as `docs/post-training.md` §7 says.
Clearing the moments on an RL warm start would change the stage for every
caller to make a test simpler, so the test pays the second leg instead.

## 5. The pipeline module

```python
# python/nanochat_cpp/pipeline.py

PROFILES = {"demo": Config(num_layers=2, num_heads=4, num_kv_heads=2,
                           hidden_dim=64, seq_len=64, vocab_size=487,
                           padded_vocab_size=512, window_pattern="L")}

@dataclass(frozen=True)
class Job:                 # JSON round trip; one file for all three hosts
    run_id: str
    profile: str = "demo"
    base_steps: int = 100
    sft_steps: int = 100
    rl_steps: int = 5
    num_prompts: int = 4
    num_samples: int = 4
    max_tokens: int = 32
    seed: int = 42
    out_root: str = ""
    corpus: str = "fixture"      # "fixture" today; "reference" is a follow-on
    deliver: bool = True

@dataclass(frozen=True)
class Stage:               # one stage's facts
    name: str
    steps: int
    losses: tuple[float, ...]
    checkpoint: str
    sha256: str
    metadata: dict
    seconds: float
    extra: dict = field(default_factory=dict)   # means rewards, dropped groups

@dataclass(frozen=True)
class Report:
    version: int               # the report schema, 1 today
    job: Job
    stages: tuple[Stage, ...]
    delivered: str
    report_path: str
    total_seconds: float
    host: dict                 # backend, arch, precision, source revision

def stage_base(job) -> Stage
def stage_sft(job, base: Stage) -> Stage
def stage_rl(job, sft: Stage) -> Stage
def deliver(job, stages, *, out_root) -> tuple[str, str]
def run(job, *, out_root=None) -> Report
def main(argv=None) -> int        # python -m nanochat_cpp.pipeline job.json
```

Each stage function is short and calls one facade:

| Stage | Call |
|---|---|
| Base | `runtime.MeshTrainConfig(config(job), steps=job.base_steps, ...)` with `train_parquet`, `tokenizer`, `out_dir`, `checkpoint_path` |
| SFT | `sft.pack(...)` then `sft.run(shard, ckpt, config=config(job), warm_start=base.checkpoint, num_iterations=job.sft_steps)` |
| RL | `rl.run(reward, prompts, num_samples=..., steps=..., config=config(job), warm_start=sft.checkpoint, checkpoint=rl_ckpt, max_tokens=...)` |

`config(job)` is `PROFILES[job.profile]`, so the shape has one home.

The reward is a class with a `reward` method (`_normalize_task` accepts one), so
no Hub dataset loads. It is graded: 1.0 for the right number, 0.5 for the
`####` marker, otherwise the character overlap with the reference. A graded
reward keeps the advantages from collapsing (risk 6.3 of `rl-modal-plan.md`).

The corpus builder copies the committed parquet and tokenizer into
`<run_root>/<run_id>/corpus/`, and writes `<NANOCHAT_BASE_DIR>/tokenizer/tokenizer.nctoken`
because `data.nctoken1_path()` reads that exact path.

The pipeline never reads `NANOCHAT_BASE_DIR` for output. The report names the
root it used.

## 6. The three hosts

### 6.1 Local CPU

```
tools/nanochat build //src:sft_main //src:rl_worker
NANOCHAT_CPP_BACKEND=cpu tools/nanochat train -- \
    python -m nanochat_cpp.pipeline /tmp/job.json
```

One `train` scope (6 GB, 800 % CPU) covers all three stages. B5 keeps the stage
children inside that scope instead of starting a second scope inside the first.
Without B5 the chain still runs here, but the inner profile hides its budget from
the outer one. With B5 the profile is the real ceiling.

Acceptance: the chain finishes, `deliver/pipeline_report.json` exists, and the
three stages record `base`@100, `sft`@100, `rl`@105.

### 6.2 Local GPU

The same command with `NANOCHAT_CPP_BACKEND=cuda`, under
`tools/nanochat gpu --profile t1-gpu -- ...`. `tools/nanochat doctor` reports
`gpu` as name, driver, memory, use, and utilisation, and no compute capability
(`DOCTOR_KEYS`, `python/nanochat_cpp/cli.py:74`), so the arch comes from
`nvidia-smi --query-gpu=compute_cap --format=csv,noheader`. That probe already
exists at `tools/modal/app.py:228`.

Acceptance: the same report, with `host.backend` equal to `cuda`.

### 6.3 Colab with a T4

New `notebooks/colab_pipeline.ipynb`, four cells:

1. Setup: clone the tree, install Bazelisk with `tools/kaggle/bootstrap.sh`,
   then `notebooks/colab_setup.setup(backend="cuda", arch=probe(),
   prefer_installed=False)`, which resolves the library, then
   `_build.build_stages()`, which builds the two stage binaries. Two calls, not
   one: `setup` resolves the library only.
2. Environment: `NANOCHAT_BASE_DIR=/content/cache`,
   `NANOCHAT_RUNS_DIR=/content/runs`, `NANOCHAT_CPP_BACKEND=cuda`,
   `NANOCHAT_CUDA_ARCH=probe()`, `NANOCHAT_SANDBOX_BACKEND=none`. Colab has no
   systemd user manager, so the sandbox is off, as on Kaggle.
3. Run:
   `tools/nanochat train -- python -m nanochat_cpp.pipeline /content/runs/job.json`.
4. Collect: print the report, and offer one download of `deliver/`.

Cell 3 must wrap the module. The backend value satisfies the library check in
`nanochat_init` only: both stage executables call `RequireSandboxOrDie`, which
accepts `NANOCHAT_SANDBOX` or the override and not the backend value
(`bindings/nanochat_capi.cc:103`, `include/nanochat/sandbox.h:35`). So the entry
is the tool, which sets the sentinel to `off` on a host without systemd. Do not
set `NANOCHAT_ALLOW_UNSANDBOXED` in the notebook: §6.4 of `rl-modal-plan.md` bans
it in the image, and the same reason holds here.

`probe()` is `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`, mapped
to `sm_<major><minor>`. Free Colab gives a T4, so `sm_75`. A paid session can
give a V100 (`sm_70`) or an A100 (`sm_80`), and an `sm_75` build then fails
inside `nanochat_init` with a message about the architecture, not about the job.

`prefer_installed=False` is required, not optional. `setup` prefers an
installed distribution by default (`notebooks/kaggle_setup.py:266`), so a session
that already carries the wheel returns the install root as `repo`. That root has
no `tools/nanochat` and no stage binaries, so it contradicts D8 and breaks D10.

`notebooks/colab_setup.py` is additive: a thin module that calls
`kaggle_setup.setup(...)` with the `/content` defaults, then `build_stages()`.
`kaggle_setup.py` keeps
its name, because `notebooks/session.py:33` and five documents name it, and a
rename would leave the shim alive forever. A rename is a separate, mechanical
change with no behavior, and it is not part of this plan.

Acceptance: `/content/runs/<run_id>/deliver/pipeline_report.json` exists, and
`host.backend` is `cuda` with `host.arch` equal to the probed value.

### 6.4 Modal with a T4

`tools/modal/app.py` gains a second function, `pipeline(job_json, run_id,
stage_json="", publish_json="")`. It mirrors `train`:

- it rewrites `out_root` to `/runs/<run_id>`, because the app owns the Volume
  path;
- it starts the work as
  `tools/nanochat client --profile train -- python -m nanochat_cpp.pipeline <job.json>`,
  which is the container entry pattern of D3 in `rl-modal-plan.md`;
- it keeps `max_containers=1`, `_check_cards`, the commit keeper, and the
  `worker.log`;
- it reuses `_publish` for the Hub push of the final stage, so the chain gains
  no second uploader.

The client half is `ModalTarget(function="pipeline")`, which needs the
`function` argument that phase 1 of `rl-modal-plan.md` already lists.
`ModalTarget.prepare` reads `output.out_dir` and raises without it
(`notebooks/target.py:266`), so the target gains one `job_kind` branch whose
`"pipeline"` case reads `out_root`. The job carries no host path either way,
because the app rewrites `out_root` to `/runs/<run_id>`. `target.map_job` and
`target.job_paths` need no new field table, because a fixture job holds no input
path.

Acceptance: `modal run tools/modal/app.py::pipeline_demo` prints a report with
`host.arch` `sm_75`, and the `nanochat-runs` Volume holds `deliver/`.

### 6.5 The host matrix

| Host | Backend | Arch | Sandbox | Build | Gate |
|---|---|---|---|---|---|
| Local CPU | `cpu` | none | `train` scope | Bazel | `//python:pipeline_test`, default loop |
| Local GPU | `cuda` | card | `t1-gpu` queue job | Bazel | `//python:pipeline_gpu_test`, `gpu` tag |
| Colab | `cuda` | `sm_75` | `none` | bootstrap + Bazel | manual run, recorded in `docs/host-portability.md` |
| Modal | `cuda` | `sm_75` | `none` | image build | `modal run ...::pipeline_demo`, recorded in `rl-modal-plan.md` |
| Modal CPU | `cpu` | none | `none` | image build, CPU config | verification only; see below |

The Modal CPU row is a verification host, not a product path. The image of
`tools/modal/app.py` pins one compiled backend: `NANOCHAT_CPP_LIB` names the CUDA
library and the build passes `--config=cuda`. A container with no card therefore
fails inside `nanochat_init`, and `_check_cards` calls `nvidia-smi`, which does
not exist there. So a CPU leg needs a second image whose build passes
`--config=cpu` and whose function omits `gpu=`. The spike of section 13 does
exactly that, and it must keep the two images apart, because one Bazel output
base can hold only one build of `libnanochat_shared.so`.

## 7. The integration test

`python/tests/pipeline_test.py` calls `pipeline.run(job)` with the `demo`
profile. The assertions are the contract, not the loss values. The parity rungs
own the numbers.

| # | Assertion | Catches |
|---|---|---|
| A1 | Each stage returns status 0 with the step count it was asked for | A shortened schedule |
| A2 | Every loss and reward is finite; the base loss falls from step 1 to step 100 | A dead optimizer |
| A3 | Metadata chain: `base`@100, `sft`@100, `rl`@105, and every recorded config equals the job's | B1, B2, a wrong warm start |
| A4 | The parameter L2 norm differs between base and SFT, and between SFT and RL | A stage that only rewrites its file. It does not prove the warm start, because the loaded moments move the weights on their own |
| A5 | One group of rollout rewards holds two distinct values, so its advantages are not all zero | B4. Read through `rl.run(..., on_step=...)`, which receives the per-rollout rewards; `RlReport` carries no advantage |
| A6 | Two runs from the same SFT checkpoint and the same seed, one with the graded reward and one with a constant reward, give different final parameters | A gradient path that the inherited moments hide |
| A7 | `deliver/chatrl.ckpt` equals the RL checkpoint byte for byte, and the report digests match the bytes | A lost artifact |
| A8 | The report names the stage order, the seeds, and the host facts | An unreviewable run |
| A9 | The delivered file loads through the public container reader, and its metadata reports `rl` at step 105 | A container only the pipeline can read |

A6 cannot be "a constant reward leaves the parameters unchanged". A warm start
carries the optimizer state (`docs/post-training.md` §7), so the inherited AdamW
moments move the weights with an advantage vector of exactly zero. I measured
it: the graded and the constant runs gave the same L2 norm to the last digit.
A6 is therefore a two-run comparison, and it is the one assertion that isolates
the advantage term.

A5 and A6 fail today. A4 passes today for the wrong reason. A3 and the argv that
`nc.sft.run` builds are what prove the chain.

Two targets, one source:

- `//python:pipeline_test`: the `demo` profile, `size = "medium"`, data
  `//bindings:nanochat_shared`, `//src:sft_main`, `//src:rl_worker`, the parquet
  fixture, and the tokenizer fixture. About 5 s, so it stays in the default
  loop.
- `//python:pipeline_gpu_test`: the same source, `NANOCHAT_CPP_BACKEND=cuda`,
  tags `gpu` and `manual`.

## 8. The order of work

One commit per row. Each row ends with `tools/nanochat check` green.

| # | Work | Gate | Small? |
|---|---|---|---|
| 0 | B7, the capability gate on the fp8 probe and a status return instead of `std::exit` | `//src:runtime_gpu_test` on a T4, then the chain of section 13 | small |
| 0b | B8, the `std::string` comparison in `train_main` | a parser case per value | two lines |
| 0c | B9, the host rope leg stages its table | `//tests:sft_step_parity_test` and the chain | done |
| 0d | B10, the `SpanList` staging in `src/ops.cc` | `//src:runtime_gpu_test` on a card, then the chain of section 13 | done |
| 1 | B5, the in-sandbox branch in `Toolchain` | `//python:toolchain_test` | one helper, six call sites |
| 2 | B2, `--value-embedding` | `//src:harness_test` | one flag, one usage line per binary |
| 3 | B4, the seed mix in `GenerateBatch` and `GenerateMultiPrompt` | `//src:generate_test` | four lines and three cases |
| 4 | B1, `config=` on `nc.sft.run`, shared `api.model_flags` | `//python:sft_test` | small |
| 5 | B3, `warm_start=` on `nc.rl.run` | `//python:rl_test` | small |
| 6 | B6, `_build.stage_binary`, `build_stages`, image and Colab build lines | `//python:build_test` | small |
| 7 | `pipeline.py`, the report, and `deliver()` | `//python:pipeline_test` A1 to A4, A7 to A9 | medium |
| 8 | The reward and A5, A6 | `//python:pipeline_test` A5, A6 | small |
| 9 | `notebooks/colab_setup.py` and the Colab notebook | manual run, recorded | medium |
| 10 | The Modal `pipeline` function and its contract test | `//tools/modal:all`, then `modal run` | medium |
| 11 | Documents: `rl-notebook.md`, `host-portability.md`, `post-training.md` §5, `parity.md` E1, `python.md` §3.1 and §3.2, `_lib.py:6`, `README.md`, `AGENTS.md` owner rows | `tools/nanochat lint` | small |

Rows 1 to 6 change no training arithmetic except B4, and B4 changes only the
sampled stream. So the numeric fixtures stay valid, and the wave-boundary parity
runs do not move.

## 9. Risks

**The seed change moves a sampled baseline.** Any recorded sampled generation
would change. I checked the committed fixtures: `generate_parity.bin` is greedy,
and `rl_parity.bin` replays recorded rows. Nothing records a sampled stream.
Confirm with a grep before row 3 lands.

**The reward can be uniform.** If the four samples of a group still coincide,
A5 fails. The graded reward and the per-row seeds make it unlikely, not
impossible. A5 should report the group size and the distinct reward values, so a
failure is diagnosable.

**Colab has no systemd.** The sandbox is off there, as on Kaggle, so the
container quota is the only bound. The `demo` shape is small, and the notebook
sets `max_tokens` and the step counts.

**The image build gets slower.** Two more Bazel targets in the Modal image add
seconds, and the image is already the slow path. The two binaries are small
beside the library.

**A wheel host still cannot run the chain.** B6 makes the error clear and the
fix one line. The wheel path stays out of scope (D8).

**Two documentation sources for the chain.** `post-training.md` §8 still says
"Implementation: not in the tree yet" while §11 says Done. Row 11 fixes that,
because the pipeline makes the gap visible.

**A6 is two legs, and the rollout shape is a flake surface.** The second RL run
adds about 1 s and doubles the samples the assertion sees. Keep `num_samples`,
`max_tokens`, the temperature, and the `top_k` value pinned in `Job`, and print
them in the failure message, so a failure names the shape it used.

**The two sandbox checks differ on purpose.** `nanochat_init` accepts
`NANOCHAT_SANDBOX_BACKEND=none` and `RequireSandboxOrDie` does not. That
difference is what lets a notebook call the library while a stage executable
still refuses to start unwrapped. Do not unify them: either the notebook path or
the stage guard would disappear.

**One Modal image holds one backend.** A CPU container cannot load the CUDA
library, so a CPU verification leg is a second image with `--config=cpu`. Do not
add a CPU branch to `tools/modal/app.py`: `_check_cards` and the pinned
`NANOCHAT_CPP_LIB` both assume a card, and the CPU leg has no product job.

**CUDA training is not bit-reproducible across runs.** Three identical T4 runs
of the demo job gave base L2 525.0440, 525.0414, and 525.0388. The CPU leg gives
the same digits every time. So a repeat-and-compare-digest check belongs on the
CPU leg only, and a GPU comparison must use a tolerance. The cause is the
reduction ordering of the device kernels, not the seed.

**The GPU column was red for tree reasons, not host reasons.** B7 stopped every
card below compute capability 8.9 at startup, and B10 stopped every card at the
first backward pass. Both are fixed, and the T4 leg of section 13 now passes.
The lesson is the one the tier rule already states: a GPU gate that nobody runs
is not a gate. `//src:runtime_gpu_test` was red before this work and is the check
that proves it now.

## 10. Non-goals

- Distributed training of any stage.
- The Hub corpus kind, and the reference data mixture. `corpus` accepts
  `"fixture"` only in this plan.
- A binary wheel, and a published CUDA wheel per architecture.
- Asynchronous rollouts, clipped proximal policy optimization, and a reference
  model. Those stay in `rl-modal-plan.md` §7.
- A performance number. The chain is a correctness and portability gate.

## 11. Open questions

**Q1. Should the seed mix live in `GenerateBatch` or in `SampleParams`?**

Recommendation: in `GenerateBatch`. `SampleParams` is a public struct in
`include/nanochat/sampler.h`, and `Decode` is declared in the frozen
`include/nanochat/model.h`, so a per-row seed field would either need a frozen
header change or a field no caller sets. The mix needs neither.

**Q2. Should `--value-embedding` join the model flags, or should the job forbid
value embeddings?**

Recommendation: add the flag. The Track 3 baseline disables value embeddings,
and `MeshTrainConfig` can already set the field through the job, so a driver
that cannot reproduce it is an asymmetry. A ban would hide a real model
configuration from the SFT and RL stages.

**Q3. Does the legacy load-from-`checkpoint` behavior stay?**

Recommendation: keep it for one release and note the deprecation in
`docs/python.md`. `warm_start` wins when both are given. A silent removal breaks
a notebook in the field.

**Q4. Does the pipeline run the base stage through the mesh driver or through a
new single-device entry?**

Recommendation: the mesh driver, because it is the only owner of the loop, the
record, the summary, and the resume, and its `packed_rows_path` and parquet
paths both work today. The pipeline passes `world_size=1`.

**Q5. Where does the demo job file live?**

Recommendation: in `python/nanochat_cpp/pipeline.py` as `Job()` defaults, plus
one committed example at `notebooks/pipeline_job.json`. The test builds the job
in code, so a committed file cannot drift into a gate; the example is for a
person.

**Q6. Should the Colab notebook be a build-from-source path or the wheel?**

Recommendation: source, per D8. The notebook is four cells, and the setup cell
reuses the Kaggle helper.

**Q7. Does the GPU leg need a recorded baseline report?**

Recommendation: no. Record the `host` block of one run per host in
`docs/host-portability.md`, with the date, the commit, and the seconds. A
throughput number that moves is information, not a gate.

**Q8. Is 5 RL steps enough to prove the stage works?**

Recommendation: yes, with A5 and A6 beside it. A4 proves the parameters moved,
A5 proves the rollouts differ, and A6 proves the movement came from the
advantages and not from the moments. Five steps of nothing would pass A1 and A2
alone.

## 12. The review corrections

The oracle agent reviewed sections 3 to 9 against the tree. Each correction is
applied in place, so the body of this plan is the corrected version. The table
records what moved and why, so a later reader does not re-open a settled point.

| # | Correction | Where |
|---|---|---|
| C1 | The pipeline entry is always wrapped by `tools/nanochat`. The container form in §6.3 was unwrapped, and the first stage binary would exit 2 | §6.3, D10 |
| C2 | A6 cannot be "a constant reward leaves the parameters unchanged". A warm start carries the moments, and they move the weights with a zero advantage | §7, D11 |
| C3 | A5 asserted a quantity `RlReport` does not carry. It reads the rewards through `on_step` | §7 |
| C4 | The seed mix belongs in `GenerateMultiPrompt` too, or two prompts draw the same quantiles | B4 |
| C5 | `src/cli.h` is unassigned, and the two per-directory `BUILD.bazel` files are not the architect's | D9 |
| C6 | The arch probe is `compute_cap`, which `tools/nanochat doctor` does not report; two citations were stale | §6.2, §6.3, B6, D9 |
| C7 | B2's gate extends `TestModelFlagParsing`, which already exists, instead of adding a second home | B2 |
| C8 | B5 prevents a nested scope and a doubled budget; it is not what makes the chain run. The `kaggle_setup.py` rename is dropped | B5, §6.1, §6.3 |
| C9 | A7 passes for a container only the pipeline can read, so A9 loads the delivered file through the public reader | §7 |

Two items here are not from the review. Both came out of checking the corrected
text: the `ModalTarget` output-field branch of §6.4, and the report schema
version in §5.

A second, bounded read of the corrected sections found eight smaller items, and
all are applied above: two line citations (`src/generate.cc:559`,
`src/harness_test.cc:388`), two more (`bindings/nanochat_capi.cc:103`,
`include/nanochat/sandbox.h:35`), the case count in row 3, the missing A8 and A9
in row 7, the `build_stages` and `prefer_installed=False` steps in §6.3, the
"distinct reward values" wording in §9, and the review scope of this section.
None of them changes a decision.

The review also confirmed four areas with nothing to change: no committed
fixture records a sampled stream, so B4 moves no gate; `_build_command`
centralizes the build, so B6 is one signature change; the two facades match the
stage calls once B1 and B3 land, and an in-process base stage beside spawned
children is sound; and the row order of §8 does not disturb the wave-boundary
parity runs.

## 13. Verification before implementation

The six blockers are not in the tree yet, so the chain cannot run as
`pipeline.run(job)` today. Two checks therefore exist, and they answer different
questions.

### 13.1 Phase 0: the premises, proven now

A throwaway harness runs the three stages against the binaries directly and
records which facade feature it bypassed. It proves the environment, not the
module. Files: `chain.py` and `app.py` in a scratch directory, not in this tree,
so they need no owner and no gate.

Already run on the local host, with the demo profile of D5 and the corpus of D4:

| Stage | Steps | Recorded | L2 | Seconds | Loss |
|---|---|---|---|---|---|
| Base | 100 | `base`@100 | 524.4127 | 1.16 | 6.1874 to 0.0049 |
| SFT | 100 | `sft`@100 | 515.7249 | 0.91 | 9.9730 to 0.0027 |
| RL | 5 | `rl`@105 | 513.6772 | 0.76 | 8.067, 7.989, 7.801, 7.486, 7.229 |

Total 2.90 s. The largest absolute advantage is `0.0`, which is the B4 control:
the run is correct and the RL stage still trains on nothing. Three premises are
proven by the same run:

- The sentinel reaches the children. The harness started under
  `tools/nanochat train --` and both stage executables passed
  `RequireSandboxOrDie` (D10).
- The chain crosses stages. The driver accepted the base checkpoint as a warm
  start, and its recorded config matched the SFT argv (B1, B2).
- The delivery path works. `<out_root>/deliver/` holds the copied checkpoint and
  the report.

A second premise needs no card: `NANOCHAT_SANDBOX_BACKEND=none tools/nanochat
client --profile train -- ...` sets `NANOCHAT_SANDBOX=off`, so the container
entry of D10 satisfies the child guard.

### 13.2 Phase 0 on Modal: the same harness, two images

The Modal run answers the questions the local host cannot: does the image build
the two binaries, does the CUDA build load on a T4, does the tokenizer resolve
from `NANOCHAT_BASE_DIR` inside a Volume, and how long does each stage take on a
card.

```bash
pip install modal && modal setup          # once, per account
modal deploy /tmp/pipeline-spike/app.py   # builds both images, cached after
modal run /tmp/pipeline-spike/app.py::probe_cpu
modal run /tmp/pipeline-spike/app.py::probe_gpu
modal run /tmp/pipeline-spike/app.py::chain_cpu
modal run /tmp/pipeline-spike/app.py::chain_gpu
```

The Volume is `nanochat-spike-runs`, so a spike cannot touch `nanochat-runs`.
Both functions carry `max_containers=1`, and each starts its work through
`tools/nanochat client --profile train`, which is the container entry of D10.

Pass criteria per leg: `returncode` 0; the three stages record `base`@100,
`sft`@100, `rl`@105; the two config blocks are equal; the L2 norm differs between
neighbouring stages; `total_seconds` is under 60 s; and `probe` reports the
expected `backend`, `arch`, and both binaries present. The RL advantage stays
`0.0` until B4 lands, on every host; that is expected, not a failure.

### 13.3 Phase 1: the module, after the blockers

After rows 1 to 8 of section 8, the same two legs run `pipeline.run(job)` instead
of the harness, and the gate list of section 7 becomes the pass criteria. The
harness then retires: it exists to de-risk the environment before the module
exists, and it must not become a second implementation.

Two Modal rounds are therefore expected: one now for the environment, one after
the blockers for the module. The alternative is one round, taken after the
blockers land, and it is cheaper. Take it if the environment questions can wait;
take phase 0 first if a Modal surprise would change the plan.

### 13.4 What the Modal run found

Run on 2026-10-31 with Modal client 1.6.3, the demo profile, and the harness of
sections 13.1 and 13.2. Two images, one per backend, and the Volume
`nanochat-spike-runs`.

| Leg | Result | Evidence |
|---|---|---|
| CPU probe | Pass | Library loads, `backend_reported` `cpu`, both binaries built, no `nvidia-smi` |
| CPU chain | Pass | 100/100/5 in 6.07 s, `base`@100, `sft`@100, `rl`@105, `max_abs_advantage` `0.0` |
| T4 probe | Pass | `Tesla T4`, compute capability `7.5`, library loads, both binaries built |
| T4 chain | Fail | `Segmentation fault` in `MeshTrainConfig.run`, first base step |

Four findings that the plan now carries:

1. **The image builds the stage binaries.** Both `sft_main` and `rl_worker` exist
   after the image build, on both backends. B6's premise holds, and the Modal
   line of B6 is one `run_commands` argument.
2. **The container entry satisfies the child guard.** The chain log opens with
   `sandbox: backend=none profile=train (not sandboxed)`, and both stage
   executables start. D10 holds.
3. **The CPU leg is hermetic and fast.** 6.07 s in a two-core container, with the
   same base loss trace as the local host, 6.1874 to 0.0049.
4. **The GPU leg is blocked by B7.** The T4 cannot start, and the same failure
   reproduces on the local 1080 Ti, so it is a tree defect and not a host one.

Two smaller findings, both now fixed in the spike and worth fixing in the tree:

- `.bazelrc.local` must not travel into an image. The host file pins
  `/usr/bin/gcc-12`, and the first CPU image build died because an Ubuntu 22.04
  image has no such compiler. `tools/modal/app.py` copies the tree with the same
  ignore list, so the defect is latent there too: a deploy from a host that ran
  `tools/kaggle/bootstrap.sh` builds an image that cannot compile.
- The committed parquet holds too few tokens for a 64-token batch of 2. The base
  stage ran zero steps and exited 0. A stage that silently shortens the schedule
  is exactly what assertion A1 must catch, so the harness now fails on it.

### 13.5 After the B7, B8, and B9 fixes

Re-measured on 2026-10-31, same harness and same images:

| Leg | Before | After |
|---|---|---|
| 1080 Ti `train_main` | cuBLAS fast-accum abort, then a segfault in `QkPrepForward` | gate skips 6 fp8 cases, forward runs, then B10 |
| T4 chain | `Segmentation fault`, first base step | `fatal: CUDA error in cudaMemcpy`, first base step |
| CPU chain | green | green, 3 steps at 0.86 s, same L2 per stage as before |

The three fixes are independent, and each one removed exactly one fault. What
remains is B10, and it is a pre-existing defect in the CUDA train step: the
tree's own `//src:runtime_gpu_test` fails with the same message. So the GPU
column of the host matrix is red for a reason that has nothing to do with Modal,
Colab, or this plan.

### 13.6 The GPU column is green

The same demo job on the pinned card, after the four fixes:

| Item | Modal CPU | Modal T4 |
|---|---|---|
| `returncode` | 0 | 0 |
| Stages | `base`@100, `sft`@100, `rl`@105 | `base`@100, `sft`@100, `rl`@105 |
| L2 per stage | 524.4127, 515.7249, 513.6772 | 525.0440, 514.2361, 512.2022 |
| Total | 6.07 s | 8.60 s |
| Base loss | 6.1874 to 0.0049 | 6.1874 to 0.0049 |
| `max_abs_advantage` | 0.0 | 0.0 |

The two L2 columns differ in the third digit, which is the arithmetic order of
the two backends, not a divergence: both stages record the same chain, the same
config, and the same monotone L2 movement. The `0.0` advantage is B4 on both
hosts, so it is the one item this run cannot sign off.

Local checks after the last change: `//src:runtime_gpu_test` passes on the
1080 Ti, `train_main --hw-check auto` runs two steps on the card, and
`//tests:sft_step_parity_test`, `//tests:sft_loop_parity_test`, and
`//src:harness_test` pass on the CPU.

What this buys the plan: the four defects that stood between the tree and a
GPU host are closed, and the remaining blockers (B1 to B6) are the Python-side
gaps the plan was written for. The CPU and T4 legs can now be re-run as a
regression check after each row of section 8.

### 13.7 Three T4 runs, and what they say

| Run | `returncode` | Base L2 | SFT L2 | RL L2 |
|---|---|---|---|---|
| `spike-gpu-b10` | 0 | 525.0440 | 514.2361 | 512.2022 |
| `spike-gpu-r1` | 0 | 525.0414 | 514.5446 | 512.5101 |
| `spike-gpu-r2` | 0 | 525.0388 | 514.5324 | 512.4969 |

All three finish the chain and write the report, so the GPU column is green and
stable in outcome. The L2 spread in the fourth digit is the reduction ordering of
the device kernels: the CPU leg repeats to the last digit, the card does not. A
plan-level consequence, recorded as a risk in section 9: a determinism check that
compares digests belongs on the CPU leg, and a GPU check compares with a
tolerance.

The remaining flake is B11, which the chain does not touch at `world_size=1`.
