# The pipeline plan: one chain, three hosts

Status: executed. Rows 0 to 11 have landed, and section 3 records what each
blocker's fix turned out to be. The owners of D9 and the four review decisions
of section 11 are settled. Two small questions remain, and neither blocked a
row.

Extended: [pipeline-scaleup-plan.md](pipeline-scaleup-plan.md) took the chain to
a bigger shape, a fourth stage, and a second card. D3 of section 4 is now a
sequence of two legs, section 5 holds the `standard` profile and the longer
schedule, section 6.3 holds the two Modal legs, section 7 holds A10 to A13, and
section 11 holds the decisions that came with the extension. The measured
numbers of section 2 and section 6.3 are the demo-shape runs of this plan, and
they stay exactly where they are: they are a record, not a forecast.

This plan extends [post-training.md](post-training.md), which holds the RL design,
and [host-portability.md](host-portability.md), which holds the host contract.
This document covers the whole chain: base training, supervised fine-tuning
(SFT), RL, then an evaluation, on a local CPU, on Colab with a T4, and on Modal,
where two legs run the same job: a T4 at fp32 and an H100 at bf16.

## 1. Goal

One job file runs base training, then SFT, then RL, then an evaluation, on any
of three hosts. Each stage reads the weights of the stage before it. The run
delivers one final checkpoint and one report.

The model is small, so a person can watch the chain end to end in seconds on the
CPU and in under a minute on a card.

Two legs, and not one, run on Modal. The first leg is a T4 at fp32: the pinned
triple of D3, the card a free Colab session hands out, and the cheap leg a
reviewer reads while the second one runs. The second leg is an H100 at bf16:
the same job, its own image, and the precision its own card declares. Each leg
writes one report under its own run id, `<run_id>-t4` and `<run_id>-h100`, so
both stay on the runs Volume and a failed H100 leg leaves the T4 report
readable. The entry point compares the two reports and fails when a gate fails
(section 6.3 and gates G1 to G3).

The fourth stage is the evaluation. It runs in the chain's own process, after
RL, and writes no checkpoint: it scores the base checkpoint and the RL
checkpoint with one evaluator batch, so the report holds both bits-per-byte
numbers side by side (P4 and P5 of
[pipeline-scaleup-plan.md](pipeline-scaleup-plan.md) section 4, and section 5
below).

## 2. What runs today

The chain works by hand at the demo shape of D5. A throwaway harness ran both
hosts on 2026-10-31:

| Leg | Stages | L2 per stage | Seconds |
|---|---|---|---|
| Modal CPU, 2 cores | `base`@100, `sft`@100, `rl`@105 | 524.4127, 515.7249, 513.6772 | 6.07 |
| Modal T4 | `base`@100, `sft`@100, `rl`@105 | 525.0440, 514.2361, 512.2022 | 8.60 |

Four facts come out of those runs:

- The base loss falls from 6.1874 to 0.0049 on both hosts.
- The warm start works, so the recorded config matches the SFT argv.
- `deliver/` holds the checkpoint and the report.
- The largest absolute advantage is `0.0`, which is B4.

Three identical T4 runs gave base L2 525.0440, 525.0414, and 525.0388. The device
kernels reduce in a different order each run, and the seed is not the cause. The
CPU leg repeats to the last digit. So a digest comparison belongs on the CPU
leg, and a GPU comparison uses a tolerance.

## 3. The blockers

Six defects blocked the automated chain. All six have landed, and four more
landed on 2026-10-31. The last column names the row of section 8 that landed
each one, because the reading of a blocker and the fix that landed are not
always the same thing (B5).

| # | Blocker | State |
|---|---|---|
| B1 | The SFT stage cannot set the model shape | landed, row 4 |
| B2 | No model flag sets `value_embedding` | landed, row 2 |
| B3 | The RL stage overwrites the checkpoint it warm starts from | landed, row 5 |
| B4 | Every rollout in a group is identical | landed, row 3 |
| B5 | A nested `tools/nanochat` starts a nested sandbox scope | landed, row 1, with the premise corrected |
| B6 | The two stage executables exist only in a Bazel tree | landed, row 6 |
| B7 | The startup gate probed a fast-accumulate fp8 GEMM on every card | landed |
| B8 | `--hw-check` compared a `const char*` with a literal | landed |
| B9 | The host rope leg read the device table | landed |
| B10 | The CUDA train step built span lists in stack storage | landed |
| B11 | The two-replica path mismatches `lm_head.weight` about once in 18 runs | out of scope, Runtime owner |

### B1. The SFT stage cannot set the model shape

`nc.sft.run` forwards no model flag (`python/nanochat_cpp/sft.py`). The driver
therefore builds the 12-layer default, and `Checkpointer::LoadModel` refuses the
warm start with "the recorded run config differs".

Change: add `config: Config | None = None` to `nc.sft.run`. Move the flag list
out of `rl.worker_argv` into one helper, `api.model_flags(config) -> list[str]`,
and call it from both facades, so the two stages cannot drift. Raise when
`config` and `seq_len` disagree, because the shard owns the sequence length.

Test: a new `//python:sft_test` pins the argument list with a fake toolchain, in
the style of `python/tests/toolchain_test.py`. The facade has no test today.

### B2. No model flag sets `value_embedding`

`nanochat::cli::IsModelFlag` and `ApplyModelFlag` (`src/cli.h`) cover nine
fields. `Config::value_embedding` is not one of them, so no stage command line
reproduces a base run with value embeddings off.

Change: add `--value-embedding <0|1>`, default true, and return false for a bad
value. Add the flag to the usage line of every binary that prints the model
flags; seven binaries print them today. The Python half of the flag list lands
in row 4 with `api.model_flags`, so this row changes no Python file.

Test: extend `TestModelFlagParsing` in `src/harness_test.cc`, which already
covers `ApplyModelFlag` and which `//src:harness_test` runs in the default
suite. Add the new flag, then loop over every flag in `IsModelFlag` and assert
that `ApplyModelFlag` accepts it. A flag that one function knows and the other
does not is then a failure. Do not add a second test target: it would give the
flag table two homes.

### B3. The RL stage overwrites the checkpoint it warm starts from

`nc.rl.run` derives the load path from `checkpoint` when that file exists, and
it saves to the same path (`python/nanochat_cpp/rl.py`). A warm start therefore
destroys its parent.

Change: add `warm_start: str | os.PathLike | None = None`. It becomes the
`--checkpoint` load argument. Keep `checkpoint` as the save target only, and
never derive a load from it. Remove the legacy derivation, because a load path
must not come from the save path. This is a breaking change, and row 11 records
it in `docs/python.md`.

Test: `python/tests/rl_test.py` gains a fake-worker case. `warm_start` must
reach the worker as `--checkpoint`, and the `save` message must name
`checkpoint`. A second case passes `checkpoint` alone and asserts that the
worker gets no load flag.

### B4. Every rollout in a group is identical

`SampleToken` builds a fresh generator from `params.seed` on every call
(`src/generate.cc`, `Rand local(params.seed)`). `GenerateBatch` copies one
`SampleParams` for every row and keeps one seed for every decode step. So row
`r` and row `r + 1` draw the same quantile at the same step, forever.

Consequence: the four samples of a group hold the same text, every advantage is
zero, and the RL stage trains on nothing. A graded reward and a constant reward
gave the same final parameters to the last bit.

Change: mix the row index and the decode step into the seed before each `Decode`
call in `GenerateBatch`:

```
sample.seed = MixSeed(params.seed, r, step)
```

`MixSeed` is a file-local helper in `src/generate.cc`, so no frozen header
moves. `include/nanochat/model.h` and `include/nanochat/sampler.h` stay as they
are. The greedy path is unchanged, because it draws no random value.

`GenerateMultiPrompt` copies `params` per prompt and keeps the same seed
(`src/generate.cc`), so row `r` of prompt 0 and row `r` of prompt 1 draw the same
quantile. Mix the prompt index in the same way,
`per_prompt.seed = MixSeed(params.seed, prompt, 0)`. A single-prompt call is
then unchanged.

Three properties must hold, and all become tests:

- Row independence. Row `r` of a batch equals row `r` of a one-row run at the
  same seed.
- Reproducibility. One seed, one row, one step gives one draw.
- Prompt independence. Row `r` of prompt 0 and row `r` of prompt 1 differ, in the
  same way that two rows of one prompt differ.

Test: `//src:generate_test` gains three cases. Four rows at one seed give four
distinct streams. Row 0 of a one-row batch equals row 0 of a four-row batch. Two
prompts at one seed give different first rows.

`//tests:generate_parity_test` stays green, because its fixture is greedy
(`temperature = 0.0f`). The committed `tests/data/generate_parity.bin` records
greedy output only, and `rl_parity.bin` replays recorded rows, so no fixture
changes. Confirm this with a grep before row 3 lands.

The blast radius is wider than the RL rollout path. Every recorded sampled
evaluation number moves once, and `ChatEvaluator` is the one consumer that ships
(`docs/eval.md` §4). The edit to parity item E1 is two sentences: the stream is
derived per row, per prompt, and per step, and sampled evaluation numbers moved
once at this change.

### B5. A nested `tools/nanochat` starts a nested sandbox scope

Landed as row 1, and the premise was narrower than the title. Two facts about
the tree, each one command to check, correct it:

- The entry point already skips its own scope when a sentinel is set:
  `cli.Context.submit` execs the child under `in_sandbox()`, so
  `NANOCHAT_SANDBOX=t0-cpu tools/nanochat train -- /bin/echo ok` prints `ok`
  and starts no profile. The six facades therefore started no second scope, and
  the chain did not have the nesting defect this row was written for.
- The entry point is not a passthrough: `tools/nanochat /bin/echo hi` prints
  `unknown command '/bin/echo'` and exits 2. So handing a raw child argv to it
  cannot be the inside-scope arm.

What was left is the real defect, and it is a smaller one. `Toolchain.gpu` ran
its child by giving the entry point a raw argv, which exited 2 and never ran the
child, and a bare sandbox sentinel still re-entered the *queue*: `gpu` under
`NANOCHAT_SANDBOX`, and `profile`, whose handler submits the job itself.

Change, as it landed: one private helper, `Toolchain._dispatch(direct, wrapped)`,
holds the in-scope branch and the six training facades call it, so the rule is
stated at the facade instead of in a docstring. When `NANOCHAT_SANDBOX` or
`NANOCHAT_GPU_QUEUED` is set, `direct` — one child executable and its arguments
— runs as a child of this process; otherwise `wrapped` runs through
`tools/nanochat`. `profile` passes `direct=None`, because its argv carries flags
and not a child executable, so both of its arms run the entry point.

Test, as it landed: `python/tests/toolchain_test.py` has one case per method,
beside `test_gpu_inside_sandbox_skips_the_queue`.

The row still earns its place in the order, for the reason it always had: one
sandbox profile is the real ceiling for the whole chain once the stage launches
run by hand, and a child that leaves the scope hides its budget from the parent.
It is not what makes the chain run, and the fix that landed is not the one the
defect statement named. That is why the state column of section 3 says so.

### B6. The two stage executables exist only in a Bazel tree

`nc.sft.run` defaults its binary to `bazel-bin/src/sft_main`, and `nc.rl.run`
defaults to `bazel-bin/src/rl_worker`. A wheel install has neither. The Modal
image keeps the tree, so both paths exist there; a Colab clone must build them.

Change: one resolver, `_build.stage_binary(name) -> Path`. It follows the order
that `_lib.load` already implements, not the five-entry list in `docs/python.md`
§3.2, which omits the runfiles tree and the wheel. Row 11 fixes that list.

1. the `NANOCHAT_SFT_MAIN` and `NANOCHAT_RL_WORKER` overrides;
2. the Bazel runfiles tree;
3. `bazel-bin/src/<name>`;
4. `shutil.which("nanochat-" + name)`, for a future wheel;
5. an error that names the build command.

Both facades call it. Add a sibling builder, `_build.build_stages(options)`,
which builds the library and the two binaries in one `tools/nanochat build`
call, so `_build_command` keeps one home for the command line. The Modal image
extends its existing `.run_commands` build call (`tools/modal/app.py`) to name
`//src:sft_main` and `//src:rl_worker`. The Colab setup runs the same call.

Test: a unit test for `stage_binary` with a fake tree, covering the override,
the runfiles miss, and the error text.

### B7 to B10. Landed on 2026-10-31

The four fixes are in the tree, and each one removed exactly one fault.

- B7 gates the fp8 probe and the fp8 path on compute capability 8.9, which
  `docs/fp8-training.md` §5.3 already requires. An unsupported probe is now a
  skip, and no device thread calls `std::exit`.
- B8 compares `std::string(value)` with `"auto"`, `"on"`, and `"off"` in
  `src/train_main.cc`.
- B9 stages the two rope tables to host in both legs of `src/ops.cc`.
- B10 adds a `SpanList` type to `src/ops.cc`, which stages the span pointers, the
  second operand pointers, and the counts to device memory for the life of one
  call.

The gates that prove them: `//src:runtime_gpu_test` passes on a 1080 Ti and on a
T4, `train_main --hw-check auto` runs two steps on both cards, and
`//tests:sft_step_parity_test`, `//tests:sft_loop_parity_test`, and
`//src:harness_test` pass on the CPU.

### B11. Out of scope

`//src:runtime_gpu_test` builds two replicas on one card and compares their
parameters after a step. It passed 17 of 18 runs and failed once with
`replica parameter mismatch: lm_head.weight`. `kernels::Alloc` is
`cudaMalloc`, which is thread-safe, so the per-call staging of B9 and B10 is not
the cause. The mismatch is a nondeterministic difference between two replicas
that share one card, so either the in-process collective or a per-thread buffer
is racy.

This does not block the pipeline: the chain runs `world_size=1`. It blocks any
distributed claim, and it belongs to the Runtime owner. The test is the
reproducer: run it in a loop and keep the failing output.

## 4. Decisions

| # | Decision |
|---|---|
| D1 | One orchestration module, `python/nanochat_cpp/pipeline.py`. Each stage calls an existing facade. It holds no training logic, no model flag list of its own, and no host name. |
| D2 | Every host builds the two stage binaries, through the resolver of B6. A tree build is the only route until the dependency plan publishes a binary wheel. |
| D3 | The pinned triple is T4, one card, `sm_75`, fp32, and it is the first of two legs: a second leg runs the same job on one H100 card, `sm_90`, at bf16, in its own image and under its own run id. A local card uses its own architecture through `NANOCHAT_CUDA_ARCH`. |
| D4 | The corpus is the committed fixture. Base reads `src/parquet/testdata/text.parquet` with `tests/data/loader_tokenizer.nctoken`. SFT reads one shard that `nc.sft.pack` writes, and RL reads four of those conversations. No stage reads the Hub. |
| D5 | The demo shape has 2 layers, hidden 64, 4 heads, 2 key/value heads, sequence 64, and vocabulary 487 padded to 512. It costs about 4 s of CPU for the whole chain. |
| D6 | `deliver/` is the artifact contract, below. |
| D7 | No frozen header changes. B2 lands in `src/cli.h`, B4 stays inside `src/generate.cc`, and B1, B3, B5, and B6 are Python. |
| D8 | Colab builds from source, not from the wheel. A wheel path lands when a binary wheel exists, and B6 makes that a one-line change. |
| D9 | One owner per file, below. |
| D10 | The entry is always wrapped by `tools/nanochat`, in two forms. See below. |
| D11 | A6 compares two runs, and no warm start changes. The optimizer state stays part of what a warm start carries, as `docs/post-training.md` §7 says. Clearing the moments would change the stage for every caller to simplify one test. |

The artifact contract of D6:

```
<run_root>/<run_id>/base/checkpoint.ckpt
<run_root>/<run_id>/sft/checkpoint.ckpt
<run_root>/<run_id>/rl/checkpoint.ckpt
<run_root>/<run_id>/deliver/chatrl.ckpt
<run_root>/<run_id>/deliver/pipeline_report.json
```

The eval stage of section 5 writes no checkpoint and no directory: its whole
record is one `stage` entry inside `deliver/pipeline_report.json`.

The report is the review artifact. It holds these items:

- the stage order: base, SFT, RL, then eval, with the step counts;
- the loss traces and the mean rewards;
- the checkpoint digests and the metadata blocks;
- the seconds and the seeds;
- the model shape and the tokenizer digest;
- the source revision.

D10 has two entry forms. On a host:

```
tools/nanochat train -- python -m nanochat_cpp.pipeline <job>
```

In a container:

```
tools/nanochat client --profile train -- python -m nanochat_cpp.pipeline <job>
```

Two child executables check the sentinel with `RequireSandboxOrDie`, and no host
may satisfy that check with the override.

The owners of D9:

| File | Owner |
|---|---|
| `python/nanochat_cpp/pipeline.py`, `python/tests/pipeline_test.py` | Python surface |
| `python/nanochat_cpp/sft.py`, `rl.py`, `api.py`, `toolchain.py`, `_build.py` | Python surface |
| `src/cli.h` | Harness |
| `src/generate.cc` | Workflow |
| `tools/modal/app.py`, `tools/modal/modal_app_test.py` | Python surface |
| `notebooks/colab_pipeline.ipynb`, `notebooks/colab_setup.py` | Notebooks |
| `docs/**`, `AGENTS.md`, and the root `BUILD.bazel` files | Architect |

A per-directory `BUILD.bazel` belongs to that directory's owner (`AGENTS.md`
§6), so `python/BUILD.bazel` is the Python surface's and `src/BUILD.bazel` is
the owning workstream's.

## 5. The pipeline module

The module as it stands after the extension, whose design is
[pipeline-scaleup-plan.md](pipeline-scaleup-plan.md) sections 5 and 6:

```python
# python/nanochat_cpp/pipeline.py

PROFILES = {
    "demo": Config(num_layers=2, num_heads=4, num_kv_heads=2,
                   hidden_dim=64, seq_len=64, vocab_size=487,
                   padded_vocab_size=512, window_pattern="L"),
    "standard": Config(num_layers=8, num_heads=8, num_kv_heads=4,
                       hidden_dim=512, seq_len=64, vocab_size=487,
                       padded_vocab_size=512, window_pattern="L"),
}
DEFAULT_PROFILE = "standard"

@dataclass(frozen=True)
class Job:                 # JSON round trip; one file for all three hosts
    run_id: str = "demo"
    profile: str = DEFAULT_PROFILE
    base_steps: int = 200
    sft_steps: int = 200
    rl_steps: int = 8
    num_prompts: int = 4
    num_samples: int = 4
    max_tokens: int = 32
    eval_batches: int = 1        # one batch; the corpus allows no more
    eval_samples: int = 2
    seed: int = 42
    out_root: str = ""
    corpus: str = "fixture"      # "fixture" today; "reference" is a follow-on
    deliver: bool = True
    temperature: float = 1.5
    top_k: int = 0

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
    version: int               # the report schema, 2 today
    job: Job
    stages: tuple[Stage, ...]
    delivered: str
    report_path: str
    total_seconds: float
    host: dict                 # backend, arch, precision, source revision

def stage_base(job) -> Stage
def stage_sft(job, base: Stage) -> Stage
def stage_rl(job, sft: Stage) -> Stage
def stage_eval(job, base: Stage, rl: Stage) -> Stage
def deliver(job, stages, *, out_root) -> tuple[str, str]
def run(job, *, out_root=None) -> Report
def main(argv=None) -> int        # python -m nanochat_cpp.pipeline job.json
```

`demo` is the shape of D5 and it stays, because the CPU test loop runs it in
about five seconds. `standard` is the shape a throughput number is read at, and
`Job()` defaults to it: 8 layers, hidden 512, 8 heads, 4 key/value heads,
sequence 64, the same vocabulary 487 padded to 512, and the `L` window pattern.
The sequence length and the base batch stay where they are, because the
committed corpus fills exactly one 65-token row per epoch
([pipeline-scaleup-plan.md](pipeline-scaleup-plan.md) P1, P2 and section 5).

The schedule grows by about a factor of two: 200 base steps, 200 SFT steps, 8
RL steps, one evaluator batch, and two sampled completions. One step of the
`standard` shape does about ten times the device work of one `demo` step, so a
local CPU chain still costs tens of seconds.

Each stage function is short and calls one facade:

| Stage | Call |
|---|---|
| Base | `runtime.MeshTrainConfig(config(job), steps=job.base_steps, ...)` with `train_parquet`, `tokenizer`, `out_dir`, `checkpoint_path` |
| SFT | `sft.pack(...)` then `sft.run(shard, ckpt, config=config(job), warm_start=base.checkpoint, num_iterations=job.sft_steps)` |
| RL | `rl.run(reward, prompts, num_samples=..., steps=..., config=config(job), warm_start=sft.checkpoint, checkpoint=rl_ckpt, max_tokens=...)` |
| Eval | `api.Model.load(checkpoint)` then `api.evaluate(model, corpus, tokenizer, steps=job.eval_batches, batch=1, seed=job.seed)`, once for the base checkpoint and once for the RL one, then `api.Model.generate` for `job.eval_samples` completions |

`config(job)` is `PROFILES[job.profile]`, so the shape has one home.

The eval stage is the one stage with no executable. It runs in the chain's own
process through the in-process API, so it adds no third stage binary and no new
build target (P4). It writes no checkpoint: its record names the RL file, so
`deliver()` still copies `chatrl.ckpt`. The one limit worth knowing is the
batch: `eval_batches` is 1 and it is not a tunable number in this plan. The
corpus holds 119 tokens, so one epoch fills one 65-token row, and a batch of two
rows cannot form. An evaluator that is asked for two rows reports no bytes and
returns infinity, which is why the module passes `batch=1` by hand: the default
of `api.evaluate` is 8. A tenfold larger corpus would allow more batches, and
that corpus needs a producer, which is a separate work item
([pipeline-scaleup-plan.md](pipeline-scaleup-plan.md) section 6).

The reward is a class with a `reward` method, which `_normalize_task` accepts, so
no Hub dataset loads. It is graded: 1.0 for the right number, 0.5 for the `####`
marker, and otherwise the character overlap with the reference. A graded reward
keeps the advantages from collapsing (Phase 6 of `post-training.md` §12).

The corpus builder copies the committed parquet and tokenizer into
`<run_root>/<run_id>/corpus/`, and writes
`<NANOCHAT_BASE_DIR>/tokenizer/tokenizer.nctoken`, because `data.nctoken1_path()`
reads that exact path. The pipeline never reads `NANOCHAT_BASE_DIR` for output,
and the report names the root it used.

## 6. The three hosts

Every host runs the same four stages of the same job file. The job's counts are
its own: the committed example job of section 5 asks for 200 base steps, 200 SFT
steps, 8 RL steps, and one evaluator batch, and a test job asks for the `demo`
shape and whatever schedule the test pins. The measured numbers quoted in this
section are the 2026-10-31 runs of the `demo` shape at 100, 100, and 5 steps,
and they stay as recorded.

| Host | Backend | Arch | Sandbox | Build | Gate |
|---|---|---|---|---|---|
| Local CPU | `cpu` | none | `train` scope | Bazel | `//python:pipeline_test`, default loop |
| Local GPU | `cuda` | card | `t1-gpu` queue job | Bazel | `//python:pipeline_gpu_test`, `gpu` tag |
| Colab | `cuda` | `sm_75` | `none` | bootstrap + Bazel | manual run; owed, and recorded as owed in `docs/host-portability.md` section 9.2 |
| Modal, T4 leg | `cuda` | `sm_75` | `none` | image build, fp32 | `modal run ...::pipeline_demo`; recorded on 2026-10-31, with the host block of `docs/host-portability.md` section 9.3 |
| Modal, H100 leg | `cuda` | `sm_90` | `none` | image build, bf16 | the same entry point; G1 to G3 pass, with the host block of `docs/host-portability.md` section 9.3 |
| Modal CPU | `cpu` | none | `none` | image build, CPU config | verification only |

The two Modal rows are one run of one job: the T4 leg first, then the H100 leg.

### 6.1 Local CPU and local GPU

```
tools/nanochat build //src:sft_main //src:rl_worker
NANOCHAT_CPP_BACKEND=cpu tools/nanochat train -- \
    python -m nanochat_cpp.pipeline /tmp/job.json
```

One `train` scope, at 6 GB and 800 % CPU, covers all four stages. B5 keeps the
stage children inside that scope: the entry point already skips its own scope
under a sentinel, so a facade runs its child directly instead of re-entering the
entry point, and nothing asks for a second profile (section 3, B5). The eval
stage starts no child at all, so the scope covers whatever it reads.

For the local GPU, run the same command with `NANOCHAT_CPP_BACKEND=cuda` under
`tools/nanochat gpu --profile t1-gpu -- ...`. `tools/nanochat doctor` reports
the name, driver, memory, use, and utilisation, and no compute capability, so the
arch comes from `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`.

Acceptance: the chain finishes, `deliver/pipeline_report.json` exists, the four
stages record the counts the job asked for (the example job: `base`@200,
`sft`@200, `rl`@8, and `eval` on its one batch), and `host.backend` reports the
backend that ran.

### 6.2 Colab with a T4

New `notebooks/colab_pipeline.ipynb`, four cells:

1. Setup: clone the tree, install Bazelisk with `tools/kaggle/bootstrap.sh`, then
   `notebooks/colab_setup.setup(backend="cuda", arch=probe(),
   prefer_installed=False)`, then `_build.build_stages()`. Two calls, not one:
   `setup` resolves the library only.
2. Environment: `NANOCHAT_BASE_DIR=/content/cache`,
   `NANOCHAT_RUNS_DIR=/content/runs`, `NANOCHAT_CPP_BACKEND=cuda`,
   `NANOCHAT_CUDA_ARCH=probe()`, `NANOCHAT_SANDBOX_BACKEND=none`.
3. Run: `tools/nanochat train -- python -m nanochat_cpp.pipeline
   /content/runs/job.json`.
4. Collect: print the report, and offer one download of `deliver/`.

Cell 3 must wrap the module. The backend value satisfies the library check in
`nanochat_init` only: both stage executables call `RequireSandboxOrDie`, which
accepts `NANOCHAT_SANDBOX` or the override and not the backend value. So the
entry is the tool, which sets the sentinel to `off` on a host without systemd.
Do not set `NANOCHAT_ALLOW_UNSANDBOXED` in the notebook: `docs/sandbox.md` and
`docs/host-portability.md` forbid it, and the image has the same boundary.

`probe()` maps the output of `nvidia-smi --query-gpu=compute_cap` to
`sm_<major><minor>`. Free Colab gives a T4, so `sm_75`. A paid session can give
a V100, `sm_70`, or an A100, `sm_80`, and an `sm_75` build then fails inside
`nanochat_init` with a message about the architecture, not about the job.

`prefer_installed=False` is required, not optional. `setup` prefers an installed
distribution by default (`notebooks/kaggle_setup.py`), so a session that already
carries the wheel returns the install root as `repo`. That root has no
`tools/nanochat` and no stage binaries, so it contradicts D8 and breaks D10.

`notebooks/colab_setup.py` is additive: a thin module that calls
`kaggle_setup.setup(...)` with the `/content` defaults, then `build_stages()`.
`kaggle_setup.py` keeps its name, because `notebooks/session.py` and five
documents name it. A rename is a separate, mechanical change, and it is not part
of this plan.

Acceptance: `/content/runs/<run_id>/deliver/pipeline_report.json` exists, it
names four stages with the counts the job asked for and one evaluator batch, and
`host.backend` is `cuda` with `host.arch` equal to the probed value.

### 6.3 Modal: two legs, a T4 at fp32 and an H100 at bf16

`tools/modal/app.py` gains four functions over two images, and the entry point
takes both legs of one job.

| Leg | Card | Arch | Precision | Build command |
|---|---|---|---|---|
| `t4` | `T4` | `sm_75` | `fp32` | `--config=cuda --define=precision=fp32 --@rules_cuda//cuda:archs=sm_75` |
| `h100` | `H100` | `sm_90` | `bf16` | `--config=cuda --define=precision=bf16 --@rules_cuda//cuda:archs=sm_90` |

A frozen `Leg` record holds the name, the card type, the card count, the
architecture, the precision, and the build configuration. `image_for(leg)`
returns one image per leg, and the stage executables, the precision, the
architecture, and the build call all read the leg. One library per image is
what keeps the architecture and the precision honest: one Bazel output base
holds one build of `libnanochat_shared.so` (section 6.4 makes the same argument
for the CPU leg).

The four functions are `train` and `pipeline` on the T4 leg, and `train_h100`
and `pipeline_h100` on the H100 leg. The first two keep their names, because
`notebooks/target.py` looks the name `train` up through
`modal.Function.from_name`. `notebooks/target.py` also gains a `leg` argument on
`ModalTarget`, so a notebook selects one leg.

Each leg mirrors what the single leg did:

- `pipeline` and `pipeline_h100` are thin faces of one `_pipeline_on(leg, ...)`,
  so the two legs cannot drift. The leg selects the image, the card, the
  precision, the architecture, and the roofline device.
- Both rewrite `out_root` to `/runs/<run_id>`, because the app owns the Volume
  path.
- Both start the work as `tools/nanochat client --profile train -- python -m
  nanochat_cpp.pipeline <job.json>`, which is the container entry of D10.
- Both keep `max_containers=1`, `_check_cards(leg, ...)`, the commit keeper, and
  the `worker.log`, and both reuse `_publish` for the Hub push of the final
  stage, so the chain gains no second uploader.
- Both set the same `cpu=` value. Modal sizes the CPU by memory when `cpu=` is
  absent, so two legs with different memory would get different host budgets,
  and gate G2 compares the two base step times: the shared `cpu=` is what makes
  that comparison meaningful.
- Both legs keep the default collective, `nccl`, because the chain runs
  `world_size=1`. The H100 leg must not use the `h100` named config, which is the
  simulator profile and swaps the collective to `native`. The `hopper` config of
  [build.md](build.md) is the same build as the flag triple of the table above,
  and the build helper still emits the triple, never a named config.
- `pipeline_demo` runs the T4 leg first, under the run id `<run_id>-t4`, then
  the H100 leg under `<run_id>-h100`, so both reports stay on the
  `nanochat-runs` Volume and the cheap artifact survives a failure of the
  expensive leg. It prints one document with three keys -- `legs`,
  `comparison`, and `gates` -- and exits non-zero when a gate fails.
- The image build of a leg names `//src:roofline_main` beside the library and the
  two stage executables, so gate G3 runs the tool inside the H100 container and
  starts no second Bazel build there.

Gates G1 to G3 are section 7.1's subject. Their first two-leg run is row 9 of
[pipeline-scaleup-plan.md](pipeline-scaleup-plan.md), and it ran on 2026-10-31:
the T4 leg measured 128.74 ms per base step and the H100 leg 124.39 ms, and all
three gates passed. The chain reports `startup_check: not_run`, because the mesh
trainer does not call the hardware correctness gate; the fp8 path on Hopper needs
its own row.

The client half is `ModalTarget(function="pipeline")`, which needs a `function`
argument that the class does not have today.
`ModalTarget.prepare` reads `output.out_dir` and raises without it
(`notebooks/target.py`), so the target gains one `job_kind` branch whose
`"pipeline"` case reads `out_root`. The job carries no host path either way,
because the app rewrites `out_root` to `/runs/<run_id>`. `target.map_job` and
`target.job_paths` need no new field table, because a fixture job holds no input
path.

Acceptance: `modal run tools/modal/app.py::pipeline_demo` prints one document
whose T4 leg names `host.arch` `sm_75` at fp32 and whose H100 leg names `sm_90`
at bf16, all three gates pass, and the `nanochat-runs` Volume holds `deliver/`
twice, under `<run_id>-t4` and `<run_id>-h100`.

### 6.4 The Modal CPU leg is a second image

The Modal image of `tools/modal/app.py` pins one compiled backend:
`NANOCHAT_CPP_LIB` names the CUDA library and the build passes `--config=cuda`.
A container with no card therefore fails inside `nanochat_init`, and
`_check_cards` calls `nvidia-smi`, which does not exist there.

So a CPU leg needs a second image whose build passes `--config=cpu` and whose
function omits `gpu=`. Do not add a CPU branch to `tools/modal/app.py`. Keep the
two images apart, because one Bazel output base holds only one build of
`libnanochat_shared.so`.

## 7. The integration gate

`python/tests/pipeline_test.py` calls `pipeline.run(job)` with the `demo`
profile. The assertions are the contract, not the loss values. The parity rungs
own the numbers.

| # | Assertion | Catches |
|---|---|---|
| A1 | Each stage returns status 0 with the step count it was asked for | A shortened schedule |
| A2 | Every loss and reward is finite; the base loss falls from step 1 to step 100 | A dead optimizer |
| A3 | Metadata chain: `base`@100, `sft`@100, `rl`@105, and every recorded config equals the job's | B1, B2, a wrong warm start |
| A4 | The parameter L2 norm differs between base and SFT, and between SFT and RL | A stage that only rewrites its file |
| A5 | One group of rollout rewards holds two distinct values, so its advantages are not all zero | B4 |
| A6 | Two runs from the same SFT checkpoint and the same seed, one with the graded reward and one with a constant reward, give different final parameters | A gradient path that the inherited moments hide |
| A7 | `deliver/chatrl.ckpt` equals the RL checkpoint byte for byte, and the report digests match the bytes | A lost artifact |
| A8 | The report names the stage order, the seeds, and the host facts | An unreviewable run |
| A9 | The delivered file loads through the public container reader, and its metadata reports `rl` at step 105 | A container only the pipeline can read |
| A10 | The eval stage ran, both `bpb` values are finite and inside `[0, 8]`, and each sample is non-empty text | A silent eval, and a wrong `eval_batch` |
| A11 | The RL checkpoint digest is the same before and after the eval stage | An eval that writes |
| A12 | The report names the evaluated checkpoint, its digest, the corpus path, the batch size, and the batch count | An unreviewable eval |
| A13 | `deliver/chatrl.ckpt` holds the RL weights, and its metadata reports stage `rl` | A delivery that follows the wrong stage |

A4 does not prove the warm start, because the loaded moments move the weights on
their own. Read A5 through `rl.run(..., on_step=...)`, which receives the per
rollout rewards; `RlReport` carries no advantage.

A6 cannot be "a constant reward leaves the parameters unchanged". A warm start
carries the optimizer state (`docs/post-training.md` §7), so the inherited AdamW
moments move the weights with an advantage vector of exactly zero. A6 is
therefore a two-run comparison, and it is the one assertion that isolates the
advantage term.

A5 and A6 fail today. A4 passes today for the wrong reason. A3 and the argv that
`nc.sft.run` builds are what prove the chain.

A10 to A13 are the eval stage's gate, and they land with the stage
([pipeline-scaleup-plan.md](pipeline-scaleup-plan.md) section 10). A10 reads the
two bits-per-byte numbers and the sampled text out of the eval record, so an
evaluator that reports no bytes, or a batch that cannot form, is a failure and
not a quiet infinity. A11 compares the digest the report carries against the
file after the stage, which is how "the eval writes no checkpoint" (section 5)
stays true. A12 is the reviewability condition: the report names the file it
scored, that file's digest, the corpus, the batch size, and the batch count. A13
is the one that guards the decision to name the RL file in the eval record: the
delivered checkpoint has to be the RL weights at stage `rl`. The batch count of
A12 is one, and a report that claims eight is wrong for the corpus of this plan.

Two targets, one source:

- `//python:pipeline_test`: the `demo` profile, `size = "medium"`, data
  `//bindings:nanochat_shared`, `//src:sft_main`, `//src:rl_worker`, the parquet
  fixture, and the tokenizer fixture. About 5 s, so it stays in the default
  loop. It holds A1 to A13: the `demo` shape and a short schedule, so the eval
  stage runs at the same size as the three training stages.
- `//python:pipeline_gpu_test`: the same source, `NANOCHAT_CPP_BACKEND=cuda`,
  tags `gpu` and `manual`.

### 7.1 The three gates of the two legs

A1 to A13 are the gate of one chain on one host. The two Modal legs need three
more, and they live in `tools/modal/app.py` beside the comparison, not in the
pipeline test: the entry point reads two reports and judges them
([pipeline-scaleup-plan.md](pipeline-scaleup-plan.md) section 8).

| Gate | Claim | Check |
|---|---|---|
| G1 | bf16 ran on the H100 | `host.precision` is `bf16`, the RL checkpoint metadata records `bf16`, and `host.arch` is `sm_90` |
| G2 | the H100 leg did not regress | the H100 base `step_ms` is at most 1.25 times the T4 base `step_ms` for the same job, and both legs ran with the same `cpu=` |
| G3 | the run matches the model | the H100 container ran `roofline --device h100 --precision bf16 --json` for the job shape and recorded `predicted_step_ms` and `limit`; the measured ratio is inside the recorded band |

G2 is a smoke check, not a throughput claim. Both legs are bound by the host
program, so the gate catches a large regression and a broken device path; it
does not measure the card. G3 starts with the band `[1, 200]`, and the first
two-leg run measured a ratio of 162.8. Twice the measured value is wider than
200, so the band stays `[1, 200]` (row 9 of
[pipeline-scaleup-plan.md](pipeline-scaleup-plan.md)).

The document records two numbers that no gate reads: the MFU, and the cross-leg
speed ratio. At 64 tokens per step, one row of sequence 64, the device bound is
about one millisecond inside a step of tens of milliseconds, so the MFU reads
near 0.01 percent; and the chain is bound by launches, so a speed ratio below one
is as correct as one above it. Recording them keeps a reader from reading either
as a pass.

## 8. The order of work

One commit per row. Each row ends with `tools/nanochat check` green. Rows 1 to 6
change no training arithmetic except B4, and B4 changes only the sampled stream,
so the numeric fixtures stay valid.

| # | Work | Gate | Size |
|---|---|---|---|
| 1 | B5, the in-sandbox branch in `Toolchain` | `//python:toolchain_test` | one helper, six call sites |
| 2 | B2, `--value-embedding` | `//src:harness_test` | one flag, one usage line per binary |
| 3 | B4, the seed mix in `GenerateBatch` and `GenerateMultiPrompt` | `//src:generate_test` | four lines, three cases |
| 4 | B1, `config=` on `nc.sft.run`, shared `api.model_flags` | `//python:sft_test` | small |
| 5 | B3, `warm_start=` on `nc.rl.run` | `//python:rl_test` | small |
| 6 | B6, `_build.stage_binary`, `build_stages`, image and Colab build lines | `//python:build_test` | small |
| 7 | `pipeline.py`, the report, and `deliver()` | `//python:pipeline_test` A1 to A4, A7 to A9 | medium |
| 8 | The reward and A5, A6 | `//python:pipeline_test` A5, A6 | small |
| 9 | `notebooks/colab_setup.py` and the Colab notebook | manual run, recorded | medium |
| 10 | The Modal `pipeline` function and its contract test | `//tools/modal:all`, then `modal run` | medium |
| 11 | Documents: `rl-notebook.md`, `host-portability.md` with a Modal host block, `post-training.md` §5, `parity.md` E1, `python.md` §3.1 and §3.2 with the removed RL load, `_lib.py`, `README.md`, `AGENTS.md` owner rows | `tools/nanochat lint` | small |

Row 7 carries one open check: confirm that the committed parquet fills 100 base
steps at the demo shape. It holds 9 rows and about 530 characters, so the data
cursor must cycle epochs for the run to reach step 100. If the loader stops
short, either the corpus grows or `base_steps` falls. A1 is the gate either way.

Row 10 carries a latent image defect. `.bazelrc.local` must not travel into an
image: the host file pins `/usr/bin/gcc-12`, and an Ubuntu 22.04 image has no
such compiler. `tools/modal/app.py` copies the tree with an ignore list, so the
defect is latent there too: a deploy from a host that ran
`tools/kaggle/bootstrap.sh` builds an image that cannot compile.

### 8.1 The rows that followed

[pipeline-scaleup-plan.md](pipeline-scaleup-plan.md) section 9 carries ten more
rows, numbered 0 to 9, and they run before the rows above are extended. They are
what makes the chain bigger and gives it a second leg:

| Row | Work | Gate |
|---|---|---|
| 0 | The non-ASCII characters in the tracked C++ comments become ASCII | `tools/nanochat lint` |
| 1 | The `hopper` named config for a hand run | `//python:build_test` |
| 2 | The init error names the build precision, not a literal `fp32` | `//bindings:nanochat_capi_test` |
| 3 | The `standard` profile, the longer schedule, report version 2, and the example job | `//python:pipeline_test` |
| 4 | The base stage records `step_ms`, `tok/s`, `mfu`, and the hardware-check facts | `//python:pipeline_test` |
| 5 | The eval stage, and A10 to A13 | `//python:pipeline_test` |
| 6 | The two legs, the two images, the four functions, the comparison, and G1 to G3 | `//tools/modal:all` |
| 7 | The client leg argument on `notebooks/target.py` | `//notebooks:all` |
| 8 | These documents | `tools/nanochat lint` |
| 9 | Run both legs, record the numbers, and narrow the G3 band | `modal run tools/modal/app.py::pipeline_demo`, manual |

Rows 0 to 9 have landed. Row 9 ran on 2026-10-31: the `standard` step time is
measured, the G3 ratio is 162.8, and all three gates pass. The band stays
`[1, 200]` because twice the measured ratio is wider. The run also found two
Hopper defects, both fixed (section 7.1 of that document).

## 9. Risks

- **The seed change moves a sampled baseline.** Any recorded sampled generation
  changes. The committed fixtures are safe, because `generate_parity.bin` is
  greedy and `rl_parity.bin` replays rows. Confirm with a grep before row 3.
- **The reward can be uniform.** If the four samples of a group still coincide,
  A5 fails. The graded reward and the per-row seeds make that unlikely, not
  impossible. A5 reports the group size and the distinct reward values, so a
  failure is diagnosable.
- **A6 is two legs.** The second RL run adds about 1 s and doubles the samples
  the assertion sees. Keep `num_samples`, `max_tokens`, the temperature, and the
  `top_k` value pinned in `Job`, and print them in the failure message.
- **Colab has no systemd.** The sandbox is off there, as on Kaggle, so the
  container quota is the only bound. The demo shape is small, and the notebook
  sets `max_tokens` and the step counts.
- **The image build gets slower.** Two more Bazel targets add seconds, and the
  image is already the slow path. The two binaries are small beside the library.
- **A wheel host still cannot run the chain.** B6 makes the error clear and the
  fix one line. The wheel path stays out of scope (D8).
- **The two sandbox checks differ on purpose.** `nanochat_init` accepts
  `NANOCHAT_SANDBOX_BACKEND=none` and `RequireSandboxOrDie` does not. That
  difference lets a notebook call the library while a stage executable still
  refuses to start unwrapped. Do not unify them.
- **The two legs are not bit-identical.** The T4 leg is fp32 and the H100 leg is
  bf16, and the device kernels reduce in a different order on each card, so the
  two runs differ. Compare `bpb` and step times with a tolerance, and never
  compare digests across legs (section 7.1).
- **The second leg costs a second build and a second cached image.** The first
  deploy is slow and later deploys use the cache. The chain itself is under a
  minute on either card, so the cost is the container start, not the training.
- **CUDA training is not bit-reproducible.** Three T4 runs of one job gave base
  L2 525.0440, 525.0414, and 525.0388. The cause is the reduction order of the
  device kernels, not the seed. A digest comparison belongs on the CPU leg, and
  a GPU comparison uses a tolerance.
- **B11 stays open.** It is a flake in the two-replica path, it is outside this
  plan, and it needs an owner before any distributed claim.

## 10. Non-goals

- Distributed training of any stage.
- The Hub corpus kind, and the reference data mixture. `corpus` accepts
  `"fixture"` only in this plan.
- A binary wheel, and a published CUDA wheel per architecture.
- Asynchronous rollouts, clipped proximal policy optimization, and a reference
  model. Those stay in `post-training.md` section 9.
- A performance number. The chain is a correctness and portability gate. The
  extension records a step time per leg and gates the two legs against each
  other (section 7.1), and it still claims no throughput: G2 is a smoke check and
  the cross-leg ratio is information.

## 11. Open questions and review decisions

Decided on review, so no row waits on them:

- The owner of `src/cli.h` is the Harness workstream, which also owns
  `src/train_main.cc`. Row 2 is unblocked.
- The seed mix of B4 lives inside `GenerateBatch`, not in a `SampleParams`
  field, because `SampleParams` sits in a frozen header.
- The legacy RL load from `checkpoint` is removed now, not deprecated. A load
  path must not come from the save path, and the row 11 note in
  `docs/python.md` names the change.
- The owner of `tools/modal/app.py` is the Python surface. Row 10 is unblocked.

Two questions remain, and neither blocks a row:

**Q1. Does the base stage run through the mesh driver or through a new
single-device entry?**
Recommendation: the mesh driver. It owns the loop, the record, the summary, and
the resume, and its parquet paths work today. The pipeline passes
`world_size=1`.

**Q2. Where does the demo job file live?**
Recommendation: in `pipeline.py` as the `Job()` defaults, plus one committed
example at `notebooks/pipeline_job.json`. The test builds the job in code, so a
committed file cannot drift into a gate.

Settled by this plan, and recorded so a later reader does not re-open them:

- The Colab notebook builds from source (D8).
- The GPU legs record a host block, not a baseline report. A throughput number
  that moves is information, not a gate.
- Five RL steps are enough, with A5 and A6 beside them. A4 proves the parameters
  moved, A5 proves the rollouts differ, and A6 proves the movement came from the
  advantages. The extension's job asks for 8; the CPU gate still pins 5.
- A CPU verification leg is a second image, not a branch in `app.py` (6.4).

Settled by the extension, and recorded for the same reason
([pipeline-scaleup-plan.md](pipeline-scaleup-plan.md) section 4, decisions P1 to
P9):

- A second profile, `standard`, exists, and `Job()` defaults to it. `demo` stays
  for the CPU test loop, which pins the profile and a short schedule by hand.
- The sequence length stays 64 and the base batch stays one row. The committed
  corpus fills one 65-token row per epoch, so the evaluator reads one batch,
  `eval_batches` is 1, and it is not a tunable number in this plan (section 5).
- The eval stage runs in process through the API, so it adds no third stage
  executable (P4). It scores the base checkpoint and the RL checkpoint, and its
  record names the RL file, so delivery still copies `chatrl.ckpt` (P5).
- Two legs, not one: T4 first at fp32, then H100 at bf16, each from its own
  image and each under its own run id (P6). A second leg at fp8 is a non-goal.
- The H100 leg runs bf16 and not fp8. The build helper supports bf16 today, the
  chain is launch bound at this shape, and the bf16 leg still runs the fp8
  operators through the startup hardware check, which keys on the card and not
  on the build (P3 and
  [pipeline-scaleup-plan.md](pipeline-scaleup-plan.md) section 4.1).
- The `hopper` named config is a hand-run shorthand only. The build helper keeps
  its explicit flag triple, and the H100 leg must never use the `h100` config,
  which is the simulator (P7 and section 6.3).
- The gates are G1 to G3 of section 7.1: no gate reads an fp8 speedup and no gate
  reads the MFU (P8). G3's ratio band stays `[1, 200]`, because the first H100
  run measured a ratio of 162.8 and twice that value is wider than the band.
- No frozen header changes for the extension. `caps.json` already declares bf16
  for the H100 (P9).

## 12. Proven premises

The harness of section 2 ran before the module exists, so it answers the
environment questions and not the module questions. Four premises hold:

- The sentinel reaches the children. The harness started under
  `tools/nanochat train --`, and both stage executables passed
  `RequireSandboxOrDie` (D10).
- The chain crosses stages. The driver accepted the base checkpoint as a warm
  start, and its recorded config matched the SFT argv.
- The delivery path works. `<out_root>/deliver/` holds the copied checkpoint and
  the report.
- The container entry satisfies the child guard.
  `NANOCHAT_SANDBOX_BACKEND=none tools/nanochat client --profile train -- ...`
  sets `NANOCHAT_SANDBOX=off`.

The image builds both stage binaries on both backends, so the Modal line of B6 is
one `run_commands` argument. The GPU leg is green since the B7 to B10 fixes.

Section 13 of the previous version of this plan held the dated measurement log
and the table of applied review corrections. That history is in the git commits
of this file, and this version keeps only its conclusions.
