# Python API

Status: implemented. This document replaced `python-bridge.md` and
`python-api.md`, which are removed. It specifies one Python surface for
`nanochat.cpp`.

## 1. Purpose

The package supports one Python surface with three layers. Two layers run in
the notebook process. The third layer uses the shell entry point.

| Layer | Module | Task |
|---|---|---|
| Compute | `nanochat_cpp.api` | Train, evaluate, and generate in one process. |
| Planning | `nanochat_cpp.plan` | Derive the model shape, the horizon, the batch size, and the rates. |
| Orchestration | `nanochat_cpp.toolchain` | Call `tools/nanochat` for build, test, lint, and doctor. |

The command-line mirrors (`python -m nanochat_cpp.base_train`, `base_eval`,
and `chat_eval`) are not part of the surface. The `tools/nanochat_cpp` wrapper
is not part of the surface. The shell tool `tools/nanochat` stays the only
process entry point. `AGENTS.md` requires that.

## 2. Package layout

```text
python/nanochat_cpp/
  __init__.py        public names
  api.py             compute layer (unchanged)
  plan.py            planning layer (new)
  chat.py            chat evaluation (new)
  toolchain.py       orchestration layer (new)
  _entry.py          the command table for toolchain (new)
  checkpoint.py      reference .pt to NCHKPT01 (tool only)
  data.py            NCTOKEN1 reader and parquet listing (internal)
  tasks.py           ARC, MMLU, GSM8K, HumanEval (internal)
  eval_fixture.py    fixture wire format (test only)
  reference.py       reference checkout lookup (test only)
  _core.py           ctypes declarations (internal)
  _lib.py            library search and load (internal)
  _build.py          on-demand builder (internal)
```

## 3. Compute layer

The compute layer does not change. It holds `Config`, `Tokenizer`, `Model`,
`Optimizer`, `TokenData`, `Trainer`, `Evaluator`, `no_grad`, and `build`. It
loads the shared library with `ctypes`. It never imports `torch`.

### 3.1 Sandbox rule

The C entry points call `nanochat::RequireSandboxOrDie`. The shared library
keeps the same guarantee. `nanochat_init` checks three conditions at start:

- The environment defines `NANOCHAT_SANDBOX`, or
- `NANOCHAT_SANDBOX_BACKEND` holds the value `none`, or
- `NANOCHAT_ALLOW_UNSANDBOXED` holds a true value.

The library stops with a corrective message when the three checks fail. A
Kaggle Notebook sets the `none` backend. A workstation run goes through
`tools/nanochat`. Section 5.4 gives the chat evaluation route.

### 3.2 Kaggle delivery

The Kaggle install cell sets the library, the cache, and the sandbox backend:

```python
import os, sys
root = "/kaggle/input/nanochat-cpp"
sys.path.insert(0, f"{root}/python")
os.environ["NANOCHAT_CPP_LIB"] = f"{root}/lib/libnanochat_sm75_fp16.so"
os.environ["NANOCHAT_CPP_CACHE"] = "/kaggle/working/.nanochat_cpp"
os.environ["NANOCHAT_SANDBOX_BACKEND"] = "none"
import nanochat_cpp as nc
```

The library search order is:

1. the path in `NANOCHAT_CPP_LIB`;
2. a cached library that matches the build key;
3. the development tree under `bazel-bin/bindings/`;
4. the package directory;
5. a build from the sources.

## 4. Planning layer

The reference plan needs two model-derived numbers:

- the scaling parameter count (`transformer_matrices + lm_head`);
- the FLOP estimate for one token.

The reference computes both in PyTorch. The package must not import PyTorch.
So these two numbers come from C. The rest of the plan is arithmetic with no
torch, so it stays in `plan.py`.

### 4.1 The C surface

Add one struct and one function to `include/nanochat/capi.h`:

```c
// Model-derived planning inputs (docs/python.md section 4.1). `total` counts
// every allocated parameter, including the token and value embeddings.
// `transformer_matrices` and `lm_head` mirror the reference
// GPT.num_scaling_params().
typedef struct {
  int64_t total;
  int64_t transformer_matrices;
  int64_t lm_head;
  int64_t embeddings;
  int64_t scalars;
  double flops_per_token;
} nanochat_params;

// Fills `out` from `config`. Allocates no model. Returns a status. The shim
// calls CountParams and EstimateFlopsPerToken.
nanochat_status nanochat_params_get(const nanochat_config* config,
                                    nanochat_params* out);
```

The C++ side gains one function. Reuse the existing FLOP estimate.

```c
// include/nanochat/mfu.h, or a new harness header. Defined in src/train.cc.
struct ParamBreakdown {
  int64_t total;
  int64_t transformer_matrices;
  int64_t lm_head;
  int64_t embeddings;
  int64_t scalars;
};

ParamBreakdown CountParams(const Config& config);
// Already declared in include/nanochat/mfu.h:16.
double EstimateFlopsPerToken(const Config& config);
```

The design must state which reference function each count mirrors.
`CountParams` mirrors `GPT.num_scaling_params()`. `NumMatmulParams`
(`src/train.cc:114`) mirrors `GPT.num_matmul_params()`. The two are different
functions, and they can drift. A fixture pins both.

`CountParams` is a structural count of `Config`. It is not a walk of runtime
parameter names. It lands in harness-owned `src/train.cc`, beside the
file-local `NumMatmulParams`.

### 4.2 Build wiring

The narrow call keeps the build wiring small.

| File | Change |
|---|---|
| `include/nanochat/capi.h` | Add `nanochat_params` and `nanochat_params_get`. Architect-owned. |
| `bindings/nanochat_capi.cc` | Implement the wrapper over `CountParams` and `EstimateFlopsPerToken`. |
| `bindings/nanochat_capi_test.cc` | Call `nanochat_params_get` in the drift test. |
| `src/train.cc` | Add `CountParams`, beside `NumMatmulParams`. Harness-owned. |
| `include/nanochat/mfu.h` | Declare `CountParams`, or add a new harness header. |
| `python/nanochat_cpp/_core.py` | Declare the struct and the function. |
| `src/BUILD.bazel` | No change when `CountParams` lands in `src/train.cc`. |
| `bindings/BUILD.bazel` | No change. `//bindings:nanochat_shared` already deps `//src:model`. |

### 4.3 Parameter categories and parity fixture

`CountParams` mirrors the reference `num_scaling_params`. `NumMatmulParams`
mirrors `num_matmul_params`. The two functions use different groups, and they
can drift. `total` counts every allocated parameter. It equals
`embeddings + transformer_matrices + lm_head + scalars`. The reference asserts
this sum against `sum(p.numel() for p in self.parameters())`
(`nanochat/gpt.py:408`).

The category assignment follows `nanochat/gpt.py:390-420`:

| C++ name | Category |
|---|---|
| `transformer.h.N.*` (attention and MLP) | `transformer_matrices` |
| `transformer.h.N.attn.ve_gate.weight` | `transformer_matrices` |
| `lm_head.weight` | `lm_head` |
| `transformer.wte.weight` | `embeddings` |
| `value_embeds.N.weight` | `embeddings` |
| `resid_lambdas`, `x0_lambdas` | `scalars` |
| `smear_gate.weight` | `scalars` |
| `smear_lambda`, `backout_lambda` | `scalars` |

Two traps follow from the reference:

- `ve_gate` sits inside `transformer.h`, so it belongs to
  `transformer_matrices`, not to `scalars`.
- `smear_gate` is a top-level `Linear`. `num_scaling_params` puts it in
  `scalars`, but `num_matmul_params` counts it as a matrix. `CountParams`
  follows `num_scaling_params`.

A new tool `tools/dump_plan_fixture.py` writes `tests/data/plan_fixture.bin`
from the reference. The Oracle workstream owns the tool and the fixture
(`AGENTS.md` section 1). The test `plan_test.py` compares every field. The
existing `tools/dump_train_fixture.py` holds token and loss data only, so it
cannot pin the plan.

### 4.4 The Python surface

```python
@dataclasses.dataclass(frozen=True)
class TrainPlan:
    config: Config
    padded_vocab_size: int
    device_batch_size: int
    total_batch_size: int
    grad_accum: int
    num_iterations: int
    target_tokens: int
    scaling_params: int
    flops_per_token: float
    batch_lr_scale: float
    optimizer: Mapping[str, float]   # the five base rates, scaled
    scheduler: Mapping[str, float]   # warmup, warmdown, final fraction

def params(config: Config) -> Params:
    """Call nanochat_params_get and return the two C numbers."""

def compute_plan(depth: int, *, aspect_ratio: int = 64, head_dim: int = 128,
                 seq_len: int = 2048, vocab_size: int,
                 device_batch_size: int, window_pattern: str = "SSSL",
                 num_kv_heads: int | None = None, rope_base: float = 100000.0,
                 total_batch_size: int = -1, num_iterations: int = 0,
                 target_flops: int = 0,
                 target_param_data_ratio: int = 12,
                 **rates) -> TrainPlan:
    """Build the Config, read the two C numbers, and derive the plan."""
```

The Python layer builds `Config` from the arguments. It calls `params` for the
target depth and for depth 12. It keeps the policy constants `B_REF`, the
`0.383` exponent, and the weight-decay formula, because no C++ code uses them.
It does the horizon, batch, learning-rate, and weight-decay arithmetic.

The reference plan forces `n_kv_head = num_heads`. The C++ `Config` carries
`num_kv_heads`. `compute_plan` must pin that difference.

A notebook uses the result:

```python
plan = nc.plan.compute_plan(depth=8, vocab_size=32768, seq_len=512,
                            device_batch_size=8)
model = nc.Model(plan.config)
optimizer = nc.Optimizer(model, **plan.optimizer, **plan.scheduler)
```

## 5. Chat evaluation

The chat evaluation becomes part of the Python API. The Python side owns the
task datasets, the prompt rendering, the answer extraction, and the metric
rules. The C side owns the forward pass. This split is the current behavior of
`chat_eval.py`, moved under the API.

### 5.1 The Python surface

```python
class ChatEvaluator:
    def __init__(self, model: Model, *, tokenizer: Tokenizer | None = None):
        ...
    def run(self, tasks: Sequence[str] | None = None, *,
            max_problems: int | None = None, seed: int = 0,
            data_dir: str | None = None) -> ChatReport:
        ...
    def task(self, name: str, **options) -> TaskReport:
        ...

@dataclasses.dataclass(frozen=True)
class TaskReport:
    name: str
    correct: int
    total: int
    accuracy: float
    baseline: float
    skipped: bool = False
    error: str | None = None

@dataclasses.dataclass(frozen=True)
class ChatReport:
    tasks: tuple[TaskReport, ...]
    chat_core: float | None          # mean centered accuracy
    categorical_core: float | None   # the ARC and MMLU subset
```

The default task list is `ARC-Easy`, `ARC-Challenge`, `MMLU`, `GSM8K`, and
`HumanEval`. The baselines are ARC and MMLU `0.25`, and GSM8K and HumanEval
`0.0`. These rules mirror the reference post-training loop.

### 5.2 The two task paths

| Task type | Tasks | C call |
|---|---|---|
| Categorical | ARC-Easy, ARC-Challenge, MMLU | `Evaluator.score` with a focus set |
| Generative | GSM8K, HumanEval | `Model.generate` |

`nanochat_cpp/tasks.py` stays as the internal module. It holds the datasets,
`render_mc`, `extract_answer`, `extract_program`, and the HumanEval runner.

### 5.3 HumanEval and failure isolation

HumanEval runs generated code. The runner calls
`nanochat.execution.execute_code` from the reference package. That import is
lazy. `docs/eval.md:167-173` documents it as the vetted guarded executor. A
local runner needs its own sandbox story, so it is a later change (section 12).

The reference checkout can be absent. `reference.find_reference_repo` raises
`SystemExit` at `python/nanochat_cpp/reference.py:43-46`. `SystemExit` is not
an `Exception`. The loop at `python/nanochat_cpp/chat_eval.py:410-424` has no
per-task handler, so one unavailable task stops the other four.

`ChatEvaluator.run` isolates each task:

- wrap each task in its own handler;
- catch `BaseException`, because `SystemExit` escapes `Exception`;
- re-raise `KeyboardInterrupt` and `GeneratorExit`;
- record a `TaskReport` with `skipped=True` and an `error` string;
- continue to the next task, and return normally.

`chat_core` is a mean over all five tasks (`chat_eval.py:303-310`, `:429-434`).
So `chat_core` stays `None` when any task reports a skip. `categorical_core`
stays valid when its three tasks ran. Both stay `None` rather than averaging
over fewer tasks.

The same isolation policy covers the other stop sites: the vocabulary mismatch
at `python/nanochat_cpp/chat_eval.py:399-402`, and a library load failure.

### 5.4 Sandbox route

A workstation evaluation must run under the sandbox. The module
`nanochat_cpp/chat.py` keeps a `main()` entry point for that purpose. The
orchestration layer starts it:

```python
toolchain.run("t2-parity", ["-m", "nanochat_cpp.chat", "--tasks", "ARC-Easy"])
```

An in-process call also obeys the `nanochat_init` rule. So a bare
`ChatEvaluator` on a workstation stops unless a sandbox is active. A Kaggle
Notebook uses the `none` backend and calls `ChatEvaluator` directly.

## 6. Orchestration layer

The orchestration layer exposes every `tools/nanochat` command as a Python
function. It calls the shell tool through `subprocess`. It never calls Bazel,
the sandbox, or the GPU broker directly.

| Python call | Shell command |
|---|---|
| `build(targets=None)` | `build [targets...]` |
| `test(targets=None, gpu=False)` | `test [--gpu] [targets...]` |
| `check(gpu=False)` | `check [--gpu]` |
| `lint(paths=None)` | `lint [file...]` |
| `shutdown()` | `shutdown` |
| `prune(apply=False, worktree=None)` | `prune [--apply] [--worktree DIR]` |
| `run(profile, argv)` | `run <profile> -- <cmd...>` |
| `train(argv)`, `eval(argv)`, `bench(argv)` | the matching alias |
| `verify(argv)` | `verify -- <cmd...>` |
| `profile(argv=())` | `profile [opts]` |
| `doctor()` | `doctor` |
| `gpu(argv, profile=None)` | `gpu [--profile P] -- <cmd...>` |
| `dispatch(name, *args)` | any command, by name |
| `commands()` | the command list |

The command table in `_entry.py` is the single source of truth. Each command
function reads the table. A new shell subcommand is one table row.

`CommandResult` holds `argv`, `returncode`, `stdout`, `stderr`, and `seconds`.
`ToolError` carries the failing `argv`, the return code, and `stderr`.
`doctor()` returns a `DoctorReport`.

The shell tool gains one flag, `doctor --json`. The Python layer parses the
JSON. A text fallback covers an older shell tool.

When the process already runs inside a sandbox, the layer must not acquire the
broker again. The layer reads `NANOCHAT_SANDBOX`, as `launcher.launch` does
today. The change removes `launcher.py`.

## 7. Checkpoint conversion

`nanochat_cpp/checkpoint.py` stays. It converts a reference `.pt` state
dictionary to the `NCHKPT01` container. It imports `torch` on use only. The
tool `tools/convert_checkpoint.py` stays. The compute API does not import it.

## 8. Removed files

| Path | Action |
|---|---|
| `python/nanochat_cpp/base_train.py` | Remove. |
| `python/nanochat_cpp/base_eval.py` | Remove. |
| `python/nanochat_cpp/chat_eval.py` | Remove. |
| `python/nanochat_cpp/config.py` | Remove. `plan.py` replaces its math without torch. |
| `python/nanochat_cpp/launcher.py` | Remove. The orchestration layer replaces it. |
| `python/nanochat_cpp/selftest.py` | Remove. The `py_test` targets replace it. |
| `tools/nanochat_cpp` | Remove. |
| `tools/nanochat_cpp_selftest.sh` | Remove. A `py_test` target replaces it. |
| `docs/python-bridge.md` | Remove. This document replaces it. |
| `docs/python-api.md` | Remove. This document replaces it. |

## 9. Documentation

This document must also carry the content that the removed documents hold
today. Section 3.1 carries the `nanochat_init` sandbox rule. Section 3.2
carries the Kaggle install cell and the library search order. A later edit
adds the C ABI contract from `python-api.md` section 4.

Update these references.

| File | Change |
|---|---|
| `docs/README.md` | Replace the two Python rows with `python.md`. |
| `docs/eval.md` sections 4 and 5 | Point at the `ChatEvaluator` surface. |
| `docs/post-training.md` | Point at `python.md`. |
| `docs/parity.md` | Point at `python.md`. |
| `docs/testing.md` | Point at `python.md`. |
| `docs/build.md` at line 39 | Remove `base_train.py` and `launcher.py` from the layout list. |
| `docs/cpu-performance.md` | Repoint the command examples. |
| `docs/cpu-baseline.json` at line 32 | Repoint the recorded command. |
| `docs/parquet-native.md` at line 219 | Repoint the command. |
| `README.md` at lines 45 and 54 | Replace the `tools/nanochat_cpp base_train` quickstart. |
| `tools/check_tokenizer_parity.py` at line 12 | Point at `python.md`. |
| `tools/convert_checkpoint.py` | Point at `python.md`. |

## 10. Migration

1. Add `nanochat_params` and `nanochat_params_get` to `include/nanochat/capi.h`.
   Add `CountParams` to `src/train.cc` and declare it in
   `include/nanochat/mfu.h` or a new harness header. Implement the wrapper in
   `bindings/nanochat_capi.cc`. Call it in `bindings/nanochat_capi_test.cc`,
   and declare it in `_core.py`.
2. Add `plan.py`, `tools/dump_plan_fixture.py`, `tests/data/plan_fixture.bin`,
   and `plan_test.py`.
3. Add `chat.py` and a fixture test.
4. Add `toolchain.py`, `_entry.py`, and `doctor --json`.
5. Write this document. Mark the two old documents as deprecated.
6. Remove the files in section 8 and their build targets.
7. Delete the two old documents and fix the references in section 9.
8. Run `tools/nanochat check`.

Steps 1 to 4 add the replacement. Steps 6 and 7 remove the old surface.

## 11. Tests

| Test | Task |
|---|---|
| `python/tests/plan_test.py` | `compute_plan` matches the reference fixture. |
| `python/tests/chat_test.py` | `ChatEvaluator` matches the committed chat fixture. |
| `python/tests/toolchain_test.py` | The argv for every command, against a fake entry script. |
| `python/tests/api_test.py` | Unchanged. |
| `python/tests/core_test.py` | Unchanged. |

### 11.1 Coverage from the removed self-test

The change removes `selftest.py`. Assign each check to a target:

| Removed check | New target |
|---|---|
| Configuration math | `python/tests/plan_test.py` |
| `NCTOKEN1` reader | `python/tests/data_test.py` |
| Checkpoint name, dtype, container, and path | `python/tests/checkpoint_test.py` |
| `eval_fixture` wire format | `python/tests/eval_fixture_test.py` |
| `tasks` prompt and extraction | `python/tests/chat_test.py` |

Update `python/BUILD.bazel` `test_suite(name = "all")` and `tools/BUILD.bazel`
at line 139 when the targets land.

The `toolchain_test.py` target is hermetic. It needs no library, no Bazel, and
no GPU.

## 12. Risks

| Risk | Response |
|---|---|
| The two count functions drift. | One fixture pins `CountParams`, `NumMatmulParams`, and `flops_per_token`. |
| `total` is ambiguous. | Define `total` as every allocated parameter. The fixture pins it. |
| `smear_gate` disagrees between the two reference functions. | Section 4.3 pins the assignment. The fixture pins both functions. |
| HumanEval needs the reference package. | A skipped `TaskReport` and a `None` chat_core. A local runner is a later step. |
| One task failure stops the run. | Per-task `BaseException` isolation in `ChatEvaluator.run`. |
| The orchestration layer duplicates the shell logic. | The layer only builds `argv` and reads the result. |
| The shell text changes. | `toolchain_test.py` checks the argv for every command. |

## 13. Decisions

The council resolved the design decisions. Five owner decisions remain.

Resolved:

1. **D1, the C interface scope.** Use the narrow call. `nanochat_params_get`
   returns the counts and `flops_per_token`. The policy constants stay in
   `plan.py`. Reason: only two functions need torch, and
   `EstimateFlopsPerToken` already exists in C.
2. **D2, HumanEval.** Keep the lazy reference `execute_code` in this redesign.
   Add the per-task isolation in section 5.3.
3. **D3, ownership.** `CountParams` lands in harness-owned `src/train.cc`.
   `include/nanochat/*.h` and `capi.h` stay architect-owned. `bindings/` and
   `python/**` need owner rows.
4. **The parameter categories.** Section 4.3 pins `total` and the category for
   every C++ name.

Owner decisions:

1. Confirm the narrow D1 scope.
2. Confirm that the harness owns `CountParams` in `src/train.cc`. Decide the
   declaration site: `mfu.h` or a new harness header.
3. Add owner rows for `bindings/**` and `python/**` to `AGENTS.md` section 1.
   Name the `src/BUILD.bazel` editor.
4. Confirm that the Oracle workstream owns `tests/data/plan_fixture.bin` and
   `tools/dump_plan_fixture.py`.
5. Decide whether the deferred local HumanEval runner becomes a tracked
   follow-up.
