# Notebook workflow: hyperparameter and ablation studies

Status: implemented. The runner is `notebooks/lab.py`. The checkpoint policy
is the best only.

This document specifies a productive experiment loop for a full-stack
researcher on a Kaggle notebook. The notebook stays the front end. The run
logic leaves the cells. The content notebook runs one trial and evaluates it.
The ablation sweep is an optional facility of the runner.

## 1. Problem

`notebooks/nanochat-cpp-on-t4-gpu.ipynb` holds the run sequence in cells.
Three problems follow.

- A new experiment means a copy of the notebook. Parallel copies drift.
- The result lives in a cell output. A cell output is not queryable.
- A crash in the last section loses the whole run.

## 2. Goal

One stable notebook. One run function. One run directory per trial. One
results table. An optional trial list drives an ablation sweep.

## 3. Design

### 3.1 The trial is data

`Trial` is a frozen data class. It holds the independent knobs only. The
derived values come from `nanochat_cpp.plan.compute_plan`.

```python
@dataclass(frozen=True)
class Trial:
    name: str
    depth: int = 8
    seq_len: int = 512
    window_pattern: str = "SSSL"
    device_batch_size: int = 8
    total_batch_size: int = 4096
    num_iterations: int = 50
    embedding_lr: float | None = None
    unembedding_lr: float | None = None
    matrix_lr: float | None = None
    scalar_lr: float | None = None
    weight_decay: float | None = None
    warmup_steps: int | None = None
    warmdown_ratio: float | None = None
    final_lr_frac: float | None = None
    model_seed: int = 42
```

Four rules follow.

- Pass each rate as a keyword to `compute_plan`. `compute_plan` applies
  `batch_lr_scale` to every rate and rejects an unknown name. Do not patch
  `plan.optimizer` after the call, or the scale is lost.
- Keep the batch sizes consistent with `seq_len`. `compute_plan` raises when
  `total_batch_size` is not a multiple of `device_batch_size * seq_len`.
- Backend and precision are per process. A process loads the shared library
  once and refuses a second backend. A precision ablation uses one process per
  value. The build cache reuses the compiled library.
- A `None` rate keeps the plan default. A value overrides the base rate. Then
  `compute_plan` applies the scale.

`compute_plan` derives the horizon, the batch size, the learning-rate scale,
and the weight-decay scale. A small search space gives a result sooner. See
`docs/python.md`, section 4.

### 3.2 The shared prefix

The build, the data download, and the tokenizer train are identical for every
trial. They run once per session.

```python
@dataclass
class Prefix:
    tokenizer: nc.Tokenizer
    train_parquet: str
    val_parquet: str
    data_seed: int = 7
```

The prefix is a plain data class, not frozen, because it holds a live
tokenizer handle.

The data seed stays fixed across trials. The model seed varies. Both are
recorded. A fair ablation changes one factor at a time.

The notebook caches the prefix under `/kaggle/working/nanochat-cache`.

### 3.3 The run is a function

`run_trial(trial, prefix, out_root, device="cuda", *, keep_model=False,
force=False, eval_tokens=None, threads=4, build_key=None,
progress=None) -> TrialResult`

The function does this:

1. Compute the fingerprint of the trial and the resolved plan.
2. Check the stored result against the fingerprint (section 3.6).
3. Make the run directory. Write `config.json` and `revision`.
4. Build the plan with `nc.plan.compute_plan`.
5. Build the model, the optimizer, and the trainer.
6. Append one `metrics.jsonl` record per step. Call `progress` after each
   step when you pass it.
7. Evaluate the validation split with `nc.evaluate`.
8. Write `summary.json` through a temporary file and `os.replace`.
9. Return a `TrialResult`.

The function is a module, not a cell. A test can call it.

`progress` is an optional callable `(row, total)`. The runner calls it after
each step. `ProgressReporter` prints step 1, the last step, and every `every`
steps between. Each line shows the loss, the learning rate, the gradient norm,
the throughput, and the MFU. The reporter averages the throughput and the MFU
over the interval. A line before the last step ends with a carriage return, so
a notebook overwrites one line.

### 3.4 The result object

```python
@dataclass
class TrialResult:
    trial: Trial
    summary: dict
    metrics: list[dict] = field(default_factory=list)
    model: "nc.Model | None" = None
    plan: "nc.plan.TrainPlan | None" = None
```

`model` is present only for a fresh run with `keep_model`. `plan` is always
present. A sweep keeps the best model in memory and drops the others, so at
most one model is alive. The demo sets `keep_model=True`, then plots the
metrics and runs the supervised fine-tuning on the same model.

This keeps the plot cell and the fine-tuning cell working. A summary alone
cannot feed them.

### 3.5 The run-directory contract

`out_root / trial.name` holds:

| File | Content |
|---|---|
| `config.json` | The trial fields, the resolved plan rates, and the fingerprint |
| `revision` | The git commit hash |
| `metrics.jsonl` | One record per step: `step`, `loss`, `lr`, `grad_norm`, `tokens_per_sec`, `mfu`, `step_seconds`, `seconds` |
| `summary.json` | `bpb`, `final_loss`, `tokens_per_sec`, `mfu`, seconds, seeds, revision |
| `error.json` | Present only when the trial failed |

Write `summary.json` and `error.json` atomically, through a temporary file and
`os.replace`. A crash then never leaves a partial file. Flush `metrics.jsonl`
after each step. A fresh run removes the old `summary.json` before it trains,
so an interrupted run cannot look finished.

The contract is the point of the design. A crash loses one trial. A resumed
session skips a finished trial. The table comes from the files, not from cell
outputs.

### 3.6 The resume rule

Resume only when `summary.json` exists and its fingerprint matches the current
trial. Raise a clear error on a mismatch, and name the changed fields. Run
again when `force=True`.

The fingerprint holds the trial fields, the resolved plan rates, the library
build key, and the run context. The context holds the train and validation
parquet paths, the data seed, the device, and the evaluation size. The rule
prevents a stale result after a change to any of them with the same trial name.

A change to the training loop in `notebooks/lab.py` is not in the fingerprint.
Use `force=True` after such a change.

### 3.7 The results table

`load_results(out_root) -> list[dict]` reads every `summary.json` and every
`error.json`. The notebook builds a pandas frame and sorts by `bpb`. A failed
trial shows as a row with its error message.

### 3.8 The sweep

`run_sweep` takes the `run_trial` options, plus `metric` and
`keep_best_checkpoint`. The source is `notebooks/lab.py`.

The function catches an exception for each trial, writes `error.json`, and
continues to the next trial. One bad trial does not stop a time-limited
session. A stale trial becomes an error row, not a fatal error.

The sweep writes the best checkpoint only, to `out_root/best.nchkpt01`. A
resumed trial has no live model, so it cannot supply the checkpoint.

### 3.9 The two-card option: deferred

The plan defers the two-card fan-out. The WSL host serializes GPU jobs through
`tools/gpu.sh --device N`. A raw subprocess would bypass that broker. If the
option returns, it must launch each trial through `tools/gpu.sh --device N` or
`tools/nanochat gpu`. On Kaggle the sandbox backend is `none`, so the notebook
process is not brokered. Document the constraint before the feature.

## 4. The notebook shape

The content notebook keeps these sections.

1. Setup: the loader cell, then `kaggle_setup.setup()`.
2. Data and tokenizer: the shared prefix.
3. Demo trial: `run_trial(Trial("demo"), prefix, RUNS, keep_model=True,
   progress=ProgressReporter())`.
4. Pretraining metrics: the loss, the throughput, and the MFU from
   `result.metrics`.
5. Base evaluation: bits per byte on the real ClimbMix train and validation
   splits, and the reference sample prompts.
6. Fine-tune: from `result.model`, `result.plan`, and `prefix.tokenizer`.
7. Chat evaluation: `nanochat_cpp.chat.ChatEvaluator` on the real chat tasks.

The demo path and the evaluation path use the same model. The sweep is
optional. Add a trial list and call `run_sweep` for an ablation study.

## 5. Files

| File | Change | Owner |
|---|---|---|
| `notebooks/lab.py` | New. `Trial`, `Prefix`, `TrialResult`, `ProgressReporter`, `run_trial`, `run_sweep`, `load_results`. | Notebooks |
| `notebooks/lab_test.py` | New. Tests for the pure helpers. | Notebooks |
| `notebooks/kaggle_setup.py` | New. The session setup: fetch, bootstrap, host environment, build. | Notebooks |
| `notebooks/kaggle_setup_test.py` | New. Tests for the pure setup helpers. | Notebooks |
| `notebooks/BUILD.bazel` | New. Two `py_library` targets, two CPU `py_test` targets, and a `test_suite`. | Notebooks |
| `notebooks/nanochat-cpp-on-t4-gpu.ipynb` | Use `lab` for the run sequence. | Notebooks |
| `notebooks/kaggle_demo.py` | Use `nanochat_cpp.sft_data` for the real SFT mixture. Fall back to the offline demo set when the hub is unreachable. | Notebooks |
| `docs/notebook-workflow.md` | This document. | Architect |
| `docs/README.md` | Add the index row. | Architect |

`notebooks/lab.py` imports `nanochat_cpp` inside the run functions, not at
module load. The pure helpers then import without the shared library.

`notebooks/lab.py` stays inside the Notebooks workstream. No `python/` change
is necessary. See `AGENTS.md`, section 1.

## 6. Test strategy

- Pure helpers: the fingerprint, the resume check, the atomic write, the
  summary aggregation, and the name validation. A CPU `py_test`. No GPU. This
  is the T0 loop.
- The end-to-end trial is T2, not T1. T1 is a tiny single-kernel shape. A full
  model, optimizer, and evaluation run is T2. Use the notebook smoke run with
  `num_iterations=2` as the manual check. Gate at T2 for a merge, not T1.
- No change to the C++ tiers.

The repository rule is to test logic on the CPU first. This design follows the
rule. See `AGENTS.md`, section 5.

## 7. Definition of Done

1. `tools/nanochat test` covers the pure helpers.
2. The base and chat evaluations run in the notebook, end to end.
3. A three-trial sweep runs through `run_sweep`.
4. The results table loads from `summary.json` and `error.json`.
5. A second run skips a finished trial.
6. A changed hyperparameter with the same name raises a clear error.
7. The style checker passes on the document.

## 8. Risks and open questions

| Risk | Effect | Response |
|---|---|---|
| Session time limit | A long sweep stops halfway | Resume by fingerprint within a session. Order the trials by value |
| Output quota | Many checkpoints fail to save | Save the best checkpoint only |
| Stale resume | A wrong result after a rate change | Compare the fingerprint. Raise on a mismatch |
| Rate override | A silent error | Pass the rate to `compute_plan`, which validates it |
| Data seed | An unfair comparison | Fix the data seed. Vary the model seed only |
| Precision ablation | One process, one backend | Use one process per precision value |
| Name path escape | A write outside `out_root` | Slugify and validate the name |
| Two-card fan-out | A bypass of the GPU broker | Deferred. Specify the broker path first |
| Module location | `notebooks/` has no Bazel coverage | Add `notebooks/BUILD.bazel` |

Decision: `notebooks/lab.py` lives under the Notebooks workstream. It is
specific to the Kaggle demo. No `python/` change is necessary.

## 9. Non-goals

- No distributed training.
- No notebook-to-script conversion.
- No C++ change.
- No new third-party runtime dependency. `pandas` and `matplotlib` stay
  notebook-side.
- No automatic checkpoint retention policy in this phase.
