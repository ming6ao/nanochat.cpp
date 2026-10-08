"""The trial runner for the Kaggle notebook workflow.

One :class:`Trial` holds the independent knobs. :func:`run_trial` runs one
trial and writes a run directory. :func:`run_sweep` runs a list of trials and
keeps the best checkpoint only. :func:`load_results` reads every run directory
into rows for a table.

The module imports ``nanochat_cpp`` inside the run functions, not at module
load. The pure helpers then import without the shared library. See
``docs/notebook-workflow.md``.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

if TYPE_CHECKING:  # pragma: no cover - typing only
    import nanochat_cpp as nc

__all__ = [
    "BEST_CHECKPOINT",
    "RATE_FIELDS",
    "Prefix",
    "StaleRunError",
    "Trial",
    "TrialResult",
    "changed_fields",
    "fingerprint",
    "is_better",
    "load_results",
    "metric_value",
    "rate_overrides",
    "read_json",
    "resume_decision",
    "run_dir_for",
    "run_sweep",
    "run_trial",
    "slugify",
    "write_json",
]

#: The file name of the best checkpoint under the run root.
BEST_CHECKPOINT = "best.nchkpt01"

#: The ``Trial`` fields that are not rate knobs.
_TRIAL_NON_RATE_FIELDS = frozenset({
    "name", "depth", "seq_len", "window_pattern", "device_batch_size",
    "total_batch_size", "num_iterations", "model_seed",
})

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class StaleRunError(ValueError):
    """A stored run does not match the current trial configuration."""


@dataclass(frozen=True)
class Trial:
    """The independent knobs of one run.

    A ``None`` rate keeps the plan default. A value overrides the base rate.
    ``compute_plan`` then applies the batch scale. See
    ``docs/notebook-workflow.md``.
    """

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


#: The rate fields a trial may override. Each name is a ``compute_plan``
#: keyword. A ``None`` field keeps the plan default.
RATE_FIELDS = tuple(
    field.name for field in dataclasses.fields(Trial)
    if field.name not in _TRIAL_NON_RATE_FIELDS)


@dataclass
class Prefix:
    """The shared, cached inputs that every trial reuses."""

    tokenizer: "nc.Tokenizer"
    train_parquet: str
    val_parquet: str
    data_seed: int = 7


@dataclass
class TrialResult:
    """The outcome of one trial.

    ``model`` is set only for a fresh run with ``keep_model``. ``plan`` is
    always set.
    """

    trial: Trial
    summary: dict
    metrics: list[dict] = field(default_factory=list)
    model: "nc.Model | None" = None
    plan: "nc.plan.TrainPlan | None" = None


def slugify(name: str) -> str:
    """Return a safe run-directory name, or raise ``ValueError``."""
    slug = _SAFE_NAME.sub("-", str(name).strip()).strip("-")
    if not slug or slug in {".", ".."}:
        raise ValueError(f"invalid trial name: {name!r}")
    return slug


def run_dir_for(out_root: str | Path, trial: Trial) -> Path:
    """The run directory of ``trial`` under ``out_root``."""
    return Path(out_root) / slugify(trial.name)


def rate_overrides(trial: Trial) -> dict[str, float | int]:
    """The non-None rate fields, as ``compute_plan`` keyword arguments."""
    return {name: getattr(trial, name)
            for name in RATE_FIELDS
            if getattr(trial, name) is not None}


def _flatten(prefix: str, value: object, out: dict[str, object]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            _flatten(name, item, out)
    else:
        out[prefix] = value


def _flat(mapping: Mapping) -> dict[str, object]:
    out: dict[str, object] = {}
    _flatten("", mapping, out)
    return out


def changed_fields(stored: Mapping, current: Mapping) -> list[str]:
    """The dotted keys whose values differ between two config mappings."""
    left = _flat(stored)
    right = _flat(current)
    changes: list[str] = []
    for key in sorted(set(left) | set(right)):
        if left.get(key, "<missing>") != right.get(key, "<missing>"):
            changes.append(
                f"{key}: {left.get(key, '<missing>')!r} -> "
                f"{right.get(key, '<missing>')!r}")
    return changes


def fingerprint(trial: Trial, rates: Mapping[str, float],
                build_key: str | None = None,
                context: Mapping | None = None) -> str:
    """A short digest of a trial, its rates, the build, and the context."""
    payload = {
        "trial": dataclasses.asdict(trial),
        "rates": {str(key): float(value) for key, value in rates.items()},
        "build": build_key,
        "context": dict(context) if context else {},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def metric_value(summary: Mapping, metric: str = "bpb") -> float | None:
    """The selection value: the metric, then ``final_loss``, then ``None``."""
    value = summary.get(metric)
    if value is None:
        value = summary.get("final_loss")
    return None if value is None else float(value)


def is_better(value: float | None, best: float | None) -> bool:
    """True when ``value`` beats ``best``. Lower is better. A tie keeps best."""
    return value is not None and (best is None or value < best)


def resume_decision(summary_present: bool, stored_fingerprint: str | None,
                    digest: str, force: bool) -> str:
    """Return ``"run"``, ``"resume"``, or ``"stale"`` for one run directory."""
    if force or not summary_present:
        return "run"
    if stored_fingerprint == digest:
        return "resume"
    return "stale"


def read_json(path: str | Path) -> dict:
    """Read a JSON object from ``path``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Mapping) -> None:
    """Write JSON through a temporary file, then rename it into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def load_results(out_root: str | Path) -> list[dict]:
    """Read every run directory under ``out_root`` into one row.

    A failed trial carries ``error`` and hides its earlier summary.
    """
    out_root = Path(out_root)
    if not out_root.is_dir():
        return []
    rows: list[dict] = []
    for run_dir in sorted(path for path in out_root.iterdir() if path.is_dir()):
        error_path = run_dir / "error.json"
        summary_path = run_dir / "summary.json"
        if error_path.is_file():
            row = read_json(error_path)
        elif summary_path.is_file():
            row = read_json(summary_path)
        else:
            continue
        row.setdefault("name", run_dir.name)
        row["path"] = str(run_dir)
        rows.append(row)
    return rows


@functools.lru_cache(maxsize=1)
def _build_key() -> str | None:
    """The library build key, or ``None`` when the accessor is absent."""
    try:
        import nanochat_cpp as nc
        return nc._build.build_key()
    except (AttributeError, ImportError):
        return None


def _revision() -> str:
    """The short git revision of the repository, or ``unknown``."""
    root = Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _read_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _train(trainer, run_dir: Path, tokens_per_step: int) -> list[dict]:
    """Iterate ``trainer`` and write one metrics record per step."""
    rows: list[dict] = []
    previous = time.perf_counter()
    start = previous
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as sink:
        for step, loss in trainer:
            now = time.perf_counter()
            row = {
                "step": int(step),
                "loss": float(loss),
                "tokens_per_sec": tokens_per_step / max(now - previous, 1e-9),
                "seconds": now - start,
            }
            previous = now
            rows.append(row)
            sink.write(json.dumps(row, sort_keys=True) + "\n")
            sink.flush()
    return rows


def run_trial(trial: Trial, prefix: Prefix, out_root: str | Path,
              device: str = "cuda", *,
              keep_model: bool = False, force: bool = False,
              eval_tokens: int | None = None, threads: int = 4,
              build_key: str | None = None) -> TrialResult:
    """Run one trial and write its run directory.

    Resume a finished trial when its fingerprint matches. Raise
    :class:`StaleRunError` when the fingerprint differs. Run again when
    ``force`` is true.

    ``model`` is present only for a fresh run with ``keep_model`` set.
    ``plan`` is always present.
    """
    import nanochat_cpp as nc

    out_root = Path(out_root)
    run_dir = run_dir_for(out_root, trial)
    summary_path = run_dir / "summary.json"
    error_path = run_dir / "error.json"

    plan = nc.plan.compute_plan(
        depth=trial.depth, seq_len=trial.seq_len,
        window_pattern=trial.window_pattern,
        vocab_size=prefix.tokenizer.vocab_size,
        device_batch_size=trial.device_batch_size,
        total_batch_size=trial.total_batch_size,
        num_iterations=trial.num_iterations,
        **rate_overrides(trial))

    if eval_tokens is None:
        eval_tokens = plan.device_batch_size * trial.seq_len * 4
    rates: dict[str, float] = dict(plan.optimizer)
    rates.update(dict(plan.scheduler))
    context = {"train": str(prefix.train_parquet),
               "val": str(prefix.val_parquet),
               "data_seed": int(prefix.data_seed),
               "device": str(device),
               "eval_tokens": int(eval_tokens)}
    current = {"trial": dataclasses.asdict(trial),
               "rates": {str(key): value for key, value in rates.items()},
               "context": context}
    digest = fingerprint(trial, rates,
                         build_key if build_key is not None else _build_key(),
                         context)

    stored = read_json(run_dir / "config.json") \
        if (run_dir / "config.json").is_file() else {}
    decision = resume_decision(summary_path.is_file(),
                               stored.get("fingerprint"), digest, force)
    if decision == "stale":
        changes = changed_fields(stored.get("config", {}), current)
        detail = "; ".join(changes) if changes else "no stored config"
        raise StaleRunError(
            f"trial {trial.name!r} differs from its stored result: "
            f"{detail}. Use force=True to run it again.")
    if decision == "resume":
        error_path.unlink(missing_ok=True)
        return TrialResult(trial=trial, summary=read_json(summary_path),
                           metrics=_read_metrics(run_dir), plan=plan)

    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path.unlink(missing_ok=True)
    error_path.unlink(missing_ok=True)
    revision = _revision()
    write_json(run_dir / "config.json",
               {"fingerprint": digest, "config": current})
    (run_dir / "revision").write_text(revision + "\n", encoding="utf-8")

    model = nc.Model(plan.config, device=device, seed=trial.model_seed)
    data = nc.TokenData(parquet=prefix.train_parquet,
                        tokenizer=prefix.tokenizer, seq_len=trial.seq_len,
                        batch=plan.device_batch_size, seed=prefix.data_seed,
                        threads=threads)
    optimizer = nc.Optimizer(model, num_iterations=plan.num_iterations,
                             **dict(plan.optimizer), **dict(plan.scheduler))
    trainer = nc.Trainer(model, data, num_iterations=plan.num_iterations,
                         optimizer=optimizer, grad_accum=plan.grad_accum)

    metrics = _train(trainer, run_dir, plan.total_batch_size)

    bpb = nc.evaluate(model, parquet=prefix.val_parquet,
                      tokenizer=prefix.tokenizer, tokens=eval_tokens,
                      batch=plan.device_batch_size, seed=trial.model_seed,
                      threads=threads).bpb

    last = metrics[-1] if metrics else {}
    summary = {
        "name": trial.name,
        "bpb": float(bpb),
        "final_loss": None if "loss" not in last else float(last["loss"]),
        "tokens_per_sec": float(last.get("tokens_per_sec", 0.0)),
        "seconds": float(last.get("seconds", 0.0)),
        "iterations": len(metrics),
        "parameters": int(model.param_count()),
        "device": str(model.device),
        "model_seed": int(trial.model_seed),
        "data_seed": int(prefix.data_seed),
        "revision": revision,
    }
    write_json(summary_path, summary)
    return TrialResult(trial=trial, summary=summary, metrics=metrics,
                       model=model if keep_model else None, plan=plan)


def run_sweep(trials: Iterable[Trial], prefix: Prefix, out_root: str | Path,
              device: str = "cuda", *, metric: str = "bpb",
              keep_best_checkpoint: bool = True, force: bool = False,
              eval_tokens: int | None = None,
              threads: int = 4,
              build_key: str | None = None) -> list[TrialResult]:
    """Run many trials and keep the best checkpoint only.

    One failed trial does not stop the sweep. The failure becomes an
    ``error.json`` in the run directory and a row in :func:`load_results`.

    The best checkpoint is written to ``out_root/best.nchkpt01`` at the end.
    A resumed trial has no model, so it cannot supply the checkpoint.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    best_path = out_root / BEST_CHECKPOINT
    trials = list(trials)
    seen: dict[str, str] = {}
    for trial in trials:
        slug = slugify(trial.name)
        if slug in seen:
            raise ValueError(
                f"trial names {seen[slug]!r} and {trial.name!r} share the "
                f"run directory {slug!r}")
        seen[slug] = trial.name

    results: list[TrialResult] = []
    best_result: TrialResult | None = None
    best_value: float | None = None

    for trial in trials:
        try:
            result = run_trial(trial, prefix, out_root, device,
                               keep_model=keep_best_checkpoint, force=force,
                               eval_tokens=eval_tokens, threads=threads,
                               build_key=build_key)
        except Exception as exc:  # noqa: BLE001 - isolate one trial
            error = {"name": trial.name,
                     "error": f"{type(exc).__name__}: {exc}"}
            try:
                write_json(run_dir_for(out_root, trial) / "error.json", error)
            except ValueError:
                pass  # an invalid name has no run directory
            results.append(TrialResult(trial=trial, summary=error))
            continue

        value = metric_value(result.summary, metric)
        if is_better(value, best_value):
            best_value = value
            if best_result is not None:
                best_result.model = None
            best_result = result
        else:
            result.model = None
        results.append(result)

    if best_result is not None and best_result.model is not None:
        best_result.model.save(str(best_path))
        best_result.summary["checkpoint"] = str(best_path)
        write_json(run_dir_for(out_root, best_result.trial) / "summary.json",
                   best_result.summary)
    return results
