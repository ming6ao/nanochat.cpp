"""The one-job chain: base training, then SFT, then RL, then eval.

One job file runs the whole chain on any host: a local CPU, a Colab T4, or a
Modal T4 (``docs/pipeline-plan.md`` section 1). Each stage calls one existing
facade and reads the checkpoint of the stage before it:

============================  =========================================
stage                         call
============================  =========================================
``stage_base``                :class:`nanochat_cpp.runtime.MeshTrainConfig`
``stage_sft``                 :func:`nanochat_cpp.sft.pack` then
                              :func:`nanochat_cpp.sft.run`
``stage_rl``                  :func:`nanochat_cpp.rl.run`
``stage_eval``                :func:`nanochat_cpp.api.evaluate` then
                              :meth:`nanochat_cpp.api.Model.generate`
============================  =========================================

The eval stage is the one stage with no executable and no checkpoint: it runs
in this process over the base checkpoint and the RL checkpoint, so the report
holds both numbers and a reader sees the two bits per byte side by side
(design decisions P4 and P5 of ``docs/pipeline-scaleup-plan.md``).

The module holds no training logic, no model flag list of its own, and no host
name (design decision D1). The model shape has one home here, :data:`PROFILES`,
and every stage receives it as a :class:`nanochat_cpp.api.Config`, which turns
it into the model flags the stage binaries parse. The reward is graded, so the
advantages of a rollout group do not collapse into zeros (section 5).

``deliver()`` writes the artifact contract of D6: the RL checkpoint copied byte
for byte to ``deliver/chatrl.ckpt`` and one ``deliver/pipeline_report.json`` a
reviewer reads instead of the run.

The entry point is always wrapped by ``tools/nanochat`` (D10)::

    tools/nanochat train -- python -m nanochat_cpp.pipeline job.json

The module imports only the Python standard library and the sibling
``nanochat_cpp`` modules.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import _build, api, checkpoint, data, rl, runtime, sft, tasks

__all__ = [
    "CORPUS_DIRNAME",
    "DEMO_CONVERSATIONS",
    "GradedReward",
    "Job",
    "PROFILES",
    "PipelineError",
    "REPORT_VERSION",
    "Report",
    "Stage",
    "build_corpus",
    "config",
    "deliver",
    "main",
    "run",
    "stage_base",
    "stage_eval",
    "stage_rl",
    "stage_sft",
]

#: The report schema. It moves only when the meaning of a report field moves,
#: and it is not the checkpoint container version.
#:
#: 2: the eval stage, and the ``step_ms``, tokens per second, ``mfu``, and
#:    hardware check facts of the base stage (``docs/pipeline-scaleup-plan.md``
#:    section 9 rows 4 and 5). No fixture moves: the report version is not the
#:    container version, so an old checkpoint still reads.
REPORT_VERSION = 2

#: The directory the corpus builder fills, under the run directory.
CORPUS_DIRNAME = "corpus"

#: The run root variable of the notebook environments (section 6.2).
RUNS_DIR_ENV = "NANOCHAT_RUNS_DIR"

#: The base directory the tokenizer artifact is published under, because
#: ``nanochat_cpp.data.nctoken1_path()`` reads
#: ``<NANOCHAT_BASE_DIR>/tokenizer/tokenizer.nctoken`` and the RL bridge loads
#: the tokenizer from there.
BASE_DIR_ENV = "NANOCHAT_BASE_DIR"
BASE_DIR_DEFAULT = Path(".cache") / "nanochat"
TOKENIZER_DIRNAME = "tokenizer"

#: The committed fixture corpus (design decision D4), relative to the tree.
FIXTURE_PARQUET = Path("src") / "parquet" / "testdata" / "text.parquet"
FIXTURE_TOKENIZER = Path("tests") / "data" / "loader_tokenizer.nctoken"

#: The conversations the corpus builder writes beside the fixture data.
CONVERSATIONS_NAME = "conversations.json"

#: The global batch of the base stage, in sequences. ``world_size`` is 1, so
#: this is the whole batch of one step. It stays at one row because the
#: committed corpus is small: ``src/parquet/testdata/text.parquet`` holds 10
#: documents and 109 bytes, so one epoch is at most 119 tokens with one BOS per
#: document. A row is ``seq + 1`` = 65 tokens at either profile, so two rows
#: would need 130 tokens and a batch would never form. The loader resets once
#: per step and a fresh epoch always fills one 65-token row
#: (docs/pipeline-plan.md D5, row 7; docs/pipeline-scaleup-plan.md P2).
BASE_BATCH = 1

#: The evaluator batch of the eval stage, in sequences. It is passed by hand
#: because the default of :func:`nanochat_cpp.api.evaluate` is 8: no evaluator
#: call resets the data loader, a fresh epoch fills one 65-token row, and a
#: batch of eight would report no bytes and return infinity
#: (docs/pipeline-scaleup-plan.md section 6).
EVAL_BATCH = 1

#: The micro-batch of the SFT stage.
SFT_BATCH = 2

#: The model shapes, the one home of a shape (design decision D5).
#:
#: ``demo`` is 2 layers, hidden 64, 4 heads, 2 key/value heads, sequence 64,
#: and vocabulary 487 padded to 512. The 487 is the committed fixture
#: tokenizer's id count, so no id is out of range, and the cost is about four
#: seconds of CPU for the whole chain. The default test loop keeps it.
#:
#: ``standard`` is 8 layers, hidden 512, 8 heads, 4 key/value heads, sequence
#: 64, and the same vocabulary 487 padded to 512
#: (``docs/pipeline-scaleup-plan`` P1 and section 5). The sequence length and
#: the base batch stay where they are, because the committed corpus fills
#: exactly one 65-token row per epoch (P2). The window pattern is ``L`` -- one
#: full-attention pass at this length -- at both shapes.
PROFILES = {
    "demo": api.Config(num_layers=2, num_heads=4, num_kv_heads=2,
                       hidden_dim=64, seq_len=64, vocab_size=487,
                       padded_vocab_size=512, window_pattern="L"),
    "standard": api.Config(num_layers=8, num_heads=8, num_kv_heads=4,
                           hidden_dim=512, seq_len=64, vocab_size=487,
                           padded_vocab_size=512, window_pattern="L"),
}

#: The profile a job runs unless it names another one. ``standard`` is the
#: default because it is the shape the throughput facts are read at; ``demo``
#: stays for the CPU test loop (P1).
DEFAULT_PROFILE = "standard"

#: The fixture conversations, in corpus order. Each is a grade-school-math
#: example: the user asks a question and the assistant answer carries the
#: ``####`` marked number the graded reward reads. They are committed here
#: rather than fetched, because no stage of the chain reads the Hub (D4).
DEMO_CONVERSATIONS = (
    {"messages": [
        {"role": "user", "content": "What is 8 * 5?"},
        {"role": "assistant",
         "content": [{"type": "text", "text": "#### 40"}]},
    ]},
    {"messages": [
        {"role": "user", "content": "What is 3 + 4?"},
        {"role": "assistant",
         "content": [{"type": "text", "text": "#### 7"}]},
    ]},
    {"messages": [
        {"role": "user", "content": "What is 12 - 5?"},
        {"role": "assistant", "content": [{"type": "text", "text": "#### 7"}]},
    ]},
    {"messages": [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant",
         "content": [{"type": "text", "text": "#### 4"}]},
    ]},
)


class PipelineError(RuntimeError):
    """A stage of the chain did not produce the artifact the contract names."""


# ---------------------------------------------------------------------------
# The model shape
# ---------------------------------------------------------------------------


def config(job: "Job") -> api.Config:
    """The model shape of the job's profile: the single home of the shape.

    Every stage receives the shape through this function and hands it to its
    facade, which renders it with :func:`nanochat_cpp.api.model_flags`. So a
    shape is written once (design decision D5) and read by every stage.
    """
    try:
        return PROFILES[job.profile]
    except KeyError:
        raise PipelineError(
            f"unknown profile {job.profile!r}; the profiles are "
            f"{sorted(PROFILES)}") from None


# ---------------------------------------------------------------------------
# The job, a stage, and the report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    """One chain: the four schedules, the rollout shape, and the run root.

    A job is plain data with a JSON round trip (:meth:`to_dict`,
    :meth:`from_dict`, :meth:`to_json`, and :meth:`from_json`), so one file
    drives every host. It holds no host name and no host path: ``out_root`` is
    empty on a host that rewrites it, and the app or the notebook fills it
    (design decision D10). ``run_id`` names the directory under ``out_root``.

    The schedule is the one of ``docs/pipeline-scaleup-plan.md`` section 5:
    200 base steps, 200 SFT steps, 8 RL steps, one evaluator batch, and two
    sampled completions. The profile defaults to ``standard``, so a job file
    that names nothing runs the shape a throughput number can be read at.
    """

    run_id: str = "demo"
    profile: str = DEFAULT_PROFILE
    base_steps: int = 200
    sft_steps: int = 200
    rl_steps: int = 8
    num_prompts: int = 4
    num_samples: int = 4
    max_tokens: int = 32
    # The eval stage: one batch of one row over the corpus, and two greedy
    # completions. ``eval_batches`` is one because the corpus holds 119 tokens
    # and one epoch fills one 65-token row, and it is not a tunable number in
    # this plan (section 5).
    eval_batches: int = 1
    eval_samples: int = 2
    seed: int = 42
    out_root: str = ""
    corpus: str = "fixture"
    deliver: bool = True
    # Sampling temperature of the rollouts. It stays above 1.0 because the
    # demo model answers a fixture conversation and emits its stop token, so at
    # 1.0 every rollout of a group scores the same text and every advantage is
    # zero (docs/pipeline-plan.md B4 and section 7 A5). Measured on the local
    # CUDA card: 1.0 and 1.2 give one distinct reward per group, 1.5 gives three
    # of four, and 2.0 gives four of four at the cost of unreadable text.
    temperature: float = 1.5
    top_k: int = 0

    def to_dict(self) -> dict:
        """The job as a plain dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping) -> "Job":
        """Rebuild a job from a dictionary; an absent key keeps its default."""
        fields = {field.name for field in dataclasses.fields(cls)}
        return cls(**{key: values[key] for key in fields if key in values})

    def to_json(self) -> str:
        """The job as one JSON object."""
        return _json_text(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "Job":
        """Rebuild a job from one JSON object."""
        return cls.from_dict(_json_object(json.loads(text), "job"))

    @classmethod
    def read(cls, path: str | os.PathLike) -> "Job":
        """Read a job from the JSON file at ``path``."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def write(self, path: str | os.PathLike) -> Path:
        """Write the job to ``path`` and return it."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json() + "\n", encoding="utf-8")
        return target


@dataclass(frozen=True)
class Stage:
    """The facts of one stage: what it ran, what it wrote, and what it cost.

    ``losses`` is the per-step loss trace, empty when the driver does not
    return one. ``checkpoint`` and ``sha256`` name the artifact and its digest,
    ``metadata`` is the container's run metadata block
    (:func:`nanochat_cpp.checkpoint.read_metadata`), and ``seconds`` is the
    wall-clock time of the stage. ``extra`` carries what only one stage has:
    the corpus and the reward statistics.
    """

    name: str
    steps: int
    losses: tuple[float, ...]
    checkpoint: str
    sha256: str
    metadata: dict
    seconds: float
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """The stage as a plain dictionary; the loss trace is a list."""
        return {
            "name": self.name,
            "steps": int(self.steps),
            "losses": [float(value) for value in self.losses],
            "checkpoint": self.checkpoint,
            "sha256": self.sha256,
            "metadata": dict(self.metadata),
            "seconds": float(self.seconds),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, values: Mapping) -> "Stage":
        """Rebuild a stage from a dictionary."""
        return cls(
            name=str(values["name"]),
            steps=int(values["steps"]),
            losses=tuple(float(value) for value in values.get("losses") or ()),
            checkpoint=str(values.get("checkpoint", "")),
            sha256=str(values.get("sha256", "")),
            metadata=dict(values.get("metadata") or {}),
            seconds=float(values.get("seconds", 0.0)),
            extra=dict(values.get("extra") or {}),
        )

    def to_json(self) -> str:
        """The stage as one JSON object."""
        return _json_text(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "Stage":
        """Rebuild a stage from one JSON object."""
        return cls.from_dict(_json_object(json.loads(text), "stage"))


@dataclass(frozen=True)
class Report:
    """The review artifact of one chain (design decision D6).

    ``job`` is the job that ran, so the seeds and the shapes travel with the
    numbers. ``stages`` is in stage order. ``delivered`` and ``report_path``
    name the two files of ``deliver/``, ``total_seconds`` is the sum of the
    stage times, and ``host`` holds ``backend``, ``arch``, ``precision``, and
    ``revision`` of the machine that ran it.
    """

    version: int
    job: Job
    stages: tuple[Stage, ...]
    delivered: str
    report_path: str
    total_seconds: float
    host: dict

    def to_dict(self) -> dict:
        """The report as a plain dictionary."""
        return {
            "version": int(self.version),
            "job": self.job.to_dict(),
            "stages": [stage.to_dict() for stage in self.stages],
            "delivered": self.delivered,
            "report_path": self.report_path,
            "total_seconds": float(self.total_seconds),
            "host": dict(self.host),
        }

    @classmethod
    def from_dict(cls, values: Mapping) -> "Report":
        """Rebuild a report from a dictionary."""
        return cls(
            version=int(values.get("version", REPORT_VERSION)),
            job=Job.from_dict(values.get("job") or {}),
            stages=tuple(Stage.from_dict(stage)
                         for stage in values.get("stages") or ()),
            delivered=str(values.get("delivered", "")),
            report_path=str(values.get("report_path", "")),
            total_seconds=float(values.get("total_seconds", 0.0)),
            host=dict(values.get("host") or {}),
        )

    def to_json(self) -> str:
        """The report as one JSON object."""
        return _json_text(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "Report":
        """Rebuild a report from one JSON object."""
        return cls.from_dict(_json_object(json.loads(text), "report"))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _json_text(values: Mapping) -> str:
    """One JSON object, indented and with sorted keys, for a stable diff."""
    return json.dumps(values, indent=2, sort_keys=True)


def _json_object(values, label: str) -> Mapping:
    """``values`` when it is a JSON object, and an error otherwise."""
    if not isinstance(values, Mapping):
        raise PipelineError(f"the {label} JSON is not an object")
    return values


def _sha256(path: str | os.PathLike) -> str:
    """The SHA-256 hex digest of one file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runs_dir(job: "Job") -> Path:
    """The directory that holds one directory per run.

    ``job.out_root`` wins; otherwise ``NANOCHAT_RUNS_DIR``; otherwise a
    temporary root, so a test run never writes into the source tree. The report
    names whatever root it used (section 5).
    """
    if job.out_root:
        return Path(job.out_root).expanduser()
    value = os.environ.get(RUNS_DIR_ENV, "").strip()
    if value:
        return Path(value).expanduser()
    return Path(tempfile.gettempdir()) / "nanochat_cpp" / "runs"


def _run_dir(job: "Job") -> Path:
    """The run directory: ``<run_root>/<run_id>`` (design decision D6)."""
    if not job.run_id:
        raise PipelineError(
            "the job has no run_id, so it has no run directory")
    return _runs_dir(job) / job.run_id


def _published_tokenizer() -> Path:
    """The path ``nanochat_cpp.data.nctoken1_path()`` reads.

    The rule is that function's rule, which is the only one that matters to the
    readers: ``<NANOCHAT_BASE_DIR>/tokenizer/tokenizer.nctoken``.
    """
    base = os.environ.get(BASE_DIR_ENV, "").strip()
    root = Path(base).expanduser() if base else Path.home() / BASE_DIR_DEFAULT
    return root / TOKENIZER_DIRNAME / data.NCTOKEN1_NAME


def _metric_records(path: str | os.PathLike) -> tuple[dict, ...]:
    """The per-step records of the mesh driver's ``metrics.jsonl``.

    One JSON object per step (``src/mesh_train.h``), and the driver writes the
    step time, the tokens per second, and the model FLOPs utilization of every
    step beside the loss. A line that does not parse or is not an object is
    skipped, so a truncated final line costs one record and not the file.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return ()
    records: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return tuple(records)


def _metric_losses(records: Sequence[Mapping]) -> tuple[float, ...]:
    """The per-step losses of a driver's metric records.

    A record that carries no ``loss``, or one whose ``loss`` is not a number,
    is skipped, so a truncated final line costs one entry and not the trace.
    """
    losses: list[float] = []
    for record in records:
        if not isinstance(record.get("loss"), (int, float)):
            continue
        losses.append(float(record["loss"]))
    return tuple(losses)


def _mean(values: Sequence[float]) -> float:
    """The arithmetic mean of ``values``, or 0.0 when there is none."""
    numbers = [float(value) for value in values]
    if not numbers:
        return 0.0
    return float(sum(numbers) / len(numbers))


def _stage(name: str, steps: int, losses, path: str | os.PathLike,
           seconds: float, extra: Mapping | None = None) -> Stage:
    """One :class:`Stage` from a finished stage's artifact."""
    artifact = Path(path)
    if not artifact.is_file():
        raise PipelineError(
            f"the {name} stage wrote no checkpoint at {artifact}")
    return Stage(
        name=name,
        steps=int(steps),
        losses=tuple(float(value) for value in losses),
        checkpoint=str(artifact),
        sha256=_sha256(artifact),
        metadata=checkpoint.read_metadata(artifact),
        seconds=float(seconds),
        extra=dict(extra or {}),
    )


def _total_seconds(stages: Sequence[Stage]) -> float:
    """The sum of the stage times, which is the run time of the chain."""
    return float(sum(stage.seconds for stage in stages))


def _rates(records: Sequence[Mapping], seconds: float, steps_ran: int,
           shape: api.Config, steady_mfu: float = 0.0) -> dict:
    """The rate facts of the base stage: ``step_ms``, tokens per second, mfu.

    ``seconds`` is the stage's own wall clock, the number the stage records, so
    ``step_ms`` is that time divided by the steps that ran and the throughput
    is the tokens those steps moved divided by the same time. ``mfu`` is read
    from what the driver wrote and never computed here from a guessed peak: the
    driver turns the shape's FLOPs per token
    (``EstimateFlopsPerToken``) and the measured tokens per second into the
    number (``ComputeMfu``, ``src/mesh_train.cc``), and an unknown device names
    no peak and so records 0.0. The per-step values come first and the run
    summary's steady value is the fallback, so a driver that wrote no metrics
    still reports the number it recorded (``docs/pipeline-scaleup-plan.md``
    section 9 row 4).
    """
    steps = max(1, int(steps_ran))
    elapsed = float(seconds)
    tokens = float(steps) * BASE_BATCH * int(shape.seq_len)
    step_ms = 1000.0 * elapsed / steps
    tokens_per_sec = tokens / elapsed if elapsed > 0.0 else 0.0
    per_step = [float(record["mfu"]) for record in records
                if isinstance(record.get("mfu"), (int, float))]
    mfu = _mean(per_step) if per_step else float(steady_mfu)
    return {
        "step_ms": step_ms,
        "tokens_per_sec": tokens_per_sec,
        "mfu": mfu,
    }


def _hardware() -> dict:
    """The hardware-check facts of the base stage.

    ``backend``, ``arch``, and ``precision`` are the build's own facts, the
    same three the report's ``host`` block carries. ``startup_check`` and
    ``fp8_cases`` say whether the hardware correctness gate ran
    (``src/hw_gate.cc``, ``docs/correctness-gates.md`` section 4).

    The chain's base stage is the mesh trainer. That driver builds its own
    runtime and does not call ``RunHardwareCorrectnessGate``, which is a
    start-up step of the single-device ``TrainRunner`` only. A device report
    therefore says ``not_run`` for both fields and names the reason. Do not
    report ``ran`` from the build facts: the build is not the check.
    """
    backend = _build.backend()
    arch = _build.architecture()
    if backend != "cuda":
        startup = "skipped"
        reason = ("the hardware correctness gate runs only on a device, and "
                  f"this build runs on the {backend or 'unknown'} backend")
    else:
        startup = "not_run"
        reason = ("the chain runs the mesh trainer, which does not call the "
                  "hardware correctness gate")
    return {
        "backend": backend,
        "arch": arch,
        "precision": _build.precision(),
        "startup_check": startup,
        "startup_check_reason": reason,
        "fp8_cases": "not_run",
    }


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------


def build_corpus(job: "Job") -> dict[str, str]:
    """Copy the fixture corpus into the run and publish the tokenizer.

    The corpus is the committed fixture (design decision D4): the parquet and
    the ``NCTOKEN1`` tokenizer of the tree, copied into
    ``<run_root>/<run_id>/corpus/``, plus the fixture conversations beside
    them, so a stage reads only the run directory.

    The tokenizer is also written to the path
    :func:`nanochat_cpp.data.nctoken1_path` reads, because the RL bridge loads
    the artifact from there and a stage cannot be told another path. The
    pipeline never reads ``NANOCHAT_BASE_DIR`` for output: the run root is the
    one output root, and the report names it.

    Returns the paths a stage needs, under the keys ``dir``, ``parquet``,
    ``tokenizer``, ``conversations``, and ``published``.
    """
    if job.corpus != "fixture":
        raise PipelineError(
            f"unknown corpus {job.corpus!r}; the reference corpus is a "
            f"follow-on and only 'fixture' is available")
    root = _run_dir(job) / CORPUS_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    tree = _build.repo_root()
    parquet = root / FIXTURE_PARQUET.name
    tokenizer = root / FIXTURE_TOKENIZER.name
    shutil.copyfile(tree / FIXTURE_PARQUET, parquet)
    shutil.copyfile(tree / FIXTURE_TOKENIZER, tokenizer)
    conversations = root / CONVERSATIONS_NAME
    conversations.write_text(
        _json_text({"conversations": list(DEMO_CONVERSATIONS)}) + "\n",
        encoding="utf-8")
    published = _published_tokenizer()
    published.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tokenizer, published)
    return {
        "dir": str(root),
        "parquet": str(parquet),
        "tokenizer": str(tokenizer),
        "conversations": str(conversations),
        "published": str(published),
    }


def _conversations(corpus: Mapping[str, str]) -> list[dict]:
    """The conversations of a built corpus, in corpus order."""
    payload = json.loads(
        Path(corpus["conversations"]).read_text(encoding="utf-8"))
    corpus_object = _json_object(payload, "corpus")
    return [dict(conversation)
            for conversation in corpus_object["conversations"]]


def _corpus_facts(corpus: Mapping[str, str]) -> dict:
    """The corpus block of a stage's ``extra``: the paths and the digests."""
    return {
        "dir": corpus["dir"],
        "parquet": corpus["parquet"],
        "tokenizer": corpus["tokenizer"],
        "tokenizer_id": data.NCTOKEN1_NAME,
        "tokenizer_sha256": _sha256(corpus["tokenizer"]),
        "published": corpus["published"],
    }


# ---------------------------------------------------------------------------
# The graded reward
# ---------------------------------------------------------------------------


def _reference_text(conversation: Mapping) -> str:
    """The reference answer of a conversation, as text.

    The last message carries the answer: a string, or the parts of a rendered
    GSM8K solution. The parts are joined, so the ``####`` marker is found
    wherever the renderer put it.
    """
    messages = conversation.get("messages") or ()
    if not messages:
        return ""
    content = messages[-1].get("content", "")
    if isinstance(content, str):
        return content
    texts = [str(part.get("text", "")) for part in content
             if isinstance(part, Mapping)]
    return " ".join(text for text in texts if text)


def _overlap(reference: str, completion: str) -> float:
    """The character overlap of two strings, in ``[0, 1]``."""
    if not reference or not completion:
        return 0.0
    matcher = difflib.SequenceMatcher(None, reference, completion,
                                     autojunk=False)
    return float(min(1.0, max(0.0, matcher.ratio())))


class GradedReward:
    """The graded reward of the chain (``docs/pipeline-plan.md`` section 5).

    A constant reward makes every advantage of a rollout group zero, and so
    does an all-or-nothing reward when no rollout is right. This one is graded:

    * ``1.0`` when the number after the ``####`` marker equals the reference
      answer's number;
    * ``0.5`` when the completion carries the marker at all, so a wrong number
      still outscores a rollout that never answers;
    * otherwise the character overlap with the reference answer, in ``[0, 1]``.

    :func:`nanochat_cpp.rl.run` accepts an object with a ``reward`` method
    (``_normalize_task``), so this class loads no Hub dataset and needs no
    registered adapter.
    """

    def reward(self, conversation: Mapping, completion: str) -> float:
        """Score one rollout against the reference answer of its prompt."""
        reference = _reference_text(conversation)
        expected = tasks.extract_answer(reference)
        predicted = tasks.extract_answer(completion)
        if expected is not None and predicted is not None and \
                predicted.strip() == expected.strip():
            return 1.0
        if "####" in completion:
            return 0.5
        return _overlap(reference, completion)


# ---------------------------------------------------------------------------
# The stages
# ---------------------------------------------------------------------------


def stage_base(job: "Job") -> Stage:
    """Run base training over the committed parquet (design decision D4).

    One :class:`nanochat_cpp.runtime.MeshTrainConfig` call with a mesh of one
    device. The driver writes ``metrics.jsonl``, ``summary.json``, and the
    checkpoint, so the loss trace of the stage is read back from the metrics
    file it wrote, and so are the rates: the step time, the tokens per second,
    and the model FLOPs utilization of every step. The stage's ``extra``
    therefore holds the rates and the hardware-check facts beside the corpus
    block (``docs/pipeline-scaleup-plan.md`` section 9 row 4), and no number in
    it is estimated here.
    """
    shape = config(job)
    corpus = build_corpus(job)
    out_dir = _run_dir(job) / "base"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "checkpoint.ckpt"
    started = time.monotonic()
    client = runtime.MeshTrainConfig(
        shape,
        world_size=1,
        per_rank_batch=BASE_BATCH,
        grad_accum=1,
        steps=int(job.base_steps),
        tokenizer=corpus["tokenizer"],
        train_parquet=[corpus["parquet"]],
        seq=shape.seq_len,
        seed=int(job.seed),
        data_seed=int(job.seed),
        log_every=1,
        save_every=0,
        out_dir=str(out_dir),
        checkpoint_path=str(target),
    )
    with client:
        summary = client.run()
        records = _metric_records(client.metrics_path())
        metrics_path = client.metrics_path()
    seconds = time.monotonic() - started
    losses = _metric_losses(records)
    steps_ran = int(summary.steps)
    extra = _corpus_facts(corpus)
    extra.update(_rates(records, seconds, steps_ran, shape,
                        float(summary.steady_mfu)))
    extra.update({
        "mean_loss": float(summary.mean_loss),
        "steps_ran": steps_ran,
        "interrupted": bool(summary.interrupted),
        "metrics": str(metrics_path),
        "hardware": _hardware(),
    })
    return _stage("base", job.base_steps, losses, target, seconds, extra)


def stage_sft(job: "Job", base: Stage) -> Stage:
    """Pack one shard from the corpus, then fine-tune from ``base``'s weights.

    One :func:`nanochat_cpp.sft.pack` call and one
    :func:`nanochat_cpp.sft.run` call. ``config`` carries the model shape, so
    the driver builds the shape the warm start records (B1), and ``warm_start``
    names the base checkpoint as the load path while ``checkpoint`` names the
    SFT file as the save path (B3).

    The SFT driver reports its per-step loss on its own stdout, which the
    facade returns only as an exit status, so the trace of this stage is empty
    and the report says so through ``extra``.
    """
    shape = config(job)
    corpus = build_corpus(job)
    out_dir = _run_dir(job) / "sft"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "checkpoint.ckpt"
    shard = out_dir / "sft_rows.nchsft"
    tokenizer = api.Tokenizer.load(corpus["tokenizer"])
    sft.pack(shard, _conversations(corpus), tokenizer, shape.seq_len)
    started = time.monotonic()
    status = sft.run(shard, target, config=shape, warm_start=base.checkpoint,
                     num_iterations=int(job.sft_steps), batch=SFT_BATCH,
                     seq_len=shape.seq_len, seed=int(job.seed))
    seconds = time.monotonic() - started
    if status != 0:
        raise PipelineError(
            f"the sft stage exited with status {status}; the driver's own "
            f"message is on its stderr")
    extra = {
        "exit_status": int(status),
        "shard": str(shard),
        "shard_sha256": _sha256(shard),
        "warm_start": base.checkpoint,
        "loss_trace": "not returned by nanochat_cpp.sft.run",
    }
    return _stage("sft", job.sft_steps, (), target, seconds, extra)


def _groups(rewards: Sequence[float], size: int) -> list[list[float]]:
    """The rollout rewards of one step, in groups of ``size``."""
    values = [float(reward) for reward in rewards]
    return [values[index:index + size]
            for index in range(0, len(values), size)]


def _dropped_groups(step_rewards: Sequence[Sequence[float]],
                    samples: int) -> int:
    """Count the rollout groups whose advantages are all zero.

    A group whose rewards all agree has zero advantages, which is what defect
    B4 produced for every group and what an ungraded reward produces again. The
    report keeps the count so a reader sees the reward still grades.
    """
    count = 0
    for rewards in step_rewards:
        for group in _groups(rewards, samples):
            if len(set(group)) < 2:
                count += 1
    return count


def stage_rl(job: "Job", sft_stage: Stage) -> Stage:
    """Run the RL stage from the SFT checkpoint, saving to its own file.

    One :func:`nanochat_cpp.rl.run` call: the graded reward, the corpus
    conversations as prompts, and ``warm_start`` naming the SFT checkpoint as
    the load path while ``checkpoint`` names the RL file as the save path, so
    the parent survives the child (B3).
    """
    shape = config(job)
    corpus = build_corpus(job)
    prompts = _conversations(corpus)
    if int(job.num_prompts) > len(prompts):
        raise PipelineError(
            f"the job asks for {int(job.num_prompts)} prompts and the "
            f"{job.corpus} corpus holds {len(prompts)} conversations")
    out_dir = _run_dir(job) / "rl"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "checkpoint.ckpt"
    step_rewards: list[list[float]] = []

    def on_step(step: int, loss: float, grad_norm: float,
                rewards: list[float]) -> None:
        # The rollout rewards of the step, which `RlReport` does not carry: the
        # integration gate reads a group here to see the advantages move.
        step_rewards.append([float(reward) for reward in rewards])

    started = time.monotonic()
    report = rl.run(
        GradedReward(),
        prompts[:int(job.num_prompts)],
        num_samples=int(job.num_samples),
        steps=int(job.rl_steps),
        checkpoint=target,
        warm_start=sft_stage.checkpoint,
        config=shape,
        max_tokens=int(job.max_tokens),
        temperature=float(job.temperature),
        top_k=int(job.top_k),
        seed=int(job.seed),
        on_step=on_step,
    )
    seconds = time.monotonic() - started
    extra = {
        "warm_start": sft_stage.checkpoint,
        "num_prompts": int(report.num_prompts),
        "num_samples": int(report.num_samples),
        "max_tokens": int(job.max_tokens),
        "temperature": float(job.temperature),
        "top_k": int(job.top_k),
        "mean_rewards": [float(value) for value in report.mean_rewards],
        "grad_norms": [float(value) for value in report.grad_norms],
        "rewards": step_rewards,
        "dropped_groups": _dropped_groups(step_rewards, int(job.num_samples)),
    }
    return _stage("rl", job.rl_steps, report.losses, target, seconds, extra)


def _loaded(shape: api.Config, path: str) -> api.Model:
    """One model at the job's shape, loaded from a checkpoint container."""
    model = api.Model(shape)
    model.load(str(path))
    return model


def _bpb(model: api.Model, corpus: Mapping[str, str], tokenizer,
         job: "Job") -> float:
    """The bits per byte of one loaded model over the corpus's parquet.

    ``EVAL_BATCH`` is passed by hand because the default of
    :func:`nanochat_cpp.api.evaluate` is 8 and the committed corpus fills one
    65-token row per epoch; the report names the batch it used (section 6).
    """
    report = api.evaluate(model, corpus["parquet"], tokenizer,
                          steps=int(job.eval_batches), batch=EVAL_BATCH,
                          seed=int(job.seed))
    return float(report.bpb)


def _completions(conversations: Sequence[Mapping], tokenizer, model: api.Model,
                 job: "Job") -> list[str]:
    """``job.eval_samples`` greedy completions of the corpus, decoded to text.

    The prompt is the RL stage's prompt: :func:`nanochat_cpp.rl.render_prompts`
    renders each conversation up to the assistant marker, so the two stages ask
    the model the same question and the samples read as the chain's own output.
    Greedy sampling (``temperature=0.0``) keeps the report reproducible, and
    only the sampled ids are decoded, which is the mask rule of
    :class:`nanochat_cpp.api.GeneratedRow`.
    """
    prompts = rl.render_prompts(list(conversations), tokenizer)
    if not prompts:
        return []
    texts: list[str] = []
    for index in range(int(job.eval_samples)):
        prompt = prompts[index % len(prompts)]
        rows = model.generate(prompt, temperature=0.0,
                              max_tokens=int(job.max_tokens),
                              seed=int(job.seed) + index)
        if not rows:
            continue
        row = rows[0]
        generated = [token for token, mask
                     in zip(row.tokens, row.mask) if mask]
        # A stop token the tokenizer decodes to nothing still names the row, so
        # a sample is never an empty string a reader has to guess at.
        texts.append(tokenizer.decode(generated or row.tokens))
    return texts


def stage_eval(job: "Job", base: Stage, rl: Stage) -> Stage:
    """Evaluate the base checkpoint and the RL checkpoint in this process.

    The one stage with no executable (design decision P4): the model is built
    with :class:`nanochat_cpp.api.Model` at the job's shape, loaded from a
    checkpoint container, and scored with :func:`nanochat_cpp.api.evaluate`.
    Both checkpoints are scored, so the report holds the two bits per byte side
    by side and a reader sees what the RL stage did to the base model (P5).

    No checkpoint is written: the record names the RL file, because
    ``deliver()`` copies the checkpoint of the last stage and the delivered
    artifact has to stay ``deliver/chatrl.ckpt`` with the RL weights. The
    digest in ``evaluated_sha256`` is read after the eval work, so a pass that
    wrote to the container would show up as a mismatch against the RL stage's
    own digest (section 6).
    """
    shape = config(job)
    corpus = build_corpus(job)
    conversations = _conversations(corpus)
    tokenizer = api.Tokenizer.load(corpus["tokenizer"])
    started = time.monotonic()
    model = _loaded(shape, base.checkpoint)
    bpb_base = _bpb(model, corpus, tokenizer, job)
    # The rebinding releases the base model before the RL one is built, so the
    # stage holds one model arena at a time.
    model = _loaded(shape, rl.checkpoint)
    bpb_rl = _bpb(model, corpus, tokenizer, job)
    samples = _completions(conversations, tokenizer, model, job)
    seconds = time.monotonic() - started
    extra = {
        "bpb_base": bpb_base,
        "bpb_rl": bpb_rl,
        "eval_batches": int(job.eval_batches),
        "eval_batch": EVAL_BATCH,
        "eval_corpus": corpus["parquet"],
        "evaluated": rl.checkpoint,
        "evaluated_sha256": _sha256(rl.checkpoint),
        "samples": samples,
    }
    return _stage("eval", job.eval_batches, (), rl.checkpoint, seconds, extra)


# ---------------------------------------------------------------------------
# Delivery and the run
# ---------------------------------------------------------------------------


def _source_revision() -> str:
    """The short revision of the tree, or an empty string off a checkout.

    ``NANOCHAT_SOURCE_REVISION`` wins, so a container that copies the tree
    without ``.git`` can still record what it ran. Under a sandboxed test the
    ``git`` call fails and the report carries the empty string, which is the
    honest answer.
    """
    override = os.environ.get("NANOCHAT_SOURCE_REVISION", "").strip()
    if override:
        return override
    try:
        result = subprocess.run(
            ["git", "-C", str(_build.repo_root()), "rev-parse", "--short",
             "HEAD"], capture_output=True, text=True, check=False)
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _host(stages: Sequence[Stage]) -> dict:
    """The host block of the report: backend, arch, precision, revision."""
    revision = _source_revision()
    if not revision:
        for stage in stages:
            recorded = (stage.metadata or {}).get("git_revision") or ""
            if recorded:
                revision = str(recorded)
                break
    return {
        "backend": _build.backend(),
        "arch": _build.architecture(),
        "precision": _build.precision(),
        "revision": revision,
    }


def _report(job: "Job", stages: Sequence[Stage], *, delivered: str = "",
            report_path: str = "") -> Report:
    """The report of one finished chain."""
    ordered = tuple(stages)
    return Report(version=REPORT_VERSION, job=job, stages=ordered,
                  delivered=delivered, report_path=report_path,
                  total_seconds=_total_seconds(ordered),
                  host=_host(ordered))


def deliver(job: "Job", stages: Sequence[Stage], *,
            out_root: str | os.PathLike | None = None
            ) -> tuple[str, str]:
    """Write the artifact contract of D6 and return the two file paths.

    ``deliver/chatrl.ckpt`` is the RL checkpoint copied byte for byte, and
    ``deliver/pipeline_report.json`` is the report. The D6 items live here:

    * the stage order and the step counts: ``stages[].name`` and
      ``stages[].steps``, in chain order;
    * the loss traces and the mean rewards: ``stages[].losses`` and
      ``stages[].extra["mean_rewards"]`` (the RL stage also keeps the rollout
      rewards and the count of groups whose advantages are all zero);
    * the checkpoint digests and the metadata blocks: ``stages[].sha256`` and
      ``stages[].metadata``, whose ``config`` is the recorded model shape;
    * the seconds and the seeds: ``stages[].seconds``, ``total_seconds``, and
      ``job.seed``;
    * the model shape and the tokenizer digest: ``stages[].metadata["config"]``
      and ``stages[].extra["tokenizer_sha256"]`` of the base stage's corpus;
    * the base stage's rates and the hardware-check facts:
      ``stages[].extra["step_ms"]``, ``stages[].extra["tokens_per_sec"]``,
      ``stages[].extra["mfu"]``, and ``stages[].extra["hardware"]``;
    * the eval line: ``stages[].extra["bpb_base"]``, ``bpb_rl``,
      ``eval_batches``, ``eval_batch``, ``eval_corpus``, ``evaluated``,
      ``evaluated_sha256``, and ``samples``;
    * the source revision: ``host.revision``, beside ``host.backend``,
      ``host.arch``, and ``host.precision``.

    The copy is verified against the digest the stage recorded, so a report
    that names a digest names the bytes beside it. The last stage is the eval
    stage, which names the RL checkpoint as its artifact -- the file it did not
    write -- so the delivered copy stays the RL file (section 6).
    """
    if not stages:
        raise PipelineError("there is no stage to deliver")
    final = stages[-1]
    source = Path(final.checkpoint)
    if not source.is_file():
        raise PipelineError(
            f"the {final.name} stage checkpoint {source} is missing, so "
            f"there is nothing to deliver")
    root = Path(out_root).expanduser() if out_root is not None \
        else _run_dir(job)
    target_dir = root / "deliver"
    target_dir.mkdir(parents=True, exist_ok=True)
    delivered = target_dir / "chatrl.ckpt"
    shutil.copyfile(source, delivered)
    if _sha256(delivered) != final.sha256:
        raise PipelineError(
            f"the delivered copy {delivered} does not match the digest of "
            f"{source}")
    report_path = target_dir / "pipeline_report.json"
    report = _report(job, stages, delivered=str(delivered),
                     report_path=str(report_path))
    report_path.write_text(report.to_json() + "\n", encoding="utf-8")
    return str(delivered), str(report_path)


def _check_profile(name: str, shape: api.Config) -> None:
    """Reject a profile that ``Config::Validate`` would reject.

    The rules are the ones ``include/nanochat/config.h`` states, and a
    profile that breaks one otherwise fails inside a stage executable, where
    the message names the C flag and not the profile. Each message here names
    the rule and the profile (``docs/pipeline-scaleup-plan.md`` section 5).
    """
    heads = int(shape.num_heads)
    kv_heads = int(shape.num_kv_heads)
    hidden = int(shape.hidden_dim)
    if heads < 1 or kv_heads < 1:
        raise PipelineError(
            f"the {name!r} profile breaks the Config::Validate rule that "
            f"model dimensions must be positive: num_heads {heads} and "
            f"num_kv_heads {kv_heads}")
    if hidden % heads != 0:
        raise PipelineError(
            f"the {name!r} profile breaks the Config::Validate rule that "
            f"hidden_dim must be divisible by num_heads: hidden_dim {hidden} "
            f"and num_heads {heads}")
    if heads % kv_heads != 0:
        raise PipelineError(
            f"the {name!r} profile breaks the Config::Validate rule that "
            f"num_heads must be divisible by num_kv_heads: num_heads {heads} "
            f"and num_kv_heads {kv_heads}")
    if int(shape.vocab_size) > int(shape.padded_vocab_size):
        raise PipelineError(
            f"the {name!r} profile breaks the Config::Validate rule that "
            f"vocab_size must not be above padded_vocab_size: vocab_size "
            f"{int(shape.vocab_size)} and padded_vocab_size "
            f"{int(shape.padded_vocab_size)}")


def _check(job: "Job") -> None:
    """Reject a job that cannot run, before any stage starts."""
    shape = config(job)
    for name in ("base_steps", "sft_steps", "rl_steps", "num_prompts",
                 "num_samples", "max_tokens", "eval_batches",
                 "eval_samples"):
        if int(getattr(job, name)) < 1:
            raise PipelineError(f"the job's {name} must be positive")
    if int(shape.seq_len) < 1:
        raise PipelineError("the profile's seq_len must be positive")
    _check_profile(job.profile, shape)


def run(job: "Job", *, out_root: str | os.PathLike | None = None) -> Report:
    """Run base, then SFT, then RL, then eval, and return the report.

    ``out_root`` overrides ``job.out_root`` for this call, so a host that owns
    the volume rewrites the root without rewriting the job. The run directory
    is ``<run_root>/<run_id>/``, which holds one directory per stage, the
    corpus, and ``deliver/`` (D6).

    The eval stage comes last and writes no checkpoint, so it names the RL
    file as its own artifact: ``deliver()`` copies the checkpoint of the last
    stage and the delivered file stays ``deliver/chatrl.ckpt`` carrying the RL
    weights (``docs/pipeline-scaleup-plan.md`` section 6).

    With ``job.deliver`` the two deliverable files are written and the report
    that is returned is the report that was written, read back from the file.
    """
    if not isinstance(job, Job):
        raise PipelineError("run() takes a Job")
    if out_root is not None:
        job = dataclasses.replace(job, out_root=str(out_root))
    if not job.out_root:
        job = dataclasses.replace(job, out_root=str(_runs_dir(job)))
    _check(job)
    base = stage_base(job)
    sft_stage = stage_sft(job, base)
    rl_stage = stage_rl(job, sft_stage)
    eval_stage = stage_eval(job, base, rl_stage)
    stages = (base, sft_stage, rl_stage, eval_stage)
    if not job.deliver:
        return _report(job, stages)
    # `deliver` names its `out_root` the run directory, not the root that
    # holds one directory per run, so the call takes its default. Passing the
    # root here wrote `<run_root>/deliver` while every stage wrote
    # `<run_root>/<run_id>/`, and the Modal app and the notebook collect cell
    # read the folder the design names (D6).
    _delivered, report_path = deliver(job, stages)
    return Report.from_json(Path(report_path).read_text(encoding="utf-8"))


def _print_summary(report: Report) -> None:
    """Print one line per stage and the delivered paths."""
    for stage in report.stages:
        if stage.name == "eval":
            # The eval stage has no loss trace of its own; the two bits per
            # byte are what a reader compares.
            print(f"eval: batches={stage.steps} "
                  f"bpb_base={stage.extra.get('bpb_base')} "
                  f"bpb_rl={stage.extra.get('bpb_rl')} "
                  f"seconds={stage.seconds:.2f} "
                  f"evaluated={stage.checkpoint}")
            continue
        last = stage.losses[-1] if stage.losses else float("nan")
        print(f"{stage.name}: steps={stage.steps} last_loss={last:.6f} "
              f"seconds={stage.seconds:.2f} checkpoint={stage.checkpoint}")
    if report.delivered:
        print(f"delivered: {report.delivered}")
    if report.report_path:
        print(f"report: {report.report_path}")


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m nanochat_cpp.pipeline <job.json>`` (design decision D10).

    The module is always wrapped by ``tools/nanochat``, which starts the
    sandbox scope and sets the sentinel the stage executables require. The
    exit status is 0 on a finished chain, 1 on a failure the chain reports, and
    2 for a bad command line.
    """
    parser = argparse.ArgumentParser(
        prog="python -m nanochat_cpp.pipeline",
        description="Run the base, SFT, RL, and eval chain of one job file.")
    parser.add_argument("job", help="the job JSON file (see nanochat_cpp."
                                   "pipeline.Job.to_json)")
    parser.add_argument("--out-root", default=None,
                        help="override the job's out_root for this run")
    parser.add_argument("--json", action="store_true",
                        help="print the whole report as one JSON object")
    options = parser.parse_args(list(argv) if argv is not None else None)
    try:
        job = Job.read(options.job)
        report = run(job, out_root=options.out_root)
    except (PipelineError, OSError, ValueError) as error:
        print(f"pipeline: {error}", file=sys.stderr)
        return 1
    if options.json:
        print(report.to_json())
    else:
        _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
