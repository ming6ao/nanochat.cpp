"""``python -m nanochat_cpp.base_eval``: a nanochat-compatible base evaluation CLI.

Mirrors ``scripts/base_eval.py`` through the process seam described in
``docs/eval.md``: Python owns everything that is not tensor math (the
checkpoint resolution, the tokenizer, the CORE eval bundle, jinja prompt
rendering, few-shot sampling, and every metric decision), while the C++
binaries own the forward pass. The C++ side receives token ids only; torch and
jinja never leave this module.

Three modes, comma-separated in ``--eval`` (default all three):

* ``bpb``    -- run ``eval_main`` over the train/val parquet splits
                (``nanochat::EvalBpb``), mapping the reference ``--split-tokens``
                to a step count.
* ``sample`` -- tokenize the fixed reference prompt list and run
                ``generate_main`` (``nanochat::GenerateBatch``): greedy for the
                conditioned prompts, temperature 1.0 with eight samples for the
                unconditioned prompt.
* ``core``   -- reproduce ``nanochat.core_eval.evaluate_core`` exactly, scoring
                each candidate span with ``score_main`` (``ScoreBatch``).

The checkpoint is resolved by ``source``/``model-tag``/``step`` (mirroring
``nanochat.checkpoint_manager``). A reference torch ``.pt`` is converted to the
``NCHKPT01`` container into the bridge's cache; an already-converted container
is used in place. ``--model`` overrides the resolution with an explicit path.

Example::

    tools/nanochat_cpp base_eval --eval bpb,sample,core --max-examples 100
    python -m nanochat_cpp.base_eval --model-tag d8 --source base --step 1000 \
        --split-tokens 524288 --eval bpb
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from . import checkpoint, data, eval_fixture, launcher, reference

# ---------------------------------------------------------------------------
# Constants (mirrors scripts/base_eval.py and nanochat/core_eval.py)
# ---------------------------------------------------------------------------

#: The CORE bundle the reference downloads on first use.
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

#: ``scripts/base_eval.py`` default for ``--split-tokens``.
DEFAULT_SPLIT_TOKENS = 40 * 524288

#: The fixed conditioning prompts from ``scripts/base_eval.py``.
DEFAULT_PROMPTS = (
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
)

CONDITIONED_MAX_TOKENS = 16
UNCONDITIONED_MAX_TOKENS = 128
UNCONDITIONED_SAMPLES = 8
SAMPLE_SEED = 42

EVAL_MODES = ("core", "bpb", "sample")

# ---------------------------------------------------------------------------
# CORE prompt rendering (verbatim from nanochat/core_eval.py)
# ---------------------------------------------------------------------------

_MC_TEMPLATE = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()

_SCHEMA_TEMPLATE = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()

_LM_TEMPLATE = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()


def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render one prompt per multiple-choice candidate (reference verbatim)."""
    from jinja2 import Template

    template = Template(_MC_TEMPLATE)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    return [template.render(choice=choice, **context) for choice in item["choices"]]


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render one prompt per schema context option (reference verbatim)."""
    from jinja2 import Template

    template = Template(_SCHEMA_TEMPLATE)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    return [template.render(context=context_option, **context)
            for context_option in item["context_options"]]


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """Render the without/with-continuation prompt pair (reference verbatim)."""
    from jinja2 import Template

    template = Template(_LM_TEMPLATE)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction="left"):
    """Length of the common prefix/suffix across token sequences (reference)."""
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        "left": range(min_len),
        "right": range(-1, -min_len - 1, -1),
    }[direction]
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


# ---------------------------------------------------------------------------
# C++ binary runner
# ---------------------------------------------------------------------------


def _autodetect_backend(device_type: str) -> str:
    """``cuda``/``cpu`` from an explicit device type or torch availability."""
    if device_type == "cuda":
        return "cuda"
    if device_type in ("cpu", "mps"):
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - torch is optional for this decision
        return "cpu"


class Runners:
    """Locate, build, and run the evaluation binaries under the sandbox.

    Mirrors ``launcher.launch`` for the subprocess case so the caller can read
    the binary's stdout (``bpb``) and output files. When the bridge already runs
    inside the sandbox (``NANOCHAT_SANDBOX`` is set), the binary is executed
    directly so the existing cgroup covers it; otherwise the launch goes through
    ``tools/nanochat`` (through the GPU broker for a CUDA run).
    """

    def __init__(self, repo_root: Path, cuda: bool, profile: str,
                 build: bool = True) -> None:
        self.repo_root = repo_root
        self.cuda = cuda
        self.profile = profile
        self.build = build
        self._paths: dict[str, Path] = {}
        self.plan: list[list[str]] = []

    def path(self, name: str) -> Path:
        if name not in self._paths:
            self._paths[name] = launcher.ensure_built(
                self.repo_root, self.cuda, name, build=self.build)
        return self._paths[name]

    def command(self, name: str, arguments: list[str]) -> list[str]:
        binary = self.path(name)
        if os.environ.get("NANOCHAT_SANDBOX"):
            return [str(binary)] + list(arguments)
        if self.cuda:
            return [str(launcher.nanochat_entry(self.repo_root)), "gpu",
                    "--profile", self.profile, "--", str(binary)] + list(arguments)
        return [str(launcher.nanochat_entry(self.repo_root)), "run",
                self.profile, "--", str(binary)] + list(arguments)

    def run(self, name: str, arguments: list[str],
            capture: bool = True) -> subprocess.CompletedProcess:
        command = self.command(name, arguments)
        self.plan.append(command)
        print("[nanochat_cpp] " + " ".join(shlex.quote(part) for part in command),
              file=sys.stderr)
        sys.stderr.flush()
        try:
            return subprocess.run(command, capture_output=capture, text=True,
                                  check=True)
        except subprocess.CalledProcessError as error:
            # Surface the binary's own diagnostics before the traceback, so a
            # failing evaluation points at the C++ side directly.
            if error.stdout:
                print(error.stdout, file=sys.stderr)
            if error.stderr:
                print(error.stderr, file=sys.stderr)
            raise


# ---------------------------------------------------------------------------
# Checkpoint resolution and conversion
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """A resolved NCHKPT01 container plus its model configuration."""

    path: Path
    config: dict
    step: int
    source: str
    model_tag: str
    base_dir: Path

    def model_flags(self) -> list[str]:
        return model_flags(self.config)


def model_flags(config: dict) -> list[str]:
    """The shared ``Config`` flags every evaluation binary accepts."""
    flags = [
        "--layers", str(config["n_layer"]),
        "--heads", str(config["n_head"]),
        "--kv-heads", str(config["n_kv_head"]),
        "--hidden", str(config["n_embd"]),
        "--seq", str(config["sequence_len"]),
        "--vocab", str(config["vocab_size"]),
        "--padded-vocab", str(config["padded_vocab_size"]),
        "--window-pattern", str(config["window_pattern"]),
    ]
    rope_base = config.get("rope_base")
    if rope_base is not None:
        flags += ["--rope-base", repr(float(rope_base))]
    return flags


def default_base_dir() -> Path:
    override = os.environ.get("NANOCHAT_BASE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "nanochat"


def _is_container(path: Path) -> bool:
    """True when ``path`` starts with the ``NCHKPT01`` magic."""
    try:
        with open(path, "rb") as handle:
            return handle.read(8) == b"NCHKPT01"
    except OSError:
        return False


def _step_from_name(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def _find_meta(path: Path) -> Path | None:
    """A metadata sidecar for an explicit checkpoint path, if one exists."""
    for candidate in (path.with_suffix(".json"),
                      path.parent / "meta.json"):
        if candidate.is_file():
            return candidate
    match = re.search(r"(?:model_)?(\d+)", path.stem)
    if match:
        sidecar = path.parent / f"meta_{int(match.group(1)):06d}.json"
        if sidecar.is_file():
            return sidecar
    metas = sorted(path.parent.glob("meta_*.json"))
    return metas[-1] if metas else None


def _read_config(meta_path: Path | None) -> dict:
    if meta_path is None or not meta_path.is_file():
        return {}
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    config = dict(meta.get("model_config", {}))
    return config


def _convert_to_cache(pt_path: Path, cache_dir: Path, step: int) -> Path:
    """Convert a reference torch ``.pt`` into a cached ``NCHKPT01`` container."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{pt_path.stem}.nchkpt01"
    if out_path.is_file():
        return out_path
    print(f"[nanochat_cpp] converting {pt_path} -> {out_path} (torch, host-side)",
          file=sys.stderr)
    checkpoint.main(["--input", str(pt_path), "--output", str(out_path),
                     "--force", "--quiet", "--verify"])
    return out_path


def _apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    overrides = {
        "n_layer": args.layers,
        "n_head": args.heads,
        "n_kv_head": args.kv_heads,
        "n_embd": args.hidden,
        "sequence_len": args.max_seq_len,
        "vocab_size": args.vocab,
        "window_pattern": args.window_pattern,
        "padded_vocab_size": args.padded_vocab,
        "rope_base": args.rope_base,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if "padded_vocab_size" not in config and "vocab_size" in config:
        from .config import padded_vocab
        config["padded_vocab_size"] = padded_vocab(int(config["vocab_size"]))
    return config


def resolve_model(args: argparse.Namespace) -> ModelSpec:
    """Resolve ``--model`` or ``--source``/``--model-tag``/``--step`` to weights."""
    base_dir = Path(args.base_dir) if args.base_dir else default_base_dir()
    base_dir = base_dir.resolve()
    cache_dir = base_dir / "base_eval" / "checkpoints"

    if args.model:
        path = Path(args.model).resolve()
        if not path.is_file():
            raise SystemExit(f"checkpoint not found: {path}")
        meta_path = Path(args.meta).resolve() if args.meta else _find_meta(path)
        config = _read_config(meta_path)
        with open(path, "rb") as handle:
            magic = handle.read(8)
        step = _step_from_name(path)
        if magic == b"NCHKPT01" or args.dry_run:
            weights = path
        else:
            weights = _convert_to_cache(path, cache_dir, step)
        if not config:
            raise SystemExit(
                "no model metadata found; pass --meta or the model config "
                "overrides (--layers/--heads/--hidden/--max-seq-len/...)")
        return ModelSpec(path=weights, config=_apply_overrides(config, args),
                         step=step, source=args.source,
                         model_tag=path.parent.name, base_dir=base_dir)

    resolved = checkpoint.resolve(args.source, args.model_tag, args.step, base_dir)
    meta = {}
    if resolved.meta_path.is_file():
        with open(resolved.meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    config = _read_config(resolved.meta_path)
    if _is_container(resolved.model_path) or args.dry_run:
        weights = resolved.model_path
    else:
        weights = _convert_to_cache(resolved.model_path, cache_dir, resolved.step)
    if not config:
        raise SystemExit(
            f"no model_config in {resolved.meta_path}; pass --meta or the "
            "model config overrides")
    return ModelSpec(path=weights, config=_apply_overrides(config, args),
                     step=int(meta.get("step", resolved.step)),
                     source=resolved.source, model_tag=resolved.model_tag,
                     base_dir=base_dir)


# ---------------------------------------------------------------------------
# Tokenizer / eval bundle helpers
# ---------------------------------------------------------------------------


def load_tokenizer():
    """Return the container-backed tokenizer for evaluation."""
    artifact = data.nctoken1_path()
    if artifact is None:
        raise SystemExit(
            "evaluation needs the NCTOKEN1 container; run tok_train_main first "
            "(see docs/tokenizer.md)")
    return data.load_nctoken1(artifact)


def _download_locked(url: str, target: Path, timeout: float = 3600.0) -> None:
    """Download ``url`` to ``target`` once, serialized by an exclusive lock."""
    lock_path = target.with_suffix(target.suffix + ".lock")
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if target.is_file():
                return
            if time.time() - start > timeout:
                raise SystemExit(f"timed out waiting for {lock_path}")
            time.sleep(1.0)
    try:
        if target.is_file():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        print(f"[nanochat_cpp] downloading {url}", file=sys.stderr)
        with urllib.request.urlopen(url) as response, open(temporary, "wb") as out:
            shutil.copyfileobj(response, out)
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(lock_path)
        except OSError:
            pass


def ensure_eval_bundle(base_dir: Path) -> Path:
    """Ensure ``base_dir/eval_bundle`` holds the CORE task data."""
    bundle_dir = base_dir / "eval_bundle"
    if (bundle_dir / "core.yaml").is_file():
        return bundle_dir
    zip_path = base_dir / "eval_bundle.zip"
    if not zip_path.is_file():
        _download_locked(EVAL_BUNDLE_URL, zip_path)
    with tempfile.TemporaryDirectory(dir=str(base_dir)) as tmp:
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(tmp)
        extracted = Path(tmp) / "eval_bundle"
        if not extracted.is_dir():
            extracted = Path(tmp)
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        shutil.move(str(extracted), str(bundle_dir))
    print(f"[nanochat_cpp] placed eval bundle at {bundle_dir}", file=sys.stderr)
    return bundle_dir


# ---------------------------------------------------------------------------
# BPB evaluation
# ---------------------------------------------------------------------------

_BPB_LINE = re.compile(r"eval_main:\s+(\w+)\s+bpb\s+([0-9.eE+-]+)")


def run_bpb(runners: Runners, spec: ModelSpec, args: argparse.Namespace) -> dict:
    """Run ``EvalBpb`` over the train and val splits, returning ``{split: bpb}``."""
    batch = int(args.device_batch_size)
    seq = int(spec.config["sequence_len"])
    tokens_per_step = batch * seq
    if tokens_per_step <= 0:
        raise SystemExit("device-batch-size and sequence length must be > 0")
    split_tokens = int(args.split_tokens)
    if split_tokens % tokens_per_step != 0:
        split_tokens = (split_tokens // tokens_per_step) * tokens_per_step
    steps = split_tokens // tokens_per_step
    if steps <= 0:
        steps = 1
    adjusted = steps * tokens_per_step

    print(f"[nanochat_cpp] bpb: batch={batch} seq={seq} steps={steps} "
          f"tokens={adjusted:,}", file=sys.stderr)
    artifact = data.nctoken1_path()
    if artifact is None:
        raise SystemExit(
            "bpb evaluation needs the NCTOKEN1 container; run tok_train_main "
            "first (see docs/tokenizer.md)")
    train_parquet = ",".join(data.parquet_files("train"))
    val_parquet = ",".join(data.parquet_files("val"))
    if args.dry_run:
        arguments = [
            "--train-parquet", "<train-parquet>",
            "--val-parquet", "<val-parquet>",
            "--tokenizer", str(artifact),
            "--batch", str(batch),
            "--steps", str(steps),
            "--seed", str(args.seed),
            "--model", str(spec.path),
        ] + spec.model_flags()
        print("[nanochat_cpp] bpb dry run:")
        print("  " + " ".join(arguments))
        return {}

    arguments = [
        "--train-parquet", train_parquet,
        "--val-parquet", val_parquet,
        "--tokenizer", str(artifact),
        "--batch", str(batch),
        "--steps", str(steps),
        "--seed", str(args.seed),
        "--model", str(spec.path),
    ] + spec.model_flags()
    completed = runners.run("eval_main", arguments)
    results: dict[str, float] = {}
    for line in completed.stdout.splitlines():
        match = _BPB_LINE.search(line)
        if match:
            results[match.group(1)] = float(match.group(2))
    if not results:
        raise SystemExit(
            "eval_main did not report a bpb value:\n" + completed.stdout)
    for split, value in results.items():
        print(f"{split} bpb: {value:.6f}")
    return results


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _parse_generation_file(path: Path) -> list[list[int]]:
    """Parse the ``--out`` file written by ``generate_main``."""
    rows: list[list[int]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            token_part = line.split("|", 1)[0].strip()
            if not token_part:
                rows.append([])
                continue
            rows.append([int(tok) for tok in token_part.split()])
    return rows


def _generate(runners: Runners, spec: ModelSpec, prompt_ids,
              max_tokens: int, num_samples: int, temperature: float,
              stop_id: int, out_path: Path) -> list[list[int]]:
    prompt_file = out_path.with_suffix(".prompt")
    prompt_file.write_text(" ".join(str(i) for i in prompt_ids), encoding="utf-8")
    arguments = [
        "--prompt-file", str(prompt_file),
        "--max-tokens", str(max_tokens),
        "--num-samples", str(num_samples),
        "--temperature", repr(float(temperature)),
        "--top-k", "0",
        "--stop-id", str(stop_id),
        "--bos-id", "-1",
        "--seed", str(SAMPLE_SEED),
        "--out", str(out_path),
        "--model", str(spec.path),
    ] + spec.model_flags()
    runners.run("generate_main", arguments)
    return _parse_generation_file(out_path)


def run_samples(runners: Runners, spec: ModelSpec, tokenizer,
                args: argparse.Namespace, work_dir: Path) -> tuple[list[str], list[str]]:
    """Run the conditioned and unconditioned sample prompts from the reference."""
    bos = tokenizer.get_bos_token_id()
    try:
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
    except Exception:  # noqa: BLE001 - older tokenizers may lack the token
        assistant_end = -1

    conditioned: list[str] = []
    print("\n" + "=" * 80)
    print("Model Samples")
    print("=" * 80)
    print("\nConditioned samples:")
    for index, prompt in enumerate(DEFAULT_PROMPTS):
        tokens = tokenizer(prompt, prepend="<|bos|>")
        out_path = work_dir / f"sample_cond_{index}.txt"
        rows = _generate(runners, spec, tokens,
                         CONDITIONED_MAX_TOKENS, 1, 0.0, assistant_end, out_path)
        sample_str = tokenizer.decode(rows[0]) if rows else ""
        print("-" * 80)
        print(sample_str)
        conditioned.append(sample_str)

    unconditioned: list[str] = []
    print("\nUnconditioned samples:")
    tokens = tokenizer("", prepend="<|bos|>")
    out_path = work_dir / "sample_uncond.txt"
    rows = _generate(runners, spec, tokens,
                     UNCONDITIONED_MAX_TOKENS, UNCONDITIONED_SAMPLES, 1.0,
                     assistant_end, out_path)
    for row in rows:
        sample_str = tokenizer.decode(row)
        print("-" * 80)
        print(sample_str)
        unconditioned.append(sample_str)
    return conditioned, unconditioned


# ---------------------------------------------------------------------------
# CORE evaluation
# ---------------------------------------------------------------------------


@dataclass
class CoreExample:
    """One CORE example's packed cases and the data needed to score it."""

    cases: list
    task_type: str
    gold: int | None
    starts: list[int]
    ends: list[int]
    tokens: list[list[int]] = field(default_factory=list)
    row_offset: int = 0
    index: int = 0


def _fewshot_examples(data, idx: int, num_fewshot: int):
    if num_fewshot <= 0:
        return []
    rng = random.Random(1234 + idx)
    available_indices = [i for i in range(len(data)) if i != idx]
    fewshot_indices = rng.sample(available_indices, num_fewshot)
    return [data[i] for i in fewshot_indices]


def build_core_example(idx: int, data, task_meta: dict, tokenizer) -> CoreExample:
    """Pack one CORE example into ``ScoreBatch`` cases (reference logic)."""
    item = data[idx]
    task_type = task_meta["task_type"]
    continuation_delimiter = task_meta["continuation_delimiter"]
    fewshot = _fewshot_examples(data, idx,
                                int(task_meta["num_fewshot"]))
    bos = tokenizer.get_bos_token_id()

    if task_type == "multiple_choice":
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot)
        tokens = tokenizer(prompts, prepend=bos)
        start = find_common_length(tokens, direction="left")
        starts = [start] * len(tokens)
        ends = [len(row) for row in tokens]
    elif task_type == "schema":
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot)
        tokens = tokenizer(prompts, prepend=bos)
        suffix_length = find_common_length(tokens, direction="right")
        ends = [len(row) for row in tokens]
        starts = [end - suffix_length for end in ends]
    elif task_type == "language_modeling":
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot)
        rows = tokenizer(prompts, prepend=bos)
        tokens_without, tokens_with = rows
        start = len(tokens_without)
        end = len(tokens_with)
        if not start < end:
            raise ValueError("prompt without is not a prefix of prompt with")
        if tokens_without != tokens_with[:start]:
            raise ValueError("prompt without is not a prefix of prompt with")
        tokens = [tokens_with]
        starts = [start]
        ends = [end]
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    cases = [eval_fixture.EvalCase(tokens=tuple(row), start=si, end=ei)
             for row, si, ei in zip(tokens, starts, ends)]
    return CoreExample(cases=cases, task_type=task_type,
                       gold=item.get("gold"), starts=starts, ends=ends,
                       tokens=tokens)


def _score_cases(runners: Runners, spec: ModelSpec, cases: list,
                 pad_id: int, work_dir: Path, tag: str):
    """Write a cases fixture, run ``score_main``, and read the results."""
    in_path = work_dir / f"score_in_{tag}.bin"
    out_path = work_dir / f"score_out_{tag}.bin"
    eval_fixture.write_cases(in_path, cases, pad_id=pad_id)
    arguments = [
        "--fixture", str(in_path),
        "--out", str(out_path),
        "--model", str(spec.path),
    ] + spec.model_flags()
    runners.run("score_main", arguments)
    return eval_fixture.read_results(out_path)


def _example_is_correct(example: CoreExample, results) -> bool:
    if example.task_type == "language_modeling":
        start, end = example.starts[0], example.ends[0]
        predicted = list(results[0].argmax[start - 1:end - 1])
        actual = list(example.tokens[0][start:end])
        return predicted == actual
    mean_losses = []
    for index, (start, end) in enumerate(zip(example.starts, example.ends)):
        span = list(results[index].nll[start - 1:end - 1])
        mean_losses.append(sum(span) / len(span) if span else math.inf)
    pred_idx = mean_losses.index(min(mean_losses))
    return pred_idx == example.gold


def _evaluate_task(runners: Runners, spec: ModelSpec, data, task_meta: dict,
                   tokenizer, max_tokens: int, work_dir: Path,
                   tag: str) -> float:
    """Score one CORE task and return its accuracy (single process).

    ``ScoreBatch`` materialises a ``batch * seq * vocab`` logits tensor, so the
    examples are packed into chunks whose padded token count
    (``sum(rows) * max_seq``) stays under ``max_tokens``. Examples are sorted by
    length first so a chunk's padded width is close to its rows' real length.
    The decision is per example, so the grouping only perturbs the summation
    order.
    """
    bos = tokenizer.get_bos_token_id()
    examples = []
    for idx in range(len(data)):
        example = build_core_example(idx, data, task_meta, tokenizer)
        example.index = idx
        examples.append(example)
    if not examples:
        return float("nan")
    examples.sort(key=lambda example: max(len(case.tokens)
                                          for case in example.cases))

    budget = max(1, int(max_tokens))
    chunks: list[list[CoreExample]] = []
    current: list[CoreExample] = []
    current_rows = 0
    current_seq = 0
    for example in examples:
        rows = len(example.cases)
        seq = max(len(case.tokens) for case in example.cases)
        if current and (current_rows + rows) * max(current_seq, seq) > budget:
            chunks.append(current)
            current = []
            current_rows = 0
            current_seq = 0
        current.append(example)
        current_rows += rows
        current_seq = max(current_seq, seq)
    if current:
        chunks.append(current)

    correct = [False] * len(examples)
    for chunk_index, block in enumerate(chunks):
        cases = []
        for example in block:
            example.row_offset = len(cases)
            cases.extend(example.cases)
        results = _score_cases(runners, spec, cases, bos, work_dir,
                               f"{tag}_{chunk_index}")
        for example in block:
            rows = len(example.cases)
            view = results[example.row_offset:example.row_offset + rows]
            correct[example.index] = _example_is_correct(example, view)
    return sum(1 for value in correct if value) / len(correct)


def evaluate_core(runners: Runners, spec: ModelSpec, tokenizer, args,
                  work_dir: Path) -> dict:
    """Reproduce ``nanochat.core_eval.evaluate_core`` with ``ScoreBatch``."""
    import yaml

    if args.dry_run:
        print("[nanochat_cpp] core dry run; evaluate_core would run every "
              "core.yaml task")
        return {}

    bundle_dir = ensure_eval_bundle(spec.base_dir)
    config_path = bundle_dir / "core.yaml"
    data_base_path = bundle_dir / "eval_data"
    meta_path = bundle_dir / "eval_meta_data.csv"

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    tasks = config["icl_tasks"]

    random_baselines: dict[str, float] = {}
    with open(meta_path, "r", encoding="utf-8") as handle:
        import csv as _csv
        for row in _csv.DictReader(handle):
            random_baselines[row["Eval Task"]] = float(row["Random baseline"])

    results: dict[str, float] = {}
    centered_results: dict[str, float] = {}
    for task in tasks:
        label = task["label"]
        num_fewshot = task["num_fewshot"]
        if isinstance(num_fewshot, (list, tuple)):
            num_fewshot = num_fewshot[0]
        task_meta = {
            "task_type": task["icl_task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": int(num_fewshot),
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        print(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: "
              f"{task_meta['task_type']})...")

        data_path = data_base_path / task_meta["dataset_uri"]
        with open(data_path, "r", encoding="utf-8") as handle:
            data = [json.loads(line.strip()) for line in handle if line.strip()]

        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if args.max_per_task > 0:
            data = data[:args.max_per_task]

        accuracy = _evaluate_task(runners, spec, data, task_meta, tokenizer,
                                  args.score_max_tokens, work_dir, label)
        results[label] = accuracy
        baseline = random_baselines[label]
        centered = (accuracy - 0.01 * baseline) / (1.0 - 0.01 * baseline)
        centered_results[label] = centered
        print(f"  accuracy: {accuracy:.4f} | centered: {centered:.4f}")

    core_metric = (sum(centered_results.values()) / len(centered_results)
                   if centered_results else float("nan"))
    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
    }


def write_core_csv(core_results: dict, spec: ModelSpec, args) -> Path:
    """Write the reference-format CORE CSV and return its path."""
    if args.output_csv:
        output_csv_path = Path(args.output_csv)
    else:
        model_slug = f"base_model_{spec.step:06d}"
        output_csv_path = spec.base_dir / "base_eval" / f"{model_slug}.csv"
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
        for label in core_results["results"]:
            accuracy = core_results["results"][label]
            centered = core_results["centered_results"][label]
            handle.write(f"{label:<35}, {accuracy:<10.6f}, {centered:<10.6f}\n")
        handle.write(f"{'CORE':<35}, {'':<10}, "
                     f"{core_results['core_metric']:<10.6f}\n")
    return output_csv_path


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nanochat_cpp.base_eval",
        description="Base model evaluation (nanochat CLI): CORE, bpb, samples")
    parser.add_argument("--eval", type=str, default="core,bpb,sample",
                        help="comma-separated evaluations: core,bpb,sample")
    # Checkpoint selection (mirrors nanochat.checkpoint_manager).
    parser.add_argument("--source", type=str, default="base",
                        choices=sorted(checkpoint.CHECKPOINT_SOURCES),
                        help="checkpoint family (default: base)")
    parser.add_argument("--model-tag", type=str, default=None,
                        help="model tag to identify the checkpoint directory")
    parser.add_argument("--step", type=int, default=None,
                        help="model step to load (default = last)")
    parser.add_argument("--model", type=str, default=None,
                        help="explicit .pt or NCHKPT01 checkpoint path")
    parser.add_argument("--meta", type=str, default=None,
                        help="explicit model metadata JSON sidecar")
    # Model config overrides (normally read from the metadata sidecar).
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--kv-heads", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--vocab", type=int, default=None)
    parser.add_argument("--padded-vocab", type=int, default=None)
    parser.add_argument("--window-pattern", type=str, default=None)
    parser.add_argument("--rope-base", type=float, default=None)
    # Evaluation knobs (mirrors scripts/base_eval.py).
    parser.add_argument("--max-per-task", "--max-examples", dest="max_per_task",
                        type=int, default=-1,
                        help="max examples per CORE task (-1 = all)")
    parser.add_argument("--device-batch-size", type=int, default=32,
                        help="per-device batch size for the bpb evaluation")
    parser.add_argument("--split-tokens", type=int, default=DEFAULT_SPLIT_TOKENS,
                        help="tokens to evaluate per bpb split")
    parser.add_argument("--device-type", type=str, default="",
                        help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--score-max-tokens", type=int, default=4096,
                        help="max padded tokens (rows * seq) per score_main "
                             "call; bounds the logits workspace memory")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="override the CORE CSV output path")
    # Bridge-specific.
    parser.add_argument("--base-dir", type=str, default=None,
                        help="nanochat base directory override")
    parser.add_argument("--backend", choices=["cpu", "cuda"], default=None,
                        help="C++ backend for the evaluation binaries")
    parser.add_argument("--profile", type=str, default="t2-parity",
                        help="sandbox profile for a CUDA launch")
    parser.add_argument("--no-build", action="store_true",
                        help="skip the Bazel build and use the current binaries")
    parser.add_argument("--force-data", action="store_true",
                        help="re-tokenize even if a cached shard exists")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the planned commands and exit")
    return parser


def _parse_modes(text: str) -> list[str]:
    modes = [mode.strip() for mode in text.split(",") if mode.strip()]
    invalid = [mode for mode in modes if mode not in EVAL_MODES]
    if invalid:
        raise SystemExit(
            f"invalid eval modes: {invalid}; valid: {list(EVAL_MODES)}")
    ordered = []
    for mode in ("sample", "bpb", "core"):
        if mode in modes:
            ordered.append(mode)
    return ordered


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    modes = _parse_modes(args.eval)

    if args.base_dir:
        os.environ["NANOCHAT_BASE_DIR"] = str(Path(args.base_dir).resolve())

    backend = args.backend or _autodetect_backend(args.device_type)
    cuda = backend == "cuda"
    repo_root = reference.repository_root()
    runners = Runners(repo_root, cuda, args.profile, build=not args.no_build)
    spec = resolve_model(args)

    print(f"[nanochat_cpp] checkpoint: {spec.path}")
    print(f"[nanochat_cpp] source={spec.source} tag={spec.model_tag} "
          f"step={spec.step} backend={backend}")
    print(f"[nanochat_cpp] model: layers={spec.config['n_layer']} "
          f"heads={spec.config['n_head']} hidden={spec.config['n_embd']} "
          f"seq={spec.config['sequence_len']} "
          f"vocab={spec.config['vocab_size']} "
          f"window={spec.config['window_pattern']}")

    tokenizer = None
    if modes:
        tokenizer = load_tokenizer()
        if tokenizer.get_vocab_size() != int(spec.config["vocab_size"]):
            raise SystemExit(
                f"tokenizer vocab {tokenizer.get_vocab_size()} does not match "
                f"model vocab {spec.config['vocab_size']}")

    if args.dry_run:
        # Resolve the planned commands without executing them.
        if "sample" in modes:
            print("[nanochat_cpp] sample dry run; "
                  f"{len(DEFAULT_PROMPTS)} conditioned + "
                  f"{UNCONDITIONED_SAMPLES} unconditioned samples")
        if "bpb" in modes:
            run_bpb(runners, spec, args)
        if "core" in modes:
            evaluate_core(runners, spec, tokenizer, args, Path("."))
        print("[nanochat_cpp] dry run complete")
        return 0

    bpb_results: dict[str, float] = {}
    core_results = None
    with tempfile.TemporaryDirectory(prefix="nanochat_base_eval_") as tmp:
        work_dir = Path(tmp)
        if "sample" in modes:
            run_samples(runners, spec, tokenizer, args, work_dir)
        if "bpb" in modes:
            bpb_results = run_bpb(runners, spec, args)
        if "core" in modes:
            core_results = evaluate_core(runners, spec, tokenizer, args, work_dir)
            output_csv_path = write_core_csv(core_results, spec, args)
            print(f"\nResults written to: {output_csv_path}")
            print(f"CORE metric: {core_results['core_metric']:.4f}")

    if bpb_results:
        print("[nanochat_cpp] bpb: " + ", ".join(
            f"{split}={value:.6f}" for split, value in sorted(bpb_results.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
