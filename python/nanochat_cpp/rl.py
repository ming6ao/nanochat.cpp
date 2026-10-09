"""The reinforcement-learning bridge (docs/rl-notebook.md sections 3 and 4).

``nc.rl`` sits beside the other facades. One call starts an RL run::

    nc.rl.run(
        task="gsm8k",
        prompts=conversations,
        num_samples=4,
        steps=1000,
        checkpoint="chatrl.nchkpt01",
        checkpoint_interval=100,
    )

Python owns the prompts and the reward only. It renders a batch of
conversations, starts the persistent C++ worker
(``docs/rl-notebook.md`` section 4 and ``docs/training-seam.md`` section 6.4),
speaks the worker's pipe protocol, scores each rollout with the task reward,
and forms the advantage. It never steps the optimizer and never imports
``torch``: the worker holds the model and the optimizer, and the RL step and
its divisor live in C++.

One step has the seven parts of ``docs/rl-notebook.md`` section 4. The
bridge sends a ``rollout`` message, receives the generated rows, scores them
and computes ``advantage = reward - mean(reward)`` per example, then sends an
``advantage`` message. The worker returns the loss and the gradient norm.

The worker is started directly when ``NANOCHAT_SANDBOX_BACKEND`` is ``none``
(the Kaggle notebook, ``docs/python.md`` section 3.1). Otherwise it is started
through ``tools/nanochat train -- <worker>`` so the GPU broker and the resource
sandbox apply (``docs/rl-notebook.md`` section 6).

This module imports only the Python standard library and the sibling
``nanochat_cpp`` modules.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
from pathlib import Path
from typing import Callable, Sequence

from . import _build, chat, data
from . import toolchain as _toolchain
from .api import Config, Tokenizer

__all__ = [
    "RlError",
    "RlReport",
    "advantages",
    "render_prompts",
    "run",
    "worker_argv",
]

#: The sandbox backend value that starts the worker directly (Kaggle).
DIRECT_BACKEND = "none"

#: The environment variable that overrides the worker binary path.
WORKER_ENV = "NANOCHAT_RL_WORKER"

#: The worker binary, relative to the repository root.
WORKER_RELATIVE = Path("bazel-bin") / "src" / "rl_worker"

#: The ``NANOCHAT_SANDBOX_BACKEND`` variable (docs/python.md section 3.1).
BACKEND_ENV = "NANOCHAT_SANDBOX_BACKEND"

#: The default rollout shape (docs/rl-notebook.md section 6).
DEFAULT_NUM_SAMPLES = 4
DEFAULT_MAX_TOKENS = 256
DEFAULT_SEED = 42


class RlError(RuntimeError):
    """The RL bridge could not start the worker or complete a request."""


@dataclasses.dataclass(frozen=True)
class RlReport:
    """The result of one :func:`run` call.

    ``losses`` and ``grad_norms`` hold one entry per step, in order.
    ``mean_rewards`` holds the mean reward over every rollout of the step.
    """

    steps: int
    num_prompts: int
    num_samples: int
    losses: tuple[float, ...]
    grad_norms: tuple[float, ...]
    mean_rewards: tuple[float, ...]
    checkpoint: str | None = None

    @property
    def loss(self) -> float:
        """The last step's loss, or ``nan`` when no step ran."""
        return self.losses[-1] if self.losses else float("nan")

    @property
    def grad_norm(self) -> float:
        """The last step's gradient norm, or ``nan`` when no step ran."""
        return self.grad_norms[-1] if self.grad_norms else float("nan")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def advantages(rewards: Sequence[float]) -> list[float]:
    """``reward - mean(reward)`` (docs/post-training.md section 5.1).

    The baseline is the mean reward over the samples of one example, so the
    advantages of an example sum to zero.
    """
    values = [float(value) for value in rewards]
    if not values:
        return []
    mean = sum(values) / len(values)
    return [value - mean for value in values]


def _special_id(tokenizer, name: str) -> int:
    """The id of a chat special token, or ``-1`` when it is absent."""
    specials = getattr(tokenizer, "special_tokens", None)
    if specials and name in specials:
        return int(specials[name])
    try:
        return int(tokenizer.encode_special(name))
    except Exception:  # noqa: BLE001 - a missing token is reported by the caller
        return -1


def render_prompts(conversations: Sequence[dict], tokenizer) -> list[list[int]]:
    """Render a batch of conversations to completion prompts.

    Each conversation is rendered with the public
    :func:`nanochat_cpp.chat.render_conversation`. The reference answer of the
    final assistant turn is dropped so the rows start at the
    ``<|assistant_start|>`` marker the generator extends. A conversation that
    is already a prompt (no assistant turn) gains that marker.
    """
    marker = _special_id(tokenizer, "<|assistant_start|>")
    prompts: list[list[int]] = []
    for conversation in conversations:
        ids, _mask = chat.render_conversation(conversation, tokenizer)
        cut = None
        if marker >= 0:
            for index, token in enumerate(ids):
                if token == marker:
                    cut = index
        if cut is not None:
            prompts.append(list(ids[:cut + 1]))
        elif marker >= 0:
            prompts.append(list(ids) + [marker])
        else:
            raise RlError(
                "the tokenizer lacks the <|assistant_start|> chat special "
                "token, so a completion prompt cannot be rendered")
    return prompts


def worker_argv(worker, config: Config | None = None,
                load: str | os.PathLike | None = None) -> list[str]:
    """The worker argument list: the binary, the model flags, and the load.

    The flags match ``nanochat::cli::IsModelFlag`` so the worker recreates the
    exact model shape the bridge used to render the prompts. ``load`` is an
    ``NCHKPT01`` checkpoint to warm-start the model and the optimizer state
    from.
    """
    argv = [str(worker)]
    if config is not None:
        argv.extend([
            "--layers", str(config.num_layers),
            "--heads", str(config.num_heads),
            "--kv-heads", str(config.num_kv_heads),
            "--hidden", str(config.hidden_dim),
            "--seq", str(config.seq_len),
            "--vocab", str(config.vocab_size),
            "--padded-vocab", str(config.padded_vocab_size),
            "--window-pattern", str(config.window_pattern),
            "--rope-base", _format_float(config.rope_base),
        ])
    if load is not None:
        argv.extend(["--checkpoint", str(load)])
    return argv


# ---------------------------------------------------------------------------
# Pipe protocol
# ---------------------------------------------------------------------------


def _format_float(value: float) -> str:
    """The protocol's float format (``%.9g``), which round-trips a float32."""
    return format(float(value), ".9g")


def _format_ints(values: Sequence[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def _format_floats(values: Sequence[float]) -> str:
    return ",".join(_format_float(value) for value in values)


def _parse_ints(text: str) -> list[int]:
    if not text:
        return []
    return [int(part) for part in text.split(",") if part]


def _parse_message(line: str) -> tuple[str, dict[str, str]]:
    """Split one protocol line into ``(type, fields)``."""
    stripped = line.strip()
    if not stripped:
        raise RlError("the RL worker returned an empty line")
    parts = stripped.split(" ")
    fields: dict[str, str] = {}
    for token in parts[1:]:
        key, separator, value = token.partition("=")
        if not separator:
            raise RlError(f"malformed protocol field {token!r} in: {stripped}")
        fields[key] = value
    return parts[0], fields


class _Worker:
    """The client half of the worker pipe protocol (src/rl_worker.h)."""

    def __init__(self, process: subprocess.Popen) -> None:
        self._process = process

    def _send(self, kind: str, fields: dict[str, object]) -> dict[str, str]:
        parts = [kind]
        for key, value in fields.items():
            if value is None:
                continue
            parts.append(f"{key}={value}")
        line = " ".join(parts)
        stdin = self._process.stdin
        if stdin is None:
            raise RlError("the RL worker pipe is closed")
        stdin.write(line + "\n")
        stdin.flush()
        return self._receive(kind)

    def _receive(self, expected: str) -> dict[str, str]:
        stdout = self._process.stdout
        if stdout is None:
            raise RlError("the RL worker pipe is closed")
        line = stdout.readline()
        if not line:
            code = self._process.poll()
            raise RlError(
                f"the RL worker closed the pipe (exit status {code}); pass a "
                "warm-start checkpoint and check the worker's stderr")
        kind, fields = _parse_message(line)
        if kind != expected:
            raise RlError(
                f"expected a {expected!r} reply, got {kind!r}: {line.strip()}")
        if fields.get("ok") != "1":
            raise RlError(
                f"the RL worker reported a {kind} error: "
                f"{fields.get('error', 'unknown')}")
        return fields

    def rollout(self, prompts: Sequence[int], prompt_len: int, num_prompts: int,
                num_samples: int, max_tokens: int, temperature: float,
                top_k: int, seed: int, stop_ids: Sequence[int] | None,
                bos_id: int, stop_id: int) -> dict[str, str]:
        fields: dict[str, object] = {
            "prompt_len": prompt_len,
            "num_prompts": num_prompts,
            "prompts": _format_ints(prompts),
            "num_samples": num_samples,
            "max_tokens": max_tokens,
            "temperature": _format_float(temperature),
            "top_k": top_k,
            "seed": seed,
            "bos_id": bos_id,
            "stop_id": stop_id,
        }
        if stop_ids is not None:
            fields["stop_ids"] = _format_ints(stop_ids)
        return self._send("rollout", fields)

    def advantage(self, tokens: Sequence[int], targets: Sequence[int],
                  weights: Sequence[float], batch: int, seq: int,
                  num_passes: int, examples_per_rank: int) -> dict[str, str]:
        return self._send("advantage", {
            "batch": batch,
            "seq": seq,
            "num_passes": num_passes,
            "examples_per_rank": examples_per_rank,
            "tokens": _format_ints(tokens),
            "targets": _format_ints(targets),
            "advantages": _format_floats(weights),
        })

    def save(self, path: str | os.PathLike) -> dict[str, str]:
        return self._send("save", {"path": str(path)})

    def close(self) -> None:
        try:
            self._send("quit", {})
        except RlError:
            pass
        finally:
            if self._process.stdin is not None:
                self._process.stdin.close()
            if self._process.stdout is not None:
                self._process.stdout.close()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()


# ---------------------------------------------------------------------------
# Startup and scoring
# ---------------------------------------------------------------------------


def _load_tokenizer() -> Tokenizer:
    artifact = data.nctoken1_path()
    if artifact is None:
        raise RlError(
            "the RL bridge needs the NCTOKEN1 tokenizer; run tok_train_main "
            "first (see docs/tokenizer.md)")
    return Tokenizer.load(artifact)


def _normalize_task(task):
    """A task name or a task object, resolved to a task object."""
    if isinstance(task, str):
        for candidate in chat.ALL_TASKS:
            if candidate.lower() == task.lower():
                return chat.build_task(candidate)
        raise RlError(
            f"unknown task {task!r}; valid: {list(chat.ALL_TASKS)}")
    if hasattr(task, "reward") or hasattr(task, "evaluate"):
        return task
    raise RlError("task must be a task name or a chat task object")


def _reward(task, conversation: dict, completion: str) -> float:
    """Score one rollout with the task reward (or its ``evaluate`` rule)."""
    reward = getattr(task, "reward", None)
    if callable(reward):
        return float(reward(conversation, completion))
    evaluate = getattr(task, "evaluate", None)
    if callable(evaluate):
        return float(evaluate(conversation, completion))
    raise RlError("the task has neither a reward nor an evaluate method")


def _split_rows(fields: dict[str, str], expected_rows: int
                ) -> tuple[list[list[int]], list[list[int]]]:
    lengths = _parse_ints(fields.get("lengths", ""))
    masks = _parse_ints(fields.get("masks", ""))
    tokens = _parse_ints(fields.get("tokens", ""))
    if len(lengths) != expected_rows:
        raise RlError(
            f"the worker returned {len(lengths)} lengths for "
            f"{expected_rows} rows")
    if len(masks) != len(tokens):
        raise RlError("the worker returned misaligned masks and tokens")
    rows_tokens: list[list[int]] = []
    rows_masks: list[list[int]] = []
    offset = 0
    for length in lengths:
        rows_tokens.append(tokens[offset:offset + length])
        rows_masks.append(masks[offset:offset + length])
        offset += length
    if offset != len(tokens):
        raise RlError("the worker lengths do not cover the token list")
    return rows_tokens, rows_masks


def _advantage_batch(rows_tokens: list[list[int]], rows_masks: list[list[int]],
                     row_advantages: Sequence[float]
                     ) -> tuple[list[int], list[int], list[float], int]:
    """Flatten the rows into the ``advantage`` message's batch.

    A sampled position carries the token as its target and the row's advantage
    as its weight. A prompt position (or padding) carries ``-1`` (the one mask
    channel) and a zero weight, per ``docs/post-training.md`` section 5.2.
    """
    seq = max((len(row) for row in rows_tokens), default=1)
    tokens: list[int] = []
    targets: list[int] = []
    weights: list[float] = []
    for row, mask, advantage in zip(rows_tokens, rows_masks, row_advantages):
        for position in range(seq):
            if position < len(row):
                tokens.append(int(row[position]))
                if mask[position]:
                    targets.append(int(row[position]))
                    weights.append(float(advantage))
                else:
                    targets.append(-1)
                    weights.append(0.0)
            else:
                tokens.append(0)
                targets.append(-1)
                weights.append(0.0)
    return tokens, targets, weights, seq


def _spawn(argv: Sequence[str], tool: _toolchain.Toolchain,
           env: dict[str, str], direct: bool) -> subprocess.Popen:
    """Start the worker, directly or through ``tools/nanochat train``."""
    if direct:
        command = list(argv)
        child_env = env
    else:
        command = [str(tool.entry), "train", "--", *argv]
        child_env = dict(tool.env)
    try:
        return subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, env=child_env, cwd=str(tool.repo_root))
    except OSError as error:
        raise RlError(
            f"cannot start the RL worker {command[0]!r}: {error}") from error


# ---------------------------------------------------------------------------
# The facade
# ---------------------------------------------------------------------------


def run(task, prompts: Sequence[dict], num_samples: int = DEFAULT_NUM_SAMPLES,
        steps: int = 1, checkpoint: str | os.PathLike | None = None,
        checkpoint_interval: int | None = None, *,
        config: Config | None = None, model=None, tokenizer=None,
        worker: str | os.PathLike | None = None,
        toolchain: _toolchain.Toolchain | None = None,
        env: dict[str, str] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = 1.0,
        top_k: int = 0, seed: int = DEFAULT_SEED,
        stop_ids: Sequence[int] | None = None, stop_id: int | None = None,
        num_passes: int = 1, examples_per_rank: int | None = None,
        save_final: bool = True,
        on_step: Callable[[int, float, float, list[float]], None] | None = None,
        ) -> RlReport:
    """Run RL for ``steps`` optimizer steps and report the loss trajectory.

    ``task`` is a task name (``"gsm8k"``) or a task object that owns the
    reward. ``prompts`` is a batch of conversations (each a full example with
    its reference answer). Python renders the prompts, drives the worker, and
    scores the rollouts; the worker holds the model and the optimizer and runs
    the step.

    ``checkpoint`` is the ``NCHKPT01`` path the worker saves to. It is used as
    the warm start when it already exists. ``checkpoint_interval`` saves every
    ``checkpoint_interval`` steps. ``config`` (or ``model.config``) gives the
    worker the model shape. ``tokenizer`` defaults to the ``NCTOKEN1``
    artifact.

    The worker is started directly when ``NANOCHAT_SANDBOX_BACKEND`` is
    ``none``, and through ``tools/nanochat train -- <worker>`` otherwise. The
    remaining keyword arguments tune the rollout and the normalization.
    """
    prompt_list = list(prompts)
    if not prompt_list:
        raise RlError("prompts must not be empty")
    steps = int(steps)
    num_samples = int(num_samples)
    if steps < 1:
        raise RlError("steps must be positive")
    if num_samples < 1:
        raise RlError("num_samples must be positive")
    if examples_per_rank is not None and int(examples_per_rank) < 1:
        raise RlError("examples_per_rank must be positive")

    task_object = _normalize_task(task)
    if tokenizer is None:
        tokenizer = _load_tokenizer()
    if config is None and model is not None:
        config = model.config
    if config is None:
        config = Config()

    environ = dict(os.environ) if env is None else {
        key: str(value) for key, value in env.items()}
    tool = toolchain if toolchain is not None else _toolchain.Toolchain()
    direct = environ.get(BACKEND_ENV, "").strip().lower() == DIRECT_BACKEND
    if worker is not None:
        worker_path = worker
    elif environ.get(WORKER_ENV):
        worker_path = environ[WORKER_ENV]
    else:
        worker_path = _build.repo_root() / WORKER_RELATIVE

    load: Path | None = None
    if checkpoint is not None:
        candidate = Path(checkpoint).expanduser()
        if candidate.is_file():
            load = candidate

    argv = worker_argv(worker_path, config, load)
    process = _spawn(argv, tool, environ, direct)
    client = _Worker(process)

    num_prompts = len(prompt_list)
    losses: list[float] = []
    grad_norms: list[float] = []
    mean_rewards: list[float] = []
    try:
        prompt_rows = render_prompts(prompt_list, tokenizer)
        prompt_len = max(len(row) for row in prompt_rows)
        pad_id = _special_id(tokenizer, "<|bos|>")
        if pad_id < 0:
            pad_id = 0
        # Left-pad every prompt to a common length; the trailing token stays
        # the assistant marker so the first sampled id follows the prompt.
        padded = [[pad_id] * (prompt_len - len(row)) + row
                  for row in prompt_rows]
        flat_prompts = [token for row in padded for token in row]

        if stop_ids is None:
            terminal = stop_id if stop_id is not None else \
                _special_id(tokenizer, "<|assistant_end|>")
            terminals: list[int] | None = [terminal] * num_samples
        else:
            terminals = [int(value) for value in stop_ids]
            if len(terminals) != num_samples:
                raise RlError(
                    "stop_ids must hold one id per sample "
                    f"({num_samples})")
        terminal_id = terminals[0] if terminals else -1

        if examples_per_rank is None:
            examples_per_rank = num_prompts
        examples_per_rank = int(examples_per_rank)

        for step in range(1, steps + 1):
            reply = client.rollout(
                flat_prompts, prompt_len, num_prompts, num_samples, max_tokens,
                temperature, top_k, seed + step - 1, terminals, -1,
                terminal_id)
            rows_tokens, rows_masks = _split_rows(reply, num_prompts * num_samples)
            rewards: list[float] = []
            for index in range(num_prompts):
                for sample in range(num_samples):
                    row = index * num_samples + sample
                    completion = tokenizer.decode(
                        [token for token, keep in zip(rows_tokens[row],
                                                      rows_masks[row]) if keep])
                    rewards.append(_reward(task_object, prompt_list[index],
                                           completion))
            row_advantages: list[float] = []
            for index in range(num_prompts):
                group = rewards[index * num_samples:(index + 1) * num_samples]
                row_advantages.extend(advantages(group))
            tokens, targets, weights, seq = _advantage_batch(
                rows_tokens, rows_masks, row_advantages)
            result = client.advantage(tokens, targets, weights,
                                      len(rows_tokens), seq, int(num_passes),
                                      examples_per_rank)
            loss = float(result["loss"])
            grad_norm = float(result["grad_norm"])
            losses.append(loss)
            grad_norms.append(grad_norm)
            mean_rewards.append(sum(rewards) / len(rewards) if rewards
                                else 0.0)
            if on_step is not None:
                on_step(step, loss, grad_norm, rewards)
            if (checkpoint is not None and checkpoint_interval
                    and step % int(checkpoint_interval) == 0):
                client.save(checkpoint)
        if checkpoint is not None and save_final:
            client.save(checkpoint)
    finally:
        client.close()

    return RlReport(
        steps=steps,
        num_prompts=num_prompts,
        num_samples=num_samples,
        losses=tuple(losses),
        grad_norms=tuple(grad_norms),
        mean_rewards=tuple(mean_rewards),
        checkpoint=str(checkpoint) if checkpoint is not None else None,
    )
