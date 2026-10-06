"""Chat evaluation for the Python API (docs/python.md section 5).

The Python side owns the task datasets, the prompt rendering, the answer
extraction, and the metric rules. The C side owns the forward pass. This is
the split of the removed ``chat_eval.py``, moved under the in-process API.

Two task paths:

* **Categorical** (ARC-Easy, ARC-Challenge, MMLU) -- render the prompt, then
  call :meth:`nanochat_cpp.Evaluator.score` with the final prompt position as
  the focus and the answer-letter ids as the focus set, and argmax the focused
  logits.
* **Generative** (GSM8K, HumanEval) -- render the prompt, then call
  :meth:`nanochat_cpp.Model.generate`, decode the completions, and apply the
  task criterion.

:meth:`ChatEvaluator.run` isolates each task: a task that raises (including a
``SystemExit`` from a missing reference checkout) becomes a skipped
:class:`TaskReport`, and the loop continues. ``chat_core`` stays ``None`` when
any task is skipped; ``categorical_core`` stays valid when its three tasks ran.

``main`` keeps a workstation entry point. It runs under the sandbox:

    tools/nanochat eval -- python -m nanochat_cpp.chat --tasks ARC-Easy
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import os
import re
import sys
from pathlib import Path
from typing import Sequence

from . import data
from . import tasks as _tasks
from .api import Config, Evaluator, Model, Tokenizer

__all__ = [
    "ALL_TASKS",
    "BASELINES",
    "CATEGORICAL_TASKS",
    "ChatEvaluator",
    "ChatReport",
    "TaskReport",
    "build_task",
    "build_parser",
    "main",
]

#: The default task list, in evaluation order.
ALL_TASKS = ("ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval")
#: The tasks that score with a focused multiple-choice logit set.
CATEGORICAL_TASKS = ("ARC-Easy", "ARC-Challenge", "MMLU")
#: The tasks that sample and decode a completion.
GENERATIVE_TASKS = ("GSM8K", "HumanEval")
#: The random-guess baseline for each task (mirrors scripts/chat_eval.py).
BASELINES = {
    "ARC-Easy": 0.25,
    "ARC-Challenge": 0.25,
    "MMLU": 0.25,
    "GSM8K": 0.0,
    "HumanEval": 0.0,
}


@dataclasses.dataclass(frozen=True)
class TaskReport:
    """The result of one chat task. ``skipped`` marks an isolated failure."""

    name: str
    correct: int
    total: int
    accuracy: float
    baseline: float
    skipped: bool = False
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class ChatReport:
    """The aggregate chat result. A core value is None when a task skipped."""

    tasks: tuple[TaskReport, ...]
    chat_core: float | None
    categorical_core: float | None


def build_task(task_name: str) -> _tasks.Task:
    """Construct the reference task object for ``task_name``.

    The subset and split choices match ``scripts/chat_eval.py`` exactly.
    """
    if task_name == "HumanEval":
        return _tasks.HumanEval()
    if task_name == "MMLU":
        return _tasks.MMLU(subset="all", split="test")
    if task_name == "ARC-Easy":
        return _tasks.ARC(subset="ARC-Easy", split="test")
    if task_name == "ARC-Challenge":
        return _tasks.ARC(subset="ARC-Challenge", split="test")
    if task_name == "GSM8K":
        return _tasks.GSM8K(subset="main", split="test")
    raise ValueError(f"unknown task: {task_name}")


def _specials(tokenizer) -> dict[str, int]:
    """The special-token id map for an API or a reference tokenizer."""
    specials = getattr(tokenizer, "special_tokens", None)
    if specials:
        return {name: int(value) for name, value in specials.items()}
    # A reference-style tokenizer exposes `encode_special`.
    names = ("<|bos|>", "<|user_start|>", "<|user_end|>",
             "<|assistant_start|>", "<|assistant_end|>", "<|python_start|>",
             "<|python_end|>", "<|output_start|>", "<|output_end|>")
    result: dict[str, int] = {}
    for name in names:
        try:
            result[name] = int(tokenizer.encode_special(name))
        except Exception:  # noqa: BLE001 - a missing token is reported later
            continue
    return result


class _PromptRenderer:
    """Render a conversation to a completion prompt (reference template).

    This mirrors ``nanochat.tokenizer.Tokenizer.render_for_completion`` so the
    focused position and the prompt ids match the reference evaluation. The
    renderer uses the API tokenizer for ordinary text and the special-token id
    map for the template markers.
    """

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.specials = _specials(tokenizer)
        missing = [name for name in ("<|bos|>", "<|user_start|>",
                                     "<|user_end|>", "<|assistant_start|>",
                                     "<|assistant_end|>")
                   if name not in self.specials]
        if missing:
            raise ValueError(
                f"the tokenizer lacks the chat special tokens {missing}")

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text))

    def render_for_completion(self, conversation: dict) -> list[int]:
        """The prompt ids, ending with ``<|assistant_start|>``."""
        messages = copy.deepcopy(conversation["messages"])
        assert messages and messages[-1]["role"] == "assistant", (
            "the last message must be from the assistant")
        messages.pop()  # drop the reference assistant answer
        if messages and messages[0]["role"] == "system":
            assert messages[1]["role"] == "user", (
                "a system message must be followed by a user message")
            messages[1]["content"] = (messages[0]["content"] + "\n\n"
                                      + messages[1]["content"])
            messages = messages[1:]

        ids = [self.specials["<|bos|>"]]
        for index, message in enumerate(messages):
            must_be = "user" if index % 2 == 0 else "assistant"
            assert message["role"] == must_be, (
                f"message {index} is from {message['role']}, expected "
                f"{must_be}")
            content = message["content"]
            if message["role"] == "user":
                assert isinstance(content, str), "user messages are strings"
                ids.append(self.specials["<|user_start|>"])
                ids.extend(self._encode(content))
                ids.append(self.specials["<|user_end|>"])
            else:
                ids.append(self.specials["<|assistant_start|>"])
                if isinstance(content, str):
                    ids.extend(self._encode(content))
                else:
                    for part in content:
                        if part["type"] == "text":
                            ids.extend(self._encode(part["text"]))
                        elif part["type"] == "python":
                            ids.append(self.specials["<|python_start|>"])
                            ids.extend(self._encode(part["text"]))
                            ids.append(self.specials["<|python_end|>"])
                        elif part["type"] == "python_output":
                            ids.append(self.specials["<|output_start|>"])
                            ids.extend(self._encode(part["text"]))
                            ids.append(self.specials["<|output_end|>"])
                        else:
                            raise ValueError(
                                f"unknown part type: {part['type']}")
                ids.append(self.specials["<|assistant_end|>"])
        ids.append(self.specials["<|assistant_start|>"])
        return ids


def _load_tokenizer() -> Tokenizer:
    """Load the ``NCTOKEN1`` tokenizer that the model was trained with."""
    artifact = data.nctoken1_path()
    if artifact is None:
        raise RuntimeError(
            "chat evaluation needs the NCTOKEN1 container; run tok_train_main "
            "first (see docs/tokenizer.md)")
    return Tokenizer.load(artifact)


class ChatEvaluator:
    """Run the chat tasks against one model, with per-task isolation."""

    def __init__(self, model: Model, *,
                 tokenizer: Tokenizer | None = None) -> None:
        if not isinstance(model, Model):
            raise TypeError("model must be a nanochat_cpp.Model")
        self.model = model
        self._evaluator = Evaluator(model)
        if tokenizer is None:
            tokenizer = _load_tokenizer()
        self.tokenizer = tokenizer
        self.renderer = _PromptRenderer(tokenizer)
        # Rows per focused `Evaluator.score` call. Each row is padded to the
        # model context, so a small batch keeps the logits workspace bounded.
        self.score_batch = 8

    # -- One task ---------------------------------------------------------

    def task(self, name: str, **options) -> TaskReport:
        """Evaluate one task. A failure propagates to the caller."""
        if name not in ALL_TASKS:
            raise ValueError(f"unknown task {name!r}; valid: {list(ALL_TASKS)}")
        task_object = build_task(name)
        max_problems = options.get("max_problems")
        seed = int(options.get("seed", 0))
        if task_object.eval_type == "categorical":
            correct, total = self._categorical(task_object, max_problems)
        elif task_object.eval_type == "generative":
            correct, total = self._generative(task_object, max_problems, seed,
                                              options)
        else:
            raise ValueError(
                f"unsupported evaluation type: {task_object.eval_type}")
        accuracy = correct / total if total else float("nan")
        return TaskReport(name=name, correct=correct, total=total,
                          accuracy=accuracy, baseline=BASELINES[name])

    def _categorical(self, task_object, max_problems) -> tuple[int, int]:
        seq = self.model.config.seq_len
        num = (len(task_object) if max_problems is None
               else min(len(task_object), max_problems))
        letter_cache: dict[str, int] = {}
        problems = []
        for index in range(num):
            conversation = task_object[index]
            letters = conversation["letters"]
            letter_ids = self._letter_ids(letters, letter_cache)
            prompt_ids = self.renderer.render_for_completion(conversation)
            if len(prompt_ids) > seq:
                raise ValueError(
                    f"prompt of {len(prompt_ids)} tokens exceeds the model "
                    f"context of {seq}")
            row = list(prompt_ids) + [0] * (seq - len(prompt_ids))
            problems.append((row, len(prompt_ids), letter_ids, letters,
                             conversation))

        correct = total = 0
        for start in range(0, len(problems), self.score_batch):
            chunk = problems[start:start + self.score_batch]
            flat = [token for row, *_ in chunk for token in row]
            lengths = [length for _, length, *_ in chunk]
            focus = [(length - 1, letter_ids)
                     for _, length, letter_ids, *_ in chunk]
            report = self._evaluator.score(flat, lengths=lengths, focus=focus)
            for (_, _, _, letters, conversation), logits in zip(
                    chunk, report.focus_logits):
                assert len(logits) == len(letters), (
                    f"focus logits {len(logits)} != letters {len(letters)}")
                best = max(range(len(logits)), key=lambda k: logits[k])
                total += 1
                correct += int(task_object.evaluate(conversation,
                                                    letters[best]))
        return correct, total

    def _generative(self, task_object, max_problems, seed: int,
                    options: dict) -> tuple[int, int]:
        num = (len(task_object) if max_problems is None
               else min(len(task_object), max_problems))
        max_new_tokens = int(options.get("max_new_tokens", 512))
        temperature = float(options.get("temperature", 0.0))
        top_k = int(options.get("top_k", 50))
        num_samples = int(options.get("num_samples", 1))
        stop_id = self.tokenizer.encode_special("<|assistant_end|>")
        correct = total = 0
        for index in range(num):
            conversation = task_object[index]
            prompt_ids = self.renderer.render_for_completion(conversation)
            rows = self.model.generate(
                prompt_ids, max_tokens=max_new_tokens, temperature=temperature,
                top_k=top_k, num_samples=num_samples, seed=seed + index,
                stop_id=stop_id, bos_id=-1)
            completions = []
            for row in rows:
                generated = [token for token, mask in zip(row.tokens, row.mask)
                             if mask]
                completions.append(self.tokenizer.decode(generated))
            outcomes = [task_object.evaluate(conversation, completion)
                        for completion in completions]
            total += 1
            correct += int(any(outcomes))
        return correct, total

    def _letter_ids(self, letters, cache: dict) -> list[int]:
        """Answer-letter token ids, cached and asserted single-token."""
        ids = []
        for letter in letters:
            if letter not in cache:
                encoded = self.tokenizer.encode(letter)
                assert len(encoded) == 1, (
                    "each answer letter must be a single token")
                cache[letter] = encoded[0]
            ids.append(cache[letter])
        return ids

    # -- Every task -------------------------------------------------------

    def run(self, tasks: Sequence[str] | None = None, *,
            max_problems: int | None = None, seed: int = 0,
            data_dir: str | None = None) -> ChatReport:
        """Evaluate ``tasks`` with per-task failure isolation.

        A task that raises ``SystemExit`` (a missing reference checkout) or any
        other ``BaseException`` becomes a skipped report. ``KeyboardInterrupt``
        and ``GeneratorExit`` are re-raised.
        """
        if data_dir is not None:
            os.environ["NANOCHAT_BASE_DIR"] = str(Path(data_dir).resolve())
        names = list(tasks) if tasks is not None else list(ALL_TASKS)
        reports: list[TaskReport] = []
        for name in names:
            try:
                reports.append(self.task(name, max_problems=max_problems,
                                         seed=seed))
            except (KeyboardInterrupt, GeneratorExit):
                raise
            except BaseException as error:  # noqa: BLE001 - isolate a task
                reports.append(TaskReport(
                    name=name, correct=0, total=0, accuracy=float("nan"),
                    baseline=BASELINES.get(name, 0.0), skipped=True,
                    error=f"{type(error).__name__}: {error}"))
        return ChatReport(
            tasks=tuple(reports),
            chat_core=_centered_mean(reports, ALL_TASKS),
            categorical_core=_centered_mean(reports, CATEGORICAL_TASKS))


def _centered_mean(reports, names) -> float | None:
    """The mean centered accuracy over ``names``, or None if any is skipped."""
    selected = {report.name: report for report in reports}
    values = []
    for name in names:
        report = selected.get(name)
        if report is None or report.skipped:
            return None
        baseline = BASELINES[name]
        denominator = 1.0 - baseline
        if denominator == 0.0:
            return None
        values.append((report.accuracy - baseline) / denominator)
    if not values:
        return None
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Workstation entry point (runs under tools/nanochat)
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nanochat_cpp.chat",
        description="Chat evaluation over the in-process nanochat.cpp API "
                    "(docs/python.md section 5.4)")
    parser.add_argument("--model", required=True, type=str,
                        help="NCHKPT01 checkpoint path")
    parser.add_argument("--tasks", type=str, default=None,
                        help="comma-separated task names (default: all)")
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="NANOCHAT_BASE_DIR override")
    parser.add_argument("--device-type", type=str, default="",
                        help="cuda|cpu (empty = autodetect)")
    # Model config (the checkpoint container holds weights, not the config).
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--aspect-ratio", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--vocab", type=int, default=32768)
    parser.add_argument("--window-pattern", type=str, default="SSSL")
    return parser


def _parse_tasks(text: str | None) -> list[str]:
    if text is None:
        return list(ALL_TASKS)
    names = [name.strip() for name in re.split(r"[|,]", text) if name.strip()]
    invalid = [name for name in names if name not in ALL_TASKS]
    if invalid:
        raise SystemExit(f"unknown task(s) {invalid}; valid: {list(ALL_TASKS)}")
    return names


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data_dir:
        os.environ["NANOCHAT_BASE_DIR"] = str(Path(args.data_dir).resolve())
    device = "cuda" if args.device_type == "cuda" else None
    config = Config(depth=args.layers, aspect_ratio=args.aspect_ratio,
                    head_dim=args.head_dim, seq_len=args.max_seq_len,
                    vocab_size=args.vocab,
                    window_pattern=args.window_pattern)
    model = Model(config, device=device)
    model.load(args.model)
    evaluator = ChatEvaluator(model)
    report = evaluator.run(_parse_tasks(args.tasks),
                           max_problems=args.max_problems, seed=args.seed)
    for task_report in report.tasks:
        if task_report.skipped:
            print(f"{task_report.name}: skipped ({task_report.error})")
        else:
            print(f"{task_report.name}: {100 * task_report.accuracy:.2f}% "
                  f"({task_report.correct}/{task_report.total})")
    if report.chat_core is not None:
        print(f"ChatCORE metric: {report.chat_core:.4f}")
    if report.categorical_core is not None:
        print(f"ChatCORE_cat metric: {report.categorical_core:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
