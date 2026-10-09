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
import ast
import collections
import copy
import dataclasses
import os
import re
import sys
from pathlib import Path
from typing import Sequence

from . import data
from . import tasks as _tasks
from .api import Config, Evaluator, GeneratedRow, Model, Tokenizer

__all__ = [
    "ALL_TASKS",
    "BASELINES",
    "CATEGORICAL_TASKS",
    "ChatEngine",
    "ChatEvaluator",
    "ChatReport",
    "TaskReport",
    "build_task",
    "build_parser",
    "main",
    "render_conversation",
    "use_calculator",
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


def render_conversation(conversation, tokenizer) -> tuple[list[int], list[int]]:
    """Render one conversation to ``(ids, mask)`` for a supervised dataset.

    ``mask`` is 1 over the assistant text, the assistant end token, and the
    python tool-call tokens, and 0 over BOS, the user text, and the tool-output
    tokens (docs/training-seam.md section 6.5). This mirrors the notebook
    template in ``notebooks/nanochat-cpp-on-t4-gpu.ipynb`` so the packed rows
    and the reference renderer agree.
    """
    specials = _specials(tokenizer)
    missing = [name for name in ("<|bos|>", "<|user_start|>",
                                 "<|user_end|>", "<|assistant_start|>",
                                 "<|assistant_end|>", "<|python_start|>",
                                 "<|python_end|>", "<|output_start|>",
                                 "<|output_end|>")
               if name not in specials]
    if missing:
        raise ValueError(
            f"the tokenizer lacks the chat special tokens {missing}")

    messages = copy.deepcopy(conversation["messages"])
    if messages and messages[0]["role"] == "system":
        messages[1]["content"] = (messages[0]["content"] + "\n\n"
                                  + messages[1]["content"])
        messages = messages[1:]

    ids = [specials["<|bos|>"]]
    mask = [0]
    for message in messages:
        if message["role"] == "user":
            ids.append(specials["<|user_start|>"])
            mask.append(0)
            for token in tokenizer.encode(message["content"]):
                ids.append(token)
                mask.append(0)
            ids.append(specials["<|user_end|>"])
            mask.append(0)
            continue

        ids.append(specials["<|assistant_start|>"])
        mask.append(0)
        content = message["content"]
        if isinstance(content, str):
            for token in tokenizer.encode(content):
                ids.append(token)
                mask.append(1)
        else:
            for part in content:
                if part["type"] == "python":
                    ids.append(specials["<|python_start|>"])
                    mask.append(1)
                    body, keep = tokenizer.encode(part["text"]), 1
                elif part["type"] == "python_output":
                    ids.append(specials["<|output_start|>"])
                    mask.append(0)
                    body, keep = tokenizer.encode(part["text"]), 0
                else:
                    body, keep = tokenizer.encode(part["text"]), 1
                for token in body:
                    ids.append(token)
                    mask.append(keep)
                if part["type"] == "python":
                    ids.append(specials["<|python_end|>"])
                    mask.append(1)
                elif part["type"] == "python_output":
                    ids.append(specials["<|output_end|>"])
                    mask.append(0)
        ids.append(specials["<|assistant_end|>"])
        mask.append(1)
    return ids, mask


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


# ---------------------------------------------------------------------------
# Tool forcing (docs/parity.md item P1)
# ---------------------------------------------------------------------------

#: Name fragments the reference calculator rejects. They keep the forced
#: expression away from the interpreter; the AST walk below only ever sees
#: literals, arithmetic, and a string ``.count`` call.
_DANGEROUS_NAMES = ("__", "import", "exec", "eval", "compile", "open",
                    "file", "input", "raw_input", "globals", "locals", "vars",
                    "dir", "getattr", "setattr", "delattr", "hasattr")

#: The binary operators the calculator accepts. ``**`` is deliberately absent.
_ARITHMETIC_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}


def _eval_node(node):
    """Evaluate one arithmetic or ``str.count`` AST node."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(
                node.value, (int, float, str)):
            raise ValueError("unsupported literal")
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC_OPS:
        return _ARITHMETIC_OPS[type(node.op)](_eval_node(node.left),
                                              _eval_node(node.right))
    if (isinstance(node, ast.Call) and not node.keywords
            and len(node.args) == 1
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "count"):
        target = _eval_node(node.func.value)
        needle = _eval_node(node.args[0])
        if not isinstance(target, str) or not isinstance(needle, str):
            raise ValueError("unsupported count call")
        return target.count(needle)
    raise ValueError("unsupported expression")


def use_calculator(expr: str) -> str | None:
    """Evaluate the expression of one python tool call, or return None.

    Mirrors ``nanochat.engine.use_calculator`` (the reference calculator tool,
    docs/parity.md item P1) so a forced output matches the reference: pure
    arithmetic is allowed except ``**``, and only the string ``.count`` method
    is otherwise supported. The result is ``str(result)``, exactly the text the
    reference forces.
    """
    expr = expr.replace(",", "")
    if all(character in "0123456789*+-/.() " for character in expr):
        if "**" in expr:
            return None
    else:
        allowed = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                   "0123456789'\"()._ ")
        if not all(character in allowed for character in expr):
            return None
        if any(name in expr.lower() for name in _DANGEROUS_NAMES):
            return None
        if ".count(" not in expr:
            return None
    try:
        value = _eval_node(ast.parse(expr, mode="eval"))
    except Exception:  # noqa: BLE001 - an unusable expression is no result
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    return str(value)


@dataclasses.dataclass
class _ToolRowState:
    """The per-row state of the streaming ``chat_engine`` protocol."""

    tokens: list[int]
    mask: list[int]
    forced: collections.deque = dataclasses.field(
        default_factory=collections.deque)
    in_python_block: bool = False
    expression: list[int] = dataclasses.field(default_factory=list)
    completed: bool = False


class ChatEngine:
    """The streaming ``chat_engine`` protocol (docs/parity.md item P1).

    The reference ``nanochat.engine.Engine.generate`` yields one token column at
    a time. It watches for ``<|python_start|>`` ... ``<|python_end|>``, decodes
    the expression, runs the calculator, and forces
    ``<|output_start|> result <|output_end|>`` back into the row with mask 0.
    This class reproduces that state machine on top of the public
    :meth:`Model.generate`: :meth:`stream` yields ``(column, masks)`` per step
    and :meth:`generate` collects rows, dropping the terminal token exactly like
    :meth:`Model.generate`.

    The C++ engine exposes the same seam (``src/generate.h``), with the forced
    ids fed through the KV cache. Until the C ABI surfaces it, this driver
    steps the model one token at a time, so keep the rollout shapes small
    (docs/rl-notebook.md section 6): each step re-prefills the row.
    """

    def __init__(self, model: Model, tokenizer: Tokenizer) -> None:
        if not isinstance(model, Model):
            raise TypeError("model must be a nanochat_cpp.Model")
        self.model = model
        self.tokenizer = tokenizer
        self.specials = _specials(tokenizer)
        needed = ("<|python_start|>", "<|python_end|>", "<|output_start|>",
                  "<|output_end|>", "<|assistant_end|>")
        missing = [name for name in needed if name not in self.specials]
        if missing:
            raise ValueError(
                f"the tokenizer lacks the chat special tokens {missing}")

    def _sample(self, tokens, *, seed: int, temperature: float,
                top_k: int) -> int:
        """The next token after ``tokens``, one incremental step.

        ``Model.generate`` with ``max_tokens=1`` prefills ``tokens[:-1]`` and
        decodes ``tokens[-1]``, which is exactly one step of the reference
        loop; the engine re-seeds from ``seed`` on every call, so the sampled
        step is reproducible.
        """
        rows = self.model.generate(tokens, max_tokens=1,
                                   temperature=temperature, top_k=top_k,
                                   num_samples=1, seed=seed, stop_id=-1,
                                   bos_id=-1)
        return rows[0].tokens[-1]

    def _advance(self, state: _ToolRowState, token: int) -> None:
        """Update one row's tool state after ``token`` was emitted."""
        specials = self.specials
        if token == specials["<|assistant_end|>"]:
            state.completed = True
        if token == specials["<|python_start|>"]:
            state.in_python_block = True
            state.expression = []
        elif token == specials["<|python_end|>"] and state.in_python_block:
            state.in_python_block = False
            if state.expression:
                expr = self.tokenizer.decode(state.expression)
                result = use_calculator(expr)
                if result is not None:
                    state.forced.append(specials["<|output_start|>"])
                    state.forced.extend(self.tokenizer.encode(result))
                    state.forced.append(specials["<|output_end|>"])
            state.expression = []
        elif state.in_python_block:
            state.expression.append(token)

    def stream(self, tokens, *, num_samples: int = 1, max_tokens: int = 256,
               temperature: float = 1.0, top_k: int = 50, seed: int = 42):
        """Yield ``(column, masks)`` one generation step at a time.

        A completed row yields ``None`` in the column with mask 0; the stream
        stops once every row is complete or after ``max_tokens`` steps. A
        forced token has mask 0 and a sampled token has mask 1.
        """
        states = [_ToolRowState(list(tokens), [0] * len(tokens))
                  for _ in range(num_samples)]
        for _ in range(max_tokens):
            if all(state.completed for state in states):
                break
            column: list[int | None] = []
            masks: list[int] = []
            for state in states:
                if state.completed:
                    column.append(None)
                    masks.append(0)
                    continue
                if state.forced:
                    token = state.forced.popleft()
                    mask = 0
                else:
                    token = self._sample(state.tokens, seed=seed,
                                         temperature=temperature, top_k=top_k)
                    mask = 1
                state.tokens.append(token)
                state.mask.append(mask)
                self._advance(state, token)
                column.append(token)
                masks.append(mask)
            yield column, masks

    def generate(self, tokens, *, num_samples: int = 1, max_tokens: int = 256,
                 temperature: float = 1.0, top_k: int = 50,
                 seed: int = 42) -> list[GeneratedRow]:
        """Run :meth:`stream` to completion and return one row per sample."""
        rows = [GeneratedRow(tokens=list(tokens), mask=[0] * len(tokens))
                for _ in range(num_samples)]
        terminal = self.specials["<|assistant_end|>"]
        for column, masks in self.stream(
                tokens, num_samples=num_samples, max_tokens=max_tokens,
                temperature=temperature, top_k=top_k, seed=seed):
            for index, token in enumerate(column):
                if token is None or token == terminal:
                    continue
                rows[index].tokens.append(token)
                rows[index].mask.append(masks[index])
        return rows


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
