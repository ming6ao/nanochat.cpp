"""Chat evaluation tasks (mirrors the reference ``tasks/`` package).

The core owns the task classes, the evaluation criteria, the prompt renderers
and the answer parsers of chat evaluation, exactly as ``scripts/chat_eval.py``
imports them from the reference ``tasks`` package: ``ARC``, ``MMLU``, ``GSM8K``
and ``HumanEval``, plus the shared :func:`render_mc` prompt renderer and the
GSM8K/HumanEval answer extraction. Only the model compute is C++ (see
``docs/eval.md`` section 5).

The module imports only the standard library, at import time and after it
(rule 1 of ``docs/dependency-layering-plan.md`` section 4). Two things a task
class cannot do alone are handed in through the registration seam of
:mod:`nanochat_cpp.seam`:

* the hub dataset loader -- ``pyarrow`` and ``filelock`` live behind it, so a
  dataset access needs no import here;
* the code executor -- HumanEval is the one place chat evaluation runs
  generated code, and it inherits the reference guards (a fresh interpreter,
  rlimits, a scrubbed environment, a timeout) rather than reimplementing them.

A host registers both once (`seam.register_hub_loader`, ...). When nothing is
registered, a dataset-backed class raises
:class:`nanochat_cpp.seam.MissingAdapter` naming the extra to install, and the
prompt/extraction logic stays importable and testable anywhere. A test hands
in a fake loader, so it needs no network and no wheel.
"""

from __future__ import annotations

import re

from . import seam

# ---------------------------------------------------------------------------
# Dataset access (mirrors nanochat/tasks/common.py)
# ---------------------------------------------------------------------------


def load_hub_dataset(repo_id, subset="default", split="train"):
    """Load a hub dataset through the registered loader.

    The dataset access itself -- the parquet shard listing, the download under
    a file lock, and the ``pyarrow`` read -- belongs to the caller's loader,
    which it registers once with :func:`nanochat_cpp.seam.register_hub_loader`.
    This function is the one place the task classes reach it, so a host and a
    test swap the loader in one move. An unregistered loader raises
    :class:`nanochat_cpp.seam.MissingAdapter` naming the extra to install.
    """
    loader = seam.hub_loader()
    return loader(repo_id, subset, split=split)


class Task:
    """Base class of a task: a dataset plus an evaluation criterion.

    Mirrors ``tasks.common.Task``, including the lightweight slicing view.
    """

    def __init__(self, start=0, stop=None, step=1):
        assert start >= 0, f"Start must be non-negative, got {start}"
        assert stop is None or stop >= start, (
            f"Stop should be greater than or equal to start, got {stop} and "
            f"{start}")
        assert step >= 1, f"Step must be strictly positive, got {step}"
        self.start = start
        self.stop = stop
        self.step = step

    @property
    def eval_type(self):
        """One of ``'generative'`` or ``'categorical'``."""
        raise NotImplementedError

    def num_examples(self):
        raise NotImplementedError

    def get_example(self, index):
        raise NotImplementedError

    def __len__(self):
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        step = self.step
        span = stop - start
        num = (span + step - 1) // step
        assert num >= 0, f"Negative number of examples???: {num}"
        return num

    def __getitem__(self, index: int):
        assert isinstance(index, int), (
            f"Index must be an integer, got {type(index)}")
        physical_index = self.start + index * self.step
        return self.get_example(physical_index)

    def evaluate(self, problem, completion):
        raise NotImplementedError


def render_mc(question, letters, choices):
    """The shared multiple-choice prompt (mirrors ``tasks.common.render_mc``).

    Two deliberate details matter for parity: the letter comes *after* the
    choice, and there is no whitespace before the letter. The assistant answer
    is the bare letter (``"A"``), so the prompt must contain that exact token,
    not ``" A"``.
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join(f"- {choice}={letter}\n"
                     for letter, choice in zip(letters, choices))
    query += "\nRespond only with the letter of the correct answer."
    return query


# ---------------------------------------------------------------------------
# ARC (mirrors tasks/arc.py)
# ---------------------------------------------------------------------------


class ARC(Task):
    """The ARC dataset from Allen AI (mirrors ``tasks.arc.ARC``)."""

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["ARC-Easy", "ARC-Challenge"], (
            "ARC subset must be ARC-Easy or ARC-Challenge")
        assert split in ["train", "validation", "test"], (
            "ARC split must be train|validation|test")
        self.ds = load_hub_dataset("allenai/ai2_arc", subset,
                                   split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]
        choices = row["choices"]["text"]
        answer_string = row["answerKey"]
        letters = row["choices"]["label"]
        assert answer_string in letters, (
            f"ARC answer {answer_string} must be one of {letters}")
        user_message = render_mc(question, letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_string},
        ]
        return {
            "messages": messages,
            "letters": letters,
        }

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in conversation["letters"], (
            f"ARC answer {assistant_response} is expected to be one of "
            f"{conversation['letters']}")
        assistant_message = conversation["messages"][-1]["content"]
        return assistant_response == assistant_message


# ---------------------------------------------------------------------------
# MMLU (mirrors tasks/mmlu.py)
# ---------------------------------------------------------------------------


class MMLU(Task):
    """The MMLU dataset (mirrors ``tasks.mmlu.MMLU``)."""

    letters = ("A", "B", "C", "D")

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["all"], f"subset {subset} must be all"
        assert split in ["auxiliary_train", "validation", "dev", "test"], (
            f"split {split} must be auxiliary_train|validation|dev|test")
        self.subset = subset
        self.split = split
        self.ds = load_hub_dataset("cais/mmlu", subset, split=split).shuffle(
            seed=42)

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]
        choices = row["choices"]
        answer = row["answer"]
        subject = row["subject"]
        assert len(choices) == 4, "MMLU should have 4 choices"
        user_message = render_mc(question, self.letters, choices)
        assistant_message = self.letters[answer]
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
        return {
            "messages": messages,
            "subject": subject,
            "letters": self.letters,
        }

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in self.letters, (
            f"MMLU answer {assistant_response} is expected to be one of "
            f"{self.letters}")
        assistant_message = conversation["messages"][-1]["content"]
        return assistant_response == assistant_message


# ---------------------------------------------------------------------------
# GSM8K (mirrors tasks/gsm8k.py)
# ---------------------------------------------------------------------------

GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer(completion):
    """Extract the numerical answer after the ``####`` marker.

    Follows the official grade-school-math normalization (strip, drop commas).
    Returns ``None`` when the marker is absent.
    """
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


class GSM8K(Task):
    """GSM8K evaluation (mirrors ``tasks.gsm8k.GSM8K``)."""

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], (
            "GSM8K subset must be main|socratic")
        assert split in ["train", "test"], "GSM8K split must be train|test"
        self.ds = load_hub_dataset("openai/gsm8k", subset, split=split).shuffle(
            seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]
        answer = row["answer"]
        # GSM8K solutions embed calculator tool calls in << expression=result >>
        # tags, which the reference renders as python/python_output parts.
        assistant_message_parts = []
        parts = re.split(r"(<<[^>]+>>)", answer)
        for part in parts:
            if part.startswith("<<") and part.endswith(">>"):
                inner = part[2:-2]
                if "=" in inner:
                    expr, result = inner.rsplit("=", 1)
                else:
                    expr, result = inner, ""
                assistant_message_parts.append({"type": "python", "text": expr})
                assistant_message_parts.append(
                    {"type": "python_output", "text": result})
            else:
                assistant_message_parts.append({"type": "text", "text": part})
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_message_parts},
        ]
        return {"messages": messages}

    def evaluate(self, conversation, assistant_response):
        assert isinstance(assistant_response, str), (
            "Assuming simple string response for now")
        assistant_message = conversation["messages"][-1]
        assert assistant_message["role"] == "assistant", (
            "Last message must be from the Assistant")
        assert isinstance(assistant_message["content"], list), (
            "This is expected to be a list of parts")
        last_text_part = assistant_message["content"][-1]["text"]
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        return int(pred_num == ref_num)

    def reward(self, conversation, assistant_response):
        return float(self.evaluate(conversation, assistant_response))


# ---------------------------------------------------------------------------
# HumanEval (mirrors tasks/humaneval.py)
# ---------------------------------------------------------------------------


def extract_imports(prompt):
    """Extract the import statements at the top of a code block."""
    imports = []
    for line in prompt.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(stripped)
        elif stripped and not stripped.startswith("#"):
            break
    return "\n".join(imports)


def extract_program(completion):
    """Extract the first Python code block, or the whole completion.

    Handles ```python ... ``` and ``` ... ``` fences and bare code; extra text
    before or after the first block is ignored.
    """
    pattern = r"```(?:python)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, completion, re.DOTALL)
    if matches:
        return matches[0].strip()
    return completion.strip()


def _execute_code(code: str):
    """Run generated code through the registered executor.

    The executor -- a fresh interpreter, rlimits, a scrubbed environment, a
    timeout -- belongs to the caller, which registers one with
    :func:`nanochat_cpp.seam.register_code_executor`. An unregistered executor
    raises :class:`nanochat_cpp.seam.MissingAdapter` naming the extra to
    install; the criterion below reads only the ``success`` attribute, so a
    test can hand in a stub.
    """
    return seam.code_executor()(code)


class HumanEval(Task):
    """HumanEval code benchmark (mirrors ``tasks.humaneval.HumanEval``)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # This dataset has no named subsets on the hub; its parquet config is
        # "openai_humaneval".
        self.ds = load_hub_dataset("openai/openai_humaneval",
                                   subset="openai_humaneval",
                                   split="test").shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        prompt = row["prompt"]
        solution = row["canonical_solution"]
        entry_point = row["entry_point"]
        test = row["test"]
        complete_solution = f"{prompt}\n{solution}"
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": complete_solution},
        ]
        return {
            "messages": messages,
            "entry_point": entry_point,
            "test": test,
        }

    def evaluate(self, conversation, completion):
        imports = extract_imports(conversation["messages"][0]["content"])
        completion_code = extract_program(completion)
        program = (
            imports
            + "\n\n"
            + completion_code
            + "\n\n"
            + conversation["test"]
            + "\n"
            + f"check({conversation['entry_point']})"
        )
        result = _execute_code(program)
        return result.success


__all__ = [
    "load_hub_dataset",
    "Task",
    "render_mc",
    "ARC",
    "MMLU",
    "GSM8K",
    "HumanEval",
    "GSM_RE",
    "extract_answer",
    "extract_imports",
    "extract_program",
]
