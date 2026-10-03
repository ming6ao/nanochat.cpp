"""Chat evaluation tasks (mirrors the reference ``tasks/`` package).

The bridge owns the task datasets and the evaluation criteria for chat
evaluation, exactly as ``scripts/chat_eval.py`` imports them from the reference
``tasks`` package: ``ARC``, ``MMLU``, ``GSM8K`` and ``HumanEval``, plus the
shared :func:`render_mc` prompt renderer and the GSM8K/HumanEval answer
extraction. Only the model compute is C++ (see ``docs/eval.md`` section 5).

The module is stdlib-only at import time. ``numpy``, ``pyarrow``,
``filelock`` and the reference ``nanochat`` package are imported lazily inside
the functions that need them, so the torch-free bridge self-test can import and
exercise the pure prompt/extraction logic anywhere.

HumanEval is the one place chat evaluation executes generated code. It goes
through the reference ``nanochat.execution.execute_code`` (fresh interpreter,
rlimits, scrubbed environment, timeout) and therefore inherits the reference
guards rather than reimplementing them. When the bridge runs under
``tools/nanochat`` the child process inherits the resource sandbox cgroup.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

# ---------------------------------------------------------------------------
# Dataset access (mirrors nanochat/tasks/common.py)
# ---------------------------------------------------------------------------


def get_base_dir() -> str:
    """The nanochat cache root (``NANOCHAT_BASE_DIR`` or ``~/.cache/nanochat``).

    Copied from ``nanochat.common.get_base_dir`` so the task datasets land in
    the same place as the reference implementation without importing torch.
    """
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ["NANOCHAT_BASE_DIR"]
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir


class HubDataset:
    """A minimal HuggingFace ``datasets.Dataset`` stand-in over a pyarrow table.

    Offers lazy row access and a seeded shuffle that matches
    ``datasets.Dataset.shuffle(seed=seed)`` exactly, so the example order the
    reference evaluates is reproduced. Mirrors ``tasks.common.HubDataset``.
    """

    def __init__(self, table, permutation=None):
        self.table = table
        self.permutation = permutation

    def __len__(self):
        return self.table.num_rows

    def shuffle(self, seed):
        import numpy as np

        permutation = np.random.default_rng(seed).permutation(len(self))
        return HubDataset(self.table, permutation)

    def __getitem__(self, index):
        physical_index = (index if self.permutation is None
                          else int(self.permutation[index]))
        return {column: self.table[column][physical_index].as_py()
                for column in self.table.column_names}


def load_hub_dataset(repo_id, subset="default", split="train"):
    """Load a hub dataset through its auto-generated parquet export.

    Mirrors ``tasks.common.load_hub_dataset``: list the parquet shards through
    the hub API, download them once into the base directory, and read them with
    pyarrow. The manifest is written last, so its presence means the download
    completed; a file lock serializes concurrent ranks.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from filelock import FileLock

    base_dir = get_base_dir()
    slug = repo_id.replace("/", "--")
    shards_dir = os.path.join(base_dir, "task_data", slug, subset, split)
    manifest_path = os.path.join(shards_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        os.makedirs(shards_dir, exist_ok=True)
        with FileLock(manifest_path + ".lock"):
            # Only a single rank acquires the lock and downloads; the others
            # block here and then skip because the manifest now exists.
            if not os.path.exists(manifest_path):
                listing_url = (f"https://huggingface.co/api/datasets/{repo_id}"
                               f"/parquet/{subset}/{split}")
                with urllib.request.urlopen(listing_url) as response:
                    shard_urls = json.loads(response.read())
                filenames = []
                for shard_index, shard_url in enumerate(shard_urls):
                    filename = f"{shard_index:05d}.parquet"
                    print(f"Downloading {shard_url} ...")
                    with urllib.request.urlopen(shard_url) as response:
                        content = response.read()
                    with open(os.path.join(shards_dir, filename), "wb") as f:
                        f.write(content)
                    filenames.append(filename)
                with open(manifest_path, "w") as f:
                    json.dump(filenames, f)
    with open(manifest_path, "r") as f:
        filenames = json.load(f)
    shard_paths = [os.path.join(shards_dir, filename) for filename in filenames]
    tables = [pq.read_table(path) for path in shard_paths]
    table = pa.concat_tables(tables)
    return HubDataset(table)


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
    """Run generated code through the reference guarded executor.

    Imported lazily so this module stays importable without the reference
    package; the call itself is the reference ``execute_code`` (fresh
    interpreter, rlimits, scrubbed environment, timeout), not a copy.
    """
    from . import reference

    reference.ensure_reference_on_path()
    from nanochat.execution import execute_code

    return execute_code(code)


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
    "HubDataset",
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
