"""Supervised fine-tuning data: the reference mixture and the packer.

This module mirrors ``scripts/chat_sft.py``, ``tasks/smoltalk.py``, and
``tasks/common.py`` in the PyTorch ``nanochat`` tree. The bridge downloads and
mixes the datasets, renders each conversation, and packs the rows (docs/
post-training.md sections 3 and 4). The C++ side receives the token ids and a
``-1`` target mask.

The reference training mixture has three parts:

* SmolTalk (``HuggingFaceTB/smol-smoltalk``), the train split, about 460K
  general conversations;
* MMLU (``cais/mmlu``), the ``auxiliary_train`` split, three passes by default;
* GSM8K (``openai/gsm8k``), the ``main`` train split, four passes by default.

Validation uses the matching test splits. The reference caps the MMLU test
split at 5.2K rows and the GSM8K test split at 420 rows, so the validation
mixture has the same task ratios as the training mixture.

The module is standard-library only at import time. ``numpy``, ``pyarrow``,
and ``filelock`` load lazily inside the functions that need them. This is the
same policy as :mod:`nanochat_cpp.tasks`, so the torch-free bridge can import
this module anywhere.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from .tasks import GSM8K, MMLU, Task, load_hub_dataset

SMOLLTALK_REPO = "HuggingFaceTB/smol-smoltalk"
MMLU_EPOCHS = 3
GSM8K_EPOCHS = 4
VAL_MMLU_STOP = 5200
VAL_GSM8K_STOP = 420


class SmolTalk(Task):
    """SmolTalk general conversations (mirrors ``tasks.smoltalk.SmolTalk``)."""

    def __init__(self, split, **kwargs) -> None:
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SmolTalk split must be train|test"
        self.ds = load_hub_dataset(SMOLLTALK_REPO, split=split).shuffle(seed=42)

    def num_examples(self) -> int:
        return len(self.ds)

    def get_example(self, index: int) -> dict:
        messages = self.ds[index]["messages"]
        # The reference sanity checks. An optional system message stays.
        assert len(messages) >= 1
        rest = messages[1:] if messages[0]["role"] == "system" else messages
        assert len(rest) >= 2, "SmolTalk messages must have at least 2 messages"
        for i, message in enumerate(rest):
            expected = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected, (
                f"Message {i} has role {message['role']} but should be "
                f"{expected}")
            assert isinstance(message["content"], str), (
                "Content must be a string")
        return {"messages": messages}


class TaskMixture(Task):
    """A deterministic shuffle of several tasks.

    Mirrors ``tasks.common.TaskMixture``. Pass a task more than once to
    oversample it. This is how the reference applies the MMLU and GSM8K
    multipliers.
    """

    def __init__(self, tasks, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tasks = list(tasks)
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)
        index_map = []
        for task_index, length in enumerate(self.lengths):
            for local_index in range(length):
                index_map.append((task_index, local_index))
        # The fixed seed makes the example order reproducible.
        random.Random(42).shuffle(index_map)
        self.index_map = index_map

    def num_examples(self) -> int:
        return self.num_conversations

    def get_example(self, index: int) -> dict:
        assert 0 <= index < self.num_conversations, (
            f"Index {index} out of range for mixture with "
            f"{self.num_conversations} conversations")
        task_index, local_index = self.index_map[index]
        return self.tasks[task_index][local_index]


def build_train_mixture(mmlu_epochs: int = MMLU_EPOCHS,
                        gsm8k_epochs: int = GSM8K_EPOCHS) -> TaskMixture:
    """The reference SFT training mixture.

    The default multipliers match ``scripts/chat_sft.py``: SmolTalk once, MMLU
    three times, and GSM8K four times.
    """
    tasks = [SmolTalk(split="train")]
    tasks += [MMLU(subset="all", split="auxiliary_train")
              for _ in range(mmlu_epochs)]
    tasks += [GSM8K(subset="main", split="train")
              for _ in range(gsm8k_epochs)]
    return TaskMixture(tasks)


def build_val_mixture() -> TaskMixture:
    """The reference SFT validation mixture.

    The test splits match the training task ratios: all of SmolTalk test, the
    first 5.2K MMLU test rows, and the first 420 GSM8K test rows.
    """
    return TaskMixture([
        SmolTalk(split="test"),
        MMLU(subset="all", split="test", stop=VAL_MMLU_STOP),
        GSM8K(subset="main", split="test", stop=VAL_GSM8K_STOP),
    ])


def demo_mixture(limit: int = 64) -> TaskMixture:
    """A small real-data mixture for a notebook or a smoke run.

    This calls the same datasets as :func:`build_train_mixture`, but it takes
    only the first ``limit`` rows of each task. The hub shards still download in
    full, because the loader fetches a shard at a time.
    """
    return TaskMixture([
        SmolTalk(split="train", stop=limit),
        MMLU(subset="all", split="auxiliary_train", stop=limit),
        GSM8K(subset="main", split="train", stop=limit),
    ])


def iter_packed_rows(conversations, tokenizer, seq_len: int,
                     render=None) -> Iterator[tuple[list[int], list[int]]]:
    """Yield ``(tokens, targets)`` rows of length ``seq_len``.

    Each row is BOS-aligned and best-fit packed. A conversation that fits the
    remainder of a row goes in whole. When no conversation fits, the remainder
    is padded with BOS instead of cropped, because SFT must not discard tokens.
    The padding targets and the masked user targets are ``-1``. A conversation
    that is longer than a full row is skipped.

    ``render`` defaults to :func:`nanochat_cpp.chat.render_conversation`. A
    caller can pass a different renderer, for example in a test.

    This mirrors ``sft_data_generator_bos_bestfit`` in ``scripts/chat_sft.py``.
    """
    if render is None:
        from .chat import render_conversation as render  # noqa: PLC0415

    bos = tokenizer.get_bos_token_id()
    row_capacity = seq_len + 1  # the last position is the target only
    source = iter(conversations)
    buffer = []
    exhausted = False

    def refill() -> None:
        nonlocal exhausted
        while not exhausted:
            try:
                conversation = next(source)
            except StopIteration:
                exhausted = True
                break
            ids, mask = render(conversation, tokenizer)
            if len(ids) > row_capacity:
                continue  # this conversation can never fit a row
            buffer.append((ids, mask))

    while True:
        row, mask_row = [], []
        while len(row) < row_capacity:
            if not buffer:
                refill()
                if not buffer:
                    break
            remaining = row_capacity - len(row)
            best_index, best_length = -1, 0
            for index, (ids, _) in enumerate(buffer):
                if len(ids) <= remaining and len(ids) > best_length:
                    best_index, best_length = index, len(ids)
            if best_index < 0:
                break  # no conversation fits; pad the remainder below
            ids, mask = buffer.pop(best_index)
            row.extend(ids)
            mask_row.extend(mask)

        if not row:
            return
        content_len = len(row)
        if len(row) < row_capacity:
            pad = row_capacity - len(row)
            row.extend([bos] * pad)
            mask_row.extend([0] * pad)

        inputs = row[:seq_len]
        targets = [token if keep else -1
                   for token, keep in zip(row[1:row_capacity],
                                          mask_row[1:row_capacity])]
        # Mask the padding targets at the end of the row.
        for position in range(max(content_len - 1, 0), seq_len):
            targets[position] = -1
        yield inputs, targets


def pack_rows(conversations, tokenizer, seq_len: int, render=None) -> list:
    """Materialize :func:`iter_packed_rows` into a list of rows."""
    return list(iter_packed_rows(conversations, tokenizer, seq_len,
                                 render=render))


__all__ = [
    "GSM8K",
    "MMLU",
    "SMOLLTALK_REPO",
    "MMLU_EPOCHS",
    "GSM8K_EPOCHS",
    "VAL_MMLU_STOP",
    "VAL_GSM8K_STOP",
    "SmolTalk",
    "TaskMixture",
    "build_train_mixture",
    "build_val_mixture",
    "demo_mixture",
    "iter_packed_rows",
    "pack_rows",
]
