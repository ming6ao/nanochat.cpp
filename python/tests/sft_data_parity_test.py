"""T0: the R1 data parity gate of ``docs/sft-parity-plan.md`` section 7 C.1.

The pack fixture ``//tests:data/sft_pack_fixture.bin`` records what the
reference tokenizer and the reference ``sft_data_generator_bos_bestfit``
produce for the eight conversations of section 6.1. The test reproduces every
section from it and compares:

* ``render`` -- the :func:`nanochat_cpp.chat.render_conversation` ids and mask
  of each conversation. The case above 2048 tokens pins the truncation
  default of :data:`nanochat_cpp.sft_data.RENDER_MAX_TOKENS`.
* ``pack`` -- :func:`nanochat_cpp.sft_data.pack_rows` over the recorded ids
  and mask, which pins the largest-fit choice, the BOS padding, the
  ``seq_len + 1`` row capacity, and the ``-1`` masked and padding targets.
* ``mixture`` and ``validation_mixture`` -- the seed-42
  :class:`nanochat_cpp.sft_data.TaskMixture` order of D8 and the two
  validation stops.

The test is hermetic. It rebuilds the fixture tokenizer from the ``pieces``
table and the special-token map, so it needs no tokenizer artifact, no torch,
no pyarrow, no shared library, and no network.
"""

from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path

from nanochat_cpp import _build, _lib, chat, sft_data

FIXTURE = "tests/data/sft_pack_fixture.bin"
MAGIC = b"NANCHAT1"
VERSION = 1

#: The nine chat special tokens, in the reference order. The fixture
#: tokenizer owns the ids below the first text id.
SPECIAL_TOKENS = (
    "<|bos|>", "<|user_start|>", "<|user_end|>", "<|assistant_start|>",
    "<|assistant_end|>", "<|python_start|>", "<|python_end|>",
    "<|output_start|>", "<|output_end|>",
)

#: The conversation count, the row count, and the per-row width of section
#: 5.5. The model of that section has ``seq_len=32``, so a row capacity is
#: ``seq_len + 1 = 33`` tokens and a packed row is 32 tokens wide.
CONVERSATIONS = 8
ROWS = 4
SEQ_LEN = 32

#: The largest-fit choice of the reference packer, one entry per packed row:
#: the conversation indexes it packs, in order, and the number of BOS tokens
#: that pad the row to the capacity. The first row takes the longest
#: conversation ("smoltalk", 21 tokens), then the longest one that still fits
#: the remaining 12 ("short", 7 tokens). Nothing fits the last 5, so they are
#: BOS. The second row does the same with "gsm8k_tool" (21 tokens). The cursor
#: wraps after the seven packable conversations and the packer consumes one
#: conversation per pick, so the third and fourth rows repeat the first two
#: and the eighth pick reaches the ``consumed >= size`` stop rule.
ROW_PLAN = (
    ((0, 4), 5),
    ((2, 4), 5),
    ((0, 4), 5),
    ((2, 4), 5),
)


def resolve(relative: str) -> Path:
    """A runfiles path when the test runs under Bazel, else a source path."""
    found = _lib.find_in_runfiles(relative)
    if found is not None:
        return found
    return _build.repo_root() / relative


def load_fixture(path: Path) -> dict:
    """Read ``NANCHAT1``, one version word, and the UTF-8 JSON body."""
    raw = path.read_bytes()
    assert raw[:8] == MAGIC, f"not a {MAGIC!r} fixture: {path}"
    version = struct.unpack("<I", raw[8:12])[0]
    assert version == VERSION, f"unsupported pack fixture version {version}"
    return json.loads(raw[12:].decode("utf-8"))


class _FixtureTokenizer:
    """The fixture tokenizer, rebuilt from the ``pieces`` table.

    The fixture pins the tokenizer by name and SHA-256 but stores no artifact.
    The ``pieces`` table maps every fixture text fragment to its reference
    token ids, and the special-token map carries the nine chat tokens, so the
    two together are the whole surface the renderer and the packer call.
    """

    def __init__(self, record: dict) -> None:
        self._pieces = {text: list(ids)
                        for text, ids in record["pieces"].items()}
        self.special_tokens = {name: int(value) for name, value
                               in record["special_tokens"].items()}
        self._vocab_size = len(record["token_bytes"])

    def encode(self, text: str, prepend=None, append=None) -> list[int]:
        """The reference ids of one fragment, one id per recorded piece.

        ``prepend`` and ``append`` mirror the reference signature. A fragment
        outside the table is a generator bug: the table covers every text the
        conversations use.
        """
        if text not in self._pieces:
            raise AssertionError(
                "the fixture tokenizer has no piece for "
                f"{text[:48]!r} ({len(text)} characters)")
        ids = [] if prepend is None else [int(prepend)]
        ids.extend(self._pieces[text])
        if append is not None:
            ids.append(int(append))
        return ids

    def get_bos_token_id(self) -> int:
        """The id of ``<|bos|>``, the token that pads every row."""
        return self.special_tokens["<|bos|>"]

    def get_vocab_size(self) -> int:
        """The pinned model vocabulary size, including the unused ids."""
        return self._vocab_size


def _recorded_render(conversation, tokenizer):
    """The injected renderer: the fixture holds the ids and the mask."""
    return list(conversation["ids"]), list(conversation["mask"])


class _SyntheticTask:
    """A stand-in task with a fixed length, the D8 synthetic dataset.

    :class:`nanochat_cpp.sft_data.TaskMixture` reads only ``len(task)`` and
    ``task[index]``, so a fixed-length stand-in reproduces the reference
    shuffle without a hub download. The fixture pins the mixture algorithm,
    not the hub order.
    """

    def __init__(self, name, split, length, stop=None) -> None:
        self.name = name
        self.split = split
        self.stop = stop
        self.length = int(length)
        self.examples = [
            {"messages": [
                {"role": "user",
                 "content": f"{name} {split} example {index}"},
                {"role": "assistant", "content": f"{name} answer {index}"},
            ]}
            for index in range(self.length)
        ]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return self.examples[index]


class SftDataParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = load_fixture(resolve(FIXTURE))
        cls.tokenizer = _FixtureTokenizer(cls.fixture["tokenizer"])
        cls.conversations = list(cls.fixture["conversations"])
        cls.render = list(cls.fixture["render"])
        cls.names = [conversation["name"]
                     for conversation in cls.conversations]
        cls.long_index = cls.names.index("long")

    def packed_rows(self) -> list[tuple[list[int], list[int]]]:
        """``pack_rows`` over the recorded ids and the recorded mask."""
        settings = self.fixture["settings"]
        recorded = [{"ids": entry["ids"], "mask": entry["mask"]}
                    for entry in self.render]
        return sft_data.pack_rows(
            recorded, self.tokenizer, settings["seq_len"], _recorded_render,
            buffer_size=settings["buffer_size"], rank=settings["rank"],
            world_size=settings["world_size"])

    def mixture_order(self, record: dict) -> list[list[int]]:
        """The ``TaskMixture`` order of a recorded synthetic task set."""
        mixture = sft_data.TaskMixture(
            [_SyntheticTask(**task) for task in record["tasks"]])
        self.assertEqual(mixture.num_conversations, record["conversations"])
        return [[task_index, local_index]
                for task_index, local_index in mixture.index_map]

    # -- the pinned settings and the rebuilt tokenizer --------------------

    def test_settings_pin_the_reference_packer(self) -> None:
        settings = self.fixture["settings"]
        self.assertEqual(settings["conversations"], CONVERSATIONS)
        self.assertEqual(len(self.conversations), CONVERSATIONS)
        self.assertEqual(settings["seq_len"], SEQ_LEN)
        self.assertEqual(settings["rows"], ROWS)
        # The reference defaults of section 5.1, as the module pins them.
        self.assertEqual(settings["buffer_size"], sft_data.PACK_BUFFER_SIZE)
        self.assertEqual(settings["render_max_tokens"],
                         sft_data.RENDER_MAX_TOKENS)
        self.assertEqual(settings["render_max_tokens"], 2048)
        self.assertEqual(settings["rank"], 0)
        self.assertEqual(settings["world_size"], 1)
        self.assertEqual(settings["stop_rule"], "last_step")
        self.assertEqual(self.fixture["pack"]["rows"], settings["rows"])
        self.assertEqual(self.fixture["version"], VERSION)

    def test_the_fixture_tokenizer_covers_the_conversations(self) -> None:
        record = self.fixture["tokenizer"]
        # The fixture sorts its keys, so compare the map, which pins the
        # reference order of the nine ids.
        self.assertEqual(sorted(record["special_tokens"]),
                         sorted(SPECIAL_TOKENS))
        self.assertEqual(record["special_tokens"],
                         {name: index
                          for index, name in enumerate(SPECIAL_TOKENS)})
        self.assertEqual(self.tokenizer.get_bos_token_id(), 0)
        # The tiny model of section 5.5 has a 256-entry vocabulary, so the
        # 256-entry byte table is the vocab size the ids stay below.
        vocab_size = self.tokenizer.get_vocab_size()
        self.assertEqual(len(record["token_bytes"]), vocab_size)
        self.assertEqual(vocab_size, 256)
        self.assertGreater(len(record["pieces"]), 0)
        for text, ids in record["pieces"].items():
            self.assertTrue(ids, f"the piece table holds no ids for {text!r}")
            self.assertLess(max(ids), vocab_size, text)

    # -- the render section ----------------------------------------------

    def test_renderer_matches_the_reference(self) -> None:
        for index, conversation in enumerate(self.conversations):
            with self.subTest(conversation=conversation["name"]):
                ids, mask = chat.render_conversation(conversation,
                                                     self.tokenizer)
                self.assertEqual(len(ids), len(mask))
                self.assertEqual(ids, self.render[index]["ids"])
                self.assertEqual(mask, self.render[index]["mask"])

    def test_the_long_conversation_pins_the_truncation_default(self) -> None:
        conversation = self.conversations[self.long_index]
        self.assertEqual(conversation["name"], "long")
        ids, mask = chat.render_conversation(conversation, self.tokenizer)
        # No max_tokens argument: this is the pinned 2048-token default.
        self.assertEqual(len(ids), sft_data.RENDER_MAX_TOKENS)
        self.assertEqual(ids, self.render[self.long_index]["ids"])
        self.assertEqual(mask, self.render[self.long_index]["mask"])
        # The conversation is longer than the truncation, and the renderer
        # cuts the prefix of the ids and the mask together.
        untruncated_ids, untruncated_mask = chat.render_conversation(
            conversation, self.tokenizer, max_tokens=4096)
        self.assertGreater(len(untruncated_ids), sft_data.RENDER_MAX_TOKENS)
        self.assertEqual(untruncated_ids[:len(ids)], ids)
        self.assertEqual(untruncated_mask[:len(mask)], mask)

    # -- the pack section ------------------------------------------------

    def test_pack_matches_the_reference(self) -> None:
        rows = self.packed_rows()
        self.assertEqual(len(rows), ROWS)
        self.assertEqual([list(tokens) for tokens, _ in rows],
                         self.fixture["pack"]["inputs"])
        self.assertEqual([list(targets) for _, targets in rows],
                         self.fixture["pack"]["targets"])
        for tokens, targets in rows:
            self.assertEqual(len(tokens), SEQ_LEN)
            self.assertEqual(len(targets), SEQ_LEN)

    def test_the_largest_fit_choice_and_the_bos_padding(self) -> None:
        settings = self.fixture["settings"]
        capacity = settings["seq_len"] + 1
        bos = self.tokenizer.get_bos_token_id()
        rows = self.packed_rows()
        self.assertEqual(len(rows), len(ROW_PLAN))
        # The plan names the conversations the largest-fit search takes.
        self.assertEqual([self.names[content[0]] for content, _ in ROW_PLAN],
                         ["smoltalk", "gsm8k_tool", "smoltalk", "gsm8k_tool"])
        for index, ((content, padding), (tokens, targets)) in enumerate(
                zip(ROW_PLAN, rows)):
            with self.subTest(row=index):
                self.assertTrue(all(self.names[which] != "long"
                                    for which in content))
                assembled = [value for which in content
                             for value in self.render[which]["ids"]]
                mask_row = [value for which in content
                            for value in self.render[which]["mask"]]
                # The row holds exactly seq_len + 1 tokens: the content, then
                # the BOS padding for the remainder of the capacity.
                assembled += [bos] * padding
                mask_row += [0] * padding
                self.assertEqual(len(assembled), capacity)
                self.assertEqual(assembled[-1], bos)
                # The inputs are the row without its last token, which only
                # ever serves as the final target.
                self.assertEqual(tokens, assembled[:-1])
                # A mask of 1 supervises the token that follows it; the masked
                # user span, the conversation boundary BOS, and the padding
                # targets are -1.
                expected = [value if keep else -1
                            for value, keep in zip(assembled[1:],
                                                   mask_row[1:])]
                for position in range(capacity - 1 - padding, len(tokens)):
                    expected[position] = -1
                self.assertEqual(targets, expected)
                self.assertEqual(targets[-1], -1)

    def test_the_oversized_conversation_is_skipped(self) -> None:
        capacity = self.fixture["settings"]["seq_len"] + 1
        oversized = self.render[self.long_index]["ids"]
        self.assertGreater(len(oversized), capacity)
        # The reference packer drops a conversation longer than one row at
        # fetch time, so no row holds it. It consumes one conversation per
        # pick, so the four rows pack the eight conversations of the fixture
        # and the last pick reaches the stop rule.
        picks = [which for content, _ in ROW_PLAN for which in content]
        self.assertNotIn(self.long_index, picks)
        self.assertEqual(len(picks), CONVERSATIONS)

    # -- the two mixtures (D8) -------------------------------------------

    def test_train_mixture_matches_the_reference_order(self) -> None:
        record = self.fixture["mixture"]
        self.assertEqual(record["seed"], 42)
        self.assertEqual([task["name"] for task in record["tasks"]],
                         ["smoltalk", "mmlu", "mmlu", "gsm8k", "gsm8k"])
        self.assertEqual([task["split"] for task in record["tasks"]],
                         ["train", "auxiliary_train", "auxiliary_train",
                          "train", "train"])
        self.assertEqual([task["length"] for task in record["tasks"]],
                         [7, 3, 3, 2, 2])
        self.assertEqual([task.get("stop") for task in record["tasks"]],
                         [None] * 5)
        self.assertEqual(record["conversations"], 17)
        self.assertEqual(self.mixture_order(record), record["order"])

    def test_validation_mixture_matches_the_reference_order(self) -> None:
        record = self.fixture["validation_mixture"]
        self.assertEqual(record["seed"], 42)
        self.assertEqual([task["name"] for task in record["tasks"]],
                         ["smoltalk", "mmlu", "gsm8k"])
        self.assertEqual([task["split"] for task in record["tasks"]],
                         ["test", "test", "test"])
        self.assertEqual([task["length"] for task in record["tasks"]],
                         [5, 3, 2])
        # The two validation stops of section 5.1.
        self.assertEqual([task.get("stop") for task in record["tasks"]],
                         [None, sft_data.VAL_MMLU_STOP,
                          sft_data.VAL_GSM8K_STOP])
        self.assertEqual([task.get("stop") for task in record["tasks"]],
                         [None, 5200, 420])
        self.assertEqual(record["conversations"], 10)
        self.assertEqual(self.mixture_order(record), record["order"])


if __name__ == "__main__":
    unittest.main()
