"""T0: the SFT packer and the task mixture.

The test is hermetic. It uses a fake tokenizer and a fake renderer, so it needs
no hub download, no pyarrow, and no shared library. See docs/post-training.md
section 4 and docs/training-seam.md section 6.5.
"""

from __future__ import annotations

import unittest

from nanochat_cpp import sft_data
from nanochat_cpp.tasks import Task


class _FakeTokenizer:
    """The one tokenizer method the packer needs."""

    def get_bos_token_id(self) -> int:
        return 0


def _fake_render(conversation, tokenizer):
    """Render a conversation dict that already holds ids and a mask."""
    return ([tokenizer.get_bos_token_id()] + list(conversation["ids"]),
            [0] + list(conversation["mask"]))


def _conversation(ids, mask):
    return {"ids": ids, "mask": mask}


class _FakeTask(Task):
    """A task over a fixed list of values."""

    def __init__(self, values, **kwargs) -> None:
        super().__init__(**kwargs)
        self.values = list(values)

    def num_examples(self) -> int:
        return len(self.values)

    def get_example(self, index: int):
        return self.values[index]


class SftDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def pack(self, conversations, seq_len):
        return sft_data.pack_rows(conversations, self.tokenizer, seq_len,
                                  render=_fake_render)

    def test_two_short_conversations_pad_the_row(self) -> None:
        # seq_len=4 gives a row capacity of 5. Two conversations of length 2
        # fill four tokens, and the packer pads the fifth with BOS.
        rows = self.pack([
            _conversation([1], [1]),
            _conversation([2], [1]),
        ], seq_len=4)
        self.assertEqual(len(rows), 1)
        tokens, targets = rows[0]
        self.assertEqual(tokens, [0, 1, 0, 2])
        # Position 0 predicts the assistant token 1. Position 1 predicts the
        # BOS of the second conversation, which the mask ignores. Position 2
        # predicts the assistant token 2. Position 3 is padding.
        self.assertEqual(targets, [1, -1, 2, -1])

    def test_user_tokens_are_masked(self) -> None:
        rows = self.pack([_conversation([1, 2], [0, 1])], seq_len=4)
        tokens, targets = rows[0]
        self.assertEqual(tokens, [0, 1, 2, 0])
        # The user token 1 is masked. The assistant token 2 is supervised.
        self.assertEqual(targets, [-1, 2, -1, -1])

    def test_oversized_conversation_is_skipped(self) -> None:
        rows = self.pack([
            _conversation([1, 2, 3, 4, 5], [1, 1, 1, 1, 1]),
            _conversation([9], [1]),
        ], seq_len=4)
        self.assertEqual(len(rows), 1)
        tokens, targets = rows[0]
        self.assertEqual(tokens, [0, 9, 0, 0])
        self.assertEqual(targets, [9, -1, -1, -1])

    def test_a_full_row_has_no_padding(self) -> None:
        # Five tokens exactly fill the capacity, so no position is padding.
        rows = self.pack([_conversation([1, 2, 3, 4], [1, 1, 1, 1])],
                         seq_len=4)
        tokens, targets = rows[0]
        self.assertEqual(tokens, [0, 1, 2, 3])
        self.assertEqual(targets, [1, 2, 3, 4])

    def test_empty_input_yields_no_rows(self) -> None:
        self.assertEqual(self.pack([], seq_len=4), [])

    def test_task_mixture_len_and_order(self) -> None:
        mixture = sft_data.TaskMixture([
            _FakeTask(["a", "b"], step=1),
            _FakeTask(["c"], step=1),
        ])
        self.assertEqual(len(mixture), 3)
        self.assertEqual(sorted(mixture.get_example(i) for i in range(3)),
                         ["a", "b", "c"])

    def test_defaults_match_the_reference(self) -> None:
        self.assertEqual(sft_data.MMLU_EPOCHS, 3)
        self.assertEqual(sft_data.GSM8K_EPOCHS, 4)
        self.assertEqual(sft_data.SMOLLTALK_REPO,
                         "HuggingFaceTB/smol-smoltalk")


if __name__ == "__main__":
    unittest.main()
