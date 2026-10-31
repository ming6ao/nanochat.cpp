"""T0: the SFT packer and the task mixture.

The test is hermetic. It uses a fake tokenizer and a fake renderer, so it needs
no hub download, no pyarrow, and no shared library. See docs/post-training.md
section 4 and docs/training-seam.md section 6.5.

The packer is a pinned mirror of ``sft_data_generator_bos_bestfit`` in
``scripts/chat_sft.py``. :func:`reference_rows` is a direct transcription of
that generator, and :class:`ReferenceEquivalenceTest` compares the two over
fixed seeds, buffer sizes, ranks, and iteration limits.
"""

from __future__ import annotations

import random
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


def _conversation(ids, mask=None):
    return {"ids": list(ids), "mask": [1] * len(ids) if mask is None else mask}


def reference_rows(conversations, tokenizer, seq_len, render, *,
                   buffer_size=sft_data.PACK_BUFFER_SIZE, batch=1,
                   num_iterations=0, rank=0, world_size=1):
    """A direct transcription of ``sft_data_generator_bos_bestfit``.

    It follows ``scripts/chat_sft.py:178-300`` statement by statement, so it
    is the oracle for :class:`nanochat_cpp.sft_data.PackedRowStream`. The
    transcription stops after the batch that sets ``last_step``, which is the
    reference loop's stop rule.
    """
    dataset_size = len(conversations)
    row_capacity = seq_len + 1
    bos_token = tokenizer.get_bos_token_id()
    conv_buffer = []
    cursor = rank
    consumed = rank
    epoch = 1
    it = 0
    last_step = False
    out = []

    def refill_buffer():
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            ids, mask = render(conversations[cursor], tokenizer)
            cursor += world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1
            if len(ids) > row_capacity:
                continue
            conv_buffer.append((ids, mask))

    while True:
        rows = []
        for _ in range(batch):
            row, mask_row = [], []
            content_len = row_capacity
            while len(row) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - len(row)
                best_idx, best_len = -1, 0
                for i, (conv, _) in enumerate(conv_buffer):
                    if len(conv) <= remaining and len(conv) > best_len:
                        best_idx, best_len = i, len(conv)
                if best_idx >= 0:
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                    consumed += world_size
                else:
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    mask_row.extend([0] * remaining)
                    break
            inputs = row[:seq_len]
            targets = [token if keep else -1
                       for token, keep in zip(row[1:row_capacity],
                                              mask_row[1:row_capacity])]
            if content_len < row_capacity:
                for position in range(content_len - 1, seq_len):
                    targets[position] = -1
            rows.append((inputs, targets))
        it += 1
        if num_iterations > 0:
            approx_progress = it / num_iterations
        else:
            approx_progress = consumed / dataset_size
        del approx_progress  # the driver takes the maximum; `it` is enough here
        if consumed >= dataset_size:
            last_step = True
        if num_iterations > 0 and it >= num_iterations:
            last_step = True
        out.extend(rows)
        if last_step:
            return out


class SftDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def pack(self, conversations, seq_len, **kwargs):
        return sft_data.pack_rows(conversations, self.tokenizer, seq_len,
                                  render=_fake_render, **kwargs)

    def test_two_short_conversations_pad_the_row(self) -> None:
        # seq_len=4 gives a row capacity of 5. Two conversations of length 2
        # fill four tokens, and the packer pads the fifth with BOS.
        rows = self.pack([
            _conversation([1]),
            _conversation([2]),
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
        # The oversized conversation never fits a row, so the reference packer
        # skips it at fetch time and wraps the cursor. One epoch therefore
        # packs the short conversation twice, and `consumed` ends at 2.
        rows = self.pack([
            _conversation([1, 2, 3, 4, 5]),
            _conversation([9]),
        ], seq_len=4)
        self.assertEqual(len(rows), 1)
        tokens, targets = rows[0]
        self.assertEqual(tokens, [0, 9, 0, 9])
        self.assertEqual(targets, [9, -1, 9, -1])

    def test_a_full_row_has_no_padding(self) -> None:
        # Five tokens exactly fill the capacity, so no position is padding.
        rows = self.pack([_conversation([1, 2, 3, 4])], seq_len=4)
        tokens, targets = rows[0]
        self.assertEqual(tokens, [0, 1, 2, 3])
        self.assertEqual(targets, [1, 2, 3, 4])

    def test_empty_input_is_an_error(self) -> None:
        # The reference asserts `dataset_size > 0`; an empty shard is a bug.
        with self.assertRaises(ValueError):
            self.pack([], seq_len=4)

    def test_a_list_with_no_packable_conversation_is_an_error(self) -> None:
        # The reference would loop forever. The stream raises instead.
        with self.assertRaises(ValueError):
            self.pack([_conversation([1, 2, 3, 4, 5])], seq_len=4)

    def test_buffer_size_selects_the_largest_conversation(self) -> None:
        conversations = [_conversation([5]), _conversation([1, 2, 3])]
        # A buffer of 100 holds both, so best-fit takes the longer one first.
        greedy = self.pack(conversations, seq_len=4)
        self.assertEqual(greedy[0][0], [0, 1, 2, 3])
        # A buffer of one cannot look ahead: the short one goes first.
        lookahead = self.pack(conversations, seq_len=4, buffer_size=1)
        self.assertEqual(lookahead[0][0], [0, 5, 0, 0])

    def test_the_iteration_limit_stops_the_stream(self) -> None:
        conversations = [_conversation([value]) for value in range(1, 5)]
        rows = self.pack(conversations, seq_len=4, num_iterations=1)
        self.assertEqual(len(rows), 1)

    def test_world_size_sets_the_cursor_stride(self) -> None:
        conversations = [_conversation([value]) for value in range(1, 5)]
        rows = self.pack(conversations, seq_len=4, rank=1, world_size=2)
        self.assertEqual(rows[0][0], [0, 2, 0, 4])

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
        self.assertEqual(sft_data.PACK_BUFFER_SIZE, 100)
        self.assertEqual(sft_data.RENDER_MAX_TOKENS, 2048)


class PackedRowStreamTest(unittest.TestCase):
    """The counters a driver reads: epoch, consumed, iterations, progress."""

    def setUp(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def test_counters_end_the_stream_at_one_epoch(self) -> None:
        conversations = [_conversation([value]) for value in range(1, 7)]
        stream = sft_data.PackedRowStream(
            conversations, self.tokenizer, 4, _fake_render)
        rows = list(stream)
        # Two length-2 conversations fill a row of capacity 5, so six
        # conversations make three rows.
        self.assertEqual(len(rows), 3)
        self.assertEqual(stream.rows, 3)
        self.assertEqual(stream.iterations, 3)
        self.assertGreaterEqual(stream.consumed, 6)
        self.assertGreaterEqual(stream.epoch, 2)
        self.assertTrue(stream.last_step)
        self.assertEqual(stream.approx_progress, 1.0)
        self.assertEqual(stream.progress, 1.0)

    def test_progress_is_the_running_maximum(self) -> None:
        conversations = [_conversation([value]) for value in range(1, 7)]
        stream = sft_data.PackedRowStream(
            conversations, self.tokenizer, 4, _fake_render,
            num_iterations=2)
        rows = list(stream)
        self.assertEqual(len(rows), 2)
        self.assertEqual(stream.iterations, 2)
        self.assertEqual(stream.approx_progress, 1.0)
        self.assertEqual(stream.progress, 1.0)


class ReferenceEquivalenceTest(unittest.TestCase):
    """The Python packer must match the transcription of the reference."""

    def test_packer_matches_the_reference(self) -> None:
        tokenizer = _FakeTokenizer()
        for seed in range(8):
            rng = random.Random(seed)
            conversations = []
            for index in range(rng.randint(2, 12)):
                length = rng.randint(1, 7)
                ids = [rng.randint(1, 50) for _ in range(length)]
                mask = [rng.randint(0, 1) for _ in range(length)]
                conversations.append(_conversation(ids, mask))
            for buffer_size in (1, 2, 5, 100):
                for batch in (1, 2):
                    for world_size, rank in ((1, 0), (2, 0), (2, 1)):
                        expected = reference_rows(
                            conversations, tokenizer, 8, _fake_render,
                            buffer_size=buffer_size, batch=batch,
                            rank=rank, world_size=world_size)
                        got = list(sft_data.PackedRowStream(
                            conversations, tokenizer, 8, _fake_render,
                            buffer_size=buffer_size, batch=batch, rank=rank,
                            world_size=world_size))
                        self.assertEqual(
                            got, expected,
                            f"seed {seed} buffer {buffer_size} batch {batch} "
                            f"rank {rank} world {world_size}")


class _FakeTask(Task):
    """A task over a fixed list of values."""

    def __init__(self, values, **kwargs) -> None:
        super().__init__(**kwargs)
        self.values = list(values)

    def num_examples(self) -> int:
        return len(self.values)

    def get_example(self, index: int):
        return self.values[index]


if __name__ == "__main__":
    unittest.main()
