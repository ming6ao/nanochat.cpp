"""T0: the chat prompt renderer and the per-task isolation.

The fixture ``//tests:data/chat_fixture.bin`` records the completion prompt
that the reference ``RustBPETokenizer.render_for_completion`` produces for a
few synthetic conversations. The test renders the same conversations with
:class:`nanochat_cpp.chat._PromptRenderer` and compares the ids. It also checks
the per-task failure isolation and the core aggregation. See docs/python.md
sections 5 and 11.
"""

from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path

from nanochat_cpp import _build, _lib, chat, tasks
from nanochat_cpp.api import Config, Model, Tokenizer

FIXTURE = "tests/data/chat_fixture.bin"
TOKENIZER = "tests/data/loader_tokenizer.nctoken"
MAGIC = b"NANCHAT1"


def resolve(relative: str) -> Path:
    """A runfiles path when the test runs under Bazel, else a source path."""
    found = _lib.find_in_runfiles(relative)
    if found is not None:
        return found
    return _build.repo_root() / relative


def load_fixture(path: Path) -> dict:
    raw = path.read_bytes()
    assert raw[:8] == MAGIC, f"not a {MAGIC!r} fixture: {path}"
    version = struct.unpack("<I", raw[8:12])[0]
    assert version == 1, f"unsupported chat fixture version {version}"
    return json.loads(raw[12:].decode("utf-8"))


def _tiny_model() -> Model:
    """A two-layer model whose vocabulary covers the fixture tokenizer."""
    config = Config(num_layers=2, num_heads=2, num_kv_heads=2, hidden_dim=32,
                    seq_len=8, vocab_size=512, padded_vocab_size=512)
    return Model(config, seed=1234)


class _NoSpecials:
    """A tokenizer stand-in that lacks every chat special token."""

    special_tokens: dict = {}

    def encode(self, text):  # noqa: ANN001 - test double
        return []

    def encode_special(self, name):  # noqa: ANN001 - test double
        raise KeyError(name)


class ChatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = load_fixture(resolve(FIXTURE))
        cls.tokenizer = Tokenizer.load(resolve(TOKENIZER))

    def test_renderer_matches_reference(self) -> None:
        renderer = chat._PromptRenderer(self.tokenizer)
        self.assertTrue(self.fixture["cases"])
        for case in self.fixture["cases"]:
            with self.subTest(case=case["name"]):
                got = renderer.render_for_completion(case["conversation"])
                self.assertEqual(got, case["prompt_ids"])

    def test_renderer_rejects_missing_special_tokens(self) -> None:
        with self.assertRaises(ValueError):
            chat._PromptRenderer(_NoSpecials())

    def test_render_conversation_masks(self) -> None:
        conversation = {"messages": [
            {"role": "user", "content": "Say hi"},
            {"role": "assistant", "content": "Hello there"}]}
        ids, mask = chat.render_conversation(conversation, self.tokenizer)
        self.assertEqual(len(ids), len(mask))
        specials = chat._specials(self.tokenizer)
        self.assertEqual(ids[0], specials["<|bos|>"])
        self.assertEqual(mask[0], 0)
        self.assertEqual(ids[-1], specials["<|assistant_end|>"])
        self.assertEqual(mask[-1], 1)
        # The user span, markers included, is not a target.
        user_start = ids.index(specials["<|user_start|>"])
        user_end = ids.index(specials["<|user_end|>"])
        self.assertEqual(mask[user_start:user_end + 1],
                         [0] * (user_end - user_start + 1))
        # Everything after the assistant marker is a target.
        assistant_start = ids.index(specials["<|assistant_start|>"])
        self.assertEqual(mask[assistant_start], 0)
        self.assertTrue(all(flag == 1
                            for flag in mask[assistant_start + 1:]))

    def test_render_conversation_python_mask(self) -> None:
        conversation = {"messages": [
            {"role": "user", "content": "compute"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me "},
                {"type": "python", "text": "print(1)"},
                {"type": "python_output", "text": "1"}]}]}
        ids, mask = chat.render_conversation(conversation, self.tokenizer)
        self.assertEqual(len(ids), len(mask))
        specials = chat._specials(self.tokenizer)
        py_start = ids.index(specials["<|python_start|>"])
        py_end = ids.index(specials["<|python_end|>"])
        self.assertTrue(all(flag == 1 for flag in mask[py_start:py_end + 1]))
        out_start = ids.index(specials["<|output_start|>"])
        out_end = ids.index(specials["<|output_end|>"])
        self.assertTrue(all(flag == 0
                            for flag in mask[out_start:out_end + 1]))
        self.assertEqual(mask[-1], 1)

    def test_baselines_and_task_lists(self) -> None:
        self.assertEqual(chat.ALL_TASKS, ("ARC-Easy", "ARC-Challenge", "MMLU",
                                          "GSM8K", "HumanEval"))
        self.assertEqual(chat.CATEGORICAL_TASKS,
                         ("ARC-Easy", "ARC-Challenge", "MMLU"))
        for name, baseline in chat.BASELINES.items():
            self.assertIn(name, chat.ALL_TASKS)
            self.assertGreaterEqual(baseline, 0.0)

    def test_centered_mean(self) -> None:
        reports = [
            chat.TaskReport("ARC-Easy", 0, 0, 0.5, 0.25),
            chat.TaskReport("ARC-Challenge", 0, 0, 0.5, 0.25),
            chat.TaskReport("MMLU", 0, 0, 0.5, 0.25),
        ]
        expected = ((0.5 - 0.25) / 0.75) * 3 / 3
        value = chat._centered_mean(reports, chat.CATEGORICAL_TASKS)
        self.assertAlmostEqual(value, expected)
        # A missing task makes the mean undefined.
        self.assertIsNone(chat._centered_mean(reports, chat.ALL_TASKS))
        # A skipped generative task does not invalidate the categorical core.
        skipped = reports + [chat.TaskReport("GSM8K", 0, 0, 0.0, 0.0,
                                             skipped=True, error="x")]
        self.assertAlmostEqual(
            chat._centered_mean(skipped, chat.CATEGORICAL_TASKS), expected)
        # A skipped categorical task invalidates it.
        broken = [reports[0], reports[1],
                  chat.TaskReport("MMLU", 0, 0, 0.0, 0.25, skipped=True,
                                  error="x")]
        self.assertIsNone(
            chat._centered_mean(broken, chat.CATEGORICAL_TASKS))

    def test_run_isolates_a_failing_task(self) -> None:
        evaluator = chat.ChatEvaluator(_tiny_model(),
                                       tokenizer=self.tokenizer)
        original = chat.build_task

        def unavailable(name):  # noqa: ANN001 - test double
            raise SystemExit("no reference checkout")

        chat.build_task = unavailable
        try:
            report = evaluator.run(["MMLU", "GSM8K"])
        finally:
            chat.build_task = original
        self.assertEqual([r.name for r in report.tasks], ["MMLU", "GSM8K"])
        for task_report in report.tasks:
            self.assertTrue(task_report.skipped)
            self.assertIsNotNone(task_report.error)
        self.assertIsNone(report.chat_core)
        self.assertIsNone(report.categorical_core)

    def test_run_re_raises_keyboard_interrupt(self) -> None:
        evaluator = chat.ChatEvaluator(_tiny_model(),
                                       tokenizer=self.tokenizer)
        original = chat.build_task

        def interrupt(name):  # noqa: ANN001 - test double
            raise KeyboardInterrupt

        chat.build_task = interrupt
        try:
            with self.assertRaises(KeyboardInterrupt):
                evaluator.run(["MMLU"])
        finally:
            chat.build_task = original

    def test_task_rejects_unknown_name(self) -> None:
        evaluator = chat.ChatEvaluator(_tiny_model(),
                                       tokenizer=self.tokenizer)
        with self.assertRaises(ValueError):
            evaluator.task("Nonsense")

    def test_prompt_and_extraction_rules(self) -> None:
        # The multiple-choice prompt is whitespace-exact: the letter follows the
        # choice with no space, matching the bare-letter assistant answer.
        prompt = tasks.render_mc("What?", ["A", "B"], ["first", "second"])
        self.assertEqual(prompt, (
            "Multiple Choice question: What?\n"
            "- first=A\n"
            "- second=B\n"
            "\nRespond only with the letter of the correct answer."))

        self.assertEqual(tasks.extract_answer("steps #### 1,234"), "1234")
        self.assertEqual(tasks.extract_answer("#### -3.5"), "-3.5")
        self.assertIsNone(tasks.extract_answer("no marker"))

        self.assertEqual(
            tasks.extract_program("```python\nprint(1)\n```\nafter"),
            "print(1)")
        self.assertEqual(tasks.extract_program("```\nprint(2)\n```"),
                         "print(2)")
        self.assertEqual(tasks.extract_program("print(3)"), "print(3)")
        self.assertEqual(
            tasks.extract_imports("import math\nfrom os import path\nx = 1"),
            "import math\nfrom os import path")

    def test_task_slicing(self) -> None:
        class _Stub(tasks.Task):
            def num_examples(self):
                return 10

            def get_example(self, index):
                return index

        sliced = _Stub(start=2, stop=8, step=2)
        self.assertEqual(len(sliced), 3)
        self.assertEqual([sliced[i] for i in range(len(sliced))],
                         [2, 4, 6])

    def test_parse_tasks(self) -> None:
        self.assertEqual(chat._parse_tasks("ARC-Easy|GSM8K"),
                         ["ARC-Easy", "GSM8K"])
        self.assertEqual(chat._parse_tasks("ARC-Easy,GSM8K"),
                         ["ARC-Easy", "GSM8K"])
        self.assertEqual(chat._parse_tasks(None), list(chat.ALL_TASKS))
        with self.assertRaises(SystemExit):
            chat._parse_tasks("Nonsense")


if __name__ == "__main__":
    unittest.main()
