"""T1: the chat evaluation paths run against the CUDA backend.

The test builds a tiny model on the CUDA library, then drives the categorical
(``Evaluator.score`` with a focus set) and generative (``Model.generate``)
paths with a stub task. It proves the new ``ChatEvaluator`` reaches the device
kernels, not only the host code. Run it with
``tools/nanochat test --gpu //python:chat_gpu_test``.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from nanochat_cpp import _build, _lib, chat
from nanochat_cpp.api import Config, Model, Tokenizer

TOKENIZER = "tests/data/loader_tokenizer.nctoken"


def resolve(relative: str) -> Path:
    found = _lib.find_in_runfiles(relative)
    if found is not None:
        return found
    return _build.repo_root() / relative


class _CategoricalTask:
    """One synthetic multiple-choice problem."""

    eval_type = "categorical"

    def __init__(self, conversation: dict) -> None:
        self._conversation = conversation

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        return self._conversation

    def evaluate(self, conversation: dict, response: str) -> int:
        return int(response == conversation["messages"][-1]["content"])


class _GenerativeTask:
    """One synthetic generative problem that always passes."""

    eval_type = "generative"

    def __init__(self, conversation: dict) -> None:
        self._conversation = conversation

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        return self._conversation

    def evaluate(self, conversation: dict, completion: str) -> int:
        return 1


class ChatGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = Tokenizer.load(resolve(TOKENIZER))

    def tiny_model(self) -> Model:
        config = Config(num_layers=2, num_heads=2, num_kv_heads=2, hidden_dim=32,
                        seq_len=64, vocab_size=512, padded_vocab_size=512)
        return Model(config, seed=7)

    def test_backend_is_cuda(self) -> None:
        model = self.tiny_model()
        self.assertTrue(model.device.startswith("cuda"),
                        f"expected the CUDA backend, got {model.device}")

    def test_categorical_path(self) -> None:
        model = self.tiny_model()
        evaluator = chat.ChatEvaluator(model, tokenizer=self.tokenizer)
        conversation = {
            "messages": [
                {"role": "user",
                 "content": "Pick one.\n- a=A\n- b=B\n"
                            "Respond only with the letter."},
                {"role": "assistant", "content": "A"},
            ],
            "letters": ["A", "B"],
        }
        original = chat.build_task
        chat.build_task = lambda name: _CategoricalTask(conversation)
        try:
            report = evaluator.task("ARC-Easy")
        finally:
            chat.build_task = original
        self.assertEqual(report.total, 1)
        self.assertIn(report.accuracy, (0.0, 1.0))

    def test_generative_path(self) -> None:
        model = self.tiny_model()
        evaluator = chat.ChatEvaluator(model, tokenizer=self.tokenizer)
        conversation = {
            "messages": [
                {"role": "user", "content": "Say a number."},
                {"role": "assistant", "content": "#### 4"},
            ],
        }
        original = chat.build_task
        chat.build_task = lambda name: _GenerativeTask(conversation)
        try:
            report = evaluator.task("GSM8K", max_new_tokens=4)
        finally:
            chat.build_task = original
        self.assertEqual(report.total, 1)
        self.assertEqual(report.accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
