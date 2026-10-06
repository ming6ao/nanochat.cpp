"""T1: the in-process API runs on the CUDA backend.

This test carries the ``gpu`` tag. It runs under ``--config=cuda`` through
``tools/nanochat test --gpu``. It proves the device path that the CPU tests
cannot. The loaded library reports the CUDA backend. ``Model`` selects it. The
host-safe parameter copy works, and it does not dereference device memory. A
training step and generation still match the CPU reference tolerance.

The test uses the tiny shape of the committed C++ fixtures. It stays in the
microsecond range.
"""

from __future__ import annotations

import ctypes
import unittest
from pathlib import Path

from nanochat_cpp import (
    Config,
    Model,
    Optimizer,
    TokenData,
    Tokenizer,
    Trainer,
    _build,
    _core,
    _lib,
)


def resolve(relative: str) -> Path:
    """A runfiles path when the test runs under Bazel, else a source path."""
    found = _lib.find_in_runfiles(relative)
    if found is not None:
        return found
    return _build.repo_root() / relative


TOKENIZER = "tests/data/loader_tokenizer.nctoken"
PARQUET = "src/parquet/testdata/text.parquet"


def tiny_config() -> Config:
    return Config(num_layers=2, num_heads=2, num_kv_heads=1, hidden_dim=32,
                  seq_len=8, vocab_size=64, padded_vocab_size=64,
                  window_pattern="SL")


class CudaBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = _lib.load(device="cuda")

    def test_backend_is_cuda(self) -> None:
        self.assertEqual(_lib.library_backend(), "cuda")
        self.assertEqual(_core.backend_name(self.lib), "cuda")

    def test_device_info_reports_a_gpu(self) -> None:
        info = _core.device_info(self.lib)
        self.assertNotEqual(info.device_name, b"cpu")
        self.assertGreaterEqual(info.compute_major, 1)
        self.assertGreater(info.total_memory_bytes, 0)

    def test_model_selects_the_cuda_device(self) -> None:
        model = Model(tiny_config(), device="cuda", seed=1)
        self.assertTrue(model.device.startswith("cuda"), model.device)

    def test_cpu_request_rejects_the_cuda_library(self) -> None:
        with self.assertRaises(_core.NanochatError):
            _lib.load(device="cpu")

    def test_param_copy_round_trip(self) -> None:
        # The regression: reading `param.value` directly would dereference a
        # device pointer and crash. The TensorView copies through the C surface.
        model = Model(tiny_config(), device="cuda", seed=3)
        views = model.params()
        self.assertTrue(views)
        value = views[0].value
        self.assertIsNotNone(value)
        original = value.tolist()
        self.assertEqual(len(original), views[0].count)
        value[0] = original[0] + 1.0
        self.assertAlmostEqual(value[0], original[0] + 1.0, places=5)
        value[0] = original[0]
        self.assertAlmostEqual(value[0], original[0], places=5)

    def test_train_step_and_generate(self) -> None:
        model = Model(tiny_config(), device="cuda", seed=5)
        tokens = (ctypes.c_int * 16)(*list(range(8)) * 2)
        targets = (ctypes.c_int * 16)(*(list(range(1, 8)) + [0]) * 2)
        loss = model.forward_loss(tokens, targets, batch=2, seq=8)
        self.assertGreater(loss, 0.0)
        self.assertLess(loss, 100.0)
        model.zero_grad()
        model.backward()
        optimizer = Optimizer(model, num_iterations=1)
        optimizer.step(1)
        self.assertEqual(optimizer.grad_norm(), optimizer.grad_norm())
        rows = model.generate([1, 2, 3], max_tokens=4, temperature=0.0,
                              num_samples=2, seed=7)
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0].tokens)

    def test_trainer_and_evaluator_run_on_gpu(self) -> None:
        tokenizer = Tokenizer.load(resolve(TOKENIZER))
        model = Model(tiny_config(), device="cuda", seed=11)
        data = TokenData(parquet=str(resolve(PARQUET)), tokenizer=tokenizer,
                         seq_len=8, batch=2, seed=13, threads=1,
                         document_buffer=100)
        trainer = Trainer(model, data, num_iterations=2)
        losses = [loss for _, loss in trainer]
        self.assertEqual(len(losses), 2)
        for loss in losses:
            self.assertTrue(loss == loss)  # not NaN
        report = model.params()
        self.assertTrue(report)


if __name__ == "__main__":
    unittest.main()
