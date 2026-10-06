"""T0: the ctypes core loads the CPU library and runs a forward/backward step.

The test uses the same tiny shape as the committed C++ fixtures: two layers,
eight query heads, two key/value heads, hidden width 32, sequence length 8,
and a vocabulary of 64. See docs/python-api.md section 11.
"""

from __future__ import annotations

import ctypes
import unittest

from nanochat_cpp import _core, _lib


def tiny_config() -> _core.Config:
    return _core.Config(2, 8, 2, 32, 8, 64, 64, 100000.0, b"SL")


def optimizer_config() -> _core.OptimizerConfig:
    config = _core.OptimizerConfig()
    config.unembedding_lr = 0.004
    config.embedding_lr = 0.2
    config.matrix_lr = 0.02
    config.scalar_lr = 0.5
    config.muon_ns_steps = 5
    config.muon_beta2 = 0.9
    config.adam_eps = 1e-10
    config.num_iterations = 1
    config.warmdown_ratio = 0.65
    config.muon_momentum_start = 0.85
    config.muon_momentum_peak = 0.97
    config.muon_momentum_final = 0.90
    return config


class CoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = _lib.load()

    def test_version_is_a_nonempty_string(self) -> None:
        version = self.lib.nanochat_version()
        self.assertIsInstance(version, bytes)
        self.assertTrue(version)

    def test_struct_field_order(self) -> None:
        # A field-order drift would silently corrupt the C conversion.
        self.assertEqual(_core.Config.num_layers.offset, 0)
        self.assertEqual(_core.Config.rope_base.offset, 7 * 4)
        self.assertEqual(_core.Param.count.offset, 24)
        self.assertEqual(_core.OptimizerConfig.muon_ns_steps.offset, 5 * 4)
        self.assertEqual(_core.OptimizerConfig.num_iterations.offset, 9 * 4)

    def test_null_config_raises(self) -> None:
        with self.assertRaises(_core.NanochatError):
            handle = self.lib.nanochat_model_create(None, 0)
            _core.check_handle(self.lib, handle, "nanochat_model_create")

    def test_forward_and_backward(self) -> None:
        config = tiny_config()
        handle = _core.check_handle(
            self.lib,
            self.lib.nanochat_model_create(ctypes.byref(config),
                                           ctypes.c_uint64(1234)),
            "nanochat_model_create")
        try:
            self.assertGreater(self.lib.nanochat_param_count(handle), 0)
            tokens = (ctypes.c_int * 16)(*list(range(8)) * 2)
            targets = (ctypes.c_int * 16)(*(list(range(1, 8)) + [0]) * 2)
            loss = self.lib.nanochat_forward_loss(handle, tokens, targets, 2, 8)
            self.assertGreater(loss, 0.0)
            self.assertLess(loss, 100.0)
            self.lib.nanochat_zero_grad(handle)
            self.lib.nanochat_backward(handle)
        finally:
            self.lib.nanochat_model_free(handle)

    def test_param_info(self) -> None:
        config = tiny_config()
        handle = _core.check_handle(
            self.lib,
            self.lib.nanochat_model_create(ctypes.byref(config),
                                           ctypes.c_uint64(7)),
            "nanochat_model_create")
        try:
            count = self.lib.nanochat_param_count(handle)
            param = _core.Param()
            _core.check_code(
                self.lib,
                self.lib.nanochat_param_info(handle, 0, ctypes.byref(param)),
                "nanochat_param_info")
            self.assertTrue(param.name)
            self.assertGreater(param.count, 0)
        finally:
            self.lib.nanochat_model_free(handle)

    def test_optimizer_step(self) -> None:
        config = tiny_config()
        model = _core.check_handle(
            self.lib,
            self.lib.nanochat_model_create(ctypes.byref(config),
                                           ctypes.c_uint64(11)),
            "nanochat_model_create")
        try:
            options = optimizer_config()
            optimizer = _core.check_handle(
                self.lib,
                self.lib.nanochat_optim_create(model, ctypes.byref(options)),
                "nanochat_optim_create")
            try:
                tokens = (ctypes.c_int * 16)(*list(range(8)) * 2)
                targets = (ctypes.c_int * 16)(*(list(range(1, 8)) + [0]) * 2)
                self.lib.nanochat_forward_loss(model, tokens, targets, 2, 8)
                self.lib.nanochat_backward(model)
                self.lib.nanochat_optim_step(optimizer, 1)
                norm = self.lib.nanochat_optim_grad_norm(optimizer)
                self.assertEqual(norm, norm)  # not NaN
            finally:
                self.lib.nanochat_optim_free(optimizer)
        finally:
            self.lib.nanochat_model_free(model)


if __name__ == "__main__":
    unittest.main()
