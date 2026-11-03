"""T0: the three C entry points that ``docs/api-surface.md`` R11a found unbound.

``nanochat_generate_multi``, ``nanochat_rl_step``, and
``nanochat_optim_config_size`` were declared in ``include/nanochat/capi.h`` and
had no row in the ``ctypes`` function table of ``nanochat_cpp._core``, so a
Python caller could not reach them and nothing noticed. Each test below calls
one of them against the loaded library, which is what keeps the row from going
missing again. The last two also drive the guard that ``nanochat_cpp._lib.load``
applies on every load, so a library built from another
``nanochat_optim_config`` layout fails at import instead of misreading the
optimizer fields.

The tiny shape is the one of the committed C++ fixtures: two layers, eight
query heads, two key/value heads, hidden width 32, sequence length 8, and a
vocabulary of 64.
"""

from __future__ import annotations

import ctypes
import math
import unittest

from nanochat_cpp import _core, _lib


def tiny_config() -> _core.Config:
    # The trailing 1 enables the value embeddings, so the tiny shape keeps the
    # odd-layer value gate that the C++ fixture counts (bindings test).
    return _core.Config(2, 8, 2, 32, 8, 64, 64, 100000.0, b"SL", 1)


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


class WrongSizeLibrary:
    """A library whose ``nanochat_optim_config`` is four bytes longer."""

    def nanochat_optim_config_size(self) -> int:
        return ctypes.sizeof(_core.OptimizerConfig) + 4


class StaleLibrary:
    """A library that predates ``nanochat_optim_config_size``."""


class CapiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = _lib.load()

    def _model(self, seed: int = 5):
        config = tiny_config()
        return _core.check_handle(
            self.lib,
            self.lib.nanochat_model_create(ctypes.byref(config),
                                           ctypes.c_uint64(seed)),
            "nanochat_model_create", _core.KIND_MODEL)

    def test_generate_multi_returns_a_row_per_prompt(self) -> None:
        model = self._model()
        try:
            # Three prompts of two ids, one sample each: the C documentation
            # promises `num_prompts * params.num_samples` rows, prompt-major,
            # and each row starts with its own prompt.
            params = _core.GenerateMultiParams()
            params.num_prompts = 3
            params.prompt_len = 2
            params.row_stops = None
            params.params.num_samples = 1
            params.params.max_tokens = 4
            params.params.temperature = 1.0
            params.params.top_k = 0
            params.params.seed = 7
            params.params.stop_id = -1
            params.params.bos_id = -1
            prompts = [1, 2, 3, 4, 5, 6]
            prompt_buffer = (ctypes.c_int * len(prompts))(*prompts)
            out = _core.Sequences()
            self.lib.nanochat_generate_multi(
                model, prompt_buffer, ctypes.byref(params), ctypes.byref(out))
            self.assertEqual(out.count,
                             params.num_prompts * params.params.num_samples)
            for row in range(int(out.count)):
                offset = int(out.offsets[row])
                length = int(out.lengths[row])
                self.assertEqual(length,
                                 params.prompt_len + params.params.max_tokens)
                start = row * params.prompt_len
                self.assertEqual(
                    [int(out.tokens[offset + i])
                     for i in range(params.prompt_len)],
                    prompts[start:start + params.prompt_len])
        finally:
            self.lib.nanochat_model_free(model)

    def test_rl_step_returns_the_forward_loss(self) -> None:
        model = self._model()
        optimizer = _core.check_handle(
            self.lib,
            self.lib.nanochat_optim_create(
                model, ctypes.byref(optimizer_config())),
            "nanochat_optim_create", _core.KIND_OPTIMIZER)
        try:
            batch, seq = 2, 4
            tokens = [3, 5, 7, 9, 11, 13, 15, 17]
            targets = [5, 7, 9, 11, 13, 15, 17, 19]
            advantages = [1.0, -0.5, 0.25, 0.0, 0.75, -0.25, 0.5, 0.125]
            token_buffer = (ctypes.c_int * len(tokens))(*tokens)
            target_buffer = (ctypes.c_int * len(targets))(*targets)
            advantage_buffer = (ctypes.c_float * len(advantages))(*advantages)

            # `nanochat_rl_step` returns the forward mean loss, and the forward
            # it runs is the one `nanochat_forward_loss` runs, so a second
            # opinion on the same weights pins both the result and the order of
            # the two input buffers.
            expected = float(self.lib.nanochat_forward_loss(
                model, token_buffer, target_buffer, batch, seq))
            loss = float(self.lib.nanochat_rl_step(
                model, optimizer, token_buffer, target_buffer,
                advantage_buffer, batch, seq, 1, 1, 1))
            self.assertTrue(math.isfinite(loss))
            self.assertGreater(loss, 0.0)
            self.assertAlmostEqual(loss, expected, places=5)
        finally:
            self.lib.nanochat_optim_free(optimizer)
            self.lib.nanochat_model_free(model)

    def test_optim_config_size_matches_the_python_mirror(self) -> None:
        mirrored = ctypes.sizeof(_core.OptimizerConfig)
        self.assertEqual(_core.check_optim_config_size(self.lib), mirrored)

    def test_the_size_guard_rejects_a_wrong_size(self) -> None:
        # The guard is the import-time gate of `docs/api-surface.md` R11a: a
        # library whose struct is another size must fail before a caller fills
        # the mirror in, because a one-field shift renames every field after it.
        with self.assertRaises(_core.NanochatError) as caught:
            _core.check_optim_config_size(WrongSizeLibrary())
        self.assertEqual(caught.exception.function,
                         "nanochat_optim_config_size")
        self.assertIn("nanochat_optim_config", str(caught.exception))

        # A library that predates the query cannot be checked either, so it is
        # the same gate and not a missing symbol at the first optimizer call.
        with self.assertRaises(_core.NanochatError) as stale:
            _core.check_optim_config_size(StaleLibrary())
        self.assertIn("does not export", str(stale.exception))


if __name__ == "__main__":
    unittest.main()
