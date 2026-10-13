"""T0: the single-client multi-device runtime client.

Two and four replicas that split one global batch equal one replica on the whole
batch, and the replicas of one runtime stay equal. The test uses a simulated CPU
mesh, so it needs no GPU and no broker
(docs/client-runtime-design.md sections 10 and 16).
"""

from __future__ import annotations

import array
import unittest

from nanochat_cpp import _core, _lib, mesh, runtime


def tiny_config() -> _core.Config:
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
    config.clip = 1.0
    config.num_iterations = 2
    config.warmdown_ratio = 0.65
    config.muon_momentum_start = 0.85
    config.muon_momentum_peak = 0.97
    config.muon_momentum_final = 0.90
    return config


def ids(rows: int, vocab: int, a: int, b: int) -> list[int]:
    return [(i * a + b) % vocab for i in range(rows)]


class RuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = _lib.load()

    def make_runtime(self, size: int, batch: int, grad_accum: int):
        devices = mesh.devices()
        # A simulated mesh of the requested size: the runtime only needs the
        # mesh size, so a CPU host can run a multi-device test.
        simulated = mesh.Mesh(tuple(devices[0] for _ in range(size)))
        return runtime.MeshRuntime(
            simulated,
            tiny_config(),
            optimizer_config(),
            batch=batch,
            grad_accum=grad_accum,
            seq=8,
            seed=20241001,
        )

    def test_accepts_high_level_config(self) -> None:
        from nanochat_cpp import Config

        devices = mesh.devices()
        simulated = mesh.Mesh(tuple(devices[0] for _ in range(2)))
        config = Config(
            num_layers=2,
            num_heads=8,
            num_kv_heads=2,
            hidden_dim=32,
            seq_len=8,
            vocab_size=64,
            padded_vocab_size=64,
            window_pattern="SL",
        )
        with runtime.MeshRuntime(simulated, config, optimizer_config(),
                                 batch=4, grad_accum=2, seq=8) as rt:
            self.assertEqual(rt.world_size, 2)
            self.assertGreater(rt.param_count(0), 0)

    def test_runtime_shape(self) -> None:
        with self.make_runtime(2, batch=4, grad_accum=2) as rt:
            self.assertEqual(rt.world_size, 2)
            self.assertEqual(rt.mesh.size, 2)
            self.assertEqual(rt.param_count(0), rt.param_count(1))
            self.assertGreater(rt.param_count(0), 0)
            self.assertTrue(rt.param_name(0, 0))

    def test_two_replicas_equal_one(self) -> None:
        batch, seq, accum = 4, 8, 2
        rows = batch * seq
        tokens = ids(rows, 64, 7, 3)
        targets = ids(rows, 64, 11, 5)

        single = self.make_runtime(1, batch, accum)
        pair = self.make_runtime(2, batch, accum)
        try:
            loss_single = single.train_step(tokens, targets)
            loss_pair = pair.train_step(tokens, targets)
            self.assertAlmostEqual(loss_single, loss_pair, places=5)
            for replica in (0, 1):
                self._assert_same_parameters(single, 0, pair, replica)
        finally:
            single.close()
            pair.close()

    def test_four_replicas_equal_one(self) -> None:
        batch, seq, accum = 8, 8, 1
        rows = batch * seq
        tokens = ids(rows, 64, 7, 3)
        targets = ids(rows, 64, 11, 5)

        single = self.make_runtime(1, batch, accum)
        quad = self.make_runtime(4, batch, accum)
        try:
            single.train_step(tokens, targets)
            quad.train_step(tokens, targets)
            for replica in range(4):
                self._assert_same_parameters(single, 0, quad, replica)
        finally:
            single.close()
            quad.close()

    def test_replicas_stay_equal(self) -> None:
        batch, seq, accum = 4, 8, 2
        rows = batch * seq
        tokens = ids(rows, 64, 13, 1)
        targets = ids(rows, 64, 17, 9)

        with self.make_runtime(2, batch, accum) as pair:
            for _ in range(3):
                pair.train_step(tokens, targets)
            self._assert_same_parameters(pair, 0, pair, 1, tolerance=0.0)

    def test_save_and_load(self) -> None:
        import os
        import tempfile

        batch, seq, accum = 4, 8, 2
        rows = batch * seq
        tokens = ids(rows, 64, 7, 3)

        single = self.make_runtime(1, batch, accum)
        pair = self.make_runtime(2, batch, accum)
        try:
            single.train_step(tokens, tokens)
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "checkpoint.ncmdl")
                single.save(path)
                pair.load(path)
            self._assert_same_parameters(single, 0, pair, 0, tolerance=0.0)
            self._assert_same_parameters(pair, 0, pair, 1, tolerance=0.0)
        finally:
            single.close()
            pair.close()

    def test_metrics_after_step(self) -> None:
        batch, seq, accum = 4, 8, 2
        rows = batch * seq
        tokens = ids(rows, 64, 7, 3)
        targets = ids(rows, 64, 11, 5)

        with self.make_runtime(2, batch, accum) as pair:
            loss = pair.train_step(tokens, targets)
            metrics = pair.metrics()

        self.assertEqual(len(metrics.loss), 2)
        self.assertAlmostEqual(metrics.mean_loss, loss, places=5)
        self.assertGreaterEqual(metrics.grad_norm, 0.0)
        self.assertGreater(metrics.step_seconds, 0.0)

    def test_train_step_accepts_array_batches(self) -> None:
        # The parquet loader returns `array.array` batches. The int buffer must
        # iterate one, not hand the whole buffer to `int()`.
        batch, seq, accum = 4, 8, 2
        rows = batch * seq
        tokens = array.array("i", ids(rows, 64, 7, 3))
        targets = array.array("i", ids(rows, 64, 11, 5))

        with self.make_runtime(1, batch, accum) as single:
            loss = single.train_step(tokens, targets)
        self.assertTrue(loss == loss, "the loss must be finite")

    def test_train_step_accepts_raw_buffers(self) -> None:
        # A caller may pass the raw int32 bytes or a byte memoryview. The
        # helper must reinterpret those as int32, not iterate the bytes.
        batch, seq, accum = 4, 8, 2
        rows = batch * seq
        tokens = array.array("i", ids(rows, 64, 7, 3)).tobytes()
        targets = memoryview(array.array("i", ids(rows, 64, 11, 5)).tobytes())

        with self.make_runtime(1, batch, accum) as single:
            loss = single.train_step(tokens, targets)
        self.assertTrue(loss == loss, "the loss must be finite")

    def test_replica_digest_matches(self) -> None:
        batch, seq, accum = 4, 8, 2
        rows = batch * seq
        tokens = ids(rows, 64, 7, 3)
        targets = ids(rows, 64, 11, 5)

        with self.make_runtime(2, batch, accum) as pair:
            before = pair.replica_digest(0)
            self.assertEqual(before, pair.replica_digest(1))
            pair.train_step(tokens, targets)
            after = pair.replica_digest(0)
            self.assertEqual(after, pair.replica_digest(1))
            self.assertNotEqual(before, after,
                                "a step must change the parameters")

    def test_optimizer_config_builder(self) -> None:
        from nanochat_cpp import optimizer_config

        config = optimizer_config(num_iterations=7, matrix_lr=0.5)
        self.assertEqual(config.num_iterations, 7)
        self.assertAlmostEqual(config.matrix_lr, 0.5, places=6)
        self.assertAlmostEqual(config.embedding_lr, 0.2, places=6)
        with self.assertRaises(TypeError):
            optimizer_config(not_a_field=1)

    def test_track3_preset_matches_the_harness(self) -> None:
        from nanochat_cpp import preset_optimizer_config

        config = preset_optimizer_config("track3")
        self.assertAlmostEqual(config.embedding_lr, 0.7, places=6)
        self.assertAlmostEqual(config.unembedding_lr, 0.004, places=6)
        self.assertAlmostEqual(config.scalar_lr, 0.015, places=6)
        self.assertAlmostEqual(config.matrix_lr, 0.025, places=6)
        self.assertAlmostEqual(config.warmdown_ratio, 0.7, places=6)
        self.assertAlmostEqual(config.final_lr_frac, 0.0, places=6)
        self.assertAlmostEqual(config.weight_decay_base, 0.05, places=6)
        with self.assertRaises(_core.NanochatError):
            preset_optimizer_config("unknown")

    def test_save_and_load_resumes_the_step(self) -> None:
        import os
        import tempfile

        batch, seq, accum = 4, 8, 2
        rows = batch * seq
        tokens = ids(rows, 64, 7, 3)
        targets = ids(rows, 64, 11, 5)

        uninterrupted = self.make_runtime(1, batch, accum)
        resumed = self.make_runtime(1, batch, accum)
        try:
            uninterrupted.train_step(tokens, targets)
            uninterrupted.train_step(tokens, targets)
            self.assertEqual(uninterrupted.step, 2)

            resumed.train_step(tokens, targets)
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "checkpoint.nchkpt01")
                resumed.save(path)
                resumed.load(path)
            self.assertEqual(resumed.step, 1)
            resumed.train_step(tokens, targets)
            self.assertEqual(resumed.step, 2)

            self._assert_same_parameters(uninterrupted, 0, resumed, 0,
                                         tolerance=0.0)
        finally:
            uninterrupted.close()
            resumed.close()

    def _assert_same_parameters(self, a, a_replica, b, b_replica,
                                tolerance: float = 1e-4) -> None:
        count = a.param_count(a_replica)
        self.assertEqual(count, b.param_count(b_replica))
        worst = 0.0
        for index in range(count):
            left = a.param_read(index, a_replica)
            right = b.param_read(index, b_replica)
            self.assertEqual(len(left), len(right))
            for x, y in zip(left, right):
                worst = max(worst, abs(x - y))
        self.assertLessEqual(worst, tolerance,
                             f"parameter mismatch, worst = {worst}")


if __name__ == "__main__":
    unittest.main()
