"""T0: the pure helpers of the mesh training loop.

The test covers the batch mapping, the model FLOPs utilization, the summary
statistics, and the loop itself against a fake runtime and loader. It needs no
shared library, no GPU, and no broker. See ``docs/client-runtime-design.md``
and ``docs/notebook-workflow.md``.
"""

from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

import mesh_train


class GlobalBatchTest(unittest.TestCase):
    def test_matches_the_multi_process_total(self) -> None:
        # One sequence per rank, eight micro-batches, two ranks: 65536 tokens
        # at seq 4096.
        self.assertEqual(mesh_train.global_batch(1, 8, 2), 16)
        self.assertEqual(mesh_train.global_batch(1, 8, 2) * 4096, 65536)

    def test_rejects_a_nonpositive_factor(self) -> None:
        for values in ((0, 1, 1), (1, 0, 1), (1, 1, 0)):
            with self.assertRaises(ValueError):
                mesh_train.global_batch(*values)


class MfuTest(unittest.TestCase):
    def test_scales_with_throughput(self) -> None:
        self.assertAlmostEqual(mesh_train.mfu(100.0, 2.0, 1000.0), 0.2)

    def test_clamps_above_one(self) -> None:
        self.assertEqual(mesh_train.mfu(1e9, 1.0, 1.0), 1.0)

    def test_unknown_peak_is_zero(self) -> None:
        self.assertEqual(mesh_train.mfu(100.0, 2.0, 0.0), 0.0)


class SummaryTest(unittest.TestCase):
    def test_mean_loss(self) -> None:
        rows = [{"loss": 4.0}, {"loss": 2.0}]
        self.assertEqual(mesh_train.mean_loss(rows), 3.0)
        self.assertEqual(mesh_train.mean_loss([]), 0.0)

    def test_steady_mfu_uses_the_second_half(self) -> None:
        rows = [{"mfu": 0.0}, {"mfu": 0.0}, {"mfu": 0.4}, {"mfu": 0.6}]
        self.assertAlmostEqual(mesh_train.steady_mfu(rows), 0.5)


class _Metrics:
    def __init__(self, loss, grad_norm=1.0):
        self.loss = tuple(loss)
        self.grad_norm = grad_norm
        self.mean_loss = sum(loss) / len(loss)


class _FakeRuntime:
    def __init__(self) -> None:
        self.steps = 0
        self.saved: list[str] = []
        self._last = _Metrics([10.0, 10.0])

    def train_step(self, tokens, targets) -> float:
        self.steps += 1
        loss = 10.0 - self.steps
        self._last = _Metrics([loss, loss])
        return loss

    def metrics(self):
        return self._last

    def save(self, path) -> None:
        self.saved.append(str(path))


class _FakeData:
    def __init__(self, batches: int) -> None:
        self.batches = batches
        self.count = 0

    def next(self):
        if self.count >= self.batches:
            return None
        self.count += 1
        return [0] * 8, [0] * 8

    def reset(self) -> None:
        self.count = 0


def _plan(tmp: str, **overrides) -> mesh_train.MeshTrainPlan:
    values = dict(
        train_parquet="train.parquet",
        tokenizer=object(),
        out_dir=tmp,
        config=types.SimpleNamespace(seq_len=8),
        per_rank_batch=1,
        grad_accum=2,
        world_size=2,
        steps=4,
        save_every=2,
        flops_per_token=2.0,
        peak_flops=1000.0,
    )
    values.update(overrides)
    return mesh_train.MeshTrainPlan(**values)


def _fake_clock(step_seconds: float = 0.1):
    """A deterministic clock that advances by ``step_seconds`` per call."""
    state = {"now": 0.0}

    def clock() -> float:
        state["now"] += step_seconds
        return state["now"]

    return clock


class IterateTest(unittest.TestCase):
    def test_writes_one_record_per_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _plan(tmp)
            runtime = _FakeRuntime()
            seen: list[int] = []

            def progress(row, total):
                seen.append(row["step"])

            records = mesh_train.iterate(runtime, _FakeData(plan.steps), plan,
                                         progress=progress,
                                         clock=_fake_clock())

            self.assertEqual(len(records), plan.steps)
            self.assertEqual(runtime.steps, plan.steps)
            self.assertEqual(seen, [1, 2, 3, 4])
            self.assertEqual(records[0]["loss"], 9.0)
            self.assertEqual(records[0]["device_loss"], [9.0, 9.0])
            # 32 tokens per step over 0.1 seconds, times 2 FLOP per token,
            # over 1000 peak FLOP: 0.64.
            self.assertAlmostEqual(records[0]["mfu"], 0.64)
            self.assertAlmostEqual(records[0]["tokens_per_sec"], 320.0)
            # The save cadence fires on steps 2 and 4.
            self.assertEqual(len(runtime.saved), 2)

            written = plan.metrics_path().read_text(encoding="utf-8")
            lines = [json.loads(line) for line in written.splitlines()]
            self.assertEqual(len(lines), plan.steps)
            self.assertEqual(lines[-1]["step"], plan.steps)

    def test_resets_a_short_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _plan(tmp, steps=4, save_every=0)
            runtime = _FakeRuntime()
            # The loader holds one batch, so every step reuses it.
            mesh_train.iterate(runtime, _FakeData(1), plan)
            self.assertEqual(runtime.steps, 4)

    def test_tokens_per_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _plan(tmp)
            self.assertEqual(plan.sequence_length, 8)
            self.assertEqual(plan.global_sequences, 4)
            self.assertEqual(plan.tokens_per_step, 32)

    def test_rejects_an_empty_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _plan(tmp, steps=2, save_every=0)
            with self.assertRaises(RuntimeError):
                mesh_train.iterate(_FakeRuntime(), _FakeData(0), plan)


if __name__ == "__main__":
    unittest.main()
