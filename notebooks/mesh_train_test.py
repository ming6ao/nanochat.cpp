"""T0: the pure helpers of the mesh training loop and the child launch.

The test covers the batch mapping, the model FLOPs utilization, the summary
statistics, the loop against a fake runtime, the plan job round trip, and the
child launch against a fake ``tools/nanochat`` entry. It needs no shared
library, no GPU, and no broker. See ``docs/notebook-workflow.md``.
"""

from __future__ import annotations

import json
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path

import mesh_train
import mesh_worker
import nanochat_cpp as nc


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


class LogTest(unittest.TestCase):
    def test_format_row_reports_every_field(self) -> None:
        row = {"step": 3, "loss": 4.5, "grad_norm": 0.25, "step_seconds": 22.4,
               "compute_seconds": 21.9, "tokens_per_sec": 2926.0,
               "mfu": 0.3012}
        text = mesh_train.format_row(row)
        self.assertIn("loss 4.5000", text)
        self.assertIn("grad_norm 0.25", text)
        self.assertIn("step 22.40s", text)
        self.assertIn("compute 21.90s", text)
        self.assertIn("2,926 tok/s", text)
        self.assertIn("mfu 30.12%", text)

    def test_log_progress_writes_one_line(self) -> None:
        import contextlib
        import io

        row = {"step": 1, "loss": 1.0, "grad_norm": 1.0, "step_seconds": 1.0,
               "compute_seconds": 0.9, "tokens_per_sec": 10.0, "mfu": 0.1}
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            mesh_train.log_progress(row, 5)
        self.assertEqual(buffer.getvalue().count("\n"), 1)
        self.assertIn("[    1/5]", buffer.getvalue())


class JobTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        plan = _plan("/tmp/mesh-out")
        job = plan.job()
        self.assertEqual(job["tokenizer_path"], plan.tokenizer)
        self.assertEqual(job["config"]["num_layers"], plan.config.num_layers)
        self.assertEqual(job["config"]["window_pattern"],
                         plan.config.window_pattern)

        rebuilt = mesh_train.MeshTrainPlan.from_job(job)
        self.assertEqual(rebuilt.train_parquet, plan.train_parquet)
        self.assertEqual(rebuilt.tokenizer, plan.tokenizer)
        self.assertEqual(rebuilt.global_sequences, plan.global_sequences)
        self.assertEqual(rebuilt.sequence_length, plan.sequence_length)
        self.assertEqual(rebuilt.config.value_embedding,
                         plan.config.value_embedding)
        self.assertEqual(rebuilt.preset, plan.preset)
        self.assertEqual(rebuilt.resume, plan.resume)
        self.assertEqual(rebuilt.flops_per_token, plan.flops_per_token)


class _Metrics:
    def __init__(self, loss, grad_norm=1.0, step_seconds=0.05):
        self.loss = tuple(loss)
        self.grad_norm = grad_norm
        self.step_seconds = step_seconds
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


def _config() -> nc.Config:
    return nc.Config(num_layers=2, num_heads=2, num_kv_heads=1, hidden_dim=32,
                     seq_len=8, vocab_size=64, window_pattern="SL",
                     value_embedding=False)


def _plan(tmp: str, **overrides) -> mesh_train.MeshTrainPlan:
    values = dict(
        train_parquet="train.parquet",
        tokenizer="tokenizer.nctoken",
        out_dir=tmp,
        config=_config(),
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
            self.assertAlmostEqual(records[0]["compute_seconds"], 0.05)
            # The save cadence fires on steps 2 and 4.
            self.assertEqual(len(runtime.saved), 2)

            written = mesh_train.read_metrics(plan.metrics_path())
            self.assertEqual(len(written), plan.steps)
            self.assertEqual(written[-1]["step"], plan.steps)

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

    def test_resume_appends_and_starts_later(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # The first run reaches step 2. A resume of a longer plan starts
            # at step 3 and appends to the same metrics file.
            mesh_train.iterate(_FakeRuntime(), _FakeData(1),
                               _plan(tmp, steps=2, save_every=0),
                               clock=_fake_clock())
            runtime = _FakeRuntime()
            mesh_train.iterate(runtime, _FakeData(1),
                               _plan(tmp, steps=4, save_every=0),
                               start_step=3, clock=_fake_clock())
            written = mesh_train.read_metrics(
                Path(tmp) / "metrics.jsonl")
            self.assertEqual([row["step"] for row in written], [1, 2, 3, 4])
            self.assertEqual(runtime.steps, 2)

    def test_rejects_an_empty_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _plan(tmp, steps=2, save_every=0)
            with self.assertRaises(RuntimeError):
                mesh_train.iterate(_FakeRuntime(), _FakeData(0), plan)


_FAKE_ENTRY = """\
import sys
import time

print("fake client", " ".join(sys.argv[1:]), flush=True)
time.sleep(0.3)
print("fake done", flush=True)
"""

_SLEEP_ENTRY = """\
import time

print("sleeping", flush=True)
time.sleep(60)
"""


def _fake_entry(root: Path, body: str) -> Path:
    entry = root / "fake-nanochat"
    entry.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    entry.chmod(entry.stat().st_mode | stat.S_IEXEC)
    return entry


class LaunchTest(unittest.TestCase):
    def test_launch_runs_a_child_and_writes_a_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = _fake_entry(root, _FAKE_ENTRY)
            plan = _plan(str(root / "out"))
            live = mesh_train.launch(plan, root=root, entry=str(entry))
            code = live.wait(timeout=20)

            self.assertEqual(code, 0)
            self.assertIn("fake client", live.tail())
            self.assertTrue((Path(plan.out_dir) / "worker.log").is_file())
            job = json.loads(
                (Path(plan.out_dir) / "job.json").read_text(encoding="utf-8"))
            self.assertEqual(job["tokenizer_path"], plan.tokenizer)
            self.assertEqual(job["per_rank_batch"], plan.per_rank_batch)

    def test_terminate_stops_a_sleeping_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = _fake_entry(root, _SLEEP_ENTRY)
            plan = _plan(str(root / "out"))
            live = mesh_train.launch(plan, root=root, entry=str(entry))
            time.sleep(0.3)
            self.assertTrue(live.running())
            live.terminate(timeout=10)
            self.assertFalse(live.running())


class WorkerTest(unittest.TestCase):
    def test_usage_error(self) -> None:
        self.assertEqual(mesh_worker.main([]), 2)

    def test_missing_job_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                mesh_worker.main([str(Path(tmp) / "missing.json")])


if __name__ == "__main__":
    unittest.main()
