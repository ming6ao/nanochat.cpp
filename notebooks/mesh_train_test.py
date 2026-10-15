"""T0: the pure helpers of the mesh launcher and the child launch.

The test covers the batch mapping, the metrics and summary readers, the plan
job JSON through the C++ serializer, and the child launch against a fake
``tools/nanochat`` entry. The serializer and the plan config need the shared
library; the rest needs nothing. See ``docs/client-runtime-design.md``.
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
from nanochat_cpp import runtime as rt


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


class ReaderTest(unittest.TestCase):
    def test_read_metrics_missing_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(mesh_train.read_metrics(Path(tmp) / "none.jsonl"),
                             [])

    def test_read_metrics_skips_a_bad_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            path.write_text('{"step": 1}\nnot json\n{"step": 2}\n',
                            encoding="utf-8")
            rows = mesh_train.read_metrics(path)
            self.assertEqual([row["step"] for row in rows], [1, 2])

    def test_read_summary_missing_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(mesh_train.read_summary(tmp), {})


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


class PlanTest(unittest.TestCase):
    def test_derived_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _plan(tmp)
            self.assertEqual(plan.sequence_length, 8)
            self.assertEqual(plan.global_sequences, 4)
            self.assertEqual(plan.tokens_per_step, 32)

    def test_job_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _plan(tmp)
            job = json.loads(plan.job_json())
            self.assertEqual(job["data"]["tokenizer_path"], plan.tokenizer)
            self.assertEqual(job["run"]["batch"], plan.global_sequences)
            self.assertEqual(job["run"]["grad_accum"], plan.grad_accum)
            self.assertEqual(job["mesh"]["count"], 2)
            self.assertEqual(job["model"]["num_layers"],
                             plan.config.num_layers)

            rebuilt = rt.MeshTrainConfig.from_dict(
                json.loads(plan.job_json()))
            try:
                again = json.loads(rebuilt.job_json())
            finally:
                rebuilt.close()
            self.assertEqual(again["run"]["batch"], job["run"]["batch"])
            self.assertEqual(again["run"]["seed"], job["run"]["seed"])
            self.assertEqual(again["model"]["window_pattern"],
                             job["model"]["window_pattern"])


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
            self.assertEqual(job["data"]["tokenizer_path"], plan.tokenizer)
            self.assertEqual(job["run"]["batch"], plan.global_sequences)

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

    def test_bad_job_returns_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            self.assertEqual(mesh_worker.main([str(path)]), 2)


if __name__ == "__main__":
    unittest.main()
