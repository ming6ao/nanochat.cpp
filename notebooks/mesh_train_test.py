"""T0: the pure helpers of the mesh launcher and the child launch.

The test covers the global batch mapping, the metrics and summary readers, the
driver config and its job round trip, and the child launch against a fake
``tools/nanochat`` entry. The config needs the shared library; the rest needs
nothing. See ``docs/client-runtime-design.md``.
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
import target
from nanochat_cpp import runtime as rt


class GlobalBatchTest(unittest.TestCase):
    def test_matches_the_multi_process_total(self) -> None:
        # One sequence per rank, eight micro-batches, two ranks: 65536 tokens
        # at seq 4096.
        self.assertEqual(rt.global_batch(1, 8, 2), 16)
        self.assertEqual(rt.global_batch(1, 8, 2) * 4096, 65536)

    def test_rejects_a_nonpositive_factor(self) -> None:
        for values in ((0, 1, 1), (1, 0, 1), (1, 1, 0)):
            with self.assertRaises(ValueError):
                rt.global_batch(*values)


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


def _model() -> nc.Config:
    return nc.Config(num_layers=2, num_heads=2, num_kv_heads=1, hidden_dim=32,
                     seq_len=8, vocab_size=64, window_pattern="SL",
                     value_embedding=False)


def _config(tmp: str, **overrides) -> rt.MeshTrainConfig:
    values = dict(
        tokenizer="tokenizer.nctoken",
        train_parquet="train.parquet",
        per_rank_batch=1,
        grad_accum=2,
        world_size=2,
        steps=4,
        save_every=2,
        out_dir=tmp,
        flops_per_token=2.0,
        peak_flops=1000.0,
    )
    values.update(overrides)
    return rt.MeshTrainConfig(_model(), **values)


class ConfigTest(unittest.TestCase):
    def test_derived_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            self.assertEqual(config.sequence_length, 8)
            self.assertEqual(config.global_sequences, 4)
            self.assertEqual(config.tokens_per_step, 32)
            self.assertEqual(config.metrics_path(),
                             Path(tmp) / "metrics.jsonl")
            self.assertEqual(config.checkpoint_path(),
                             Path(tmp) / "checkpoint.ckpt")

    def test_job_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            job = json.loads(config.job_json())
            self.assertEqual(job["data"]["tokenizer_path"],
                             "tokenizer.nctoken")
            self.assertEqual(job["run"]["batch"], 4)
            self.assertEqual(job["run"]["grad_accum"], 2)
            self.assertEqual(job["mesh"]["count"], 2)
            self.assertEqual(job["model"]["num_layers"], 2)

            rebuilt = rt.MeshTrainConfig.from_dict(job)
            try:
                self.assertEqual(rebuilt.tokens_per_step,
                                 config.tokens_per_step)
                self.assertEqual(rebuilt.out_dir, tmp)
            finally:
                rebuilt.close()


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
            config = _config(str(root / "out"))
            live = mesh_train.launch(config, root=root, entry=str(entry))
            code = live.wait(timeout=20)

            self.assertEqual(code, 0)
            self.assertIn("fake client", live.tail())
            self.assertTrue((Path(config.out_dir) / "worker.log").is_file())
            job = json.loads(
                (Path(config.out_dir) / "job.json").read_text(encoding="utf-8"))
            self.assertEqual(job["data"]["tokenizer_path"],
                             "tokenizer.nctoken")
            self.assertEqual(job["run"]["batch"], config.global_sequences)

    def test_stream_prints_the_child_log(self) -> None:
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = _fake_entry(root, _FAKE_ENTRY)
            config = _config(str(root / "out"))
            live = mesh_train.launch(config, root=root, entry=str(entry))
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                live.stream(interval=0.05)
            live.wait(timeout=20)
            self.assertIn("fake client", buffer.getvalue())

    def test_terminate_stops_a_sleeping_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = _fake_entry(root, _SLEEP_ENTRY)
            config = _config(str(root / "out"))
            live = mesh_train.launch(config, root=root, entry=str(entry))
            time.sleep(0.3)
            self.assertTrue(live.running())
            live.terminate(timeout=10)
            self.assertFalse(live.running())


class DispatchTest(unittest.TestCase):
    """``launch`` picks the run host and keeps the local path unchanged."""

    class Recording(target.Target):
        name = "recording"
        needs_local_devices = False

        def __init__(self) -> None:
            self.calls = []

        def run(self, config, **options):
            self.calls.append((config, options))
            return "remote-handle"

    def test_a_named_host_runs_remotely(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(str(Path(tmp) / "out"))
            host = self.Recording()
            result = mesh_train.launch(config, root=tmp, target=host,
                                       timeout=7)
        self.assertEqual(result, "remote-handle")
        self.assertEqual(host.calls[0][1], {"timeout": 7, "extra_env": None,
                                            "publish": None})

    def test_the_default_host_still_runs_a_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = _fake_entry(root, _FAKE_ENTRY)
            config = _config(str(root / "out"))
            live = mesh_train.launch(config, root=root, entry=str(entry),
                                     target="local")
            self.assertIsInstance(live, mesh_train.MeshLiveRun)
            self.assertEqual(live.wait(timeout=20), 0)

    def test_an_unknown_host_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(str(Path(tmp) / "out"))
            with self.assertRaises(ValueError):
                mesh_train.launch(config, root=tmp, target="slurm")


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
