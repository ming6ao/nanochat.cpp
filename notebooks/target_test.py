"""T0: the run-host seam of ``target.py``.

The test covers the job translation, the plan, the registry, the two roots, the
record parsers, the Modal upload and spawn, and the live handle. The Modal SDK
is a fake and the Volume is a file table, so the test needs no SDK, no account,
no card, and no network. See ``docs/host-portability.md``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import mesh_train
import target


class FakeConfig:
    """A config the seam reads: one job document and one run directory."""

    def __init__(self, out_dir, job):
        self.out_dir = str(out_dir)
        self._job = job

    def job_json(self) -> str:
        return json.dumps(self._job, sort_keys=True)


def _job(stage="/stage") -> dict:
    return {
        "warm_start_path": "",
        "data": {
            "parquet": [f"{stage}/data/train.parquet",
                        f"{stage}/data/val.parquet"],
            "packed_rows_path": "",
            "tokenizer_path": f"{stage}/data/tokenizer.nctoken",
            "data_seed": 42,
        },
        "output": {
            "out_dir": f"{stage}/runs/base",
            "checkpoint_path": "",
        },
    }


class FakeSDK:
    """The two Modal calls the target makes, and the Volume file table."""

    def __init__(self, files=None) -> None:
        self.files = dict(files or {})
        self.uploads = []
        self.spawned = []
        self.cancelled = False
        self.Volume = types.SimpleNamespace(from_name=self._volume)
        self.Function = types.SimpleNamespace(from_name=self._function)

    def _volume(self, *_args, **_kwargs):
        return types.SimpleNamespace(batch_upload=self._batch,
                                     read_file=self._read)

    def _function(self, *_args, **_kwargs):
        return types.SimpleNamespace(spawn=self._spawn)

    def _batch(self):
        sdk = self

        class Batch:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def put_file(self, local, remote):
                sdk.uploads.append((local, remote))

        return Batch()

    def _read(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        data = self.files[path]

        class Handle:
            def read(self):
                return data

        return Handle()

    def _spawn(self, *args):
        self.spawned.append(args)
        sdk = self

        class Call:
            def __init__(self):
                self.finished = False
                self.cancelled = False

            def get(self, timeout=None):
                if not self.finished and timeout == 0:
                    raise TimeoutError("running")
                self.finished = True
                return "done"

            def cancel(self):
                self.cancelled = True
                sdk.cancelled = True

        return Call()


class MapJobTest(unittest.TestCase):
    def test_without_roots_every_path_stays(self) -> None:
        job = _job()
        self.assertEqual(target.map_job(job), job)

    def test_with_roots_every_path_moves(self) -> None:
        mapped = target.map_job(_job(), Path("/stage"), Path("/runs"))
        self.assertEqual(mapped["output"]["out_dir"], "/runs/runs/base")
        self.assertEqual(mapped["data"]["tokenizer_path"],
                         "/runs/data/tokenizer.nctoken")
        self.assertEqual(mapped["data"]["parquet"],
                         ["/runs/data/train.parquet",
                          "/runs/data/val.parquet"])
        self.assertEqual(mapped["data"]["data_seed"], 42)

    def test_the_input_is_not_modified(self) -> None:
        job = _job()
        target.map_job(job, Path("/stage"), Path("/runs"))
        self.assertEqual(job, _job())

    def test_a_path_outside_the_root_raises(self) -> None:
        with self.assertRaises(ValueError):
            target.map_job(_job("/kaggle/input"), Path("/stage"),
                           Path("/runs"))

    def test_an_empty_value_stays_empty(self) -> None:
        mapped = target.map_job(_job(), Path("/stage"), Path("/runs"))
        self.assertEqual(mapped["warm_start_path"], "")


class JobPathsTest(unittest.TestCase):
    def test_inputs_are_flattened(self) -> None:
        self.assertEqual(target.job_paths(_job(), target.INPUT_FIELDS), [
            "/stage/data/train.parquet",
            "/stage/data/val.parquet",
            "/stage/data/tokenizer.nctoken",
        ])

    def test_outputs_are_separate(self) -> None:
        self.assertEqual(target.job_paths(_job(), target.OUTPUT_FIELDS),
                         ["/stage/runs/base"])

    def test_the_groups_cover_every_field(self) -> None:
        self.assertEqual(target.PATH_FIELDS,
                         target.INPUT_FIELDS + target.OUTPUT_FIELDS)


class ParseTest(unittest.TestCase):
    def test_metrics_skips_a_bad_line(self) -> None:
        text = '{"step": 1}\nnot json\n\n{"step": 2}\n'
        self.assertEqual([row["step"] for row in target.parse_metrics(text)],
                         [1, 2])

    def test_summary_tolerates_empty_and_bad_text(self) -> None:
        self.assertEqual(target.parse_summary('{"steps": 4}'), {"steps": 4})
        self.assertEqual(target.parse_summary(""), {})
        self.assertEqual(target.parse_summary("{not json"), {})

    def test_the_path_readers_use_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "metrics.jsonl").write_text('{"step": 1}\n',
                                               encoding="utf-8")
            (run / "summary.json").write_text('{"steps": 1}',
                                              encoding="utf-8")
            self.assertEqual(mesh_train.read_metrics(run / "metrics.jsonl"),
                             [{"step": 1}])
            self.assertEqual(mesh_train.read_summary(run), {"steps": 1})
            self.assertEqual(mesh_train.read_metrics(run / "none.jsonl"), [])
            self.assertEqual(mesh_train.read_summary(run / "none"), {})


class RootsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.saved = {}
        for name in ("NANOCHAT_BASE_DIR", "NANOCHAT_RUNS_DIR"):
            self.saved[name] = os.environ.pop(name, None)

    def tearDown(self) -> None:
        for name, value in self.saved.items():
            os.environ.pop(name, None)
            if value is not None:
                os.environ[name] = value

    def test_the_staging_root(self) -> None:
        self.assertEqual(target.LocalTarget().cache_root(),
                         target.NOTEBOOK_BASE)
        os.environ["NANOCHAT_BASE_DIR"] = "/content/cache"
        self.assertEqual(target.LocalTarget().cache_root(),
                         Path("/content/cache"))

    def test_the_run_root_is_beside_the_staging_root(self) -> None:
        os.environ["NANOCHAT_BASE_DIR"] = "/kaggle/working/cache"
        self.assertEqual(target.LocalTarget().run_root(),
                         Path("/kaggle/working"))

    def test_the_environment_names_the_run_root(self) -> None:
        os.environ["NANOCHAT_RUNS_DIR"] = "/content/runs"
        self.assertEqual(target.LocalTarget().run_root(),
                         Path("/content/runs"))


class PlanTest(unittest.TestCase):
    def test_a_local_plan_keeps_every_path(self) -> None:
        config = FakeConfig("/stage/runs/base", _job())
        plan = target.LocalTarget().prepare(config)
        self.assertEqual(plan.job_json, config.job_json())
        self.assertEqual(plan.out_dir, "/stage/runs/base")
        self.assertEqual(plan.job_path, Path("/stage/runs/base/job.json"))

    def test_a_modal_plan_moves_them(self) -> None:
        config = FakeConfig("/stage/runs/base", _job())
        plan = target.ModalTarget().prepare(config, Path("/stage"))
        self.assertEqual(plan.out_dir, "/runs/runs/base")
        self.assertEqual(plan.job_path, Path("/stage/runs/base/job.json"))
        self.assertEqual(json.loads(plan.job_json)["output"]["out_dir"],
                         "/runs/runs/base")

    def test_a_modal_plan_needs_an_out_dir(self) -> None:
        job = _job()
        job["output"]["out_dir"] = ""
        with self.assertRaises(RuntimeError):
            target.ModalTarget().prepare(FakeConfig("/stage", job),
                                         Path("/stage"))


class UploadsTest(unittest.TestCase):
    def test_only_the_input_fields_travel(self) -> None:
        config = FakeConfig("/stage/runs/base", _job())
        pairs = target.ModalTarget().uploads(config, Path("/stage"))
        self.assertEqual(pairs, [
            (Path("/stage/data/train.parquet"), "/runs/data/train.parquet"),
            (Path("/stage/data/val.parquet"), "/runs/data/val.parquet"),
            (Path("/stage/data/tokenizer.nctoken"),
             "/runs/data/tokenizer.nctoken"),
        ])


class SpawnTest(unittest.TestCase):
    def test_it_uploads_then_spawns(self) -> None:
        sdk = FakeSDK()
        host = target.ModalTarget()
        config = FakeConfig("/stage/runs/base", _job())
        plan = host.prepare(config, Path("/stage"))
        with mock.patch.object(target, "client", lambda: sdk):
            live = host.submit(plan, host.uploads(config, Path("/stage")))
        self.assertEqual([remote for _local, remote in sdk.uploads],
                         ["/runs/data/train.parquet",
                          "/runs/data/val.parquet",
                          "/runs/data/tokenizer.nctoken"])
        self.assertEqual(sdk.spawned, [(plan.job_json, "runs/base", "")])
        self.assertIsInstance(live, target.ModalLiveRun)

    def test_a_publish_travels_with_the_job(self) -> None:
        sdk = FakeSDK()
        host = target.ModalTarget()
        config = FakeConfig("/stage/runs/base", _job())
        plan = host.prepare(config, Path("/stage"))
        publish = {"stage": "base", "repo_id": "owner/model"}
        with mock.patch.object(target, "client", lambda: sdk):
            host.submit(plan, [], publish=publish)
        self.assertEqual(json.loads(sdk.spawned[0][2]), publish)

    def test_a_run_writes_the_job_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sdk = FakeSDK()
            config = FakeConfig(Path(tmp) / "runs" / "base", _job(tmp))
            with mock.patch.object(target, "client", lambda: sdk):
                target.ModalTarget().run(config, stage_root=Path(tmp))
            job = json.loads((Path(tmp) / "runs" / "base"
                              / "job.json").read_text(encoding="utf-8"))
        self.assertEqual(job["output"]["out_dir"], "/runs/runs/base")


class LiveRunTest(unittest.TestCase):
    def make(self, files=None):
        sdk = FakeSDK(files)
        host = target.ModalTarget()
        call = sdk._spawn()
        return target.ModalLiveRun(call, remote_dir="runs/base",
                                   reader=host.reader(sdk._volume()))

    def test_rows_and_summary_read_the_volume(self) -> None:
        live = self.make({
            "runs/base/metrics.jsonl": b'{"step": 1}\n{"step": 2}\n',
            "runs/base/summary.json": b'{"steps": 2}',
        })
        self.assertEqual([row["step"] for row in live.rows()], [1, 2])
        self.assertEqual(live.summary(), {"steps": 2})

    def test_an_absent_file_is_empty(self) -> None:
        live = self.make()
        self.assertEqual(live.rows(), [])
        self.assertEqual(live.summary(), {})
        self.assertEqual(live.tail(), "")

    def test_tail_returns_the_last_lines(self) -> None:
        text = "\n".join(f"line {index}" for index in range(30))
        live = self.make({"runs/base/worker.log": text.encode()})
        self.assertEqual(live.tail(2), "line 28\nline 29")

    def test_pull_writes_the_files_that_exist(self) -> None:
        live = self.make({
            "runs/base/checkpoint.ckpt": b"NCHKPT01 binary",
            "runs/base/summary.json": b'{"steps": 2}',
        })
        with tempfile.TemporaryDirectory() as tmp:
            written = live.pull(Path(tmp) / "base")
            names = sorted(path.name for path in written)
            self.assertEqual(names, ["checkpoint.ckpt", "summary.json"])
            self.assertEqual((Path(tmp) / "base" / "checkpoint.ckpt")
                             .read_bytes(), b"NCHKPT01 binary")

    def test_running_wait_and_cancel(self) -> None:
        live = self.make()
        self.assertTrue(live.running())
        self.assertEqual(live.wait(timeout=5), 0)
        self.assertFalse(live.running())
        live.terminate()
        self.assertTrue(live.call.cancelled)


class ClientTest(unittest.TestCase):
    def test_a_missing_sdk_explains_the_install(self) -> None:
        with mock.patch.dict(sys.modules, {"modal": None}):
            with self.assertRaises(RuntimeError) as caught:
                target.client()
        self.assertIn("pip install modal", str(caught.exception))


class ResolveTest(unittest.TestCase):
    def test_the_default_is_local(self) -> None:
        self.assertIsInstance(target.resolve(environ={}), target.LocalTarget)

    def test_the_environment_names_the_host(self) -> None:
        resolved = target.resolve(environ={"NANOCHAT_TARGET": "modal"})
        self.assertIsInstance(resolved, target.ModalTarget)
        self.assertFalse(resolved.needs_local_devices)

    def test_an_explicit_name_beats_the_environment(self) -> None:
        resolved = target.resolve("local",
                                  environ={"NANOCHAT_TARGET": "modal"})
        self.assertEqual(resolved.name, "local")

    def test_an_instance_passes_through(self) -> None:
        instance = target.ModalTarget()
        self.assertIs(target.resolve(instance), instance)

    def test_both_hosts_are_named(self) -> None:
        self.assertEqual(target.names(), ("local", "modal"))

    def test_an_unknown_name_raises(self) -> None:
        with self.assertRaises(ValueError) as caught:
            target.resolve("slurm", environ={})
        self.assertIn("unknown run target", str(caught.exception))


class LaunchTest(unittest.TestCase):
    def test_a_remote_host_gets_the_job(self) -> None:
        calls = []

        class Host(target.Target):
            name = "recording"
            needs_local_devices = False

            def run(self, config, **options):
                calls.append(options)
                return "remote-handle"

        config = FakeConfig("/stage/runs/base", _job())
        result = mesh_train.launch(config, root="/repo", target=Host(),
                                   timeout=7)
        self.assertEqual(result, "remote-handle")
        self.assertEqual(calls, [{"timeout": 7, "extra_env": None,
                                  "publish": None}])

    def test_the_local_host_rejects_a_publish(self) -> None:
        config = FakeConfig("/stage/runs/base", _job())
        with self.assertRaises(ValueError) as caught:
            mesh_train.launch(config, target="local", publish={"stage": "b"})
        self.assertIn("hub_checkpoint", str(caught.exception))

    def test_the_local_host_runs_a_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            entry = Path(tmp) / "fake-nanochat"
            entry.write_text(
                f"#!{sys.executable}\nprint('fake', flush=True)\n",
                encoding="utf-8")
            entry.chmod(0o755)
            config = FakeConfig(Path(tmp) / "out", _job())
            live = mesh_train.launch(config, root=tmp, entry=str(entry))
            self.assertIsInstance(live, mesh_train.MeshLiveRun)
            self.assertEqual(live.wait(timeout=20), 0)

    def test_an_unknown_host_raises(self) -> None:
        with self.assertRaises(ValueError):
            mesh_train.launch(FakeConfig("/stage/runs/base", _job()),
                              target="slurm")


if __name__ == "__main__":
    unittest.main()
