"""T0: the pure helpers and the Hub call sequence of ``hub_checkpoint.py``.

The test injects a fake ``api`` and a fake ``download``, so it needs no
network and no ``huggingface_hub`` wheel. See ``docs/checkpoint-format.md``
section 22.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import hub_checkpoint


class FakeApi:
    """A stand-in for ``HfApi`` that records every call in order."""

    def __init__(self, name="alice", whoami=None):
        self.name = name
        self.whoami_value = whoami
        self.calls = []

    def whoami(self):
        self.calls.append(("whoami", {}))
        if self.whoami_value is not None:
            return self.whoami_value
        return {"name": self.name}

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))

    def upload_file(self, **kwargs):
        self.calls.append(("upload_file", kwargs))

    def create_tag(self, **kwargs):
        self.calls.append(("create_tag", kwargs))


class FakeDownload:
    """A stand-in for ``snapshot_download``.

    It writes one file for each ``allow_patterns`` entry, unless the pattern is
    in ``missing`` or ``raise_error`` is set.
    """

    def __init__(self, missing=(), raise_error=False):
        self.missing = set(missing)
        self.raise_error = raise_error
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_error:
            raise RuntimeError("not found")
        dest = Path(kwargs["local_dir"])
        for pattern in kwargs.get("allow_patterns") or []:
            if pattern in self.missing:
                continue
            path = dest / pattern
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
        return str(dest)


class TokenizerFileTest(unittest.TestCase):
    def test_name(self) -> None:
        self.assertEqual(hub_checkpoint.tokenizer_file(32768),
                         "vocab-32768.nctoken")
        self.assertEqual(hub_checkpoint.tokenizer_file("4096"),
                         "vocab-4096.nctoken")


class RepoIdTest(unittest.TestCase):
    def test_explicit_namespace(self) -> None:
        self.assertEqual(
            hub_checkpoint.repo_id_for("base-L16-H1024", namespace="alice"),
            "alice/nanochat-base-L16-H1024")

    def test_environment_override(self) -> None:
        self.assertEqual(
            hub_checkpoint.repo_id_for("base", environ={"HF_USERNAME": "bob"}),
            "bob/nanochat-base")

    def test_whoami(self) -> None:
        api = FakeApi("carol")
        self.assertEqual(
            hub_checkpoint.repo_id_for("base", api=api, environ={}),
            "carol/nanochat-base")
        self.assertEqual(api.calls, [("whoami", {})])

    def test_missing_token(self) -> None:
        api = FakeApi(whoami={})
        with self.assertRaises(RuntimeError) as caught:
            hub_checkpoint.repo_id_for("base", api=api, environ={})
        self.assertIn("HF_USERNAME", str(caught.exception))


class StageDirTest(unittest.TestCase):
    def test_path(self) -> None:
        self.assertEqual(hub_checkpoint.stage_dir("/root", "base"),
                         Path("/root/base"))
        self.assertEqual(hub_checkpoint.stage_dir(Path("/root"), "sft"),
                         Path("/root/sft"))


class BuildRunRecordTest(unittest.TestCase):
    def test_base_record(self) -> None:
        record = hub_checkpoint.build_run_record(
            stage="base", run_name="base-L16-H1024", git_revision="b59ca93",
            build_key="key", cuda_arch="sm_75", step=500,
            created_utc="2026-01-01T00:00:00Z")
        self.assertEqual(record["stage"], "base")
        self.assertEqual(record["container"], "NCHKPT01")
        self.assertEqual(record["step"], 500)
        self.assertEqual(record["parent_repo_id"], "")
        self.assertEqual(record["parent_revision"], "")

    def test_sft_record_names_the_parent(self) -> None:
        record = hub_checkpoint.build_run_record(
            stage="sft", run_name="sft-L16-H1024", step=40,
            parent_repo_id="alice/nanochat-base-L16-H1024",
            parent_revision="base-500")
        self.assertEqual(record["stage"], "sft")
        self.assertEqual(record["parent_repo_id"],
                         "alice/nanochat-base-L16-H1024")
        self.assertEqual(record["parent_revision"], "base-500")

    def test_defaults_are_typed(self) -> None:
        record = hub_checkpoint.build_run_record(stage="base", run_name="run")
        self.assertEqual(record["step"], 0)
        self.assertEqual(record["git_revision"], "")


class PushStageTest(unittest.TestCase):
    def test_call_sequence_and_tag(self) -> None:
        api = FakeApi()
        result = hub_checkpoint.push_stage(
            "alice/nanochat-base", "base", "/local/base", token="t",
            tag="base-500", api=api)
        self.assertEqual(result, "base")
        self.assertEqual([name for name, _ in api.calls],
                         ["create_repo", "upload_folder", "create_tag"])
        self.assertEqual(api.calls[1][1]["path_in_repo"], "base")
        self.assertEqual(api.calls[1][1]["folder_path"], "/local/base")
        self.assertEqual(api.calls[2][1]["tag"], "base-500")

    def test_without_tag(self) -> None:
        api = FakeApi()
        hub_checkpoint.push_stage("alice/nanochat-base", "sft", "/local/sft",
                                  api=api)
        self.assertEqual([name for name, _ in api.calls],
                         ["create_repo", "upload_folder"])


class PushFileTest(unittest.TestCase):
    def test_report(self) -> None:
        api = FakeApi()
        path = hub_checkpoint.push_file(
            "alice/nanochat-base", "reports/base-base-500.json",
            "/local/report.json", token="t", api=api)
        self.assertEqual(path, "reports/base-base-500.json")
        self.assertEqual(api.calls[0][0], "upload_file")
        self.assertEqual(api.calls[0][1]["path_in_repo"],
                         "reports/base-base-500.json")


class PushTokenizerTest(unittest.TestCase):
    def test_flat_name(self) -> None:
        api = FakeApi()
        name = hub_checkpoint.push_tokenizer(
            "alice/nanochat-tokenizer", 32768, "/local/tok", api=api)
        self.assertEqual(name, "vocab-32768.nctoken")
        self.assertEqual([n for n, _ in api.calls],
                         ["create_repo", "upload_file"])
        self.assertEqual(api.calls[1][1]["path_in_repo"],
                         "vocab-32768.nctoken")


class PullStageTest(unittest.TestCase):
    def test_pattern_list_and_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download = FakeDownload()
            folder = hub_checkpoint.pull_stage(
                "alice/nanochat-base", "base", tmp, download=download)
            self.assertEqual(folder, Path(tmp) / "base")
            self.assertEqual(
                download.calls[0]["allow_patterns"],
                ["base/checkpoint.ckpt", "base/model.json",
                 "base/tokenizer.nctoken"])
            self.assertTrue((folder / "checkpoint.ckpt").is_file())
            self.assertTrue((folder / "model.json").is_file())
            self.assertTrue((folder / "tokenizer.nctoken").is_file())

    def test_stage_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download = FakeDownload()
            folder = hub_checkpoint.pull_stage(
                "alice/nanochat-base", "sft", tmp,
                patterns=hub_checkpoint.FULL_STAGE_PATTERNS, download=download)
            self.assertEqual(folder, Path(tmp) / "sft")
            self.assertIn("sft/metrics.jsonl",
                          download.calls[0]["allow_patterns"])


class PullTokenizerTest(unittest.TestCase):
    def test_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download = FakeDownload()
            path = hub_checkpoint.pull_tokenizer(
                "alice/nanochat-tokenizer", 32768, tmp, download=download)
            self.assertEqual(path, Path(tmp) / "vocab-32768.nctoken")
            self.assertTrue(path.is_file())
            self.assertEqual(download.calls[0]["allow_patterns"],
                             ["vocab-32768.nctoken"])

    def test_missing_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download = FakeDownload(raise_error=True)
            self.assertIsNone(hub_checkpoint.pull_tokenizer(
                "alice/nanochat-tokenizer", 32768, tmp, download=download))


if __name__ == "__main__":
    unittest.main()
