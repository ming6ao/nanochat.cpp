"""T0: the contract of the Modal app in ``tools/modal/app.py``.

The test imports the app with a stub Modal SDK, so it needs no SDK, no account,
and no network. It gates the three things a client cannot see: the card and the
architecture are one pair, the names match the client, and the image installs
the package while the weights Volume holds the stages
(``docs/dependency-layering-plan.md`` section 6.4 and D8).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

_HERE = Path(__file__).resolve().parent


class _Chain:
    """An SDK object that accepts any call, attribute, or subscript."""

    def __call__(self, *_args, **_kwargs) -> "_Chain":
        return self

    def __getattr__(self, _name) -> "_Chain":
        return self

    def __getitem__(self, _key) -> "_Chain":
        return self


def _decorator(*_args, **_kwargs):
    """The stub of ``app.function``: it returns the wrapped function."""

    def apply(func):
        return func

    return apply


def _stub_sdk() -> types.ModuleType:
    """A Modal SDK whose images, volumes, and secrets accept any call.

    The stub records the Volume names and the ``app.function`` keywords, so a
    test reads which volumes the app declares and which one each function
    mounts.
    """
    module = types.ModuleType("modal")
    module.volume_names = []
    module.function_calls = []

    def from_name(name, create_if_missing=False, **_kwargs):
        module.volume_names.append((str(name), bool(create_if_missing)))
        return _Chain()

    def function(**kwargs):
        module.function_calls.append(kwargs)
        return _decorator()

    module.Image = _Chain()
    module.Volume = types.SimpleNamespace(from_name=from_name)
    module.Secret = _Chain()
    module.App = lambda *_args, **_kwargs: types.SimpleNamespace(
        function=function)
    return module


def _load_app(with_sdk: bool = False):
    """Import ``app.py`` with the stub SDK on ``sys.modules``.

    With ``with_sdk`` the return value is ``(module, stub)``, so a test can
    read what the module asked the SDK for.
    """
    path = _HERE / "app.py"
    spec = importlib.util.spec_from_file_location("modal_app", path)
    module = importlib.util.module_from_spec(spec)
    sdk = _stub_sdk()
    with mock.patch.dict(sys.modules, {"modal": sdk}):
        spec.loader.exec_module(module)
    return (module, sdk) if with_sdk else module


class AppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _load_app()

    def test_the_entry_points_exist(self) -> None:
        for name in ("doctor", "train"):
            self.assertTrue(callable(getattr(self.app, name)), name)

    def test_the_type_and_the_architecture_are_one_pair(self) -> None:
        self.assertEqual(self.app.GPU_TYPE, "T4")
        self.assertEqual(self.app.CUDA_ARCH, "sm_75")
        self.assertEqual(self.app._capability(self.app.CUDA_ARCH), "7.5")

    def test_the_gpu_request_names_the_count(self) -> None:
        self.assertEqual(self.app.gpu_request(), "T4")
        self.assertEqual(self.app.gpu_request("A100-80GB", 4), "A100-80GB:4")
        self.assertEqual(self.app.gpu_request("T4", 1), "T4")
        self.assertEqual(self.app.GPU_COUNT, 1)

    def test_the_contract_names_match_the_client(self) -> None:
        import target

        self.assertEqual(self.app.APP_NAME, target.APP_NAME)
        self.assertEqual(self.app.RUNS_VOLUME, target.RUNS_VOLUME)
        self.assertEqual(self.app.REMOTE_ROOT, target.REMOTE_ROOT)
        self.assertEqual(self.app.SECRET_NAME, target.SECRET_NAME)

    def test_the_image_carries_the_library_path(self) -> None:
        self.assertEqual(self.app.LIBRARY,
                         f"{self.app.REPO_ROOT}/bazel-bin/bindings/"
                         "libnanochat_shared.so")


class CheckCardsTest(unittest.TestCase):
    """The card guard: the function, the container, and the job must agree."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _load_app()

    def setUp(self) -> None:
        self.app._device_capability = lambda: "7.5"
        self.app._device_count = lambda: 1

    def job(self, replicas: int) -> str:
        return json.dumps({"mesh": {"count": replicas}})

    def test_one_replica_on_one_card_passes(self) -> None:
        self.app._check_cards(self.job(1))

    def test_a_wrong_architecture_stops(self) -> None:
        self.app._device_capability = lambda: "8.0"
        with self.assertRaises(RuntimeError) as caught:
            self.app._check_cards(self.job(1))
        self.assertIn("deploy an image", str(caught.exception))

    def test_a_count_that_disagrees_with_the_function_stops(self) -> None:
        self.app._device_count = lambda: 4
        with self.assertRaises(RuntimeError) as caught:
            self.app._check_cards(self.job(4))
        self.assertIn("set GPU_COUNT", str(caught.exception))

    def test_a_job_that_disagrees_with_the_count_stops(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            self.app._check_cards(self.job(4))
        self.assertIn("set world_size", str(caught.exception))


class PublishTest(unittest.TestCase):
    """A publish runs in the container, beside the checkpoint."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _load_app()

    def setUp(self) -> None:
        self.calls = []
        fake = types.ModuleType("hub_checkpoint")
        fake.now_utc = lambda: "2026-01-01T00:00:00Z"
        fake.write_stage = lambda folder, **kwargs: (
            self.calls.append(("write_stage", str(folder), kwargs))
            or {"stage": kwargs["stage"]})
        fake.push_stage = lambda *args, **kwargs: (
            self.calls.append(("push_stage",) + args) or "base")
        self.saved = sys.modules.get("hub_checkpoint")
        sys.modules["hub_checkpoint"] = fake
        self.saved_token = os.environ.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = "token"

    def tearDown(self) -> None:
        if self.saved is None:
            sys.modules.pop("hub_checkpoint", None)
        else:
            sys.modules["hub_checkpoint"] = self.saved
        os.environ.pop("HF_TOKEN", None)
        if self.saved_token is not None:
            os.environ["HF_TOKEN"] = self.saved_token

    def meta(self) -> str:
        return json.dumps({
            "stage": "base", "run_name": "base-1", "repo_id": "o/m",
            "tag": "base-500", "revision": "abc123", "step": 500,
            "description": "base model", "model_json": "{}",
            "train_json": "{}",
        })

    def job(self) -> str:
        return json.dumps({"data": {"tokenizer_path": "/runs/tok.nctoken"}})

    def test_it_writes_the_stage_beside_the_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.app._publish(self.job(), Path(tmp), self.meta())
        kind, folder, kwargs = self.calls[0]
        self.assertEqual((kind, folder), ("write_stage", tmp))
        self.assertEqual(kwargs["tokenizer"], "/runs/tok.nctoken")
        self.assertEqual(kwargs["git_revision"], "abc123")
        self.assertEqual(kwargs["cuda_arch"], self.app.CUDA_ARCH)
        self.assertEqual(kwargs["step"], 500)
        self.assertEqual(result["stage"], "base")

    def test_it_pushes_the_stage_and_tags_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.app._publish(self.job(), Path(tmp), self.meta())
        self.assertEqual(self.calls[1],
                         ("push_stage", "o/m", "base", Path(tmp)))

    def test_a_missing_token_stops_before_the_upload(self) -> None:
        os.environ.pop("HF_TOKEN", None)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as caught:
                self.app._publish(self.job(), Path(tmp), self.meta())
        self.assertIn("Modal Secret", str(caught.exception))
        self.assertEqual(self.calls, [])


class ImageTest(unittest.TestCase):
    """The image must install the package and keep the tree as the fallback."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = (_HERE / "app.py").read_text(encoding="utf-8")

    def test_the_distribution_is_installed_at_build(self) -> None:
        self.assertIn('.pip_install(f"{REPO_ROOT}/python[hub]")', self.text)

    def test_the_wheel_line_awaits_d6(self) -> None:
        self.assertIn("nanochat_cpp_cuda_sm75", self.text)
        self.assertIn('.pip_install("nanochat-cpp[hub] @', self.text)

    def test_the_repository_tree_stays_baked_in(self) -> None:
        self.assertIn("add_local_dir", self.text)
        self.assertIn("copy=True", self.text)

    def test_the_notebook_modules_are_importable(self) -> None:
        self.assertIn("{REPO_ROOT}/notebooks", self.text)

    def test_the_train_function_accepts_a_publish(self) -> None:
        self.assertIn('publish_json: str = ""', self.text)

    def test_the_train_function_accepts_a_stage(self) -> None:
        self.assertIn('stage_json: str = ""', self.text)


class WeightsTest(unittest.TestCase):
    """The weights Volume holds ``<repo_id>/<stage>/`` (section 6.4, D8)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app, cls.sdk = _load_app(with_sdk=True)

    def test_the_volume_is_declared_beside_the_runs_volume(self) -> None:
        self.assertEqual([name for name, _ in self.sdk.volume_names],
                         ["nanochat-runs", "nanochat-weights"])
        self.assertTrue(all(create for _, create in self.sdk.volume_names))
        self.assertEqual(self.app.WEIGHTS_ROOT, "/weights")

    def test_both_volumes_are_mounted_on_the_worker(self) -> None:
        mounted = [call["volumes"] for call in self.sdk.function_calls
                   if "volumes" in call]
        self.assertEqual(len(mounted), 1)
        self.assertEqual(sorted(mounted[0]), ["/runs", "/weights"])
        self.assertIs(mounted[0]["/weights"], self.app.weights)
        self.assertIs(mounted[0]["/runs"], self.app.runs)

    def test_a_stage_is_a_repository_and_a_stage_folder(self) -> None:
        self.assertEqual(self.app.stage_dir("owner/model", "base"),
                         Path("/weights/owner/model/base"))

    def test_a_cached_revision_is_not_downloaded_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.app.stage_dir("owner/model", "base", root=tmp)
            folder.mkdir(parents=True)
            (folder / "checkpoint.ckpt").write_bytes(b"ckpt")
            (folder / self.app.REVISION_FILE).write_text("abc123\n")
            calls = []
            got = self.app.fetch_stage(
                "owner/model", "base", revision="abc123", root=tmp,
                pull=lambda *args, **kwargs: calls.append(args),
                commit=lambda: calls.append(("commit",)))
        self.assertEqual(got, folder)
        self.assertEqual(calls, [])

    def test_a_new_revision_is_one_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.app.stage_dir("owner/model", "base", root=tmp)
            folder.mkdir(parents=True)
            (folder / "checkpoint.ckpt").write_bytes(b"old")
            (folder / self.app.REVISION_FILE).write_text("abc123\n")
            calls = []

            def pull(repo_id, stage, dest, **kwargs):
                calls.append((repo_id, stage, str(dest), kwargs))
                return folder

            committed = []
            got = self.app.fetch_stage(
                "owner/model", "base", revision="def456", root=tmp,
                pull=pull, commit=lambda: committed.append(True))
            self.assertEqual(got, folder)
            self.assertEqual(calls[0][:3],
                             ("owner/model", "base", str(folder.parent)))
            self.assertEqual(calls[0][3]["revision"], "def456")
            self.assertEqual(calls[0][3]["patterns"],
                             self.app.STAGE_FILES)
            self.assertEqual(self.app.cached_revision(folder), "def456")
            self.assertEqual(committed, [True])

    def test_a_warm_start_points_at_the_cached_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.app.stage_dir("owner/model", "base", root=tmp)
            folder.mkdir(parents=True)
            (folder / "checkpoint.ckpt").write_bytes(b"ckpt")

            def fetch(repo_id, stage, **kwargs):
                return folder

            with contextlib.redirect_stderr(io.StringIO()):
                job = self.app._warm_start(
                    {"data": {}},
                    json.dumps({"repo_id": "owner/model", "stage": "base",
                                "revision": "abc123"}),
                    fetch=fetch)
        self.assertEqual(job["warm_start_path"],
                         str(folder / "checkpoint.ckpt"))
        self.assertEqual(job["data"], {})

    def test_an_empty_stage_keeps_the_job_alone(self) -> None:
        job = {"warm_start_path": "/runs/earlier/checkpoint.ckpt"}
        self.assertEqual(self.app._warm_start(job, ""), job)

    def test_a_stage_without_a_checkpoint_stops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.app.stage_dir("owner/model", "base", root=tmp)
            folder.mkdir(parents=True)
            with self.assertRaises(RuntimeError) as caught:
                self.app._warm_start(
                    {}, json.dumps({"repo_id": "owner/model",
                                    "stage": "base"}),
                    fetch=lambda *_args, **_kwargs: folder)
        self.assertIn("warm start needs a checkpoint", str(caught.exception))


class SourceTest(unittest.TestCase):
    """Which nanochat.cpp the container runs: the install, else the tree."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _load_app()

    def run_stdout(self, text, code=0):
        return lambda *_args, **_kwargs: types.SimpleNamespace(
            stdout=text, returncode=code)

    def test_an_installed_package_is_preferred(self) -> None:
        origin = ("/usr/local/lib/python3.11/site-packages/"
                  "nanochat_cpp/__init__.py\n")
        self.assertEqual(
            self.app.package_source(environ={}, run=self.run_stdout(origin)),
            "installed")

    def test_no_install_keeps_the_repository_path(self) -> None:
        self.assertEqual(
            self.app.package_source(environ={}, run=self.run_stdout("\n")),
            "repository")

    def test_a_failed_probe_keeps_the_repository_path(self) -> None:
        def run(*_args, **_kwargs):
            raise OSError("no interpreter")

        self.assertEqual(self.app.package_source(environ={}, run=run),
                         "repository")

    def test_the_variable_forces_one_path(self) -> None:
        self.assertEqual(
            self.app.package_source(
                environ={self.app.SOURCE_ENV: "repository"},
                run=self.run_stdout("/site/nanochat_cpp/__init__.py\n")),
            "repository")
        self.assertEqual(
            self.app.package_source(
                environ={self.app.SOURCE_ENV: "installed"},
                run=self.run_stdout("\n")),
            "installed")

    def test_the_worker_environment_follows_the_source(self) -> None:
        root = self.app.REPO_ROOT
        installed = self.app.worker_env("installed", {"PYTHONPATH": "/x"})
        self.assertEqual(installed["PYTHONPATH"],
                         f"{root}/notebooks{os.pathsep}/x")
        tree = self.app.worker_env("repository", {})
        self.assertEqual(
            tree["PYTHONPATH"],
            f"{root}/python{os.pathsep}{root}/notebooks")

    def test_the_train_function_reports_the_source(self) -> None:
        text = (_HERE / "app.py").read_text(encoding="utf-8")
        self.assertIn('"source": source', text)
        self.assertIn('worker_env(source)', text)


if __name__ == "__main__":
    unittest.main()
