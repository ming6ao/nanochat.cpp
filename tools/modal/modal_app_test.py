"""T0: the contract of the Modal app in ``tools/modal/app.py``.

The test imports the app with a stub Modal SDK, so it needs no SDK, no account,
and no network. It gates the two things a client cannot see: the card and the
architecture are one pair, and the names match the client.
"""

from __future__ import annotations

import importlib.util
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
    """A Modal SDK whose images, volumes, and secrets accept any call."""
    module = types.ModuleType("modal")
    module.Image = _Chain()
    module.Volume = _Chain()
    module.Secret = _Chain()
    module.App = lambda *_args, **_kwargs: types.SimpleNamespace(
        function=_decorator)
    return module


def _load_app():
    """Import ``app.py`` with the stub SDK on ``sys.modules``."""
    path = _HERE / "app.py"
    spec = importlib.util.spec_from_file_location("modal_app", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"modal": _stub_sdk()}):
        spec.loader.exec_module(module)
    return module


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
    """The image must carry the Hub client and the notebook modules."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = (_HERE / "app.py").read_text(encoding="utf-8")

    def test_the_hub_client_is_installed(self) -> None:
        self.assertIn('pip_install("huggingface_hub")', self.text)

    def test_the_notebook_modules_are_importable(self) -> None:
        self.assertIn("{REPO_ROOT}/notebooks", self.text)

    def test_the_train_function_accepts_a_publish(self) -> None:
        self.assertIn('publish_json: str = ""', self.text)


if __name__ == "__main__":
    unittest.main()
