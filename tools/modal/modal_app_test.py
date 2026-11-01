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
import subprocess
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


def _stub_sdk() -> types.ModuleType:
    """A Modal SDK whose images, volumes, and secrets accept any call.

    The stub records the Volume names, the ``app.function`` keywords and the
    name of each decorated function, and the ``app.local_entrypoint`` calls,
    so a test reads which volumes the app declares, which one each function
    mounts, and which entry point a person runs by hand.
    """
    module = types.ModuleType("modal")
    module.volume_names = []
    module.function_calls = []
    module.local_entrypoints = []

    def from_name(name, create_if_missing=False, **_kwargs):
        module.volume_names.append((str(name), bool(create_if_missing)))
        return _Chain()

    def function(**kwargs):
        def apply(func):
            module.function_calls.append(dict(kwargs, name=func.__name__))
            return func

        return apply

    def local_entrypoint(**_kwargs):
        def apply(func):
            module.local_entrypoints.append(func.__name__)
            return func

        return apply

    module.Image = _Chain()
    module.Volume = types.SimpleNamespace(from_name=from_name)
    module.Secret = _Chain()
    module.App = lambda *_args, **_kwargs: types.SimpleNamespace(
        function=function, local_entrypoint=local_entrypoint)
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
        for name in ("doctor", "train", "pipeline", "pipeline_demo"):
            self.assertTrue(callable(getattr(self.app, name)), name)

    def test_the_demo_is_a_local_entry_point(self) -> None:
        _module, sdk = _load_app(with_sdk=True)
        self.assertEqual(sdk.local_entrypoints, ["pipeline_demo"])

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

    def test_the_image_carries_the_two_stage_executables(self) -> None:
        # The chain launches both children, and one build produces all three
        # outputs (docs/pipeline-plan.md B6 and D2).
        self.assertEqual(self.app.STAGE_TARGETS,
                         ("//src:sft_main", "//src:rl_worker"))
        self.assertEqual(
            self.app.STAGE_TARGETS,
            tuple(f"//src:{name}" for name in self.app.STAGE_NAMES))

    def test_the_image_names_the_two_stage_executables(self) -> None:
        # The environment override wins in `_build.stage_binary`, which is what
        # makes them findable when the install, and not the tree, answers
        # `import nanochat_cpp` (section 6.4).
        root = self.app.REPO_ROOT
        self.assertEqual(
            self.app.STAGE_ENV,
            {"NANOCHAT_SFT_MAIN": f"{root}/bazel-bin/src/sft_main",
             "NANOCHAT_RL_WORKER": f"{root}/bazel-bin/src/rl_worker"})


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

    def test_a_job_without_a_mesh_is_the_function_count(self) -> None:
        # A pipeline job declares no mesh: the chain runs one replica of each
        # stage (docs/pipeline-plan.md B11), so the function's count governs.
        self.app._check_cards(json.dumps({"run_id": "demo",
                                          "profile": "demo"}))

    def test_a_wrong_architecture_stops_a_pipeline_job(self) -> None:
        self.app._device_capability = lambda: "8.0"
        with self.assertRaises(RuntimeError) as caught:
            self.app._check_cards(json.dumps({"run_id": "demo"}))
        self.assertIn("deploy an image", str(caught.exception))


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


class PipelineTest(unittest.TestCase):
    """The chain entry of section 6.3: one job file, one report, one uploader.

    The stub keeps the function callable, so the body runs here with a
    temporary run root, a fake Volume, and a fake launcher. The launcher writes
    the ``deliver/`` folder and the corpus the real chain writes.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.app, cls.sdk = _load_app(with_sdk=True)

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = temporary.name
        for name, value in (("REMOTE_ROOT", self.root),
                            ("package_source", lambda: "repository"),
                            ("_device_capability", lambda: "7.5"),
                            ("_device_count", lambda: 1)):
            patcher = mock.patch.object(self.app, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.commits = []
        patcher = mock.patch.object(
            self.app, "runs",
            types.SimpleNamespace(commit=lambda: self.commits.append(True)))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.commands = []
        self.environments = []
        self.returncode = 0
        self.report = {"version": 1, "stages": [],
                       "host": {"arch": "sm_75", "backend": "cuda"}}
        patcher = mock.patch.object(
            self.app, "subprocess",
            types.SimpleNamespace(run=self.launch,
                                  STDOUT=subprocess.STDOUT))
        patcher.start()
        self.addCleanup(patcher.stop)

    def launch(self, argv, **kwargs):
        """Stand in for the ``tools/nanochat client`` launch."""
        self.commands.append(list(argv))
        self.environments.append(kwargs.get("env"))
        job_path = Path(argv[-1])
        job = json.loads(job_path.read_text(encoding="utf-8"))
        folder = self.app.chain_dir(job_path.parent, job)
        deliver = folder / self.app.DELIVER_DIRNAME
        deliver.mkdir(parents=True, exist_ok=True)
        if self.report is not None:
            (deliver / self.app.REPORT_NAME).write_text(
                json.dumps(self.report), encoding="utf-8")
        corpus = folder / self.app.CORPUS_DIRNAME
        corpus.mkdir(parents=True, exist_ok=True)
        (corpus / "loader_tokenizer.nctoken").write_text("tok")
        (folder / "rl").mkdir(parents=True, exist_ok=True)
        return types.SimpleNamespace(returncode=self.returncode)

    def call(self, name):
        """The ``app.function`` keywords of the function ``name``."""
        for keywords in self.sdk.function_calls:
            if keywords["name"] == name:
                return keywords
        raise AssertionError(f"no function named {name}")

    def run_once(self, job=None, run_id="run-1", **kwargs):
        """Run the function once and return its parsed answer."""
        text = self.app.pipeline(
            json.dumps({"run_id": "demo"} if job is None else job),
            run_id, **kwargs)
        return json.loads(text)

    def test_the_argv_names_the_client_and_the_chain_module(self) -> None:
        # Design decision D10: the module is always wrapped by tools/nanochat.
        self.run_once()
        argv = self.commands[0]
        self.assertEqual(argv[:7], ["tools/nanochat", "client", "--profile",
                                    "train", "--", sys.executable, "-m"])
        self.assertEqual(argv[7], "nanochat_cpp.pipeline")
        self.assertEqual(argv[8],
                         str(Path(self.root) / "run-1" / "job.json"))
        self.assertEqual(len(argv), 9)

    def test_the_chain_runs_from_the_one_source_environment(self) -> None:
        self.run_once()
        self.assertEqual(self.environments[0],
                         self.app.worker_env("repository"))

    def test_out_root_is_rewritten_to_the_volume(self) -> None:
        self.run_once({"run_id": "demo", "base_steps": 100,
                       "out_root": "/elsewhere/runs/demo"})
        job = json.loads(
            (Path(self.root) / "run-1" / "job.json").read_text())
        self.assertEqual(job["out_root"], f"{self.root}/run-1")
        self.assertEqual(job["base_steps"], 100)
        self.assertEqual(job["run_id"], "demo")

    def test_the_run_directory_and_the_job_file_are_kept(self) -> None:
        self.run_once()
        run_dir = Path(self.root) / "run-1"
        self.assertTrue((run_dir / "job.json").is_file())
        self.assertTrue((run_dir / "worker.log").is_file())

    def test_the_run_directory_is_committed_for_a_poll(self) -> None:
        self.run_once()
        self.assertEqual(self.commits, [True])

    def test_the_card_check_runs_on_the_rewritten_job(self) -> None:
        seen = []
        with mock.patch.object(self.app, "_check_cards", seen.append):
            self.run_once()
        self.assertEqual(json.loads(seen[0])["out_root"],
                         f"{self.root}/run-1")

    def test_one_chain_at_a_time_on_one_card(self) -> None:
        # Another chain at the same time would fight for the Volume and the
        # card, so the function keeps the `train` limit (section 6.3).
        keywords = self.call("pipeline")
        self.assertEqual(keywords["max_containers"], 1)
        self.assertEqual(keywords["gpu"], self.app.gpu_request())
        self.assertEqual(keywords["timeout"], self.app.TIMEOUT_SECONDS)
        self.assertIsInstance(keywords["secrets"], list)

    def test_the_report_of_the_chain_comes_back(self) -> None:
        answer = self.run_once()
        self.assertEqual(answer["report"]["host"]["arch"], "sm_75")
        self.assertEqual(answer["run_dir"],
                         str(Path(self.root) / "run-1" / "demo"))
        self.assertEqual(answer["source"], "repository")
        self.assertEqual(answer["publish"], {})

    def test_an_undelivered_chain_fails_the_call(self) -> None:
        # A zero exit with no delivery is a failed gate. This case used to
        # return an empty document, so a chain that wrote nothing looked like a
        # successful run with no data, and the empty report reached the caller.
        self.report = None
        with self.assertRaises(RuntimeError) as caught:
            self.app.pipeline(json.dumps({"run_id": "demo"}), "run-1")
        self.assertIn("wrote no deliver/pipeline_report.json",
                      str(caught.exception))

    def test_a_failed_chain_stops_the_function(self) -> None:
        self.returncode = 2
        with self.assertRaises(RuntimeError) as caught:
            self.run_once()
        self.assertIn("the chain exited 2", str(caught.exception))

    def test_the_final_stage_publishes_through_the_one_uploader(self) -> None:
        calls = []

        def publish(job_json, run_dir, publish_json):
            calls.append((json.loads(job_json), str(run_dir), publish_json))
            return {"stage": "chatrl"}

        meta = json.dumps({"stage": "chatrl", "repo_id": "o/m"})
        with mock.patch.object(self.app, "_publish", publish):
            answer = self.run_once(publish_json=meta)
        job, folder, publish_json = calls[0]
        chain = Path(self.root) / "run-1" / "demo"
        self.assertEqual(folder, str(chain / "rl"))
        self.assertEqual(publish_json, meta)
        self.assertEqual(
            job["data"]["tokenizer_path"],
            str(Path(self.root) / "run-1" / "demo" / "corpus"
                / "loader_tokenizer.nctoken"))
        self.assertEqual(answer["publish"], {"stage": "chatrl"})

    def test_the_stage_argument_is_noted_and_ignored(self) -> None:
        # The signature carries it so one client call shape serves both
        # functions; the chain warm starts each stage from the one before it.
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            answer = self.run_once(stage_json=json.dumps({"stage": "base"}))
        self.assertIn("takes no stage_json", stderr.getvalue())
        self.assertEqual(answer["report"]["host"]["arch"], "sm_75")

    def test_the_report_names_the_tokenizer_of_the_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            corpus = folder / self.app.CORPUS_DIRNAME
            corpus.mkdir()
            (corpus / "loader_tokenizer.nctoken").write_text("tok")
            named = corpus / "named.nctoken"
            named.write_text("tok")
            report = {"stages": [{"name": "base",
                                 "extra": {"tokenizer": str(named)}}]}
            self.assertEqual(self.app.stage_tokenizer(report, folder),
                             str(named))

    def test_the_corpus_is_the_fallback_for_the_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            corpus = folder / self.app.CORPUS_DIRNAME
            corpus.mkdir()
            tokenizer = corpus / "loader_tokenizer.nctoken"
            tokenizer.write_text("tok")
            self.assertEqual(self.app.stage_tokenizer({}, folder),
                             str(tokenizer))
        self.assertEqual(self.app.stage_tokenizer({}, folder / "nope"), "")

    def test_the_run_directory_nests_the_job_run_id(self) -> None:
        self.assertEqual(self.app.chain_dir("/runs/x", {"run_id": "demo"}),
                         Path("/runs/x/demo"))
        self.assertEqual(self.app.chain_dir("/runs/x", {}), Path("/runs/x"))

    def test_the_demo_job_runs_the_demo_profile(self) -> None:
        job = self.app.demo_job()
        self.assertEqual(job["run_id"], "demo")
        self.assertEqual(job["profile"], "demo")
        self.assertEqual(job["out_root"], "")
        self.assertTrue(job["deliver"])

    def test_the_demo_job_falls_back_to_the_copy_in_this_file(self) -> None:
        # A workstation that installed the Modal SDK and not the package.
        with mock.patch.dict(sys.modules, {"nanochat_cpp.pipeline": None}):
            job = self.app.demo_job("other")
        self.assertEqual(job, dict(self.app.DEMO_JOB, run_id="other"))

    def test_the_demo_job_matches_the_package_defaults(self) -> None:
        try:
            from nanochat_cpp import pipeline as chain  # noqa: PLC0415
        except Exception:  # noqa: BLE001 - the test host may not hold it
            self.skipTest("nanochat_cpp is not importable here")
        self.assertEqual(self.app.DEMO_JOB, chain.Job().to_dict())


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

    def test_the_image_build_names_the_stage_executables(self) -> None:
        self.assertIn("_build.build_stages(options=_build.BuildOptions",
                      self.text)
        self.assertIn("//src:sft_main", self.text)
        self.assertIn("//src:rl_worker", self.text)

    def test_the_host_file_stays_out_of_the_image(self) -> None:
        # The host file pins /usr/bin/gcc-12, which the image does not have, so
        # a deploy from a host that ran tools/kaggle/bootstrap.sh would build
        # an image that cannot compile (section 8 row 10).
        self.assertRegex(self.text, r"ignore=\[[^\]]*\.bazelrc\.local")


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
        mounted = {call["name"]: call["volumes"]
                   for call in self.sdk.function_calls if "volumes" in call}
        self.assertEqual(sorted(mounted), ["pipeline", "train"])
        self.assertEqual(sorted(mounted["train"]), ["/runs", "/weights"])
        self.assertIs(mounted["train"]["/weights"], self.app.weights)
        self.assertIs(mounted["train"]["/runs"], self.app.runs)
        # The chain reads and writes the run root only: its stages warm start
        # from each other, so it never reads the stage cache of `train`.
        self.assertEqual(sorted(mounted["pipeline"]), ["/runs"])
        self.assertIs(mounted["pipeline"]["/runs"], self.app.runs)

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
