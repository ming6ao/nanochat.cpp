"""T0: the contract of the Modal app in ``tools/modal/app.py``.

The test imports the app with a stub Modal SDK, so it needs no SDK, no account,
and no network. It gates what a client cannot see:

- the two legs of ``docs/pipeline-scaleup-plan.md`` section 7 -- one card, one
  architecture, one precision, and the flag triple each build passes;
- one image per leg, so a T4 container declares ``sm_75`` at fp32 and an H100
  container declares ``sm_90`` at bf16;
- the four function names ``notebooks/target.py`` resolves, and the ``cpu=``
  value both legs share;
- the three gates G1 to G3 of section 8, each of which must fail on a report
  document that was doctored.
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
    read what the module asked the SDK for. The module is registered in
    ``sys.modules`` while it executes, because ``dataclasses`` resolves a
    string annotation through it (the module holds one frozen record, ``Leg``).
    """
    path = _HERE / "app.py"
    spec = importlib.util.spec_from_file_location("modal_app", path)
    module = importlib.util.module_from_spec(spec)
    sdk = _stub_sdk()
    with mock.patch.dict(sys.modules, {"modal": sdk, "modal_app": module}):
        spec.loader.exec_module(module)
    return (module, sdk) if with_sdk else module


def _source() -> str:
    """The text of ``app.py``, for the contracts that are not a call."""
    return (_HERE / "app.py").read_text(encoding="utf-8")


def _report_document(*, precision="bf16", arch="sm_90", checkpoint="bf16",
                     step_ms=40.0, mfu=0.0001) -> dict:
    """One chain report, shaped the way ``nanochat_cpp.pipeline`` writes it.

    The base stage carries the rate facts of row 4 (``step_ms``, ``mfu``) and
    the hardware block, and the RL stage carries the checkpoint metadata the
    container wrote, so a gate reads exactly what the chain recorded.
    """
    return {
        "version": 2,
        "job": {"run_id": "demo", "profile": "standard"},
        "stages": [
            {"name": "base", "steps": 200, "seconds": 30.0,
             "checkpoint": "/runs/x/base/checkpoint.ckpt",
             "extra": {"step_ms": step_ms, "tokens_per_sec": 1.6,
                       "mfu": mfu,
                       "hardware": {"backend": "cuda", "arch": arch,
                                    "precision": precision}}},
            {"name": "sft", "steps": 200, "seconds": 20.0},
            {"name": "rl", "steps": 8, "seconds": 10.0,
             "checkpoint": "/runs/x/rl/checkpoint.ckpt",
             "metadata": {"stage": "rl", "precision": checkpoint,
                          "step": 8}},
            {"name": "eval", "steps": 1, "seconds": 1.0},
        ],
        "host": {"backend": "cuda", "arch": arch, "precision": precision,
                 "revision": "abc123"},
        "total_seconds": 61.0,
        "delivered": "/runs/x/deliver/chatrl.ckpt",
        "report_path": "/runs/x/deliver/pipeline_report.json",
    }


class AppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _load_app()

    def test_the_four_functions_and_the_entry_points_exist(self) -> None:
        for name in ("doctor", "train", "train_h100", "pipeline",
                     "pipeline_h100", "pipeline_demo"):
            self.assertTrue(callable(getattr(self.app, name)), name)

    def test_the_demo_is_a_local_entry_point(self) -> None:
        _module, sdk = _load_app(with_sdk=True)
        self.assertEqual(sdk.local_entrypoints, ["pipeline_demo"])

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


class LegTest(unittest.TestCase):
    """The two legs of section 7: one record each, and one home for the triple."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _load_app()

    def test_the_two_legs_are_the_ones_the_client_names(self) -> None:
        import target

        self.assertEqual(sorted(self.app.LEGS), sorted(target.LEGS))
        self.assertEqual(tuple(leg.name for leg in self.app.LEG_ORDER),
                         (target.LEG_T4, target.LEG_H100))

    def test_the_t4_leg_runs_first(self) -> None:
        # The T4 artifact survives an H100 failure, and a reviewer reads the
        # cheap leg while the expensive one runs (section 7).
        self.assertIs(self.app.LEG_ORDER[0], self.app.T4)
        self.assertIs(self.app.LEG_ORDER[1], self.app.H100)

    def test_the_t4_leg_is_one_card_at_sm_75_in_fp32(self) -> None:
        leg = self.app.T4
        self.assertEqual((leg.name, leg.card, leg.count), ("t4", "T4", 1))
        self.assertEqual((leg.arch, leg.precision), ("sm_75", "fp32"))
        self.assertEqual(leg.capability, "7.5")

    def test_the_h100_leg_is_one_card_at_sm_90_in_bf16(self) -> None:
        leg = self.app.H100
        self.assertEqual((leg.name, leg.card, leg.count), ("h100", "H100", 1))
        self.assertEqual((leg.arch, leg.precision), ("sm_90", "bf16"))
        self.assertEqual(leg.capability, "9.0")

    def test_the_t4_flags_are_the_table_of_section_7(self) -> None:
        self.assertEqual(self.app.T4.build,
                         ("--config=cuda", "--define=precision=fp32",
                          "--@rules_cuda//cuda:archs=sm_75"))

    def test_the_h100_flags_are_the_table_of_section_7(self) -> None:
        self.assertEqual(self.app.H100.build,
                         ("--config=cuda", "--define=precision=bf16",
                          "--@rules_cuda//cuda:archs=sm_90"))

    def test_the_builder_emits_the_two_triples(self) -> None:
        # The triple is a contract with `_build._build_command`, which is the
        # one home of the flag line. The host of this test may not hold the
        # package, so an absent one skips instead of inventing a pass.
        try:
            from nanochat_cpp import _build  # noqa: PLC0415 - optional here
        except Exception:  # noqa: BLE001 - the test host may not hold it
            self.skipTest("nanochat_cpp is not importable here")
        for leg in self.app.LEG_ORDER:
            options = _build.BuildOptions.resolve(
                backend="cuda", precision=leg.precision, arch=leg.arch)
            command = _build._build_command("tools/nanochat", options, ())
            self.assertEqual(tuple(command[2:]), leg.build, leg.name)

    def test_a_leg_is_frozen(self) -> None:
        # One home for the triple: a run cannot move a leg under a deployment.
        with self.assertRaises(Exception):
            self.app.T4.arch = "sm_90"

    def test_the_gpu_request_names_the_count(self) -> None:
        self.assertEqual(self.app.T4.gpu, "T4")
        self.assertEqual(self.app.H100.gpu, "H100")
        self.assertEqual(self.app.gpu_request("A100-80GB", 4), "A100-80GB:4")
        self.assertEqual(self.app.gpu_request("T4", 1), "T4")

    def test_the_roofline_device_is_the_card_of_the_leg(self) -> None:
        # The device limit table is keyed by the card in lower case.
        self.assertEqual(self.app.T4.device, "t4")
        self.assertEqual(self.app.H100.device, "h100")

    def test_the_run_ids_carry_the_leg_name(self) -> None:
        self.assertEqual(self.app.leg_run_id("run-1", self.app.T4), "run-1-t4")
        self.assertEqual(self.app.leg_run_id("run-1", self.app.H100),
                         "run-1-h100")


class ImageTest(unittest.TestCase):
    """One image per leg, and the build line each one passes (sections 7)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _load_app()

    def test_one_image_belongs_to_one_leg(self) -> None:
        # The cache is keyed by the leg name, so two legs never share a build.
        self.assertIs(self.app.image_for(self.app.T4),
                      self.app.image_for(self.app.T4))
        self.app.image_for(self.app.H100)
        self.assertEqual(sorted(self.app._IMAGES), ["h100", "t4"])

    def test_the_t4_image_takes_its_precision_and_arch_from_the_leg(self) -> None:
        env = self.app.image_env(self.app.T4)
        self.assertEqual(env["NANOCHAT_CPP_PRECISION"], "fp32")
        self.assertEqual(env["NANOCHAT_CUDA_ARCH"], "sm_75")
        self.assertEqual(env["NANOCHAT_CPP_BACKEND"], "cuda")

    def test_the_h100_image_takes_its_precision_and_arch_from_the_leg(self) -> None:
        env = self.app.image_env(self.app.H100)
        self.assertEqual(env["NANOCHAT_CPP_PRECISION"], "bf16")
        self.assertEqual(env["NANOCHAT_CUDA_ARCH"], "sm_90")

    def test_the_two_images_declare_different_builds(self) -> None:
        self.assertNotEqual(self.app.image_env(self.app.T4),
                            self.app.image_env(self.app.H100))

    def test_each_build_command_passes_its_own_leg(self) -> None:
        t4 = self.app.image_build_command(self.app.T4)
        h100 = self.app.image_build_command(self.app.H100)
        self.assertIn("precision='fp32'", t4)
        self.assertIn("arch='sm_75'", t4)
        self.assertIn("precision='bf16'", h100)
        self.assertIn("arch='sm_90'", h100)
        # The builder stays the one home of the flag line.
        self.assertIn("_build._build_command(", t4)
        self.assertIn("check=True", t4)

    def test_the_build_command_names_the_roofline_tool(self) -> None:
        # Gate G3 runs the tool inside the container, so the image build names
        # it beside the library and the two stage executables (sections 7, 8).
        self.assertIn("//src:roofline_main", self.app.BUILD_TARGETS)
        self.assertEqual(self.app.ROOFLINE_TARGET, "//src:roofline_main")
        for leg in self.app.LEG_ORDER:
            self.assertIn("//src:roofline_main",
                          self.app.image_build_command(leg))

    def test_the_build_names_the_library_and_the_stage_executables(self) -> None:
        self.assertEqual(self.app.BUILD_TARGETS[0], "//bindings:nanochat_shared")
        for target in self.app.STAGE_TARGETS:
            self.assertIn(target, self.app.BUILD_TARGETS)

    def test_the_distribution_is_installed_at_build(self) -> None:
        self.assertIn('.pip_install(f"{REPO_ROOT}/python[hub]")', _source())

    def test_the_wheel_line_awaits_d6(self) -> None:
        self.assertIn("nanochat_cpp_cuda_sm75", _source())
        self.assertIn('.pip_install("nanochat-cpp[hub] @', _source())

    def test_the_repository_tree_stays_baked_in(self) -> None:
        self.assertIn("add_local_dir", _source())
        self.assertIn("copy=True", _source())

    def test_the_notebook_modules_are_importable(self) -> None:
        self.assertIn("{REPO_ROOT}/notebooks", _source())

    def test_the_train_function_accepts_a_publish(self) -> None:
        self.assertIn('publish_json: str = ""', _source())

    def test_the_train_function_accepts_a_stage(self) -> None:
        self.assertIn('stage_json: str = ""', _source())

    def test_the_host_file_stays_out_of_the_image(self) -> None:
        # The host file pins /usr/bin/gcc-12, which the image does not have, so
        # a deploy from a host that ran tools/kaggle/bootstrap.sh would build
        # an image that cannot compile (section 8 row 10).
        self.assertRegex(_source(), r"ignore=\[[^\]]*\.bazelrc\.local")

    def test_no_leg_passes_a_named_simulator_config(self) -> None:
        # `h100` in .bazelrc is the simulator, and it also sets the native
        # collective, so the hardware leg takes the explicit triple instead
        # (section 3, trap 1). No leg builds fp8 either (P3).
        text = _source()
        for forbidden in ("config=h100", "define=simulator",
                          "precision=fp8", "config=fp8"):
            self.assertNotIn(forbidden, text)

    def test_the_old_globals_are_gone(self) -> None:
        # The triple has one home, the leg record: a module global that also
        # named the leg would give it two.
        text = _source()
        for gone in ("GPU_TYPE", "GPU_COUNT", "CUDA_ARCH ="):
            self.assertNotIn(gone, text)


class CheckCardsTest(unittest.TestCase):
    """The card guard: the leg, the container, and the job must agree."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _load_app()

    def setUp(self) -> None:
        self.app._device_capability = lambda: "7.5"
        self.app._device_count = lambda: 1

    def job(self, replicas: int) -> str:
        return json.dumps({"mesh": {"count": replicas}})

    def test_one_replica_on_one_card_passes(self) -> None:
        self.app._check_cards(self.app.T4, self.job(1))

    def test_the_h100_leg_checks_sm_90(self) -> None:
        self.app._device_capability = lambda: "9.0"
        self.app._check_cards(self.app.H100, self.job(1))

    def test_a_t4_container_rejects_an_h100_card(self) -> None:
        # A mismatched deployment stops before the chain starts (section 7).
        self.app._device_capability = lambda: "9.0"
        with self.assertRaises(RuntimeError) as caught:
            self.app._check_cards(self.app.T4, self.job(1))
        self.assertIn("the t4 image is built for sm_75", str(caught.exception))
        self.assertIn("deploy an image", str(caught.exception))

    def test_an_h100_container_rejects_a_t4_card(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            self.app._check_cards(self.app.H100, self.job(1))
        self.assertIn("the h100 image is built for sm_90",
                      str(caught.exception))

    def test_a_count_that_disagrees_with_the_leg_stops(self) -> None:
        self.app._device_count = lambda: 4
        with self.assertRaises(RuntimeError) as caught:
            self.app._check_cards(self.app.T4, self.job(4))
        self.assertIn("deploy the leg with its own count",
                      str(caught.exception))

    def test_a_job_that_disagrees_with_the_count_stops(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            self.app._check_cards(self.app.T4, self.job(4))
        self.assertIn("world_size 1", str(caught.exception))

    def test_a_job_without_a_mesh_is_the_leg_count(self) -> None:
        # A pipeline job declares no mesh: the chain runs one replica of each
        # stage (docs/pipeline-plan.md B11), so the leg's count governs.
        self.app._check_cards(self.app.T4, json.dumps({"run_id": "demo",
                                                       "profile": "demo"}))

    def test_a_wrong_architecture_stops_a_pipeline_job(self) -> None:
        self.app._device_capability = lambda: "8.0"
        with self.assertRaises(RuntimeError) as caught:
            self.app._check_cards(self.app.H100, json.dumps({"run_id": "demo"}))
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
            result = self.app._publish(self.job(), Path(tmp), self.meta(),
                                       self.app.T4)
        kind, folder, kwargs = self.calls[0]
        self.assertEqual((kind, folder), ("write_stage", tmp))
        self.assertEqual(kwargs["tokenizer"], "/runs/tok.nctoken")
        self.assertEqual(kwargs["git_revision"], "abc123")
        self.assertEqual(kwargs["cuda_arch"], "sm_75")
        self.assertEqual(kwargs["step"], 500)
        self.assertEqual(result["stage"], "base")

    def test_the_publish_records_the_architecture_of_the_leg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.app._publish(self.job(), Path(tmp), self.meta(),
                              self.app.H100)
        self.assertEqual(self.calls[0][2]["cuda_arch"], "sm_90")

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


class _RemoteFixture(unittest.TestCase):
    """A container stand-in: a temporary run root, a fake Volume, a launcher.

    ``self.launch`` replaces ``subprocess.run``, so the body of a function runs
    here and the argv, the environment, and the files it wrote are readable.

    The Modal SDK is loaded twice overall, once per subclass, because the
    decorated function objects of one load are the ones its own answers came
    from.
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
        patcher = mock.patch.object(
            self.app, "subprocess",
            types.SimpleNamespace(run=self.launch,
                                  STDOUT=subprocess.STDOUT))
        patcher.start()
        self.addCleanup(patcher.stop)

    def launch(self, argv, **kwargs):
        raise NotImplementedError


class TrainTest(_RemoteFixture):
    """``train`` and ``train_h100``: one stage on either leg, one body."""

    def launch(self, argv, **kwargs):
        self.commands.append(list(argv))
        self.environments.append(kwargs.get("env"))
        job_path = Path(argv[-1])
        (job_path.parent / "summary.json").write_text(
            json.dumps({"steps": 1}), encoding="utf-8")
        return types.SimpleNamespace(returncode=self.returncode)

    def run_train(self, name, run_id="run-1"):
        return json.loads(getattr(self.app, name)(
            json.dumps({"run_id": "demo"}), run_id))

    def test_the_t4_worker_entry_is_the_notebook_worker(self) -> None:
        answer = self.run_train("train")
        argv = self.commands[0]
        self.assertEqual(argv[:7], ["tools/nanochat", "client", "--profile",
                                    "train", "--", sys.executable,
                                    "notebooks/mesh_worker.py"])
        self.assertEqual(argv[7], str(Path(self.root) / "run-1" / "job.json"))
        self.assertEqual(answer["summary"], {"steps": 1})
        self.assertEqual(answer["source"], "repository")

    def test_the_h100_twin_runs_the_same_entry(self) -> None:
        # The twins differ in the leg they pass and in nothing else.
        self.app._device_capability = lambda: "9.0"
        answer = self.run_train("train_h100")
        self.assertEqual(answer["leg"], "h100")
        argv = self.commands[0]
        self.assertEqual(argv[:7], ["tools/nanochat", "client", "--profile",
                                    "train", "--", sys.executable,
                                    "notebooks/mesh_worker.py"])

    def test_the_two_legs_pass_one_argv(self) -> None:
        self.run_train("train")
        self.app._device_capability = lambda: "9.0"
        self.run_train("train_h100")
        self.assertEqual(self.commands[0], self.commands[1])

    def test_the_run_directory_is_kept_and_committed(self) -> None:
        self.run_train("train")
        run_dir = Path(self.root) / "run-1"
        self.assertTrue((run_dir / "job.json").is_file())
        self.assertTrue((run_dir / "worker.log").is_file())
        self.assertEqual(self.commits, [True])

    def test_the_environment_is_the_one_source_path(self) -> None:
        self.run_train("train")
        self.assertEqual(self.environments[0],
                         self.app.worker_env("repository"))

    def test_a_failed_worker_stops_the_function(self) -> None:
        self.returncode = 2
        with self.assertRaises(RuntimeError) as caught:
            self.run_train("train")
        self.assertIn("the worker exited 2", str(caught.exception))

    def test_the_publish_takes_the_leg(self) -> None:
        seen = []

        def publish(job_json, run_dir, publish_json, leg=None):
            seen.append(leg)
            return {"stage": "chatrl"}

        self.app._device_capability = lambda: "9.0"
        with mock.patch.object(self.app, "_publish", publish):
            answer = json.loads(self.app.train_h100(
                json.dumps({"run_id": "demo"}), "run-1",
                json.dumps({"stage": "chatrl", "repo_id": "o/m"})))
        self.assertIs(seen[0], self.app.H100)
        self.assertEqual(answer["publish"], {"stage": "chatrl"})


class PipelineTest(_RemoteFixture):
    """The chain entry of section 6.3: one job file, one report, one uploader.

    The stub keeps the function callable, so the body runs here with a
    temporary run root, a fake Volume, a fake launcher, and a fake roofline
    record. The launcher writes the ``deliver/`` folder and the corpus the real
    chain writes.
    """

    def setUp(self) -> None:
        super().setUp()
        self.roofline_calls = []
        patcher = mock.patch.object(self.app, "measure_roofline",
                                    self.measure)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.report = {"version": 2, "stages": [],
                       "host": {"arch": "sm_75", "backend": "cuda",
                                "precision": "fp32"}}

    def measure(self, leg, job, run=None):
        self.roofline_calls.append(leg)
        return {"device": leg.device, "precision": leg.precision,
                "predicted_step_ms": 0.764 if leg is self.app.H100 else 3.6,
                "limit": "launch", "band": [1.0, 200.0]}

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

    @classmethod
    def setUpClass(cls) -> None:
        cls.app, cls.sdk = _load_app(with_sdk=True)

    def run_once(self, job=None, run_id="run-1", name="pipeline", **kwargs):
        """Run the function once and return its parsed answer."""
        text = getattr(self.app, name)(
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

    def test_the_h100_twin_runs_the_same_entry(self) -> None:
        self.app._device_capability = lambda: "9.0"
        answer = self.run_once(name="pipeline_h100")
        self.assertEqual(answer["leg"], "h100")
        self.assertEqual(self.commands[0][:9],
                         ["tools/nanochat", "client", "--profile", "train",
                          "--", sys.executable, "-m", "nanochat_cpp.pipeline",
                          str(Path(self.root) / "run-1" / "job.json")])

    def test_the_four_functions_are_declared(self) -> None:
        names = [keywords["name"] for keywords in self.sdk.function_calls]
        self.assertEqual(sorted(names), ["doctor", "pipeline", "pipeline_h100",
                                         "train", "train_h100"])

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
        with mock.patch.object(self.app, "_check_cards",
                               lambda leg, job_json: seen.append((leg,
                                                                  job_json))):
            self.run_once()
        leg, job_json = seen[0]
        self.assertIs(leg, self.app.T4)
        self.assertEqual(json.loads(job_json)["out_root"],
                         f"{self.root}/run-1")

    def test_the_h100_function_checks_the_h100_leg(self) -> None:
        seen = []
        with mock.patch.object(self.app, "_check_cards",
                               lambda leg, job_json: seen.append(leg)):
            self.run_once(name="pipeline_h100")
        self.assertIs(seen[0], self.app.H100)

    def test_one_chain_at_a_time_on_one_card(self) -> None:
        # Another chain at the same time would fight for the Volume and the
        # card, so the function keeps the `train` limit (section 6.3).
        for name, leg in (("pipeline", "T4"), ("pipeline_h100", "H100"),
                          ("train", "T4"), ("train_h100", "H100")):
            keywords = self.call(name)
            self.assertEqual(keywords["max_containers"], 1, name)
            self.assertEqual(keywords["gpu"], getattr(self.app, leg).gpu, name)
            self.assertEqual(keywords["timeout"], self.app.TIMEOUT_SECONDS)
            self.assertIsInstance(keywords["secrets"], list)

    def test_both_legs_get_the_same_cpu(self) -> None:
        # Modal sizes the CPU by memory when `cpu=` is absent, and gate G2
        # compares step times across the legs under the same host budget.
        values = {name: self.call(name)["cpu"]
                  for name in ("train", "train_h100", "pipeline",
                               "pipeline_h100")}
        self.assertEqual(set(values.values()), {self.app.CPU_CORES})

    def test_one_image_per_leg_on_the_four_functions(self) -> None:
        self.assertIs(self.call("train")["image"],
                      self.app.image_for(self.app.T4))
        self.assertIs(self.call("pipeline")["image"],
                      self.app.image_for(self.app.T4))
        self.assertIs(self.call("train_h100")["image"],
                      self.app.image_for(self.app.H100))
        self.assertIs(self.call("pipeline_h100")["image"],
                      self.app.image_for(self.app.H100))

    def test_the_report_of_the_chain_comes_back(self) -> None:
        answer = self.run_once()
        self.assertEqual(answer["report"]["host"]["arch"], "sm_75")
        self.assertEqual(answer["run_dir"],
                         str(Path(self.root) / "run-1" / "demo"))
        self.assertEqual(answer["source"], "repository")
        self.assertEqual(answer["publish"], {})
        self.assertEqual(answer["leg"], "t4")

    def test_the_roofline_prediction_comes_back_with_the_report(self) -> None:
        # G3 reads the prediction the container recorded (section 8).
        answer = self.run_once()
        self.assertEqual(answer["roofline"]["predicted_step_ms"], 3.6)
        self.assertEqual(answer["roofline"]["device"], "t4")
        self.assertIs(self.roofline_calls[0], self.app.T4)

    def test_the_h100_leg_records_the_h100_prediction(self) -> None:
        self.app._device_capability = lambda: "9.0"
        answer = self.run_once(name="pipeline_h100")
        self.assertEqual(answer["roofline"]["predicted_step_ms"], 0.764)
        self.assertEqual(answer["roofline"]["limit"], "launch")
        self.assertIs(self.roofline_calls[0], self.app.H100)

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

        def publish(job_json, run_dir, publish_json, leg=None):
            calls.append((json.loads(job_json), str(run_dir), publish_json,
                          leg))
            return {"stage": "chatrl"}

        meta = json.dumps({"stage": "chatrl", "repo_id": "o/m"})
        with mock.patch.object(self.app, "_publish", publish):
            answer = self.run_once(publish_json=meta)
        job, folder, publish_json, leg = calls[0]
        chain = Path(self.root) / "run-1" / "demo"
        self.assertEqual(folder, str(chain / "rl"))
        self.assertEqual(publish_json, meta)
        self.assertIs(leg, self.app.T4)
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

    def test_the_demo_job_is_the_package_default(self) -> None:
        job = self.app.demo_job()
        self.assertEqual(job["run_id"], "demo")
        self.assertEqual(job["profile"], "standard")
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


class RooflineTest(unittest.TestCase):
    """The G3 measurement: the command, the prediction, and the limit."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _load_app()

    def argv(self, leg):
        with mock.patch.object(self.app, "shape_of", lambda job: "shape"), \
                mock.patch.object(self.app, "model_flags",
                                  lambda shape: ["--layers", "8"]), \
                mock.patch.object(self.app, "base_batch", lambda: 1):
            return self.app.roofline_argv(leg, {"profile": "standard"})

    def test_the_h100_command_names_the_device_and_the_precision(self) -> None:
        self.assertEqual(
            self.argv(self.app.H100),
            ["tools/nanochat", "roofline", "--device", "h100",
             "--precision", "bf16", "--batch", "1", "--layers", "8", "--json"])

    def test_the_t4_command_names_its_own_device_and_precision(self) -> None:
        self.assertEqual(
            self.argv(self.app.T4)[:7],
            ["tools/nanochat", "roofline", "--device", "t4",
             "--precision", "fp32", "--batch"])

    def test_the_prediction_and_the_limit_are_read(self) -> None:
        document = {"schema": "nanochat.roofline.v1",
                    "rows": [{"name": "gemm", "bound_seconds": 0.0005,
                              "limit": "launch"},
                             {"name": "attn", "bound_seconds": 0.0002,
                              "limit": "compute"}],
                    "step_seconds": 0.000764, "steps": 1}
        self.assertEqual(self.app.predicted_step_ms(document),
                         (0.764, "launch"))

    def test_a_document_without_a_bound_answers_no_prediction(self) -> None:
        self.assertEqual(self.app.predicted_step_ms({}), (None, ""))
        self.assertEqual(self.app.predicted_step_ms({"step_seconds": 0.0}),
                         (None, ""))

    def test_the_record_holds_the_prediction_the_limit_and_the_band(self) -> None:
        document = json.dumps({"schema": "nanochat.roofline.v1",
                               "rows": [{"name": "gemm",
                                         "bound_seconds": 0.000764,
                                         "limit": "launch"}],
                               "step_seconds": 0.000764})

        def run(argv, **_kwargs):
            self.assertIn("--json", argv)
            return types.SimpleNamespace(stdout=document, returncode=0,
                                         stderr="")

        with mock.patch.object(self.app, "roofline_argv",
                               lambda leg, job: ["tools/nanochat",
                                                 "roofline", "--json"]):
            record = self.app.measure_roofline(self.app.H100, {}, run=run)
        self.assertEqual(record["predicted_step_ms"], 0.764)
        self.assertEqual(record["limit"], "launch")
        self.assertEqual(record["band"], [1.0, 200.0])
        self.assertEqual(record["device"], "h100")
        self.assertNotIn("error", record)

    def test_a_failing_tool_records_the_error(self) -> None:
        def run(_argv, **_kwargs):
            return types.SimpleNamespace(stdout="", returncode=2,
                                         stderr="unknown option\n")

        with mock.patch.object(self.app, "roofline_argv",
                               lambda leg, job: ["tools/nanochat"]):
            record = self.app.measure_roofline(self.app.H100, {}, run=run)
        self.assertIn("the tool exited 2", record["error"])
        self.assertNotIn("predicted_step_ms", record)

    def test_a_tool_that_writes_no_json_records_the_error(self) -> None:
        def run(_argv, **_kwargs):
            return types.SimpleNamespace(stdout="not json", returncode=0,
                                         stderr="")

        with mock.patch.object(self.app, "roofline_argv",
                               lambda leg, job: ["tools/nanochat"]):
            record = self.app.measure_roofline(self.app.H100, {}, run=run)
        self.assertIn("wrote no JSON", record["error"])

    def test_a_tool_that_does_not_start_records_the_error(self) -> None:
        def run(_argv, **_kwargs):
            raise OSError("no such file")

        with mock.patch.object(self.app, "roofline_argv",
                               lambda leg, job: ["tools/nanochat"]):
            record = self.app.measure_roofline(self.app.H100, {}, run=run)
        self.assertIn("did not start", record["error"])


class GateTest(unittest.TestCase):
    """G1, G2, and G3 of section 8, each on a doctored report document."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _load_app()

    def report(self, **kwargs) -> dict:
        return _report_document(**kwargs)

    def roofline(self, predicted=0.764, limit="launch"):
        return {"predicted_step_ms": predicted, "limit": limit,
                "device": "h100", "precision": "bf16", "band": [1.0, 200.0]}

    # -- G1: bf16 ran on the H100 ------------------------------------------

    def test_g1_passes_on_a_bf16_h100_report(self) -> None:
        gate = self.app.gate_bf16_ran(self.report())
        self.assertEqual((gate["gate"], gate["status"]), ("G1", "pass"))
        self.assertEqual(gate["read"]["host_precision"], "bf16")
        self.assertEqual(gate["read"]["host_arch"], "sm_90")
        self.assertEqual(gate["read"]["checkpoint_precision"], "bf16")

    def test_g1_fails_when_the_host_precision_is_fp32(self) -> None:
        gate = self.app.gate_bf16_ran(self.report(precision="fp32"))
        self.assertEqual(gate["status"], "fail")
        self.assertIn("host_precision", gate["reason"])

    def test_g1_fails_when_the_checkpoint_records_fp32(self) -> None:
        gate = self.app.gate_bf16_ran(self.report(checkpoint="fp32"))
        self.assertEqual(gate["status"], "fail")
        self.assertIn("checkpoint_precision", gate["reason"])

    def test_g1_fails_on_a_turing_architecture(self) -> None:
        gate = self.app.gate_bf16_ran(self.report(arch="sm_75"))
        self.assertEqual(gate["status"], "fail")
        self.assertIn("host_arch", gate["reason"])

    def test_g1_fails_on_an_empty_report(self) -> None:
        self.assertEqual(self.app.gate_bf16_ran({})["status"], "fail")

    # -- G2: the H100 leg did not regress ----------------------------------

    def test_g2_passes_when_the_two_legs_are_equal(self) -> None:
        gate = self.app.gate_step_ratio(self.report(step_ms=40.0),
                                        self.report(step_ms=40.0))
        self.assertEqual((gate["gate"], gate["status"]), ("G2", "pass"))
        self.assertEqual(gate["read"]["step_ms_ratio"], 1.0)

    def test_g2_passes_at_the_limit(self) -> None:
        gate = self.app.gate_step_ratio(self.report(step_ms=40.0),
                                        self.report(step_ms=50.0))
        self.assertEqual(gate["status"], "pass")

    def test_g2_fails_past_the_limit(self) -> None:
        gate = self.app.gate_step_ratio(self.report(step_ms=40.0),
                                        self.report(step_ms=50.4))
        self.assertEqual(gate["status"], "fail")
        self.assertIn("above 1.25", gate["reason"])

    def test_g2_fails_when_a_leg_recorded_no_step_time(self) -> None:
        t4 = self.report()
        t4["stages"] = [stage for stage in t4["stages"]
                        if stage["name"] != "base"]
        gate = self.app.gate_step_ratio(t4, self.report())
        self.assertEqual(gate["status"], "fail")
        self.assertIn("no positive base step_ms", gate["reason"])

    def test_g2_names_the_shared_cpu(self) -> None:
        gate = self.app.gate_step_ratio(self.report(), self.report())
        self.assertEqual(gate["read"]["cpu"], self.app.CPU_CORES)

    # -- G3: the run matches the model -------------------------------------

    def test_g3_passes_inside_the_band(self) -> None:
        gate = self.app.gate_roofline(self.report(step_ms=40.0),
                                      self.roofline(predicted=0.764))
        self.assertEqual((gate["gate"], gate["status"]), ("G3", "pass"))
        self.assertEqual(gate["read"]["predicted_step_ms"], 0.764)
        self.assertEqual(gate["read"]["limit"], "launch")
        self.assertEqual(gate["read"]["band"], [1.0, 200.0])
        self.assertGreater(gate["read"]["ratio"], 1.0)

    def test_g3_fails_below_the_band(self) -> None:
        gate = self.app.gate_roofline(self.report(step_ms=0.5),
                                      self.roofline(predicted=0.764))
        self.assertEqual(gate["status"], "fail")
        self.assertIn("outside", gate["reason"])

    def test_g3_fails_above_the_band(self) -> None:
        gate = self.app.gate_roofline(self.report(step_ms=400.0),
                                      self.roofline(predicted=0.764))
        self.assertEqual(gate["status"], "fail")

    def test_g3_fails_without_a_prediction(self) -> None:
        gate = self.app.gate_roofline(self.report(),
                                      {"error": "the tool exited 2"})
        self.assertEqual(gate["status"], "fail")
        self.assertIn("the tool exited 2", gate["reason"])

    def test_g3_fails_without_a_measured_step_time(self) -> None:
        report = self.report()
        report["stages"] = [{"name": "rl", "steps": 8}]
        gate = self.app.gate_roofline(report, self.roofline())
        self.assertEqual(gate["status"], "fail")
        self.assertIn("no base step_ms", gate["reason"])

    # -- The three gates together ------------------------------------------

    def test_the_three_gates_are_reported_in_order(self) -> None:
        gates = self.app.gates_of(self.report(step_ms=10.0),
                                  self.report(precision="fp32", arch="sm_75",
                                              checkpoint="fp32",
                                              step_ms=40.0),
                                  self.roofline())
        self.assertEqual([gate["gate"] for gate in gates],
                         ["G1", "G2", "G3"])
        self.assertEqual([gate["status"] for gate in gates],
                         ["fail", "fail", "pass"])

    def test_no_gate_reads_the_mfu_or_the_speed_ratio(self) -> None:
        # P8: the two numbers are recorded and no claim rests on them.
        quiet = self.report(mfu=0.0)
        loud = self.report(mfu=99.0)
        quiet_gates = self.app.gates_of(quiet, quiet, self.roofline())
        loud_gates = self.app.gates_of(loud, loud, self.roofline())
        self.assertEqual([gate["status"] for gate in quiet_gates],
                         [gate["status"] for gate in loud_gates])
        for gate in loud_gates:
            self.assertNotIn("mfu", gate["read"])
            self.assertNotIn("speed_ratio", gate["read"])

    def test_the_comparison_records_what_no_gate_reads(self) -> None:
        legs = {"t4": self.app.leg_summary(self.app.T4, {
                    "report": self.report(precision="fp32", arch="sm_75",
                                          checkpoint="fp32", step_ms=40.0,
                                          mfu=0.001)}),
                "h100": self.app.leg_summary(self.app.H100, {
                    "report": self.report(step_ms=20.0, mfu=0.004)})}
        numbers = self.app.comparison(legs)
        self.assertEqual(numbers["base_step_ms"], {"t4": 40.0, "h100": 20.0})
        self.assertEqual(numbers["step_ms_ratio"], 0.5)
        self.assertEqual(numbers["speed_ratio"], 2.0)
        self.assertEqual(numbers["mfu"], {"t4": 0.001, "h100": 0.004})
        self.assertEqual(numbers["cpu"], self.app.CPU_CORES)


class DocumentTest(unittest.TestCase):
    """The one document the entry point prints, and the exit code it sets."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _load_app()

    def report(self, **kwargs) -> dict:
        return _report_document(**kwargs)

    def answer(self, *, step_ms=40.0, predicted=0.764, precision="bf16",
               arch="sm_90"):
        return {"leg": "h100",
                "report": self.report(step_ms=step_ms, precision=precision,
                                      arch=arch, checkpoint=precision),
                "roofline": {"predicted_step_ms": predicted,
                             "limit": "launch", "device": "h100",
                             "precision": "bf16"},
                "source": "repository", "run_dir": "/runs/x-h100/demo"}

    def answers(self, **kwargs):
        return {"t4": self.answer(step_ms=40.0, predicted=3.6,
                                  precision="fp32", arch="sm_75"),
                "h100": self.answer(**kwargs)}

    def test_the_document_has_the_three_keys(self) -> None:
        document = self.app.demo_document(self.answers())
        self.assertEqual(sorted(document), ["comparison", "gates", "legs"])
        self.assertEqual(sorted(document["legs"]), ["h100", "t4"])
        self.assertEqual([gate["gate"] for gate in document["gates"]],
                         ["G1", "G2", "G3"])
        self.assertEqual(self.app.gate_failures(document), [])

    def test_the_leg_summary_names_the_leg_and_its_report(self) -> None:
        summary = self.app.demo_document(self.answers())["legs"]["h100"]
        self.assertEqual((summary["leg"], summary["card"],
                          summary["precision"]), ("h100", "H100", "bf16"))
        self.assertEqual(summary["arch"], "sm_90")
        self.assertEqual(summary["host"]["precision"], "bf16")
        self.assertEqual(summary["base"]["step_ms"], 40.0)
        self.assertEqual(summary["rl"]["precision"], "bf16")
        self.assertEqual(summary["run_dir"], "/runs/x-h100/demo")

    def test_a_failed_leg_keeps_the_other_summary(self) -> None:
        answers = self.answers()
        answers["h100"] = {"leg": "h100", "error": "the chain exited 2"}
        document = self.app.demo_document(answers)
        self.assertEqual(document["legs"]["h100"]["error"],
                         "the chain exited 2")
        self.assertEqual(document["legs"]["t4"]["base"]["step_ms"], 40.0)
        self.assertEqual(self.app.gate_failures(document), ["G1", "G2", "G3"])

    def test_one_doctored_report_fails_one_gate(self) -> None:
        document = self.app.demo_document(self.answers(precision="fp32"))
        failed = [gate["gate"] for gate in document["gates"]
                  if gate["status"] == "fail"]
        self.assertEqual(failed, ["G1"])

    def test_the_t4_leg_runs_first_and_a_failure_keeps_the_other_run(self) -> None:
        seen = []

        def runner(leg, job_json, run_id):
            seen.append((leg.name, run_id))
            if leg is self.app.H100:
                raise RuntimeError("the H100 container failed")
            return self.answer(step_ms=40.0, predicted=3.6,
                               precision="fp32", arch="sm_75")

        answers = self.app.run_legs("{}", "run-1", runner=runner)
        self.assertEqual(seen, [("t4", "run-1-t4"), ("h100", "run-1-h100")])
        self.assertIn("report", answers["t4"])
        self.assertIn("the H100 container failed", answers["h100"]["error"])

    def test_the_entry_point_exits_non_zero_when_a_gate_fails(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(self.app, "run_legs",
                               lambda job_json, run_id: self.answers(
                                   step_ms=80.0, predicted=0.764)):
            with mock.patch.object(self.app, "demo_job",
                                   lambda run_id="demo": {"run_id": "demo"}):
                with contextlib.redirect_stdout(stdout), \
                        contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as caught:
                        self.app.pipeline_demo()
        self.assertNotEqual(caught.exception.code, 0)
        document = json.loads(stdout.getvalue())
        self.assertEqual(sorted(document), ["comparison", "gates", "legs"])
        self.assertIn("G2", stderr.getvalue())

    def test_the_entry_point_passes_when_every_gate_passes(self) -> None:
        stdout = io.StringIO()
        with mock.patch.object(self.app, "run_legs",
                               lambda job_json, run_id: self.answers()):
            with mock.patch.object(self.app, "demo_job",
                                   lambda run_id="demo": {"run_id": "demo"}):
                with contextlib.redirect_stdout(stdout):
                    self.app.pipeline_demo()
        document = json.loads(stdout.getvalue())
        self.assertEqual(self.app.gate_failures(document), [])


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
        self.assertEqual(sorted(mounted), ["pipeline", "pipeline_h100",
                                           "train", "train_h100"])
        for name in ("train", "train_h100"):
            self.assertEqual(sorted(mounted[name]), ["/runs", "/weights"])
            self.assertIs(mounted[name]["/weights"], self.app.weights)
            self.assertIs(mounted[name]["/runs"], self.app.runs)
        # The chain reads and writes the run root only: its stages warm start
        # from each other, so it never reads the stage cache of `train`.
        for name in ("pipeline", "pipeline_h100"):
            self.assertEqual(sorted(mounted[name]), ["/runs"])
            self.assertIs(mounted[name]["/runs"], self.app.runs)

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
        self.assertIn('"source": source', _source())
        self.assertIn('worker_env(source)', _source())


if __name__ == "__main__":
    unittest.main()
