"""The Modal app that runs the two legs of one nanochat.cpp chain.

The image holds the package and the compiled shared library; the weights
Volume holds the weights. That split is section 6.4 of
``docs/dependency-layering-plan.md`` and D8: a new checkpoint then costs a
download, not an image build.

- ``nanochat-runs`` at ``/runs`` is the run directory the notebook polls.
- ``nanochat-weights`` at ``/weights`` caches one stage per folder,
  ``<repo_id>/<stage>/``, keyed by the stage revision it holds in
  ``.revision``. ``fetch_stage`` downloads a revision it does not hold yet and
  returns the folder, so a warm start is a download and never a rebuild.

Two legs run here, and section 7 of ``docs/pipeline-scaleup-plan.md`` is the
table they follow. A leg is a frozen :class:`Leg` record: a name, a card type,
a card count, an architecture, a precision, and the flag triple the builder
emits. ``t4`` is a T4 at ``sm_75`` and fp32, and it runs first so its artifact
survives an H100 failure; ``h100`` is an H100 at ``sm_90`` and bf16.
:func:`image_for` returns one image for one leg, and one image builds one
library, so the architecture and the precision the environment declares are
the facts of the library that ran (P6). A second precision in one image would
need a second build of the same library, and the environment would name one of
the two.

Four functions run here. ``train`` and ``pipeline`` are the T4 leg and keep
their names, because ``notebooks/target.py`` resolves ``train`` through
``modal.Function.from_name``; ``train_h100`` and ``pipeline_h100`` are the H100
leg and mirror them. ``train`` runs one stage of a job; ``pipeline`` runs the
whole base, SFT, RL, and eval chain of one job file through
``python -m nanochat_cpp.pipeline``, which is the container entry of design
decision D10 (``docs/pipeline-plan.md`` section 6.3). One build per image
produces the library, the two stage executables the chain launches, and
``//src:roofline_main``, so the chain and gate G3 both find a binary already
built (sections 6.4 and B6).

Both legs keep the default collective, ``nccl``, because the chain runs
``world_size=1``, and no leg builds a simulator: the ``h100`` entry of
``.bazelrc`` selects the simulator and the native collective, so the hardware
leg passes the builder's explicit flag triple and never a named config
(section 3, trap 1, and section 7).

The image installs the distribution of ``python/pyproject.toml`` with the
``hub`` extra at build time, so site-packages answers ``import nanochat_cpp``
and ``import nanochat_cpp_bridge``. The repository tree stays baked in as the
fallback, because this campaign publishes no binary wheel: `tools/`,
`notebooks/`, and the Bazel build all need the source tree. ``package_source``
reports which of the two answered, and ``train`` and ``pipeline`` log it at
run start; the worker then runs from that one path, and
``NANOCHAT_CPP_SOURCE`` forces the tree when a run needs it. One commented
line below replaces the tree install with the ``nanochat_cpp_cuda_sm75`` wheel
once D6 publishes it.

Deploy and check it once, from a workstation or a notebook:

    pip install modal && modal setup
    modal deploy tools/modal/app.py
    modal run tools/modal/app.py::doctor
    modal run tools/modal/app.py::pipeline_demo

The last line runs both legs of one job, the T4 leg first, and prints one
document with three keys: ``legs`` (each leg's report summary), ``comparison``
(the cross-leg step-time ratio and the recorded MFU), and ``gates`` (G1 to G3
with pass or fail and the numbers each one read). It exits non-zero when a
gate fails, and a leg that fails still leaves the other leg's summary in the
document. Each leg writes its own run directory on the ``nanochat-runs``
Volume, at ``<run_id>-t4`` and ``<run_id>-h100``.

The client in ``notebooks/target.py`` uploads the job inputs and spawns
``train`` or ``pipeline``. The run directory arrives on the ``nanochat-runs``
Volume at ``/runs``, so the notebook polls it. A publish also runs here,
beside the checkpoint, so the checkpoint never crosses the notebook. See
``docs/host-portability.md``.

Create the Secret once, so a publish can reach the Hub:

    modal secret create huggingface HF_TOKEN=hf_...

The card type, the card count, and the architecture live in one record, the
:class:`Leg` of the leg the function deploys, because the image compiles for
exactly one architecture. :func:`_check_cards` reads that leg, so a T4
container checks ``sm_75`` at fp32 and an H100 container checks ``sm_90`` at
bf16 against the same job, and a mismatched deployment stops before the chain
starts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import modal

#: The app name. It must match ``notebooks/target.APP_NAME``.
APP_NAME = "nanochat"

#: The runs Volume and its mount point. The client uses the same names.
RUNS_VOLUME = "nanochat-runs"
REMOTE_ROOT = "/runs"

#: The weights Volume and its mount point (D8). It caches one stage per folder,
#: ``<repo_id>/<stage>/``, so a new step of a stage is a download.
WEIGHTS_VOLUME = "nanochat-weights"
WEIGHTS_ROOT = "/weights"

#: The file a cached stage folder carries: the revision it was fetched at.
REVISION_FILE = ".revision"

#: The two answers of :func:`package_source`, and the variable that forces one
#: of them, so a run can fall back to the tree without a new deployment.
SOURCES = ("installed", "repository")
SOURCE_ENV = "NANOCHAT_CPP_SOURCE"

#: The files a warm start reads from a stage. It mirrors ``STAGE_PATTERNS`` of
#: ``docs/checkpoint-format.md`` section 16; the rest is provenance a run does
#: not read.
STAGE_FILES = ("checkpoint.ckpt", "model.json", "tokenizer.nctoken")

#: The Secret that holds ``HF_TOKEN``, for a publish from the container.
SECRET_NAME = "huggingface"

#: The repository inside the image.
REPO_ROOT = "/repo"

#: The shared library that the image build produces.
LIBRARY = f"{REPO_ROOT}/bazel-bin/bindings/libnanochat_shared.so"

#: The two stage executables the chain launches, beside the library: the Bazel
#: labels the one image build names (``docs/pipeline-plan.md`` B6), so a
#: container needs no second build and the wheel line below stays a later
#: one-line change.
STAGE_NAMES = ("sft_main", "rl_worker")
STAGE_TARGETS = ("//src:sft_main", "//src:rl_worker")

#: The Bazel label of the roofline tool. Gate G3 runs
#: ``tools/nanochat roofline`` inside the container, and the tool must find its
#: binary already built there instead of starting a second build
#: (``docs/pipeline-scaleup-plan.md`` sections 7 and 8). It is not a stage
#: executable: no stage launches it, and only the gate reads its report.
ROOFLINE_TARGET = "//src:roofline_main"

#: The labels the one build of an image names: the shared library, the two
#: stage executables the chain launches (``docs/pipeline-plan.md`` B6 and D2),
#: and the roofline tool G3 measures with.
BUILD_TARGETS = ("//bindings:nanochat_shared", *STAGE_TARGETS,
                 ROOFLINE_TARGET)

#: The directory of the image build that holds them, and the variables that
#: name them. ``nanochat_cpp._build.stage_binary`` prefers the override, and
#: the image sets it because the installed distribution answers ``import
#: nanochat_cpp`` from site-packages, where ``repo_root()`` is not the image
#: tree (section 6.4). The names follow ``_build.stage_override``: the program
#: name upper-cased behind the prefix.
STAGE_BIN = f"{REPO_ROOT}/bazel-bin/src"
STAGE_ENV = {f"NANOCHAT_{name.upper()}": f"{STAGE_BIN}/{name}"
             for name in STAGE_NAMES}

#: The directory the chain delivers into, the report it writes, and the corpus
#: folder whose tokenizer a stage upload reads. The names mirror
#: ``nanochat_cpp.pipeline`` (``deliver()`` and ``CORPUS_DIRNAME``).
DELIVER_DIRNAME = "deliver"
REPORT_NAME = "pipeline_report.json"
CORPUS_DIRNAME = "corpus"
TOKENIZER_SUFFIX = ".nctoken"

#: The commit-seconds of one live volume commit. A poll of the client sees a
#: metric record only after the container commits the Volume.
COMMIT_SECONDS = 10

#: The longest run the deployment allows, in seconds.
TIMEOUT_SECONDS = 8 * 60 * 60

#: The CPU allocation of every leg, in cores. Modal sizes the CPU by memory
#: when ``cpu=`` is absent, so two legs with different memory would get
#: different host budgets; gate G2 compares step times across the legs, so both
#: legs take this one value (``docs/pipeline-scaleup-plan.md`` section 7).
CPU_CORES = 4.0

#: The gate constants of section 8. G2 is a smoke check: the H100 base step
#: time may be at most this many times the T4 base step time for the same job
#: and the same ``cpu=``. G3 holds the measured step time to a band around the
#: roofline prediction; the band starts wide and row 9 narrows it to twice the
#: measured ratio once a real H100 run records one.
G2_STEP_RATIO_LIMIT = 1.25
G3_RATIO_BAND = (1.0, 200.0)

#: The local repository this file is deployed from. `modal run` mounts the
#: entry module at `/root/app.py`, so `Path(__file__).parents[2]` raised
#: `IndexError` in the container and the app never started. Resolve the tree
#: instead of assuming where this file sits (`docs/host-portability.md` row 10).
def _repo_root() -> Path:
    """The tree this file belongs to, on the host and inside the image.

    Three sources, in order: `NANOCHAT_REPO`, the nearest ancestor that holds
    `MODULE.bazel` (the host layout, `tools/modal/app.py`), and the path the
    image copies the tree to. A layout none of them explain fails here with a
    message, not with an `IndexError` from `Path.parents`.
    """
    override = os.environ.get("NANOCHAT_REPO")
    if override:
        return Path(override).expanduser().resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "MODULE.bazel").is_file():
            return candidate
    image_tree = Path(REPO_ROOT)
    if image_tree.is_dir():
        return image_tree
    raise RuntimeError(
        f"cannot find the repository: no MODULE.bazel above {here} and "
        f"{REPO_ROOT} is not a directory; set NANOCHAT_REPO")


REPO = _repo_root()


def _capability(arch: str) -> str:
    """The compute capability of an architecture name, such as ``7.5``."""
    digits = arch.removeprefix("sm_")
    return f"{digits[0]}.{digits[1:]}"


def gpu_request(kind: str, count: int) -> str:
    """The Modal ``gpu=`` value: a type alone for one card, else type:count."""
    return kind if int(count) == 1 else f"{kind}:{int(count)}"


@dataclass(frozen=True)
class Leg:
    """One leg of the two-leg run: one card, one architecture, one precision.

    The record is frozen and it is the one home of the triple that used to be
    three module globals. ``card``, ``count``, and ``arch`` are a pair with the
    image, because the image compiles for exactly one architecture, and
    ``precision`` is the build precision the image sets in the environment, so
    the report's ``host.precision`` and ``host.arch`` are the facts of the
    library that ran (``docs/pipeline-scaleup-plan.md`` P6 and section 7).

    ``build`` is the flag triple the builder emits today
    (``nanochat_cpp._build._build_command``), written out here so a reader sees
    the flags without following the builder. The test contracts the two
    against each other, and the image build passes the leg's precision and
    architecture back through ``BuildOptions.resolve``, so the builder stays
    the one home of the command line.
    """

    name: str
    card: str
    count: int
    arch: str
    precision: str
    build: tuple[str, ...]

    @property
    def capability(self) -> str:
        """The compute capability of the leg's architecture, such as ``7.5``."""
        return _capability(self.arch)

    @property
    def gpu(self) -> str:
        """The Modal ``gpu=`` value of the leg."""
        return gpu_request(self.card, self.count)

    @property
    def device(self) -> str:
        """The roofline device name of the leg's card, such as ``h100``.

        The device limit table is keyed by the card in lower case
        (``backends/cuda/caps.json``), so ``T4`` reads ``t4`` and ``H100``
        reads ``h100``.
        """
        return self.card.lower()


#: The two legs of section 7, in the order the entry point runs them. The T4
#: leg is first, so its artifact survives an H100 failure and a reviewer reads
#: the cheap leg while the expensive one runs. Both legs run one card, keep the
#: default ``nccl`` collective, and add no simulator define and no fp8 build.
T4 = Leg(name="t4", card="T4", count=1, arch="sm_75", precision="fp32",
         build=("--config=cuda", "--define=precision=fp32",
                "--@rules_cuda//cuda:archs=sm_75"))
H100 = Leg(name="h100", card="H100", count=1, arch="sm_90", precision="bf16",
           build=("--config=cuda", "--define=precision=bf16",
                  "--@rules_cuda//cuda:archs=sm_90"))

#: The legs by name, and the order they run in.
LEGS = {T4.name: T4, H100.name: H100}
LEG_ORDER = (T4, H100)

#: The separator between the run id of a two-leg run and the leg name, so both
#: reports stay on the ``nanochat-runs`` Volume and a reviewer names one.
RUN_ID_SEPARATOR = "-"


def leg_run_id(run_id, leg) -> str:
    """The run id of one leg: ``<run_id>-t4`` or ``<run_id>-h100``."""
    return f"{run_id}{RUN_ID_SEPARATOR}{leg.name}"


app = modal.App(APP_NAME)

runs = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)
weights = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)

def image_env(leg: Leg) -> dict:
    """The environment of one leg's image.

    ``NANOCHAT_CPP_PRECISION`` and ``NANOCHAT_CUDA_ARCH`` are the leg's own two
    fields, so the library resolves its build precision and its architecture
    from the image, and the report's ``host`` block states the facts of that
    build (section 7). The stage executables are named where the one build put
    them, because the installed distribution resolves
    ``bazel-bin/src/<name>`` from its own site-packages location, which is not
    the image tree (section 6.4 and B6).
    """
    return {
        "CUDA_HOME": "/usr/local/cuda",
        "NANOCHAT_CPP_BACKEND": "cuda",
        "NANOCHAT_CPP_LIB": LIBRARY,
        **STAGE_ENV,
        "NANOCHAT_CPP_PRECISION": leg.precision,
        "NANOCHAT_CUDA_ARCH": leg.arch,
        "NANOCHAT_SANDBOX_BACKEND": "none",
        "NANOCHAT_TARGET": "local",
        "PYTHONPATH": f"{REPO_ROOT}/python:{REPO_ROOT}/notebooks",
    }


def image_build_command(leg: Leg, targets: Sequence[str] = BUILD_TARGETS,
                        root: str = REPO_ROOT) -> str:
    """The one build command of one leg's image.

    The command is a one-liner that calls ``_build._build_command``, so the
    builder stays the one home of the flag line and the image passes the leg's
    precision and architecture through ``BuildOptions.resolve`` (section 7).
    The targets are the library, the two stage executables the chain launches,
    and ``//src:roofline_main``, so gate G3 finds the tool already built inside
    the container.

    The command names no Bazel config. The ``h100`` entry of ``.bazelrc`` is
    the simulator, and it also sets the native collective, so the hardware leg
    takes the builder's explicit triple instead (section 3, trap 1).
    """
    labels = "(" + "".join(f"{label!r}, " for label in targets) + ")"
    return (
        f"cd {root} && python3 -c \""
        "import subprocess; from nanochat_cpp import _build; "
        "_options = _build.BuildOptions.resolve("
        f"backend='cuda', precision='{leg.precision}', arch='{leg.arch}'); "
        "subprocess.run(_build._build_command("
        f"'tools/nanochat', _options, {labels}), check=True)\"")


def _build_image(leg: Leg):
    """The image of one leg, before :func:`image_for` caches it."""
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
        .apt_install("build-essential", "curl", "git")
        .run_commands(
            "curl -fsSL -o /usr/local/bin/bazelisk "
            "https://github.com/bazelbuild/bazelisk/releases/download/"
            "v1.22.1/bazelisk-linux-amd64",
            "chmod 0755 /usr/local/bin/bazelisk",
        )
        .add_local_dir(str(REPO), remote_path=REPO_ROOT, copy=True,
                       ignore=[".git", "bazel-*", "**/__pycache__",
                               # The host file of `docs/host-portability.md`
                               # section 8.3 pins the compiler and the budget of
                               # the workstation, such as /usr/bin/gcc-12, which
                               # an Ubuntu 22.04 image does not have. A deploy
                               # from a host that ran `tools/kaggle/bootstrap.sh`
                               # would otherwise build an image that cannot
                               # compile (`docs/pipeline-plan.md` section 8
                               # row 10).
                               ".bazelrc.local"])
        # Install the package at image build (section 6.4). The tree install
        # builds the distribution of python/pyproject.toml in place, so both
        # import packages and the `hub` extra - huggingface_hub and
        # safetensors - reach site-packages. D6 publishes one binary wheel per
        # architecture; while it does not exist this line is the working path,
        # and one line replaces it when it does (sections 6.3 and Q5 pin one
        # revision):
        #   .pip_install("nanochat-cpp[hub] @ "
        #                "https://.../nanochat_cpp_cuda_sm75-0.1.0.whl")
        .pip_install(f"{REPO_ROOT}/python[hub]")
        .env(image_env(leg))
        # One build names the library, the two stage executables, and the
        # roofline tool, so the image satisfies D2, the chain finds both
        # children (B6), and gate G3 measures without a second build.
        .run_commands(image_build_command(leg))
        .workdir(REPO_ROOT)
    )


#: The images of the legs, keyed by leg name. One image per leg keeps the
#: architecture and the precision honest (P6); the cache keeps one deployment
#: from building the same image twice.
_IMAGES: dict = {}


def image_for(leg: Leg):
    """The image of one leg: the one home of its build and its environment.

    A caller never names the card, the architecture, or the precision here:
    the leg holds all three (section 7). A second precision in one image would
    need a second build of the same library, and the environment would name one
    of the two, so one image belongs to one leg.
    """
    built = _IMAGES.get(leg.name)
    if built is None:
        built = _IMAGES[leg.name] = _build_image(leg)
    return built


#: The probe behind :func:`package_source`. The repository root arrives as
#: ``argv[1]`` and leaves ``sys.path``, so the copy beside the install cannot
#: answer in its place.
_SOURCE_PROBE = (
    "import importlib.util, sys\n"
    "root = sys.argv[1]\n"
    "sys.path[:] = [p for p in sys.path if not p.startswith(root)]\n"
    "spec = importlib.util.find_spec('nanochat_cpp')\n"
    "print(spec.origin if spec and spec.origin else '')\n"
)


def package_source(python=None, root=None, run=None, environ=None) -> str:
    """``installed`` when the distribution answers, else ``repository``.

    The image installs the package at build time and keeps the repository
    tree on ``PYTHONPATH`` for ``tools/`` and ``notebooks/``, so the probe
    leaves the tree out of ``sys.path``: the answer is the installation, not
    the copy beside it. An absent install, a failed probe, and an empty answer
    all mean the repository path runs, which is the fallback of section 6.4.

    ``NANOCHAT_CPP_SOURCE`` forces one of the two, so a run can pick the tree
    without a new deployment.
    """
    environ = os.environ if environ is None else environ
    forced = str(environ.get(SOURCE_ENV, "")).strip().lower()
    if forced in SOURCES:
        return forced
    python = sys.executable if python is None else python
    root = REPO_ROOT if root is None else str(root)
    run = subprocess.run if run is None else run
    try:
        result = run([python, "-c", _SOURCE_PROBE, root],
                     capture_output=True, text=True, check=False)
    except OSError:
        return "repository"
    origin = str(getattr(result, "stdout", "") or "").strip()
    return "installed" if origin else "repository"


def worker_env(source, environ=None) -> dict:
    """The worker environment for one :func:`package_source` answer.

    ``notebooks/`` stays on the path either way, because the worker is a
    notebook module. Under ``installed`` the repository ``python/`` stays off
    the path, so the distribution answers the worker's import; under
    ``repository`` it comes first, exactly as before. The worker is a child
    process, so this is the order it inherits.
    """
    environ = os.environ if environ is None else environ
    env = dict(environ)
    roots = [f"{REPO_ROOT}/notebooks"]
    if source != "installed":
        roots.insert(0, f"{REPO_ROOT}/python")
    inherited = str(env.get("PYTHONPATH", ""))
    env["PYTHONPATH"] = os.pathsep.join(
        roots + ([inherited] if inherited else []))
    return env


def _device_capability() -> str:
    """The compute capability of the card, such as ``7.5``."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True, check=True)
    return result.stdout.splitlines()[0].strip()


def _device_count() -> int:
    """The number of cards the container sees."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True, text=True, check=True)
    return len([line for line in result.stdout.splitlines() if line.strip()])


def _check_cards(leg: Leg, job_json: str) -> None:
    """Stop unless the container matches the leg and the job.

    The leg names one card type, one card count, and one architecture, and the
    image compiled for exactly that architecture, so a T4 container checks
    ``sm_75`` at fp32 and an H100 container checks ``sm_90`` at bf16 against
    the same job, and a mismatched deployment stops before the chain starts
    (``docs/pipeline-scaleup-plan.md`` section 7).

    A train job declares one replica per device in ``mesh.count``. A pipeline
    job declares no mesh, because the chain runs one replica of each stage
    (``docs/pipeline-plan.md`` B11), so the count the leg declares is the
    answer. A mismatch fails here, before the upload, not inside the driver.
    """
    expected = leg.capability
    actual = _device_capability()
    if actual != expected:
        raise RuntimeError(
            f"the {leg.name} image is built for {leg.arch} ({expected}), but "
            f"the card reports {actual}; deploy an image for that "
            f"architecture")
    count = _device_count()
    if count != leg.count:
        raise RuntimeError(
            f"the {leg.name} function asks for {leg.count} card(s), but the "
            f"container has {count}; deploy the leg with its own count")
    mesh = json.loads(job_json).get("mesh") or {}
    replicas = int(mesh.get("count", count))
    if replicas != count:
        raise RuntimeError(
            f"the job asks for {replicas} replicas, but the container has "
            f"{count} card(s); run the chain at world_size {leg.count}")


def _commit_loop(volume, stop: threading.Event, seconds: float) -> None:
    """Commit the Volume until ``stop`` is set, so a poll sees new records."""
    while not stop.wait(seconds):
        volume.commit()


def _secrets() -> list:
    """The Secret that holds ``HF_TOKEN``. Absence blocks a publish only."""
    try:
        return [modal.Secret.from_name(SECRET_NAME)]
    except Exception:  # noqa: BLE001 - a run without a publish needs none
        print(f"modal: secret {SECRET_NAME} is absent", file=sys.stderr)
        return []


def stage_dir(repo_id, stage, root=WEIGHTS_ROOT) -> Path:
    """The weights Volume folder of one stage: ``<repo_id>/<stage>/``."""
    return Path(root) / str(repo_id) / str(stage)


def cached_revision(folder) -> str:
    """The revision a cached stage folder holds, or an empty string."""
    marker = Path(folder) / REVISION_FILE
    if not marker.is_file():
        return ""
    return marker.read_text(encoding="utf-8").strip()


def stage_pull():
    """The Hub stage downloader: the bridge package, else the notebook host.

    The installed distribution holds the call (``docs/checkpoint-format.md``
    section 16); the fallback is the notebook module on ``PYTHONPATH``, which
    forwards to the same place.
    """
    try:
        from nanochat_cpp_bridge.hf_hub import pull_stage  # noqa: PLC0415

        return pull_stage
    except Exception:  # noqa: BLE001 - a tree without the install falls back
        import hub_checkpoint  # noqa: PLC0415 - notebooks/ is on the path

        return hub_checkpoint.pull_stage


def fetch_stage(repo_id, stage, *, revision="main", token=None,
                patterns=STAGE_FILES, pull=None, commit=None,
                root=WEIGHTS_ROOT) -> Path:
    """Return the cached stage folder, downloading it when the revision moved.

    The Volume holds ``<repo_id>/<stage>/`` and, in ``.revision``, the revision
    fetched last. A matching revision and a checkpoint on disk cost nothing; a
    new step costs one download. Either way no image build is involved, which
    is the point of D8.
    """
    folder = stage_dir(repo_id, stage, root)
    if (cached_revision(folder) == str(revision)
            and (folder / "checkpoint.ckpt").is_file()):
        return folder
    pull = stage_pull() if pull is None else pull
    folder.parent.mkdir(parents=True, exist_ok=True)
    pulled = Path(pull(str(repo_id), str(stage), folder.parent,
                       revision=str(revision), token=token,
                       patterns=tuple(patterns)))
    marker = folder / REVISION_FILE
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{revision}\n", encoding="utf-8")
    if commit is None:
        commit = weights.commit
    commit()
    return pulled


def _warm_start(job: dict, stage_json: str, *, fetch=None,
                token=None) -> dict:
    """Point one job at the stage the weights Volume holds.

    ``stage_json`` is ``{"repo_id", "stage", "revision"}``; the fetch caches
    ``<repo_id>/<stage>/`` on the Volume, so the same revision costs one
    download and a new step costs another. An empty value keeps the job's own
    warm start, which is what the notebook client sends today.
    """
    if not stage_json:
        return job
    stage = json.loads(stage_json)
    fetch = fetch_stage if fetch is None else fetch
    folder = Path(fetch(str(stage["repo_id"]), str(stage["stage"]),
                        revision=str(stage.get("revision", "main")),
                        token=token))
    checkpoint = folder / "checkpoint.ckpt"
    if not checkpoint.is_file():
        raise RuntimeError(
            f"the stage {stage['stage']!r} of {stage['repo_id']!r} holds no "
            f"{checkpoint.name}; a warm start needs a checkpoint")
    started = dict(job)
    started["warm_start_path"] = str(checkpoint)
    print(f"modal: warm start from {checkpoint}", file=sys.stderr)
    return started


def _build_key(leg: Leg) -> str:
    """The build key of the library in this leg's image, or an empty string."""
    try:
        from nanochat_cpp import _build  # noqa: PLC0415 - image only

        options = _build.BuildOptions.resolve(
            backend="cuda", precision=leg.precision, arch=leg.arch)
        return _build.build_key(options=options)
    except Exception:  # noqa: BLE001 - the key is provenance, not a gate
        return ""


def _publish(job_json: str, run_dir: Path, publish_json: str,
             leg: Leg = T4) -> dict:
    """Write the stage folder beside the checkpoint and push it to the Hub.

    The client sends the metadata it alone can know: the model arguments, the
    job, the Git revision, and the stage name. The image supplies the build
    key and the architecture, because the container holds the library that
    trained the model, and both are the leg's own facts (section 7).
    """
    import hub_checkpoint  # noqa: PLC0415 - resolved inside the image

    meta = json.loads(publish_json)
    job = json.loads(job_json)
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            f"a publish needs HF_TOKEN: create the Modal Secret "
            f"{SECRET_NAME} with HF_TOKEN, then deploy again")
    stage = str(meta["stage"])
    tag = str(meta.get("tag", ""))
    record = hub_checkpoint.write_stage(
        run_dir, stage=stage, run_name=meta["run_name"],
        model_json=meta["model_json"], train_json=meta["train_json"],
        tokenizer=job["data"]["tokenizer_path"],
        git_revision=meta.get("revision", ""), build_key=_build_key(leg),
        cuda_arch=leg.arch, step=int(meta.get("step", 0)),
        created_utc=hub_checkpoint.now_utc(),
        description=meta.get("description", ""))
    hub_checkpoint.push_stage(meta["repo_id"], stage, run_dir, token=token,
                              tag=tag)
    return {"repo_id": meta["repo_id"], "stage": stage, "tag": tag,
            "record": record}


@app.function(image=image_for(T4), gpu=T4.gpu, cpu=CPU_CORES, timeout=600)
def doctor() -> str:
    """Report the device, the architecture, and the sandbox of the worker."""
    result = subprocess.run(["tools/nanochat", "doctor"],
                            capture_output=True, text=True, check=False)
    return (result.stdout + result.stderr).strip()


def _train_on(leg: Leg, job_json: str, run_id: str, publish_json: str = "",
              stage_json: str = "") -> str:
    """Run one job file on one leg and return the run result.

    ``train`` and ``train_h100`` are thin faces of this function, so the two
    legs cannot drift: the only difference between them is the :class:`Leg`
    they pass, which selects the image, the card, the precision, and the
    architecture of the deployment (section 7).

    ``max_containers=1`` keeps one job at a time, which matches the local GPU
    queue rule. A caller that wants a retry sets ``resume`` in the job.

    An empty ``publish_json`` stops after the run. A non-empty value writes the
    stage folder beside the checkpoint and pushes it to the Hub, so the
    checkpoint never leaves the Volume.

    ``stage_json`` names a stage to warm start from on the weights Volume. An
    empty value, which is what the notebook client sends today, keeps the warm
    start the job already carries.

    The return value is one JSON document with ``leg``, ``summary``,
    ``publish``, and ``source``: the section 6.4 answer of which nanochat.cpp
    the worker ran, and the leg it ran on.
    """
    source = package_source()
    print(f"modal: nanochat.cpp runs from the {source} path, on the "
          f"{leg.name} leg", file=sys.stderr)
    job = _warm_start(json.loads(job_json), stage_json,
                      token=os.environ.get("HF_TOKEN"))
    job_json = json.dumps(job, sort_keys=True)
    _check_cards(leg, job_json)
    run_dir = Path(REMOTE_ROOT) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    job_path = run_dir / "job.json"
    job_path.write_text(job_json + "\n", encoding="utf-8")

    stop = threading.Event()
    keeper = threading.Thread(target=_commit_loop,
                              args=(runs, stop, COMMIT_SECONDS), daemon=True)
    keeper.start()
    log_path = run_dir / "worker.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                ["tools/nanochat", "client", "--profile", "train", "--",
                 sys.executable, "notebooks/mesh_worker.py", str(job_path)],
                env=worker_env(source), stdout=log, stderr=subprocess.STDOUT,
                check=False)
    finally:
        stop.set()
        runs.commit()

    if result.returncode != 0:
        tail = "\n".join(
            log_path.read_text(encoding="utf-8").splitlines()[-20:])
        raise RuntimeError(f"the worker exited {result.returncode}:\n{tail}")
    summary_path = run_dir / "summary.json"
    summary = (json.loads(summary_path.read_text(encoding="utf-8"))
               if summary_path.is_file() else {})
    published = _publish(job_json, run_dir, publish_json, leg) if publish_json \
        else {}
    return json.dumps({"leg": leg.name, "summary": summary,
                       "publish": published, "source": source},
                      sort_keys=True)


@app.function(
    image=image_for(T4),
    gpu=T4.gpu,
    cpu=CPU_CORES,
    volumes={REMOTE_ROOT: runs, WEIGHTS_ROOT: weights},
    secrets=_secrets(),
    timeout=TIMEOUT_SECONDS,
    max_containers=1,
)
def train(job_json: str, run_id: str, publish_json: str = "",
          stage_json: str = "") -> str:
    """Run one job file on the T4 leg and return the run result.

    The name stays bare, and the leg is the T4 one, because
    ``notebooks/target.py`` resolves ``train`` through
    ``modal.Function.from_name`` and a call that names no leg must keep
    working (section 7). :func:`_train_on` is the body.
    """
    return _train_on(T4, job_json, run_id, publish_json, stage_json)


@app.function(
    image=image_for(H100),
    gpu=H100.gpu,
    cpu=CPU_CORES,
    volumes={REMOTE_ROOT: runs, WEIGHTS_ROOT: weights},
    secrets=_secrets(),
    timeout=TIMEOUT_SECONDS,
    max_containers=1,
)
def train_h100(job_json: str, run_id: str, publish_json: str = "",
               stage_json: str = "") -> str:
    """Run one job file on the H100 leg and return the run result.

    The mirror of :func:`train`: the same body, the same volumes, the same
    ``cpu=``, and the same run-directory layout, on the second image at bf16
    (section 7). ``notebooks/target.py`` reaches it as ``train_h100``.
    """
    return _train_on(H100, job_json, run_id, publish_json, stage_json)


def chain_dir(run_root, job) -> Path:
    """The directory the chain writes: ``<out_root>/<job run_id>``.

    ``nanochat_cpp.pipeline`` nests its own ``run_id`` under ``out_root``, so
    the app names the outer directory on the Volume and the job names the inner
    one. The report and the delivered checkpoint sit under it, in ``deliver/``.
    """
    return Path(run_root) / str((job or {}).get("run_id", ""))


def read_report(folder) -> dict:
    """The report the chain delivered, or raise when it delivered none.

    ``deliver()`` writes ``deliver/pipeline_report.json`` beside the delivered
    checkpoint (design decision D6). An absent report used to come back as an
    empty document, so a chain that exited 0 without delivering looked like a
    successful run with no data. The gate must fail instead: a run is reviewed
    by the report, and there is nothing to review.
    """
    path = Path(folder) / DELIVER_DIRNAME / REPORT_NAME
    if not path.is_file():
        raise RuntimeError(
            f"the chain wrote no {DELIVER_DIRNAME}/{REPORT_NAME} under {folder}, "
            f"so there is no report to return")
    return json.loads(path.read_text(encoding="utf-8"))


def stage_tokenizer(report, folder) -> str:
    """The tokenizer path the chain copied, for the stage uploader.

    ``_publish`` reads ``data.tokenizer_path`` from a train job, and a pipeline
    job holds no such field: the chain builds its corpus and copies the
    tokenizer beside it. The report names that path in the base stage's
    ``extra``, and the corpus folder is the fallback.
    """
    for stage in report.get("stages") or ():
        extra = (stage or {}).get("extra") or {}
        value = str(extra.get("tokenizer") or "")
        if value and Path(value).is_file():
            return value
    corpus = Path(folder) / CORPUS_DIRNAME
    found = sorted(corpus.glob(f"*{TOKENIZER_SUFFIX}")) \
        if corpus.is_dir() else []
    return str(found[0]) if found else ""


def publish_job_json(job, tokenizer) -> str:
    """The job document ``_publish`` reads, built from a pipeline job.

    The uploader reads ``data.tokenizer_path`` and nothing else of the job, so
    the app fills that one field from the chain's corpus and keeps the rest.
    """
    document = dict(job)
    data = dict(document.get("data") or {})
    data["tokenizer_path"] = str(tokenizer)
    document["data"] = data
    return json.dumps(document, sort_keys=True)


def shape_of(job: Mapping):
    """The model shape of one job, from the chain's own profile table.

    The shape has one home, ``nanochat_cpp.pipeline.PROFILES`` (P1), so the
    roofline command and the stage commands read the same numbers and neither
    keeps a copy of the shape. The import is inside the call because this file
    is also imported on a workstation that holds only the Modal SDK.
    """
    from nanochat_cpp import pipeline as chain  # noqa: PLC0415 - image only

    return chain.config(chain.Job.from_dict(dict(job)))


def model_flags(shape) -> list:
    """The model flags of one shape, from the one Python home of the list."""
    from nanochat_cpp import api  # noqa: PLC0415 - image only

    return list(api.model_flags(shape))


def base_batch() -> int:
    """The global batch of the base stage, which is the batch of one step."""
    from nanochat_cpp import pipeline as chain  # noqa: PLC0415 - image only

    return int(chain.BASE_BATCH)


def roofline_argv(leg: Leg, job: Mapping) -> list:
    """The ``tools/nanochat roofline`` command gate G3 runs.

    The device is the leg's card and the precision is the leg's precision, so
    the prediction is the bound of the hardware that ran: ``--device h100
    --precision bf16`` for the H100 leg (section 8). The model flags come from
    ``nanochat_cpp.api.model_flags``, the one Python home of the model flag
    list, and the batch is the chain's own base batch, so the shape the tool
    models is the shape the stages ran.
    """
    return ["tools/nanochat", "roofline",
            "--device", leg.device,
            "--precision", leg.precision,
            "--batch", str(base_batch()),
            *model_flags(shape_of(job)),
            "--json"]


def predicted_step_ms(document: Mapping) -> tuple:
    """The step bound and the binding limit of one roofline JSON document.

    ``step_seconds`` is the sum of the rows plus the gradient sync, and the
    limit of the row with the largest bound is the limit the step time reads.
    Every row of a report at this shape answers ``launch``, so the chain is
    bound by launches and by the host program, not by arithmetic (section 2 of
    ``docs/pipeline-scaleup-plan.md``). The step bound answers ``None`` when
    the document carries no positive ``step_seconds``.
    """
    seconds = document.get("step_seconds")
    rows = [row for row in document.get("rows") or ()
            if isinstance(row, Mapping)]
    worst = max(rows,
                key=lambda row: float(row.get("bound_seconds") or 0.0),
                default={})
    limit = str((worst or {}).get("limit") or "")
    if not isinstance(seconds, (int, float)) or not float(seconds) > 0.0:
        return None, limit
    return 1000.0 * float(seconds), limit


def measure_roofline(leg: Leg, job: Mapping, run=None) -> dict:
    """Run the roofline tool in this container and record its prediction.

    The record is what gate G3 reads: the command, the leg's device and
    precision, the ``predicted_step_ms`` step bound, the binding ``limit``, and
    the ratio band of the gate (section 8). A tool that fails records
    ``error`` and no prediction, and the chain still answers with its report: a
    measurement that did not run is a failed gate and never a lost run.
    """
    argv = roofline_argv(leg, job)
    run = subprocess.run if run is None else run
    record = {"command": argv, "device": leg.device,
              "precision": leg.precision, "band": list(G3_RATIO_BAND)}
    try:
        result = run(argv, capture_output=True, text=True, check=False)
    except OSError as error:
        return dict(record, error=f"the tool did not start: {error}")
    code = int(getattr(result, "returncode", 1))
    if code != 0:
        lines = str(getattr(result, "stderr", "") or "").strip().splitlines()
        return dict(record,
                    error=f"the tool exited {code}: {' | '.join(lines[-3:])}")
    try:
        document = json.loads(str(getattr(result, "stdout", "") or ""))
    except ValueError as error:
        return dict(record, error=f"the tool wrote no JSON: {error}")
    predicted, limit = predicted_step_ms(document)
    if predicted is None:
        return dict(record, limit=limit,
                    error="the tool wrote no positive step_seconds")
    return dict(record, schema=str(document.get("schema") or ""),
                limit=limit, predicted_step_ms=predicted,
                rows=len(document.get("rows") or ()))


def _pipeline_on(leg: Leg, job_json: str, run_id: str, stage_json: str = "",
                 publish_json: str = "") -> str:
    """Run the whole chain of one job file on one leg and return its report.

    ``pipeline`` and ``pipeline_h100`` are thin faces of this function, so the
    two legs cannot drift: the :class:`Leg` they pass selects the image, the
    card, the precision, the architecture, and the roofline device (section 7).

    One job file runs the stages of ``docs/pipeline-plan.md`` section 1, and
    the container entry is the one design decision D10 names:
    ``tools/nanochat client --profile train -- python -m
    nanochat_cpp.pipeline <job.json>``. The function mirrors ``train``:
    ``max_containers=1`` keeps one chain at a time, ``_check_cards`` stops a
    container that does not match the leg, the commit keeper makes the run
    directory visible to a poll, and ``worker.log`` holds the chain's output.

    The app rewrites ``out_root`` to ``/runs/<run_id>``, because the app owns
    the Volume path and one job runs several stages, so no stage owns the
    directory. The chain nests the job's own ``run_id`` under it.

    An empty ``publish_json`` stops after the chain, which has already
    delivered ``deliver/``. A non-empty value reuses :func:`_publish` for the
    Hub push of the final stage, so the chain gains no second uploader. The
    metadata is the same document a train publish sends; the tokenizer comes
    from the corpus the chain built.

    ``stage_json`` is accepted and ignored: the signature carries it so one
    client call shape serves every function, and the chain warm starts each
    stage from the checkpoint of the stage before it, so a Modal stage cache
    has nothing to point at. A non-empty value is noted on the log.

    The return value is one JSON document with ``leg``, ``report``,
    ``roofline``, ``publish``, ``source``, and ``run_dir``: the section 6.4
    answer of which nanochat.cpp the worker ran, the directory the chain wrote
    on the Volume, and the prediction gate G3 compares with the run.
    """
    source = package_source()
    print(f"modal: nanochat.cpp runs from the {source} path, on the "
          f"{leg.name} leg", file=sys.stderr)
    if stage_json:
        print("modal: a pipeline call takes no stage_json; the chain warm "
              "starts each stage from the one before it", file=sys.stderr)
    job = json.loads(job_json)
    job["out_root"] = str(Path(REMOTE_ROOT) / str(run_id))
    job_json = json.dumps(job, sort_keys=True)
    _check_cards(leg, job_json)
    run_dir = Path(REMOTE_ROOT) / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    job_path = run_dir / "job.json"
    job_path.write_text(job_json + "\n", encoding="utf-8")

    stop = threading.Event()
    keeper = threading.Thread(target=_commit_loop,
                              args=(runs, stop, COMMIT_SECONDS), daemon=True)
    keeper.start()
    log_path = run_dir / "worker.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                ["tools/nanochat", "client", "--profile", "train", "--",
                 sys.executable, "-m", "nanochat_cpp.pipeline",
                 str(job_path)],
                env=worker_env(source), stdout=log, stderr=subprocess.STDOUT,
                check=False)
    finally:
        stop.set()
        runs.commit()

    if result.returncode != 0:
        tail = "\n".join(
            log_path.read_text(encoding="utf-8").splitlines()[-20:])
        raise RuntimeError(f"the chain exited {result.returncode}:\n{tail}")
    folder = chain_dir(run_dir, job)
    # A missing report raises here, before the publish. A zero exit with no
    # delivery is a failed gate, not an empty success.
    report = read_report(folder)
    # The prediction is measured after the chain, so a tool that fails costs a
    # gate and never the report the chain already delivered.
    roofline = measure_roofline(leg, job)
    published = {}
    if publish_json:
        published = _publish(
            publish_job_json(job, stage_tokenizer(report, folder)),
            folder / "rl", publish_json, leg)
    return json.dumps({"leg": leg.name, "report": report,
                       "roofline": roofline, "publish": published,
                       "source": source, "run_dir": str(folder)},
                      sort_keys=True)


@app.function(
    image=image_for(T4),
    gpu=T4.gpu,
    cpu=CPU_CORES,
    volumes={REMOTE_ROOT: runs},
    secrets=_secrets(),
    timeout=TIMEOUT_SECONDS,
    max_containers=1,
)
def pipeline(job_json: str, run_id: str, stage_json: str = "",
             publish_json: str = "") -> str:
    """Run the whole chain on the T4 leg and return its report.

    The name stays bare, and the leg is the T4 one, because
    ``notebooks/target.py`` resolves it through ``modal.Function.from_name``
    (section 7). :func:`_pipeline_on` is the body.
    """
    return _pipeline_on(T4, job_json, run_id, stage_json, publish_json)


@app.function(
    image=image_for(H100),
    gpu=H100.gpu,
    cpu=CPU_CORES,
    volumes={REMOTE_ROOT: runs},
    secrets=_secrets(),
    timeout=TIMEOUT_SECONDS,
    max_containers=1,
)
def pipeline_h100(job_json: str, run_id: str, stage_json: str = "",
                  publish_json: str = "") -> str:
    """Run the whole chain on the H100 leg and return its report.

    The mirror of :func:`pipeline` at bf16 on ``sm_90``, with the same
    ``cpu=``, the same volumes, and the same ``out_root`` rewrite, so gate G2
    compares the two legs under the same host budget (section 7). The answer
    also carries the roofline prediction gate G3 reads, measured in this
    container with ``tools/nanochat roofline --device h100 --precision bf16``
    (section 8).
    """
    return _pipeline_on(H100, job_json, run_id, stage_json, publish_json)


# ---------------------------------------------------------------------------
# The two-leg document: the summaries, the comparison, and the gates
# ---------------------------------------------------------------------------


def _number(value, default=None):
    """``value`` as a float when it is a number a gate can read."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def stage_facts(report, name: str) -> dict:
    """The record of the stage ``name`` in one report, or an empty document."""
    for stage in (report or {}).get("stages") or ():
        if isinstance(stage, Mapping) and str(stage.get("name")) == name:
            return dict(stage)
    return {}


def base_facts(report) -> dict:
    """The base stage's rates and hardware facts, as numbers a gate can read.

    ``step_ms`` comes from the base stage's ``extra`` block, which row 4 of the
    plan fills from the metrics the driver wrote, so nothing here estimates it.
    A report without a base stage answers ``None`` for every number, and a gate
    that reads one fails with a reason instead of comparing a guess.
    """
    extra = dict(stage_facts(report, "base").get("extra") or {})
    return {"step_ms": _number(extra.get("step_ms")),
            "tokens_per_sec": _number(extra.get("tokens_per_sec")),
            "mfu": _number(extra.get("mfu")),
            "hardware": dict(extra.get("hardware") or {})}


def leg_summary(leg: Leg, answer=None, error: str = "") -> dict:
    """The report summary of one leg, for the document's ``legs`` key.

    The summary names the leg and then quotes what the report and the answer
    say: the ``host`` block, the base stage's rates, the RL checkpoint's stage
    and precision, what was delivered, and the leg's own roofline record. A leg
    whose run failed carries ``error`` instead of numbers, and the other leg's
    summary still prints (section 7).
    """
    answer = dict(answer or {})
    report = dict(answer.get("report") or {})
    host = dict(report.get("host") or {})
    rl = stage_facts(report, "rl")
    metadata = dict(rl.get("metadata") or {})
    summary = {
        "leg": leg.name,
        "card": leg.card,
        "arch": leg.arch,
        "precision": leg.precision,
        "run_dir": str(answer.get("run_dir") or ""),
        "source": str(answer.get("source") or ""),
        "host": {"backend": str(host.get("backend") or ""),
                 "arch": str(host.get("arch") or ""),
                 "precision": str(host.get("precision") or ""),
                 "revision": str(host.get("revision") or "")},
        "base": base_facts(report),
        "rl": {"steps": rl.get("steps"),
               "checkpoint": str(rl.get("checkpoint") or ""),
               "sha256": str(rl.get("sha256") or ""),
               "stage": str(metadata.get("stage") or ""),
               "precision": str(metadata.get("precision") or "")},
        "total_seconds": _number(report.get("total_seconds")),
        "delivered": str(report.get("delivered") or ""),
        "report_path": str(report.get("report_path") or ""),
        "publish": dict(answer.get("publish") or {}),
        "roofline": dict(answer.get("roofline") or {}),
    }
    if error:
        summary["error"] = str(error)
    return summary


def comparison(legs: Mapping) -> dict:
    """The cross-leg numbers: the step-time ratio, the MFU, and the speed ratio.

    ``step_ms_ratio`` is the H100 base step time over the T4 one, which is the
    quantity gate G2 bounds. ``speed_ratio`` is its inverse: how many times
    faster the H100 leg ran the same job. ``mfu`` is each leg's recorded value.
    No gate reads the MFU or the speed ratio (P8): at 256 tokens per step the
    device bound is about one millisecond inside a step of tens of
    milliseconds, so MFU reads near 0.01 percent, and the chain is bound by
    launches, so a speed ratio below one is as correct as one above it. Both
    numbers are recorded, and no claim rests on them (section 8).
    """
    def facts(name):
        return ((legs or {}).get(name) or {}).get("base") or {}

    t4_ms = _number(facts(T4.name).get("step_ms"))
    h100_ms = _number(facts(H100.name).get("step_ms"))
    ratio = h100_ms / t4_ms if (t4_ms and h100_ms is not None) else None
    speed = t4_ms / h100_ms if (h100_ms and t4_ms is not None) else None
    return {
        "cpu": CPU_CORES,
        "base_step_ms": {T4.name: t4_ms, H100.name: h100_ms},
        "step_ms_ratio": ratio,
        "speed_ratio": speed,
        "mfu": {T4.name: _number(facts(T4.name).get("mfu")),
                H100.name: _number(facts(H100.name).get("mfu"))},
    }


#: The claims of the three gates, word for word from section 8.
G1_CLAIM = "bf16 ran on the H100"
G2_CLAIM = "the H100 leg did not regress"
G3_CLAIM = "the run matches the model"


def _gate(name: str, claim: str, ok: bool, read: Mapping, expects: Mapping,
          reason: str = "") -> dict:
    """One gate record: pass or fail, with the numbers it read."""
    return {"gate": name, "claim": claim,
            "status": "pass" if ok else "fail",
            "read": dict(read), "expects": dict(expects),
            "reason": "" if ok else str(reason)}


def gate_bf16_ran(h100_report, leg: Leg = H100) -> dict:
    """G1: the H100 leg ran bf16, in the report and in the checkpoint.

    Three facts, all of them read: ``host.precision`` is ``bf16``,
    ``host.arch`` is ``sm_90``, and the RL checkpoint metadata records
    ``bf16``. The checkpoint is the file the chain delivered, so the third fact
    says the artifact was written by the bf16 build rather than only that a
    bf16 build ran something (section 8).
    """
    host = dict((h100_report or {}).get("host") or {})
    metadata = dict(stage_facts(h100_report, "rl").get("metadata") or {})
    read = {"host_precision": str(host.get("precision") or ""),
            "host_arch": str(host.get("arch") or ""),
            "checkpoint_precision": str(metadata.get("precision") or "")}
    expects = {"host_precision": leg.precision,
               "host_arch": leg.arch,
               "checkpoint_precision": leg.precision}
    wrong = [key for key in expects if read[key] != expects[key]]
    reason = ""
    if wrong:
        reason = "the H100 report does not say " + "; ".join(
            f"{key} is {expects[key]!r}" for key in wrong)
    return _gate("G1", G1_CLAIM, not wrong, read, expects, reason)


def gate_step_ratio(t4_report, h100_report,
                    limit: float = G2_STEP_RATIO_LIMIT) -> dict:
    """G2: the H100 base step time is at most ``limit`` times the T4 one.

    A smoke check, not a throughput claim: both legs are bound by the host
    program, so the gate catches a large regression and a broken device path
    and it does not measure the card (section 8). The two functions took the
    same ``cpu=`` value, so the host budget of the two numbers is the same. A
    leg that reports no base step time fails the gate with a reason, so a chain
    that recorded no rates is not a pass.
    """
    t4_ms = base_facts(t4_report)["step_ms"]
    h100_ms = base_facts(h100_report)["step_ms"]
    read = {"t4_base_step_ms": t4_ms, "h100_base_step_ms": h100_ms,
            "cpu": CPU_CORES}
    expects = {"step_ms_ratio_at_most": float(limit),
               "same_cpu_on_both_legs": True}
    if t4_ms is None or h100_ms is None or not t4_ms > 0.0:
        return _gate("G2", G2_CLAIM, False, read, expects,
                     "a leg reports no positive base step_ms, so the ratio is "
                     "unknown")
    ratio = h100_ms / t4_ms
    read["step_ms_ratio"] = ratio
    ok = ratio <= float(limit)
    reason = "" if ok else (
        f"the H100 base step is {ratio:.3f} times the T4 base step, above "
        f"{float(limit):.2f}")
    return _gate("G2", G2_CLAIM, ok, read, expects, reason)


def gate_roofline(h100_report, roofline=None,
                  band: Sequence[float] = G3_RATIO_BAND) -> dict:
    """G3: the measured H100 step time is inside the band of the prediction.

    The H100 container ran ``tools/nanochat roofline --device h100 --precision
    bf16 --json`` for the job shape and recorded ``predicted_step_ms`` and the
    binding ``limit``; this gate divides the measured base step time by the
    prediction. The band starts at ``[1, 200]`` and row 9 narrows it to twice
    the ratio the first two-leg run measures (section 8). A record with no
    prediction, or with an error, fails the gate.
    """
    roofline = dict(roofline or {})
    predicted = _number(roofline.get("predicted_step_ms"))
    measured = base_facts(h100_report)["step_ms"]
    read = {"predicted_step_ms": predicted,
            "measured_step_ms": measured,
            "limit": str(roofline.get("limit") or ""),
            "band": [float(band[0]), float(band[1])]}
    expects = {"ratio_at_least": float(band[0]),
               "ratio_at_most": float(band[1])}
    if predicted is None or not predicted > 0.0:
        error = str(roofline.get("error") or "")
        return _gate("G3", G3_CLAIM, False, read, expects,
                     "the container recorded no roofline prediction"
                     + (f": {error}" if error else ""))
    if measured is None:
        return _gate("G3", G3_CLAIM, False, read, expects,
                     "the report carries no base step_ms to compare")
    ratio = measured / predicted
    read["ratio"] = ratio
    ok = float(band[0]) <= ratio <= float(band[1])
    reason = "" if ok else (
        f"the measured step is {ratio:.3f} times the prediction, outside "
        f"{float(band[0]):.0f} to {float(band[1]):.0f}")
    return _gate("G3", G3_CLAIM, ok, read, expects, reason)


def gates_of(t4_report, h100_report, roofline=None) -> list:
    """The three gates of section 8, in order: G1, G2, and G3."""
    return [gate_bf16_ran(h100_report),
            gate_step_ratio(t4_report, h100_report),
            gate_roofline(h100_report, roofline)]


def gate_failures(document: Mapping) -> list:
    """The names of the gates that did not pass in one printed document."""
    return [str(gate.get("gate"))
            for gate in (document or {}).get("gates") or ()
            if str(gate.get("status")) != "pass"]


def demo_document(answers: Mapping) -> dict:
    """The one document the entry point prints: legs, comparison, and gates.

    A leg whose answer carries ``error`` still contributes its summary, so a
    failed H100 leg leaves the T4 report readable. The gates read the two
    reports and the H100 leg's roofline record, and a missing one is a failure
    with a reason rather than a crash, so the document always has its three
    keys (section 7).
    """
    legs = {}
    for leg in LEG_ORDER:
        answer = dict((answers or {}).get(leg.name) or {})
        legs[leg.name] = leg_summary(leg, answer,
                                     error=str(answer.get("error") or ""))
    t4_report = dict(((answers or {}).get(T4.name) or {}).get("report") or {})
    h100_answer = dict((answers or {}).get(H100.name) or {})
    h100_report = dict(h100_answer.get("report") or {})
    return {"legs": legs,
            "comparison": comparison(legs),
            "gates": gates_of(t4_report, h100_report,
                              h100_answer.get("roofline"))}


def pipeline_function(leg: Leg):
    """The deployed chain function of one leg."""
    return {T4.name: pipeline, H100.name: pipeline_h100}[leg.name]


def run_leg(leg: Leg, job_json: str, run_id: str) -> dict:
    """Run one leg's chain on Modal and return its parsed answer document."""
    return json.loads(pipeline_function(leg).remote(job_json, str(run_id)))


def run_legs(job_json: str, run_id: str, runner=None) -> dict:
    """Run both legs, T4 first, and return one answer per leg name.

    The order is the promise of section 7: the T4 artifact is on the Volume
    before the H100 container starts, so a reviewer reads the cheap leg while
    the expensive one runs, and an H100 failure leaves the T4 report intact.
    The run ids are ``<run_id>-t4`` and ``<run_id>-h100``, so both reports stay
    on the ``nanochat-runs`` Volume. A leg that raises answers with ``error``
    and the other leg still runs, so one container's failure is a line in the
    document and not a lost run.
    """
    runner = run_leg if runner is None else runner
    answers = {}
    for leg in LEG_ORDER:
        try:
            answers[leg.name] = runner(leg, job_json, leg_run_id(run_id, leg))
        except Exception as error:  # noqa: BLE001 - a leg's failure is a report
            print(f"modal: the {leg.name} leg failed: {error}", file=sys.stderr)
            answers[leg.name] = {"leg": leg.name, "error": str(error)}
    return answers


#: The run id, and the job, of :func:`pipeline_demo`. The copy is the fallback
#: of :func:`demo_job` on a workstation that installed the Modal SDK and not
#: the package, so the entry point runs from anywhere. It mirrors the defaults
#: of ``nanochat_cpp.pipeline.Job``: the ``standard`` profile, the longer
#: schedule, and the eval stage (P1 and section 5).
DEMO_RUN_ID = "pipeline-demo"
DEMO_JOB = {
    "run_id": "demo",
    "profile": "standard",
    "base_steps": 200,
    "sft_steps": 200,
    "rl_steps": 8,
    "num_prompts": 4,
    "num_samples": 4,
    "max_tokens": 32,
    "eval_batches": 1,
    "eval_samples": 2,
    "seed": 42,
    "out_root": "",
    "corpus": "fixture",
    "deliver": True,
    "temperature": 1.5,
    "top_k": 0,
}


def demo_job(run_id: str = "demo") -> dict:
    """The demo job: ``nanochat_cpp.pipeline.Job``'s defaults when they answer.

    The package owns the job shape (design decision D5), so a host that holds
    it reads the defaults from there; :data:`DEMO_JOB` is the fallback of a
    workstation that installed only the Modal SDK. Either way ``out_root``
    stays empty, because the function rewrites it onto the Volume.
    """
    try:
        from nanochat_cpp.pipeline import Job  # noqa: PLC0415 - optional
    except Exception:  # noqa: BLE001 - a host without the package falls back
        return dict(DEMO_JOB, run_id=str(run_id))
    return dict(Job(run_id=str(run_id)).to_dict())


@app.local_entrypoint()
def pipeline_demo(job: str = "", run_id: str = DEMO_RUN_ID) -> None:
    """Run both legs and print the one document: legs, comparison, and gates.

        modal run tools/modal/app.py::pipeline_demo
        modal run tools/modal/app.py::pipeline_demo --job my-job.json

    ``job`` names a job file on this host; without one, the demo job runs. The
    T4 leg runs first and writes ``<run_id>-t4/<job run_id>/deliver/`` on the
    ``nanochat-runs`` Volume, then the H100 leg writes ``<run_id>-h100/...``, so
    both reports stay and a reviewer reads one without waiting for the other.
    The document holds ``legs``, ``comparison``, and ``gates``, and the command
    exits non-zero when any gate fails (section 7 and section 8).
    """
    payload = (json.loads(Path(job).read_text(encoding="utf-8")) if job
               else demo_job())
    if not payload.get("run_id"):
        payload["run_id"] = dict(DEMO_JOB)["run_id"]
    answers = run_legs(json.dumps(payload, sort_keys=True), str(run_id))
    document = demo_document(answers)
    print(json.dumps(document, indent=1, sort_keys=True))
    failed = gate_failures(document)
    if failed:
        print(f"modal: the run failed {', '.join(failed)}", file=sys.stderr)
        raise SystemExit(1)
