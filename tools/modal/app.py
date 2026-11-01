"""The Modal app that runs one nanochat.cpp job, and the chain of one job file.

The image holds the package and the compiled shared library; the weights
Volume holds the weights. That split is section 6.4 of
``docs/dependency-layering-plan.md`` and D8: a new checkpoint then costs a
download, not an image build.

- ``nanochat-runs`` at ``/runs`` is the run directory the notebook polls.
- ``nanochat-weights`` at ``/weights`` caches one stage per folder,
  ``<repo_id>/<stage>/``, keyed by the stage revision it holds in
  ``.revision``. ``fetch_stage`` downloads a revision it does not hold yet and
  returns the folder, so a warm start is a download and never a rebuild.

Two functions run here. ``train`` runs one stage of a job; ``pipeline`` runs
the whole base, SFT, and RL chain of one job file through
``python -m nanochat_cpp.pipeline``, which is the container entry of
design decision D10 (``docs/pipeline-plan.md`` section 6.3). One image holds
both, and one build produces the library and the two stage executables the
chain launches, so the CPU and the CUDA leg of that chapter share the build
line and differ only in its configuration (sections 6.4 and B6).

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

The last line runs one demo chain in the image and prints the report the chain
delivers, with ``host.arch`` of ``sm_75`` and ``deliver/`` on the
``nanochat-runs`` Volume.

The client in ``notebooks/target.py`` uploads the job inputs and spawns
``train`` or ``pipeline``. The run directory arrives on the ``nanochat-runs``
Volume at ``/runs``, so the notebook polls it. A publish also runs here,
beside the checkpoint, so the checkpoint never crosses the notebook. See
``docs/host-portability.md``.

Create the Secret once, so a publish can reach the Hub:

    modal secret create huggingface HF_TOKEN=hf_...

``GPU_TYPE``, ``GPU_COUNT``, and ``CUDA_ARCH`` form one triple. The type and
the architecture are a pair, because the image compiles for exactly that
architecture. Both functions check all three against the container and the
job.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

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

#: The card type, the card count, and the compiled architecture.
#: ``GPU_TYPE`` and ``CUDA_ARCH`` are one pair. The image compiles for exactly
#: that architecture, so a different type needs a new deployment.
GPU_TYPE = "T4"
GPU_COUNT = 1
CUDA_ARCH = "sm_75"

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


def gpu_request(kind=GPU_TYPE, count=GPU_COUNT) -> str:
    """The Modal ``gpu=`` value: a type alone for one card, else type:count."""
    return kind if int(count) == 1 else f"{kind}:{int(count)}"


app = modal.App(APP_NAME)

runs = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)
weights = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)

image = (
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
                           # an Ubuntu 22.04 image does not have. A deploy from
                           # a host that ran `tools/kaggle/bootstrap.sh` would
                           # otherwise build an image that cannot compile
                           # (`docs/pipeline-plan.md` section 8 row 10).
                           ".bazelrc.local"])
    # Install the package at image build (section 6.4). The tree install builds
    # the distribution of python/pyproject.toml in place, so both import
    # packages and the `hub` extra — huggingface_hub and safetensors — reach
    # site-packages. D6 publishes one binary wheel per architecture; while it
    # does not exist this line is the working path, and one line replaces it
    # when it does (sections 6.3 and Q5 pin one revision):
    #   .pip_install("nanochat-cpp[hub] @ "
    #                "https://.../nanochat_cpp_cuda_sm75-0.1.0.whl")
    .pip_install(f"{REPO_ROOT}/python[hub]")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "NANOCHAT_CPP_BACKEND": "cuda",
        "NANOCHAT_CPP_LIB": LIBRARY,
        # The chain launches both stage executables, so the image names them
        # where they were built (B6). The installed distribution resolves
        # `bazel-bin/src/<name>` from its own site-packages location, which is
        # not the image tree, so the override is the one path that holds.
        **STAGE_ENV,
        "NANOCHAT_CPP_PRECISION": "fp32",
        "NANOCHAT_CUDA_ARCH": CUDA_ARCH,
        "NANOCHAT_SANDBOX_BACKEND": "none",
        "NANOCHAT_TARGET": "local",
        "PYTHONPATH": f"{REPO_ROOT}/python:{REPO_ROOT}/notebooks",
    })
    # One build names the library and the two stage executables, so the image
    # satisfies D2 and the chain finds both children (B6). The builder
    # `_build.build_stages` is the one home of that command line, so the Colab
    # setup and this image ask for the same three targets.
    .run_commands(
        f"cd {REPO_ROOT} && python3 -c "
        "\"from nanochat_cpp import _build; "
        "_build.build_stages(options=_build.BuildOptions.resolve("
        f"backend='cuda', precision='fp32', arch='{CUDA_ARCH}'))\"",
    )
    .workdir(REPO_ROOT)
)


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


def _capability(arch: str) -> str:
    """The compute capability of an architecture name, such as ``7.5``."""
    digits = arch.removeprefix("sm_")
    return f"{digits[0]}.{digits[1:]}"


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


def _check_cards(job_json: str) -> None:
    """Stop unless the container matches the function and the job.

    The function declares one type and one count. A train job declares one
    replica per device in ``mesh.count``. A pipeline job declares no mesh,
    because the chain runs one replica of each stage (``docs/pipeline-plan.md``
    B11), so the count the function declares is the answer. A mismatch fails
    here, before the upload, not inside the driver.
    """
    expected = _capability(CUDA_ARCH)
    actual = _device_capability()
    if actual != expected:
        raise RuntimeError(
            f"the image is built for {CUDA_ARCH} ({expected}), but the card "
            f"reports {actual}; deploy an image for that architecture")
    count = _device_count()
    if count != GPU_COUNT:
        raise RuntimeError(
            f"the function asks for {GPU_COUNT} card(s), but the container "
            f"has {count}; set GPU_COUNT to the value the job uses")
    mesh = json.loads(job_json).get("mesh") or {}
    replicas = int(mesh.get("count", count))
    if replicas != count:
        raise RuntimeError(
            f"the job asks for {replicas} replicas, but the container has "
            f"{count} card(s); set world_size to {GPU_COUNT}")


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


def _build_key() -> str:
    """The build key of the library in this image, or an empty string."""
    try:
        from nanochat_cpp import _build  # noqa: PLC0415 - image only

        options = _build.BuildOptions.resolve(
            backend="cuda", precision="fp32", arch=CUDA_ARCH)
        return _build.build_key(options=options)
    except Exception:  # noqa: BLE001 - the key is provenance, not a gate
        return ""


def _publish(job_json: str, run_dir: Path, publish_json: str) -> dict:
    """Write the stage folder beside the checkpoint and push it to the Hub.

    The client sends the metadata it alone can know: the model arguments, the
    job, the Git revision, and the stage name. The image supplies the build
    key and the architecture, because the container holds the library that
    trained the model.
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
        git_revision=meta.get("revision", ""), build_key=_build_key(),
        cuda_arch=CUDA_ARCH, step=int(meta.get("step", 0)),
        created_utc=hub_checkpoint.now_utc(),
        description=meta.get("description", ""))
    hub_checkpoint.push_stage(meta["repo_id"], stage, run_dir, token=token,
                              tag=tag)
    return {"repo_id": meta["repo_id"], "stage": stage, "tag": tag,
            "record": record}


@app.function(image=image, gpu=gpu_request(), timeout=600)
def doctor() -> str:
    """Report the device, the architecture, and the sandbox of the worker."""
    result = subprocess.run(["tools/nanochat", "doctor"],
                            capture_output=True, text=True, check=False)
    return (result.stdout + result.stderr).strip()


@app.function(
    image=image,
    gpu=gpu_request(),
    volumes={REMOTE_ROOT: runs, WEIGHTS_ROOT: weights},
    secrets=_secrets(),
    timeout=TIMEOUT_SECONDS,
    max_containers=1,
)
def train(job_json: str, run_id: str, publish_json: str = "",
          stage_json: str = "") -> str:
    """Run one job file on the runs Volume and return the run result.

    ``max_containers=1`` keeps one job at a time, which matches the local GPU
    queue rule. A caller that wants a retry sets ``resume`` in the job.

    An empty ``publish_json`` stops after the run. A non-empty value writes the
    stage folder beside the checkpoint and pushes it to the Hub, so the
    checkpoint never leaves the Volume.

    ``stage_json`` names a stage to warm start from on the weights Volume. An
    empty value, which is what the notebook client sends today, keeps the warm
    start the job already carries.

    The return value is one JSON document with ``summary``, ``publish``, and
    ``source``: the section 6.4 answer of which nanochat.cpp the worker ran.
    """
    source = package_source()
    print(f"modal: nanochat.cpp runs from the {source} path", file=sys.stderr)
    job = _warm_start(json.loads(job_json), stage_json,
                      token=os.environ.get("HF_TOKEN"))
    job_json = json.dumps(job, sort_keys=True)
    _check_cards(job_json)
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
    published = _publish(job_json, run_dir, publish_json) if publish_json \
        else {}
    return json.dumps({"summary": summary, "publish": published,
                       "source": source}, sort_keys=True)


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


@app.function(
    image=image,
    gpu=gpu_request(),
    volumes={REMOTE_ROOT: runs},
    secrets=_secrets(),
    timeout=TIMEOUT_SECONDS,
    max_containers=1,
)
def pipeline(job_json: str, run_id: str, stage_json: str = "",
             publish_json: str = "") -> str:
    """Run the whole base, SFT, and RL chain and return its report.

    One job file runs three stages (``docs/pipeline-plan.md`` section 1), and
    the container entry is the one design decision D10 names:
    ``tools/nanochat client --profile train -- python -m
    nanochat_cpp.pipeline <job.json>``. The function mirrors ``train``:
    ``max_containers=1`` keeps one chain at a time, ``_check_cards`` stops a
    container that does not match the image, the commit keeper makes the run
    directory visible to a poll, and ``worker.log`` holds the chain's output.

    The app rewrites ``out_root`` to ``/runs/<run_id>``, because the app owns
    the Volume path and one job runs three stages, so no stage owns the
    directory. The chain nests the job's own ``run_id`` under it.

    An empty ``publish_json`` stops after the chain, which has already
    delivered ``deliver/``. A non-empty value reuses :func:`_publish` for the
    Hub push of the final stage, so the chain gains no second uploader. The
    metadata is the same document a train publish sends; the tokenizer comes
    from the corpus the chain built.

    ``stage_json`` is accepted and ignored: the signature carries it so one
    client call shape serves both functions, and the chain warm starts each
    stage from the checkpoint of the stage before it, so a Modal stage cache
    has nothing to point at. A non-empty value is noted on the log.

    The return value is one JSON document with ``report``, ``publish``,
    ``source``, and ``run_dir``: the section 6.4 answer of which nanochat.cpp
    the worker ran, and the directory the chain wrote on the Volume.
    """
    source = package_source()
    print(f"modal: nanochat.cpp runs from the {source} path", file=sys.stderr)
    if stage_json:
        print("modal: a pipeline call takes no stage_json; the chain warm "
              "starts each stage from the one before it", file=sys.stderr)
    job = json.loads(job_json)
    job["out_root"] = str(Path(REMOTE_ROOT) / str(run_id))
    job_json = json.dumps(job, sort_keys=True)
    _check_cards(job_json)
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
    published = {}
    if publish_json:
        published = _publish(
            publish_job_json(job, stage_tokenizer(report, folder)),
            folder / "rl", publish_json)
    return json.dumps({"report": report, "publish": published,
                       "source": source, "run_dir": str(folder)},
                      sort_keys=True)


#: The run id, and the job, of :func:`pipeline_demo`: the shape of the ``demo``
#: profile that ``docs/pipeline-plan.md`` section 7 gates. The copy is the
#: fallback of :func:`demo_job` on a workstation that installed the Modal SDK
#: and not the package, so the entry point runs from anywhere.
DEMO_RUN_ID = "pipeline-demo"
DEMO_JOB = {
    "run_id": "demo",
    "profile": "demo",
    "base_steps": 100,
    "sft_steps": 100,
    "rl_steps": 5,
    "num_prompts": 4,
    "num_samples": 4,
    "max_tokens": 32,
    "seed": 42,
    "out_root": "",
    "corpus": "fixture",
    "deliver": True,
    "temperature": 1.0,
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
    """Run one chain in the image and print the report it delivered.

        modal run tools/modal/app.py::pipeline_demo
        modal run tools/modal/app.py::pipeline_demo --job my-job.json

    ``job`` names a job file on this host; without one, the demo job runs. The
    chain writes ``<run_id>/<job run_id>/deliver/`` on the ``nanochat-runs``
    Volume, and the printed document is the report of that folder, so a run is
    checked without a notebook (``docs/pipeline-plan.md`` section 6.3).
    """
    payload = (json.loads(Path(job).read_text(encoding="utf-8")) if job
               else demo_job())
    if not payload.get("run_id"):
        payload["run_id"] = dict(DEMO_JOB)["run_id"]
    result = json.loads(pipeline.remote(
        json.dumps(payload, sort_keys=True), str(run_id)))
    print(json.dumps(result, indent=1, sort_keys=True))
