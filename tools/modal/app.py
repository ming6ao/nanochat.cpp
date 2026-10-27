"""The Modal app that runs one nanochat.cpp training job.

The image bakes the repository and the compiled shared library at image build
time. A run therefore pays no Bazel cost, which is the point of this host.

Deploy and check it once, from a workstation or a notebook:

    pip install modal && modal setup
    modal deploy tools/modal/app.py
    modal run tools/modal/app.py::doctor

The client in ``notebooks/target.py`` uploads the job inputs and spawns
``train``. The run directory arrives on the ``nanochat-runs`` Volume at
``/runs``, so the notebook polls it. A publish also runs here, beside the
checkpoint, so the checkpoint never crosses the notebook. See
``docs/host-portability.md``.

Create the Secret once, so a publish can reach the Hub:

    modal secret create huggingface HF_TOKEN=hf_...

``GPU_TYPE``, ``GPU_COUNT``, and ``CUDA_ARCH`` form one triple. The type and
the architecture are a pair, because the image compiles for exactly that
architecture. ``train`` checks all three against the container and the job.
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

#: The commit-seconds of one live volume commit. A poll of the client sees a
#: metric record only after the container commits the Volume.
COMMIT_SECONDS = 10

#: The longest run the deployment allows, in seconds.
TIMEOUT_SECONDS = 8 * 60 * 60

#: The local repository this file is deployed from.
REPO = Path(__file__).resolve().parents[2]


def gpu_request(kind=GPU_TYPE, count=GPU_COUNT) -> str:
    """The Modal ``gpu=`` value: a type alone for one card, else type:count."""
    return kind if int(count) == 1 else f"{kind}:{int(count)}"


app = modal.App(APP_NAME)

runs = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)

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
                   ignore=[".git", "bazel-*", "**/__pycache__"])
    .pip_install("huggingface_hub")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "NANOCHAT_CPP_BACKEND": "cuda",
        "NANOCHAT_CPP_LIB": LIBRARY,
        "NANOCHAT_CPP_PRECISION": "fp32",
        "NANOCHAT_CUDA_ARCH": CUDA_ARCH,
        "NANOCHAT_SANDBOX_BACKEND": "none",
        "NANOCHAT_TARGET": "local",
        "PYTHONPATH": f"{REPO_ROOT}/python:{REPO_ROOT}/notebooks",
    })
    .run_commands(
        f"cd {REPO_ROOT} && python3 -c "
        "\"from nanochat_cpp import _build; "
        "_build.build(options=_build.BuildOptions.resolve("
        f"backend='cuda', precision='fp32', arch='{CUDA_ARCH}'))\"",
    )
    .workdir(REPO_ROOT)
)


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

    The function declares one type and one count. The job declares one
    replica per device in ``mesh.count``. A mismatch fails here, before the
    upload, not inside the driver.
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
    replicas = int(json.loads(job_json)["mesh"]["count"])
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
    volumes={REMOTE_ROOT: runs},
    secrets=_secrets(),
    timeout=TIMEOUT_SECONDS,
    max_containers=1,
)
def train(job_json: str, run_id: str, publish_json: str = "") -> str:
    """Run one job file on the runs Volume and return the run result.

    ``max_containers=1`` keeps one job at a time, which matches the local GPU
    queue rule. A caller that wants a retry sets ``resume`` in the job.

    An empty ``publish_json`` stops after the run. A non-empty value writes the
    stage folder beside the checkpoint and pushes it to the Hub, so the
    checkpoint never leaves the Volume. The return value is one JSON document
    with ``summary`` and ``publish``.
    """
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
                stdout=log, stderr=subprocess.STDOUT, check=False)
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
    return json.dumps({"summary": summary, "publish": published},
                      sort_keys=True)
