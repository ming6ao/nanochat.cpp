"""The run host: where one training job runs, and how its records come back.

A notebook is the user surface. The run host is where the job executes. Today
there are two hosts: ``local``, the notebook container, and ``modal``, a remote
function with a Volume. A new host adds one class and one registry row.

The host boundary carries data, not objects:

- a job dictionary and a path map, because ``MeshTrainConfig`` needs the shared
  library and a remote host has none;
- the two run records, ``metrics.jsonl`` and ``summary.json``, parsed here so a
  local run and a remote run report the same rows.

The Modal SDK is imported inside :func:`client`, so this module loads, and its
tests run, without the SDK and without an account. See
``docs/host-portability.md``.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import time
from pathlib import Path

__all__ = [
    "APP_NAME",
    "DEFAULT",
    "FUNCTION",
    "INPUT_FIELDS",
    "JobPlan",
    "LocalTarget",
    "ModalLiveRun",
    "ModalTarget",
    "NOTEBOOK_BASE",
    "OUTPUT_FIELDS",
    "PATH_FIELDS",
    "REMOTE_ROOT",
    "RUN_FILES",
    "RUNS_VOLUME",
    "SECRET_NAME",
    "Target",
    "job_paths",
    "map_job",
    "names",
    "parse_metrics",
    "parse_summary",
    "resolve",
]

#: The run host a caller gets when it names none.
DEFAULT = "local"

#: The staging root of a notebook host, when the environment names none.
NOTEBOOK_BASE = Path("/kaggle/working/nanochat-cache")

#: The job fields a run reads. They must reach the host before it starts.
INPUT_FIELDS = (
    ("data", "parquet"),
    ("data", "packed_rows_path"),
    ("data", "tokenizer_path"),
    ("warm_start_path",),
)

#: The job fields a run writes. They must arrive back on the client.
OUTPUT_FIELDS = (
    ("output", "checkpoint_path"),
    ("output", "out_dir"),
)

#: Every job field that holds a path. ``data.parquet`` holds a list of paths.
PATH_FIELDS = INPUT_FIELDS + OUTPUT_FIELDS

#: The Modal app, its function, the runs Volume it mounts, and the Secret that
#: holds ``HF_TOKEN`` for a publish inside the container.
APP_NAME = "nanochat"
FUNCTION = "train"
RUNS_VOLUME = "nanochat-runs"
REMOTE_ROOT = "/runs"
SECRET_NAME = "huggingface"

#: The files a run writes. A client pulls the ones it needs back.
RUN_FILES = (
    "checkpoint.ckpt",
    "summary.json",
    "metrics.jsonl",
    "worker.log",
    "job.json",
)


def _field_value(job, field):
    """The value of one dotted ``field`` in ``job``, or ``None``."""
    container = job
    for key in field[:-1]:
        container = container.get(key) if isinstance(container, dict) else None
        if container is None:
            return None
    if not isinstance(container, dict):
        return None
    return container.get(field[-1])


def _to_host(path, local_root, host_root) -> str:
    """Map one path from the local root to the host root.

    An empty value stays empty, because it means "not set". A path outside
    ``local_root`` raises ``ValueError``, so a host never reads a file it
    cannot have.
    """
    text = str(path)
    try:
        relative = Path(text).relative_to(local_root)
    except ValueError as error:
        raise ValueError(
            f"{text} is outside the run root {local_root}; "
            "copy it under the run root first") from error
    return str(Path(host_root) / relative)


def map_job(job, local_root=None, host_root=None) -> dict:
    """Return a copy of ``job`` with every path field mapped.

    Without both roots the copy keeps every path, which is what a local run
    needs. An absent field and an empty value are left alone.
    """
    copied = copy.deepcopy(dict(job))
    if local_root is None or host_root is None:
        return copied
    for field in PATH_FIELDS:
        value = _field_value(copied, field)
        if not value:
            continue
        container = copied
        for key in field[:-1]:
            container = container.setdefault(key, {})
        if isinstance(value, (list, tuple)):
            container[field[-1]] = [
                _to_host(item, local_root, host_root) for item in value]
        else:
            container[field[-1]] = _to_host(value, local_root, host_root)
    return copied


def job_paths(job, fields=PATH_FIELDS) -> list:
    """The non-empty path values of ``fields`` in ``job``, in field order."""
    found = []
    for field in fields:
        value = _field_value(job, field)
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            if item:
                found.append(str(item))
    return found


def parse_metrics(text: str) -> list[dict]:
    """Parse the per-step JSON lines. A bad line is skipped."""
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def parse_summary(text: str) -> dict:
    """Parse one run summary. A bad or empty text returns an empty dict."""
    if not text.strip():
        return {}
    try:
        return json.loads(text)
    except ValueError:
        return {}


@dataclasses.dataclass(frozen=True)
class JobPlan:
    """One job, ready for one run host.

    ``job_json`` is the one-line job document. ``out_dir`` is the run
    directory on the host. ``job_path`` is the local file the client writes.
    """

    job_json: str
    out_dir: str
    job_path: Path


class Target:
    """One run host.

    ``needs_local_devices`` tells the session whether to check a card count.
    ``cache_root`` is the staging root for the data. ``run_root`` is the local
    root that maps to the host root. ``prepare`` builds the plan of one job.
    """

    #: The name a caller passes as ``target``.
    name = ""

    #: True when the host runs the job on a local card.
    needs_local_devices = True

    def cache_root(self) -> Path:
        """The staging root: ``NANOCHAT_BASE_DIR``, else the notebook root."""
        value = os.environ.get("NANOCHAT_BASE_DIR")
        return Path(value) if value else NOTEBOOK_BASE

    def run_root(self) -> Path:
        """The local root that maps to the host run root.

        ``NANOCHAT_RUNS_DIR`` when set. Otherwise the parent of the staging
        root, which is the notebook working directory beside the cache.
        """
        value = os.environ.get("NANOCHAT_RUNS_DIR")
        if value:
            return Path(value).expanduser()
        return self.cache_root().parent

    def prepare(self, config, stage_root=None) -> JobPlan:
        """Build the plan of one job. The local host keeps every path."""
        del stage_root  # One process reads and writes the same files.
        out_dir = Path(config.out_dir)
        return JobPlan(job_json=config.job_json(), out_dir=str(out_dir),
                       job_path=out_dir / "job.json")


class LocalTarget(Target):
    """The notebook container runs the job in a child process.

    :func:`mesh_train.launch` runs this host through its own launcher, so this
    class holds no ``run`` method.
    """

    name = DEFAULT


class ModalTarget(Target):
    """A Modal function runs the job on a remote card.

    The path map is built per run, because the remote root is a fixed mount
    point and the local root depends on the notebook.
    """

    name = "modal"
    needs_local_devices = False

    def __init__(self, *, app_name=APP_NAME, function=FUNCTION,
                 volume=RUNS_VOLUME, remote_root=REMOTE_ROOT) -> None:
        self.app_name = app_name
        self.function = function
        self.volume = volume
        self.remote_root = str(remote_root)

    def local_root(self, stage_root=None) -> Path:
        """The local run root: the caller value, else :meth:`run_root`."""
        return Path(stage_root) if stage_root is not None \
            else self.run_root()

    def prepare(self, config, stage_root=None) -> JobPlan:
        """Build the plan of one job, with every path on the host root.

        A run needs an ``out_dir``, because the run directory is the Volume
        path. An empty value raises ``RuntimeError``.
        """
        job = map_job(json.loads(config.job_json()),
                      self.local_root(stage_root), Path(self.remote_root))
        out_dir = _field_value(job, ("output", "out_dir"))
        if not out_dir:
            raise RuntimeError(
                "a modal run needs an out_dir on the config: the run "
                "directory is the path on the runs Volume")
        return JobPlan(job_json=json.dumps(job, sort_keys=True),
                       out_dir=str(out_dir),
                       job_path=Path(config.out_dir) / "job.json")

    def relative_dir(self, plan) -> str:
        """The run directory inside the Volume, such as ``runs/base``."""
        return Path(plan.out_dir).relative_to(self.remote_root).as_posix()

    def uploads(self, config, stage_root=None) -> list:
        """The ``(local, remote)`` file pairs the run reads.

        The pairs come from the input fields of the job, so an output file is
        never uploaded. A run that resumes reads the checkpoint from the
        Volume, and it needs no upload.
        """
        local_root = self.local_root(stage_root)
        job = json.loads(config.job_json())
        return [(Path(local),
                 _to_host(local, local_root, self.remote_root))
                for local in job_paths(job, INPUT_FIELDS)]

    def reader(self, volume):
        """A reader of one Volume file, by Volume-relative path.

        The reader returns ``None`` for an absent file. A poll before the first
        metric then returns an empty list, and the cell does not fail.
        """

        def read(relative):
            try:
                handle = volume.read_file(relative)
            except FileNotFoundError:
                return None
            data = handle.read() if hasattr(handle, "read") else handle
            return data if isinstance(data, bytes) else str(data).encode()

        return read

    def submit(self, plan, uploads, *, publish=None,
               timeout=None) -> "ModalLiveRun":
        """Upload the inputs, spawn the function, and return the handle.

        ``publish`` is a dictionary of stage metadata, or ``None``. A publish
        runs inside the container, beside the checkpoint.
        """
        sdk = client()
        volume = sdk.Volume.from_name(self.volume, create_if_missing=True)
        if uploads:
            with volume.batch_upload() as batch:
                for local, remote in uploads:
                    batch.put_file(str(local), remote)
        function = sdk.Function.from_name(self.app_name, self.function)
        publish_json = json.dumps(publish, sort_keys=True) if publish else ""
        call = function.spawn(plan.job_json, self.relative_dir(plan),
                              publish_json)
        return ModalLiveRun(call, remote_dir=self.relative_dir(plan),
                            reader=self.reader(volume), timeout=timeout)

    def run(self, config, *, stage_root=None, timeout=None,
            extra_env=None, publish=None) -> "ModalLiveRun":
        """Upload, spawn, and return the live handle of one job.

        ``publish`` is the stage metadata, or ``None``. A publish runs inside
        the container, so the checkpoint is read from the Volume and pushed
        from there.

        ``extra_env`` is not used. The container environment comes from the
        image, so a local value must not leak into the remote job.
        """
        del extra_env  # The image owns the container environment.
        plan = self.prepare(config, stage_root)
        uploads = self.uploads(config, stage_root)
        plan.job_path.parent.mkdir(parents=True, exist_ok=True)
        plan.job_path.write_text(plan.job_json + "\n", encoding="utf-8")
        return self.submit(plan, uploads, publish=publish, timeout=timeout)


class ModalLiveRun:
    """A job that runs in a Modal function.

    The notebook polls the run directory through the Volume, so the cell never
    blocks on the function. ``rows``, ``summary``, and ``tail`` return empty
    values until the container writes the file and commits the Volume.
    """

    def __init__(self, call, *, remote_dir, reader, timeout=None) -> None:
        self.call = call
        self.remote_dir = str(remote_dir)
        self._reader = reader
        self._timeout = timeout

    def _read(self, name: str) -> str:
        """The text of one file in the run directory, or an empty string."""
        data = self._reader(f"{self.remote_dir}/{name}")
        if data is None:
            return ""
        if isinstance(data, bytes):
            return data.decode("utf-8", "replace")
        return str(data)

    def running(self) -> bool:
        """True while the function runs."""
        try:
            self.call.get(timeout=0)
        except TimeoutError:
            return True
        except Exception:  # noqa: BLE001 - a failed call is not running
            return False
        return False

    def result(self, timeout=None):
        """Block until the function returns and return its value."""
        return self.call.get(timeout=self._timeout if timeout is None
                             else timeout)

    def rows(self) -> list[dict]:
        """The per-step metrics written so far."""
        return parse_metrics(self._read("metrics.jsonl"))

    def summary(self) -> dict:
        """The run summary, or an empty dict before the run finishes."""
        return parse_summary(self._read("summary.json"))

    def tail(self, lines: int = 15) -> str:
        """The last ``lines`` lines of the worker log. Missing returns ""."""
        text = self._read("worker.log")
        return "\n".join(text.splitlines()[-lines:])

    def pull(self, dest, names=RUN_FILES) -> list:
        """Copy the run files from the Volume into ``dest``.

        A file the run has not written yet is skipped. The call returns the
        local paths it wrote, so a caller can publish the stage from them.
        """
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        written = []
        for name in names:
            data = self._reader(f"{self.remote_dir}/{name}")
            if data is None:
                continue
            path = dest / name
            path.write_bytes(data if isinstance(data, bytes)
                             else str(data).encode())
            written.append(path)
        return written

    def stream(self, interval: float = 5.0) -> None:
        """Print the worker log as it grows, until the function returns."""
        position = 0
        try:
            while self.running():
                text = self._read("worker.log")
                if len(text) > position:
                    print(text[position:], end="", flush=True)
                    position = len(text)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\ninterrupt received; cancelling the run", flush=True)
            self.terminate()

    def wait(self, timeout: float | None = None) -> int:
        """Block until the function returns and return 0."""
        self.result(timeout)
        return 0

    def terminate(self, timeout: float = 30.0) -> None:
        """Cancel the function. The driver saves nothing on a cancel.

        A cancel stops the container without an interrupt path, so the last
        checkpoint is the last committed one. Use the resume flag of the job
        to continue from it.
        """
        del timeout  # The SDK cancel takes no timeout.
        try:
            self.call.cancel()
        except Exception:  # noqa: BLE001 - a finished call cannot be cancelled
            pass


def client():
    """The Modal SDK module, or a ``RuntimeError`` when it is absent."""
    try:
        import modal  # noqa: PLC0415 - the SDK is optional
    except ImportError as error:
        raise RuntimeError(
            "the modal run host needs the Modal client: "
            "pip install modal, then modal setup") from error
    return modal


#: The run hosts. A new host adds one row.
_HOSTS = {LocalTarget.name: LocalTarget, ModalTarget.name: ModalTarget}


def names() -> tuple:
    """The run host names, in a stable order."""
    return tuple(sorted(_HOSTS))


def resolve(value=None, environ=None) -> Target:
    """The target of ``value``, else of ``NANOCHAT_TARGET``, else ``local``.

    ``value`` may be a name or a :class:`Target` instance. An unknown name
    raises ``ValueError``, because a typo must not quietly select a host that
    the caller did not name.
    """
    if isinstance(value, Target):
        return value
    environ = os.environ if environ is None else environ
    name = value if value is not None else environ.get("NANOCHAT_TARGET", "")
    name = str(name or DEFAULT).strip().lower()
    if name not in _HOSTS:
        raise ValueError(f"unknown run target {name!r}; "
                         f"known: {', '.join(names())}")
    return _HOSTS[name]()
