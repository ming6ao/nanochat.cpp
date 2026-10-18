"""The ``tools/nanochat`` command line.

``tools/nanochat`` is a thin shim that starts this module. The command table,
the argument assembly, the Bazel wiring, the build lock, and the sandbox and
queue routing live here. The shell keeps the resource launcher
(``tools/sandbox.sh``). See ``docs/script-consolidation-plan.md``.

The observable contract is frozen (decision 4 of the plan): the environment
variable names, the two log paths and their line formats, the exit codes, the
two ``--run_under`` strings, the ``NANOCHAT_SANDBOX`` sentinel, the
``doctor --json`` keys, the fixture subcommand names, and the fixture binary
layouts. ``python/tests/contract_test.py`` pins them.
"""

from __future__ import annotations

import contextlib
import fcntl
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# `tools/nanochat` runs this file directly, so `python/` is not on the path.
# A `-m nanochat_cpp.cli` run already has it.
_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from nanochat_cpp import _entry, gpu_queue, prune  # noqa: E402

__all__ = ["main"]

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
TOOLS = REPO_ROOT / "tools"
ENTRY = TOOLS / "nanochat"
SANDBOX = TOOLS / "sandbox.sh"
BAZEL_BIN = REPO_ROOT / "bazel-bin"

BUILD_LOCK = os.environ.get("NANOCHAT_BUILD_LOCK", "/tmp/nanochat-build.lock")

#: The commands that `--dry-run` covers.
DRY_RUN_COMMANDS = ("build", "test", "check", "run", "simulate")

#: The `doctor --json` keys, in output order.
DOCTOR_KEYS = ("repo", "revision", "branch", "nvcc", "bazel", "gpu",
               "gpu_procs", "sandbox", "sandbox_backend", "gpu_queue", "sim")

#: The sandbox profiles that `run` sends to the queue instead of the launcher.
DIRECT_PROFILE = "t0-cpu"


class Failure(Exception):
    """A command-line error. ``main`` prints it and returns 2."""


def die(message: str) -> None:
    """Raise the command-line error that ``main`` reports."""
    raise Failure(message)


def in_sandbox() -> bool:
    """True inside a sandbox scope or a queued job."""
    return bool(os.environ.get("NANOCHAT_SANDBOX") or
                os.environ.get("NANOCHAT_GPU_QUEUED"))


def sandbox_backend() -> str:
    """The configured backend name, or ``auto`` when unset."""
    return os.environ.get("NANOCHAT_SANDBOX_BACKEND", "auto")


def find_bazel() -> str:
    """The Bazel launcher. Prefer Bazelisk so `.bazelversion` is honored."""
    for name in ("bazelisk", "bazel"):
        path = shutil.which(name)
        if path:
            return path
    die("neither bazelisk nor bazel found on PATH")
    raise AssertionError("unreachable")


def have_nccl() -> bool:
    """True when the host has the NCCL headers. See third_party/nccl.bzl."""
    prefixes = []
    override = os.environ.get("NANOCHAT_NCCL_PATH")
    if override:
        prefixes.append(override)
    prefixes += ["/usr", "/usr/local", "/opt/nccl"]
    return any((Path(prefix) / "include" / "nccl.h").is_file()
               for prefix in prefixes)


def with_build_lock(argv: list[str], stdout=None, stderr=None) -> int:
    """Run ``argv`` under the build lock.

    The lock lives on a descriptor that Python opens close-on-exec, so the
    detached Bazel server cannot inherit it. That inheritance deadlocked the
    old GPU broker.
    """
    with open(BUILD_LOCK, "a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return subprocess.run(argv, check=False, stdout=stdout,
                              stderr=stderr).returncode


def _run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run ``argv`` and never raise on a nonzero status."""
    return subprocess.run(argv, check=False, **kwargs)


def _capture(argv: list[str]) -> str:
    """The trimmed stdout of ``argv``, or ``""`` when it fails."""
    result = _run(argv, capture_output=True, text=True)
    return result.stdout.strip()


def exec_replace(argv: list[str]) -> None:
    """Replace this process with ``argv``."""
    os.execvp(argv[0], argv)


class Context:
    """The shared state of one command run."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run

    def plan(self, layer: str, *argv: str) -> None:
        """Print one ``--dry-run`` line: the layer and the argument list."""
        print(json.dumps({"layer": layer, "argv": [str(a) for a in argv]}))

    def bazel(self, argv: list[str]) -> int:
        """Run Bazel under the lock, or print the plan line in dry-run mode."""
        if self.dry_run:
            self.plan("bazel", "bazel", *argv)
            return 0
        return with_build_lock([find_bazel(), *argv])

    def sandbox(self, profile: str, cmd: list[str]) -> int:
        """Run ``cmd`` under the resource launcher."""
        if self.dry_run:
            self.plan("sandbox", str(SANDBOX), "--profile", profile, "--",
                      *cmd)
            return 0
        return exec_replace([str(SANDBOX), "--profile", profile, "--", *cmd])

    def submit(self, profile: str, cmd: list[str]) -> int:
        """Submit one GPU job, or run the command when already inside one."""
        if self.dry_run:
            self.plan("gpuq", str(ENTRY), "gpuq", "submit", "--profile",
                      profile, "--", *cmd)
            return 0
        if in_sandbox():
            return exec_replace(cmd)
        return exec_replace([str(ENTRY), "gpuq", "submit", "--profile",
                             profile, "--", *cmd])


# -- Commands -------------------------------------------------------------

def cmd_help(argv: list[str], ctx: Context) -> int:
    sys.stdout.write(_entry.usage())
    return 0


def _build_argv(targets: list[str]) -> list[str]:
    return ["build", *(targets or ["//..."])]


def cmd_build(argv: list[str], ctx: Context) -> int:
    return ctx.bazel(_build_argv(argv))


def _test_argv(argv: list[str]) -> list[str]:
    """The Bazel argument list for ``test``."""
    gpu = False
    targets: list[str] = []
    configs: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--gpu":
            gpu = True
            index += 1
        elif argument == "--config":
            if index + 1 >= len(argv):
                die("--config needs a name")
            configs += ["--config", argv[index + 1]]
            index += 2
        elif argument.startswith("--config="):
            configs.append(argument)
            index += 1
        else:
            targets.append(argument)
            index += 1
    if not targets:
        targets = ["//..."]
    if gpu:
        profile, tag_filter = "t1-gpu", "gpu"
        run_under = f"{ENTRY} gpuq submit --profile={profile} --"
    else:
        profile, tag_filter = "t0-cpu", "-gpu"
        run_under = f"{SANDBOX} --profile={profile} --"
    arguments = [*targets, f"--test_tag_filters={tag_filter}",
                 f"--run_under={run_under}", "--spawn_strategy=local",
                 *configs]
    if gpu:
        # A GPU run selects the CUDA backend, so the shared workflow
        # (//src:model) links the kernels under test. One action at a time.
        arguments += ["--config=cuda", "--local_test_jobs=1"]
        if not have_nccl():
            arguments.append("--config=native")
    return ["test", *arguments]


def cmd_test(argv: list[str], ctx: Context) -> int:
    return ctx.bazel(_test_argv(argv))


def cmd_check(argv: list[str], ctx: Context) -> int:
    gpu = argv[:1] == ["--gpu"]
    if ctx.dry_run:
        ctx.bazel(_build_argv([]))
        ctx.bazel(_test_argv([]))
        if gpu:
            ctx.bazel(_test_argv(["--gpu"]))
        return 0
    children = [["build"], ["test"]]
    if gpu:
        children.append(["test", "--gpu"])
    for child in children:
        code = _run([str(ENTRY), *child]).returncode
        if code != 0:
            return code
    return 0


def cmd_lint(argv: list[str], ctx: Context) -> int:
    return exec_replace([str(TOOLS / "lint.sh"), *argv])


def cmd_shutdown(argv: list[str], ctx: Context) -> int:
    return exec_replace([find_bazel(), "shutdown"])


def cmd_prune(argv: list[str], ctx: Context) -> int:
    return prune.main(argv)


def cmd_gpuq(argv: list[str], ctx: Context) -> int:
    return gpu_queue.main(argv)


def cmd_run(argv: list[str], ctx: Context) -> int:
    if not argv:
        die("run needs a profile")
    profile, rest = argv[0], argv[1:]
    if rest[:1] == ["--"]:
        rest = rest[1:]
    if not rest:
        die(f"no command given; use: tools/nanochat run {profile} -- <cmd...>")
    if profile == DIRECT_PROFILE:
        return ctx.sandbox(profile, rest)
    return ctx.submit(profile, rest)


def _alias(profile: str):
    """A command that submits its argument list under one profile."""

    def command(argv: list[str], ctx: Context) -> int:
        rest = argv[1:] if argv[:1] == ["--"] else argv
        if not rest:
            die("no command given; use: tools/nanochat -- <cmd...>")
        return ctx.submit(profile, rest)

    return command


def _int(value: str, message: str) -> int:
    """Parse a base-10 integer, or report a command-line error."""
    try:
        return int(value, 10)
    except ValueError:
        die(message)
        raise AssertionError("unreachable")


def cmd_dist(argv: list[str], ctx: Context) -> int:
    nproc = 2
    device = ""
    device_given = False
    profile = "train"
    master = "127.0.0.1"
    port = 29500
    log_dir = ""
    as_json = False
    command: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument in ("--nproc-per-node", "--device", "--profile", "--master",
                        "--port", "--log-dir"):
            if index + 1 >= len(argv):
                die(f"dist: {argument} needs a value")
            value = argv[index + 1]
            if argument == "--nproc-per-node":
                nproc = _int(value, f"dist: --nproc-per-node needs a number, "
                                    f"not '{value}'")
            elif argument == "--device":
                device, device_given = value, True
            elif argument == "--profile":
                profile = value
            elif argument == "--master":
                master = value
            elif argument == "--port":
                port = _int(value, f"dist: --port needs a number, "
                                   f"not '{value}'")
            else:
                log_dir = value
            index += 2
        elif argument == "--json":
            as_json = True
            index += 1
        elif argument == "--":
            command = argv[index + 1:]
            break
        else:
            die(f"dist: unknown option {argument}")
    if not command:
        die("dist: no command given; use: tools/nanochat dist -- <cmd...>")
    if nproc < 1:
        die("dist: --nproc-per-node must be >= 1")
    if not device and os.environ.get("NANOCHAT_GPU_DEVICES"):
        pool = os.environ["NANOCHAT_GPU_DEVICES"].split(",")
        if len(pool) < nproc:
            die(f"dist: the worker assigned {len(pool)} device(s), need {nproc}")
        device = ",".join(pool[:nproc])
    if not device:
        device = ",".join(str(rank) for rank in range(nproc))
    devices = device.split(",")
    if len(devices) != nproc:
        die(f"dist: --device has {len(devices)} entries, need {nproc}")
    if not log_dir:
        log_dir = _tempdir("nanochat-dist.")
    os.makedirs(log_dir, exist_ok=True)

    # One queue job for the whole rank group. Kaggle has no queue, so skip
    # when the backend is `none` or the job is already queued.
    if not in_sandbox() and sandbox_backend() != "none":
        reexec = [str(ENTRY), "dist", "--nproc-per-node", str(nproc),
                  "--profile", profile, "--master", master,
                  "--port", str(port), "--log-dir", log_dir]
        if device_given:
            reexec += ["--device", device]
        if as_json:
            reexec.append("--json")
        return exec_replace([str(ENTRY), "gpuq", "submit", "--profile",
                             profile, "--", *reexec, "--", *command])

    # Inside a queued job the outer scope applies the budget and the affinity.
    launcher: list[str] = []
    if not os.environ.get("NANOCHAT_GPU_QUEUED"):
        launcher = [str(SANDBOX), "--profile", profile, "--"]

    start = time.monotonic()
    processes = []
    stdio_paths = []
    for rank in range(nproc):
        expanded = [value.replace("{rank}", str(rank)) for value in command]
        stdio = Path(log_dir) / f"rank{rank}.stdio.log"
        stdio_paths.append(str(stdio))
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = devices[rank]
        env["NANOCHAT_WORLD_SIZE"] = str(nproc)
        env["NANOCHAT_RANK"] = str(rank)
        env["NANOCHAT_LOCAL_RANK"] = str(rank)
        env["NANOCHAT_MASTER_ADDR"] = master
        env["NANOCHAT_MASTER_PORT"] = str(port)
        handle = open(stdio, "w", encoding="utf-8")
        processes.append(subprocess.Popen([*launcher, *expanded], env=env,
                                          stdout=handle, stderr=handle))
        # Rank 0 binds the rendezvous port; give it a moment before the peers
        # start so the connect does not race the bind.
        if rank == 0 and nproc > 1:
            time.sleep(0.5)

    codes = [process.wait() for process in processes]
    seconds = f"{time.monotonic() - start:.3f}"
    returncode = next((code for code in codes if code != 0), 0)

    if as_json:
        ranks = ",".join(
            json.dumps({"rank": rank, "returncode": codes[rank],
                        "stdio": stdio_paths[rank], "seconds": float(seconds)},
                       separators=(",", ":"))
            for rank in range(nproc))
        print(f'{{"ranks":[{ranks}]}}')
    else:
        for rank in range(nproc):
            print(f"dist: rank {rank} exit {codes[rank]} "
                  f"(stdio {stdio_paths[rank]})")
    return returncode


def cmd_client(argv: list[str], ctx: Context) -> int:
    device = ""
    device_given = False
    profile = "train"
    command: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument in ("--device", "--profile"):
            if index + 1 >= len(argv):
                die(f"client: {argument} needs a value")
            if argument == "--device":
                device, device_given = argv[index + 1], True
            else:
                profile = argv[index + 1]
            index += 2
        elif argument.startswith("--device="):
            device, device_given = argument[len("--device="):], True
            index += 1
        elif argument.startswith("--profile="):
            profile = argument[len("--profile="):]
            index += 1
        elif argument == "--":
            command = argv[index + 1:]
            break
        else:
            die(f"client: unknown option {argument}")
    if not command:
        die("client: no command given; use: tools/nanochat client -- <cmd...>")
    if not device:
        if os.environ.get("NANOCHAT_GPU_DEVICES"):
            device = os.environ["NANOCHAT_GPU_DEVICES"]
        elif shutil.which("nvidia-smi"):
            device = _nvidia_smi_indices()
        else:
            device = "0"
    # One queue job for the whole device set. The worker owns the devices, so
    # the client does not name one.
    if not in_sandbox() and sandbox_backend() != "none":
        reexec = [str(ENTRY), "client", "--profile", profile]
        if device_given:
            reexec += ["--device", device]
        return exec_replace([str(ENTRY), "gpuq", "submit", "--profile",
                             profile, "--", *reexec, "--", *command])
    if os.environ.get("NANOCHAT_GPU_QUEUED"):
        return exec_replace(command)
    return ctx.sandbox(profile, command)


def _nvidia_smi_indices() -> str:
    output = _run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                  capture_output=True, text=True).stdout
    indices = ",".join(line.strip() for line in output.splitlines()
                       if line.strip())
    return indices or "0"


def cmd_profile(argv: list[str], ctx: Context) -> int:
    code = ctx.bazel(["build", "//backends/cuda/kernels:attention_benchmark"])
    if ctx.dry_run:
        return code
    if code != 0:
        return code
    binary = BAZEL_BIN / "backends" / "cuda" / "kernels" / "attention_benchmark"
    return exec_replace([str(ENTRY), "gpuq", "submit", "--profile", "t3-bench",
                         "--", str(binary), *argv])


def cmd_roofline(argv: list[str], ctx: Context) -> int:
    code = ctx.bazel(["build", "//src:roofline_main"])
    if code != 0:
        return code
    binary = BAZEL_BIN / "src" / "roofline_main"
    return exec_replace([str(SANDBOX), "--profile", "t0-cpu", "--",
                         str(binary), *argv])


def _tempdir(prefix: str) -> str:
    import tempfile
    return tempfile.mkdtemp(prefix=prefix, dir="/tmp")


def cmd_doctor(argv: list[str], ctx: Context) -> int:
    as_json = argv[:1] == ["--json"]
    values = _doctor_values()
    if as_json:
        print(json.dumps({key: values[key] for key in DOCTOR_KEYS}))
        return 0
    _print_doctor(values)
    return 0


def _doctor_values() -> dict:
    """Collect the environment report. Every field is a string or an object."""
    revision = branch = ""
    if _run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True).returncode == 0:
        revision = _capture(["git", "-C", str(REPO_ROOT), "rev-parse",
                             "--short", "HEAD"])
        branch = _capture(["git", "-C", str(REPO_ROOT), "branch",
                           "--show-current"])
    if shutil.which("nvcc"):
        found = re.search(r"release [0-9.]*",
                          _run(["nvcc", "--version"],
                               capture_output=True, text=True).stdout)
        nvcc = found.group(0) if found else ""
    else:
        nvcc = "NOT FOUND"
    bazel = ""
    if shutil.which("bazelisk"):
        bazel = _run(["bazelisk", "version"], capture_output=True,
                     text=True).stdout.splitlines()[:1]
        bazel = bazel[0] if bazel else ""
    if shutil.which("nvidia-smi"):
        gpu = _run(["nvidia-smi", "--query-gpu=name,driver_version,"
                    "memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader"], capture_output=True,
                   text=True).stdout.splitlines()[:1]
        gpu = gpu[0] if gpu else ""
        gpu_procs = _run(["nvidia-smi", "--query-compute-apps=pid,"
                          "process_name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.splitlines()
        gpu_procs = "; ".join(line.strip() for line in gpu_procs
                              if line.strip())
    else:
        gpu, gpu_procs = "nvidia-smi NOT FOUND", ""
    sandbox = " ".join(_run(
        ["systemctl", "--user", "show", "nanochat.slice", "-p",
         "MemoryCurrent", "-p", "MemoryMax", "-p", "CPUQuotaPerSecUSec"],
        capture_output=True, text=True).stdout.splitlines())
    sim = os.environ.get("NANOCHAT_SIM_PROFILE", "unset")
    if os.environ.get("NANOCHAT_SANDBOX_BACKEND"):
        backend = os.environ["NANOCHAT_SANDBOX_BACKEND"]
    elif _run(["systemd-run", "--user", "--scope", "--quiet", "true"],
              capture_output=True).returncode == 0:
        backend = "systemd"
    else:
        backend = "none (systemd unavailable)"
    return {
        "repo": str(REPO_ROOT),
        "revision": revision,
        "branch": branch,
        "nvcc": nvcc,
        "bazel": bazel,
        "gpu": gpu,
        "gpu_procs": gpu_procs,
        "sandbox": sandbox,
        "sandbox_backend": backend,
        "gpu_queue": _queue_status(),
        "sim": sim,
    }


def _queue_status() -> dict:
    """The queue status object, or an empty object when the query fails."""
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer), \
                contextlib.redirect_stderr(io.StringIO()):
            gpu_queue.main(["status", "--json"])
    except SystemExit:
        return {}
    try:
        payload = json.loads(buffer.getvalue() or "{}")
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _print_doctor(values: dict) -> None:
    print("nanochat.cpp doctor")
    print("====================")
    print(f"repo:      {values['repo']}")
    if values["revision"]:
        print(f"revision:  {values['revision']} "
              f"({values['branch'] or 'detached'})")
    print(f"nvcc:      {values['nvcc']}")
    if values["bazel"]:
        print(f"bazel:     {values['bazel']}")
    print(f"gpu:       {values['gpu']}")
    if values["gpu_procs"]:
        print(f"gpu procs: {values['gpu_procs']}")
    if values["sandbox"]:
        print(f"sandbox:   {values['sandbox']}")
    print(f"sandbox backend: {values['sandbox_backend']}")
    print(f"sim profile: {values['sim']}")
    queue = values["gpu_queue"]
    if queue.get("alive"):
        print(f"gpu queue: worker {queue.get('worker_id', '')} "
              f"devices {queue.get('devices', '')} "
              f"queued {queue.get('queued', 0)} "
              f"running {queue.get('running_job') or '-'}")
    else:
        print("gpu queue: idle (no worker)")
    for log in ("/tmp/nanochat-gpu.log", "/tmp/nanochat-sandbox.log"):
        if not os.path.isfile(log):
            continue
        print(f"--- tail {log} ---")
        for line in _tail(log, 5):
            print(line)
    for binary in (BAZEL_BIN / "src" / "train_main",
                   BAZEL_BIN / "backends" / "cuda" / "kernels"
                   / "attention_benchmark"):
        if not binary.exists():
            continue
        stamp = time.strftime("%Y-%m-%d %H:%M",
                              time.localtime(binary.stat().st_mtime))
        print(f"built:     {binary} ({stamp})")


def _tail(path: str, count: int) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n") for line in handle.readlines()[-count:]]


SIM_DEVICES = ("h100", "h200", "a100", "t4", "gtx1080ti")
SIM_SUITES = {
    "correctness": ["//tests:oracle_test", "//tests:train_parity_test",
                    "//tests:sim_numerics_test", "//tests:numerics_trace_test"],
    "fp8-fast": ["//tests:fp8_rounding_test", "//tests:fp8_gemm_test",
                 "//tests:fp8_graph_test", "//tests:device_profile_test"],
    "collective": ["//tests:collective_sim_test"],
    "kernel": ["//tests:kernel_emu_test"],
    "profile": ["//tests:device_profile_test"],
    "all": ["//tests:oracle_test", "//tests:train_parity_test",
            "//tests:sim_numerics_test", "//tests:numerics_trace_test",
            "//tests:fp8_rounding_test", "//tests:fp8_gemm_test",
            "//tests:fp8_graph_test", "//tests:collective_sim_test",
            "//tests:kernel_emu_test", "//tests:device_profile_test"],
}
SIM_GATE_TARGETS = ["//backends/cuda:all_kernels",
                    "//backends/cuda/collective:native_sync",
                    "//backends/cuda:cuda_runtime_gpu_test",
                    "//backends/cuda/kernels:precision_gpu_test",
                    "//backends/cuda/kernels:row_kernels_gpu_test",
                    "//backends/cuda/kernels:sequence_oracle_gpu_test"]


def _simulate_args(argv: list[str]) -> tuple[str, str, str, str]:
    mode, device, suite, precision = "run", "h100", "all", "fp32"
    for argument in argv:
        if argument.startswith("--device="):
            device = argument.split("=", 1)[1]
        elif argument.startswith("--suite="):
            suite = argument.split("=", 1)[1]
        elif argument.startswith("--mode="):
            mode = argument.split("=", 1)[1]
        elif argument.startswith("--precision="):
            precision = argument.split("=", 1)[1]
        else:
            die(f"simulate: unknown argument '{argument}'")
    return mode, device, suite, precision


def cmd_simulate(argv: list[str], ctx: Context) -> int:
    # `--name value` comes as separate tokens; fold them onto the `=` form.
    folded: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        match = re.fullmatch(r"--(device|suite|mode|precision)", argument)
        if match:
            if index + 1 >= len(argv):
                die(f"simulate: {argument} needs a value")
            folded.append(f"{argument}={argv[index + 1]}")
            index += 2
        else:
            folded.append(argument)
            index += 1
    mode, device, suite, precision = _simulate_args(folded)
    if device not in SIM_DEVICES:
        die(f"simulate: unknown device '{device}' ({' '.join(SIM_DEVICES)})")

    if mode == "compile":
        ptxas_log = os.environ.get(
            "NANOCHAT_SIM_PTXAS_LOG", f"/tmp/nanochat-sim-ptxas-{device}.log")
        gate_argv = ["build", f"--config={device}", "--config=sim-gate",
                     *SIM_GATE_TARGETS]
        if ctx.dry_run:
            ctx.plan("bazel", "bazel", *gate_argv)
            return 0
        return _simulate_compile(gate_argv, ptxas_log, device)
    if mode != "run":
        die(f"simulate: unknown mode '{mode}' (run compile)")

    if suite in ("api", "all"):
        code = _simulate_api(ctx, device)
        if code != 0:
            return code
    if suite == "api":
        return 0
    if suite not in SIM_SUITES:
        die(f"simulate: unknown suite '{suite}'")
    configs = ["--config=sim"]
    if precision == "bf16":
        configs.append("--config=bf16")
    elif precision == "fp8":
        configs.append("--config=fp8")
    elif precision != "fp32":
        die(f"simulate: unknown precision '{precision}' (fp32 bf16 fp8)")
    return ctx.bazel([
        "test", *SIM_SUITES[suite], "--test_tag_filters=-gpu",
        f"--run_under={SANDBOX} --profile=t0-cpu --", "--spawn_strategy=local",
        f"--test_env=NANOCHAT_SIM_PROFILE={device}", *configs])


def _simulate_compile(gate_argv: list[str], ptxas_log: str,
                      device: str) -> int:
    """Run the compile gate, keep the ptxas resource lines, and report."""
    import tempfile
    handle = tempfile.NamedTemporaryFile(
        mode="w", prefix="nanochat-sim-gate.", suffix=".log", delete=False,
        encoding="utf-8")
    try:
        code = with_build_lock([find_bazel(), *gate_argv], stdout=handle,
                               stderr=handle)
    finally:
        handle.close()
    text = Path(handle.name).read_text(encoding="utf-8", errors="replace")
    if code != 0:
        sys.stderr.write(text)
        os.unlink(handle.name)
        return code
    # Keep the ptxas resource lines for the record. A cached build compiles
    # nothing, so keep the previous log instead of erasing it.
    fresh = [line for line in text.splitlines(keepends=True)
             if "ptxas info" in line or _REGISTERS.search(line)]
    fresh_count = sum(1 for line in fresh if _REGISTERS.search(line))
    if fresh_count > 0:
        Path(ptxas_log).write_text("".join(fresh), encoding="utf-8")
        count, note = fresh_count, "fresh"
    elif os.path.isfile(ptxas_log) and os.path.getsize(ptxas_log) > 0:
        previous = Path(ptxas_log).read_text(encoding="utf-8",
                                             errors="replace")
        count = sum(1 for line in previous.splitlines()
                    if _REGISTERS.search(line))
        note = "build cached; kept the previous log"
    else:
        count, note = 0, "build cached and no earlier log"
    print(f"simulate: compile gate OK ({device}, -Werror, ptxas records: "
          f"{count}; {note})")
    print(f"simulate: ptxas resource log -> {ptxas_log}")
    os.unlink(handle.name)
    return 0


_REGISTERS = re.compile(r"Used [0-9]+ registers")


def _simulate_api(ctx: Context, device: str) -> int:
    """Build the mock, run the symbol gate, and run the two API tests."""
    mock = (BAZEL_BIN / "tools" / "cuda_sim"
            / "libcuda_sim_interposer.so").resolve()
    real = (BAZEL_BIN / "backends" / "cuda" / "cuda_runtime_gpu_test").resolve()
    gate = str(TOOLS / "cuda_sim" / "check_symbols.sh")
    sim_log = "<sim-api>.jsonl" if ctx.dry_run else str(
        Path(_tempdir("nanochat-sim-api.")) / "events.jsonl")
    build_argv = ["build", "--config=cuda",
                  "//tools/cuda_sim:cuda_sim_interposer",
                  "//backends/cuda:cuda_runtime_gpu_test",
                  "//tests:cuda_sim_preload_test"]
    preload_argv = [
        "test", "//tests:cuda_sim_preload_test", "--config=cuda",
        "--test_tag_filters=sim",
        f"--run_under={SANDBOX} --profile=t0-cpu --",
        "--spawn_strategy=local",
        f"--test_env=NANOCHAT_SIM_PROFILE={device}",
        f"--test_env=LD_PRELOAD={mock}",
        f"--test_env=NANOCHAT_SIM_LOG={sim_log}"]
    unit_argv = [
        "test", "//tests:cuda_sim_test", "--test_tag_filters=sim",
        f"--run_under={SANDBOX} --profile=t0-cpu --",
        "--spawn_strategy=local",
        f"--test_env=NANOCHAT_SIM_PROFILE={device}"]
    if ctx.dry_run:
        ctx.plan("bazel", "bazel", *build_argv)
        for extra in ([], [str(real)], ["--negative-self-test"]):
            ctx.plan("host", gate, *extra)
        ctx.plan("bazel", "bazel", *preload_argv)
        ctx.plan("bazel", "bazel", *unit_argv)
        return 0
    code = ctx.bazel(build_argv)
    if code != 0:
        return code
    if not mock.is_file():
        die(f"simulate: the interposer .so was not built ({mock})")
    if not real.is_file():
        die(f"simulate: {real} was not built")
    for extra in ([], [str(real)], ["--negative-self-test"]):
        status = _run([gate, *extra]).returncode
        if status != 0:
            return status
    code = ctx.bazel(preload_argv)
    if code != 0:
        return code
    return ctx.bazel(unit_argv)


def cmd_gpu(argv: list[str], ctx: Context) -> int:
    if in_sandbox():
        profile = "t1-gpu"
        rest = list(argv)
        while rest:
            argument = rest[0]
            if argument == "--profile":
                if len(rest) < 2:
                    die("--profile needs a name")
                profile, rest = rest[1], rest[2:]
            elif argument.startswith("--profile="):
                profile, rest = argument.split("=", 1)[1], rest[1:]
            elif argument == "--":
                rest = rest[1:]
                break
            else:
                break
        if not rest:
            die("gpu: no command given")
        if os.environ.get("NANOCHAT_GPU_QUEUED"):
            return exec_replace(rest)
        return ctx.sandbox(profile, rest)
    return exec_replace([str(ENTRY), "gpuq", "submit", *argv])


#: The handler for every ``_entry`` row, in the table order. The keys must
#: equal `_entry.names()`; `python/tests/contract_test.py` compares them.
COMMANDS = {
    "build": cmd_build,
    "test": cmd_test,
    "check": cmd_check,
    "lint": cmd_lint,
    "shutdown": cmd_shutdown,
    "prune": cmd_prune,
    "gpuq": cmd_gpuq,
    "run": cmd_run,
    "dist": cmd_dist,
    "client": cmd_client,
    "train": _alias("train"),
    "rl": _alias("train"),
    "eval": _alias("eval"),
    "bench": _alias("t3-bench"),
    "profile": cmd_profile,
    "roofline": cmd_roofline,
    "doctor": cmd_doctor,
    "simulate": cmd_simulate,
    "verify": _alias("t2-parity"),
    "gpu": cmd_gpu,
    "help": cmd_help,
}


def _split_dry_run(argv: list[str]) -> tuple[bool, list[str]]:
    """Remove ``--dry-run`` up to the first ``--`` and report whether it ran."""
    dry_run = False
    kept: list[str] = []
    seen_separator = False
    for argument in argv:
        if seen_separator:
            kept.append(argument)
            continue
        if argument == "--dry-run":
            dry_run = True
        elif argument == "--":
            seen_separator = True
            kept.append(argument)
        else:
            kept.append(argument)
    return dry_run, kept


def main(argv: list[str] | None = None) -> int:
    """Run one command and report a command-line error as exit code 2."""
    try:
        return _dispatch(argv)
    except Failure as error:
        sys.stderr.write(f"nanochat: {error}\n")
        return 2


def _dispatch(argv: list[str] | None) -> int:
    """Split the flags, resolve the command, and run it."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    dry_run, kept = _split_dry_run(arguments)
    if not kept:
        sys.stdout.write(_entry.usage())
        return 2
    command, rest = kept[0], kept[1:]
    if command in ("-h", "--help"):
        command, rest = "help", []
    handler = COMMANDS.get(command)
    if handler is None:
        die(f"unknown command '{command}' (try: tools/nanochat help)")
    ctx = Context(dry_run=dry_run)
    if dry_run:
        if command not in DRY_RUN_COMMANDS:
            die("--dry-run covers build, test, check, run, and simulate")
        # A nested invocation (from `check`) prints only its external commands.
        nested = os.environ.get("NANOCHAT_PLAN_NESTED") == "1"
        if not nested:
            ctx.plan("argv", command, *rest)
    return handler(rest, ctx)


if __name__ == "__main__":
    sys.exit(main())
