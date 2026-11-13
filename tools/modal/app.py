"""The Modal app that runs the two legs of one nanochat.cpp chain.

The image holds the package and the compiled shared library; the weights
Volume holds the weights. That split is section 6.4 of
``docs/build.md``, the Python distribution: a new checkpoint then costs a
download, not an image build.

- ``nanochat-runs`` at ``/runs`` is the run directory the notebook polls.
- ``nanochat-weights`` at ``/weights`` caches one stage per folder,
  ``<repo_id>/<stage>/``, keyed by the stage revision it holds in
  ``.revision``. ``fetch_stage`` downloads a revision it does not hold yet and
  returns the folder, so a warm start is a download and never a rebuild.

Two legs run here, and section 7 of ``docs/post-training.md`` is the
table they follow. A leg is a frozen :class:`Leg` record: a name, a card type,
a card count, an architecture, a precision, and the flag triple the builder
emits. ``t4`` is a T4 at ``sm_75`` and fp32, and it runs first so its artifact
survives an H100 failure; ``h100`` is an H100 at ``sm_90`` and bf16.
:func:`image_for` returns one image for one leg, and one leg builds one
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
decision D10 (``docs/post-training.md`` section 6.3). One build per container
produces the library and ``//src:nanochat_cli``, the one program every stage
subcommand and the roofline gate run, so the chain and gate G3 both find a
binary already built (sections 6.4 and B6). The image holds no compiled
artifact: a Modal Volume cannot reach an image build, so each container builds
once against the cache Volume (see :func:`_ensure_build`).

Both legs link the native ring, which is the one collective, and the chain runs
``world_size=1`` so the ring reduces to a no-op. No leg builds a simulator: a
named config could carry the engine or another leg's device values, so the
hardware leg passes the builder's explicit flag triple and never a named config
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
    modal run tools/modal/app.py::card_job --name gemm_check

``card_job`` is the one entrypoint of the four gates and benchmarks: a name
selects the job, and the answer prints. ``pipeline_demo`` runs both legs of one
job, the T4 leg first, and prints one document with three keys: ``legs`` (each
leg's report summary), ``comparison`` (the cross-leg step-time ratio and the
recorded MFU), and ``gates`` (G1 to G3 with pass or fail and the numbers each
one read). It exits non-zero when a gate fails, and a leg that fails still
leaves the other leg's summary in the document. Each leg writes its own run
directory on the ``nanochat-runs`` Volume, at ``<run_id>-t4`` and
``<run_id>-h100``.

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

import functools
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
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

#: The Bazel cache Volume and its mount point. An image build cannot reach a
#: Volume and a Volume cannot reach an image build, so the image holds no
#: compiled artifact: each container builds its leg once, against this cache
#: (``docs/build.md``, the Python distribution). One folder per leg, because a leg is one
#: architecture at one precision.
CACHE_VOLUME = "nanochat-build-cache"
CACHE_ROOT = "/cache"

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

#: The shared library that the container build produces.
LIBRARY = f"{REPO_ROOT}/bazel-bin/bindings/libnanochat_shared.so"

#: The one stage command binary the chain launches, beside the library: the
#: Bazel label the container build names (``docs/post-training.md`` B6), so a
#: container needs no second build and the wheel line below stays a later
#: one-line change. One program runs every stage, so the subcommands are a
#: table rather than a target each.
#: The stage names of the chain, in pipeline order. A name is the stage
#: identifier and the stem of its environment override, not a Bazel target:
#: one program runs every stage (``docs/post-training.md`` B6).
STAGE_NAMES = ("sft_main", "rl_worker")

CLI_TARGET = "//src:nanochat_cli"
STAGE_TARGETS = (CLI_TARGET,)

#: The Bazel label of the roofline tool. Gate G3 runs
#: ``tools/nanochat roofline`` inside the container, and the tool must find its
#: binary already built there instead of starting a second build
#: (``docs/post-training.md`` sections 7 and 8). It is not a stage:
#: no stage launches it, and only the gate reads its report.
ROOFLINE_TARGET = CLI_TARGET

#: The Bazel label of the native tokenizer trainer, and the binary it produces.
#: It is not a stage executable either: only :func:`corpus` calls it, to build
#: the ``NCTOKEN1`` artifact a ``files`` corpus needs. It sits in its own
#: directory, so the path is not the ``bazel-bin/src`` one the stage
#: executables take (``docs/post-training.md`` section 9 row 3).
TOKENIZER_TARGET = "//src/tokenizer:tok_train_main"
TOKENIZER_BIN = f"{REPO_ROOT}/bazel-bin/src/tokenizer/tok_train_main"

#: The Bazel label of the single-device trainer, and its binary. The chain runs
#: the mesh trainer, which reports one number per step. This one reports the
#: phase split: data, forward, backward, sync, and optimizer. A step that sits
#: far from its roofline bound is attributed with it, on the card that ran it,
#: because the split is the only thing that says which phase to fix.
TRAIN_TARGET = CLI_TARGET
TRAIN_BIN = f"{REPO_ROOT}/bazel-bin/src/nanochat_cli"

def _bazel_bin(target: str) -> str:
    """The path one target label takes inside the container build."""
    return f"{REPO_ROOT}/bazel-bin/{target.removeprefix('//').replace(':', '/')}"

#: The Bazel label of the Muon optimizer benchmark, and its binary. The
#: benchmark times the Polar Express GEMM shapes and one whole optimizer phase,
#: so a card that is slow in the optimizer reports its own rate instead of
#: hiding inside a step time. It is not a stage executable: only
#: :func:`bench_h100` runs it.
BENCH_TARGET = "//backends/cuda/kernels:muon_benchmark"
BENCH_BIN = _bazel_bin(BENCH_TARGET)

#: The attention benchmark and its family test. The H100 leg is the only card
#: with tensor cores this repository can reach, so the bf16 tile of
#: ``docs/attention.md`` Part 6 is measured and verified there: one job runs the
#: test binary, and the other reports the rows the promotion gate reads.
ATTENTION_BENCH_TARGET = "//backends/cuda/kernels:attention_benchmark"
ATTENTION_TEST_TARGET = "//backends/cuda/kernels:attention_gpu_test"
ATTENTION_BENCH_BIN = _bazel_bin(ATTENTION_BENCH_TARGET)
ATTENTION_TEST_BIN = _bazel_bin(ATTENTION_TEST_TARGET)

#: The hand tensor-core GEMM and its two card jobs
#: (``docs/tensor-core-gemm-plan.md`` sections B4 and B5). The tile is bf16
#: only, so the H100 leg is the one image that can build it: the T4 leg is
#: fp32 and leaves the tile out. One job runs the device gate, which compares
#: the tile with cuBLAS; the other reports the sweep and the fraction of the
#: cuBLAS rate that the promotion gate reads.
GEMM_BENCH_TARGET = "//backends/cuda/gemm:gemm_benchmark"
GEMM_TEST_TARGET = "//backends/cuda/gemm:gemm_gpu_test"
GEMM_BENCH_BIN = _bazel_bin(GEMM_BENCH_TARGET)
GEMM_TEST_BIN = _bazel_bin(GEMM_TEST_TARGET)

#: The fp32 shared-memory tile and its card gate
#: (``docs/kernels.md`` Part 3 section 6 phase 3). The T4 is the leg this tile
#: exists for: that leg is fp32 at ``sm_75``, where the bf16 tile and the
#: warpgroup mainloop are both compiled out. The T4 image builds this test, and
#: the benchmark of the same leg times the tile through its ``hand`` arm.
GEMM_FP32_TEST_TARGET = "//backends/cuda/gemm:gemm_fp32_gpu_test"
GEMM_FP32_TEST_BIN = _bazel_bin(GEMM_FP32_TEST_TARGET)

#: Phase 1 of ``docs/kernels.md`` Part 3: the gate of the `ldmatrix` and
#: `mma.sync` micro-tile. One binary holds a host row and a device row over one
#: reference, so this job reports both and the device row is the instruction
#: pair in one warp. The target builds at every architecture, because below
#: `sm_80` the header's plain arm is the one that compiles.
GEMM_MMA_TARGET = "//backends/cuda/gemm:mma_bf16_test"
GEMM_MMA_BIN = _bazel_bin(GEMM_MMA_TARGET)

#: Phase 0 of ``docs/kernels.md`` Part 3 and Phase H1 of
#: ``docs/attention.md`` Part 8 in one program, ``dev/kernels/wgmma_bench.cu``.
#: It absorbed the standalone ``dev/kernels/wgmma_probe.cu``, so one job builds
#: and runs both row sets. It is the one target in the tree that needs
#: ``sm_90a``, and the image's own build emits ``sm_90``, so
#: :func:`bench_wgmma_h100` builds it inside the container with the tree's
#: builder before it runs it. The probe is a device-only prototype: the
#: emulation engine maps no WGMMA, so there is no off-device leg.
WGMMA_BENCH_TARGET = "//dev/kernels:wgmma_bench"
WGMMA_BENCH_BIN = _bazel_bin(WGMMA_BENCH_TARGET)
#: The architecture the probe needs. It is stated by the target
#: ``archs = ["sm_90a"]``, so no build command carries a flag, and the name is
#: kept here only to document what the target asserts.
WGMMA_ARCH = "sm_90a"

#: The labels the one build of an image names: the shared library, the stage
#: binary the chain launches (``docs/post-training.md`` B6 and D2), the
#: tokenizer trainer the corpus function runs, the optimizer benchmark
#: :func:`bench_h100` times, the two attention jobs, and the two hand GEMM jobs.
#: The roofline tool and the trainer are the same program and the same label,
#: so each appears once.
BUILD_TARGETS = ("//bindings:nanochat_shared", *STAGE_TARGETS,
                 TOKENIZER_TARGET, BENCH_TARGET, ATTENTION_BENCH_TARGET,
                 ATTENTION_TEST_TARGET, GEMM_BENCH_TARGET, GEMM_TEST_TARGET,
                 GEMM_FP32_TEST_TARGET, GEMM_MMA_TARGET)

#: The files the module graph reads before it needs a package of the tree. The
#: fetch layer holds these and the tooling, and nothing else, so a source change
#: does not invalidate the fetch (``docs/build.md``, the Python distribution).
MANIFEST_FILES = ("MODULE.bazel", "MODULE.bazel.lock", ".bazelversion",
                  ".bazelrc", ".bazelignore")

#: The two directories of the tree that the module graph and the entry point
#: read: the package behind ``tools/nanochat``, and the two repository rules.
MANIFEST_DIRS = ("python", "third_party")

#: The two files of ``tools/`` the entry point needs: the program, and the
#: profile table that ``nanochat_cpp.sandbox`` reads when it loads.
MANIFEST_TOOL_FILES = ("tools/nanochat", "tools/sandbox/profiles.json")

#: The external labels the fetch layer resolves. They are the two archives this
#: tree downloads, and resolving them walks the whole module graph, so one
#: command fills the output base with every repository the build needs.
FETCH_TARGETS = ("@cutlass//:cutlass", "@zstd//:zstd")

#: The directory of the container build that holds them, and the variables
#: that name them. ``nanochat_cpp._build.stage_binary`` prefers the override, and
#: the image sets it because the installed distribution answers ``import
#: nanochat_cpp`` from site-packages, where ``repo_root()`` is not the image
#: tree (section 6.4). The names follow ``_build.stage_override``: the stage
#: name upper-cased behind the prefix, and both name the one program.
STAGE_BIN = f"{REPO_ROOT}/bazel-bin/src"
STAGE_ENV = {f"NANOCHAT_{name.upper()}": f"{STAGE_BIN}/nanochat_cli"
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
#: legs take this one value (``docs/post-training.md`` section 7).
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
    library that ran (``docs/post-training.md`` P6 and section 7).

    ``build`` is the flag triple the builder emits today
    (``nanochat_cpp._build._build_command``), written out here so a reader sees
    the flags without following the builder. The test contracts the two
    against each other, and the container build reads the leg's precision and
    architecture from the image environment through ``BuildOptions.from_env``,
    so the builder stays the one home of the command line.
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
#: the cheap leg while the expensive one runs. Both legs run one card, link the
#: native ring, and add no simulator define and no fp8 build.
T4 = Leg(name="t4", card="T4", count=1, arch="sm_75", precision="fp32",
         build=("--config=cuda", "--define=precision=fp32",
                "--@rules_cuda//cuda:archs=sm_75"))
H100 = Leg(name="h100", card="H100", count=1, arch="sm_90", precision="bf16",
           build=("--config=cuda", "--define=precision=bf16",
                  "--@rules_cuda//cuda:archs=sm_90"))

#: The fp8 leg, and why it is a leg and not a flag. The fp8 arm of
#: ``docs/kernels.md`` section 10.4 is compiled only under
#: ``--define=precision=fp8``, and a report's ``host.precision`` is a fact of the
#: library that ran, so its image is its own. It is not a leg of the
#: post-training chain: nothing in ``notebooks/target.py`` deploys it, and it
#: stays out of :data:`LEG_ORDER` so the two-leg run is unchanged.
H100_FP8 = Leg(name="h100-fp8", card="H100", count=1, arch="sm_90",
               precision="fp8",
               build=("--config=cuda", "--define=precision=fp8",
                      "--@rules_cuda//cuda:archs=sm_90"))

#: The legs by name, and the two the chain runs in order. A leg outside
#: :data:`LEG_ORDER` is an image one card job names, so a run never picks it up.
LEGS = {T4.name: T4, H100.name: H100, H100_FP8.name: H100_FP8}
LEG_ORDER = (T4, H100)

#: The separator between the run id of a two-leg run and the leg name, so both
#: reports stay on the ``nanochat-runs`` Volume and a reviewer names one.
RUN_ID_SEPARATOR = "-"


def leg_run_id(run_id, leg) -> str:
    """The run id of one leg: ``<run_id>-t4`` or ``<run_id>-h100``."""
    return f"{run_id}{RUN_ID_SEPARATOR}{leg.name}"


def selected_legs(name: str) -> tuple:
    """The legs a run asks for: ``both``, ``t4``, or ``h100``.

    A name the table does not hold is a ValueError that lists the answers, so a
    typo costs one line here and not a container. One card is the answer when a
    reader wants one number and gate G2 has nothing to compare (section 8).
    """
    wanted = str(name or "both").strip().lower()
    if wanted in ("", "both", "all"):
        return tuple(LEG_ORDER)
    if wanted in LEGS:
        return (LEGS[wanted],)
    raise ValueError(
        f"unknown legs {name!r}; the legs are 'both' (the default), 't4', "
        f"'h100', and 'h100-fp8'")


app = modal.App(APP_NAME)

runs = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)
weights = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
build_cache = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)


def _ensure_build() -> str:
    """Build the leg's targets in this container, and return the cache path.

    The image holds no compiled artifact, because a Modal Volume cannot reach an
    image build. Every container therefore builds its leg once, against the
    cache that :data:`CACHE_VOLUME` holds. The leg is not a parameter: the
    image's own environment names the cache folder, the precision, and the
    architecture, and ``BuildOptions.from_env`` reads them, so a container
    builds what it declares.

    The cost is the measurement of the T4 leg on 2026-11-10. Its 190 actions
    read 133 disk cache hits, and the build cost 19.1 seconds on a warm Volume
    and 101.9 seconds on the first read after :func:`warm_cache` wrote the
    cache. The hits are downloads, so the Volume's latency per file is the cost,
    and a local disk cache would read it as 3.4 seconds. A warm container
    therefore pays tens of seconds, and the image build pays nothing.

    The Volume is committed, because a container that exits without a commit
    throws its cache away, and the next container would compile again.

    The output is streamed and not captured. A cold build of a leg takes
    minutes, and a captured one prints nothing until it ends: the container log
    is where an operator reads the failing action. One summary line states the
    cache, the elapsed time, and the exit status, so a run report carries the
    cost of its own build.
    """
    from nanochat_cpp import _build  # noqa: PLC0415 - the image only

    options = _build.BuildOptions.from_env()
    command = _build._build_command(Path("tools/nanochat"), options,
                                    BUILD_TARGETS)
    started = time.monotonic()
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    elapsed = time.monotonic() - started
    build_cache.commit()
    print(f"build: {options.bazel_cache} in {elapsed:.1f}s "
          f"(exit {result.returncode})", flush=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"the container build failed ({result.returncode}) after "
            f"{elapsed:.1f}s with the cache at {options.bazel_cache}; the "
            f"output above names the failing action")
    return options.bazel_cache


def _function(**kwargs):
    """``app.function`` with the build cache and the container build.

    One wrapper, so no function can forget either: every container mounts the
    cache Volume, and every container builds its leg before its body runs. A
    function that skipped the build would find no library, and one that skipped
    the mount would compile into its own disk and keep nothing.
    """
    volumes = dict(kwargs.pop("volumes", {}))
    volumes[CACHE_ROOT] = build_cache

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **inner):
            _ensure_build()
            return func(*args, **inner)

        return app.function(volumes=volumes, **kwargs)(wrapper)

    return decorate


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
        "NANOCHAT_BAZEL_CACHE": f"{CACHE_ROOT}/{leg.name}",
        "NANOCHAT_SANDBOX_BACKEND": "none",
        "NANOCHAT_TARGET": "local",
        "PYTHONPATH": f"{REPO_ROOT}/python:{REPO_ROOT}/notebooks",
    }


def _builder_command(leg: Leg, targets: Sequence[str], flags: Sequence[str],
                     pythonpath: str | None = None,
                     root: str = REPO_ROOT) -> str:
    """The one ``python3 -c`` command that runs the tree's builder in a container.

    One home for the quoting, so the fetch command and the probe command cannot
    drift. ``flags`` is appended to the list that ``_build._build_command``
    returns, so the builder stays the one home of the flag line. The flags are a
    list literal because that function returns a list, and a tuple would raise
    ``TypeError`` inside the container and not here.

    ``pythonpath`` names the tree's ``python/`` directory for a command that
    runs before the install. The fetch layer is such a command: it copies the
    tooling and fetches before the package reaches site-packages, so the import
    resolves from the tree.
    """
    labels = "(" + "".join(f"{label!r}, " for label in targets) + ")"
    extra = "[" + "".join(f"{flag!r}, " for flag in flags) + "]"
    prefix = f"PYTHONPATH={pythonpath} " if pythonpath else ""
    return (
        f"cd {root} && {prefix}python3 -c \""
        "import subprocess; from nanochat_cpp import _build; "
        "_options = _build.BuildOptions.resolve("
        f"backend='cuda', precision='{leg.precision}', arch='{leg.arch}'); "
        "_command = _build._build_command("
        f"'tools/nanochat', _options, {labels}) + {extra}; "
        "subprocess.run(_command, check=True)\"")


def image_fetch_command(leg: Leg, targets: Sequence[str] = FETCH_TARGETS,
                        root: str = REPO_ROOT) -> str:
    """The one fetch command of one leg's image.

    It runs the tree's builder with ``--nobuild``, so the module graph resolves
    and every external repository arrives while no action runs. The container
    that later builds the leg starts from this image, so its output base already
    holds those repositories and it compiles without a fetch. A measured cold
    fetch costs 18 seconds and a fetch against a warm cache 4 seconds.

    No action pool flag is passed: this command runs no action.
    """
    return _builder_command(leg, targets, ("--nobuild",),
                            pythonpath=f"{root}/python", root=root)


def probe_build_command(leg: Leg, target: str, root: str = REPO_ROOT) -> str:
    """The one build command of a prototype.

    The image builds every target of a leg at the leg's architecture, and a
    prototype that needs another one cannot share that: WGMMA is architecture
    conditional, so ``dev/kernels/wgmma_bench.cu`` compiles only at
    ``sm_90a``. The target states that architecture for itself, so this command
    names no flag and the leg's own architecture is the resolved default. The
    command calls the tree's builder, so the flag line stays in
    ``nanochat_cpp._build``. ``docs/build.md`` records the per-target form, and
    the CUTLASS candidates of ``backends/cuda/gemm/BUILD.bazel`` are the
    precedent.
    """
    return (
        f"cd {root} && python3 -c \""
        "import subprocess; from nanochat_cpp import _build; "
        "_options = _build.BuildOptions.resolve("
        f"backend='cuda', precision='{leg.precision}'); "
        "subprocess.run(_build._build_command("
        f"'tools/nanochat', _options, ({target!r},)), check=True)\"")


#: The patterns that keep local state out of the image. ``add_local_dir``
#: hashes the tree, so a file that changes on every run rebuilds the copy layer
#: and every layer above it, including the layer that runs Bazel. The list is a
#: module constant, so ``tools/modal/modal_app_test.py`` reads the same list
#: that the call passes (``docs/build.md``, the Python distribution).
IMAGE_IGNORE = (
    # The history is not source, and one checkout is tens of megabytes.
    ".git",
    # Bazel's convenience symlinks and output tree of the host.
    "bazel-*",
    # Bytecode of the host interpreter.
    "**/__pycache__",
    # The profile artifacts of ``docs/performance.md``: local measurements, and
    # one trace is hundreds of megabytes. The loop passes its job file as a
    # JSON string, so the image never reads them.
    "profiles",
    # The coding agent logs into ``.pi``, and a file that changes under the
    # build fails it.
    ".pi",
    # The state of the agent runner: a SQLite database that it rewrites on
    # every action, and one git worktree per campaign. It is untracked, it
    # changes on every run, and no target reads it, so this entry is the
    # difference between a cached build layer and a full rebuild.
    ".sliceme",
    # The sandbox configuration of the agent runner, which no target reads.
    "sliceme.sandbox.json",
    # The host file of ``docs/host-portability.md`` section 8.3 pins the
    # compiler and the budget of the workstation, such as /usr/bin/gcc-12, which
    # an Ubuntu 22.04 image does not have. A deploy from a host that ran
    # ``tools/kaggle/bootstrap.sh`` would otherwise build an image that cannot
    # compile.
    ".bazelrc.local",
)


def _manifest_layer(image, leg: Leg):
    """Copy the manifest and the tooling, then resolve the external repositories.

    The tree arrives in two layers. This one holds what the module graph reads
    and what the entry point needs, so a source change does not touch it. The
    next layer is the tree itself, and the build above it then reads a warm
    output base: the workspace path is the same, so the external repositories
    of this layer are the ones it uses.

    The command is chmod'ed first, because ``add_local_file`` copies one file
    and the mode of ``tools/nanochat`` is what makes it the entry point.
    """
    for name in MANIFEST_FILES:
        image = image.add_local_file(str(REPO / name),
                                     f"{REPO_ROOT}/{name}", copy=True)
    for name in MANIFEST_DIRS:
        image = image.add_local_dir(str(REPO / name),
                                    f"{REPO_ROOT}/{name}", copy=True,
                                    ignore=["**/__pycache__"])
    for name in MANIFEST_TOOL_FILES:
        image = image.add_local_file(str(REPO / name),
                                     f"{REPO_ROOT}/{name}", copy=True)
    return image.run_commands(f"chmod 0755 {REPO_ROOT}/tools/nanochat",
                              image_fetch_command(leg))


#: The apt packages of every image: ``build-essential`` compiles, ``curl``
#: fetches Bazelisk, and ``git`` lets a run name the revision it ran.
APT_PACKAGES = ("build-essential", "curl", "git")

#: The profiler package of the two profile images. It sits beside
#: :data:`APT_PACKAGES` and below the tree, because a layer above the tree copy
#: is rebuilt by every source change: one file cost 215 and 63 seconds of apt on
#: 2026-11-10, for a profiler that a card job never runs.
NSYS_PACKAGE = "nsight-systems-2023.4.4"


def _apt_command(packages: Sequence[str]) -> str:
    """The apt line of an image, with three attempts.

    ``Image.apt_install`` emits a bare ``apt-get update``, and a builder whose
    index fetch fails fails the whole deployment. One attempt on 2026-11-09 read
    "Some index files failed to download" and exited 100, which is the same
    class as the ``504`` this tree records for the GitHub fetch. The loop keeps
    one layer, and a retry costs seconds instead of a run.

    ``--no-install-recommends`` keeps the layer to the packages the build names.
    """
    names = " ".join(packages)
    return (
        "for attempt in 1 2 3; do "
        f"apt-get update && apt-get install -y --no-install-recommends {names} "
        "&& exit 0; "
        'echo "apt attempt ${attempt} failed"; sleep 15; '
        "done; exit 1")


def _toolkit_image(leg: Leg, profile: bool = False):
    """The base of one leg's image: the toolkit, the apt packages, and Bazelisk.

    ``profile`` adds Nsight Systems here, below the tree, and not above it. The
    order is the cost: a layer above the tree copy is rebuilt by every source
    change, and the profiler is a 335 MB apt install that a card job never uses.
    """
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
        .run_commands(_apt_command(APT_PACKAGES))
    )
    if profile:
        image = image.run_commands(_apt_command((NSYS_PACKAGE,)))
    return image.run_commands(
        "curl -fsSL --retry 3 --retry-all-errors "
        "-o /usr/local/bin/bazelisk "
        "https://github.com/bazelbuild/bazelisk/releases/download/"
        "v1.22.1/bazelisk-linux-amd64",
        "chmod 0755 /usr/local/bin/bazelisk",
    )


def _build_image(leg: Leg, profile: bool = False):
    """The image of one leg, before :func:`image_for` caches it.

    The image holds the toolkit, the tree, the installed package, and the
    external repositories. It holds no compiled artifact: a Modal Volume cannot
    reach an image build, so a cache in the image would be cold on every
    rebuild, and :func:`_ensure_build` builds in the container instead.
    """
    image = _toolkit_image(leg, profile=profile)
    # The manifest, the tooling, and the fetch come before the tree, so a source
    # change rebuilds one layer and not three (docs/build.md, the Python distribution).
    image = _manifest_layer(image, leg)
    return (
        image
        .add_local_dir(str(REPO), remote_path=REPO_ROOT, copy=True,
                       ignore=list(IMAGE_IGNORE))
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
    (``docs/post-training.md`` section 7).

    A train job declares one replica per device in ``mesh.count``. A pipeline
    job declares no mesh, because the chain runs one replica of each stage
    (``docs/post-training.md`` B11), so the count the leg declares is the
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


@_function(image=image_for(T4), gpu=T4.gpu, cpu=CPU_CORES, timeout=1800)
def doctor() -> str:
    """Report the device, the architecture, and the sandbox of the worker.

    The timeout matches a card job, because the container builds its leg before
    this body runs, and a cold cache makes that build the long part.
    """
    result = subprocess.run(["tools/nanochat", "doctor"],
                            capture_output=True, text=True, check=False)
    return (result.stdout + result.stderr).strip()


#: The longest a warm build may take. A cold cache compiles every target of one
#: leg in a container with four cores, and the first run of a leg pays it. The
#: bound is generous because the call is one-off, and a card is not needed to
#: compile: ``nvcc`` crosses the architecture on any host.
WARM_TIMEOUT_SECONDS = 60 * 60


@_function(image=image_for(T4), cpu=CPU_CORES, timeout=WARM_TIMEOUT_SECONDS)
def warm_t4() -> str:
    """Build the T4 leg into the cache Volume and return the cache path."""
    return os.environ.get("NANOCHAT_BAZEL_CACHE", "")


@_function(image=image_for(H100), cpu=CPU_CORES, timeout=WARM_TIMEOUT_SECONDS)
def warm_h100() -> str:
    """Build the H100 leg into the cache Volume and return the cache path."""
    return os.environ.get("NANOCHAT_BAZEL_CACHE", "")


@_function(image=image_for(H100_FP8), cpu=CPU_CORES,
           timeout=WARM_TIMEOUT_SECONDS)
def warm_h100_fp8() -> str:
    """Build the fp8 leg into the cache Volume and return the cache path."""
    return os.environ.get("NANOCHAT_BAZEL_CACHE", "")


#: The warm function of each leg, by leg name.
WARM_FUNCTIONS = {"t4": warm_t4, "h100": warm_h100, "h100-fp8": warm_h100_fp8}


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


@_function(
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


@_function(
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


def _run_in_image(command: list, argv_json: str, row_prefixes: tuple) -> str:
    """Run one in-image command and return its answer as one JSON document.

    ``argv_json`` is a JSON list of extra arguments for the program, so a caller
    selects a shape without a new deployment. ``row_prefixes`` names the output
    lines worth lifting out of the full text. The image sets
    ``NANOCHAT_SANDBOX_BACKEND=none``, so ``tools/nanochat`` runs the program in
    place and needs no queue.
    """
    argv = [str(item) for item in json.loads(argv_json or "[]")]
    full = [*command, *argv]
    result = subprocess.run(full, env=worker_env(package_source()),
                            capture_output=True, text=True, check=False)
    rows = [line for line in result.stdout.splitlines()
            if line.startswith(row_prefixes)]
    return json.dumps({"command": full, "status": int(result.returncode),
                       "rows": rows, "stdout": result.stdout.strip(),
                       "tail": process_tail(result, lines=8)},
                      sort_keys=True)


@_function(
    image=image_for(H100),
    gpu=H100.gpu,
    cpu=CPU_CORES,
    timeout=TIMEOUT_SECONDS,
    max_containers=1,
)
def bench_h100(argv_json: str = "[]") -> str:
    """Time the Muon benchmark on the H100 leg and return what it printed.

    The benchmark runs in the H100 image, so the rate belongs to the library
    that image built: sm_90 at bf16, the same build the chain runs. ``argv_json``
    is a JSON list of the benchmark's own arguments, such as
    ``["--profile", "big", "--muon"]``, so a caller selects the shape without
    a new deployment.

    It answers one JSON document with the command, the exit status, the
    benchmark's report lines, and the tail of its output. The benchmark needs
    no GPU queue in the container: ``NANOCHAT_SANDBOX_BACKEND`` is ``none``
    there, so ``tools/nanochat bench`` runs the binary in place (section 7).
    """
    return _run_in_image(["tools/nanochat", "bench", "--", BENCH_BIN],
                         argv_json, ("polar:", "muon:"))


@_function(
    image=image_for(H100),
    gpu=H100.gpu,
    cpu=CPU_CORES,
    timeout=1800,
    max_containers=1,
)
def check_attention_h100() -> str:
    """Run the attention family test on the H100 leg and return its output.

    The image built the test binary from the same source and the same flags as
    the library, so this is the bf16 tensor-core tile of ``docs/attention.md``
    Part 6 measured against the host reference on the card the tile targets. The
    test takes the sandbox profile ``t1-gpu``; the image sets
    ``NANOCHAT_SANDBOX_BACKEND=none``, so the profile is a marker and the binary
    runs in place.
    """
    return _run_in_image(["tools/nanochat", "run", "t1-gpu", "--",
                          ATTENTION_TEST_BIN], "[]", ())


@_function(
    image=image_for(H100),
    gpu=H100.gpu,
    cpu=CPU_CORES,
    timeout=1800,
    max_containers=1,
)
def bench_attention_h100(argv_json: str = "[]") -> str:
    """Time the attention benchmark on the H100 leg and return what it printed.

    The mirror of :func:`bench_h100` for the attention family. The benchmark
    runs in the H100 image, so every row belongs to the library that image
    built: ``sm_90`` at bf16, the same build the chain runs, which is the build
    in which the tensor-core tile replaces the dense cuBLAS forward on the
    windowed layers (``docs/attention.md`` Part 6).

    ``argv_json`` is a JSON list of the benchmark's own arguments, so a caller
    selects the shape without a new deployment.
    """
    return _run_in_image(["tools/nanochat", "bench", "--",
                          ATTENTION_BENCH_BIN], argv_json, ("attention:",))


@_function(
    image=image_for(H100),
    gpu=H100.gpu,
    cpu=CPU_CORES,
    timeout=1800,
    max_containers=1,
)
def check_gemm_h100() -> str:
    """Run the hand tensor-core GEMM device gate on the H100 leg.

    The image built the test from the same source and the same flags as the
    library, so this is the bf16 tile of ``backends/cuda/gemm/`` measured
    against cuBLAS on the card it targets: every contract case of
    ``docs/tensor-core-gemm-plan.md`` section B.2, and every schedule of the
    sweep (section B.4). A T4 image would refuse the run, because the tile is
    compiled only in a bf16 build.
    """
    return _run_in_image(["tools/nanochat", "run", "t1-gpu", "--",
                          GEMM_TEST_BIN], "[]", ())


@_function(
    image=image_for(H100),
    gpu=H100.gpu,
    cpu=CPU_CORES,
    timeout=1800,
    max_containers=1,
)
def bench_gemm_h100(argv_json: str = "[]") -> str:
    """Time the hand tensor-core GEMM against cuBLAS on the H100 leg.

    The mirror of :func:`bench_attention_h100` for the hand GEMM. The report
    holds one row per shape of the closed list, twice: once with the tile and
    once with cuBLAS, through the shipped ``kernels::Gemm`` and the
    ``NANOCHAT_GEMM_LIB`` switch, so the ratio of the two is the fraction the
    promotion gate reads (``docs/tensor-core-gemm-plan.md`` section B.5).

    ``argv_json`` is a JSON list of the benchmark's own arguments, such as
    ``["--schedule", "all", "--json"]``, so a caller sweeps the five schedules
    without a new deployment.
    """
    return _run_in_image(["tools/nanochat", "bench", "--", GEMM_BENCH_BIN],
                         argv_json, ("gemm:",))


@_function(
    image=image_for(T4),
    gpu=T4.gpu,
    cpu=CPU_CORES,
    timeout=1800,
    max_containers=1,
)
def check_gemm_fp32_t4() -> str:
    """Run the fp32 shared-memory tile's device gate on the T4 leg.

    The tile of ``backends/cuda/gemm/gemm_fp32.cu`` against a host reference in
    double, over the four operand forms, a k tail, a batch above one, the alpha
    and beta terms, and both schedules (``docs/kernels.md`` Part 3 section 6
    phase 3). The T4 image is the one that builds it: the leg is fp32 at
    ``sm_75``.
    """
    return _run_in_image(["tools/nanochat", "run", "t1-gpu", "--",
                          GEMM_FP32_TEST_BIN], "[]", ())


@_function(
    image=image_for(T4),
    gpu=T4.gpu,
    cpu=CPU_CORES,
    timeout=1800,
    max_containers=1,
)
def bench_gemm_fp32_t4(argv_json: str = "[]") -> str:
    """Time the fp32 shared-memory tile against cuBLAS on the T4 leg.

    The same protocol and the same rows as :func:`bench_gemm_h100`, on the leg
    whose compute type is fp32. The ``hand`` arm routes through the seam's
    ``NANOCHAT_GEMM_LIB`` switch to the fp32 tile, and the ``cublas`` arm is the
    reference, so the ratio of the two columns is the fraction the phase 3 gate
    of ``docs/kernels.md`` reads.

    ``argv_json`` is a JSON list of the benchmark's own arguments, such as
    ``["--rows", "production", "--schedule", "all", "--json"]``.
    """
    return _run_in_image(["tools/nanochat", "bench", "--", GEMM_BENCH_BIN],
                         argv_json, ("gemm:",))


@_function(
    image=image_for(H100_FP8),
    gpu=H100_FP8.gpu,
    cpu=CPU_CORES,
    timeout=1800,
    max_containers=1,
)
def bench_gemm_fp8_h100(argv_json: str = "[]") -> str:
    """Time the fp8 arm against cublasLt on the fp8 H100 leg.

    The same sweep as :func:`bench_gemm_h100`, on the image that builds at fp8,
    with the benchmark's ``--fp8`` arm added. That arm quantizes each row's own
    operands and times the call twice: once on cublasLt, which ships, and once
    on the Hopper mainloop of ``cutlass_fp8.cu``. The ratio of the two columns is
    the fraction the fp8 gate of ``docs/kernels.md`` section 10.4 reads.

    ``argv_json`` is a JSON list of the benchmark's own arguments, such as
    ``["--rows", "production", "--json"]``, so the run states its rows without
    a new deployment.
    """
    return _run_in_image(
        ["tools/nanochat", "bench", "--", GEMM_BENCH_BIN, "--fp8"], argv_json,
        ("gemm:",))


@_function(
    image=image_for(H100),
    gpu=H100.gpu,
    cpu=CPU_CORES,
    timeout=1800,
    max_containers=1,
)
def check_mma_h100() -> str:
    """Run the phase 1 micro-tile gate on the H100 leg.

    The image built the test from the same source and the same flags as the
    library, so the device row is the `ldmatrix` and `mma.sync` pair in one warp
    on the card, checked against the same reference as the host row. A row for
    `transpose_a` and `transpose_b` each of the four combinations, because the
    staged layouts of `gemm_tile.cuh` produce all four.
    """
    return _run_in_image(["tools/nanochat", "run", "t1-gpu", "--",
                          GEMM_MMA_BIN], "[]", ("mma_bf16",))


def prototype_probe(target: str, binary: str, argv_json: str = "[]") -> str:
    """Build one prototype in the container, then run it with the caller's argv.

    A prototype needs an architecture the leg's own build does not emit, so the
    build comes first and inside the container: one build of one target is
    cheaper than a second image. The target states that architecture for itself,
    so the command names no flag.

    The program reports a ``nanochat.bench.v3`` document, so the answer carries
    it in ``stdout`` and no line prefix is lifted. ``argv_json`` is the probe's
    own arguments, such as ``["--rows", "phase0", "--json"]``, so a caller
    selects a row set without a new deployment.
    """
    build = subprocess.run(
        ["bash", "-c", probe_build_command(H100, target)],
        env=worker_env(package_source()), capture_output=True, text=True,
        check=False)
    if build.returncode != 0:
        return json.dumps({"stage": "build",
                           "status": int(build.returncode),
                           "tail": process_tail(build, lines=20)},
                          sort_keys=True)
    return _run_in_image(["tools/nanochat", "run", "t1-gpu", "--", binary],
                         argv_json, ())


@_function(
    image=image_for(H100),
    gpu=H100.gpu,
    cpu=CPU_CORES,
    timeout=3600,
    max_containers=1,
)
def bench_wgmma_h100(argv_json: str = "[]") -> str:
    """Build and run the WGMMA probe on the H100 leg, both row sets.

    One binary answers two phases. The ``wgmma:`` rows are Phase 0 of
    ``docs/kernels.md`` Part 3: how fast one warpgroup issues the warpgroup
    matrix multiply-accumulate from a resident shared-memory tile, and what
    share of its SM that is. The ``attention-wgmma:`` rows are Phase H1 of
    ``docs/attention.md`` Part 8: the operand form, the two operand maps, and
    the rate of a key tile that the copy engine stages.

    ``argv_json`` is the probe's own arguments, such as
    ``["--rows", "phase0", "--json"]``, so one gate reads its own rows from
    the one document.

    The build comes first and it is inside the container, because the image
    built every other target at ``sm_90`` and this probe needs ``sm_90a``.
    """
    return prototype_probe(WGMMA_BENCH_TARGET, WGMMA_BENCH_BIN, argv_json)


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


def base_shape(job) -> tuple:
    """The base stage's batch triple: rows per device, accumulation, devices.

    The three fields have one home, the chain's own
    :class:`nanochat_cpp.pipeline.Job`, so the roofline command models the batch
    the stages ran and never a copy of it. One missing field reads as the job's
    default, which is one row on one device, so an older job file still answers.
    """
    from nanochat_cpp import pipeline as chain  # noqa: PLC0415 - image only

    job = chain.Job.from_dict(dict(job))
    return int(job.batch), int(job.grad_accum), int(job.devices)


def roofline_argv(leg: Leg, job: Mapping) -> list:
    """The ``tools/nanochat roofline`` command gate G3 runs.

    The device is the leg's card and the precision is the leg's precision, so
    the prediction is the bound of the hardware that ran: ``--device h100
    --precision bf16`` for the H100 leg (section 8). The model flags come from
    ``nanochat_cpp.api.model_flags``, the one Python home of the model flag
    list, and the batch triple is the chain's own, so the shape the tool models
    is the shape the stages ran.
    """
    batch, grad_accum, devices = base_shape(job)
    return ["tools/nanochat", "roofline",
            "--device", leg.device,
            "--precision", leg.precision,
            "--batch", str(batch),
            "--grad-accum", str(grad_accum),
            "--devices", str(devices),
            *model_flags(shape_of(job)),
            "--json"]


def predicted_step_ms(document: Mapping) -> tuple:
    """The step bound and the binding limit of one roofline JSON document.

    ``step_seconds`` is the sum of the rows plus the gradient sync, and the
    limit of the row with the largest bound is the limit the step time reads.
    Every row of a report at this shape answers ``launch``, so the chain is
    bound by launches and by the host program, not by arithmetic (section 2 of
    ``docs/post-training.md``). The step bound answers ``None`` when
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

    One job file runs the stages of ``docs/post-training.md`` section 1, and
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


@_function(
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


@_function(
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
# The corpus: the download and the tokenizer of the H100 leg
# ---------------------------------------------------------------------------

#: The directory of the runs Volume that holds the corpus of every job. It sits
#: beside a run directory, not inside one, because the corpus outlives one run
#: and a second job reads it again (``docs/post-training.md`` section 9
#: row 3). It is the same word as the run-level name above and a different
#: directory, so it has its own name.
CORPUS_VOLUME_DIRNAME = "corpus"

#: The characters the tokenizer trainer reads by default. A 32768-merge
#: vocabulary needs tens of millions of characters to use its merges, and 200
#: million is a few minutes of one core and a bounded first deploy.
TOKENIZER_CHARS = 200_000_000


#: The document one corpus download writes beside its shards. The chain reads
#: it as the manifest of the directory, so a shard left by an earlier download
#: cannot join a run (``docs/post-training.md`` section 9 row 3).
CORPUS_MANIFEST = "corpus.json"


def corpus_dir_name(urls: Sequence[str], name: str = "h100") -> str:
    """The Volume directory of one URL set: the name and a digest of the URLs.

    Two datasets must never share a directory. The FineWeb-Edu shard and the
    ClimbMix shard were both planned as ``00_<basename>.parquet`` under one
    directory keyed by the profile name, so the loader read the first in name
    order and answered ``only a one-column flat schema is supported`` for the
    other dataset's file. The digest keeps one URL set in one place and makes
    the download cache exact (section 9 row 3).
    """
    digest = hashlib.sha256("\n".join(str(url) for url in urls)
                            .encode("utf-8")).hexdigest()[:12]
    return f"{name}-{digest}"


def corpus_plan(urls: Sequence[str], root) -> dict:
    """The file each URL downloads to, and the tokenizer path beside them.

    The index prefix keeps the download order the order of the list, so the
    loader reads the shards the operator named in the order they named them. A
    URL whose last segment is not a parquet name takes a generated one, so a
    presigned link still lands somewhere sensible.
    """
    root = Path(root)
    files = []
    for index, url in enumerate(urls):
        name = str(url).rstrip("/").rsplit("/", 1)[-1]
        if not name.endswith(".parquet"):
            name = f"shard{index:03d}.parquet"
        files.append(root / f"{index:02d}_{name}")
    return {"dir": str(root),
            "files": [str(path) for path in files],
            "tokenizer": str(root / "tokenizer.nctoken")}


def download_argv(url: str, target) -> list:
    """The ``curl`` line that writes one corpus file.

    ``-f`` turns an HTTP error into a non-zero status, so a wrong URL fails the
    function instead of leaving an empty file that the tokenizer trainer then
    reads. ``-L`` follows the redirect Hugging Face answers with.
    """
    return ["curl", "-f", "-L", "--retry", "3", "--create-dirs",
            "--output", str(target), str(url)]


def tokenizer_argv(files: Sequence[str], out, vocab_size: int,
                   max_chars: int = TOKENIZER_CHARS) -> list:
    """The ``tok_train_main`` line that writes the ``NCTOKEN1`` artifact.

    The trainer runs through ``tools/nanochat``, which applies the sandbox and
    sets the sentinel every executable requires, so the corpus function uses
    the same entry point as the chain.
    """
    return ["tools/nanochat", "run", "t0-cpu", "--", TOKENIZER_BIN,
            "--parquet", ",".join(str(path) for path in files),
            "--text-column", "text",
            "--vocab-size", str(int(vocab_size)),
            "--max-chars", str(int(max_chars)),
            "--out", str(out)]


def tokenizer_ids(path) -> int:
    """The identifier count of an ``NCTOKEN1`` artifact, or 0 when it is absent.

    The count comes from the tree's own reader (``nanochat_cpp.data``), which
    needs no third party, so the corpus function can decide whether a cached
    artifact already matches the vocabulary it was asked for.
    """
    from nanochat_cpp import data  # noqa: PLC0415 - image only

    try:
        return len(data.token_byte_lengths(data.read_nctoken1(path)))
    except (OSError, ValueError):
        return 0


def process_tail(result, lines: int = 5) -> str:
    """The last lines a finished process wrote, for a failure message.

    A child that fails in a container writes its reason to stderr, and that
    stderr is the only place the reason exists. A bare exit status sends the
    operator to the Modal log for it; the tail puts the reason beside the
    status (``docs/post-training.md`` section 9 row 3).
    """
    text = "\n".join(part for part in (getattr(result, "stdout", ""),
                                      getattr(result, "stderr", "")) if part)
    kept = text.strip().splitlines()[-max(1, int(lines)):]
    return " | ".join(kept)


@_function(
    image=image_for(H100),
    cpu=CPU_CORES,
    volumes={REMOTE_ROOT: runs},
    secrets=_secrets(),
    timeout=TIMEOUT_SECONDS,
    max_containers=1,
)
def corpus(job_json: str, urls_json: str, name: str = "h100",
           vocab_size: int = 0, max_chars: int = TOKENIZER_CHARS) -> str:
    """Download a parquet corpus to the Volume and train its tokenizer.

    The corpus is the input of the H100 leg and it outlives one run, so it
    lives on the ``nanochat-runs`` Volume and a second job reuses it: a file the
    function downloaded before is not fetched again, and a tokenizer whose
    identifier count already matches ``vocab_size`` is not trained again. The
    returned document is what :func:`pipeline_demo` merges into the job as
    ``corpus_path`` and ``tokenizer_path``.

    ``vocab_size`` 0 reads the job's own profile here, in the container, which
    holds the installed package. The local entry point therefore needs no
    package of its own: the vocabulary, the profile, and the shape have one
    home, and it is the image.

    The parquet must hold one flat column named ``text``. The native reader
    refuses a multi-column schema, so a public dataset that ships its metadata
    beside the text must be reshaped first
    (``docs/data.md``). ``karpathy/climbmix-400b-shuffle`` is the
    corpus the tree already names, and ``shard_00000.parquet`` is 92 MB.

    This function holds no device. It runs on the H100 image because that image
    holds ``curl`` and the tokenizer trainer, and it commits the Volume so the
    answer survives the container.
    """
    urls = [str(url) for url in json.loads(urls_json)]
    if not urls:
        raise ValueError("corpus: give at least one parquet URL")
    if int(vocab_size) <= 0:
        vocab_size = int(shape_of(json.loads(job_json)).vocab_size)
    plan = corpus_plan(urls, Path(REMOTE_ROOT) / CORPUS_VOLUME_DIRNAME
                       / corpus_dir_name(urls, str(name)))
    Path(plan["dir"]).mkdir(parents=True, exist_ok=True)
    for url, target in zip(urls, plan["files"]):
        cached = Path(target)
        if cached.is_file() and cached.stat().st_size > 0:
            print(f"corpus: {cached.name} is on the Volume already",
                  file=sys.stderr)
            continue
        print(f"corpus: downloading {url}", file=sys.stderr)
        result = subprocess.run(download_argv(url, target), check=False,
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"corpus: curl exited {result.returncode} for {url}: "
                f"{process_tail(result)}")
    plan["vocab_size"] = int(vocab_size)
    plan["tokenizer_ids"] = tokenizer_ids(plan["tokenizer"])
    if plan["tokenizer_ids"] != int(vocab_size):
        print(f"corpus: training a {int(vocab_size)}-identifier tokenizer",
              file=sys.stderr)
        argv = tokenizer_argv(plan["files"], plan["tokenizer"], vocab_size,
                              max_chars)
        result = subprocess.run(argv, env=worker_env(package_source()),
                                check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"corpus: the tokenizer trainer exited {result.returncode}: "
                f"{process_tail(result)}")
        plan["tokenizer_ids"] = tokenizer_ids(plan["tokenizer"])
        if plan["tokenizer_ids"] != int(vocab_size):
            raise RuntimeError(
                f"corpus: the trainer wrote {plan['tokenizer_ids']} identifiers "
                f"and the job asks for {int(vocab_size)}; raise --max-chars")
    runs.commit()
    document = Path(plan["dir"]) / CORPUS_MANIFEST
    document.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")
    runs.commit()
    return json.dumps(plan, sort_keys=True)


def step_split_argv(job: Mapping, corpus: Mapping, steps: int = 6) -> list:
    """The ``train_main`` line that reports the phase split of one step.

    The trainer is the single-device one, because it is the only entry point
    that times the phases: ``data_ms``, ``fwd_ms``, ``bwd_ms``, ``sync_ms``, and
    ``opt_ms``. The shape and the batch come from the job, so the split is the
    split of the run being explained and not of a copy of it.
    """
    batch, grad_accum, devices = base_shape(job)
    if int(devices) != 1:
        raise ValueError(
            f"step_split: one device at a time, and the job names {devices}")
    files = [str(name) for name in (corpus or {}).get("files") or ()]
    tokenizer = str((corpus or {}).get("tokenizer") or "")
    if not files or not tokenizer:
        raise ValueError(
            "step_split: the corpus document needs `files` and `tokenizer`")
    return ["tools/nanochat", "train", "--", TRAIN_BIN, "train",
            *model_flags(shape_of(job)),
            "--batch", str(int(batch)),
            "--grad-accum", str(int(grad_accum)),
            "--num-iterations", str(int(steps)),
            "--log-every", "1",
            "--tokenizer", tokenizer,
            "--train-parquet", ",".join(files),
            "--text-column", "text",
            "--device", H100.device,
            "--hw-check", "off",
            "--checkpoint", f"{REMOTE_ROOT}/split/checkpoint.ckpt"]


@_function(
    image=image_for(H100),
    gpu=H100.gpu,
    cpu=CPU_CORES,
    volumes={REMOTE_ROOT: runs},
    secrets=_secrets(),
    timeout=TIMEOUT_SECONDS,
    max_containers=1,
)
def step_split(job_json: str, corpus_json: str, steps: int = 6) -> str:
    """Run one short training on the H100 and return its phase split.

    The chain reports one number per step, so a step far from its roofline bound
    cannot be attributed from the chain's report. This function runs the phase
    timer on the same card, at the same shape and batch, and returns the lines.
    It reads the corpus the ``corpus`` function left on the Volume, so it costs
    one container start and a few seconds of training.
    """
    job = json.loads(job_json)
    corpus = json.loads(corpus_json)
    Path(f"{REMOTE_ROOT}/split").mkdir(parents=True, exist_ok=True)
    argv = step_split_argv(job, corpus, steps)
    result = subprocess.run(argv, env=worker_env(package_source()),
                            capture_output=True, text=True, check=False)
    measured = [line for line in result.stdout.splitlines()
                if line.startswith("step ")]
    runs.commit()
    return json.dumps({"command": argv, "status": int(result.returncode),
                       "steps": measured,
                       "tail": process_tail(result, lines=8)}, sort_keys=True)


# ---------------------------------------------------------------------------
# The H100 profile: one Nsight Systems trace of the base stage
# ---------------------------------------------------------------------------

#: The directory of the trace inside one run directory.
PROFILE_DIRNAME = "profile"

#: The base stage alone, as one driver module. ``stage_base`` is the chain's
#: own base stage, so the trace belongs to the run being explained and not to a
#: copy of it. ``sys.argv`` carries the job file and the step count. It is a
#: file and not ``-c`` source, so the traced command line is the same shape a
#: person reads.
BASE_STAGE_DRIVER = (
    '"""Trace the base stage of one job file."""\n'
    "import dataclasses, json, sys\n"
    "from nanochat_cpp import pipeline as pl\n"
    "job = pl.Job.read(sys.argv[1])\n"
    "job = dataclasses.replace(job, base_steps=int(sys.argv[2]))\n"
    "stage = pl.stage_base(job)\n"
    "print(json.dumps(stage.extra, indent=1, sort_keys=True))\n"
    "print('profile: checkpoint', stage.checkpoint)\n"
)

#: The report names the trace is cut into. The two sums are small and the two
#: traces are the timeline: the GPU kernels and the CPU driver API calls. The
#: cuBLAS sum is asked for only when the trace collected that domain, because
#: the library call is where a small GEMM's host time hides.
NSYS_SUM_REPORTS = ("cuda_gpu_kern_sum", "cuda_api_sum", "osrt_sum",
                    "cuda_gpu_mem_time_sum")
NSYS_TRACE_REPORTS = ("cuda_gpu_trace", "cuda_api_trace")

#: The default trace domains and the default sampling mode. The CUDA activity
#: and the OS runtime calls answer a device question. A host question takes the
#: ``cublas`` domain, because a cuBLAS call is not a CUDA runtime call and the
#: time inside it is invisible to every other domain.
NSYS_TRACE_DOMAINS = "cuda,osrt"
NSYS_SAMPLE = "none"

#: The images of the legs with Nsight Systems on top, keyed by leg name. The
#: cache is separate from :data:`_IMAGES`, so a trace costs one layer and the
#: training functions keep the image they already build.
_PROFILE_IMAGES: dict = {}


def profile_image_for(leg: Leg):
    """The image of one leg with Nsight Systems installed.

    The profiler is an apt layer of the toolkit, below the tree copy, so a
    source change does not reinstall it. The trace is still a measurement of the
    library the chain ran, because the container builds that library itself.
    """
    built = _PROFILE_IMAGES.get(leg.name)
    if built is None:
        built = _PROFILE_IMAGES[leg.name] = _build_image(leg, profile=True)
    return built


def nsys_argv(stem, driver: Sequence[str],
              domains: str = NSYS_TRACE_DOMAINS,
              sample: str = NSYS_SAMPLE) -> list:
    """The ``nsys profile`` line that traces one driver command.

    ``domains`` selects the activity collected: the CUDA activity and the OS
    runtime calls by default, and ``cublas`` as well when the question is the
    host time inside a library call. ``sample`` is ``none`` for a device trace
    and ``cpu`` for a host trace; a CPU sample adds the stack of the thread, so
    a host cost inside the trainer's own code is attributed to a function.

    ``--force-overwrite`` lets a second trace of the same run replace the first
    instead of stopping for a question. The application follows the options
    with no ``--`` separator: version 2023.4.4 reads a bare ``--`` as an empty
    long option and rejects the whole line as ambiguous.
    """
    argv = ["nsys", "profile", "--force-overwrite=true",
            f"--output={stem}",
            f"--trace={domains}",
            f"--sample={sample}",
            "--cpuctxsw=none",
            "--cuda-memory-usage=false", "--stats=false"]
    if str(sample).lower() != "none":
        # A stack is what turns a host sample into an attribution. The frame
        # pointer is the cheap method, and a release build keeps enough of it
        # to name the trainer's own function.
        argv.append("--backtrace=fp")
    return [*argv, *driver]


def nsys_export_argv(report, stem) -> list:
    """The ``nsys export`` line that writes the SQLite copy of one trace."""
    return ["nsys", "export", "--force-overwrite=true", "--type=sqlite",
            f"--output={stem}.sqlite", str(report)]


def nsys_stats_argv(report, stem, reports) -> list:
    """The ``nsys stats`` line that writes the CSV reports of one trace."""
    return ["nsys", "stats", "--force-overwrite=true", "--format=csv",
            f"--report={','.join(reports)}", f"--output={stem}",
            str(report)]


def _run_step(argv, env, log) -> int:
    """Run one step of the profile and write its output to ``log``.

    A step that fails is recorded and the profile continues, because a missing
    timeline report costs one fact and never the trace the other steps wrote.
    """
    log.write(f"$ {' '.join(str(item) for item in argv)}\n")
    log.flush()
    try:
        result = subprocess.run(argv, env=env, stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    except OSError as error:
        log.write(f"modal: {error}\n")
        return 127
    log.write(f"modal: exit {result.returncode}\n")
    log.flush()
    return int(result.returncode)


def _csv_rows(path, limit: int = 25) -> list:
    """The rows of one ``nsys stats`` CSV file, the longest first.

    The two sum reports are sorted by their total time column, so a reader sees
    the cost a kernel or a call owns before the row that is a hundredth of it.
    """
    import csv  # noqa: PLC0415 - one small read of one report

    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)]
    total_key = next((name for name in (
        "Total Time (ns)", "Total Time", "Time (%)", "Total (MB)")
        if rows and name in rows[0]), None)
    if total_key is not None:
        def key(row):
            try:
                return float(str(row.get(total_key, 0)).replace(",", ""))
            except ValueError:
                return 0.0
        rows.sort(key=key, reverse=True)
    return rows[:limit]


@_function(
    image=profile_image_for(H100),
    gpu=H100.gpu,
    cpu=CPU_CORES,
    volumes={REMOTE_ROOT: runs},
    secrets=_secrets(),
    timeout=TIMEOUT_SECONDS,
    max_containers=1,
)
def profile_h100(job_json: str, run_id: str, steps: int = 20,
                 domains: str = NSYS_TRACE_DOMAINS,
                 sample: str = NSYS_SAMPLE) -> str:
    """Trace the base stage of one job file on the H100 leg and return it.

    The base stage is the chain's own :func:`nanochat_cpp.pipeline.stage_base`,
    so the trace is of the run the report describes: the same shape, the same
    batch, and the same driver. The trace is cut into the device timeline and
    the driver's own API calls, which is the one attribution the chain's
    ``step_ms`` and ``compute_ms`` cannot give.

    The artifacts land in ``/runs/<run_id>/profile``: the ``.nsys-rep`` report,
    its SQLite copy, the sum CSVs, and the two timeline CSVs. The answer names
    them with their sizes and carries the first rows of each sum, so a reader
    with the Volume has the numbers without a second container.

    ``steps`` keeps the trace short: the driver's start-up is one checkpoint
    write and its steady state is one shape, so twenty steps carry both.
    """
    source = package_source()
    job = json.loads(job_json)
    job["out_root"] = str(Path(REMOTE_ROOT) / str(run_id))
    job["base_steps"] = int(steps)
    job_json = json.dumps(job, sort_keys=True)
    _check_cards(H100, job_json)
    run_dir = Path(REMOTE_ROOT) / str(run_id)
    trace_dir = run_dir / PROFILE_DIRNAME
    trace_dir.mkdir(parents=True, exist_ok=True)
    job_path = run_dir / "job.json"
    job_path.write_text(job_json + "\n", encoding="utf-8")
    report = trace_dir / "base.nsys-rep"
    env = worker_env(source)

    stop = threading.Event()
    keeper = threading.Thread(target=_commit_loop,
                              args=(runs, stop, COMMIT_SECONDS), daemon=True)
    keeper.start()
    log_path = run_dir / "profile.log"
    driver_path = run_dir / "profile_driver.py"
    driver_path.write_text(BASE_STAGE_DRIVER, encoding="utf-8")
    driver = [sys.executable, str(driver_path), str(job_path),
              str(int(steps))]
    status = {}
    sums = list(NSYS_SUM_REPORTS)
    if "cublas" in str(domains):
        sums.append("cublas_api_sum")
    try:
        with log_path.open("w", encoding="utf-8") as log:
            status["profile"] = _run_step(
                nsys_argv(trace_dir / "base", driver, domains, sample),
                env, log)
            if report.is_file():
                status["sqlite"] = _run_step(
                    nsys_export_argv(report, trace_dir / "base"), env, log)
                status["sums"] = _run_step(
                    nsys_stats_argv(report, trace_dir / "base", sums),
                    env, log)
                status["traces"] = _run_step(
                    nsys_stats_argv(report, trace_dir / "timeline",
                                    NSYS_TRACE_REPORTS), env, log)
    finally:
        stop.set()
        runs.commit()

    files = []
    for path in sorted(trace_dir.iterdir()):
        files.append({"name": path.name, "bytes": path.stat().st_size})
    answer = {
        "leg": H100.name,
        "run_dir": str(run_dir),
        "trace_dir": str(trace_dir),
        "report": str(report),
        "steps": int(steps),
        "domains": str(domains),
        "sample": str(sample),
        "source": source,
        "status": status,
        "files": files,
        "kernel_sum": _csv_rows(trace_dir / "base_cuda_gpu_kern_sum.csv"),
        "api_sum": _csv_rows(trace_dir / "base_cuda_api_sum.csv"),
        "cublas_sum": _csv_rows(trace_dir / "base_cublas_api_sum.csv"),
        "osrt_sum": _csv_rows(trace_dir / "base_osrt_sum.csv"),
        "mem_sum": _csv_rows(trace_dir / "base_cuda_gpu_mem_time_sum.csv"),
        "log_tail": "\n".join(
            log_path.read_text(encoding="utf-8").splitlines()[-25:]),
    }
    return json.dumps(answer, sort_keys=True)


# ---------------------------------------------------------------------------
# The optimization loop: one stage, one device, one trace or one measurement
# ---------------------------------------------------------------------------

#: The stages the loop targets. Three of them train and one only measures, so
#: ``eval`` reports a throughput and no MFU.
PGO_STAGES = ("base", "sft", "rl", "eval")

#: One driver module for all four stages, so a stage costs a branch and not a
#: second file. It runs the chain's own stage function where the chain has one,
#: and it prints one JSON document whatever the stage: the loop reads that
#: document, so a stage that reports nothing is a fault the loop sees at once.
#:
#: A stage that the chain always runs after another one runs here on a fresh
#: model. The trace measures a throughput, and the shapes are the same whether
#: the weights came from a checkpoint or from the initializer.
PGO_DRIVER = (
    '"""Run one stage of one job file for the optimization loop."""\n'
    "import dataclasses, json, sys, time\n"
    "from pathlib import Path\n"
    "from nanochat_cpp import api, pipeline as pl, rl, sft\n"
    "\n"
    "\n"
    "def metric_times(path):\n"
    "    \"\"\"The per-step seconds a driver wrote, in milliseconds.\"\"\"\n"
    "    times = []\n"
    "    if not path or not Path(path).is_file():\n"
    "        return times\n"
    "    for line in Path(path).read_text(encoding='utf-8').splitlines():\n"
    "        line = line.strip()\n"
    "        if not line:\n"
    "            continue\n"
    "        try:\n"
    "            record = json.loads(line)\n"
    "        except ValueError:\n"
    "            continue\n"
    "        value = record.get('step_seconds')\n"
    "        if value:\n"
    "            times.append(1000.0 * float(value))\n"
    "    return times\n"
    "\n"
    "\n"
    "def deltas(marks):\n"
    "    \"\"\"The gap between successive timestamps, in milliseconds.\"\"\"\n"
    "    return [1000.0 * (after - before)\n"
    "            for before, after in zip(marks, marks[1:])]\n"
    "\n"
    "\n"
    "def run(stage, job, steps):\n"
    "    \"\"\"One stage of one job at a step budget.\"\"\"\n"
    "    shape = pl.config(job)\n"
    "    corpus = pl.build_corpus(job)\n"
    "    answer = {'stage': stage, 'steps': int(steps)}\n"
    "    if stage == 'base':\n"
    "        record = pl.stage_base(dataclasses.replace(job, base_steps=steps))\n"
    "        answer['stage_seconds'] = float(record.seconds)\n"
    "        answer['step_ms'] = metric_times(record.extra.get('metrics'))\n"
    "        answer['mfu'] = record.extra.get('mfu')\n"
    "        answer['tokens_per_second'] = record.extra.get('tokens_per_sec')\n"
    "        return answer\n"
    "    tokenizer = api.Tokenizer.load(corpus['tokenizer'])\n"
    "    if stage == 'sft':\n"
    "        folder = Path(str(pl._run_dir(job))) / 'sft'\n"
    "        folder.mkdir(parents=True, exist_ok=True)\n"
    "        shard = folder / 'pgo_rows.nchsft'\n"
    "        sft.pack(shard, pl._conversations(corpus), tokenizer, shape.seq_len)\n"
    "        started = time.monotonic()\n"
    "        status = sft.run(shard, folder / 'pgo.ckpt', config=shape,\n"
    "                         num_iterations=int(steps), batch=pl.SFT_BATCH,\n"
    "                         seq_len=shape.seq_len, seed=int(job.seed))\n"
    "        answer['stage_seconds'] = time.monotonic() - started\n"
    "        answer['status'] = int(status)\n"
    "        return answer\n"
    "    if stage == 'rl':\n"
    "        prompts = pl._conversations(corpus)[:int(job.num_prompts)]\n"
    "        marks = [time.monotonic()]\n"
    "\n"
    "        def on_step(step, loss, grad_norm, rewards):\n"
    "            marks.append(time.monotonic())\n"
    "\n"
    "        report = rl.run(pl.GradedReward(), prompts,\n"
    "                        num_samples=int(job.num_samples), steps=int(steps),\n"
    "                        config=shape, max_tokens=int(job.max_tokens),\n"
    "                        temperature=float(job.temperature),\n"
    "                        top_k=int(job.top_k), seed=int(job.seed),\n"
    "                        on_step=on_step, save_final=False)\n"
    "        answer['step_ms'] = deltas(marks)\n"
    "        answer['stage_seconds'] = sum(answer['step_ms']) / 1000.0\n"
    "        answer['mean_rewards'] = [float(value)\n"
    "                                  for value in report.mean_rewards]\n"
    "        return answer\n"
    "    model = api.Model(shape)\n"
    "    started = time.monotonic()\n"
    "    report = api.evaluate(model, corpus['parquet'], tokenizer,\n"
    "                          steps=int(steps), batch=pl.EVAL_BATCH,\n"
    "                          seed=int(job.seed))\n"
    "    answer['stage_seconds'] = max(time.monotonic() - started, 1e-9)\n"
    "    answer['samples_per_second'] = float(steps) / answer['stage_seconds']\n"
    "    answer['bpb'] = float(report.bpb)\n"
    "    return answer\n"
    "\n"
    "\n"
    "def main():\n"
    "    stage, job_path, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])\n"
    "    job = pl.Job.read(job_path)\n"
    "    print(json.dumps(run(stage, job, steps), sort_keys=True), flush=True)\n"
    "\n"
    "\n"
    "main()\n"
)


def pgo_driver_argv(driver_path, stage, job_path, steps) -> list:
    """The command the container runs for one stage of one job."""
    return [sys.executable, str(driver_path), str(stage), str(job_path),
            str(int(steps))]


def _timeout_run(argv, env, log, seconds: int) -> int:
    """Run one command with a wall-clock cap, and stop it politely.

    Nsight Systems writes its report when the application exits, so the cap
    sends ``SIGINT`` to the process group first. That is the signal the driver
    already handles: it saves its checkpoint and returns. A process still alive
    after the grace period is killed, and the answer says which happened.
    """
    log.write(f"$ {' '.join(str(item) for item in argv)}\n")
    log.flush()
    try:
        started = time.monotonic()
        process = subprocess.Popen(argv, env=env, stdout=log,
                                   stderr=subprocess.STDOUT,
                                   start_new_session=True)
    except OSError as error:
        log.write(f"modal: {error}\n")
        return 127
    try:
        return int(process.wait(timeout=max(30, int(seconds))))
    except subprocess.TimeoutExpired:
        log.write(f"modal: the cap of {seconds} s reached; sending SIGINT\n")
        log.flush()
        for signal_number, grace in ((signal.SIGINT, 60), (signal.SIGKILL, 15)):
            try:
                os.killpg(os.getpgid(process.pid), signal_number)
            except OSError:
                break
            try:
                return int(process.wait(timeout=grace))
            except subprocess.TimeoutExpired:
                continue
        return -int(time.monotonic() - started)


def _pgo_paths(job: dict, run_id: str, leg: Leg, stage: str, steps: int):
    """The run directory, the job file it froze, and the driver it runs."""
    job["out_root"] = str(Path(REMOTE_ROOT) / str(run_id))
    run_dir = Path(REMOTE_ROOT) / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    job_path = run_dir / "job.json"
    job_path.write_text(json.dumps(job, sort_keys=True) + "\n",
                        encoding="utf-8")
    driver_path = run_dir / "pgo_driver.py"
    driver_path.write_text(PGO_DRIVER, encoding="utf-8")
    return run_dir, job_path, driver_path


def _trace_on(leg: Leg, stage: str, job_json: str, run_id: str,
              steps: int, seconds: int, domains: str = NSYS_TRACE_DOMAINS,
              sample: str = NSYS_SAMPLE) -> str:
    """Trace one stage of one job file on one leg and return the artifacts.

    The driver is the chain's own stage function, so the trace belongs to the
    run the report describes. The cap of ``seconds`` arrives as a signal and not
    as a kill, so a run that the cap stops still writes its report.
    """
    if stage not in PGO_STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    source = package_source()
    job = json.loads(job_json)
    _check_cards(leg, json.dumps(job, sort_keys=True))
    run_dir, job_path, driver_path = _pgo_paths(job, run_id, leg, stage, steps)
    trace_dir = run_dir / PROFILE_DIRNAME
    trace_dir.mkdir(parents=True, exist_ok=True)
    report = trace_dir / f"{stage}.nsys-rep"
    env = worker_env(source)
    log_path = run_dir / "profile.log"
    status = {}
    sums = list(NSYS_SUM_REPORTS)
    if "cublas" in str(domains):
        sums.append("cublas_api_sum")
    stop = threading.Event()
    keeper = threading.Thread(target=_commit_loop,
                              args=(runs, stop, COMMIT_SECONDS), daemon=True)
    keeper.start()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            status["profile"] = _timeout_run(
                nsys_argv(trace_dir / stage,
                          pgo_driver_argv(driver_path, stage, job_path, steps),
                          domains, sample), env, log, seconds)
            if report.is_file():
                status["sqlite"] = _run_step(
                    nsys_export_argv(report, trace_dir / stage), env, log)
                status["sums"] = _run_step(
                    nsys_stats_argv(report, trace_dir / stage, sums), env, log)
                status["traces"] = _run_step(
                    nsys_stats_argv(report, trace_dir / "timeline",
                                    NSYS_TRACE_REPORTS), env, log)
    finally:
        stop.set()
        runs.commit()
    files = [{"name": path.name, "bytes": path.stat().st_size}
             for path in sorted(trace_dir.iterdir())]
    answer = {"leg": leg.name, "stage": stage, "steps": int(steps),
              "seconds": int(seconds), "domains": str(domains),
              "sample": str(sample), "source": source,
              "run_dir": str(run_dir), "trace_dir": str(trace_dir),
              "report": str(report), "status": status, "files": files,
              "kernel_sum": _csv_rows(trace_dir / f"{stage}_cuda_gpu_kern_sum.csv"),
              "api_sum": _csv_rows(trace_dir / f"{stage}_cuda_api_sum.csv"),
              "log_tail": "\n".join(
                  log_path.read_text(encoding="utf-8").splitlines()[-20:])}
    return json.dumps(answer, sort_keys=True)


def _measure_on(leg: Leg, stage: str, job_json: str, run_id: str,
                steps: int, seconds: int, times: int = 2) -> str:
    """Run one stage untraced, ``times`` times, and return what it reported.

    The profiler costs about 28 percent of the step rate, so the loop scores
    this answer and never the trace. Each run gets its own output root, so a
    metric file of one run cannot be read as the metric file of the next.
    """
    if stage not in PGO_STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    source = package_source()
    base_job = json.loads(job_json)
    _check_cards(leg, json.dumps(base_job, sort_keys=True))
    env = worker_env(source)
    run_dir = Path(REMOTE_ROOT) / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "measure.log"
    results = []
    stop = threading.Event()
    keeper = threading.Thread(target=_commit_loop,
                              args=(runs, stop, COMMIT_SECONDS), daemon=True)
    keeper.start()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            for index in range(max(1, int(times))):
                job = dict(base_job)
                driver_path = run_dir / f"pgo_driver_{index}.py"
                driver_path.write_text(PGO_DRIVER, encoding="utf-8")
                job["out_root"] = str(run_dir / f"run-{index}")
                job_path = run_dir / f"job-{index}.json"
                job_path.write_text(json.dumps(job, sort_keys=True) + "\n",
                                    encoding="utf-8")
                code = _timeout_run(
                    pgo_driver_argv(driver_path, stage, job_path, steps),
                    env, log, seconds)
                answer = _last_json_block(
                    log_path.read_text(encoding="utf-8"))
                results.append({"run": index, "status": int(code),
                                **(answer or {})})
    finally:
        stop.set()
        runs.commit()
    document = {"leg": leg.name, "stage": stage, "steps": int(steps),
                "seconds": int(seconds), "source": source,
                "run_dir": str(run_dir), "runs": results,
                "log_tail": "\n".join(
                    log_path.read_text(encoding="utf-8").splitlines()[-20:])}
    return json.dumps(document, sort_keys=True)


def _last_json_block(text: str):
    """The last JSON object on one or more lines of a log."""
    for line in reversed(str(text or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


@_function(
    image=profile_image_for(H100),
    gpu=H100.gpu,
    cpu=CPU_CORES,
    volumes={REMOTE_ROOT: runs},
    secrets=_secrets(),
    timeout=TIMEOUT_SECONDS,
    max_containers=1,
)
def profile_stage_h100(stage: str, job_json: str, run_id: str, steps: int = 50,
                       seconds: int = 180, domains: str = NSYS_TRACE_DOMAINS,
                       sample: str = NSYS_SAMPLE) -> str:
    """Trace one stage of one job file on the H100 leg."""
    return _trace_on(H100, stage, job_json, run_id, steps, seconds, domains,
                     sample)


@_function(
    image=profile_image_for(T4),
    gpu=T4.gpu,
    cpu=CPU_CORES,
    volumes={REMOTE_ROOT: runs},
    secrets=_secrets(),
    timeout=TIMEOUT_SECONDS,
    max_containers=1,
)
def profile_stage_t4(stage: str, job_json: str, run_id: str, steps: int = 50,
                     seconds: int = 180, domains: str = NSYS_TRACE_DOMAINS,
                     sample: str = NSYS_SAMPLE) -> str:
    """Trace one stage of one job file on the T4 leg."""
    return _trace_on(T4, stage, job_json, run_id, steps, seconds, domains,
                     sample)


@_function(
    image=profile_image_for(H100),
    gpu=H100.gpu,
    cpu=CPU_CORES,
    volumes={REMOTE_ROOT: runs},
    secrets=_secrets(),
    timeout=TIMEOUT_SECONDS,
    max_containers=1,
)
def measure_stage_h100(stage: str, job_json: str, run_id: str,
                       steps: int = 50, seconds: int = 180,
                       times: int = 2) -> str:
    """Measure one stage of one job file on the H100 leg, untraced.

    It takes the profile image so one iteration builds one image: the tree hash
    decides the build, and a second image would double it for no gain.
    """
    return _measure_on(H100, stage, job_json, run_id, steps, seconds, times)


@_function(
    image=profile_image_for(T4),
    gpu=T4.gpu,
    cpu=CPU_CORES,
    volumes={REMOTE_ROOT: runs},
    secrets=_secrets(),
    timeout=TIMEOUT_SECONDS,
    max_containers=1,
)
def measure_stage_t4(stage: str, job_json: str, run_id: str,
                     steps: int = 50, seconds: int = 180,
                     times: int = 2) -> str:
    """Measure one stage of one job file on the T4 leg, untraced."""
    return _measure_on(T4, stage, job_json, run_id, steps, seconds, times)


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
    ``compute_ms``, ``data_ms``, and ``other_ms`` are the split of that step and
    ``window_steps`` names the steady window it was read over, so a reader of
    the two-leg document sees where the time went instead of one total. A
    report without a base stage answers ``None`` for every number, and a gate
    that reads one fails with a reason instead of comparing a guess.
    """
    extra = dict(stage_facts(report, "base").get("extra") or {})
    return {"step_ms": _number(extra.get("step_ms")),
            "compute_ms": _number(extra.get("compute_ms")),
            "data_ms": _number(extra.get("data_ms")),
            "other_ms": _number(extra.get("other_ms")),
            "window_steps": _number(extra.get("window_steps")),
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


def gates_of(t4_report, h100_report, roofline=None, pair=True) -> list:
    """The gates of section 8, in order: G1, G2, and G3.

    G2 compares the two legs, so a single-leg run has no G2 to answer. ``pair``
    names the request and not the data: a two-leg run whose T4 container failed
    still answers G2 as a failure with its reason, instead of silently dropping
    the gate that would have caught the regression.
    """
    gates = [gate_bf16_ran(h100_report)]
    if pair:
        gates.append(gate_step_ratio(t4_report, h100_report))
    gates.append(gate_roofline(h100_report, roofline))
    return gates


def gate_failures(document: Mapping) -> list:
    """The names of the gates that did not pass in one printed document."""
    return [str(gate.get("gate"))
            for gate in (document or {}).get("gates") or ()
            if str(gate.get("status")) != "pass"]


def demo_document(answers: Mapping, legs=LEG_ORDER) -> dict:
    """The one document the entry point prints: legs, comparison, and gates.

    A leg whose answer carries ``error`` still contributes its summary, so a
    failed H100 leg leaves the T4 report readable. The gates read the two
    reports and the H100 leg's roofline record, and a missing one is a failure
    with a reason rather than a crash, so the document always has its three
    keys (section 7).

    ``legs`` names the legs the run asked for. A single-leg request answers G1
    and G3 and leaves G2 out, because the gate compares the two legs and a run
    that started one of them cannot answer it (section 8).
    """
    selected = tuple(legs)
    names = {leg.name for leg in selected}
    summaries = {}
    for leg in selected:
        answer = dict((answers or {}).get(leg.name) or {})
        summaries[leg.name] = leg_summary(
            leg, answer, error=str(answer.get("error") or ""))
    t4_report = dict(((answers or {}).get(T4.name) or {}).get("report") or {})
    h100_answer = dict((answers or {}).get(H100.name) or {})
    h100_report = dict(h100_answer.get("report") or {})
    return {"legs": summaries,
            "comparison": comparison(summaries),
            "gates": gates_of(t4_report, h100_report,
                              h100_answer.get("roofline"),
                              pair=T4.name in names and H100.name in names)}


def pipeline_function(leg: Leg):
    """The deployed chain function of one leg."""
    return {T4.name: pipeline, H100.name: pipeline_h100}[leg.name]


def run_leg(leg: Leg, job_json: str, run_id: str) -> dict:
    """Run one leg's chain on Modal and return its parsed answer document."""
    return json.loads(pipeline_function(leg).remote(job_json, str(run_id)))


def run_legs(job_json: str, run_id: str, runner=None, legs=LEG_ORDER) -> dict:
    """Run the selected legs, T4 first, and return one answer per leg name.

    The order is the promise of section 7: the T4 artifact is on the Volume
    before the H100 container starts, so a reviewer reads the cheap leg while
    the expensive one runs, and an H100 failure leaves the T4 report intact.
    The run ids are ``<run_id>-t4`` and ``<run_id>-h100``, so both reports stay
    on the ``nanochat-runs`` Volume. A leg that raises answers with ``error``
    and the other leg still runs, so one container's failure is a line in the
    document and not a lost run. ``legs`` selects which legs to start, so a
    throughput run on one card starts one container.
    """
    runner = run_leg if runner is None else runner
    answers = {}
    for leg in legs:
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
    "batch": 1,
    "grad_accum": 1,
    "devices": 1,
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
    "corpus_path": "",
    "tokenizer_path": "",
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


def _print_answer(answer_json: str, what: str) -> None:
    """Print one job's answer, and fail the run when its status is not zero."""
    answer = json.loads(answer_json)
    print(json.dumps(answer, indent=1, sort_keys=True))
    if int(answer.get("status", 1)) != 0:
        print(f"modal: {what} reported a failure", file=sys.stderr)
        raise SystemExit(1)


#: One thing the card entrypoint can run: the Modal Function, the message a
#: failure prints, and whether the job reports a JSON document of its own
#: arguments.
@dataclass(frozen=True)
class CardJob:
    job: object
    what: str
    takes_argv: bool = False


#: The card jobs, by the name a command spells. A name is `<family>_<kind>`:
#:   `_check`  a test target that compares against a reference;
#:   `_bench`  a rate report that states a `nanochat.bench.v3` document and
#:             takes `--argv`;
#:   `_probe`  unused at present, kept for a prototype that reports neither.
#: Each one is the gate or the benchmark of one tensor-core family: the tile's
#: own test against its reference, and the rate report whose rows a promotion
#: gate reads. One job may carry two row sets, which is why `wgmma_bench` takes
#: `--rows`.
CARD_JOBS = {
    "attention_check": CardJob(check_attention_h100, "the attention test"),
    "attention_bench": CardJob(bench_attention_h100, "the attention benchmark",
                               takes_argv=True),
    "gemm_check": CardJob(check_gemm_h100, "the hand GEMM device gate"),
    "gemm_bench": CardJob(bench_gemm_h100, "the hand GEMM benchmark",
                          takes_argv=True),
    "gemm_bench_fp8": CardJob(bench_gemm_fp8_h100, "the fp8 arm benchmark",
                              takes_argv=True),
    "gemm_fp32_check": CardJob(check_gemm_fp32_t4,
                               "the fp32 tile device gate"),
    "gemm_fp32_bench": CardJob(bench_gemm_fp32_t4, "the fp32 tile benchmark",
                               takes_argv=True),
    "mma_check": CardJob(check_mma_h100, "the Part 3 phase 1 micro-tile gate"),
    "wgmma_bench": CardJob(
        bench_wgmma_h100,
        "the WGMMA probe: the Part 3 phase 0 ceiling rows and the Part 8 H1 "
        "copy-path rows, from one binary",
        takes_argv=True),
}


@app.local_entrypoint()
def card_job(name: str = "", argv: str = "[]") -> None:
    """Run one card job on its leg and print its answer.

        modal run tools/modal/app.py::card_job
        modal run tools/modal/app.py::card_job --name gemm_check
        modal run tools/modal/app.py::card_job --name gemm_bench \\
            --argv '["--schedule","all","--json"]'
        modal run tools/modal/app.py::card_job --name gemm_bench_fp8 \\
            --argv '["--rows","production","--json"]'

    One entrypoint serves the gates and benchmarks, because the wrapper is the
    whole point and it is the same for all of them: print the answer and turn a
    non-zero status into a failed run. A ``modal run`` of a Function target does
    neither. ``argv`` is a JSON list of the job's own arguments, for the jobs
    that report a document, and a name the table does not hold is refused with
    the list. Each job names its own leg's image, so ``gemm_bench_fp8`` runs the
    build at fp8 and every other job runs the bf16 one.
    """
    if not name:
        print("card jobs: " + ", ".join(CARD_JOBS))
        return
    entry = CARD_JOBS.get(name)
    if entry is None:
        print(f"modal: unknown card job {name!r}; the jobs are "
              + ", ".join(CARD_JOBS), file=sys.stderr)
        raise SystemExit(2)
    if entry.takes_argv:
        _print_answer(entry.job.remote(argv), entry.what)
        return
    if argv != "[]":
        print(f"modal: {name} takes no --argv", file=sys.stderr)
        raise SystemExit(2)
    _print_answer(entry.job.remote(), entry.what)


@app.local_entrypoint()
def warm_cache(legs: str = "both") -> None:
    """Build each leg once, so the first real run is not the cold one.

        modal run tools/modal/app.py::warm_cache
        modal run tools/modal/app.py::warm_cache --legs h100
        modal run tools/modal/app.py::warm_cache --legs h100-fp8

    A container compiles its leg against the cache Volume, and a cold cache
    compiles every target of that leg. The first run of a leg therefore pays
    minutes, and every later run pays seconds. This entry point pays that cost
    once, in a function with an hour to spend, so a card job is never the first
    thing to meet a cold cache. A card is not needed: ``nvcc`` crosses the
    architecture on any host.
    """
    for leg in selected_legs(legs):
        function = WARM_FUNCTIONS.get(leg.name)
        if function is None:
            raise SystemExit(f"no warm function for the leg {leg.name!r}")
        print(f"{leg.name}: {function.remote()}")


@app.local_entrypoint()
def profile_stage(stage: str = "base", device: str = "h100",
                  run_id: str = "", job: str = "", steps: int = 50,
                  seconds: int = 180, domains: str = NSYS_TRACE_DOMAINS,
                  sample: str = NSYS_SAMPLE) -> None:
    """Trace one stage of one job file on one device and print the answer.

        modal run tools/modal/app.py::profile_stage --stage base \\
            --device h100 --job job.json --steps 50 --seconds 180

    One Nsight Systems container writes the report, its SQLite copy, the sum
    CSVs, and the two timeline CSVs to ``/runs/<run_id>/profile``. Copy them
    with ``modal volume get nanochat-runs <run_id>/profile ./trace``, or let
    ``python -m nanochat_cpp.pgo trace`` do it.
    """
    payload = (json.loads(Path(job).read_text(encoding="utf-8"))
               if job else demo_job())
    tag = str(run_id or f"pgo-{stage}-{device}")
    function = {"h100": profile_stage_h100, "t4": profile_stage_t4}
    if device not in function:
        raise SystemExit(f"unknown device {device!r}; use h100 or t4")
    answer = json.loads(function[device].remote(
        str(stage), json.dumps(payload, sort_keys=True), tag, int(steps),
        int(seconds), str(domains), str(sample)))
    print(json.dumps(answer, indent=1, sort_keys=True))
    if any(code != 0 for code in (answer.get("status") or {}).values()):
        print("modal: one profile step did not finish", file=sys.stderr)
        raise SystemExit(1)


@app.local_entrypoint()
def measure_stage(stage: str = "base", device: str = "h100",
                  run_id: str = "", job: str = "", steps: int = 50,
                  seconds: int = 180, runs: int = 2) -> None:
    """Measure one stage of one job file untraced and print the answer.

        modal run tools/modal/app.py::measure_stage --stage base \\
            --device h100 --job job.json --runs 2

    The answer holds one entry per run. ``nanochat_cpp.pgo measure`` scores the
    better of them, because interference only ever slows a run.
    """
    payload = (json.loads(Path(job).read_text(encoding="utf-8"))
               if job else demo_job())
    tag = str(run_id or f"pgo-{stage}-{device}-measure")
    function = {"h100": measure_stage_h100, "t4": measure_stage_t4}
    if device not in function:
        raise SystemExit(f"unknown device {device!r}; use h100 or t4")
    answer = json.loads(function[device].remote(
        str(stage), json.dumps(payload, sort_keys=True), tag, int(steps),
        int(seconds), int(runs)))
    print(json.dumps(answer, indent=1, sort_keys=True))


@app.local_entrypoint()
def profile_demo(job: str = "", run_id: str = "profile", steps: int = 20,
                 profile: str = "", batch: int = 0,
                 corpus_urls: str = "", domains: str = NSYS_TRACE_DOMAINS,
                 sample: str = NSYS_SAMPLE) -> None:
    """Trace the base stage on the H100 leg and print where its time went.

        modal run tools/modal/app.py::profile_demo
        modal run tools/modal/app.py::profile_demo --job my-job.json
        modal run tools/modal/app.py::profile_demo --steps 30 \\
            --profile h100 --batch 16 --corpus-urls "https://..."

    One Nsight Systems container traces the chain's own base stage and writes
    the artifacts to ``/runs/<run_id>/profile``. The document holds the file
    list with sizes, the exit status of each profile step, and the first rows
    of the kernel, API, and memory sums. Copy the artifacts with:

        modal volume get nanochat-runs <run_id>/profile ./profile
    """
    payload = (json.loads(Path(job).read_text(encoding="utf-8")) if job
               else demo_job())
    if profile:
        payload["profile"] = str(profile)
    if int(batch) > 0:
        payload["batch"] = int(batch)
    urls = [url.strip() for url in str(corpus_urls).split(",") if url.strip()]
    if urls:
        plan = json.loads(corpus.remote(
            json.dumps(payload, sort_keys=True), json.dumps(urls),
            str(payload.get("profile") or "h100")))
        payload["corpus"] = "files"
        payload["corpus_path"] = plan["dir"]
        payload["tokenizer_path"] = plan["tokenizer"]
    answer = json.loads(profile_h100.remote(
        json.dumps(payload, sort_keys=True), str(run_id), int(steps),
        str(domains), str(sample)))
    print(json.dumps(answer, indent=1, sort_keys=True))
    if any(code != 0 for code in (answer.get("status") or {}).values()):
        print("modal: one profile step did not finish", file=sys.stderr)
        raise SystemExit(1)


@app.local_entrypoint()
def pipeline_demo(job: str = "", run_id: str = DEMO_RUN_ID,
                  corpus_urls: str = "", profile: str = "",
                  batch: int = 0, legs: str = "both", split_job: str = "",
                  split_corpus: str = "", split_steps: int = 6,
                  bench: str = "") -> None:
    """Run the selected legs, or one phase split, and print the answer.

        modal run tools/modal/app.py::pipeline_demo
        modal run tools/modal/app.py::pipeline_demo --job my-job.json
        modal run tools/modal/app.py::pipeline_demo --profile big --batch 16 \\
            --legs h100 --corpus-urls "https://.../shard_00000.parquet"
        modal run tools/modal/app.py::pipeline_demo --split-job '{"profile":...}' \\
            --split-corpus '{"files":[...],"tokenizer":"..."}'
        modal run tools/modal/app.py::pipeline_demo --bench '["--profile","big"]'

    ``job`` names a job file on this host; without one, the demo job runs.
    ``profile`` and ``batch`` override that job's shape and its base batch, so
    the throughput shape of the H100 leg runs without writing a file.
    ``legs`` chooses the cards: ``both`` (the default), ``t4``, or ``h100``. A
    single-leg request starts one container and leaves gate G2 out, because G2
    compares the two legs (section 8).
    ``corpus_urls`` is a comma-separated list of parquet URLs: it calls
    :func:`corpus` first, which downloads and tokenizes on the Volume, and the
    paths it returns become the job's ``corpus_path`` and ``tokenizer_path``.
    A second run with the same URLs reuses the download and the tokenizer, so
    the flag costs nothing after the first call.

    ``bench`` switches to the Muon benchmark and returns it: the argument is a
    JSON list of the benchmark's own flags, such as
    ``["--profile", "big", "--muon"]``, and :func:`bench_h100` times it on the
    H100 leg. The rows it prints are the optimizer's own rates, so a step that
    is slow in the optimizer is attributed on the card that ran it.

    ``split_job`` switches to the phase split and returns it: the single-device
    trainer runs ``split_steps`` steps of that job on the H100 and the answer
    carries ``fwd_ms``, ``bwd_ms``, ``opt_ms``, and ``sync_ms``. The chain
    reports one number per step, so a step far from its roofline bound is
    attributed this way (section 9).

    The document holds ``legs``, ``comparison``, and ``gates``, and the command
    exits non-zero when any gate fails (section 7 and section 8).
    """
    if bench:
        answer = json.loads(bench_h100.remote(bench))
        print(json.dumps(answer, indent=1, sort_keys=True))
        if int(answer.get("status") or 0) != 0:
            print("modal: the benchmark did not finish", file=sys.stderr)
            raise SystemExit(1)
        return
    if split_job:
        answer = json.loads(step_split.remote(split_job, split_corpus,
                                              int(split_steps)))
        print(json.dumps(answer, indent=1, sort_keys=True))
        if int(answer.get("status") or 0) != 0:
            print("modal: the trainer did not finish", file=sys.stderr)
            raise SystemExit(1)
        return
    payload = (json.loads(Path(job).read_text(encoding="utf-8")) if job
               else demo_job())
    if not payload.get("run_id"):
        payload["run_id"] = dict(DEMO_JOB)["run_id"]
    if profile:
        payload["profile"] = str(profile)
    if int(batch) > 0:
        payload["batch"] = int(batch)
    selected = selected_legs(legs)
    urls = [url.strip() for url in str(corpus_urls).split(",") if url.strip()]
    if urls:
        # The vocabulary comes from the job's profile, and the profile is read
        # in the container: this host may hold the Modal SDK and not the
        # package, which is the one thing the entry point must not require.
        plan = json.loads(corpus.remote(
            json.dumps(payload, sort_keys=True), json.dumps(urls),
            str(payload.get("profile") or "h100")))
        payload["corpus"] = "files"
        payload["corpus_path"] = plan["dir"]
        payload["tokenizer_path"] = plan["tokenizer"]
        print(f"modal: the corpus is {plan['dir']} and the tokenizer is "
              f"{plan['tokenizer']}", file=sys.stderr)
    answers = run_legs(json.dumps(payload, sort_keys=True), str(run_id),
                       legs=selected)
    document = demo_document(answers, legs=selected)
    print(json.dumps(document, indent=1, sort_keys=True))
    failed = gate_failures(document)
    if failed:
        print(f"modal: the run failed {', '.join(failed)}", file=sys.stderr)
        raise SystemExit(1)
