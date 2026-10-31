"""The shared Hugging Face Hub helper for the three notebooks.

The Hub save and load calls have one home: ``nanochat_cpp_bridge.hf_hub``,
the adapter zone of ``docs/dependency-layering-plan.md`` section 4, so a
notebook, a Modal container, and a script run the same sequence. This module
is the host's handle on that home. It keeps only what a host owns:

- the pure helpers the notebook cells and the CPU tests name
  (``tokenizer_file``, ``repo_id_for``, ``build_run_record``, ``now_utc``,
  ``stage_dir``), which import nothing outside the standard library;
- ``write_stage``, which writes the local stage folder beside the container;
- ``load_stage``, which rebuilds a model from a pulled stage and so needs this
  host's ``args`` module and the released package. A package never imports a
  host (rule 4 of section 4), which is why the model wrapper stays here.

Every Hub save and load call forwards to the Z2 module, which imports
``huggingface_hub`` inside the call. The CPU tests inject a fake ``api`` or a
fake ``download``, so they run without the wheel and without network access.
See ``docs/checkpoint-format.md`` section 16.
"""

from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path

from nanochat_cpp_bridge import hf_hub

__all__ = [
    "CLIMBMIX_REPO",
    "CONTAINER",
    "FULL_STAGE_PATTERNS",
    "STAGE_PATTERNS",
    "TRAIN_SHARD",
    "VAL_SHARD",
    "build_run_record",
    "default_namespace",
    "download_climbmix",
    "load_stage",
    "now_utc",
    "pull_stage",
    "pull_tokenizer",
    "push_file",
    "push_stage",
    "push_tokenizer",
    "repo_id_for",
    "stage_dir",
    "tokenizer_file",
    "write_stage",
]

# The patterns, the container magic, the repository prefix, and the shard names
# come from the Z2 Hub module, so a notebook pulls exactly what the same module
# pushes. One home for the layout of ``docs/checkpoint-format.md`` section 16.
STAGE_PATTERNS = hf_hub.STAGE_PATTERNS
FULL_STAGE_PATTERNS = hf_hub.FULL_STAGE_PATTERNS
CONTAINER = hf_hub.CONTAINER
PREFIX = hf_hub.PREFIX
CLIMBMIX_REPO = hf_hub.CLIMBMIX_REPO
TRAIN_SHARD = hf_hub.TRAIN_SHARD
VAL_SHARD = hf_hub.VAL_SHARD


def tokenizer_file(vocab_size) -> str:
    """The tokenizer repository file name for one vocabulary size."""
    return f"vocab-{int(vocab_size)}.nctoken"


def default_namespace(api=None, environ=None) -> str:
    """The namespace for a new repository.

    ``HF_USERNAME`` overrides the account name. Otherwise the name comes from
    ``HfApi.whoami()``, through the Z2 Hub module.
    """
    return hf_hub.default_namespace(api=api, environ=environ)


def repo_id_for(run_name, namespace=None, prefix=PREFIX, api=None,
                environ=None) -> str:
    """The repository id of one model family run.

    ``repo_id`` is ``<namespace>/nanochat-<run_name>``.
    """
    if namespace is None:
        namespace = default_namespace(api=api, environ=environ)
    return f"{namespace}/{prefix}-{run_name}"


def stage_dir(root, stage) -> Path:
    """The local folder of one stage, ``<root>/<stage>``."""
    return Path(root) / str(stage)


def build_run_record(*, stage, run_name, git_revision="", build_key="",
                     cuda_arch="", step=0, parent_repo_id="",
                     parent_revision="", created_utc="") -> dict:
    """Assemble the provenance record. Pure; no Hub call.

    The record names the stage, the run, the source revision, the build, the
    checkpoint container, the parent stage, and the step.
    """
    return {
        "stage": str(stage),
        "run_name": str(run_name),
        "git_revision": str(git_revision),
        "build_key": str(build_key),
        "cuda_arch": str(cuda_arch),
        "container": CONTAINER,
        "parent_repo_id": str(parent_repo_id),
        "parent_revision": str(parent_revision),
        "step": int(step),
        "created_utc": str(created_utc),
    }


def now_utc() -> str:
    """The current time in UTC as an ISO 8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _stage_card(record, description) -> str:
    """The model card text of one stage record."""
    lines = [f"# {record['run_name']}", ""]
    if description:
        lines += [str(description), ""]
    lines += [
        f"- stage: `{record['stage']}`",
        f"- container: `{record['container']}`",
        f"- source revision: `{record['git_revision']}`",
        f"- step: {record['step']}",
    ]
    if record["parent_repo_id"]:
        lines.append(f"- parent: `{record['parent_repo_id']}` "
                     f"at `{record['parent_revision']}`")
    return "\n".join(lines) + "\n"


def write_stage(folder, *, stage, run_name, model_json, train_json,
                tokenizer, git_revision="", build_key="", cuda_arch="",
                step=0, parent_repo_id="", parent_revision="",
                created_utc="", description="") -> dict:
    """Write one local stage folder and return the provenance record.

    The folder gets ``model.json``, ``train.json``, ``tokenizer.nctoken``,
    ``run.json``, and ``README.md``. The caller then calls :func:`push_stage`
    to upload the folder. No Hub call runs here, so the write works offline.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.json").write_text(str(model_json) + "\n",
                                       encoding="utf-8")
    (folder / "train.json").write_text(str(train_json) + "\n",
                                       encoding="utf-8")
    shutil.copyfile(tokenizer, folder / "tokenizer.nctoken")
    record = build_run_record(
        stage=stage, run_name=run_name, git_revision=git_revision,
        build_key=build_key, cuda_arch=cuda_arch, step=step,
        parent_repo_id=parent_repo_id, parent_revision=parent_revision,
        created_utc=created_utc)
    (folder / "run.json").write_text(
        json.dumps(record, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    (folder / "README.md").write_text(_stage_card(record, description),
                                      encoding="utf-8")
    return record


def push_stage(repo_id, stage, folder, **options) -> str:
    """Upload one stage folder into the repository and optionally tag it.

    The upload runs in ``nanochat_cpp_bridge.hf_hub``. ``options`` are its
    ``token``, ``private``, ``commit_message``, ``tag``, and ``api``.
    """
    return hf_hub.push_stage(repo_id, stage, folder, **options)


def push_file(repo_id, path_in_repo, local_path, **options) -> str:
    """Upload one file into the repository.

    The upload runs in ``nanochat_cpp_bridge.hf_hub``. ``options`` are its
    ``token``, ``commit_message``, and ``api``.
    """
    return hf_hub.push_file(repo_id, path_in_repo, local_path, **options)


def pull_stage(repo_id, stage, dest, **options) -> Path:
    """Download the named files of one stage and return the stage folder.

    The download runs in ``nanochat_cpp_bridge.hf_hub``. ``options`` are its
    ``revision``, ``token``, ``patterns``, and ``download``.
    """
    return hf_hub.pull_stage(repo_id, stage, dest, **options)


def download_climbmix(dest, **options) -> Path:
    """Download the ClimbMix shards into ``dest`` and return the folder.

    The download runs in ``nanochat_cpp_bridge.hf_hub``, which skips a shard
    that ``dest`` already holds, so a notebook can call it on every run.
    ``options`` are its ``repo_id``, ``shards``, ``token``, and ``download``.
    """
    return hf_hub.download_climbmix(dest, **options)


def pull_tokenizer(repo_id, vocab_size, dest, **options):
    """Download the tokenizer file for one vocabulary size.

    Return the local path, or ``None`` when the repository does not hold it.
    A missing tokenizer means the notebook trains one. The download runs in
    ``nanochat_cpp_bridge.hf_hub``.
    """
    return hf_hub.pull_tokenizer(repo_id, vocab_size, dest, **options)


def push_tokenizer(repo_id, vocab_size, local_path, **options) -> str:
    """Upload one tokenizer file to the tokenizer repository.

    The upload runs in ``nanochat_cpp_bridge.hf_hub``.
    """
    return hf_hub.push_tokenizer(repo_id, vocab_size, local_path, **options)


def load_stage(repo_id, stage, dest, *, revision="main", token=None,
               device="cuda", download=None):
    """Download one stage and build the model, the tokenizer, and the args.

    Return ``(model, tokenizer, model_args)``. The load is parameter only, so
    the pretraining optimizer section is not read. The download forwards to
    ``nanochat_cpp_bridge.hf_hub``; the model comes from the released package,
    which is why this wrapper stays on the host.
    """
    import args as model_args_module  # noqa: PLC0415
    import nanochat_cpp as nc  # noqa: PLC0415

    folder = pull_stage(repo_id, stage, dest, revision=revision, token=token,
                        download=download)
    model_path = folder / "model.json"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"the stage {stage!r} has no {model_path.name}")
    tokenizer_path = folder / "tokenizer.nctoken"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"the stage {stage!r} has no {tokenizer_path.name}")
    checkpoint_path = folder / "checkpoint.ckpt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"the stage {stage!r} has no {checkpoint_path.name}")
    model_args = model_args_module.ModelArgs.from_json(
        model_path.read_text(encoding="utf-8"))
    model = nc.Model(model_args.to_config(), device=device, seed=0)
    model.load(str(checkpoint_path))
    tokenizer = nc.Tokenizer.load(str(tokenizer_path))
    return model, tokenizer, model_args
