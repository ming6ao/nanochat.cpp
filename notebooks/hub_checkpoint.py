"""The shared Hugging Face Hub helper for the three notebooks.

The module holds every Hugging Face Hub call, so the notebooks show the flow
and the calls have one home. Each function imports ``huggingface_hub`` inside
the body and takes an ``api`` or ``download`` argument. The CPU tests inject a
fake and run without the wheel and without network access. See
``docs/checkpoint-format.md`` section 16.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
from pathlib import Path

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

#: The files a notebook needs to load a stage.
STAGE_PATTERNS = ("checkpoint.ckpt", "model.json", "tokenizer.nctoken")

#: STAGE_PATTERNS plus the provenance files the post-training notebook copies.
FULL_STAGE_PATTERNS = STAGE_PATTERNS + (
    "train.json", "run.json", "metrics.jsonl", "summary.json", "README.md")

#: The container name the notebooks write.
CONTAINER = "NCHKPT01"

#: The repository name prefix of one model family.
PREFIX = "nanochat"

#: The ClimbMix dataset repository.
CLIMBMIX_REPO = "karpathy/climbmix-400b-shuffle"

#: The training shard the pretraining notebook reads.
TRAIN_SHARD = "shard_00000.parquet"

#: The validation shard the pretraining and evaluation notebooks read.
VAL_SHARD = "shard_06542.parquet"


def tokenizer_file(vocab_size) -> str:
    """The tokenizer repository file name for one vocabulary size."""
    return f"vocab-{int(vocab_size)}.nctoken"


def default_namespace(api=None, environ=None) -> str:
    """The namespace for a new repository.

    ``HF_USERNAME`` overrides the account name. Otherwise the name comes from
    ``HfApi.whoami()``.
    """
    environ = os.environ if environ is None else environ
    override = environ.get("HF_USERNAME")
    if override:
        return override
    if api is None:
        from huggingface_hub import HfApi  # noqa: PLC0415

        api = HfApi()
    info = api.whoami() or {}
    name = info.get("name") or info.get("user")
    if not name:
        raise RuntimeError(
            "cannot resolve the Hugging Face namespace; set HF_USERNAME")
    return name


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
    container, the parent stage, and the step.
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
    ``run.json``, and ``README.md``. The caller then calls
    :func:`push_stage` to upload the folder.
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


def push_stage(repo_id, stage, folder, *, token=None, private=False,
               commit_message="", tag=None, api=None) -> str:
    """Upload one stage folder into the repository and optionally tag it."""
    if api is None:
        from huggingface_hub import HfApi  # noqa: PLC0415

        api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private,
                    exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="model",
                      folder_path=str(folder), path_in_repo=str(stage),
                      commit_message=commit_message, token=token)
    if tag:
        api.create_tag(repo_id=repo_id, tag=tag, repo_type="model",
                       token=token)
    return str(stage)


def push_file(repo_id, path_in_repo, local_path, *, token=None,
              commit_message="", api=None) -> str:
    """Upload one file into the repository."""
    if api is None:
        from huggingface_hub import HfApi  # noqa: PLC0415

        api = HfApi(token=token)
    api.upload_file(repo_id=repo_id, repo_type="model",
                    path_or_fileobj=str(local_path),
                    path_in_repo=str(path_in_repo),
                    commit_message=commit_message, token=token)
    return str(path_in_repo)


def pull_stage(repo_id, stage, dest, *, revision="main", token=None,
               patterns=STAGE_PATTERNS, download=None) -> Path:
    """Download the named files of one stage and return the stage folder."""
    if download is None:
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        download = snapshot_download
    local = download(repo_id=repo_id, repo_type="model", revision=revision,
                     local_dir=str(dest), token=token,
                     allow_patterns=[f"{stage}/{name}" for name in patterns])
    return Path(local) / str(stage)


def download_climbmix(dest, *, repo_id=CLIMBMIX_REPO,
                      shards=(TRAIN_SHARD, VAL_SHARD), token=None,
                      download=None) -> Path:
    """Download the ClimbMix shards into ``dest`` and return the folder.

    The call skips a shard that is already present, so a notebook can call it
    on every run.
    """
    dest = Path(dest)
    missing = [name for name in shards if not (dest / name).is_file()]
    if not missing:
        return dest
    if download is None:
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        download = snapshot_download
    download(repo_id=repo_id, repo_type="dataset", local_dir=str(dest),
             allow_patterns=list(missing), token=token)
    return dest


def pull_tokenizer(repo_id, vocab_size, dest, *, revision="main", token=None,
                   download=None):
    """Download the tokenizer file for one vocabulary size.

    Return the local path, or ``None`` when the repository does not hold it.
    A missing tokenizer means the notebook trains one.
    """
    if download is None:
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        download = snapshot_download
    name = tokenizer_file(vocab_size)
    try:
        local = download(repo_id=repo_id, repo_type="model",
                         revision=revision, local_dir=str(dest), token=token,
                         allow_patterns=[name])
    except Exception:  # noqa: BLE001 - a miss means "train it"
        return None
    path = Path(local) / name
    return path if path.is_file() else None


def push_tokenizer(repo_id, vocab_size, local_path, *, token=None,
                   commit_message="", api=None) -> str:
    """Upload one tokenizer file to the tokenizer repository."""
    if api is None:
        from huggingface_hub import HfApi  # noqa: PLC0415

        api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    name = tokenizer_file(vocab_size)
    api.upload_file(repo_id=repo_id, repo_type="model",
                    path_or_fileobj=str(local_path), path_in_repo=name,
                    commit_message=commit_message, token=token)
    return name


def load_stage(repo_id, stage, dest, *, revision="main", token=None,
               device="cuda", download=None):
    """Download one stage and build the model, the tokenizer, and the args.

    Return ``(model, tokenizer, model_args)``. The load is parameter only, so
    the pretraining optimizer section is not read.
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
