"""The reference task surface: the hub loader, the shuffle, the code executor.

``nanochat_cpp.tasks`` (Z1) owns the task classes, the multiple-choice prompt
renderer, and the answer parsers, all of which are standard-library only. The
three things it cannot do without an extra live here
(``docs/dependency-layering-plan.md`` section 2.1):

``HubDataset`` and ``load_hub_dataset``
    The dataset access of ``tasks.common``: list the parquet shards of a hub
    repository, download them once into the base directory under a file lock,
    and read them with ``pyarrow``. The seeded shuffle matches
    ``datasets.Dataset.shuffle(seed=seed)`` exactly, so the example order the
    reference evaluates is reproduced. This module therefore needs
    ``filelock`` as well as ``numpy`` and ``pyarrow``; section 2.1's table
    misses that leak and the ``hfdata`` extra of section 6.2 carries it.
``_execute_code``
    The reference ``nanochat.execution.execute_code`` of HumanEval: a fresh
    interpreter, rlimits, a scrubbed environment, and a timeout. The bridge
    inherits the reference guards rather than reimplementing them, and under
    ``tools/nanochat`` the child process inherits the resource sandbox cgroup.

The core reaches both through a named seam, never through an import (gate G5):
:func:`hub_loader` returns :func:`load_hub_dataset` and :func:`code_executor`
returns the executor, and a host injects the callable it wants. That is how
``ARC``, ``MMLU``, ``GSM8K``, and ``HumanEval`` stay importable with no extra,
and how a test hands in a fake table with no network and no wheel.

Extras: ``reference`` (``numpy``), ``hfdata`` (``pyarrow`` and ``filelock``),
and ``nanochat`` for the code executor.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Callable

from . import _extras

__all__ = [
    "EXTRAS",
    "HubDataset",
    "code_executor",
    "execute_code",
    "get_base_dir",
    "hub_loader",
    "load_hub_dataset",
]

#: The extras this module needs, in the order a caller installs them.
EXTRAS = ("reference", "hfdata", "nanochat")


def get_base_dir() -> str:
    """The nanochat cache root (``NANOCHAT_BASE_DIR`` or ``~/.cache/nanochat``).

    Mirrors ``nanochat.common.get_base_dir`` and the copy the core's task
    module carries, so the task datasets land in the same place as the
    reference implementation without importing it. The loader below needs the
    path, and Z2 does not reach into Z1 for a helper the core is free to drop,
    so this module keeps its own copy.
    """
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ["NANOCHAT_BASE_DIR"]
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir


class HubDataset:
    """A minimal HuggingFace ``datasets.Dataset`` stand-in over a pyarrow table.

    Offers lazy row access and a seeded shuffle that matches
    ``datasets.Dataset.shuffle(seed=seed)`` exactly, so the example order the
    reference evaluates is reproduced. Mirrors ``tasks.common.HubDataset``.
    """

    def __init__(self, table, permutation=None):
        self.table = table
        self.permutation = permutation

    def __len__(self):
        return self.table.num_rows

    def shuffle(self, seed):
        numpy = _extras.import_package("numpy")

        permutation = numpy.random.default_rng(seed).permutation(len(self))
        return HubDataset(self.table, permutation)

    def __getitem__(self, index):
        physical_index = (index if self.permutation is None
                          else int(self.permutation[index]))
        return {column: self.table[column][physical_index].as_py()
                for column in self.table.column_names}


def load_hub_dataset(repo_id, subset="default", split="train") -> HubDataset:
    """Load a hub dataset through its auto-generated parquet export.

    Mirrors ``tasks.common.load_hub_dataset``: list the parquet shards through
    the hub API, download them once into the base directory, and read them with
    pyarrow. The manifest is written last, so its presence means the download
    completed; a file lock serializes concurrent ranks.
    """
    pyarrow = _extras.import_package("pyarrow")
    parquet = _extras.import_package("pyarrow.parquet")
    filelock = _extras.import_package("filelock")

    base_dir = get_base_dir()
    slug = repo_id.replace("/", "--")
    shards_dir = os.path.join(base_dir, "task_data", slug, subset, split)
    manifest_path = os.path.join(shards_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        os.makedirs(shards_dir, exist_ok=True)
        with filelock.FileLock(manifest_path + ".lock"):
            # Only a single rank acquires the lock and downloads; the others
            # block here and then skip because the manifest now exists.
            if not os.path.exists(manifest_path):
                listing_url = (f"https://huggingface.co/api/datasets/{repo_id}"
                               f"/parquet/{subset}/{split}")
                with urllib.request.urlopen(listing_url) as response:
                    shard_urls = json.loads(response.read())
                filenames = []
                for shard_index, shard_url in enumerate(shard_urls):
                    filename = f"{shard_index:05d}.parquet"
                    print(f"Downloading {shard_url} ...")
                    with urllib.request.urlopen(shard_url) as response:
                        content = response.read()
                    with open(os.path.join(shards_dir, filename), "wb") as f:
                        f.write(content)
                    filenames.append(filename)
                with open(manifest_path, "w") as f:
                    json.dump(filenames, f)
    with open(manifest_path, "r") as f:
        filenames = json.load(f)
    shard_paths = [os.path.join(shards_dir, filename) for filename in filenames]
    tables = [parquet.read_table(path) for path in shard_paths]
    table = pyarrow.concat_tables(tables)
    return HubDataset(table)


def hub_loader() -> Callable[..., HubDataset]:
    """The hub dataset loader, as one callable a core caller can be handed.

    The core task classes never import this package (gate G5). A host that
    wants hub-backed tasks injects this callable through its own seam, and a
    test injects a fake table instead, so the class stays hermetic.
    """
    return load_hub_dataset


def execute_code(code: str) -> Any:
    """Run generated code through the reference guarded executor.

    The reference checkout goes on ``sys.path`` first, so the call is the
    reference ``nanochat.execution.execute_code`` (fresh interpreter, rlimits,
    scrubbed environment, timeout), not a copy of it.
    """
    _extras.reference_on_path()
    execution = _extras.import_package("nanochat.execution")
    return execution.execute_code(code)


def code_executor() -> Callable[[str], Any]:
    """The reference code executor, as one callable a core caller is handed.

    The returned object has the reference result shape, so a task class reads
    ``result.success`` exactly as it does today.
    """
    return execute_code
