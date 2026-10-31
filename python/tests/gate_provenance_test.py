"""G4: every fixture record carries its provenance.

Gate G4 of ``docs/dependency-layering-plan.md`` section 8, and the invariant of
section 7.2: a manifest record names the environment that produced the bytes
(``producer``) and the module or generator that makes them (``generated_by``).
Without the pair, a parity claim in [parity.md](parity.md) rests on an
environment the manifest cannot name.

The gate reads every ``models/*/fixtures/manifest.json`` and fails naming the
file and the record when a record has no ``producer`` object or no
``generated_by``. It is standard-library only, and it resolves ``models/``
through the runfiles tree under Bazel, so the packaging node can hand the test
``//models/nanochat-default:files`` as data.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

#: One manifest, relative to the repo root. Its position finds ``models/``.
MANIFEST = "models/nanochat-default/fixtures/manifest.json"

#: Every manifest under a ``models/`` directory, relative to it.
PATTERN = "*/fixtures/manifest.json"


def _runfiles_paths() -> list[Path]:
    """The repository root in the Bazel runfiles tree, if there is one."""
    paths: list[Path] = []
    bases = (os.environ.get("TEST_SRCDIR"), os.environ.get("RUNFILES_DIR"))
    for base in bases:
        if not base:
            continue
        workspace = os.environ.get("TEST_WORKSPACE")
        if workspace:
            paths.append(Path(base) / workspace)
        paths.append(Path(base))
    return paths


def models_dir() -> Path:
    """``models/``: the runfiles tree under Bazel, else the repo root."""
    for root in _runfiles_paths():
        candidate = root / MANIFEST
        if candidate.is_file():
            return candidate.parents[2]
    return Path(__file__).resolve().parents[2] / "models"


def manifests() -> list[Path]:
    """Every model manifest, in name order."""
    return sorted(models_dir().glob(PATTERN))


def provenance_violations(path: Path) -> list[str]:
    """The records of ``path`` that carry no provenance, as text."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    reports: list[str] = []
    for record in manifest.get("fixtures") or ():
        name = record.get("file", "<unnamed record>")
        producer = record.get("producer")
        if not isinstance(producer, dict) or not producer:
            reports.append(f"{path}: {name}: no producer object")
        generated_by = record.get("generated_by")
        if not isinstance(generated_by, str) or not generated_by.strip():
            reports.append(f"{path}: {name}: no generated_by")
    return reports


class ProvenanceTest(unittest.TestCase):
    def test_every_record_names_its_producer(self) -> None:
        found = manifests()
        self.assertTrue(
            found,
            f"no manifest under {models_dir()}; every model package records "
            f"its fixture provenance in fixtures/manifest.json (section 7.2)")
        reports: list[str] = []
        for path in found:
            reports.extend(provenance_violations(path))
        self.assertEqual(
            reports, [],
            "a fixture record names no environment; add the producer pair of "
            "section 7.2 to each record below:\n" + "\n".join(reports))

    def test_the_scan_notices_a_record_without_a_producer(self) -> None:
        """The gate can fail: a record with neither field is reported twice."""
        manifest = {
            "schema_version": 1,
            "fixtures": [
                {"file": "fixtures/ok.bin",
                 "producer": {"python": "3.12.4"},
                 "generated_by": "nanochat_cpp_bridge.fixtures.train_parity"},
                {"file": "fixtures/bare.bin", "sha256": "0" * 64},
            ],
        }
        with tempfile.TemporaryDirectory(prefix="nanochat_g4_") as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            reports = provenance_violations(path)
        self.assertEqual(len(reports), 2, reports)
        self.assertTrue(all("fixtures/bare.bin" in report
                            for report in reports), reports)
        self.assertTrue(any("producer" in report for report in reports))
        self.assertTrue(any("generated_by" in report for report in reports))


if __name__ == "__main__":
    unittest.main()
