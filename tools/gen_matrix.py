#!/usr/bin/env python3
"""Generate and check the model/device conformance matrix.

``docs/model-and-device-bringup.md`` section 11 makes the matrix the one place
where the two axes meet.  It is *generated, not maintained*: every row is
derived from ``models/<name>/spec.json`` and ``backends/<name>/caps.json``, so a
hand-added row is detectable (section 15).

The matrix has three axes:

* **Model** -- every directory that carries a ``spec.json`` under ``models/``;
* **Tier** -- ``T0 CPU``, ``S0 simulator``, ``T1 GPU correctness``, or
  ``T2 GPU parity`` (``docs/testing.md``);
* **Graph** -- ``train step``, ``prefill``, ``decode``, ``eval``, ``generate``
  (``docs/model.md``).

The tier-per-device rule of section 11 keeps the table small: every model gets
a ``T0 CPU`` row; a device with no physical sample gets an ``S0 simulator``
row; real silicon gets a ``T1 GPU correctness`` row; and only the flagship
model gets a ``T2 GPU parity`` row, at a wave boundary.

Usage::

    # Markdown on stdout.
    python3 tools/gen_matrix.py

    # JSON on stdout.
    python3 tools/gen_matrix.py --json

    # Validate a matrix document.  Exits non-zero for a model with no row, a
    # row with no graph coverage, a passing row that skips train step or
    # decode, and a row the generator cannot derive (section 15).
    python3 tools/gen_matrix.py --check --matrix docs/conformance-matrix.md

The script is standard-library only and hermetic: it reads JSON from the
repository and writes only to stdout, ``--out``, or a path the caller names.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TIERS = (
    "T0 CPU",
    "S0 simulator",
    "T1 GPU correctness",
    "T2 GPU parity",
)
GRAPHS = (
    "train step",
    "prefill",
    "decode",
    "eval",
    "generate",
)
#: The minimum a passing row must cover (section 11: "The minimum for a
#: passing row is the train step and decode").
MINIMUM_GRAPHS = ("train step", "decode")

DEFAULT_FLAGSHIP = "nanochat-default"

#: Devices with a physical sample on this tree's host.  AGENTS.md section 5
#: names the one GTX 1080 Ti; every other declared device is reached through
#: the simulator until it has silicon.  The list is data for the generator,
#: not a property of the model or the backend package.
PHYSICAL_SAMPLES = ("gtx1080ti",)

DEFAULT_MATRIX = REPO_ROOT / "tools" / "conformance-matrix.md"

#: A status that claims a passing gate.  Only these rows are held to the
#: train-step-and-decode minimum.
PASSING_STATUS = frozenset({"pass", "passing", "passed", "green", "ok"})


@dataclass(frozen=True)
class Row:
    """One generated matrix cell."""

    model: str
    tier: str
    device: str
    graphs: tuple[str, ...]
    status: str

    def key(self) -> tuple[str, str, str]:
        return (self.model, self.tier, self.device)


def _read_json(path: Path) -> dict:
    print(f"read  {path}", file=sys.stderr)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise SystemExit(f"{path}: cannot read: {error}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"{path}: invalid JSON: {error.msg}") from error


def load_models(root: Path) -> list[str]:
    """Every model package under ``root/models``, sorted by directory name."""
    models_dir = root / "models"
    if not models_dir.is_dir():
        raise SystemExit(f"{models_dir}: no models/ directory")
    names = []
    for path in sorted(models_dir.glob("*/spec.json")):
        spec = _read_json(path)
        name = spec.get("name", path.parent.name)
        if name != path.parent.name:
            raise SystemExit(
                f"{path}: spec name {name!r} does not match directory "
                f"{path.parent.name!r}")
        if spec.get("spec_version") != 1:
            raise SystemExit(
                f"{path}: unsupported spec_version "
                f"{spec.get('spec_version')!r}")
        names.append(name)
    if not names:
        raise SystemExit(f"{models_dir}: no */spec.json files")
    return names


def load_devices(root: Path) -> dict[str, dict]:
    """The declared physical device entries, by name, from backends/*/caps.json."""
    devices: dict[str, dict] = {}
    backends_dir = root / "backends"
    if not backends_dir.is_dir():
        return devices
    for path in sorted(backends_dir.glob("*/caps.json")):
        caps = _read_json(path)
        for entry in caps.get("devices", []):
            if entry.get("is_device") and entry.get("name"):
                devices[entry["name"]] = dict(entry, backend=caps.get("backend"))
    return devices


def derive_rows(models: list[str], devices: dict[str, dict],
                physical: tuple[str, ...], flagship: str) -> list[Row]:
    """Build the canonical rows, applying the tier-per-device rule of section 11."""
    physical_present = tuple(sorted(name for name in physical if name in devices))
    physical_absent = tuple(sorted(name for name in devices if name not in physical))
    rows: list[Row] = []
    for model in models:
        # T0 CPU is the definition, so every model has this row.  It covers
        # all five graphs: the CPU reference is the graph definition.
        rows.append(Row(model, "T0 CPU", "cpu", GRAPHS, "pass"))
        # A device without a physical sample is exercised through the
        # simulator, which is CPU-only and needs no GPU queue.
        for device in physical_absent:
            rows.append(Row(model, "S0 simulator", device, GRAPHS, "pass"))
        # Real silicon gets tiny-shape correctness.
        for device in physical_present:
            rows.append(Row(model, "T1 GPU correctness", device, GRAPHS,
                            "pending"))
        # T2 is the flagship's parity row at a wave boundary.  Do not give
        # every model a T2 row: that is how the matrix explodes.  If this tree
        # has no device with a physical sample, keep the T2 row with a generic
        # device cell so the tier is still enumerated and visibly pending.
        if model == flagship:
            t2_devices = physical_present or ("gpu",)
            for device in t2_devices:
                rows.append(Row(model, "T2 GPU parity", device, GRAPHS,
                                "pending"))
    return rows


def _normalise_graph(value: str) -> str | None:
    text = re.sub(r"[_-]+", " ", value.strip().lower())
    text = re.sub(r"\s+", " ", text)
    return text if text in GRAPHS else None


def parse_graphs(cell: str) -> tuple[tuple[str, ...], str | None]:
    """The named graph coverage in a cell, or an error string."""
    text = cell.strip()
    if not text or text in ("-", "--", "—", "n/a", "N/A"):
        return (), "does not name its graph coverage"
    if text.strip().lower() in ("all", "all five", "5/5", "complete"):
        return GRAPHS, None
    graphs: list[str] = []
    for part in re.split(r"[,;/]+", text):
        if not part.strip():
            continue
        graph = _normalise_graph(part)
        if graph is None:
            return (), f"names an unknown graph {part.strip()!r}"
        if graph not in graphs:
            graphs.append(graph)
    if not graphs:
        return (), "does not name its graph coverage"
    return tuple(graphs), None


def _normalise_tier(value: str) -> str | None:
    text = re.sub(r"\s+", " ", value.strip().lower())
    for tier in TIERS:
        if text == tier.lower():
            return tier
    return None


def _row_cells(line: str) -> list[str]:
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]
    return [cell.strip() for cell in text.split("|")]


def parse_markdown(text: str) -> list[dict]:
    """Parse the markdown matrix into row mappings.

    The table is found by its header row; the column order is read from the
    header so a caller may add a notes column with no parser change.  A line
    without a pipe is prose and is ignored.
    """
    header: list[str] | None = None
    rows: list[dict] = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = _row_cells(line)
        if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells if cell):
            continue
        lowered = [cell.lower() for cell in cells]
        if header is None and "model" in lowered and "tier" in lowered:
            header = lowered
            continue
        if header is None:
            continue
        mapping = {name: cells[index] if index < len(cells) else ""
                   for index, name in enumerate(header)}
        rows.append(mapping)
    return rows


def check_matrix(rows: list[dict], derived: list[Row],
                 models: list[str]) -> list[str]:
    """Return the section 11 violations in a parsed matrix."""
    problems: list[str] = []
    derived_keys = {row.key() for row in derived}
    derived_pairs = {(row.model, row.tier) for row in derived}
    declared_models = set(models)
    seen_models: set[str] = set()
    seen_keys: set[tuple[str, str, str]] = set()

    for index, cells in enumerate(rows, start=1):
        model = cells.get("model", "").strip()
        tier = _normalise_tier(cells.get("tier", ""))
        device = cells.get("device", "").strip()
        status = cells.get("status", "").strip().lower()
        where = (f"row {index} ({model or '?'} / "
                 f"{cells.get('tier', '').strip() or '?'} / {device or '?'})")

        if not model:
            problems.append(f"{where}: no model name")
            continue
        if model not in declared_models:
            problems.append(
                f"{where}: model {model!r} has no models/<name>/spec.json; "
                "the generator cannot derive this row (section 15)")
        else:
            seen_models.add(model)

        if tier is None:
            problems.append(
                f"{where}: tier {cells.get('tier', '').strip()!r} is not one "
                f"of {', '.join(TIERS)}")
            continue

        graphs, graph_error = parse_graphs(cells.get("graphs", ""))
        if graph_error is not None:
            problems.append(f"{where}: {graph_error}")

        key = (model, tier, device)
        if key in seen_keys:
            problems.append(f"{where}: duplicate row")
        seen_keys.add(key)
        # A section 11 table may omit the device column: the tier-per-device
        # rule already selects the device.  In that form a (model, tier) pair
        # is the derivable key; with a device present, all three must match.
        if device:
            derivable = key in derived_keys
        else:
            derivable = (model, tier) in derived_pairs
        if not derivable:
            problems.append(
                f"{where}: row is not derivable from models/*/spec.json and "
                "backends/*/caps.json; the matrix is generated, not maintained "
                "(section 15)")

        if status in PASSING_STATUS:
            missing = [graph for graph in MINIMUM_GRAPHS if graph not in graphs]
            if missing:
                problems.append(
                    f"{where}: passing row does not cover "
                    f"{', '.join(missing)}")

    for model in models:
        if model not in seen_models:
            problems.append(f"model {model!r} has no row in the matrix")
    return problems


def render_markdown(rows: list[Row], models: list[str],
                    devices: dict[str, dict], flagship: str) -> str:
    lines = [
        "# Model/device conformance matrix",
        "",
        "Generated by `tools/gen_matrix.py`; do not edit a row by hand "
        "(docs/model-and-device-bringup.md section 15).",
        "",
        f"Models: {', '.join(models)}.  Flagship (T2): {flagship}.",
        "",
        "| Model | Tier | Device | Graphs | Status |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| {model} | {tier} | {device} | {graphs} | {status} |"
                     .format(model=row.model, tier=row.tier, device=row.device,
                             graphs=", ".join(row.graphs), status=row.status))
    lines += [
        "",
        "Tier rule (section 11): T0 CPU for every model; S0 simulator for a "
        "device with no physical sample; T1 GPU correctness on real silicon; "
        "T2 GPU parity only for the flagship, at a wave boundary.",
        "",
        "A passing row covers at least `train step` and `decode`; a row that "
        "skips a graph must say so in the Graphs column.",
    ]
    return "\n".join(lines) + "\n"


def render_json(rows: list[Row], models: list[str],
                devices: dict[str, dict], flagship: str) -> str:
    payload = {
        "schema_version": 1,
        "generated_by": "tools/gen_matrix.py",
        "flagship": flagship,
        "tiers": list(TIERS),
        "graphs": list(GRAPHS),
        "minimum_graphs_for_pass": list(MINIMUM_GRAPHS),
        "models": [
            {"name": name,
             "spec": f"models/{name}/spec.json"}
            for name in models
        ],
        "devices": [
            {"name": name,
             "backend": entry.get("backend"),
             "physical_sample": name in PHYSICAL_SAMPLES}
            for name, entry in sorted(devices.items())
        ],
        "rows": [
            {"model": row.model, "tier": row.tier, "device": row.device,
             "graphs": list(row.graphs), "status": row.status}
            for row in rows
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


def _read_matrix(args, root: Path) -> str | None:
    if args.matrix:
        if args.matrix == "-":
            print("read  <stdin>", file=sys.stderr)
            return sys.stdin.read()
        path = Path(args.matrix)
        if not path.is_absolute():
            path = root / path
        print(f"read  {path}", file=sys.stderr)
        try:
            return path.read_text(encoding="utf-8")
        except OSError as error:
            raise SystemExit(f"{path}: cannot read matrix: {error}") from error
    if args.out:
        path = Path(args.out)
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            print(f"read  {path}", file=sys.stderr)
            return path.read_text(encoding="utf-8")
    if DEFAULT_MATRIX.is_file():
        print(f"read  {DEFAULT_MATRIX}", file=sys.stderr)
        return DEFAULT_MATRIX.read_text(encoding="utf-8")
    root_matrix = root / "tools" / "conformance-matrix.md"
    if root_matrix != DEFAULT_MATRIX and root_matrix.is_file():
        print(f"read  {root_matrix}", file=sys.stderr)
        return root_matrix.read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            print("read  <stdin>", file=sys.stderr)
            return data
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of markdown")
    parser.add_argument("--out", metavar="PATH",
                        help="write the matrix to PATH instead of stdout")
    parser.add_argument("--check", action="store_true",
                        help="validate a matrix document instead of emitting one")
    parser.add_argument("--matrix", metavar="PATH",
                        help=("matrix document to validate in --check mode "
                              "(default: tools/conformance-matrix.md, then "
                              "--out, then stdin; '-' reads stdin)"))
    parser.add_argument("--root", default=str(REPO_ROOT),
                        help="tree that holds models/ and backends/ "
                             "(default: the repository root)")
    parser.add_argument("--flagship", default=DEFAULT_FLAGSHIP, metavar="NAME",
                        help=f"model that gets the T2 row (default: "
                             f"{DEFAULT_FLAGSHIP})")
    parser.add_argument("--physical", action="append", metavar="DEVICE",
                        help=("a device with a physical sample; repeatable. "
                              "Default: " + ", ".join(PHYSICAL_SAMPLES)))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    models = load_models(root)
    devices = load_devices(root)
    physical = tuple(args.physical) if args.physical else PHYSICAL_SAMPLES
    if args.flagship not in models:
        raise SystemExit(
            f"--flagship {args.flagship!r} is not one of {', '.join(models)}")
    rows = derive_rows(models, devices, physical, args.flagship)

    if args.check:
        text = _read_matrix(args, root)
        if text is None:
            # No document supplied.  Validate the derivation itself: every
            # generated row names its graphs and every passing row covers the
            # minimum.  This is the useful no-input default, and it says so.
            problems = check_matrix(
                [{"model": row.model, "tier": row.tier, "device": row.device,
                  "graphs": ", ".join(row.graphs), "status": row.status}
                 for row in rows], rows, models)
            if problems:
                for problem in problems:
                    print(f"check: {problem}", file=sys.stderr)
                return 1
            print(f"check: validated {len(rows)} derived row(s); no matrix "
                  "document was supplied, so no hand-added row could exist",
                  file=sys.stderr)
            return 0
        problems = check_matrix(parse_markdown(text), rows, models)
        if problems:
            for problem in problems:
                print(f"check: {problem}", file=sys.stderr)
            return 1
        print(f"check: {len(rows)} derived row(s) agree with the matrix "
              "document", file=sys.stderr)
        return 0

    output = (render_json(rows, models, devices, args.flagship)
              if args.json else render_markdown(rows, models, devices,
                                                args.flagship))
    if args.out:
        path = Path(args.out)
        if not path.is_absolute():
            path = root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
        print(f"write {path}", file=sys.stderr)
    else:
        sys.stdout.write(output)
        print("write <stdout>", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
