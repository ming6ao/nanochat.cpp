"""Writes the GEMM decision table from a sweep report.

``docs/kernels.md`` section 11.4 item 5 states the design: the table of
``backends/cuda/gemm/cutlass_rules.h`` maps a production shape to the arm the
sweep chose, and a tool writes it from the report, so the table is data and a
hand edit fails a gate instead of changing a shipped decision.

The input is one ``gemm_bench --json`` report of ``--rows production``, which is
the closed shape list of ``dispatch.cc``. The report holds one row for each
measurement, named ``gemm:<row>:<arm>``, and a ``:median`` companion row:

   ``gemm:<row>:cublas``             cuBLAS, the reference
   ``gemm:<row>:<candidate>:cutlass`` one candidate of the registry
   ``gemm:<row>:<schedule>:hand``     one schedule of the hand tile

The primary row states the minimum of the rounds, and this tool reads only those
rows: a decision is made on the statistic that a kernel repeatability supports,
and the median companion stays in the report for a reader.

**The order is candidate, then hand tile, then cuBLAS**, as section 11.4 states.
A candidate wins a shape when it beats cuBLAS by the margin and is no slower
than the hand tile. The hand tile wins when it beats cuBLAS by the margin and no
candidate did. cuBLAS wins every other shape.

**The margin is measured, and it is why a near tie promotes nothing.** Two runs
of one sweep disagree by up to 4 percent on a row whose arms are close: on the
H100 of section 11.8, five of the 28 production rows changed their winner
between two runs. A win smaller than the margin is therefore a coin toss and not
a measurement, so the default margin is 0.04 of the cuBLAS time and a caller can
state another one.

The measurement rows of ``bench_shapes.h`` never enter the table: they answer a
question about a kernel and not about a model, so the tool refuses a report whose
``rows`` field names another family.

Usage::

    tools/gemm_rules.py --report bench.json --check
    tools/gemm_rules.py --report bench.json --write

``--check`` compares the committed block with the block the report would write,
so the commit and the report cannot drift. ``--write`` rewrites the block and
leaves every other byte alone.

Hermetic and CPU-only: the standard library, no card, no CUDA toolchain.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

#: The header the table lives in, relative to the repository root.
DEFAULT_HEADER = Path("backends/cuda/gemm/cutlass_rules.h")

#: The two lines that bound the generated block.
BEGIN_MARKER = "// --- BEGIN GENERATED RULES ---"
END_MARKER = "// --- END GENERATED RULES ---"

#: The report field that states which row family was measured. Only the
#: production family may write the table. The report's `rows` key holds the
#: measurements themselves, so the family is its own field.
FAMILY_FIELD = "row_family"

#: The one family that may write the table.
PRODUCTION_ROWS = "production"

#: The prefix every measurement row name carries.
ROW_PREFIX = "gemm:"

#: The smallest win over cuBLAS that a promotion needs, as a fraction of the
#: cuBLAS time. `docs/kernels.md` section 11.4 item 5 states the rule, and
#: section 11.8 states the spread it comes from.
DEFAULT_MARGIN = 0.04

#: The four operand forms, in the spelling of that function.
FORMS = {"tn": (True, False), "nt": (False, True), "tt": (True, True)}


class RulesError(Exception):
    """A report the tool cannot use, or a header with no block."""


@dataclass(frozen=True)
class RowKey:
    """The key of one decision-table row: what a call must match."""

    m: int
    n: int
    k: int
    batch: int
    transpose_a: bool
    transpose_b: bool
    fp32_c: bool


@dataclass(frozen=True)
class Decision:
    """The arm one production shape takes, with the evidence of the choice."""

    key: RowKey
    row_name: str
    candidate: int
    route: str
    of_cublas: float
    hand_of_cublas: float


def parse_shape(text: str) -> RowKey:
    """The key a report's ``shape`` text states.

    The text comes from one function, ``ExtentText`` in ``gemm_benchmark.cc``, so
    this parser and that writer are one contract. A text that does not match is a
    report the tool must not read, because a mistyped key would build a table row
    that no call ever matches.
    """
    stamp = re.match(
        r"^(?P<m>\d+)x(?P<n>\d+)x(?P<k>\d+)"
        r"(?: batch (?P<batch>\d+))?"
        r"(?: (?P<form>tn|nt|tt))?"
        r"(?P<fp32> fp32 out)?$",
        text,
    )
    if stamp is None:
        raise RulesError(f"cannot read the extents of shape {text!r}")
    form = stamp.group("form") or ""
    if form and form not in FORMS:
        raise RulesError(f"unknown operand form {form!r} in shape {text!r}")
    transpose_a, transpose_b = FORMS.get(form, (False, False))
    return RowKey(
        m=int(stamp.group("m")),
        n=int(stamp.group("n")),
        k=int(stamp.group("k")),
        batch=int(stamp.group("batch") or 1),
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        fp32_c=stamp.group("fp32") is not None,
    )


def split_row_name(name: str) -> tuple[str, str, str]:
    """The row, the arm kind, and the arm's own name of one report row.

    A row name itself holds colons (``muon:640x1280:A``), so the suffix decides.
    A ``:median`` companion row is not a measurement and is rejected here.
    """
    if not name.startswith(ROW_PREFIX):
        raise RulesError(f"row {name!r} is not a sweep row")
    if name.endswith(":median"):
        raise RulesError(f"row {name!r} is a median companion, not a measurement")
    body = name[len(ROW_PREFIX) :]
    if body.endswith(":seam"):
        raise RulesError(
            f"row {name!r} is the shipped seam, which is a verification row and "
            "not a decision input"
        )
    for kind in ("cublas", "cutlass", "hand"):
        suffix = ":" + kind
        if not body.endswith(suffix):
            continue
        rest = body[: -len(suffix)]
        if kind == "cublas":
            return rest, kind, ""
        if ":" not in rest:
            raise RulesError(f"row {name!r} names no {kind} arm")
        row, arm = rest.rsplit(":", 1)
        return row, kind, arm
    raise RulesError(f"row {name!r} names no arm")


def read_report(path: Path) -> dict[str, dict]:
    """The report's rows, grouped by row name, with the arm of each row."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RulesError(f"cannot read the report {path}: {error}") from error
    family = document.get(FAMILY_FIELD)
    if family != PRODUCTION_ROWS:
        raise RulesError(
            f"the report names the {family!r} row family; the table takes only "
            f"{PRODUCTION_ROWS!r} rows, because a measurement row answers a "
            "question about a kernel and not about a model"
        )
    rows: dict[str, dict] = {}
    for row in document.get("rows", []):
        # The document's `rows` list holds the measurements; its `row_family`
        # field names the family. A document that states neither is one this
        # tool does not read.
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise RulesError("a row of the report states no name")
        # A median companion is the same measurement stated twice. It is not a
        # decision input, so the loop skips it instead of reading it twice.
        if row["name"].endswith(":median"):
            continue
        name, kind, arm = split_row_name(row["name"])
        entry = rows.setdefault(name, {"shape": row.get("shape", "")})
        entry["shape"] = entry["shape"] or row.get("shape", "")
        arms = entry.setdefault("arms", {})
        arms[kind] = float(row.get("ms", 0.0))
        if kind == "cutlass":
            arms.setdefault("candidates", {})[arm] = float(row.get("ms", 0.0))
        if kind == "hand":
            arms.setdefault("schedules", {})[arm] = float(row.get("ms", 0.0))
    return rows


def decide(
    row_name: str,
    entry: dict,
    registry: list[str],
    margin: float = DEFAULT_MARGIN,
) -> Decision:
    """The arm one production shape takes, by the order of section 11.4.

    ``margin`` is the smallest win a promotion needs. A win below it is inside
    the spread of two runs of one sweep, so the row keeps cuBLAS.
    """
    arms = entry["arms"]
    cublas_ms = float(arms.get("cublas", 0.0))
    if not cublas_ms > 0.0:
        raise RulesError(f"row {row_name!r} states no cuBLAS measurement")
    candidates = arms.get("candidates", {})
    schedules = arms.get("schedules", {})

    best_candidate = -1
    best_candidate_ms = 0.0
    for name, ms in sorted(candidates.items()):
        if name not in registry:
            raise RulesError(
                f"row {row_name!r} measured {name!r}, which is no candidate of "
                f"the registry ({', '.join(registry)})"
            )
        if ms < best_candidate_ms or best_candidate < 0:
            best_candidate = registry.index(name)
            best_candidate_ms = ms
    hand_ms = min(schedules.values()) if schedules else 0.0

    fastest = cublas_ms * (1.0 - margin)
    if (
        best_candidate >= 0
        and 0.0 < best_candidate_ms < fastest
        and (hand_ms <= 0.0 or best_candidate_ms <= hand_ms)
    ):
        route = "kCandidate"
        chosen_ms = best_candidate_ms
    elif 0.0 < hand_ms < fastest:
        route = "kHand"
        chosen_ms = hand_ms
        best_candidate = -1
    else:
        route = "kCublas"
        chosen_ms = cublas_ms
        best_candidate = -1
    of_cublas = cublas_ms / chosen_ms if chosen_ms > 0.0 else 0.0
    hand_of_cublas = cublas_ms / hand_ms if hand_ms > 0.0 else 0.0
    return Decision(
        key=parse_shape(entry["shape"]),
        row_name=row_name,
        candidate=best_candidate,
        route=route,
        of_cublas=of_cublas,
        hand_of_cublas=hand_of_cublas,
    )


def read_candidates(path: Path) -> list[str]:
    """The registry's candidate names, in registry order.

    ``backends/cuda/gemm/cutlass_candidates.h`` holds one row for each
    candidate, and its first field is the name. The table's index is the
    candidate's index, so the order is the contract.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise RulesError(f"cannot read the registry {path}: {error}") from error
    start = text.find("kGemmCandidates[]")
    if start < 0:
        raise RulesError(f"the registry {path} holds no kGemmCandidates table")
    end = text.find("};", start)
    if end < 0:
        raise RulesError(f"the registry {path} has no end of kGemmCandidates")
    names = re.findall(r'^\s*\{"([a-z0-9-]+)",', text[start:end], re.MULTILINE)
    if not names:
        raise RulesError(f"the registry {path} names no candidate")
    return names


def render(decisions: list[Decision]) -> str:
    """The generated block, between its two markers.

    One field stands on each line. The block is data that a card run writes once
    and a reader compares with `docs/baselines.json`, so a stable layout matters
    more than a short one: this form is a fixed point of the tree's
    `clang-format` style, which a single-line row is not.
    """
    lines = [
        BEGIN_MARKER,
        "// Written by `tools/gemm_rules.py --write` from a `gemm_bench --json`",
        "// report of the production rows. A row promotes the arm that beat cuBLAS",
        "// by the measured margin of section 11.4 item 5, and every other row",
        "// keeps cuBLAS. The fractions are that run's, so the block is the",
        "// evidence of the decision and not a summary of it.",
        "inline constexpr std::array<GemmRuleRow, "
        f"{len(decisions)}> kGemmRules{{{{",
    ]
    for decision in decisions:
        key = decision.key
        lines.append(f"    // {decision.row_name}")
        lines.append("    {")
        for field in (
            str(key.m),
            str(key.n),
            str(key.k),
            str(key.batch),
            "true" if key.transpose_a else "false",
            "true" if key.transpose_b else "false",
            "true" if key.fp32_c else "false",
            str(decision.candidate),
            f"GemmRoute::{decision.route}",
            f"{decision.of_cublas:.4f}",
            f"{decision.hand_of_cublas:.4f}",
        ):
            lines.append(f"        {field},")
        lines.append("    },")
    lines.append("}};")
    lines.append(END_MARKER)
    return "\n".join(lines)


def committed_block(text: str) -> tuple[int, int]:
    """The byte range of the block, markers included."""
    start = text.find(BEGIN_MARKER)
    end = text.find(END_MARKER)
    if start < 0 or end < 0 or end < start:
        raise RulesError("the header holds no generated block")
    return start, end + len(END_MARKER)


def generate(
    report: Path, registry: Path, margin: float = DEFAULT_MARGIN
) -> list[Decision]:
    """Every decision the report states, in report order."""
    names = read_candidates(registry)
    entries = read_report(report)
    decisions = [decide(name, entries[name], names, margin) for name in entries]
    if not decisions:
        raise RulesError(f"the report {report} holds no production row")
    return decisions


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--header", type=Path, default=DEFAULT_HEADER)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("backends/cuda/gemm/cutlass_candidates.h"),
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=DEFAULT_MARGIN,
        help=(
            "the smallest win over cuBLAS that promotes a row, as a fraction "
            f"of the cuBLAS time (default {DEFAULT_MARGIN})"
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    options = parser.parse_args(argv)

    try:
        decisions = generate(options.report, options.registry,
                             options.min_margin)
        block = render(decisions)
        text = options.header.read_text(encoding="utf-8")
        start, end = committed_block(text)
    except (RulesError, OSError) as error:
        print(f"gemm rules: {error}", file=sys.stderr)
        return 2

    committed = text[start:end]
    promoted = [d for d in decisions if d.route == "kCandidate"]
    if options.check:
        if committed != block:
            print(
                f"gemm rules: {options.header} does not state the decision table "
                f"of {options.report}",
                file=sys.stderr,
            )
            for line in _difference(committed, block):
                print(line, file=sys.stderr)
            return 1
        print(
            f"gemm rules: {len(decisions)} row(s) agree, "
            f"{len(promoted)} promoted to a candidate"
        )
        return 0

    updated = text[:start] + block + text[end:]
    options.header.write_text(updated, encoding="utf-8")
    print(
        f"gemm rules: wrote {len(decisions)} row(s) to {options.header}, "
        f"{len(promoted)} promoted to a candidate"
    )
    return 0


def _difference(committed: str, wanted: str) -> list[str]:
    """The lines two blocks disagree on, for a gate's message."""
    left = committed.splitlines()
    right = wanted.splitlines()
    lines: list[str] = []
    for index in range(max(len(left), len(right))):
        have = left[index] if index < len(left) else "<missing>"
        want = right[index] if index < len(right) else "<missing>"
        if have != want:
            lines.append(f"  line {index + 1}: have {have}")
            lines.append(f"  line {index + 1}: want {want}")
    return lines[:20]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
