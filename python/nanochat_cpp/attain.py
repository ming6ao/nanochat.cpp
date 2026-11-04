"""The attainment record: join a bound with a measurement.

The roofline model reports a bound and the benchmark tool reports a
measurement. This module is the one place that puts the two side by side. It
reads a `nanochat.roofline.v3` document and a `nanochat.bench.v3` document, joins
their rows on the four identity keys (`family`, `phase`, `precision`, `shape`),
and prints the fraction of the bound the device reached.

The record it prints is the row of `docs/performance-model.md` section 10. A
hand-kept table drifts from the numbers it claims, so the row is generated from
the two files instead of transcribed.

Two rules come from the model document and are enforced here:

* The fraction is the bound over the measurement. A value above 1.0 is a
  finding, not a success: it means the declared peak rate is too low or the
  measurement does not describe the work the model counted
  (`docs/performance-model.md` section 9). It is reported unclamped.
* A bound is only a prediction once a device has tested it. Each joined row
  states the device that validated it (`validated_on`), which is the device the
  benchmark ran on. When that differs from the device the bound was computed
  for, the row says so, because a bound for one card says nothing about another
  (`docs/performance-model.md` section 10).
"""

import json

ROOFLINE_SCHEMA = "nanochat.roofline.v3"
BENCH_SCHEMA = "nanochat.bench.v3"

# The four keys a joined row carries. A row that states none of them takes part
# in no join, so a benchmark that names no family still reports its time.
IDENTITY_KEYS = ("family", "phase", "precision", "shape")


def _load(path: str, expected: str) -> dict:
    """Read one report and check its schema version."""
    try:
        with open(path, encoding="utf-8") as handle:
            body = json.load(handle)
    except OSError as error:
        raise SystemExit(f"attain: cannot read {path}: {error}") from error
    except ValueError as error:
        raise SystemExit(f"attain: {path} is not JSON: {error}") from error
    if not isinstance(body, dict):
        raise SystemExit(f"attain: {path} is not a JSON object")
    schema = body.get("schema", "")
    if schema != expected:
        raise SystemExit(
            f"attain: {path} states schema '{schema}', expected '{expected}'. "
            "Re-run the report at the current revision: the identity keys and "
            "the millisecond unit arrived with version 3.")
    return body


def _shape_tokens(shape: str) -> dict:
    """Parse a `key=value` shape string into its tokens.

    Both reports state a shape as space separated `key=value` tokens in one
    canonical order. Comparing the tokens a pair of rows shares is what lets a
    bound for `B=8 T=512 H=4 KV=4 D=128 wl=-1 L=8` meet a measured row that
    states the same workload without the layer count.
    """
    tokens = {}
    for part in str(shape).split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        tokens[key] = value
    return tokens


def shape_matches(left: str, right: str) -> bool:
    """True when two shapes agree on every token they both state.

    A pair with no token in common does not match: two empty shapes are not
    evidence that the rows describe the same work.
    """
    left_tokens = _shape_tokens(left)
    right_tokens = _shape_tokens(right)
    shared = set(left_tokens) & set(right_tokens)
    if not shared:
        return False
    return all(left_tokens[key] == right_tokens[key] for key in shared)


def _want_join(bound: dict, measured: dict) -> bool:
    """True when a bound row and a measured row describe the same work."""
    for key in IDENTITY_KEYS:
        if key == "shape":
            continue
        if not measured.get(key):
            return False
        if bound.get(key) != measured.get(key):
            return False
    return shape_matches(bound.get("shape", ""), measured.get("shape", ""))


def join(bound_report: dict, measured_report: dict) -> list[dict]:
    """Every measured row with the bound row it belongs to, or without one."""
    rows = []
    for measured in measured_report.get("rows", []):
        match = None
        for bound in bound_report.get("rows", []):
            if _want_join(bound, measured):
                match = bound
                break
        rows.append({"measured": measured, "bound": match})
    return rows


def attainment_row(joined: dict, bound_device: str,
                   measured_device: str) -> dict:
    """One record row: the fraction of the bound, and what limits the row."""
    measured = joined["measured"]
    bound = joined["bound"]
    measured_ms = float(measured.get("ms", 0.0))
    row = {
        "family": measured.get("family", ""),
        "phase": measured.get("phase", ""),
        "precision": measured.get("precision", ""),
        "shape": measured.get("shape", ""),
        "measured_ms": measured_ms,
        "measured_gflops": float(measured.get("gflops", 0.0)),
        "measured_work_ratio": float(measured.get("work_ratio", 0.0)),
        "measured_mfu": float(measured.get("mfu", 0.0)),
        # The guarantee the row rests on: the card that actually ran the work.
        "validated_on": measured_device,
    }
    if bound is None:
        row["bound_ms"] = None
        row["fraction"] = None
        row["limit"] = ""
        row["note"] = "no bound row for this family, phase, precision, and shape"
        return row
    bound_ms = float(bound.get("ms", 0.0))
    row["bound_ms"] = bound_ms
    row["limit"] = bound.get("limit", "")
    row["bound_work_ratio"] = float(bound.get("work_ratio", 0.0))
    if measured_ms > 0.0 and bound_ms > 0.0:
        # Unclamped: a fraction above 1.0 is the finding section 9 describes.
        row["fraction"] = bound_ms / measured_ms
    else:
        row["fraction"] = None
    notes = []
    if measured_ms <= 0.0:
        notes.append("the row states no time")
    if bound_device and measured_device and bound_device != measured_device:
        notes.append(
            f"the bound is for '{bound_device}' and the measurement is from "
            f"'{measured_device}': the bound was not validated on its target")
    if row.get("fraction") is not None and row["fraction"] > 1.0:
        notes.append("the fraction exceeds 1.0: the declared peak is too low, "
                     "or the measurement does not count the work the bound "
                     "counts")
    row["note"] = "; ".join(notes)
    return row


def summarize(bound_report: dict, measured_report: dict) -> dict:
    """The whole attainment record for one pair of reports."""
    bound_device = str(bound_report.get("device", ""))
    measured_device = str(measured_report.get("device", ""))
    joined = join(bound_report, measured_report)
    rows = [attainment_row(item, bound_device, measured_device)
            for item in joined]
    matched = [row for row in rows if row["fraction"] is not None]
    worst = min(matched, key=lambda row: row["fraction"]) if matched else None
    return {
        "schema": "nanochat.attainment.v1",
        "bound_device": bound_device,
        "bound_schema": bound_report.get("schema", ""),
        "measured_device": measured_device,
        "measured_schema": measured_report.get("schema", ""),
        # The device that validates the record is the one the benchmark ran on.
        "validated_on": measured_device,
        "rows": rows,
        "unbound_rows": len(rows) - len(matched),
        "worst_family": worst["family"] if worst else "",
        "worst_fraction": worst["fraction"] if worst else None,
    }


def _format_ms(value) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _format_fraction(value) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def markdown_rows(record: dict) -> str:
    """The rows of `docs/performance-model.md` section 10, as Markdown."""
    lines = [
        "| Family | Phase | Precision | Shape | Bound (ms) | Measured (ms) "
        "| Fraction | Issue/need | Limit | Validated on | Gap cause |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in record["rows"]:
        lines.append(
            "| {family} | {phase} | {precision} | {shape} | {bound} | {measured} "
            "| {fraction} | {ratio} | {limit} | {device} | {note} |".format(
                family=row["family"] or "-",
                phase=row["phase"] or "-",
                precision=row["precision"] or "-",
                shape=row["shape"] or "-",
                bound=_format_ms(row["bound_ms"]),
                measured=_format_ms(row["measured_ms"]),
                fraction=_format_fraction(row["fraction"]),
                ratio=(f"{row['measured_work_ratio']:.3f}"
                       if row["measured_work_ratio"] > 0.0 else "-"),
                limit=row["limit"] or "-",
                device=row["validated_on"] or "-",
                note=row["note"] or "-"))
    return "\n".join(lines)


def _text(record: dict) -> str:
    lines = [
        f"bound from   : {record['bound_schema']}"
        f" on {record['bound_device'] or 'an unnamed device'}",
        f"measured from: {record['measured_schema']}"
        f" on {record['measured_device'] or 'an unnamed device'}",
        f"validated on : {record['validated_on'] or 'an unnamed device'}",
    ]
    if record["unbound_rows"]:
        lines.append(
            f"unjoined     : {record['unbound_rows']} measured row(s) have no "
            "bound row")
    if record["worst_fraction"] is not None:
        lines.append(
            f"worst family : {record['worst_family']} at "
            f"{record['worst_fraction']:.3f} of its bound")
    lines.append("")
    lines.append(markdown_rows(record))
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    bench_path = ""
    roofline_path = ""
    as_json = False
    markdown_only = False
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument in ("--help", "-h"):
            print("usage: tools/nanochat attain --bench FILE --roofline FILE "
                  "[--json] [--markdown]")
            print("")
            print("Joins a nanochat.bench.v3 report with a nanochat.roofline.v3")
            print("report and prints the attainment record (docs/performance.md,")
            print("docs/performance-model.md section 10).")
            return 0
        if argument == "--json":
            as_json = True
        elif argument == "--markdown":
            markdown_only = True
        elif argument.startswith("--bench="):
            bench_path = argument.split("=", 1)[1]
        elif argument.startswith("--roofline="):
            roofline_path = argument.split("=", 1)[1]
        elif argument in ("--bench", "--roofline"):
            if index + 1 >= len(argv):
                raise SystemExit(f"attain: {argument} needs a value")
            index += 1
            if argument == "--bench":
                bench_path = argv[index]
            else:
                roofline_path = argv[index]
        else:
            raise SystemExit(f"attain: unknown option {argument}")
        index += 1
    if not bench_path or not roofline_path:
        raise SystemExit(
            "attain: --bench FILE and --roofline FILE are both required")
    record = summarize(_load(roofline_path, ROOFLINE_SCHEMA),
                       _load(bench_path, BENCH_SCHEMA))
    if as_json:
        print(json.dumps(record, indent=2, sort_keys=True))
    elif markdown_only:
        print(markdown_rows(record))
    else:
        print(_text(record))
    return 0
