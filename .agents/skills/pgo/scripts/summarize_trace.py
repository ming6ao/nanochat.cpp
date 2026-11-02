"""Attribute an Nsight Systems trace of the nanochat.cpp base stage.

The input is the directory ``tools/modal/app.py::profile_stage`` writes on the
runs Volume. ``nanochat_cpp.pgo analyze`` is the tested instrument, and this is
the standalone form that came first; the skill's ``analyze.py`` calls that one.

The tool answers three questions from the trace bytes and never from a
stopwatch:

* how much of the wall time the device was busy, and how much it was idle;
* where each idle stretch comes from, by naming the driver call that was in
  flight while the device had nothing to do;
* which kernels own the device time, and whether the step launches too many
  kernels that are too short to pay for their own launch.

It also writes a Chrome trace JSON, the format ``torch.profiler`` emits, so the
timeline opens in Perfetto or ``chrome://tracing``.

    python .agents/skills/pgo/scripts/summarize_trace.py \
        --dir /path/to/trace_dir --steps 20 --json trace.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

#: A gap shorter than this is the launch gap between two kernels, not an idle
#: stretch: the host queues the next kernel while the device finishes the last.
GAP_NS = 200_000


def _rows(path: Path) -> list[dict]:
    """The rows of one ``nsys stats`` CSV file, or no rows when it is absent."""
    if not Path(path).is_file():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row]


def _column(rows: list[dict], *names: str) -> str:
    """The first column of ``rows`` that matches one of ``names``.

    A name matches a column that contains it, case-insensitively, so a report
    whose header moved from ``Start (ns)`` to ``Start`` still answers.
    """
    if not rows:
        return ""
    header = list(rows[0])
    for name in names:
        for column in header:
            if name.lower() in column.lower():
                return column
    return ""


def _number(row: dict, column: str) -> float:
    """One cell as a float, with the report's thousands separators removed."""
    text = str(row.get(column, "") or "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


def _events(rows: list[dict], name_key: str = "Name") -> list[dict]:
    """The trace rows as ``{start, end, dur, name}`` intervals in nanoseconds."""
    start_column = _column(rows, "Start")
    duration_column = _column(rows, "Duration")
    name_column = _column(rows, name_key)
    events = []
    for row in rows:
        start = _number(row, start_column)
        duration = _number(row, duration_column)
        events.append({"start": start, "end": start + duration,
                       "dur": duration,
                       "name": str(row.get(name_column, "") or "?"),
                       "row": row})
    events.sort(key=lambda event: event["start"])
    return events


def _union_ns(events: list[dict]) -> float:
    """The time at least one event covers, so two streams never count twice."""
    total = 0.0
    current_start = None
    current_end = None
    for event in events:
        if current_start is None:
            current_start, current_end = event["start"], event["end"]
            continue
        if event["start"] <= current_end:
            current_end = max(current_end, event["end"])
            continue
        total += current_end - current_start
        current_start, current_end = event["start"], event["end"]
    if current_start is not None:
        total += current_end - current_start
    return total


def _gaps(kernels: list[dict]) -> list[dict]:
    """The idle stretches between two consecutive kernels."""
    gaps = []
    for before, after in zip(kernels, kernels[1:]):
        idle = after["start"] - before["end"]
        if idle > 0:
            gaps.append({"start": before["end"], "end": after["start"],
                         "ns": idle, "before": before["name"],
                         "after": after["name"]})
    return gaps


def _overlapping(api: list[dict], start: float, end: float) -> list[dict]:
    """The driver calls that were in flight inside one window."""
    inside = [call for call in api
              if call["start"] < end and call["end"] > start]
    inside.sort(key=lambda call: -call["dur"])
    return inside


def _ms(nanoseconds: float) -> float:
    """Nanoseconds as milliseconds, rounded for a report."""
    return round(nanoseconds / 1e6, 3)


def _bucket(duration_ns: float) -> str:
    """The size class of one kernel, by the launch overhead it must repay."""
    if duration_ns < 3_000:
        return "under 3 us"
    if duration_ns < 10_000:
        return "3 to 10 us"
    if duration_ns < 100_000:
        return "10 to 100 us"
    if duration_ns < 1_000_000:
        return "100 us to 1 ms"
    return "1 ms and over"


def _sum_report(rows: list[dict], limit: int) -> list[dict]:
    """The rows of one ``nsys stats`` sum report, the costliest first."""
    time_column = _column(rows, "Total Time")
    count_column = _column(rows, "Instances", "Count")
    name_column = _column(rows, "Name")
    parsed = []
    for row in rows:
        parsed.append({"name": str(row.get(name_column, "") or "?"),
                       "instances": int(_number(row, count_column)),
                       "total_ms": _ms(_number(row, time_column))})
    parsed.sort(key=lambda row: -row["total_ms"])
    return parsed[:limit]


def _kind(name: str) -> str:
    """The direction of one copy, or the word for anything else."""
    for kind in ("Device-to-Host", "Host-to-Device", "Device-to-Device"):
        if kind in name:
            return kind
    if "memset" in name:
        return "memset"
    return "kernel"


def _step_bounds(kernels: list[dict]) -> list[tuple[int, int]]:
    """The span of each step, from the trace itself and not from a clock.

    The trainer copies its per-row losses to the host once per forward pass, so
    that copy marks a step. Two of them appear per step in this tree, and the
    pair is what the bounds take: a step starts at one copy and ends at the
    same position in the next, which is why the pairs are counted and not the
    copies.
    """
    marks = [event["start"] for event in kernels
             if _kind(event["name"]) == "Device-to-Host"
             and 15_000 < _size_bytes(event) < 18_000]
    if len(marks) < 4:
        return []
    return list(zip(marks[2::2], marks[4::2]))


def _size_bytes(event: dict) -> int:
    """The payload of one copy, from the report's decimal megabyte column."""
    for column, value in event.get("row", {}).items():
        if "bytes" in column.lower():
            try:
                return int(round(float(str(value).replace(",", "")) * 1e6))
            except ValueError:
                return 0
    return 0


def step_profile(folder: Path, index: int = 8) -> dict:
    """The composition of one steady step.

    The window is bounded by the trace's own step marks, so a reader gets the
    step the machine ran and not one this tool guessed at. ``index`` picks a
    step by number; the first and the last are outside the steady window.
    """
    kernels = _events(_rows(folder / "timeline_cuda_gpu_trace.csv"))
    api = _events(_rows(folder / "timeline_cuda_api_trace.csv"))
    bounds = _step_bounds(kernels)
    if not bounds:
        raise SystemExit("no step marks in the GPU trace")
    start, end = bounds[min(index, len(bounds) - 1)]
    inside = [event for event in kernels if start <= event["start"] < end]
    calls = [event for event in api if start <= event["start"] < end]
    busy = _union_ns(inside)
    copies: dict[tuple[str, int], list] = {}
    for event in inside:
        kind = _kind(event["name"])
        if kind == "kernel":
            continue
        key = (kind, _size_bytes(event))
        entry = copies.setdefault(key, [0, 0])
        entry[0] += 1
        entry[1] += _size_bytes(event)
    stalls = []
    for before, after in zip(inside, inside[1:]):
        idle = after["start"] - (before["start"] + before["dur"])
        if idle < 50_000:
            continue
        during = _overlapping(calls, before["start"] + before["dur"],
                              after["start"])
        stalls.append({"idle_ms": _ms(idle),
                       "after": before["name"],
                       "after_bytes": _size_bytes(before),
                       "host_cuda_ms": _ms(sum(call["dur"] for call in during)),
                       "host_elsewhere_ms": _ms(
                           idle - sum(call["dur"] for call in during))})
    stalls.sort(key=lambda stall: -stall["idle_ms"])
    by_kind: dict[str, int] = {}
    for event in inside:
        by_kind[_kind(event["name"])] = by_kind.get(_kind(event["name"]), 0) + 1
    return {
        "step_index": min(index, len(bounds) - 1),
        "steps_found": len(bounds),
        "wall_ms": _ms(end - start),
        "operations": len(inside),
        "operations_by_kind": by_kind,
        "device_busy_ms": _ms(busy),
        "device_busy_percent": round(100.0 * busy / (end - start), 1),
        "driver_calls": len(calls),
        "host_in_cuda_api_ms": _ms(sum(call["dur"] for call in calls)),
        "copies": [{"kind": kind, "bytes": size, "count": count,
                    "total_mb": round(total / 1e6, 2)}
                   for (kind, size), (count, total) in sorted(
                       copies.items(), key=lambda item: -item[1][1])],
        "stalls": stalls[:10],
        "stall_total_ms": round(sum(stall["idle_ms"] for stall in stalls), 3),
        "host_own_code_ms": round(
            sum(stall["host_elsewhere_ms"] for stall in stalls), 3),
    }


def summarize(folder: Path, steps: int, json_path: Path | None) -> dict:
    """Read one trace directory and return the attribution document."""
    gpu_rows = _rows(folder / "timeline_cuda_gpu_trace.csv")
    api_rows = _rows(folder / "timeline_cuda_api_trace.csv")
    kernels = _events(gpu_rows)
    api = _events(api_rows)
    if not kernels:
        raise SystemExit(f"no GPU kernels in {folder}/timeline_cuda_gpu_trace.csv")

    span = max(event["end"] for event in api + kernels) - kernels[0]["start"]
    busy = _union_ns(kernels)
    gaps = _gaps(kernels)
    real_gaps = [gap for gap in gaps if gap["ns"] >= GAP_NS]
    idle = span - busy

    buckets: dict[str, dict] = {}
    for kernel in kernels:
        name = _bucket(kernel["dur"])
        bucket = buckets.setdefault(name, {"kernels": 0, "total_ns": 0.0})
        bucket["kernels"] += 1
        bucket["total_ns"] += kernel["dur"]

    biggest = sorted(real_gaps, key=lambda gap: -gap["ns"])[:8]
    for gap in biggest:
        calls = _overlapping(api, gap["start"], gap["end"])
        gap["calls"] = [{"name": call["name"], "ms": _ms(call["dur"])}
                        for call in calls[:4]]
        gap["host_only_ms"] = _ms(gap["ns"] - sum(call["dur"] for call in calls))

    per_name: dict[str, dict] = {}
    for kernel in kernels:
        entry = per_name.setdefault(kernel["name"],
                                    {"kernels": 0, "total_ns": 0.0,
                                     "durs": []})
        entry["kernels"] += 1
        entry["total_ns"] += kernel["dur"]
        entry["durs"].append(kernel["dur"])
    top = sorted(per_name.items(), key=lambda item: -item[1]["total_ns"])[:15]

    document = {
        "trace_dir": str(folder),
        "steps": int(steps),
        "wall_ms": _ms(span),
        "wall_per_step_ms": _ms(span) / max(1, steps),
        "gpu_busy_ms": _ms(busy),
        "gpu_busy_percent": round(100.0 * busy / span, 2) if span else 0.0,
        "gpu_idle_ms": _ms(idle),
        "kernels": len(kernels),
        "kernels_per_step": round(len(kernels) / max(1, steps), 1),
        "kernel_mean_us": round(sum(event["dur"] for event in kernels)
                                / max(1, len(kernels)) / 1e3, 2),
        "kernels_per_stream": len({_number(event["row"],
                                          _column(gpu_rows, "Strm", "Stream"))
                                  for event in kernels}),
        "buckets": {name: {"kernels": data["kernels"],
                           "total_ms": _ms(data["total_ns"]),
                           "percent_of_device": round(
                               100.0 * data["total_ns"] / busy, 2) if busy
                           else 0.0}
                    for name, data in buckets.items()},
        "top_kernels": [
            {"name": name, "kernels": data["kernels"],
             "total_ms": _ms(data["total_ns"]),
             "mean_us": round(data["total_ns"] / data["kernels"] / 1e3, 2),
             "max_us": round(max(data["durs"]) / 1e3, 1)}
            for name, data in top],
        "gaps_over_200us": {
            "count": len(real_gaps),
            "total_ms": _ms(sum(gap["ns"] for gap in real_gaps)),
            "biggest": biggest,
        },
        "api_calls": len(api),
        "api_top": _sum_report(_rows(folder / "base_cuda_api_sum.csv"), 15),
        "kernel_sum": _sum_report(_rows(folder / "base_cuda_gpu_kern_sum.csv"),
                                  15),
        "memory_sum": _sum_report(_rows(folder / "base_cuda_gpu_mem_time_sum.csv"),
                                  10),
    }
    if json_path is not None:
        _write_chrome_trace(json_path, kernels, api)
        document["chrome_trace"] = str(json_path)
    return document


def _write_chrome_trace(path: Path, kernels: list[dict],
                        api: list[dict]) -> None:
    """Write the trace as Chrome trace JSON, the shape Perfetto opens.

    One track holds the device kernels and one holds the driver calls, so an
    idle stretch on the device shows the call the host was inside at the time.
    """
    if not kernels:
        return
    origin = min(event["start"] for event in kernels + api or kernels)
    entries = []
    for event in kernels:
        entries.append({"ph": "X", "cat": "kernel",
                        "name": event["name"],
                        "ts": round((event["start"] - origin) / 1e3, 3),
                        "dur": round(event["dur"] / 1e3, 3),
                        "pid": 1, "tid": 1})
    for event in api:
        entries.append({"ph": "X", "cat": "cuda_api",
                        "name": event["name"],
                        "ts": round((event["start"] - origin) / 1e3, 3),
                        "dur": round(event["dur"] / 1e3, 3),
                        "pid": 2, "tid": 2})
    entries.sort(key=lambda entry: entry["ts"])
    document = {
        "traceEvents": entries,
        "displayTimeUnit": "ms",
        "otherData": {"process_names": [
            {"name": "H100 device kernels", "process_name": "gpu"},
            {"name": "CUDA driver calls", "process_name": "host"}]},
    }
    Path(path).write_text(json.dumps(document), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Read one trace directory and print the attribution."""
    parser = argparse.ArgumentParser(prog="summarize_trace.py",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=".",
                        help="the trace directory on this host")
    parser.add_argument("--steps", type=int, default=20,
                        help="the traced step count, for the per-step rates")
    parser.add_argument("--json", default="",
                        help="write a Chrome trace JSON at this path")
    parser.add_argument("--step", type=int, default=-1,
                        help="also report one steady step by index")
    options = parser.parse_args(argv)
    folder = Path(options.dir)
    document = summarize(folder, options.steps,
                         Path(options.json) if options.json else None)
    if options.step >= 0:
        document["step"] = step_profile(folder, options.step)
    print(json.dumps(document, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
