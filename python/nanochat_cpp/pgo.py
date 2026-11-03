"""Profile-guided optimization: measure, trace, analyze, compare, record.

The loop of ``docs/profile-guided-optimization.md`` has two halves. One half
needs judgement: which two ideas the trace supports, and how to write them. The
other half must be exact: the step time, the ceilings, the comparison, and the
state. This module is the exact half, so no number in a verdict comes from a
language model.

Four facts come out of one trace directory:

* how much of a step the device spent working, and how much it spent waiting;
* which device-to-host copies and which stalls own the waiting;
* a *signature* for each measured cost, with a ceiling: the share of the step
  that the signature owns;
* the two ideas with the largest ceilings, ranked by that ceiling.

The module reads the CSV reports ``nsys stats`` writes, the ``metrics.jsonl``
the driver writes, and the JSON documents the Modal functions return. It uses
the standard library only, so ``gate_stdlib_imports_test`` keeps passing and the
Bazel test needs no network, no GPU, and no Modal account.

Typical use, one iteration::

    python -m nanochat_cpp.pgo init --stage base --device h100 --job job.json
    python -m nanochat_cpp.pgo measure --before      # two untraced runs
    python -m nanochat_cpp.pgo trace                 # Nsight Systems on Modal
    python -m nanochat_cpp.pgo analyze               # signatures and ceilings
    #   the analyst writes ideas.json, the implementers write patch.diff
    python -m nanochat_cpp.pgo measure --after
    python -m nanochat_cpp.pgo compare --apply       # verdict.json, loop.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

#: The Modal command line. A host that keeps the SDK in a virtual environment
#: names the executable here, so the loop works without a second ``PATH`` entry.
MODAL = os.environ.get("NANOCHAT_MODAL", "modal")

#: The stages a target may name, in chain order. ``eval`` is here although it
#: trains nothing: its throughput is measurable and its trace is useful.
STAGES = ("base", "sft", "rl", "eval")

#: The devices a target may name. One card each, on Modal.
DEVICES = ("h100", "t4")

MEASUREMENT_SCHEMA = "nanochat.pgo.measurement.v1"
ANALYSIS_SCHEMA = "nanochat.pgo.analysis.v1"
IDEAS_SCHEMA = "nanochat.pgo.ideas.v1"
LOOP_SCHEMA = "nanochat.pgo.loop.v1"
VERDICT_SCHEMA = "nanochat.pgo.verdict.v1"

#: The bar the pair must clear, as a fraction of the before value. The pair is
#: the unit, so one idea of the pair may be smaller than this.
DEFAULT_THRESHOLD = 0.05

#: The baseline window: 50 steps or 180 seconds, whichever ends first. The
#: traced run uses both. The scored run uses the seconds alone, because 50 steps
#: is about 20 seconds here and its median is noise.
DEFAULT_TRACE_STEPS = 50
DEFAULT_SECONDS = 180

#: The step count of the scored run. Fifty steps is about 20 seconds here, and
#: its median moves by more than the 5 percent bar. Two hundred steps is about
#: 65 seconds, which sits inside the 180 second cap and leaves a steady window.
DEFAULT_MEASURE_STEPS = 200

#: Two runs per measurement point, and the better one stands. Interference only
#: ever slows a run, so the minimum is the better estimate of the true rate.
DEFAULT_RUNS = 2

#: The iteration cap when the operator names none.
DEFAULT_MAX_ITERATIONS = 5

#: The stages whose step time is a training step, and the one that measures a
#: throughput instead. ``eval`` trains nothing, so it has no MFU.
METRICS = {
    "base": ("step_ms", "lower"),
    "sft": ("step_ms", "lower"),
    "rl": ("step_ms", "lower"),
    "eval": ("samples_per_second", "higher"),
}

#: The step records at the start of a run carry start-up, and the last one
#: carries the checkpoint write. Neither belongs in the median.
WARMUP_STEPS = 2

#: A gap shorter than this is the launch gap between two kernels, not a stall.
GAP_NS = 200_000

#: A kernel shorter than this costs more to launch than it spends working.
SMALL_KERNEL_NS = 3_000

#: A copy smaller than this is a scalar the host could hold instead.
SCALAR_COPY_BYTES = 1024

#: A signature below this share of the step is not worth a slot in the pair.
SIGNATURE_FLOOR = 0.005

#: A signature that no idea answers. `idle_unexplained` is the device time that
#: no stall accounts for: the trace cannot see it, so the next trace must sample
#: the CPU. It belongs in the table and never in the pair.
UNActionable = frozenset({"idle_unexplained"})

#: A signature whose cost includes another one's. Pairing the two would count
#: the same time twice, so the ranking keeps the larger one and drops the other.
#: ``driver_call_overhead`` is the host inside a driver call, and a blocking
#: copy and a launch are what those calls are.
CONTAINS = {
    "driver_call_overhead": ("blocking_dtoh_scalars", "dtoh_bytes",
                             "launches_under_3us"),
}

#: The artifacts ``profile_stage`` writes beside the report, by suffix. The
#: trace the loop reads is ``timeline_cuda_gpu_trace.csv``; the sums name the
#: temperature of the run without a second query.
TRACE_FILES = (
    "{stage}.nsys-rep",
    "{stage}.sqlite",
    "{stage}_cuda_gpu_kern_sum.csv",
    "{stage}_cuda_api_sum.csv",
    "{stage}_cuda_gpu_mem_time_sum.csv",
    "{stage}_osrt_sum.csv",
    "timeline_cuda_gpu_trace.csv",
    "timeline_cuda_api_trace.csv",
)


# ---------------------------------------------------------------------------
# CSV reports from Nsight Systems
# ---------------------------------------------------------------------------


def _rows(path) -> list[dict]:
    """The rows of one report CSV, or no rows when the file is absent."""
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row]


def _report(trace_dir, suffix: str) -> list[dict]:
    """The rows of the one report whose name ends with ``suffix``."""
    for path in sorted(Path(trace_dir).glob(f"*{suffix}")):
        rows = _rows(path)
        if rows:
            return rows
    return []


def _column(rows: Sequence[Mapping], *names: str) -> str:
    """The first column that matches one of ``names``, case-insensitively."""
    if not rows:
        return ""
    header = list(rows[0])
    for name in names:
        for column in header:
            if name.lower() in column.lower():
                return column
    return ""


def _number(row: Mapping, column: str) -> float:
    """One cell as a float, with the report's separators removed."""
    text = str(row.get(column, "") or "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


def _size_bytes(event: Mapping) -> int:
    """The payload of one copy, from the report's decimal megabyte column."""
    for column, value in event.get("row", {}).items():
        if "bytes" in column.lower():
            try:
                return int(round(float(str(value).replace(",", "")) * 1e6))
            except ValueError:
                return 0
    return 0


def _kind(name: str) -> str:
    """The direction of one copy, or the word for anything else."""
    for kind in ("Device-to-Host", "Host-to-Device", "Device-to-Device"):
        if kind in name:
            return kind
    if "memset" in name:
        return "memset"
    return "kernel"


def _events(rows: Sequence[Mapping]) -> list[dict]:
    """The rows as ``{start, end, dur, name, size}`` intervals in nanoseconds."""
    start_column = _column(rows, "Start")
    duration_column = _column(rows, "Duration")
    name_column = _column(rows, "Name")
    events = []
    for row in rows:
        start = _number(row, start_column)
        duration = _number(row, duration_column)
        events.append({"start": start, "end": start + duration,
                       "dur": duration,
                       "name": str(row.get(name_column, "") or "?"),
                       "size": _size_bytes({"row": row}),
                       "row": row})
    events.sort(key=lambda event: event["start"])
    return events


def _union_ns(events: Sequence[Mapping]) -> float:
    """The time at least one event covers, so two streams never count twice."""
    total = 0.0
    start = None
    end = None
    for event in events:
        if start is None:
            start, end = event["start"], event["end"]
        elif event["start"] <= end:
            end = max(end, event["end"])
        else:
            total += end - start
            start, end = event["start"], event["end"]
    if start is not None:
        total += end - start
    return total


def _ms(nanoseconds: float) -> float:
    """Nanoseconds as milliseconds, rounded for a report."""
    return round(float(nanoseconds) / 1e6, 3)


def _share(part: float, whole: float) -> float:
    """``part`` over ``whole``, clamped to a fraction a ceiling may hold."""
    if whole <= 0:
        return 0.0
    return round(min(1.0, max(0.0, float(part) / float(whole))), 4)


def _bucket(duration_ns: float) -> str:
    """The size class of one kernel, by the launch overhead it must repay."""
    if duration_ns < SMALL_KERNEL_NS:
        return "under 3 us"
    if duration_ns < 10_000:
        return "3 to 10 us"
    if duration_ns < 100_000:
        return "10 to 100 us"
    if duration_ns < 1_000_000:
        return "100 us to 1 ms"
    return "1 ms and over"


# ---------------------------------------------------------------------------
# One step, cut out of the trace by the trace's own marks
# ---------------------------------------------------------------------------


def step_bounds(kernels: Sequence[Mapping],
                mark_bytes: int | None = None) -> list[tuple[int, int]]:
    """The span of each step, taken from the trace and not from a clock.

    The trainer copies its per-row losses to the host once per forward pass, so
    that copy marks a step. Two of them appear per step in this tree, so a step
    starts at one pair and ends at the same position in the next pair.

    That copy is four bytes per row, so its size follows the shape: 16 KB at
    the 4096-row reference profile, and 64 KB at 16384 rows. ``mark_bytes``
    names the size when the shape is not the reference one. The trace states a
    payload in decimal megabytes, so the explicit case allows one percent of
    rounding.
    """
    low, high = 15_001, 17_999
    if mark_bytes is not None:
        slack = max(1_000, int(mark_bytes) // 100)
        low, high = int(mark_bytes) - slack, int(mark_bytes) + slack
    marks = [event["start"] for event in kernels
             if _kind(event["name"]) == "Device-to-Host"
             and low <= event["size"] <= high]
    if len(marks) < 4:
        return []
    return list(zip(marks[2::2], marks[4::2]))


def _stalls(step: Sequence[Mapping], api: Sequence[Mapping],
            floor_ns: int = 50_000) -> list[dict]:
    """Every idle stretch long enough to be a stall, with the host's calls."""
    stalls = []
    for before, after in zip(step, step[1:]):
        idle = after["start"] - (before["start"] + before["dur"])
        if idle < floor_ns:
            continue
        inside = [call for call in api
                  if call["start"] < after["start"]
                  and call["end"] > before["start"] + before["dur"]]
        host_api = sum(call["dur"] for call in inside)
        stalls.append({
            "start": before["start"] + before["dur"],
            "idle_ns": idle,
            "idle_ms": _ms(idle),
            "after": before["name"],
            "after_bytes": before["size"],
            "host_cuda_ms": _ms(host_api),
            "host_own_ms": _ms(idle - host_api),
        })
    return stalls


def step_profile(kernels: Sequence[Mapping], api: Sequence[Mapping],
                 index: int = 6, mark_bytes: int | None = None) -> dict:
    """The composition of one steady step: the unit the loop scores.

    ``index`` picks the step by number. The first steps of a run carry the
    library load and the first batch, and the last one carries the checkpoint
    write, so a caller reads the middle. ``mark_bytes`` is the size of the
    per-row loss copy, or ``None`` for the reference profile's 16 KB.
    """
    bounds = step_bounds(kernels, mark_bytes=mark_bytes)
    if not bounds:
        raise PgoError("no step marks in the GPU trace: the trainer copied no "
                       "per-row losses, so this trace is not a step loop")
    start, end = bounds[min(index, len(bounds) - 1)]
    inside = [event for event in kernels if start <= event["start"] < end]
    calls = [event for event in api if start <= event["start"] < end]
    busy = _union_ns(inside)
    by_kind: dict = {}
    copies: dict = {}
    for event in inside:
        kind = _kind(event["name"])
        by_kind[kind] = by_kind.get(kind, 0) + 1
        if kind == "kernel":
            continue
        key = f"{kind} {event['size']}"
        entry = copies.setdefault(key, {"kind": kind, "bytes": event["size"],
                                        "count": 0, "total_bytes": 0,
                                        "device_ms": 0.0})
        entry["count"] += 1
        entry["total_bytes"] += event["size"]
        entry["device_ms"] = round(entry["device_ms"] + _ms(event["dur"]), 3)
    stalls = sorted(_stalls(inside, calls), key=lambda stall: -stall["idle_ns"])
    per_name: dict = {}
    for event in inside:
        if _kind(event["name"]) != "kernel":
            continue
        entry = per_name.setdefault(event["name"], {"count": 0, "ns": 0.0})
        entry["count"] += 1
        entry["ns"] += event["dur"]
    top = sorted(per_name.items(), key=lambda item: -item[1]["ns"])[:12]
    return {
        "wall_ms": _ms(end - start),
        "operations": len(inside),
        "operations_by_kind": by_kind,
        "device_busy_ms": _ms(busy),
        "device_busy_percent": round(100.0 * busy / (end - start), 1),
        "driver_calls": len(calls),
        "host_in_cuda_api_ms": _ms(sum(call["dur"] for call in calls)),
        "host_own_code_ms": round(sum(stall["host_own_ms"] for stall in stalls),
                                  3),
        "copies": sorted(copies.values(), key=lambda row: -row["total_bytes"]),
        "kernels": [{"name": name, "count": data["count"],
                     "device_ms": _ms(data["ns"]),
                     "mean_us": round(data["ns"] / data["count"] / 1e3, 2)}
                    for name, data in top],
        "stalls": stalls[:8],
        "stall_total_ms": round(sum(stall["idle_ms"] for stall in stalls), 3),
    }


# ---------------------------------------------------------------------------
# Signatures: the measured cost, and the ceiling an idea may claim
# ---------------------------------------------------------------------------


def signatures(step: Mapping, api_rows: Sequence[Mapping]) -> list[dict]:
    """The cost signatures of one step, each with a ceiling.

    A ceiling is the share of the step that one signature owns, and it is the
    only reason an idea may enter the pair. The measurement decides whether the
    idea reaches it: a ceiling is an upper bound, not a promise.
    """
    wall = float(step["wall_ms"]) or 1.0
    found = []

    def add(identifier, title, ceiling_ms, evidence, hint=""):
        ceiling = _share(ceiling_ms, wall)
        if ceiling < SIGNATURE_FLOOR:
            return
        found.append({"id": identifier, "title": title, "ceiling": ceiling,
                      "ceiling_ms": round(float(ceiling_ms), 3),
                      "evidence": evidence, "hint": hint})

    host_own = float(step["host_own_code_ms"])
    add("host_own_code", "the host spends its own code time while the device "
        "waits", host_own,
        {"stalls": [{"idle_ms": stall["idle_ms"],
                     "host_own_ms": stall["host_own_ms"],
                     "after": stall["after"],
                     "after_bytes": stall["after_bytes"]}
                    for stall in step["stalls"][:6]]},
        "the largest stall names the copy that ends before it")

    add("driver_call_overhead", "the host sits inside a driver call",
        float(step["host_in_cuda_api_ms"]),
        {"calls": _api_counts(api_rows, 6)}, "one call per queued operation")

    scalar_stalls = [stall for stall in step["stalls"]
                     if _kind(stall["after"]) == "Device-to-Host"
                     and 0 <= stall["after_bytes"] <= SCALAR_COPY_BYTES]
    add("blocking_dtoh_scalars",
        "a scalar is read back with a blocking copy, which drains the queue",
        sum(stall["idle_ms"] for stall in scalar_stalls),
        {"copies": len(scalar_stalls),
         "idle_ms": [stall["idle_ms"] for stall in scalar_stalls[:8]]},
        "one read per layer of a value the device already holds")

    dtoh = [row for row in step["copies"] if row["kind"] == "Device-to-Host"]
    add("dtoh_bytes", "the host reads a bulk buffer it reduces or converts",
        sum(row["device_ms"] for row in dtoh),
        {"total_mb": round(sum(row["total_bytes"] for row in dtoh) / 1e6, 2),
         "largest": [{"bytes": row["bytes"], "count": row["count"]}
                     for row in dtoh[:4]]},
        "a reduction over these bytes belongs on the device")

    small = [kernel for kernel in step["kernels"]
             if kernel["mean_us"] * 1e3 < SMALL_KERNEL_NS]
    add("launches_under_3us",
        "kernels too short to repay their own launch",
        sum(kernel["device_ms"] for kernel in small),
        {"kernels": sum(kernel["count"] for kernel in small),
         "share_of_launches": _share(
             sum(kernel["count"] for kernel in small),
             float(step["operations_by_kind"].get("kernel", 1))),
         "names": [kernel["name"] for kernel in small[:5]]},
        "fuse the pointwise chain, or batch the call")

    tiny = [kernel for kernel in step["kernels"]
            if "32x32" in kernel["name"] or "16x16" in kernel["name"]]
    add("tiny_gemm_tile", "a GEMM runs a tile far below the card's shape",
        sum(kernel["device_ms"] for kernel in tiny),
        {"kernels": sum(kernel["count"] for kernel in tiny),
         "names": [kernel["name"] for kernel in tiny[:3]]},
        "one batched call, or a retile")

    dtod = [row for row in step["copies"] if row["kind"] == "Device-to-Device"]
    add("dtod_gather", "the optimizer gathers a group into one buffer",
        sum(row["device_ms"] for row in dtod),
        {"total_mb": round(sum(row["total_bytes"] for row in dtod) / 1e6, 2)},
        "bandwidth, not a stall: expect no gain")

    stall_covered = float(step["stall_total_ms"])
    idle = float(step["wall_ms"]) - float(step["device_busy_ms"])
    add("idle_unexplained",
        "the device waits and no stall names the cause", idle - stall_covered,
        {"idle_ms": round(idle, 3), "stalls_ms": stall_covered},
        "a host cost the trace cannot see: collect --sample=cpu next time")

    found.sort(key=lambda row: -row["ceiling"])
    return found


def _api_counts(rows: Sequence[Mapping], limit: int) -> list[dict]:
    """The costliest driver calls of one report, by total time."""
    time_column = _column(rows, "Total Time")
    count_column = _column(rows, "Num Calls", "Count")
    name_column = _column(rows, "Name")
    parsed = []
    for row in rows:
        parsed.append({"name": str(row.get(name_column, "") or "?"),
                       "calls": int(_number(row, count_column)),
                       "total_ms": _ms(_number(row, time_column))})
    parsed.sort(key=lambda row: -row["total_ms"])
    return parsed[:limit]


def analyze_trace(trace_dir, *, stage: str = "base", device: str = "h100",
                  steps: int = DEFAULT_TRACE_STEPS,
                  step_index: int = 6,
                  mark_bytes: int | None = None) -> dict:
    """Read one trace directory and return the analysis document.

    The document is the only thing an analyst reads about the trace: the whole
    trace for context, one steady step for the ceilings, and the signature table
    that the ranking sorts.
    """
    gpu_rows = _report(trace_dir, "timeline_cuda_gpu_trace.csv")
    api_rows = _report(trace_dir, "timeline_cuda_api_trace.csv")
    kernels = _events(gpu_rows)
    api = _events(api_rows)
    if not kernels:
        raise PgoError(f"no GPU kernels in {trace_dir}: the report is missing "
                       "timeline_cuda_gpu_trace.csv")
    span = max(event["end"] for event in api + kernels) - kernels[0]["start"]
    busy = _union_ns(kernels)
    buckets: dict = {}
    for event in kernels:
        if _kind(event["name"]) != "kernel":
            continue
        name = _bucket(event["dur"])
        entry = buckets.setdefault(name, {"kernels": 0, "ns": 0.0})
        entry["kernels"] += 1
        entry["ns"] += event["dur"]
    step = step_profile(kernels, api, index=step_index, mark_bytes=mark_bytes)
    return {
        "schema": ANALYSIS_SCHEMA,
        "stage": stage,
        "device": device,
        "steps": int(steps),
        "trace_dir": str(trace_dir),
        "wall_ms": _ms(span),
        "gpu_idle_percent": round(100.0 * (1.0 - busy / span), 1) if span else 0.0,
        "device_busy_percent": round(100.0 * busy / span, 1) if span else 0.0,
        "operations": len(kernels),
        "bucket": {name: {"kernels": data["kernels"],
                          "device_ms": _ms(data["ns"]),
                          "share": _share(data["ns"], busy)}
                   for name, data in sorted(buckets.items(),
                                            key=lambda item: -item[1]["ns"])},
        "step": step,
        "signatures": signatures(step, api_rows),
    }


def chrome_trace(trace_dir, out) -> Path:
    """Write the trace as Chrome trace JSON, the shape Perfetto opens.

    One track holds the device operations and one holds the driver calls, so an
    idle stretch on the device shows the call the host was inside at the time.
    This is the same format ``torch.profiler`` emits.
    """
    kernels = _events(_report(trace_dir, "timeline_cuda_gpu_trace.csv"))
    api = _events(_report(trace_dir, "timeline_cuda_api_trace.csv"))
    if not kernels:
        raise PgoError(f"no GPU kernels in {trace_dir}")
    origin = min(event["start"] for event in kernels + api)
    entries = []
    for event in kernels:
        entries.append({"ph": "X", "cat": "device", "name": event["name"],
                        "ts": round((event["start"] - origin) / 1e3, 3),
                        "dur": round(event["dur"] / 1e3, 3),
                        "pid": 1, "tid": 1})
    for event in api:
        entries.append({"ph": "X", "cat": "host", "name": event["name"],
                        "ts": round((event["start"] - origin) / 1e3, 3),
                        "dur": round(event["dur"] / 1e3, 3),
                        "pid": 2, "tid": 2})
    entries.sort(key=lambda entry: entry["ts"])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "traceEvents": entries,
        "displayTimeUnit": "ms",
        "otherData": {"process_names": [
            {"name": "device operations", "process_name": "device"},
            {"name": "CUDA driver calls", "process_name": "host"}]},
    }, separators=(",", ":")), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# The measurement: what the loop scores
# ---------------------------------------------------------------------------


def measurement_report(document: Mapping, *, stage: str,
                       device: str) -> dict:
    """Turn one container answer into the scored measurement.

    ``document`` carries one entry per run. An entry with per-step times is
    reduced to the median of its steady window. An entry with only the stage's
    own seconds is divided by its step count, because SFT and eval report no
    per-step time yet (``docs/profile-guided-optimization.md`` row PGO.4).
    """
    metric, direction = METRICS[stage]
    runs = []
    for raw in document.get("runs") or ():
        steps = max(1, int(raw.get("steps") or 0))
        times = [float(value) for value in (raw.get("step_ms") or ())]
        if times:
            steady = times[WARMUP_STEPS:] or times
            step_ms = round(statistics.median(steady), 4)
            source = "metrics"
        else:
            step_ms = round(1000.0 * float(raw.get("stage_seconds") or 0.0)
                            / steps, 4)
            source = "stage_seconds"
        runs.append({
            "step_ms": step_ms,
            "stage_seconds": round(float(raw.get("stage_seconds") or 0.0), 3),
            "tokens_per_second": raw.get("tokens_per_second"),
            "samples_per_second": raw.get("samples_per_second"),
            "mfu": raw.get("mfu"),
            "source": source,
        })
    if not runs:
        raise PgoError("the container reported no run")
    values = [run[metric] for run in runs if run.get(metric) is not None]
    if not values:
        raise PgoError(f"the container reported no {metric}")
    best = min(values) if direction == "lower" else max(values)
    throughput = [run["tokens_per_second"] for run in runs
                  if run.get("tokens_per_second")]
    mfu = [run["mfu"] for run in runs if run.get("mfu")]
    return {
        "schema": MEASUREMENT_SCHEMA,
        "stage": stage,
        "device": device,
        "steps": int(document.get("steps") or 0),
        "metric": {"name": metric, "direction": direction},
        "value": best,
        "step_ms": best if metric == "step_ms" else min(
            run["step_ms"] for run in runs),
        "tokens_per_second": max(throughput) if throughput else None,
        "mfu": max(mfu) if mfu else None,
        "runs": runs,
        "source": str(document.get("source") or ""),
    }


def compare(before: Mapping, after: Mapping,
            threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Compare two measurements and decide whether the pair stays.

    The pair wins when the metric moved by at least ``threshold`` **and** the
    runs of the two points do not overlap. The second condition is the one that
    makes a 5 percent decision possible: the measured steady steps of one
    configuration spread by more than that.
    """
    metric = str(before["metric"]["name"])
    direction = str(before["metric"]["direction"])
    was = float(before["value"])
    now = float(after["value"])
    gain = ((was - now) / was if direction == "lower" else (now - was) / was)
    was_runs = [float(run[metric]) for run in before["runs"]]
    now_runs = [float(run[metric]) for run in after["runs"]]
    if direction == "lower":
        disjoint = max(now_runs) < min(was_runs)
    else:
        disjoint = min(now_runs) > max(was_runs)
    mfu_before = before.get("mfu")
    mfu_after = after.get("mfu")
    mfu_gain = None
    if mfu_before and mfu_after:
        mfu_gain = round((float(mfu_after) - float(mfu_before))
                         / float(mfu_before), 4)
    kept = bool(gain >= threshold and disjoint)
    return {
        "schema": VERDICT_SCHEMA,
        "metric": metric,
        "direction": direction,
        "threshold": float(threshold),
        "before": round(was, 4),
        "after": round(now, 4),
        "gain": round(gain, 4),
        "overlap": not disjoint,
        "mfu_before": mfu_before,
        "mfu_after": mfu_after,
        "mfu_gain": mfu_gain,
        "decision": "kept" if kept else "reverted",
        "reason": _verdict_reason(gain, disjoint, threshold),
    }


def _verdict_reason(gain: float, disjoint: bool, threshold: float) -> str:
    """One sentence for the verdict, so a reader need not re-derive it."""
    if gain < threshold and not disjoint:
        return (f"the gain is {gain:.1%}, below the {threshold:.0%} bar, and "
                "the two points overlap")
    if gain < threshold:
        return f"the gain is {gain:.1%}, below the {threshold:.0%} bar"
    if not disjoint:
        return (f"the gain is {gain:.1%} but the two points overlap, so the "
                "difference is inside the noise")
    return f"the gain is {gain:.1%} and the two points do not overlap"


# ---------------------------------------------------------------------------
# The pair
# ---------------------------------------------------------------------------


def rank(signatures_: Sequence[Mapping],
         rejected: Sequence[str] = ()) -> dict:
    """The two ideas with the largest ceilings, and their pair ceiling.

    An individual ceiling may sit below the bar. The pair is the unit, so the
    two are summed and clamped to 0.9: no pair of ideas removes a whole step.
    """
    skipped = set(rejected) | UNActionable
    usable = [row for row in signatures_ if row["id"] not in skipped]
    top: list = []
    dropped: set = set()
    for row in usable:
        if row["id"] in dropped or len(top) == 2:
            continue
        top.append(row)
        dropped.update(CONTAINS.get(row["id"], ()))
        dropped.add(row["id"])
    total = sum(float(row["ceiling"]) for row in top)
    return {
        "ids": [row["id"] for row in top],
        "ceilings": [row["ceiling"] for row in top],
        "pair_ceiling": round(min(0.9, total), 4),
        "bar": DEFAULT_THRESHOLD,
        "proceed": bool(total >= DEFAULT_THRESHOLD),
        "dropped": sorted(dropped - {row["id"] for row in top}),
        "unactionable": sorted(UNActionable),
        "rejected": list(rejected),
    }


# ---------------------------------------------------------------------------
# The loop state
# ---------------------------------------------------------------------------


class PgoError(Exception):
    """A fault the loop reports as a message and an exit status."""


def run_root(root, stage: str, device: str) -> Path:
    """The directory of one target: ``<root>/<stage>-<device>``."""
    return Path(root) / f"{stage}-{device}"


def iteration_dir(folder, iteration: int) -> Path:
    """The directory of one iteration: ``iter-01``, ``iter-02``, ..."""
    return Path(folder) / f"iter-{int(iteration):02d}"


def new_loop(stage: str, device: str, *, job: str = "",
             max_iterations: int = DEFAULT_MAX_ITERATIONS,
             threshold: float = DEFAULT_THRESHOLD,
             trace_steps: int = DEFAULT_TRACE_STEPS,
             seconds: int = DEFAULT_SECONDS,
             runs: int = DEFAULT_RUNS) -> dict:
    """The state document of one target, before its first measurement."""
    if stage not in STAGES:
        raise PgoError(f"unknown stage {stage!r}; the stages are "
                       f"{', '.join(STAGES)}")
    if device not in DEVICES:
        raise PgoError(f"unknown device {device!r}; the devices are "
                       f"{', '.join(DEVICES)}")
    metric, direction = METRICS[stage]
    return {
        "schema": LOOP_SCHEMA,
        "target": {"stage": stage, "device": device},
        "job": str(job),
        "budget": {"trace_steps": int(trace_steps),
                   "seconds": int(seconds), "measure_runs": int(runs)},
        "metric": {"name": metric, "direction": direction},
        "threshold": float(threshold),
        "max_iterations": int(max_iterations),
        "iteration": 1,
        "baseline": None,
        "history": [],
        "rejected": [],
    }


def load_loop(path) -> dict:
    """Read a loop document, and refuse one this module cannot drive."""
    path = Path(path)
    if not path.is_file():
        raise PgoError(f"no loop state at {path}; run `pgo init` first")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != LOOP_SCHEMA:
        raise PgoError(f"{path} is not a {LOOP_SCHEMA} document")
    return document


def save_loop(loop: Mapping, path) -> Path:
    """Write a loop document where the next session reads it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(loop, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def record(loop: dict, verdict: Mapping, *, iteration: int | None = None,
           pair: Sequence[str] = (), pair_ceiling: float | None = None,
           after: Mapping | None = None) -> dict:
    """Append one verdict to the state and advance the iteration.

    A kept pair becomes the new baseline, so the next iteration compares
    against the tree it now has. A reverted pair does not, and its identifiers
    join the rejection list, so the next ranking cannot choose them again.
    """
    number = int(loop["iteration"] if iteration is None else iteration)
    entry = {
        "iteration": number,
        "pair": list(pair),
        "pair_ceiling": pair_ceiling,
        "before": (after or {}).get("before_value"),
        "after": (after or {}).get("after_value"),
        "gain": float(verdict["gain"]),
        "mfu_before": verdict.get("mfu_before"),
        "mfu_after": verdict.get("mfu_after"),
        "overlap": bool(verdict.get("overlap")),
        "decision": str(verdict["decision"]),
    }
    if verdict["decision"] == "kept":
        loop["baseline"] = {"iteration": number,
                            "value": float(verdict["after"]),
                            "metric": verdict["metric"],
                            "mfu": verdict.get("mfu_after")}
        if after is not None and after.get("measurement") is not None:
            loop["baseline"]["measurement"] = after["measurement"]
    else:
        loop["rejected"].extend(str(name) for name in pair)
    loop["history"].append(entry)
    loop["iteration"] = number + 1
    return loop


def next_action(loop: Mapping, folder) -> dict:
    """What the loop needs next, from the files on disk.

    The answer is one word, so an operator or an agent never guesses which
    phase is missing. ``stop`` carries the reason.
    """
    if int(loop["iteration"]) > int(loop["max_iterations"]):
        return {"action": "stop", "reason": "the iteration cap is reached"}
    if loop.get("rejected") and _pair_below_bar(loop):
        return {"action": "stop",
                "reason": "the best remaining pair is below the threshold"}
    folder = Path(folder)
    current = iteration_dir(folder, int(loop["iteration"]))
    if not (current / "before" / "measurement.json").is_file():
        return {"action": "measure", "point": "before", "iteration":
                int(loop["iteration"])}
    if not sorted((current / "before" / "trace").glob("*.nsys-rep")):
        return {"action": "trace", "iteration": int(loop["iteration"])}
    if not (current / "before" / "analysis.json").is_file():
        return {"action": "analyze", "iteration": int(loop["iteration"])}
    if not (current / "ideas.json").is_file():
        return {"action": "rank",
                "reason": "the analyst writes ideas.json from analysis.json"}
    if not (current / "after" / "measurement.json").is_file():
        return {"action": "measure", "point": "after",
                "iteration": int(loop["iteration"])}
    if not (current / "verdict.json").is_file():
        return {"action": "compare", "iteration": int(loop["iteration"])}
    return {"action": "record", "iteration": int(loop["iteration"])}


def _pair_below_bar(loop: Mapping) -> bool:
    """True when the recorded analysis holds no pair above the threshold."""
    analysis = loop.get("analysis")
    if not analysis:
        return False
    return not rank(analysis.get("signatures") or (),
                    loop.get("rejected") or ()).get("proceed", True)


# ---------------------------------------------------------------------------
# The commands the loop runs, and the answers it reads
# ---------------------------------------------------------------------------


def last_json(text: str) -> dict:
    """The last top-level JSON object in one command's output.

    A Modal local entrypoint prints its document after the container's own
    output, and the CLI then prints its exit lines, so the document sits in the
    middle of the text. ``raw_decode`` reads one value and ignores what follows,
    which a whole-text ``loads`` cannot do.
    """
    text = str(text or "")
    decoder = json.JSONDecoder()
    for marker in ("\n{\n", "\n{"):
        start = text.rfind(marker)
        while start >= 0:
            try:
                value, _end = decoder.raw_decode(text[start + 1:])
            except ValueError:
                start = text.rfind(marker, 0, start)
                continue
            if isinstance(value, dict):
                return value
            start = text.rfind(marker, 0, start)
    raise PgoError("no JSON document in the command output")


def trace_command(*, stage: str, device: str, run_id: str, job_path,
                  steps: int, seconds: int, domains: str = "cuda,osrt",
                  sample: str = "none") -> list[str]:
    """The ``modal run`` line that collects one trace."""
    return [MODAL, "run", "tools/modal/app.py::profile_stage",
            "--stage", str(stage), "--device", str(device),
            "--run-id", str(run_id), "--job", str(job_path),
            "--steps", str(int(steps)), "--seconds", str(int(seconds)),
            "--domains", str(domains), "--sample", str(sample)]


def measure_command(*, stage: str, device: str, run_id: str, job_path,
                    steps: int, seconds: int, runs: int) -> list[str]:
    """The ``modal run`` line that measures one point, untraced."""
    return [MODAL, "run", "tools/modal/app.py::measure_stage",
            "--stage", str(stage), "--device", str(device),
            "--run-id", str(run_id), "--job", str(job_path),
            "--steps", str(int(steps)), "--seconds", str(int(seconds)),
            "--runs", str(int(runs))]


def fetch_commands(*, stage: str, run_id: str, destination) -> list[list[str]]:
    """The ``modal volume get`` lines that bring one trace home."""
    destination = Path(destination)
    return [[MODAL, "volume", "get", "--force", "nanochat-runs",
             f"{run_id}/profile/{name.format(stage=stage)}",
             str(destination / name.format(stage=stage))]
            for name in TRACE_FILES]


def _run(argv: Sequence[str], *, dry_run: bool = False) -> str:
    """Run one command and return its output, or print the line it would run."""
    printable = " ".join(argv)
    if dry_run:
        print(printable)
        return ""
    print(f"pgo: {printable}", file=sys.stderr)
    result = subprocess.run(list(argv), capture_output=True, text=True,
                            check=False)
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-12:])
        raise PgoError(f"{argv[0]} exited {result.returncode}:\n{tail}")
    return result.stdout


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def _target(options) -> tuple[dict, Path]:
    """The loop document and its folder, from one parsed command line."""
    folder = run_root(options.root, options.stage, options.device)
    return load_loop(folder / "loop.json"), folder


def _write(path, document) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def _cmd_init(options) -> int:
    folder = run_root(options.root, options.stage, options.device)
    loop_path = folder / "loop.json"
    if loop_path.is_file() and not options.force:
        raise PgoError(f"{loop_path} exists; pass --force to replace it")
    job_copy = ""
    if options.job:
        source = Path(options.job)
        if not source.is_file():
            raise PgoError(f"no job file at {source}")
        folder.mkdir(parents=True, exist_ok=True)
        _write(folder / "job.json",
               json.loads(source.read_text(encoding="utf-8")))
        job_copy = str(folder / "job.json")
    loop = new_loop(options.stage, options.device, job=job_copy,
                    max_iterations=options.iterations,
                    threshold=options.threshold,
                    trace_steps=options.steps,
                    seconds=options.seconds, runs=options.runs)
    save_loop(loop, loop_path)
    print(json.dumps({"loop": str(loop_path), "target": loop["target"],
                      "job": job_copy, "next": next_action(loop, folder)},
                     indent=1, sort_keys=True))
    return 0


def _cmd_trace(options) -> int:
    loop, folder = _target(options)
    current = iteration_dir(folder, int(loop["iteration"]))
    job_path = loop.get("job") or options.job
    if not job_path:
        raise PgoError("the loop names no job file; run `pgo init --job`")
    run_id = f"pgo-{loop['target']['stage']}-{loop['target']['device']}" \
             f"-i{int(loop['iteration']):02d}"
    argv = trace_command(stage=loop["target"]["stage"],
                         device=loop["target"]["device"], run_id=run_id,
                         job_path=job_path,
                         steps=int(loop["budget"]["trace_steps"]),
                         seconds=int(loop["budget"]["seconds"]),
                         domains=options.domains, sample=options.sample)
    _run(argv, dry_run=options.dry_run)
    if options.dry_run:
        return 0
    destination = current / "before" / "trace"
    destination.mkdir(parents=True, exist_ok=True)
    for fetch in fetch_commands(stage=loop["target"]["stage"], run_id=run_id,
                                destination=destination):
        _run(fetch)
    loop["last_trace"] = {"run_id": run_id, "dir": str(destination)}
    save_loop(loop, folder / "loop.json")
    print(json.dumps({"trace": str(destination),
                      "files": sorted(path.name for path in
                                      destination.iterdir())},
                     indent=1, sort_keys=True))
    return 0


def _cmd_measure(options) -> int:
    loop, folder = _target(options)
    point = options.point
    current = iteration_dir(folder, int(loop["iteration"]))
    if point == "after" and not (current / "before" / "measurement.json").is_file():
        # A compare advances the iteration, so an after point asked for twice
        # would land one iteration ahead of the before it is compared with.
        raise PgoError(
            f"iteration {int(loop['iteration'])} has no before point; an after "
            "point compares against one. Finish the iteration, or start a new "
            "target with `pgo init --force`")
    job_path = loop.get("job") or options.job
    if not job_path:
        raise PgoError("the loop names no job file; run `pgo init --job`")
    run_id = f"pgo-{loop['target']['stage']}-{loop['target']['device']}" \
             f"-i{int(loop['iteration']):02d}-{point}"
    argv = measure_command(stage=loop["target"]["stage"],
                           device=loop["target"]["device"], run_id=run_id,
                           job_path=job_path,
                           steps=int(options.steps),
                           seconds=int(loop["budget"]["seconds"]),
                           runs=int(options.runs))
    output = _run(argv, dry_run=options.dry_run)
    if options.dry_run:
        return 0
    document = last_json(output)
    measurement = measurement_report(
        document, stage=loop["target"]["stage"],
        device=loop["target"]["device"])
    path = _write(current / point / "measurement.json", measurement)
    if point == "before" and not loop.get("baseline"):
        loop["baseline"] = {"iteration": int(loop["iteration"]),
                            "value": measurement["value"],
                            "metric": measurement["metric"]["name"],
                            "mfu": measurement.get("mfu"),
                            "measurement": measurement}
        save_loop(loop, folder / "loop.json")
    print(json.dumps({"measurement": str(path),
                      "value": measurement["value"],
                      "metric": measurement["metric"]["name"],
                      "step_ms": measurement["step_ms"],
                      "tokens_per_second": measurement.get("tokens_per_second"),
                      "mfu": measurement.get("mfu")},
                     indent=1, sort_keys=True))
    return 0


def _cmd_analyze(options) -> int:
    loop, folder = _target(options)
    current = iteration_dir(folder, int(loop["iteration"]))
    trace_dir = Path(options.trace or (current / "before" / "trace"))
    analysis = analyze_trace(trace_dir, stage=loop["target"]["stage"],
                             device=loop["target"]["device"],
                             steps=int(loop["budget"]["trace_steps"]),
                             step_index=int(options.step_index),
                             mark_bytes=options.mark_bytes)
    analysis["rank"] = rank(analysis["signatures"], loop.get("rejected") or ())
    trace_json = chrome_trace(trace_dir, current / "before" / "chrome.json")
    path = _write(current / "before" / "analysis.json", analysis)
    loop["analysis"] = {"signatures": analysis["signatures"],
                        "rank": analysis["rank"]}
    save_loop(loop, folder / "loop.json")
    print(json.dumps({"analysis": str(path), "chrome": str(trace_json),
                      "step_wall_ms": analysis["step"]["wall_ms"],
                      "device_busy_percent":
                          analysis["step"]["device_busy_percent"],
                      "signatures": [{"id": row["id"],
                                      "ceiling": row["ceiling"]}
                                     for row in analysis["signatures"]],
                      "rank": analysis["rank"]}, indent=1, sort_keys=True))
    return 0


def _cmd_compare(options) -> int:
    loop, folder = _target(options)
    current = iteration_dir(folder, int(loop["iteration"]))
    before = json.loads((current / "before" / "measurement.json")
                        .read_text(encoding="utf-8"))
    after = json.loads((current / "after" / "measurement.json")
                       .read_text(encoding="utf-8"))
    ideas = {}
    if (current / "ideas.json").is_file():
        ideas = json.loads((current / "ideas.json").read_text(encoding="utf-8"))
    verdict = compare(before, after, float(loop["threshold"]))
    verdict["iteration"] = int(loop["iteration"])
    verdict["pair"] = [idea.get("id", "") for idea in ideas.get("ideas") or ()]
    verdict["pair_ceiling"] = ideas.get("pair_ceiling")
    path = _write(current / "verdict.json", verdict)
    loop = record(loop, verdict,
                  pair=verdict["pair"], pair_ceiling=verdict["pair_ceiling"],
                  after={"before_value": verdict["before"],
                         "after_value": verdict["after"],
                         "measurement": after})
    save_loop(loop, folder / "loop.json")
    print(json.dumps({"verdict": str(path), "decision": verdict["decision"],
                      "gain": verdict["gain"], "reason": verdict["reason"],
                      "next": next_action(loop, folder)},
                     indent=1, sort_keys=True))
    return 0


def _cmd_status(options) -> int:
    loop, folder = _target(options)
    print(json.dumps({"target": loop["target"],
                      "iteration": loop["iteration"],
                      "max_iterations": loop["max_iterations"],
                      "baseline": loop.get("baseline"),
                      "history": loop["history"],
                      "next": next_action(loop, folder)},
                     indent=1, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nanochat_cpp.pgo",
        description="The deterministic half of the profile-guided "
                    "optimization loop (docs/profile-guided-optimization.md).")
    #: ``--root`` reads before or after the subcommand. The subcommand copy
    #: suppresses its default, so it never overwrites the value the top-level
    #: option already parsed.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=argparse.SUPPRESS,
                        help="the directory that holds one folder per target")
    parser.add_argument("--root", default="profiles/pgo",
                        help="the directory that holds one folder per target")
    sub = parser.add_subparsers(dest="command", required=True)

    def target_arguments(cmd):
        cmd.add_argument("--stage", required=True, choices=STAGES)
        cmd.add_argument("--device", required=True, choices=DEVICES)
        return cmd

    def command(name: str, **kwargs):
        return target_arguments(sub.add_parser(name, parents=[common],
                                               **kwargs))

    init = command("init", help="create one target")
    init.add_argument("--job", default="", help="the job file to freeze")
    init.add_argument("--iterations", type=int,
                      default=DEFAULT_MAX_ITERATIONS)
    init.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    init.add_argument("--steps", type=int, default=DEFAULT_TRACE_STEPS)
    init.add_argument("--seconds", type=int, default=DEFAULT_SECONDS)
    init.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    init.add_argument("--force", action="store_true")
    init.set_defaults(handler=_cmd_init)

    trace_cmd = command("trace", help="trace the stage on Modal")
    trace_cmd.add_argument("--domains", default="cuda,osrt")
    trace_cmd.add_argument("--sample", default="none")
    trace_cmd.add_argument("--job", default="")
    trace_cmd.add_argument("--dry-run", action="store_true")
    trace_cmd.set_defaults(handler=_cmd_trace)

    measure = command("measure", help="measure one point, untraced")
    measure.add_argument("--point", choices=("before", "after"),
                         default="before")
    measure.add_argument("--steps", type=int, default=DEFAULT_MEASURE_STEPS)
    measure.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    measure.add_argument("--job", default="")
    measure.add_argument("--dry-run", action="store_true")
    measure.set_defaults(handler=_cmd_measure)

    analyze = command("analyze", help="read the trace and rank")
    analyze.add_argument("--trace", default="")
    analyze.add_argument("--step-index", type=int, default=6)
    analyze.add_argument("--mark-bytes", type=int, default=None,
                         help="the size of the per-row loss copy, in bytes, "
                              "when the shape is not the reference profile")
    analyze.set_defaults(handler=_cmd_analyze)

    compare_cmd = command("compare", help="decide the pair and advance")
    compare_cmd.set_defaults(handler=_cmd_compare)

    status = command("status", help="what the loop needs next")
    status.set_defaults(handler=_cmd_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand and report a fault as an exit status."""
    options = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        return int(options.handler(options))
    except PgoError as error:
        print(f"pgo: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
