"""Attribute a nsys trace directory to a per-step budget.

Steps are cut at the optimizer phase, which every step has and which no other
phase has: the AdamW and Muon kernels are grouped into clusters, one cluster per
step, and a step window runs from one cluster midpoint to the next. That needs no
gradient-norm burst, so the same script measures a run whose step stages its
gradients and one whose step does not.

    python analyze.py <trace_dir> [steps]
"""

import bisect
import collections
import csv
import statistics
import sys
from pathlib import Path


def f(x):
    try:
        return float(str(x).replace(",", "") or 0)
    except ValueError:
        return 0.0


def load(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row]


def union(intervals):
    total, cs, ce = 0.0, None, None
    for s, e in sorted(intervals):
        if cs is None:
            cs, ce = s, e
        elif s <= ce:
            ce = max(ce, e)
        else:
            total += ce - cs
            cs, ce = s, e
    if cs is not None:
        total += ce - cs
    return total


def cluster(times, gap=20e6):
    groups = [[times[0]]]
    for value in times[1:]:
        if value - groups[-1][-1] > gap:
            groups.append([value])
        else:
            groups[-1].append(value)
    return groups


def main(directory, steps=20):
    directory = Path(directory)
    gpu = load(directory / "timeline_cuda_gpu_trace.csv")
    api = load(directory / "timeline_cuda_api_trace.csv")

    ops = []
    for row in gpu:
        start = f(row["Start (ns)"])
        name = row["Name"]
        kind = ("D2H" if "Device-to-Host" in name else
                "H2D" if "Host-to-Device" in name else
                "D2D" if "Device-to-Device" in name else
                "MSET" if "memset" in name else "KERNEL")
        ops.append((start, start + f(row["Duration (ns)"]),
                    f(row["Duration (ns)"]), name, kind,
                    f(row.get("Bytes (MB)", "0")), str(row["CorrId"])))
    ops.sort()

    calls = []
    for row in api:
        start = f(row["Start (ns)"])
        calls.append((start, start + f(row["Duration (ns)"]),
                      f(row["Duration (ns)"]), row["Name"], str(row["CorrID"])))
    calls.sort()
    call_starts = [c[0] for c in calls]

    # One optimizer cluster per step. The midpoint marks the step boundary.
    optimizer = [o for o in ops
                 if o[4] == "KERNEL"
                 and ("AdamW" in o[3] or "Muon" in o[3])]
    groups = cluster([o[0] for o in optimizer])
    mids = [statistics.mean([g[0], g[-1]]) for g in groups]
    if len(mids) < steps:
        raise SystemExit(f"only {len(mids)} optimizer clusters, need {steps}")
    mids = mids[:steps]
    first = operator_start = groups[0][0]
    origin = first
    # The first window starts at the first real kernel of the run.
    starts = [next(o[0] for o in ops if o[4] == "KERNEL")]
    for index in range(1, len(mids)):
        starts.append((mids[index - 1] + mids[index]) / 2)
    ends = starts[1:] + [mids[-1] + (mids[-1] - mids[-2])]

    copied = {o[6]: (o[4], o[5]) for o in ops if o[4] in ("D2H", "H2D", "D2D")}
    op_starts = [o[0] for o in ops]

    def window(begin, finish):
        lo = bisect.bisect_left(op_starts, begin)
        inside_ops = []
        index = lo
        while index < len(ops) and ops[index][0] < finish:
            inside_ops.append(ops[index])
            index += 1
        lo = bisect.bisect_left(call_starts, begin)
        inside_calls = []
        index = lo
        while index < len(calls) and calls[index][0] < finish:
            inside_calls.append(calls[index])
            index += 1

        merged = []
        for s, e, d, name, corr in inside_calls:
            s, e = max(s, begin), min(e, finish)
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e, name, corr])
        host_only = collections.Counter()
        if merged:
            host_only["lead_after_copy"] += merged[0][0] - begin
            for index, (s, e, name, corr) in enumerate(merged):
                nxt = (merged[index + 1][0] if index + 1 < len(merged)
                       else finish)
                gap = nxt - e
                if gap <= 0:
                    continue
                if name == "cudaMemcpy":
                    direction, megabytes = copied.get(corr, ("?", 0.0))
                    key = ("after_D2H_ge20MB"
                           if direction == "D2H" and megabytes >= 20
                           else "after_D2H_span" if direction == "D2H"
                           else "after_copy_" + direction)
                elif "LaunchKernel" in name:
                    key = "after_launch"
                elif "KernelGetFunction" in name:
                    key = "after_lookup"
                else:
                    key = "after_other_api"
                host_only[key] += gap
        else:
            host_only["lead_after_copy"] += finish - begin

        d2h = [o for o in inside_ops if o[4] == "D2H"]
        return {
            "wall": finish - begin,
            "busy": union([(o[0], o[1]) for o in inside_ops]),
            "kernels": sum(o[2] for o in inside_ops if o[4] == "KERNEL"),
            "launches": len([o for o in inside_ops if o[4] == "KERNEL"]),
            "d2h_time": sum(o[2] for o in d2h),
            "d2h_count": len(d2h),
            "d2h_bytes": sum(o[5] for o in d2h),
            "h2d_time": sum(o[2] for o in inside_ops if o[4] == "H2D"),
            "d2d_time": sum(o[2] for o in inside_ops if o[4] == "D2D"),
            "api": sum(e - s for s, e, _, _ in merged),
            "host_only": sum(host_only.values()),
            "breakdown": host_only,
        }

    measured = [window(starts[i], ends[i]) for i in range(steps)]
    # The first step carries the corpus read and the warm-up; the steady step is
    # the rest.
    steady = measured[1:]

    def mean(key):
        return statistics.mean(w[key] for w in steady) / 1e6

    def median(key):
        return statistics.median(w[key] for w in steady) / 1e6

    print(f"trace        {directory}")
    print(f"steps        {len(measured)} cut at the optimizer phase; "
          f"window 1 {measured[0]['wall']/1e6:.1f} ms, "
          f"steady {median('wall'):.1f} ms (median), "
          f"{mean('wall'):.1f} ms (mean)")
    print()
    print(f"{'per steady step':<26}{'ms':>9}{'% step':>9}")
    wall = median("wall")
    rows = [
        ("step wall", median("wall")),
        ("device busy (union)", median("busy")),
        ("  kernels", median("kernels")),
        ("  D2H copies", median("d2h_time")),
        ("  H2D + D2D", median("h2d_time") + median("d2d_time")),
        ("host inside driver calls", median("api")),
        ("host in its own code", median("host_only")),
    ]
    for label, value in rows:
        print(f"{label:<26}{value:>9.1f}{100*value/wall:>8.1f}%")
    print()
    print("host in its own code, by what precedes it:")
    keys = set()
    for w in steady:
        keys |= set(w["breakdown"])
    for key in sorted(keys, key=lambda k: -mean_breakdown(steady, k)):
        print(f"  {key:<22}{mean_breakdown(steady, key):>9.1f} ms")
    print()
    print(f"launches/step            {statistics.median(w['launches'] for w in steady):>9.0f}")
    print(f"D2H copies/step          {statistics.median(w['d2h_count'] for w in steady):>9.0f}")
    print(f"D2H bytes/step           {statistics.median(w['d2h_bytes'] for w in steady):>9.0f} MB")


def mean_breakdown(steady, key):
    return statistics.mean(w["breakdown"][key] for w in steady) / 1e6


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20)
