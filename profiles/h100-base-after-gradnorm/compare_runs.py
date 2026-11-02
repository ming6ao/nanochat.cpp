"""Compare two runs' per-step metrics: loss, gradient norm, and step time.

The two files are the engine's own ``metrics.jsonl``, one JSON object per step.
The comparison is the numeric gate for a change that touches the optimizer: the
loss and the pre-clip gradient norm must agree, and the step time is what the
change is for.

    python compare.py <before.jsonl> <after.jsonl>
"""

import json
import sys


def read(path):
    steps = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                steps[int(record["step"])] = record
    return steps


def main(before_path, after_path):
    before = read(before_path)
    after = read(after_path)
    common = sorted(set(before) & set(after))
    print(f"{'step':>5}{'loss before':>14}{'loss after':>14}{'d(loss)':>12}"
          f"{'norm before':>13}{'norm after':>12}{'d(norm)':>12}")
    worst_loss = 0.0
    worst_norm = 0.0
    for step in common:
        b, a = before[step], after[step]
        dl = a["loss"] - b["loss"]
        dn = a["grad_norm"] - b["grad_norm"]
        worst_loss = max(worst_loss, abs(dl))
        worst_norm = max(worst_norm, abs(dn))
        print(f"{step:>5}{b['loss']:>14.7f}{a['loss']:>14.7f}{dl:>12.2e}"
              f"{b['grad_norm']:>13.7f}{a['grad_norm']:>12.7f}{dn:>12.2e}")
    print()
    print(f"steps compared      {len(common)}")
    print(f"worst |d(loss)|     {worst_loss:.3e}")
    print(f"worst |d(norm)|     {worst_norm:.3e}")
    for label, key in (("step_seconds", "step_seconds"),):
        b = sum(before[s][key] for s in common if key in before[s])
        a = sum(after[s][key] for s in common if key in after[s])
        if b and a:
            print(f"{label:<19} before {b/len(common)*1e3:8.1f} ms   "
                  f"after {a/len(common)*1e3:8.1f} ms   "
                  f"ratio {b/a:5.2f}x")
    if common and "mfu" in after[common[-1]]:
        b = sum(before[s]["mfu"] for s in common) / len(common)
        a = sum(after[s]["mfu"] for s in common) / len(common)
        print(f"{'mfu':<19} before {100*b:8.3f} %    "
              f"after {100*a:8.3f} %    ratio {a/b if b else 0:5.2f}x")
    if common and "tokens_per_sec" in after[common[-1]]:
        b = sum(before[s]["tokens_per_sec"] for s in common) / len(common)
        a = sum(after[s]["tokens_per_sec"] for s in common) / len(common)
        print(f"{'tokens/s':<19} before {b:8.0f}      after {a:8.0f}"
              f"      ratio {a/b if b else 0:5.2f}x")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
