---
name: model-fit
description: Answer which nanochat model architectures are the best fit for particular hardware and topology. Use when you size a model for a GPU, device count, memory size, or interconnect.
---

# Model fit for hardware and topology

`scripts/model_fit.py` answers one question. For a given accelerator, device
count, and interconnect, which nanochat architectures are the largest, the
longest-context, and the fastest that fit?

The engine has three inputs:

- a hardware profile: memory, and the fp32, fp16, and bf16 rates;
- a topology: the device count and the interconnect;
- a model spec: the nanochat shape and the sequence length.

## When to use it

Use this skill to size a run for hardware:

- "What is the biggest model that fits two T4 cards?"
- "What sequence length can an 80 GB A100 hold at 12 layers?"
- "Does the interconnect change the best architecture?"
- "Which precision should this device use?"

Do not use it for a single measured benchmark. Use the measurement tools for
that.

## Run it

```bash
python3 .agents/skills/model-fit/scripts/model_fit.py \
    --hardware t4 --devices 2 --interconnect pcie3 --precision fp32
```

The command prints the three corners and the balanced pick. Each row has the
parameter count, the per-device memory, the estimated global tokens per second,
the scaling efficiency, and the sync fraction.

Useful options:

- `--hardware NAME` selects a device: `h200`, `h100`, `a100`, `t4`,
  `gtx1080ti`. Use `--hardware custom --memory-gib N --fp32-tflops T` for a
  device that is not in the table.
- `--devices N` sets the data-parallel device count.
- `--interconnect NAME` selects the link: `nvlink1` to `nvlink4`, `pcie3` to
  `pcie5`, `shared-memory`, `ethernet-100g`, `ethernet-400g`,
  `infiniband-ndr`. Add `--bandwidth-gbps N` to override it.
- `--overlap F` sets the fraction of the reduction that hides behind the
  backward pass. Use `0.9` for a well-overlapped run.
- `--precision fp32|fp16|bf16` selects the compute precision.
- `--min-seq N` and `--min-params N` set the minimums for the balanced pick.
- `--presets quick|default|wide` sets the search size.
- `--json` prints the full machine-readable answer.
- `--list-devices` and `--list-interconnects` print the tables.

## Interpret the answer

- **biggest** is the largest parameter count that fits.
- **longest** is the longest sequence that fits.
- **fastest** is the highest tokens per second.
- **balanced** is the fastest point at 250M parameters or more and 4096 tokens
  or more.

The three goals conflict. Tell the user which corner you report, and why.

`scaling_efficiency` compares the group rate with `device_count` times the
one-device rate. The gap is the gradient reduction. A value near 1.0 means the
interconnect is not the limit. A low value means the run needs gradient
bucketing, overlap, or a fast link.

## Use it as a library

```python
import model_fit

hw = model_fit.hardware("t4")
topo = model_fit.KAGGLE_T4X2
result = model_fit.answer(hw, topo, "fp32")
best = model_fit.recommend(hw, topo, "fp32")
flags = model_fit.shape_flags(model_fit.BEST_FIT_T4)
```

`shape_flags` returns the `train_main` model flags for a spec. Pass the
`track3` preset first, so the value embeddings stay off. The launch plan owns
`--batch` and `--grad-accum`.

## Grounding

The memory model is exact. It mirrors the nanochat.cpp source:

- the training arena: `BuildTrainWorkspace` in `src/model.cc`;
- the parameter counts: `CountParams` in `src/train.cc`;
- the optimizer state: `NanochatOptimizer::Build` in `src/optim.cc`;
- the arithmetic: `EstimateFlopsPerToken` in `src/train.cc`.

The device table mirrors `src/device_profile.cc`. See
`references/devices.md`.

The throughput model is an estimate. It uses the rate tables and a fixed
efficiency. Replace it with a measurement when a device is available. The
measurement method is in `docs/performance.md`.

## Caveats

- The model is data parallel. Every device holds the full model. The topology
  changes the sync cost, not the per-device memory.
- The precision is fp32 unless the caller selects another rate. A device with
  no tensor cores cannot use fp16 at a useful rate.
- The attention kernel uses no tensor cores, so its rate is separate.
- The reduced-precision gradient and master bytes follow the nanochat.cpp
  layout, not a generic mixed-precision recipe.
- The estimate omits the data loader, the optimizer kernel time, and the
  launch overhead. Treat a value as a comparison, not a promise.

## Test it

```bash
cd .agents/skills/model-fit/scripts
python3 -m unittest discover -s . -p 'model_fit_test.py'
```

The test pins the arena against the committed 36 GB figure and the T4 corners.

## Files

| File | Purpose |
|---|---|
| `scripts/model_fit.py` | The engine and the command line |
| `scripts/model_fit_test.py` | The tests |
| `references/devices.md` | The device table and the sources |
| `references/topology.md` | The interconnect table and the sync model |
