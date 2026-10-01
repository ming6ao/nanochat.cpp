# Documentation

Reference and how-to material for `nanochat.cpp`. Design rationale, invariants,
and evolution rules live in [../DESIGN.md](../DESIGN.md); this directory is the
detailed reference that the design points at.

| Document | Contents |
|---|---|
| [kernels.md](kernels.md) | The backend seam: kernel inventory, API, device helpers |
| [model.md](model.md) | Model API, workspace/memory, the four graphs |
| [optimizer.md](optimizer.md) | AdamW/Muon parameter grouping and schedules |
| [precision.md](precision.md) | FP32/FP16 selection and capability reporting |
| [backends.md](backends.md) | Backend selection and how to add one |
| [build.md](build.md) | File layout, Bazel, Makefile fallback |
| [data.md](data.md) | Token shards, checkpoints, the oracle fixture |
| [testing.md](testing.md) | Test tiers, oracle harness, finite-difference checks |
| [sandbox.md](sandbox.md) | Execution sandbox and host resource governance |

New here? Start with [../README.md](../README.md) for the quickstart, then
[../DESIGN.md](../DESIGN.md) for the shape of the system.
