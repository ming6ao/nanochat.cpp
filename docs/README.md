# Documentation

Reference and how-to material for `nanochat.cpp`. Design rationale, invariants,
and evolution rules live in [../DESIGN.md](../DESIGN.md); this directory is the
detailed reference that the design points at.

| Document | Contents |
|---|---|
| [kernels.md](kernels.md) | The backend seam: kernel inventory, API, device helpers |
| [model.md](model.md) | Model API, workspace/memory, the graphs, batched generation |
| [optimizer.md](optimizer.md) | AdamW/Muon parameter grouping and schedules |
| [precision.md](precision.md) | FP32/FP16 selection and capability reporting |
| [backends.md](backends.md) | Backend selection and how to add one |
| [build.md](build.md) | File layout, Bazel, Makefile fallback |
| [data.md](data.md) | Token shards, checkpoints, the oracle fixture |
| [python-bridge.md](python-bridge.md) | nanochat-compatible Python entry points |
| [eval.md](eval.md) | Design: base and chat evaluation (forward-only) |
| [post-training.md](post-training.md) | Design: SFT and reinforcement learning |
| [plan-eval-rl.md](plan-eval-rl.md) | Implementation plan for eval.md and post-training.md |
| [parity.md](parity.md) | Known differences from the PyTorch reference, with status |
| [testing.md](testing.md) | Test tiers, oracle harness, finite-difference checks |
| [performance.md](performance.md) | Performance measurement protocol, debugging interface, benchmarks |
| [flash-attention-pascal.md](flash-attention-pascal.md) | Flash attention on Pascal: feasibility, baseline, and staged plan |
| [sandbox.md](sandbox.md) | Execution sandbox and host resource governance |

New here? Start with [../README.md](../README.md) for the quickstart, then
[../DESIGN.md](../DESIGN.md) for the shape of the system.
