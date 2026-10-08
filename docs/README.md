# Documentation

Reference and how-to material for `nanochat.cpp`. Design rationale, invariants,
and evolution rules live in [../DESIGN.md](../DESIGN.md); this directory is the
detailed reference that the design points at.

| Document | Contents |
|---|---|
| [build.md](build.md) | File layout, Bazel, Makefile fallback, backends, precision, Turing |
| [kernels.md](kernels.md) | The backend seam: kernel inventory, API, device helpers |
| [model.md](model.md) | Model API, graphs, optimizer, data and checkpoints, workspace |
| [testing.md](testing.md) | Test tiers, oracle harness, finite-difference checks |
| [parity.md](parity.md) | Known differences from the PyTorch reference, with status |
| [distributed-design.md](distributed-design.md) | Data-parallel training on two devices: the gradient-sync seam, sharding, and the launch |
| [performance.md](performance.md) | Performance measurement protocol and the debugging interface |
| [cpu-performance.md](cpu-performance.md) | CPU backend baseline, cause, design, and phase results |
| [attention.md](attention.md) | Attention baseline, Pascal feasibility, the fused window, results |
| [optimizations.md](optimizations.md) | Planned recompute, fused-classifier, and row-kernel work |
| [eval.md](eval.md) | Base and chat evaluation, and the forward-only workspace |
| [tokenizer.md](tokenizer.md) | The native BPE tokenizer: design, artifacts, training, and status |
| [parquet-native.md](parquet-native.md) | The native parquet reader and on-the-fly tokenization plan |
| [post-training.md](post-training.md) | Design and status: supervised fine-tuning and reinforcement learning |
| [python.md](python.md) | The single Python surface: in-process compute API, planning, chat evaluation, and the `tools/nanochat` toolchain |
| [sandbox.md](sandbox.md) | Execution sandbox and host resource governance |
| [host-portability.md](host-portability.md) | Toolchain, optional sandbox, devices, and Kaggle |
| [simulator.md](simulator.md) | The S0 simulator: device profiles, the reference engine, the emulation engine, the API interposer, and the collective mock |

New here? Start with [../README.md](../README.md) for the quickstart, then
[../DESIGN.md](../DESIGN.md) for the shape of the system.
