# Documentation

Reference and how-to material for `nanochat.cpp`. Design rationale, invariants,
and evolution rules live in [../DESIGN.md](../DESIGN.md); this directory is the
detailed reference that the design points at.

| Document | Contents |
|---|---|
| [build.md](build.md) | File layout, Bazel, Makefile fallback, backends, precision, Turing |
| [precision.md](precision.md) | The precision policy: fp32 on Pascal, Turing, and the CPU; bf16 on Ampere and newer; fp16 removed from the tree |
| [fp8-training.md](fp8-training.md) | The fp8 tier: the storage tags, the scale rule, the scaled GEMM, the layer set, and the backward pass |
| [kernels.md](kernels.md) | The backend seam: kernel inventory, **the tier column and the 16-of-30 bring-up cost**, API, device helpers |
| [model.md](model.md) | Model API, graphs, the model description and its block kinds, optimizer, data and checkpoints, workspace |
| [model-and-device-bringup.md](model-and-device-bringup.md) | The two axes: the directory contract, the kernel tiers, the model description, the planner, isolation, the conformance matrix, the checklists, and the open questions with their decision records |
| [conformance-matrix.md](conformance-matrix.md) | The model/device/tier/graph matrix. **Generated** by `tools/gen_matrix.py`; do not edit a row |
| [examples.md](examples.md) | The two smallest examples of the public API: what each one shows, how to run it, and its line budget |
| [checkpoint-format.md](checkpoint-format.md) | The checkpoint container: the version 1 layout and the version 0 layout, the compatibility rules, the parameter-only read, and the three pipeline notebooks |
| [checkpoint-metadata-and-examples-plan.md](checkpoint-metadata-and-examples-plan.md) | The plan for versioned run metadata in the container and the two smallest examples of the public API |
| [testing.md](testing.md) | Test tiers, oracle harness, finite-difference checks |
| [parity.md](parity.md) | Known differences from the PyTorch reference, with status |
| [distributed-design.md](distributed-design.md) | Data-parallel training on two devices: the gradient-sync seam, sharding, the data cursor, and the launch |
| [distributed-launch.md](distributed-launch.md) | The launch of a data-parallel run: the rank contract, the `tools/nanochat dist` command, and the Python helper |
| [distributed-native-plan.md](distributed-native-plan.md) | The execution plan for a native all-reduce over NVLink and RDMA: the transport interface, the ring schedule, ownership, and the broker |
| [client-runtime-design.md](client-runtime-design.md) | The single-client multi-device runtime: one program, a device mesh, placement, the in-process collective, the C++ mesh training driver, and the compatibility process runtime |
| [performance.md](performance.md) | Performance measurement protocol and the debugging interface |
| [performance-model.md](performance-model.md) | The analytic throughput model: the operation counters, the device rates, the roofline, and the prediction record |
| [cpu-performance.md](cpu-performance.md) | CPU backend baseline, cause, design, and phase results |
| [attention.md](attention.md) | Attention baseline, Pascal feasibility, the fused window, results |
| [attention-tensor-core.md](attention-tensor-core.md) | Tensor-core attention: the fused tile in a bf16 or fp8 build, and the fp8 coupling |
| [optimizations.md](optimizations.md) | Planned recompute, fused-classifier, and row-kernel work |
| [eval.md](eval.md) | Base and chat evaluation, and the forward-only workspace |
| [tokenizer.md](tokenizer.md) | The native BPE tokenizer: design, artifacts, training, and status |
| [parquet-native.md](parquet-native.md) | The native parquet reader and on-the-fly tokenization plan |
| [post-training.md](post-training.md) | Design and status: supervised fine-tuning and reinforcement learning |
| [sft-parity-plan.md](sft-parity-plan.md) | Earning reference parity for supervised fine-tuning: the four rungs, the fixtures, and the phases |
| [training-seam.md](training-seam.md) | The Python and C++ boundary for training: the seam rule, the current defects, the mesh driver, and the target interface |
| [rl-notebook.md](rl-notebook.md) | Reinforcement learning in a notebook: the hand-rolled loop, the `nc.rl` worker bridge, and the open items |
| [pipeline-plan.md](pipeline-plan.md) | The plan for one chain on three hosts: base, then SFT, then RL, with the six blockers and the host matrix |
| [self-evolution.md](self-evolution.md) | The self-evolution loop: the goal intake, the stuck detector, the human escalation, the intervention dataset, the training round, and the Hub transport |
| [python.md](python.md) | The single Python surface: in-process compute API, planning, chat evaluation, and the `tools/nanochat` toolchain |
| [sandbox.md](sandbox.md) | Execution sandbox and host resource governance |
| [host-portability.md](host-portability.md) | Toolchain, optional sandbox, devices, and Kaggle |
| [script-consolidation-plan.md](script-consolidation-plan.md) | The plan to give each host task one home: the command table, the fixture generators, and the Python dispatch |
| [dependency-layering-plan.md](dependency-layering-plan.md) | The plan to keep the core free of third-party packages: the four zones, the packaging, the binary wheels, and the fixture provenance |
| [gpu-queue-plan.md](gpu-queue-plan.md) | The GPU job queue: one worker owns the GPU, a SQLite store holds the jobs, and `tools/nanochat gpuq` submits them |
| [node-h100.md](node-h100.md) | The H100 node run book: the memory plan, the batch plan, the launch, and the acceptance run |
| [simulator.md](simulator.md) | The S0 simulator: device profiles, the reference engine, the emulation engine, the API interposer, and the collective mock |
| [simulator-precision.md](simulator-precision.md) | The simulator at bf16 and fp8: the profile rows, the fp8 rounding, the cuBLASLt coverage, the emulator intrinsics, and the trace leg |
| [correctness-gates.md](correctness-gates.md) | The correctness split: what the simulator proves, the Hopper startup gate, and the speed budget |
| [numerics-integration.md](numerics-integration.md) | The staged numeric trace: CPU, simulator, and Pascal GPU agreement |

## The contract vocabulary, `spec/`

This directory is the architect-owned reference: it says what the tree does.
[`../spec/`](../spec/) beside it holds the *vocabulary* — the names the two axes
declare against, and the authority a declaration is reviewed by. It carries
prose and data and no code. Where a name matters, the document here links into
`spec/` instead of copying the table. A second copy of a closed list is the
drift the design warns about (§17 of
[model-and-device-bringup.md](model-and-device-bringup.md)).

| File | Contents |
|---|---|
| [../spec/versions.md](../spec/versions.md) | The four contract versions and the rules that move them: additions are free, a removal needs a deprecation release, a meaning change needs a new name |
| [../spec/capability_bits.md](../spec/capability_bits.md) | The fifteen capability bits, their meanings, their obligations, and the one spelling rule |
| [../spec/block_kinds.md](../spec/block_kinds.md) | The block kinds, what each computes, the saved slots its backward reads, and the workspace-sharing rule |
| [../spec/pointwise_ops.md](../spec/pointwise_ops.md) | The nine pointwise op codes and the declared accuracy bound of each |
| [../spec/decisions.md](../spec/decisions.md) | The nine decision records D-001 to D-009 that the bring-up document cites |

Where each is read from: the tier table is [kernels.md](kernels.md), not here
and not in `spec/`; the block kinds and their slots are
[model.md](model.md) § The model description and
[../spec/block_kinds.md](../spec/block_kinds.md); the matrix is
[conformance-matrix.md](conformance-matrix.md).

New here? Start with [../README.md](../README.md) for the quickstart, then
[../DESIGN.md](../DESIGN.md) for the shape of the system.
