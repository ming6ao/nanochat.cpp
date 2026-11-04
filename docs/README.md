# Documentation

Reference and how-to material for `nanochat.cpp`. Design rationale, invariants,
and evolution rules live in [../DESIGN.md](../DESIGN.md); this directory is the
detailed reference that the design points at.

One document covers one subject. When a subject needs a campaign plan, the plan
is absorbed by the reference document when its last row lands, and the outcome
goes to [campaigns.md](campaigns.md). The rule is at the end of this page.

## Seams, build, and models

| Document | Contents |
|---|---|
| [build.md](build.md) | File layout, Bazel, Makefile fallback, backends, precision, Turing, and Part 1: the dependency zones and the host packaging |
| [kernels.md](kernels.md) | The backend seam: kernel inventory, **the tier column and the 16-of-30 bring-up cost**, API, device helpers |
| [model.md](model.md) | Model API, graphs, the model description and its block kinds, optimizer, data and checkpoints, workspace |
| [model-and-device-bringup.md](model-and-device-bringup.md) | The two axes: the directory contract, the kernel tiers, the model description, the planner, isolation, the conformance matrix, the checklists, and the open questions with their decision records |
| [conformance-matrix.md](conformance-matrix.md) | The model/device/tier/graph matrix. **Generated** by `tools/gen_matrix.py`; do not edit a row |
| [examples.md](examples.md) | The two smallest examples of the public API: what each one shows, how to run it, the six design decisions, and its line budget |
| [checkpoint-format.md](checkpoint-format.md) | The checkpoint container: the version 1 layout and the version 0 layout, the eight run-metadata decisions, the compatibility rules, and the parameter-only read |
| [api-surface.md](api-surface.md) | The API of each component at HEAD, the smaller API proposed for each one, and the review rows `R1` to `R14` that the code cites |

## Precision and numerics

| Document | Contents |
|---|---|
| [precision.md](precision.md) | The precision policy: fp32 on Pascal, Turing, and the CPU; bf16 on Ampere and newer; fp16 removed from the tree |
| [fp8-training.md](fp8-training.md) | The fp8 tier: the storage tags, the scale rule, the scaled GEMM, the layer set, and the backward pass |
| [testing.md](testing.md) | Test tiers, oracle harness, finite-difference checks, and Part 1: the correctness gates. Part 2: the numerics integration test |
| [simulator.md](simulator.md) | The S0 simulator: device profiles, the reference engine, the emulation engine, the API interposer, and the collective mock. Part 7: the simulator at bf16 and fp8 |
| [parity.md](parity.md) | Known differences from the PyTorch reference, with status. Part 1: supervised fine-tuning reference parity |

## Training and the pipeline

| Document | Contents |
|---|---|
| [post-training.md](post-training.md) | Supervised fine-tuning and reinforcement learning. Part 1: reinforcement learning in a notebook. Part 2: one chain on three hosts. Part 3: the scale-up of that chain |
| [distributed-design.md](distributed-design.md) | Data-parallel training on two devices: the gradient-sync seam, sharding, and the data cursor. Part 1: the launch. Part 2: the Python and C++ seam. Part 3: native collectives over NVLink and RDMA |
| [client-runtime-design.md](client-runtime-design.md) | The single-client multi-device runtime: one program, a device mesh, placement, the in-process collective, the C++ mesh training driver, and the compatibility process runtime |
| [node-h100.md](node-h100.md) | The H100 node run book: the memory plan, the batch plan, the launch, and the acceptance run |
| [tokenizer.md](tokenizer.md) | The native BPE tokenizer: design, artifacts, training, and status |
| [parquet-native.md](parquet-native.md) | The native parquet reader and on-the-fly tokenization |
| [eval.md](eval.md) | Base and chat evaluation, and the forward-only workspace |

## Performance

| Document | Contents |
|---|---|
| [performance.md](performance.md) | How to measure, and the debugging interface. Part 1: the profile-guided optimization loop. Part 2: the CPU backend record |
| [performance-model.md](performance-model.md) | The analytic throughput model: the operation counters, the device rates, the roofline, and the prediction record |
| [attention.md](attention.md) | Attention baseline, Pascal feasibility, the fused window, and the results. Part 5: tensor-core attention |
| [scalar-arguments.md](scalar-arguments.md) | The device-resident scalar in the seam: `ScalarArg`, the optimizer migration, the pointwise arms, and the model-side fp32 view |
| [backlog.md](backlog.md) | Work that is designed and not implemented: Part 1 the training optimizations, Part 2 the self-evolution loop |

## Hosts and tools

| Document | Contents |
|---|---|
| [python.md](python.md) | The single Python surface: in-process compute API, planning, chat evaluation, and the `tools/nanochat` toolchain. Part 1: host task consolidation |
| [sandbox.md](sandbox.md) | Execution sandbox and host resource governance. Part 1: the GPU job queue |
| [host-portability.md](host-portability.md) | Toolchain, optional sandbox, devices, and Kaggle |

## The records

Two files hold the history. Neither is a reference for how the tree behaves
today.

| File | Contents |
|---|---|
| [campaigns.md](campaigns.md) | One section per finished campaign: the dates, the outcome, the defects it found, the open items, and the evidence path. Read it before you reopen a settled question |
| [baselines.json](baselines.json) | The measured baselines: `attention`, `attention_e2e`, `cpu_train`, and `cpu_gemm`. Each entry names the command that produced it and the revision. Update an entry only with a real run |

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

## How to add a document, and how to retire one

Add a document only when a subject has no home. A new subject earns a new file;
a new campaign does not, because a campaign is a record and not a reference.

A plan is retired as soon as its last row lands:

1. The evidence goes to `.sliceme/feat--<campaign>.evidence.md`.
2. The durable sections move into the reference document, under a part heading.
   The section numbers of an absorbed document do not change, so every existing
   reference still resolves. A reference then reads "Part N, section K".
3. The outcome, the defect list, and the open items go to
   [campaigns.md](campaigns.md).
4. The plan file is deleted, and every reference to it is retargeted.

New here? Start with [../README.md](../README.md) for the quickstart, then
[../DESIGN.md](../DESIGN.md) for the shape of the system.
