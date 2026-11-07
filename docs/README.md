# Documentation

Reference and design material for `nanochat.cpp`. Each document states what a
subject is, how it works, and how to use it. A measured claim cites
[baselines.json](baselines.json), not prose.

New here? Start with [../README.md](../README.md) for the quickstart, then
[build.md](build.md) § Layers for the shape of the system.

## Seams, build, and models

| Document | Contents |
|---|---|
| [build.md](build.md) | The layers and the dependency invariants, the file layout, Bazel, backends, precision selection, and the Turing target |
| [kernels.md](kernels.md) | The backend seam: the kernel inventory, the tier column and the bring-up cost, the fusion boundary rule, the compute strategy, the API, `ScalarArg`, and the two kernel designs the tree does not implement |
| [model.md](model.md) | The five graphs, the no-autograd policy, the model description and its block kinds, the workspace and recompute, the public API, and the Muon step with its cuBLAS seam |
| [model-and-device-bringup.md](model-and-device-bringup.md) | The two axes: the directory contract, the kernel tiers, the model description, the planner, isolation, the conformance matrix, the checklists, and §18 the evolution rules |
| [conformance-matrix.md](conformance-matrix.md) | The model/device/tier/graph matrix. **Generated** by `tools/gen_matrix.py`; do not edit a row |
| [examples.md](examples.md) | The two smallest examples of the public API: what each one shows, how to run it, the six design decisions, and its line budget |
| [checkpoint-format.md](checkpoint-format.md) | The checkpoint container: the version 1 and version 0 layouts, the run-metadata decisions, the compatibility rules, and the notebook lifecycle |

## Precision and numerics

| Document | Contents |
|---|---|
| [precision.md](precision.md) | The precision policy: fp32, bf16, and the fp8 tier. Part 1 is fp8: the storage tags, the scale rule, the scaled GEMM, the layer set, and the backward pass |
| [testing.md](testing.md) | The test tiers, the oracle harness, the finite-difference checks, and the parity gates. Part 1: the correctness gates. Part 2: the numerics integration test |
| [simulator.md](simulator.md) | The S0 simulator: device profiles, the reference engine, the emulation engine, the API interposer, and the collective mock. Part 7: bf16 and fp8 |
| [parity.md](parity.md) | Known differences from the PyTorch reference, with status. Part 1: supervised fine-tuning reference parity |

## Data, training, and the pipeline

| Document | Contents |
|---|---|
| [data.md](data.md) | The data path: the native parquet reader, the tokenizer, the artifact format, row packing, and the command-line flags. Part 1: the reader design |
| [post-training.md](post-training.md) | Supervised fine-tuning and reinforcement learning. Part 1: RL in a notebook. Part 2: one chain on three hosts. Part 3: the scale-up of that chain |
| [distributed.md](distributed.md) | Data-parallel training and the device mesh: the gradient seam, both implementations, sharding, resume, and launch. Part 1: the Python/C++ seam. Part 2: native collectives. Part 3: the single-client runtime |
| [eval.md](eval.md) | Base and chat evaluation, and the forward-only workspace |
| [python.md](python.md) | The single Python surface: the in-process compute API, planning, chat evaluation, checkpoints, and the self-evolution loop |

## Performance

| Document | Contents |
|---|---|
| [performance.md](performance.md) | How to measure, and the debugging interface. Part 1: the profile-guided optimization loop. Part 2: the CPU backend record |
| [performance-model.md](performance-model.md) | The analytic throughput model: the operation counters, the device rates, the roofline, and the prediction record |
| [attention.md](attention.md) | The three attention paths, the window saving, the tensor-core decision, and the bf16 tile with its H100 measurement |

## Hosts and tools

| Document | Contents |
|---|---|
| [host-portability.md](host-portability.md) | Toolchain, optional sandbox, devices, Kaggle, and the pipeline chain on three hosts. Part 1: the H100 node run book |
| [sandbox.md](sandbox.md) | Execution sandbox and host resource governance. Part 1: the GPU job queue |

## The records

One file holds the measured history. It is not a reference for how the tree
behaves today.

| File | Contents |
|---|---|
| [baselines.json](baselines.json) | The measured baselines: `attention_bf16`, `cpu_train`, and `cpu_gemm`. Each entry names the command that produced it and the revision. Update an entry only with a real run, and keep `validated_on` naming the device that measured it. A card the tree no longer supports contributes no entry |

## The contract vocabulary, `spec/`

This directory is the architect-owned reference: it says what the tree does.
[`../spec/`](../spec/) beside it holds the *vocabulary* — the names the two axes
declare against, and the authority a declaration is reviewed by. Where a name
matters, the document here links into `spec/` instead of copying the table. A
second copy of a closed list is the drift the design warns about (§17 of
[model-and-device-bringup.md](model-and-device-bringup.md)).

| File | Contents |
|---|---|
| [../spec/versions.md](../spec/versions.md) | The four contract versions and the rules that move them: additions are free, a removal needs a deprecation release, a meaning change needs a new name |
| [../spec/capability_bits.md](../spec/capability_bits.md) | The fifteen capability bits, their meanings, their obligations, and the one spelling rule |
| [../spec/block_kinds.md](../spec/block_kinds.md) | The block kinds, what each computes, the saved slots its backward reads, and the workspace-sharing rule |
| [../spec/pointwise_ops.md](../spec/pointwise_ops.md) | The pointwise op codes and the declared accuracy bound of each |
| [../spec/decisions.md](../spec/decisions.md) | The nine decision records D-001 to D-009 that the bring-up document cites |
| [../spec/api-surface.md](../spec/api-surface.md) | The API of each component at HEAD, the smaller API proposed for each one, and the review rows `R1` to `R20` that the code and the generators cite |

Where each is read from: the tier table is [kernels.md](kernels.md), not here
and not in `spec/`; the block kinds and their slots are
[model.md](model.md) § The model description and
[../spec/block_kinds.md](../spec/block_kinds.md); the matrix is
[conformance-matrix.md](conformance-matrix.md).

## How to add a document, and to retire one

Add a document only when a subject has no home. One document covers one subject.
An unimplemented design lives as a clearly marked section of the document that
owns its subject, not in a separate plan file.

Retire a document when its subject moves:

1. The durable sections move into the owning document under a part heading. The
   section numbers do not change, so every existing reference still resolves as
   "Part N, section K". The retired file is deleted, and every reference to it
   is retargeted.
2. A measurement leaves prose for [baselines.json](baselines.json). A phase
   table, an owner list, and an open-question list do not move: they are not
   reference material.
