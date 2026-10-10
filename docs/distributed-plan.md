# Distributed training: implementation plan

This plan covers the remaining distributed work: the NCCL backend, the overlap
of the gradient reduction with the backward pass, and the checkpoint document
index. It also removes `docs/distributed-t4-plan.md` and folds its live content
into `docs/distributed-design.md`.

`docs/distributed-design.md` is the design authority. This file is the
execution plan for the NCCL backend. [distributed-native-plan.md](distributed-native-plan.md)
supersedes it on the CUDA path: the native collective replaces NCCL behind the
same seam, and `collective=nccl` stays the default until the native path
reaches parity.

## 0. Status

The project implements all phases. The implementation differs from the first
draft in two places:

- Phase 2 overlaps the bucket reductions on a side stream, not the backward
  itself. A per-layer gradient-ready hook would overlap the backward, but it
  needs a change to the frozen `include/nanochat/model.h`. Section 16 of the
design document lists it as an open question.
- Phase 3 stores the encoded, not-yet-packed buffer beside the source position.
  The reference best-fit packing reorders documents inside the buffer, so a
document index alone cannot reproduce the exact stream.

The CPU gates pass: `//src:distributed_test`, `//src:train_parallel_test`, and
`//src:data_test`. The NCCL source compiles, and the NCCL gate and the two-rank
parity gate run on Kaggle.

## 1. Scope

In scope:

- A native NCCL gradient-sync backend for the CUDA build.
- Overlap of the gradient reduction with the backward pass.
- A checkpoint document index for an exact resume.
- The removal of `docs/distributed-t4-plan.md`.

Out of scope:

- Distributed evaluation.
- Tensor parallel and pipeline parallel.
- More than one node.

## 2. Constraints

- `src` must not include a vendor header. NCCL code lives in `backends/cuda`.
- The freeze covers `include/nanochat/model.h`. Sync overlap needs a new
  backward callback, so the architect owns that change.
- The CUDA backend uses NCCL only. There is no host-reference fallback on the
  CUDA path. The host reference stays for the CPU tests.
- The WSL2 host has no NCCL and one card. It cannot build a CUDA config after
  this change, and it cannot run a two-rank NCCL test. The CUDA build and the
  GPU gates run on Kaggle.
- The `tools/cuda_sim` collective mock (`tools/cuda_sim/collective/nccl_mock.h`)
  tests the NCCL call sequence on the CPU, with no vendor library.

## 3. Phase 0 — Select the factory at link time

Goal: link the host reference on the CPU backend and NCCL on the CUDA backend.
The build supports no other combination.

The factory `CreateGradientSync` is the link-time seam. Each backend library
provides the symbol. Bazel selects the library. The `src` layer never sees a
vendor header.

1. Move `src/distributed.cc` out of `//src:model` into a new target
   `//src:distributed_host`. That target defines the host factory
   `CreateGradientSync` and the CPU no-op object.
2. `backends/cuda/nccl_sync.cu` defines the NCCL factory `CreateGradientSync`.
3. `//src:model` depends on a two-way select:

```python
deps = select({
    "//:backend_cuda": ["//backends/cuda:nccl_sync"],
    "//:backend_cpu": ["//src:distributed_host"],
}) + [
    ":device_profile",
    "//src/tokenizer:tokenizer",
    "//src/parquet:parquet",
    "//include/nanochat:nanochat",
]
```

4. Do not add a `//:backend_cuda` branch for `//src:distributed_host`. That
   branch is the removed host-reference fallback.
5. Do not add a separate `//:nccl` config. The CUDA backend always links NCCL,
   so the config is redundant. Add the NCCL repository to `MODULE.bazel`, and
   make it a hard dependency of every CUDA config.
6. Remove the device staging from the host reference. Delete
   `DistributedConfig::device_buffers` from `src/distributed.h`, and delete the
   `kernels::Memcpy` staging in `HostGradientSync`. The CUDA backend no longer
   calls the host reference, so the staging is dead code. The CPU backend
   reports `is_device` false and never used it.
7. Keep `NoopGradientSync` in each library, or share one small private header.
   Do not put the no-op in `src/distributed.h`, because that would make a
   backend depend on `src`.

Gate: `//src:distributed_test` and `//src:train_parallel_test` pass under
`--config=cpu`. A CUDA config builds with NCCL and fails to build without it.
Expect this build failure without NCCL, and keep the behavior.

## 4. Phase 1 — The NCCL backend

New file `backends/cuda/nccl_sync.cu`. This is the only file that includes
`nccl.h`.

1. Define `class NcclGradientSync final : public GradientSync`.
2. Rendezvous. Rank 0 calls `ncclGetUniqueId`. Every rank exchanges the
   128-byte identifier over the existing rendezvous sockets.
3. Extract the rendezvous into a small POSIX-only helper, for example
   `ExchangeBootstrap(config, void* id, int bytes)`, in `src/distributed_host`.
   The helper exposes an opaque byte exchange, so `src` stays vendor-free.
4. Communicator. Each rank calls
   `ncclCommInitRank(&comm, world_size, id, rank)`.
5. Device selection. The launch sets `CUDA_VISIBLE_DEVICES=rank`, so each
   process sees one card as device 0. Call `cudaSetDevice(0)`. Document the
   assumption.
6. Reduction. `AllReduceSum` calls
   `ncclAllReduce(buffer, buffer, count, dtype, ncclSum, comm, stream)`. Use the
   backend default stream first. Call `kernels::Synchronize()` after the call.
   The asynchronous form comes in Phase 2.
7. Data type. `ComputeType` is `float` in the fp32 build. Add a static check, or
   select `ncclHalf` in the fp16 build.
8. Teardown. Call `ncclCommDestroy` in the destructor.
9. Build. Add `cuda_library(name = "nccl_sync", ...)` in
   `backends/cuda/BUILD.bazel`, with `alwayslink = True`, a dependency on
   `:device`, and a dependency on the NCCL target.

Tests:

- CPU and simulator. Extend the collective suite. `//tests:collective_sim_test`
  runs four ranks against the mock. Add a case that drives the new factory and
  checks the sum, the rank, and the deadlock behavior.
- Kaggle GPU. Add `//backends/cuda:nccl_sync_gpu_test` with
  `tags = ["gpu", "manual"]`. Two processes, two cards. Check that the NCCL sum
  matches the host reference.
- Kaggle end to end. A two-rank run against a one-rank run. Check the
  validation loss.

Gate: `ncclAllReduce` matches the host reference. The two-rank validation loss
matches the one-rank run.

## 5. Phase 2 — Overlap the reduction with the backward

This phase depends on Phase 1 and on an architect change to
`include/nanochat/model.h`.

Implemented: the seam is asynchronous and `TrainLoop` pipelines the bucket
reductions on a side stream. The model callback is not implemented (section 0).

1. Extend the seam in `src/distributed.h`:
   - `virtual void AllReduceSumAsync(ComputeType* buffer, std::int64_t count)`.
   - `virtual void Wait()`.
   - The host reference implements both synchronously. The asynchronous call
     runs the existing path, and `Wait` returns.
2. Add a ready callback to the model. Add
   `Model::BackwardAccumulate(float scale, const GradientReadyCallback& on_ready)`
   or an equivalent hook. The model calls the hook once per parameter when the
   gradient is final. This touches `include/nanochat/model.h` and
   `src/model.cc`.
3. Build the bucket plan once. `TrainLoop` already buckets gradients. Keep the
   plan across steps, because `model_->params()` is deterministic.
4. Use two streams. Run the reduction on a dedicated side stream. The side
   stream waits on an event recorded after the last gradient copy of each
   bucket. The main stream continues the backward.
5. The optimizer waits on one event that marks every bucket complete.
6. The order of the sum does not change the result, so overlap is numerically
   identical. Add a CPU test that compares the bucket scheduler against the
   synchronous path with a mock sync.

Gate: the sync fraction in the phase timer is below 0.05. Two-rank parameter
equality holds.

## 6. Phase 3 — The checkpoint document index

Goal: a resumed run reads the data stream at the exact document, not at the
epoch start.

Implemented: the cursor is `(epoch, source_position, encoded buffer)`, stored in
a sidecar beside the checkpoint (section 0).

Position definition: the number of documents this rank consumed from the
sharded stream and packed into completed batches. The stride rule is
deterministic, so rank `r`'s k-th accepted document is global index
`k * world_size + r`.

1. `include/nanochat/dataloader.h` and `src/data.cc`:
   - Count documents removed from the encoded buffer in `NextDocuments`,
     including cropped documents.
   - Add `std::int64_t documents_consumed() const` to `DataLoader`.
   - Add an `epoch` counter that increments on `Reset`.
   - Add a start position to the loader construction.
2. `ShardedDocumentSource`:
   - Accept a start global index. Set `index_` to it.
   - Add `DocumentSource::Seek(std::int64_t index)`. The parquet source seeks by
     row or row group. The default implementation drops documents.
   - On resume, the loader discards the prefetched but unpacked documents and
     re-reads them. Nothing repeats and nothing is lost.
3. Checkpoint format:
   - Store `(epoch, documents_consumed)` beside the optimizer step. Mirror
     `SaveOptimizerStep` in `src/optim.cc`. Use one record with eight raw
     bytes, because the container has no integer dtype.
   - Add overloads `Checkpointer::SaveModel(model, optimizer, step,
     data_position, path)` and the matching load. Keep the old overloads.
4. `src/train.cc`:
   - On save, pass `train_loader_->documents_consumed()` and the epoch.
   - On resume, read the stored position and pass it to the loader.
5. Each rank stores its own count. The counts differ by rank, because the
   stride differs. Each rank resumes its own stream.

Tests:

- CPU: save at step N, resume, and check that the first batch after resume
  equals the batch of an uninterrupted run.
- CPU: two ranks save equal parameters and resume to the same stream.

Gate: the CPU resume test passes. A resumed two-rank run reads no repeated
document.

## 7. Phase 4 — Remove `docs/distributed-t4-plan.md`

1. Move the live content into `docs/distributed-design.md`:
   - The best-fit configuration and the measured run go in a new "Measured run"
     section.
   - The execution phases become a "Status and roadmap" section, updated to
     this plan.
   - The reproduction commands move to the design document or the notebook.
2. Update every reference. The current list is:
   - `docs/README.md` (the table row)
   - `docs/distributed-design.md` (four references)
   - `include/nanochat/logger.h` (the phase B1 comment)
   - `src/BUILD.bazel` (the phase B1 comment)
   - `src/train.cc` (two comments)
   - `src/train.h` (the phase C1 comment)
   - `notebooks/nanochat-cpp-on-2x-t4.ipynb` (three references)
3. Repoint the phase labels to the design-document section, or delete them.
   After the file is gone, "phase B1" has no target.
4. Delete `docs/distributed-t4-plan.md`.
5. Verify with `rg "distributed-t4-plan" .`. The command must return nothing.
   Do not touch the copy under `.sliceme/worktrees/`; that is a separate
   worktree, and `tools/nanochat prune` handles it.
6. Register this plan in `docs/README.md`, or state the status in the design
   document and delete this file.

This phase is architect-owned, because `docs/**`, `include/nanochat/logger.h`,
and the root build files are architect-owned.

## 8. Order of work

- Phase 0 and Phase 3 are independent. Run them together.
- Phase 1 depends on Phase 0.
- Phase 2 depends on Phase 1 and on the frozen-header change.
- Phase 4 comes last, so the design document absorbs the new status.

## 9. Definition of Done

1. NCCL builds under a CUDA config. The sum matches the host reference. The
   two-rank validation loss matches one rank.
2. The overlap path is numerically identical to the synchronous path. The sync
   fraction is below 0.05.
3. A resume from a mid-epoch checkpoint produces the exact next batch.
4. No reference to `docs/distributed-t4-plan.md` remains. The design document
   carries the status and the roadmap.
5. `tools/nanochat lint` passes. The CPU tests pass under `t0-cpu`. The
   simulator collective suite passes.

## 10. Risks and controls

| Risk | Control |
|---|---|
| NCCL is absent on WSL2 | Build and test the CPU config there. Gate on Kaggle |
| NCCL is absent at build time on Kaggle | Point the build at the NCCL in the torch wheel, or install it in the notebook |
| One card cannot run two real ranks | Do not run the real NCCL test on WSL2. Gate on Kaggle |
| The backward callback changes a frozen header | The architect owns the change |
| Prefetch breaks resume exactness | Count documents on the consumer side, not the producer side |
| The CUDA build no longer builds without NCCL | This is intended. Document the new requirement in `docs/build.md` |
| Deleting the plan loses history | Git history keeps it. Fold the live status into the design document |

## 11. Ownership summary

| Phase | Owner | Files |
|---|---|---|
| 0 | Architect plus build | `BUILD.bazel`, `.bazelrc`, `MODULE.bazel`, `src/BUILD.bazel`, `src/distributed.*` |
| 1 | Kernel agent | `backends/cuda/nccl_sync.cu`, `backends/cuda/BUILD.bazel` |
| 2 | Architect plus workflow | `include/nanochat/model.h`, `src/model.cc`, `src/train.cc`, `src/distributed.h` |
| 3 | Harness plus data pipeline | `src/train.cc`, `src/train.h`, `include/nanochat/dataloader.h`, `src/data.cc`, `src/optim.cc`, `src/optim_state.h` |
| 4 | Architect | `docs/**`, `include/nanochat/logger.h`, `src/BUILD.bazel`, `src/train.cc`, `src/train.h`, the notebook |
