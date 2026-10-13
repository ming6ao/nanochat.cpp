# Single-client multi-device runtime design

Status: partly delivered. This document describes the target and section 21
records the delivered part. The C++ mesh runtime, the in-process collective,
the C application binary interface, and the Python `mesh`/`runtime` client are
implemented and CPU-tested. The CUDA thread-local change, the loader
integration, and the compatibility process runtime are later phases. The kernel
seam and the public model API do not change.

Related: [distributed-design.md](distributed-design.md),
[distributed-launch.md](distributed-launch.md), [python.md](python.md), and
[../DESIGN.md](../DESIGN.md).

## 1. Purpose

This document gives a design for one client program that controls every local
accelerator. The client sees all devices as one system. The user starts one
program. The user needs no `torchrun`, no rank variable, and no rendezvous
port.

The design keeps two properties:

- **Single point of control.** One client program is the controller.
- **Unified view.** The client treats every local device as one machine.

The reference model is JAX on one host. The API shape follows
`jax.devices()`, `Mesh`, `NamedSharding`, and `PartitionSpec`.

## 2. Scope

In scope:

- One host.
- Two to eight local accelerators.
- One client process.
- Data-parallel training.
- Replicated inference and evaluation.
- The current fixed model graph.

Out of scope:

- More than one host.
- Tensor parallel and pipeline parallel.
- A general parallel array library.
- A sharding compiler.
- An autograd engine.
- Dynamic shapes and runtime dtype dispatch.

## 3. The current model and its cost

The project uses a single program, multiple data model today.
`tools/nanochat dist` starts one process per device. The launcher sets
`CUDA_VISIBLE_DEVICES` and the `NANOCHAT_*` variables for each process.
`src/train_main.cc` reads those variables. The gradient-sync seam reduces the
gradients.

The user pays a cost for this model:

- The rank is visible in the shape of the run.
- The user must start the group.
- The user must pick a master address and a port.
- The model exists one time per process.
- A checkpoint exists one time per rank.

The client design removes the rank from the user program. The runtime owns the
rank.

## 4. Decision

Two runtimes can give a single control point.

### 4.1 Runtime A: one process, one device thread

One process owns every device. One host thread binds to one device. The thread
calls `cudaSetDevice` one time. The CUDA runtime then uses that device for every
later call on that thread. Each thread builds one model replica on its device.
The collective runs between threads in one address space.

### 4.2 Runtime B: one process, one device process

One client process owns every device. The client starts one worker process per
device. The workers use the current runtime. The client is a supervisor. The
client sets the device variables for each worker.

### 4.3 The choice

The project selects Runtime A as the target. Runtime A gives a true unified
address space. It removes the rendezvous and the rank variables.

Runtime B stays as a compatibility backend behind the same client interface.
Use Runtime B when a driver forbids more than one CUDA context in one process.

The rest of this document describes Runtime A. Section 15 describes Runtime B.

## 5. Programming model

The client exposes four concepts. They match JAX.

### 5.1 Device list

`nanochat.devices()` returns every local accelerator. The call does not create a
CUDA context. The runtime reads the device count and the device properties.

```python
devices = nanochat.devices()
# (cuda:0, cuda:1, cuda:2, cuda:3)
```

### 5.2 Mesh

`Mesh(devices, axis_names)` binds a named axis to the device list.

```python
mesh = nanochat.Mesh(devices, ("data",))
```

The runtime accepts a rank-1 mesh only. The one axis must have the name
`"data"`. The `MeshSpec` type carries a shape vector and a name vector, so a
rank-N mesh is a later additive change. The runtime rejects a rank-N mesh and
an unknown axis name at construction time. The error names the supported axis.

The project puts tensor parallel and pipeline parallel out of scope (section
2). A second axis has no executor, because the kernel seam has no device
argument. Section 20.1 records this decision.

### 5.3 Placement

`PartitionSpec` names the sharded axes of one tensor. A value of `None` means
replicated. A name means sharded along that mesh axis.

```python
from nanochat import PartitionSpec as P

params = nanochat.NamedSharding(mesh, P())        # replicated
batch = nanochat.NamedSharding(mesh, P("data"))   # sharded on 'data'
```

### 5.4 Supported placements

The project has a fixed graph. It has no sharding compiler. So the runtime
supports a small placement set.

| Object | Placement | Meaning |
|---|---|---|
| Model parameters | `P()` | One full replica per device |
| Optimizer state | `P()` | One full state per device |
| Global batch | `P("data")` | Contiguous slices, one per device |
| Activations | derived | Follow the batch placement |
| Gradients | derived | Follow the parameter placement |

The runtime rejects any other placement at construction time. The error names
the supported set. Parameter sharding is tensor parallel. Section 2 puts that
work out of scope.

### 5.5 The program

A training program mentions no rank.

```python
import nanochat

devices = nanochat.devices()
mesh = nanochat.Mesh(devices, ("data",))

config = nanochat.Config(num_layers=12, hidden_dim=768)
with nanochat.MeshRuntime(mesh, config, batch=32, grad_accum=4) as run:
    for step in range(steps):
        loss = run.train_step(tokens, targets)  # one global batch
        print(step, loss)
```

The `MeshRuntime` splits the global batch across the mesh, reduces the
gradients in process, and returns one loss. The client never names a rank.

`trainer.step()` returns one loss for the whole mesh. The loss is the global
mean over the batch. `trainer.metrics()` returns the per-device breakdown and
the timing fields. Section 20.4 records this decision.

## 6. What the model is not

The client is not a general parallel array library.

- There is no `jnp`.
- There is no `jit` over user code. The graph is the code.
- There is no `vmap` or `pmap` over user functions.
- There is no `psum` or `pmean` in user code.

The runtime reduces the gradients itself. The parallel entry points are the
fixed operations: `forward_loss`, `step`, `generate`, and `evaluate`.

## 7. Architecture

The runtime adds one host layer. The layer sits above the per-replica driver. It
never touches the kernel seam.

```text
App / CLI        train.py   (one process, one program)
Runtime (new)    DeviceMesh  DeviceContext  ReplicaGroup  Collective
Train driver     TrainLoop   Checkpointer  Logger      (one per replica)
Optim            AdamW  Muon  ANVIL  Scheduler
Model / graphs   Model  Prefill  Decode  KvCache
Ops
KERNEL API       <-- unchanged seam
Backend          backends/cpu/   backends/cuda/
```

New files:

| File | Purpose | Owner |
|---|---|---|
| `include/nanochat/mesh.h` | `DeviceInfo`, `EnumerateDevices`, `MeshSpec` | Architect |
| `src/mesh.cc` | The mesh validation shared by the runtime and the client | Architect |
| `src/mesh_collective.h`, `src/mesh_collective.cc` | The in-process all-reduce | Collective |
| `src/mesh_runtime.h`, `src/mesh_runtime.cc` | Device threads, replicas, step dispatch | Harness |
| `capi.h` additions | Device enumeration, the mesh handle, the runtime handle | Architect |
| `python/nanochat_cpp/mesh.py` | `devices`, `Mesh`, `NamedSharding`, `PartitionSpec` | Python surface |
| `python/nanochat_cpp/runtime.py` | `MeshRuntime`, the step loop | Python surface |

The frozen header `kernels.h` does not change. Section 8 explains why.

## 8. Thread-affine device selection

The seam has no device argument. Every kernel call carries a pointer and a
shape. The runtime does not change the seam. The runtime uses one rule instead:
one device thread owns one device.

- The thread calls `cudaSetDevice(index)` one time.
- The CUDA runtime uses that device for every later call on the thread.
- `kernels::Alloc`, `kernels::Gemm`, and the other calls use the calling
  thread's device.
- Each thread builds its own `Model`. The buffers land on the right device.
- Each thread destroys its own model and its own workspace.

This rule keeps the seam frozen.

### 8.1 Backend state that is thread-local

The CUDA backend holds process-global state in the first draft. Each item is
thread-local now:

| Item | File |
|---|---|
| cuBLAS handle | `backends/cuda/gemm.cu` |
| Tensor-core flag | `backends/cuda/gemm.cu` |
| Device name buffer | `backends/cuda/device.cu` |
| ANVIL scratch | `backends/cuda/kernels/anvil.cu` |
| Muon scratch | `backends/cuda/kernels/muon.cu` |
| Attention scratch | `backends/cuda/kernels/attention.cu` |

The device globals in `backends/cuda/kernels/global_norm.cu` stay device
globals. Each device has its own copy of the module state. One thread drives one
device, so the order is safe.

### 8.2 The seam stays frozen

A thread-local change is invisible above the seam. The `src/**` layer sees no
change. The `kernels.h` contract is unchanged. `DESIGN.md` section 7 permits
this class of change.

## 9. The in-process collective

The in-process collective replaces NCCL and the TCP host reference on one host.

### 9.1 Interface

The collective satisfies `GradientSync` in `src/distributed.h`. The seam does
not change.

### 9.2 Structure

One `CollectiveGroup` holds every device thread.

- A barrier synchronizes the group.
- One pinned host buffer holds one slot per device.
- Each device copies its gradient bucket to its slot.
- After the barrier, each device sums every slot in rank order.
- Each device copies the sum back to its bucket.

The sum order is the rank order. Every replica sees the same bytes. So every
optimizer step is identical.

### 9.3 Memory

One bucket holds a fixed element count. The staging buffer holds `world_size`
slots. A large gradient needs several buckets. The current harness already
buckets the gradients. The runtime reuses that code.

### 9.4 The fast path

When a device pair has peer access, the runtime can copy between devices with
`cudaMemcpyPeerAsync`. The ring schedule in
`backends/cuda/collective/ring_schedule.cc` already exists. A later phase can
reuse it. The host path is correct first.

## 10. One training step

1. The controller reads one global micro-batch from the loader.
2. The controller splits the batch into `world_size` contiguous slices.
3. The controller puts one slice on each device thread queue.
4. Each device thread runs `forward_loss` and `backward_accumulate` with the
   scale `1 / (accum * world_size)`.
5. Each device thread packs its gradient buckets.
6. The collective sums every bucket. Section 9.2 gives the order.
7. Each device thread runs one optimizer step with the same reduced gradient.
8. The controller collects the per-device losses and returns the global mean.

The global batch stays constant on any device count. Only the work per device
changes. This matches `docs/distributed-design.md` section 3.

## 11. Data and the loader

The loader yields a global batch. The runtime owns the split. The user does not
see a rank.

In phase 2 the controller reads one global batch and slices it. This path
matches the reference rule in `docs/distributed-design.md` section 7: one global
token batch splits into contiguous slices. The CPU mesh test needs no overlap,
so the simple path is enough. The host batch is small. A 524288-token batch
needs about 4 MiB for the tokens and the targets.

The controller path needs a small interface change. The per-replica `TrainLoop`
gains an entry point that accepts an injected batch. The current `Run()` reads
its own data, so it cannot serve the mesh. Section 17 phase 2 covers this
change.

In phase 5 the read moves behind the runtime. Each device thread reads its own
shard through a pinned double buffer. That change helps the overlap of the data
read with the step. It is not necessary for correctness. Section 20.2 records
this decision.

## 12. Checkpoints

One checkpoint is canonical. The runtime writes the checkpoint from replica 0.
The runtime loads a checkpoint into every replica. The replicas stay
byte-identical. The current design already gives this property. The runtime
keeps the sidecar data cursor for an exact resume.

## 13. Inference and evaluation

Replicated inference: every device holds a full copy of the model. The runtime
puts one prompt on each device and fans out the rows. `GenerateBatch` and
`GenerateMultiPrompt` already batch rows.

Evaluation: the runtime splits the validation set across the devices and
reduces the byte-per-byte totals.

These cases need no gradient reduction. They use the barrier and a small scalar
reduction only.

## 14. The entry point and the sandbox

The client is the entry point. The runtime keeps the sandbox rule.

- The client process runs inside one sandbox scope. It takes one broker lock
  for the device set.
- `tools/nanochat run train -- python train.py` is the strict form.
- The console script `python -m nanochat_cpp.run train.py` re-enters
  `tools/nanochat client`. The user sees one command.

The `nanochat_init()` sandbox check stays. A run outside the entry point still
fails fast.

The broker key is the full device set, for example `0,1,2,3`. One lock covers
the group. A second run on the same set waits.

## 15. The compatibility runtime

Runtime B starts one worker process per device behind the same client
interface.

- `Mesh` and `PartitionSpec` behave the same.
- The client sets `CUDA_VISIBLE_DEVICES` and the rank variables per worker.
- The client uses `tools/nanochat dist` today.
- The reduction uses NCCL or the native ring.

The user program does not change. The runtime backend is a construction option.

```python
runtime = nanochat.Runtime(mesh, backend="inproc")     # default
runtime = nanochat.Runtime(mesh, backend="process")    # compatibility
```

Runtime B stays a supported backend. It is not a second API. Three cases need
it:

- Some drivers forbid more than one CUDA context in one process.
- Kaggle runs two T4 cards, and a notebook can prefer process isolation.
- One bad device stops one worker, not the whole client.

Runtime B shares the mesh code and the placement code. So the extra cost is
small. Section 20.3 records this decision.

## 16. Determinism and parity

- The reduction sums the buckets in rank order.
- The gradient is the mean over the global batch.
- Every replica applies the same update. So the replicas stay equal.
- The validation loss of a multi-device run equals the validation loss of a
  one-device run on the fixed set.

Section 18 pins these properties.

## 17. Phasing

### Phase 0 — Design and ownership

1. Register this document in `docs/README.md`.
2. Confirm the ownership of the new files in section 7.
3. Confirm the supported placement set in section 5.4.

Gate: the architect approves the API names and the placement set.

### Phase 1 — Device enumeration and the mesh

1. Add `EnumerateDevices` and `MeshSpec` to `include/nanochat/mesh.h`.
2. Add the device count and the mesh entries to `capi.h`.
3. Add `Mesh`, `NamedSharding`, and `PartitionSpec` to
   `python/nanochat_cpp/mesh.py`.

Gate: a CPU test lists the devices and builds a mesh. A single-device GPU test
reports the real device.

### Phase 2 — The in-process runtime and the CPU mesh

1. Add the device threads and the replica group in `src/runtime.cc`.
2. Add the in-process collective.
3. Add an injected-batch entry point to the per-replica `TrainLoop`.
4. Add `Trainer.step()` over the mesh.
5. Run the two-device case on the CPU backend.

Gate: a two-device CPU run equals a one-device CPU run on the loss and the
parameters.

### Phase 3 — Thread-local CUDA state

1. Make the items in section 8.1 thread-local.
2. Add a two-device CUDA test on one card with two contexts. Mark it `gpu` and
   `manual`.

Gate: the numerical trace passes on the device mesh.

### Phase 4 — The client entry point and the compatibility runtime

1. Add the console script.
2. Add the broker coverage for the full device set.
3. Add Runtime B behind the same interface.

Gate: one command starts a two-device run. One broker lock covers the group.

### Phase 5 — Fast collective and data overlap

1. Add the peer-to-peer fast path.
2. Move the loader behind the runtime.

Gate: a benchmark records the reduction cost and the step time.

## 18. Tests

| Tier | Test | Gate |
|---|---|---|
| T0 CPU | `//src:mesh_test` | Device enumeration and mesh construction |
| T0 CPU | `//src:runtime_test` | Two CPU devices equal one CPU device |
| T0 CPU | `//backends/host/collective:inproc_sync_test` | The in-process sum matches the host reference |
| T0 CPU | `//python:mesh_test` | The mesh, the placement, and the rejection of a bad placement |
| Simulator | `//tests:collective_sim_test` | The call sequence and the deadlock check |
| T1 GPU | `//src:runtime_gpu_test` | Two device threads on one card |
| T2 GPU | A two-device run versus one device | The validation loss matches one device |

The CPU tests carry the mesh logic. The development host has one card, so the
CPU path is the main gate. This matches `DESIGN.md` section 4 and `AGENTS.md`
section 5.1.

## 19. Risks

| Risk | Control |
|---|---|
| The CUDA runtime forbids two contexts in one process | The compatibility runtime B |
| Global backend state corrupts a second device | Phase 3 makes the state thread-local. A two-device test pins it |
| The host-staged reduction is slow | Section 9.4 adds the peer path. Measure it first |
| One process uses too much host memory | The sandbox profile sizes the run |
| A bad device stops the whole process | Runtime A accepts this cost for the unified view. Runtime B isolates it |
| The placement set grows into a compiler | Section 5.4 fixes the set. A new placement needs a design review |
| The client API drifts from the frozen headers | The seam stays frozen. The runtime is additive |

## 20. Resolved decisions

This section answers the four questions that the first draft left open. Each
answer gives the decision and the reason.

### 20.1 The mesh has one axis

Decision: the runtime accepts a rank-1 mesh. The one axis is named `"data"`. A
rank-N mesh is rejected at construction time.

Reasons:

- Tensor parallel and pipeline parallel stay out of scope (section 2).
- The kernel seam has no device argument, so a second axis has no executor.
- The current collective reduces one gradient group along one axis.

`MeshSpec` already carries vectors for the shape and the names. So a later
second axis is an additive change. The first axis keeps the name `"data"`.

### 20.2 The controller reads the global batch in phase 2

Decision: in phase 2 the controller reads one global batch and slices it. Phase
5 moves the read behind the runtime.

Reasons:

- The contiguous global slice matches the reference rule in
  `docs/distributed-design.md` section 7.
- The CPU mesh test needs no overlap, and phase 2 must stay small.
- The host batch is small. A 524288-token batch needs about 4 MiB.

Phase 5 adds a pinned double buffer per device and reads each shard on its
device thread. That change is an optimization. It does not change the result.

### 20.3 The compatibility runtime stays

Decision: Runtime B stays a supported backend. It is a construction option, not
a second API. The project does not remove it at phase 5.

Reasons:

- Some drivers forbid more than one CUDA context in one process.
- Kaggle runs two T4 cards, and a notebook can prefer process isolation.
- One bad device stops one worker, not the whole client.

Runtime B shares the mesh code and the placement code. So the cost is small. A
later removal needs a design review.

### 20.4 One loss, with a per-device breakdown

Decision: `MeshRuntime.train_step()` returns one global mean loss. A later
`metrics()` accessor returns the per-device split and the timing fields.

Reasons:

- The unified view wants one result for one step.
- The JAX reference returns one value from a jitted function.
- Debugging still needs the per-device split. The harness already collects it.

The runtime computes the global loss with a scalar reduction over the valid
tokens. The delivered client returns one loss, and `MeshRuntime.param_read`
exposes one replica at a time. A later `metrics()` follows the `LogRecord`
fields in `src/train.cc` and adds one loss entry per device.

## 21. Delivery status

This section records what the tree implements.

### Delivered

- `include/nanochat/mesh.h`: the device list, `MeshSpec` with the device
  indices, `SelectDevice`, and the rank-1 data rule. The CPU and CUDA backends
  implement `EnumerateDevices` and `SelectDevice`.
- `src/mesh.cc`: the mesh validation shared by the runtime and the client.
- `src/mesh_collective.h` and `src/mesh_collective.cc`: the in-process
  all-reduce over a reusable barrier and one host staging buffer. It stages
  every bucket through the kernel-seam copy, so a device gradient never reaches
  the host as a raw pointer. It satisfies the `GradientSync` seam, so
  `src/distributed.h` does not change.
- `src/mesh_runtime.h` and `src/mesh_runtime.cc`: one device thread per
  replica, the device binding, the contiguous global batch split, the
  in-process reduction, the equal optimizer step, checkpoint save and load, the
  `MeshMetrics` report, and the `ParamDigest` replica check.
- `include/nanochat/capi.h`: device enumeration, the mesh handle, the runtime
  handle, the runtime save and load, the named optimizer preset
  (`nanochat_optim_config_preset`), the metrics (`nanochat_runtime_metrics`),
  and the replica digest (`nanochat_runtime_param_digest`). The C application
  binary interface version moves to `0.8.0`.
- `python/nanochat_cpp/mesh.py`: `devices`, `Device`, `Mesh`, `NamedSharding`,
  `PartitionSpec`, and the placement checks.
- `python/nanochat_cpp/runtime.py`: `MeshRuntime`, which runs one global batch
  over the mesh, returns one loss, reports `StepMetrics`, digests a replica,
  and saves or loads a checkpoint.
- `python/nanochat_cpp/api.py`: `optimizer_config` and `preset_optimizer_config`,
  so `Optimizer` and `MeshRuntime` share one field list and the `track3` preset
  has one C source of truth.
- `python/nanochat_cpp/run.py`: the client entry point. It re-enters
  `tools/nanochat client`.
- `tools/nanochat client`: one broker lock for the full device set, then one
  sandbox scope.
- `notebooks/mesh_train.py`: the notebook loop over the runtime, the
  `MeshTrainPlan` job round trip, and the child `launch` through
  `tools/nanochat client`. `notebooks/mesh_worker.py` is the child entry point,
  and `notebooks/nanochat-cpp-on-2x-t4.ipynb` is the two-card end-to-end
  notebook.
- Phase 3: the CUDA backend state is thread-local, and each device thread binds
  with `SelectDevice`. The cuBLAS handle, the tensor-core flag, the device name
  buffer, and the attention, Muon, and ANVIL scratch buffers are per thread.
- Tests: `//src:mesh_test`, `//src:mesh_runtime_test`, `//src:runtime_gpu_test`,
  `//python:mesh_test`, `//python:runtime_test`, `//python:run_test`,
  `//notebooks:mesh_train_test`, and the device enumeration in
  `//bindings:nanochat_capi_test`.

### The evidence

The two-device and the four-device CPU runs equal the one-device run on the
loss and on every parameter. The replicas of one runtime stay byte-identical
over several steps. A save from one replica loads into every replica. The
Python client reproduces the same result through the C application binary
interface. `//src:runtime_gpu_test` runs two device threads on one card, and the
replicas stay byte-identical. The CUDA compile gate passes with warnings as
errors.

### Not delivered

- Phase 5: the peer-to-peer fast path. The kernel-seam staging path is correct
  first (section 9.4).
- The controller drives the loader, so a step does not overlap the data read.
  The overlap needs the loader behind the runtime (section 11).
- A two-card CUDA run on real T4 cards is not measured here. It needs the
  `t1-gpu` and `t2-parity` gates on a multi-card host. `//src:runtime_gpu_test`
  covers two device threads on one card.
