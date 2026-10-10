# Native collective plan: NVLink and RDMA

This document is the execution plan for a native all-reduce backend. The backend
replaces NCCL on the CUDA path. [distributed-design.md](distributed-design.md)
holds the current seam design. [distributed-plan.md](distributed-plan.md) holds
the NCCL plan that this document supersedes on the CUDA path.

Status: plan. The project implements no phase yet.

## 1. Goal and scope

**Goal.** Remove NCCL as a hard dependency of the CUDA build. Provide a native
all-reduce behind the existing `GradientSync` seam. The training loop calls
`AllReduceSumAsync` and later `Wait` on that asynchronous seam. Use NVLink
inside one node. Use remote direct memory access (RDMA) as the second transport
to reach 8 ranks on one H100 node.

**In scope.**

- A backend-private collective under `backends/cuda/collective/`.
- An asynchronous, completion-based transport interface.
- Two transports: NVLink peer copy and RDMA.
- A ring schedule with full-duplex send and receive.
- A single-precision reduction only.
- Support for world sizes 1, 2, 4, and 8.

**Out of scope.**

- Tensor parallel and pipeline parallel.
- Distributed evaluation.
- Elastic rendezvous.
- A replacement for the reserved collectives: `Broadcast`, `AllGather`,
  `ReduceScatter`, `Send`, and `Recv`.

**Assumptions.**

- The 8 H100 cards sit in one node with NVLink 4.0 and NVSwitch. This node is the
  fast path.
- RDMA is the second transport. It serves a peer that NVLink cannot reach, or a
  future second node. One schedule runs on both transports.
- The node has at least one RDMA adapter with GPUDirect RDMA. Section 11 records
  how to confirm this.
- The reduction operand is single precision. The collective does not use fp16.

## 2. Why the transport is completion-based

A CUDA stream does not describe an RDMA transfer. RDMA posts a work request and
polls a completion queue. So the transport must not take a `cudaStream_t` in its
main interface. The interface exposes a completion token instead. The NVLink
transport implements the token with stream events. The RDMA transport implements
the token with a work completion.

This is the most important choice in the plan. A stream-shaped interface would
force the RDMA path into a poor design.

## 3. The private interface

New directory `backends/cuda/collective/`. The interface header is backend
private. It is not part of the frozen seam, so it may include CUDA headers.

The transport is pairwise. One `Transport` connects this rank to one peer. A
group with mixed links holds one instance for each ring neighbor. The next
subsection defines the selection. `Rendezvous` is the control plane. A
`TimePoint` is `std::chrono::steady_clock::time_point`.

```cpp
namespace nanochat {
namespace cuda_backend {

// A completion handle. The transport that issued the token defines the
// encoding. A token names one posted operation: one direction, one slot, and
// one stream or queue pair. The value 0 is invalid. The token is single-use
// and bound to the issuing transport.
struct Token {
  std::uint64_t value = 0;
};

enum class WaitResult { kCompleted, kTimeout, kError };

enum class TransportKind { kNone, kNvlink, kRdma, kHost };

// Moves chunks between this rank and one peer. One implementation per kind.
class Transport {
 public:
  virtual ~Transport() = default;
  virtual int self_rank() const = 0;
  virtual int peer_rank() const = 0;
  // Binds the control plane and opens the data plane to the one peer. Returns
  // false on failure.
  virtual bool Connect(const DistributedConfig& config,
                       Rendezvous* rendezvous) = 0;
  // Publishes the local transfer window to the peer and imports the peer's
  // window. `window` is owned by the adapter. Call once, and again after a
  // growth (section 5.4).
  virtual bool RegisterWindow(void* window, std::size_t bytes) = 0;
  // The number of receive slots for this direction. A ring needs two.
  virtual int SlotCount() const = 0;
  // Posts a send of `bytes` from `local` into the peer's receive slot `slot`.
  // The call returns at once.
  virtual Token Send(int slot, const void* local, std::size_t bytes) = 0;
  // Posts a receive of `bytes` into `local`, the local receive slot `slot`.
  virtual Token Recv(int slot, void* local, std::size_t bytes) = 0;
  // Returns true when the transfer completed. Never blocks.
  virtual bool Done(Token token) = 0;
  // Waits for the transfer until `deadline`. It returns kCompleted, kTimeout,
  // or kError. It never hangs past the deadline.
  virtual WaitResult Wait(Token token, TimePoint deadline) = 0;
  // Frees the receive slot. The transfer must be complete. The call
  // invalidates the token.
  virtual void Release(Token token) = 0;
};

}  // namespace cuda_backend
}  // namespace nanochat
```

The schedule uses only this interface. A memory transport backs the central
processing unit test. The memory transport can reuse the POSIX shared-memory
rendezvous from `tools/cuda_sim/collective/`.

### 3.1 Token lifetime and slot ownership

A slot is a receive-side resource. `Recv(slot, ...)` posts into the local
receive slot. `Send(slot, ...)` targets the peer's receive slot with the same
index. One index pairs a send with its matching receive.

Each direction owns its own slots. The NVLink transport keeps two receive slots
for each stream. The RDMA transport keeps two receive slots for each queue pair.
The send side has no slot. The caller owns the send buffer until completion.

A token has three states.

| State | Allowed operations |
|---|---|
| Issued | `Done`, `Wait` |
| Completed | `Done`, `Wait` again, `Release` |
| Released | none; the token is invalid |

Rules:

- `Done` and `Wait` do not consume the token. The schedule may call either
  more than once.
- `Release` requires completion. It returns the receive slot to the free list.
- A released token is invalid. A later use is a programming error. The debug
  build asserts; the release build returns a failure.
- A token is bound to the transport that issued it. Do not pass a token to
  another transport.
- The schedule must release every issued token exactly once. A leaked receive
  slot stalls the ring.

### 3.2 Pairwise transports and the selection rule

The ring needs two peers. The next rank is the send direction. The previous
rank is the receive direction. The group holds one transport for each neighbor.

```cpp
// Holds the two ring neighbors. It builds the neighbor table at Connect.
class RingTransports {
 public:
  bool Connect(const DistributedConfig& config, Rendezvous* rendezvous);
  Transport* Next();  // the send direction
  Transport* Prev();  // the receive direction
};
```

`RingTransports::Connect` selects the kind for each neighbor, in this order.

1. `cudaDeviceCanAccessPeer` is true for the pair: NVLink.
2. An RDMA adapter reaches the peer: RDMA.
3. Otherwise: host staging.

The selection is per neighbor. So one group can use NVLink on one edge and RDMA
on the other. A mixed group is the normal case at a node boundary. Section 11.2
records the rule.

## 4. The control plane

The socket rendezvous becomes a small, vendor-free helper in
`src/rendezvous.{h,cc}`. It exposes one class.

```cpp
// The control plane. One instance per process. Every call is a collective: all
// ranks call it with the same length and in the same order.
class Rendezvous {
 public:
  static std::unique_ptr<Rendezvous> Create(const DistributedConfig& config);
  // Exchanges `bytes` of opaque data across every rank. Rank 0 gathers and
  // broadcasts. Returns false on failure.
  bool ExchangeBytes(void* data, std::size_t bytes);
};
```

The helper carries both payloads.

- The NVLink payload: the CUDA IPC memory handles and the device identifiers.
- The RDMA payload: the queue pair number, the global identifier or local
  identifier, the remote key, and the buffer addresses.

The helper includes only POSIX sockets. The `src` layer stays vendor-free. The
backend links the helper. The host reference in `src/distributed.cc` uses the
same helper.

## 5. The schedule

Use a ring reduce-scatter, then an all-gather. It is bandwidth-optimal. Each
rank moves about `2 * (N - 1) / N` of the buffer. For 8 ranks that is 1.75 times
the buffer.

The schedule is pure host logic. It sends on the `Next` transport and receives
on the `Prev` transport (section 3.2). It calls `Send`, `Recv`, `Done`, `Wait`,
`Release`, and one device add. A memory transport exercises it exactly.

### 5.1 Double buffering

Split the bucket into `N` chunks. Each direction keeps two receive slots, so
`SlotCount()` is 2. Slot reuse gives the pipeline. A send has no slot. Its
buffer stays alive until the token completes.

For each ring step `s`:

1. Receive the incoming chunk into local slot `s % 2`. The receive waits until
   the add of step `s - 2` released that slot.
2. Add the received chunk into the local chunk.
3. Send the local chunk to the next rank, into the peer's receive slot
   `s % 2`. The send waits until the add for that chunk completed.

### 5.2 Two streams

The NVLink transport uses two streams.

- `recv_stream`: every incoming copy and every add.
- `send_stream`: every outgoing copy.

A receive slot belongs to one direction and one stream, as section 3.1 states.

The dependency rules:

- Before `Recv(s)` into slot `s % 2`, `recv_stream` waits on the event recorded
  after the add of step `s - 2`.
- The add of step `s` runs on `recv_stream` after `Recv(s)`.
- After the add of step `s`, record an event on `recv_stream`.
- Before `Send(chunk c)`, `send_stream` waits on the event for chunk `c`.

The two streams let the outgoing copy of one chunk overlap the incoming copy of
the next. One stream would serialize them and halve the link use.

The RDMA transport keeps the same schedule. Each direction uses its own queue
pair, so the work requests overlap without stream events. `Done` and `Wait` poll
the completion queue.

### 5.3 Small messages

The ring latency is `2 * (N - 1)` steps. For a small bucket, use a direct
exchange or a recursive-doubling schedule. Add this only after the measurement
shows a need. Keep the ring as the default.

### 5.4 The registered transfer window

The NVLink transport moves bytes between two processes. A device pointer from
one process is not valid in another process. The transport must export the
memory that a peer writes into and import the peer's memory. This rule applies
to NVLink and to RDMA (`ibv_reg_mr`).

The seam passes an arbitrary bucket pointer to `AllReduceSumAsync` on every
step. The training loop allocates the buckets each step and frees them after.
Per-step registration is too expensive. A per-step `cudaIpcOpenMemHandle` also
leaks a mapping.

So `NativeGradientSync` owns a pool of registered transfer windows.

- Allocate the pool with `kernels::Alloc`. The pool size is the
  `DistributedConfig.max_inflight_reductions` field, default 8.
- Export each window once with `cudaIpcGetMemHandle`.
- Exchange the handle and the device identifier with every rank through
  `Rendezvous::ExchangeBytes`. The transport keeps the entry for its neighbor.
- Import the peer handles once with `cudaIpcOpenMemHandle`, and keep the
  mappings.
- Register the same windows with `ibv_reg_mr` on the RDMA path.
- Pass a pointer inside a local window to `Send` and `Recv`. The transport maps
  it to the same offset in the peer window.

The adapter calls `Transport::RegisterWindow` with each window at connect time
and after a growth.

The pool size matches the harness bucket count. `train.cc` uses four to eight
buckets, so eight windows let every bucket be in flight. A reduction in excess
of the pool drains one completion first. That case is rare.

The window holds one bucket. The adapter copies the caller's bucket into a free
window, runs the ring, and copies the sum back. Both copies are device-to-device
on the local card. They cost local bandwidth only.

A window grows when a reduction is larger than the window.

- The growth condition is `count` alone. Every rank sees the same `count` in the
  same order, because the model and the bucket plan are equal. So every rank
  grows on the same call and the handle exchange is symmetric.
- Grow every window in the pool to the next power of two of the new size. So
  the exchange is rare.
- The host-staging transport has no window. It stages through host memory.

### 5.5 The GradientSync adapter

`NativeGradientSync` implements `GradientSync` and drives the ring schedule.
The adapter is the bridge between the contiguous seam and the chunked token
schedule.

- `AllReduceSumAsync(buffer, count)`:
  1. Record an event on the main stream. The event fires after the caller
     produced `buffer`.
  2. Take a free window from the pool. If the pool is empty, drain one
     completion first.
  3. Make the adapter stream wait on the event.
  4. Copy `buffer` into the window.
  5. Enqueue the ring reduce-scatter and all-gather for the window.
  6. Append the request to a FIFO queue and return at once.
- `Wait()`:
  1. Drain the FIFO in order. For each request, poll `Done`, then call `Wait`
     with the watchdog deadline (section 5.6).
  2. After a ring completes, copy its window back into the caller buffer on the
     adapter stream. Return the window to the pool.
  3. Synchronize the adapter streams.

The ring runs on the adapter's streams, so the main stream is free. The pack
copy of the next bucket overlaps the reduction of the current bucket. This is
the overlap the asynchronous seam promises (docs/distributed-design.md
section 6).

The adapter also implements `AllReduceSum` for a synchronous caller. The call
issues `AllReduceSumAsync` and then `Wait`.

### 5.6 The watchdog

A rank failure must not hang the group. The seam `Wait()` takes no deadline, so
the adapter enforces one.

- `DistributedConfig` gains a `timeout_ms` field. The default is 600000
  milliseconds. That value gives one bucket ample time.
- `Wait()` computes a deadline as `now + timeout_ms` before it drains the FIFO.
- On `kTimeout`, the adapter prints the rank, the step, the slot, and the peer,
  then calls `Die`. A missing peer is a fatal group error, not a retry.
- The memory transport returns `kTimeout` on demand, so the CPU test covers the
  path. Section 13 adds the test.

## 6. Single precision only

The collective reduces single precision only. `ComputeType` is `float` in the
fp32 build. Remove the fp16 dtype branch from the collective.

[precision.md](precision.md) section 6 defines the policy. Stage 1 removes fp16
from the production path and keeps `--config=fp16` for the simulator only. So
the production reduction is fp32. The collective needs fp32 only.

The device target for this work is `--config=h100`, which is fp32. The `h200`
config is out of the collective path. The architect decides the `h200` fate as a
separate question.

## 7. Phases

### Phase 0 — Decision and scaffolding

Owner: architect, with the collective agent.

- Record the decision to drop NCCL as a hard dependency. Update
  [distributed-design.md](distributed-design.md),
  [distributed-plan.md](distributed-plan.md), and
  [distributed-launch.md](distributed-launch.md).
- Add a build define, for example `--define=collective=native`. Keep `nccl` as
  the default until parity.
- Move the rendezvous into `src/rendezvous.{h,cc}`. Update `src/distributed.cc`
  to use it.
- Add the `backends/cuda/collective/` directory and its `BUILD.bazel`.
- Add the RDMA repository rule `third_party/rdma.bzl` for libibverbs, in the
  style of `third_party/nccl.bzl`.
- Add the collective workstream to `AGENTS.md` section 1 (section 8 below).

Gate: the tree builds under both `--define=collective=native` and
`--define=collective=nccl`.

### Phase 1 — The schedule and the memory transport

Owner: collective agent, with the oracle agent for tests. This phase is CPU
only.

- Implement `RingAllReduceSum` against the `Transport` interface.
- Implement the memory transport. Reuse the shared-memory rendezvous from the
  NCCL mock.
- Add `//backends/cuda/collective:schedule_test`. It runs world sizes 1, 2, 4,
  and 8 against a reference sum.
- Add `//backends/cuda/collective:transport_select_test`. It checks the
  per-neighbor selection, including a mixed group.
- Use a relative tolerance. A ring adds in a different order than a single sum.
- Check that every rank ends with equal bytes.
- Add the schedule trace (section 10).

Gate: `tools/nanochat test --config=cpu` passes. The test needs no GPU.

### Phase 2 — The NVLink transport

Owner: collective agent.

- Add the registered transfer window and the CUDA IPC handle exchange
  (section 5.4).
- Add `cudaDeviceCanAccessPeer` and `cudaDeviceEnablePeerAccess`.
- Add the two streams, the event dependencies, and the add kernel.
- Add the host-staged fallback for a failed peer access.
- Implement `NativeGradientSync`, the seam adapter, and `CreateGradientSync` in
  `backends/cuda/collective/native_sync.cu` (section 5.5).
- Add `//backends/cuda/collective:nvlink_test` with the `gpu` and `manual`
  tags. Two processes, two cards, tiny shapes.

Gate: the two-rank test passes and matches the host reference within tolerance.

### Phase 3 — The RDMA transport

Owner: collective agent.

- Add the RDMA connection setup over the control plane.
- Register the transfer window for GPUDirect RDMA (section 5.4).
- Post one-sided RDMA writes with immediate, or two-sided send and receive. Poll
  the completion queue.
- Add `//backends/cuda/collective:rdma_test` with the `gpu` and `manual` tags.
  Two ranks over the adapter.
- Keep the NVLink transport as the first choice. Use RDMA when a peer is not
  reachable by NVLink.

Gate: the RDMA transport matches the host reference within tolerance. The
transport selection is correct for a mixed group.

### Phase 4 — Overlap and performance

Owner: collective agent, with the harness agent.

- Tune the chunk count and the slot count per bucket size.
- Measure the bus bandwidth and the sync fraction. Use `tools/nanochat doctor`
  first, then the `t3-bench` profile.
- Add the small-message schedule only if the measurement asks.
- Keep the bucket plan in `src/train.cc` unchanged unless the measurement asks.

Gate: the sync fraction is below 0.05. Record the bus bandwidth with the commit
hash.

### Phase 5 — Eight ranks on H100

Owner: collective agent, with the architect for the build.

- Build with `--config=h100`. Check the compile under `--config=sim-gate`.
- Run an 8-rank GPU test on the 8x H100 node.
- Run the T2 parity gate: an 8-rank loss curve against a 1-rank loss curve,
  within tolerance.

Gate: the 8-rank test and the T2 parity test pass.

### Phase 6 — Make native the default and clean up

Owner: architect.

- Make `collective=native` the default for the `h100` config.
- Keep `nccl` selectable for the `t4` config during the transition.
- Remove `@nccl//:nccl` from the CUDA config after the transition.
- Update the design documents.
- Decide the fate of the NCCL mock in `tools/cuda_sim/collective/`.

Gate: the merge gate in `AGENTS.md` section 5.5 passes with NCCL absent from the
default CUDA build.

## 8. Ownership

The current map in `AGENTS.md` section 1 has no row for the collective. The
kernel agents own `backends/cuda/kernels/<family>.cu`. The map does not list an
owner for `device.cu` and `gemm.cu`. This work needs a clear owner.

Add one workstream to the map.

| Workstream | Owns | Depends on |
|---|---|---|
| **Collective** | `backends/cuda/collective/**`, `dev/collective/**` | frozen `kernels.h`, `src/rendezvous.h` |

Suggested file split.

| File | Owner |
|---|---|
| `backends/cuda/collective/transport.h` | Collective |
| `backends/cuda/collective/nvlink_transport.cu` | Collective |
| `backends/cuda/collective/rdma_transport.cc` | Collective |
| `backends/cuda/collective/ring_schedule.{h,cc}` | Collective |
| `backends/cuda/collective/native_sync.cu` | Collective |
| `backends/cuda/collective/BUILD.bazel` | Collective |
| `src/rendezvous.{h,cc}` | Workflow |
| `third_party/rdma.bzl` | Architect |
| `.bazelrc`, root `BUILD`, `MODULE.bazel` | Architect |
| `docs/**`, `AGENTS.md` | Architect |

`include/nanochat/kernels.h` and `include/nanochat/model.h` do not change. The
asynchronous seam already exists. Keep the add kernel backend-private.

The RDMA work needs a device and a driver that the WSL2 host does not have. The
collective agent develops the schedule on the central processing unit and the
transports on the H100 node.

## 9. Broker and launch

The local broker in `tools/gpu.sh` serves one GTX 1080 Ti. It is not the
broker for an 8x H100 node. Treat the node as one resource.

- One `tools/nanochat dist` invocation holds one lock for the whole rank group.
  [distributed-launch.md](distributed-launch.md) already defines this. Keep that
  rule. Do not take one lock per rank.
- The lock covers all 8 cards. The `nvidia-smi` gate must see all 8 cards as
  idle before the group starts.
- Add a node-class sandbox profile to `tools/sandbox/profiles.conf`, for example
  `node-train`. Size the memory, the central processing unit quota, and the
  process count for eight ranks on one node. Do not reuse the `train` profile,
  which serves the WSL2 host.
- One rank per card. Set `CUDA_VISIBLE_DEVICES` to the rank, as the NCCL path
  already does.
- The T1 and T2 gates run on the node, under the broker. Keep the GPU tiers
  serialized, as `AGENTS.md` section 5.1 requires.
- Deploy the broker on the node. Do not point the node at the WSL2 broker
  configuration.

If a scheduler already serializes jobs on the node, do not add a second lock.
Integrate the rank group as one scheduler job. A second lock would deadlock
against the scheduler.

If the node has no broker and no scheduler, run the T1 and T2 gates through a
managed job on the node. Record the result with the commit hash.

## 10. The schedule trace

The two-stream event logic is the largest correctness risk. Build a trace, and
check it on the central processing unit.

The NCCL mock already has the pattern. `tools/cuda_sim/collective/nccl_mock.h`
exposes `AppendCollectiveRecord`, `ReadCollectiveLog`, and
`JudgeCollectiveRecords`. The deadlock case is already a test in
`tests/collective_sim_test.cc`.

Extend the pattern. The schedule emits one record per send, receive, and add.
Each record carries the rank, the direction, the transport kind, the step, the
chunk, the slot, and the event dependencies. A central processing unit test
with the memory transport replays the records and checks four rules.

1. The schedule uses every slot once at a time.
2. The dependency graph has no cycle.
3. Every wait has a matching completion.
4. A peer that never completes returns `kTimeout`, and the adapter stops.

The simulator agent owns the trace convention. The oracle agent owns the test.

## 11. Open questions and how to close them

Each question is either a discovery task or a decision.

- A discovery task has a command or a measurement. Run it, record the output,
  and close the question.
- A decision needs an owner and a deadline. Write a short record, then close the
  question.

### 11.1 The RDMA adapter and GPUDirect RDMA

Type: discovery. Run the probes on the 8x H100 node.

- `ibv_devinfo -v` — the adapter list, the active speed, and the active width.
- `ibv_devices` and `rdma link show` — the device names and the link state.
- `nvidia-smi topo -m` — the GPU-to-adapter affinity. A `PIX` or `NODE` entry is
  good. A `SYS` entry crosses the central processing unit and is slow.
- `nvidia-smi nvlink -s` — the NVLink status and the per-link speed.
- `lspci | grep -i -E 'mellanox|infiniband|ethernet'` — the adapter vendor and
  model.
- `lsmod | grep -E 'nvidia_peermem|nv_peer_mem'` — the peer-memory module.

A list of adapters is not proof of GPUDirect RDMA. Write a small test that calls
`cudaMalloc`, then `ibv_reg_mr` on the device pointer, then frees the region. A
failure at `ibv_reg_mr` means no device registration. Put this test in the
collective directory and tag it `manual`.

Extend `tools/nanochat doctor` with an RDMA section. It already prints the GPU,
the driver, the sandbox, and the lock state. Add the adapter list, the link
speed, and the peer-memory module. Keep the `--json` shape.

Record the result in a new `docs/node-h100.md`, or extend
[host-portability.md](host-portability.md).

### 11.2 RDMA as a fallback or as a required transport

Type: decision with a reversible design.

Make transport selection a runtime rule.

| Condition | Transport |
|---|---|
| Every peer pair passes `cudaDeviceCanAccessPeer` | NVLink |
| A peer is on another node, or peer access fails | RDMA |
| No RDMA adapter and no peer access | Host staging |

The rule is per edge, not per group. `RingTransports` applies it to each of the
two ring neighbors, so a mixed group works (section 3.2).

With this rule, the answer changes only the default preference and the tests.
The schedule does not change.

Ask the roadmap owner one question in writing: will a second node join within
the current plan horizon? If the answer is no, RDMA is a fallback and its
performance work can wait. If the answer is yes, RDMA is a first-class path and
every phase tests it.

Record the rule in [distributed-design.md](distributed-design.md) beside the
current section 5 table.

### 11.3 Retire fp16, or only remove it from the collective

Type: decision, mostly closed.

[precision.md](precision.md) section 6 already accepts Stage 1. Stage 1 removes
fp16 from the production path and keeps `--config=fp16` for the simulator only.
So the production reduction is fp32, and the collective needs fp32 only.

Scope the collective to fp32 and close this question for this work. Add one
sentence to [precision.md](precision.md) section 6. Let the architect decide the
`h200` config as a separate question. That question does not block the
collective.

### 11.4 Node ownership and the broker

Type: discovery plus an access contract.

Ask the node owner four questions, in writing.

1. Is the node exclusive to this project, or shared?
2. Does a scheduler already serialize jobs on it, for example Slurm or
   Kubernetes?
3. What are the driver and toolkit versions?
4. Is a systemd user manager available for the sandbox?

Probe the node. Extend `tools/nanochat doctor` to report the card count, the
driver, the toolkit, the interconnect, and whether a scheduler is present.

Then pick one broker case, as section 9 describes. Record the access contract in
`docs/node-h100.md`, in the style of the Kaggle section in
[host-portability.md](host-portability.md).

### 11.5 The schedule trace

Type: recommendation, accepted.

Build the trace in phase 1, as section 10 describes. It is cheap, and it removes
the largest correctness risk in the two-stream design.

## 12. Summary of open questions

| Question | Type | How to close | Owner | Blocks |
|---|---|---|---|---|
| RDMA adapter and GPUDirect | Discovery | `ibv_devinfo`, `nvidia-smi topo -m`, and an `ibv_reg_mr` probe | Collective | Phase 3 |
| RDMA fallback or required | Decision | One written roadmap answer, plus a reversible selection rule | Architect | Phase 3 default |
| Retire fp16 | Decision, mostly closed | One sentence in [precision.md](precision.md); the `h200` fate is separate | Architect | Nothing |
| Node ownership and broker | Discovery and contract | Four written answers; a `doctor` extension; `docs/node-h100.md` | Architect | Phase 5 gates |
| Schedule trace | Recommendation | Reuse the mock's JSONL and coordinator pattern | Collective and Oracle | Phase 1 |

## 13. Test matrix

| Tier | Target | What it checks |
|---|---|---|
| T0 CPU | `//backends/cuda/collective:schedule_test` | The ring at world sizes 1, 2, 4, 8; the trace; the missing-peer timeout |
| T0 CPU | `//backends/cuda/collective:transport_select_test` | The per-neighbor selection, including a mixed group |
| T0 CPU | `//src:distributed_test` | The host reference, unchanged |
| T0 CPU | `//src:train_parallel_test` | The mean scale and the resume, unchanged |
| S0 | `//tests:collective_sim_test` | The NCCL mock, only while NCCL stays selectable |
| T1 GPU | `//backends/cuda/collective:nvlink_test` | A two-rank NVLink sum against the host reference |
| T1 GPU | `//backends/cuda/collective:rdma_test` | A two-rank RDMA sum against the host reference |
| T2 GPU | An 8-rank loss curve against 1 rank | The full training parity |

The schedule test is the primary new gate. It carries the algorithm. The
transport tests carry only data movement.

## 14. Risks and controls

| Risk | Effect | Control |
|---|---|---|
| A ring adds in a different order | The result differs from the reference by a few ulps | Use a relative tolerance in every test |
| Peer access is unavailable | The NVLink copy fails | Detect with `cudaDeviceCanAccessPeer`, then choose RDMA or host staging |
| GPUDirect RDMA needs a kernel module | The RDMA path registers no device memory | Check the peer-memory module at setup. Fail with a clear message |
| A rank fails during a collective | The job hangs | Bound every wait with the watchdog deadline (section 5.6). Test the timeout path |
| Per-step buffer registration | The NVLink path is slow or leaks mappings | Register a fixed window pool once (section 5.4). Grow only when `count` grows |
| Two streams deadlock each other | The pipeline stalls | Document the two event rules. Test with the schedule trace |
| The H100 node is not available locally | The T1 and T2 gates cannot run here | Use `--config=sim-gate` for the compile, and the node for the gates |
| The RDMA adapter topology changes | The tuned chunk count is wrong | Derive the chunk count from the measured bandwidth |
| The node broker is misconfigured | Concurrent jobs collide | One lock per group. Deploy the node broker separately |

## 15. Definition of Done

1. `tools/nanochat test --config=cpu` passes with the new schedule test.
2. The T1 NVLink test passes with two ranks.
3. The T1 RDMA test passes with two ranks.
4. The 8-rank T2 parity test passes on the 8x H100 node.
5. The sync fraction is below 0.05 for the target model.
6. The default CUDA build needs no NCCL.
7. The `h100` compile gate passes under `--config=sim-gate`.
8. `tools/nanochat lint` passes.
9. The design documents match the code.
