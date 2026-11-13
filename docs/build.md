# Build

One Bazel workspace builds every host and every precision. `tools/nanochat`
selects the configuration, applies the sandbox, and runs the command.

```bash
tools/nanochat build                      # the default fp32 build
tools/nanochat build --config=bf16        # the bf16 compute type
tools/nanochat build --config=hopper      # a real Hopper build
tools/nanochat test                       # every non-GPU test
tools/nanochat lint                       # the style gate
```

## What the build produces

The workspace produces four things.

- **One stage binary.** It is the single command binary of the tree, and it
  holds nine subcommands. One dispatch path guards every subcommand with the
  sandbox check, so a handler never repeats that check.
- **Two backends.** The CPU backend is the reference and the correctness
  baseline. The CUDA backend is the tuned path. Both implement the same seam.
- **One Python distribution.** It reaches the binary through a small C ABI and
  a shared library, and it adds no third-party package to the base install.
- **One test tree.** The CPU loop runs under a sandbox profile, and the device
  tiers run through a queue.

The subcommand table is the authority for the binary's surface. A new
subcommand adds one row here and one handler function.

<!-- BEGIN GENERATED: subcommands -->
| Subcommand | Purpose |
|---|---|
| `train` | one training run over packed parquet rows |
| `eval` | forward-only bits-per-byte evaluation |
| `generate` | sample tokens from a checkpoint |
| `score` | stateless scorer over one evaluation fixture |
| `sft` | supervised fine-tuning on a packed SFT shard |
| `rl_worker` | the reinforcement-learning worker the Python bridge drives |
| `rl_step` | one file-driven reinforcement-learning step |
| `roofline` | the performance model and the roofline report |
| `eval_bench` | the forward-only evaluation benchmark |
<!-- END GENERATED: subcommands -->

Every executable calls the sandbox guard at startup. A binary that starts
outside `tools/nanochat` exits with the corrective command. The guard applies
the CPU, memory, and device budget of the profile, so a test or a short run
cannot take the host from another worker.

## Layers and the one seam

Dependencies point downward. The tree has one interchangeable seam, and it is
the kernel API at level 1.

```text
L6  App / CLI        the one binary and its subcommand table
L5  Train driver     the training loop, the checkpointer, the logger, the MFU counter
L4  Optim            AdamW, Muon, the scheduler, the parameter groups
L3  Model / graphs   the model, prefill, decode, and the KV cache
L2  Ops (internal)   the composite operations the graphs call
L1  KERNEL API       <-- THE BACKEND SEAM
L0  Backend          allocation, streams, launch, and the vendor runtime
```

Host utilities sit beside levels 2 to 5 and never above the kernel API. They
are the tokenizer, the data loader, the random generator, the sampler, the
scheduler, the logger, and the MFU counter. A host utility needs no device.

Level 1 is a header of function declarations. Every name in it is a pointer to
`ComputeType`, an opaque stream, or a plain scalar. The seam names no vendor
type and no model. That is what makes a backend a build and link choice: the
workflow above the seam compiles once and links against either backend.

## Dependency invariants

The build graph enforces five rules.

1. **The workflow reaches down only.** Source above the seam includes
   `nanochat/**` and no vendor header. One host package may include the
   host-only compression header, and the native parquet reader is the only user
   of it. The model, the kernels, and the training runtime never link a
   database library.
2. **A backend reaches down only.** Source under `backends/` includes the
   kernel header and the tensor header, and nothing else that carries model
   knowledge. A kernel knows nothing about the model. Two leaf headers are the
   named exception: one declares the simulated capability table, and one
   declares the local device list. A leaf depends on the tensor header alone
   and holds no arithmetic, so the reference backend still computes every
   value. A new edge of that shape needs the same review as any other change
   to this rule.
3. **The seam holds no vendor type.** A stream is opaque, a buffer is
   `ComputeType*`, and no vendor header is reachable from the seam.
4. **No cross naming.** Nothing above the seam names a model or a device, and
   nothing below it names a graph. The seam names neither.
5. **A bring-up adds a directory.** A model bring-up edits one model directory.
   A device bring-up edits one backend directory. Neither edits a frozen
   header, a shared build file, or another package without a contract
   extension first.

Rules 1 to 3 are load-bearing. They let a backend be replaced as a build and
link decision rather than as a source change, so a stable interface is the
default and cheap by construction. Rules 4 and 5 are the isolation promise:
they keep two workstreams out of each other's files. The evolution rules for a
frozen header are in
[model-and-device-bringup.md](model-and-device-bringup.md#18-evolution-rules).

## Build configurations

Architecture, precision, backend, collective, and simulator are build and link
choices, not run-time switches. A named configuration sets one or more of them,
and named configurations are additive: `--config=hopper --config=sim-gate` is
one build.

The table states the effective build of one named configuration, added to the
default. The simulator column is prose, because it names an engine rather than
a flag value. A script checks the other four columns against the tracked Bazel
configuration, so a row cannot drift from the build.

<!-- BEGIN GENERATED: build-configurations -->
| Configuration | Architecture | Precision | Backend | Collective | Simulator |
|---|---|---|---|---|---|
| _the default build_ | `sm_75` | fp32 | cpu | native | off |
| `--config=bf16` | `sm_75` | bf16 | cpu | native | off |
| `--config=fp8` | `sm_75` | fp8 | cpu | native | off |
| `--config=cuda` | `sm_75` | fp32 | cuda | native | off |
| `--config=native` | `sm_75` | fp32 | cuda | native | off |
| `--config=sim` | `sm_75` | fp32 | cpu | native | the reference engine |
| `--config=sim-gate` | `sm_75` | fp32 | cpu | native | the reference engine, with the compile gate |
| `--config=h100` | `sm_90` | bf16 | cuda | native | the H100 device profile |
| `--config=h200` | `sm_90` | bf16 | cuda | native | the H200 device profile |
| `--config=hopper` | `sm_90` | bf16 | cuda | native | off |
<!-- END GENERATED: build-configurations -->

**A simulator profile is not a device build.** The H100 and H200
configurations compile the capability override, and `GetCaps()` answers from
the device profile table instead of from the card. A Hopper build with the
simulator off compiles the real CUDA backend. A check that needs a card chooses
that build. A report that is read without a card chooses a simulator profile.

There is one collective. Both the simulator profile and the device build link
it, so a one-device run reduces the collective to no operation.

The default architecture is the oldest card the tree supports. A build for an
older card is not a target, so a binary built here holds no code for one.

## Resource bounds

One small host serves several agents, and every worktree is its own workspace
with its own long-lived server. Left unbounded, those servers accumulate, and
each one sizes its action pool to the whole machine. The tracked Bazel
configuration therefore bounds the servers, and `tools/nanochat` serializes the
builds. The tracked file holds the values; this section holds the reason for
each one.

- The server heap is capped.
- An idle server exits instead of lingering for the default three hours.
- The kernel may stop a server under memory pressure.
- One server's action pool and memory hint are bounded.
- `tools/nanochat build` and `tools/nanochat test` take one exclusive lock, so
  at most one worktree compiles at a time. The lock descriptor is closed on the
  client so the detached server cannot inherit it.
- `tools/nanochat shutdown` stops the current workspace's server.

The host budget that these bounds protect is in [sandbox.md](sandbox.md).

### One device compile at a time

A device compile of a large template target holds two to three GiB. Four such
actions at the default action count exceed the host, and the memory hint of the
build is a scheduling hint and not a limit. Two rules follow.

- Build a heavy template target with one action. A local configuration may
  define its own action cap, so the tracked default stays unchanged.
- A stalled build is not a slow build. Read the memory pressure before you
  wait. The kernel reports the share of time that a task stalled on memory.
  The process statistics report the processor time that a compile has used. A
  high stall share beside a low processor time means that the build waits on
  memory. Stop it and restart with one action, because a longer wait does not
  finish it.

### A build inside a session

A session worktree places the build in the session slice and in a private
network namespace. The kernel then stops the compile and not the session. An
external repository that is not already fetched stops a build with no network,
and one command with the host network fetches it. A repeat build reuses the
server. The shared headers stay read-only in a session, and one flag makes them
writable for the workstream that owns them. Outside a session there is no slice
and no namespace, so a long build wants a hand scope.

## Third-party code

The tree has three third-party dependencies, and only one of them reaches a
shipped binary.

| Dependency | Kind | Cost to the binary |
|---|---|---|
| the CUDA runtime and cuBLAS | linked library | none at build time, and the vendor runtime at run time |
| the host compression library | linked library, host tools only | one link edge in the native parquet reader |
| CUTLASS | headers only | no library and no symbol |

The CPU build links the math library and the host thread runtime. Neither
backend links cuDNN or a vendor collective: the native ring needs no vendor
library.

### CUTLASS is a build-time dependency

CUTLASS supplies the Hopper mainloop of the hand GEMM. It is headers and
nothing else, so the build links no library and no shipped binary holds one
CUTLASS symbol.

| Property | Value |
|---|---|
| Pin | commit `b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd`, tag `v4.8.0-23`, `version.h` 4.8.1 |
| Checksum | the SHA-256 of the archive, checked on every fetch |
| Licence | BSD-3-Clause, copied into the fetched repository as `LICENSE.txt` |
| Contents | `include/` only |
| Override | `NANOCHAT_CUTLASS_PATH`, for a host with no network |

The rule fetches the tree as an external repository and does not track it. That
is a build decision with three effects. The lint pass never reads the vendor
headers, and those headers break several of its rules on purpose. The format
gate never reads them. A diff never holds them.

The vendor headers also emit many warnings, and the compile gate builds with
warnings as errors. A target that compiles against CUTLASS therefore states its
own options and does not use the project CUDA options.

A Hopper target names its own architecture. The rule that compiles a CUDA
target takes an architecture list, and that value replaces the command-line
default for that target alone. The mainloop needs the Hopper tensor-core
architecture spelling, so its target states it and the tree's default build
stays at the oldest supported architecture. No tracked configuration line
changes, and no command line gains a flag.

## Backends, precision, and devices

A backend is selected at build and link time.

| Backend | Implements | Notes |
|---|---|---|
| the CPU backend | every function of the seam, with plain loops | the reference, the correctness baseline, and the fixture producer; the math library and the host thread runtime |
| the CUDA backend | every function of the seam; cuBLAS for GEMM | fp32 on the oldest card, bf16 on newer cards; no cuDNN, no vendor collective |

### Precision

One precision exists per build, and no run-time type dispatch occurs. The build
selects the source, or the code reads the build's compute type. No source tests
the numeric define; the one header that defines the value is the only file that
may branch on it.

- The oldest card and the CPU compute in fp32.
- Ampere and newer cards compute in bf16.
- The fp8 compute type computes in bf16 and quantizes the eligible matrix
  multiplications.
- The persistent state stays fp32 in every build, so the parameter gradient is
  always a single-precision value even when the compute type is bf16.

The backend reports the type set it supports. A bf16 build on a device without
bf16 fails at model creation, with the device name in the message. The
precision policy, the fp8 tier, and the accuracy bounds are in
[precision.md](precision.md).

### Devices

A card is a row of a backend declaration file, and that file is the single
device declaration of the tree. The profile table that the simulator reads and
the planner both derive their rows from those files, so a device bring-up adds
a file and no header row. The capability bit names come from the capability
vocabulary.

Two cards serve the tree today: a Hopper training card and the oldest supported
Turing card. The Turing card has fp32 arithmetic, tensor-core instructions, and
no bf16 arithmetic. Hopper adds bf16 arithmetic. A device declares only the
precision and the capability bits that its silicon honors, because a
declaration may not grant what the card lacks. The planner and the roofline
report refuse a device that has no row, and they name it.

### Adding a backend

A backend is a directory under `backends/` and one configuration value. The
workflow sources and the public headers need no change. The directory provides
every function of the seam, a declaration file for each device it serves, and
the tests that compare its kernels against the reference backend on small
shapes.

A new kernel family reaches below the replacement boundary of the seam. It
therefore touches a frozen header, under the evolution rules in
[model-and-device-bringup.md](model-and-device-bringup.md#18-evolution-rules).

## The Python distribution

The Python surface is one distribution with two import packages and four
boundaries. A boundary is a promise about imports, and the base install carries
every boundary at no third-party cost.

| Boundary | May import | Counted dependencies |
|---|---|---|
| the C++ core | the C runtime and the vendor runtime | three to five libraries |
| the Python core | the standard library, and the C++ core through the C ABI | zero packages |
| the adapters | the standard library, the Python core, and the adapter extras | one to six packages |
| the hosts | any boundary | per host |

The arrow points down. A boundary imports a lower boundary and never a higher
one. Five rules hold this shape.

1. The Python core imports the standard library and nothing else.
2. An adapter module imports its heavy package inside the function that needs
   it, so the module loads with no extra installed and the base install stays
   pure.
3. A host imports either package. A notebook is a host.
4. No package imports a host.
5. A public header names no vendor type. That rule is invariant 3 of the build.

An adapter is the only place that names the reference implementation of the
model. The Python core reaches an adapter by being handed a callable, never by
naming the module.

### The gates that hold the boundaries

Five gates run in the CPU test loop, so a broken boundary fails a build and not
a review.

| Gate | Fails when |
|---|---|
| the standard-library scan | a file of the Python core imports a name outside the standard library, including an import inside a function |
| the isolated import | a fresh interpreter with no site packages cannot import the package and resolve every public name |
| the pure base install | the package description declares a base dependency, or the file is missing |
| the fixture provenance | a fixture record names no producer environment or no producing module |
| the core independence | a file of the Python core mentions the adapter package, even in prose |

### Installing

The base install carries both import packages and no dependency. A task
installs one extra, and the extras name a job rather than a package list.

| Task | Install line |
|---|---|
| read a stage from the Hub | `pip install "nanochat-cpp[hub]"` |
| train on a rented device | `pip install "nanochat-cpp[hub]"` and a binary wheel |
| produce a parity fixture | `pip install "nanochat-cpp[reference,nanochat]"` |
| build SFT data from the Hub | `pip install "nanochat-cpp[reference,nanochat,hfdata]"` |
| develop on a workstation | the source tree and Bazel |

The build resolves the interpreter through the path, so an activated
environment becomes the test interpreter and its site packages can hide a
layering break. Keep the tree's interpreter free of third-party packages, and
install an extra in one environment outside the tree.

### The shared library

A pure wheel cannot carry one shared library for every architecture, so a
binary wheel states its architecture in the package name. The search order of
the loader, the fallback build, and the cache are in
[python.md](python.md#32-kaggle-delivery). A wrong architecture fails at
initialization with the corrective command.

### A container

A rented container installs the package at image build and fetches the weights
at run start. The weights live in a volume keyed by the stage revision, so a
new checkpoint costs a download and not an image build.

A build cannot run at image build time against a cold cache, because a volume
attaches to a function and not to an image. The image therefore copies the
module graph and the tooling, resolves the external repositories, and the build
runs inside the container against a cache volume. A warm cache makes the second
run cheap, so one warm function pays the first compile.

The toolkit is a layer above the source tree, so a source change rebuilds it.
Keep it out of the image once the binary wheel is the install path.

### A notebook

A notebook cell installs the package and one binary wheel, and it sets the
sandbox backend to none. The cell does no clone and no build. The logic stays
in the notebook package, and a cell holds no logic.

## The parity seam

A comparison with the reference implementation is a file, not an import.

```text
      producer (an adapter, offline)        consumer (the Python core, no torch)
torch, nanochat  -->  fixtures/*.bin  -->  a CPU test
a data library   -->  packed rows     -->  the supervised fine-tuning command
```

The producer runs in an environment that holds the reference checkout and the
adapter extras. It writes the bytes and records that environment. The consumer
reads the committed file with the standard library only, in the CPU loop.

A fixture record names two things beside its hash and its size. The producer
record names the interpreter, the reference library versions, and the reference
revision. The generator record names the module that wrote the bytes. A gate
fails a record that lacks either one, so a parity claim always names an
environment that a reader can rebuild.

One command runs a producer: `tools/nanochat parity produce <kind>`. The tool
resolves the reference environment from the extras of the distribution and
runs the adapter module. No producer runs under `tools/nanochat test`, so the
build lock and the sandbox cover every long job.

The producer lives in the adapter package, and the fixture travels as data. A
new comparison adds a kind to that package and a test that reads the bytes. The
known differences from the reference implementation, with their status, are in
[parity.md](parity.md).

## Not yet built

These designs have no code yet. Each states the change and the check that
accepts it.

**Tensor cores in the attention kernel.** The attention kernel computes fp32
statistics with an online softmax and does not use tensor cores. A tensor-core
variant is the largest available gain on a small card. It belongs in the
unpromoted prototype area first, and it must beat the current kernel at the
training shape on a real measurement.

**Two cards from one notebook.** A notebook host provides two cards. Two
independent jobs, one per card, need no change. Data-parallel training over
both cards needs the device mesh and the gradient seam of
[distributed.md](distributed.md).

**An image with no toolkit.** The container currently holds the source tree,
the toolchain, and a cache volume. An image that installs only the binary wheel
needs no toolkit, no Bazel, and no cache, and it shortens every rebuild. The
change is one install line, and the check is a cold container that trains one
step.

## Logs

A dated record of a real run. An entry states the command, what happened, and
the rule it produced. A rule that an entry justifies appears in the section
named at the end of the entry. This is a record and not a reference: it holds
the measurements that no other file holds.

### 2026-11-08 — three device compiles, and the kernel stopped one

**Command.** `tools/nanochat build --jobs=4` over the Hopper GEMM targets.

**Observed.** Three device compiles ran at once and held 2.3 to 2.9 GiB each.
The kernel ran the out-of-memory killer on one compile. The swap file was
already full, so the host thrashed before it killed the process. An operator
feels that thrash as a freeze.

**Insight.** The default action count bounds the count and not the memory. One
compile of a large template target is the unit that must fit.

**Rule.** [Resource bounds](#one-device-compile-at-a-time).

### 2026-11-10 — the build cache on a rented volume

**Command.** `modal run tools/modal/app.py::warm_cache`, then one build of the
Turing leg inside a container.

**Observed.** The build had 190 actions and read 133 disk cache hits. A warm
volume took 19.1 seconds and the first read after the warm function took 101.9
seconds. The same cache on a local disk read in 3.4 seconds. One package layer
above the source tree cost 215 and 63 seconds of package installation for a
profiler that a card job never runs.

**Insight.** The volume's latency per file is the cost, so a warm volume is
cheap and a cold one is expensive. The first run of a leg pays the whole
compile in a container with four cores, so one function with a long timeout
pays it once.

**Rule.** [A container](#a-container).

### 2026-11-12 — a stalled build is not a slow build

**Command.** `tools/nanochat build --jobs=4` over two device targets, beside two
editor processes.

**Observed.** The two compiles held 3.5 GiB and the editors held 1.0 GiB, so the
host of 7.8 GiB had 233 MiB free. The build computed almost nothing: one
compiler process used 29.7 seconds of processor time in 1122 seconds of wall
clock. There was no error and no out-of-memory kill. A restart with one action
took the memory pressure from 84 percent to 3 percent and raised the available
memory from 1.7 GiB to 5.2 GiB.

**Insight.** The memory pressure file reports the share of time that a task
stalled on memory. Field 14 of the process statistics reports processor time. A
high stall share beside a low processor time identifies a build that waits on
memory.

**Rule.** [One device compile at a time](#one-device-compile-at-a-time).
