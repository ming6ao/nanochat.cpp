# nanochat.cpp — Design

A minimal, hardware-portable, from-scratch training and inference stack for the
nanochat architecture, in the spirit of [llm.c](https://github.com/karpathy/llm.c)
and [llama.cpp](https://github.com/ggml-org/llama.cpp). No PyTorch, no autograd
engine, no graph compiler. One model definition, swappable hardware backends.

Status: design. Nothing built yet.

This document is the design rationale: goals, invariants, and the reasoning
behind the decisions. Interface reference and how-to material lives under
[docs/](docs/README.md).

---

## 1. Goals and non-goals

### Goals

- Train and run the **nanochat architecture** (RMSNorm, RoPE, QK-norm, GQA,
  relu^2 MLP, sliding-window attention, value residual, Muon + AdamW) with no
  framework dependency.
- **Hardware portability by replacement, not by configuration**: adding a
  backend means adding a directory that implements one header. The seam is
  designed so the public API and model definition are stable by default:
  changes to them are exceptional, additive where possible, and handled under
  the evolution rules in §7.
- **First-class on Pascal (GTX 1080 Ti, sm_61)** and **Turing (T4, sm_75)**;
  later NVIDIA generations are a backend variant, not a rewrite.
- Keep the mainline small and readable enough to audit in one sitting.
- Every operator independently testable against a fixed oracle.

### Non-goals

- General-purpose tensor library or model framework.
- Autograd / differentiation engine (see §5).
- Graph IR, fusion passes, scheduling compiler (see [docs/model.md](docs/model.md)).
- Dynamic shapes, runtime dtype dispatch, runtime kernel selection.
- TPU / systolic-array backends.
- Non-NVIDIA accelerators (ROCm/Metal are future `backends/<x>/` packages).
- Whole-block megakernels for training (see §4.3).

---

## 2. Architecture overview

Dependencies point strictly downward. There is exactly one interchangeable
seam.

```text
L6  App / CLI        train_main.cc  generate_main.cc  eval_main.cc
L5  Train driver     TrainLoop  Scheduler  Checkpointer  Logger  Mfu        train.cc
L4  Optim            AdamW  Muon  param groups                              optim.cc
L3  Model / graphs   Model (forward+backward)  Prefill  Decode  KvCache    model.cc generate.cc
L2  Ops (internal)   Linear  Block  Mlp  ValueResidual  Smear  Backout      ops.cc
L1  KERNEL API       <-- THE BACKEND SEAM  (nanochat/kernels.h)
L0  Backend          backends/cpu/   backends/cuda/   alloc . stream . launch
    Vendor           libm / OpenMP   |   cudart + cuBLAS (+ cuDNN, NCCL)
```

Host utilities sit beside L2-L5 and never above the kernel API: `Tokenizer`,
`DataLoader`, `Rand` (xorshift), `Sampler`, `Scheduler`, `Logger`, `Mfu`. The
concrete file layout is in [docs/build.md](docs/build.md); the kernel seam is
specified in [docs/kernels.md](docs/kernels.md).

### 2.1 Dependency invariants (enforced by the build graph)

1. `src/**` includes only `nanochat/**`. **No vendor headers outside `backends/`,
   with one exception: `src/tokenizer/` is the one host package that may include
   the host-only DuckDB header.** The model, the kernels, and the training
   runtime never link DuckDB.
2. `backends/**` includes only `nanochat/kernels.h` and `nanochat/tensor.h`.
   **A kernel knows nothing about GPT.**
3. `kernels.h` contains no vendor types: pointers are `ComputeType*`, streams are
   opaque, and no CUDA header is ever included from it.

These three rules are the load-bearing invariants: they let a backend be
replaced as a build/link decision rather than a source change. API stability is
the default and cheap by construction, not a guarantee that the API cannot
move — when it must move, see §7.

### 2.2 Execution entry

Running anything — build, test, training, eval, benchmark, or verification —
goes through one host-tooling command, `tools/nanochat`. It selects the resource
profile, applies the sandbox, acquires the GPU broker when needed, and wires the
Bazel test wrapper. Executables call `nanochat::RequireSandboxOrDie` at startup,
so a process launched outside the entry point fails loudly instead of consuming
the host. This is outside L0-L6: it changes neither the kernel seam nor the
public API. See [docs/sandbox.md](docs/sandbox.md).

---

## 3. Fusion boundary rule

Fuse ops `A -> B` with intermediate `Y` **iff all five hold**:

1. **Decomposition match** — one (grid, block) config efficiently covers both;
   the iteration spaces map onto the same CTA/thread ownership.
2. **Residency** — the part of `Y` that `B` needs fits in registers + shared
   memory for that config, so the HBM round trip is genuinely avoided.
3. **No library displacement** — neither `A` nor `B` is better served by
   cuBLAS/cuDNN, which cannot participate in the fusion.
4. **Occupancy survives** — the fused kernel's register/shared usage does not
   drop occupancy past the break-even point.
5. **Derivable backward** — the fused backward is derivable and testable (see §5).

Consequences for nanochat:

| Boundary | Fuse? | Reason |
|---|---|---|
| `RmsNorm -> RoPE -> scale` on q,k | **yes** (`QkPrep`) | row-local, on-chip, no library |
| `residual add -> RmsNorm` | **yes** | row-local, on-chip |
| `softcap -> cross-entropy` | **yes** (`Classifier`) | row-local over vocab |
| `relu^2 -> c_proj GEMM` | no | cuBLASLt epilogues cannot express relu^2 |
| `c_q GEMM -> RmsNorm(q)` | no | GEMM tiles vs whole-row reduction |
| `attention` | separate | own T x T pattern |
| `embedding gather` | separate | indexed access |
| `GEMM -> GEMM` | no | cuBLAS owns both |

Performance **validates** a boundary; it does not choose it. Structure
(decomposition, residency, library ownership) determines which boundaries are
legal.

### 3.1 Candidate v2 fusion

`QkPrep` can fold into the `Attention` kernel, because the RMSNorm reduction is
over `head_dim`, which is contained within a tile row (it does not span CTAs).
Legal under the rule; defer to v2 because it couples the attention backward to
the norm/rope backward.

---

## 4. Compute strategy

### 4.1 GEMM

Keep cuBLAS/cuBLASLt as the GEMM engine. It is near-peak on GP102 (11.34 TFLOPS
fp32), per-shape tuned, and portable across architectures. Do not hand-write a
GEMM.

### 4.2 Launch overhead

At `d8_s512` the full step is roughly 0.9 s and kernel launches are under 1% of
it. CUDA Graphs are therefore an **optional internal optimization** of the CUDA
backend, not an architectural element. Add only if profiling demands it.

### 4.3 No megakernels for training

Whole-block megakernels require hand-written GEMMs (cuBLAS is not callable
device-side), collapse occupancy, serialize phases behind grid barriers, and do
not avoid HBM round trips across tile boundaries. They are the wrong trade for
throughput-bound training.

**Exception:** the **decode** graph is batch-1 and latency-bound. A fused
persistent decode kernel is a legitimate `dev/kernels/` experiment there, on
MMA-capable hardware (Turing+). It must beat a `cuBLAS + CUDA Graphs` baseline
before promotion, and it does not apply to Pascal (no `mma` instructions).

### 4.4 Direct-MMA / persistent kernels

Not applicable to Pascal (sm_61 has no tensor cores or `mma`/`wmma`). The win in
systems like Mirage MPK is whole-model inference fusion on Hopper, not GEMM
throughput. Keep cuBLAS; revisit only for decode on sm_75+.

---

## 5. Autograd policy

**There is no autograd engine.** No tape, no source-to-source AD pass, no
reverse-mode template metaprogramming.

- The graphs are fixed, so backward is a fixed program that you author.
- A runtime engine would reintroduce the per-op bookkeeping that profiling
  showed is pure overhead for a graph whose topology is fixed at build time.
- Fused kernels must have their backward derived as a unit; a per-op engine
  cannot differentiate through them.

**Mechanism:**

- **Save-for-backward workspace.** Each forward kernel writes exactly the
  tensors its backward needs (e.g. `rstd` for `RmsNorm`, softmax stats `(m, l)`
  for `Attention`, the pre-activation for `Pointwise`) into pre-allocated, named
  workspace slots. The slot set is static. See [docs/model.md](docs/model.md).
- **`Model::Backward()`** is a hand-written sequence of kernel backward calls in
  reverse order. The graph is the code.
- **Macros/templates only for trivial pointwise fwd/bwd pairs.** Do not
  generalize beyond pointwise.
- **Validation:** a finite-difference gradient check per kernel in
  `backends/cuda/kernels/`, plus the torch oracle in `tests/`. See
  [docs/testing.md](docs/testing.md).

---

## 6. What is deliberately absent

- Autograd engine / tape / AD pass.
- Graph IR, fusion passes, memory planner, plan cache.
- Runtime kernel registry / vtable / dynamic dispatch.
- Dynamic shapes; runtime dtype switching; runtime recompute.
- TPU / systolic backends; ROCm/Metal (future backend packages).
- Whole-block megakernels and persistent direct-MMA kernels for training.
- Third-party runtime libraries beyond CUDA and cuBLAS.
- Container runtimes and sandboxing daemons (Docker, Podman, bubblewrap,
  firejail). Isolation is host cgroup v2 plus user namespaces, applied by
  `tools/sandbox.sh`; no image, no daemon, no new dependency.

---

## 7. Evolution rules

The public API (`model.h`, `config.h`, `tensor.h`, `kernels.h`) is stable by
default. Prefer additive changes, and keep any edit to a frozen header
architect-owned and local to the seam (`AGENTS.md` §1). The rules below cover
the common cases.

- **Add hardware**: new `backends/<x>/` implementing `kernels.h`. Nothing above
  needs to change. See [docs/build.md](docs/build.md).
- **Fuse more**: only if the §3 rule holds and the fused backward is derivable.
  Develop in `dev/kernels/`, prove with a benchmark, then promote.
- **Add a graph variant** (SFT/RL head): add to `model.cc`; topology stays in one
  place.
- **Optimize**: profile on native Linux (CUPTI is unavailable under WSL2).
  Priority order — attention fwd/bwd, Muon, QkPrep, sparse embedding backward.

---

## 8. References

- llm.c — single-TU-per-backend simplicity, pre-tokenized `.bin` data,
  `debug_state.bin` oracle, bump-allocator workspace, Makefile autodetection.
- llama.cpp / ggml — backend as a swappable implementation; op-level capability
  fallback.
- Mirage MPK — why whole-model persistent/direct-MMA kernels pay for
  latency-bound inference on Hopper, and why they do not apply here.
