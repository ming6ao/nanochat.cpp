# Python API

Status: design. The process bridge in [python-bridge.md](python-bridge.md)
exists. This document specifies the in-process API, the build from Python, and
the Kaggle delivery.

The Python API lets a notebook train and evaluate `nanochat.cpp` without a
shell. A notebook holds a model, reads a live loss, and passes arrays. The API
uses a C application binary interface (ABI) and the Python `ctypes` module. It
does not use `pybind11` and it does not use a wheel. The package can also build
the shared library from the C++ sources on demand.

## 1. Goals and non-goals

Goals:

- Train and evaluate end to end in one Python process.
- Keep all compute in the existing C++ graph.
- Run on a Kaggle Notebook with two T4 cards.
- Add no runtime dependency to the C++ tree.
- Build the shared library from Python when a prebuilt library is absent.

Non-goals:

- A general tensor library.
- A second model definition in Python.
- A replacement for the process bridge.

## 2. Why a C interface and ctypes

The C++ model API in `include/nanochat/model.h` takes raw `int*` buffers and
returns `std::vector`. A thin C layer over it is small. The `ctypes` module is
part of the Python standard library, so the bridge needs no new build
dependency.

### 2.1 The alternative, pybind11

`pybind11` gives a shorter wrapper. It is a third-party build dependency.
`docs/python-bridge.md` rejects it, and this design keeps that rule.

### 2.2 The alternative, a Python extension

A CPython extension module binds one build to one Python interpreter. A plain
shared library does not. The Kaggle image changes its Python version between
images. One `libnanochat.so` works with every Python 3 build.

### 2.3 The alternative, a wheel

A wheel is not necessary. The API uses `ctypes`, so the library is not a Python
extension. Section 8 gives the delivery without a wheel.

A correct CUDA wheel must also vendor `libcudart` and `libcublas` and set a
run-time search path. That work duplicates the Kaggle image. The dataset
delivery avoids it.

### 2.4 The sandbox rule

The C++ entry points call `nanochat::RequireSandboxOrDie`. The library keeps
that guarantee. `nanochat_init` checks the same environment variables at start:

- The environment defines `NANOCHAT_SANDBOX`, or
- `NANOCHAT_SANDBOX_BACKEND` holds the value `none`, or
- `NANOCHAT_ALLOW_UNSANDBOXED` holds a true value.

The library stops with a corrective message when all three tests fail. A Kaggle
Notebook uses the `none` backend. A workstation run goes through
`tools/nanochat`.

## 3. Layers

```text
Notebook or user script
  nanochat_cpp.api        Model, Trainer, Evaluator, Tokenizer, TokenData
  nanochat_cpp._core      ctypes declarations; status -> Python exception
  nanochat_cpp._build     compile the library on demand
  libnanochat.so          C ABI; one handle per object
  //src:model             existing C++ graph (unchanged)
  backends/cpu | cuda     existing backends (unchanged)
```

The bindings workstream owns `bindings/` and the new Python modules. It does
not touch `src/**` or the frozen headers.

## 4. The C interface

The bindings use a C surface and a C++ implementation. The header
`bindings/nanochat_c.h` declares `extern "C"` functions. The file
`bindings/nanochat_capi.cc` defines them in C++. That file calls `Model`,
`Optimizer`, and `DataLoader`.

A C surface has four advantages:

- `ctypes` needs unmangled names. C++ names carry type information.
- A C++ exception must not cross into Python. The C layer catches each
  exception and returns a status.
- The C++ standard library interface changes between compilers and versions. A C
  surface hides `std::string` and `std::vector`.
- A C surface works with any Python 3 build and any language binding.

The header keeps the declarations inside a C linkage guard:

```c
#ifdef __cplusplus
extern "C" {
#endif

nanochat_status nanochat_init(void);
nanochat_model* nanochat_model_create(const nanochat_config*, uint64_t seed);

#ifdef __cplusplus
}
#endif
```

The bindings workstream owns the header. Opaque handles cross the boundary. No
C++ type crosses the boundary.

```c
nanochat_status nanochat_init(void);
const char*     nanochat_last_error(void);
const char*     nanochat_version(void);

typedef struct {
  int num_layers, num_heads, num_kv_heads, hidden_dim;
  int seq_len, vocab_size, padded_vocab_size;
  float rope_base;
  const char* window_pattern;
} nanochat_config;

nanochat_model* nanochat_model_create(const nanochat_config*, uint64_t seed);
void            nanochat_model_free(nanochat_model*);
float           nanochat_forward_loss(nanochat_model*, const int* tokens,
                                      const int* targets, int batch, int seq);
void            nanochat_backward(nanochat_model*);
void            nanochat_zero_grad(nanochat_model*);
int             nanochat_param_count(nanochat_model*);
int             nanochat_param_info(nanochat_model*, int index,
                                    nanochat_param* out);
void            nanochat_save(nanochat_model*, const char* path);
void            nanochat_load(nanochat_model*, const char* path);

nanochat_optim* nanochat_optim_create(nanochat_model*,
                                      const nanochat_optim_config*);
void            nanochat_optim_step(nanochat_optim*, int step);
float           nanochat_optim_grad_norm(nanochat_optim*);
void            nanochat_optim_free(nanochat_optim*);

nanochat_loader* nanochat_loader_create(const char** parquet, int count,
                                        const char* text_column,
                                        nanochat_tokenizer* tokenizer,
                                        int batch, int seq, uint64_t seed,
                                        int tokenizer_threads,
                                        size_t document_buffer);
int             nanochat_loader_next(nanochat_loader*, int* tokens,
                                     int* targets);
const uint8_t*  nanochat_loader_token_bytes(nanochat_loader*, int* vocab);
void            nanochat_loader_free(nanochat_loader*);

void  nanochat_score_batch(nanochat_model*, const int* tokens, int batch,
                           int seq, const int* lengths,
                           const nanochat_focus*,
                           nanochat_score_result* out);
void  nanochat_generate(nanochat_model*, const int* prompt, int length,
                        const nanochat_generate_params*,
                        nanochat_sequences* out);
float nanochat_eval_bpb(nanochat_model*, nanochat_loader*, int steps);

nanochat_tokenizer* nanochat_tokenizer_load(const char* path);
int  nanochat_encode(nanochat_tokenizer*, const char* text, int* out,
                     int capacity);
int  nanochat_decode(nanochat_tokenizer*, const int* ids, int count,
                     char* out, int capacity);
void nanochat_tokenizer_free(nanochat_tokenizer*);
```

### 4.1 Ownership

Each `create` function returns an owned handle. The Python wrapper releases the
handle in `__del__`. Each function that returns an array fills a caller buffer,
or returns an owned result with a matching `free` function.

### 4.2 Errors

Each function returns `nanochat_status`. The value `0` means success. A nonzero
value means failure. `nanochat_last_error` returns the message for the current
thread. The Python wrapper turns a nonzero status into a Python exception.

### 4.3 Data conversion

The C layer converts `nanochat_config` to `nanochat::Config` in one place. It
copies `ParamView`, `ScoreResult`, and `GeneratedSequence` to flat C structs.
The C++ headers stay frozen.

## 5. Build from Python

The package can compile the shared library on demand. This build is similar to
a just-in-time build. The Python module `nanochat_cpp._build` holds the logic.

### 5.1 When the build runs

`nanochat_cpp._lib.load` calls `_build.ensure_library`. The function follows
this order:

1. Use the path in `NANOCHAT_CPP_LIB` when the file exists.
2. Use a cached library when the build key matches.
3. Build the library.
4. Stop with a clear message when no compiler and no prebuilt library exist.

A build key is a hash of the source contents, the compiler version, the CUDA
architecture, the precision, and the backend. A change in one input gives a new
key.

### 5.2 Two builders

| Builder | Command | Use |
|---|---|---|
| `bazel` | `tools/nanochat build //bindings:nanochat_shared --config=<cfg>` | The canonical path. The flags match the tree. The default in a source checkout. |
| `cc` | `nvcc` and `g++` with a flag manifest | A tree without Bazel. For a small Kaggle dataset. |

The `bazel` builder keeps one source of truth for the flags. The `cc` builder
reads `bindings/flags.json` and needs a parity test, because it can drift.

### 5.3 Architecture and precision

The builder selects the target from the host:

- `NANOCHAT_CUDA_ARCH=sm_75` sets the architecture by hand.
- Otherwise the builder reads `nvidia-smi --query-gpu=compute_cap` and maps the
  result.
- A CPU request selects the reference backend and no architecture.

The precision comes from the configuration, in the same way as
`tools/nanochat`.

### 5.4 Cache

The builder writes the library under `NANOCHAT_CPP_CACHE`, then
`~/.cache/nanochat_cpp`. On Kaggle, set
`NANOCHAT_CPP_CACHE=/kaggle/working/.nanochat_cpp`. The library then survives
the session. A second run in the same session reuses the library.

### 5.5 Triggers

- `nanochat_cpp.build()` builds the library and reports the path.
- `nanochat_cpp.Model(...)` starts a build when no library exists, and prints
  one progress line.
- `NANOCHAT_CPP_PREBUILT=1` forbids a build. The load then stops when no
  library exists.

### 5.6 Why not cffi

`cffi` can compile C at run time. It adds a third-party runtime dependency.
`ctypes` plus a subprocess build does the same work with the standard library.

## 6. The Python API

The target notebook code:

```python
import nanochat_cpp as nc

tokenizer = nc.Tokenizer.load("/kaggle/input/nanochat-cpp/tokenizer.nctoken")
config = nc.Config(depth=8, seq_len=512, vocab_size=tokenizer.vocab_size)
model = nc.Model(config, device="cuda", seed=42)

data = nc.TokenData(parquet="/kaggle/input/nanochat-cpp/shard-*.parquet",
                    tokenizer=tokenizer, seq_len=512, threads=4)
trainer = nc.Trainer(model, data, batch=8, total_batch=4096, warmup_steps=3)
for step, loss in trainer:
    print(step, loss)

trainer.save("/kaggle/working/model.nchkpt")

report = nc.evaluate(model, parquet="/kaggle/input/nanochat-cpp/val-*.parquet",
                     tokenizer=tokenizer, tokens=40 * 524288)
print(report.bpb, report.core)

prompt = tokenizer.encode("The capital of France is")
row = model.generate(prompt, max_tokens=16, temperature=0.0)[0]
print(tokenizer.decode(row.tokens))
```

| Python object | C calls | Notes |
|---|---|---|
| `Tokenizer` | `nanochat_tokenizer_load`, `nanochat_encode`, `nanochat_decode` | `encode` takes `str` or a list. |
| `Config` | none | A dataclass. `depth` sets the shape. |
| `Model` | `nanochat_model_create`, `nanochat_forward_loss`, `nanochat_backward`, `nanochat_save`, `nanochat_load` | `params()` gives NumPy views. |
| `Optimizer` | `nanochat_optim_create`, `nanochat_optim_step`, `nanochat_optim_grad_norm` | Mirrors `OptimizerConfig`. |
| `TokenData` | `nanochat_loader_create`, `nanochat_loader_next` | Reads parquet documents and packs `int32` rows. |
| `Trainer` | the calls above | Iterable. Yields `(step, loss)`. |
| `Evaluator` | `nanochat_score_batch`, `nanochat_eval_bpb`, `nanochat_generate` | Returns dataclasses. |
| `no_grad` | none | A context manager over `SetGradEnabled`. |

The Python package does not import `torch`. This keeps the notebook light and
avoids a second CUDA context.

## 7. Data and tokenization

The C++ runtime reads parquet documents and tokenizes during the run
([parquet-native.md](parquet-native.md)). The native tokenizer in
`include/nanochat/tokenizer.h` removes the `torch` requirement. The API loads
the portable `NCTOKEN1` artifact. The format is in
[tokenizer.md](tokenizer.md) section 5.

`TokenData` names the parquet files and the tokenizer artifact. The loader
reads the documents and packs rows with the reference best-fit packer. The API
does not read parquet in Python and does not write a shard. A Kaggle Dataset
holds the parquet files and the artifact.

## 8. Delivery on Kaggle

The repository prefers a prebuilt shared library. A continuous integration job
builds it, and a Kaggle Dataset carries the result. When the prebuilt library
does not match the host, the package builds one from the sources. Section 5
gives the build.

### 8.1 Dataset layout

```text
/kaggle/input/nanochat-cpp/          (read-only, versioned)
  src/ include/ backends/ bindings/  C++ sources for a build from Python
  nanochat.bzl                       the shared compiler options
  python/nanochat_cpp/               the package
  lib/libnanochat_sm75_fp16.so       optional prebuilt library
  lib/libnanochat_sm75_fp32.so
  data/*.parquet                     the parquet documents
  tokenizer.nctoken                  the NCTOKEN1 artifact
  BUILD_INFO                         version, arch, precision, toolchain
```

### 8.2 Install cell

```python
import os, sys
root = "/kaggle/input/nanochat-cpp"
sys.path.insert(0, f"{root}/python")
os.environ["NANOCHAT_CPP_LIB"] = f"{root}/lib/libnanochat_sm75_fp16.so"
os.environ["NANOCHAT_CPP_CACHE"] = "/kaggle/working/.nanochat_cpp"
os.environ["NANOCHAT_SANDBOX_BACKEND"] = "none"
import nanochat_cpp as nc
```

The Kaggle image puts the CUDA libraries on the loader path. If a run fails to
find them, prepend `/usr/local/cuda/lib64` to `LD_LIBRARY_PATH`. Section 4.6 of
[host-portability.md](host-portability.md) already forwards this value for
tests.

### 8.3 Library discovery

`nanochat_cpp._lib` finds the shared library in this order:

1. The path in `NANOCHAT_CPP_LIB`.
2. A cached library that matches the build key.
3. The development tree under `bazel-bin/bindings/`.
4. The package directory, for a bundled library.
5. A build from the sources.

`nanochat_init` reads `BUILD_INFO` and compares the architecture to the device.
A mismatch starts a build, or stops the run with a clear message.

### 8.4 Why no wheel

A wheel adds a package manager step. The dataset is already a versioned
artifact. The `ctypes` design removes the Python version constraint. The
dataset also avoids wheel tags, vendored CUDA libraries, and a run-time search
path. Add a wheel later for a public package index. The package layout supports
that step without a code change.

## 9. GPU rules

The CUDA backend uses the default stream and synchronous host copies. Follow
these rules:

- Use one Python process for each card.
- Call `nanochat_init` before the first CUDA call.
- Do not create a `torch` CUDA tensor in the same process.
- Do not run two heavy jobs on one card.

Two T4 cards give two options. Option A runs two independent jobs, one on each
card. Option B is data-parallel training. Section 10 gives the design.

## 10. Multi-GPU training

Data-parallel training gives each rank a full model replica. Each rank reads a
different batch. The ranks all-reduce the gradients, then each rank steps its
optimizer. This is the distributed data parallel (DDP) schedule.

### 10.1 One process for each card

The CUDA backend keeps process-global buffers that a device index does not key:

- `backends/cuda/kernels/muon.cu` holds `g_muon_workspace` and `g_muon_scratch`.
- `backends/cuda/kernels/attention.cu` holds `static ComputeType* buffers[2]`.

Two threads on two cards would race on these buffers. `kernels::Alloc` calls
`cudaMalloc` on the current card, so a switch between cards would also alias the
cached buffers.

Use one process for each card. Each process has its own globals and its own
card. This rule matches the DDP launcher.

### 10.2 The loop

```python
# train_ddp.py
def main():
    dist = nc.dist.init()                # RANK, LOCAL_RANK, WORLD_SIZE
    model = nc.Model(config, device=dist.local_rank, seed=42)
    dist.broadcast_params(model)         # optional; seed 42 already matches
    data = nc.TokenData(parquet=PARQUET, tokenizer=tokenizer, seq_len=512,
                        seed=42 + dist.rank)
    optimizer = nc.Optimizer(model, **plan.optimizer)
    for step in range(plan.num_iterations):
        tokens, targets = data.next()
        loss = model.forward_loss(tokens, targets)
        model.backward()
        dist.all_reduce_grads(model)     # average across ranks
        optimizer.step(step + 1)
        if dist.rank == 0 and step % 10 == 0:
            print(step, loss)
    if dist.rank == 0:
        model.save("/kaggle/working/model.nchkpt")
    dist.shutdown()

nc.dist.run(main, nproc_per_node=2)
```

The all-reduce sits between `backward` and `optimizer.step`. Muon is safe. Every
rank gives it the same averaged gradient, so the Newton-Schulz iterations agree.

### 10.3 The launcher

`nc.dist.run` starts one process for each card. Each process reads `RANK`,
`LOCAL_RANK`, and `WORLD_SIZE`. Do not mask the cards with
`CUDA_VISIBLE_DEVICES`. Each process calls `cudaSetDevice(local_rank)` instead.

Rank 0 writes the NCCL unique ID to a rendezvous file. The other rank reads it.
The launcher passes the path in `NANOCHAT_DIST_RENDEZVOUS`.

A notebook kernel cannot use `multiprocessing` with the `spawn` start method.
Write the loop to a file, then launch it:

```text
%%writefile /kaggle/working/train_ddp.py
...
```

```python
nc.dist.run("/kaggle/working/train_ddp.py", nproc_per_node=2)
```

### 10.4 Required changes

| Layer | Change | Owner |
|---|---|---|
| `include/nanochat/kernels.h` | Additive `SetDevice(int)` and `DeviceCount()`. | Architect |
| `backends/cuda/device.cu` | `cudaSetDevice` and `cudaGetDeviceCount`. | Runtime |
| `backends/cpu/kernels.cc` | A no-op device and a count of one. | Runtime |
| `bindings/nanochat_c.h` | `nanochat_set_device` and the `nanochat_dist_*` group. | Bindings |
| `bindings/nanochat_dist.cc` | The NCCL communicator, `all_reduce_grads`, `broadcast_params`, `barrier`. | Bindings |
| `python/nanochat_cpp/dist.py` | The launcher, the environment, and the rendezvous. | Bindings |
| `MODULE.bazel` and the bindings BUILD file | The NCCL link dependency. | Build |

The new C surface:

```c
int  nanochat_device_count(void);
void nanochat_set_device(int index);

nanochat_dist* nanochat_dist_create(int rank, int world_size,
                                    const void* unique_id, size_t id_len);
void nanochat_dist_all_reduce_grads(nanochat_dist*, nanochat_model*);
void nanochat_dist_broadcast_params(nanochat_dist*, nanochat_model*);
void nanochat_dist_barrier(nanochat_dist*);
void nanochat_dist_free(nanochat_dist*);
```

`all_reduce_grads` walks `model->params()`. For each `ParamView` with a
gradient, it calls `ncclAllReduce` with `ncclSum` on the backend stream. It then
scales the result by `1 / world_size`. The first cut uses one call for each
tensor. Bucket the tensors later when the launch overhead shows up.

NCCL is the NVIDIA Collective Communications Library. It is a new third-party
runtime dependency. It ships with the Kaggle CUDA image. The project rule needs
architect sign-off for the dependency.

### 10.5 Weight and gradient rules

- Give every rank the same `seed` for `InitWeights`. The function is
  deterministic, so the replicas start equal.
- Give every rank a different data seed, or split the parquet files by rank.
- Keep the per-rank batch size equal. The average is correct.
- All-reduce after the full gradient accumulation, not after each micro-batch.
- Log and checkpoint on rank 0 only.

### 10.6 Testing

The workstation has one card, so multi-GPU runs happen only on Kaggle. Two
tests:

- **T0 on the CPU backend.** Two processes, a host-staged all-reduce over a
  socket, and a check that the averaged gradient equals a single-process run
  over the combined batch. This proves the loop and the launcher.
- **T2 on Kaggle with two T4 cards.** The same check with NCCL. Compare the
  first step and the weights after that step against a single-card run.

Add `//tests:ddp_parity_test` with the `gpu` tag. A two-process launch needs a
broker mode for a card pair. On Kaggle the sandbox is `none`, so the broker
stays out of the way.

### 10.7 The alternative without NCCL

A host-staged all-reduce needs no new dependency. Each step copies every
gradient to host memory, exchanges it over a socket, averages it, and copies it
back. The PCIe link is the limit. For a small model on two T4 cards this path
may be fast enough. Use it as the first implementation and the T0 gate. Add
NCCL for speed.

### 10.8 Phases

| Phase | Work | Gate |
|---|---|---|
| 5a | `SetDevice` and `DeviceCount`. Two independent jobs, one for each card. | Both cards train at once. |
| 5b | The host-staged all-reduce, the `dist` module, and the CPU parity test. | The T0 parity check passes. |
| 5c | The NCCL backend behind the same API. | The T2 parity check passes on two T4 cards. |
| 5d | Gradient bucketing and overlap with the backward pass. | A measured step-time win. |

## 11. Tests

| Tier | Test | Gate |
|---|---|---|
| T0 | A `sh_test` loads the CPU library and runs one forward step. | The loss matches `train_main` for a fixed seed. |
| T0 | A build from Python with the `cc` builder. | The result matches the Bazel build for one fixed-seed step. |
| T0 | A `sh_test` drives `Trainer` and `Evaluator` on a tiny document set. | The API matches the T1 fixture. |
| T1 | A tiny-shape CUDA run through `_core`. | Same result as the CPU reference. |
| T2 | A full train-parity run through the API. | The loss curve matches `//tests:train_parity`. |
| T0 | A two-process host-staged all-reduce on the CPU backend. | The averaged gradient matches a single-process run. |
| T2 | Two T4 cards with NCCL through `dist`. | The first step matches a single-card run. |

Use a `sh_test` wrapper for the Python tests. The tree does not declare
`rules_python`, and the existing `nanochat_cpp_selftest` target shows the
pattern.

## 12. Phases

| Phase | Deliverable | Proof |
|---|---|---|
| 0 | `bindings/nanochat_c.h` with `nanochat_init`, `nanochat_model_create`, and `nanochat_forward_loss`. A CPU shared library. | T0: `ctypes` loads the library and prints a loss. |
| 1 | The full C interface for the CPU backend. | T0: one fixed-seed step matches `train_main`. |
| 2 | `_core.py`, `_build.py`, and `api.py`. | T0: the API drives a tiny run. |
| 3 | The CUDA shared library, the device selection, and the `cc` builder. | T1 and T2 gates pass. |
| 4 | The dataset bundle, an example notebook, and this document. | A headless script runs the notebook cells. |

Each phase rebases on `main` before the merge request. Section 10.8 holds the
multi-GPU phases.

## 13. Risks

| Risk | Effect | Response |
|---|---|---|
| A CUDA context shared with `torch` | A driver or allocator fault | The API does not import `torch`. Document one CUDA user for each process. |
| Two T4 cards | The model is single-device | Use Option A now. Keep Option B separate. |
| ABI drift from `model.h` | A silent mismatch | A C++ test calls every function. Bump `nanochat_version` on each change. |
| A `cc` builder that drifts from Bazel | A wrong flag or a missing source | Keep a parity test for the `cc` build. |
| A cold build cache | A slow first cell | Keep the prebuilt library in the dataset. |
| A library for the wrong arch | A load failure | Read `BUILD_INFO` and compare at start. |
| The Kaggle session limit | A lost checkpoint | Save to `/kaggle/working`. Keep one small checkpoint. |

## 14. Open items

1. The header location. This document uses `bindings/nanochat_c.h`. The
   alternative is `include/nanochat/capi.h` under the architect.
2. The role of the process bridge. This document keeps it for sandboxed runs
   on the workstation.
3. The tokenizer source. This document bundles `tokenizer.nctoken` in the
   dataset.
4. The flag source for the `cc` builder. This document uses
   `bindings/flags.json`. A generated file from Bazel is the other option.
5. The NCCL link dependency. This document needs architect sign-off for the
   new third-party runtime dependency.
6. The broker mode for a card pair. A multi-GPU run needs both cards at once.
