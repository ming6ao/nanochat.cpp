# Testing

## Test tiers

| Tier | What | Cost | Scheduling |
|---|---|---|---|
| T0 CPU | reference kernels, workflow, oracle on CPU | ms–s | parallel, each under `t0-cpu` |
| T1 GPU correctness | tiny shapes (`B=2, T=8`), single kernel | < 100 ms | serialized via broker |
| T2 GPU parity | full oracle, loss curve, small training | seconds–minutes | serialized, exclusive |
| T3 GPU benchmark/profile | throughput, Nsight | minutes+ | scheduled, exclusive, native Linux |
| S0 simulator | device profiles, the reference engine, the emulated kernels, the API interposer, the collective mock | ms-s | CPU only, `t0-cpu`, no broker |

Rule: **the inner development loop is T0.** GPU tiers are gates, not iteration
tools. If you are waiting on the GPU to test logic, that logic should have been
tested on CPU. All tiers run under a resource profile; see
[sandbox.md](sandbox.md).

### The four conformance tiers

Four of the tiers above are the **conformance** tiers, and those four are the
Tier axis of the matrix in [conformance-matrix.md](conformance-matrix.md):

| Tier | Proves | Needs silicon |
|---|---|---|
| `T0 CPU` | The reference semantics of every graph, on the CPU backend | no |
| `S0 simulator` | A device profile's behaviour, through the reference engine, the emulated kernels, the API interposer, and the collective mock | no |
| `T1 GPU correctness` | One tiny shape per primitive (`B=2, T=8`) on the card | yes |
| `T2 GPU parity` | The full oracle, the loss curve, and a small training run, against the committed goldens | yes |

`T3 GPU benchmark/profile` is deliberately not among them: a throughput number
that moves is information, not a pass or a fail, so no matrix row is a
benchmark row.

**Tier per device, not tier per model.** This is the rule of
[model-and-device-bringup.md](model-and-device-bringup.md) §11, and it is what
keeps a table of models × devices × tiers from growing into a slow gate:

- `T0 CPU` for every model, always. This is the definition of conformant.
- `S0 simulator` for a declared device with no physical sample on this host.
- `T1 GPU correctness` for a device that has silicon here, at tiny shapes.
- `T2 GPU parity` only for the flagship model, at a wave boundary.

**Graph coverage is a column of every row.** A row names which of the five
graphs of [model.md](model.md) it proves. The minimum for a passing row is the
train step *and* decode; a complete row covers all five, and a row that skips a
graph names the one it skips. Without that column a model that runs its train
step and breaks in prefill looks green.

**The matrix is generated and checked, never maintained.** It is rendered by
`python3 tools/gen_matrix.py` from `models/*/spec.json` and
`backends/*/caps.json`, and a document is validated against the derivation with
`python3 tools/gen_matrix.py --check --matrix docs/conformance-matrix.md`. A
row for an undeclared model, a tier the tier rule does not produce, a duplicate
row, and a passing row that skips the train step or decode all fail, with the
offending label named. `//tests:matrix_gate_test` runs the check in the T0 loop,
so a hand-edited row breaks a gate instead of drifting, and
`tools/isolation_gate.py` is the companion gate of §10.4: it regenerates a
model's fixtures, diffs them against the committed bytes, and confirms every
other model is unchanged.

### S0: the simulator tier

S0 is the CPU-only tier that exercises H100-class behavior without a Hopper
device ([simulator.md](simulator.md)). It takes no GPU queue, so it runs
like T0 -- under the `t0-cpu` profile, in parallel with other T0 work. The
suites are:

```bash
# Everything: oracle, training parity, API, emulated kernels, profile
# consistency. Every run is fp32 (docs/precision.md).
tools/nanochat test --profile h100
tools/nanochat test --profile t4

# One suite.
tools/nanochat test --profile h100 --suite correctness   # reference engine
tools/nanochat test --profile h100 --suite api           # API interposer
tools/nanochat test --profile h100 --suite kernel        # emulated kernels
tools/nanochat test --profile h100 --suite gemm          # emulated tensor-core GEMM
tools/nanochat test --profile h100 --suite profile       # device table

# Compile the CUDA backend for the target architecture, and stop.
tools/nanochat build --config=cuda --@rules_cuda//cuda:archs=sm_90 \
    --define=precision=bf16 //backends/cuda:device_compile
```

The reference-engine suites add `--config=sim` and pass
`--test_env=NANOCHAT_SIM_PROFILE=<device>`, which makes `kernels::GetCaps()`
answer from `include/nanochat/device_profile.h` while every value is still
computed by the CPU reference loops. `//tests:sim_numerics_test` asserts that
the caps change and the numerics do not, so a profile can never silently move a
result.

The simulator runs the reference engine once, and every suite compares against
the committed fp32 fixtures. The numeric trace covers fp32 only
(`docs/testing.md` Part 2, section 6.2). The simulated device reports the
bf16 capability of its profile through `Caps::supports_bf16`
([precision.md](precision.md) section 6).

The collective mock rendezvouses through POSIX shared memory and bounds its
wait (`NANOCHAT_SIM_TIMEOUT_MS`), so a rank that never issues the all-reduce is
reported as a timeout instead of hanging the suite.

The native collective (`backends/cuda/collective/`, on every CUDA build) adds
two CPU-only tests to the default T0 loop, and one T1
GPU test. `//backends/cuda/collective:schedule_test` runs the ring at world
sizes 1, 2, 4, and 8 over the in-process memory transport, then judges the
schedule trace. `//backends/cuda/collective:transport_select_test` checks the
per-neighbor selection, including a mixed group.
`//backends/cuda/collective:nvlink_test` (tags `gpu`, `manual`) reduces across
two cards. On a host without NVLink peer access the T1 test exercises the
host-staged fallback. See
[distributed.md](distributed.md) Part 2, section 13.

The suite targets carry the `sim` tag. They are CPU-only and fast, so they stay
in the default `tools/nanochat test` loop; the tag exists so a merge gate can
select or exclude them by name. Two exceptions carry `manual`.
`//tests:cuda_sim_preload_test` links the real CUDA backend, and its own runner
supplies `LD_PRELOAD`, `NANOCHAT_SIM_LOG`, and the profile, so
`tools/nanochat test //tests:sim_api` reaches it. `//tests:numerics_trace_test`
replays a 10-layer trace, which is too slow for the inner loop; run it through
`tools/nanochat test --suite correctness` or by name.

The fabricated `cudaDeviceProp` layout is pinned by
`//tests:cuda_device_prop_abi_test`, a compile-time cross-check of
`cudaDevicePropPrefix` against the real `<driver_types.h>`; it needs the CUDA
toolkit headers but no device.

## How tests are sandboxed

`tools/nanochat test` runs Bazel with
`--run_under='python/nanochat_cpp/sandbox.py --profile=… --'`, so every test action gets its
own cgroup, and forces `--spawn_strategy=local` so the wrapper can reach the
systemd user manager. A GPU suite uses
`--run_under='tools/nanochat gpuq submit --profile=t1-gpu --'` instead: the
queue worker sandboxes each test action and owns the device
([sandbox.md](sandbox.md)). Sandboxing per test action means the
result does not depend on whether a Bazel server is already running. The
trade-off is that Bazel's own filesystem sandbox is disabled for these runs;
the resource sandbox is the priority. See [sandbox.md](sandbox.md).

One level up, a test run in a session worktree rides the session boundary. The
client and its Bazel server start inside `nanochat-agent-<id>.slice`, so the
build phase of the suite has a memory ceiling. Test actions still get their own
profile scope, so a run is bounded twice. The network is denied by default,
because the session supervisor holds one namespace and the suite then costs
what a normal run costs, 2.8 s measured. A session without a supervisor falls
back to the host network and says so ([sandbox.md](sandbox.md) section 19).

## Fixture generator commands

The offline fixture generator is one grouped script. Pass a fixture name after
the script name.

| Script | Fixture names |
|---|---|
| `tools/dump_fixtures.py` | `oracle`, `train`, `rl`, `generate`, `eval`, `plan`, `tokenizer`, `loader`, `chat`, `api`, `numerics`, `checkpoint`, `sft`, `sft_loop` |

For example, run `python3 tools/dump_fixtures.py oracle --out tests/data/debug_state.bin`.
The shared `tools/fixture_format.py` writer keeps the common record layouts stable,
and `tools/fixture_common.py` holds the shared helpers.
Each fixture kind keeps its own format and runtime requirements.

### The two SFT fixture kinds

`tools/dump_fixtures.py sft` writes four files, and
`tools/dump_fixtures.py sft_loop` writes the fifth. Both are tier T0.

`tests/data/sft_pack_fixture.bin` is `NANCHAT1` plus JSON: the eight
conversations, the reference render, the transcribed packer, the synthetic
mixture order, and the fixture tokenizer.

`tests/data/sft_step_fixture.bin` is a `NANOORC1` record file: the pinned
settings and the per-step loss, gradient norm, parameter L2, rate multiplier,
and Muon momentum of an eight-step reference trajectory.

`tests/data/sft_rows_fixture.nchsft` is the `NCSFT001` packed row shard that the
same packer produced.

`tests/data/sft_base_fixture.ckpt` is the `NCHKPT01` checkpoint with the
reference parameters and the reference optimizer moments at step 0.

`tests/data/sft_loop_fixture.bin` is a second `NANOORC1` file: the progress
stream, the rate multiplier, the momentum, the epoch, the loss curve, and the
packed bits-per-byte of the same eight steps.

Both kinds are hermetic. They read a reference checkout to generate the bytes
and never again afterwards: every gate that consumes a fixture imports no
torch, opens no network socket, and needs no tokenizer artifact. `sft_loop`
reads the shard, the base checkpoint, and the pack fixture, so it runs second.
The pinned settings live in one module, `tools/sft_fixture.py`, which both kinds
import, so the step and the loop fixtures cannot disagree.

The layout of every record, its dtype, its shape, the pinned settings, and the
SHA-256 of each file are recorded in `tests/data/sft_fixtures.txt`, because a
binary cannot carry a header comment. `docs/parity.md` Part 1, section 6 is the
design. Both kinds print the digest of each file they write, so a re-pin is
copied from the generator's output instead of derived a second time.

## Oracle fixtures

`tests/debug_state.bin` is produced offline by nanochat's `gpt.py` (see
[model.md](model.md)). `tests/oracle_test.cc` compares logits, loss, and
gradients, then runs a few optimizer steps and matches losses. Tolerances are
per backend: fp32 CUDA ~1e-5.

The multi-step trajectory is a strict gate (loss `1e-5`, parameter `1e-4`).
`tools/dump_fixtures.py oracle` records the AdamW `eps` it used as
`config/opt/adam_eps` (currently `1e-4`), and the C++ test matches it. Several
fixture parameters have a true gradient of exactly zero at this initialization.
At nanochat's default `eps=1e-10` their fp32 roundoff would be amplified into
unreproducible updates. Regenerate the fixture with
`python tools/dump_fixtures.py oracle --adam-eps 1e-4`.

Fixtures are data, so GPU tests never need torch at runtime. Use fixed seeds.

## Training-parity harness

`tools/dump_fixtures.py train` records a multi-step *training trajectory* from
nanochat's PyTorch `gpt.py`. It holds the initial parameters and the exact
token/target batch at every step. After every step it holds the loss, the
global gradient norm, and the L2 norm of every parameter (and optionally the
final parameter tensors).
`tests/train_parity_test.cc` drives `nanochat.cpp`'s `TrainStep` from the same
initial parameters over the same batches with the same optimizer and
learning-rate schedule, and compares the loss, gradient, and parameter
trajectories. This is the scaled-up version of the oracle trajectory gate. It
catches a bug that only appears after the first update, when the
zero-initialized output projections activate the attention and MLP backward.

The committed fixture `tests/data/train_parity.bin` is the compact CPU case,
with `--nonzero-projections`, so the whole graph is active at step 0. It runs as
`//tests:train_parity_test` (CPU) and `//tests:train_parity_cuda_test` (GPU).
Because the fixture shares the exact batches, this gate covers the model,
backward pass, optimizer, and schedules. It does **not** cover the data
pipeline, and the loader difference is entry D1 in [parity.md](parity.md).
The same source builds as the standalone `//tests:train_parity` binary for an
arbitrary production fixture:

```bash
python3 tools/dump_fixtures.py train --out /tmp/train_parity_d8_s512_50.bin \
    --layers 8 --heads 4 --kv-heads 4 --embd 512 --vocab 32768 --pad-to 32768 \
    --seq 512 --batch 8 --steps 50 --window L --adam-eps 1e-4
tools/nanochat gpu --profile t2-parity -- \
    ./bazel-bin/tests/train_parity /tmp/train_parity_d8_s512_50.bin
```

A 50-step run at `d8_s512` matches the reference: loss `6e-6`, parameter L2
`1.7e-3`, gradient L2 `4.3e-5`.

`TRAIN_PARITY_STOP=1` stops at the first divergent step, which is useful for
bisecting a shape- or schedule-dependent bug. Like the oracle, the fixture is
generated on the CPU with torch and the runtime tests never import torch.

## SFT parity gates

The supervised-fine-tuning rungs of `docs/parity.md` Part 1, section 3 have one
gate each. Every one compares one backend against one committed fixture, as
`docs/parity.md` requires.

| Gate | Tier | Fixture | What it compares |
|---|---|---|---|
| `//python:sft_data_parity_test` | T0 CPU | `sft_pack_fixture.bin` | The render ids and mask, the packed rows, the mixture order, and the two `stop` values |
| `//tests:sft_fixture_load_test` | T0 CPU | All five `sft_*` files | That the containers, the shard, the checkpoint, and the pack JSON all read |
| `//tests:sft_step_parity_test` | T0 CPU | `sft_step_fixture.bin`, `sft_base_fixture.ckpt` | The per-step loss, gradient norm, parameter L2, rate, and momentum |
| `//tests:sft_step_parity_gpu_test` | T1 GPU | The same two files | The same eight-step trajectory on the CUDA backend |
| `//tests:sft_step_parity` | T2 | A fixture named on the command line | A larger trajectory through the standalone binary |
| `//tests:sft_loop_parity_test` | T0 CPU | `sft_loop_fixture.bin`, `sft_step_fixture.bin` | `LrMultiplierForProgress`, the momentum, the epoch, and the packed bits-per-byte |
| `//tests:sft_parity_test` | T2 GPU | The loop, step, rows, and base fixtures | The real `nanochat_cli sft` output lines and the checkpoint it writes |

Every fixture name above is under `tests/data/`. The gates compare against the
recorded values, not against each other:

- `//python:sft_data_parity_test` compares `render_conversation` and `pack_rows`
  with the transcribed reference packer, and `TaskMixture` with the recorded
  seed-42 order.
- `//tests:sft_fixture_load_test` proves the fixtures load on this host, with no
  reference checkout, no torch, and no tokenizer artifact.
- `//tests:sft_step_parity_test` drives `ForwardLoss`, `BackwardWeighted`, and
  `Optimizer::Step` for eight steps, and then compares every final parameter.
- `//tests:sft_loop_parity_test` recomputes the rate multiplier from the
  recorded progress and the momentum from the step, then compares the two
  against the trace and reproduces the recorded bits-per-byte.
- `//tests:sft_parity_test` runs the real binary, so it pins the flags, the
  accumulation, the progress stream, the evaluation cadence, and the `Recipe`
  block of the checkpoint it writes.

`//tests:sft_parity_test` carries the `gpu` and `manual` tags, so the default T0
loop and the wildcard GPU sweep both skip it. Run it by name at a wave boundary.
`//tests:sft_step_parity_gpu_test` carries the same two tags.

State of the gates, as of the last recorded run in this tree:

| Gate | Result |
|---|---|
| `//python:sft_data_parity_test` | Passed at T0, 9 test cases |
| `//tests:sft_fixture_load_test` | Passed at T0 |
| `//tests:sft_step_parity_test` | Passed at T0; the measured maxima are printed per run |
| `//tests:sft_loop_parity_test` | Passed at T0; the maximum schedule difference is `5.96e-08` |
| `//tests:sft_step_parity_gpu_test` | Not run on this host |
| `//tests:sft_parity_test` | Not run on this host |
| `//tests:sft_step_parity` | Not run on this host |

A GPU tier that did not run is not evidence. `docs/parity.md` records `P5` for
the CPU backend only, and it leaves `P6` and `P7` open.

### The bounds these gates read

The first three bounds are the pretraining numbers. The SFT step trajectory
reads a wider class, and the reason is one setting: section 5.1 of
`docs/parity.md` pins `adam_eps = 1e-10`, the reference value. A
near-zero gradient then divides its moment by `sqrt(v) + 1e-10`, so the last
bits of a gradient that both sides compute in a different order become a
full-size step. The pretraining oracle fixture avoids this by pinning `1e-4`
(`tools/dump_fixtures.py oracle`); the SFT recipe cannot.

| Quantity | Bound | Where it applies |
|---|---|---|
| Loss | `1e-5` absolute and relative | The pretraining fixture (`//tests:oracle_test`, `//tests:train_parity_test`) |
| Parameter L2 | `1e-4` relative | The pretraining trajectory (`//tests:train_parity_test`) |
| fp32 CUDA | about `1e-5` relative | The GPU leg of a pretraining parity gate (`//tests:train_parity_cuda_test`) |
| Schedule values (`lr`, momentum, progress) | `1e-6` absolute and relative | Every SFT gate. These are exact settings, so only the fp32 round trip separates the two sides |
| SFT loss, per step | `2e-3` absolute and relative | `//tests:sft_step_parity_test`, `//tests:sft_parity_test` |
| SFT gradient norm, per step | `1e-3` absolute, `2e-2` relative | The same two gates |
| SFT parameter L2, per step | `1e-3` absolute, `5e-3` relative | The same two gates |
| SFT validation loss, against the train curve | `5e-2` relative | `//tests:sft_loop_parity_test`: the evaluation reads every row while `train/loss/<k>` logs one micro-batch |

`//tests:sft_step_parity_test` prints its measured maxima on every run, so a
drift is visible even when it passes. The measured values of the recorded run
are `9.05e-4` for the loss, `8.4e-3` for the gradient norm, `1.56e-1` absolute
(`1.3e-3` relative) for the parameter L2, and `5.96e-8` for the schedule.

The total parameter L2 is the gate, and the per-parameter norms are diagnostics.
A few scalars do disagree by percent-scale relative differences
(`x0_lambdas` `4.2e-1`, `backout_lambda` at a norm of `0.058` against `0.404`),
because the reference gradient of those parameters is itself roundoff at this
scale. A wrong rate, a wrong epsilon, or a wrong parameter grouping moves the
total parameter L2 by percent, not by `1e-3`: the same trajectory at
`adam_eps = 1e-4` misses the first step's parameter L2 by 5 percent.

## Tokenizer tests

The tokenizer, the trainer, and the parquet reader are host
utilities. Every test is tier T0. The tests need no GPU and no broker. The exact
commands are:

```bash
# The tokenizer package: round trip, splitter, reference parity, trainer,
# reader. The suite label runs all five targets.
tools/nanochat test //src/tokenizer:all

# The `LoadTokenizer` contract and the `DataLoader` shift.
tools/nanochat test //src:data_test

# The `NCTOKEN1` reader and merge reconstruction, torch-free.
tools/nanochat test //python:data_test

# The whole CPU suite and the style gate.
tools/nanochat test
tools/nanochat lint
```

`//src/tokenizer:tokenizer_parity_test` and `//src/tokenizer:bpe_trainer_test`
read the committed fixture `tests/data/tokenizer_fixture.bin`, so they never
import Python. The native command lines build and run through the sandbox:

```bash
tools/nanochat build //src/tokenizer:tok_train_main
tools/nanochat run t0-cpu -- <tok_train_main> --text corpus.txt \
    --vocab-size 512 --out /tmp/tokenizer.nctoken
tools/nanochat run t0-cpu -- <tok_train_main> --parquet 'data/*.parquet' \
    --vocab-size 512 --max-chars 2000000 --out /tmp/tokenizer.nctoken
```

The Python tests need the system `python3` only. A full run needs the reference
virtual environment (torch, pyarrow) and the parquet dataset; see
[python.md](python.md).

## Python API tests

The Python surface ([python.md](python.md)) has one `py_test` per module and one
C surface test. Every target is tier T0. The commands are:

```bash
# The C surface: every function in `nanochat/capi.h`, the sandbox rule, and
# the owned handles.
tools/nanochat test //bindings:nanochat_capi_test

# The ctypes core loads the CPU library and runs one forward/backward step.
tools/nanochat test //python:core_test

# The on-demand builder key and the library search order. Hermetic: it never
# compiles.
tools/nanochat test //python:build_test

# The `Trainer` loss curve and the `Evaluator` bits-per-byte result match the
# committed `//tests:api_fixture`.
tools/nanochat test //python:api_test

# The planning layer against `//tests:plan_fixture`, the chat prompt renderer
# and the per-task isolation, the orchestration argv, the checkpoint logic, the
# evaluation fixture, and the `NCTOKEN1` reader.
tools/nanochat test //python:plan_test //python:chat_test \
    //python:toolchain_test //python:checkpoint_test \
    //python:eval_fixture_test //python:data_test
```

`//python:all` runs every Python target. `//bindings:all` runs the C surface
test. Both ride `//:all_tests` and the default `tools/nanochat test`.

The notebooks directory adds two T0 tests:
`//notebooks:args_test` for the model and training argument data classes, and
`//notebooks:kaggle_setup_test` for the host-contract parser. `//notebooks:all`
runs both. They ride `//:all_tests` and the default `tools/nanochat test`.

The Python targets use the local `python3` toolchain from `rules_python`, so
they need no third-party wheel. `tools/dump_fixtures.py api`,
`tools/dump_fixtures.py plan`, and `tools/dump_fixtures.py chat` produce the
committed fixtures offline.

## Finite-difference checks

Kernel families with a backward are checked with a finite-difference gradient
test next to the kernel in `backends/cuda/kernels/`. The fused backward is
derived and tested as a unit; see [model.md](model.md) § No autograd engine.
`backends/cuda/kernels/README.md` describes the per-kernel test and benchmark
convention.

## Bazel tagging

```python
cc_test(
    name = "rms_norm_gpu_test",
    srcs = ["rms_norm_test.cc"],
    tags = ["gpu"],   # excluded from the default CPU loop
    ...
)
```

- `-gpu` filter for the fast default test loop.
- `--local_test_jobs=1` for GPU runs.
- `tools/nanochat test --gpu //...` discovers every `gpu`-tagged test. Do not
  add the `manual` tag to a test that this command must discover, because Bazel
  excludes `manual` targets from wildcard expansion.

---

## Part 1. Correctness gates: the simulator and the Hopper startup check

_Absorbed from the former `correctness-gates.md`. Its section numbers are
unchanged, so a reference to a section of that document resolves to the same
number here._

This part states the correctness gates for the simulator and the Hopper startup
check. Parts 1 to 6 are in [simulator.md](simulator.md). Part
7 is [simulator.md](simulator.md).

This document answers three questions:

1. Which part of the fp8 and bf16 correctness can the simulator prove without a
   GPU?
2. How does the project prove the remaining part on a real Hopper card?
3. How does the project keep the simulator fast?

The reference designs are [precision.md](precision.md),
[precision.md](precision.md), [testing.md](testing.md),
and [attention.md](attention.md).

### 1. The split

The simulator is the S0 tier. It is CPU-only and it takes no GPU queue
([testing.md](testing.md) section "S0: the simulator tier"). The simulator
computes real values on the CPU reference backend. It cannot compute a
tensor-core result, because the CPU has no tensor core.

The plan therefore has two halves:

- **The first part** extends the simulator, so the off-GPU proof is as complete as
  possible.
- **The second part** adds a fast hardware check at the start of a training run.

The third part keeps the simulator fast.

### 2. The coverage boundary

The table states which surface proves which claim. The last column names the
only gate that can prove the claim.

| Surface | Simulator after the extension | Hopper startup gate |
|---|---|---|
| fp8 rounding, E4M3 and E5M2 | Exact, through a shared header | Confirm exact |
| The amax, scale, and cast kernel | Emulator against the CPU reference | Confirm |
| The scaled GEMM, forward | The CPU reference value only | The real value |
| The scaled GEMM, dgrad and wgrad | The CPU reference value only | The real value |
| The fp8 layout, three GEMMs | Named, not proven | Proven |
| The `fast_accum` flag | The call shape only | Proven |
| The fp8 layer set | Exact on the graph | Confirm |
| The graph: loss, gradients, optimizer | Exact against the CPU golden | Confirm within 1e-2 |
| The tensor-core attention tile | Not possible | Proven |
| The `cublasLt` call shape | Interposer | Confirm on the device |
| The arena and the workspace size | The size model only | Proven |
| The collective numerics | The host reference, exact | Confirm on NVLink |

Two limits are load bearing. `simulator.md` Part 7, section 5.3 states that
the interposer proves the call shape and computes nothing. `simulator.md`
section 14 states that only hardware proves a tensor-core layout.

### 3. Extend the simulator

This part follows phases P0 to P6 of
[simulator.md](simulator.md). Each subsection names the
deliverable and the gate.

#### 3.1 The profile and the capability fields

The `peak_fp8_flops` field already exists at
`include/nanochat/device_profile.h:47`. Add the rest:

- `supports_fp8` to `DeviceProfile`.
- `supports_fp8` to `Caps`, and an fp8 branch in `Caps::Supports`
  (`include/nanochat/tensor.h:199`).

Extend `//tests:device_profile_test` with the assertions in
[simulator.md](simulator.md) Part 7, section 3. Two assertions hold
today: a nonzero fp8 rate matches the flag, and an fp8 device reports compute
capability 8.9 or higher. The third assertion, the startup rejection of an fp8
build on a device without support, needs the fp8 build, so it lands in A3.

Gate: the test passes.

#### 3.2 The shared rounding header

The header is `include/nanochat/fp8.h`. It sits under `include/nanochat/`, so
the CPU backend and the emulation engine both reach it below the seam. This
resolves open question 1 of [simulator.md](simulator.md).

The header holds the encoding, the decoding, and the round trip for both
formats:

- `Fp8E4M3Encode`, `Fp8E4M3Decode`, `Fp8E4M3Round`;
- `Fp8E5M2Encode`, `Fp8E5M2Decode`, `Fp8E5M2Round`.

Both formats round to nearest with ties to even. Both saturate at the format
maximum, 448 for E4M3 and 57344 for E5M2. The emulation engine needs the encode
and the decode for the CUDA conversion intrinsics
([simulator.md](simulator.md) Part 7, section 6.1).

`simulator.md` Part 7, invariant 2 requires one implementation. Two
implementations would break the exact simulator leg.

The unit test is `//tests:fp8_rounding_test`. It pins the hand-computed
encodings, the round trip over every finite code, the tie rule against an
independent direct scan, saturation, the infinity, the NaN, and the sign bit.

Gate: the test passes. It passes today.

#### 3.3 The fp8 graph in the reference engine

Add the storage tags and the scaled GEMM seam
([precision.md](precision.md) sections 3 and 5). Implement
`GemmScaledFp8` in `backends/cpu/kernels.cc`. The implementation quantizes,
accumulates in fp32, and dequantizes.

The seam and both engines are present, except the operand reuse across calls. The seam, both
backends, the build config, and the model plumbing are in. `--config=fp8` and
`test --config=fp8` select the fp8 build. `src/ops.cc` quantizes the
eligible forward and backward matmuls, and `src/model.cc` owns the workspace
slots (one byte per operand, one fp32 per scale, rebuilt per step).
`//tests:fp8_graph_test` compares the fp8 forward and backward against the dense
path at an eligible shape.

The operand reuse, rules 2 to 4 of [precision.md](precision.md)
section 7, is the remaining follow-up: the code quantizes a tensor again even
when the same step already quantized it. The result is correct; the extra casts
are a cost only.

Gate:

```bash
tools/nanochat test --profile h100 --suite correctness --config=fp8
```

The run must reach the committed fp8 golden. The golden is the CPU emitter's
output (section 3.6), so the CPU and simulator legs hold today. The device leg
waits for a Hopper measurement.

#### 3.4 The `cublasLt` coverage

The fp8 path calls `cublasLtMatmul`. Add the 15 symbols in
[simulator.md](simulator.md) Part 7, section 5.1 to
`tools/cuda_sim/cuda_sim.map` under the `libcublasLt.so.12` node. Add the eight
validation rules in section 5.2.

The map exports the family, the mock implements it in
`tools/cuda_sim/interposer.cc`, and the ABI lives in
`tools/cuda_sim/cuda_sim_abi.h`. `//tests:cuda_sim_preload_test` issues a scaled
GEMM through the real `gemm.cu` and asserts the recorded call shape.
`check_symbols.py` resolves 57 required symbols.

Gate: `tools/cuda_sim/check_symbols.py` resolves every symbol, and the negative
case still fails. Both hold today.

#### 3.5 The emulation engine

Add `tools/cuda_sim/emu/cuda_fp8.h` with the six intrinsics in
[simulator.md](simulator.md) Part 7, section 6.1. Each intrinsic
uses the shared rounding header.

Map `extern __shared__`. This mapping unblocks the value gate and the fused
attention forward. Map `mma` to a scalar loop for one narrow case only. Record
in [attention.md](attention.md) Part 6, section 6, that the
tile still needs hardware.

The tree holds it except the `mma` scalar loop. `cuda_fp8.h` maps the conversion
intrinsics to `nanochat/fp8.h`. `cuda_emu.h` carries a per-block dynamic shared
buffer, and the BUILD file rewrites the `extern __shared__` declaration in the
emulated copy of `value_gate.cu` and `attention.cu`. The device sources do not
change; only the copies do.

`//tests:kernel_emu_test` compares the emulated cast, the value gate, and the
attention forward (the cuBLAS path and the fused windowed path) against the CPU
reference. It also asserts that the dynamic shared path ran.

The `mma` scalar loop for fp8 stays open: that instruction needs hardware
([attention.md](attention.md) Part 6, section 6, and
[tensor-core-gemm-plan.md](tensor-core-gemm-plan.md) decision D3). `mma.h` itself
is mapped, and `//tools/cuda_sim/emu:gemm_emu_test` is the leg that uses it: it
runs the hand bf16 GEMM of `backends/cuda/gemm/` through the block runtime over
the four operand layouts, a ragged shape with a k tail, a batch, both stores, and
every schedule of the sweep. It sits in its own target because the tile is bf16
by construction and this test is fp32, and `tools/nanochat test --suite gemm`
runs it.

Gate: the emulated cast, the value gate, and the attention forward match the
CPU reference. All three hold today.

#### 3.6 The fp8 trace leg

The fixture carries an fp8 section beside the fp32 and bf16 sections, so the fp8
build compares the discrete records byte for byte. The section is produced by the
CPU emitter under `--config=fp8`, through `tools/dump_fixtures.py numerics`.

The fixture is the **CPU emitter's output**, not a PyTorch fixture. The trace
compares the discrete results across targets; the PyTorch parity gates are the
oracle and the training parity. The floating-point fp8 bound is
[simulator.md](simulator.md) Part 7, section 7, applied by the fp8
parity tests, not by this gate.

The tie rule of [testing.md](testing.md) Part 2, section 6.4
guards the fp32 leg. A bf16 or fp8 build has ties at the bf16 resolution. The
test prints the margin and does not fail on it for a bf16 or fp8 build. The
greedy identifiers are still compared exactly against the fp8 section.

Gate: every leg compares its own section byte for byte.

### 4. The Hopper startup gate

#### 4.1 The goal

Prove, once per fresh run on the real card, the four things that the simulator
cannot. Abort before the first optimizer step on a mismatch. Do not fall back to
bf16, because [../README.md](../README.md) § What this is not forbids a runtime
precision branch.

`src/hw_gate.cc` implements the cases and a host reference.
`src/train.cc` runs it before the first step when the device is real, and exits
with code 2 on a mismatch. `nanochat_cli train --hw-check {auto,on,off}` selects
the
behaviour. `//tests:hw_gate_cuda_test` is the standalone twin. There is no
stamp, by decision: the gate always runs on a fresh run.

#### 4.2 The shape of the gate

Write one function, `RunHardwareCorrectnessGate`, in `src/hw_gate.cc` with its
header in `src/hw_gate.h`. The Harness workstream owns `src/`. The same source
also builds as a standalone test target, so the project tests the logic once.

The gate uses the real workspace and the real seam calls. Every case uses a tiny
shape: `B=2`, `T=8`, `hidden=64`, `vocab=128`.

| Step | Check | Bound |
|---|---|---|
| 1 | The device fp8 conversion against the shared rounding table | Exact |
| 2 | The scaled GEMM, forward, E4M3, `fast_accum` on | 1e-2 relative |
| 3 | The scaled GEMM, dgrad, E5M2, `fast_accum` off | 1e-2 relative |
| 4 | The scaled GEMM, wgrad, E5M2, `fast_accum` off | 1e-2 relative |
| 5 | The three layouts on a non-square matrix | 1e-2 relative |
| 6 | The `fast_accum` pair: the flag changes the result, and both stay inside the bound | 1e-2 relative |
| 7 | The attention forward and backward at one small shape | 1e-2 relative |
| 8 | A two-step training step: the loss and the gradient norm | 1e-2 relative |

Step 5 uses a non-square matrix, so a wrong stride shows. Step 6 is the
fast-accumulation contract. Step 7 is the tensor-core attention. Step 8 ties the
graph to the device.

The golden values come from the CPU reference with a fixed seed. They live in
`src/hw_gate.cc`, not in a committed fixture, so the gate needs no torch at run
time.

#### 4.3 The integration into training

`nanochat_cli train` gains `--hw-check {auto,on,off}`. The default is `auto`.

`auto` runs the gate when three conditions hold:

- the build has an fp8 or a bf16 path;
- the device is Hopper-class;
- the run is not a resume.

`auto` skips the gate on a resume when the checkpoint carries a passing stamp
for the same commit and the same device.

Put the call in `src/train.cc` inside `Train()`. Put it after the model and the
workspace exist, and before the first step. The analytic bound report at
`src/train.cc:973` is the place to put it beside.

On a mismatch, the gate logs the failing case with the two values. The process
then exits with code 2.

The gate writes three fields into the checkpoint: `hw_check/commit`,
`hw_check/device`, and `hw_check/result`.

The gate is in-process. It reuses the CUDA context and the `cublasLt` handle
that training needs. A separate process would pay a second context setup.

#### 4.4 The cost budget

| Item | Cost |
|---|---|
| The CUDA context and the handle | Already paid by training |
| The first `cublasLt` heuristic query | 0.1 to 0.5 s |
| The conversion table | Less than 1 ms |
| The three GEMMs and the layouts | Less than 10 ms |
| The small attention forward and backward | Less than 20 ms |
| The two training steps at depth 2 | About 50 ms |
| **Total added to a fresh run** | **Less than 1 s** |

The gate runs once per fresh run. The cost is far below one training step at the
production shape.

#### 4.5 The warmup variant

The learning-rate warmup is a schedule, not a correctness phase
(`src/optim.cc:638`). Keep it separate.

If the operator wants a live check, add `--hw-check=warmup`. That mode logs the
loss and the gradient norm of the first `N` steps. The operator compares them
against the golden offline. Keep it off by default, because it costs the
production step time.

#### 4.6 The continuous integration twin

Add `//tests:hw_gate_cuda_test` with the tags `gpu` and `manual`. The target
links the same `src/hw_gate.cc`. Run it through
`tools/nanochat test --gpu`. It is the same check, without a training run around
it.

#### 4.7 The fault injection

The gate needs a way to prove that it fails. Add a debug flag,
`--hw-check-inject=case`, that corrupts one golden value. The test asserts the
exit code 2.

### 5. Keep the simulator fast

#### 5.1 The suite split

Do not put the fp8 work into the slow `correctness` suite alone. Split the work
by cost.

| Suite | Contents | Target | When |
|---|---|---|---|
| `fp8-fast` | The rounding unit, the seam GEMM at a tiny shape, the profile rows, the shared-header equality | Less than 2 s | Every edit, in the default T0 loop |
| `correctness` | The oracle, the training parity, the fp32 trace | The current cost | Per commit |
| `fp8-full` | The fp8 oracle, the fp8 training parity, the fp8 trace at a tiny shape, the emulated cast kernel | Less than 60 s | Merge gate |
| `api` | The `cublasLt` call shape, the symbol gate | Less than 30 s | Merge gate only |
| `compile` | `sm_90` under `-Werror` | The build time | Merge gate only |

#### 5.2 The speed measures

1. **One process per suite.** Build the model once and run every fp8 case in it.
   The current `correctness` suite runs four separate binaries.
2. **Tiny shapes in the fast suite.** Use `B=2, T=8, hidden=64, vocab=128`. The
   trace config in [testing.md](testing.md) section
   6.2 is 10 layers, hidden 384, sequence 256, and batch 4. That config is too
   slow for the inner loop, so it stays `manual`.
3. **Committed, exact goldens.** Compare the discrete results byte for byte.
   The trace holds the exact class only, so it needs no hash and no tolerance.
4. **No torch at run time.** The fixtures are data.
5. **Pin the reference thread count to one.** The CPU reductions depend on the
   thread count. The pin also removes contention.
6. **Do not rebuild the CUDA backend for a CPU-only suite.** Only the `api` and
   `compile` suites need it.
7. **Run under `t0-cpu` at 200 percent with `--local_test_jobs=2`.** Two
   CPU-bound jobs run at once. The aggregate slice still bounds the host.
8. **Emulate only what the gate needs.** Map `extern __shared__` and one scalar `mma`
   case. Do not emulate the attention at the production shape.
9. **Record the time.** Print the wall time per suite from the profiled run, so
   a regression is visible.

#### 5.3 The reason this stays fast

The reference engine computes every value on the CPU. Its cost scales with the
shape, the number of steps, and the number of layers. The fast suite pays none
of that, because it checks the rounding and one tiny GEMM. The heavy graph
checks move to the merge gate.

## Part 2. The numerics integration test

_Absorbed from the former `numerics-integration.md`. Its section numbers are
unchanged, so a reference to a section of that document resolves to the same
number here._

This document defines an integration test for numeric agreement across three
targets. The three targets are the CPU backend, the CUDA simulator, and a
CUDA card. The test uses a 10-layer model. It covers training, evaluation,
supervised fine-tuning (SFT), and reinforcement learning (RL).

The test follows Path A. Path A is a staged precision contract. This document
also records what Path A does not do.

The trace library, the emitter, the two Bazel targets, the fixture, the
simulator wiring, and the merge-gate entry are present, and the fixture holds
one section per precision. The optional `--trace` option of the harness is not
implemented.

### 1. The reduction, stated first

Path A does not compare floating-point results across targets. It compares the
results that two arithmetic implementations can agree on. State them before the
design.

1. Every leg compares the fixture byte for byte. There is one comparison mode,
   and a difference is a failure.
2. The fixture holds the **exact class** only: configuration, inputs, token ids,
   masks, greedy identifiers, argmax, and valid-target counts. Every one of them
   is an integer or a fixed constant.
3. The fixture carries one section per precision (`fp32/`, `bf16/`, `fp8/`), and
   each build compares the section that names its own precision.
4. The caps invariant is exact, and `//tests:sim_numerics_test` proves it in one
   process. That test changes `NANOCHAT_SIM_PROFILE` and requires every recorded
   value to be bit-identical. It is the guard that no numeric path reads
   `GetCaps()`.
5. Floating-point results are not records. Their contract is the per-stage
   parity tests of [testing.md](testing.md), each with one bound and one
   committed fixture.
6. The device leg repeats across two runs and compares the discrete records
   again. The device kernels use atomic scatter-add operations, so a byte order
   is not a contract, but a discrete result is.

The status table in [post-training.md](post-training.md) marks the SFT renderer,
the SFT packer, the SFT loop, the `rl_step` binary, and the RL fixture as
**Done**. This trace gives kernel-level coverage of the discrete results only.
The SFT arithmetic evidence is `//tests:sft_step_parity_test`, not this test.

### 2. What the recon established

Five facts shape the design.

1. The CPU backend uses `double` accumulators. See the GEMM microkernel in
   `backends/cpu/kernels.cc`. The CUDA kernels use `float` accumulators, and
   GEMM calls cuBLAS. [parity.md](parity.md) records this difference as `D2`
   and `D3` with status `equivalent`.
2. Under `--config=sim`, the CPU loops compute every value. The cap override is
   the only change. `tests/sim_numerics_test.cc` asserts the invariant. The
   "simulator" leg is therefore a cap-override regression guard. It is not a
   third numeric implementation.
3. No test compares two backends against each other. Every parity gate
   compares one backend against a committed fixture. See [testing.md](testing.md).
4. The CUDA device kernels use atomic operations. The attention key and value
   gradients, the embedding scatter-add, the value-gate weight gradient, and
   the block combine all use atomics. Atomic order is not a contract.
5. A bf16 or fp8 build rounds the logits to 7 mantissa bits, so the top two
   logits can tie exactly. The fixture therefore carries one section per
   precision, and each build compares its own section.

One more fact fixes the architecture. Bazel selects the backend at link time
(see `src/BUILD.bazel`). The CPU backend and the CUDA backend define the same
`nanochat::kernels::*` symbols. One process cannot hold both. Each leg compares
its own section of one committed fixture.

### 3. The contract

The contract has four clauses.

1. Each leg equals its fixture section, byte for byte.
2. The device leg reproduces its section across two runs.
3. The greedy token identifiers and the argmax equal the section on every
   target.
4. The caps invariant is exact: a simulated profile changes no value.

Clause 3 uses parity entry `E1`. [parity.md](parity.md) records greedy
generation as bit-identical. Clause 3 holds only for a peaked model head.
Section 6.3 defines the tie rule.

### 4. Record classes and where the bounds live

The fixture has one class of record, and two kinds of value that are not
records.

| Value | Where it lives | Comparison |
|---|---|---|
| Configuration, inputs, token ids, masks, greedy identifiers, argmax, valid-target counts | The fixture | Byte for byte, on every backend and every precision |
| Logit margins | The run, printed | The tie rule of section 6.3. The fp32 leg fails below `1e-5` |
| Loss, gradient norm, parameter L2, bits per byte, reward, advantage | The parity fixtures | One bound per test, below |

| Artifact | Test | Bound |
|---|---|---|
| Train loss, gradient norm, parameter L2 | `//tests:train_parity_test` | [testing.md](testing.md) |
| The same on the device | `//tests:train_parity_cuda_test` | [testing.md](testing.md) |
| Evaluation bits per byte and per-batch loss | `//tests:score_parity_test` | [testing.md](testing.md) |
| SFT loss, gradient norm, parameter L2, learning rate, momentum | `//tests:sft_step_parity_test` | [testing.md](testing.md) |
| RL reward, advantage, loss, gradient norm | `//tests:rl_parity_test` | Exact against `tests/data/rl_parity.bin` |
| The fp8 floating-point bounds | The fp8 parity tests | [simulator.md](simulator.md) Part 7, section 7 |

This gate applies no numeric tolerance. It has no per-group table, no skip, no
warning path, and no host pin. A record mismatch is a failure.

### 5. Test architecture: fixture replay

The test follows the existing `oracle_test` pattern. One source file builds as
two Bazel targets. One committed fixture is the reference.

- The **fixture** `tests/data/numerics_golden_10l.bin` comes from the CPU
  emitter. A reviewer checks the file. A developer commits the file. The file
  does not need PyTorch at runtime.
- The **CPU target** `//tests:numerics_trace_test` runs the model, then
  compares its section against the fixture in process.
- The **simulator leg** runs the same target under `--config=sim`.
- The **CUDA target** `//tests:numerics_trace_cuda_test` shares the source and
  carries the tag `gpu`.
- The **comparison runs in process**, by record name.

The fixture is portable by construction. Every record is an integer or a fixed
constant, so the same bytes come out of any host. A re-emission that differs is
a real change, not a host difference. The generator checks this: a re-emission
on this host reproduced the committed sha256 byte for byte.

#### 5.1 Fixture format

The fixture reuses the little-endian record layout of `src/container.h` under
its own magic `NANONUM1`. The container is `[8-byte magic][u32 version][u32
count]`, then one record per `[u16 name length][name][u8 dtype][u8 width][u64
shape...][payload]`.

Every record name carries its section as a prefix. The fixture holds the
sections in the order `fp32`, `bf16`, `fp8`, and each section carries the same
63 names in the same order.

| Record group | Names |
|---|---|
| Identity | `header/version`, `header/magic`, `header/seed`, `header/steps` |
| Configuration | `config/layers`, `config/heads`, `config/kv_heads`, `config/hidden`, `config/seq`, `config/batch`, `config/vocab`, `config/padded_vocab`, `config/window_pattern`, `config/value_embedding`, `config/rope_base`, `config/parameters` |
| Optimizer and schedule | `opt/*`, `sched/*` |
| Phase inputs | `train/batches`, `eval/steps`, `gen/prompt_len`, `gen/tokens`, `gen/num_samples`, `sft/ignore_every`, `rl/num_samples`, `rl/prompt_len`, `rl/tokens`, `rl/num_passes`, `rl/examples_per_rank`, `rl/reward_rule` |
| Evaluation | `eval/tokens/<b>`, `eval/argmax/<b>` |
| Generation | `gen/prompt`, `gen/ids`, `gen/mask` |
| SFT | `sft/valid_targets`, `sft/zero_weights` |
| RL | `rl/prompt/<r>`, `rl/ids/<r>`, `rl/valid_targets` |

The trace pins the host thread count to one. The CPU reference reductions depend
on the thread count, and a pinned count keeps a re-emission stable.

The schema check runs before any value is compared. It requires all three
sections to exist, requires the three suffix lists to be identical, and requires
the dtype and the shape of every record to match the run.

#### 5.2 Re-emission

One command builds the emitter under each precision and writes the single file.

```bash
python3 tools/dump_fixtures.py numerics
```

The generator runs the emitter through `tools/nanochat run t0-cpu --`, prefixes
each record name with its section, and writes the container. `--check`
regenerates into a temporary directory and compares the bytes with the committed
file. The manifest records the size, the SHA-256, the FNV-1a digest, the record
count, the record names, and the producer facts.

A change to the graph, the configuration, the seed, or the tokenizer changes the
fixture. Re-emit it in the same change, and update the manifest and
`tolerances.json` with it. A change that moves only the floating-point results
does not touch this fixture.

### 6. Trace contents and model configuration

#### 6.1 Trace contents

The trace drives five optimizer steps, then records the discrete results of the
four phases.

- The header holds the magic string, the seed, and the step count.
- The configuration holds the `Config` values and the parameter count.
- The optimizer and schedule groups hold the reference recipe as data.
- The evaluation section holds the token ids of two batches and their argmax.
- The generation section holds the prompt, the greedy identifiers, and the mask.
- The SFT section holds the valid-target count and the zero-weight count.
- The RL section holds four rollouts and the valid-target count.

#### 6.2 Reference configuration

The device leg needs a CUDA card at `sm_75` or newer, so it runs on the Kaggle
T4 or the H100 node. The sandbox caps memory at 5 GiB high and 6 GiB max.

```text
num_layers = 10
num_heads = 6
num_kv_heads = 3
hidden_dim = 384
seq_len = 256
batch = 4
vocab_size = 512
padded_vocab_size = 512
window_pattern = "SSSL"
value_embedding = true
seed = 42
steps = 5
```

`padded_vocab_size` is a separate field. `src/model.cc` reads it directly. If
the test leaves the field at the default 32768, the classifier stays wide and
the logits use about 268 MB.

Use `fp32` only. Keep the shapes small. The test measures agreement, not speed.

#### 6.3 Greedy tie rule

A random 10-layer model can produce near-ties at the argmax. One unit of
roundoff can flip the result. The test records the logit margin and requires the
margin to exceed a threshold.

The implementation measures the margin at the position that predicts each
sampled token. It reads the logits from the training graph over the whole row,
so one forward covers every step of a rollout. The threshold is `1e-5`. The
measured minima of the committed fixture are:

| Section | Evaluation | Generation | RL |
|---|---|---|---|
| fp32 | `3.73e-3` | `5.75e-2` | `3.73e-2` |
| bf16 | `0` | `4.69e-2` | `1.56e-2` |
| fp8 | `0` | `6.25e-2` | `0` |

The fp32 head is peaked, so the rule passes. A bf16 or fp8 build rounds the
logits to 7 mantissa bits, so the top two logits can be exactly equal and the
margin is exactly zero. The test prints the margin for those builds and does not
fail on it. The discrete records still compare byte for byte, and the committed
sections are stable.

#### 6.4 Trace inputs

Every input comes from the seed 42 and the committed tokenizer, so the fixture
needs no data file. The trace records no batch and no weight vector.

- The six pretraining batches come from a xoshiro256\*\* source seeded with
  `seed ^ 0x9e3779b97f4a7c15`. Each row draws `seq + 1` ids in `[0, vocab)`, and
  the row is the input and its one-position shift is the target.
- The evaluation batches come from the committed tokenizer
  `tests/data/loader_tokenizer.nctoken` and 128 fixed documents. The loader
  refills its buffer to 64 documents, so the packing does not depend on
  producer timing.
- The generation prompt is 16 ids from a source seeded with
  `seed ^ 0x2545f4914f6cdd1d`. Greedy decoding adds 16 ids and no stop token.
- The RL rollout uses four prompts of 16 ids each, from a source seeded with
  `seed ^ 0x5bf03635ef1b0a0d`. Greedy decoding adds 32 ids per row. The reward
  rule is a weighted mean of the generated ids, mapped to `[0, 1]`. The
  advantage is the reward minus the mean reward over the four rows.
- The SFT section reuses the first evaluation batch. The weight is 1 on the
  second half of each row and 0 on the first half, and every fifth target is the
  ignore index. The recorded normalizer is the valid-target count.
- The RL section takes the weights from the advantages on the sampled positions
  and 0 elsewhere. The scale is `1 / (num_passes * examples_per_rank)`, so the
  normalizer is `num_valid * num_passes * examples_per_rank`.

### 8. Commands and wall-clock budget

```bash
tools/nanochat test //tests:numerics_trace_test
tools/nanochat test --gpu //tests:numerics_trace_cuda_test
tools/nanochat test --profile t4 --suite correctness
```

A 10-layer CPU run with `double` GEMM over five steps is the slow part. It takes
about 60 seconds under the `t0-cpu` profile. The GPU leg takes about 10 seconds,
including the repeat run of clause 2.

The CPU target carries the `manual` tag, so it stays out of the default test
loop. `tools/nanochat test --suite correctness` runs it, and
`tools/nanochat test //tests:numerics_trace_test` runs it alone. Keep the
per-merge GPU budget in [AGENTS.md](../AGENTS.md) section 5.5.

### 9. Risks

- The discrete results are derived from floats, so a change that flips one
  argmax changes the fixture. That is a real change and a re-emission, not a
  host difference.
- A low-precision build can tie at the argmax. Section 6.3 records the measured
  margins. If a tie becomes unstable, the fixture is not usable for that
  section.
- A cuBLAS version change can change the device floating-point results. The
  device floating-point contract is `//tests:train_parity_cuda_test`, and the
  tolerance record pins the CUDA and cuBLAS versions.
- The fixture depends on the pinned host thread count and on the tokenizer
  artifact of the evaluation phase. Regenerate it after a change to either.
- The trace covers the discrete results only. A pure floating-point drift that
  flips no identifier is caught by the parity tests, not here.
- The card's memory and the 5 GiB sandbox cap bound the sequence length and the
  vocabulary. Keep `seq_len` at or below 256.

### 10. Open questions

1. Does clause 3 hold on a CUDA card for a 10-layer model? Resolved by Phase 3:
   the identifiers agree on all three targets.
2. Does the operator accept the reduction in section 1? If the operator needs
   bit equality on the device leg, choose Path B.
3. Should the comparison run in continuous integration, or only at merge? The
   GPU tier suggests merge only, and section 5.5 settles it. The `manual` tag
   keeps the CPU target out of the inner loop.
4. Is `vocab_size = 512` acceptable for the parity goal? The production model
   uses 32768.
