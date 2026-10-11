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

### S0: the simulator tier

S0 is the CPU-only tier that exercises H100/H200-class behavior without a
Hopper device ([simulator.md](simulator.md)). It takes no GPU broker, so it runs
like T0 -- under the `t0-cpu` profile, in parallel with other T0 work. The
suites are:

```bash
# Everything: oracle, training parity, API, collective, emulated kernels,
# profile consistency. A numerical run defaults to fp32; the compile gate uses
# the device's named target (h200 is fp16).
tools/nanochat simulate --device h100
tools/nanochat simulate --device h200

# One suite.
tools/nanochat simulate --device h100 --suite correctness   # reference engine
tools/nanochat simulate --device h100 --suite api           # API interposer
tools/nanochat simulate --device h100 --suite collective    # NCCL mock
tools/nanochat simulate --device h100 --suite kernel        # emulated kernels
tools/nanochat simulate --device h100 --suite profile       # device table

# Compile the CUDA backend for the target architecture (no GPU, no broker).
tools/nanochat simulate --device h200 --mode compile
```

The reference-engine suites add `--config=sim` and pass
`--test_env=NANOCHAT_SIM_PROFILE=<device>`, which makes `kernels::GetCaps()`
answer from `include/nanochat/device_profile.h` while every value is still
computed by the CPU reference loops. `//tests:sim_numerics_test` asserts that
the caps change and the numerics do not, so a profile can never silently move a
result.

The half-storage path is explicit: `--precision fp16` adds `--config=fp16`.
The committed oracle and training-parity fixtures are the fp32 CPU case, and the
fp16 CPU build records its own looser tolerance, so the fp16 *correctness* suite
is expected to diverge from the fp32 fixture -- a pre-existing property of the
fp16 reference backend, not of the simulator. The fp16 profile is still
exercised: `//tests:sim_numerics_test` runs in both precisions and asserts that
the profile changes the caps and not a single value. `//tests:numerics_trace_test`
skips on an fp16 build, because the numeric trace covers fp32 only
(`docs/numerics-integration.md` section 6.2).

The collective mock rendezvouses through POSIX shared memory and bounds its
wait (`NANOCHAT_SIM_TIMEOUT_MS`), so a rank that never issues the all-reduce is
reported as a timeout instead of hanging the suite.

The native collective (`backends/cuda/collective/`, selected with
`--config=native`) adds two CPU-only tests to the default T0 loop and one T1
GPU test: `//backends/cuda/collective:schedule_test` runs the ring at world
sizes 1, 2, 4, and 8 over the in-process memory transport and judges the
schedule trace; `//backends/cuda/collective:transport_select_test` checks the
per-neighbor selection including a mixed group; and
`//backends/cuda/collective:nvlink_test` (tags `gpu`, `manual`) reduces across
two cards. On a host without NVLink peer access the T1 test exercises the
host-staged fallback. See
[distributed-native-plan.md](distributed-native-plan.md) section 13.

The suite targets carry the `sim` tag. They are CPU-only and fast, so they stay
in the default `tools/nanochat test` loop; the tag exists so a merge gate can
select or exclude them by name. Two exceptions carry `manual`.
`//tests:cuda_sim_preload_test` links the real CUDA backend and must run under
`tools/nanochat simulate --suite api`, which supplies `LD_PRELOAD`,
`NANOCHAT_SIM_LOG`, and the profile. `//tests:numerics_trace_test` replays a
10-layer trace, which is too slow for the inner loop; run it through
`tools/nanochat simulate --suite correctness` or by name.

The fabricated `cudaDeviceProp` layout is pinned by
`//tests:cuda_device_prop_abi_test`, a compile-time cross-check of
`cudaDevicePropPrefix` against the real `<driver_types.h>`; it needs the CUDA
toolkit headers but no device.

## Definition of done

Every workstream must pass, in order:

1. `tools/nanochat build` — no warnings.
2. `tools/nanochat test` — CPU tests.
3. `tools/nanochat test --gpu <target>` — small-shape GPU correctness.
4. Finite-difference gradient check (kernel families with a backward).
5. Oracle fixture match within the backend tolerance.
6. `tools/nanochat lint` — the Google C++ Style gate.

A workstream is not done until 1–6 pass on the **CPU** backend and at least
small-shape GPU correctness passes.

## How tests are sandboxed

`tools/nanochat test` runs Bazel with
`--run_under='tools/sandbox.sh --profile=… --'`, so every test action gets its
own cgroup, and forces `--spawn_strategy=local` so the wrapper can reach the
systemd user manager. GPU suites hold the broker lock for the whole invocation
with `tools/gpu.sh --lock-only`. Sandboxing per test action means the result
does not depend on whether a Bazel server is already running. The trade-off is
that Bazel's own filesystem sandbox is disabled for these runs; the resource
sandbox is the priority. See [sandbox.md](sandbox.md).

## Oracle fixtures

`tests/debug_state.bin` is produced offline by nanochat's `gpt.py` (see
[model.md](model.md)). `tests/oracle_test.cc` compares logits, loss, and
gradients, then runs a few optimizer steps and matches losses. Tolerances are
per backend: fp32 CUDA ~1e-5; the fp16 storage test is looser.

The multi-step trajectory is a strict gate (loss `1e-5`, parameter `1e-4`).
`tools/dump_oracle.py` records the AdamW `eps` it used as
`config/opt/adam_eps` (currently `1e-4`) and the C++ test matches it: several
fixture parameters have a true gradient of exactly zero at this initialization,
so at nanochat's default `eps=1e-10` their fp32 roundoff would be amplified
into unreproducible updates. Regenerate the fixture with
`python tools/dump_oracle.py --adam-eps 1e-4`.

Fixtures are data, so GPU tests never need torch at runtime. Use fixed seeds.

## Training-parity harness

`tools/dump_train_fixture.py` records a multi-step *training trajectory* from
nanochat's PyTorch `gpt.py`: the initial parameters, the exact token/target
batch at every step, and, after every step, the loss, the global gradient norm,
and the L2 norm of every parameter (and optionally the final parameter tensors).
`tests/train_parity_test.cc` drives `nanochat.cpp`'s `TrainStep` from the same
initial parameters over the same batches with the same optimizer and
learning-rate schedule, and compares the loss, gradient, and parameter
trajectories. This is the scaled-up version of the oracle trajectory gate: it
is what catches a bug that only appears after the first update, when the
zero-initialized output projections activate the attention and MLP backward.

The committed fixture `tests/data/train_parity.bin` is the compact CPU case
(`--nonzero-projections` so the whole graph is active at step 0) and runs as
`//tests:train_parity_test` (CPU) and `//tests:train_parity_cuda_test` (GPU).
Because the fixture shares the exact batches, this gate covers the model,
backward pass, optimizer, and schedules but **not** the data pipeline; the
loader difference is tracked as entry D1 in [parity.md](parity.md).
The same source builds as the standalone `//tests:train_parity` binary for an
arbitrary production fixture:

```bash
python3 tools/dump_train_fixture.py --out /tmp/train_parity_d8_s512_50.bin \
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

The notebook workflow ([notebook-workflow.md](notebook-workflow.md)) adds two
T0 tests: `//notebooks:kaggle_setup_test` for the host-contract parser and
`//notebooks:lab_test` for the trial runner. `//notebooks:all` runs both. They
ride `//:all_tests` and the default `tools/nanochat test`.

The Python targets use the local `python3` toolchain from `rules_python`, so
they need no third-party wheel. `tools/dump_api_fixture.py`,
`tools/dump_plan_fixture.py`, and `tools/dump_chat_fixture.py` produce the
committed fixtures offline.

## Finite-difference checks

Kernel families with a backward are checked with a finite-difference gradient
test next to the kernel in `backends/cuda/kernels/`. The fused backward is
derived and tested as a unit; see [DESIGN.md §5](../DESIGN.md).
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
