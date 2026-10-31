# Dependency layering and host packaging: plan

Status: implemented, except D6. Owner: Architect.

The campaign of 2026-10-30 landed stages S0 to S7 of section 9. All eight phases
passed their recorded checks, and the evidence is
`.sliceme/feat--dependency-layering-plan.evidence.md`. Section 12 records what
is in the tree and what is still open.

The C++ core must stay lean. Notebooks, Modal training, and the PyTorch parity
work must still run. This plan reaches both goals with one rule: each dependency
enters at one zone, and the build proves the zone boundary.

Section 2 records the four leaks in the tree as it stood. Section 3 records
the decisions. Section 9 records the stages and the ownership handoffs.

The plan separates three artifacts. They change on different schedules.

| Artifact | Changes with | Lives in |
|---|---|---|
| The package | a source commit | a wheel |
| The weights | a training run | a checkpoint on a volume or the Hub |
| The job | an experiment | one JSON record |

Background: [python.md](python.md) holds the Python surface. [build.md](build.md)
holds the file layout. [parity.md](parity.md) holds the known differences from
the reference.

Related documents: [sft-parity-plan.md](sft-parity-plan.md) holds the four rungs
of supervised fine-tuning parity. [host-portability.md](host-portability.md)
holds the second host. [checkpoint-format.md](checkpoint-format.md) holds the
container.

---

## 0. Scope

| In scope | Out of scope |
|---|---|
| A `pyproject.toml` with an empty base dependency list | A change to the C++ core |
| A second Python package for the heavy adapters | A change to a frozen header |
| Five gates that fail on a broken boundary | A new checkpoint format |
| Per-architecture binary wheels, and a search entry for them | A second Python surface |
| A weights volume for the Modal deployment | A public PyPI release |
| Fixture provenance in the model manifests | New parity fixtures beyond the provenance field |
| A doc row in [README.md](README.md) | A change to the sandbox or the GPU queue |

## 1. Goals and non-goals

### Goals

- Keep `pip install nanochat-cpp` free of third-party packages.
- Make a broken dependency boundary fail a test, not a review.
- Drive a notebook, a Modal container, and a test gate from one package.
- Compare with the PyTorch reference without a torch install at gate time.
- Install the package at image build, and the weights at run start.

### Non-goals

- Adding numpy to the core. The current `array` and `ctypes` code is why the
  wheel stays pure.
- Moving a source file under `src/`, `include/`, or `backends/`.
- A language server, a formatter, or a packaging rewrite.
- A published binary wheel for every platform. The first release covers sm_75
  and sm_61 only.

## 2. The defect

The C++ core is already lean. `DESIGN.md` section 2.1 enforces the layering, and
`backends/**` links no vendor library beyond CUDA. The leak is in the Python
tree.

### 2.1 Leak one: lazy third-party imports in core modules

A lazy import hides a dependency from every tool. The import succeeds in the
test loop only because the module name resolves late.

| File | Line | Import | Correct zone |
|---|---|---|---|
| `python/nanochat_cpp/checkpoint.py` | 943 | `torch` | Z2 |
| `python/nanochat_cpp/checkpoint.py` | 962 | `torch` | Z2 |
| `python/nanochat_cpp/checkpoint.py` | 986 | `torch` | Z2 |
| `python/nanochat_cpp/checkpoint.py` | 1005 | `torch` | Z2 |
| `python/nanochat_cpp/data.py` | 42 | `nanochat.dataset` | Z2 |
| `python/nanochat_cpp/data.py` | 163 | `tiktoken` | Z2 |
| `python/nanochat_cpp/tasks.py` | 65 | `numpy` | Z2 |
| `python/nanochat_cpp/tasks.py` | 409 | `nanochat.execution` | Z2 |

### 2.2 Leak two: no package description

The tree has no `pyproject.toml`. A reader cannot tell which packages the base
install needs. A host cannot install the package without a repository clone.
`tools/modal/app.py` and `notebooks/kaggle_setup.py` both clone and build the
tree for that reason.

### 2.3 Leak three: an eager package import

`python/nanochat_cpp/__init__.py` line 18 imports thirteen modules. A caller
that needs only `api` pays for `chat`, `rl`, `sft`, `sft_data`, and `toolchain`.
The cost is small today. It grows when `Z2` re-exports arrive.

### 2.4 Leak four: no provenance in a fixture

`models/nanochat-default/fixtures/manifest.json` records a hash, a size, and a
record count. It does not record the torch version or the reference commit that
produced the bytes. A parity claim in [parity.md](parity.md) then rests on an
environment that the manifest cannot name.

## 3. Decisions

| ID | Decision | Recommendation |
|---|---|---|
| D1 | How many import packages? | Two: `nanochat_cpp` and `nanochat_cpp_bridge`. |
| D2 | How many distributions? | One. The base list is empty and the extras carry the rest. |
| D3 | Where is the boundary tested? | In the test loop, by five gates. See section 8. |
| D4 | Which module moves first? | `checkpoint.py`. It has the largest torch surface. |
| D5 | Does the bridge live under `python/`? | Yes. One `py_library`, one test area, one owner. |
| D6 | How does a wheel carry the shared library? | A per-architecture wheel, `nanochat_cpp_cuda_<arch>`. |
| D7 | Where does the wheel sit in the search order? | New item 5, before the cache and the on-demand build. |
| D8 | How does Modal get the weights? | A volume keyed by the stage revision, not the image. |
| D9 | How does a notebook install? | One `pip install` cell and one environment cell. |
| D10 | How does a parity gate compare? | It reads a committed file, never an import. |
| D11 | Where does the producer code live? | `python/nanochat_cpp_bridge/fixtures/`, run by `tools/nanochat parity`. |
| D12 | Is a public PyPI release in scope? | No. The first channel is a Hugging Face repository. |

The owner decides D1, D2, and D12. D6 and D7 need a decision from the Build
workstream, because a wheel changes `.bazelrc` and the release command.

## 4. The four zones

```text
Z3  Hosts        modal, kaggle, jupyter        tools/**, notebooks/**
Z2  Adapters     torch, numpy, tiktoken,      nanochat_cpp_bridge
                 safetensors, huggingface_hub,
                 datasets, nanochat
Z1  Python core  standard library only        nanochat_cpp
Z0  C++ core     libm, cudart, cuBLAS,        src/**, include/**, backends/**
                 zstd (host only), NCCL, RDMA
```

The arrow points down. A zone imports a lower zone and never a higher one.

| Zone | May import | Must not import | Counted dependency |
|---|---|---|---|
| Z0 | the C runtime and the vendor runtime | any Python | 3 to 5 libraries |
| Z1 | the standard library, and `Z0` through `ctypes` | anything in `Z2` | 0 packages |
| Z2 | the standard library, `Z1`, and its extras | `Z3` | 1 to 6 packages |
| Z3 | any zone | nothing imports it | per host |

Five rules hold the zones.

1. `python/nanochat_cpp/**` imports only the standard library.
2. `python/nanochat_cpp_bridge/**` imports the standard library, `nanochat_cpp`,
   and its extra packages.
3. `tools/**` and `notebooks/**` import either package.
4. No package imports `tools/` or `notebooks/`.
5. A header under `include/` names no vendor type. This rule already holds.

## 5. Target tree

```text
python/
  pyproject.toml                  # new: base list empty, extras below
  BUILD.bazel                     # one new py_library and one test area
  nanochat_cpp/                   # Z1, same role
    __init__.py                   # lazy __getattr__ re-exports
    api.py chat.py rl.py mesh.py runtime.py plan.py toolchain.py
    data.py                       # the NCTOKEN1 reader stays
    checkpoint.py                 # the container read and write stay
    sft_data.py tasks.py eval_fixture.py
    _core.py _lib.py _build.py
  nanochat_cpp_bridge/            # Z2, new
    __init__.py
    torch_checkpoint.py           # from checkpoint.py lines 915 to 1046
    reference_data.py             # from data.py lines 42 and 163
    reference_tasks.py            # from tasks.py lines 65 and 409
    hf_hub.py                     # the Hub save and load path
    sft_reference.py              # the reference renderer and mixture
    fixtures/                     # the producers. See section 7
```

The model manifests gain one field. `models/<name>/fixtures/manifest.json`
records the producer beside the hash. See section 7.2.

## 6. Packaging

### 6.1 The base install

```toml
[project]
name = "nanochat-cpp"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []          # Z1. This line is the invariant.

[project.optional-dependencies]
hub       = ["huggingface_hub>=0.24", "safetensors>=0.4"]
reference = ["torch>=2.4", "numpy>=1.26", "tiktoken>=0.8"]
nanochat  = ["nanochat @ git+https://github.com/karpathy/nanochat"]
hfdata    = ["datasets>=2.20", "tokenizers>=0.20", "pyarrow>=16"]
modal     = ["modal>=0.64"]
notebook  = ["jupyterlab", "ipykernel"]

[tool.setuptools.packages.find]
include = ["nanochat_cpp*", "nanochat_cpp_bridge*"]
```

The base install carries both import packages. The bridge files are inert
without an extra, because a bridge module imports its heavy package inside the
function that needs it.

### 6.2 Extras by task

| Task | Command |
|---|---|
| Notebook, evaluate a stage | `pip install "nanochat-cpp[hub]"` |
| Modal image, train | `pip install "nanochat-cpp[hub]"` and a binary wheel |
| Produce a parity fixture | `pip install "nanochat-cpp[reference,nanochat]"` |
| SFT data from the Hub | `pip install "nanochat-cpp[reference,nanochat,hfdata]"` |
| Workstation development | the source tree, plus Bazel |

### 6.3 The shared library

A pure wheel cannot carry one `.so` for every card. Use one binary wheel per
architecture. The package name states the architecture, as `nanochat_cpp_cuda_sm75`.

`_lib.py` searches in this order after the change:

1. the path in `NANOCHAT_CPP_LIB`;
2. the Bazel runfiles tree;
3. the development tree under `bazel-bin/bindings`;
4. a library next to the package;
5. an installed `nanochat_cpp_cuda_<arch>` wheel;  **new**
6. the cache or an on-demand build.

A wrong architecture fails at `nanochat_init`, with the corrective command.

### 6.4 Modal

Install the package at image build. Fetch the weights at run start.

```python
image = (
    modal.Image.from_registry("nvidia/cuda:12.0-devel-ubuntu22.04")
    .pip_install("nanochat-cpp[hub] @ https://.../nanochat_cpp_cuda_sm75-0.1.0.whl")
)
runs   = modal.Volume.from_name("nanochat-runs",   create_if_missing=True)
weights = modal.Volume.from_name("nanochat-weights", create_if_missing=True)
```

The image holds the package. The weights volume holds `<repo_id>/<stage>/`, keyed
by the stage revision. A new checkpoint then costs a download, not an image
build.

The current `tools/modal/app.py` bakes the repository and the Bazel build into
the image. Keep that path as a fallback until the wheels exist. Add the weights
volume now, because the image path costs minutes per run today.

### 6.5 Notebook

```python
%pip install -q "nanochat-cpp[hub]" "https://.../nanochat_cpp_cuda_sm75-0.1.0.whl"
import os; os.environ.setdefault("NANOCHAT_SANDBOX_BACKEND", "none")
```

The cell does no clone and no build. The logic stays in `notebooks/session.py`
and `notebooks/target.py`. A cell holds no logic.

## 7. The parity seam

### 7.1 The rule

A comparison with the PyTorch world is a file, not an import.

```text
      producer (Z2, offline)              consumer (Z1, no torch)
torch nanochat  -->  fixtures/*.bin  -->  tests/
HF datasets     -->  packed rows     -->  src/sft_main
```

The producer runs in a virtual environment with the reference checkout. The
consumer runs in the T0 loop with no third-party package. The tree already
follows this pattern for `debug_state.bin`, `train_parity.bin`,
`tokenizer_fixture.bin`, and `numerics_golden_10l.bin`.

### 7.2 Provenance

Add two fields to each manifest record.

```json
"producer": {
  "torch": "2.5.1",
  "numpy": "1.26.4",
  "nanochat_commit": "<sha>",
  "python": "3.12.4"
},
"generated_by": "nanochat_cpp_bridge.fixtures.train_parity"
```

A gate reads the manifest and fails when a record has no producer. The claim in
[parity.md](parity.md) then names an environment that a reader can rebuild.

### 7.3 The producer entry point

`tools/nanochat parity produce <kind>` is the only producer command. It resolves
the reference virtual environment and runs the bridge fixture module. No
producer runs under `tools/nanochat test`. The lock and the sandbox therefore
still cover every long job.

### 7.4 What this unblocks

| Row | Rung | Fixture |
|---|---|---|
| `P4` | SFT data parity | `sft_rows_fixture.bin` |
| `P5` | SFT arithmetic parity | `sft_trajectory.bin` |
| `P7` | SFT validation parity | `sft_val.bin` |

Do not write those fixtures before stage S4 lands. They must land on the
consumer side of the boundary.

## 8. Gates

A boundary that no test reads is a comment. Add these gates to
`python/tests/`, and to `python/BUILD.bazel` as `py_test` targets.

| ID | Gate | Fails when |
|---|---|---|
| G1 | `python3 -I -S -c "import nanochat_cpp"` | any third-party import runs |
| G2 | An abstract syntax tree scan of `nanochat_cpp` | an import is outside `sys.stdlib_module_names` |
| G3 | A read of `pyproject.toml` | `dependencies` is not empty |
| G4 | A manifest read | a fixture record has no producer |
| G5 | A text scan of `nanochat_cpp` | a file names `nanochat_cpp_bridge` |

G2 is the load bearing gate. It is about thirty lines and it catches every leak
in section 2.1. G1 is the independent check, because a scan can miss a dynamic
import.

## 9. Stages

Land one stage per commit. Run the full test loop after each stage.

| Stage | Work | Files | Owner |
|---|---|---|---|
| S0 | Add `pyproject.toml` with an empty base list and the extras | `pyproject.toml` | Architect |
| S1 | Add gates G1, G2, G3, and G5 | `python/tests/`, `python/BUILD.bazel` | Python surface |
| S2 | Create `nanochat_cpp_bridge`, and move the torch code from `checkpoint.py` | `python/**` | Python surface |
| S3 | Move `reference_data` and `reference_tasks` | `python/**` | Python surface |
| S4 | Make `nanochat_cpp/__init__.py` lazy | `python/**` | Python surface |
| S5 | Add the binary wheel, and the search entry in `_lib.py` | `python/**`, `.bazelrc` | Python surface and Build |
| S6 | Add the weights volume, and the install at image build | `tools/modal/**` | Python surface and Notebooks |
| S7 | Add fixture provenance, then gate G4 | `models/**`, `python/**` | Architect and Model owners |

Stages S0 to S4 are small and independent of the wheels. Do them before the SFT
parity work, so a new fixture lands on the right side of the boundary.

S7 touches `models/<name>/fixtures/manifest.json`, which a Model owner holds.
Coordinate that one edit.

## 10. Open questions

| ID | Question | Recommendation |
|---|---|---|
| Q1 | Is a public PyPI release wanted? | Not now. Use a Hugging Face repository. |
| Q2 | Does the bridge get its own distribution? | No. One distribution, two import packages. |
| Q3 | Which architecture ships first? | `sm_75`, because Kaggle and the Modal image use the T4. |
| Q4 | Does the wheel carry the CPU backend? | Yes. One wheel, both backends, same runpath. |
| Q5 | Does a notebook pin a revision? | Yes. Set `NANOCHAT_CPP_LIB` or the wheel URL to one version. |

## 11. What not to do

- Do not add torch to `Z1` "just for conversion". Conversion is an adapter in
  both directions.
- Do not add numpy to the base list. It breaks the pure wheel.
- Do not add a root `requirements.txt`. It repeats the extras and drifts.
- Do not let a notebook become a gate. A notebook is a host, not a test.
- Do not put reference task logic in `tests/`. It belongs in `Z2`, behind the
  fixture.
- Do not import the bridge from the core to save one function.

## 12. Status of the tree

| Item | State |
|---|---|
| `pyproject.toml` | present, in `python/`; the base dependency list is empty |
| `nanochat_cpp_bridge` | present, six modules and `fixtures/` |
| Gate G1 to G5 | present, `python/tests/gate_*_test.py` |
| The parity dispatch test | present, `python/tests/parity_dispatch_test.py` |
| Binary wheel | absent; D6 is open and needs the Build workstream |
| The wheel search entry in `_lib.py` | present; it falls through while no wheel is installed |
| Weights volume in the Modal app | present, `nanochat-weights`, keyed by the stage revision |
| Fixture provenance | present; `producer` and `generated_by` in all seven records |
| The container read and write | present, in `python/nanochat_cpp/checkpoint.py` |
| The Hub publish path | present, in `python/nanochat_cpp_bridge/hf_hub.py` |
| The fixture hash record | present, in the model manifest |

Open after the campaign:

- D6, the per-architecture wheel. D7 added its search entry, so the release is
the only missing half.
- Q1 to Q5 of section 10.
- The CPU tier of a campaign check. The campaign submitted every check through
the GPU queue, including two that only read JSON, which section 5.1 of
`AGENTS.md` puts at `t0-cpu`.
