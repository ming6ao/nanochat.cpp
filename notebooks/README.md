# Notebooks

Three pipeline notebooks run the model pipeline on a Kaggle session with two
T4 cards. Each notebook performs its own setup, so any one notebook runs alone
in a fresh session. The shared code lives in `session.py` and
`hub_checkpoint.py`.

A fourth notebook, `modal_pretrain.ipynb`, runs the pretraining on Modal and
keeps the Kaggle notebook as the client.

## Run order

1. `pretrain.ipynb` downloads the data and the tokenizer, pretrains the base
   model, and publishes the `base` stage.
2. `eval.ipynb` downloads a published stage and measures it. It writes no
   checkpoint.
3. `posttrain.ipynb` downloads the `base` stage, fine-tunes it, and publishes
   the `sft` stage.

`modal_pretrain.ipynb` replaces step 1 when the training runs on Modal. The
notebook stays on Kaggle and the C++ driver runs in a Modal container. Set
`NANOCHAT_TARGET=modal`, add the two `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`
secrets, and deploy `tools/modal/app.py` first. It publishes the same `base`
stage, so steps 2 and 3 are unchanged.

`setup.ipynb` is a manual session check and the `pi` agent. It is optional,
because every pipeline notebook runs its own setup.

## Checkpoint flow

The project writes one checkpoint container with the `.ckpt` suffix. The
layout is in `docs/checkpoint-format.md`.

`nc.Model.load` and `MeshRuntime.load(optimizer=False)` read the parameter
records only. They skip the optimizer section, so an evaluation or a
fine-tuning warm start does not allocate the pretraining moments. The mesh
driver warm start uses the same parameter-only read.

The C++ mesh training driver writes `metrics.jsonl` and `summary.json` in the
run directory. The notebook polls the two files, so a cell never blocks on the
training loop. See `docs/training-seam.md` section 6.6.

## Hugging Face Hub layout

One repository holds the model family: `<namespace>/nanochat-<run_name>`.

```
<repo_id>/
  base/
    checkpoint.ckpt     the checkpoint container: parameters, optimizer
                        state, and step
    model.json          the ModelArgs
    train.json          the training job or the TrainArgs
    run.json            the lineage record
    tokenizer.nctoken
    metrics.jsonl
    summary.json
    README.md
  sft/
    (the same file names)
  reports/
    <stage>-<revision>.json
```

The tokenizer lives in a separate repository,
`<namespace>/nanochat-tokenizer`. Each vocabulary size is one root file, for
example `vocab-32768.nctoken`.

## Shared modules

- `session.py` holds the one session entry point. Each notebook installs the
distribution with one cell and configures the environment with one cell, then
calls `session.setup`. The call prefers the installed package: the shared
library comes from the search order of
`docs/dependency-layering-plan.md` section 6.3, so nothing is cloned and
nothing is built. When the package is absent it falls back to
`session.clone_and_build`, which fetches the repository, runs the bootstrap,
and builds the library. The returned session says which path ran. The call
reads the Kaggle secrets and checks the device count. `session.describe`
prints the mode, the library, the secret names, and the devices.
- `kaggle_setup.py` owns the repository fetch, the bootstrap, the host
  environment, the library build, and `revision(root)` for the short Git
  revision. Its `setup` prefers the installed package too, and
  `prefer_installed=False` forces the clone fallback.
- `target.py` owns the run host. `resolve` reads `NANOCHAT_TARGET`, and
  `local` is the default. `Target.prepare` maps every job path to the root the
  host reads, so a remote host needs no notebook change. The module also parses
  the two run records, so both hosts report the same rows.
- The `modal` host is in `target.py` too. `ModalTarget` uploads the input files
  to the runs Volume, spawns the deployed function in `tools/modal/app.py`, and
  returns a `ModalLiveRun`. That handle polls the run directory through the
  Volume, so the cell returns and never blocks.
- `hub_checkpoint.py` is the host's handle on the Hugging Face Hub. Every save
  and load call forwards to `nanochat_cpp_bridge.hf_hub`, which imports
  `huggingface_hub` inside the call, so the tests run without the wheel. The
  module keeps the pure helpers a notebook owns (`tokenizer_file`,
  `repo_id_for`, `build_run_record`, `now_utc`, `stage_dir`, `write_stage`) and
  the model wrapper `load_stage`, which needs this host's `args` module.
  `download_climbmix` fetches the data shards. `load_stage` and `pull_stage`
  fetch a published stage. `write_stage` writes the local metadata and the
  model card. `push_stage` and `push_file` upload the result.
- `kaggle_demo.py` holds the supervised fine-tuning helpers. `print_samples`
  prints one greedy completion for each fixed prompt.
- `mesh_train.py` launches the C++ mesh driver in a child process. The
  notebook cell polls `metrics.jsonl` and `summary.json` through the returned
  handle. `launch` resolves the run host from `target.py`, and `launch_local`
  is the local host.

## Tests

`tools/nanochat test //notebooks:session_test //notebooks:hub_checkpoint_test`
runs the shared-module tests. They inject a fake secrets client, a fake Hub
API, and a fake download, so they need no network, no Kaggle package, and no
GPU.

## Running a notebook

In a Kaggle session, set the accelerator to T4 and turn Internet on. Add the
`HF_TOKEN` secret. Each notebook's first cell installs the distribution
(`pip install "nanochat-cpp[hub]"`) and its second cell sets
`NANOCHAT_SANDBOX_BACKEND`. The shared modules (`session.py`,
`hub_checkpoint.py`) live in this directory, so open the notebook from the
repository's `notebooks/` directory; the package itself comes from the
install. A session without the package takes the clone and build fallback.

The C++ mesh driver still runs from a source tree (`tools/nanochat client`), so
a notebook that trains locally needs the clone fallback even when the package
is installed. An evaluation of a published stage needs neither.
