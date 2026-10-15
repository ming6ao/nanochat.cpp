# Notebooks

Three notebooks run the model pipeline on a Kaggle session with two T4 cards.
Each notebook performs its own setup, so any one notebook runs alone in a
fresh session. The shared code lives in `session.py` and `hub_checkpoint.py`.

## Run order

1. `pretrain.ipynb` downloads the data and the tokenizer, pretrains the base
   model, and publishes the `base` stage.
2. `eval.ipynb` downloads a published stage and measures it. It writes no
   checkpoint.
3. `posttrain.ipynb` downloads the `base` stage, fine-tunes it, and publishes
   the `sft` stage.

`setup.ipynb` is a manual session check and the `pi` agent. It is optional,
because every pipeline notebook runs its own setup.

## Checkpoint flow

The project writes one container, `NCHKPT01`, with the `.ckpt` suffix. The
layout is in `docs/checkpoint-format.md`.

`nc.Model.load` and `MeshRuntime.load(optimizer=False)` read the parameter
records only. They skip the optimizer section, so an evaluation or a
fine-tuning warm start does not allocate the pretraining moments.

## Hugging Face Hub layout

One repository holds the model family: `<namespace>/nanochat-<run_name>`.

```
<repo_id>/
  base/
    checkpoint.ckpt     NCHKPT01: parameters, optimizer state, and step
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

- `session.py` reads the Kaggle secrets, clones the repository, builds the
  shared library, and checks the device count. `session.setup` is the one
  entry point.
- `hub_checkpoint.py` holds every Hugging Face Hub call. It imports
  `huggingface_hub` inside each function, so the tests run without the wheel.

## Tests

`tools/nanochat test //notebooks:session_test //notebooks:hub_checkpoint_test`
runs the shared-module tests. They inject a fake secrets client, a fake Hub
API, and a fake download, so they need no network, no Kaggle package, and no
GPU.

## Running a notebook

In a Kaggle session, set the accelerator to T4 and turn Internet on. Add the
`HF_TOKEN` secret. Each notebook clones the repository, builds the shared
library for `sm_75`, and runs.
