#!/usr/bin/env python3
"""Generate the offline fixtures for every parity gate.

One module holds every fixture kind (``docs/script-consolidation-plan.md``
section 7, P4). The kinds are ``oracle``, ``train``, ``rl``, ``generate``,
``eval``, ``plan``, ``tokenizer``, ``loader``, ``chat``, ``api``, ``numerics``,
``checkpoint``, ``sft``, and ``sft_loop``. Run
``python3 tools/dump_fixtures.py <kind> [options]``. Each kind keeps its own
``--help`` text.

Model-scoped kinds (``oracle``, ``train``, ``rl``, ``generate``, ``eval``, and
``numerics``) take ``--model NAME`` (default ``nanochat-default``), read
``models/<name>/spec.json``, and default their output to
``models/<name>/fixtures/<fixture>``. A caller can still pass an explicit
``--out``. ``--check`` regenerates into a temporary directory and compares the
bytes with the committed model fixture; in a tree without torch/numpy it
falls back to verifying the committed bytes against
``models/<name>/fixtures/manifest.json`` and says which proof ran.

The two supervised fine-tuning kinds (``sft`` and ``sft_loop``) are harness
data, not model definitions, so they always write into ``tests/data/``. Both
are hermetic at test time: a runtime test reads the committed bytes with no
reference checkout, no network, and no torch. ``sft`` writes the pack fixture,
the packed shard, the base checkpoint, and the eight-step trajectory;
``sft_loop`` reads the shard and the checkpoint and writes the loop trace. The
pinned settings and the shared model live in ``tools/sft_fixture.py``, so the
two kinds cannot drift apart (``docs/sft-parity-plan.md`` sections 5 to 7).

The module imports no third-party package at import time. Each kind imports
its own reference package, so a kind that does not touch PyTorch stays fast.
The helpers that need one of those packages live in the adapter package
``nanochat_cpp_bridge`` (``docs/dependency-layering-plan.md`` section 5): a
kind imports ``reference_data`` for the ``tiktoken`` encoding and the
reference tokenizer, and the core keeps the ``NCTOKEN1`` reader and the
container writer.
"""

from __future__ import annotations

import sys
import textwrap

from fixture_common import (COMMANDS, MODEL_FIXTURES, USAGE,
                            add_check_argument, add_model_argument,
                            as_f32 as _as_f32, as_i32 as _as_i32,
                            check_committed_manifest, check_generated,
                            build_config, compare_fixture_bytes,
                            model_fixtures_dir,
                            model_from_argv, resolve_out,
                            scalar as _scalar, selected_command,
                            torch_available)

if __name__ == "__main__":
    _argv = sys.argv[1:]
    _command = selected_command(_argv)
else:
    _argv = []
    _command = None

# A model-scoped ``--check`` can prove its own bytes only when the reference
# torch/numpy pair is importable. Without it, verify the committed fixture
# against models/<name>/fixtures/manifest.json before any kind imports torch;
# this keeps the check useful in the CPU-only sandbox and says which proof ran.
if (_command is not None and _command in MODEL_FIXTURES
        and _command != "numerics" and "--check" in _argv[1:]
        and not torch_available()):
    raise SystemExit(check_committed_manifest(_command,
                                              model_from_argv(_argv[1:])))

if _command == 'oracle':
    __doc__ = textwrap.dedent("""\
        Generate the CPU oracle fixture, ``models/<name>/fixtures/debug_state.bin``.

        The fixture is *data*: it is produced once, offline, from nanochat's reference
        ``gpt.py`` (PyTorch) and committed to the tree. Runtime tests never import
        torch; they parse this file with the dependency-free reader in
        ``tests/oracle_fixture.h``.

        What it contains
        ----------------
        For one fixed seed and one tiny configuration:

        * ``config/*``       the exact architecture and optimizer hyperparameters;
        * ``input/tokens``   the int32 token ids fed to the model;
        * ``input/targets``  the int32 next-token targets;
        * ``forward/raw_logits``   the pre-softcap classifier output;
        * ``forward/logits``       the post-softcap output (what ``gpt.py`` returns);
        * ``forward/loss``         the batch-mean cross-entropy;
        * ``grad/raw_logits``      d(loss)/d(raw logits), analytic;
        * ``grad/logits``          d(loss)/d(post-softcap logits), analytic;
        * ``param/<name>``         every initial parameter, named as in ``gpt.py``;
        * ``grad/<name>``          the autograd gradient of every parameter;
        * ``step/<k>/loss``        the loss after ``k`` optimizer steps (``k`` = 0..N);
        * ``step/<k>/param/<name>`` every parameter after ``k`` optimizer steps.

        The optimizer trajectory uses nanochat's canonical ``setup_optimizer``
        grouping, with *no* gradient clipping and a constant learning rate, so a C++
        optimizer can reproduce it exactly. It overrides the AdamW ``eps`` (default
        ``1e-4`` instead of nanochat's ``1e-10``): several fixture parameters have a
        true gradient of exactly zero at this initialization, so their fp32 backward
        is pure roundoff at the ``1e-10`` scale, which ``eps=1e-10`` amplifies into
        unreproducible updates. The chosen ``eps`` is recorded as
        ``config/opt/adam_eps`` so the C++ test can match it.

        Binary format (little-endian)
        -----------------------------
        ::

            header   : magic[8] = b"NANOORC1", uint32 version, uint32 num_records
            record   : uint16 name_len, name bytes,
                       uint8 dtype (0=fp32, 1=int32, 2=fp64, 3=int64, 4=uint8),
                       uint8 ndim, uint64 shape[ndim],
                       payload (product(shape) * dtype_size bytes)

        Run it from anywhere; it locates the repository root from ``__file__``::

            python3 tools/dump_training_fixtures.py oracle
            python3 tools/dump_training_fixtures.py oracle --seed 1234 --steps 3 --out tests/data/debug_state.bin

        The script forces the CPU and disables ``torch.compile`` so the output is
        deterministic and never touches the local GPU.

    """)
    from fixture_format import FixtureWriter
    import argparse
    import os
    import struct
    import sys
    from pathlib import Path

    # Force CPU *before* importing torch: the fixture is the CPU oracle and the
    # host has one shared GPU that must not be touched here.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # The reference optimizer is compiled with @torch.compile in gpt.py; run it
    # eagerly so the fixture is deterministic and needs no C++ compiler.
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    import torch.nn.functional as F  # noqa: E402

    # Reuse the repository's reference checkout finder.
    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO_ROOT / "python"))
    from nanochat_cpp.reference import ensure_reference_on_path  # noqa: E402

    ensure_reference_on_path()
    from nanochat.gpt import GPT, GPTConfig  # noqa: E402

    # Tiny but structurally complete: two blocks, grouped-query attention, value
    # embeddings on the last layer, sliding-window pattern. Small enough that the
    # whole fixture is a few hundred kilobytes.
    # n_embd must be at least 24: gpt.py's smear gate reads x[..., :24] and the
    # value-embedding gate reads x[..., :12]. head_dim = n_embd / n_head = 4.
    DEFAULT_SEED = 1234
    DEFAULT_LAYERS = 2
    DEFAULT_HEADS = 8
    DEFAULT_KV_HEADS = 2
    DEFAULT_EMBD = 32
    DEFAULT_VOCAB = 64
    DEFAULT_SEQ = 8
    DEFAULT_BATCH = 2
    DEFAULT_STEPS = 3
    DEFAULT_WINDOW = "SL"
    DEFAULT_PAD_TO = 64
    SOFTCAP = 15.0

    # Dtype codes shared with tests/oracle_fixture.h.


    # _as_f32 comes from tools/fixture_common.py; see the import above.


    # _as_i32 comes from tools/fixture_common.py; see the import above.


    # _scalar comes from tools/fixture_common.py; see the import above.


    # build_config comes from tools/fixture_common.py; see the import
    # above.


    def generate(args: argparse.Namespace) -> Path:
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        config = build_config(args, GPTConfig)
        model = GPT(config, pad_vocab_size_to=args.pad_to)
        model.init_weights()
        model.to(torch.float32)
        model.eval()

        padded_vocab = model.lm_head.weight.shape[0]

        generator = torch.Generator().manual_seed(args.seed + 1)
        tokens = torch.randint(
            0, args.vocab, (args.batch, args.seq), generator=generator, dtype=torch.long
        )
        targets = torch.randint(
            0, args.vocab, (args.batch, args.seq), generator=generator, dtype=torch.long
        )

        # Capture the pre-softcap classifier output from the lm_head forward hook.
        raw_holder: dict[str, torch.Tensor] = {}

        def capture_raw(_module, _inputs, output):
            raw_holder["raw"] = output

        handle = model.lm_head.register_forward_hook(capture_raw)

        model.zero_grad(set_to_none=True)
        capped = model(tokens)  # post-softcap logits, requires grad
        loss = F.cross_entropy(
            capped.reshape(-1, args.vocab), targets.reshape(-1), reduction="mean"
        )
        loss.backward()
        handle.remove()

        raw = raw_holder["raw"][..., : args.vocab]
        with torch.no_grad():
            recapped = SOFTCAP * torch.tanh(raw / SOFTCAP)
        if not torch.allclose(capped.detach(), recapped, atol=1e-6, rtol=1e-5):
            raise AssertionError("softcap reconstruction does not match gpt.py forward")

        # Analytic classifier gradients: softmax cross-entropy through the softcap.
        with torch.no_grad():
            probs = F.softmax(capped.detach(), dim=-1)
            one_hot = F.one_hot(targets, num_classes=args.vocab).to(probs.dtype)
            count = float(args.batch * args.seq)
            grad_capped = (probs - one_hot) / count
            softcap_derivative = 1.0 - (capped.detach() / SOFTCAP).square()
            grad_raw = grad_capped * softcap_derivative

        writer = FixtureWriter(Path(args.out))

        # --- configuration -----------------------------------------------------
        writer.add("config/version", np.asarray([1], dtype=np.int32))
        writer.add("config/seed", np.asarray([args.seed], dtype=np.int64))
        writer.add("config/layers", np.asarray([args.layers], dtype=np.int32))
        writer.add("config/heads", np.asarray([args.heads], dtype=np.int32))
        writer.add("config/kv_heads", np.asarray([args.kv_heads], dtype=np.int32))
        writer.add("config/embd", np.asarray([args.embd], dtype=np.int32))
        writer.add("config/vocab", np.asarray([args.vocab], dtype=np.int32))
        writer.add("config/padded_vocab", np.asarray([padded_vocab], dtype=np.int32))
        writer.add("config/seq", np.asarray([args.seq], dtype=np.int32))
        writer.add("config/batch", np.asarray([args.batch], dtype=np.int32))
        writer.add("config/steps", np.asarray([args.steps], dtype=np.int32))
        writer.add(
            "config/window_pattern",
            np.frombuffer(args.window.encode("ascii"), dtype=np.uint8).copy(),
        )
        # Optimizer hyperparameters (nanochat setup_optimizer defaults, no clipping).
        writer.add("config/opt/unembedding_lr", _scalar(0.004))
        writer.add("config/opt/embedding_lr", _scalar(0.2))
        writer.add("config/opt/matrix_lr", _scalar(0.02))
        writer.add("config/opt/scalar_lr", _scalar(0.5))
        writer.add("config/opt/weight_decay", _scalar(0.0))
        writer.add("config/opt/clip", _scalar(0.0))
        writer.add("config/opt/adam_eps", _scalar(args.adam_eps))

        # --- inputs ------------------------------------------------------------
        writer.add("input/tokens", _as_i32(tokens))
        writer.add("input/targets", _as_i32(targets))

        # --- reference forward and backward ------------------------------------
        writer.add("forward/raw_logits", _as_f32(raw))
        writer.add("forward/logits", _as_f32(capped))
        writer.add("forward/loss", _scalar(float(loss)))
        writer.add("grad/raw_logits", _as_f32(grad_raw))
        writer.add("grad/logits", _as_f32(grad_capped))

        # Initial parameters and their gradients, named as in gpt.py.
        for name, parameter in model.named_parameters():
            writer.add(f"param/{name}", _as_f32(parameter))
            if parameter.grad is None:
                raise AssertionError(f"parameter {name!r} has no gradient")
            writer.add(f"grad/{name}", _as_f32(parameter.grad))

        # --- a few optimizer steps --------------------------------------------
        writer.add("step/0/loss", _scalar(float(loss)))
        for name, parameter in model.named_parameters():
            writer.add(f"step/0/param/{name}", _as_f32(parameter))

        optimizer = model.setup_optimizer()
        # `setup_optimizer` hardcodes eps=1e-10. Several fixture parameters have a
        # true gradient of exactly zero at this initialization (the zero-initialized
        # output projections plus RMSNorm scale invariance), so their fp32 backward
        # is pure roundoff at the ~1e-10 scale. At eps=1e-10 AdamW amplifies that
        # roundoff into large spurious updates and the trajectory is not
        # reproducible. A larger eps freezes those directions while leaving the
        # smallest real gradient (mlp.c_proj, ~1.7e-4) essentially intact.
        for group in optimizer.param_groups:
            if group.get("kind") == "adamw":
                group["eps"] = args.adam_eps
        for step in range(1, args.steps + 1):
            optimizer.step()
            model.zero_grad(set_to_none=True)
            step_loss = model(tokens, targets=targets)
            step_loss.backward()
            writer.add(f"step/{step}/loss", _scalar(float(step_loss)))
            for name, parameter in model.named_parameters():
                writer.add(f"step/{step}/param/{name}", _as_f32(parameter))

        writer.write()
        return Path(args.out)


    def parse_args(argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        add_model_argument(parser)
        add_check_argument(parser)
        parser.add_argument("--out", default=None,
                            help="output .bin path (default: "
                                 "models/<model>/fixtures/debug_state.bin)")
        parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
        parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
        parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
        parser.add_argument("--kv-heads", type=int, default=DEFAULT_KV_HEADS)
        parser.add_argument("--embd", type=int, default=DEFAULT_EMBD)
        parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
        parser.add_argument("--seq", type=int, default=DEFAULT_SEQ)
        parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
        parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
        parser.add_argument("--window", default=DEFAULT_WINDOW)
        parser.add_argument("--pad-to", type=int, default=DEFAULT_PAD_TO)
        parser.add_argument("--adam-eps", type=float, default=1e-4)
        args = parser.parse_args(argv)
        args.out = str(resolve_out(args, "debug_state.bin"))
        return args


    def main(argv: list[str]) -> int:
        args = parse_args(argv)
        if args.check:
            return check_generated("oracle", args, generate)
        if torch.cuda.is_available():
            raise RuntimeError(
                "the oracle fixture must be generated on the CPU; "
                "set CUDA_VISIBLE_DEVICES='' and retry"
            )
        path = generate(args)
        size = path.stat().st_size
        print(
            f"wrote {path} ({size} bytes, seed={args.seed}, "
            f"layers={args.layers}, heads={args.heads}, kv_heads={args.kv_heads}, "
            f"embd={args.embd}, vocab={args.vocab}, seq={args.seq}, "
            f"batch={args.batch}, steps={args.steps})"
        )
        return 0
    _selected_main = main

if _command == 'train':
    __doc__ = textwrap.dedent("""\
        Generate a multi-step *training-parity* fixture,
        ``models/<name>/fixtures/train_parity.bin``.

        This is the scaled-up cousin of ``tools/dump_training_fixtures.py oracle``. Where the oracle
        fixture records one forward/backward and a couple of optimizer steps, this
        fixture records a fixed *training trajectory* of ``N`` steps: the initial
        parameters, the exact token/target batch used at every step, and, after every
        step, the loss, the global gradient norm, and the L2 norm of every parameter.

        Both implementations are then run from the *same* initial parameters over the
        *same* batches with the *same* optimizer and learning-rate schedule:

        * the reference side is this script, which drives nanochat's PyTorch ``GPT``
          with ``setup_optimizer`` and the same scheduler formulas as
          ``scripts/base_train.py``;
        * the candidate side is ``tests/train_parity_test.cc``, which drives
          ``nanochat.cpp``'s ``Model::TrainStep``.

        Because the initial weights and the data are shared, the two runs are
        comparable step by step. The per-step parameter L2 norm is a compact trajectory
        checksum that catches divergence without storing hundreds of megabytes of
        weights; pass ``--final-params`` to also record the final parameter tensors.

        Output is the same little-endian ``NANOORC1`` container the oracle fixture uses
        (see ``tools/dump_training_fixtures.py oracle``), so the C++ side reuses
        ``tests/oracle_fixture.h``.

        The fixture is generated on the CPU and deterministically, like the oracle.
        Run it from anywhere::

            python3 tools/dump_training_fixtures.py train --out /tmp/train_parity.bin

        For the production comparison against the measured ``d8_s512`` profile::

            python3 tools/dump_training_fixtures.py train         --out /tmp/train_parity_d8_s512.bin         --layers 8 --heads 4 --kv-heads 4 --embd 512 --vocab 32768         --pad-to 32768 --seq 512 --batch 8 --steps 50 --window L         --adam-eps 1e-4 --final-params

    """)
    from fixture_format import FixtureWriter
    import argparse
    import math
    import os
    import struct
    import sys
    from pathlib import Path

    # Force the CPU reference before importing torch. The fixture is the oracle and
    # the host has one shared GPU that must stay free for the candidate run.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # The reference optimizer is compiled with @torch.compile in gpt.py; run it
    # eagerly so the fixture is deterministic and needs no C++ compiler.
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    import numpy as np  # noqa: E402
    import torch  # noqa: E402


    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT / "python") not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT / "python"))
    from nanochat_cpp.reference import ensure_reference_on_path  # noqa: E402

    REFERENCE_REPO = ensure_reference_on_path()
    from nanochat.gpt import GPT, GPTConfig  # noqa: E402


    # Record data uses the shared writer in tools/fixture_format.py.


    # _as_f32 comes from tools/fixture_common.py; see the import above.


    # _as_i32 comes from tools/fixture_common.py; see the import above.


    # _scalar comes from tools/fixture_common.py; see the import above.

    # A compact but structurally complete default: grouped-query attention, value
    # embeddings on odd layers, a sliding window pattern. Small enough that the
    # committed fixture is well under a megabyte and the CPU test runs in ms.
    DEFAULT_SEED = 1234
    DEFAULT_LAYERS = 2
    DEFAULT_HEADS = 8
    DEFAULT_KV_HEADS = 2
    DEFAULT_EMBD = 32
    DEFAULT_VOCAB = 64
    DEFAULT_SEQ = 8
    DEFAULT_BATCH = 2
    DEFAULT_STEPS = 20
    DEFAULT_WINDOW = "SL"
    DEFAULT_PAD_TO = 64
    # See the oracle note: at eps=1e-10 the fp32 roundoff of exactly-zero gradient
    # directions is amplified into unreproducible updates at this initialization.
    DEFAULT_ADAM_EPS = 1e-4


    # build_config comes from tools/fixture_common.py; see the import
    # above.


    def make_stream(count: int, vocab: int, seed: int, mode: str) -> torch.Tensor:
        """A deterministic token stream.

        ``cycle`` has a learnable bigram structure (``next = 7*token + 3 mod V``)
        so the loss descends and the optimizer schedules are actually exercised;
        ``random`` is uniform noise, useful when only numerical parity matters.
        """
        if mode == "random":
            generator = torch.Generator().manual_seed(seed)
            return torch.randint(0, vocab, (count,), generator=generator, dtype=torch.long)
        stream = torch.empty(count, dtype=torch.long)
        value = 0
        for i in range(count):
            stream[i] = value
            value = (value * 7 + 3) % vocab
        return stream


    # --- scheduler mirrors of scripts/base_train.py -----------------------------


    def _warmdown_iters(num_iterations: int, warmdown_ratio: float) -> int:
        return int(round(warmdown_ratio * num_iterations))


    def get_lr_multiplier(it: int, num_iterations: int, warmup_steps: int,
                          warmdown_ratio: float, final_lr_frac: float) -> float:
        warmdown_iters = _warmdown_iters(num_iterations, warmdown_ratio)
        if warmup_steps > 0 and it < warmup_steps:
            return (it + 1) / warmup_steps
        if warmdown_iters <= 0 or it <= num_iterations - warmdown_iters:
            return 1.0
        progress = (num_iterations - it) / warmdown_iters
        return progress + (1.0 - progress) * final_lr_frac


    def get_muon_momentum(it: int, num_iterations: int, warmdown_ratio: float,
                          warmup_steps: float, start: float, peak: float,
                          final: float) -> float:
        warmdown_iters = _warmdown_iters(num_iterations, warmdown_ratio)
        warmdown_start = num_iterations - warmdown_iters
        if warmup_steps > 0 and it < warmup_steps:
            frac = it / warmup_steps
            return (1.0 - frac) * start + frac * peak
        if warmdown_iters > 0 and it >= warmdown_start:
            progress = (it - warmdown_start) / warmdown_iters
            return peak * (1.0 - progress) + final * progress
        return peak


    def get_weight_decay(it: int, num_iterations: int, base: float) -> float:
        if num_iterations <= 0:
            return base
        return base * 0.5 * (1.0 + math.cos(math.pi * it / num_iterations))


    def total_grad_norm(model: torch.nn.Module) -> float:
        total = 0.0
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            total += float(parameter.grad.detach().double().pow(2).sum())
        return math.sqrt(total)


    def param_l2(parameter: torch.Tensor) -> float:
        return float(parameter.detach().double().pow(2).sum()) ** 0.5


    def generate(args: argparse.Namespace) -> Path:
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        config = build_config(args, GPTConfig)
        model = GPT(config, pad_vocab_size_to=args.pad_to)
        model.init_weights()
        model.to(torch.float32)
        model.train()
        padded_vocab = model.lm_head.weight.shape[0]

        # `gpt.py` zero-initializes both output projections and the smear scalar,
        # which hides the attention and MLP backward behind a zero path at step 0.
        # For a stronger single-step parity check, optionally give them small
        # nonzero values so the whole graph participates from the first step.
        if args.nonzero_projections:
            generator = torch.Generator().manual_seed(args.seed + 7)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name.endswith("c_proj.weight"):
                        noise = torch.rand(parameter.shape, generator=generator)
                        parameter.copy_(0.05 * (noise * 2.0 - 1.0))
                    elif name == "smear_lambda":
                        parameter.fill_(0.7)
                    elif name == "backout_lambda":
                        parameter.fill_(0.3)

        rows = args.batch * args.seq
        stream = make_stream(rows * args.steps + 1, args.vocab, args.seed + 1,
                             args.data)
        tokens = []
        targets = []
        for step in range(args.steps):
            window = stream[step * rows:step * rows + rows + 1]
            tokens.append(window[:-1].reshape(args.batch, args.seq))
            targets.append(window[1:].reshape(args.batch, args.seq))

        writer = FixtureWriter(Path(args.out))

        # --- configuration -----------------------------------------------------
        writer.add("config/version", np.asarray([1], dtype=np.int32))
        writer.add("config/seed", np.asarray([args.seed], dtype=np.int64))
        writer.add("config/layers", np.asarray([args.layers], dtype=np.int32))
        writer.add("config/heads", np.asarray([args.heads], dtype=np.int32))
        writer.add("config/kv_heads", np.asarray([args.kv_heads], dtype=np.int32))
        writer.add("config/embd", np.asarray([args.embd], dtype=np.int32))
        writer.add("config/vocab", np.asarray([args.vocab], dtype=np.int32))
        writer.add("config/padded_vocab", np.asarray([padded_vocab], dtype=np.int32))
        writer.add("config/seq", np.asarray([args.seq], dtype=np.int32))
        writer.add("config/batch", np.asarray([args.batch], dtype=np.int32))
        writer.add("config/steps", np.asarray([args.steps], dtype=np.int32))
        writer.add(
            "config/window_pattern",
            np.frombuffer(args.window.encode("ascii"), dtype=np.uint8).copy(),
        )
        writer.add("config/opt/unembedding_lr", _scalar(args.unembedding_lr))
        writer.add("config/opt/embedding_lr", _scalar(args.embedding_lr))
        writer.add("config/opt/matrix_lr", _scalar(args.matrix_lr))
        writer.add("config/opt/scalar_lr", _scalar(args.scalar_lr))
        writer.add("config/opt/weight_decay", _scalar(args.weight_decay))
        writer.add("config/opt/clip", _scalar(args.clip))
        writer.add("config/opt/adam_eps", _scalar(args.adam_eps))
        writer.add("config/opt/muon_ns_steps", np.asarray([args.muon_ns_steps],
                                                          dtype=np.int32))
        writer.add("config/opt/muon_beta2", _scalar(args.muon_beta2))
        writer.add("config/sched/num_iterations",
                   np.asarray([args.steps], dtype=np.int32))
        writer.add("config/sched/warmup_steps",
                   np.asarray([args.warmup_steps], dtype=np.int32))
        writer.add("config/sched/warmdown_ratio", _scalar(args.warmdown_ratio))
        writer.add("config/sched/final_lr_frac", _scalar(args.final_lr_frac))
        writer.add("config/sched/weight_decay_base", _scalar(args.weight_decay_base))
        writer.add("config/sched/muon_momentum_warmup_steps",
                   _scalar(args.muon_momentum_warmup_steps))
        writer.add("config/sched/muon_momentum_start", _scalar(args.muon_momentum_start))
        writer.add("config/sched/muon_momentum_peak", _scalar(args.muon_momentum_peak))
        writer.add("config/sched/muon_momentum_final", _scalar(args.muon_momentum_final))

        # --- initial parameters and the exact batches --------------------------
        parameters = list(model.named_parameters())
        for name, parameter in parameters:
            writer.add(f"param/{name}", _as_f32(parameter))
        for step in range(args.steps):
            writer.add(f"batch/{step}/tokens", _as_i32(tokens[step]))
            writer.add(f"batch/{step}/targets", _as_i32(targets[step]))

        # --- optimizer and schedule -------------------------------------------
        optimizer = model.setup_optimizer(
            unembedding_lr=args.unembedding_lr,
            embedding_lr=args.embedding_lr,
            matrix_lr=args.matrix_lr,
            weight_decay=args.weight_decay,
            scalar_lr=args.scalar_lr,
        )
        # setup_optimizer hardcodes a 1e-10 eps and a 0.95 Muon momentum; make both
        # configurable so the recorded trajectory is reproducible and matched.
        for group in optimizer.param_groups:
            if group.get("kind") == "adamw":
                group["eps"] = args.adam_eps
            if group.get("kind") == "muon":
                group["ns_steps"] = args.muon_ns_steps
                group["beta2"] = args.muon_beta2

        def record_trajectory(step: int, loss_value: float) -> None:
            writer.add(f"step/{step}/loss", _scalar(float(loss_value)))
            for name, parameter in parameters:
                writer.add(f"step/{step}/param_l2/{name}", _scalar(param_l2(parameter)))

        # Step 0: the initial loss and the initial parameter trajectory point. The
        # backward also produces the gradients the first update will consume.
        model.zero_grad(set_to_none=True)
        loss = model(tokens[0], targets=targets[0])
        loss.backward()
        record_trajectory(0, float(loss))

        for it in range(args.steps):
            # The gradients in `.grad` were produced by the forward/backward on
            # `tokens[it]` at the parameters after `it` updates: exactly what
            # `TrainStep` consumes on its `it + 1`-th call.
            assert all(p.grad is not None for _, p in parameters)
            writer.add(f"step/{it + 1}/grad_norm", _scalar(total_grad_norm(model)))
            for name, parameter in parameters:
                writer.add(f"step/{it + 1}/grad_l2/{name}",
                           _scalar(param_l2(parameter.grad)))

            lrm = get_lr_multiplier(it, args.steps, args.warmup_steps,
                                    args.warmdown_ratio, args.final_lr_frac)
            momentum = get_muon_momentum(
                it, args.steps, args.warmdown_ratio,
                args.muon_momentum_warmup_steps, args.muon_momentum_start,
                args.muon_momentum_peak, args.muon_momentum_final)
            weight_decay = get_weight_decay(it, args.steps, args.weight_decay_base)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lrm
                if group.get("kind") == "muon":
                    group["momentum"] = momentum
                    group["weight_decay"] = weight_decay

            optimizer.step()
            model.zero_grad(set_to_none=True)

            # Post-update loss and parameter checksum on the same batch. No
            # gradient is needed, so run it under `no_grad`.
            with torch.no_grad():
                eval_loss = model(tokens[it], targets=targets[it])
            record_trajectory(it + 1, float(eval_loss))

            # The next update consumes the next batch, so build its gradients now.
            if it + 1 < args.steps:
                next_loss = model(tokens[it + 1], targets=targets[it + 1])
                next_loss.backward()

        if args.final_params:
            for name, parameter in parameters:
                writer.add(f"final/param/{name}", _as_f32(parameter))

        writer.write()
        return Path(args.out)


    def parse_args(argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        add_model_argument(parser)
        add_check_argument(parser)
        parser.add_argument("--out", default=None,
                            help="output .bin path (default: "
                                 "models/<model>/fixtures/train_parity.bin)")
        parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
        parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
        parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
        parser.add_argument("--kv-heads", type=int, default=DEFAULT_KV_HEADS)
        parser.add_argument("--embd", type=int, default=DEFAULT_EMBD)
        parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
        parser.add_argument("--seq", type=int, default=DEFAULT_SEQ)
        parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
        parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
        parser.add_argument("--window", default=DEFAULT_WINDOW)
        parser.add_argument("--pad-to", type=int, default=DEFAULT_PAD_TO)
        parser.add_argument("--data", choices=["cycle", "random"], default="random")
        parser.add_argument("--nonzero-projections", action="store_true",
                            help="give c_proj/mlp.c_proj/smear_lambda nonzero "
                                 "values so the whole graph is active at step 0")
        parser.add_argument("--final-params", action="store_true",
                            help="also record the final parameter tensors")
        # Optimizer.
        parser.add_argument("--unembedding-lr", type=float, default=0.004)
        parser.add_argument("--embedding-lr", type=float, default=0.2)
        parser.add_argument("--matrix-lr", type=float, default=0.02)
        parser.add_argument("--scalar-lr", type=float, default=0.5)
        parser.add_argument("--weight-decay", type=float, default=0.0)
        parser.add_argument("--clip", type=float, default=0.0)
        parser.add_argument("--adam-eps", type=float, default=DEFAULT_ADAM_EPS)
        parser.add_argument("--muon-ns-steps", type=int, default=5)
        parser.add_argument("--muon-beta2", type=float, default=0.9)
        # Scheduler.
        parser.add_argument("--warmup-steps", type=int, default=2)
        parser.add_argument("--warmdown-ratio", type=float, default=0.5)
        parser.add_argument("--final-lr-frac", type=float, default=0.1)
        parser.add_argument("--weight-decay-base", type=float, default=0.05)
        parser.add_argument("--muon-momentum-warmup-steps", type=float, default=4.0)
        parser.add_argument("--muon-momentum-start", type=float, default=0.85)
        parser.add_argument("--muon-momentum-peak", type=float, default=0.97)
        parser.add_argument("--muon-momentum-final", type=float, default=0.90)
        args = parser.parse_args(argv)
        args.out = str(resolve_out(args, "train_parity.bin"))
        return args


    def main(argv: list[str]) -> int:
        args = parse_args(argv)
        if args.check:
            return check_generated("train", args, generate)
        if torch.cuda.is_available():
            raise RuntimeError(
                "the training fixture must be generated on the CPU; "
                "set CUDA_VISIBLE_DEVICES='' and retry")
        path = generate(args)
        size = path.stat().st_size
        print(
            f"wrote {path} ({size} bytes, layers={args.layers}, heads={args.heads}, "
            f"kv_heads={args.kv_heads}, embd={args.embd}, vocab={args.vocab}, "
            f"seq={args.seq}, batch={args.batch}, steps={args.steps}, "
            f"data={args.data}, adam_eps={args.adam_eps:g})")
        return 0
    _selected_main = main

if _command == 'rl':
    __doc__ = textwrap.dedent("""\
        Generate the reinforcement-learning parity fixture,
        ``models/<name>/fixtures/rl_parity.bin``.

        This is the RL cousin of ``tools/dump_training_fixtures.py train``. It records the inputs
        of one short RL run so the file-driven ``rl_step`` binary and
        ``//tests:rl_parity_test`` can replay it:

        * the model configuration and the optimizer/schedule configuration,
        * the initial parameter tensors (one ``param/<name>`` record per parameter),
        * per step, the recorded rollout (``rollout/<step>/tokens``), the targets with
          ``-1`` at every ignored position (``rollout/<step>/targets``), and the
          per-position advantage weights (``rollout/<step>/advantages``), and
        * the model-independent normalization the RL step must use, computed by the
          pure-Python reference below: ``ref/<step>/valid_targets`` and
          ``ref/<step>/divisor`` = ``valid * num_passes * examples_per_rank``
          (docs/training-seam.md section 5.7).

        The reference side of a full parity run is the PyTorch ``scripts/chat_rl.py``
        rollout and objective. **This host has neither torch nor numpy installed**, so
        the script falls back to a deterministic pure-Python reference: it draws the
        parameters and the rollout from a fixed 64-bit LCG, and it computes the
        normalization reference (the valid-target count and the divisor) in pure
        Python. The model trajectory itself is then checked by
        ``//tests:rl_parity_test`` against the shared C++ step, and the recorded
        ``ref/`` records pin the normalization that the C++ step owns. The torch path
        that records a reference trajectory is a documented future extension, not a
        requirement for this fixture.

        The container is the same little-endian ``NANOORC1`` format the oracle and
        training fixtures use (see ``tools/dump_training_fixtures.py oracle``), so the C++ side reads it
        with ``tests/oracle_fixture.h``.

        Run it from anywhere::

            python3 tools/dump_training_fixtures.py rl --out tests/data/rl_parity.bin

    """)
    from fixture_format import FixtureWriter
    import argparse
    import struct
    import sys
    from pathlib import Path

    # Record data uses the shared writer in tools/fixture_format.py.


    def _f32(values) -> bytes:
        values = list(values)
        return struct.pack("<%df" % len(values), *values)


    def _i32(values) -> bytes:
        values = list(values)
        return struct.pack("<%di" % len(values), *values)


    def _i64(values) -> bytes:
        values = list(values)
        return struct.pack("<%dq" % len(values), *values)


    def _f32_scalar(value: float) -> bytes:
        return _f32([float(value)])


    def _i32_scalar(value: int) -> bytes:
        return _i32([int(value)])


    # --- deterministic pure-Python reference RNG --------------------------------


    class Rng:
        """A tiny deterministic 64-bit linear congruential generator.

        It is deliberately not `random`: the fixture must be byte-identical across
        Python builds, and the LCG is small enough to audit.
        """

        def __init__(self, seed: int) -> None:
            self.state = seed & 0xFFFFFFFFFFFFFFFF

        def next_u64(self) -> int:
            self.state = (self.state * 6364136223846793005 +
                          1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            return self.state

        def uniform(self, low: float, high: float) -> float:
            return low + (high - low) * (self.next_u64() / float(1 << 64))

        def randrange(self, count: int) -> int:
            return self.next_u64() % count


    # --- fixture configuration --------------------------------------------------

    # A compact but structurally complete model: grouped-query attention, value
    # embeddings on the odd layer, a sliding-window pattern. Small enough that the
    # committed fixture is a few tens of kilobytes and the CPU test runs in ms.
    DEFAULT_LAYERS = 2
    DEFAULT_HEADS = 8
    DEFAULT_KV_HEADS = 2
    DEFAULT_EMBD = 32
    DEFAULT_VOCAB = 64
    DEFAULT_PAD_TO = 64
    DEFAULT_SEQ = 8
    DEFAULT_BATCH = 2
    DEFAULT_STEPS = 3
    DEFAULT_WINDOW = "SL"
    DEFAULT_SEED = 1234
    DEFAULT_NUM_PASSES = 2
    DEFAULT_EXAMPLES_PER_RANK = 1
    # The prompt positions are ignored; the sampled positions carry the advantage.
    DEFAULT_PROMPT_LEN = 2

    # `Config::has_value_embedding` (include/nanochat/config.h): value embeddings
    # are on the layers whose parity matches the last layer's.
    _SMEAR_CHANNELS = 24
    _VE_GATE_CHANNELS = 12


    def has_value_embedding(layer: int, layers: int) -> bool:
        return layer % 2 == (layers - 1) % 2


    def parameter_specs(layers: int, heads: int, kv_heads: int, embd: int,
                        padded_vocab: int) -> list[tuple[str, tuple[int, ...]]]:
        """The parameter name and shape list, in the optimizer grouping order.

        It mirrors the constructor order in ``src/model.cc`` so every
        ``param/<name>`` record the C++ model looks up is present.
        """
        head_dim = embd // heads
        query_dim = heads * head_dim
        kv_dim = kv_heads * head_dim
        mlp_dim = 4 * embd
        specs: list[tuple[str, tuple[int, ...]]] = []
        specs.append(("lm_head.weight", (padded_vocab, embd)))
        specs.append(("transformer.wte.weight", (padded_vocab, embd)))
        for layer in range(layers):
            if has_value_embedding(layer, layers):
                specs.append((f"value_embeds.{layer}.weight", (padded_vocab,
                                                               kv_dim)))
        specs.append(("resid_lambdas", (layers,)))
        specs.append(("x0_lambdas", (layers,)))
        specs.append(("smear_gate.weight", (_SMEAR_CHANNELS, 1)))
        specs.append(("smear_lambda", (1,)))
        specs.append(("backout_lambda", (1,)))
        for layer in range(layers):
            specs.append((f"transformer.h.{layer}.attn.c_q.weight",
                          (query_dim, embd)))
            specs.append((f"transformer.h.{layer}.attn.c_k.weight",
                          (kv_dim, embd)))
            specs.append((f"transformer.h.{layer}.attn.c_v.weight",
                          (kv_dim, embd)))
            specs.append((f"transformer.h.{layer}.attn.c_proj.weight",
                          (embd, embd)))
            if has_value_embedding(layer, layers):
                specs.append((f"transformer.h.{layer}.attn.ve_gate.weight",
                              (kv_heads, _VE_GATE_CHANNELS)))
            specs.append((f"transformer.h.{layer}.mlp.c_fc.weight",
                          (mlp_dim, embd)))
            specs.append((f"transformer.h.{layer}.mlp.c_proj.weight",
                          (embd, mlp_dim)))
        return specs


    def _parameter_values(name: str, shape: tuple[int, ...], rng: Rng) -> bytes:
        count = 1
        for extent in shape:
            count *= extent
        if name == "resid_lambdas":
            return _f32([1.0] * count)
        if name == "x0_lambdas":
            return _f32([0.2] * count)
        if name == "smear_lambda":
            return _f32([0.0] * count)
        if name == "backout_lambda":
            return _f32([0.2] * count)
        if name.endswith("ve_gate.weight"):
            return _f32([rng.uniform(0.0, 0.02) for _ in range(count)])
        if name == "smear_gate.weight":
            return _f32([rng.uniform(0.0, 0.02) for _ in range(count)])
        return _f32([rng.uniform(-0.1, 0.1) for _ in range(count)])


    def _advantages(values: list[float]) -> list[float]:
        """The per-row advantage: the reward minus the batch mean reward."""
        mean = sum(values) / len(values)
        return [value - mean for value in values]


    def generate(args: argparse.Namespace) -> Path:
        layers = args.layers
        heads = args.heads
        kv_heads = args.kv_heads
        embd = args.embd
        vocab = args.vocab
        padded_vocab = args.pad_to
        seq = args.seq
        batch = args.batch
        steps = args.steps
        prompt_len = args.prompt_len
        if padded_vocab < vocab:
            raise SystemExit("--pad-to must be >= --vocab")
        if prompt_len < 0 or prompt_len >= seq:
            raise SystemExit("--prompt-len must be in [0, seq)")

        writer = FixtureWriter(Path(args.out))

        # --- configuration -----------------------------------------------------
        writer.add("config/version", "i4", (1,), _i32_scalar(1))
        writer.add("config/layers", "i4", (1,), _i32_scalar(layers))
        writer.add("config/heads", "i4", (1,), _i32_scalar(heads))
        writer.add("config/kv_heads", "i4", (1,), _i32_scalar(kv_heads))
        writer.add("config/embd", "i4", (1,), _i32_scalar(embd))
        writer.add("config/vocab", "i4", (1,), _i32_scalar(vocab))
        writer.add("config/padded_vocab", "i4", (1,), _i32_scalar(padded_vocab))
        writer.add("config/seq", "i4", (1,), _i32_scalar(seq))
        writer.add("config/batch", "i4", (1,), _i32_scalar(batch))
        writer.add("config/steps", "i4", (1,), _i32_scalar(steps))
        writer.add("config/num_passes", "i4", (1,), _i32_scalar(args.num_passes))
        writer.add("config/examples_per_rank", "i4", (1,),
                   _i32_scalar(args.examples_per_rank))
        writer.add("config/window_pattern", "u1", (len(args.window),),
                   args.window.encode("ascii"))
        writer.add("config/opt/unembedding_lr", "f4", (1,),
                   _f32_scalar(args.unembedding_lr))
        writer.add("config/opt/embedding_lr", "f4", (1,),
                   _f32_scalar(args.embedding_lr))
        writer.add("config/opt/matrix_lr", "f4", (1,),
                   _f32_scalar(args.matrix_lr))
        writer.add("config/opt/scalar_lr", "f4", (1,),
                   _f32_scalar(args.scalar_lr))
        writer.add("config/opt/weight_decay", "f4", (1,),
                   _f32_scalar(args.weight_decay))
        writer.add("config/opt/clip", "f4", (1,), _f32_scalar(args.clip))
        writer.add("config/opt/adam_eps", "f4", (1,),
                   _f32_scalar(args.adam_eps))
        writer.add("config/opt/muon_ns_steps", "i4", (1,),
                   _i32_scalar(args.muon_ns_steps))
        writer.add("config/opt/muon_beta2", "f4", (1,),
                   _f32_scalar(args.muon_beta2))
        writer.add("config/sched/num_iterations", "i4", (1,), _i32_scalar(steps))
        writer.add("config/sched/warmup_steps", "i4", (1,),
                   _i32_scalar(args.warmup_steps))
        writer.add("config/sched/warmdown_ratio", "f4", (1,),
                   _f32_scalar(args.warmdown_ratio))
        writer.add("config/sched/final_lr_frac", "f4", (1,),
                   _f32_scalar(args.final_lr_frac))
        writer.add("config/sched/weight_decay_base", "f4", (1,),
                   _f32_scalar(args.weight_decay_base))
        writer.add("config/sched/muon_momentum_warmup_steps", "f4", (1,),
                   _f32_scalar(args.muon_momentum_warmup_steps))
        writer.add("config/sched/muon_momentum_start", "f4", (1,),
                   _f32_scalar(args.muon_momentum_start))
        writer.add("config/sched/muon_momentum_peak", "f4", (1,),
                   _f32_scalar(args.muon_momentum_peak))
        writer.add("config/sched/muon_momentum_final", "f4", (1,),
                   _f32_scalar(args.muon_momentum_final))

        # --- initial parameters ------------------------------------------------
        rng = Rng(args.seed)
        for name, shape in parameter_specs(layers, heads, kv_heads, embd,
                                           padded_vocab):
            writer.add(f"param/{name}", "f4", shape,
                       _parameter_values(name, shape, rng))

        # --- recorded rollouts and the pure-Python normalization reference -----
        rows = batch * seq
        for step in range(steps):
            tokens = [rng.randrange(vocab) for _ in range(rows)]
            targets = [-1] * rows
            advantages = [0.0] * rows
            rewards = [rng.uniform(0.0, 1.0) for _ in range(batch)]
            row_advantage = _advantages(rewards)
            for row in range(batch):
                for position in range(seq):
                    index = row * seq + position
                    if position < prompt_len:
                        continue
                    # A sampled position: a valid target and the row's advantage.
                    targets[index] = tokens[index]
                    advantages[index] = row_advantage[row]
            valid = sum(1 for target in targets if target != -1)
            divisor = valid * args.num_passes * args.examples_per_rank
            writer.add(f"rollout/{step}/tokens", "i4", (batch, seq),
                       _i32(tokens))
            writer.add(f"rollout/{step}/targets", "i4", (batch, seq),
                       _i32(targets))
            writer.add(f"rollout/{step}/advantages", "f4", (batch, seq),
                       _f32(advantages))
            # The model-independent reference the C++ step must reproduce.
            writer.add(f"ref/{step}/valid_targets", "i4", (1,),
                       _i32_scalar(valid))
            writer.add(f"ref/{step}/divisor", "i8", (1,), _i64([divisor]))

        writer.write()
        return Path(args.out)


    def parse_args(argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        add_model_argument(parser)
        add_check_argument(parser)
        parser.add_argument("--out", default=None,
                            help="output .bin path (default: "
                                 "models/<model>/fixtures/rl_parity.bin)")
        parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
        parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
        parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
        parser.add_argument("--kv-heads", type=int, default=DEFAULT_KV_HEADS)
        parser.add_argument("--embd", type=int, default=DEFAULT_EMBD)
        parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
        parser.add_argument("--pad-to", type=int, default=DEFAULT_PAD_TO)
        parser.add_argument("--seq", type=int, default=DEFAULT_SEQ)
        parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
        parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
        parser.add_argument("--window", default=DEFAULT_WINDOW)
        parser.add_argument("--prompt-len", type=int, default=DEFAULT_PROMPT_LEN)
        parser.add_argument("--num-passes", type=int, default=DEFAULT_NUM_PASSES)
        parser.add_argument("--examples-per-rank", type=int,
                            default=DEFAULT_EXAMPLES_PER_RANK)
        # Optimizer.
        parser.add_argument("--unembedding-lr", type=float, default=0.004)
        parser.add_argument("--embedding-lr", type=float, default=0.2)
        parser.add_argument("--matrix-lr", type=float, default=0.02)
        parser.add_argument("--scalar-lr", type=float, default=0.5)
        parser.add_argument("--weight-decay", type=float, default=0.0)
        parser.add_argument("--clip", type=float, default=0.0)
        parser.add_argument("--adam-eps", type=float, default=1e-4)
        parser.add_argument("--muon-ns-steps", type=int, default=5)
        parser.add_argument("--muon-beta2", type=float, default=0.9)
        # Scheduler.
        parser.add_argument("--warmup-steps", type=int, default=1)
        parser.add_argument("--warmdown-ratio", type=float, default=1.0)
        parser.add_argument("--final-lr-frac", type=float, default=0.0)
        parser.add_argument("--weight-decay-base", type=float, default=0.0)
        parser.add_argument("--muon-momentum-warmup-steps", type=float, default=0.0)
        parser.add_argument("--muon-momentum-start", type=float, default=0.85)
        parser.add_argument("--muon-momentum-peak", type=float, default=0.97)
        parser.add_argument("--muon-momentum-final", type=float, default=0.90)
        args = parser.parse_args(argv)
        args.out = str(resolve_out(args, "rl_parity.bin"))
        return args


    def main(argv: list[str]) -> int:
        args = parse_args(argv)
        if args.check:
            return check_generated("rl", args, generate)
        path = generate(args)
        size = path.stat().st_size
        print(
            f"wrote {path} ({size} bytes, layers={args.layers}, heads={args.heads}, "
            f"kv_heads={args.kv_heads}, embd={args.embd}, vocab={args.vocab}, "
            f"seq={args.seq}, batch={args.batch}, steps={args.steps}, "
            f"num_passes={args.num_passes}, "
            f"examples_per_rank={args.examples_per_rank})")
        return 0
    _selected_main = main

if _command == 'generate':
    __doc__ = textwrap.dedent("""\
        Generate the *generation-parity* fixture, ``models/<name>/fixtures/generate_parity.bin``.

        This is the inference analogue of ``tools/dump_training_fixtures.py oracle`` (one forward/backward)
        and ``tools/dump_training_fixtures.py train`` (a training trajectory). It records a fixed
        greedy-decoding trace from nanochat's PyTorch ``GPT``:

        * the architecture and generation hyperparameters,
        * every model parameter, named as in ``gpt.py``,
        * the reference prompt,
        * the generated token ids (the effective prompt followed by the sampled ids),
        * the per-position mask (``0`` for a prompt id, ``1`` for a sampled id).

        ``tests/generate_parity_test.cc`` rebuilds the same model from the recorded
        parameters and drives ``nanochat.cpp``'s ``GenerateBatch`` in greedy mode. The
        expected ids and mask are compared exactly. Because the fixture is data, the
        test never imports torch; the same source runs against the CPU reference backend
        by default and the CUDA backend under the ``gpu`` tag.

        Two details make the discrete comparison robust. First, the recorded parameters
        are *trained* for ``--train-steps`` steps on a learnable bigram stream before the
        trace is taken. Random initialization leaves the classifier almost uniform, so
        two implementations that agree to within fp32 roundoff can still disagree on an
        argmax; a short, deterministic PyTorch training run sharpens the distribution
        and the top-1 margin by orders of magnitude. Second, the trace is produced by the
        reference ``GPT.generate`` with ``temperature=0`` (greedy), which is exactly the
        single-row loop ``GenerateBatch`` is built to reproduce.

        Output is the same little-endian ``NANOORC1`` container as the oracle and
        training fixtures (see ``tools/dump_training_fixtures.py oracle``), so the C++ side reuses
        ``tests/oracle_fixture.h``.

        The fixture is generated on the CPU and deterministically. Run it from
        anywhere::

            python3 tools/dump_model_fixtures.py generate --out tests/data/generate_parity.bin

    """)
    from fixture_format import FixtureWriter
    import argparse
    import os
    import struct
    import sys
    from pathlib import Path

    # Force the CPU reference before importing torch. The fixture is the oracle and
    # the host has one shared GPU that must stay free for the candidate run.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # The reference optimizer is compiled with @torch.compile in gpt.py; run it
    # eagerly so the fixture is deterministic and needs no C++ compiler.
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    import numpy as np  # noqa: E402
    import torch  # noqa: E402


    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT / "python") not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT / "python"))
    from nanochat_cpp.reference import ensure_reference_on_path  # noqa: E402

    REFERENCE_REPO = ensure_reference_on_path()
    from nanochat.gpt import GPT, GPTConfig  # noqa: E402


    # Record data uses the shared writer in tools/fixture_format.py.


    # _as_f32 comes from tools/fixture_common.py; see the import above.


    # _scalar comes from tools/fixture_common.py; see the import above.


    # A compact but structurally complete default: grouped-query attention, value
    # embeddings on odd layers, a sliding window pattern. Matches the oracle and
    # training fixtures (2 layers, 8 query heads, 2 key/value heads, hidden 32,
    # vocabulary 64) apart from `sequence_len`, which is 16 here so the trace can
    # extend past the 8-token training fixtures without leaving the context.
    DEFAULT_SEED = 1234
    DEFAULT_LAYERS = 2
    DEFAULT_HEADS = 8
    DEFAULT_KV_HEADS = 2
    DEFAULT_EMBD = 32
    DEFAULT_VOCAB = 64
    # `sequence_len` is 16 and the committed trace is `prompt_len + max_tokens = 14`
    # tokens, so the run stays inside one training context. That matters: the
    # reference sets the "L" (full-context) window to `sequence_len` rather than
    # unlimited, so once generation runs past `sequence_len` its L layers attend to
    # only the last `sequence_len` positions, while nanochat.cpp's L layers are
    # unlimited. Keeping the trace within `sequence_len` makes the two attend the
    # same keys. `generate` below enforces this.
    DEFAULT_SEQ = 16
    DEFAULT_WINDOW = "SL"
    DEFAULT_PAD_TO = 64
    DEFAULT_PROMPT_LEN = 6
    DEFAULT_MAX_TOKENS = 8
    DEFAULT_TRAIN_STEPS = 300
    DEFAULT_BATCH = 2


    # build_config comes from tools/fixture_common.py; see the import
    # above.


    def make_stream(count: int, vocab: int) -> torch.Tensor:
        """A deterministic, learnable bigram stream (``next = 7*token + 3 mod V``).

        The model can drive the loss near zero on it, which is what sharpens the
        classifier and makes the greedy trace robust to roundoff.
        """
        stream = torch.empty(count, dtype=torch.long)
        value = 0
        for i in range(count):
            stream[i] = value
            value = (value * 7 + 3) % vocab
        return stream


    def train_reference(model: GPT, args: argparse.Namespace) -> float:
        """Runs a short deterministic training loop to sharpen the classifier."""
        if args.train_steps <= 0:
            return float("nan")
        rows = args.batch * args.seq
        stream = make_stream(rows * (args.train_steps + 1) + 1, args.vocab)
        optimizer = model.setup_optimizer()
        model.train()
        loss_value = float("nan")
        for it in range(args.train_steps):
            window = stream[it * rows:it * rows + rows + 1]
            tokens = window[:-1].reshape(args.batch, args.seq)
            targets = window[1:].reshape(args.batch, args.seq)
            loss = model(tokens, targets=targets)
            loss.backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
            loss_value = float(loss)
        return loss_value


    def generate(args: argparse.Namespace) -> Path:
        if args.prompt_len + args.max_tokens > args.seq:
            raise SystemExit(
                "prompt_len + max_tokens must not exceed seq: the reference caps "
                "its L layers at sequence_len, so a longer trace would compare "
                "two different attention windows")
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        config = build_config(args, GPTConfig)
        model = GPT(config, pad_vocab_size_to=args.pad_to)
        model.init_weights()
        model.to(torch.float32)

        train_reference(model, args)
        model.eval()

        padded_vocab = model.lm_head.weight.shape[0]

        prompt_generator = torch.Generator().manual_seed(args.seed + 1)
        prompt = torch.randint(0, args.vocab, (args.prompt_len,),
                               generator=prompt_generator, dtype=torch.long)
        prompt_ids = prompt.tolist()

        # The reference greedy decode. `generate` is the batch-1 streaming loop in
        # gpt.py: temperature <= 0 selects an argmax over the soft-capped logits at
        # the last position, exactly the `GenerateBatch` greedy contract.
        generated = list(model.generate(prompt_ids, max_tokens=args.max_tokens,
                                        temperature=0.0, top_k=None,
                                        seed=args.seed))
        effective = prompt_ids + generated
        mask = [0] * len(prompt_ids) + [1] * len(generated)

        writer = FixtureWriter(Path(args.out))

        # --- configuration -----------------------------------------------------
        writer.add("config/version", np.asarray([1], dtype=np.int32))
        writer.add("config/seed", np.asarray([args.seed], dtype=np.int64))
        writer.add("config/layers", np.asarray([args.layers], dtype=np.int32))
        writer.add("config/heads", np.asarray([args.heads], dtype=np.int32))
        writer.add("config/kv_heads", np.asarray([args.kv_heads], dtype=np.int32))
        writer.add("config/embd", np.asarray([args.embd], dtype=np.int32))
        writer.add("config/vocab", np.asarray([args.vocab], dtype=np.int32))
        writer.add("config/padded_vocab", np.asarray([padded_vocab], dtype=np.int32))
        writer.add("config/seq", np.asarray([args.seq], dtype=np.int32))
        writer.add("config/train_steps",
                   np.asarray([args.train_steps], dtype=np.int32))
        writer.add(
            "config/window_pattern",
            np.frombuffer(args.window.encode("ascii"), dtype=np.uint8).copy(),
        )
        writer.add("config/prompt_len", np.asarray([args.prompt_len],
                                                   dtype=np.int32))
        writer.add("config/max_tokens", np.asarray([args.max_tokens],
                                                   dtype=np.int32))
        writer.add("config/bos_id", np.asarray([-1], dtype=np.int32))
        writer.add("config/temperature", _scalar(0.0))

        # --- trained parameters, prompt, and the greedy trace ------------------
        for name, parameter in model.named_parameters():
            writer.add(f"param/{name}", _as_f32(parameter))
        writer.add("prompt/tokens",
                   np.asarray(prompt_ids, dtype=np.int32))
        writer.add("generate/tokens", np.asarray(effective, dtype=np.int32))
        writer.add("generate/mask", np.asarray(mask, dtype=np.uint8))

        writer.write()
        return Path(args.out)


    def parse_args(argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        add_model_argument(parser)
        add_check_argument(parser)
        parser.add_argument("--out", default=None,
                            help="output .bin path (default: "
                                 "models/<model>/fixtures/generate_parity.bin)")
        parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
        parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
        parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
        parser.add_argument("--kv-heads", type=int, default=DEFAULT_KV_HEADS)
        parser.add_argument("--embd", type=int, default=DEFAULT_EMBD)
        parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
        parser.add_argument("--seq", type=int, default=DEFAULT_SEQ)
        parser.add_argument("--window", default=DEFAULT_WINDOW)
        parser.add_argument("--pad-to", type=int, default=DEFAULT_PAD_TO)
        parser.add_argument("--prompt-len", type=int, default=DEFAULT_PROMPT_LEN)
        parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
        parser.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS)
        parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
        args = parser.parse_args(argv)
        args.out = str(resolve_out(args, "generate_parity.bin"))
        return args


    def main(argv: list[str]) -> int:
        args = parse_args(argv)
        if args.check:
            return check_generated("generate", args, generate)
        if torch.cuda.is_available():
            raise RuntimeError(
                "the generation fixture must be generated on the CPU; "
                "set CUDA_VISIBLE_DEVICES='' and retry")
        path = generate(args)
        size = path.stat().st_size
        print(
            f"wrote {path} ({size} bytes, layers={args.layers}, heads={args.heads}, "
            f"kv_heads={args.kv_heads}, embd={args.embd}, vocab={args.vocab}, "
            f"seq={args.seq}, prompt_len={args.prompt_len}, "
            f"max_tokens={args.max_tokens}, train_steps={args.train_steps})")
        return 0
    _selected_main = main

if _command == 'eval':
    __doc__ = textwrap.dedent("""\
        Generate the *score-parity* fixture, ``models/<name>/fixtures/core_eval.bin``.

        This is the evaluation analogue of ``tools/dump_model_fixtures.py generate``. It
        records a tiny, fully synthetic CORE-shaped scoring problem and the PyTorch
        reference result, so ``tests/score_parity_test.cc`` can drive ``score_main``
        without the network and without the real ``eval_bundle.zip``.

        The fixture mirrors the byte layout owned by
        ``python/nanochat_cpp/eval_fixture.py`` (``NANOEVL1``; see its module
        docstring). On top of that schema it carries the model configuration and every
        model parameter (``param/<name>``, the names from ``GPT.named_parameters()``),
        so the committed file is self-contained: the C++ test rebuilds the same model,
        writes a checkpoint, and runs ``score_main`` on the recorded input.

        The scored batch exercises the three CORE candidate-span shapes from
        ``nanochat/core_eval.py``:

        * ``multiple_choice``: several rows share the prompt prefix and differ in the
          continuation, so ``input/spans[i] = [common_prefix_len, len_i)``;
        * ``schema``: several rows share the continuation suffix and differ in the
          context, so ``input/spans[i] = [len_i - common_suffix_len, len_i)``;
        * ``language_modeling``: one row is the prompt with the continuation, so
          ``input/spans[0] = [len(prompt_without), len(prompt_with)]``.

        Two rows carry an ``input/focus_*`` candidate set (the categorical-chat shape),
        so the test also covers ``result/focus_logits``.

        The parameters are trained briefly on the same learnable bigram stream the
        other fixtures use. That is not needed for the loss values, but it sharply
        separates the top-1 logit so the exact argmax comparison is stable against the
        fp32 roundoff that separates the two implementations.

        Output is little-endian and deterministic, and the generation is CPU-only:

            python3 tools/dump_model_fixtures.py eval --out tests/data/core_eval.bin

    """)
    from fixture_format import FixtureWriter
    import argparse
    import os
    import struct
    import sys
    from array import array
    from pathlib import Path

    # Force the CPU reference before importing torch. The fixture is the oracle and
    # the host has one shared GPU that must stay free for the candidate run.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # The reference optimizer is compiled with @torch.compile in gpt.py; run it
    # eagerly so the fixture is deterministic and needs no C++ compiler.
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    import numpy as np  # noqa: E402
    import torch  # noqa: E402


    # Reuse the repository's reference checkout finder and fixture reader.
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT / "python") not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT / "python"))
    from nanochat_cpp.reference import ensure_reference_on_path  # noqa: E402

    REFERENCE_REPO = ensure_reference_on_path()
    from nanochat.gpt import GPT, GPTConfig  # noqa: E402
    from nanochat_cpp import eval_fixture as ef  # noqa: E402

    # A compact but structurally complete default: grouped-query attention, value
    # embeddings on the odd layer, and a sliding-window pattern, matching the other
    # committed fixtures.
    DEFAULT_SEED = 1234
    DEFAULT_LAYERS = 2
    DEFAULT_HEADS = 8
    DEFAULT_KV_HEADS = 2
    DEFAULT_EMBD = 32
    DEFAULT_VOCAB = 64
    DEFAULT_TRAIN_STEPS = 20
    DEFAULT_BATCH = 2
    DEFAULT_PAD_TO = 64
    DEFAULT_PAD_ID = 0
    # The reference caps its L layers at `sequence_len`; the fixture width is the
    # longest row and every row stays within it, so the C++ model sees the same
    # attention window.
    DEFAULT_SEQ = 11


    def build_plan_config(seq: int, vocab: int, layers: int, heads: int,
                          kv_heads: int, embd: int, window: str) -> GPTConfig:
        return GPTConfig(
            sequence_len=seq,
            vocab_size=vocab,
            n_layer=layers,
            n_head=heads,
            n_kv_head=kv_heads,
            n_embd=embd,
            window_pattern=window,
        )


    def make_stream(count: int, vocab: int) -> torch.Tensor:
        """A deterministic, learnable bigram stream (``next = 7*token + 3 mod V``)."""
        stream = torch.empty(count, dtype=torch.long)
        value = 0
        for i in range(count):
            stream[i] = value
            value = (value * 7 + 3) % vocab
        return stream


    def train_reference(model: GPT, steps: int, batch: int, seq: int,
                        vocab: int) -> float:
        """Runs a short deterministic training loop to sharpen the classifier."""
        if steps <= 0:
            return float("nan")
        rows = batch * seq
        stream = make_stream(rows * (steps + 1) + 1, vocab)
        optimizer = model.setup_optimizer()
        model.train()
        loss_value = float("nan")
        for it in range(steps):
            window = stream[it * rows:it * rows + rows + 1]
            tokens = window[:-1].reshape(batch, seq)
            targets = window[1:].reshape(batch, seq)
            loss = model(tokens, targets=targets)
            loss.backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
            loss_value = float(loss)
        return loss_value


    def _cycle(vocab: int) -> list[int]:
        """The bigram cycle ``next = 7*token + 3 mod V`` starting at 0."""
        cycle: list[int] = []
        value = 0
        for _ in range(vocab):
            cycle.append(value)
            value = (value * 7 + 3) % vocab
            if value == 0:
                break
        if len(cycle) < 2:
            raise SystemExit("the requested vocabulary has a trivial bigram cycle")
        return cycle


    def _pick(cycle: list[int], start: int, count: int) -> list[int]:
        return [cycle[(start + i) % len(cycle)] for i in range(count)]


    def _find_common_length(token_sequences: list[list[int]],
                            direction: str) -> int:
        """Length of the common prefix (``left``) or suffix (``right``).

        Mirrors ``nanochat.core_eval.find_common_length`` so the fixture's spans
        are exactly what the reference CORE batcher produces.
        """
        min_len = min(len(seq) for seq in token_sequences)
        if direction == "left":
            indices = range(min_len)
        else:
            indices = range(-1, -min_len - 1, -1)
        for i, idx in enumerate(indices):
            token = token_sequences[0][idx]
            if not all(seq[idx] == token for seq in token_sequences):
                return i
        return min_len


    def batch_sequences_mc(sequences: list[list[int]]) -> tuple[list[int],
                                                               list[int]]:
        """Candidate spans for multiple choice: shared prefix, ends at each length."""
        start = _find_common_length(sequences, "left")
        return [start] * len(sequences), [len(seq) for seq in sequences]


    def batch_sequences_schema(
            sequences: list[list[int]]) -> tuple[list[int], list[int]]:
        """Candidate spans for schema: shared suffix, starts at end - suffix."""
        suffix = _find_common_length(sequences, "right")
        ends = [len(seq) for seq in sequences]
        return [end - suffix for end in ends], ends


    def batch_sequences_lm(without: list[int],
                           with_continuation: list[int]) -> tuple[list[int],
                                                                 list[int]]:
        """Candidate span for language modeling: the continuation alone."""
        return [len(without)], [len(with_continuation)]


    def build_cases(vocab: int) -> tuple[list[ef.EvalCase], int]:
        """The synthetic CORE batch for the three candidate-span shapes."""
        cycle = _cycle(vocab)

        # multiple_choice: a shared prefix and continuations that differ at their
        # first token, so the common prefix is exactly the prompt.
        prefix = _pick(cycle, 0, 5)
        mc_sequences = [
            prefix + _pick(cycle, 5, 3),
            prefix + _pick(cycle, 11, 4),
        ]
        mc_starts, mc_ends = batch_sequences_mc(mc_sequences)

        # schema: different contexts and a shared continuation suffix. The token
        # before the suffix differs between the rows, so the common suffix is
        # exactly the continuation.
        suffix = _pick(cycle, 8, 3)
        schema_sequences = [
            _pick(cycle, 0, 8) + suffix,
            _pick(cycle, 11, 5) + suffix,
        ]
        schema_starts, schema_ends = batch_sequences_schema(schema_sequences)

        # language_modeling: the prompt without and with the continuation.
        lm_without = _pick(cycle, 0, 6)
        lm_with = lm_without + _pick(cycle, 6, 4)
        lm_starts, lm_ends = batch_sequences_lm(lm_without, lm_with)

        cases: list[ef.EvalCase] = []
        for tokens, start, end in zip(mc_sequences, mc_starts, mc_ends):
            cases.append(ef.EvalCase(tokens=tuple(tokens), start=start, end=end))
        for tokens, start, end in zip(schema_sequences, schema_starts,
                                      schema_ends):
            cases.append(ef.EvalCase(tokens=tuple(tokens), start=start, end=end))
        cases.append(ef.EvalCase(tokens=tuple(lm_with), start=lm_starts[0],
                                 end=lm_ends[0]))

        # A categorical focus at the answer position of the first multiple-choice
        # row and at the final language-modeling position.
        cases[0] = ef.EvalCase(
            tokens=cases[0].tokens, start=cases[0].start, end=cases[0].end,
            focus_position=mc_starts[0] - 1,
            focus_ids=(mc_sequences[0][mc_starts[0]], 3, 11, 32))
        cases[-1] = ef.EvalCase(
            tokens=cases[-1].tokens, start=cases[-1].start, end=cases[-1].end,
            focus_position=lm_ends[0] - 1,
            focus_ids=(lm_with[lm_ends[0] - 1], 0, 7, 19))

        seq = max(case.length for case in cases)
        return cases, seq


    def _fp32(name: str, shape: tuple[int, ...],
              values: list[float]) -> ef.Record:
        buf = array("f", (float(value) for value in values))
        if buf.itemsize != 4:
            raise SystemExit("this host's C float is not 32 bits")
        if sys.byteorder != "little":
            buf.byteswap()
        return ef.Record(name, ef.DTYPE_FP32, shape, buf.tobytes())


    def _i32(name: str, shape: tuple[int, ...],
             values: list[int]) -> ef.Record:
        buf = array("i", (int(value) for value in values))
        if buf.itemsize != 4:
            raise SystemExit("this host's C int is not 32 bits")
        if sys.byteorder != "little":
            buf.byteswap()
        return ef.Record(name, ef.DTYPE_INT32, shape, buf.tobytes())


    def reference_results(model: GPT, cases: list[ef.EvalCase],
                          batch: int, seq: int,
                          pad_id: int) -> list[ef.EvalResult]:
        """Runs the reference forward and returns the per-row scoring result.

        Mirrors ``nanochat.core_eval.forward_model``: the model output is the
        post-softcap logits, the targets are the input rolled left by one, and the
        per-position cross-entropy is read off with ``reduction='none'``. Positions
        at or past a row's valid length (padding and the row's final token) carry
        the masked loss and the ignore index.
        """
        tokens = torch.full((batch, seq), pad_id, dtype=torch.long)
        lengths: list[int] = []
        for row, case in enumerate(cases):
            tokens[row, :case.length] = torch.tensor(case.tokens, dtype=torch.long)
            lengths.append(case.length)

        with torch.no_grad():
            logits = model(tokens)  # (B, T, vocab), already soft-capped
            targets = torch.roll(tokens, shifts=-1, dims=1)
            losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            ).reshape(batch, seq)
            predictions = logits.argmax(dim=-1)

        losses = losses.to(torch.float32).cpu().numpy()
        predictions = predictions.cpu().numpy()
        logits = logits.to(torch.float32).cpu().numpy()

        results: list[ef.EvalResult] = []
        for row, case in enumerate(cases):
            nll = [float(losses[row, p]) for p in range(seq)]
            argmax = [int(predictions[row, p]) for p in range(seq)]
            for p in range(lengths[row], seq):
                nll[p] = ef.MASKED_NLL
                argmax[p] = ef.IGNORE_INDEX
            # The row's final token has no autoregressive target.
            final = lengths[row] - 1
            if final >= 0:
                nll[final] = ef.MASKED_NLL
                argmax[final] = ef.IGNORE_INDEX
            focus_logits: tuple[float, ...] = ()
            if case.has_focus:
                focus_logits = tuple(
                    float(logits[row, case.focus_position, token])
                    for token in case.focus_ids)
            results.append(ef.EvalResult(nll=tuple(nll), argmax=tuple(argmax),
                                         focus_logits=focus_logits))
        return results


    def _min_argmax_margin(model: GPT, cases: list[ef.EvalCase], batch: int,
                           seq: int, pad_id: int) -> float:
        """Smallest top-1 minus top-2 post-softcap logit over valid positions."""
        tokens = torch.full((batch, seq), pad_id, dtype=torch.long)
        for row, case in enumerate(cases):
            tokens[row, :case.length] = torch.tensor(case.tokens, dtype=torch.long)
        with torch.no_grad():
            logits = model(tokens)
        margin = float("inf")
        for row, case in enumerate(cases):
            for p in range(case.length - 1):
                top2 = torch.topk(logits[row, p], 2).values
                margin = min(margin, float(top2[0] - top2[1]))
        return margin


    def build_fixture(args: argparse.Namespace) -> Path:
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        config = build_plan_config(args.seq, args.vocab, args.layers, args.heads,
                              args.kv_heads, args.embd, args.window)
        model = GPT(config, pad_vocab_size_to=args.pad_to)
        model.init_weights()
        model.to(torch.float32)
        train_reference(model, args.train_steps, args.batch, args.seq, args.vocab)
        model.eval()

        cases, seq = build_cases(args.vocab)
        if seq > args.seq:
            raise SystemExit(
                f"the synthetic cases are {seq} tokens wide, wider than --seq "
                f"{args.seq}")
        # A too-small top-1 margin would make the exact argmax comparison flaky.
        margin = _min_argmax_margin(model, cases, len(cases), args.seq,
                                    args.pad_id)
        if margin < 1e-3:
            raise SystemExit(
                f"the trained classifier is not peaked enough (top-1 margin "
                f"{margin:.3e}); increase --train-steps")

        results = reference_results(model, cases, len(cases), args.seq,
                                    args.pad_id)

        records: list[ef.Record] = list(
            ef.fixture_to_records(cases, results, seq=args.seq,
                                  pad_id=args.pad_id))
        padded_vocab = model.lm_head.weight.shape[0]
        records += [
            _i32("config/layers", (1,), [args.layers]),
            _i32("config/heads", (1,), [args.heads]),
            _i32("config/kv_heads", (1,), [args.kv_heads]),
            _i32("config/embd", (1,), [args.embd]),
            _i32("config/vocab", (1,), [args.vocab]),
            _i32("config/padded_vocab", (1,), [padded_vocab]),
            _fp32("config/softcap", (1,), [15.0]),
        ]
        pattern = args.window.encode("ascii")
        records.append(ef.Record("config/window_pattern", ef.DTYPE_UINT8,
                                 (len(pattern),), pattern))
        for name, parameter in model.named_parameters():
            values = parameter.detach().to(torch.float32).cpu().numpy()
            records.append(_fp32(f"param/{name}", tuple(values.shape),
                                 values.reshape(-1).tolist()))

        ef.write(args.out, records)
        return Path(args.out)


    def parse_args(argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        add_model_argument(parser)
        add_check_argument(parser)
        parser.add_argument("--out", default=None,
                            help="output .bin path (default: "
                                 "models/<model>/fixtures/core_eval.bin)")
        parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
        parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
        parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
        parser.add_argument("--kv-heads", type=int, default=DEFAULT_KV_HEADS)
        parser.add_argument("--embd", type=int, default=DEFAULT_EMBD)
        parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
        parser.add_argument("--seq", type=int, default=DEFAULT_SEQ)
        parser.add_argument("--window", default="SL")
        parser.add_argument("--pad-to", type=int, default=DEFAULT_PAD_TO)
        parser.add_argument("--pad-id", type=int, default=DEFAULT_PAD_ID)
        parser.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS)
        parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
        args = parser.parse_args(argv)
        args.out = str(resolve_out(args, "core_eval.bin"))
        return args


    def main(argv: list[str]) -> int:
        args = parse_args(argv)
        if args.check:
            return check_generated("eval", args, build_fixture)
        if torch.cuda.is_available():
            raise RuntimeError(
                "the score fixture must be generated on the CPU; "
                "set CUDA_VISIBLE_DEVICES='' and retry")
        path = build_fixture(args)
        size = path.stat().st_size
        print(
            f"wrote {path} ({size} bytes, layers={args.layers}, "
            f"heads={args.heads}, kv_heads={args.kv_heads}, embd={args.embd}, "
            f"vocab={args.vocab}, seq={args.seq}, train_steps={args.train_steps})")
        return 0
    _selected_main = main

if _command == 'plan':
    __doc__ = textwrap.dedent("""\
        Generate the planning fixture, ``tests/data/plan_fixture.bin``.

        The fixture is *data*: it is produced once, offline, from the reference PyTorch
        nanochat checkout and committed to the tree. The runtime test never imports
        torch; it reads the fixture and compares the C-based planning layer
        (``python/nanochat_cpp/plan.py``) against it (docs/python.md sections 4.3 and
        11).

        For each case the fixture records:

        * the planning inputs (the arguments of ``compute_plan``);
        * the reference parameter counts from ``GPT.num_scaling_params`` and the
          reference ``GPT.num_matmul_params`` for the same config;
        * the reference ``GPT.estimate_flops`` value for one token;
        * the reference plan outputs, computed with the ``scripts/base_train.py``
          arithmetic.

        The whole file is little-endian JSON inside a small container::

            header : magic[8] = b"NANOPLAN", uint32 version
            body   : UTF-8 JSON

        The generation is CPU-only:

            /path/to/reference/.venv/bin/python tools/dump_model_fixtures.py plan         --out tests/data/plan_fixture.bin

    """)
    import argparse
    import json
    import math
    import os
    import struct
    import sys
    from pathlib import Path

    # Force the CPU reference before importing torch. The fixture is the oracle and
    # the host has one shared GPU that must stay free for the candidate run.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    MAGIC = b"NANOPLAN"
    VERSION = 1

    # Policy constants copied from scripts/base_train.py.
    B_REF = 2**19

    #: The planning cases. Each set of inputs is a ``compute_plan`` call.
    CASES = [
        {
            "depth": 8,
            "aspect_ratio": 64,
            "head_dim": 128,
            "seq_len": 512,
            "vocab_size": 32768,
            "window_pattern": "SSSL",
            "device_batch_size": 8,
            "total_batch_size": -1,
            "num_iterations": 0,
            "target_flops": 0,
            "target_param_data_ratio": 12,
        },
        {
            "depth": 12,
            "aspect_ratio": 64,
            "head_dim": 128,
            "seq_len": 2048,
            "vocab_size": 32768,
            "window_pattern": "SSSL",
            "device_batch_size": 32,
            "total_batch_size": 524288,
            "num_iterations": 100,
            "target_flops": 0,
            "target_param_data_ratio": 12,
        },
        {
            "depth": 20,
            "aspect_ratio": 64,
            "head_dim": 128,
            "seq_len": 2048,
            "vocab_size": 50304,
            "window_pattern": "SSL",
            "device_batch_size": 16,
            "total_batch_size": -1,
            "num_iterations": 0,
            "target_flops": 0,
            "target_param_data_ratio": 12,
        },
    ]

    #: Default base rates and schedule, copied from scripts/base_train.py.
    RATES = {
        "embedding_lr": 0.3,
        "unembedding_lr": 0.008,
        "weight_decay": 0.28,
        "matrix_lr": 0.02,
        "scalar_lr": 0.5,
        "warmup_steps": 40,
        "warmdown_ratio": 0.65,
        "final_lr_frac": 0.05,
    }


    def _reference_model(depth: int, aspect_ratio: int, head_dim: int,
                         seq_len: int, vocab_size: int, window_pattern: str):
        """A reference ``GPT`` on the meta device for the requested config."""
        import torch

        base_dim = depth * aspect_ratio
        model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
        num_heads = model_dim // head_dim
        config = GPTConfig(sequence_len=seq_len, vocab_size=vocab_size,
                           n_layer=depth, n_head=num_heads, n_kv_head=num_heads,
                           n_embd=model_dim, window_pattern=window_pattern)
        with torch.device("meta"):
            return GPT(config)


    def _reference_counts(depth: int, aspect_ratio: int, head_dim: int,
                          seq_len: int, vocab_size: int,
                          window_pattern: str) -> dict:
        model = _reference_model(depth, aspect_ratio, head_dim, seq_len,
                                 vocab_size, window_pattern)
        counts = model.num_scaling_params()
        return {
            "total": int(counts["total"]),
            "transformer_matrices": int(counts["transformer_matrices"]),
            "lm_head": int(counts["lm_head"]),
            "embeddings": int(counts["wte"] + counts["value_embeds"]),
            "scalars": int(counts["scalars"]),
            "num_matmul_params": int(model.num_matmul_params()),
            "flops_per_token": float(model.estimate_flops()),
        }


    def _reference_plan(case: dict) -> dict:
        """The reference plan outputs for one case, from base_train.py."""
        counts = _reference_counts(
            case["depth"], case["aspect_ratio"], case["head_dim"], case["seq_len"],
            case["vocab_size"], case["window_pattern"])
        reference = _reference_counts(
            12, case["aspect_ratio"], case["head_dim"], case["seq_len"],
            case["vocab_size"], case["window_pattern"])

        scaling = counts["transformer_matrices"] + counts["lm_head"]
        target_tokens = int(case["target_param_data_ratio"] * scaling)
        d_ref = case["target_param_data_ratio"] * (
            reference["transformer_matrices"] + reference["lm_head"])

        total_batch_size = case["total_batch_size"]
        if total_batch_size == -1:
            batch_size_ratio = target_tokens / d_ref
            predicted = B_REF * batch_size_ratio**0.383
            total_batch_size = 2 ** round(math.log2(predicted))

        batch_ratio = total_batch_size / B_REF
        batch_lr_scale = batch_ratio**0.5 if batch_ratio != 1.0 else 1.0
        weight_decay_scaled = (RATES["weight_decay"]
                               * math.sqrt(total_batch_size / B_REF)
                               * (d_ref / target_tokens))

        if case["num_iterations"] > 0:
            iterations = case["num_iterations"]
        elif case["target_flops"] > 0:
            iterations = round(
                case["target_flops"]
                / (counts["flops_per_token"] * total_batch_size))
        else:
            iterations = target_tokens // total_batch_size

        tokens_per_micro = case["device_batch_size"] * case["seq_len"]
        grad_accum = total_batch_size // tokens_per_micro

        padded_vocab = ((case["vocab_size"] + 63) // 64) * 64
        optimizer = {
            "embedding_lr": RATES["embedding_lr"] * batch_lr_scale,
            "unembedding_lr": RATES["unembedding_lr"] * batch_lr_scale,
            "matrix_lr": RATES["matrix_lr"] * batch_lr_scale,
            "scalar_lr": RATES["scalar_lr"] * batch_lr_scale,
            "weight_decay": weight_decay_scaled,
        }
        scheduler = {
            "warmup_steps": RATES["warmup_steps"],
            "warmdown_ratio": RATES["warmdown_ratio"],
            "final_lr_frac": RATES["final_lr_frac"],
            "weight_decay_base": weight_decay_scaled,
        }
        return {
            "padded_vocab_size": padded_vocab,
            "total_batch_size": int(total_batch_size),
            "grad_accum": int(grad_accum),
            "num_iterations": int(iterations),
            "target_tokens": int(target_tokens),
            "scaling_params": int(scaling),
            "batch_lr_scale": float(batch_lr_scale),
            "optimizer": optimizer,
            "scheduler": scheduler,
        }


    def main(argv: list[str] | None = None) -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        add_model_argument(parser)
        parser.add_argument("--out", default="tests/data/plan_fixture.bin")
        args = parser.parse_args(argv)

        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root / "python"))
        from nanochat_cpp.reference import ensure_reference_on_path  # noqa: E402

        ensure_reference_on_path()
        global GPT, GPTConfig
        from nanochat.gpt import GPT, GPTConfig  # noqa: E402

        cases = []
        for case in CASES:
            counts = _reference_counts(
                case["depth"], case["aspect_ratio"], case["head_dim"],
                case["seq_len"], case["vocab_size"], case["window_pattern"])
            cases.append({
                "inputs": dict(case),
                "counts": {key: counts[key] for key in (
                    "total", "transformer_matrices", "lm_head", "embeddings",
                    "scalars", "num_matmul_params")},
                "flops_per_token": counts["flops_per_token"],
                "plan": _reference_plan(case),
            })

        body = json.dumps({"version": VERSION, "cases": cases},
                          sort_keys=True, indent=2).encode("utf-8")
        payload = MAGIC + struct.pack("<I", VERSION) + body
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(payload)
        print(f"wrote {out} ({len(cases)} cases, {len(payload)} bytes)")
        return 0
    _selected_main = main

if _command == 'tokenizer':
    __doc__ = textwrap.dedent("""\
        Generate the reference tokenizer fixture, ``tests/data/tokenizer_fixture.bin``.

        The fixture is *data*: the script trains a small reference tokenizer with
        ``rustbpe`` and records the inference behavior of the matching ``tiktoken``
        encoding. Runtime C++ tests never import Python; they parse this file with
        ``tests/oracle_fixture.h``.

        What it contains
        ----------------
        * ``config/*``          the format version, the split pattern, the vocabulary
                                sizes, the merge hash, the corpus hash, and the
                                ``rustbpe`` commit and ``tiktoken`` version;
        * ``merges/*``          the ordered merges: the left and right ids, and the
                                token bytes of every merge;
        * ``tokens/*``          the token bytes of every base vocabulary token;
        * ``special/*``         the nine special token names and their ids;
        * ``split/*``           the reference split of a curated string set;
        * ``encode/*``          encode cases, including a leading or trailing special;
        * ``decode/*``          decode cases, including specials and partial UTF-8;
        * ``token_bytes/*``     the source byte length of every token.

        Binary format (little-endian)
        -----------------------------
        ::

            header   : magic[8] = b"NANOTOK1", uint32 version, uint32 num_records
            record   : uint16 name_len, name bytes,
                       uint8 dtype (1=int32, 4=uint8),
                       uint8 ndim, uint64 shape[ndim],
                       payload (product(shape) * dtype_size bytes)

        The container matches the ``NANOORC1`` oracle fixture, so the same reader
        (``tests/oracle_fixture.h``) parses both.

        Determinism
        -----------
        The corpus is fixed and the reference merge loop is deterministic. The script
        trains the reference twice and compares the two merge hashes. It fails when the
        hashes differ. The split check encodes each piece of the Python ``regex`` split
        and compares the result with the full ``tiktoken`` encoding for every curated
        string.

        The curated set stays inside the Unicode 16.0.0 letter, number, and whitespace
        ranges, so the Python split and the tiktoken split agree for these strings. The
        Python ``regex`` module may use a newer Unicode release, so do not add a
        version-sensitive code point to the curated set without a check.

        Run it from the repository root::

            python3 tools/dump_data_fixtures.py tokenizer
            python3 tools/dump_data_fixtures.py tokenizer --vocab-size 512 --out /tmp/f.bin

        The script needs ``rustbpe``, ``tiktoken``, and ``regex``. It never imports
        ``torch`` and never touches the GPU.

    """)
    from fixture_format import FixtureWriter
    import argparse
    import hashlib
    import struct
    import sys
    from pathlib import Path

    import regex
    import rustbpe
    import tiktoken

    # --- reference constants ---------------------------------------------------

    MERGE_BASE_VOCAB_SIZE = 256


    def recover_merge_pairs(
        mergeable_ranks: dict[bytes, int],
    ) -> list[tuple[int, int]]:
        """Recover the ordered merge pairs from a token-bytes-to-rank map.

        Rank ``i`` is the token id. The first 256 ranks are the single byte tokens.
        The function returns one ``(left, right)`` pair for every rank at or above
        256, in rank order.
        """
        ranks = dict(mergeable_ranks)
        inverse: dict[int, bytes] = {}
        for token, rank in ranks.items():
            if rank in inverse:
                raise ValueError(f"duplicate rank {rank}")
            inverse[rank] = bytes(token)
        if sorted(inverse) != list(range(len(ranks))):
            raise ValueError("ranks must be contiguous and start at zero")

        pairs: list[tuple[int, int]] = []
        for rank in range(MERGE_BASE_VOCAB_SIZE, len(ranks)):
            token = inverse[rank]
            candidates: list[tuple[int, int]] = []
            for split in range(1, len(token)):
                left = ranks.get(token[:split])
                right = ranks.get(token[split:])
                if left is None or right is None:
                    continue
                if left < rank and right < rank:
                    candidates.append((left, right))
            if not candidates:
                raise ValueError(f"no merge pair reconstructs rank {rank}")
            candidates.sort()
            pairs.append(candidates[0])
        return pairs


    #: The nine special tokens, in the reference order (docs/tokenizer.md 2.4).
    SPECIAL_TOKENS = [
        "<|bos|>",
        "<|user_start|>",
        "<|user_end|>",
        "<|assistant_start|>",
        "<|assistant_end|>",
        "<|python_start|>",
        "<|python_end|>",
        "<|output_start|>",
        "<|output_end|>",
    ]

    #: The nanochat split pattern from docs/tokenizer.md section 2.5.
    SPLIT_PATTERN = (
        r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}"
        r"| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
    )

    #: The pinned rustbpe source. The tag v0.1.0 of karpathy/rustbpe.
    RUSTBPE_COMMIT = "8bcf3131ab2ed157ac1510201e620cc36068aa66"
    RUSTBPE_VERSION = "0.1.0"

    DEFAULT_VOCAB_SIZE = 512

    #: A fixed English-like corpus. It is long enough to train the small fixture
    #: vocabulary and it exercises letters, digits, punctuation, and whitespace.
    CORPUS = [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!",
        "Sphinx of black quartz, judge my vow.",
        "The five boxing wizards jump quickly.",
        "Bright vixens jump; dozy fowl quack.",
        "Jinxed wizards pluck ivy from the big quilt.",
        "Waltz, bad nymph, for quick jigs vex.",
        "Quick zephyrs blow, vexing daft Jim.",
        "Two driven jocks help fax my big quiz.",
        "It was a bright cold day in April, and the clocks were striking thirteen.",
        "All happy families are alike; each unhappy family is unhappy in its own way.",
        "Call me Ishmael. Some years ago, never mind how long precisely.",
        "It is a truth universally acknowledged, that a single man in possession of "
        "a good fortune, must be in want of a wife.",
        "The sky above the port was the color of television, tuned to a dead channel.",
        "I've seen things you people wouldn't believe.",
        "Don't panic. The answer is 42, not 1234.",
        "We're going to need a bigger boat, aren't we?",
        'She said, "Hello, World!" and waved. Numbers: 1 22 333 4444.',
        "cafe naive resume jalapeno cooperate",
        "The value is 3.14 and 2,718, plus 100% of it.",
        "Line one\nline two\r\nline three\n",
        "   leading spaces and trailing spaces   ",
        "Tabs\tand\tspaces\t mixed together.",
        "Symbols: @#$%^&*()_+-=[]{}|;:',.<>/?`~",
        "def train_model(x, y): return x + y  # comment",
        "for i in range(10): print(i * 2)",
        "The rain in Spain stays mainly in the plain.",
        "Peter Piper picked a peck of pickled peppers.",
        "She sells seashells by the seashore.",
        "How much wood would a woodchuck chuck?",
        "A man, a plan, a canal: Panama!",
        "Was it a car or a cat I saw?",
        "Never odd or even. Madam, I'm Adam.",
        "The llama is a quadruped, related to the camel.",
        "Pack my box with five dozen liquor jugs, again.",
    ]

    #: A curated string set that covers the seven pattern alternatives and the
    #: known edge cases in docs/tokenizer.md section 6.
    CURATED_STRINGS = [
        # Apostrophe and contraction.
        "don't", "I'll", "we've", "they're", "he's", "won't", "Mary's", "isn't",
        # Optional symbol, then letters.
        " hello", " !abc", "  !abc", "-word", "@user", "_private",
        # One or two digits.
        "1", "12", "123", "1234", "3.14", "007",
        # Optional space, symbols, and line ends.
        "!!!", "??\n", "...", " \t  ", "```", "a!!",
        # Whitespace and line ends.
        "\n", "\r\n", "  \n", "a\nb", "one\r\ntwo",
        # Trailing whitespace.
        "word ", "a  ", "end\t",
        # Whitespace runs.
        "   ", "\t\t", " \t ",
        # Mixed text.
        "Hello, World!", "naive cafe", "\u65e5\u672c\u8a9e", "123 456", "",
    ]

    #: Encode cases: ``(text, prepend special name or None, append name or None)``.
    ENCODE_CASES = (
        [(text, None, None) for text in CURATED_STRINGS]
        + [
            ("Hello, World!", "<|bos|>", None),
            ("Hello, World!", None, "<|assistant_end|>"),
            ("", "<|bos|>", "<|assistant_end|>"),
            ("naive cafe", "<|bos|>", "<|assistant_end|>"),
            ("def f(x): return x", "<|python_start|>", "<|python_end|>"),
            ("user message", "<|user_start|>", "<|user_end|>"),
        ]
    )

    # Dtype codes shared with tests/oracle_fixture.h.
    _MAGIC = b"NANOTOK1"


    def _flatten(items: list[bytes]) -> tuple[bytes, list[int]]:
        """Concatenate byte strings and return the payload plus prefix offsets."""
        payload = bytearray()
        offsets = [0]
        for item in items:
            payload += item
            offsets.append(len(payload))
        return bytes(payload), offsets


    def _flatten_ids(rows: list[list[int]]) -> tuple[list[int], list[int]]:
        """Concatenate integer lists and return the payload plus prefix offsets."""
        payload: list[int] = []
        offsets = [0]
        for row in rows:
            payload.extend(row)
            offsets.append(len(payload))
        return payload, offsets


    def merge_hash(mergeable_ranks: list[tuple[bytes, int]]) -> str:
        """Hash the ordered merges (rank >= 256), so two runs can be compared."""
        digest = hashlib.sha256()
        for token, rank in mergeable_ranks:
            if rank < 256:
                continue
            digest.update(struct.pack("<II", rank, len(token)))
            digest.update(token)
        return digest.hexdigest()


    def train_reference(vocab_size: int) -> list[tuple[bytes, int]]:
        """Train the reference tokenizer and return the mergeable ranks."""
        base_vocab_size = vocab_size - len(SPECIAL_TOKENS)
        if base_vocab_size < 256:
            raise ValueError("vocab_size must leave at least 256 base tokens")
        tokenizer = rustbpe.Tokenizer()
        tokenizer.train_from_iterator(
            iter(CORPUS), base_vocab_size, pattern=SPLIT_PATTERN)
        if tokenizer.vocab_size != base_vocab_size:
            raise AssertionError(
                f"rustbpe returned {tokenizer.vocab_size} tokens, "
                f"expected {base_vocab_size}")
        pattern = tokenizer.get_pattern()
        if pattern != SPLIT_PATTERN:
            raise AssertionError("rustbpe returned a different split pattern")
        return list(tokenizer.get_mergeable_ranks())


    def build_encoding(
        mergeable_ranks: list[tuple[bytes, int]], vocab_size: int
    ) -> tiktoken.Encoding:
        """Build the matching reference tiktoken encoding."""
        ranks = {bytes(token): rank for token, rank in mergeable_ranks}
        offset = len(ranks)
        specials = {
            name: offset + index for index, name in enumerate(SPECIAL_TOKENS)
        }
        encoding = tiktoken.Encoding(
            name="nanochat-fixture",
            pat_str=SPLIT_PATTERN,
            mergeable_ranks=ranks,
            special_tokens=specials,
        )
        if encoding.n_vocab != vocab_size:
            raise AssertionError(
                f"tiktoken vocabulary is {encoding.n_vocab}, expected {vocab_size}")
        return encoding


    def check_split(encoding: tiktoken.Encoding) -> None:
        """Prove that the Python split agrees with the tiktoken split."""
        for text in CURATED_STRINGS:
            pieces = regex.findall(SPLIT_PATTERN, text)
            manual: list[int] = []
            for piece in pieces:
                manual.extend(encoding.encode_ordinary(piece))
            reference = encoding.encode_ordinary(text)
            if manual != reference:
                raise AssertionError(
                    f"split mismatch for {text!r}: {pieces} -> {manual} "
                    f"!= {reference}")


    def _resolve(value: int | str, specials: dict[str, int]) -> int:
        if isinstance(value, str):
            return specials[value]
        return value


    def decode_cases(specials: dict[str, int]) -> list[list[int]]:
        """Return the fixed decode cases, with specials resolved to ids."""
        raw: list[list[int | str]] = [
            [0xE2, 0x80],                      # partial UTF-8: one U+FFFD
            [0xE2, 0x82, 0xAC],                # complete euro sign
            [ord("h"), ord("i")],              # "hi"
            ["<|bos|>"],
            ["<|assistant_end|>"],
            ["<|user_start|>", ord("d"), "<|user_end|>"],
            ["<|bos|>", "<|user_start|>", ord("H"), ord("i"),
             "<|user_end|>", "<|assistant_start|>", ord("y"),
             "<|assistant_end|>"],
        ]
        return [[_resolve(value, specials) for value in row] for row in raw]


    def build_fixture(args: argparse.Namespace) -> tuple[Path, str]:
        first = train_reference(args.vocab_size)
        second = train_reference(args.vocab_size)
        first_hash = merge_hash(first)
        second_hash = merge_hash(second)
        if first_hash != second_hash:
            raise AssertionError(
                f"reference runs disagree: {first_hash} != {second_hash}")

        encoding = build_encoding(first, args.vocab_size)
        check_split(encoding)

        specials = {
            name: encoding.encode_single_token(name) for name in SPECIAL_TOKENS
        }
        base_vocab_size = len(first)
        mergeable = {bytes(token): rank for token, rank in first}
        merge_pairs = recover_merge_pairs(mergeable)
        if len(merge_pairs) != base_vocab_size - 256:
            raise AssertionError("merge pair count does not match the vocabulary")

        # Base token bytes, in rank order.
        inverse = {rank: bytes(token) for token, rank in first}
        base_bytes = [inverse[rank] for rank in range(base_vocab_size)]
        merge_bytes = [inverse[rank] for rank in range(256, base_vocab_size)]

        # Source byte length of every token. A special token has length zero.
        special_ids = set(specials.values())
        token_lengths = []
        for token_id in range(args.vocab_size):
            if token_id in special_ids:
                token_lengths.append(0)
            else:
                token_lengths.append(
                    len(encoding.decode_single_token_bytes(token_id)))

        # The reference split of the curated set.
        split_rows = [regex.findall(SPLIT_PATTERN, text) for text in CURATED_STRINGS]
        split_pieces = [piece.encode("utf-8") for row in split_rows for piece in row]
        string_piece_offsets = [0]
        for row in split_rows:
            string_piece_offsets.append(string_piece_offsets[-1] + len(row))

        # Encode cases.
        encode_inputs = [text.encode("utf-8") for text, _, _ in ENCODE_CASES]
        encode_ids: list[list[int]] = []
        encode_prepend: list[int] = []
        encode_append: list[int] = []
        for text, prepend_name, append_name in ENCODE_CASES:
            ids = list(encoding.encode_ordinary(text))
            prepend = -1 if prepend_name is None else specials[prepend_name]
            append = -1 if append_name is None else specials[append_name]
            if prepend >= 0:
                ids.insert(0, prepend)
            if append >= 0:
                ids.append(append)
            encode_ids.append(ids)
            encode_prepend.append(prepend)
            encode_append.append(append)

        # Decode cases.
        raw_decode = decode_cases(specials)
        decode_outputs = [encoding.decode(row).encode("utf-8") for row in raw_decode]

        corpus_text = "\n".join(CORPUS).encode("utf-8")
        pattern_bytes = SPLIT_PATTERN.encode("utf-8")
        commit_bytes = RUSTBPE_COMMIT.encode("ascii")
        version_bytes = tiktoken.__version__.encode("ascii")
        names_payload, names_offsets = _flatten(
            [name.encode("utf-8") for name in SPECIAL_TOKENS])
        strings_payload, strings_offsets = _flatten(
            [text.encode("utf-8") for text in CURATED_STRINGS])
        pieces_payload, piece_offsets = _flatten(split_pieces)
        encode_inputs_payload, encode_input_offsets = _flatten(encode_inputs)
        encode_ids_payload, encode_id_offsets = _flatten_ids(encode_ids)
        decode_ids_payload, decode_id_offsets = _flatten_ids(raw_decode)
        decode_outputs_payload, decode_output_offsets = _flatten(decode_outputs)
        base_bytes_payload, base_bytes_offsets = _flatten(base_bytes)
        merge_bytes_payload, merge_bytes_offsets = _flatten(merge_bytes)

        writer = FixtureWriter(args.out, magic=_MAGIC)
        writer.add_int32("config/format_version", [1])
        writer.add_uint8("config/pattern", pattern_bytes)
        writer.add_uint8("config/rustbpe_commit", commit_bytes)
        writer.add_uint8("config/rustbpe_version", RUSTBPE_VERSION.encode("ascii"))
        writer.add_uint8("config/tiktoken_version", version_bytes)
        writer.add_uint8("config/merge_hash", first_hash.encode("ascii"))
        writer.add_uint8("config/corpus_sha256",
                         hashlib.sha256(corpus_text).hexdigest().encode("ascii"))
        writer.add_int32("config/vocab_size", [args.vocab_size])
        writer.add_int32("config/base_vocab_size", [base_vocab_size])
        writer.add_int32("config/special_count", [len(SPECIAL_TOKENS)])
        writer.add_int32("config/merge_count", [len(merge_pairs)])

        writer.add_int32("merges/ranks",
                         [256 + index for index in range(len(merge_pairs))])
        writer.add_int32("merges/left", [left for left, _ in merge_pairs])
        writer.add_int32("merges/right", [right for _, right in merge_pairs])
        writer.add_int32("merges/token_bytes_offsets", merge_bytes_offsets)
        writer.add_uint8("merges/token_bytes", merge_bytes_payload)

        writer.add_int32("tokens/bytes_offsets", base_bytes_offsets)
        writer.add_uint8("tokens/bytes", base_bytes_payload)

        writer.add_int32("special/names_offsets", names_offsets)
        writer.add_uint8("special/names", names_payload)
        writer.add_int32("special/ids",
                         [specials[name] for name in SPECIAL_TOKENS])

        writer.add_int32("split/strings_offsets", strings_offsets)
        writer.add_uint8("split/strings", strings_payload)
        writer.add_int32("split/string_piece_offsets", string_piece_offsets)
        writer.add_int32("split/piece_offsets", piece_offsets)
        writer.add_uint8("split/pieces", pieces_payload)

        writer.add_int32("encode/input_offsets", encode_input_offsets)
        writer.add_uint8("encode/inputs", encode_inputs_payload)
        writer.add_int32("encode/prepend", encode_prepend)
        writer.add_int32("encode/append", encode_append)
        writer.add_int32("encode/id_offsets", encode_id_offsets)
        writer.add_int32("encode/ids", encode_ids_payload)

        writer.add_int32("decode/id_offsets", decode_id_offsets)
        writer.add_int32("decode/ids", decode_ids_payload)
        writer.add_int32("decode/output_offsets", decode_output_offsets)
        writer.add_uint8("decode/outputs", decode_outputs_payload)

        writer.add_uint8("token_bytes/lengths", bytes(token_lengths))

        writer.write()
        return args.out, first_hash


    def parse_args(argv: list[str]) -> argparse.Namespace:
        root = Path(__file__).resolve().parents[1]
        parser = argparse.ArgumentParser(
            description="Generate the reference tokenizer fixture.")
        add_model_argument(parser)
        parser.add_argument(
            "--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE,
            help="full vocabulary size, including the nine special tokens")
        parser.add_argument(
            "--out", type=Path,
            default=root / "tests" / "data" / "tokenizer_fixture.bin",
            help="output fixture path")
        return parser.parse_args(argv)


    def main(argv: list[str]) -> int:
        args = parse_args(argv)
        path, merge_digest = build_fixture(args)
        payload = path.read_bytes()
        print(f"wrote {path} ({len(payload)} bytes)")
        print(f"merge hash: {merge_digest}")
        print(f"fixture sha256: {hashlib.sha256(payload).hexdigest()}")
        return 0
    _selected_main = main

if _command == 'loader':
    __doc__ = textwrap.dedent("""\
        Dump the loader-parity fixture (docs/parquet-native.md phase 8).

        The fixture pins the C++ document-mode loader against the reference
        ``tokenizing_distributed_data_loader_bos_bestfit`` (``nanochat/dataloader.py``).
        Only the document source is replaced; the reference packing loop runs
        unchanged.

        The tokenizer comes from a portable ``NCTOKEN1`` artifact, so the C++ side and
        the ``tiktoken`` side use the same merge list. Run this tool in the reference
        virtual environment (torch, tiktoken) because ``nanochat.dataloader`` imports
        torch.

            # 1. Dump the fixed document corpus for the trainer.
            python tools/dump_data_fixtures.py loader --dump-docs /tmp/loader_docs.txt

            # 2. Build the NCTOKEN1 artifact with the native trainer.
            tools/nanochat run t0-cpu -- <tok_train_main> --text /tmp/loader_docs.txt \
                --vocab-size 512 --out tests/data/loader_tokenizer.nctoken

            # 3. Write the fixture with the reference venv.
            /home/egrader/repos/nanochat/.venv/bin/python tools/dump_data_fixtures.py loader \
                --tokenizer tests/data/loader_tokenizer.nctoken \
                --out tests/data/loader_parity.bin

        The output format is little-endian:

            magic       8 bytes  "NCLDRP01"
            version     u32      1
            batch       u32
            seq         u32
            buffer      u32      the encoded-document buffer size
            source_batch u32     documents per source batch
            doc_count   u32
            documents   doc_count * (u32 length, bytes)
            steps       u32
            rows        steps * (batch*seq i32 tokens, batch*seq i32 targets)

    """)
    import argparse
    import struct
    import sys
    from pathlib import Path

    MAGIC = b"NCLDRP01"
    VERSION = 1

    # The fixed loader shape. `buffer` and `source_batch` must match the C++ test.
    BATCH = 2
    SEQ = 16
    BUFFER = 16
    SOURCE_BATCH = 256
    STEPS = 3
    DOC_COUNT = 256

    BASE_DOCS = [
        "The quick brown fox jumps over the lazy dog.",
        "a",
        "ab cd ef",
        "Caf\u00e9 \u2615 time.",
        "Short and sweet.",
        "A document that is intentionally very long so that it exceeds one single "
        "row and must be cropped somewhere in the middle of the text.",
        "0123456789",
        "Hello, world!",
    ]


    def documents() -> list[str]:
        """The fixed corpus: every base document with a unique suffix."""
        return [f"{BASE_DOCS[i % len(BASE_DOCS)]} [{i}]" for i in range(DOC_COUNT)]


    def dump_docs(path: Path) -> None:
        path.write_text("\n".join(documents()) + "\n", encoding="utf-8")
        print(f"wrote {DOC_COUNT} documents to {path}")


    def reference_rows(reference: Path, bridge: Path, tokenizer_path: Path):
        """Runs the reference best-fit loader over the fixed corpus."""
        sys.path.insert(0, str(reference))
        sys.path.insert(0, str(bridge))
        import torch
        from nanochat import dataloader
        # The ``tiktoken`` tokenizer over an NCTOKEN1 artifact lives in the
        # adapter package; the core's reader, ``nanochat_cpp.data.read_nctoken1``,
        # stays in the standard library only (docs/dependency-layering-plan.md
        # sections 2.1 and 5).
        from nanochat_cpp_bridge.reference_data import load_nctoken1

        tokenizer = load_nctoken1(str(tokenizer_path))
        docs = documents()

        def fake_document_batches(split, resume_state_dict, tokenizer_batch_size):
            while True:
                for start in range(0, len(docs), tokenizer_batch_size):
                    batch = docs[start:start + tokenizer_batch_size]
                    yield batch, (0, 0, 1)

        dataloader._document_batches = fake_document_batches

        loader = dataloader.tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, BATCH, SEQ, split="train",
            tokenizer_threads=1, tokenizer_batch_size=SOURCE_BATCH,
            device="cpu", buffer_size=BUFFER)
        rows = []
        for _ in range(STEPS):
            inputs, targets = next(loader)
            assert inputs.shape == (BATCH, SEQ), inputs.shape
            rows.append((inputs.flatten().tolist(), targets.flatten().tolist()))
        return tokenizer.get_vocab_size(), rows


    def write_fixture(path: Path, rows) -> None:
        docs = documents()
        with path.open("wb") as handle:
            handle.write(MAGIC)
            handle.write(struct.pack("<IIIII", VERSION, BATCH, SEQ, BUFFER,
                                     SOURCE_BATCH))
            handle.write(struct.pack("<I", len(docs)))
            for doc in docs:
                data = doc.encode("utf-8")
                handle.write(struct.pack("<I", len(data)))
                handle.write(data)
            handle.write(struct.pack("<I", len(rows)))
            for tokens, targets in rows:
                handle.write(struct.pack(f"<{len(tokens)}i", *tokens))
                handle.write(struct.pack(f"<{len(targets)}i", *targets))
        print(f"wrote {len(rows)} steps of {BATCH}x{SEQ} to {path}")


    def main(argv: list[str]) -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        add_model_argument(parser)
        parser.add_argument("--dump-docs", type=Path,
                            help="write the fixed corpus and exit")
        parser.add_argument("--tokenizer", type=Path,
                            help="the NCTOKEN1 artifact")
        parser.add_argument("--out", type=Path, help="the fixture to write")
        parser.add_argument("--reference", type=Path,
                            default=Path("/home/egrader/repos/nanochat"),
                            help="the reference nanochat checkout")
        parser.add_argument("--bridge", type=Path, default=Path("python"),
                            help="the directory that holds the nanochat_cpp bridge")
        args = parser.parse_args(argv)

        if args.dump_docs is not None:
            dump_docs(args.dump_docs)
            return 0
        if args.tokenizer is None or args.out is None:
            parser.error("give --dump-docs, or --tokenizer and --out")

        vocab_size, rows = reference_rows(args.reference, args.bridge,
                                          args.tokenizer)
        print(f"reference tokenizer vocab size: {vocab_size}")
        write_fixture(args.out, rows)
        return 0
    _selected_main = main

if _command == 'chat':
    __doc__ = textwrap.dedent("""\
        Generate the chat prompt fixture, ``tests/data/chat_fixture.bin``.

        The fixture pins the completion prompt that the reference
        ``nanochat.tokenizer.Tokenizer.render_for_completion`` produces for a few
        synthetic conversations. The test ``python/tests/chat_test.py`` renders the
        same conversations with :class:`nanochat_cpp.chat._PromptRenderer` and compares
        the ids. See docs/python.md sections 5 and 11.

        The file is a small container::

            header : magic[8] = b"NANCHAT1", uint32 version
            body   : UTF-8 JSON

        The generation needs the reference package and ``tiktoken`` only:

            /path/to/reference/.venv/bin/python tools/dump_data_fixtures.py chat         --out tests/data/chat_fixture.bin

    """)
    import argparse
    import json
    import struct
    import sys
    from pathlib import Path

    MAGIC = b"NANCHAT1"
    VERSION = 1

    #: The committed tokenizer artifact the fixture is rendered with.
    TOKENIZER = "tests/data/loader_tokenizer.nctoken"

    #: Synthetic conversations. Each renders with the reference template.
    CASES = [
        {
            "name": "simple",
            "conversation": {
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "Paris"},
                ],
            },
        },
        {
            "name": "system",
            "conversation": {
                "messages": [
                    {"role": "system",
                     "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Name a color."},
                    {"role": "assistant", "content": "Blue"},
                ],
            },
        },
        {
            "name": "tool_parts",
            "conversation": {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": [
                        {"type": "python", "text": "2+2"},
                        {"type": "python_output", "text": "4"},
                        {"type": "text", "text": " #### 4"},
                    ]},
                    {"role": "user", "content": "Now double it."},
                    {"role": "assistant", "content": "#### 8"},
                ],
            },
        },
        {
            "name": "multiple_choice",
            "conversation": {
                "messages": [
                    {"role": "user", "content": (
                        "Multiple Choice question: What is 1+1?\n"
                        "- 1=A\n- 2=B\n\n"
                        "Respond only with the letter of the correct answer.")},
                    {"role": "assistant", "content": "B"},
                ],
            },
        },
    ]


    def main(argv: list[str] | None = None) -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        add_model_argument(parser)
        parser.add_argument("--out", default="tests/data/chat_fixture.bin")
        parser.add_argument("--tokenizer", default=TOKENIZER)
        args = parser.parse_args(argv)

        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root / "python"))

        from nanochat_cpp import data  # noqa: E402
        from nanochat_cpp.reference import ensure_reference_on_path  # noqa: E402

        ensure_reference_on_path()
        from nanochat.tokenizer import RustBPETokenizer  # noqa: E402
        # ``build_tiktoken_encoding`` needs ``tiktoken``, so it lives in the
        # adapter package; the artifact reader is the core's (section 5).
        from nanochat_cpp_bridge import reference_data  # noqa: E402

        artifact = data.read_nctoken1(repo_root / args.tokenizer)
        encoding = reference_data.build_tiktoken_encoding(artifact)
        tokenizer = RustBPETokenizer(encoding, "<|bos|>")

        cases = []
        for case in CASES:
            prompt_ids = tokenizer.render_for_completion(
                json.loads(json.dumps(case["conversation"])))
            cases.append({
                "name": case["name"],
                "conversation": case["conversation"],
                "prompt_ids": [int(token) for token in prompt_ids],
            })

        body = json.dumps({"version": VERSION, "tokenizer": args.tokenizer,
                           "cases": cases},
                          sort_keys=True, indent=2).encode("utf-8")
        payload = MAGIC + struct.pack("<I", VERSION) + body
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(payload)
        print(f"wrote {out} ({len(cases)} cases, {len(payload)} bytes)")
        return 0
    _selected_main = main

if _command == 'api':
    __doc__ = textwrap.dedent("""\
        Generate the Python-API fixture, ``tests/data/api_fixture.bin``.

        The in-process Python API must reproduce the C++ trainer and the C++
        evaluator. This script records the reference values for that check. It runs
        the committed CPU binaries through ``tools/nanochat``:

        * ``train_main`` trains a tiny model on the committed document set
          ``src/parquet/testdata/text.parquet`` with the committed tokenizer
          ``tests/data/loader_tokenizer.nctoken``, and writes a checkpoint;
        * ``eval_main`` loads that checkpoint and prints the bits-per-byte result.

        The script records the model config, the optimizer config, the scheduler
        config, the seed, the data paths, the per-step loss, and the bits-per-byte
        result. The container is the little-endian ``NANOEVL1`` record format owned by
        ``python/nanochat_cpp/eval_fixture.py``. Every record name starts with ``api/``
        so the generic record reader stays separate from the semantic evaluation
        reader.

        The fixture is deterministic. A fixed seed and one tokenizer thread remove the
        run-to-run variation. Two runs give the same file.

        The parameter trajectory and the synthetic optimizer check already live in
        ``tests/data/train_parity.bin``. This fixture records only the document-mode
        values, and it names that file in the ``api/data/train_parity`` record. The
        optimizer and scheduler values below copy the committed train-parity fixture,
        so the two files describe the same update rule.

        Run the generator from the repository root::

            tools/nanochat build //src:train_main //src:eval_main
            python3 tools/dump_runtime_fixtures.py api --out tests/data/api_fixture.bin

        Read the file with ``nanochat_cpp.eval_fixture.read`` and index the records by
        name. Do not call ``records_to_fixture``: the ``api/`` records are not the
        evaluation schema.

    """)
    import argparse
    import os
    import re
    import struct
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    # The repository root holds the `tools/` entry point and the `python/` package.
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT / "python") not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT / "python"))

    from nanochat_cpp import eval_fixture as ef  # noqa: E402
    from nanochat_cpp.data import read_nctoken1  # noqa: E402

    # The fixture schema version, recorded as `api/version`.
    API_FIXTURE_VERSION = 1

    # A tiny, structurally complete model. It copies the committed train-parity
    # architecture. Two layers use grouped-query attention and a sliding-window
    # pattern. The tokenizer vocabulary has 487 tokens. The padded vocabulary is
    # the next multiple of 64.
    DEFAULT_SEED = 1234
    DEFAULT_LAYERS = 2
    DEFAULT_HEADS = 8
    DEFAULT_KV_HEADS = 2
    DEFAULT_HIDDEN = 32
    DEFAULT_SEQ = 8
    DEFAULT_BATCH = 2
    DEFAULT_GRAD_ACCUM = 1
    DEFAULT_STEPS = 4
    DEFAULT_EVAL_STEPS = 3
    DEFAULT_WINDOW = "SL"
    #: One encode worker keeps the token stream deterministic.
    DEFAULT_TOKENIZER_THREADS = 1
    #: The committed default encoded-document buffer.
    DEFAULT_DOCUMENT_BUFFER = 1000

    # The optimizer and scheduler values of the committed train-parity fixture.
    # tests/data/train_parity.bin already covers these values.
    OPTIMIZER = {
        "unembedding_lr": 0.004,
        "embedding_lr": 0.2,
        "matrix_lr": 0.02,
        "scalar_lr": 0.5,
        "weight_decay": 0.0,
        "clip": 0.0,
        "adam_eps": 1e-4,
        "muon_ns_steps": 5,
        "muon_beta2": 0.9,
    }
    SCHEDULER = {
        "warmup_steps": 2,
        "warmdown_ratio": 0.5,
        "final_lr_frac": 0.1,
        "weight_decay_base": 0.05,
        "muon_momentum_warmup_steps": 4.0,
        "muon_momentum_start": 0.85,
        "muon_momentum_peak": 0.97,
        "muon_momentum_final": 0.90,
    }

    # The recorded reference for the parameter trajectory.
    TRAIN_PARITY_PATH = "tests/data/train_parity.bin"

    #: One ``step NNNNNN | loss X | ...`` line from the training log.
    _STEP_LOSS = re.compile(
        r"^step\s+(?P<step>\d+)\s+\|\s+loss\s+(?P<loss>[0-9.eE+-]+)\s+\|")
    #: The training summary line.
    _FINAL_LOSS = re.compile(
        r"^train_main: final loss (?P<loss>[0-9.eE+-]+) at step (?P<step>\d+)",
        re.MULTILINE)
    #: One ``eval_main: <split> bpb X`` line from the evaluator.
    _BPB = re.compile(r"^eval_main: (?P<split>train|val) bpb (?P<bpb>[0-9.eE+-]+)",
                      re.MULTILINE)


    def _i32(values) -> bytes:
        values = [int(value) for value in values]
        return struct.pack("<" + "i" * len(values), *values)


    def _i64(values) -> bytes:
        values = [int(value) for value in values]
        return struct.pack("<" + "q" * len(values), *values)


    def _f32(values) -> bytes:
        values = [float(value) for value in values]
        return struct.pack("<" + "f" * len(values), *values)


    def _bytes_record(name: str, text: str) -> ef.Record:
        """A UTF-8 string record, used for the paths."""
        payload = text.encode("utf-8")
        return ef.Record(name, ef.DTYPE_UINT8, (len(payload),), payload)


    def _int_record(name: str, value: int) -> ef.Record:
        return ef.Record(name, ef.DTYPE_INT32, (1,), _i32([value]))


    def _float_record(name: str, value: float) -> ef.Record:
        return ef.Record(name, ef.DTYPE_FP32, (1,), _f32([value]))


    def _vector_record(name: str, values) -> ef.Record:
        values = [float(value) for value in values]
        return ef.Record(name, ef.DTYPE_FP32, (len(values),), _f32(values))


    def _padded_vocab(vocab_size: int, multiple: int = 64) -> int:
        return ((vocab_size + multiple - 1) // multiple) * multiple


    def _relative(repo_root: Path, path: Path) -> str:
        """A stable repository-relative path, or the absolute path as a fallback."""
        try:
            return path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            return path.resolve().as_posix()


    def vocab_size(tokenizer_path: Path) -> int:
        """The token count of an ``NCTOKEN1`` artifact."""
        artifact = read_nctoken1(tokenizer_path)
        return 256 + len(artifact.merge_pairs) + len(artifact.special_tokens)


    def run_nanochat(repo_root: Path, arguments) -> subprocess.CompletedProcess:
        """Run one command through the single entry point and capture its output."""
        command = [str(repo_root / "tools" / "nanochat"), *arguments]
        result = subprocess.run(command, cwd=str(repo_root), capture_output=True,
                                text=True)
        if result.returncode != 0:
            sys.stderr.write(result.stdout)
            sys.stderr.write(result.stderr)
            raise SystemExit(
                f"command failed ({result.returncode}): " + " ".join(command))
        return result


    def build_binaries(repo_root: Path, build: bool) -> None:
        if build:
            run_nanochat(repo_root, ["build", "//src:train_main", "//src:eval_main"])


    def binary_path(repo_root: Path, name: str) -> Path:
        path = repo_root / "bazel-bin" / "src" / name
        if not path.is_file():
            raise SystemExit(
                f"missing built binary {path}; run "
                "tools/nanochat build //src:train_main //src:eval_main")
        return path


    def model_flags(args: argparse.Namespace, vocab: int) -> list[str]:
        return [
            "--layers", str(args.layers),
            "--heads", str(args.heads),
            "--kv-heads", str(args.kv_heads),
            "--hidden", str(args.hidden),
            "--seq", str(args.seq),
            "--vocab", str(vocab),
            "--padded-vocab", str(_padded_vocab(vocab)),
            "--window-pattern", args.window,
        ]


    def optimizer_flags() -> list[str]:
        flags: list[str] = []
        for key in ("unembedding_lr", "embedding_lr", "matrix_lr", "scalar_lr",
                    "weight_decay", "clip", "adam_eps", "muon_ns_steps",
                    "muon_beta2"):
            flags += ["--" + key.replace("_", "-"), repr(OPTIMIZER[key])]
        return flags


    def scheduler_flags() -> list[str]:
        flags: list[str] = []
        for key in ("warmup_steps", "warmdown_ratio", "final_lr_frac",
                    "weight_decay_base", "muon_momentum_warmup_steps",
                    "muon_momentum_start", "muon_momentum_peak",
                    "muon_momentum_final"):
            flags += ["--" + key.replace("_", "-"), repr(SCHEDULER[key])]
        return flags


    def run_train(repo_root: Path, train_main: Path, parquet: Path,
                  tokenizer: Path, args: argparse.Namespace, vocab: int,
                  checkpoint: Path) -> tuple[list[tuple[int, float]], int, float]:
        """Train the tiny model and return the per-step losses and the final loss."""
        train_args = [
            "--train-parquet", str(parquet),
            "--tokenizer", str(tokenizer),
            "--tokenizer-threads", str(args.tokenizer_threads),
            "--buffer-docs", str(args.document_buffer),
            *model_flags(args, vocab),
            *optimizer_flags(),
            *scheduler_flags(),
            "--batch", str(args.batch),
            "--grad-accum", str(args.grad_accum),
            "--num-iterations", str(args.steps),
            "--log-every", "1",
            "--save-every", "0",
            "--eval-every", "0",
            "--seed", str(args.seed),
            "--checkpoint", str(checkpoint),
        ]
        result = run_nanochat(repo_root, ["run", "t0-cpu", "--", str(train_main),
                                          *train_args])
        steps: list[tuple[int, float]] = []
        for line in result.stdout.splitlines():
            step_match = _STEP_LOSS.match(line)
            if step_match:
                steps.append((int(step_match.group("step")),
                              float(step_match.group("loss"))))
        steps.sort()
        if len(steps) != args.steps:
            raise SystemExit(
                f"expected {args.steps} loss lines, found {len(steps)}")
        final_match = _FINAL_LOSS.search(result.stdout)
        if final_match is None:
            raise SystemExit("the training output has no final-loss line")
        final_step = int(final_match.group("step"))
        final_loss = float(final_match.group("loss"))
        return steps, final_step, final_loss


    def run_eval(repo_root: Path, eval_main: Path, parquet: Path,
                 tokenizer: Path, args: argparse.Namespace, vocab: int,
                 checkpoint: Path) -> dict[str, float]:
        """Evaluate the checkpoint and return the bits-per-byte for each split."""
        eval_args = [
            "--train-parquet", str(parquet),
            "--val-parquet", str(parquet),
            "--tokenizer", str(tokenizer),
            "--tokenizer-threads", str(args.tokenizer_threads),
            "--buffer-docs", str(args.document_buffer),
            *model_flags(args, vocab),
            "--batch", str(args.batch),
            "--steps", str(args.eval_steps),
            "--model", str(checkpoint),
            "--seed", str(args.seed),
        ]
        result = run_nanochat(repo_root, ["run", "t0-cpu", "--", str(eval_main),
                                          *eval_args])
        values: dict[str, float] = {}
        for line in result.stdout.splitlines():
            match = _BPB.match(line)
            if match:
                values[match.group("split")] = float(match.group("bpb"))
        missing = [split for split in ("train", "val") if split not in values]
        if missing:
            raise SystemExit(f"the evaluator output has no bpb for: {missing}")
        return values


    def build_records(args: argparse.Namespace, vocab: int,
                      steps: list[tuple[int, float]], final_step: int,
                      final_loss: float, bpb: dict[str, float],
                      train_parquet: str, val_parquet: str,
                      tokenizer: str) -> list[ef.Record]:
        records: list[ef.Record] = []
        # Header and identity.
        records.append(_int_record("api/version", API_FIXTURE_VERSION))
        records.append(ef.Record("api/seed", ef.DTYPE_INT64, (1,),
                                 _i64([args.seed])))
        # Data paths, repository-relative so the file is portable.
        records.append(_bytes_record("api/data/train_parquet", train_parquet))
        records.append(_bytes_record("api/data/val_parquet", val_parquet))
        records.append(_bytes_record("api/data/tokenizer", tokenizer))
        records.append(_bytes_record("api/data/train_parity", TRAIN_PARITY_PATH))
        # Model config.
        records.append(_int_record("api/config/layers", args.layers))
        records.append(_int_record("api/config/heads", args.heads))
        records.append(_int_record("api/config/kv_heads", args.kv_heads))
        records.append(_int_record("api/config/hidden", args.hidden))
        records.append(_int_record("api/config/seq", args.seq))
        records.append(_int_record("api/config/vocab", vocab))
        records.append(_int_record("api/config/padded_vocab",
                                   _padded_vocab(vocab)))
        records.append(_int_record("api/config/batch", args.batch))
        records.append(_int_record("api/config/grad_accum", args.grad_accum))
        records.append(_int_record("api/config/steps", args.steps))
        records.append(_bytes_record("api/config/window_pattern", args.window))
        records.append(_int_record("api/config/tokenizer_threads",
                                   args.tokenizer_threads))
        records.append(_int_record("api/config/document_buffer",
                                   args.document_buffer))
        # Optimizer config (the values of tests/data/train_parity.bin).
        for key in ("unembedding_lr", "embedding_lr", "matrix_lr", "scalar_lr",
                    "weight_decay", "clip", "adam_eps", "muon_beta2"):
            records.append(_float_record(f"api/config/opt/{key}",
                                         OPTIMIZER[key]))
        records.append(_int_record("api/config/opt/muon_ns_steps",
                                   OPTIMIZER["muon_ns_steps"]))
        # Scheduler config (the values of tests/data/train_parity.bin).
        records.append(_int_record("api/config/sched/warmup_steps",
                                   SCHEDULER["warmup_steps"]))
        for key in ("warmdown_ratio", "final_lr_frac", "weight_decay_base",
                    "muon_momentum_warmup_steps", "muon_momentum_start",
                    "muon_momentum_peak", "muon_momentum_final"):
            records.append(_float_record(f"api/config/sched/{key}",
                                         SCHEDULER[key]))
        # Training results.
        records.append(_int_record("api/train/steps", args.steps))
        records.append(_vector_record("api/train/loss",
                                      [loss for _, loss in steps]))
        records.append(_int_record("api/train/final_step", final_step))
        records.append(_float_record("api/train/final_loss", final_loss))
        # Evaluation results.
        records.append(_int_record("api/eval/steps", args.eval_steps))
        records.append(_float_record("api/eval/train/bpb", bpb["train"]))
        records.append(_float_record("api/eval/val/bpb", bpb["val"]))
        return records


    def parse_args(argv) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        add_model_argument(parser)
        parser.add_argument("--out", default="tests/data/api_fixture.bin",
                            help="output fixture path")
        parser.add_argument("--parquet",
                            default="src/parquet/testdata/text.parquet",
                            help="the committed document set")
        parser.add_argument("--tokenizer",
                            default="tests/data/loader_tokenizer.nctoken",
                            help="the committed NCTOKEN1 artifact")
        parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
        parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
        parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
        parser.add_argument("--kv-heads", type=int, default=DEFAULT_KV_HEADS)
        parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
        parser.add_argument("--seq", type=int, default=DEFAULT_SEQ)
        parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
        parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
        parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
        parser.add_argument("--eval-steps", type=int, default=DEFAULT_EVAL_STEPS)
        parser.add_argument("--window", default=DEFAULT_WINDOW)
        parser.add_argument("--tokenizer-threads", type=int,
                            default=DEFAULT_TOKENIZER_THREADS)
        parser.add_argument("--document-buffer", type=int,
                            default=DEFAULT_DOCUMENT_BUFFER)
        parser.add_argument("--no-build", action="store_true",
                            help="use the current bazel-bin binaries")
        return parser.parse_args(argv)


    def main(argv) -> int:
        args = parse_args(argv)
        # The fixture is the CPU reference. Keep the shared GPU free.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        repo_root = _REPO_ROOT
        parquet = (repo_root / args.parquet).resolve()
        tokenizer = (repo_root / args.tokenizer).resolve()
        out = Path(args.out or DEFAULT_OUT_BY_PRECISION[args.precision])
        if not out.is_absolute():
            out = repo_root / out
        for path, what in ((parquet, "parquet"), (tokenizer, "tokenizer")):
            if not path.is_file():
                raise SystemExit(f"missing {what} file: {path}")

        vocab = vocab_size(tokenizer)
        build_binaries(repo_root, build=not args.no_build)
        train_main = binary_path(repo_root, "train_main")
        eval_main = binary_path(repo_root, "eval_main")

        with tempfile.TemporaryDirectory(prefix="api_fixture_") as workdir:
            checkpoint = Path(workdir) / "model.nchkpt"
            steps, final_step, final_loss = run_train(
                repo_root, train_main, parquet, tokenizer, args, vocab, checkpoint)
            bpb = run_eval(repo_root, eval_main, parquet, tokenizer, args, vocab,
                           checkpoint)

        train_parquet = _relative(repo_root, parquet)
        tokenizer_path = _relative(repo_root, tokenizer)
        records = build_records(args, vocab, steps, final_step, final_loss, bpb,
                                train_parquet, train_parquet, tokenizer_path)
        ef.write(out, records)
        print(f"wrote {out} ({out.stat().st_size} bytes, "
              f"layers={args.layers}, heads={args.heads}, "
              f"kv_heads={args.kv_heads}, hidden={args.hidden}, seq={args.seq}, "
              f"vocab={vocab}, steps={args.steps}, seed={args.seed})")
        print("per-step loss: " + ", ".join(f"{loss:.6f}" for _, loss in steps))
        print("final loss: %.6f" % final_loss)
        print("bpb: train %.6f val %.6f" % (bpb["train"], bpb["val"]))
        return 0
    _selected_main = main

if _command == 'numerics':
    __doc__ = textwrap.dedent("""\
        Generate the staged numeric golden trace. The default output is
        ``models/<name>/fixtures/numerics_golden_10l.bin`` at fp32;
        ``--precision fp8`` writes
        ``models/<name>/fixtures/numerics_golden_10l_fp8.bin`` from the fp8 CPU build.

        The trace of ``docs/numerics-integration.md`` is one flat list of named records
        in the little-endian container that ``tests/oracle_fixture.h`` reads, under the
        magic ``NANONUM1``. The C++ emitter owns the exact bytes, because it computes
        the values; this script builds the emitter, runs it once through
        ``tools/nanochat`` under the ``t0-cpu`` sandbox profile, verifies the container,
        and moves the file into place. The emitter pins the host thread count to one, so
        the file does not depend on the sandbox thread budget.

        The fixture is the fp32 CPU reference of section 6.2: 10 layers, 6 heads, 3 KV
        heads, hidden dimension 384, sequence length 256, batch 4, vocabulary 512, the
        window pattern ``SSSL``, value embeddings on, seed 42, and five optimizer steps.
        The trace holds the training trajectory (loss, gradient norm, and the L2 norm
        and byte hash of every parameter and gradient at every step), the evaluation
        section (bits per byte, per-batch loss, logits hash, greedy argmax identifiers),
        the greedy generation section (identifiers, mask, and logit margins), the SFT
        arithmetic, and the RL arithmetic.

        Run the generator from the repository root::

            python3 tools/dump_runtime_fixtures.py numerics

        Then read the file with ``tests/numerics_trace_test.cc`` (the CPU and simulator
        legs compare byte for byte; the GPU leg compares within the tolerance table) or
        with ``read_records`` below.

    """)
    import argparse
    import hashlib
    import os
    import struct
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    # The repository root holds the `tools/` entry point and the emitter binary.
    _REPO_ROOT = Path(__file__).resolve().parents[1]

    # The fixture magic and version of the container.
    MAGIC = b"NANONUM1"
    VERSION = 1

    #: One element size per dtype code (`tests/oracle_fixture.h`).
    _ELEM_SIZE = {0: 4, 1: 4, 2: 8, 3: 8, 4: 1}

    # The fp32 golden keeps the original name. Each other precision has its own
    # file, because a bf16 or fp8 build cannot reproduce the fp32 golden
    # (docs/precision.md section 6).
    DEFAULT_OUT_BY_PRECISION = {
        "fp32": "tests/data/numerics_golden_10l.bin",
        "bf16": "tests/data/numerics_golden_10l_bf16.bin",
        "fp8": "tests/data/numerics_golden_10l_fp8.bin",
    }
    DEFAULT_TOKENIZER = "tests/data/loader_tokenizer.nctoken"
    EMIT_TARGET = "//tests:numerics_trace_emit"
    EMIT_BINARY = "bazel-bin/tests/numerics_trace_emit"


    def read_records(path: Path) -> list[tuple[str, int, tuple, bytes]]:
        """Reads the flat record container: name, dtype, shape, payload."""
        data = path.read_bytes()
        if data[:8] != MAGIC:
            raise SystemExit(f"{path}: bad magic {data[:8]!r}")
        version, count = struct.unpack_from("<II", data, 8)
        if version != VERSION:
            raise SystemExit(f"{path}: unsupported version {version}")
        offset = 16
        records = []
        for _ in range(count):
            (name_len,) = struct.unpack_from("<H", data, offset)
            offset += 2
            name = data[offset:offset + name_len].decode("utf-8")
            offset += name_len
            dtype = data[offset]
            ndim = data[offset + 1]
            offset += 2
            shape = struct.unpack_from("<" + "q" * ndim, data, offset) if ndim else ()
            offset += 8 * ndim
            numel = 1
            for extent in shape:
                numel *= extent
            nbytes = numel * _ELEM_SIZE[dtype]
            payload = data[offset:offset + nbytes]
            offset += nbytes
            records.append((name, dtype, shape, payload))
        if offset != len(data):
            raise SystemExit(f"{path}: {len(data) - offset} trailing bytes")
        return records


    def scalar(values: dict[str, tuple], name: str) -> float:
        dtype, _shape, payload = values[name]
        if dtype == 0:
            return struct.unpack("<f", payload)[0]
        if dtype == 1:
            return struct.unpack("<i", payload)[0]
        if dtype == 3:
            return struct.unpack("<q", payload)[0]
        raise SystemExit(f"{name}: not a scalar record (dtype {dtype})")


    def run_nanochat(repo_root: Path, arguments) -> subprocess.CompletedProcess:
        """Runs one command through the single entry point and captures its output."""
        command = [str(repo_root / "tools" / "nanochat"), *arguments]
        result = subprocess.run(command, cwd=str(repo_root), capture_output=True,
                                text=True)
        if result.returncode != 0:
            sys.stderr.write(result.stdout)
            sys.stderr.write(result.stderr)
            raise SystemExit(
                f"command failed ({result.returncode}): " + " ".join(command))
        return result


    def summarize(records, out: Path, expected_precision: str) -> None:
        """Prints the fixture summary and the checks a reviewer needs."""
        values = {name: (dtype, shape, payload)
                  for name, dtype, shape, payload in records}
        backend = values["header/backend"][2].decode()
        precision = values["header/precision"][2].decode()
        steps = int(scalar(values, "header/steps"))
        print(f"wrote {out} ({out.stat().st_size} bytes, {len(records)} records, "
              f"sha256 {hashlib.sha256(out.read_bytes()).hexdigest()[:16]})")
        print(f"identity: backend {backend}, precision {precision}, "
              f"seed {int(scalar(values, 'header/seed'))}, "
              f"parameters {int(scalar(values, 'config/parameters'))}")
        losses = [scalar(values, f"train/step{step}/loss") for step in range(steps + 1)]
        print("train loss: " + ", ".join(f"{loss:.6f}" for loss in losses))
        print("train gradient norm: %.6f" % scalar(values, "train/step1/grad_norm"))
        print("eval: bpb %.6f, loss %.6f, min logit margin %.6g"
              % (scalar(values, "eval/bpb"), scalar(values, "eval/loss/0"),
                 scalar(values, "eval/margin_min")))
        print("generation: min logit margin %.6g"
              % scalar(values, "gen/margin_min"))
        print("sft: loss %.6f, gradient norm %.6f, valid targets %d"
              % (scalar(values, "sft/loss"), scalar(values, "sft/grad_norm"),
                 int(scalar(values, "sft/valid_targets"))))
        print("rl: loss %.6g, gradient norm %.6g, mean reward %.6f, "
              "min logit margin %.6g"
              % (scalar(values, "rl/loss"), scalar(values, "rl/grad_norm"),
                 scalar(values, "rl/mean_reward"), scalar(values, "rl/margin_min")))
        for row in range(4):
            print("rl row %d: reward %.6f, advantage %.6g"
                  % (row, scalar(values, f"rl/reward/{row}"),
                     scalar(values, f"rl/advantage/{row}")))
        if backend != "cpu" or precision != expected_precision:
            raise SystemExit("the golden trace must come from the "
                             + expected_precision + " CPU build")


    def parse_args(argv) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        add_model_argument(parser)
        add_check_argument(parser)
        parser.add_argument("--out", default=None,
                            help="output fixture path (default: "
                                 "models/<model>/fixtures/<precision name>)")
        parser.add_argument("--precision", default="fp32",
                            choices=("fp32", "bf16", "fp8"),
                            help="the build precision of the golden")
        parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                            help="the committed NCTOKEN1 artifact of the eval phase")
        parser.add_argument("--no-build", action="store_true",
                            help="use the current bazel-bin emitter")
        return parser.parse_args(argv)


    def main(argv) -> int:
        args = parse_args(argv)
        # The fixture is the CPU reference. Keep the shared GPU free.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        repo_root = _REPO_ROOT
        committed_dir = model_fixtures_dir(args.model)
        default_name = DEFAULT_OUT_BY_PRECISION[args.precision]
        committed = committed_dir / Path(default_name).name
        if args.out is None:
            out = committed
        else:
            out = Path(args.out)
            if not out.is_absolute():
                out = repo_root / out
        if args.check:
            check_dir_obj = tempfile.TemporaryDirectory(
                prefix="nanochat-numerics-check-")
            out = Path(check_dir_obj.name) / committed.name
        tokenizer = Path(args.tokenizer)
        if not tokenizer.is_absolute():
            tokenizer = repo_root / tokenizer
        if not tokenizer.is_file():
            raise SystemExit(f"missing tokenizer file: {tokenizer}")
        if not args.no_build:
            config = [] if args.precision == "fp32" else ["--config=" + args.precision]
            run_nanochat(repo_root, ["build", *config, EMIT_TARGET])
        emitter = repo_root / EMIT_BINARY
        if not emitter.is_file():
            raise SystemExit(f"missing emitter {emitter}; run "
                             f"tools/nanochat build {EMIT_TARGET}")

        with tempfile.TemporaryDirectory(prefix="numerics_golden_") as workdir:
            staged = Path(workdir) / "numerics_golden_10l.bin"
            result = run_nanochat(repo_root, [
                "run", "t0-cpu", "--", str(emitter),
                "--out", str(staged),
                "--tokenizer", str(tokenizer),
            ])
            sys.stdout.write(result.stdout)
            records = read_records(staged)
            out.parent.mkdir(parents=True, exist_ok=True)
            # The temporary directory may live on another filesystem, so stage
            # beside the target and replace inside one directory.
            beside = out.parent / (out.name + ".tmp")
            beside.write_bytes(staged.read_bytes())
            os.replace(beside, out)

        # Re-read the committed path, so the summary describes the file on disk.
        summarize(read_records(out), out, args.precision)
        if args.check:
            if not compare_fixture_bytes(out, committed):
                return 1
        return 0
    _selected_main = main

if _command == 'checkpoint':
    __doc__ = textwrap.dedent("""\
        Write the version 0 checkpoint fixture, ``tests/data/checkpoint_v0.ckpt``.

        The fixture is the *version 0 pin* for the metadata work: it records the
        bytes the current C writer (``src/data.cc``) produces, taken before
        anything changes that writer
        (docs/checkpoint-metadata-and-examples-plan.md stage A.2).
        ``//tests:checkpoint_compat_test`` reads this file with the current reader
        and asserts the table below, so a change that moves a version 0 byte fails
        in the tree instead of on a Hub artifact.

        The bytes come from ``nanochat_cpp.checkpoint.write_checkpoint``, which
        calls the C writer through ``nanochat_checkpoint_write``. This script holds
        no copy of the layout: the magic, the version, the dtype codes, and the
        record framing stay single-sourced in ``src/data.cc`` (docs/model.md).

        What it holds
        -------------
        Two parameter records and no optimizer-state section, drawn from the fixed
        seed ``20250601`` (``random.Random(SEED)``), one value per element, each
        uniform in ``[-1, 1)``::

            name             dtype      shape   elements  payload bytes
            fixture/weights  float32    (2, 3)  6         24
            fixture/bias     bfloat16   (3,)    3         6

        A float32 element is the value rounded to the nearest fp32. A bfloat16
        element is the round-to-nearest-even narrowing of that same fp32 value, so
        the two dtypes describe one number stream.

        Expected payload checksums
        --------------------------
        FNV-1a, 64 bit (offset basis ``0xcbf29ce484222325``, prime
        ``0x100000001b3``), the digest the numerics trace and the mesh runtime
        already use, over the payload bytes in record order::

            fixture/weights   0x7610708711193cf0
            fixture/bias      0x702bf156049a2035
            concatenated      0x41273f0935abaaa8

        The generation is deterministic: two runs produce identical bytes, and the
        seed is fixed, so the committed file matches these checksums.

        Run it from anywhere; the script locates the repository root from
        ``__file__``::

            python3 tools/dump_fixtures.py checkpoint
            python3 tools/dump_fixtures.py checkpoint --out /tmp/checkpoint_v0.ckpt

        The script imports no third-party package: the record set is built with
        ``struct`` and ``random``, and the C library owns the container bytes.

    """)
    import argparse
    import random
    import struct
    from pathlib import Path

    #: The fixture seed. A fixed seed keeps the payload bytes identical on every
    #: host, so the committed file and the checksums above stay valid.
    SEED = 20250601

    #: The records, in record order: name, storage dtype, shape.
    RECORDS = (
        ("fixture/weights", "float32", (2, 3)),
        ("fixture/bias", "bfloat16", (3,)),
    )

    # FNV-1a, 64 bit: the offset basis and the prime.
    _FNV_OFFSET = 0xCBF29CE484222325
    _FNV_PRIME = 0x100000001B3
    _U64_MASK = 0xFFFFFFFFFFFFFFFF


    def _repo_root() -> Path:
        """The repository root, found from this file."""
        return Path(__file__).resolve().parents[1]


    def _checkpoint_module():
        """The bridge module that owns the container writer."""
        import sys
        sys.path.insert(0, str(_repo_root() / "python"))
        from nanochat_cpp import checkpoint  # noqa: E402
        return checkpoint


    def fnv1a64(data: bytes) -> int:
        """The 64-bit FNV-1a digest of ``data``."""
        digest = _FNV_OFFSET
        for byte in data:
            digest = ((digest ^ byte) * _FNV_PRIME) & _U64_MASK
        return digest


    def payload(values: list[float], dtype: str) -> bytes:
        """The little-endian payload of ``values`` stored as ``dtype``."""
        if dtype == "float32":
            return b"".join(struct.pack("<f", value) for value in values)
        if dtype == "bfloat16":
            out = bytearray()
            for value in values:
                bits = struct.unpack("<I", struct.pack("<f", value))[0]
                # Round to nearest even on the 16-bit narrowing: add half an ulp
                # plus the low bit, then drop the low half.
                bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
                out += struct.pack("<H", (bits >> 16) & 0xFFFF)
            return bytes(out)
        raise SystemExit("unsupported fixture dtype: " + dtype)


    def build_records(checkpoint) -> list:
        """The two fixed-seed records, as the writer's own record type."""
        rng = random.Random(SEED)
        records = []
        for name, dtype, shape in RECORDS:
            elements = 1
            for dim in shape:
                elements *= int(dim)
            values = [rng.uniform(-1.0, 1.0) for _ in range(elements)]
            records.append(checkpoint.TensorRecord(
                name, checkpoint.dtype_code(dtype), shape,
                payload(values, dtype)))
        return records


    def summarize(records) -> int:
        """Print the record table and the checksums the fixture pins."""
        joined = b""
        for record in records:
            joined += record.data
            print(f"{record.name:<16} {record.dtype_name:<8} "
                  f"{str(record.shape):<7} {len(record.data):>3} bytes  "
                  f"fnv1a 0x{fnv1a64(record.data):016x}")
        print(f"{'concatenated':<16} {'':<8} {'':<7} {len(joined):>3} bytes  "
              f"fnv1a 0x{fnv1a64(joined):016x}")
        return 0


    def main(argv: list[str] | None = None) -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        add_model_argument(parser)
        parser.add_argument("--out", default="tests/data/checkpoint_v0.ckpt",
                            help="the fixture to write (default: %(default)s)")
        args = parser.parse_args(argv)

        out = Path(args.out)
        if not out.is_absolute():
            out = _repo_root() / out
        checkpoint = _checkpoint_module()
        records = build_records(checkpoint)
        checkpoint.write_checkpoint(out, records)
        print(f"wrote {len(records)} records to {out}")
        return summarize(records)
    _selected_main = main

if _command == 'sft':
    __doc__ = textwrap.dedent("""\
        Generate the supervised fine-tuning fixtures of
        ``docs/sft-parity-plan.md`` section 6, phase B items 1 to 3.

        One run writes four committed files from one pinned tiny model and one
        pinned conversation corpus. They make the SFT rungs ``R1`` and ``R2``
        hermetic: a runtime test reads them with no reference checkout, no network,
        and no torch.

        What it writes
        --------------
        ``tests/data/sft_pack_fixture.bin`` (section 6.1)
            ``NANCHAT1``, one ``uint32`` version word, then UTF-8 JSON: the pinned
            ``settings``, the ``tokenizer`` pin, the eight ``conversations``, the
            reference ``render`` ids and mask of every conversation, the reference
            ``pack`` inputs and targets, and the two mixtures of D8.

        ``tests/data/sft_rows_fixture.nchsft`` (section 6.4)
            The same packed rows as an ``NCSFT001`` shard of shape ``[rows, 32]``,
            written through ``nanochat_cpp.sft.pack``. The generator also proves the
            transcription below and the merged packer agree row for row.

        ``tests/data/sft_base_fixture.ckpt`` (section 6.4)
            The reference parameters and the reference optimizer moments at step 0,
            written through the path of ``tools/convert_checkpoint.py``, so the
            container bytes come from the C library and the names are canonical.

        ``tests/data/sft_step_fixture.bin`` (section 6.2)
            The eight-step reference trajectory as ``tools/fixture_format.py``
            records: the configuration (including the effective per-group rate
            table), the micro-batches of every step, the loss (the mean over the
            accumulation), the gradient norm, the parameter L2, the rate multiplier,
            the momentum, and the final parameters.

        Pinned settings (section 5)
        ---------------------------
        Model: 2 layers, ``hidden_dim=32``, 2 heads, 2 KV heads, ``vocab_size=256``,
        ``padded_vocab_size=256``, ``seq_len=32``, ``window_pattern="L"``. Pack: 8
        conversations, ``buffer_size=100``, ``render_max_tokens=2048``, rank 0, world
        size 1. Step: 8 steps, ``device_batch_size=1``, ``grad_accum=2``. Optimizer:
        the four rates, ``init_lr_frac=0.8``, ``adam_eps=1e-10``, no gradient clip.
        Schedule: ``warmup_ratio=0``, ``warmdown_ratio=0.5``, ``final_lr_frac=0``,
        and the Muon momentum ramp 0.85 to 0.95 over 300 steps. The constants live in
        ``tools/sft_fixture.py``, which the ``sft_loop`` kind reads too, so the two
        kinds cannot drift apart.

        The reference packer (section 6.1)
        ----------------------------------
        The ``pack`` object is a transcription of ``scripts/chat_sft.py:178-300``
        (``sft_data_generator_bos_bestfit``): a buffer refill before every
        largest-fit search, the largest conversation that fits the row, BOS padding
        of the remainder, a row capacity of ``max_seq_len + 1``, inputs as the first
        ``max_seq_len`` positions and targets as the last, ``-1`` for a masked or a
        padded target, a cursor advance by the world size with an epoch wrap, a
        too-long conversation skipped at fetch time, and the ``last_step`` stop rule.
        It runs with rank 0 and world size 1. The run fails when the transcription
        and ``nanochat_cpp.sft_data`` disagree.

        The tokenizer (section 6.1)
        ---------------------------
        The pinned model has a 256-entry vocabulary, which cannot hold a trained BPE
        vocabulary, so the fixture tokenizer is a piece table: the distinct pieces of
        the fixture corpus, in first-appearance order, after the nine chat special
        tokens. The pack fixture records the artifact name, its SHA-256, the
        special-token map, the vocabulary, the per-id byte table, and a ``pieces``
        table from every string the renderer encodes to its reference ids. The phase
        C.1 test builds a tokenizer from that table, so the ids it compares are the
        ids this run recorded. There is no tokenizer artifact to drift.

        What the run needs
        ------------------
        The reference PyTorch checkout (``NANOCHAT_REPO``), ``torch``, ``numpy``, and
        the built ``nanochat_cpp`` shared library, on the CPU. Run it from anywhere::

            python3 tools/dump_fixtures.py sft
            python3 tools/dump_fixtures.py sft --dir /tmp/sft --seed 7

        The script forces the CPU and disables ``torch.compile``, so the trajectory is
        deterministic. It ends by printing the SHA-256 of every file it wrote, so the
        ``fixtures`` node can copy the digests into ``tests/data/sft_fixtures.txt``
        without a second derivation.

        ``--real-mixture`` (D8)
        -----------------------
        The default ``mixture`` and ``validation_mixture`` objects pin the reference
        mixture algorithm over a synthetic five-task and three-task set, so the gate
        is hermetic. ``--real-mixture`` replaces those two objects with the real hub
        order; the conversations and the packed rows stay the pinned synthetic
        corpus either way. That mode needs a network download of SmolTalk, MMLU, and
        GSM8K plus a pinned ``datasets`` version. It is a manual check, not part of
        any gate.

    """)
    import argparse
    from pathlib import Path

    from fixture_format import FixtureWriter
    import sft_fixture as fx

    #: The rank and the world size the reference packer runs at (section 6.1).
    RANK = fx.PACK["rank"]
    WORLD_SIZE = fx.PACK["world_size"]
    #: The placeholder the transcription puts in a padded position before it
    #: substitutes the BOS id, the way the reference pads with BOS tokens.
    PAD = -1


    def repo_path(value: str) -> Path:
        """One fixture path, resolved against the repository root."""
        path = Path(value)
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[1] / path


    def reference_pack(conversations, render, seq_len, buffer_size, rank,
                       world_size):
        """The reference packer, transcribed from ``scripts/chat_sft.py:178-300``.

        ``render`` turns one conversation into ``(ids, mask)``, exactly as
        ``tokenizer.render_conversation`` does with the default 2048-token
        truncation. The result is one epoch of rows, each ``(row, mask_row,
        content_len)``: the reference ``last_step`` stop rule ends the stream after
        the row in which ``consumed >= len(conversations)``.
        """
        dataset_size = len(conversations)
        if dataset_size <= 0:
            raise SystemExit("the SFT conversation list is empty")
        # +1 for the target at the last position of the row.
        row_capacity = seq_len + 1
        buffer = []
        cursor = rank
        consumed = rank
        rows = []
        # The reference skips a too-long conversation at fetch time. When every
        # conversation is too long the reference would refill forever, so the
        # guard below reports the corpus instead of hanging. It cannot trigger
        # for the pinned corpus.
        skips = 0
        while True:
            row = []
            mask_row = []
            content_len = row_capacity
            while len(row) < row_capacity:
                # Refill the buffer *before* the largest-fit search, as the
                # reference generator does.
                while len(buffer) < buffer_size:
                    ids, mask = render(conversations[cursor])
                    cursor += world_size
                    if cursor >= dataset_size:
                        cursor %= dataset_size
                    if len(ids) > row_capacity:
                        # Too long to ever fit one row: skip it at fetch time.
                        skips += 1
                        if skips >= dataset_size:
                            raise SystemExit(
                                "no conversation fits one row of "
                                f"{row_capacity} tokens")
                        continue
                    skips = 0
                    buffer.append((ids, mask))
                remaining = row_capacity - len(row)
                best_index, best_length = -1, 0
                for index, (ids, _) in enumerate(buffer):
                    if len(ids) <= remaining and len(ids) > best_length:
                        best_index, best_length = index, len(ids)
                if best_index >= 0:
                    ids, mask = buffer.pop(best_index)
                    row.extend(ids)
                    mask_row.extend(mask)
                    consumed += world_size
                    continue
                # Nothing fits: pad the remainder with BOS instead of cropping,
                # so no supervised token is ever discarded.
                content_len = len(row)
                row.extend([PAD] * remaining)
                mask_row.extend([0] * remaining)
                break
            rows.append((row, mask_row, content_len))
            if consumed >= dataset_size:
                break
        return rows


    def pack_rows(rows, seq_len, bos_token):
        """Turn the transcribed rows into ``(inputs, targets)`` pairs.

        The reference slices each row into inputs (the first ``max_seq_len``
        positions) and targets (the last ``max_seq_len`` positions), masks a
        target whose source position is not supervised, and masks the padding tail
        after ``content_len - 1``.
        """
        packed = []
        for row, mask_row, content_len in rows:
            full = [bos_token if token == PAD else token for token in row]
            targets = [token if keep else -1
                       for token, keep in zip(full[1:], mask_row[1:])]
            for position in range(max(content_len - 1, 0), seq_len):
                targets[position] = -1
            packed.append((full[:seq_len], targets))
        return packed


    def write_step_fixture(path, torch, model, optimizer, rows):
        """Write the eight-step trajectory of section 6.2."""
        steps = fx.STEP["steps"]
        accumulate = fx.STEP["grad_accum"]
        batch = fx.STEP["device_batch_size"]
        seq = fx.STEP["seq_len"]

        stream = fx.row_cycle(rows)
        micro_batches = []
        for _ in range(steps):
            micro = [next(stream)[0] for _ in range(accumulate * batch)]
            micro_batches.append(micro)

        writer = FixtureWriter(Path(path))
        writer.add("config/version", fx.i32([1]))
        writer.add("config/seed", fx.i32([fx.SEED]))
        writer.add("config/layers", fx.i32([fx.MODEL["layers"]]))
        writer.add("config/heads", fx.i32([fx.MODEL["heads"]]))
        writer.add("config/kv_heads", fx.i32([fx.MODEL["kv_heads"]]))
        writer.add("config/embd", fx.i32([fx.MODEL["embd"]]))
        writer.add("config/seq", fx.i32([seq]))
        writer.add("config/vocab", fx.i32([fx.MODEL["vocab"]]))
        writer.add("config/padded_vocab", fx.i32([fx.MODEL["padded_vocab"]]))
        writer.add("config/window_pattern", fx.u8(fx.MODEL["window_pattern"]))
        writer.add("config/device_batch_size", fx.i32([batch]))
        writer.add("config/grad_accum", fx.i32([accumulate]))
        writer.add("config/steps", fx.i32([steps]))
        writer.add("config/opt/unembedding_lr",
                   _scalar(fx.OPT["unembedding_lr"]))
        writer.add("config/opt/embedding_lr", _scalar(fx.OPT["embedding_lr"]))
        writer.add("config/opt/matrix_lr", _scalar(fx.OPT["matrix_lr"]))
        writer.add("config/opt/scalar_lr", _scalar(fx.OPT["scalar_lr"]))
        writer.add("config/opt/init_lr_frac", _scalar(fx.OPT["init_lr_frac"]))
        writer.add("config/opt/adam_eps", _scalar(fx.OPT["adam_eps"]))
        writer.add("config/opt/clip", _scalar(fx.OPT["clip"]))
        writer.add("config/opt/lr_scale",
                   _scalar((fx.MODEL["embd"] / 768) ** -0.5))
        writer.add("config/schedule/warmup_ratio",
                   _scalar(fx.SCHEDULE["warmup_ratio"]))
        writer.add("config/schedule/warmdown_ratio",
                   _scalar(fx.SCHEDULE["warmdown_ratio"]))
        writer.add("config/schedule/final_lr_frac",
                   _scalar(fx.SCHEDULE["final_lr_frac"]))
        writer.add("config/schedule/muon_momentum_steps",
                   fx.i32([fx.SCHEDULE["muon_momentum_steps"]]))
        for name in ("start", "peak", "final"):
            writer.add(f"config/schedule/muon_momentum_{name}",
                       _scalar(fx.SCHEDULE[f"muon_momentum_{name}"]))
        # The rate table of section 5.2 as the pinned run uses it: the listed
        # rate, times the (hidden_dim / 768) ** -0.5 scale, times
        # init_lr_frac, which D3 applies once on the optimizer configuration.
        for name, rate in fx.group_rates(optimizer).items():
            writer.add(f"config/opt/rate/{name}", _scalar(rate))

        for step, micro in enumerate(micro_batches, start=1):
            tokens = [pair[0] for pair in micro]
            targets = [pair[1] for pair in micro]
            writer.add(f"data/step/{step}/tokens",
                       fx.i32([value for row in tokens for value in row])
                       .reshape(accumulate, batch, seq))
            writer.add(f"data/step/{step}/targets",
                       fx.i32([value for row in targets for value in row])
                       .reshape(accumulate, batch, seq))

        for step, micro in enumerate(micro_batches, start=1):
            tokens = torch.tensor([pair[0] for pair in micro],
                                  dtype=torch.long).reshape(accumulate, batch,
                                                            seq)
            targets = torch.tensor([pair[1] for pair in micro],
                                   dtype=torch.long).reshape(accumulate, batch,
                                                             seq)
            model.zero_grad(set_to_none=True)
            losses = []
            for index in range(accumulate):
                loss = model(tokens[index], targets=targets[index])
                losses.append(float(loss))
                (loss / accumulate).backward()
            grad_norm = fx.total_grad_norm(model)
            multiplier, momentum = fx.apply_schedule(
                optimizer, step / steps, step - 1)
            optimizer.step()
            model.zero_grad(set_to_none=True)
            writer.add(f"step/{step}/loss",
                       _scalar(sum(losses) / len(losses)))
            writer.add(f"step/{step}/grad_norm", _scalar(grad_norm))
            writer.add(f"step/{step}/param_l2",
                       _scalar(fx.total_param_l2(model)))
            writer.add(f"step/{step}/lr", _scalar(multiplier))
            writer.add(f"step/{step}/momentum", _scalar(momentum))

        for name, parameter in model.named_parameters():
            writer.add(f"step/{steps}/param/{name}", _as_f32(parameter))
        writer.write()
        return Path(path)


    def write_pack_fixture(path, tokenizer, render, rows, mixture,
                           validation_mixture):
        """Write the ``NANCHAT1`` JSON of section 6.1."""
        renders = [render(conversation) for conversation in fx.CONVERSATIONS]
        payload = {
            "version": 1,
            "settings": {
                **fx.PACK,
                "seed": fx.SEED,
                "conversations": len(fx.CONVERSATIONS),
                "rows": len(rows),
            },
            "tokenizer": fx.tokenizer_record(tokenizer),
            "conversations": [
                {"name": conversation["name"],
                 "messages": conversation["messages"]}
                for conversation in fx.CONVERSATIONS
            ],
            "render": [{"ids": ids, "mask": mask} for ids, mask in renders],
            "pack": {
                "rows": len(rows),
                "inputs": [inputs for inputs, _ in rows],
                "targets": [targets for _, targets in rows],
            },
            "mixture": mixture,
            "validation_mixture": validation_mixture,
        }
        fx.write_json_fixture(path, payload)
        return Path(path)


    def generate(args) -> list:
        """Write the four fixtures and return their paths in command order."""
        torch, np, GPT, GPTConfig = fx.setup_reference()
        fx.python_path()
        from nanochat_cpp import sft as bridge  # noqa: E402 - path is ready
        from nanochat_cpp.chat import render_conversation  # noqa: E402

        directory = repo_path(args.dir)
        tokenizer = fx.build_tokenizer()


        def render(conversation):
            return render_conversation(
                conversation, tokenizer,
                max_tokens=fx.PACK["render_max_tokens"])

        bos_token = tokenizer.get_bos_token_id()
        transcribed = reference_pack(
            fx.CONVERSATIONS, render, fx.PACK["seq_len"],
            fx.PACK["buffer_size"], RANK, WORLD_SIZE)
        rows = pack_rows(transcribed, fx.PACK["seq_len"], bos_token)

        rows_path = Path(bridge.pack(directory / fx.ROWS_FIXTURE,
                                     fx.CONVERSATIONS, tokenizer,
                                     fx.PACK["seq_len"]))
        written = fx.load_shard(rows_path)
        if written != rows:
            raise SystemExit(
                "the transcribed reference packer and nanochat_cpp.sft_data "
                "disagree: "
                f"{len(written)} shard rows, {len(rows)} transcribed")

        mixture, validation_mixture = fx.build_mixture_records(
            real=args.real_mixture)
        pack_path = write_pack_fixture(
            directory / fx.PACK_FIXTURE, tokenizer, render, rows, mixture,
            validation_mixture)

        model = fx.build_model(GPT, GPTConfig, np, args.seed)
        optimizer = fx.build_optimizer(model)
        fx.materialize_moments(optimizer)
        base_path = directory / fx.BASE_FIXTURE
        fx.write_base_checkpoint(base_path, model, optimizer)

        step_path = write_step_fixture(directory / fx.STEP_FIXTURE, torch,
                                       model, optimizer, rows)
        return [pack_path, step_path, rows_path, base_path]


    def parse_args(argv: list) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        add_model_argument(parser)
        parser.add_argument("--dir", default=fx.FIXTURE_DIR, metavar="DIR",
                            help="the fixture directory "
                                 "(default: %(default)s)")
        parser.add_argument("--seed", type=int, default=fx.SEED,
                            help="the weight seed (default: %(default)s)")
        parser.add_argument(
            "--real-mixture", action="store_true",
            help="record the real hub mixture order instead of the "
                 "synthetic D8 task set; needs a network download plus a "
                 "pinned datasets version, and is not part of any gate")
        return parser.parse_args(argv)


    def main(argv: list) -> int:
        args = parse_args(argv)
        paths = generate(args)
        return fx.record_checksums(paths)
    _selected_main = main

if _command == 'sft_loop':
    __doc__ = textwrap.dedent("""\
        Generate the supervised fine-tuning loop fixture,
        ``tests/data/sft_loop_fixture.bin`` (``docs/sft-parity-plan.md`` section 6.3,
        phase B item 3).

        The kind reads the two files the ``sft`` kind writes — the packed shard
        ``tests/data/sft_rows_fixture.nchsft`` and the base checkpoint
        ``tests/data/sft_base_fixture.ckpt`` — and runs the reference model, the
        reference optimizer, and the reference schedule over them for eight steps. It
        records the trace and the loss curve, so ``//tests:sft_loop_parity_test``
        (phase E) can replay the schedule without torch.

        What it writes
        --------------
        ``config/num_iterations``, ``config/rows``, ``config/eval_every``
            The pinned run: eight iterations, the shard row count, and the
            evaluation interval of section 5.4.

        ``loop/<k>/step``, ``progress``, ``epoch``, ``lr_multiplier``, ``momentum``
            One set per step ``k = 1..8``: the 0-based counter the momentum schedule
            evaluates at, the progress value the rate schedule consumes, the pass
            over the shard, and the two schedule outputs.

        ``train/loss/<k>``
            The last micro-batch loss of step ``k``, which is the value the
            reference loop logs.

        ``val/bpb/8``, ``chatcore/8``
            The final-step evaluation of section 5.4. ``config/eval_every``,
            ``config/eval_tokens``, ``config/chatcore_every``, and the two ChatCORE
            caps pin the cadence next to it.

        What the run needs
        ------------------
        The reference PyTorch checkout (``NANOCHAT_REPO``), ``torch``, ``numpy``, and
        the built ``nanochat_cpp`` shared library, on the CPU::

            python3 tools/dump_fixtures.py sft_loop
            python3 tools/dump_fixtures.py sft_loop --out /tmp/loop.bin

        It reads the token-byte table from the pack fixture, because bits-per-byte
        divides the summed negative log-probability by the summed target bytes (D6).
        The evaluation pass is one pass over the shard, the packed-row validation of
        that decision.

        ``chatcore/8``
        --------------
        ChatCORE needs the chat evaluation bundle, the trained tokenizer, and the task
        datasets, so the pinned run cannot measure it: the record carries the
        sentinel ``-1`` and ``config/chatcore_measured`` is ``0``. Phase G measures
        ChatCORE on the real run; this fixture pins the cadence only.

        The kind ends by printing the SHA-256 of the file it wrote, in the style of
        the ``checkpoint`` kind.

    """)
    import argparse
    from pathlib import Path

    from fixture_format import FixtureWriter
    import sft_fixture as fx


    def repo_path(value: str) -> Path:
        """One fixture path, resolved against the repository root."""
        path = Path(value)
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[1] / path


    def generate(args) -> Path:
        """Run the pinned reference loop and write the loop fixture."""
        torch, np, GPT, GPTConfig = fx.setup_reference()

        rows = fx.load_shard(repo_path(args.rows))
        token_bytes = fx.read_pack_fixture(
            repo_path(args.pack))["tokenizer"]["token_bytes"]
        if len(token_bytes) != fx.MODEL["vocab"]:
            raise SystemExit(
                "the pack fixture pins "
                f"{len(token_bytes)} token byte lengths, the model has "
                f"{fx.MODEL['vocab']} ids")

        model = fx.build_model(GPT, GPTConfig, np)
        optimizer = fx.build_optimizer(model)
        fx.materialize_moments(optimizer)
        fx.load_state(repo_path(args.base), model, optimizer)
        model.train()

        steps = fx.LOOP["num_iterations"]
        accumulate = fx.LOOP["grad_accum"]
        batch = fx.LOOP["device_batch_size"]
        seq = fx.LOOP["seq_len"]
        stream = fx.row_cycle(rows)

        writer = FixtureWriter(repo_path(args.out))
        writer.add("config/version", fx.i32([1]))
        writer.add("config/num_iterations", fx.i32([steps]))
        writer.add("config/rows", fx.i32([len(rows)]))
        writer.add("config/eval_every", fx.i32([fx.EVAL["eval_every"]]))
        writer.add("config/eval_at", fx.i32([fx.LOOP["eval_step"]]))
        writer.add("config/eval_tokens", fx.i32([fx.EVAL["eval_tokens"]]))
        writer.add("config/chatcore_every",
                   fx.i32([fx.EVAL["chatcore_every"]]))
        writer.add("config/chatcore_max_cat",
                   fx.i32([fx.EVAL["chatcore_max_cat"]]))
        writer.add("config/chatcore_max_sample",
                   fx.i32([fx.EVAL["chatcore_max_sample"]]))
        writer.add("config/device_batch_size",
                   fx.i32([fx.LOOP["device_batch_size"]]))
        writer.add("config/grad_accum", fx.i32([accumulate]))
        writer.add("config/seq_len", fx.i32([fx.LOOP["seq_len"]]))
        writer.add("config/chatcore_measured", fx.i32([0]))
        writer.add("config/token_bytes", fx.i32(token_bytes))

        progress = 0.0
        for step in range(1, steps + 1):
            fetched = [next(stream) for _ in range(accumulate * batch)]
            epoch = fetched[-1][1]
            micro = [row for row, _ in fetched]
            tokens = torch.tensor([pair[0] for pair in micro],
                                  dtype=torch.long).reshape(accumulate, batch,
                                                            seq)
            targets = torch.tensor([pair[1] for pair in micro],
                                   dtype=torch.long).reshape(accumulate, batch,
                                                             seq)
            model.zero_grad(set_to_none=True)
            losses = []
            for index in range(accumulate):
                loss = model(tokens[index], targets=targets[index])
                losses.append(float(loss))
                (loss / accumulate).backward()
            # One optimization step advances progress: the reference drives it
            # from the data stream, and this fixture replaces the stream with a
            # fixed row list, so progress is the step fraction of the pinned 8
            # iterations.
            progress = max(progress, step / steps)
            multiplier, momentum = fx.apply_schedule(optimizer, progress,
                                                     step - 1)
            optimizer.step()
            model.zero_grad(set_to_none=True)

            writer.add(f"loop/{step}/step", fx.i32([step - 1]))
            writer.add(f"loop/{step}/progress", _scalar(progress))
            writer.add(f"loop/{step}/epoch", fx.i32([epoch]))
            writer.add(f"loop/{step}/lr_multiplier", _scalar(multiplier))
            writer.add(f"loop/{step}/momentum", _scalar(momentum))
            writer.add(f"train/loss/{step}", _scalar(losses[-1]))

        model.eval()
        with torch.no_grad():
            bpb = fx.evaluate_bpb(model, rows, token_bytes)
        writer.add("val/bpb/8", _scalar(bpb))
        writer.add("chatcore/8", _scalar(-1.0))
        writer.write()
        return repo_path(args.out)


    def parse_args(argv: list) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        add_model_argument(parser)
        parser.add_argument("--rows", default=fx.FIXTURE_DIR + "/"
                            + fx.ROWS_FIXTURE,
                            help="the packed shard the sft kind wrote "
                                 "(default: %(default)s)")
        parser.add_argument("--base", default=fx.FIXTURE_DIR + "/"
                            + fx.BASE_FIXTURE,
                            help="the base checkpoint the sft kind wrote "
                                 "(default: %(default)s)")
        parser.add_argument("--pack", default=fx.FIXTURE_DIR + "/"
                            + fx.PACK_FIXTURE,
                            help="the pack fixture that pins the token-byte "
                                 "table (default: %(default)s)")
        parser.add_argument("--out", default=fx.FIXTURE_DIR + "/"
                            + fx.LOOP_FIXTURE,
                            help="the fixture to write (default: %(default)s)")
        return parser.parse_args(argv)


    def main(argv: list) -> int:
        args = parse_args(argv)
        path = generate(args)
        return fx.record_checksums([path])
    _selected_main = main

if __name__ == "__main__":
    raise SystemExit(_selected_main(sys.argv[2:]))
