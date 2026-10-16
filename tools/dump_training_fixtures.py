#!/usr/bin/env python3
"""Generate grouped offline fixtures: oracle, train, rl."""

from __future__ import annotations

import sys
import textwrap

_COMMANDS = ('oracle', 'train', 'rl')
_USAGE = "usage: dump_training_fixtures.py {oracle|train|rl} [options]"
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(_USAGE)
        raise SystemExit(0 if len(sys.argv) >= 2 else 2)
    _command = sys.argv[1]
    if _command not in _COMMANDS:
        print("unknown fixture kind: " + _command, file=sys.stderr)
        print(_USAGE, file=sys.stderr)
        raise SystemExit(2)
else:
    _command = None

if _command == 'oracle':
    __doc__ = textwrap.dedent("""\
        Generate the CPU oracle fixture, ``tests/data/debug_state.bin``.

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


    def _as_f32(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().to(torch.float32).cpu().numpy()


    def _as_i32(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().to(torch.int32).cpu().numpy()


    def _scalar(value: float) -> np.ndarray:
        return np.asarray([value], dtype=np.float32)


    def build_config(args: argparse.Namespace) -> GPTConfig:
        return GPTConfig(
            sequence_len=args.seq,
            vocab_size=args.vocab,
            n_layer=args.layers,
            n_head=args.heads,
            n_kv_head=args.kv_heads,
            n_embd=args.embd,
            window_pattern=args.window,
        )


    def generate(args: argparse.Namespace) -> Path:
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        config = build_config(args)
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
        default_out = REPO_ROOT / "tests" / "data" / "debug_state.bin"
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--out", default=str(default_out), help="output .bin path")
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
        return parser.parse_args(argv)


    def main(argv: list[str]) -> int:
        args = parse_args(argv)
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
        Generate a multi-step *training-parity* fixture.

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


    def _as_f32(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().to(torch.float32).cpu().numpy()


    def _as_i32(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().to(torch.int32).cpu().numpy()


    def _scalar(value: float) -> np.ndarray:
        return np.asarray([value], dtype=np.float32)

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


    def build_config(args: argparse.Namespace) -> GPTConfig:
        return GPTConfig(
            sequence_len=args.seq,
            vocab_size=args.vocab,
            n_layer=args.layers,
            n_head=args.heads,
            n_kv_head=args.kv_heads,
            n_embd=args.embd,
            window_pattern=args.window,
        )


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

        config = build_config(args)
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
        parser.add_argument("--out", required=True, help="output .bin path")
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
        return parser.parse_args(argv)


    def main(argv: list[str]) -> int:
        args = parse_args(argv)
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
        ``tests/data/rl_parity.bin``.

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
        parser.add_argument("--out", required=True, help="output .bin path")
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
        return parser.parse_args(argv)


    def main(argv: list[str]) -> int:
        args = parse_args(argv)
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

if __name__ == "__main__":
    raise SystemExit(_selected_main(sys.argv[2:]))
