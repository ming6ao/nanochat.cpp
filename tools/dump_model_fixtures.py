#!/usr/bin/env python3
"""Generate grouped offline fixtures: generate, eval, plan."""

from __future__ import annotations

import sys
import textwrap

_COMMANDS = ('generate', 'eval', 'plan')
_USAGE = "usage: dump_model_fixtures.py {generate|eval|plan} [options]"
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

if _command == 'generate':
    __doc__ = textwrap.dedent("""\
        Generate the *generation-parity* fixture, ``tests/data/generate_parity.bin``.

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


    def _as_f32(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().to(torch.float32).cpu().numpy()


    def _scalar(value: float) -> np.ndarray:
        return np.asarray([value], dtype=np.float32)


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

        config = build_config(args)
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
        parser.add_argument("--out", required=True, help="output .bin path")
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
        return parser.parse_args(argv)


    def main(argv: list[str]) -> int:
        args = parse_args(argv)
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
        Generate the *score-parity* fixture, ``tests/data/core_eval.bin``.

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


    def build_config(seq: int, vocab: int, layers: int, heads: int,
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

        config = build_config(args.seq, args.vocab, args.layers, args.heads,
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
        parser.add_argument("--out", default="tests/data/core_eval.bin",
                            help="output .bin path")
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
        return parser.parse_args(argv)


    def main(argv: list[str]) -> int:
        args = parse_args(argv)
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

if __name__ == "__main__":
    raise SystemExit(_selected_main(sys.argv[2:]))
