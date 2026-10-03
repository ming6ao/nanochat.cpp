#!/usr/bin/env python3
"""Generate the *score-parity* fixture, ``tests/data/core_eval.bin``.

This is the evaluation analogue of ``tools/dump_generate_fixture.py``. It
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

    python3 tools/dump_eval_fixture.py --out tests/data/core_eval.bin
"""

from __future__ import annotations

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


def _find_reference_repo() -> Path:
    """Locate the PyTorch nanochat checkout that holds the `nanochat` package."""
    candidates = []
    if os.environ.get("NANOCHAT_REPO"):
        candidates.append(Path(os.environ["NANOCHAT_REPO"]))
    here = Path(__file__).resolve()
    candidates.append(here.parents[2] / "nanochat")  # sibling checkout
    candidates.append(Path.home() / "repos" / "nanochat")
    for candidate in candidates:
        if (candidate / "nanochat" / "gpt.py").is_file():
            return candidate
    raise SystemExit(
        "could not locate the PyTorch nanochat checkout; set NANOCHAT_REPO to "
        "the directory that contains the `nanochat` package")


REFERENCE_REPO = _find_reference_repo()
if str(REFERENCE_REPO) not in sys.path:
    sys.path.insert(0, str(REFERENCE_REPO))

# Reuse the wire format owner rather than re-encoding it here. `eval_fixture`
# is standard-library only, so this import needs no reference package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))

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


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
