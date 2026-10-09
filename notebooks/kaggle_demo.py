"""Helpers for ``notebooks/nanochat-cpp-on-t4-gpu.ipynb``.

The supervised fine-tuning cell uses these functions. They stay out of the
notebook, so the notebook shows the API and the packing logic is importable.

The real supervised fine-tuning data is the reference mixture in
``nanochat_cpp.sft_data``: SmolTalk, MMLU, and GSM8K. ``build_sft_conversations``
returns a small real slice when the Hugging Face hub is reachable. It returns
the offline ``DEMO_CONVERSATIONS`` set otherwise, so the notebook always runs.

Import this module after ``nanochat_cpp`` is on ``sys.path``.
"""

from __future__ import annotations

import nanochat_cpp as nc

__all__ = [
    "DEMO_CONVERSATIONS",
    "build_sft_conversations",
    "complete",
    "pack_conversations",
    "shifted_targets",
]


def shifted_targets(ids, mask, bos):
    """Shift by one and set a masked target to -1 (the loss ignore index)."""
    targets = ids[1:] + [bos]
    keep = mask[1:] + [0]
    return [token if flag else -1 for token, flag in zip(targets, keep)]


def pack_conversations(conversations, tokenizer, seq_len):
    """Pack conversations into padded ``(tokens, targets)`` rows.

    This now delegates to :func:`nanochat_cpp.sft_data.pack_rows`, which uses
    the reference BOS-aligned best-fit packer with a ``-1`` mask. The helper
    stays for the notebook and for older notebook revisions.
    """
    return nc.sft_data.pack_rows(conversations, tokenizer, seq_len)


def build_sft_conversations(limit=64, offline=None):
    """Return real supervised fine-tuning conversations, or the offline set.

    ``limit`` caps each of the three tasks, so the notebook stays small. The
    real mixture is ``nanochat_cpp.sft_data.demo_mixture``: SmolTalk train,
    MMLU ``auxiliary_train``, and GSM8K train. The download covers the whole
    hub shard even for a small ``limit``.

    ``offline=True`` skips the download. ``offline=None`` tries the download
    and falls back to ``DEMO_CONVERSATIONS`` when it fails. ``offline=False``
    re-raises the failure.
    """
    if offline is True:
        return list(DEMO_CONVERSATIONS)
    try:
        mixture = nc.sft_data.demo_mixture(limit=limit)
        return [mixture[index] for index in range(len(mixture))]
    except Exception as error:  # noqa: BLE001 - the offline fallback is wanted
        if offline is False:
            raise
        print(f"real SFT data unavailable ({error}); using DEMO_CONVERSATIONS")
        return list(DEMO_CONVERSATIONS)


FACTS = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"),
         ("Spain", "Madrid"), ("Egypt", "Cairo"), ("Canada", "Ottawa"),
         ("Brazil", "Brasilia"), ("Kenya", "Nairobi")]

DEMO_CONVERSATIONS = [
    {"messages": [
        {"role": "user", "content": f"What is the capital of {country}?"},
        {"role": "assistant",
         "content": f"The capital of {country} is {capital}."}]}
    for country, capital in FACTS]
DEMO_CONVERSATIONS += [
    {"messages": [
        {"role": "user", "content": f"What is {a} + {b}?"},
        {"role": "assistant", "content": f"{a} + {b} = {a + b}."}]}
    for a, b in [(2, 3), (5, 7), (8, 1), (4, 6), (9, 2), (3, 9), (7, 7), (6, 5),
                 (1, 8), (5, 5), (2, 9), (4, 4)]]


def complete(model, tokenizer, prompt, max_tokens=12):
    """Greedy completion of a prompt, decoded without the prompt tokens."""
    ids = tokenizer.encode(prompt)
    row = model.generate(ids, max_tokens=max_tokens, temperature=0.0)[0]
    return tokenizer.decode(
        [token for token, keep in zip(row.tokens, row.mask) if keep])
