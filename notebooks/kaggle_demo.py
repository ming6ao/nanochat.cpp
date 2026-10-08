"""Helpers for ``notebooks/nanochat-cpp-on-t4-gpu.ipynb``.

The supervised fine-tuning cell uses these functions. They stay out of the
notebook, so the notebook shows the API and the packing logic is importable.
``pack_conversations`` calls ``nanochat_cpp.render_conversation``, so import
this module after ``nanochat_cpp`` is on ``sys.path``.
"""

from __future__ import annotations

import nanochat_cpp as nc

__all__ = [
    "DEMO_CONVERSATIONS",
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

    ``nc.render_conversation`` returns the ids and the loss mask. This helper
    pads the row and shifts the mask onto the targets.
    """
    bos = tokenizer.get_bos_token_id()
    rows, ids, mask = [], [], []

    def flush():
        if not ids:
            return
        pad = seq_len - len(ids)
        full_ids = ids + [bos] * pad
        full_mask = mask + [0] * pad
        rows.append((full_ids, shifted_targets(full_ids, full_mask, bos)))

    for conversation in conversations:
        part_ids, part_mask = nc.render_conversation(conversation, tokenizer)
        if len(ids) + len(part_ids) > seq_len:
            flush()
            ids, mask = [], []
        ids.extend(part_ids)
        mask.extend(part_mask)
    flush()
    return rows


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
