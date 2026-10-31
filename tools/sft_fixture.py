"""Shared machinery for the two supervised fine-tuning fixture kinds.

``tools/dump_fixtures.py`` keeps one block per fixture kind. The ``sft`` and
``sft_loop`` kinds of ``docs/sft-parity-plan.md`` section 6 share the pinned
settings of section 5, the tiny model of section 5.5, the fixture tokenizer,
the conversation corpus, the reference optimizer and schedule, and the two small
containers they write. This module holds that shared code once, so the two kinds
cannot drift apart.

The module imports no third-party package at import time. ``numpy``, ``torch``,
the reference ``nanochat`` package, and ``nanochat_cpp`` load lazily inside the
functions that need them, so a ``--help`` run imports neither torch nor the C
library.

The fixtures are *data*: they are produced once, offline, and committed. A
runtime test reads them and never imports torch (``docs/sft-parity-plan.md``
section 7, phase B gate).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path

# ``tools/sft_fixture.py`` -> the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]

__all__ = [
    "BASE_FIXTURE",
    "CONVERSATIONS",
    "EVAL",
    "FIXTURE_DIR",
    "LONG_REPEAT",
    "LOOP",
    "LOOP_FIXTURE",
    "MODEL",
    "MIXTURE",
    "OPT",
    "PACK",
    "PACK_FIXTURE",
    "ROWS_FIXTURE",
    "SCHEDULE",
    "SEED",
    "SHARD_MAGIC",
    "SPECIAL_TOKENS",
    "STEP",
    "STEP_FIXTURE",
    "TOKENIZER_NAME",
    "VALIDATION_MIXTURE",
    "FixtureTokenizer",
    "apply_schedule",
    "build_mixture_records",
    "build_model",
    "build_tokenizer",
    "build_optimizer",
    "checkpoint_records",
    "conversation_texts",
    "evaluate_bpb",
    "file_sha256",
    "group_rates",
    "i32",
    "load_shard",
    "load_state",
    "lr_multiplier",
    "materialize_moments",
    "muon_momentum",
    "piece_texts",
    "python_path",
    "read_pack_fixture",
    "record_checksums",
    "render_texts",
    "restore_optimizer_state",
    "row_cycle",
    "setup_reference",
    "token_bytes_table",
    "tokenizer_record",
    "total_grad_norm",
    "total_param_l2",
    "u8",
    "write_base_checkpoint",
    "write_json_fixture",
]

# --- the pinned settings (docs/sft-parity-plan.md section 5) ----------------

#: The tiny model of section 5.5. One model serves the step fixture, the loop
#: fixture, and the packed shard, so the three describe the same run.
MODEL = {
    "layers": 2,
    "embd": 32,
    "heads": 2,
    "kv_heads": 2,
    "vocab": 256,
    "padded_vocab": 256,
    "seq_len": 32,
    "window_pattern": "L",
}

#: The packer settings of section 5.5 and the reference defaults of section
#: 5.1: a 100-conversation buffer, a 2048-token render truncation, one rank,
#: and the ``last_step`` stop rule.
PACK = {
    "conversations": 8,
    "seq_len": MODEL["seq_len"],
    "buffer_size": 100,
    "render_max_tokens": 2048,
    "rank": 0,
    "world_size": 1,
    "stop_rule": "last_step",
}

#: The step settings of section 5.5 and the accumulation of section 5.1:
#: ``total_batch_size / (device_batch_size * seq_len * world_size)``.
STEP = {
    "steps": 8,
    "device_batch_size": 1,
    "grad_accum": 2,
    "seq_len": MODEL["seq_len"],
}

#: The loop settings of section 5.5: eight iterations, one evaluation at the
#: last step.
LOOP = {
    "num_iterations": 8,
    "device_batch_size": STEP["device_batch_size"],
    "grad_accum": STEP["grad_accum"],
    "seq_len": STEP["seq_len"],
    "eval_step": 8,
}

#: The optimizer configuration of sections 5.1 to 5.3, before the
#: ``init_lr_frac`` scale. The reference ``setup_optimizer`` derives the group
#: rates from these four values and the ``(hidden_dim / 768) ** -0.5`` scale.
OPT = {
    "unembedding_lr": 0.004,
    "embedding_lr": 0.3,
    "matrix_lr": 0.02,
    "scalar_lr": 0.5,
    "init_lr_frac": 0.8,
    "adam_eps": 1e-10,
    "clip": 0.0,
    "weight_decay": 0.0,
    "muon_ns_steps": 5,
    "muon_beta2": 0.9,
}

#: The schedule of section 5.2: no warmup, a half-length warmdown, a zero final
#: rate, and the Muon momentum ramp of 0.85 to 0.95 over 300 steps.
SCHEDULE = {
    "warmup_ratio": 0.0,
    "warmdown_ratio": 0.5,
    "final_lr_frac": 0.0,
    "muon_momentum_start": 0.85,
    "muon_momentum_peak": 0.95,
    "muon_momentum_final": 0.95,
    "muon_momentum_steps": 300,
}

#: The evaluation cadence of section 5.4. The pinned 8-step run reaches only
#: the final-step evaluation, so the interval is data, not a wait.
EVAL = {
    "eval_every": 200,
    "eval_tokens": 40 * 524288,
    "chatcore_every": 200,
    "chatcore_max_cat": -1,
    "chatcore_max_sample": 24,
}

#: The fixed seed of every fixture.
SEED = 1234

#: The committed fixture files that this module names.
FIXTURE_DIR = "tests/data"
PACK_FIXTURE = "sft_pack_fixture.bin"
STEP_FIXTURE = "sft_step_fixture.bin"
LOOP_FIXTURE = "sft_loop_fixture.bin"
ROWS_FIXTURE = "sft_rows_fixture.nchsft"
BASE_FIXTURE = "sft_base_fixture.ckpt"

#: The magic of the packed SFT shard (``src/record_file.h`` via
#: ``python/nanochat_cpp/sft.py``).
SHARD_MAGIC = b"NCSFT001"

#: The magic and the version of the pack fixture: ``NANCHAT1``, one ``uint32``
#: version word, then UTF-8 JSON, like ``tests/data/chat_fixture.bin``.
PACK_MAGIC = b"NANCHAT1"
PACK_VERSION = 1


# --- the conversation corpus (section 6.1) ---------------------------------

#: The sentence the long conversation repeats. Its words recur, so the fixture
#: tokenizer needs only a handful of pieces for the 2048-token truncation case.
LONG_SENTENCE = "The quick brown fox jumps over the lazy dog again. "

#: How often the long conversation repeats :data:`LONG_SENTENCE`. One repetition
#: renders to about twelve pieces, so this comfortably exceeds the 2048-token
#: render truncation.
LONG_REPEAT = 190

#: The eight conversations of section 6.1, in fixture order: a SmolTalk-like
#: exchange, an MMLU multiple-choice question, a GSM8K solution with tool
#: parts, a system message, two short exchanges, and one conversation above
#: 2048 tokens that the packer skips at fetch time.
CONVERSATIONS = (
    {
        "name": "smoltalk",
        "messages": [
            {"role": "user",
             "content": "Hello! How are you doing today?"},
            {"role": "assistant", "content": "I am doing well, thank you."},
        ],
    },
    {
        "name": "mmlu",
        "messages": [
            {"role": "user", "content": "What is 1+1?\nA) 1\nB) 2"},
            {"role": "assistant", "content": "B"},
        ],
    },
    {
        "name": "gsm8k_tool",
        "messages": [
            {"role": "user", "content": "What is 8 * 5?"},
            {"role": "assistant", "content": [
                {"type": "python", "text": "8 * 5"},
                {"type": "python_output", "text": "40"},
                {"type": "text", "text": "#### 40"},
            ]},
        ],
    },
    {
        "name": "system",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Name a color."},
            {"role": "assistant", "content": "Blue"},
        ],
    },
    {
        "name": "short",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hey"},
        ],
    },
    {
        "name": "capital",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris"},
        ],
    },
    {
        "name": "greeting",
        "messages": [
            {"role": "user", "content": "Say hello."},
            {"role": "assistant", "content": "Hello there, friend."},
        ],
    },
    {
        "name": "long",
        "messages": [
            {"role": "user", "content": LONG_SENTENCE * LONG_REPEAT},
            {"role": "assistant", "content": "Done"},
        ],
    },
)


def conversation_texts(conversations=CONVERSATIONS):
    """Every text fragment the conversations use, in fixture order.

    A fragment is one user or system message content, or one ``text``,
    ``python``, or ``python_output`` part of an assistant message.
    """
    texts = []
    for conversation in conversations:
        for message in conversation["messages"]:
            content = message["content"]
            if isinstance(content, str):
                texts.append(content)
                continue
            for part in content:
                texts.append(part["text"])
    return texts


def render_texts(conversations=CONVERSATIONS):
    """The strings ``render_conversation`` actually encodes.

    The renderer folds a leading system message into the first user turn with a
    blank line between them, exactly as
    ``nanochat_cpp.chat.render_conversation`` does. That merged string is a text
    fragment of the rendered conversation, so the vocabulary and the ``pieces``
    table cover it too.
    """
    texts = []
    for conversation in conversations:
        messages = list(conversation["messages"])
        if messages and messages[0]["role"] == "system":
            merged = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = [{"role": messages[1]["role"], "content": merged}]
            messages += list(conversation["messages"][2:])
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                texts.append(content)
                continue
            for part in content:
                texts.append(part["text"])
    return texts


def piece_texts(conversations=CONVERSATIONS):
    """Every text the fixture tokenizer must encode, deduplicated in order."""
    texts = []
    fragments = conversation_texts(conversations) + render_texts(conversations)
    for text in fragments:
        if text not in texts:
            texts.append(text)
    return texts


# --- the fixture tokenizer (section 6.1) -----------------------------------

#: The nine chat special tokens, in the reference ``nanochat.tokenizer`` order.
SPECIAL_TOKENS = (
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
)

#: The first token id available to text. The special tokens own the ids below.
FIRST_TEXT_ID = len(SPECIAL_TOKENS)

#: The name of the fixture tokenizer artifact. It is a piece table, not a
#: trained BPE artifact: the pinned model of section 5.5 has a 256-entry
#: vocabulary, and a ``vocab_size`` that small cannot hold a real BPE
#: vocabulary. The pack fixture pins the table by SHA-256, so the ids and the
#: text pieces cannot drift.
TOKENIZER_NAME = "fixture/sft_piece_table/v1"

#: The piece split. A word takes its leading space, a number is at most two
#: digits, punctuation runs together, and a whitespace run is one piece. This
#: is the reference split granularity without the Unicode classes: the fixture
#: corpus is ASCII, and the split must be reproducible from the standard
#: library alone.
PIECE_PATTERN = re.compile(r" ?[A-Za-z]+| ?\d{1,2}| ?[^A-Za-z\d\s]+|\s+")


class FixtureTokenizer:
    """The piece-table tokenizer the pack fixture pins.

    ``nanochat_cpp.chat.render_conversation`` needs three methods of a
    reference-style tokenizer: ``encode``, ``get_bos_token_id``, and the
    special-token map it reads through ``special_tokens``. This class provides
    exactly those, so the renderer under test runs unchanged.
    """

    def __init__(self, vocabulary, special_tokens, vocab_size):
        self.special_tokens = dict(special_tokens)
        self.vocabulary = dict(vocabulary)
        self._vocab_size = int(vocab_size)

    def get_bos_token_id(self) -> int:
        """The id of ``<|bos|>``, the token that starts every row."""
        return self.special_tokens["<|bos|>"]

    def get_vocab_size(self) -> int:
        """The model vocabulary size, including the unused tail of the ids."""
        return self._vocab_size

    def pieces(self, text: str) -> list[str]:
        """The pieces of ``text``, in order."""
        return PIECE_PATTERN.findall(text)

    def encode(self, text: str, prepend=None, append=None) -> list[int]:
        """The reference ids of ``text``, one id per piece.

        ``prepend`` and ``append`` mirror the reference signature, so a caller
        that passes a token id around a text fragment keeps working.
        """
        ids = [] if prepend is None else [int(prepend)]
        for piece in self.pieces(text):
            try:
                ids.append(self.vocabulary[piece])
            except KeyError as error:
                raise SystemExit(
                    f"the fixture tokenizer has no piece {piece!r}; the "
                    "vocabulary is built from the fixture corpus, so a text "
                    "fragment outside it is a bug in the generator") from error
        if append is not None:
            ids.append(int(append))
        return ids

    def __call__(self, text: str) -> list[int]:
        return self.encode(text)


def build_tokenizer(conversations=CONVERSATIONS) -> FixtureTokenizer:
    """Build the fixture tokenizer from the fixture corpus.

    The vocabulary is the distinct pieces of every text fragment, in
    first-appearance order, so the ids are deterministic and the table needs no
    stored artifact. The build fails when the corpus needs more text ids than
    the pinned vocabulary leaves, or when a fragment has a gap the split cannot
    cover: both are generator bugs, not fixture data.
    """
    vocabulary: dict[str, int] = {}
    for text in piece_texts(conversations):
        if "".join(PIECE_PATTERN.findall(text)) != text:
            raise SystemExit(
                "the piece split does not cover this fixture text: "
                f"{text[:48]!r}")
        for piece in PIECE_PATTERN.findall(text):
            if piece not in vocabulary:
                vocabulary[piece] = FIRST_TEXT_ID + len(vocabulary)
    limit = MODEL["vocab"] - FIRST_TEXT_ID
    if len(vocabulary) > limit:
        raise SystemExit(
            f"the fixture corpus needs {len(vocabulary)} text pieces but the "
            f"pinned vocabulary of {MODEL['vocab']} leaves {limit}")
    specials = {name: index for index, name in enumerate(SPECIAL_TOKENS)}
    return FixtureTokenizer(vocabulary, specials, MODEL["vocab"])


def token_bytes_table(tokenizer: FixtureTokenizer) -> list[int]:
    """The byte length of every token id, the table ``evaluate_bpb`` needs.

    This mirrors the reference ``get_token_bytes``: a special token reports the
    length of its own text, which the loss mask excludes anyway, and an unused
    id reports zero bytes, so it never contributes to bits-per-byte.
    """
    table = [0] * tokenizer.get_vocab_size()
    for name, token_id in tokenizer.special_tokens.items():
        table[token_id] = len(name.encode("utf-8"))
    for piece, token_id in tokenizer.vocabulary.items():
        table[token_id] = len(piece.encode("utf-8"))
    return table


def _tokenizer_digest(tokenizer: FixtureTokenizer) -> str:
    """The SHA-256 of the tokenizer definition, as a canonical JSON string."""
    definition = {
        "name": TOKENIZER_NAME,
        "special_tokens": tokenizer.special_tokens,
        "vocabulary": tokenizer.vocabulary,
    }
    body = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def tokenizer_record(tokenizer: FixtureTokenizer) -> dict:
    """The ``tokenizer`` object of section 6.1.

    It carries the artifact name, its SHA-256, the special-token map, the piece
    vocabulary that produced the ids, the per-id byte table, and the ``pieces``
    table that maps every fixture text fragment to its reference token ids. The
    phase C.1 test builds a tokenizer from the ``pieces`` table and the
    special-token map, so the table covers every fragment the conversations
    use.
    """
    pieces = {}
    for text in piece_texts():
        pieces[text] = tokenizer.encode(text)
    return {
        "name": TOKENIZER_NAME,
        "sha256": _tokenizer_digest(tokenizer),
        "special_tokens": dict(tokenizer.special_tokens),
        "vocabulary": [{"piece": piece, "id": token_id}
                       for piece, token_id in tokenizer.vocabulary.items()],
        "token_bytes": token_bytes_table(tokenizer),
        "pieces": pieces,
    }


# --- the JSON container ----------------------------------------------------


def _dump_sections(payload: dict) -> str:
    """Serialize ``payload`` with one line per section and one per sub-key.

    A fixture holds lists of thousands of ids, so an indented JSON dump would
    be thousands of lines long. One line per section keeps the committed file
    reviewable, and the arrays stay compact. Sorting the keys makes the bytes
    deterministic, which the SHA-256 pin of the fixtures list relies on.
    """
    lines = ["{"]
    items = sorted(payload.items())
    for index, (key, value) in enumerate(items):
        comma = "," if index + 1 < len(items) else ""
        if isinstance(value, dict) and value:
            lines.append(f"  {json.dumps(key)}: {{")
            entries = sorted(value.items())
            for position, (sub_key, sub_value) in enumerate(entries):
                tail = "," if position + 1 < len(entries) else ""
                body = json.dumps(sub_value, sort_keys=True,
                                  separators=(",", ":"))
                lines.append(f"    {json.dumps(sub_key)}:{body}{tail}")
            lines.append(f"  }}{comma}")
            continue
        body = json.dumps(value, sort_keys=True, separators=(",", ":"))
        lines.append(f"  {json.dumps(key)}:{body}{comma}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_json_fixture(path, payload: dict) -> Path:
    """Write ``NANCHAT1``, one version word, and the UTF-8 JSON body."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = _dump_sections(payload).encode("utf-8")
    target.write_bytes(PACK_MAGIC + struct.pack("<I", PACK_VERSION) + body)
    return target


def read_pack_fixture(path) -> dict:
    """Read a ``NANCHAT1`` JSON fixture and return the parsed object."""
    raw = Path(path).read_bytes()
    if raw[:8] != PACK_MAGIC:
        raise SystemExit(f"{path}: not a NANCHAT1 fixture")
    if len(raw) < 12:
        raise SystemExit(f"{path}: truncated NANCHAT1 fixture")
    (version,) = struct.unpack_from("<I", raw, 8)
    if version != PACK_VERSION:
        raise SystemExit(f"{path}: unsupported NANCHAT1 version {version}")
    return json.loads(raw[12:].decode("utf-8"))


# --- the two mixtures (section 6.1 and D8) ---------------------------------

#: The synthetic training mixture: a task name, a split, a synthetic length,
#: and an optional stop. The five lengths are the pins of D8; the names say
#: which reference task each stands in for, and the repeated MMLU and GSM8K
#: entries mirror the multipliers of the reference mixture.
MIXTURE = (
    ("smoltalk", "train", 7, None),
    ("mmlu", "auxiliary_train", 3, None),
    ("mmlu", "auxiliary_train", 3, None),
    ("gsm8k", "train", 2, None),
    ("gsm8k", "train", 2, None),
)

#: The synthetic validation mixture: the validating task set of section 5.1,
#: with the MMLU and GSM8K test caps as stops. The lengths are synthetic for
#: the same reason as the training set (D8).
VALIDATION_MIXTURE = (
    ("smoltalk", "test", 5, None),
    ("mmlu", "test", 3, 5200),
    ("gsm8k", "test", 2, 420),
)


class _SyntheticTask:
    """A stand-in task with a fixed length.

    ``nanochat_cpp.sft_data.TaskMixture`` reads only ``len(task)`` and
    ``task[index]``, so a fixed-length stand-in reproduces the reference
    shuffle without a network download. This is D8: the fixture pins the
    mixture algorithm, not the hub order.
    """

    def __init__(self, name, split, length, stop=None):
        self.name = name
        self.split = split
        self.stop = stop
        self.length = int(length)
        self.examples = [
            {"messages": [
                {"role": "user",
                 "content": f"{name} {split} example {index}"},
                {"role": "assistant", "content": f"{name} answer {index}"},
            ]}
            for index in range(self.length)
        ]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return self.examples[index]


def python_path() -> None:
    """Make the ``nanochat_cpp`` bridge importable from a fixture run."""
    python_dir = str(REPO_ROOT / "python")
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)


def _mixture_record(spec, seed):
    """The seed-``seed`` ``TaskMixture`` order of one synthetic task set."""
    python_path()
    from nanochat_cpp import sft_data  # noqa: PLC0415 - lazy, stdlib only

    mixture = sft_data.TaskMixture(
        [_SyntheticTask(*task) for task in spec])
    return {
        "seed": seed,
        "tasks": [
            {"name": name, "split": split, "length": length,
             **({} if stop is None else {"stop": stop})}
            for name, split, length, stop in spec
        ],
        "conversations": mixture.num_conversations,
        "order": [[task_index, local_index]
                  for task_index, local_index in mixture.index_map],
    }


def build_mixture_records(seed=42, real=False) -> tuple[dict, dict]:
    """The ``mixture`` and ``validation_mixture`` objects of D8.

    ``real`` replaces the synthetic task set with the reference hub tasks:
    that path downloads SmolTalk, MMLU, and GSM8K and needs ``datasets`` and
    ``pyarrow``. It exists so a reviewer can verify the real order by hand once
    (docs/sft-parity-plan.md section 10) and is not part of any gate.
    """
    if not real:
        return _mixture_record(MIXTURE, seed), _mixture_record(
            VALIDATION_MIXTURE, seed)

    python_path()
    from nanochat_cpp import sft_data  # noqa: PLC0415 - lazy

    try:
        train = sft_data.build_train_mixture()
        validation = sft_data.build_val_mixture()
    except Exception as error:  # noqa: BLE001 - the mode is manual only
        raise SystemExit(
            "--real-mixture needs the hub download, pyarrow, and a pinned "
            f"datasets version; the reference mixture failed: {error}")
    return (
        _real_record(train, {
            "mmlu_epochs": sft_data.MMLU_EPOCHS,
            "gsm8k_epochs": sft_data.GSM8K_EPOCHS,
        }),
        _real_record(validation, {
            "stops": {"mmlu": sft_data.VAL_MMLU_STOP,
                      "gsm8k": sft_data.VAL_GSM8K_STOP},
        }),
    )


def _real_record(mixture, extra: dict) -> dict:
    """One mixture record built from the downloaded reference tasks."""
    return {
        "kind": "real",
        "seed": 42,
        **extra,
        "tasks": [{"name": type(task).__name__, "length": len(task)}
                  for task in mixture.tasks],
        "conversations": mixture.num_conversations,
        "order": [[task_index, local_index]
                  for task_index, local_index in mixture.index_map],
    }


# --- the reference model, optimizer, and schedule --------------------------


def setup_reference():
    """Force the CPU and import torch, the reference package, and numpy.

    Returns the ``(torch, numpy, GPT, GPTConfig)`` quartet. The CPU pin and the
    disabled ``torch.compile`` mirror the ``train`` kind: the reference
    optimizer's fused steps are compiled by default, and an eager run keeps the
    fixture deterministic and needs no C++ compiler.
    """
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    import numpy as np
    import torch

    python_path()
    from nanochat_cpp.reference import ensure_reference_on_path

    ensure_reference_on_path()
    from nanochat.gpt import GPT, GPTConfig

    torch.set_num_threads(1)
    return torch, np, GPT, GPTConfig


def build_model(GPT, GPTConfig, np, seed: int = SEED):
    """The reference ``GPT`` of section 5.5, initialized at ``seed``."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    config = GPTConfig(
        sequence_len=MODEL["seq_len"],
        vocab_size=MODEL["vocab"],
        n_layer=MODEL["layers"],
        n_head=MODEL["heads"],
        n_kv_head=MODEL["kv_heads"],
        n_embd=MODEL["embd"],
        window_pattern=MODEL["window_pattern"],
    )
    model = GPT(config, pad_vocab_size_to=MODEL["padded_vocab"])
    model.init_weights()
    model.to(torch.float32)
    model.train()
    return model


def build_optimizer(model):
    """The reference optimizer at the pinned rates and ``init_lr_frac``.

    ``setup_optimizer`` hardcodes ``eps=1e-10`` and a ``0.95`` Muon momentum
    and derives the group rates from the four pinned values. D3 puts the
    ``init_lr_frac`` scale on the optimizer configuration: every group rate is
    multiplied once here, and ``initial_lr`` records the scaled rate the
    schedule multiplies per step.
    """
    optimizer = model.setup_optimizer(
        unembedding_lr=OPT["unembedding_lr"],
        embedding_lr=OPT["embedding_lr"],
        matrix_lr=OPT["matrix_lr"],
        weight_decay=OPT["weight_decay"],
        scalar_lr=OPT["scalar_lr"],
    )
    for group in optimizer.param_groups:
        if group.get("kind") == "adamw":
            group["eps"] = OPT["adam_eps"]
        if group.get("kind") == "muon":
            group["ns_steps"] = OPT["muon_ns_steps"]
            group["beta2"] = OPT["muon_beta2"]
        group["lr"] = group["lr"] * OPT["init_lr_frac"]
        group["initial_lr"] = group["lr"]
    return optimizer


def materialize_moments(optimizer) -> None:
    """Create the zero moments the reference allocates on its first step.

    ``MuonAdamW`` builds its state lazily, so a fresh optimizer holds no
    moments and a checkpoint at step 0 would carry none. This creates the same
    tensors the reference would create, zero-initialized, in the single-rank
    shape: one AdamW pair per parameter and one stacked Muon pair per group.
    """
    import torch

    for group in optimizer.param_groups:
        params = list(group["params"])
        if group.get("kind") == "muon":
            first = params[0]
            shape = first.shape
            chunk = len(params)
            state = optimizer.state[first]
            state["step"] = 0
            state.setdefault("momentum_buffer", torch.zeros(
                chunk, *shape, dtype=first.dtype))
            second_shape = ((chunk, shape[-2], 1) if shape[-2] >= shape[-1]
                            else (chunk, 1, shape[-1]))
            state.setdefault("second_momentum_buffer", torch.zeros(
                second_shape, dtype=first.dtype))
            continue
        for parameter in params:
            state = optimizer.state[parameter]
            state["step"] = 0
            state.setdefault("exp_avg", torch.zeros_like(parameter))
            state.setdefault("exp_avg_sq", torch.zeros_like(parameter))


def adamw_kinds() -> tuple:
    """The reference AdamW group names, in ``setup_optimizer`` order."""
    python_path()
    from nanochat_cpp import checkpoint  # noqa: PLC0415 - lazy

    return checkpoint.ADAMW_KINDS


def group_rates(optimizer) -> dict:
    """The effective per-group rate every reference group starts with.

    This is the section 5.2 rate table after the ``(hidden_dim / 768) ** -0.5``
    scale and the ``init_lr_frac`` of D3, in the group order
    ``setup_optimizer`` builds: six AdamW groups and then the Muon groups,
    which share one rate. The C++ optimizer must reproduce these values from
    the four pinned rates.
    """
    names = adamw_kinds()
    rates = {}
    adamw_index = 0
    for group in optimizer.param_groups:
        if group.get("kind") == "muon":
            rates["muon"] = float(group["initial_lr"])
            continue
        if adamw_index >= len(names):
            raise SystemExit("the optimizer has more AdamW groups than the "
                             "reference defines")
        rates[names[adamw_index]] = float(group["initial_lr"])
        adamw_index += 1
    return rates


def lr_multiplier(progress: float) -> float:
    """The reference learning-rate multiplier at ``progress`` (D1)."""
    warmup = SCHEDULE["warmup_ratio"]
    warmdown = SCHEDULE["warmdown_ratio"]
    if progress < warmup:
        return (progress + 1e-8) / warmup
    if progress <= 1.0 - warmdown:
        return 1.0
    decay = (progress - (1.0 - warmdown)) / warmdown
    return 1.0 + decay * (SCHEDULE["final_lr_frac"] - 1.0)


def muon_momentum(it: int) -> float:
    """The reference Muon momentum at the 0-based step ``it`` (D2)."""
    start = SCHEDULE["muon_momentum_start"]
    peak = SCHEDULE["muon_momentum_peak"]
    steps = SCHEDULE["muon_momentum_steps"]
    frac = min(it / steps, 1.0)
    return (1.0 - frac) * start + frac * peak


def apply_schedule(optimizer, progress: float, it: int) -> tuple[float, float]:
    """Set the group rates and the Muon momentum, and return both values."""
    multiplier = lr_multiplier(progress)
    momentum = muon_momentum(it)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * multiplier
        if group.get("kind") == "muon":
            group["momentum"] = momentum
    return multiplier, momentum


def total_param_l2(model) -> float:
    """The L2 norm of every parameter, the compact trajectory checksum."""
    total = 0.0
    for parameter in model.parameters():
        total += float(parameter.detach().double().pow(2).sum())
    return math.sqrt(total)


def total_grad_norm(model) -> float:
    """The global gradient norm over every parameter that has a gradient."""
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().double().pow(2).sum())
    return math.sqrt(total)


# --- the base checkpoint (section 6.4) -------------------------------------


def checkpoint_records(model, optimizer) -> tuple[list, list]:
    """The parameter and the optimizer-state records of the base checkpoint.

    The records come from ``nanochat_cpp.checkpoint``, the module behind
    ``tools/convert_checkpoint.py``, so the generator holds no copy of the
    container layout and the names are the canonical C++ names. Both lists are
    sorted by the module's write-order keys, so the committed file matches a
    C++ ``Checkpointer`` output byte for byte.
    """
    python_path()
    from nanochat_cpp import checkpoint  # noqa: PLC0415 - lazy

    parameters = checkpoint.build_records(model.state_dict(), "float32")
    parameters.sort(key=lambda record: checkpoint.order_key(record.name))
    moments = checkpoint.build_optimizer_records(optimizer.state_dict())
    moments.sort(
        key=lambda record: checkpoint.optimizer_order_key(record.name))
    return parameters, moments


def write_base_checkpoint(path, model, optimizer) -> tuple[list, list]:
    """Write the reference parameters and moments at step 0."""
    python_path()
    from nanochat_cpp import checkpoint  # noqa: PLC0415 - lazy

    parameters, moments = checkpoint_records(model, optimizer)
    checkpoint.write_checkpoint(path, parameters, moments)
    return parameters, moments


def _tensor_from_record(record, torch):
    """One fp32 container record as a torch tensor of its recorded shape."""
    tensor = torch.frombuffer(bytearray(record.data), dtype=torch.float32)
    return tensor.reshape(tuple(int(dim) for dim in record.shape)).clone()


def load_state(path, model, optimizer) -> None:
    """Restore the parameters and the moments from a base checkpoint.

    The weight load is strict: the container must name exactly the parameters
    the reference model registers, so a rename on either side fails here
    instead of in a parity test.
    """
    python_path()
    import torch  # noqa: PLC0415 - lazy
    from nanochat_cpp import checkpoint  # noqa: PLC0415 - lazy

    records = checkpoint.read_checkpoint(path)
    state = {record.name: _tensor_from_record(record, torch)
             for record in records}
    model.load_state_dict(state, strict=True)
    restore_optimizer_state(optimizer, checkpoint.read_optimizer_state(path))


def restore_optimizer_state(optimizer, records) -> None:
    """Load the moment records into the reference optimizer.

    The C++ side stores the moments as group buffers: one AdamW pair per group,
    concatenated over the group's parameters, and one stacked pair per Muon
    shape. This inverts ``nanochat_cpp.checkpoint.build_optimizer_records``, so
    the two sides stay one layout definition apart.
    """
    import torch  # noqa: PLC0415 - lazy

    python_path()
    from nanochat_cpp import checkpoint  # noqa: PLC0415 - lazy

    by_name = {record.name: record for record in records}
    adamw_index = 0
    for group in optimizer.param_groups:
        params = list(group["params"])
        if group.get("kind") == "muon":
            first = params[0]
            rows, cols = int(first.shape[-2]), int(first.shape[-1])
            chunk = len(params)
            buffer1 = by_name.get(checkpoint.muon_state_name(rows, cols,
                                                             "buf1"))
            buffer2 = by_name.get(checkpoint.muon_state_name(rows, cols,
                                                             "buf2"))
            if buffer1 is None or buffer2 is None:
                raise SystemExit(
                    "the base checkpoint has no Muon moments for the "
                    f"{rows}x{cols} group")
            state = optimizer.state[first]
            state["step"] = 0
            state["momentum_buffer"] = _tensor_from_record(
                buffer1, torch).reshape(chunk, *first.shape)
            second_shape = ((chunk, rows, 1) if rows >= cols
                            else (chunk, 1, cols))
            state["second_momentum_buffer"] = _tensor_from_record(
                buffer2, torch).reshape(second_shape)
            continue
        if adamw_index >= len(checkpoint.ADAMW_KINDS):
            raise SystemExit("the optimizer has more AdamW groups than the "
                             "container defines")
        kind = checkpoint.ADAMW_KINDS[adamw_index]
        adamw_index += 1
        for key, attribute in (("m", "exp_avg"), ("v", "exp_avg_sq")):
            record = by_name.get(f"adamw.{kind}.{key}")
            if record is None:
                raise SystemExit(
                    f"the base checkpoint has no moment adamw.{kind}.{key}")
            flat = _tensor_from_record(record, torch)
            offset = 0
            for parameter in params:
                count = int(parameter.numel())
                state = optimizer.state[parameter]
                state.setdefault("step", 0)
                state[attribute] = flat[offset:offset + count].reshape(
                    parameter.shape).clone()
                offset += count
            if offset != int(flat.numel()):
                raise SystemExit(
                    f"the moment adamw.{kind}.{key} holds {flat.numel()} "
                    f"values, the group holds {offset}")


# --- the packed shard (section 6.4) ----------------------------------------


def load_shard(path) -> list[tuple[list[int], list[int]]]:
    """The rows of an ``NCSFT001`` shard as ``(tokens, targets)`` pairs.

    The reader is ``nanochat_cpp.eval_fixture.read`` with the SFT magic, so the
    generator holds no copy of the record layout (``src/record_file.h``).
    """
    python_path()
    from nanochat_cpp import eval_fixture  # noqa: PLC0415 - lazy

    records = eval_fixture.read(path, magic=SHARD_MAGIC)
    by_name = {record.name: record for record in records}
    missing = [name for name in ("tokens", "targets") if name not in by_name]
    if missing:
        raise SystemExit(f"{path}: the shard lacks the records {missing}")
    tokens = by_name["tokens"]
    targets = by_name["targets"]
    if tuple(tokens.shape) != tuple(targets.shape):
        raise SystemExit(f"{path}: tokens and targets disagree on shape")
    rows, width = int(tokens.shape[0]), int(tokens.shape[1])
    inputs = struct.unpack(f"<{rows * width}i", tokens.data)
    expected = struct.unpack(f"<{rows * width}i", targets.data)
    return [
        (list(inputs[row * width:(row + 1) * width]),
         list(expected[row * width:(row + 1) * width]))
        for row in range(rows)
    ]


def row_cycle(rows):
    """Yield the shard rows in order, wrapping with an epoch counter.

    The reference stream is finite: one pass over the conversation list yields
    about one epoch of rows. A pinned loop that runs longer than one pass
    therefore cycles the shard, and ``epoch`` counts the wraps the way the
    reference fetch cursor does.
    """
    if not rows:
        raise SystemExit("the packed shard is empty")
    epoch = 1
    while True:
        for row in rows:
            yield row, epoch
        epoch += 1


def evaluate_bpb(model, rows, token_bytes) -> float:
    """Reference bits-per-byte over one pass of the packed rows (D6).

    This transcribes ``nanochat.eval.loss_eval.evaluate_bpb``: sum the
    negative log-probability of every valid target, sum the bytes those targets
    represent, and divide. A target of ``-1`` and a token whose byte length is
    zero contribute nothing.
    """
    import torch  # noqa: PLC0415 - lazy

    table = torch.tensor(token_bytes, dtype=torch.int32)
    total_nats = 0.0
    total_bytes = 0
    for tokens, targets in rows:
        x = torch.tensor([tokens], dtype=torch.long)
        y = torch.tensor([targets], dtype=torch.long)
        losses = model(x, y, loss_reduction="none").view(-1)
        flat = y.view(-1)
        valid = flat >= 0
        safe = torch.where(valid, flat, torch.zeros_like(flat))
        counts = torch.where(valid, table[safe], torch.zeros_like(flat))
        total_nats += float((losses * (counts > 0)).sum())
        total_bytes += int(counts.sum())
    if total_bytes <= 0:
        raise SystemExit("the packed rows hold no valid target bytes")
    return total_nats / (math.log(2) * total_bytes)


# --- small value helpers ---------------------------------------------------


def i32(values) -> "object":
    """One int32 fixture record payload."""
    import numpy as np  # noqa: PLC0415 - lazy

    return np.asarray(values, dtype=np.int32)


def u8(text: str) -> "object":
    """One uint8 fixture record payload holding the bytes of ``text``."""
    import numpy as np  # noqa: PLC0415 - lazy

    return np.frombuffer(text.encode("ascii"), dtype=np.uint8).copy()


def file_sha256(path) -> str:
    """The SHA-256 of one fixture file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_checksums(paths) -> int:
    """Print the SHA-256 of every file a kind wrote, in the checkpoint style.

    The ``fixtures`` node copies these lines into
    ``tests/data/sft_fixtures.txt`` without a second derivation, so the printed
    digest is the committed pin.
    """
    for path in paths:
        size = Path(path).stat().st_size
        print(f"{Path(path).name:<24} sha256 {file_sha256(path)}  "
              f"{size} bytes")
    return 0
