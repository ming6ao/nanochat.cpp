"""The reference side of the SFT mixture and renderer (Z2).

``nanochat_cpp.sft_data`` (Z1) owns the packer and the mixture builders, and
they stay importable with no extra installed. What they cannot do is reach the
Hub, so the reference mixture -- the counterpart of the core's
``build_train_mixture`` and ``build_val_mixture`` -- lives here
(``docs/dependency-layering-plan.md`` section 11):

* SmolTalk (``HuggingFaceTB/smol-smoltalk``), the train split, about 460K
  general conversations;
* MMLU (``cais/mmlu``), the ``auxiliary_train`` split, three passes by
  default;
* GSM8K (``openai/gsm8k``), the ``main`` train split, four passes by default.

Validation uses the matching test splits, capped at the reference row counts,
so the validation mixture has the same task ratios as the training mixture.

Two seams keep the core free of an import (gate G5). :func:`mixture_builder`
returns one callable that a host hands to the core in place of an import::

    builder = sft_reference.mixture_builder()
    conversations = builder("train")          # or "val" / "demo"

and :func:`renderer` returns the reference conversation renderer, so a parity
run renders with the reference ``Tokenizer.render_conversation`` instead of the
core's pinned mirror.

Extras: ``reference``, ``nanochat``, and ``hfdata`` (the loader of
:mod:`nanochat_cpp_bridge.reference_tasks`).
"""

from __future__ import annotations

from typing import Any, Callable

from nanochat_cpp.sft_data import RENDER_MAX_TOKENS, TaskMixture
from nanochat_cpp.tasks import GSM8K, MMLU, Task

from .reference_tasks import load_hub_dataset

__all__ = [
    "EXTRAS",
    "GSM8K_EPOCHS",
    "HubGSM8K",
    "HubMMLU",
    "MIXTURE_KINDS",
    "MMLU_EPOCHS",
    "RENDER_MAX_TOKENS",
    "SMOLLTALK_REPO",
    "SmolTalk",
    "VAL_GSM8K_STOP",
    "VAL_MMLU_STOP",
    "build_mixture",
    "build_train_mixture",
    "build_val_mixture",
    "demo_mixture",
    "mixture_builder",
    "renderer",
]

#: The extras this module needs, in the order a caller installs them.
EXTRAS = ("reference", "nanochat", "hfdata")

#: The SmolTalk repository the reference mixture starts from.
SMOLLTALK_REPO = "HuggingFaceTB/smol-smoltalk"

#: The reference multipliers of ``scripts/chat_sft.py``.
MMLU_EPOCHS = 3
GSM8K_EPOCHS = 4

#: The reference validation caps: the MMLU test split at 5.2K rows and the
#: GSM8K test split at 420 rows.
VAL_MMLU_STOP = 5200
VAL_GSM8K_STOP = 420

#: The kinds :func:`mixture_builder` accepts.
MIXTURE_KINDS = ("train", "val", "demo")


class SmolTalk(Task):
    """SmolTalk general conversations (mirrors ``tasks.smoltalk.SmolTalk``).

    The dataset comes from the reference hub loader, and the row extraction is
    the reference sanity check, so a bad conversation fails here instead of
    inside the renderer.
    """

    def __init__(self, split: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SmolTalk split must be train|test"
        self.ds = load_hub_dataset(SMOLLTALK_REPO, split=split).shuffle(seed=42)

    def num_examples(self) -> int:
        return len(self.ds)

    def get_example(self, index: int) -> dict:
        messages = self.ds[index]["messages"]
        # The reference sanity checks. An optional system message stays.
        assert len(messages) >= 1
        rest = messages[1:] if messages[0]["role"] == "system" else messages
        assert len(rest) >= 2, "SmolTalk messages must have at least 2 messages"
        for i, message in enumerate(rest):
            expected = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected, (
                f"Message {i} has role {message['role']} but should be "
                f"{expected}")
            assert isinstance(message["content"], str), (
                "Content must be a string")
        return {"messages": messages}


class HubMMLU(MMLU):
    """MMLU with the reference hub loader wired in.

    The prompt rendering, the example shape, and the criterion are inherited
    from the core's :class:`nanochat_cpp.tasks.MMLU`; only the dataset access
    is this package's, because that is the half that needs ``pyarrow`` and
    ``filelock``. The constructor repeats the reference argument checks instead
    of calling the core's, which would reach for the core's loader seam.
    """

    def __init__(self, subset: str, split: str, **kwargs: Any) -> None:
        Task.__init__(self, **kwargs)
        assert subset in ["all"], f"subset {subset} must be all"
        assert split in ["auxiliary_train", "validation", "dev", "test"], (
            f"split {split} must be auxiliary_train|validation|dev|test")
        self.subset = subset
        self.split = split
        self.ds = load_hub_dataset("cais/mmlu", subset, split=split).shuffle(
            seed=42)


class HubGSM8K(GSM8K):
    """GSM8K with the reference hub loader wired in.

    The calculator-tag rendering and the answer extraction are inherited from
    the core's :class:`nanochat_cpp.tasks.GSM8K`; the dataset access is this
    package's, for the same reason as :class:`HubMMLU`.
    """

    def __init__(self, subset: str, split: str, **kwargs: Any) -> None:
        Task.__init__(self, **kwargs)
        assert subset in ["main", "socratic"], (
            "GSM8K subset must be main|socratic")
        assert split in ["train", "test"], "GSM8K split must be train|test"
        self.ds = load_hub_dataset("openai/gsm8k", subset, split=split).shuffle(
            seed=42)


def build_train_mixture(mmlu_epochs: int = MMLU_EPOCHS,
                        gsm8k_epochs: int = GSM8K_EPOCHS) -> TaskMixture:
    """The reference SFT training mixture.

    The default multipliers match ``scripts/chat_sft.py``: SmolTalk once, MMLU
    three times, and GSM8K four times.
    """
    tasks = [SmolTalk(split="train")]
    tasks += [HubMMLU(subset="all", split="auxiliary_train")
              for _ in range(mmlu_epochs)]
    tasks += [HubGSM8K(subset="main", split="train")
              for _ in range(gsm8k_epochs)]
    return TaskMixture(tasks)


def build_val_mixture() -> TaskMixture:
    """The reference SFT validation mixture.

    The test splits match the training task ratios: all of SmolTalk test, the
    first 5.2K MMLU test rows, and the first 420 GSM8K test rows.
    """
    return TaskMixture([
        SmolTalk(split="test"),
        HubMMLU(subset="all", split="test", stop=VAL_MMLU_STOP),
        HubGSM8K(subset="main", split="test", stop=VAL_GSM8K_STOP),
    ])


def demo_mixture(limit: int = 64) -> TaskMixture:
    """A small real-data mixture for a notebook or a smoke run.

    This calls the same datasets as :func:`build_train_mixture`, but it takes
    only the first ``limit`` rows of each task. The hub shards still download in
    full, because the loader fetches a shard at a time.
    """
    return TaskMixture([
        SmolTalk(split="train", stop=limit),
        HubMMLU(subset="all", split="auxiliary_train", stop=limit),
        HubGSM8K(subset="main", split="train", stop=limit),
    ])


def build_mixture(kind: str = "train", **options: Any) -> TaskMixture:
    """Build one reference mixture by name.

    ``kind`` is ``train``, ``val``, or ``demo``; ``options`` are the keyword
    arguments of the matching builder (``mmlu_epochs`` and ``gsm8k_epochs``
    for train, ``limit`` for demo).
    """
    if kind == "train":
        return build_train_mixture(**options)
    if kind == "val":
        return build_val_mixture(**options)
    if kind == "demo":
        return demo_mixture(**options)
    raise ValueError(
        f"unknown SFT mixture {kind!r}; expected one of "
        + ", ".join(MIXTURE_KINDS))


def mixture_builder() -> Callable[..., TaskMixture]:
    """The reference mixture builders, as one callable a core caller is handed.

    The returned object is :func:`build_mixture`: ``builder("train")``,
    ``builder("val")``, or ``builder("demo", limit=8)``. A host injects it
    through its own seam, so the core names no Z2 module (gate G5).
    """
    return build_mixture


def renderer(max_tokens: int = RENDER_MAX_TOKENS) -> Callable[..., Any]:
    """The reference conversation renderer, as one callable.

    The returned object takes ``(conversation, tokenizer)`` and returns
    ``(ids, mask)``, the shape the core packer calls. A reference
    ``nanochat.tokenizer.Tokenizer`` renders with its own
    ``render_conversation``, which is the parity reference; any other
    tokenizer falls back to the core's pinned mirror
    (:func:`nanochat_cpp.chat.render_conversation`), so a fake tokenizer in a
    test still works.
    """
    def render(conversation, tokenizer):
        reference_render = getattr(tokenizer, "render_conversation", None)
        if callable(reference_render):
            return reference_render(conversation, max_tokens=max_tokens)
        from nanochat_cpp.chat import (  # noqa: PLC0415 - Z1, imported on use
            render_conversation,
        )
        return render_conversation(conversation, tokenizer,
                                   max_tokens=max_tokens)

    return render
