"""Torch-free self-test for the bridge's pure logic.

Run with ``tools/nanochat_cpp selftest`` (or
``PYTHONPATH=python python3 -m nanochat_cpp.selftest``). It checks the
configuration math, the token-count estimate, the NANO shard round-trip, and
the reference-checkpoint name/dtype/container/path logic without importing
torch or the reference package, so it runs anywhere.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from . import checkpoint, config, data


def _check(condition: bool, message: str) -> int:
    if not condition:
        print(f"FAIL: {message}", file=sys.stderr)
        return 1
    return 0


def _check_raises(error: type[BaseException], function, message: str) -> int:
    try:
        function()
    except error:
        return 0
    except Exception as other:  # noqa: BLE001
        print(f"FAIL: {message} (raised {type(other).__name__})", file=sys.stderr)
        return 1
    print(f"FAIL: {message} (did not raise)", file=sys.stderr)
    return 1


def _check_config_math() -> int:
    failures = 0
    # model_shape mirrors build_model_meta: ceil(depth*aspect/head_dim)*head_dim.
    cases = [
        (2, 128, 1),
        (6, 384, 3),
        (8, 512, 4),
        (20, 1280, 10),
    ]
    for depth, expected_dim, expected_heads in cases:
        dim, heads = config.model_shape(depth, 64, 128)
        failures += _check(dim == expected_dim and heads == expected_heads,
                           f"model_shape({depth}) = {dim}/{heads}")

    failures += _check(config.padded_vocab(32768) == 32768,
                       "padded_vocab(32768)")
    failures += _check(config.padded_vocab(32769) == 32832,
                       "padded_vocab(32769)")
    failures += _check(config.padded_vocab(1) == 64, "padded_vocab(1)")

    failures += _check(
        data.tokens_for_run(4096, 50, 512) == 4096 * 50 + 512 + 2048,
        "tokens_for_run")
    return failures


def _check_shards() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for width, limit in ((2, 65535), (4, 1 << 20)):
            path = Path(tmp) / f"shard{width}.bin"
            tokens = [i % limit for i in range(1000)]
            with data.ShardWriter(path, width) as writer:
                writer.write(tokens[:400])
                writer.write(tokens[400:])
            count, got_width, got = data.read_shard(path)
            failures += _check(count == len(tokens), f"shard{width} count")
            failures += _check(got_width == width, f"shard{width} width")
            failures += _check(got == tokens, f"shard{width} tokens")
    return failures


# Every name the C++ model registers (src/model.cc); the reference state dict
# uses the same names, so the remap is an identity for these.
_CANONICAL_NAMES = [
    "lm_head.weight",
    "transformer.wte.weight",
    "value_embeds.1.weight",
    "resid_lambdas",
    "x0_lambdas",
    "smear_gate.weight",
    "smear_lambda",
    "backout_lambda",
    "transformer.h.0.attn.c_q.weight",
    "transformer.h.0.attn.c_k.weight",
    "transformer.h.0.attn.c_v.weight",
    "transformer.h.0.attn.c_proj.weight",
    "transformer.h.0.attn.ve_gate.weight",
    "transformer.h.0.mlp.c_fc.weight",
    "transformer.h.0.mlp.c_proj.weight",
]


def _check_name_remap() -> int:
    failures = 0
    for name in _CANONICAL_NAMES:
        failures += _check(checkpoint.remap_name(name) == name,
                           f"remap identity {name}")
        failures += _check(checkpoint.canonical_name(name) == name,
                           f"canonical name {name}")

    # torch.compile and DDP wrappers are stripped, repeatedly.
    failures += _check(
        checkpoint.remap_name("_orig_mod.transformer.wte.weight")
        == "transformer.wte.weight", "remap _orig_mod prefix")
    failures += _check(
        checkpoint.remap_name("module._orig_mod.lm_head.weight")
        == "lm_head.weight", "remap nested wrapper prefixes")

    # Legacy projection aliases fold onto the canonical name.
    failures += _check(
        checkpoint.remap_name("transformer.h.3.attn.proj.weight")
        == "transformer.h.3.attn.c_proj.weight", "remap attn.proj alias")
    failures += _check(
        checkpoint.remap_name("transformer.h.3.mlp.proj.weight")
        == "transformer.h.3.mlp.c_proj.weight", "remap mlp.proj alias")

    # Anything the C++ model does not define is rejected, not dropped silently.
    failures += _check(checkpoint.remap_name("optimizer.momentum") is None,
                       "unknown key returns None")
    failures += _check_raises(
        checkpoint.UnknownParameter,
        lambda: checkpoint.canonical_name("optimizer.momentum"),
        "canonical_name rejects unknown key")
    return failures


def _check_order_key() -> int:
    failures = 0
    names = [
        "lm_head.weight",
        "transformer.wte.weight",
        "value_embeds.1.weight",
        "value_embeds.7.weight",
        "resid_lambdas",
        "x0_lambdas",
        "smear_gate.weight",
        "smear_lambda",
        "backout_lambda",
        "transformer.h.0.attn.c_q.weight",
        "transformer.h.0.attn.c_k.weight",
        "transformer.h.0.attn.c_v.weight",
        "transformer.h.0.attn.c_proj.weight",
        "transformer.h.0.attn.ve_gate.weight",
        "transformer.h.0.mlp.c_fc.weight",
        "transformer.h.0.mlp.c_proj.weight",
        "transformer.h.1.attn.c_q.weight",
    ]
    keys = [checkpoint.order_key(name) for name in names]
    failures += _check(keys == sorted(keys) and len(set(keys)) == len(keys),
                       "order_key is a strict registration order")
    return failures


def _check_dtype_policy() -> int:
    failures = 0

    failures += _check(checkpoint.container_dtype("fp32") == "float32",
                       "container_dtype fp32")
    failures += _check(checkpoint.container_dtype("float16") == "float16",
                       "container_dtype float16")
    failures += _check(checkpoint.container_dtype("half") == "float16",
                       "container_dtype half")
    failures += _check(checkpoint.container_dtype("torch.float32") == "float32",
                       "container_dtype torch.float32")
    failures += _check(checkpoint.dtype_code("float32") == 0,
                       "dtype_code float32")
    failures += _check(checkpoint.dtype_code("fp16") == 1,
                       "dtype_code fp16")
    failures += _check(checkpoint.dtype_name(1) == "float16",
                       "dtype_name 1")

    # bfloat16 is a valid source but not a container dtype.
    failures += _check_raises(
        checkpoint.UnsupportedDtype,
        lambda: checkpoint.container_dtype("bfloat16"),
        "container_dtype rejects bfloat16")

    upcast = checkpoint.plan_conversion("bfloat16", "float32")
    failures += _check(upcast.code == 0 and upcast.upcast and not upcast.lossy,
                       "plan_conversion bf16 -> fp32")
    downcast = checkpoint.plan_conversion("float32", "float16")
    failures += _check(
        downcast.code == 1 and downcast.downcast and downcast.lossy,
        "plan_conversion fp32 -> fp16")
    exact = checkpoint.plan_conversion("float16", "float16")
    failures += _check(not exact.upcast and not exact.downcast and not exact.lossy,
                       "plan_conversion fp16 -> fp16")
    range_loss = checkpoint.plan_conversion("bfloat16", "float16")
    failures += _check(range_loss.lossy, "plan_conversion bf16 -> fp16 is lossy")
    return failures


def _check_container() -> int:
    failures = 0
    records = [
        # lm_head.weight: 2x3 fp32 (rank 2, 24 bytes).
        checkpoint.TensorRecord("lm_head.weight", checkpoint.DTYPE_FP32,
                                (2, 3), bytes(2 * 3 * 4)),
        # resid_lambdas: 2 fp16 (rank 1, 4 bytes).
        checkpoint.TensorRecord("resid_lambdas", checkpoint.DTYPE_FP16,
                                (2,), bytes(2 * 2)),
    ]
    failures += _check(checkpoint.serialize(records)[:8] == checkpoint.MAGIC,
                       "container magic")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.nchkpt01"
        checkpoint.write_checkpoint(path, records)
        loaded = checkpoint.read_checkpoint(path)
        failures += _check(loaded == records, "container round-trip")
        failures += _check(loaded[0].dtype_name == "float32",
                           "container dtype name")
        failures += _check(loaded[0].numel() == 6, "container numel")

    # A payload-size mismatch is rejected rather than written.
    bad = [checkpoint.TensorRecord("lm_head.weight", checkpoint.DTYPE_FP32,
                                   (2, 3), bytes(3))]
    failures += _check_raises(
        checkpoint.CheckpointError,
        lambda: checkpoint.serialize(bad),
        "container rejects short payload")
    return failures


def _check_path_resolution() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        # Largest d<N> tag wins; d8 has the last step, d20 the deeper tag.
        d8 = base / "base_checkpoints" / "d8"
        d20 = base / "base_checkpoints" / "d20"
        chat = base / "chatsft_checkpoints" / "chat"
        for directory in (d8, d20, chat):
            directory.mkdir(parents=True)
        # An empty (or optimizer-only) tag directory must not be selected.
        (base / "base_checkpoints" / "scratch").mkdir()
        (d8 / "model_000002.pt").write_bytes(b"")
        (d8 / "model_000010.pt").write_bytes(b"")
        (d8 / "meta_000010.json").write_text("{}")
        (d20 / "model_000050.pt").write_bytes(b"")
        (chat / "model_000499.pt").write_bytes(b"")

        resolved = checkpoint.resolve("base", base_dir=base)
        failures += _check(resolved.model_tag == "d20",
                           f"auto tag = {resolved.model_tag}")
        failures += _check(resolved.step == 50, f"auto step = {resolved.step}")
        failures += _check(resolved.model_path.name == "model_000050.pt",
                           "auto model path")

        resolved = checkpoint.resolve("base", "d8", base_dir=base)
        failures += _check(resolved.step == 10, "last step when unspecified")
        failures += _check(resolved.meta_path.name == "meta_000010.json",
                           "meta sidecar name")

        resolved = checkpoint.resolve("base", "d8", 2, base)
        failures += _check(resolved.model_path.name == "model_000002.pt",
                           "explicit step")

        resolved = checkpoint.resolve("sft", "chat", 499, base)
        failures += _check(
            resolved.model_path
            == base / "chatsft_checkpoints" / "chat" / "model_000499.pt",
            "sft source path")

        failures += _check_raises(
            checkpoint.CheckpointError,
            lambda: checkpoint.resolve("bogus", base_dir=base),
            "unknown source is rejected")
    return failures


def main() -> int:
    failures = 0
    failures += _check_config_math()
    failures += _check_shards()
    failures += _check_name_remap()
    failures += _check_order_key()
    failures += _check_dtype_policy()
    failures += _check_container()
    failures += _check_path_resolution()

    if failures:
        print(f"selftest: {failures} failure(s)", file=sys.stderr)
        return 1
    print("selftest: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
