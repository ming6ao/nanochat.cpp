"""T0: the reference-checkpoint name, dtype, container, and path logic.

This replaces the removed self-test checks and is hermetic: standard library
only. See docs/python.md section 11.1.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nanochat_cpp import checkpoint

CANONICAL_NAMES = [
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


class NameRemapTest(unittest.TestCase):
    def test_canonical_names_are_identity(self) -> None:
        for name in CANONICAL_NAMES:
            self.assertEqual(checkpoint.remap_name(name), name)
            self.assertEqual(checkpoint.canonical_name(name), name)

    def test_wrappers_are_stripped(self) -> None:
        self.assertEqual(checkpoint.remap_name("_orig_mod.transformer.wte.weight"),
                         "transformer.wte.weight")
        self.assertEqual(
            checkpoint.remap_name("module._orig_mod.lm_head.weight"),
            "lm_head.weight")

    def test_legacy_aliases_fold(self) -> None:
        self.assertEqual(
            checkpoint.remap_name("transformer.h.3.attn.proj.weight"),
            "transformer.h.3.attn.c_proj.weight")
        self.assertEqual(
            checkpoint.remap_name("transformer.h.3.mlp.proj.weight"),
            "transformer.h.3.mlp.c_proj.weight")

    def test_unknown_keys_are_rejected(self) -> None:
        self.assertIsNone(checkpoint.remap_name("optimizer.momentum"))
        with self.assertRaises(checkpoint.UnknownParameter):
            checkpoint.canonical_name("optimizer.momentum")


class OrderKeyTest(unittest.TestCase):
    def test_order_key_is_a_strict_registration_order(self) -> None:
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
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(set(keys)), len(keys))


class DtypePolicyTest(unittest.TestCase):
    def test_container_dtype(self) -> None:
        self.assertEqual(checkpoint.container_dtype("fp32"), "float32")
        self.assertEqual(checkpoint.container_dtype("float16"), "float16")
        self.assertEqual(checkpoint.container_dtype("half"), "float16")
        self.assertEqual(checkpoint.container_dtype("torch.float32"),
                         "float32")
        with self.assertRaises(checkpoint.UnsupportedDtype):
            checkpoint.container_dtype("bfloat16")

    def test_dtype_codes(self) -> None:
        self.assertEqual(checkpoint.dtype_code("float32"), 0)
        self.assertEqual(checkpoint.dtype_code("fp16"), 1)
        self.assertEqual(checkpoint.dtype_name(1), "float16")

    def test_plan_conversion(self) -> None:
        upcast = checkpoint.plan_conversion("bfloat16", "float32")
        self.assertEqual(upcast.code, 0)
        self.assertTrue(upcast.upcast)
        self.assertFalse(upcast.lossy)

        downcast = checkpoint.plan_conversion("float32", "float16")
        self.assertEqual(downcast.code, 1)
        self.assertTrue(downcast.downcast)
        self.assertTrue(downcast.lossy)

        exact = checkpoint.plan_conversion("float16", "float16")
        self.assertFalse(exact.upcast)
        self.assertFalse(exact.downcast)
        self.assertFalse(exact.lossy)

        self.assertTrue(checkpoint.plan_conversion("bfloat16",
                                                   "float16").lossy)


class ContainerTest(unittest.TestCase):
    def records(self):
        return [
            checkpoint.TensorRecord("lm_head.weight", checkpoint.DTYPE_FP32,
                                    (2, 3), bytes(2 * 3 * 4)),
            checkpoint.TensorRecord("resid_lambdas", checkpoint.DTYPE_FP16,
                                    (2,), bytes(2 * 2)),
        ]

    def test_magic_and_round_trip(self) -> None:
        records = self.records()
        self.assertEqual(checkpoint.serialize(records)[:8], checkpoint.MAGIC)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.nchkpt01"
            checkpoint.write_checkpoint(path, records)
            loaded = checkpoint.read_checkpoint(path)
            self.assertEqual(loaded, records)
            self.assertEqual(loaded[0].dtype_name, "float32")
            self.assertEqual(loaded[0].numel(), 6)

    def test_short_payload_is_rejected(self) -> None:
        bad = [checkpoint.TensorRecord("lm_head.weight", checkpoint.DTYPE_FP32,
                                       (2, 3), bytes(3))]
        with self.assertRaises(checkpoint.CheckpointError):
            checkpoint.serialize(bad)


class PathResolutionTest(unittest.TestCase):
    def test_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            d8 = base / "base_checkpoints" / "d8"
            d20 = base / "base_checkpoints" / "d20"
            chat = base / "chatsft_checkpoints" / "chat"
            for directory in (d8, d20, chat):
                directory.mkdir(parents=True)
            (base / "base_checkpoints" / "scratch").mkdir()
            (d8 / "model_000002.pt").write_bytes(b"")
            (d8 / "model_000010.pt").write_bytes(b"")
            (d8 / "meta_000010.json").write_text("{}")
            (d20 / "model_000050.pt").write_bytes(b"")
            (chat / "model_000499.pt").write_bytes(b"")

            resolved = checkpoint.resolve("base", base_dir=base)
            self.assertEqual(resolved.model_tag, "d20")
            self.assertEqual(resolved.step, 50)
            self.assertEqual(resolved.model_path.name, "model_000050.pt")

            resolved = checkpoint.resolve("base", "d8", base_dir=base)
            self.assertEqual(resolved.step, 10)
            self.assertEqual(resolved.meta_path.name, "meta_000010.json")

            resolved = checkpoint.resolve("base", "d8", 2, base)
            self.assertEqual(resolved.model_path.name, "model_000002.pt")

            resolved = checkpoint.resolve("sft", "chat", 499, base)
            self.assertEqual(resolved.model_path,
                             base / "chatsft_checkpoints" / "chat"
                             / "model_000499.pt")

            with self.assertRaises(checkpoint.CheckpointError):
                checkpoint.resolve("bogus", base_dir=base)


if __name__ == "__main__":
    unittest.main()
