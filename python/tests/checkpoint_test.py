"""T0: the reference-checkpoint name, dtype, container, path, and metadata.

The container half of the conversion is `nanochat_cpp.checkpoint`: the magic,
the record framing, the dtype table, the name remap, the metadata block, and
the path resolution, all standard library plus the loaded shared library. The
torch half -- reading a reference `state_dict`, casting its tensors, and
reassembling the AdamW and Muon optimizer state -- is
`nanochat_cpp_bridge.torch_checkpoint`, the Z2 package of
`docs/dependency-layering-plan.md` section 4. The conversion tests below drive
that builder with fake tensors, so this file still needs no torch and no
network; the container tests feed the records it returns into the core's write
path, which is the boundary the two halves meet at.

This replaces the removed self-test checks and is hermetic: the standard
library plus the loaded shared library, no third-party import and no network.
See docs/python.md section 11.1 and
`docs/checkpoint-metadata-and-examples-plan.md` stage A.5.
"""

from __future__ import annotations

import ctypes
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nanochat_cpp import _build, _core, _lib, checkpoint
from nanochat_cpp.api import Config, Model
from nanochat_cpp_bridge import torch_checkpoint
from nanochat_cpp_bridge.torch_checkpoint import (
    build_optimizer_records,
    build_records,
    optimizer_order_key,
)

# The storage dtype codes come from the C surface, so the test pins no layout
# constant of its own.
FP32 = checkpoint.dtype_code("float32")
BF16 = checkpoint.dtype_code("bfloat16")
INT8 = checkpoint.dtype_code("int8")

#: The committed version 0 container (`//tests:checkpoint_v0_fixture`). It was
#: written before the metadata section existed, so it carries no block and its
#: run facts read as the defaults of one (stage A.2).
FIXTURE = "tests/data/checkpoint_v0.ckpt"


def resolve(relative: str) -> Path:
    """A runfiles path when the test runs under Bazel, else a source path."""
    found = _lib.find_in_runfiles(relative)
    if found is not None:
        return found
    return _build.repo_root() / relative


def tiny_config() -> Config:
    """A valid model config with no member at its default value.

    Every member the metadata block records differs from the default of
    `nanochat::Config`, so the config in the block has to come from the model
    and not from an accident of the writer's defaults (`Config::Validate`
    accepts the shape: the hidden width is divisible by the head count, the
    per-head width is even, and the vocabulary is its own padded size).
    """
    return Config(num_layers=2, num_heads=2, num_kv_heads=1, hidden_dim=32,
                  seq_len=64, vocab_size=256, padded_vocab_size=256,
                  rope_base=10000.0, window_pattern="SL",
                  value_embedding=False)


def config_object(config) -> dict:
    """The run config as the metadata JSON renders it.

    One key per `nanochat::Config` member, in the field order `VisitConfig`
    writes (`src/checkpoint_metadata.cc`), so this is the independent copy the
    test compares the decoded metadata against. Almost every name is also the
    member name the Python mirror uses, which is why the mapping is spelled
    out here rather than derived.
    """
    return {
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "num_kv_heads": config.num_kv_heads,
        "hidden_dim": config.hidden_dim,
        "seq_len": config.seq_len,
        "vocab_size": config.vocab_size,
        "padded_vocab_size": config.padded_vocab_size,
        "rope_base": float(config.rope_base),
        "window_pattern": config.window_pattern,
        "value_embedding": bool(config.value_embedding),
    }


def with_float_rope_base(config: dict) -> dict:
    """``config`` with ``rope_base`` as a float.

    The JSON object holds the shortest number that reloads to the same value,
    so an integral base arrives as a Python ``int``; the comparison is on the
    value, never on the JSON type.
    """
    normalised = dict(config)
    normalised["rope_base"] = float(normalised["rope_base"])
    return normalised


def _text(value) -> str:
    """The text of a C string that may be null."""
    return value.decode("utf-8") if value else ""


def flat_metadata(path) -> dict:
    """The run facts through the flat C accessors, not through the mapping.

    `read_metadata` fills its mapping from the metadata struct, so the test
    reads the same block a second time through the getters that
    `include/nanochat/capi.h` declares and compares the two halves of the C
    surface. Every string belongs to the handle, so the copy is made inside
    the ``try``.
    """
    library = _lib.load()
    owner = _core.CheckpointHandle()
    raw = _core.CheckpointMetadata()
    status = library.nanochat_checkpoint_metadata_read(
        str(path).encode("utf-8"), ctypes.byref(owner), ctypes.byref(raw))
    _core.check_status(library, status, "nanochat_checkpoint_metadata_read")
    metadata = ctypes.byref(raw)
    try:
        return {
            "schema": int(raw.schema),
            "stage": int(library.nanochat_checkpoint_metadata_stage(metadata)),
            "step": int(library.nanochat_checkpoint_metadata_step(metadata)),
            "world_size": int(
                library.nanochat_checkpoint_metadata_world_size(metadata)),
            "precision": int(raw.precision),
            "config": json.loads(
                library.nanochat_checkpoint_metadata_config_json(metadata)),
            "tokenizer_id": _text(raw.tokenizer_id),
            "git_revision": _text(raw.git_revision),
            "created_utc": _text(raw.created_utc),
            "notes": [] if not raw.notes else [
                _text(raw.notes[index])
                for index in range(int(raw.note_count))],
        }
    finally:
        library.nanochat_checkpoint_free(owner)


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
        self.assertEqual(checkpoint.container_dtype("bfloat16"), "bfloat16")
        self.assertEqual(checkpoint.container_dtype("bf16"), "bfloat16")
        self.assertEqual(checkpoint.container_dtype("torch.float32"),
                         "float32")
        with self.assertRaises(checkpoint.UnsupportedDtype):
            checkpoint.container_dtype("float16")

    def test_dtype_codes(self) -> None:
        self.assertEqual(checkpoint.dtype_code("float32"), 0)
        self.assertEqual(checkpoint.dtype_code("bf16"), 1)
        self.assertEqual(checkpoint.dtype_name(1), "bfloat16")
        self.assertEqual(checkpoint.dtype_name(2), "int8")
        self.assertEqual(checkpoint.dtype_size(0), 4)
        self.assertEqual(checkpoint.dtype_size(1), 2)
        self.assertEqual(checkpoint.dtype_size(2), 1)
        with self.assertRaises(checkpoint.UnsupportedDtype):
            checkpoint.dtype_name(3)

    def test_plan_conversion(self) -> None:
        # An fp16 source upcasts into the fp32 container.
        upcast = checkpoint.plan_conversion("float16", "float32")
        self.assertEqual(upcast.code, 0)
        self.assertTrue(upcast.upcast)
        self.assertFalse(upcast.lossy)

        downcast = checkpoint.plan_conversion("float32", "bfloat16")
        self.assertEqual(downcast.code, 1)
        self.assertTrue(downcast.downcast)
        self.assertTrue(downcast.lossy)

        exact = checkpoint.plan_conversion("bfloat16", "bfloat16")
        self.assertFalse(exact.upcast)
        self.assertFalse(exact.downcast)
        self.assertFalse(exact.lossy)

        self.assertTrue(checkpoint.plan_conversion("float16",
                                                   "bfloat16").lossy)


class ContainerTest(unittest.TestCase):
    def records(self):
        return [
            checkpoint.TensorRecord("lm_head.weight", FP32,
                                    (2, 3), bytes(2 * 3 * 4)),
            checkpoint.TensorRecord("resid_lambdas", BF16,
                                    (2,), bytes(2 * 2)),
        ]

    def test_magic_and_round_trip(self) -> None:
        records = self.records()
        self.assertEqual(checkpoint.serialize(records)[:8], b"NCHKPT01")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.nchkpt01"
            checkpoint.write_checkpoint(path, records)
            loaded = checkpoint.read_checkpoint(path)
            self.assertEqual(loaded, records)
            self.assertEqual(loaded[0].dtype_name, "float32")
            self.assertEqual(loaded[1].dtype_name, "bfloat16")
            self.assertEqual(loaded[0].numel(), 6)

    def test_short_payload_is_rejected(self) -> None:
        bad = [checkpoint.TensorRecord("lm_head.weight", FP32,
                                       (2, 3), bytes(3))]
        with self.assertRaises(checkpoint.CheckpointError):
            checkpoint.serialize(bad)

    def test_attribute_block_round_trip(self) -> None:
        scale = checkpoint.TensorRecord("quant.scale", FP32,
                                        (1,), bytes(4))
        record = checkpoint.TensorRecord(
            "lm_head.weight", INT8, (2, 3), bytes(6), (scale,))
        payload = checkpoint.serialize([record])
        loaded, _ = checkpoint.parse_checkpoint(payload)
        self.assertEqual(loaded, [record])
        self.assertEqual(loaded[0].attributes[0].name, "quant.scale")
        self.assertEqual(loaded[0].element_size, 1)


class MetadataCase(unittest.TestCase):
    """The checks every `read_metadata` mapping shares."""

    def assert_metadata_keys(self, metadata: dict) -> None:
        """Every key of the mapping, in the documented order."""
        self.assertEqual(list(metadata), list(checkpoint.METADATA_KEYS))

    def assert_matches_the_c_api(self, path, metadata: dict) -> None:
        """Every field equals the value the C surface returns for it."""
        raw = flat_metadata(path)
        self.assertEqual(raw["schema"], metadata["schema"])
        self.assertEqual(checkpoint.CHECKPOINT_STAGES[raw["stage"]],
                         metadata["stage"])
        self.assertEqual(raw["step"], metadata["step"])
        self.assertEqual(raw["world_size"], metadata["world_size"])
        self.assertEqual(checkpoint.CHECKPOINT_PRECISIONS[raw["precision"]],
                         metadata["precision"])
        # The decoded config, not its JSON text: the mapping hands a caller a
        # dict so no second parser is needed (section 4).
        self.assertEqual(raw["config"], metadata["config"])
        for key in ("tokenizer_id", "git_revision", "created_utc", "notes"):
            self.assertEqual(raw[key], metadata[key], key)

    def assert_config_is(self, metadata: dict, config) -> None:
        """The decoded config is ``config``, key for key."""
        got = metadata["config"]
        self.assertEqual(set(got), set(config_object(config)))
        self.assertEqual(with_float_rope_base(got),
                         with_float_rope_base(config_object(config)))

    def assert_default_run_facts(self, metadata: dict) -> None:
        """The run facts a file with no metadata block reports.

        They are the member initializers of `CheckpointMetadata`, so they hold
        under every build: the recorded precision is fixed rather than the
        build's (section 8).
        """
        self.assertEqual(metadata["schema"], 1)
        self.assertEqual(metadata["stage"], "base")
        self.assertEqual(metadata["step"], 0)
        self.assertEqual(metadata["world_size"], 1)
        self.assertEqual(metadata["precision"], "fp32")
        self.assertEqual(metadata["tokenizer_id"], "")
        self.assertEqual(metadata["git_revision"], "")
        self.assertEqual(metadata["created_utc"], "")
        self.assertEqual(metadata["notes"], [])


class MetadataFixtureTest(MetadataCase):
    """The committed version 0 container reads as the defaults."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = resolve(FIXTURE)
        cls.metadata = checkpoint.read_metadata(cls.path)

    def test_the_mapping_holds_every_key(self) -> None:
        self.assert_metadata_keys(self.metadata)

    def test_the_detected_format_is_version_zero(self) -> None:
        self.assertEqual(checkpoint.detect_format(self.path),
                         checkpoint.CONTAINER_VERSION_0)
        self.assertEqual(self.metadata["format"],
                         checkpoint.CONTAINER_VERSION_0)

    def test_the_run_facts_are_the_defaults(self) -> None:
        self.assert_default_run_facts(self.metadata)

    def test_the_config_is_the_default_config(self) -> None:
        # A version 0 file has no block, so the config is the member
        # initializer of `CheckpointMetadata`. `Config()` spells the same
        # defaults, so the test pins no number of its own.
        self.assert_config_is(self.metadata, Config())

    def test_the_mapping_matches_the_c_api(self) -> None:
        self.assert_matches_the_c_api(self.path, self.metadata)


class MetadataVersionOneTest(MetadataCase):
    """A version 1 file, written at test time by the C writer."""

    def records(self):
        return [
            checkpoint.TensorRecord("lm_head.weight", FP32, (2, 3),
                                    bytes(2 * 3 * 4)),
        ]

    def test_the_record_writer_writes_the_default_block(self) -> None:
        # `nanochat_checkpoint_write` writes the current layout, so a caller
        # that sets no run facts still gets a complete block rather than a
        # version 0 file it would have to upgrade first.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.nchkpt01"
            checkpoint.write_checkpoint(path, self.records())
            metadata = checkpoint.read_metadata(path)
            self.assert_metadata_keys(metadata)
            self.assertEqual(checkpoint.detect_format(path),
                             checkpoint.CONTAINER_VERSION_1)
            self.assertEqual(metadata["format"],
                             checkpoint.CONTAINER_VERSION_1)
            self.assert_default_run_facts(metadata)
            self.assert_config_is(metadata, Config())
            self.assert_matches_the_c_api(path, metadata)

    def test_a_model_save_records_the_model_config(self) -> None:
        config = tiny_config()
        model = Model(config, seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.nchkpt01"
            model.save(path)
            metadata = checkpoint.read_metadata(path)
            self.assert_metadata_keys(metadata)
            self.assertEqual(checkpoint.detect_format(path),
                             checkpoint.CONTAINER_VERSION_1)
            self.assertEqual(metadata["format"],
                             checkpoint.CONTAINER_VERSION_1)
            # The block follows the model, not the writer's default config.
            self.assert_config_is(metadata, config)
            # `Checkpointer::SaveModel` fills the schema, the world size of a
            # single process, and the build precision. This save carries no
            # step and no stage, so both keep their defaults, and the run
            # facts only a caller knows stay empty.
            self.assertEqual(metadata["schema"], 1)
            self.assertEqual(metadata["stage"], "base")
            self.assertEqual(metadata["step"], 0)
            self.assertEqual(metadata["world_size"], 1)
            self.assertIn(metadata["precision"],
                          checkpoint.CHECKPOINT_PRECISIONS)
            # The recorded value is the build precision. The parameter element
            # size reports it for fp32 and bf16; the fp8 tier quantizes the
            # matmuls over bf16 parameters, so it is the one build where the
            # parameter type does not name the recorded precision.
            if metadata["precision"] != "fp8":
                self.assertEqual(metadata["precision"], model.dtype)
            self.assertEqual(metadata["tokenizer_id"], "")
            self.assertEqual(metadata["git_revision"], "")
            self.assertEqual(metadata["created_utc"], "")
            self.assertEqual(metadata["notes"], [])
            self.assert_matches_the_c_api(path, metadata)


class ParseCheckpointCompatTest(unittest.TestCase):
    """The compatibility promise of open decision O7, as a test."""

    def records(self):
        return [
            checkpoint.TensorRecord("lm_head.weight", FP32, (2, 3),
                                    bytes(2 * 3 * 4)),
            checkpoint.TensorRecord("resid_lambdas", BF16, (2,), bytes(2 * 2)),
        ]

    def test_parse_checkpoint_returns_the_two_element_pair(self) -> None:
        records = self.records()
        result = checkpoint.parse_checkpoint(checkpoint.serialize(records))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        # The unpack every existing caller uses keeps working: the metadata of
        # a version 1 file never becomes a third element, so the run facts
        # arrive through `read_metadata` instead.
        parameters, optimizer_state = result
        self.assertEqual(parameters, records)
        self.assertEqual(optimizer_state, [])

    def test_the_run_facts_come_from_read_metadata(self) -> None:
        records = self.records()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.nchkpt01"
            checkpoint.write_checkpoint(path, records)
            # One file holds both, and the second one is a separate call.
            self.assertEqual(
                len(checkpoint.parse_checkpoint(path.read_bytes())), 2)
            self.assertEqual(checkpoint.read_metadata(path)["format"],
                             checkpoint.CONTAINER_VERSION_1)


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


class _FakeTensor:
    """A stdlib stand-in for one reference tensor."""

    def __init__(self, shape, payload):
        self.shape = tuple(shape)
        self.payload = payload

    def is_floating_point(self) -> bool:
        return True


class _FakeTorch:
    """The two things the record builder asks of ``torch``.

    ``build_records`` imports torch to answer ``isinstance(tensor, Tensor)``
    and to cast a payload, and nothing else. A test class satisfies both, so
    the builder runs on a host with no wheel installed.
    """

    Tensor = _FakeTensor


def _payload(tensor, target="float32"):
    """The payload of a fake tensor, standing in for the cast to bytes.

    ``_tensor_payload`` takes the target dtype and ``_optimizer_payload`` does
    not, so the default lets one stand-in patch both.
    """
    return tensor.payload


def _optimizer_state():
    """A reference optimizer state dict: AdamW moments and one Muon shard."""
    state = {
        0: {
            "step": 3,
            "exp_avg": _FakeTensor((2, 2), b"A" * 16),
            "exp_avg_sq": _FakeTensor((2, 2), b"B" * 16),
        },
        7: {
            "step": 3,
            "exp_avg": _FakeTensor((1,), b"C" * 4),
            "exp_avg_sq": _FakeTensor((1,), b"D" * 4),
        },
        9: {
            "momentum_buffer": _FakeTensor((2, 3, 4), b"M" * 96),
            "second_momentum_buffer": _FakeTensor((2, 3, 1), b"V" * 24),
        },
    }
    groups = [
        {"kind": "adamw", "params": [0, 7]},
        {"kind": "muon", "params": [9, 10]},
    ]
    return {"state": state, "param_groups": groups}


class RecordBuildTest(unittest.TestCase):
    """A reference state dict becomes container records, with no torch."""

    def build(self, state_dict, **options):
        module = mock.patch.object(torch_checkpoint, "_torch",
                                   lambda: _FakeTorch)
        cast = mock.patch.object(torch_checkpoint, "_tensor_payload", _payload)
        with module, cast:
            return build_records(state_dict, **options)

    def test_wrapped_and_legacy_names_become_canonical_records(self) -> None:
        state = {
            "module._orig_mod.transformer.wte.weight": _FakeTensor(
                (2, 2), b"w" * 16),
            "transformer.h.0.attn.proj.weight": _FakeTensor((1, 1), b"p" * 4),
        }
        records = self.build(state)
        self.assertEqual([record.name for record in records], [
            "transformer.wte.weight",
            "transformer.h.0.attn.c_proj.weight",
        ])
        self.assertEqual(records[0].shape, (2, 2))
        self.assertEqual(records[0].dtype, FP32)
        self.assertEqual(records[0].data, b"w" * 16)

    def test_the_target_dtype_is_a_container_dtype(self) -> None:
        # The fake payload is the same for both targets: a fake tensor "casts"
        # to the bytes it was built with, and the builder only asks the target
        # dtype for its code.
        state = {"lm_head.weight": _FakeTensor((4,), bytes(8))}
        self.assertEqual(self.build(state)[0].dtype, FP32)
        self.assertEqual(self.build(state, target="bfloat16")[0].dtype, BF16)
        with self.assertRaises(checkpoint.UnsupportedDtype):
            self.build(state, target="float16")

    def test_an_unknown_key_is_rejected_unless_skipped(self) -> None:
        state = {
            "optimizer.momentum": _FakeTensor((1,), b"x" * 4),
            "lm_head.weight": _FakeTensor((1,), b"y" * 4),
        }
        with self.assertRaises(checkpoint.UnknownParameter):
            self.build(state)
        records = self.build(state, skip_unknown=True)
        self.assertEqual([record.name for record in records],
                         ["lm_head.weight"])

    def test_the_records_feed_the_container_write(self) -> None:
        """The two halves of the conversion meet at the core's write path."""
        parameters = self.build({
            "transformer.wte.weight": _FakeTensor((2, 2), b"w" * 16),
        })
        with mock.patch.object(torch_checkpoint, "_optimizer_payload",
                               _payload):
            optimizer = build_optimizer_records(_optimizer_state())
        self.assertEqual([record.name for record in optimizer],
                         ["adamw.lm_head.m", "adamw.lm_head.v",
                          "muon.3x4.buf1", "muon.3x4.buf2"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.nchkpt01"
            checkpoint.write_checkpoint(path, parameters, optimizer)
            self.assertEqual(checkpoint.read_checkpoint_full(path),
                             (parameters, optimizer))


class OptimizerStateTest(unittest.TestCase):
    """The reference optimizer shard becomes C++ optimizer-state records."""

    def build(self):
        with mock.patch.object(torch_checkpoint, "_optimizer_payload",
                               _payload):
            return build_optimizer_records(_optimizer_state())

    def test_adamw_groups_are_concatenated(self):
        records = {record.name: record for record in self.build()}
        self.assertEqual(
            set(records),
            {"adamw.lm_head.m", "adamw.lm_head.v",
             "muon.3x4.buf1", "muon.3x4.buf2"})
        self.assertEqual(records["adamw.lm_head.m"].data, b"A" * 16 + b"C" * 4)
        self.assertEqual(records["adamw.lm_head.v"].data, b"B" * 16 + b"D" * 4)
        self.assertEqual(records["adamw.lm_head.m"].shape, (5,))
        self.assertEqual(records["adamw.lm_head.m"].dtype, FP32)
        self.assertEqual(records["muon.3x4.buf1"].data, b"M" * 96)
        self.assertEqual(records["muon.3x4.buf2"].data, b"V" * 24)
        self.assertEqual(records["muon.3x4.buf1"].shape, (24,))

    def test_adamw_group_order_names_each_kind(self):
        state = {
            i: {"exp_avg": _FakeTensor((1,), b"x" * 4),
                "exp_avg_sq": _FakeTensor((1,), b"y" * 4)}
            for i in range(6)
        }
        groups = [{"kind": "adamw", "params": [i]} for i in range(6)]
        with mock.patch.object(torch_checkpoint, "_optimizer_payload",
                               _payload):
            records = build_optimizer_records(
                {"state": state, "param_groups": groups})
        self.assertEqual([record.name for record in records], [
            "adamw.lm_head.m", "adamw.lm_head.v",
            "adamw.embedding.m", "adamw.embedding.v",
            "adamw.value_embedding.m", "adamw.value_embedding.v",
            "adamw.resid.m", "adamw.resid.v",
            "adamw.x0.m", "adamw.x0.v",
            "adamw.smear.m", "adamw.smear.v",
        ])

    def test_group_without_saved_state_is_skipped(self):
        with mock.patch.object(torch_checkpoint, "_optimizer_payload",
                               _payload):
            records = build_optimizer_records(
                {"state": {}, "param_groups": [
                    {"kind": "adamw", "params": [0]},
                    {"kind": "muon", "params": [1]},
                ]})
        self.assertEqual(records, [])

    def test_optimizer_order_key_matches_the_cxx_save_order(self):
        names = [
            "adamw.lm_head.m", "adamw.lm_head.v",
            "adamw.embedding.m", "adamw.embedding.v",
            "muon.4x8.buf2", "muon.4x8.buf1", "muon.2x2.buf1",
        ]
        self.assertEqual(sorted(names, key=optimizer_order_key), [
            "adamw.lm_head.m", "adamw.lm_head.v",
            "adamw.embedding.m", "adamw.embedding.v",
            "muon.2x2.buf1", "muon.4x8.buf1", "muon.4x8.buf2",
        ])


class OptimizerContainerTest(unittest.TestCase):
    def records(self):
        return [
            checkpoint.TensorRecord("lm_head.weight", FP32, (2, 3), bytes(24)),
        ]

    def optimizer_records(self):
        return [
            checkpoint.TensorRecord("adamw.lm_head.m", FP32, (6,), bytes(24)),
            checkpoint.TensorRecord("muon.4x4.buf1", FP32, (16,), bytes(64)),
        ]

    def test_file_round_trip_with_optimizer_state(self):
        params = self.records()
        state = self.optimizer_records()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.nchkpt01"
            checkpoint.write_checkpoint(path, params, state)
            self.assertEqual(checkpoint.read_checkpoint(path), params)
            self.assertEqual(checkpoint.read_optimizer_state(path), state)
            self.assertEqual(checkpoint.read_checkpoint_full(path),
                             (params, state))

    def test_parameter_only_file_has_an_empty_optimizer_section(self):
        params = self.records()
        payload = checkpoint.serialize(params)
        got_params, got_state = checkpoint.parse_checkpoint(payload)
        self.assertEqual(got_params, params)
        self.assertEqual(got_state, [])

    def test_truncated_optimizer_section_is_rejected(self):
        payload = checkpoint.serialize(self.records(),
                                       self.optimizer_records())
        with self.assertRaises(checkpoint.CheckpointError):
            checkpoint.parse_checkpoint(payload[:-6])


if __name__ == "__main__":
    unittest.main()
