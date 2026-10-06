"""T0: ``compute_plan`` matches the committed reference plan fixture.

The fixture ``//tests:data/plan_fixture.bin`` records, for several configs, the
reference ``GPT.num_scaling_params`` and ``GPT.num_matmul_params`` counts, the
reference ``GPT.estimate_flops`` value, and the plan outputs from the
``scripts/base_train.py`` arithmetic. The test compares the C-based
``nanochat_cpp.plan`` layer against every field. See docs/python.md section 11.
"""

from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path

from nanochat_cpp import _build, _lib, plan

FIXTURE = "tests/data/plan_fixture.bin"
MAGIC = b"NANOPLAN"


def resolve(relative: str) -> Path:
    """A runfiles path when the test runs under Bazel, else a source path."""
    found = _lib.find_in_runfiles(relative)
    if found is not None:
        return found
    return _build.repo_root() / relative


def load_fixture(path: Path) -> dict:
    raw = path.read_bytes()
    assert raw[:8] == MAGIC, f"not a {MAGIC!r} fixture: {path}"
    version = struct.unpack("<I", raw[8:12])[0]
    assert version == 1, f"unsupported plan fixture version {version}"
    return json.loads(raw[12:].decode("utf-8"))


class PlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = load_fixture(resolve(FIXTURE))

    def test_every_case_matches_the_reference(self) -> None:
        self.assertTrue(self.fixture["cases"])
        for case in self.fixture["cases"]:
            with self.subTest(depth=case["inputs"]["depth"],
                              seq_len=case["inputs"]["seq_len"]):
                self.check_case(case)

    def check_case(self, case: dict) -> None:
        inputs = case["inputs"]
        result = plan.compute_plan(**inputs)
        counts = case["counts"]

        # The C parameter counts match the reference, field for field.
        got = plan.params(result.config)
        for field in ("total", "transformer_matrices", "lm_head", "embeddings",
                      "scalars"):
            self.assertEqual(getattr(got, field), counts[field],
                             f"{field} differs for depth "
                             f"{inputs['depth']}")
        self.assertAlmostEqual(got.flops_per_token, case["flops_per_token"],
                               places=6)

        # `total` is the sum of the four groups.
        self.assertEqual(got.total,
                         got.embeddings + got.transformer_matrices
                         + got.lm_head + got.scalars)

        # Pin the drift between the two reference count functions: the matmul
        # count is the scaling matrix count plus `lm_head` plus the 24-channel
        # smear gate, which `num_scaling_params` groups with the scalars.
        self.assertEqual(counts["num_matmul_params"],
                         got.transformer_matrices + got.lm_head + 24)

        # The plan arithmetic matches the reference base_train.py derivation.
        expected = case["plan"]
        self.assertEqual(result.padded_vocab_size,
                         expected["padded_vocab_size"])
        self.assertEqual(result.total_batch_size, expected["total_batch_size"])
        self.assertEqual(result.grad_accum, expected["grad_accum"])
        self.assertEqual(result.num_iterations, expected["num_iterations"])
        self.assertEqual(result.target_tokens, expected["target_tokens"])
        self.assertEqual(result.scaling_params, expected["scaling_params"])
        self.assertEqual(result.flops_per_token, got.flops_per_token)
        self.assertAlmostEqual(result.batch_lr_scale,
                               expected["batch_lr_scale"], places=12)
        for key, value in expected["optimizer"].items():
            self.assertAlmostEqual(result.optimizer[key], value, places=12)
        for key, value in expected["scheduler"].items():
            self.assertAlmostEqual(result.scheduler[key], value, places=12)

    def test_config_kv_heads_default(self) -> None:
        # The reference plan forces n_kv_head = n_head; the fixture configs
        # leave num_kv_heads unset, so the derived config must match.
        plan_result = plan.compute_plan(
            depth=8, seq_len=512, vocab_size=32768, device_batch_size=8)
        self.assertEqual(plan_result.config.num_kv_heads,
                         plan_result.config.num_heads)

    def test_unknown_rate_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            plan.compute_plan(depth=8, seq_len=512, vocab_size=32768,
                              device_batch_size=8, momentum_lr=0.1)


if __name__ == "__main__":
    unittest.main()
