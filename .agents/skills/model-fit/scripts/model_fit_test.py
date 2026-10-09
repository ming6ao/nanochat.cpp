"""Tests for the model-fit engine.

Run with: python3 -m unittest discover -s .agents/skills/model-fit/scripts
"""

from __future__ import annotations

import unittest

import model_fit


class ParameterCountsTest(unittest.TestCase):
    def test_track3_total(self) -> None:
        spec = model_fit.ModelSpec(12, 768, 6, 6, 1024)
        counts = model_fit.parameter_counts(spec)
        self.assertEqual(counts["total"], 135_266_354)
        self.assertEqual(counts["embeddings"], 25_165_824)

    def test_value_embeddings_add_parameters(self) -> None:
        plain = model_fit.ModelSpec(4, 768, 6, 6, 1024)
        ve = model_fit.ModelSpec(4, 768, 6, 6, 1024, value_embedding=True)
        extra = model_fit.parameter_counts(ve)["total"] \
            - model_fit.parameter_counts(plain)["total"]
        self.assertEqual(extra, 2 * (32768 * 768 + 6 * 12))

    def test_rejects_a_bad_head_split(self) -> None:
        with self.assertRaises(ValueError):
            model_fit.parameter_counts(model_fit.ModelSpec(2, 100, 3, 3, 128))
        with self.assertRaises(ValueError):
            model_fit.parameter_counts(model_fit.ModelSpec(2, 128, 8, 3, 128))


class ArenaTest(unittest.TestCase):
    def test_matches_the_documented_36_gb(self) -> None:
        spec = model_fit.ModelSpec(12, 768, 6, 6, 1024)
        got = model_fit.train_arena_bytes(spec, 32, value_bytes=4)
        self.assertAlmostEqual(got, 36.10e9, delta=0.4e9)

    def test_reduced_precision_shrinks_the_arena(self) -> None:
        spec = model_fit.ModelSpec(12, 1024, 8, 8, 2048)
        self.assertLess(model_fit.train_arena_bytes(spec, 4, 2),
                        model_fit.train_arena_bytes(spec, 4, 4))


class HardwareTest(unittest.TestCase):
    def test_t4_rates(self) -> None:
        hw = model_fit.hardware("t4")
        self.assertTrue(hw.supports("fp32"))
        self.assertTrue(hw.supports("fp16"))
        self.assertFalse(hw.supports("bf16"))
        self.assertAlmostEqual(hw.budget_bytes() / model_fit.GIB, 14.5, places=2)

    def test_unknown_device_raises(self) -> None:
        with self.assertRaises(KeyError):
            model_fit.hardware("voodoo")

    def test_pascal_has_no_tensor_cores(self) -> None:
        hw = model_fit.hardware("gtx1080ti")
        self.assertFalse(hw.tensor_cores)
        self.assertLess(hw.fp16_flops, hw.fp32_flops)


class TopologyTest(unittest.TestCase):
    def test_sync_grows_with_payload(self) -> None:
        topo = model_fit.Topology(world_size=2, interconnect="pcie3")
        self.assertGreater(topo.sync_seconds(1e9), topo.sync_seconds(1e8))

    def test_no_sync_for_one_device(self) -> None:
        topo = model_fit.Topology(world_size=1)
        self.assertEqual(topo.sync_seconds(1e9), 0.0)

    def test_faster_link_is_cheaper(self) -> None:
        pcie = model_fit.Topology(world_size=2, interconnect="pcie3")
        nvlink = model_fit.Topology(world_size=2, interconnect="nvlink3")
        self.assertLess(nvlink.sync_seconds(1e9), pcie.sync_seconds(1e9))


class FitTest(unittest.TestCase):
    def test_best_batch_fits_the_budget(self) -> None:
        hw = model_fit.hardware("t4")
        spec = model_fit.ModelSpec(12, 768, 6, 6, 1024)
        batch = model_fit.best_batch(spec, hw, "fp32")
        self.assertGreaterEqual(batch, 1)
        self.assertLessEqual(model_fit.card_bytes(spec, batch, "fp32"),
                             hw.budget_bytes())
        self.assertGreater(model_fit.card_bytes(spec, batch + 1, "fp32"),
                           hw.budget_bytes())

    def test_fp16_fits_more_than_fp32(self) -> None:
        hw = model_fit.hardware("t4")
        spec = model_fit.ModelSpec(16, 1024, 8, 8, 2048)
        self.assertGreaterEqual(model_fit.best_batch(spec, hw, "fp16"),
                                model_fit.best_batch(spec, hw, "fp32"))

    def test_breakdown_sums_to_the_total(self) -> None:
        spec = model_fit.ModelSpec(16, 1024, 8, 4, 4096)
        parts = model_fit.memory_breakdown(spec, 1, "fp32")
        self.assertEqual(sum(parts.values()),
                         model_fit.card_bytes(spec, 1, "fp32"))


class ScalingTest(unittest.TestCase):
    def test_the_reduction_costs_throughput(self) -> None:
        hw = model_fit.hardware("t4")
        spec = model_fit.ModelSpec(16, 1024, 8, 4, 4096)
        one = model_fit.Topology(world_size=1)
        two = model_fit.KAGGLE_T4X2
        self.assertLess(model_fit.scaling_efficiency(spec, 1, hw, two, "fp32"),
                        model_fit.scaling_efficiency(spec, 1, hw, one, "fp32"))

    def test_overlap_recovers_efficiency(self) -> None:
        hw = model_fit.hardware("t4")
        spec = model_fit.ModelSpec(12, 2560, 20, 5, 1024)
        plain = model_fit.KAGGLE_T4X2
        overlapped = model_fit.Topology(world_size=2, interconnect="pcie3",
                                        overlap=1.0)
        self.assertLess(model_fit.scaling_efficiency(spec, 1, hw, plain, "fp32"),
                        model_fit.scaling_efficiency(spec, 1, hw, overlapped,
                                                     "fp32"))

    def test_a_longer_sequence_lowers_the_sync_fraction(self) -> None:
        hw = model_fit.hardware("t4")
        short = model_fit.ModelSpec(16, 1024, 8, 4, 1024)
        long = model_fit.ModelSpec(16, 1024, 8, 4, 8192)
        topo = model_fit.KAGGLE_T4X2
        short_row = model_fit._candidate(short, 1, hw, topo, "fp32")
        long_row = model_fit._candidate(long, 1, hw, topo, "fp32")
        self.assertLess(long_row.sync_fraction, short_row.sync_fraction)


class AnswerTest(unittest.TestCase):
    def test_t4_corners(self) -> None:
        result = model_fit.answer(model_fit.hardware("t4"),
                                  model_fit.KAGGLE_T4X2, "fp32")
        corners = result["corners"]
        self.assertTrue(corners["biggest"]["spec"]["layers"] >= 12)
        self.assertEqual(corners["longest"]["spec"]["seq"], 8192)
        self.assertEqual(corners["fastest"]["label"],
                         "L12 C768 H6 KV1 T1024 b13")
        self.assertEqual(result["balanced"]["label"],
                         "L16 C1024 H8 KV4 T4096 b1")

    def test_custom_hardware_rejects_an_unknown_precision(self) -> None:
        hw = model_fit.hardware("gtx1080ti")
        result = model_fit.answer(hw, model_fit.Topology(world_size=1), "bf16")
        self.assertEqual(result["corners"], {})

    def test_recommend_returns_a_feasible_point(self) -> None:
        hw = model_fit.hardware("t4")
        row = model_fit.recommend(hw, model_fit.KAGGLE_T4X2, "fp32")
        self.assertLessEqual(row.bytes, hw.budget_bytes())
        self.assertGreater(row.tokens_per_second, 0.0)

    def test_frontier_is_speed_monotone(self) -> None:
        rows = model_fit.search(model_fit.hardware("t4"),
                                model_fit.Topology(world_size=1), "fp32",
                                presets="quick")
        frontier = model_fit.frontier(rows)
        # The frontier orders by descending size, so the speed rises.
        params = [row.params for row in frontier]
        speeds = [row.tokens_per_second for row in frontier]
        self.assertEqual(params, sorted(params, reverse=True))
        self.assertEqual(speeds, sorted(speeds))

    def test_shape_flags(self) -> None:
        flags = model_fit.shape_flags(model_fit.BEST_FIT_T4)
        self.assertEqual(flags[flags.index("--hidden") + 1], "1024")
        self.assertEqual(flags[flags.index("--seq") + 1], "4096")
        self.assertNotIn("--batch", flags)


if __name__ == "__main__":
    unittest.main()
