"""T0: the pure helpers of the notebook argument data classes.

The test covers the parts that need no shared library: the shape validation,
the JSON round trip, the rate filter, and the optimizer keyword filter.
"""

from __future__ import annotations

import dataclasses
import unittest

import args


class ModelArgsTest(unittest.TestCase):
    def test_defaults_are_valid(self) -> None:
        model = args.ModelArgs()
        self.assertEqual(model.hidden_dim % model.num_heads, 0)
        self.assertEqual(model.num_heads % model.num_kv_heads, 0)

    def test_head_dim(self) -> None:
        self.assertEqual(
            args.ModelArgs(num_layers=4, hidden_dim=256, num_heads=4,
                           num_kv_heads=2).head_dim, 64)

    def test_rejects_a_non_positive_size(self) -> None:
        for fields in ({"num_layers": 0}, {"seq_len": 0}, {"vocab_size": 0}):
            with self.assertRaises(ValueError):
                args.ModelArgs(**fields)

    def test_rejects_a_narrow_hidden(self) -> None:
        with self.assertRaises(ValueError):
            args.ModelArgs(hidden_dim=8, num_heads=1)

    def test_rejects_a_bad_head_count(self) -> None:
        with self.assertRaises(ValueError):
            args.ModelArgs(hidden_dim=256, num_heads=6)
        with self.assertRaises(ValueError):
            args.ModelArgs(num_heads=4, num_kv_heads=3)

    def test_is_frozen(self) -> None:
        model = args.ModelArgs()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            model.num_layers = 2

    def test_json_round_trip(self) -> None:
        model = args.ModelArgs(num_layers=4, hidden_dim=256, num_heads=4,
                               num_kv_heads=2, seq_len=512,
                               value_embedding=True)
        self.assertEqual(args.ModelArgs.from_json(model.to_json()), model)

    def test_replace_makes_a_variant(self) -> None:
        model = args.ModelArgs()
        self.assertEqual(dataclasses.replace(model, num_layers=16).num_layers,
                         16)


class TrainArgsTest(unittest.TestCase):
    def test_defaults_are_valid(self) -> None:
        self.assertEqual(args.TrainArgs().rate_overrides(), {})

    def test_rejects_a_non_positive_batch(self) -> None:
        with self.assertRaises(ValueError):
            args.TrainArgs(device_batch_size=0)
        with self.assertRaises(ValueError):
            args.TrainArgs(total_batch_size=0)
        with self.assertRaises(ValueError):
            args.TrainArgs(num_iterations=0)

    def test_rate_overrides_keeps_set_fields_only(self) -> None:
        train = args.TrainArgs(matrix_lr=0.01, warmdown_ratio=0.5)
        self.assertEqual(train.rate_overrides(),
                         {"matrix_lr": 0.01, "warmdown_ratio": 0.5})

    def test_optimizer_kwargs_drops_none_and_the_batch(self) -> None:
        train = args.TrainArgs(device_batch_size=8, total_batch_size=4096,
                               num_iterations=50, matrix_lr=0.02)
        fields = train.optimizer_kwargs()
        self.assertEqual(fields["matrix_lr"], 0.02)
        self.assertEqual(fields["num_iterations"], 50)
        for name in ("device_batch_size", "total_batch_size", "model_seed"):
            self.assertNotIn(name, fields)

    def test_json_round_trip(self) -> None:
        train = args.TrainArgs(num_iterations=50, matrix_lr=0.02)
        self.assertEqual(args.TrainArgs.from_json(train.to_json()), train)


if __name__ == "__main__":
    unittest.main()
