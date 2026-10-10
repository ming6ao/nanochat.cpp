"""T0: the pure helpers of the two-card launch.

The test covers the parts that need no shared library and no GPU: the argument
builder, the log parser, the loss join, and the failure report. See
``docs/distributed-design.md``.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import parallel

LOG_LINE = ("step 000010 | loss 3.141593 | lr 0.025 | grad_norm 1.5 | "
            "tok/s 12345.678 | mfu 42.50%")


def make_plan(**overrides) -> parallel.ParallelPlan:
    values = dict(
        train_parquet="/data/train.parquet",
        val_parquet="/data/val.parquet",
        tokenizer="/data/tokenizer.nctoken",
        out_dir="/runs/two-card",
        batch=8,
        grad_accum=2,
        num_iterations=20)
    values.update(overrides)
    return parallel.ParallelPlan(**values)


class CommandForTest(unittest.TestCase):
    def test_each_rank_gets_distinct_paths(self) -> None:
        plan = make_plan()
        zero = parallel.command_for(0, plan)
        one = parallel.command_for(1, plan)
        self.assertIn("/runs/two-card/rank0.log", zero)
        self.assertIn("/runs/two-card/rank1.log", one)
        self.assertIn("/runs/two-card/rank0.nchkpt01", zero)
        self.assertIn("/runs/two-card/rank1.nchkpt01", one)
        self.assertEqual(zero.count("--rank"), 1)
        self.assertIn(str(plan.world_size), zero)

    def test_uses_the_entry_point(self) -> None:
        argv = parallel.command_for(0, make_plan())
        self.assertEqual(argv[:4],
                         ["tools/nanochat", "train", "--",
                          "bazel-bin/src/train_main"])

    def test_preset_comes_before_overrides(self) -> None:
        plan = make_plan(preset="track3", flags=("--layers", "4"))
        argv = parallel.command_for(0, plan)
        self.assertLess(argv.index("--preset"), argv.index("--layers"))

    def test_no_preset_omits_the_flag(self) -> None:
        argv = parallel.command_for(0, make_plan(preset=None))
        self.assertNotIn("--preset", argv)

    def test_empty_val_parquet_omits_the_flag(self) -> None:
        argv = parallel.command_for(0, make_plan(val_parquet=""))
        self.assertNotIn("--val-parquet", argv)

    def test_rejects_a_bad_rank(self) -> None:
        plan = make_plan()
        with self.assertRaises(ValueError):
            parallel.command_for(-1, plan)
        with self.assertRaises(ValueError):
            parallel.command_for(2, plan)

    def test_rejects_a_bad_batch(self) -> None:
        with self.assertRaises(ValueError):
            parallel.command_for(0, make_plan(batch=0))
        with self.assertRaises(ValueError):
            parallel.command_for(0, make_plan(grad_accum=0))

    def test_rejects_a_bad_world_size(self) -> None:
        with self.assertRaises(ValueError):
            parallel.command_for(0, make_plan(world_size=0))


class ParseLogTest(unittest.TestCase):
    def test_reads_a_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rank0.log"
            path.write_text("[nanochat] training for 20 steps\n"
                            + LOG_LINE + "\n", encoding="utf-8")
            rows = parallel.parse_log(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["step"], 10)
        self.assertAlmostEqual(rows[0]["loss"], 3.141593)
        self.assertAlmostEqual(rows[0]["mfu"], 0.425)

    def test_reads_the_phase_split(self) -> None:
        line = (LOG_LINE +
                " | step_ms 21000.000 | data_ms 12.000 | fwd_ms 1500.000 | "
                "bwd_ms 3200.000 | sync_ms 1200.000 | opt_ms 1100.000 | "
                "eval_ms 0.000")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rank0.log"
            path.write_text(line + "\n", encoding="utf-8")
            rows = parallel.parse_log(path)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["step_ms"], 21000.0)
        self.assertAlmostEqual(rows[0]["forward_ms"], 1500.0)
        self.assertAlmostEqual(rows[0]["sync_ms"], 1200.0)
        self.assertAlmostEqual(rows[0]["eval_ms"], 0.0)

    def test_a_line_without_the_phase_split_still_parses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rank0.log"
            path.write_text(LOG_LINE + "\n", encoding="utf-8")
            rows = parallel.parse_log(path)
        self.assertEqual(len(rows), 1)
        self.assertNotIn("step_ms", rows[0])

    def test_reads_a_special_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rank0.log"
            path.write_text(
                "step 000001 | loss nan | lr 0.025 | grad_norm inf | "
                "tok/s 0.000 | mfu 0.00%\n", encoding="utf-8")
            rows = parallel.parse_log(path)
        self.assertEqual(len(rows), 1)
        self.assertNotEqual(rows[0]["loss"], rows[0]["loss"])  # nan
        self.assertEqual(rows[0]["grad_norm"], float("inf"))

    def test_ignores_information_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rank0.log"
            path.write_text("[nanochat] saved checkpoint at step 20\n",
                            encoding="utf-8")
            self.assertEqual(parallel.parse_log(path), [])

    def test_missing_file(self) -> None:
        self.assertEqual(parallel.parse_log("/no/such/file.log"), [])


class LossRowsTest(unittest.TestCase):
    def test_joins_the_common_steps(self) -> None:
        result = parallel.ParallelResult(
            out_dir=Path("/runs"),
            ranks=(
                parallel.RankResult(0, 0, Path("a"), Path("b"),
                                    [{"step": 1, "loss": 5.0},
                                     {"step": 2, "loss": 4.0}]),
                parallel.RankResult(1, 0, Path("c"), Path("d"),
                                    [{"step": 1, "loss": 5.0}]),
            ))
        self.assertEqual(parallel.loss_rows(result),
                         [{"step": 1, "rank0": 5.0, "rank1": 5.0}])


class AssertCompleteTest(unittest.TestCase):
    def test_passes_when_every_rank_is_clean(self) -> None:
        result = parallel.ParallelResult(
            out_dir=Path("/runs"),
            ranks=(parallel.RankResult(0, 0, Path("a"), Path("b"), []),
                   parallel.RankResult(1, 0, Path("c"), Path("d"), [])))
        parallel.assert_complete(result)

    def test_reports_the_failing_rank_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdio = Path(tmp) / "rank1.stdio.log"
            stdio.write_text("boom: no peer\n", encoding="utf-8")
            result = parallel.ParallelResult(
                out_dir=Path(tmp),
                ranks=(parallel.RankResult(0, 0, Path("a"), Path("b"), []),
                       parallel.RankResult(1, 3, Path("c"), stdio, [])))
            with self.assertRaises(parallel.ParallelError) as caught:
                parallel.assert_complete(result)
        self.assertIn("rank 1 exited 3", str(caught.exception))
        self.assertIn("boom: no peer", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
