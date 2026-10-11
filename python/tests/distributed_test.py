"""T0: the distributed launch plan and the JSON summary parser.

Hermetic: it needs no library, no Bazel, and no GPU. It writes a fake
``tools/nanochat`` entry that emits the launcher's JSON summary, then checks the
plan, the launch result, and the log parser. See ``docs/distributed-launch.md``
section 6.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from nanochat_cpp import distributed

FAKE_ENTRY = """#!{python}
import json
import sys

argv = sys.argv[1:]
if argv and argv[0] == "dist":
    nproc = 2
    if "--nproc-per-node" in argv:
        nproc = int(argv[argv.index("--nproc-per-node") + 1])
    print(json.dumps({{"ranks": [
        {{"rank": rank, "returncode": 0,
          "stdio": f"/tmp/rank{{rank}}.stdio.log", "seconds": 1.5}}
        for rank in range(nproc)]}}))
else:
    print(json.dumps(argv))
"""


class DistributedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="nanochat_dist_")
        cls.entry = Path(cls.tmp.name) / "fake-nanochat"
        cls.entry.write_text(FAKE_ENTRY.format(python=sys.executable),
                             encoding="utf-8")
        cls.entry.chmod(cls.entry.stat().st_mode | stat.S_IEXEC)
        cls.env = dict(os.environ)
        cls.env.pop("NANOCHAT_SANDBOX", None)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def plan(self, **overrides) -> distributed.DistributedPlan:
        values = dict(
            train_parquet="train.parquet", val_parquet="val.parquet",
            tokenizer="tokenizer.nctoken", out_dir=str(self.tmp.name),
            batch=1, grad_accum=8, num_iterations=500, preset="track3",
            flags=("--layers", "16"), world_size=2, entry=str(self.entry))
        values.update(overrides)
        return distributed.DistributedPlan(**values)

    def test_command_for_carries_the_rank_paths(self) -> None:
        plan = self.plan()
        argv = distributed.command_for(1, plan)
        self.assertEqual(argv[0], str(self.entry))
        self.assertEqual(argv[1], "train")
        self.assertIn("--rank", argv)
        self.assertIn("1", argv)
        self.assertIn("--world-size", argv)
        self.assertEqual(argv[-1], "16")
        self.assertIn(str(plan.log_path(1)), argv)
        self.assertIn(str(plan.checkpoint_path(1)), argv)

    def test_command_for_rejects_a_bad_rank(self) -> None:
        plan = self.plan()
        with self.assertRaises(ValueError):
            distributed.command_for(2, plan)
        with self.assertRaises(ValueError):
            distributed.command_for(0, self.plan(world_size=0))

    def test_launch_parses_the_json_summary(self) -> None:
        plan = self.plan()
        result = distributed.launch(plan, root=self.tmp.name, timeout=10,
                                    extra_env=self.env)
        self.assertEqual(result.returncodes, (0, 0))
        self.assertTrue(result.ok)
        self.assertEqual(len(result.ranks), 2)
        self.assertEqual(result.ranks[1].rank, 1)
        self.assertEqual(result.ranks[1].stdio_path,
                         Path("/tmp/rank1.stdio.log"))
        self.assertAlmostEqual(result.ranks[0].seconds, 1.5)
        distributed.assert_complete(result)

    def test_assert_complete_reports_a_failure(self) -> None:
        result = distributed.DistributedResult(
            out_dir=Path(self.tmp.name),
            ranks=(distributed.RankResult(
                rank=0, returncode=3, log_path=Path("missing.log"),
                stdio_path=Path("missing.stdio.log")),))
        with self.assertRaises(distributed.DistributedError):
            distributed.assert_complete(result)

    def test_parse_log_and_loss_rows(self) -> None:
        path = Path(self.tmp.name) / "rank0.log"
        path.write_text(
            "[nanochat] training for 1 steps\n"
            "step 000001 | loss 10.5 | lr 0.02 | grad_norm 0.5 | "
            "tok/s 1200.0 | mfu 25.00% | step_ms 26000.0 | data_ms 1.0 | "
            "fwd_ms 6000.0 | bwd_ms 15000.0 | sync_ms 2000.0 | "
            "opt_ms 1500.0 | eval_ms 0.0\n",
            encoding="utf-8")
        rows = distributed.parse_log(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["step"], 1)
        self.assertAlmostEqual(rows[0]["mfu"], 0.25)
        self.assertAlmostEqual(rows[0]["sync_ms"], 2000.0)

    def test_parse_log_missing_file(self) -> None:
        self.assertEqual(
            distributed.parse_log(Path(self.tmp.name) / "absent.log"), [])


if __name__ == "__main__":
    unittest.main()
