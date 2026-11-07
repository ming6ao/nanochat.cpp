"""Tests for the attainment join (docs/performance-model.md section 10).

The join is the one place a bound and a measurement meet, so these tests pin the
three rules the record rests on:

* A row joins on the four identity keys, and two shapes join when they agree on
  every token they both state.
* The fraction is the bound over the measurement, reported unclamped, so a value
  above 1.0 stays a finding.
* A row states the device that validated it, and a bound computed for one device
  and measured on another says so.
"""

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat_cpp import attain  # noqa: E402


def bound_row(family="attention", phase="forward", precision="fp32",
              shape="B=8 T=512 H=4 KV=4 D=128 wl=-1 L=8", ms=3.0):
    return {"name": family, "family": family, "phase": phase,
            "precision": precision, "shape": shape, "ms": ms,
            "work_ratio": 1.0, "limit": "compute"}


def measured_row(family="attention", phase="forward", precision="fp32",
                 shape="B=8 T=512 H=4 KV=4 D=128 wl=-1", ms=6.0,
                 work_ratio=1.5):
    return {"name": f"{family}_fwd", "family": family, "phase": phase,
            "precision": precision, "shape": shape, "ms": ms, "gflops": 100.0,
            "work_ratio": work_ratio, "mfu": 0.5}


def roofline(rows, device="t4"):
    return {"schema": attain.ROOFLINE_SCHEMA, "device": device,
            "precision": "fp32", "rows": rows}


def bench(rows, device="t4"):
    return {"schema": attain.BENCH_SCHEMA, "device": device,
            "precision": "fp32", "rows": rows}


class ShapeTokensTest(unittest.TestCase):
    def test_shared_tokens_must_agree(self):
        self.assertTrue(attain.shape_matches(
            "B=8 T=512 H=4 KV=4 D=128 wl=-1 L=8",
            "B=8 T=512 H=4 KV=4 D=128 wl=-1"))

    def test_a_disagreeing_token_breaks_the_match(self):
        self.assertFalse(attain.shape_matches(
            "B=8 T=512 H=4 KV=4 D=128 wl=mixed L=8",
            "B=8 T=512 H=4 KV=4 D=128 wl=-1"))

    def test_no_shared_token_is_no_evidence(self):
        # Two empty shapes are not evidence that the rows describe one workload.
        self.assertFalse(attain.shape_matches("", ""))
        self.assertFalse(attain.shape_matches("hidden=512", "batch=16"))


class JoinTest(unittest.TestCase):
    def test_a_matching_row_joins(self):
        record = attain.summarize(roofline([bound_row()]),
                                  bench([measured_row()]))
        self.assertEqual(len(record["rows"]), 1)
        row = record["rows"][0]
        self.assertAlmostEqual(row["bound_ms"], 3.0)
        self.assertAlmostEqual(row["fraction"], 0.5)

    def test_the_phase_separates_two_rows_of_one_family(self):
        record = attain.summarize(
            roofline([bound_row(phase="forward", ms=3.0),
                      bound_row(phase="backward", ms=9.0)]),
            bench([measured_row(phase="backward", ms=9.0)]))
        self.assertAlmostEqual(record["rows"][0]["bound_ms"], 9.0)

    def test_a_row_without_a_family_takes_part_in_no_join(self):
        record = attain.summarize(
            roofline([bound_row()]),
            bench([{"name": "row_kernel", "shape": "hidden=512", "ms": 0.2}]))
        row = record["rows"][0]
        self.assertIsNone(row["bound_ms"])
        self.assertIsNone(row["fraction"])
        self.assertIn("no bound row", row["note"])

    def test_an_unmatched_family_is_reported_not_dropped(self):
        record = attain.summarize(roofline([bound_row()]),
                                  bench([measured_row(family="classifier")]))
        self.assertEqual(record["unbound_rows"], 1)

    def test_the_worst_family_is_the_lowest_fraction(self):
        record = attain.summarize(
            roofline([bound_row(phase="forward", ms=3.0),
                      bound_row(phase="backward", ms=9.0)]),
            bench([measured_row(phase="forward", ms=6.0),
                   measured_row(phase="backward", ms=90.0)]))
        self.assertEqual(record["worst_family"], "attention")
        self.assertAlmostEqual(record["worst_fraction"], 0.1)


class FractionRuleTest(unittest.TestCase):
    def test_a_fraction_above_one_is_a_finding(self):
        # The bound is 3 ms and the measurement is 2 ms: the declared peak is
        # too low, which section 9 says must be reported, never absorbed.
        record = attain.summarize(roofline([bound_row(ms=3.0)]),
                                  bench([measured_row(ms=2.0)]))
        row = record["rows"][0]
        self.assertAlmostEqual(row["fraction"], 1.5)
        self.assertIn("exceeds 1.0", row["note"])

    def test_a_row_with_no_time_has_no_fraction(self):
        record = attain.summarize(roofline([bound_row(ms=0.0)]),
                                  bench([measured_row(ms=0.0)]))
        self.assertIsNone(record["rows"][0]["fraction"])

    def test_the_measured_work_ratio_is_carried_beside_the_predicted_one(self):
        record = attain.summarize(roofline([bound_row()]),
                                  bench([measured_row(work_ratio=1.996)]))
        row = record["rows"][0]
        self.assertAlmostEqual(row["bound_work_ratio"], 1.0)
        self.assertAlmostEqual(row["measured_work_ratio"], 1.996)


class ValidatedOnTest(unittest.TestCase):
    def test_the_validating_device_is_the_measuring_card(self):
        record = attain.summarize(roofline([bound_row()], device="h100"),
                                  bench([measured_row()], device="t4"))
        self.assertEqual(record["validated_on"], "t4")
        self.assertEqual(record["rows"][0]["validated_on"], "t4")

    def test_a_bound_for_another_card_says_so(self):
        record = attain.summarize(roofline([bound_row()], device="h100"),
                                  bench([measured_row()], device="t4"))
        note = record["rows"][0]["note"]
        self.assertIn("h100", note)
        self.assertIn("not validated", note)


class SchemaTest(unittest.TestCase):
    def _write(self, directory, name, body):
        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(body, handle)
        return path

    def test_an_old_schema_is_refused_with_the_version_it_found(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "bench.json",
                               {"schema": "nanochat.bench.v1", "rows": []})
            with self.assertRaises(SystemExit) as caught:
                attain._load(path, attain.BENCH_SCHEMA)
            self.assertIn("nanochat.bench.v1", str(caught.exception))

    def test_the_current_schema_loads(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "bench.json", bench([]))
            self.assertEqual(attain._load(path, attain.BENCH_SCHEMA)["schema"],
                             attain.BENCH_SCHEMA)


class CommandTest(unittest.TestCase):
    def test_the_command_prints_a_markdown_row(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            bench_path = os.path.join(directory, "bench.json")
            roof_path = os.path.join(directory, "roof.json")
            with open(bench_path, "w", encoding="utf-8") as handle:
                json.dump(bench([measured_row()]), handle)
            with open(roof_path, "w", encoding="utf-8") as handle:
                json.dump(roofline([bound_row()]), handle)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = attain.main(["--bench", bench_path,
                                    "--roofline", roof_path,
                                    "--markdown"])
            self.assertEqual(code, 0)
            text = buffer.getvalue()
            self.assertIn("| Family |", text)
            self.assertIn("attention", text)
            self.assertIn("0.500", text)


if __name__ == "__main__":
    unittest.main()
