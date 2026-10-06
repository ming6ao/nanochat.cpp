"""T0: the evaluation fixture wire format.

This replaces the removed self-test check and is hermetic: standard library
only. See docs/python.md section 11.1.
"""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from nanochat_cpp import eval_fixture


def sample_cases():
    return [
        eval_fixture.EvalCase(tokens=(1, 2, 3, 4), start=2, end=4,
                              focus_position=3, focus_ids=(7, 8)),
        eval_fixture.EvalCase(tokens=(5, 6, 7), start=1, end=3),
        eval_fixture.EvalCase(tokens=(8,), start=1, end=1),
    ]


def sample_results():
    return [
        eval_fixture.EvalResult(nll=(0.5, 1.5, 2.5, 0.0),
                                argmax=(2, 3, 4, -1),
                                focus_logits=(0.25, -0.25)),
        eval_fixture.EvalResult(nll=(0.125, 0.25, 0.0, 0.0),
                                argmax=(6, 7, -1, -1)),
        eval_fixture.EvalResult(nll=(0.0, 0.0, 0.0, 0.0),
                                argmax=(-1, -1, -1, -1)),
    ]


class EvalFixtureTest(unittest.TestCase):
    def test_container_header_and_round_trip(self) -> None:
        cases = sample_cases()
        results = sample_results()
        raw = eval_fixture.serialize(
            eval_fixture.fixture_to_records(cases, results))
        self.assertEqual(raw[:8], eval_fixture.MAGIC)
        self.assertEqual(raw[8:16],
                         struct.pack("<II", eval_fixture.VERSION, 14))
        parsed = eval_fixture.records_to_fixture(eval_fixture.parse(raw))
        self.assertEqual(parsed.batch, 3)
        self.assertEqual(parsed.seq, 4)
        self.assertEqual(parsed.cases, cases)
        self.assertEqual(parsed.results, results)

    def test_file_round_trip(self) -> None:
        cases = sample_cases()
        results = sample_results()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.bin"
            eval_fixture.write_fixture(path, cases, results, pad_id=99)
            loaded = eval_fixture.read_fixture(path)
            self.assertEqual(loaded.pad_id, 99)
            self.assertEqual(loaded.cases, cases)
            self.assertEqual(loaded.results, results)
            self.assertEqual(loaded.cases[1].tokens, (5, 6, 7))
            self.assertEqual(loaded.cases[2].length, 1)

            case_path = Path(tmp) / "cases.bin"
            eval_fixture.write_cases(case_path, cases)
            self.assertEqual(eval_fixture.read_cases(case_path), cases)

            result_path = Path(tmp) / "results.bin"
            eval_fixture.write_results(result_path, results)
            self.assertEqual(eval_fixture.read_results(result_path), results)
            with self.assertRaises(eval_fixture.EvalFixtureError):
                eval_fixture.read_cases(result_path)

    def test_padded_width(self) -> None:
        padded_cases = [eval_fixture.EvalCase(tokens=(1, 2, 3), start=1,
                                              end=3)]
        padded_results = [eval_fixture.EvalResult(nll=(0.0,) * 5,
                                                  argmax=(-1,) * 5)]
        padded = eval_fixture.records_to_fixture(eval_fixture.parse(
            eval_fixture.serialize(
                eval_fixture.fixture_to_records(padded_cases,
                                                padded_results))))
        self.assertEqual(padded.seq, 5)
        self.assertEqual(padded.cases[0].tokens, (1, 2, 3))

    def test_invalid_cases_are_rejected(self) -> None:
        with self.assertRaises(eval_fixture.EvalFixtureError):
            eval_fixture.EvalCase(tokens=(1, 2), start=0, end=1)
        with self.assertRaises(eval_fixture.EvalFixtureError):
            eval_fixture.EvalCase(tokens=(1, 2), start=1, end=2,
                                  focus_position=1)

    def test_mismatched_focus_counts_are_rejected(self) -> None:
        results = sample_results()
        bad = [eval_fixture.EvalResult(nll=(0.5, 1.5, 2.5, 0.0),
                                       argmax=(2, 3, 4, -1),
                                       focus_logits=(0.25,)),
               results[1], results[2]]
        with self.assertRaises(eval_fixture.EvalFixtureError):
            eval_fixture.fixture_to_records(sample_cases(), bad)

    def test_corruption_is_rejected(self) -> None:
        raw = eval_fixture.serialize(
            eval_fixture.fixture_to_records(sample_cases(), sample_results()))
        with self.assertRaises(eval_fixture.EvalFixtureError):
            eval_fixture.parse(b"NOPE" + raw[4:])
        with self.assertRaises(eval_fixture.EvalFixtureError):
            eval_fixture.parse(raw + b"\x00")
        duplicate = eval_fixture.serialize([
            eval_fixture.Record("dup", eval_fixture.DTYPE_INT32, (1,),
                                struct.pack("<i", 1)),
            eval_fixture.Record("dup", eval_fixture.DTYPE_INT32, (1,),
                                struct.pack("<i", 2)),
        ])
        with self.assertRaises(eval_fixture.EvalFixtureError):
            eval_fixture.parse(duplicate)
        with self.assertRaises(eval_fixture.EvalFixtureError):
            eval_fixture.serialize([
                eval_fixture.Record("short", eval_fixture.DTYPE_FP32, (3,),
                                    bytes(8))])


if __name__ == "__main__":
    unittest.main()
