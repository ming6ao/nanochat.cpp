"""T0: the pure helpers of the notebook trial runner.

The test covers the parts that need no shared library: the name check, the
rate filter, the fingerprint, the config diff, the metric choice, the atomic
JSON write, and the results reader. See ``docs/notebook-workflow.md``.
"""

from __future__ import annotations

import io
import tempfile
import types
import unittest
from pathlib import Path

import lab


class SlugifyTest(unittest.TestCase):
    def test_keeps_safe_characters(self) -> None:
        self.assertEqual(lab.slugify("base-lr0.01_v2"), "base-lr0.01_v2")

    def test_replaces_a_separator(self) -> None:
        self.assertEqual(lab.slugify("a/b c"), "a-b-c")

    def test_rejects_an_escape(self) -> None:
        for name in ("", "   ", ".", "..", "///"):
            with self.assertRaises(ValueError):
                lab.slugify(name)

    def test_no_slash_survives(self) -> None:
        self.assertNotIn("/", lab.slugify("../../etc/passwd"))


class RunDirTest(unittest.TestCase):
    def test_slugifies_the_name(self) -> None:
        self.assertEqual(lab.run_dir_for("/runs", lab.Trial("a b")),
                         Path("/runs/a-b"))


class RateOverridesTest(unittest.TestCase):
    def test_only_set_fields(self) -> None:
        trial = lab.Trial("x", matrix_lr=0.01, warmdown_ratio=0.5)
        self.assertEqual(lab.rate_overrides(trial),
                         {"matrix_lr": 0.01, "warmdown_ratio": 0.5})

    def test_defaults_are_empty(self) -> None:
        self.assertEqual(lab.rate_overrides(lab.Trial("x")), {})


class RateFieldsTest(unittest.TestCase):
    def test_matches_the_expected_rate_names(self) -> None:
        self.assertEqual(set(lab.RATE_FIELDS), {
            "embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr",
            "weight_decay", "warmup_steps", "warmdown_ratio",
            "final_lr_frac"})


class FingerprintTest(unittest.TestCase):
    def test_is_stable(self) -> None:
        trial = lab.Trial("x", matrix_lr=0.01)
        rates = {"matrix_lr": 0.02, "warmup_steps": 40}
        self.assertEqual(lab.fingerprint(trial, rates, "abc"),
                         lab.fingerprint(trial, rates, "abc"))

    def test_changes_on_a_rate(self) -> None:
        trial = lab.Trial("x")
        self.assertNotEqual(lab.fingerprint(trial, {"matrix_lr": 0.02}, "k"),
                            lab.fingerprint(trial, {"matrix_lr": 0.01}, "k"))

    def test_changes_on_the_build(self) -> None:
        trial = lab.Trial("x")
        self.assertNotEqual(lab.fingerprint(trial, {"matrix_lr": 0.02}, "a"),
                            lab.fingerprint(trial, {"matrix_lr": 0.02}, "b"))

    def test_changes_on_a_trial_field(self) -> None:
        self.assertNotEqual(lab.fingerprint(lab.Trial("x", depth=8), {}, "k"),
                            lab.fingerprint(lab.Trial("x", depth=12), {}, "k"))

    def test_changes_on_the_context(self) -> None:
        trial = lab.Trial("x")
        self.assertNotEqual(
            lab.fingerprint(trial, {}, "k", {"val": "a.parquet"}),
            lab.fingerprint(trial, {}, "k", {"val": "b.parquet"}))


class ChangedFieldsTest(unittest.TestCase):
    def test_reports_a_nested_change(self) -> None:
        stored = {"trial": {"depth": 8, "matrix_lr": None},
                  "rates": {"matrix_lr": 0.02}}
        current = {"trial": {"depth": 12, "matrix_lr": None},
                   "rates": {"matrix_lr": 0.02}}
        self.assertEqual(lab.changed_fields(stored, current),
                         ["trial.depth: 8 -> 12"])

    def test_reports_a_missing_key(self) -> None:
        self.assertEqual(lab.changed_fields({}, {"trial": {"depth": 8}}),
                         ["trial.depth: '<missing>' -> 8"])

    def test_reports_no_change(self) -> None:
        config = {"trial": {"depth": 8}, "rates": {"matrix_lr": 0.02}}
        self.assertEqual(lab.changed_fields(config, config), [])


class MetricValueTest(unittest.TestCase):
    def test_prefers_the_metric(self) -> None:
        self.assertEqual(lab.metric_value({"bpb": 1.5, "final_loss": 3.0}), 1.5)

    def test_falls_back_to_loss(self) -> None:
        self.assertEqual(lab.metric_value({"final_loss": 3.0}), 3.0)

    def test_returns_none(self) -> None:
        self.assertIsNone(lab.metric_value({}))


class IsBetterTest(unittest.TestCase):
    def test_first_value_wins(self) -> None:
        self.assertTrue(lab.is_better(1.0, None))

    def test_lower_value_wins(self) -> None:
        self.assertTrue(lab.is_better(1.0, 2.0))

    def test_higher_value_loses(self) -> None:
        self.assertFalse(lab.is_better(2.0, 1.0))

    def test_tie_keeps_best(self) -> None:
        self.assertFalse(lab.is_better(1.0, 1.0))

    def test_none_never_wins(self) -> None:
        self.assertFalse(lab.is_better(None, 1.0))
        self.assertFalse(lab.is_better(None, None))


class ProgressReporterTest(unittest.TestCase):
    def _row(self, step):
        return {"step": step, "loss": 1.0, "tokens_per_sec": 1000.0,
                "lr": 0.02, "grad_norm": 1.5,
                "mfu": 0.5, "step_seconds": 1.0, "seconds": float(step)}

    def test_reports_the_first_last_and_interval_steps(self) -> None:
        stream = io.StringIO()
        reporter = lab.ProgressReporter(every=10, stream=stream)
        for step in range(1, 26):
            reporter(self._row(step), 25)
        self.assertEqual(len(stream.getvalue().splitlines()), 4)

    def test_reports_every_step_at_one(self) -> None:
        stream = io.StringIO()
        reporter = lab.ProgressReporter(every=1, stream=stream)
        for step in range(1, 4):
            reporter(self._row(step), 3)
        self.assertEqual(len(stream.getvalue().splitlines()), 3)

    def test_line_reports_mfu(self) -> None:
        stream = io.StringIO()
        reporter = lab.ProgressReporter(every=10, stream=stream)
        for step in range(1, 11):
            reporter(self._row(step), 10)
        line = stream.getvalue().splitlines()[-1]
        self.assertIn("mfu 50.00%", line)

    def test_line_reports_lr_and_grad_norm(self) -> None:
        stream = io.StringIO()
        reporter = lab.ProgressReporter(every=10, stream=stream)
        reporter(self._row(1), 10)
        line = stream.getvalue().splitlines()[0]
        self.assertIn("lr 0.02", line)
        self.assertIn("grad_norm 1.5", line)

    def test_line_averages_over_the_interval(self) -> None:
        stream = io.StringIO()
        reporter = lab.ProgressReporter(every=3, stream=stream)
        reporter(self._row(1), 3)
        reporter({"step": 2, "loss": 1.0, "tokens_per_sec": 2000.0,
                  "mfu": 0.8, "step_seconds": 1.0, "seconds": 2.0}, 3)
        reporter({"step": 3, "loss": 1.0, "tokens_per_sec": 3000.0,
                  "mfu": 0.7, "step_seconds": 1.0, "seconds": 3.0}, 3)
        line = stream.getvalue().splitlines()[-1]
        self.assertIn("2,500 tok/s", line)
        self.assertIn("mfu 75.00%", line)


class PeakFlopsTest(unittest.TestCase):
    def test_known_device(self) -> None:
        self.assertEqual(lab.peak_flops("Tesla T4"), 65e12)

    def test_case_and_spacing_do_not_matter(self) -> None:
        self.assertEqual(lab.peak_flops("NVIDIA GeForce GTX 1080 Ti"),
                         11.34e12)

    def test_unknown_device(self) -> None:
        self.assertEqual(lab.peak_flops("a made up device"), 0.0)
        self.assertEqual(lab.peak_flops(""), 0.0)


class LrMultiplierTest(unittest.TestCase):
    def _config(self, **overrides):
        fields = {"warmup_steps": 0, "num_iterations": 100,
                  "warmdown_ratio": 0.5, "final_lr_frac": 0.0}
        fields.update(overrides)
        return types.SimpleNamespace(**fields)

    def test_warmup_ramps(self) -> None:
        config = self._config(warmup_steps=10)
        self.assertAlmostEqual(lab._lr_multiplier(1, config), 0.1)
        self.assertAlmostEqual(lab._lr_multiplier(10, config), 1.0)

    def test_constant_before_warmdown(self) -> None:
        self.assertEqual(lab._lr_multiplier(1, self._config()), 1.0)

    def test_warmdown_decays(self) -> None:
        config = self._config()
        self.assertAlmostEqual(lab._lr_multiplier(100, config), 0.02)

    def test_final_lr_frac_holds_the_floor(self) -> None:
        config = self._config(final_lr_frac=0.1)
        self.assertAlmostEqual(lab._lr_multiplier(100, config), 0.118)


class ResumeDecisionTest(unittest.TestCase):
    def test_no_summary_runs(self) -> None:
        self.assertEqual(lab.resume_decision(False, "a", "a", False), "run")

    def test_force_runs(self) -> None:
        self.assertEqual(lab.resume_decision(True, "a", "a", True), "run")

    def test_match_resumes(self) -> None:
        self.assertEqual(lab.resume_decision(True, "a", "a", False),
                         "resume")

    def test_mismatch_is_stale(self) -> None:
        self.assertEqual(lab.resume_decision(True, "a", "b", False), "stale")

    def test_missing_stored_fingerprint_is_stale(self) -> None:
        self.assertEqual(lab.resume_decision(True, None, "b", False), "stale")


class TempDirTest(unittest.TestCase):
    """A test case with a temporary directory under ``self.root``."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nanochat_lab_")
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()


class JsonTest(TempDirTest):
    def test_round_trip(self) -> None:
        path = self.root / "nested" / "config.json"
        lab.write_json(path, {"b": 1, "a": [1, 2]})
        self.assertEqual(lab.read_json(path), {"a": [1, 2], "b": 1})

    def test_no_temporary_file_remains(self) -> None:
        path = self.root / "summary.json"
        lab.write_json(path, {"ok": True})
        self.assertEqual([item.name for item in self.root.iterdir()],
                         ["summary.json"])


class LoadResultsTest(TempDirTest):
    def test_reads_summaries_and_errors(self) -> None:
        lab.write_json(self.root / "a" / "summary.json", {"bpb": 1.2})
        lab.write_json(self.root / "b" / "error.json", {"error": "boom"})
        (self.root / "notes.txt").write_text("ignore me")
        rows = lab.load_results(self.root)
        self.assertEqual([row["name"] for row in rows], ["a", "b"])
        self.assertEqual(rows[0]["bpb"], 1.2)
        self.assertEqual(rows[1]["error"], "boom")

    def test_error_hides_an_old_summary(self) -> None:
        lab.write_json(self.root / "a" / "summary.json", {"bpb": 1.2})
        lab.write_json(self.root / "a" / "error.json", {"error": "boom"})
        rows = lab.load_results(self.root)
        self.assertEqual(rows[0]["error"], "boom")
        self.assertNotIn("bpb", rows[0])

    def test_missing_root(self) -> None:
        self.assertEqual(lab.load_results(self.root / "absent"), [])
