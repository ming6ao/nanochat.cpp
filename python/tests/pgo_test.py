"""T0: the deterministic half of the profile-guided optimization loop.

Hermetic. The trace is a synthetic pair of CSV reports in the shape ``nsys
stats`` writes, so the test needs no GPU, no Modal account, and no 96 MB
artifact. The numbers it checks are the ones the loop decides on: the step
median, the signature ceilings, the pair ranking, the noise rule of the
comparison, and the state machine of the loop.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from nanochat_cpp import pgo

#: The shape of one step, as ``(offset_ms, duration_ns, name, bytes)``. The two
#: loss copies mark the step, because the trainer writes its per-row losses to
#: the host once per forward pass.
STEP_MS = 400
LOSS_BYTES = 16_384
BULK_BYTES = 70_369_280


def _step_layout(loss_bytes: int = LOSS_BYTES) -> list[tuple[float, int, str, int]]:
    """One synthetic step: small kernels, a bulk readback, and one scalar read."""
    layout = []
    for index in range(200):
        layout.append((0.4 * index, 500, "nc::PointwiseForwardKernel", 0))
    for index in range(3):
        layout.append((100.0 + 10 * index, 200_000,
                       "sm90_xmma_gemm_bf16bf16_bf16f32_f32_nn_n_tilesize128",
                       0))
    for index in range(2):
        layout.append((130.0 + 5 * index, 2_700,
                       "void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_"
                       "32x32_128x2_tn_align8>", 0))
    # A bulk readback of 67.1 MB, then 30 ms with nothing to run: the host is in
    # its own loop, which is the signature the first iteration chased.
    layout.append((150.0, 6_300_000, "[CUDA memcpy Device-to-Host]",
                   BULK_BYTES))
    layout.append((206.0, 6_300_000, "[CUDA memcpy Device-to-Host]",
                   BULK_BYTES))
    # A scalar read that blocks the queue for 5 ms.
    layout.append((300.0, 2_000, "[CUDA memcpy Device-to-Host]", 0))
    layout.append((350.0, 4_000, "[CUDA memcpy Device-to-Host]", loss_bytes))
    return layout


def _write_trace(folder: Path, steps: int = 6,
                 loss_bytes: int = LOSS_BYTES) -> None:
    """Write one synthetic trace directory in the ``nsys stats`` CSV shape."""
    folder.mkdir(parents=True, exist_ok=True)
    gpu = ["Start (ns),Duration (ns),GrdX,Strm,Bytes (MB),Name"]
    api = ["Start (ns),Duration (ns),Name,Result,Pid,Tid"]
    for step in range(steps):
        base = step * STEP_MS * 1_000_000
        for offset_ms, duration_ns, name, size in _step_layout(loss_bytes):
            start = base + int(offset_ms * 1_000_000)
            megabytes = f"{size / 1e6:.6f}" if size else ""
            gpu.append(f"{start},{duration_ns},1,7,{megabytes},{name}")
            if name == "[CUDA memcpy Device-to-Host]":
                # The host blocks inside the copy, which is what a synchronous
                # cudaMemcpy from pageable memory does.
                api.append(f"{start},{duration_ns},cudaMemcpy,0,49,49")
            else:
                api.append(f"{start},1500,cuLaunchKernel,0,49,49")
        # The stall after the second bulk read: no API row overlaps it.
    (folder / "base_cuda_gpu_kern_sum.csv").write_text(
        "Time (%),Total Time (ns),Instances,Avg (ns),Name\n"
        "40.0,26000000,1,26000000,[CUDA memcpy Device-to-Host]\n",
        encoding="utf-8")
    (folder / "base_cuda_api_sum.csv").write_text(
        "Time (%),Total Time (ns),Num Calls,Avg (ns),Name\n"
        "60.0,52000000,12,4333333,cudaMemcpy\n"
        "30.0,26000000,1200,21666,cuLaunchKernel\n", encoding="utf-8")
    (folder / "timeline_cuda_gpu_trace.csv").write_text("\n".join(gpu) + "\n",
                                                        encoding="utf-8")
    (folder / "timeline_cuda_api_trace.csv").write_text("\n".join(api) + "\n",
                                                        encoding="utf-8")
    (folder / "base.nsys-rep").write_bytes(b"synthetic")


class AnalyzeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="pgo-trace-"))
        _write_trace(self.root)

    def test_one_step_is_cut_out_of_the_trace_by_its_own_marks(self) -> None:
        analysis = pgo.analyze_trace(self.root, stage="base", device="h100",
                                     steps=6, step_index=1)
        self.assertEqual(analysis["schema"], pgo.ANALYSIS_SCHEMA)
        step = analysis["step"]
        self.assertGreater(step["operations"], 200)
        self.assertLess(step["device_busy_percent"], 60.0)
        self.assertGreater(step["host_own_code_ms"], 20.0)

    def test_the_host_loop_is_the_largest_signature(self) -> None:
        analysis = pgo.analyze_trace(self.root, stage="base", device="h100",
                                     steps=6, step_index=1)
        first = analysis["signatures"][0]
        self.assertEqual(first["id"], "host_own_code")
        self.assertGreater(first["ceiling"], 0.10)
        evidence = first["evidence"]["stalls"]
        self.assertEqual(evidence[0]["after_bytes"], BULK_BYTES)

    def test_every_signature_carries_a_ceiling_within_a_fraction(self) -> None:
        analysis = pgo.analyze_trace(self.root, stage="base", device="h100",
                                     steps=6, step_index=1)
        self.assertTrue(analysis["signatures"])
        for row in analysis["signatures"]:
            self.assertGreaterEqual(row["ceiling"], 0.0)
            self.assertLessEqual(row["ceiling"], 1.0)
            self.assertIn("evidence", row)

    def test_the_bulk_readback_and_the_scalar_read_are_seen(self) -> None:
        analysis = pgo.analyze_trace(self.root, stage="base", device="h100",
                                     steps=6, step_index=1)
        identifiers = {row["id"] for row in analysis["signatures"]}
        self.assertIn("dtoh_bytes", identifiers)
        self.assertIn("blocking_dtoh_scalars", identifiers)
        dtoh = next(row for row in analysis["signatures"]
                    if row["id"] == "dtoh_bytes")
        self.assertGreater(dtoh["evidence"]["total_mb"], 100.0)

    def test_the_rank_takes_the_two_largest_and_sums_them(self) -> None:
        analysis = pgo.analyze_trace(self.root, stage="base", device="h100",
                                     steps=6, step_index=1)
        ranked = pgo.rank(analysis["signatures"])
        self.assertEqual(len(ranked["ids"]), 2)
        self.assertEqual(ranked["ids"][0], "host_own_code")
        self.assertLessEqual(ranked["pair_ceiling"], 0.9)
        self.assertTrue(ranked["proceed"])

    def test_a_rejected_signature_leaves_the_ranking(self) -> None:
        analysis = pgo.analyze_trace(self.root, stage="base", device="h100",
                                     steps=6, step_index=1)
        ranked = pgo.rank(analysis["signatures"], ["host_own_code"])
        self.assertNotIn("host_own_code", ranked["ids"])

    def test_an_unactionable_signature_never_takes_a_slot(self) -> None:
        rows = [{"id": "idle_unexplained", "ceiling": 0.5},
                {"id": "blocking_dtoh_scalars", "ceiling": 0.2},
                {"id": "tiny_gemm_tile", "ceiling": 0.1}]
        ranked = pgo.rank(rows)
        self.assertEqual(ranked["ids"], ["blocking_dtoh_scalars",
                                        "tiny_gemm_tile"])

    def test_a_contained_signature_never_shares_a_pair(self) -> None:
        rows = [{"id": "driver_call_overhead", "ceiling": 0.3},
                {"id": "dtoh_bytes", "ceiling": 0.1},
                {"id": "tiny_gemm_tile", "ceiling": 0.08}]
        ranked = pgo.rank(rows)
        self.assertEqual(ranked["ids"], ["driver_call_overhead",
                                        "tiny_gemm_tile"])
        self.assertIn("dtoh_bytes", ranked["dropped"])

    def test_the_mark_size_is_the_shape_s_loss_copy(self) -> None:
        # A shape with four times the rows copies four times the losses, so the
        # reference profile's 16 KB window finds nothing and the caller has to
        # name the size its own stage wrote.
        folder = Path(tempfile.mkdtemp(prefix="pgo-marks-"))
        _write_trace(folder, loss_bytes=65_536)
        with self.assertRaises(pgo.PgoError):
            pgo.analyze_trace(folder, stage="base", device="h100", steps=6)
        analysis = pgo.analyze_trace(folder, stage="base", device="h100",
                                     steps=6, step_index=1,
                                     mark_bytes=65_536)
        self.assertGreater(analysis["step"]["operations"], 200)

    def test_a_trace_without_step_marks_is_a_fault(self) -> None:
        empty = Path(tempfile.mkdtemp(prefix="pgo-empty-"))
        (empty / "timeline_cuda_gpu_trace.csv").write_text(
            "Start (ns),Duration (ns),Name\n1000,2000,some_kernel\n",
            encoding="utf-8")
        with self.assertRaises(pgo.PgoError):
            pgo.analyze_trace(empty, stage="base", device="h100")

    def test_the_chrome_trace_holds_both_tracks(self) -> None:
        out = pgo.chrome_trace(self.root, self.root / "chrome.json")
        document = json.loads(out.read_text(encoding="utf-8"))
        processes = {entry["pid"] for entry in document["traceEvents"]}
        self.assertEqual(processes, {1, 2})
        self.assertIn("traceEvents", document)


class MeasurementTest(unittest.TestCase):
    def _document(self, runs, *, steps=50, tokens=12000.0, mfu=0.003):
        """One container answer, with one entry per run."""
        return {"stage": "base", "device": "h100", "steps": steps,
                "source": "installed",
                "runs": [{"steps": steps, "stage_seconds": 20.0,
                          "step_ms": times, "tokens_per_second": tokens,
                          "mfu": mfu} for times in runs]}

    def test_the_median_of_the_steady_window_is_the_value(self) -> None:
        report = pgo.measurement_report(
            self._document([[1.0, 0.4, 0.4, 0.4, 0.2]]), stage="base",
            device="h100")
        self.assertEqual(report["schema"], pgo.MEASUREMENT_SCHEMA)
        self.assertEqual(report["runs"][0]["step_ms"], 0.4)
        self.assertEqual(report["runs"][0]["source"], "metrics")

    def test_a_stage_without_per_step_times_divides_its_own_seconds(self) -> None:
        document = {"stage": "sft", "steps": 20,
                    "runs": [{"steps": 20, "stage_seconds": 8.0}]}
        report = pgo.measurement_report(document, stage="sft", device="t4")
        self.assertEqual(report["runs"][0]["step_ms"], 400.0)
        self.assertEqual(report["runs"][0]["source"], "stage_seconds")

    def test_two_runs_report_the_better_value(self) -> None:
        report = pgo.measurement_report(
            self._document([[0.5, 0.5], [0.4, 0.4]]), stage="base",
            device="h100")
        self.assertEqual(report["value"], 0.4)

    def test_eval_scores_a_throughput_not_a_step(self) -> None:
        document = {"stage": "eval", "steps": 4, "runs": [
            {"steps": 4, "stage_seconds": 2.0, "samples_per_second": 2.0}]}
        report = pgo.measurement_report(document, stage="eval", device="h100")
        self.assertEqual(report["metric"]["name"], "samples_per_second")
        self.assertEqual(report["metric"]["direction"], "higher")


class CompareTest(unittest.TestCase):
    def _report(self, values):
        return pgo.measurement_report(
            {"stage": "base", "steps": 50,
             "runs": [{"steps": 50, "step_ms": [value, value]} for value in
                      values]}, stage="base", device="h100")

    def test_a_pair_above_the_bar_with_disjoint_runs_is_kept(self) -> None:
        verdict = pgo.compare(self._report([0.50, 0.52]),
                             self._report([0.30, 0.32]))
        self.assertEqual(verdict["decision"], "kept")
        self.assertFalse(verdict["overlap"])
        self.assertGreater(verdict["gain"], 0.3)

    def test_a_gain_inside_the_noise_is_reverted(self) -> None:
        verdict = pgo.compare(self._report([0.50, 0.52]),
                             self._report([0.49, 0.51]))
        self.assertEqual(verdict["decision"], "reverted")
        self.assertTrue(verdict["overlap"])

    def test_a_gain_below_the_bar_is_reverted(self) -> None:
        verdict = pgo.compare(self._report([0.50]),
                             self._report([0.49]))
        self.assertEqual(verdict["decision"], "reverted")
        self.assertIn("below the 5% bar", verdict["reason"])

    def test_a_throughput_metric_reads_the_other_direction(self) -> None:
        before = pgo.measurement_report(
            {"stage": "eval", "steps": 4, "runs": [
                {"steps": 4, "samples_per_second": 2.0}]},
            stage="eval", device="h100")
        after = pgo.measurement_report(
            {"stage": "eval", "steps": 4, "runs": [
                {"steps": 4, "samples_per_second": 3.0}]},
            stage="eval", device="h100")
        verdict = pgo.compare(before, after)
        self.assertEqual(verdict["decision"], "kept")
        self.assertGreater(verdict["gain"], 0.4)


class LoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="pgo-loop-"))
        self.folder = pgo.run_root(self.root, "base", "h100")
        self.loop = pgo.new_loop("base", "h100")
        pgo.save_loop(self.loop, self.folder / "loop.json")

    def _state(self):
        return pgo.load_loop(self.folder / "loop.json")

    def test_the_state_names_the_metric_of_the_stage(self) -> None:
        self.assertEqual(self.loop["schema"], pgo.LOOP_SCHEMA)
        self.assertEqual(self.loop["metric"]["name"], "step_ms")
        self.assertEqual(pgo.new_loop("eval", "t4")["metric"]["direction"],
                         "higher")

    def test_an_unknown_stage_is_a_fault(self) -> None:
        with self.assertRaises(pgo.PgoError):
            pgo.new_loop("pretrain", "h100")
        with self.assertRaises(pgo.PgoError):
            pgo.new_loop("base", "a100")

    def test_the_next_action_walks_the_phases(self) -> None:
        self.assertEqual(pgo.next_action(self._state(), self.folder)["action"],
                         "measure")
        current = pgo.iteration_dir(self.folder, 1)
        (current / "before").mkdir(parents=True)
        (current / "before" / "measurement.json").write_text("{}")
        self.assertEqual(pgo.next_action(self._state(), self.folder)["action"],
                         "trace")
        (current / "before" / "trace").mkdir()
        (current / "before" / "trace" / "base.nsys-rep").write_bytes(b"x")
        self.assertEqual(pgo.next_action(self._state(), self.folder)["action"],
                         "analyze")
        (current / "before" / "analysis.json").write_text("{}")
        self.assertEqual(pgo.next_action(self._state(), self.folder)["action"],
                         "rank")
        (current / "ideas.json").write_text("{}")
        self.assertEqual(pgo.next_action(self._state(), self.folder)["action"],
                         "measure")
        (current / "after").mkdir()
        (current / "after" / "measurement.json").write_text("{}")
        self.assertEqual(pgo.next_action(self._state(), self.folder)["action"],
                         "compare")
        (current / "verdict.json").write_text("{}")
        self.assertEqual(pgo.next_action(self._state(), self.folder)["action"],
                         "record")

    def test_the_cap_stops_the_loop(self) -> None:
        loop = pgo.new_loop("base", "h100", max_iterations=1)
        loop["iteration"] = 2
        self.assertEqual(pgo.next_action(loop, self.folder)["action"], "stop")

    def test_a_kept_pair_becomes_the_baseline(self) -> None:
        verdict = {"decision": "kept", "gain": 0.4, "metric": "step_ms",
                   "before": 0.5, "after": 0.3, "mfu_before": 0.003,
                   "mfu_after": 0.005, "overlap": False}
        loop = pgo.record(self._state(), verdict, pair=["pgo-1", "pgo-2"],
                          pair_ceiling=0.7,
                          after={"before_value": 0.5, "after_value": 0.3})
        self.assertEqual(loop["iteration"], 2)
        self.assertEqual(loop["baseline"]["value"], 0.3)
        self.assertEqual(loop["baseline"]["mfu"], 0.005)
        self.assertEqual(len(loop["history"]), 1)
        self.assertEqual(loop["rejected"], [])

    def test_a_reverted_pair_is_not_retried(self) -> None:
        verdict = {"decision": "reverted", "gain": 0.01, "metric": "step_ms",
                   "before": 0.5, "after": 0.49, "overlap": True}
        loop = pgo.record(self._state(), verdict, pair=["pgo-3", "pgo-4"])
        self.assertEqual(loop["rejected"], ["pgo-3", "pgo-4"])
        self.assertIsNone(loop["baseline"])

    def test_an_after_point_needs_a_before_point(self) -> None:
        job = self.folder / "job.json"
        job.write_text("{}", encoding="utf-8")
        loop = self._state()
        loop["job"] = str(job)
        pgo.save_loop(loop, self.folder / "loop.json")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = pgo.main(["--root", str(self.root), "measure", "--stage",
                             "base", "--device", "h100", "--point", "after",
                             "--dry-run"])
        self.assertEqual(code, 1)
        self.assertIn("no before point", stderr.getvalue())

    def test_the_iteration_folder_carries_its_number(self) -> None:
        self.assertEqual(pgo.iteration_dir(self.folder, 3).name, "iter-03")
        self.assertEqual(pgo.run_root(self.root, "rl", "t4").name, "rl-t4")


class CommandTest(unittest.TestCase):
    def test_the_trace_command_names_the_entry_point_and_the_budget(self) -> None:
        argv = pgo.trace_command(stage="base", device="h100", run_id="r1",
                                 job_path="job.json", steps=50, seconds=180)
        # The name must be a local entrypoint of tools/modal/app.py.
        # tools/modal/modal_app_test.py checks that it is.
        self.assertTrue(argv[2].endswith("::profile_stage"))
        self.assertIn("--steps", argv)
        self.assertEqual(argv[argv.index("--seconds") + 1], "180")

    def test_the_measure_command_takes_the_run_count(self) -> None:
        argv = pgo.measure_command(stage="sft", device="t4", run_id="r2",
                                   job_path="job.json", steps=50, seconds=180,
                                   runs=2)
        self.assertTrue(argv[2].endswith("::measure_stage"))
        self.assertEqual(argv[argv.index("--runs") + 1], "2")

    def test_the_fetch_commands_name_every_artifact_once(self) -> None:
        commands = pgo.fetch_commands(stage="base", run_id="r1",
                                     destination=Path("/tmp/trace"))
        names = [command[-1].rsplit("/", 1)[-1] for command in commands]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("base.nsys-rep", names)
        self.assertIn("timeline_cuda_gpu_trace.csv", names)

    def test_the_last_json_document_wins(self) -> None:
        text = ("container chatter\n{\n \"early\": true\n}\nmore chatter\n"
                "{\n \"files\": [], \"steps\": 50\n}\n")
        self.assertEqual(pgo.last_json(text)["steps"], 50)

    def test_a_document_followed_by_cli_lines_is_read(self) -> None:
        # The shape the Modal CLI writes: the document, then its own lines.
        text = ("container noise\n{\n \"runs\": [], \"steps\": 20\n}\n"
                "Stopping app - local entrypoint completed.\n"
                "App completed.\n")
        self.assertEqual(pgo.last_json(text)["steps"], 20)

    def test_output_without_a_document_is_a_fault(self) -> None:
        with self.assertRaises(pgo.PgoError):
            pgo.last_json("no document here")


if __name__ == "__main__":
    unittest.main()
