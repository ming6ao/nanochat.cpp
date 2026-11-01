"""T0: the one-job chain -- base, SFT, RL, then eval -- at the demo profile.

The test drives the real chain: :func:`nanochat_cpp.pipeline.run` on
:data:`nanochat_cpp.pipeline.PROFILES` ``demo``, which runs the mesh training
driver, the SFT driver, and the RL worker over the committed fixture corpus and
then scores the two checkpoints in this process. The runfiles carry the shared
library, the two stage executables, the parquet document set, and the tokenizer
artifact; no stage reads the Hub (``docs/pipeline-plan.md`` D4).

The profile is named explicitly, because ``Job()`` defaults to ``standard``
(8 layers, hidden 512) -- the shape the throughput facts are read at, not the
shape a CPU gate can run in five seconds. The step counts are named with it, so
this class keeps the short demo schedule (``docs/pipeline-scaleup-plan.md``
section 5) whatever the defaults become.

The assertions are the contract of section 7, not the loss numbers: the parity
rungs own the numbers. A10 to A13 are the reviewability of the eval stage
(``docs/pipeline-scaleup-plan.md`` section 10).

=====  ===================================================================
A1     each stage returns status 0 with the step count it was asked for
A2     every loss and reward is finite, and the base loss falls
A3     the metadata chain is base@100, sft@100, rl@105, and every recorded
       config equals the job's
A4     the parameter L2 differs between base and SFT and between SFT and RL
A5     one group of rollout rewards holds two distinct values
A6     a graded reward and a constant reward, from the same SFT checkpoint
       and the same seed, give different final parameters
A7     ``deliver/chatrl.ckpt`` equals the RL checkpoint byte for byte, and
       the report digests match the bytes
A8     the report names the stage order, the seeds, and the host facts
A9     the delivered file loads through the public container reader and
       reports ``rl`` at step 105
A10    the eval stage ran, both bits-per-byte values are finite and inside
       [0, 8], and every sample is non-empty text
A11    the RL checkpoint's digest is the same before and after the eval
       stage, so an eval pass wrote nothing to the file it read
A12    the eval record names the evaluated checkpoint, its digest, the
       corpus, the batch size, and the batch count
A13    ``deliver/chatrl.ckpt`` carries the RL weights and its metadata
       reports ``rl``, because the last stage is the eval stage and that
       stage writes no checkpoint
=====  ===================================================================

A4 does not prove the warm start, because the loaded moments move the weights
on their own; A3 and the argv ``nc.sft.run`` builds are what prove the chain.
A5 is read through ``rl.run(..., on_step=...)``, which receives the per-rollout
rewards, because :class:`nanochat_cpp.rl.RlReport` carries no advantage. A6 is a
two-run comparison and not "a constant reward leaves the parameters unchanged":
a warm start carries the optimizer state (``docs/post-training.md`` section 7),
so the inherited AdamW moments move the weights even with an advantage vector of
exactly zero (``docs/pipeline-plan.md`` D11).

Row 7's open check, settled by measurement. The plan's premise was 9 rows and
about 530 characters; the committed parquet holds ten documents and 109 bytes of
text (``src/parquet/reader_test.cc`` lists them). One epoch therefore yields at
most 119 tokens -- one BOS per document, and no BPE token shorter than one byte
-- and a demo row is 65 tokens (sequence 64 plus the target). The loader returns
no partial row, so a batch of two rows, 130 tokens, cannot form: that is why
``pipeline.BASE_BATCH`` is 1, and epoch cycling does not help there because a
single batch never fits inside one epoch. Cycling does carry the run to step
100, since ``src/mesh_train.cc`` calls ``DataLoader::Reset`` when a batch is
exhausted. ``CorpusBudgetTest`` asserts both numbers independently of the chain,
so a red A1 reads as one of them, and A1 stays the gate.
"""

from __future__ import annotations

import array
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nanochat_cpp import _build, api, checkpoint, pipeline, rl

#: The environment this class owns. ``NANOCHAT_BASE_DIR`` keeps the published
#: tokenizer out of the developer's cache, and ``NANOCHAT_SANDBOX_BACKEND``
#: makes the RL stage start the worker itself: a ``py_test``'s runfiles carry no
#: ``tools/nanochat``, and the sentinel the child's ``RequireSandboxOrDie``
#: reads is ``NANOCHAT_SANDBOX``, which the sandbox wrapper sets for every test
#: action (``docs/pipeline-plan.md`` D10 and section 7).
OWNED_ENV = ("NANOCHAT_BASE_DIR", "NANOCHAT_RUNS_DIR",
             "NANOCHAT_SANDBOX_BACKEND", "NANOCHAT_SOURCE_REVISION")

#: The step count the RL stage records: the SFT stage's 100 plus its own 5.
RL_STEP = 105

#: The schedule of the demo gate: the step counts this gate runs, which are
#: half the default job's 200/200/8 (``docs/pipeline-scaleup-plan.md`` section
#: 5). They are passed by hand, so the gate keeps costing about five seconds and
#: the numbers A1 to A9 assert stay the numbers of the short schedule.
DEMO_SCHEDULE = {"base_steps": 100, "sft_steps": 100, "rl_steps": 5}


def demo_job(run_id: str, out_root, **overrides) -> pipeline.Job:
    """One job at the ``demo`` profile with the gate's short schedule.

    The profile is named because :class:`nanochat_cpp.pipeline.Job` defaults to
    ``standard``, the 8-layer shape the throughput facts are read at; a gate
    that inherited it would either cost minutes or, worse, assert the ``demo``
    numbers against another shape. ``demo`` is 2 layers of hidden 64, which is
    about five seconds of CPU for the whole chain.
    """
    fields = {"run_id": run_id, "profile": "demo",
              "out_root": str(out_root)}
    fields.update(DEMO_SCHEDULE)
    fields.update(overrides)
    return pipeline.Job(**fields)


class _ConstantReward:
    """A reward that never grades: every rollout of a group scores the same.

    It is the control leg of A6, so it has the shape
    :func:`nanochat_cpp.rl._normalize_task` accepts and no Hub dataset.
    """

    def reward(self, conversation, completion) -> float:
        return 0.0


def _float_values(record) -> list:
    """The element values of one parameter record, as floats."""
    stored = checkpoint.dtype_name(record.dtype)
    if stored == "float32":
        buffer = array.array("f")
        buffer.frombytes(record.data)
        if sys.byteorder != "little":
            buffer.byteswap()
        return buffer.tolist()
    if stored == "bfloat16":
        halves = array.array("H")
        halves.frombytes(record.data)
        if sys.byteorder != "little":
            halves.byteswap()
        wide = bytearray()
        for value in halves:
            wide.extend(struct.pack("<H", int(value)) + b"\x00\x00")
        buffer = array.array("f")
        buffer.frombytes(bytes(wide))
        return buffer.tolist()
    raise AssertionError(
        f"{record.name} is stored as {stored}, which this gate does not read")


def digest(path) -> str:
    """The SHA-256 hex digest of one file."""
    block = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            block.update(chunk)
    return block.hexdigest()


def parameter_bytes(path) -> dict:
    """The parameter records of one checkpoint, as ``name -> payload``."""
    return {record.name: record.data
            for record in checkpoint.read_checkpoint(path)}


def parameter_norm(path) -> float:
    """The L2 norm of every parameter of one checkpoint."""
    total = 0.0
    for record in checkpoint.read_checkpoint(path):
        for value in _float_values(record):
            total += float(value) * float(value)
    return math.sqrt(total)


def recorded_config(shape) -> dict:
    """The model shape as the container's metadata block records it.

    The keys are the members of ``nanochat::Config`` in declaration order
    (``bindings/nanochat_capi.cc``, ``ConfigJson``), so a recorded config that
    differs from the job's is one key away from a readable failure.
    """
    return {
        "num_layers": int(shape.num_layers),
        "num_heads": int(shape.num_heads),
        "num_kv_heads": int(shape.num_kv_heads),
        "hidden_dim": int(shape.hidden_dim),
        "seq_len": int(shape.seq_len),
        "vocab_size": int(shape.vocab_size),
        "padded_vocab_size": int(shape.padded_vocab_size),
        "rope_base": float(shape.rope_base),
        "window_pattern": str(shape.window_pattern),
        "value_embedding": bool(shape.value_embedding),
    }


class PipelineTest(unittest.TestCase):
    """The integration gate of ``docs/pipeline-plan.md`` section 7."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="nanochat_pipeline_")
        cls.root = Path(cls.tmp.name)
        cls.saved = {name: os.environ.get(name) for name in OWNED_ENV}
        os.environ["NANOCHAT_BASE_DIR"] = str(cls.root / "cache")
        os.environ["NANOCHAT_SANDBOX_BACKEND"] = "none"
        os.environ.pop("NANOCHAT_SOURCE_REVISION", None)
        cls.job = demo_job("gate", cls.root / "runs")
        cls.shape = pipeline.config(cls.job)
        cls.report = pipeline.run(cls.job)
        cls.stages = {stage.name: stage for stage in cls.report.stages}
        # A11 reads the RL file's digest now, which is after the eval stage
        # ran: the RL stage recorded the same digest before the eval stage
        # started, so the two together say the eval wrote nothing.
        cls.rl_digest_after_eval = digest(cls.stages["rl"].checkpoint)
        cls.run_dir = cls.root / "runs" / cls.job.run_id
        # The two legs of A6. The graded leg is the chain's own reward and the
        # control leg is constant, so the only difference between them is the
        # advantage term; both start from the same SFT checkpoint at the same
        # seed. A5 reads the graded leg's per-rollout rewards here, because
        # `rl.run(..., on_step=...)` is the one place they surface.
        cls.legs = {
            "graded": cls.rl_leg("graded", pipeline.GradedReward()),
            "constant": cls.rl_leg("constant", _ConstantReward()),
        }

    @classmethod
    def tearDownClass(cls) -> None:
        for name, value in cls.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        cls.tmp.cleanup()

    @classmethod
    def conversations(cls) -> list:
        """The corpus conversations the chain's RL stage rolled out."""
        path = cls.run_dir / pipeline.CORPUS_DIRNAME / \
            pipeline.CONVERSATIONS_NAME
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["conversations"]

    @classmethod
    def rl_leg(cls, name: str, reward) -> dict:
        """One RL run from the SFT checkpoint, with the job's rollout shape."""
        out_dir = cls.root / "a6" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "checkpoint.ckpt"
        steps: list = []

        def on_step(step: int, loss: float, grad_norm: float,
                    rewards: list) -> None:
            steps.append([float(value) for value in rewards])

        report = rl.run(
            reward,
            cls.conversations()[:int(cls.job.num_prompts)],
            num_samples=int(cls.job.num_samples),
            steps=int(cls.job.rl_steps),
            checkpoint=target,
            warm_start=cls.stages["sft"].checkpoint,
            config=cls.shape,
            max_tokens=int(cls.job.max_tokens),
            temperature=float(cls.job.temperature),
            top_k=int(cls.job.top_k),
            seed=int(cls.job.seed),
            on_step=on_step,
        )
        return {
            "checkpoint": str(target),
            "report": report,
            "rewards": steps,
            "parameters": parameter_bytes(target),
        }

    def rollout_shape(self) -> str:
        """The rollout settings, for a failure message (section 9)."""
        return (f"num_samples={self.job.num_samples} "
                f"max_tokens={self.job.max_tokens} "
                f"temperature={self.job.temperature} "
                f"top_k={self.job.top_k} seed={self.job.seed} "
                f"num_prompts={self.job.num_prompts} "
                f"rl_steps={self.job.rl_steps}")

    # -- A1 ----------------------------------------------------------------

    def test_a1_every_stage_ran_the_steps_it_was_asked_for(self) -> None:
        """A shortened schedule is a failure, not a faster gate."""
        base = self.stages["base"]
        self.assertEqual(base.steps, self.job.base_steps)
        self.assertEqual(base.extra["steps_ran"], self.job.base_steps)
        self.assertFalse(base.extra["interrupted"])
        sft = self.stages["sft"]
        self.assertEqual(sft.steps, self.job.sft_steps)
        self.assertEqual(sft.extra["exit_status"], 0)
        self.assertEqual(self.stages["rl"].steps, self.job.rl_steps)
        self.assertEqual(len(self.stages["rl"].losses), self.job.rl_steps)

    # -- A2 ----------------------------------------------------------------

    def test_a2_every_loss_and_reward_is_finite(self) -> None:
        """A dead optimizer shows up as a NaN or an infinity."""
        base = self.stages["base"]
        self.assertEqual(len(base.losses), self.job.base_steps)
        for loss in base.losses:
            self.assertTrue(math.isfinite(loss), f"base loss {loss}")
        for loss in self.stages["rl"].losses:
            self.assertTrue(math.isfinite(loss), f"rl loss {loss}")
        rl_extra = self.stages["rl"].extra
        for reward in rl_extra["mean_rewards"]:
            self.assertTrue(math.isfinite(reward), f"mean reward {reward}")
        for step_rewards in rl_extra["rewards"]:
            for reward in step_rewards:
                self.assertTrue(math.isfinite(reward), f"reward {reward}")

    def test_a2_the_base_loss_falls(self) -> None:
        """Step 100 is below step 1, so the stage learned something."""
        losses = self.stages["base"].losses
        self.assertLess(losses[-1], losses[0],
                        f"the base loss went from {losses[0]} to "
                        f"{losses[-1]}, so the optimizer did nothing")

    # -- A3 ----------------------------------------------------------------

    def test_a3_the_metadata_chain(self) -> None:
        """base@100, sft@100, rl@105, and every config equal to the job's.

        This is what catches B1 and B2 (a stage that cannot set the model
        shape, or a flag the binary does not know) and a warm start that
        loaded the wrong file.
        """
        expected_steps = {
            "base": self.job.base_steps,
            "sft": self.job.sft_steps,
            "rl": RL_STEP,
        }
        shape = recorded_config(self.shape)
        for name, steps in expected_steps.items():
            with self.subTest(stage=name):
                metadata = checkpoint.read_metadata(
                    self.stages[name].checkpoint)
                self.assertEqual(metadata["stage"], name)
                self.assertEqual(metadata["step"], steps)
                recorded = metadata["config"]
                self.assertEqual(sorted(recorded), sorted(shape))
                for key, value in shape.items():
                    if key == "rope_base":
                        self.assertAlmostEqual(recorded[key], value, places=4)
                    else:
                        self.assertEqual(recorded[key], value,
                                         f"the {name} stage recorded "
                                         f"{key}={recorded[key]!r}")
                # The report carries the same block, so a reviewer reads the
                # shape from the report and not from the container.
                self.assertEqual(self.stages[name].metadata["stage"], name)
                self.assertEqual(self.stages[name].metadata["step"], steps)

    def test_a3_the_sft_stage_warm_started_from_the_base_stage(self) -> None:
        self.assertEqual(self.stages["sft"].extra["warm_start"],
                         self.stages["base"].checkpoint)
        self.assertEqual(self.stages["rl"].extra["warm_start"],
                         self.stages["sft"].checkpoint)

    # -- A4 ----------------------------------------------------------------

    def test_a4_the_parameters_moved_at_every_stage(self) -> None:
        """A stage that only rewrites its file leaves the norm unchanged."""
        norms = {name: parameter_norm(self.stages[name].checkpoint)
                 for name in ("base", "sft", "rl")}
        for name, value in norms.items():
            self.assertTrue(math.isfinite(value),
                            f"the {name} norm is {value}")
            self.assertGreater(value, 0.0, f"the {name} stage has no weights")
        self.assertNotEqual(norms["base"], norms["sft"],
                            f"base and sft share the norm {norms['base']}")
        self.assertNotEqual(norms["sft"], norms["rl"],
                            f"sft and rl share the norm {norms['sft']}")

    # -- A5 ----------------------------------------------------------------

    def test_a5_a_group_of_rollouts_holds_two_distinct_rewards(self) -> None:
        """B4 gave every rollout of a group the same text, so all advantages
        were zero. A group with two distinct rewards has a non-zero advantage.
        """
        steps = self.legs["graded"]["rewards"]
        self.assertTrue(steps, "rl.run reported no step")
        samples = int(self.job.num_samples)
        distinct = []
        for step_rewards in steps:
            for start in range(0, len(step_rewards), samples):
                group = step_rewards[start:start + samples]
                distinct.append(len(set(group)))
        self.assertGreater(max(distinct), 1,
                           f"no rollout group held two distinct rewards "
                           f"({self.rollout_shape()}); the groups were "
                           f"{steps[0]}")
        # The chain's own RL stage recorded the same facts, and it counts the
        # groups whose advantages are all zero, so a uniform reward is visible
        # in the report as well.
        rl_extra = self.stages["rl"].extra
        groups = len(rl_extra["rewards"]) * int(self.job.num_prompts)
        self.assertLess(rl_extra["dropped_groups"], groups,
                        f"every one of the {groups} groups was dropped "
                        f"({self.rollout_shape()})")

    # -- A6 ----------------------------------------------------------------

    def test_a6_the_advantage_term_moves_the_parameters(self) -> None:
        """Two runs, one graded and one constant, from one warm start.

        The inherited AdamW moments move the weights even with an advantage
        vector of exactly zero, so "the parameters changed" is not the
        assertion; the assertion is that the graded run and the constant run
        end in different parameters, which isolates the advantage term
        (``docs/pipeline-plan.md`` D11).
        """
        self.assertEqual(
            (self.job.num_samples, self.job.max_tokens, self.job.temperature,
             self.job.top_k), (4, 32, 1.5, 0),
            "the rollout shape is pinned in Job, so a change here is a "
            "deliberate one and A6's comparison must be re-read")
        graded = self.legs["graded"]
        constant = self.legs["constant"]
        self.assertEqual(graded["report"].steps, constant["report"].steps)
        # The control leg never grades, and the graded leg does, so the two
        # runs saw the same rollout shape with different rewards -- which is
        # the advantage vector the comparison isolates.
        self.assertTrue(
            all(reward == 0.0 for step in constant["rewards"]
                for reward in step),
            f"the constant leg graded a rollout ({self.rollout_shape()})")
        self.assertTrue(
            any(reward != 0.0 for step in graded["rewards"]
                for reward in step),
            f"the graded leg scored every rollout zero "
            f"({self.rollout_shape()})")
        first = graded["parameters"]
        second = constant["parameters"]
        self.assertEqual(sorted(first), sorted(second))
        differing = [name for name in first if first[name] != second[name]]
        self.assertTrue(
            differing,
            f"the graded and the constant run ended in the same parameters "
            f"({self.rollout_shape()}), so the advantage term moved nothing")
        # The same comparison on the numbers a reader sees: the two legs also
        # report different mean rewards, so the grading itself is live.
        self.assertNotEqual(graded["report"].mean_rewards,
                            constant["report"].mean_rewards)

    # -- A7 ----------------------------------------------------------------

    def test_a7_the_delivered_file_is_the_rl_checkpoint(self) -> None:
        """A lost artifact is a failed delivery, not a warning."""
        delivered = Path(self.report.delivered)
        self.assertTrue(delivered.is_file(), f"{delivered} is missing")
        # D6 names the folder, so assert the layout and not only the path the
        # report carries: the chain once wrote `<run_root>/deliver` while every
        # stage wrote `<run_root>/<run_id>/`, and only the Modal app read the
        # design's folder, so every local assertion still passed.
        run_dir = Path(self.job.out_root) / self.job.run_id
        self.assertEqual(delivered.parent, run_dir / "deliver",
                         f"the delivery is not under {run_dir / 'deliver'}")
        self.assertEqual(Path(self.report.report_path).parent,
                         run_dir / "deliver")
        self.assertEqual(delivered.read_bytes(),
                         Path(self.stages["rl"].checkpoint).read_bytes())
        self.assertEqual(delivered.name, "chatrl.ckpt")
        self.assertEqual(delivered.parent.name, "deliver")
        computed = digest(delivered)
        self.assertEqual(computed, self.stages["rl"].sha256)
        # The last stage is the eval stage and it writes no checkpoint, so it
        # names the RL file as its own artifact: the delivered copy follows the
        # RL weights and not the stage that ran last (section 6).
        self.assertEqual(self.report.stages[-1].name, "eval")
        self.assertEqual(self.report.stages[-1].checkpoint,
                         self.stages["rl"].checkpoint)
        self.assertEqual(self.report.stages[-1].sha256, computed)
        self.assertEqual(self.stages["rl"].sha256, computed)
        # The report on disk names the same two files, so a reviewer who opens
        # the report finds the bytes beside it.
        on_disk = pipeline.Report.from_json(
            Path(self.report.report_path).read_text(encoding="utf-8"))
        self.assertEqual(on_disk.delivered, self.report.delivered)
        self.assertEqual(on_disk.to_dict(), self.report.to_dict())

    # -- A8 ----------------------------------------------------------------

    def test_a8_the_report_names_the_order_the_seeds_and_host(self) -> None:
        """An unreviewable run is a failed run."""
        self.assertEqual([stage.name for stage in self.report.stages],
                         ["base", "sft", "rl", "eval"])
        self.assertEqual([stage.steps for stage in self.report.stages],
                         [self.job.base_steps, self.job.sft_steps,
                          self.job.rl_steps, self.job.eval_batches])
        self.assertEqual(self.report.job.seed, self.job.seed)
        self.assertEqual(self.report.job.to_dict(), self.job.to_dict())
        self.assertEqual(self.report.version, pipeline.REPORT_VERSION)
        host = self.report.host
        self.assertEqual(sorted(host),
                         ["arch", "backend", "precision", "revision"])
        self.assertEqual(host["backend"], _build.backend())
        self.assertEqual(host["precision"], _build.precision())
        self.assertGreater(self.report.total_seconds, 0.0)
        for stage in self.report.stages:
            self.assertGreater(stage.seconds, 0.0)
            self.assertTrue(stage.sha256)
        # The tokenizer digest travels with the base stage's corpus block, so
        # the report names the bytes every stage was trained on.
        self.assertEqual(
            self.stages["base"].extra["tokenizer_sha256"],
            digest(self.run_dir / pipeline.CORPUS_DIRNAME /
                   "loader_tokenizer.nctoken"))
        # The review artifact is one JSON file a reader can diff.
        payload = json.loads(
            Path(self.report.report_path).read_text(encoding="utf-8"))
        self.assertEqual([stage["name"] for stage in payload["stages"]],
                         ["base", "sft", "rl", "eval"])
        self.assertEqual(payload["job"]["seed"], self.job.seed)

    # -- A9 ----------------------------------------------------------------

    def test_a9_the_delivered_file_loads_through_the_reader(self) -> None:
        """A container only the pipeline can read is not an artifact.

        The reader is applied to ``deliver/chatrl.ckpt``, the file the *last*
        stage names as its own artifact and did not write, so a delivery that
        followed a written eval checkpoint would read a different file or none
        at all.
        """
        delivered = self.report.delivered
        parameters = checkpoint.read_checkpoint(delivered)
        self.assertTrue(parameters, "the delivered file holds no parameter")
        metadata = checkpoint.read_metadata(delivered)
        self.assertEqual(metadata["format"], checkpoint.CONTAINER_VERSION_1)
        self.assertEqual(metadata["stage"], "rl")
        self.assertEqual(metadata["step"], RL_STEP)
        self.assertEqual(metadata["config"]["num_layers"],
                         self.shape.num_layers)

    # -- A10 ---------------------------------------------------------------

    def test_a10_the_eval_stage_scored_both_checkpoints(self) -> None:
        """A silent eval and a wrong ``eval_batch`` both fail here.

        ``eval_batch`` is 1 because no evaluator call resets the data loader and
        the committed corpus fills one 65-token row per epoch, so a batch of
        eight reports no bytes and returns infinity (section 6). The two bits
        per byte are two different numbers, because the two checkpoints are two
        different files; an eval wired to one file twice would report one
        number twice.
        """
        stage = self.stages["eval"]
        self.assertEqual(stage.name, "eval")
        self.assertEqual(stage.steps, self.job.eval_batches)
        self.assertEqual(int(stage.extra["eval_batches"]),
                         int(self.job.eval_batches))
        self.assertEqual(int(stage.extra["eval_batch"]), 1)
        self.assertEqual(pipeline.EVAL_BATCH, 1)
        values = {}
        for name in ("bpb_base", "bpb_rl"):
            self.assertIn(name, stage.extra)
            value = float(stage.extra[name])
            values[name] = value
            self.assertTrue(math.isfinite(value), f"{name} is {value}")
            self.assertGreaterEqual(
                value, 0.0,
                f"{name} is {value}, and a bits-per-byte count cannot be "
                f"negative; the eval scored no byte")
            self.assertLessEqual(
                value, 8.0,
                f"{name} is {value}, which is above the [0, 8] band of "
                f"section 10: a batch that could not form reports no bytes "
                f"and an infinity, and a wrong ``eval_batch`` is how that "
                f"happens (section 6)")
        self.assertNotEqual(
            values["bpb_base"], values["bpb_rl"],
            f"both checkpoints score {values['bpb_base']}, so the eval read "
            f"one file twice")
        samples = stage.extra["samples"]
        self.assertEqual(len(samples), int(self.job.eval_samples),
                         f"the stage sampled {len(samples)} completions")
        for index, sample in enumerate(samples):
            self.assertIsInstance(sample, str, f"sample {index} is {sample!r}")
            self.assertTrue(sample.strip(),
                            f"sample {index} is empty text, so the report "
                            f"names a completion a reader cannot read")

    # -- A11 ---------------------------------------------------------------

    def test_a11_the_eval_stage_wrote_no_checkpoint(self) -> None:
        """The RL file's digest, before and after the eval stage.

        "Before" is the digest the RL stage recorded when it closed the file:
        ``stage_rl`` hashes the container it wrote, and the eval stage starts
        after that. "After" is the digest read once the whole chain finished.
        A pass that wrote to the container it read would have moved it, which
        is why the eval record's digest is read back after its own work
        (section 6).
        """
        recorded = self.stages["rl"].sha256
        self.assertTrue(recorded, "the RL stage recorded no digest")
        self.assertEqual(
            self.rl_digest_after_eval, recorded,
            f"the RL checkpoint {self.stages['rl'].checkpoint} changed between "
            f"the RL stage and the end of the chain, so a later stage wrote "
            f"to it")
        self.assertEqual(digest(self.stages["rl"].checkpoint), recorded)
        # The eval stage names the RL file and writes nothing of its own, so
        # there is no eval directory and no second checkpoint (P4).
        self.assertFalse(
            (self.run_dir / "eval").exists(),
            f"the eval stage wrote {self.run_dir / 'eval'}, and the design "
            f"says it writes no checkpoint")

    # -- A12 ---------------------------------------------------------------

    def test_a12_the_eval_record_names_what_it_evaluated(self) -> None:
        """An unreviewable eval is a failed eval.

        ``eval_batch`` is asserted against the literal 1: the committed corpus
        allows one 65-token row per epoch, so the default batch of 8 is exactly
        the defect section 6 names, and a report that carries it would have
        recorded an infinity instead of a number.
        """
        extra = self.stages["eval"].extra
        for name in ("evaluated", "evaluated_sha256", "eval_corpus",
                     "eval_batch", "eval_batches"):
            self.assertIn(name, extra,
                          f"the eval record does not name {name}")
        self.assertEqual(extra["evaluated"], self.stages["rl"].checkpoint,
                         "the eval record does not name the RL checkpoint")
        self.assertEqual(extra["eval_corpus"],
                         self.stages["base"].extra["parquet"],
                         "the eval read a corpus the base stage did not")
        self.assertEqual(Path(extra["eval_corpus"]).parent,
                         self.run_dir / pipeline.CORPUS_DIRNAME)
        self.assertTrue(Path(extra["eval_corpus"]).is_file(),
                        f"{extra['eval_corpus']} is missing")
        self.assertEqual(extra["evaluated_sha256"], digest(extra["evaluated"]))
        self.assertEqual(extra["evaluated_sha256"], self.stages["rl"].sha256)
        self.assertEqual(int(extra["eval_batches"]), int(self.job.eval_batches))
        self.assertEqual(int(extra["eval_batch"]), 1,
                         "the eval did not name the one batch the corpus "
                         "fills: 8 is the api default and reports no bytes "
                         "on this corpus (section 6)")

    # -- A13 ---------------------------------------------------------------

    def test_a13_the_delivered_file_is_still_the_rl_weights(self) -> None:
        """The last stage is the eval stage, and the delivery follows the
        record and not the stage order.

        ``deliver()`` copies ``stages[-1].checkpoint``, and the eval stage
        names the RL file it read, so ``deliver/chatrl.ckpt`` holds the RL
        weights. The metadata is the proof a reader has: it says ``rl`` at the
        RL step, whatever ran last.
        """
        delivered = Path(self.report.delivered)
        self.assertTrue(delivered.is_file(), f"{delivered} is missing")
        self.assertEqual(self.report.stages[-1].name, "eval")
        self.assertEqual(self.stages["eval"].checkpoint,
                         self.stages["rl"].checkpoint)
        self.assertEqual(digest(delivered), self.stages["rl"].sha256)
        metadata = checkpoint.read_metadata(delivered)
        self.assertEqual(metadata["stage"], "rl")
        self.assertEqual(metadata["step"], RL_STEP)
        self.assertEqual(self.stages["eval"].metadata["stage"], "rl")


class JobValidationTest(unittest.TestCase):
    """The job and profile rules of ``docs/pipeline-scaleup-plan.md`` section 5.

    ``_check`` is the one gate a job passes before a stage starts, so a bad
    field is a message that names the field and a bad profile is a message that
    names the ``Config::Validate`` rule -- never a driver failing inside the
    chain with a C flag on its stderr. Nothing here runs a stage, so the class
    needs no library, no executable, and no runfiles data.
    """

    def test_a_non_positive_eval_schedule_is_rejected(self) -> None:
        """``eval_batches`` and ``eval_samples`` are part of the schedule."""
        for field in ("eval_batches", "eval_samples"):
            for value in (0, -1):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(pipeline.PipelineError) as caught:
                        pipeline._check(demo_job("bad", "", **{field: value}))
                    self.assertIn(field, str(caught.exception))

    def test_a_profile_that_breaks_config_validate_is_rejected(self) -> None:
        """One case per rule of ``include/nanochat/config.h``.

        The profile is injected into ``PROFILES`` for the length of the case, so
        the rule is exercised through the same path a named profile takes; a
        profile that broke a rule otherwise failed inside a stage executable,
        where the message names the C flag and not the profile (section 5,
        P1).
        """
        cases = (
            ("hidden_dim not divisible by num_heads",
             api.Config(num_layers=2, num_heads=3, num_kv_heads=1,
                        hidden_dim=64, seq_len=64, vocab_size=487,
                        padded_vocab_size=512, window_pattern="L"),
             "hidden_dim must be divisible by num_heads"),
            ("num_heads not divisible by num_kv_heads",
             api.Config(num_layers=2, num_heads=4, num_kv_heads=3,
                        hidden_dim=64, seq_len=64, vocab_size=487,
                        padded_vocab_size=512, window_pattern="L"),
             "num_heads must be divisible by num_kv_heads"),
            ("vocab_size above padded_vocab_size",
             api.Config(num_layers=2, num_heads=4, num_kv_heads=2,
                        hidden_dim=64, seq_len=64, vocab_size=600,
                        padded_vocab_size=512, window_pattern="L"),
             "vocab_size must not be above padded_vocab_size"),
        )
        for rule, shape, needle in cases:
            with self.subTest(rule=rule):
                with mock.patch.dict(pipeline.PROFILES, {"broken": shape}):
                    with self.assertRaises(pipeline.PipelineError) as caught:
                        pipeline._check(pipeline.Job(profile="broken"))
                message = str(caught.exception)
                self.assertIn("'broken'", message,
                              f"the message does not name the profile: "
                              f"{message}")
                self.assertIn(needle, message,
                              f"the message does not name the rule: "
                              f"{message}")

    def test_the_default_job_is_the_standard_profile(self) -> None:
        """P1: the default runs the shape a throughput number is read at."""
        job = pipeline.Job()
        self.assertEqual(job.profile, "standard")
        self.assertEqual((job.base_steps, job.sft_steps, job.rl_steps),
                         (200, 200, 8))
        self.assertEqual(job.eval_batches, 1)
        self.assertEqual(job.eval_samples, 2)
        shape = pipeline.PROFILES[job.profile]
        self.assertEqual(
            (shape.num_layers, shape.hidden_dim, shape.num_heads,
             shape.num_kv_heads, shape.seq_len, shape.vocab_size,
             shape.padded_vocab_size, shape.window_pattern),
            (8, 512, 8, 4, 64, 487, 512, "L"),
            f"the standard profile is {shape}")
        # The default job passes the gate, so a default that broke a rule would
        # fail here and not at the first stage (section 5).
        pipeline._check(job)

    def test_the_demo_profile_keeps_its_short_shape(self) -> None:
        """``demo`` stays the two-layer shape the CPU gate runs in seconds."""
        shape = pipeline.PROFILES["demo"]
        self.assertEqual(
            (shape.num_layers, shape.hidden_dim, shape.num_heads,
             shape.num_kv_heads, shape.seq_len, shape.vocab_size,
             shape.padded_vocab_size, shape.window_pattern),
            (2, 64, 4, 2, 64, 487, 512, "L"),
            f"the demo profile is {shape}")
        # The two batch counts stay at one row: the committed corpus fills one
        # 65-token row per epoch (P2, section 6).
        self.assertEqual(pipeline.BASE_BATCH, 1)
        self.assertEqual(pipeline.EVAL_BATCH, 1)
        pipeline._check(demo_job("demo", ""))

    def test_the_new_fields_survive_the_json_round_trip(self) -> None:
        """A job file carries the eval schedule to another host."""
        job = demo_job("roundtrip", "", eval_batches=2, eval_samples=3)
        rebuilt = pipeline.Job.from_json(job.to_json())
        self.assertEqual(rebuilt.to_dict(), job.to_dict())
        self.assertEqual((rebuilt.eval_batches, rebuilt.eval_samples), (2, 3))


#: The documents of the committed parquet, in file order, as
#: ``src/parquet/reader_test.cc`` (``Expected()``) lists them. They are PLAIN
#: encoded, so every one is a byte run of the committed file; the class asserts
#: that, so a corpus that changes breaks these numbers instead of leaving them
#: as prose. Ten documents, 109 bytes.
CORPUS_DOCUMENTS = (
    b"alpha",
    b"beta",
    b"gamma",
    b"delta",
    b"epsilon",
    b"",
    b"a longer document with spaces",
    b"unicode: caf\xc3\xa9 \xe2\x98\x95",
    b"line one\nline two",
    b"tab\tseparated value",
)


class CorpusBudgetTest(unittest.TestCase):
    """Row 7's open check, settled by measurement and not by prose.

    The check was "confirm that the committed parquet fills 100 base steps at
    the demo shape", on the plan's premise that the fixture holds 9 rows and
    about 530 characters. It holds ten documents and 109 bytes of text
    (``src/parquet/reader_test.cc`` lists them, and this class asserts each one
    is a byte run of the committed file), so one epoch yields at most 119
    tokens: one BOS per document (``src/data.cc``, ``EncodeBatch``) and no BPE
    token shorter than one byte.

    Two facts follow, and they are what the answer is:

    * a *batch* must fit inside one epoch, because the loader returns no
      partial row: ``DataLoader::NextDocuments`` gives up when the documents
      run out mid-row. A demo row is ``seq_len + 1`` = 65 tokens, so a batch of
      two rows needs 130 tokens and cannot form from 119 -- and that is why
      ``pipeline.BASE_BATCH`` is 1. Epoch cycling does not help here: a single
      batch never fits inside one epoch, so the driver's Reset-and-retry
      (``src/mesh_train.cc``, ``LoadBatch``) fails twice and the base stage
      writes no checkpoint.
    * the cursor does cycle epochs, so the *number of steps* is not bounded by
      the rows one epoch holds: ``DataLoader::Reset`` restarts the source and
      ``ResetDocuments`` begins a new epoch. With ``BASE_BATCH`` of 1 one epoch
      fills one step, and 100 steps take 100 epochs.

    The class runs before the chain and independently of it, so a red A1 reads
    as one of these two numbers and not as an unexplained missing checkpoint.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="nanochat_corpus_")
        cls.root = Path(cls.tmp.name)
        cls.saved = {name: os.environ.get(name) for name in OWNED_ENV}
        os.environ["NANOCHAT_BASE_DIR"] = str(cls.root / "cache")
        os.environ.pop("NANOCHAT_RUNS_DIR", None)
        cls.job = demo_job("corpus", cls.root / "runs")
        cls.corpus = pipeline.build_corpus(cls.job)

    @classmethod
    def tearDownClass(cls) -> None:
        for name, value in cls.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        cls.tmp.cleanup()

    def corpus_bytes(self) -> bytes:
        """The committed parquet, as the corpus builder copied it."""
        return Path(self.corpus["parquet"]).read_bytes()

    def epoch_rows(self, seq_len: int) -> int:
        """The rows one epoch of the corpus yields at ``seq_len``.

        One row per ``next()`` call, because the loader is opened with a batch
        of one; ``None`` is the end of the epoch.
        """
        tokenizer = api.Tokenizer.load(self.corpus["tokenizer"])
        loader = api.TokenData(self.corpus["parquet"], tokenizer,
                               seq_len, batch=1, seed=int(self.job.seed))
        rows = 0
        while loader.next() is not None:
            rows += 1
            if rows > 10000:
                self.fail("the loader did not reach the end of an epoch")
        return rows

    def test_row7_the_committed_corpus_is_ten_documents_and_109_bytes(self):
        """The two numbers, against the bytes and not against the plan."""
        raw = self.corpus_bytes()
        for document in CORPUS_DOCUMENTS:
            if not document:
                continue
            self.assertIn(document, raw,
                          f"the committed corpus no longer holds "
                          f"{document!r}")
        self.assertEqual(len(CORPUS_DOCUMENTS), 10)
        self.assertEqual(sum(len(document) for document in CORPUS_DOCUMENTS),
                         109)

    def test_row7_a_two_row_batch_cannot_form(self) -> None:
        """The arithmetic behind ``BASE_BATCH``: 119 tokens, not 130.

        The bound is the document bytes plus one BOS per document, because no
        BPE token is shorter than one byte; the demand is ``BASE_BATCH`` rows
        of ``seq_len + 1`` tokens. Both are asserted, so the reason for the
        batch size stays in the tree and not only in a commit message.
        """
        shape = pipeline.config(self.job)
        documents = len(CORPUS_DOCUMENTS)
        text_bytes = sum(len(document) for document in CORPUS_DOCUMENTS)
        budget = documents + text_bytes
        row_width = int(shape.seq_len) + 1
        self.assertEqual(row_width, 65)
        self.assertEqual(budget, 119)
        self.assertLess(
            budget, 2 * row_width,
            "a two-row batch of 65-token rows would fit one epoch, so the "
            "reason for BASE_BATCH = 1 no longer holds")
        self.assertGreaterEqual(
            budget // row_width, pipeline.BASE_BATCH,
            f"one epoch holds {budget} tokens at most, which is "
            f"{budget // row_width} row(s) of {row_width}, and the base stage "
            f"asks for {pipeline.BASE_BATCH} row(s) per step; the loader "
            f"returns no partial row, so the stage would stop with 'the "
            f"loader returned no batch' and write no checkpoint")

    def test_row7_one_epoch_covers_the_base_batch(self) -> None:
        """The same rule, measured through the loader the driver uses."""
        shape = pipeline.config(self.job)
        rows = self.epoch_rows(int(shape.seq_len))
        self.assertGreaterEqual(
            rows, pipeline.BASE_BATCH,
            f"the committed corpus yields {rows} row(s) of "
            f"{shape.seq_len + 1} tokens per epoch, and the base stage asks "
            f"for {pipeline.BASE_BATCH} row(s) in one step")

    def test_row7_the_cursor_cycles_epochs(self) -> None:
        """100 steps need more epochs than the corpus holds rows for.

        Cycling is what lets the run reach step 100 from a 109-byte corpus;
        it is not a substitute for a batch that fits, which is the test above.
        """
        shape = pipeline.config(self.job)
        rows = self.epoch_rows(int(shape.seq_len))
        needed = int(self.job.base_steps) * pipeline.BASE_BATCH
        self.assertLess(
            rows, needed,
            f"one epoch holds {rows} rows and the schedule needs {needed}, "
            f"so the corpus grew and this check can be retired")


if __name__ == "__main__":
    unittest.main()
