"""T0: the one-job chain -- base, then SFT, then RL -- at the demo profile.

The test drives the real chain: :func:`nanochat_cpp.pipeline.run` on
:data:`nanochat_cpp.pipeline.PROFILES` ``demo``, which runs the mesh training
driver, the SFT driver, and the RL worker over the committed fixture corpus. The
runfiles carry the shared library, the two stage executables, the parquet
document set, and the tokenizer artifact; no stage reads the Hub
(``docs/pipeline-plan.md`` D4).

The assertions are the contract of section 7, not the loss numbers: the parity
rungs own the numbers.

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
        cls.job = pipeline.Job(run_id="gate", out_root=str(cls.root / "runs"))
        cls.shape = pipeline.config(cls.job)
        cls.report = pipeline.run(cls.job)
        cls.stages = {stage.name: stage for stage in cls.report.stages}
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
             self.job.top_k), (4, 32, 1.0, 0),
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
        self.assertEqual(delivered.read_bytes(),
                         Path(self.stages["rl"].checkpoint).read_bytes())
        self.assertEqual(delivered.name, "chatrl.ckpt")
        self.assertEqual(delivered.parent.name, "deliver")
        computed = digest(delivered)
        self.assertEqual(computed, self.stages["rl"].sha256)
        self.assertEqual(self.report.stages[-1].sha256, computed)
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
                         ["base", "sft", "rl"])
        self.assertEqual([stage.steps for stage in self.report.stages],
                         [self.job.base_steps, self.job.sft_steps,
                          self.job.rl_steps])
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
                         ["base", "sft", "rl"])
        self.assertEqual(payload["job"]["seed"], self.job.seed)

    # -- A9 ----------------------------------------------------------------

    def test_a9_the_delivered_file_loads_through_the_reader(self) -> None:
        """A container only the pipeline can read is not an artifact."""
        delivered = self.report.delivered
        parameters = checkpoint.read_checkpoint(delivered)
        self.assertTrue(parameters, "the delivered file holds no parameter")
        metadata = checkpoint.read_metadata(delivered)
        self.assertEqual(metadata["format"], checkpoint.CONTAINER_VERSION_1)
        self.assertEqual(metadata["stage"], "rl")
        self.assertEqual(metadata["step"], RL_STEP)
        self.assertEqual(metadata["config"]["num_layers"],
                         self.shape.num_layers)


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
        cls.job = pipeline.Job(run_id="corpus", out_root=str(cls.root / "runs"))
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
