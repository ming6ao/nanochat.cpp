"""T0: the SFT bridge builds the driver argv the driver reads.

Hermetic: a fake toolchain records the argument list instead of running the
driver, so the test needs no shared library, no Bazel, no GPU, and no tokenizer
artifact. It pins the three facts of ``docs/pipeline-plan.md`` B1 and B3 that the
SFT stage of the chain depends on:

* ``config=`` reaches the driver as :func:`nanochat_cpp.api.model_flags`, the one
  home of the model flag list, with ``--value-embedding`` among the flags, so an
  SFT stage rebuilds the shape its warm start records and
  ``Checkpointer::LoadModel`` accepts that file;
* ``warm_start`` arrives as ``--warm-start`` and ``checkpoint`` stays the
  ``--checkpoint`` save target, because ``src/sft_main.cc`` loads only from
  ``--warm-start`` and would overwrite the base checkpoint otherwise;
* a ``config`` whose ``seq_len`` disagrees with the shard raises, because the
  packed rows own the sequence length.

The facade had no test before this file (section 3, B1).
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nanochat_cpp import _build, api, sft

#: The demo shape of design decision D5, so the flag values are the ones the
#: chain sends and not a second invented config.
DEMO = api.Config(num_layers=2, num_heads=4, num_kv_heads=2, hidden_dim=64,
                  seq_len=64, vocab_size=487, padded_vocab_size=512,
                  window_pattern="L")


@dataclasses.dataclass
class _Result:
    """The one attribute :func:`nanochat_cpp.sft.run` reads from a result."""

    returncode: int


class _FakeToolchain:
    """A toolchain that records the argv instead of running a child."""

    def __init__(self, status: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.status = status

    def train(self, argv=None) -> _Result:
        self.calls.append([str(value) for value in (argv or [])])
        return _Result(self.status)

    @property
    def argv(self) -> list[str]:
        """The argv of the one call the facade made."""
        if len(self.calls) != 1:
            raise AssertionError(
                f"the facade made {len(self.calls)} calls, not one")
        return self.calls[0]


def value_of(argv: list[str], flag: str) -> str:
    """The value of ``flag`` in ``argv``, or an error when it is absent."""
    if flag not in argv:
        raise AssertionError(f"{flag} is absent from {argv}")
    return argv[argv.index(flag) + 1]


class SftRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nanochat_sft_")
        self.root = Path(self.tmp.name)
        self.shard = self.root / "rows.nchsft"
        self.shard.write_bytes(b"")
        self.base = self.root / "base.nchkpt01"
        self.base.write_bytes(b"")
        self.target = self.root / "sft.nchkpt01"
        # A host that exports the stage override must not decide what the
        # default-binary case resolves.
        self._saved = os.environ.pop(_build.stage_override("sft_main"), None)

    def tearDown(self) -> None:
        if self._saved is not None:
            os.environ[_build.stage_override("sft_main")] = self._saved
        self.tmp.cleanup()

    def run_driver(self, tool: _FakeToolchain, **kwargs) -> int:
        return sft.run(self.shard, self.target, binary="/fake/sft_main",
                       toolchain=tool, **kwargs)

    def test_config_reaches_the_driver_as_the_model_flags(self) -> None:
        """The shape travels as ``api.model_flags``, the one flag list (B1)."""
        tool = _FakeToolchain()
        self.run_driver(tool, config=DEMO, seq_len=DEMO.seq_len)
        flags = api.model_flags(DEMO)
        self.assertTrue(flags)
        for index in range(0, len(flags), 2):
            self.assertEqual(value_of(tool.argv, flags[index]),
                             flags[index + 1],
                             f"{flags[index]} did not reach the driver")
        # The flag B2 added is among them, so a stage command line reproduces a
        # base run with value embeddings on or off.
        self.assertEqual(value_of(tool.argv, "--value-embedding"), "1")
        self.assertEqual(value_of(tool.argv, "--layers"), "2")
        self.assertEqual(value_of(tool.argv, "--window-pattern"), "L")

    def test_no_config_sends_no_model_flag(self) -> None:
        """A caller that owns no shape keeps the driver's own defaults."""
        tool = _FakeToolchain()
        self.run_driver(tool)
        self.assertEqual(api.model_flags(None), [])
        for flag in ("--layers", "--heads", "--kv-heads", "--hidden", "--seq",
                     "--vocab", "--padded-vocab", "--window-pattern",
                     "--rope-base", "--value-embedding"):
            self.assertNotIn(flag, tool.argv)

    def test_warm_start_and_checkpoint_are_two_arguments(self) -> None:
        """``warm_start`` loads and ``checkpoint`` saves, as separate flags.

        ``src/sft_main.cc`` loads only from ``--warm-start``, so a facade that
        sent the warm start as ``--checkpoint`` would overwrite the base
        checkpoint of the chain (B3).
        """
        tool = _FakeToolchain()
        self.run_driver(tool, config=DEMO, seq_len=DEMO.seq_len,
                        warm_start=str(self.base))
        self.assertEqual(value_of(tool.argv, "--warm-start"), str(self.base))
        self.assertEqual(value_of(tool.argv, "--checkpoint"), str(self.target))
        self.assertEqual(tool.argv.count("--checkpoint"), 1)
        self.assertEqual(tool.argv.count("--warm-start"), 1)

    def test_resume_and_warm_start_are_forwarded_as_named(self) -> None:
        """A resume is its own flag, so the driver can reject both at once."""
        tool = _FakeToolchain()
        self.run_driver(tool, resume=str(self.target))
        self.assertEqual(value_of(tool.argv, "--resume"), str(self.target))
        self.assertNotIn("--warm-start", tool.argv)

    def test_config_and_seq_len_disagreement_raises(self) -> None:
        """The shard owns the sequence length, so a mismatch is an error (B1)."""
        tool = _FakeToolchain()
        with self.assertRaises(ValueError) as caught:
            self.run_driver(tool, config=DEMO, seq_len=DEMO.seq_len + 1)
        self.assertIn("sequence length", str(caught.exception))
        self.assertEqual(tool.calls, [])

    def test_seq_len_alone_sends_one_seq_flag(self) -> None:
        """Without a config the shard width is the only source of ``--seq``."""
        tool = _FakeToolchain()
        self.run_driver(tool, seq_len=32)
        self.assertEqual(value_of(tool.argv, "--seq"), "32")
        self.assertEqual(tool.argv.count("--seq"), 1)

    def test_config_owns_the_seq_flag(self) -> None:
        """A config already carries ``--seq``, so a repeat is not sent."""
        tool = _FakeToolchain()
        self.run_driver(tool, config=DEMO, seq_len=DEMO.seq_len)
        self.assertEqual(tool.argv.count("--seq"), 1)
        self.assertEqual(value_of(tool.argv, "--seq"), "64")

    def test_the_driver_flags_reach_the_driver(self) -> None:
        """The schedule, the seed, and the batch are forwarded as flags."""
        tool = _FakeToolchain()
        self.run_driver(tool, num_iterations=7, batch=3, seed=11,
                        device="cpu", val_rows="/data/val.nchsft",
                        eval_every=5, eval_tokens=64, chatcore_every=9)
        self.assertEqual(value_of(tool.argv, "--rows"), str(self.shard))
        self.assertEqual(value_of(tool.argv, "--num-iterations"), "7")
        self.assertEqual(value_of(tool.argv, "--batch"), "3")
        self.assertEqual(value_of(tool.argv, "--seed"), "11")
        self.assertEqual(value_of(tool.argv, "--device"), "cpu")
        self.assertEqual(value_of(tool.argv, "--val-rows"),
                         "/data/val.nchsft")
        self.assertEqual(value_of(tool.argv, "--eval-every"), "5")
        self.assertEqual(value_of(tool.argv, "--eval-tokens"), "64")
        self.assertEqual(value_of(tool.argv, "--chatcore-every"), "9")

    def test_the_default_binary_is_the_stage_resolver(self) -> None:
        """``binary=None`` resolves ``sft_main`` through B6's one resolver."""
        override = self.root / "sft_main"
        override.write_bytes(b"")
        tool = _FakeToolchain()
        override_env = {_build.stage_override("sft_main"): str(override)}
        with mock.patch.dict(os.environ, override_env):
            status = sft.run(self.shard, self.target, toolchain=tool)
        self.assertEqual(status, 0)
        self.assertEqual(tool.argv[0], str(override))

    def test_the_status_is_returned(self) -> None:
        """The facade returns the driver's exit status and raises nothing."""
        tool = _FakeToolchain(status=3)
        self.assertEqual(self.run_driver(tool), 3)


if __name__ == "__main__":
    unittest.main()
