"""T0: the RL bridge renders prompts, speaks the pipe protocol, and reports.

Hermetic: a fake tokenizer, a fake task, and an executable Python stand-in for
the worker replace the C++ library, so the test needs no shared library, no
Bazel, and no GPU. It checks the renderer, the advantage, the worker argv, the
direct start, and the ``tools/nanochat train`` start. See
``docs/rl-notebook.md`` sections 3 and 4 and ``docs/post-training.md``
section 5.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from nanochat_cpp import _entry, rl, toolchain
from nanochat_cpp.api import Config

# A stand-in for the persistent worker. It speaks the protocol of
# `src/rl_worker.h` and records every request it receives.
FAKE_WORKER = '''#!{python}
import os
import sys

log_path = os.environ.get("FAKE_WORKER_LOG")
log = open(log_path, "a", encoding="utf-8") if log_path else None


def emit(line):
    sys.stdout.write(line + "\\n")
    sys.stdout.flush()


def record(line):
    if log is not None:
        log.write(line + "\\n")
        log.flush()


step = 0
for raw in sys.stdin:
    line = raw.rstrip("\\n")
    record(line)
    parts = line.split(" ")
    kind = parts[0]
    fields = {{}}
    for token in parts[1:]:
        key, _, value = token.partition("=")
        fields[key] = value
    if kind == "rollout":
        num_prompts = int(fields["num_prompts"])
        prompt_len = int(fields["prompt_len"])
        num_samples = int(fields["num_samples"])
        prompts = [int(value) for value in fields["prompts"].split(",")]
        lengths, masks, tokens = [], [], []
        rows = num_prompts * num_samples
        for row in range(rows):
            start = (row // num_samples) * prompt_len
            prompt = prompts[start:start + prompt_len]
            generated = [7, 8] if row % 2 == 0 else [9]
            lengths.append(prompt_len + len(generated))
            masks.extend([0] * prompt_len + [1] * len(generated))
            tokens.extend(prompt + generated)
        emit("rollout ok=1 num_prompts=%d num_samples=%d rows=%d lengths=%s "
             "masks=%s tokens=%s" % (
                 num_prompts, num_samples, rows,
                 ",".join(str(value) for value in lengths),
                 ",".join(str(value) for value in masks),
                 ",".join(str(value) for value in tokens)))
    elif kind == "advantage":
        step += 1
        targets = [int(value) for value in fields["targets"].split(",")]
        valid = sum(1 for value in targets if value != -1)
        emit("advantage ok=1 step=%d loss=%.9g grad_norm=%.9g valid_targets=%d"
             % (step, 0.5 * step, 1.5 * step, valid))
    elif kind == "save":
        with open(fields["path"], "w", encoding="utf-8") as handle:
            handle.write("checkpoint")
        emit("save ok=1")
    elif kind == "ping":
        emit("pong ok=1")
    elif kind == "quit":
        emit("quit ok=1")
        break
    else:
        emit(kind + " ok=0 error=unknown_message")
'''

# A stand-in for `tools/nanochat`: with a `--`, it execs the command after it;
# otherwise it echoes its argument list as JSON (the toolchain argv check).
FAKE_ENTRY = '''#!{python}
import json
import os
import sys

argv = sys.argv[1:]
if "--" in argv:
    index = argv.index("--")
    target = argv[index + 1]
    if os.path.isfile(target):
        os.execv(target, argv[index + 1:])
print(json.dumps(argv))
'''

CONVERSATION = {"messages": [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "answer"}]}


class _FakeTokenizer:
    """A stdlib-only stand-in for the ``NCTOKEN1`` tokenizer."""

    special_tokens = {
        "<|bos|>": 1,
        "<|user_start|>": 2,
        "<|user_end|>": 3,
        "<|assistant_start|>": 4,
        "<|assistant_end|>": 5,
        "<|python_start|>": 6,
        "<|python_end|>": 7,
        "<|output_start|>": 8,
        "<|output_end|>": 9,
    }

    def encode(self, text):
        return [10 + (ord(char) % 20) for char in text]

    def decode(self, ids):
        table = {7: "a", 8: "b", 9: "c"}
        return "".join(table.get(int(value), "?") for value in ids)

    def encode_special(self, name):
        return self.special_tokens.get(name, -1)


class _Reply:
    """A task whose reward is 1.0 for the even rows and 0.0 for the odd rows."""

    def reward(self, conversation, completion):
        return 1.0 if completion == "ab" else 0.0

    def evaluate(self, conversation, completion):
        return self.reward(conversation, completion)


def _config() -> Config:
    return Config(num_layers=2, num_heads=2, num_kv_heads=1, hidden_dim=32,
                  seq_len=16, vocab_size=64, padded_vocab_size=64,
                  window_pattern="SL")


class RlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nanochat_rl_")
        root = Path(self.tmp.name)
        self.worker = root / "fake_worker"
        self.worker.write_text(FAKE_WORKER.format(python=sys.executable),
                               encoding="utf-8")
        self.worker.chmod(0o755)
        self.entry = root / "fake-nanochat"
        self.entry.write_text(FAKE_ENTRY.format(python=sys.executable),
                              encoding="utf-8")
        self.entry.chmod(0o755)
        self.log = root / "protocol.log"
        self.env = dict(os.environ)
        self.env["NANOCHAT_SANDBOX_BACKEND"] = "none"
        self.env["NANOCHAT_RL_WORKER"] = str(self.worker)
        self.env["FAKE_WORKER_LOG"] = str(self.log)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _toolchain(self, env=None):
        return toolchain.Toolchain(entry=self.entry, repo_root=self.tmp.name,
                                   env=self.env if env is None else env)

    def test_advantages(self) -> None:
        self.assertEqual(rl.advantages([1.0, 0.0]), [0.5, -0.5])
        self.assertEqual(rl.advantages([2.0]), [0.0])
        self.assertEqual(rl.advantages([]), [])
        self.assertAlmostEqual(sum(rl.advantages([0.2, 0.4, 1.0])), 0.0)

    def test_render_prompts_drops_the_reference_answer(self) -> None:
        prompts = rl.render_prompts([CONVERSATION], _FakeTokenizer())
        self.assertEqual(len(prompts), 1)
        ids = prompts[0]
        self.assertEqual(ids[0], 1)  # <|bos|>
        self.assertEqual(ids[-1], 4)  # <|assistant_start|>
        self.assertNotIn(5, ids)  # the assistant answer is dropped

    def test_render_prompts_appends_the_marker(self) -> None:
        prompt = {"messages": [{"role": "user", "content": "hi"}]}
        ids = rl.render_prompts([prompt], _FakeTokenizer())[0]
        self.assertEqual(ids[0], 1)
        self.assertEqual(ids[-1], 4)

    def test_worker_argv(self) -> None:
        argv = rl.worker_argv("/worker", _config(), "base.nchkpt01")
        self.assertEqual(argv[0], "/worker")
        self.assertEqual(argv[argv.index("--layers") + 1], "2")
        self.assertEqual(argv[argv.index("--heads") + 1], "2")
        self.assertEqual(argv[argv.index("--kv-heads") + 1], "1")
        self.assertEqual(argv[argv.index("--hidden") + 1], "32")
        self.assertEqual(argv[argv.index("--seq") + 1], "16")
        self.assertEqual(argv[argv.index("--padded-vocab") + 1], "64")
        self.assertEqual(argv[argv.index("--window-pattern") + 1], "SL")
        self.assertEqual(argv[argv.index("--checkpoint") + 1],
                         "base.nchkpt01")
        # No load path means no --checkpoint flag.
        self.assertNotIn("--checkpoint", rl.worker_argv("/worker", _config()))

    def test_run_direct_start(self) -> None:
        checkpoint = Path(self.tmp.name) / "out.nchkpt01"
        report = rl.run(
            _Reply(), [CONVERSATION, CONVERSATION], num_samples=2, steps=2,
            checkpoint=str(checkpoint), checkpoint_interval=1,
            config=_config(), tokenizer=_FakeTokenizer(), worker=self.worker,
            env=self.env, seed=0)
        self.assertEqual(report.steps, 2)
        self.assertEqual(report.num_prompts, 2)
        self.assertEqual(report.num_samples, 2)
        self.assertEqual(report.losses, (0.5, 1.0))
        self.assertEqual(report.grad_norms, (1.5, 3.0))
        self.assertEqual(report.mean_rewards, (0.5, 0.5))
        self.assertEqual(report.loss, 1.0)
        self.assertEqual(report.grad_norm, 3.0)
        self.assertTrue(checkpoint.is_file())

    def test_run_through_tools_nanochat(self) -> None:
        # Not the `none` backend: the bridge runs `tools/nanochat train --`.
        env = dict(self.env)
        env.pop("NANOCHAT_SANDBOX_BACKEND", None)
        report = rl.run(
            _Reply(), [CONVERSATION], num_samples=2, steps=1,
            config=_config(), tokenizer=_FakeTokenizer(), worker=self.worker,
            toolchain=self._toolchain(), env=env, seed=0)
        self.assertEqual(report.losses, (0.5,))
        self.assertEqual(report.grad_norms, (1.5,))

    def test_pipe_protocol_requests(self) -> None:
        rl.run(_Reply(), [CONVERSATION, CONVERSATION], num_samples=2, steps=1,
               config=_config(), tokenizer=_FakeTokenizer(), worker=self.worker,
               env=self.env, seed=0)
        lines = self.log.read_text(encoding="utf-8").strip().splitlines()
        kinds = [line.split(" ")[0] for line in lines]
        self.assertEqual(kinds[:2], ["rollout", "advantage"])
        self.assertIn("quit", kinds)
        rollout = dict(
            token.partition("=")[::2] for token in lines[0].split(" ")[1:])
        self.assertEqual(rollout["num_prompts"], "2")
        self.assertEqual(rollout["num_samples"], "2")
        self.assertEqual(rollout["max_tokens"], "256")
        self.assertEqual(rollout["stop_id"], "5")
        self.assertEqual(rollout["stop_ids"], "5,5")
        advantage = dict(
            token.partition("=")[::2] for token in lines[1].split(" ")[1:])
        self.assertEqual(advantage["batch"], "4")
        self.assertEqual(advantage["seq"], "8")
        self.assertEqual(advantage["num_passes"], "1")
        self.assertEqual(advantage["examples_per_rank"], "2")
        targets = [int(value) for value in advantage["targets"].split(",")]
        # The even rows sample two ids and the odd rows one, for two prompts.
        self.assertEqual(sum(1 for value in targets if value != -1), 6)
        weights = [float(value) for value in advantage["advantages"].split(",")]
        self.assertEqual(weights.count(0.5), 4)
        self.assertEqual(weights.count(-0.5), 2)
        self.assertEqual(weights.count(0.0), len(weights) - 6)

    def test_run_rejects_empty_prompts(self) -> None:
        with self.assertRaises(rl.RlError):
            rl.run(_Reply(), [], steps=1, tokenizer=_FakeTokenizer(),
                   worker=self.worker, env=self.env)

    def test_run_rejects_bad_shapes(self) -> None:
        with self.assertRaises(rl.RlError):
            rl.run(_Reply(), [CONVERSATION], steps=0,
                   tokenizer=_FakeTokenizer(), worker=self.worker, env=self.env)
        with self.assertRaises(rl.RlError):
            rl.run(_Reply(), [CONVERSATION], num_samples=0,
                   tokenizer=_FakeTokenizer(), worker=self.worker, env=self.env)

    def test_entry_and_toolchain_alias(self) -> None:
        self.assertIn("rl", _entry.names())
        self.assertEqual(self._toolchain().commands(), _entry.names())
        # The `rl` alias runs the RL worker through the `train` profile, the
        # route of docs/rl-notebook.md section 6.
        result = self._toolchain().rl(["worker"])
        self.assertEqual(json.loads(result.stdout), ["train", "--", "worker"])


if __name__ == "__main__":
    unittest.main()
