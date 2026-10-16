"""T0: the Python client of the C++ mesh training driver.

The test builds the driver config through the C application binary interface,
runs a two-step job on the CPU backend, and checks that the driver wrote the
record and the summary. It needs the shared library, the committed parquet
document set, and the committed tokenizer artifact
(docs/client-runtime-design.md section 10.1, docs/training-seam.md
section 6.6).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nanochat_cpp import _build, _lib
from nanochat_cpp.api import Config, Tokenizer
from nanochat_cpp.runtime import MeshTrainConfig

TOKENIZER = "tests/data/loader_tokenizer.nctoken"
PARQUET = "src/parquet/testdata/text.parquet"


def resolve(relative: str) -> Path:
    """A runfiles path when the test runs under Bazel, else a source path."""
    found = _lib.find_in_runfiles(relative)
    if found is not None:
        return found
    return _build.repo_root() / relative


class MeshTrainClientTest(unittest.TestCase):
    def config(self) -> Config:
        vocab = int(Tokenizer.load(resolve(TOKENIZER)).vocab_size)
        return Config(num_layers=2, num_heads=2, num_kv_heads=1, hidden_dim=32,
                      seq_len=8, vocab_size=vocab,
                      padded_vocab_size=vocab, window_pattern="SL",
                      value_embedding=False)

    def job(self, out_dir: str, *, steps: int = 2) -> MeshTrainConfig:
        return MeshTrainConfig(
            self.config(), preset="track3", world_size=1,
            tokenizer=resolve(TOKENIZER), train_parquet=resolve(PARQUET),
            per_rank_batch=1, grad_accum=1, steps=steps, seq=8, seed=42,
            data_seed=42, log_every=0, save_every=0, out_dir=out_dir,
            flops_per_token=2.0, peak_flops=1000.0)

    def test_run_writes_record_and_summary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mesh_client_") as tmp:
            config = self.job(tmp)
            try:
                result = config.run()
            finally:
                config.close()
            self.assertEqual(result.steps, 2)
            self.assertFalse(result.interrupted)

            metrics = Path(tmp) / "metrics.jsonl"
            rows = [json.loads(line) for line in
                    metrics.read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["step"], 1)
            self.assertEqual(rows[0]["version"], 1)
            self.assertIn("device_loss", rows[0])
            self.assertEqual(len(rows[0]["device_loss"]), 1)

            summary = json.loads(
                (Path(tmp) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["steps"], 2)
            self.assertEqual(summary["version"], 1)
            self.assertEqual(len(summary["replica_digests"]), 1)

    def test_job_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mesh_client_") as tmp:
            config = self.job(tmp, steps=3)
            try:
                text = config.job_json()
            finally:
                config.close()
            self.assertIn("\"preset\"", text)
            rebuilt = MeshTrainConfig.from_dict(json.loads(text))
            try:
                again = json.loads(rebuilt.job_json())
            finally:
                rebuilt.close()
            self.assertEqual(again["run"]["num_iterations"], 3)
            self.assertEqual(again["run"]["batch"], 1)
            self.assertEqual(again["mesh"]["count"], 1)

    def sft_shard(self, path, *, seq_len: int = 8):
        from nanochat_cpp import sft

        class _FakeTokenizer:
            def get_bos_token_id(self) -> int:
                return 0

        def fake_render(conversation, tokenizer):
            return list(conversation), [1] * len(conversation)

        conversations = [[1, 2, 3]] * 6
        return sft.pack(path, conversations, _FakeTokenizer(), seq_len=seq_len,
                        render=fake_render)

    def test_sft_shard_run(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mesh_client_") as tmp:
            shard = self.sft_shard(Path(tmp) / "sft.nchsft")
            config = MeshTrainConfig(
                self.config(), preset="track3", world_size=1,
                packed_rows_path=str(shard), per_rank_batch=1, grad_accum=1,
                steps=2, seq=8, seed=42, log_every=0, save_every=0,
                out_dir=tmp)
            try:
                result = config.run()
            finally:
                config.close()
            self.assertEqual(result.steps, 2)
            metrics = Path(tmp) / "metrics.jsonl"
            written = [line for line in
                       metrics.read_text(encoding="utf-8").splitlines()
                       if line]
            self.assertEqual(len(written), 2)

    def test_sft_shard_reject_a_mismatched_sequence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mesh_client_") as tmp:
            shard = self.sft_shard(Path(tmp) / "sft.nchsft", seq_len=4)
            with self.assertRaises(RuntimeError):
                MeshTrainConfig(
                    self.config(), preset="track3", world_size=1,
                    packed_rows_path=str(shard), per_rank_batch=1,
                    grad_accum=1, steps=1, seq=8, out_dir=tmp).run()


if __name__ == "__main__":
    unittest.main()
