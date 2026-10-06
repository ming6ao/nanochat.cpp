"""T0: the in-process API reproduces the committed C++ fixture.

The fixture ``//tests:api_fixture`` records the per-step loss of a tiny
``train_main`` run and the bits-per-byte result of ``eval_main``. The test
drives ``Trainer`` and ``Evaluator`` through the same parquet document set and
the same ``NCTOKEN1`` artifact, then compares the results. See
docs/python-api.md section 11.
"""

from __future__ import annotations

import array
import sys
import unittest
from pathlib import Path

from nanochat_cpp import (
    Config,
    Evaluator,
    Model,
    Optimizer,
    TokenData,
    Tokenizer,
    Trainer,
    evaluate,
    no_grad,
)
from nanochat_cpp import _build, _lib
from nanochat_cpp.eval_fixture import read as read_fixture

FIXTURE = "tests/data/api_fixture.bin"
TOKENIZER = "tests/data/loader_tokenizer.nctoken"
PARQUET = "src/parquet/testdata/text.parquet"


def resolve(relative: str) -> Path:
    """A runfiles path when the test runs under Bazel, else a source path."""
    found = _lib.find_in_runfiles(relative)
    if found is not None:
        return found
    return _build.repo_root() / relative


def _int(records: dict, name: str) -> int:
    return int.from_bytes(records[name].data, "little", signed=True)


def _text(records: dict, name: str) -> str:
    return records[name].data.decode("utf-8")


def _floats(records: dict, name: str) -> list:
    buffer = array.array("f")
    buffer.frombytes(records[name].data)
    if sys.byteorder != "little":
        buffer.byteswap()
    return buffer.tolist()


def _float(records: dict, name: str) -> float:
    return float(_floats(records, name)[0])


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = {record.name: record for record in
                       read_fixture(resolve(FIXTURE))}

    def config(self) -> Config:
        records = self.records
        return Config(
            num_layers=_int(records, "api/config/layers"),
            num_heads=_int(records, "api/config/heads"),
            num_kv_heads=_int(records, "api/config/kv_heads"),
            hidden_dim=_int(records, "api/config/hidden"),
            seq_len=_int(records, "api/config/seq"),
            vocab_size=_int(records, "api/config/vocab"),
            padded_vocab_size=_int(records, "api/config/padded_vocab"),
            window_pattern=_text(records, "api/config/window_pattern"),
        )

    def optimizer_kwargs(self) -> dict:
        records = self.records
        return {
            "unembedding_lr": _float(records, "api/config/opt/unembedding_lr"),
            "embedding_lr": _float(records, "api/config/opt/embedding_lr"),
            "matrix_lr": _float(records, "api/config/opt/matrix_lr"),
            "scalar_lr": _float(records, "api/config/opt/scalar_lr"),
            "weight_decay": _float(records, "api/config/opt/weight_decay"),
            "clip": _float(records, "api/config/opt/clip"),
            "adam_eps": _float(records, "api/config/opt/adam_eps"),
            "muon_beta2": _float(records, "api/config/opt/muon_beta2"),
            "muon_ns_steps": _int(records, "api/config/opt/muon_ns_steps"),
            "num_iterations": _int(records, "api/config/steps"),
            "warmup_steps": _int(records, "api/config/sched/warmup_steps"),
            "warmdown_ratio": _float(records,
                                     "api/config/sched/warmdown_ratio"),
            "final_lr_frac": _float(records,
                                    "api/config/sched/final_lr_frac"),
            "weight_decay_base": _float(records,
                                        "api/config/sched/weight_decay_base"),
            "muon_momentum_warmup_steps": _float(
                records, "api/config/sched/muon_momentum_warmup_steps"),
            "muon_momentum_start": _float(
                records, "api/config/sched/muon_momentum_start"),
            "muon_momentum_peak": _float(
                records, "api/config/sched/muon_momentum_peak"),
            "muon_momentum_final": _float(
                records, "api/config/sched/muon_momentum_final"),
        }

    def token_data(self, seed_offset: int = 0) -> TokenData:
        records = self.records
        tokenizer = Tokenizer.load(resolve(TOKENIZER))
        self.tokenizer = tokenizer
        return TokenData(
            parquet=str(resolve(PARQUET)),
            tokenizer=tokenizer,
            seq_len=_int(records, "api/config/seq"),
            batch=_int(records, "api/config/batch"),
            seed=_int(records, "api/seed") + seed_offset,
            threads=_int(records, "api/config/tokenizer_threads"),
            document_buffer=_int(records, "api/config/document_buffer"),
        )

    def test_config_derives_from_depth(self) -> None:
        config = Config(depth=8, seq_len=512, vocab_size=1000)
        self.assertEqual(config.num_layers, 8)
        self.assertEqual(config.hidden_dim, 512)
        self.assertEqual(config.num_heads, 4)
        self.assertEqual(config.padded_vocab_size, 1024)

    def test_tokenizer_round_trip(self) -> None:
        tokenizer = Tokenizer.load(resolve(TOKENIZER))
        text = "The capital of France is"
        ids = tokenizer.encode(text)
        self.assertTrue(ids)
        self.assertEqual(tokenizer.decode(ids), text)
        self.assertEqual(tokenizer.vocab_size,
                         _int(self.records, "api/config/vocab"))

    def test_trainer_and_evaluator_match_fixture(self) -> None:
        records = self.records
        seed = _int(records, "api/seed")
        steps = _int(records, "api/config/steps")

        model = Model(self.config(), seed=seed)
        data = self.token_data()
        optimizer = Optimizer(model, **self.optimizer_kwargs())
        trainer = Trainer(model, data, num_iterations=steps,
                          optimizer=optimizer)
        losses = [loss for _, loss in trainer]
        expected = _floats(records, "api/train/loss")
        self.assertEqual(len(losses), len(expected))
        for got, want in zip(losses, expected):
            self.assertAlmostEqual(got, want, places=4)

        evaluator = Evaluator(model)
        validation = self.token_data(seed_offset=1)
        bpb = evaluator.bpb(validation, _int(records, "api/eval/steps"))
        self.assertAlmostEqual(bpb, _float(records, "api/eval/val/bpb"),
                               places=4)

    def test_no_grad_context_manager(self) -> None:
        model = Model(self.config(), seed=0)
        with no_grad(model):
            pass

    def test_evaluate_returns_a_report(self) -> None:
        records = self.records
        tokenizer = Tokenizer.load(resolve(TOKENIZER))
        self.tokenizer = tokenizer
        model = Model(self.config(), seed=_int(records, "api/seed"))
        report = evaluate(model, parquet=str(resolve(PARQUET)),
                          tokenizer=tokenizer, steps=1,
                          batch=_int(records, "api/config/batch"),
                          seed=_int(records, "api/seed"))
        self.assertGreater(report.bpb, 0.0)


if __name__ == "__main__":
    unittest.main()
