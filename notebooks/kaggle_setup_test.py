"""T0: the pure helpers of the Kaggle session setup.

The test covers the host contract parser, the environment loader, and the
``PATH`` builder. The test needs no shared library and no network. See
``docs/host-portability.md``, section 8.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import kaggle_setup


class ParseEnvFileTest(unittest.TestCase):
    def test_quoted_and_plain(self) -> None:
        text = (
            "# a comment\n"
            "\n"
            "export A=\"one\"\n"
            "export B='two'\n"
            "export C=three\n"
            "not an export\n"
            "export D=\n"
        )
        self.assertEqual(kaggle_setup.parse_env_file(text),
                         {"A": "one", "B": "two", "C": "three", "D": ""})

    def test_empty_text(self) -> None:
        self.assertEqual(kaggle_setup.parse_env_file(""), {})

    def test_value_with_equals(self) -> None:
        self.assertEqual(kaggle_setup.parse_env_file("export K=a=b\n"),
                         {"K": "a=b"})


class LoadHostEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nanochat_setup_")
        self.path = Path(self.tmp.name) / kaggle_setup.HOST_ENV_NAME
        self.saved = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.saved)
        self.tmp.cleanup()

    def test_applies_values(self) -> None:
        self.path.write_text('export NANOCHAT_CPP_BACKEND="cuda"\n',
                             encoding="utf-8")
        values = kaggle_setup.load_host_env(self.path)
        self.assertEqual(values, {"NANOCHAT_CPP_BACKEND": "cuda"})
        self.assertEqual(os.environ["NANOCHAT_CPP_BACKEND"], "cuda")

    def test_missing_file(self) -> None:
        self.assertEqual(kaggle_setup.load_host_env(self.path), {})


class ApplyHostPathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.saved = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.saved)

    def test_prepends_directories(self) -> None:
        os.environ["PATH"] = "/usr/bin"
        os.environ["CUDA_HOME"] = "/usr/local/cuda"
        entries = kaggle_setup.apply_host_path(home="/home/kaggle")
        self.assertEqual(entries, ["/home/kaggle/.local/bin",
                                   "/home/kaggle/.local/node/bin",
                                   "/usr/local/cuda/bin"])
        self.assertTrue(os.environ["PATH"].startswith("/home/kaggle/.local/bin"))
        self.assertIn("/usr/bin", os.environ["PATH"])

    def test_without_cuda_home(self) -> None:
        os.environ.pop("CUDA_HOME", None)
        entries = kaggle_setup.apply_host_path(home="/home/kaggle")
        self.assertEqual(entries, ["/home/kaggle/.local/bin",
                                   "/home/kaggle/.local/node/bin"])
