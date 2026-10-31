"""T0: the helpers of the Kaggle session setup.

The test covers the host contract parser, the environment loader, the ``PATH``
builder, and the two setup paths of ``docs/dependency-layering-plan.md``
section 6.5: the installed package, which clones nothing, and the repository
fallback. The test needs no shared library and no network. See
``docs/host-portability.md``, section 8.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import types
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


class EnvironTest(unittest.TestCase):
    """A test case that restores ``os.environ`` after each test."""

    def setUp(self) -> None:
        self.saved = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.saved)


class LoadHostEnvTest(EnvironTest):
    def setUp(self) -> None:
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory(prefix="nanochat_setup_")
        self.path = Path(self.tmp.name) / kaggle_setup.HOST_ENV_NAME

    def tearDown(self) -> None:
        self.tmp.cleanup()
        super().tearDown()

    def test_applies_values(self) -> None:
        self.path.write_text('export NANOCHAT_CPP_BACKEND="cuda"\n',
                             encoding="utf-8")
        values = kaggle_setup.load_host_env(self.path)
        self.assertEqual(values, {"NANOCHAT_CPP_BACKEND": "cuda"})
        self.assertEqual(os.environ["NANOCHAT_CPP_BACKEND"], "cuda")

    def test_missing_file(self) -> None:
        self.assertEqual(kaggle_setup.load_host_env(self.path), {})


class ApplyHostPathTest(EnvironTest):
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


class RevisionTest(unittest.TestCase):
    def test_reads_the_short_revision(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat_rev_") as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.email=t@e",
                            "-c", "user.name=t", "commit", "-q",
                            "--allow-empty", "-m", "c"], check=True)
            expected = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, check=True).stdout.strip()
            self.assertEqual(kaggle_setup.revision(root), expected)
            self.assertEqual(len(expected), 7)


class InstalledTest(EnvironTest):
    """The installed-package path of the session setup (section 6.5)."""

    def setUp(self) -> None:
        super().setUp()
        self.original_importlib = kaggle_setup.importlib
        self.original_version = self.original_importlib.metadata.version
        self.original_build = kaggle_setup.build_library
        self.original_clone = kaggle_setup.ensure_repository
        self.original_bootstrap = kaggle_setup.run_bootstrap

    def tearDown(self) -> None:
        kaggle_setup.importlib = self.original_importlib
        self.original_importlib.metadata.version = self.original_version
        kaggle_setup.build_library = self.original_build
        kaggle_setup.ensure_repository = self.original_clone
        kaggle_setup.run_bootstrap = self.original_bootstrap
        super().tearDown()

    @staticmethod
    def package(location="/opt/site-packages/nanochat_cpp/__init__.py"):
        """An imported module with the one attribute the setup reads."""
        return types.SimpleNamespace(__file__=location)

    def test_a_missing_package_returns_none(self) -> None:
        def import_module(name):
            raise ImportError(name)
        kaggle_setup.importlib = types.SimpleNamespace(
            import_module=import_module,
            metadata=self.original_importlib.metadata)
        self.assertIsNone(kaggle_setup.installed_package())

    def test_the_root_is_the_directory_above_the_package(self) -> None:
        self.assertEqual(kaggle_setup.installed_root(self.package()),
                         Path("/opt/site-packages"))

    def test_an_unknown_version_is_not_guessed(self) -> None:
        metadata = self.original_importlib.metadata

        def missing(name):
            raise metadata.PackageNotFoundError(name)
        metadata.version = missing  # restored by tearDown
        self.assertEqual(kaggle_setup.installed_version(), "")
        self.assertEqual(kaggle_setup.installed_revision(), "")

    def test_a_known_version_names_the_distribution(self) -> None:
        self.original_importlib.metadata.version = lambda name: "0.1.0"
        self.assertEqual(kaggle_setup.installed_revision(),
                         "nanochat-cpp 0.1.0")

    def test_the_installed_package_needs_no_clone(self) -> None:
        kaggle_setup.build_library = lambda **kwargs: (Path("/lib.so"), "key")
        kaggle_setup.ensure_repository = lambda **kwargs: self.fail(
            "the installed package must not clone")
        kaggle_setup.run_bootstrap = lambda root: self.fail(
            "the installed package must not bootstrap")
        host = kaggle_setup.setup(package=self.package())
        self.assertEqual(host.mode, kaggle_setup.MODE_INSTALLED)
        self.assertEqual(host.repo, Path("/opt/site-packages"))
        self.assertEqual(host.library, Path("/lib.so"))
        self.assertEqual(host.build_key, "key")

    def test_the_fallback_still_clones_and_bootstraps(self) -> None:
        bootstrapped = []
        kaggle_setup.build_library = lambda **kwargs: (Path("/lib.so"), "key")
        kaggle_setup.ensure_repository = lambda **kwargs: Path("/clone")
        kaggle_setup.run_bootstrap = bootstrapped.append
        host = kaggle_setup.setup(prefer_installed=False)
        self.assertEqual(host.mode, kaggle_setup.MODE_SOURCE)
        self.assertEqual(host.repo, Path("/clone"))
        self.assertEqual(bootstrapped, [Path("/clone")])


if __name__ == "__main__":
    unittest.main()
