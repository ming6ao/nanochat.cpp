"""T0: the on-demand builder key and the library search order.

The test is hermetic: it never compiles. It checks the build key is stable,
that ``NANOCHAT_CPP_CACHE`` selects the cache, that an explicit
``NANOCHAT_CPP_LIB`` wins, and that ``NANOCHAT_CPP_PREBUILT`` forbids a build.
See docs/python.md sections 3.2 and 3.3.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from nanochat_cpp import _build, _lib


class BuildKeyTest(unittest.TestCase):
    def test_key_is_stable_and_short(self) -> None:
        key = _build.build_key()
        self.assertEqual(key, _build.build_key())
        self.assertEqual(len(key), 16)
        int(key, 16)  # a hex string

    def test_source_hash_is_stable(self) -> None:
        self.assertEqual(_build.source_hash(), _build.source_hash())

    def test_compiler_version_is_a_string(self) -> None:
        self.assertIsInstance(_build.compiler_version(), str)


class CacheTest(unittest.TestCase):
    def test_cache_dir_honors_the_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            previous = os.environ.get("NANOCHAT_CPP_CACHE")
            os.environ["NANOCHAT_CPP_CACHE"] = directory
            try:
                self.assertEqual(_build.cache_dir(), Path(directory))
            finally:
                if previous is None:
                    os.environ.pop("NANOCHAT_CPP_CACHE", None)
                else:
                    os.environ["NANOCHAT_CPP_CACHE"] = previous

    def test_cached_library_uses_the_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _build.cached_library("abc123", Path(directory))
            self.assertEqual(path.name, "libnanochat_shared_abc123.so")
            self.assertEqual(path.parent, Path(directory))


class BuildOptionsTest(unittest.TestCase):
    _NAMES = ("NANOCHAT_CPP_BACKEND", "NANOCHAT_CPP_PRECISION",
              "NANOCHAT_CUDA_ARCH", "NANOCHAT_CPP_CACHE")

    def setUp(self) -> None:
        self._previous = {name: os.environ.get(name) for name in self._NAMES}
        for name in self._NAMES:
            os.environ.pop(name, None)

    def tearDown(self) -> None:
        for name, value in self._previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_from_env_reads_the_environment(self) -> None:
        os.environ["NANOCHAT_CPP_BACKEND"] = "cuda"
        os.environ["NANOCHAT_CPP_PRECISION"] = "fp16"
        os.environ["NANOCHAT_CUDA_ARCH"] = "sm_75"
        options = _build.BuildOptions.from_env()
        self.assertEqual(options.backend, "cuda")
        self.assertEqual(options.precision, "fp16")
        self.assertEqual(options.arch, "sm_75")
        self.assertEqual(options.cache, _build.cache_dir())

    def test_key_changes_with_precision(self) -> None:
        fp32 = _build.BuildOptions.resolve(precision="fp32")
        fp16 = _build.BuildOptions.resolve(precision="fp16")
        self.assertNotEqual(_build.build_key(options=fp32),
                            _build.build_key(options=fp16))

    def test_key_changes_with_arch(self) -> None:
        sm75 = _build.BuildOptions.resolve(arch="sm_75")
        sm80 = _build.BuildOptions.resolve(arch="sm_80")
        self.assertNotEqual(_build.build_key(options=sm75),
                            _build.build_key(options=sm80))

    def test_explicit_options_override_the_environment(self) -> None:
        os.environ["NANOCHAT_CPP_BACKEND"] = "cpu"
        os.environ["NANOCHAT_CPP_PRECISION"] = "fp32"
        os.environ["NANOCHAT_CUDA_ARCH"] = "sm_61"
        options = _build.BuildOptions.resolve(
            backend=" CUDA ", precision="FP16", arch="sm_75")
        self.assertEqual(options.backend, "cuda")
        self.assertEqual(options.precision, "fp16")
        self.assertEqual(options.arch, "sm_75")
        self.assertNotEqual(_build.build_key(options=options),
                            _build.build_key(
                                options=_build.BuildOptions.from_env()))

    def test_cache_override_selects_the_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = _build.BuildOptions.resolve(cache=directory)
            self.assertEqual(options.cache, Path(directory))
            path = _build.cached_library(_build.build_key(options=options),
                                         options.cache)
            self.assertEqual(path.parent, Path(directory))

    def test_cache_override_expands_the_user(self) -> None:
        options = _build.BuildOptions.resolve(cache="~/nanochat-build")
        self.assertEqual(options.cache, Path.home() / "nanochat-build")

    def test_no_argument_key_equals_the_environment_key(self) -> None:
        os.environ["NANOCHAT_CPP_BACKEND"] = "cuda"
        os.environ["NANOCHAT_CPP_PRECISION"] = "fp16"
        os.environ["NANOCHAT_CUDA_ARCH"] = "sm_75"
        self.assertEqual(
            _build.build_key(),
            _build.build_key(options=_build.BuildOptions.from_env()))


class BuildCommandTest(unittest.TestCase):
    def test_cuda_arch_reaches_the_command(self) -> None:
        options = _build.BuildOptions.resolve(
            backend="cuda", precision="fp32", arch="sm_75")
        command = _build._build_command(Path("tools/nanochat"), options)
        self.assertIn("--config=cuda", command)
        self.assertIn("--@rules_cuda//cuda:archs=sm_75", command)

    def test_cuda_fp16_request_raises(self) -> None:
        options = _build.BuildOptions.resolve(
            backend="cuda", precision="fp16", arch="sm_75")
        with self.assertRaises(_build.BuildError):
            _build._build_command(Path("tools/nanochat"), options)

    def test_cpu_fp16_maps_to_the_fp16_config(self) -> None:
        options = _build.BuildOptions.resolve(backend="cpu", precision="fp16")
        command = _build._build_command(Path("tools/nanochat"), options)
        self.assertIn("--config=fp16", command)

    def test_cpu_command_has_no_arch_flag(self) -> None:
        options = _build.BuildOptions.resolve(backend="cpu", arch="sm_75")
        command = _build._build_command(Path("tools/nanochat"), options)
        self.assertIn("--config=cpu", command)
        self.assertNotIn("--@rules_cuda//cuda:archs=sm_75", command)

    def test_no_arch_leaves_the_choice_to_bazel(self) -> None:
        options = _build.BuildOptions.resolve(backend="cuda", arch="")
        command = _build._build_command(Path("tools/nanochat"), options)
        self.assertFalse(
            [part for part in command if part.startswith("--@rules_cuda")])


class SearchOrderTest(unittest.TestCase):
    def _clear(self, *names: str) -> dict:
        previous = {name: os.environ.get(name) for name in names}
        for name in names:
            os.environ.pop(name, None)
        return previous

    def _restore(self, previous: dict) -> None:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_explicit_library_wins(self) -> None:
        names = ("NANOCHAT_CPP_LIB", "NANOCHAT_CPP_CACHE",
                 "NANOCHAT_CPP_PREBUILT")
        previous = self._clear(*names)
        try:
            with tempfile.TemporaryDirectory() as directory:
                library = Path(directory) / _build.LIBRARY_NAME
                library.write_bytes(b"")
                os.environ["NANOCHAT_CPP_LIB"] = str(library)
                os.environ["NANOCHAT_CPP_CACHE"] = directory
                self.assertEqual(_build.ensure_library().resolve(),
                                 library.resolve())
        finally:
            self._restore(previous)

    def test_prebuilt_only_without_a_library_raises(self) -> None:
        names = ("NANOCHAT_CPP_LIB", "NANOCHAT_CPP_CACHE",
                 "NANOCHAT_CPP_PREBUILT")
        previous = self._clear(*names)
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.environ["NANOCHAT_CPP_CACHE"] = directory
                os.environ["NANOCHAT_CPP_PREBUILT"] = "1"
                with self.assertRaises(_build.BuildError):
                    _build.ensure_library()
        finally:
            self._restore(previous)


class BackendTest(unittest.TestCase):
    def test_build_key_changes_with_the_backend(self) -> None:
        self.assertNotEqual(_build.build_key(backend_name="cpu"),
                            _build.build_key(backend_name="cuda"))

    def test_default_backend_is_the_environment(self) -> None:
        previous = os.environ.get("NANOCHAT_CPP_BACKEND")
        try:
            os.environ.pop("NANOCHAT_CPP_BACKEND", None)
            self.assertIsNone(_lib._wanted_backend(None))
            os.environ["NANOCHAT_CPP_BACKEND"] = "cuda"
            self.assertEqual(_lib._wanted_backend(None), "cuda")
        finally:
            if previous is None:
                os.environ.pop("NANOCHAT_CPP_BACKEND", None)
            else:
                os.environ["NANOCHAT_CPP_BACKEND"] = previous

    def test_device_requests_map_to_a_backend(self) -> None:
        self.assertEqual(_lib._wanted_backend("cpu"), "cpu")
        self.assertEqual(_lib._wanted_backend("cuda"), "cuda")
        self.assertEqual(_lib._wanted_backend("cuda:1"), "cuda")
        self.assertEqual(_lib._wanted_backend(2), "cuda")
        self.assertEqual(_lib._wanted_backend("0"), "cuda")
        with self.assertRaises(ValueError):
            _lib._wanted_backend("metal")


if __name__ == "__main__":
    unittest.main()
