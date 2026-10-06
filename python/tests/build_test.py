"""T0: the on-demand builder key and the library search order.

The test is hermetic: it never compiles. It checks the build key is stable,
that ``NANOCHAT_CPP_CACHE`` selects the cache, that an explicit
``NANOCHAT_CPP_LIB`` wins, and that ``NANOCHAT_CPP_PREBUILT`` forbids a build.
See docs/python-api.md sections 5.1 and 5.5.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from nanochat_cpp import _build


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


if __name__ == "__main__":
    unittest.main()
