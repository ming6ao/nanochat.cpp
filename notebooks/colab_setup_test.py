"""T0: the Colab half of the session setup, and its Bazel cache.

A Colab session builds from the source tree, so a build with no cache compiles
every action again after each edit. The cache is one directory with three
sources, in this order: the call's argument, ``NANOCHAT_BAZEL_CACHE``, and
``/content/bazel-cache``. The test needs no card, no clone, and no network: the
two calls that would clone or compile are patched.

See ``docs/build.md``, the Python distribution, and
``docs/host-portability.md`` section 8.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import kaggle_setup
from nanochat_cpp import _build

import colab_setup


class EnvironTest(unittest.TestCase):
    """A test case that restores ``os.environ`` after each test."""

    def setUp(self) -> None:
        self.saved = dict(os.environ)
        os.environ.pop(colab_setup.BAZEL_CACHE_ENV, None)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.saved)


class ResolveCacheTest(EnvironTest):
    def test_the_default_is_under_content(self) -> None:
        # `/content` is the working directory of a Colab session, so the cache
        # lands on the session's own disk and not on a mount that may be slow.
        self.assertEqual(colab_setup.resolve_bazel_cache(),
                         colab_setup.COLAB_BAZEL_CACHE)
        self.assertTrue(colab_setup.COLAB_BAZEL_CACHE.startswith("/content/"))

    def test_the_variable_overrides_the_default(self) -> None:
        # A session that mounts Drive sets the variable once and keeps the
        # cache across sessions, which is the only way it survives a restart.
        os.environ[colab_setup.BAZEL_CACHE_ENV] = \
            "/content/drive/MyDrive/nanochat-bazel-cache"
        self.assertEqual(
            colab_setup.resolve_bazel_cache(),
            "/content/drive/MyDrive/nanochat-bazel-cache")

    def test_the_call_overrides_the_variable(self) -> None:
        os.environ[colab_setup.BAZEL_CACHE_ENV] = "/from-the-variable"
        self.assertEqual(colab_setup.resolve_bazel_cache("/from-the-call"),
                         "/from-the-call")

    def test_an_empty_value_is_the_default(self) -> None:
        # An exported empty variable is what a notebook cell leaves behind when
        # it clears one, and it must not become the empty path.
        os.environ[colab_setup.BAZEL_CACHE_ENV] = "   "
        self.assertEqual(colab_setup.resolve_bazel_cache(),
                         colab_setup.COLAB_BAZEL_CACHE)

    def test_the_variable_name_is_the_builders_name(self) -> None:
        # One name, so a rename in the builder cannot leave this module
        # pointing at a variable that nothing reads.
        self.assertEqual(colab_setup.BAZEL_CACHE_ENV, _build.BAZEL_CACHE_ENV)


class StagesTest(EnvironTest):
    """The second call passes the cache to the one builder."""

    def options(self, **kwargs):
        with mock.patch.object(_build, "build_stages") as build:
            colab_setup.stages(arch="sm_75", **kwargs)
        return build.call_args.kwargs["options"]

    def test_the_default_cache_reaches_the_builder(self) -> None:
        self.assertEqual(self.options().bazel_cache,
                         colab_setup.COLAB_BAZEL_CACHE)

    def test_the_call_names_the_cache(self) -> None:
        self.assertEqual(self.options(bazel_cache="/tmp/cache").bazel_cache,
                         "/tmp/cache")

    def test_the_variable_reaches_the_builder(self) -> None:
        os.environ[colab_setup.BAZEL_CACHE_ENV] = "/content/drive/cache"
        self.assertEqual(self.options().bazel_cache, "/content/drive/cache")


class SetupTest(EnvironTest):
    """The first call exports the cache for every later build."""

    def test_setup_exports_the_cache_before_the_build(self) -> None:
        with mock.patch.object(kaggle_setup, "setup") as setup:
            colab_setup.setup(arch="sm_75")
        self.assertTrue(setup.called)
        self.assertEqual(os.environ[colab_setup.BAZEL_CACHE_ENV],
                         colab_setup.COLAB_BAZEL_CACHE)

    def test_setup_keeps_a_value_the_session_already_set(self) -> None:
        # `setdefault` and not an assignment: a cell that named a Drive cache
        # must not be overwritten by the default.
        os.environ[colab_setup.BAZEL_CACHE_ENV] = "/content/drive/cache"
        with mock.patch.object(kaggle_setup, "setup"):
            colab_setup.setup(arch="sm_75")
        self.assertEqual(os.environ[colab_setup.BAZEL_CACHE_ENV],
                         "/content/drive/cache")


class ArchTest(unittest.TestCase):
    def test_a_t4_capability_is_sm_75(self) -> None:
        self.assertEqual(colab_setup.arch_from_capability("7.5\n"), "sm_75")

    def test_a_value_that_is_not_a_pair_raises(self) -> None:
        with self.assertRaises(ValueError):
            colab_setup.arch_from_capability("7")


if __name__ == "__main__":
    unittest.main()
