"""T0: the on-demand builder key, the library search order, and the two
stage executables of the chain.

The test is hermetic: it never compiles. It checks the build key is stable,
that ``NANOCHAT_CPP_CACHE`` selects the cache, that an explicit
``NANOCHAT_CPP_LIB`` wins, and that ``NANOCHAT_CPP_PREBUILT`` forbids a build.
See docs/python.md sections 3.2 and 3.3.

The stage half is the resolver of ``docs/pipeline-plan.md`` B6: one function
finds ``sft_main`` and ``rl_worker`` in the order the library uses -- the
environment override, the runfiles tree, ``bazel-bin/src/<name>``, and the
``nanochat-<name>`` entry point of a future wheel -- and one builder builds the
library and both children with one ``tools/nanochat build`` line. Both are
covered over a fake tree: a directory, a runfiles root, a fake
``tools/nanochat`` script, and a temporary ``PATH``, so no Bazel run happens.

The last class is row 1 of ``docs/pipeline-scaleup-plan.md``: the named
``hopper`` config of ``.bazelrc``, asserted beside the ``h100`` stanza so the
hand-run Hopper build and the H100 *simulator* profile cannot be confused
again, and the explicit flag triple the build helper emits instead of either
name.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nanochat_cpp import _build, _lib


def tree_file(relative: str) -> Path:
    """One tracked file of the tree, from the runfiles or from the checkout.

    A ``py_test`` reads its data through the runfiles tree, and
    ``//python:build_test`` declares none: the ``.bazelrc`` it asserts on is
    read from the checkout that the runfiles symlinks resolve into -- the same
    path ``nanochat_cpp.sandbox`` reads ``tools/sandbox/profiles.json`` from on
    every test action. The error says what to declare, so a future target that
    hides the tree fails with the fix and not with a ``KeyError``.
    """
    found = _lib.find_in_runfiles(relative)
    if found is not None:
        return found
    candidate = _build.repo_root() / relative
    if candidate.is_file():
        return candidate
    raise AssertionError(
        f"cannot read {relative}: there is no runfiles entry and no checkout "
        f"at {candidate}; declare {relative} as data of this target")


def bazelrc_stanzas(text: str) -> dict:
    """The flags of every named config in ``.bazelrc``, as name -> [flags].

    Only ``build:NAME`` lines count. A comment above a stanza is a comment, so
    the ``h100`` comment that mentions ``simulator`` cannot make ``hopper``
    look as if it set one.
    """
    stanzas: dict = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        words = stripped.split()
        if not words[0].startswith("build:"):
            continue
        name, *flags = words
        stanzas.setdefault(name, []).extend(flags)
    return stanzas


#: A fake ``tools/nanochat``: it records its argv, and on a zero status it
#: creates the three outputs the build would create, relative to the working
#: directory the builder runs it in. ``library`` is the shared library name the
#: package resolves, so the script holds no second copy of it.
FAKE_ENTRY = """#!{python}
import json
import os
import sys

library = {library!r}
argv = sys.argv[1:]
log = os.environ.get("FAKE_BUILD_LOG")
if log:
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(argv) + "\\n")
status = os.environ.get("FAKE_BUILD_STATUS")
if status:
    sys.stderr.write("the fake build failed\\n")
    sys.exit(int(status))
wanted = [
    ("//bindings:nanochat_shared", os.path.join("bazel-bin", "bindings",
                                                library)),
    ("//src:sft_main", os.path.join("bazel-bin", "src", "sft_main")),
    ("//src:rl_worker", os.path.join("bazel-bin", "src", "rl_worker")),
]
for target, path in wanted:
    if target in argv:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(b"")
"""


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
        os.environ["NANOCHAT_CPP_PRECISION"] = "fp32"
        os.environ["NANOCHAT_CUDA_ARCH"] = "sm_75"
        options = _build.BuildOptions.from_env()
        self.assertEqual(options.backend, "cuda")
        self.assertEqual(options.precision, "fp32")
        self.assertEqual(options.arch, "sm_75")
        self.assertEqual(options.cache, _build.cache_dir())

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
            backend=" CUDA ", precision="FP32", arch="sm_75")
        self.assertEqual(options.backend, "cuda")
        self.assertEqual(options.precision, "fp32")
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
        os.environ["NANOCHAT_CPP_PRECISION"] = "fp32"
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

    def test_fp16_request_raises(self) -> None:
        options = _build.BuildOptions.resolve(
            backend="cuda", precision="fp16", arch="sm_75")
        with self.assertRaises(_build.BuildError):
            _build._build_command(Path("tools/nanochat"), options)

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

    def test_the_leg_triple_and_never_a_named_config(self) -> None:
        """The helper emits the explicit triple, not ``--config=hopper``.

        Row 1 of ``docs/pipeline-scaleup-plan.md`` adds the ``hopper`` name for
        a hand run only; the H100 leg of the Modal app keeps the triple of
        section 7, so a card build and its report name the same three flags. A
        helper that started emitting a named config would inherit whatever that
        stanza sets, which is how the simulator config got into a hardware run
        once already (section 3, trap 1).
        """
        options = _build.BuildOptions.resolve(backend="cuda",
                                              precision="bf16",
                                              arch="sm_90")
        command = _build._build_command(Path("tools/nanochat"), options)
        self.assertIn("--config=cuda", command)
        self.assertIn("--define=precision=bf16", command)
        self.assertIn("--@rules_cuda//cuda:archs=sm_90", command)
        for forbidden in ("--config=hopper", "--config=h100"):
            self.assertNotIn(forbidden, command)
        self.assertEqual(
            [part for part in command if part.startswith("--config=")],
            ["--config=cuda"],
            f"the helper named a config of .bazelrc: {command}")


class BazelConfigTest(unittest.TestCase):
    """Row 1: the ``hopper`` config, pinned beside the ``h100`` simulator.

    ``--config=h100`` is the simulator profile of ``.bazelrc``: it sets
    ``--define=simulator=1``, which compiles in the capability override, and it
    swaps the collective to ``native``. ``--config=hopper`` is the hand-run
    build for a real Hopper card, so it must set neither: the chain runs
    ``world_size=1`` and keeps the default ``nccl`` collective
    (``docs/pipeline-scaleup-plan.md`` section 3 trap 1, P7, and section 7).
    The two stanzas are read in one class, so the pair cannot drift apart
    without a red test.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = tree_file(".bazelrc")
        cls.text = cls.path.read_text(encoding="utf-8")
        cls.stanzas = bazelrc_stanzas(cls.text)

    def flags(self, name: str) -> list:
        """The flags of one named config, with a readable missing case."""
        self.assertIn(
            name, self.stanzas,
            f"{self.path} has no {name} stanza; it holds "
            f"{sorted(self.stanzas)}")
        return self.stanzas[name]

    def test_the_hopper_stanza_is_the_hand_run_build(self) -> None:
        """sm_90 at bf16 on the CUDA backend, and nothing of the simulator."""
        flags = self.flags("build:hopper")
        self.assertIn("--@rules_cuda//cuda:archs=sm_90", flags)
        self.assertIn("--define=precision=bf16", flags)
        self.assertIn("--define=backend=cuda", flags)
        for forbidden in ("--define=simulator=1", "--define=collective=native"):
            self.assertNotIn(
                forbidden, flags,
                f"the hopper stanza set {forbidden}, which is the {forbidden} "
                f"of the h100 simulator profile (section 3, trap 1): {flags}")
        for flag in flags:
            self.assertNotIn(
                "simulator", flag,
                f"the hopper stanza mentions a simulator flag: {flag}")

    def test_the_h100_stanza_is_still_the_simulator(self) -> None:
        """The neighbour: the same arch and precision, plus both simulator
        defines, so an edit that turns one into the other is visible."""
        flags = self.flags("build:h100")
        self.assertIn("--@rules_cuda//cuda:archs=sm_90", flags)
        self.assertIn("--define=precision=bf16", flags)
        self.assertIn("--define=backend=cuda", flags)
        self.assertIn("--define=simulator=1", flags)
        self.assertIn("--define=collective=native", flags)
        self.assertNotEqual(sorted(self.flags("build:hopper")), sorted(flags),
                            "the hopper and h100 stanzas are the same stanza, "
                            "so one of the two names is the other")

    def test_the_t4_host_target_is_untouched_beside_them(self) -> None:
        """``t4`` is the other complete named target: sm_75 at fp32."""
        flags = self.flags("build:t4")
        self.assertIn("--@rules_cuda//cuda:archs=sm_75", flags)
        self.assertIn("--define=precision=fp32", flags)
        self.assertIn("--define=backend=cuda", flags)


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


class StageBinaryTest(unittest.TestCase):
    """B6: one resolver finds the two stage executables of the chain.

    The order is the order of ``nanochat_cpp._lib`` and not the shorter list of
    ``docs/python.md`` section 3.2: the environment override, the runfiles
    tree, ``bazel-bin/src/<name>``, the ``nanochat-<name>`` entry point of a
    future wheel, then an error that names the build command. Every case runs
    over a temporary tree, so nothing is built and no real runfiles tree is
    read.
    """

    NAMES = ("NANOCHAT_SFT_MAIN", "NANOCHAT_RL_WORKER", "RUNFILES_DIR",
             "RUNFILES_MANIFEST_FILE", "TEST_SRCDIR", "TEST_WORKSPACE")

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nanochat_stage_")
        self.root = Path(self.tmp.name)
        self.previous = {name: os.environ.get(name) for name in self.NAMES}
        for name in self.NAMES:
            os.environ.pop(name, None)
        # An empty runfiles root, so a case that does not build one sees a
        # miss and the next step of the order runs.
        self.runfiles = self.root / "runfiles"
        self.runfiles.mkdir()
        os.environ["RUNFILES_DIR"] = str(self.runfiles)

    def tearDown(self) -> None:
        for name, value in self.previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self.tmp.cleanup()

    def _executable(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return path

    def test_the_stage_names_and_their_labels(self) -> None:
        self.assertEqual(_build.STAGE_NAMES, ("sft_main", "rl_worker"))
        self.assertEqual(_build.stage_override("sft_main"),
                         "NANOCHAT_SFT_MAIN")
        self.assertEqual(_build.stage_override("rl_worker"),
                         "NANOCHAT_RL_WORKER")
        self.assertEqual(_build.stage_target("sft_main"), "//src:sft_main")
        self.assertEqual(_build.stage_target("rl_worker"), "//src:rl_worker")

    def test_the_environment_override_wins(self) -> None:
        """The caller named the file, so it wins even when it is absent."""
        override = self.root / "named" / "sft_main"
        os.environ["NANOCHAT_SFT_MAIN"] = str(override)
        self._executable(self.runfiles / "src" / "sft_main")
        self.assertEqual(_build.stage_binary("sft_main", root=self.root),
                         override)
        worker = self.root / "named" / "rl_worker"
        os.environ["NANOCHAT_RL_WORKER"] = str(worker)
        self.assertEqual(_build.stage_binary("rl_worker", root=self.root),
                         worker)

    def test_the_runfiles_tree_wins_over_the_development_tree(self) -> None:
        """A ``py_test`` or a ``bazel run`` finds the data dependency."""
        runfiles = self._executable(self.runfiles / "src" / "sft_main")
        development = self._executable(self.root / "bazel-bin" / "src" /
                                       "sft_main")
        self.assertNotEqual(runfiles, development)
        self.assertEqual(_build.stage_binary("sft_main", root=self.root),
                         runfiles)

    def test_a_runfiles_miss_falls_through_to_bazel_bin(self) -> None:
        development = self._executable(self.root / "bazel-bin" / "src" /
                                       "rl_worker")
        self.assertEqual(_build.stage_binary("rl_worker", root=self.root),
                         development)

    def test_the_wheel_entry_point_is_the_last_source(self) -> None:
        """A wheel install resolves ``nanochat-<name>`` from ``PATH``."""
        bindir = self.root / "bin"
        installed = self._executable(bindir / "nanochat-rl_worker")
        with mock.patch.dict(os.environ,
                             {"PATH": str(bindir)}):
            self.assertEqual(_build.stage_binary("rl_worker", root=self.root),
                             installed)

    def test_the_error_names_the_build_command(self) -> None:
        """A tree with no build fails with one line a reader can paste."""
        with mock.patch.dict(os.environ, {"PATH": str(self.root / "empty")}):
            with self.assertRaises(_build.BuildError) as caught:
                _build.stage_binary("sft_main", root=self.root)
        message = str(caught.exception)
        self.assertIn("NANOCHAT_SFT_MAIN is unset", message)
        self.assertIn("there is no runfiles entry", message)
        self.assertIn(str(self.root / "bazel-bin" / "src" / "sft_main"),
                      message)
        self.assertIn("nanochat-sft_main is not on PATH", message)
        self.assertIn("build it with:", message)
        for part in ("build", "//bindings:nanochat_shared", "//src:sft_main",
                     "//src:rl_worker"):
            self.assertIn(part, message)


class BuildStagesTest(unittest.TestCase):
    """B6: one ``tools/nanochat build`` line builds the library and both
    children.

    ``_build_command`` stays the one home of the command line, so this class
    checks that :func:`nanochat_cpp._build.build_stages` adds the two stage
    targets and nothing else, and that it reports the three outputs.
    """

    NAMES = ("NANOCHAT_SFT_MAIN", "NANOCHAT_RL_WORKER", "RUNFILES_DIR",
             "RUNFILES_MANIFEST_FILE", "TEST_SRCDIR", "TEST_WORKSPACE")

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nanochat_stages_")
        self.root = Path(self.tmp.name)
        self.previous = {name: os.environ.get(name) for name in self.NAMES}
        for name in self.NAMES:
            os.environ.pop(name, None)
        empty = self.root / "runfiles"
        empty.mkdir()
        os.environ["RUNFILES_DIR"] = str(empty)
        self.log = self.root / "build.log"
        os.environ["FAKE_BUILD_LOG"] = str(self.log)
        entry = self.root / "tools" / "nanochat"
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text(FAKE_ENTRY.format(python=sys.executable,
                                          library=_build.LIBRARY_NAME),
                         encoding="utf-8")
        entry.chmod(entry.stat().st_mode | stat.S_IEXEC)
        self.options = _build.BuildOptions.resolve(backend="cpu",
                                                   precision="fp32")

    def tearDown(self) -> None:
        for name, value in self.previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self.tmp.cleanup()

    def test_one_build_line_names_the_library_and_both_stages(self) -> None:
        library, sft_main, rl_worker = _build.build_stages(
            self.options, root=self.root)
        self.assertEqual(library, self.root / "bazel-bin" / "bindings" /
                         _build.LIBRARY_NAME)
        self.assertEqual(sft_main, self.root / "bazel-bin" / "src" /
                         "sft_main")
        self.assertEqual(rl_worker, self.root / "bazel-bin" / "src" /
                         "rl_worker")
        lines = self.log.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1, "the builder ran more than one build")
        argv = json.loads(lines[0])
        self.assertEqual(argv[0], "build")
        for target in ("//bindings:nanochat_shared", "//src:sft_main",
                       "//src:rl_worker"):
            self.assertIn(target, argv)
        self.assertIn("--config=cpu", argv)
        self.assertIn("--define=precision=fp32", argv)

    def test_a_failed_build_names_the_command(self) -> None:
        os.environ["FAKE_BUILD_STATUS"] = "3"
        try:
            with self.assertRaises(_build.BuildError) as caught:
                _build.build_stages(self.options, root=self.root)
        finally:
            os.environ.pop("FAKE_BUILD_STATUS", None)
        message = str(caught.exception)
        self.assertIn("the stage build failed (3)", message)
        self.assertIn("//src:rl_worker", message)
        self.assertIn("the fake build failed", message)

    def test_a_tree_without_the_entry_point_raises(self) -> None:
        (self.root / "tools" / "nanochat").unlink()
        with self.assertRaises(_build.BuildError) as caught:
            _build.build_stages(self.options, root=self.root)
        self.assertIn("cannot build the stage executables",
                      str(caught.exception))
        self.assertFalse(self.log.exists(), "the builder ran a command")


if __name__ == "__main__":
    unittest.main()
