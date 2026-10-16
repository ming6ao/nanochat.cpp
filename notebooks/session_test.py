"""T0: the pure setup helpers of ``session.py``.

The test covers the secret loader, the device check, and the session view. It
injects a fake secrets client and a device list, so it needs no Kaggle
package, no shared library, and no network. See ``docs/checkpoint-format.md``
section 22.
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path

import kaggle_setup
import session


class FakeClient:
    """A ``UserSecretsClient`` that serves the values it holds."""

    def __init__(self, values):
        self.values = dict(values)
        self.calls = []

    def get_secret(self, name):
        self.calls.append(name)
        if name not in self.values:
            raise KeyError(name)
        return self.values[name]


class LoadSecretsTest(unittest.TestCase):
    def test_found_and_missing(self) -> None:
        client = FakeClient({"HF_TOKEN": "token"})
        environ = {}
        with contextlib.redirect_stdout(io.StringIO()) as out:
            values = session.load_secrets(("HF_TOKEN", "GEMINI_API_KEY"),
                                          client=client, environ=environ)
        self.assertEqual(values, {"HF_TOKEN": "token"})
        self.assertEqual(environ, {"HF_TOKEN": "token"})
        self.assertEqual(client.calls, ["HF_TOKEN", "GEMINI_API_KEY"])
        self.assertIn("missing secret GEMINI_API_KEY", out.getvalue())

    def test_no_names(self) -> None:
        self.assertEqual(session.load_secrets((), client=FakeClient({}),
                                              environ={}), {})


class RequireDevicesTest(unittest.TestCase):
    def test_returns_the_list(self) -> None:
        devices = ["a", "b", "c"]
        self.assertEqual(session.require_devices(2, devices=devices), devices)

    def test_too_few(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            session.require_devices(3, devices=["a", "b"])
        self.assertIn("needs 3 device(s)", str(caught.exception))

    def test_zero_needs(self) -> None:
        self.assertEqual(session.require_devices(0, devices=[]), [])


class SessionTest(unittest.TestCase):
    def make(self) -> session.Session:
        host = kaggle_setup.Host(repo=Path("/repo"), library=Path("/lib.so"),
                                 build_key="key", env={"A": "b"})
        return session.Session(host=host, secrets={"HF_TOKEN": "t"},
                               devices=[1, 2], base=Path("/base"))

    def test_views(self) -> None:
        prepared = self.make()
        self.assertEqual(prepared.repo, Path("/repo"))
        self.assertEqual(prepared.library, Path("/lib.so"))
        self.assertEqual(prepared.build_key, "key")
        self.assertEqual(prepared.base, Path("/base"))
        self.assertEqual(prepared.devices, [1, 2])

    def test_frozen(self) -> None:
        prepared = self.make()
        with self.assertRaises(Exception):
            prepared.base = Path("/other")

    def test_revision_reads_the_repository(self) -> None:
        prepared = self.make()
        calls = []
        original = kaggle_setup.revision
        kaggle_setup.revision = lambda root: calls.append(root) or "abc123"
        try:
            self.assertEqual(prepared.revision, "abc123")
        finally:
            kaggle_setup.revision = original
        self.assertEqual(calls, [Path("/repo")])


class DescribeTest(unittest.TestCase):
    def test_prints_library_secrets_and_devices(self) -> None:
        host = kaggle_setup.Host(repo=Path("/repo"), library=Path("/lib.so"),
                                 build_key="key", env={})
        device = types.SimpleNamespace(name="T4",
                                       total_memory_bytes=16 * 2**30)
        prepared = session.Session(host=host, secrets={"HF_TOKEN": "t"},
                                   devices=[device], base=Path("/base"))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            returned = session.describe(prepared)
        self.assertIs(returned, prepared)
        text = buffer.getvalue()
        self.assertIn("library: /lib.so", text)
        self.assertIn("secrets: ['HF_TOKEN']", text)
        self.assertIn("T4", text)
        self.assertIn("16.0 GiB", text)


class BootstrapTest(unittest.TestCase):
    def test_adds_the_helper_path_and_calls_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            helper = repo / ".agents" / "skills" / "model-fit" / "scripts"
            helper.mkdir(parents=True)
            captured = {}
            original = session.setup
            saved_path = list(sys.path)

            def fake_setup(**kwargs):
                captured.update(kwargs)
                return "prepared"

            session.setup = fake_setup
            try:
                result = session.bootstrap(repo_path=repo, arch="sm_75",
                                           devices=2)
                self.assertEqual(sys.path[0], str(helper))
            finally:
                session.setup = original
                sys.path[:] = saved_path
            self.assertEqual(result, "prepared")
            self.assertEqual(captured["repo_path"], repo)
            self.assertEqual(captured["devices"], 2)
            self.assertEqual(captured["arch"], "sm_75")

    def test_missing_helper_directory_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            original = session.setup
            session.setup = lambda **kwargs: "prepared"
            saved_path = list(sys.path)
            try:
                result = session.bootstrap(repo_path=Path(tmp))
            finally:
                session.setup = original
                sys.path[:] = saved_path
            self.assertEqual(result, "prepared")


if __name__ == "__main__":
    unittest.main()
