"""T0: the pure setup helpers of ``session.py``.

The test covers the secret loader, the device check, and the session view. It
injects a fake secrets client and a device list, so it needs no Kaggle
package, no shared library, and no network. See ``docs/checkpoint-format.md``
section 22.
"""

from __future__ import annotations

import contextlib
import io
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


if __name__ == "__main__":
    unittest.main()
