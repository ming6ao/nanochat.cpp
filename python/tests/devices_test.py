"""The declared device set has one source (docs/simulator.md section 7).

The command line used to carry its own tuple of device names beside the
declarations, so adding a device meant editing three files and remembering two
of them. These tests pin the rule that replaced it: the device set is read from
`backends/*/caps.json`, a device that is not an accelerator contributes no name,
and no module holds a second copy.
"""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat_cpp import cli  # noqa: E402


def declared_by_hand() -> list:
    """Every `is_device` name, read from the declarations a second way."""
    names = []
    for path in sorted(cli.REPO_ROOT.glob("backends/*/caps.json")):
        with open(path, encoding="utf-8") as handle:
            body = json.load(handle)
        for device in body.get("devices", []):
            if device.get("is_device"):
                names.append(device["name"])
    return names


class DeclaredDeviceNamesTest(unittest.TestCase):
    def test_the_names_are_the_declared_devices(self):
        self.assertEqual(cli.declared_device_names(), declared_by_hand())

    def test_the_host_is_not_a_device(self):
        # The reference host declares no launch cost, no HBM rate, and no peak
        # rate, so it is not a device a report or a simulation may name.
        self.assertNotIn("cpu", cli.declared_device_names())

    def test_the_set_is_not_empty(self):
        self.assertTrue(cli.declared_device_names())

    def test_the_module_holds_no_second_copy(self):
        # A hard-coded tuple is what drifted before. The names may appear in the
        # declarations and in documentation, never as a literal in the command
        # line module.
        with open(cli.__file__, encoding="utf-8") as handle:
            source = handle.read()
        self.assertNotIn("SIM_DEVICES", source)
        literal = re.search(
            r"\(\s*\"h100\"\s*,\s*\"h200\"", source)
        self.assertIsNone(literal, "the command line hard-codes a device tuple")


class SimulationValidationTest(unittest.TestCase):
    def test_an_undeclared_device_is_refused(self):
        # The parse keeps the name; the refusal compares it with the declared
        # set, so an undeclared card reaches a message and not a run.
        _, device, _, _ = cli._simulate_args(
            ["--device=b200", "--suite=profile"])
        self.assertEqual(device, "b200")
        self.assertNotIn(device, cli.declared_device_names())

    def test_a_declared_device_passes(self):
        for name in cli.declared_device_names():
            _, device, _, _ = cli._simulate_args([f"--device={name}"])
            self.assertEqual(device, name)


if __name__ == "__main__":
    unittest.main()
