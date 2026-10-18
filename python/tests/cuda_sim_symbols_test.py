"""T0: the symbol gate for the S0 API interposer.

The gate shells out to `nm` and `readelf`, so the suite puts fake versions of
both on `PATH` and feeds them canned output. No compiler, no toolkit, and no
mock library are involved. The real end-to-end proof is
`tools/nanochat simulate --suite api`, which builds the mock and runs the gate
three ways, including the negative self-test.

See docs/simulator.md, docs/correctness-gates.md, and
docs/python-sandbox-plan.md.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

#: A fake `nm`. The two modes read their canned output from the environment.
FAKE_NM = """#!/bin/sh
case "$*" in
  *--defined-only*) printf '%s' "$NANOCHAT_TEST_DEFINED" ;;
  *--undefined-only*) printf '%s' "$NANOCHAT_TEST_UNDEFINED" ;;
esac
"""

#: A fake `readelf -V`. It prints the version nodes of the mock.
FAKE_READELF = """#!/bin/sh
printf '%s' "$NANOCHAT_TEST_NODES"
"""

#: The map fixture: one global name and one private name.
MAP = """libcudart.so.12 {
  global:
    cudaMalloc;
  local:
    nanochatPrivateHelper;
};
"""

#: The two version nodes the gate requires.
NODES = "  0x0010  Name: libcudart.so.12\n  0x0011  Name: libcublas.so.12\n"

DEFINED = "0000000000000000 T cudaMalloc@@libcudart.so.12\n"


class CheckSymbolsTest(unittest.TestCase):
    """The required set, the version nodes, and the exit codes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="nanochat_symgate_")
        cls.root = Path(cls.tmp.name)
        cls.bin_dir = cls.root / "bin"
        cls.bin_dir.mkdir()
        for name, text in (("nm", FAKE_NM), ("readelf", FAKE_READELF)):
            tool = cls.bin_dir / name
            tool.write_text(text, encoding="utf-8")
            tool.chmod(stat.S_IRWXU)
        # The gate lives outside the Python package, next to the map it reads.
        cls.gate = (Path(__file__).resolve().parents[2]
                    / "tools" / "cuda_sim" / "check_symbols.py")
        cls.map_file = cls.root / "cuda_sim.map"
        cls.map_file.write_text(MAP, encoding="utf-8")
        cls.mock = cls.root / "libcuda_sim_interposer.so"
        cls.mock.write_text("", encoding="utf-8")
        cls.binary = cls.root / "cuda_runtime_gpu_test"
        cls.binary.write_text("", encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def run_gate(self, *argv: str, defined: str = DEFINED, nodes: str = NODES,
                 undefined: str = "", with_inputs: bool = True
                 ) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["PATH"] = f"{self.bin_dir}:{env.get('PATH', '')}"
        env["NANOCHAT_TEST_DEFINED"] = defined
        env["NANOCHAT_TEST_NODES"] = nodes
        env["NANOCHAT_TEST_UNDEFINED"] = undefined
        command = [sys.executable, str(self.gate)]
        if with_inputs:
            command += ["--map", str(self.map_file), "--mock", str(self.mock)]
        return subprocess.run([*command, *argv], capture_output=True, text=True,
                              env=env, timeout=120, check=False)

    # -- the required set -------------------------------------------------

    def test_a_global_name_is_required(self) -> None:
        result = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("check_symbols: OK (1 required symbols)", result.stdout)

    def test_a_local_name_is_not_required(self) -> None:
        """The gate reads only `global:` blocks, so a private name passes."""
        result = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("nanochatPrivateHelper", result.stderr)

    def test_a_missing_symbol_fails(self) -> None:
        result = self.run_gate(defined="")
        self.assertEqual(result.returncode, 1)
        self.assertIn("the mock does not export:", result.stderr)
        self.assertIn("cudaMalloc", result.stderr)
        self.assertIn("check_symbols: FAILED", result.stderr)

    def test_a_missing_version_node_fails(self) -> None:
        result = self.run_gate(nodes="  0x0010  Name: libcudart.so.12\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("the mock has no version node 'libcublas.so.12'",
                      result.stderr)

    def test_a_binary_that_asks_for_a_missing_symbol_fails(self) -> None:
        undefined = ("                 U cudaMalloc@libcudart.so.12\n"
                     "                 U cublasCreate_v2@libcublas.so.12\n"
                     "                 U malloc\n")
        result = self.run_gate(str(self.binary), undefined=undefined)
        self.assertEqual(result.returncode, 1)
        self.assertIn("asks for symbols the mock lacks:", result.stderr)
        self.assertIn("cublasCreate_v2", result.stderr)
        self.assertNotIn("malloc", result.stderr)

    # -- the inputs -------------------------------------------------------

    def test_a_missing_map_is_a_usage_error(self) -> None:
        result = self.run_gate("--map", str(self.root / "absent.map"),
                               "--mock", str(self.mock))
        self.assertEqual(result.returncode, 2)
        self.assertIn("is missing", result.stderr)

    def test_a_missing_mock_is_a_usage_error(self) -> None:
        result = self.run_gate("--map", str(self.map_file),
                               "--mock", str(self.root / "absent.so"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("cuda_sim_interposer", result.stderr)

    def test_a_missing_binary_is_a_usage_error(self) -> None:
        result = self.run_gate(str(self.root / "absent.bin"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("is missing", result.stderr)

    def test_a_map_without_a_global_block_is_a_usage_error(self) -> None:
        empty = self.root / "empty.map"
        empty.write_text("libcudart.so.12 {\n  local:\n    *;\n};\n",
                         encoding="utf-8")
        result = self.run_gate("--map", str(empty), "--mock", str(self.mock))
        self.assertEqual(result.returncode, 2)
        self.assertIn("lists no global symbols", result.stderr)


if __name__ == "__main__":
    unittest.main()
