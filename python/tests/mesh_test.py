"""T0: the device mesh and the placement language.

The mesh and the placement checks are hermetic. `devices()` loads the shared
library and reads the device table, so the test also covers the additive
`nanochat_device_count`/`nanochat_device_at` surface
(docs/client-runtime-design.md sections 5.1 to 5.4).
"""

from __future__ import annotations

import unittest

from nanochat_cpp import _lib, mesh


class MeshTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = _lib.load()

    def test_devices_lists_the_active_backend(self) -> None:
        found = mesh.devices()
        self.assertGreaterEqual(len(found), 1)
        for ordinal, device in enumerate(found):
            self.assertEqual(device.ordinal, ordinal)
            self.assertTrue(device.name)
            self.assertIsInstance(str(device), str)

    def test_mesh_from_devices(self) -> None:
        found = mesh.devices()
        m = mesh.Mesh(found)
        self.assertEqual(m.size, len(found))
        self.assertEqual(m.shape, (len(found),))
        self.assertEqual(m.axis_names, (mesh.DATA_AXIS,))
        self.assertEqual(len(m), len(found))

    def test_rejects_rank_two(self) -> None:
        with self.assertRaises(mesh.MeshError):
            mesh.Mesh(mesh.devices(), ("data", "model"))

    def test_rejects_unknown_axis(self) -> None:
        with self.assertRaises(mesh.MeshError):
            mesh.Mesh(mesh.devices(), ("model",))

    def test_rejects_empty_mesh(self) -> None:
        with self.assertRaises(mesh.MeshError):
            mesh.Mesh(())

    def test_replicated_placement(self) -> None:
        m = mesh.Mesh(mesh.devices())
        spec = mesh.PartitionSpec()
        self.assertTrue(spec.is_replicated())
        sharding = mesh.NamedSharding(m, spec)
        self.assertIs(mesh.validate_parameter_sharding(sharding), sharding)

    def test_batch_placement(self) -> None:
        m = mesh.Mesh(mesh.devices())
        sharding = mesh.NamedSharding(m, mesh.PartitionSpec(mesh.DATA_AXIS))
        self.assertIs(mesh.validate_batch_sharding(sharding), sharding)

    def test_rejects_sharded_parameter(self) -> None:
        m = mesh.Mesh(mesh.devices())
        sharding = mesh.NamedSharding(m, mesh.PartitionSpec(mesh.DATA_AXIS))
        with self.assertRaises(mesh.MeshError):
            mesh.validate_parameter_sharding(sharding)

    def test_rejects_replicated_batch(self) -> None:
        m = mesh.Mesh(mesh.devices())
        sharding = mesh.NamedSharding(m, mesh.PartitionSpec())
        with self.assertRaises(mesh.MeshError):
            mesh.validate_batch_sharding(sharding)

    def test_rejects_unknown_axis_in_sharding(self) -> None:
        m = mesh.Mesh(mesh.devices())
        with self.assertRaises(mesh.MeshError):
            mesh.NamedSharding(m, mesh.PartitionSpec("model"))

    def test_partition_spec_equality(self) -> None:
        self.assertEqual(mesh.PartitionSpec("data"), mesh.PartitionSpec("data"))
        self.assertNotEqual(mesh.PartitionSpec(), mesh.PartitionSpec("data"))


if __name__ == "__main__":
    unittest.main()
