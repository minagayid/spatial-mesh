import json
import subprocess
import sys
import unittest
from pathlib import Path

from spatial_mesh import (
    AStarNavigation,
    EnvironmentObject,
    Observation,
    OccupancyFusion,
    SpatialMesh,
    SpaceOperatingSystem,
    SpaceTensor,
    SpatialSceneGraph,
    World,
    raycast,
)


class SpatialMeshCoreTests(unittest.TestCase):
    def test_observation_confidence_and_channels_are_bounded(self):
        tensor = SpaceTensor(((-1, -1, -1), (1, 1, 1)), 1.0)
        tensor.apply_observation(
            (0, 0, 0),
            {
                "occupancy": 2.0,
                "reflectivity": -1.0,
                "confidence": 2.0,
                "motion": 3.0,
            },
        )
        cell = tensor.cell((0, 0, 0))
        self.assertIsNotNone(cell)
        self.assertGreaterEqual(cell.occupancy, 0.0)
        self.assertLessEqual(cell.occupancy, 1.0)
        self.assertGreaterEqual(cell.reflectivity, 0.0)
        self.assertLessEqual(cell.reflectivity, 1.0)
        self.assertGreaterEqual(cell.confidence, 0.0)
        self.assertLessEqual(cell.confidence, 1.0)
        self.assertLessEqual(cell.motion, 1.0)

    def test_tensor_occupancy_uses_bounded_hit_and_miss_updates(self):
        tensor = SpaceTensor(((0, 0, 0), (1, 1, 1)), 1.0)
        tensor.apply_observation((0, 0, 0), {"occupancy": 1.0, "confidence": 1.0})
        hit = tensor.cell((0, 0, 0)).occupancy
        tensor.apply_observation((0, 0, 0), {"occupancy": 0.0, "confidence": 1.0})
        miss = tensor.cell((0, 0, 0)).occupancy
        self.assertGreater(hit, 0.5)
        self.assertLess(miss, hit)
        self.assertGreaterEqual(miss, 0.0)
        self.assertLessEqual(miss, 1.0)

    def test_observation_api_fuses_timestamp_modality_and_uncertainty(self):
        tensor = SpaceTensor(((0, 0, 0), (2, 2, 2)), 1.0)
        obs = Observation(
            position=(0.5, 0.5, 0.5),
            occupancy=1.0,
            confidence=0.8,
            modality="depth",
            timestamp=12.5,
            uncertainty=0.2,
            object_id=7,
        )
        idx = tensor.observe(obs)
        self.assertEqual(idx, (0, 0, 0))
        cell = tensor.cell(idx)
        self.assertEqual(cell.timestamp, 12.5)
        self.assertEqual(cell.object_id, 7)
        self.assertEqual(cell.last_modality, "depth")
        self.assertAlmostEqual(cell.uncertainty, 0.2)

    def test_cell_to_spaxel_preserves_motion_velocity_confidence_and_material(self):
        tensor = SpaceTensor(((0, 0, 0), (1, 1, 1)), 1.0)
        tensor.apply_observation(
            (0, 0, 0),
            {
                "occupancy": 0.9,
                "confidence": 0.7,
                "motion": 0.8,
                "reflectivity": 0.6,
                "velocity_x": 1.25,
                "velocity_y": -0.5,
                "velocity_z": 0.25,
                "material": "metal",
            },
        )
        spaxel = tensor.cell((0, 0, 0)).to_spaxel()
        self.assertAlmostEqual(spaxel.confidence, tensor.cell((0, 0, 0)).confidence)
        self.assertEqual(spaxel.velocity, (1.25, -0.5, 0.25))
        self.assertEqual(spaxel.material, "metal")
        self.assertAlmostEqual(spaxel.motion, tensor.cell((0, 0, 0)).motion)

    def test_occupancy_fusion_is_replayable_and_uses_uncertainty(self):
        fusion = OccupancyFusion()
        first = fusion.update(0.5, 1.0, confidence=0.9, uncertainty=0.0)
        second = fusion.update(first, 0.0, confidence=0.9, uncertainty=0.0)
        self.assertGreater(first, 0.5)
        self.assertLess(second, first)
        replay = OccupancyFusion()
        replay.update(0.5, 1.0, confidence=0.9, uncertainty=0.0)
        replayed = replay.update(first, 0.0, confidence=0.9, uncertainty=0.0)
        self.assertAlmostEqual(second, replayed)

    def test_duplicate_event_is_idempotent_and_snapshot_round_trips(self):
        tensor = SpaceTensor(((0, 0, 0), (1, 1, 1)), 1.0)
        obs = Observation((0.5, 0.5, 0.5), 1.0, confidence=1.0, event_id="frame-1")
        tensor.observe(obs)
        first_digest = tensor.digest()
        tensor.observe(obs)
        self.assertEqual(first_digest, tensor.digest())
        restored = SpaceTensor.from_snapshot(tensor.snapshot())
        self.assertEqual(first_digest, restored.digest())
        restored.observe(obs)
        self.assertEqual(first_digest, restored.digest())

    def test_non_finite_confidence_is_bounded_and_zero_confidence_is_ignored(self):
        tensor = SpaceTensor(((0, 0, 0), (1, 1, 1)), 1.0)
        tensor.apply_observation((0, 0, 0), {"occupancy": 1.0, "confidence": 0.0})
        self.assertEqual(tensor.cell((0, 0, 0)).occupancy, 0.0)
        tensor.apply_observation((0, 0, 0), {"occupancy": 1.0, "confidence": float("nan")})
        self.assertEqual(tensor.cell((0, 0, 0)).occupancy, 0.0)
        tensor.apply_observation((0, 0, 0), {"occupancy": 1.0, "confidence": float("inf")})
        self.assertEqual(tensor.cell((0, 0, 0)).occupancy, 1.0)

    def test_spatial_mesh_facade_replays_in_timestamp_order(self):
        mesh = SpatialMesh(bounds=((0, 0, 0), (1, 1, 1)), cell_size=1.0)
        events = [
            Observation((0.5, 0.5, 0.5), 0.0, confidence=1.0, timestamp=2.0, event_id="miss"),
            Observation((0.5, 0.5, 0.5), 1.0, confidence=1.0, timestamp=1.0, event_id="hit"),
        ]
        mesh.replay(events)
        self.assertEqual(mesh.tensor.cell((0, 0, 0)).timestamp, 2.0)
        restored = SpatialMesh.from_snapshot(mesh.snapshot())
        self.assertEqual(mesh.digest(), restored.digest())

    def test_sparse_tensor_allocates_only_observed_cells(self):
        tensor = SpaceTensor(((0, 0, 0), (100, 100, 10)), 1.0, eager=False)
        self.assertEqual(len(tensor.cells), 0)
        tensor.apply_observation((99, 99, 9), {"occupancy": 1.0, "confidence": 1.0})
        self.assertEqual(len(tensor.cells), 1)
        self.assertTrue(AStarNavigation(tensor).plan((0.5, 0.5, 0.5), (2.5, 0.5, 0.5)))


class SceneAndNavigationTests(unittest.TestCase):
    def test_world_build_is_idempotent_and_clears_removed_objects(self):
        world = World(bounds=((0, 0, 0), (3, 1, 1)), cell_size=1.0)
        object_a = EnvironmentObject(1, "box", (0.5, 0.5, 0.5), (1, 1, 1), material="wood")
        object_b = EnvironmentObject(2, "box", (2.5, 0.5, 0.5), (1, 1, 1), material="metal")
        world.add_objects((object_a, object_b))
        world.build()
        first = world.tensor.digest()
        world.build()
        self.assertEqual(first, world.tensor.digest())
        world.objects.remove(object_a)
        world.build()
        self.assertNotIn(1, [cell.object_id for _, cell in world.tensor.occupied_cells()])

    def test_scene_graph_relations_use_real_integer_node_ids(self):
        scene = SpatialSceneGraph()
        lower = scene.add_object("table", (0, 0, 0), (1, 1, 1))
        upper = scene.add_object("cup", (0, 0, 1), (0.2, 0.2, 0.2))
        scene.relate(lower, upper, "ON")
        self.assertEqual(scene.neighbors(lower, "ON"), [upper])
        self.assertEqual(scene.graph.number_of_nodes(), 2)
        self.assertEqual(scene.graph.number_of_edges(), 1)
        with self.assertRaises(KeyError):
            scene.relate("table", "cup", "ON")

    def test_ingest_replaces_previous_frame_and_preserves_mean_velocity(self):
        tensor = SpaceTensor(((0, 0, 0), (3, 1, 1)), 1.0)
        for idx in ((0, 0, 0), (1, 0, 0)):
            tensor.apply_observation(
                idx,
                {
                    "occupancy": 1.0,
                    "confidence": 1.0,
                    "motion": 1.0,
                    "velocity_x": 2.0,
                    "object_id": 42,
                },
            )
        operating_system = SpaceOperatingSystem()
        first = operating_system.ingest(tensor)
        second = operating_system.ingest(tensor)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(operating_system.scene.graph.number_of_nodes(), 1)
        self.assertAlmostEqual(operating_system._last_clusters[0].velocity[0], 2.0)
        self.assertAlmostEqual(operating_system.predict(1.0)[0][0], operating_system._last_clusters[0].position[0] + 2.0)

    def test_navigation_rejects_invalid_start_and_avoids_occupied_cells(self):
        tensor = SpaceTensor(((0, 0, 0), (4, 2, 1)), 1.0)
        tensor.apply_observation((1, 0, 0), {"occupancy": 1.0, "confidence": 1.0})
        navigation = AStarNavigation(tensor)
        self.assertEqual(navigation.plan((-1, 0, 0), (3.5, 0.5, 0.5)), [])
        path = navigation.plan((0.5, 0.5, 0.5), (3.5, 0.5, 0.5))
        self.assertTrue(path)
        self.assertNotIn((1.5, 0.5, 0.5), path)

    def test_zero_direction_raycast_is_rejected(self):
        obj = EnvironmentObject(1, "box", (1, 0, 0), (0.5, 0.5, 0.5))
        with self.assertRaises(ValueError):
            raycast((0, 0, 0), (0, 0, 0), [obj])


class LocalRunTests(unittest.TestCase):
    def test_core_and_optional_modules_import_without_site_packages(self):
        code = "import spatial_mesh; import spatial_mesh.tracer; import spatial_mesh.visualize"
        subprocess.run([sys.executable, "-S", "-c", code], cwd=Path(__file__).parent, check=True)

    def test_headless_demo_runs_with_python_standard_library_only(self):
        demo = Path(__file__).parent / "demos" / "demo_headless.py"
        result = subprocess.run(
            [sys.executable, "-S", str(demo)],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(result.stdout)
        self.assertGreaterEqual(payload["occupied_cells"], 1)
        self.assertGreaterEqual(payload["scene_objects"], 1)
        self.assertTrue(payload["path"])


if __name__ == "__main__":
    unittest.main()
