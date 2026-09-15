import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

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
from spatial_mesh.sensors import SensorObservation, SensorSuite
from spatial_mesh.tracer import WaveBasedSpaceOS


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

    def test_first_channel_samples_are_preserved_and_zero_samples_still_fuse(self):
        tensor = SpaceTensor(((0, 0, 0), (1, 1, 1)), 1.0)
        tensor.apply_observation(
            (0, 0, 0),
            {"occupancy": 1.0, "reflectivity": 0.0, "velocity_x": 0.0, "confidence": 0.1},
        )
        cell = tensor.cell((0, 0, 0))
        self.assertGreater(cell.occupancy, 0.5)
        self.assertLess(cell.occupancy, 0.9)
        self.assertEqual(cell.reflectivity, 0.0)
        self.assertEqual(cell.velocity_x, 0.0)
        tensor.apply_observation(
            (0, 0, 0),
            {"occupancy": 0.0, "reflectivity": 1.0, "velocity_x": 1.0, "confidence": 0.1},
        )
        self.assertAlmostEqual(cell.reflectivity, 0.09)
        self.assertAlmostEqual(cell.velocity_x, 0.09)

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

    def test_large_sparse_snapshot_round_trip_stays_sparse(self):
        tensor = SpaceTensor(((0, 0, 0), (100, 100, 100)), 1.0, eager=False)
        tensor.apply_observation((10, 20, 30), {"occupancy": 1.0})
        restored = SpaceTensor.from_snapshot(tensor.snapshot())
        self.assertFalse(restored.eager)
        self.assertEqual(len(restored.cells), 1)
        self.assertEqual(restored.digest(), tensor.digest())

    def test_snapshot_preserves_observed_free_cells_and_their_event_ids(self):
        tensor = SpaceTensor(((0, 0, 0), (1, 1, 1)), 1.0)
        observation = Observation((0.5, 0.5, 0.5), 0.0, confidence=1.0, event_id="free-cell")
        tensor.observe(observation)
        self.assertLess(tensor.cell((0, 0, 0)).occupancy, 0.01)
        restored = SpaceTensor.from_snapshot(tensor.snapshot())
        self.assertTrue(restored.cell((0, 0, 0)).occupancy_initialized)
        self.assertEqual(restored.digest(), tensor.digest())
        before = restored.digest()
        restored.observe(observation)
        self.assertEqual(restored.digest(), before)

    def test_snapshot_rejects_non_boolean_channel_initialization_state(self):
        tensor = SpaceTensor(((0, 0, 0), (1, 1, 1)), 1.0)
        tensor.apply_observation((0, 0, 0), {"occupancy": 1.0, "reflectivity": 0.0})
        snapshot = tensor.snapshot()
        snapshot["cells"][0]["occupancy_initialized"] = "false"
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            SpaceTensor.from_snapshot(snapshot)

    def test_snapshot_rejects_cell_indices_with_wrong_dimensions(self):
        tensor = SpaceTensor(((0, 0, 0), (1, 1, 1)), 1.0)
        tensor.apply_observation((0, 0, 0), {"occupancy": 1.0})
        snapshot = tensor.snapshot()
        snapshot["cells"][0]["index"] = [0, 0, 0, 9]
        with self.assertRaisesRegex(ValueError, "exactly three integers"):
            SpaceTensor.from_snapshot(snapshot)

    def test_snapshot_rejects_non_positive_observation_counts(self):
        tensor = SpaceTensor(((0, 0, 0), (1, 1, 1)), 1.0)
        tensor.apply_observation((0, 0, 0), {"occupancy": 1.0})
        snapshot = tensor.snapshot()
        snapshot["cells"][0]["observation_count"] = -1
        with self.assertRaisesRegex(ValueError, "positive integer"):
            SpaceTensor.from_snapshot(snapshot)

    def test_snapshot_rejects_duplicate_cell_indices(self):
        tensor = SpaceTensor(((0, 0, 0), (1, 1, 1)), 1.0)
        tensor.apply_observation((0, 0, 0), {"occupancy": 1.0})
        snapshot = tensor.snapshot()
        snapshot["cells"].append(dict(snapshot["cells"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate cell index"):
            SpaceTensor.from_snapshot(snapshot)

    def test_snapshot_rejects_unreasonably_large_observation_count(self):
        tensor = SpaceTensor(((0, 0, 0), (1, 1, 1)), 1.0)
        tensor.apply_observation((0, 0, 0), {"occupancy": 1.0})
        snapshot = tensor.snapshot()
        snapshot["cells"][0]["observation_count"] = 10**1000
        with self.assertRaisesRegex(ValueError, "positive integer"):
            SpaceTensor.from_snapshot(snapshot)

    def test_outside_finite_coordinates_return_no_index_on_overflowing_deltas(self):
        tensor = SpaceTensor(((-1e308, 0.0, 0.0), (-9e307, 1.0, 1.0)), 1e307, eager=False)
        self.assertIsNone(tensor.index_of((1e308, 0.5, 0.5)))

    def test_list_index_is_rejected_without_dictionary_lookup_error(self):
        tensor = SpaceTensor(((0, 0, 0), (1, 1, 1)), 1.0)
        self.assertFalse(tensor.valid_index([0, 0, 0]))
        self.assertFalse(tensor.apply_observation([0, 0, 0], {"occupancy": 1.0}))
        self.assertIsNone(tensor.cell([0, 0, 0]))

    def test_tensor_construction_requires_a_boolean_eager_flag(self):
        for value in (0, 1, "false"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "eager must be a boolean"):
                    SpaceTensor(((0, 0, 0), (1, 1, 1)), 1.0, eager=value)

    def test_snapshot_rejects_present_null_eager_flag_but_keeps_legacy_fallback(self):
        tensor = SpaceTensor(((0, 0, 0), (1, 1, 1)), 1.0)
        snapshot = tensor.snapshot()
        snapshot["eager"] = None
        with self.assertRaisesRegex(ValueError, "eager must be a boolean"):
            SpaceTensor.from_snapshot(snapshot)

        del snapshot["eager"]
        self.assertTrue(SpaceTensor.from_snapshot(snapshot).eager)

    def test_non_finite_confidence_is_bounded_and_zero_confidence_is_ignored(self):
        tensor = SpaceTensor(((0, 0, 0), (1, 1, 1)), 1.0)
        tensor.apply_observation((0, 0, 0), {"occupancy": 1.0, "confidence": 0.0})
        self.assertEqual(tensor.cell((0, 0, 0)).occupancy, 0.0)
        tensor.apply_observation((0, 0, 0), {"occupancy": 1.0, "confidence": float("nan")})
        self.assertEqual(tensor.cell((0, 0, 0)).occupancy, 0.0)
        tensor.apply_observation((0, 0, 0), {"occupancy": 1.0, "confidence": float("inf")})
        self.assertAlmostEqual(tensor.cell((0, 0, 0)).occupancy, 1.0, places=5)

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

    def test_eager_tensor_rejects_a_grid_that_would_exhaust_memory(self):
        with self.assertRaisesRegex(ValueError, "eager grid exceeds"):
            SpaceTensor(((0, 0, 0), (100, 100, 100)), 0.01)

    def test_sparse_tensor_rejects_oversized_dense_materialization(self):
        tensor = SpaceTensor(((0, 0, 0), (100, 100, 100)), 1.0, eager=False)
        with self.assertRaisesRegex(ValueError, "dense tensor exceeds"):
            tensor.to_tensor()

    def test_nonstandard_channel_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "channels must be 9"):
            SpaceTensor(((0, 0, 0), (100, 100, 4)), 1.0, channels=64, eager=False)
        with self.assertRaisesRegex(ValueError, "channels must be 9"):
            SpaceTensor(((0, 0, 0), (1, 1, 1)), 1.0, channels=10**9)

    def test_snapshot_rejects_unbounded_channel_count(self):
        tensor = SpaceTensor(((0, 0, 0), (1, 1, 1)), 1.0)
        tensor.apply_observation((0, 0, 0), {"occupancy": 1.0})
        snapshot = tensor.snapshot()
        snapshot["channels"] = 10**9
        with self.assertRaisesRegex(ValueError, "channels must be 9"):
            SpaceTensor.from_snapshot(snapshot)

    def test_sparse_tensor_caps_allocated_cells_and_does_not_consume_failed_event(self):
        tensor = SpaceTensor(((0, 0, 0), (2, 1, 1)), 1.0, eager=False)
        tensor.apply_observation((0, 0, 0), {"occupancy": 1.0})
        with patch("spatial_mesh.space_tensor._MAX_EAGER_CELLS", 1):
            with self.assertRaisesRegex(ValueError, "sparse tensor exceeds"):
                tensor.observe(Observation((1.5, 0.5, 0.5), 1.0, event_id="retryable"))
        self.assertNotIn("retryable", tensor.seen_event_ids)

    def test_sparse_zero_confidence_observations_do_not_consume_cells_or_event_ids(self):
        tensor = SpaceTensor(((0, 0, 0), (2, 1, 1)), 1.0, eager=False)
        with patch("spatial_mesh.space_tensor._MAX_EAGER_CELLS", 1):
            tensor.observe(Observation((0.5, 0.5, 0.5), 1.0, confidence=0.0, event_id="ignored"))
            self.assertEqual(tensor.cells, {})
            self.assertNotIn("ignored", tensor.seen_event_ids)
            tensor.observe(Observation((1.5, 0.5, 0.5), 1.0, confidence=1.0, event_id="accepted"))
        self.assertEqual(set(tensor.cells), {(1, 0, 0)})
        self.assertIn("accepted", tensor.seen_event_ids)

    def test_malformed_observation_is_atomic_and_keeps_event_retryable(self):
        tensor = SpaceTensor(((0, 0, 0), (1, 1, 1)), 1.0)
        bad = Observation((0.5, 0.5, 0.5), 1.0, object_id="not-an-int", event_id="retry")
        with self.assertRaises(ValueError):
            tensor.observe(bad)
        cell = tensor.cell((0, 0, 0))
        self.assertEqual(cell.observation_count, 0)
        self.assertEqual(cell.occupancy, 0.0)
        self.assertNotIn("retry", tensor.seen_event_ids)
        tensor.observe(Observation((0.5, 0.5, 0.5), 1.0, object_id=7, event_id="retry"))
        self.assertEqual(cell.observation_count, 1)

    def test_event_id_history_has_a_hard_limit(self):
        tensor = SpaceTensor(((0, 0, 0), (1, 1, 1)), 1.0)
        with patch("spatial_mesh.space_tensor._MAX_SEEN_EVENT_IDS", 1):
            tensor.observe(Observation((0.5, 0.5, 0.5), 1.0, event_id="first"))
            with self.assertRaisesRegex(ValueError, "event ID history"):
                tensor.observe(Observation((0.5, 0.5, 0.5), 0.0, event_id="second"))
        self.assertEqual(tensor.seen_event_ids, {"first"})
        self.assertEqual(tensor.cell((0, 0, 0)).observation_count, 1)

    def test_wave_sensor_observations_update_the_hit_cells(self):
        tensor = SpaceTensor(((-3, -3, 0), (3, 3, 2)), 1.0, eager=False)
        system = WaveBasedSpaceOS(cell_size=1.0)
        observations = [
            SensorObservation((-2.5, -2.5, 0.5), (1.0, 0.0, 0.0), 1.0, "wall", "brick", (0.0, 0.0, 0.0), 0.1, modality="wifi"),
            SensorObservation((2.5, 2.5, 0.5), (-1.0, 0.0, 0.0), 1.0, "door", "wood", (0.0, 0.0, 0.0), 0.2, modality="uwb"),
        ]
        system.store_observations(observations, tensor)
        self.assertEqual(set(tensor.cells), {(1, 0, 0), (4, 5, 0)})
        self.assertEqual(tensor.cell((1, 0, 0)).object_type, "wall")
        self.assertEqual(tensor.cell((4, 5, 0)).object_type, "door")

    def test_sensor_scan_uses_modality_specific_wave_speed(self):
        class OneHitWorld:
            def raycast(self, origin, direction, max_range):
                return [{"distance": 1.0, "reflectivity": 0.5}]

        observations = SensorSuite(sensor_types=("ultrasound", "wifi")).scan(OneHitWorld(), 1, 1)
        self.assertAlmostEqual(observations[0].timestamp, 1.0 / 343.0)
        self.assertAlmostEqual(observations[1].timestamp, 1.0 / 299_792_458.0)
        self.assertEqual(observations[0].modality, "ultrasound")
        self.assertEqual(observations[1].modality, "wifi")


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
