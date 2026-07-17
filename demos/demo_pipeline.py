from __future__ import annotations

import os
import csv
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from spatial_mesh.spaxel import Spaxel
from spatial_mesh.space_tensor import SpaceTensor
from spatial_mesh.world import World
from spatial_mesh.environment import EnvironmentObject
from spatial_mesh.semos import SpaceOperatingSystem
from spatial_mesh.navigation import AStarNavigation


@dataclass
class Episode:
    episode_id: int
    objects: List[Dict[str, Any]] = field(default_factory=list)
    sensor_origin: Tuple[float, float, float] = (0.0, 0.0, 0.3)
    sensor_types: List[str] = field(default_factory=lambda: ["ultrasound", "wifi", "uwb"])
    episode_label: str = ""

    def to_dict(self):
        return asdict(self)


def build_default_world(world_size=3.0, cell_size=0.1):
    half = float(world_size) / 2.0
    bounds = ((-half, -half, 0.0), (half, half, 2.5))
    world = World(bounds=bounds, cell_size=cell_size)
    floor = EnvironmentObject(1, "floor", (0.0, 0.0, 0.0), (world_size, world_size, 0.1), material="wood")
    world.add_object(floor)
    wall = EnvironmentObject(2, "wall", (half - 0.5, 0.0, 1.25), (0.1, world_size, 2.5), material="wood")
    world.add_object(wall)
    table = EnvironmentObject(3, "table", (0.0, 0.0, 0.4), (1.2, 0.6, 0.05), material="wood")
    world.add_object(table)
    human = EnvironmentObject(4, "human", (0.0, 0.0, 0.8), (0.6, 0.4, 0.7), material="human", moving=True, velocity=(0.05, 0.02, 0.0))
    world.add_object(human)
    cup = EnvironmentObject(5, "cup", (0.0, 0.0, 0.75), (0.08, 0.08, 0.1), material="ceramic")
    world.add_object(cup)
    glass = EnvironmentObject(6, "glass_wall", (-half + 0.05, 0.0, 1.25), (0.05, world_size, 2.0), material="glass")
    world.add_object(glass)
    return world


def generate_dataset_episodes(output_path="data/spaxel_sets/episodes.csv", count=64):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    for episode_id in range(1, count + 1):
        world = build_default_world()
        world.build()
        entry = Episode(
            episode_id=episode_id,
            episode_label=f"episode-{episode_id}",
            sensor_origin=(0.0, 0.0, 0.3),
            sensor_types=["ultrasound", "wifi", "uwb"],
            objects=[{"id": obj.id, "kind": obj.kind, "center": obj.center, "size": obj.size, "material": obj.material} for obj in world.objects],
        )
        rows.append(entry.to_dict())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode_id", "episode_label", "sensor_origin", "sensor_types", "objects"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "episode_id": row["episode_id"],
                "episode_label": row["episode_label"],
                "sensor_origin": json.dumps(row["sensor_origin"]),
                "sensor_types": json.dumps(row["sensor_types"]),
                "objects": json.dumps(row["objects"]),
            })
    return rows


def visualize_world(world, output_name="world_grid.png"):
    fig = plt.figure(figsize=(12, 8))
    nodes = list(world.tensor.occupied_cells())
    if nodes:
        pts = np.array([(cell.x, cell.y, cell.z) for _, cell in nodes])
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Spaxelt Occupancy")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2)
    path = f"data/spaxel_sets/{output_name}"
    plt.savefig(path)
    plt.close()
    return path


def visualize_room_scene(world, output_name="room_scene.png"):
    fig = plt.figure(figsize=(12, 8))
    ax3d = fig.add_subplot(121, projection="3d")
    ax3d.set_title("Spaxel Scene")
    for obj in world.objects:
        indices = obj.voxel_indices(world.tensor)
        points = [world.tensor.cell(idx) for idx in indices if world.tensor.cell(idx)]
        if points:
            pts = np.array([(p.x, p.y, p.z) for p in points])
            ax3d.scatter(pts[::10, 0], pts[::10, 1], pts[::10, 2], s=2, label=obj.kind)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.legend(loc="upper left", fontsize=6)
    ax = fig.add_subplot(122)
    heatmap = np.zeros((world.tensor.shape[1], world.tensor.shape[0]))
    for idx, cell in world.tensor.cells.items():
        ix, iy, _ = idx
        if 0 <= ix < heatmap.shape[1] and 0 <= iy < heatmap.shape[0]:
            heatmap[iy, ix] = max(heatmap[iy, ix], cell.occupancy)
    ax.set_title("Occupancy Slice Z=0")
    ax.imshow(heatmap, cmap="viridis", origin="lower")
    plt.tight_layout()
    path = f"data/spaxel_sets/{output_name}"
    plt.savefig(path)
    plt.close()
    return path


def visualize_navigation(world, output_name="navigation.png"):
    world.build()
    nav = AStarNavigation(world.tensor)
    half = world.bounds[1][0] - 0.4
    start = (-half, -half, 0.1)
    goal = (half, half, 0.1)
    path = nav.plan(start, goal)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection="3d")
    for obj in world.objects:
        indices = obj.voxel_indices(world.tensor)
        points = [world.tensor.cell(idx) for idx in indices if world.tensor.cell(idx)]
        if points:
            pts = np.array([(p.x, p.y, p.z) for p in points])
            ax.scatter(pts[::10, 0], pts[::10, 1], pts[::10, 2], s=2)
    if path:
        pts = np.array(path)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], c="red", linewidth=3)
    ax.set_title("A* over Spaxels")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    path_out = f"data/spaxel_sets/{output_name}"
    plt.savefig(path_out)
    plt.close()
    return path_out


def visualize_scene_graph(scene, output_name="scene_graph.png"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    ax.set_title("Scene Graph")
    nodes = scene.current_objects or []
    if nodes:
        positions = np.array([node.position for node in nodes])
        ax.scatter(positions[::5, 0], positions[::5, 1], positions[::5, 2], s=60, c="steelblue")
        for node in nodes:
            ax.text(node.position[0], node.position[1], node.position[2], f" {node.label}", size=8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    path = f"data/spaxel_sets/{output_name}"
    plt.savefig(path)
    plt.close()
    return path


def visualize_predictions(scene, output_name="scene_predictions.png"):
    predictions = scene.predict(1.0)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    ax.set_title("Predicted Positions")
    nodes = scene.current_objects or []
    positions = np.array([node.position for node in nodes])
    predictions = np.array(predictions) if predictions else np.zeros((0, 3))
    if predictions.size:
      ax.scatter(predictions[:, 0], predictions[:, 1], predictions[:, 2], s=60, c="red", marker="x", label="prediction")
    ax.scatter(positions[::5, 0], positions[::5, 1], positions[::5, 2], s=60, c="steelblue", label="current")
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    path = f"data/spaxel_sets/{output_name}"
    plt.savefig(path)
    plt.close()
    return path


def visualize_sensor_scans(world, output_name="sensor_scans.png"):
    world.build()
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection="3d")
    ax.set_title("Sensor Cached Ray Hits")
    origins = [(0.0, 0.0, 0.3), (1.0, 0.0, 0.3), (-1.0, 0.0, 0.3)]
    for i, origin in enumerate(origins):
        points = []
        for az_deg in range(-90, 91, 10):
            az = math.radians(az_deg)
            direction = (math.cos(az), math.sin(az), 0.0)
            samples = world.raycast(origin, direction, max_range=6.0)
            for sample in samples:
                distance = sample.get("distance", 0.0)
                if distance <= 0.0:
                    continue
                dx = direction[0] * distance
                dy = direction[1] * distance
                dz = direction[2] * distance
                points.append((origin[0] + dx, origin[1] + dy, origin[2] + dz))
        if points:
            pts = np.array(points)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=6, label=f"origin-{i}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    path = f"data/spaxel_sets/{output_name}"
    plt.savefig(path)
    plt.close()
    return path


def generate_all(output_dir="data/spaxel_sets"):
    os.makedirs(output_dir, exist_ok=True)
    world = build_default_world()
    world.build()
    generate_dataset_episodes(f"{output_dir}/episodes.csv")
    captures = []
    captures.append(visualize_world(world))
    captures.append(visualize_room_scene(world))
    captures.append(visualize_navigation(world))
    scene = SpaceOperatingSystem()
    scene.ingest(world.tensor)
    captures.append(visualize_scene_graph(scene))
    captures.append(visualize_predictions(scene))
    captures.append(visualize_sensor_scans(world))
    return captures, world, scene


if __name__ == "__main__":
    captures, world, scene = generate_all()
    print("rendered:", captures)
    print("objects:", [obj.kind for obj in world.objects])
    print("scene nodes:", [node.label for node in scene.current_objects])
