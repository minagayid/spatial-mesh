from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from spaxels.src.spaxel import Spaxel
from spaxels.src.world import World
from spaxels.src.environment import EnvironmentObject
from spaxels.src.space_tensor import SpaceTensor
from spaxels.src.semos import SpaceOperatingSystem
from spaxels.src.navigation import AStarNavigation


def build_room(world_size=3.0, height=2.5):
    half = float(world_size) / 2.0
    bounds = ((-half, -half, 0.0), (half, half, height))
    world = World(bounds=bounds, cell_size=0.1)
    wall_thickness = 0.1
    floor_height = wall_thickness * 0.5
    floor_size = (world_size, world_size, floor_height)
    floor_center = (0.0, 0.0, -floor_height * 0.5)
    world.add_object(EnvironmentObject(1, "floor", floor_center, floor_size, material="wood"))
    center = (0.0, 0.0, 0.6)
    size = (0.7, 0.7, 0.6)
    world.add_object(EnvironmentObject(5, "human", center, size, material="human", moving=True, velocity=(0.05, 0.02, 0.0)))
    table_center = (0.0, 0.0, 0.45)
    table_size = (1.2, 0.6, 0.05)
    world.add_object(EnvironmentObject(2, "table", table_center, table_size, material="wood"))
    cup_center = (0.0, 0.0, 0.75)
    cup_size = (0.08, 0.08, 0.1)
    world.add_object(EnvironmentObject(3, "cup", cup_center, cup_size, material="ceramic"))
    wall_center = (half - 0.5, 0.0, 1.25)
    wall_size = (wall_thickness, world_size, height)
    world.add_object(EnvironmentObject(4, "wall", wall_center, wall_size, material="wood"))
    return world


def visualize_world(world):
    occupied = [cell for _, cell in world.tensor.occupied_cells()]
    positions = np.array([(cell.x, cell.y, cell.z) for cell in occupied])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Spaxel Grid")
    if positions.size:
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig("data/spaxel_sets/world_grid.png")
    plt.close()


def visualize_room_scene():
    world = build_room()
    world.build()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(121, projection="3d")
    ax.set_title("Spaxel Occupancy")
    for obj in world.objects:
        mn, _ = world.bounds
        cs = world.cell_size
        indices = obj.voxel_indices(world.tensor)
        points = [world.tensor.cell(idx) for idx in indices if world.tensor.cell(idx)]
        if points:
            pts = np.array([(p.x, p.y, p.z) for p in points])
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, label=obj.kind)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left", fontsize=6)
    ax = fig.add_subplot(122)
    heatmap = np.zeros((world.tensor.shape[1], world.tensor.shape[0]))
    for idx, cell in world.tensor.cells.items():
        ix, iy, _ = idx
        if 0 <= ix < heatmap.shape[1] and 0 <= iy < heatmap.shape[0]:
            heatmap[iy, ix] = max(heatmap[iy, ix], cell.occupancy)
    ax.set_title("Occupancy Slice z=0")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    im = ax.imshow(heatmap, cmap="viridis", origin="lower")
    plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    plt.savefig("data/spaxel_sets/room_scene.png")
    plt.close()


def visualize_navigation(world):
    world.build()
    nav = AStarNavigation(world.tensor)
    half = world.bounds[0][0] + world.bounds[1][0] / 2.0
    half = world.bounds[1][0] / 2.0
    start = (-half + 0.4, -half + 0.4, 0.1)
    goal = (half - 0.4, half - 0.4, 0.1)
    path = nav.plan(start, goal)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    for obj in world.objects:
        indices = obj.voxel_indices(world.tensor)
        points = [world.tensor.cell(idx) for idx in indices if world.tensor.cell(idx)]
        if points:
            pts = np.array([(p.x, p.y, p.z) for p in points])
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2)
    if path:
        pts = np.array(path)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], c="red", linewidth=2)
    ax.set_title("Spaxel Path")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig("data/spaxel_sets/navigation.png")
    plt.close()


def visualize_scene_graph(scene):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    positions = [node.position for node in scene.current_objects if node.position]
    if positions:
        pts = np.array(positions)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=40)
    for node in scene.current_objects:
        if node.position:
            ax.text(node.position[0], node.position[1], node.position[2], node.label, size=8)
    ax.set_title("Scene Graph")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig("data/spaxel_sets/scene_graph.png")
    plt.close()


def generate_all():
    world = build_room()
    visualize_room_scene()
    visualize_navigation(world)
    scene = SpaceOperatingSystem()
    scene.ingest(world.tensor)
    visualize_scene_graph(scene)


if __name__ == "__main__":
    generate_all()
    print("Plots saved to data/spaxel_sets")
