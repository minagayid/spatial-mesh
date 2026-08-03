"""Offline end-to-end example requiring only Python's standard library."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_mesh import AStarNavigation, EnvironmentObject, SpaceOperatingSystem, World


def main() -> None:
    world = World(bounds=((0, 0, 0), (4, 3, 1)), cell_size=1.0)
    world.add_object(EnvironmentObject(1, "wall", (2.0, 0.5, 0.5), (1.0, 1.0, 1.0), material="wood"))
    world.add_object(
        EnvironmentObject(2, "human", (0.5, 2.5, 0.5), (1.0, 1.0, 1.0), material="human", moving=True, velocity=(0.1, 0.0, 0.0))
    )
    world.build()
    scene = SpaceOperatingSystem()
    scene.ingest(world.tensor)
    path = AStarNavigation(world.tensor).plan((0.5, 0.5, 0.5), (3.5, 0.5, 0.5))
    print(json.dumps({
        "occupied_cells": sum(1 for _ in world.tensor.occupied_cells()),
        "scene_objects": len(scene.current_objects),
        "predictions": scene.predict(1.0),
        "path": path,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
