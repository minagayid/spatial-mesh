from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .space_tensor import SpaceTensor
from .environment import EnvironmentObject
from .raycasting import raycast


_REFLECTIVITY = {
    "metal": 0.95,
    "foil": 0.95,
    "wood": 0.6,
    "human": 0.55,
    "fabric": 0.35,
    "glass": 0.25,
    "plastic": 0.4,
    "default": 0.2,
}


class World:
    def __init__(self, bounds=((-3.0, -3.0, 0.0), (3.0, 3.0, 2.5)), cell_size=0.1, eager=False):
        self.bounds = tuple(tuple(float(v) for v in b) for b in bounds)
        self.cell_size = float(cell_size)
        if self.cell_size <= 0.0:
            raise ValueError("cell_size must be positive")
        self.tensor = SpaceTensor(bounds=self.bounds, cell_size=self.cell_size, eager=eager)
        self.objects: List[EnvironmentObject] = []

    def add_object(self, obj: EnvironmentObject) -> None:
        self.objects.append(obj)

    def add_objects(self, objects: Iterable[EnvironmentObject]) -> None:
        self.objects.extend(objects)

    def build(self) -> None:
        self.tensor.reset()
        for obj in self.objects:
            for idx in obj.voxel_indices(self.tensor):
                obs = {
                    "occupancy": 0.8,
                    "reflectivity": 1.0,
                    "confidence": 1.0,
                    "material": obj.material or obj.kind,
                    "object_id": obj.id,
                    "object_type": obj.kind,
                    "timestamp": float(obj.id),
                }
                if obj.temperature:
                    obs["temperature"] = obj.temperature
                if obj.moving:
                    obs["motion"] = 1.0
                    obs["velocity_x"] = obj.velocity[0]
                    obs["velocity_y"] = obj.velocity[1]
                    obs["velocity_z"] = obj.velocity[2]
                self.tensor.apply_observation(idx, obs)

    def raycast(self, origin, direction, max_range=20.0):
        result = raycast(origin, direction, self.objects)
        if not result:
            return []
        distance = float(result.distance)
        material = result.material or result.object_kind
        if distance > max_range:
            return []
        reflectivity = float(_REFLECTIVITY.get(material.lower(), _REFLECTIVITY["default"]))
        return [{
            "distance": distance,
            "object_kind": result.object_kind,
            "material": material,
            "reflectivity": reflectivity,
            "velocity": tuple(float(v) for v in result.velocity),
        }]
