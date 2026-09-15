from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

from .spaxel import Spaxel
from .space_tensor import SpaceTensor
from .world import World
from .environment import EnvironmentObject
from .raycasting import raycast
from .semos import SpaceOperatingSystem


class WaveBasedSpaceOS:
    def __init__(self, sensor_types=("ultrasound", "wifi", "uwb"), cell_size=0.25):
        self.cell_size = float(cell_size)
        self.world_model = SpaceTensor(bounds=self._default_bounds(), cell_size=self.cell_size)
        self.semos = SpaceOperatingSystem()

    def _default_bounds(self):
        return ((-3.0, -3.0, 0.0), (3.0, 3.0, 2.0))

    def world_to_idx(self, position):
        mn = self.world_model.min_bound
        cs = self.world_model.cell_size
        return tuple(int((v - m) / cs) for v, m in zip(position, mn))

    def store_observations(self, observations, tensor=None):
        if tensor is None:
            tensor = self.world_model
        for obs in observations:
            if not isinstance(obs, dict):
                origin = tuple(float(value) for value in obs.origin)
                direction = tuple(float(value) for value in obs.direction)
                distance = float(obs.distance)
                position = tuple(origin[axis] + direction[axis] * distance for axis in range(3))
                observation = {
                    "occupancy": 1.0,
                    "confidence": obs.confidence,
                    "uncertainty": 0.0,
                    "reflectivity": obs.reflectivity,
                    "material": obs.material,
                    "object_type": obs.object_kind,
                    "timestamp": obs.timestamp,
                    "velocity_x": obs.velocity[0],
                    "velocity_y": obs.velocity[1],
                    "velocity_z": obs.velocity[2],
                    "modality": obs.modality,
                }
            else:
                position = tuple(float(value) for value in obs["position"])
                observation = dict(obs)

            idx = tensor.index_of(position)
            if idx is not None:
                tensor.apply_observation(idx, observation)

    def predict_trajectory(self, tracer):
        f = tracer.get_world_model()
        if f is None:
            return []
        out = []
        t = 0.0
        for _ in range(12):
            position = (math.sin(t) * 1.2, math.cos(t) * 1.2, 0.8)
            out.append(position)
            t += 0.5
        return out


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("demos", exist_ok=True)
