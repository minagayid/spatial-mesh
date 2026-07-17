from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spaxels.src.spaxel import Spaxel
from spaxels.src.space_tensor import SpaceTensor
from spaxels.src.world import World
from spaxels.src.environment import EnvironmentObject
from spaxels.src.raycasting import raycast
from spaxels.src.semos import SpaceOperatingSystem


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
            ix, iy, iz = self.world_to_idx((0.0, 0.0, 0.0))
            for idx in tensor.cells:
                tensor.apply_observation(idx, obs)
                break

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
