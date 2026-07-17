from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .spaxel import Spaxel
from .space_tensor import SpaceTensor, SpaceTensorCell


@dataclass(frozen=True)
class EnvironmentObject:
    id: int
    kind: str          # wall, furniture, human, glass, fabric, metal
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    temperature: float = 0.0
    moving: bool = False
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    material: str = ""

    def voxel_indices(self, tensor: 'SpaceTensor') -> list[tuple[int, int, int]]:
        x0, y0, z0 = self.center
        hx, hy, hz = self.size
        x_min = x0 - hx / 2.0
        y_min = y0 - hy / 2.0
        z_min = z0 - hz / 2.0
        x_max = x0 + hx / 2.0
        y_max = y0 + hy / 2.0
        z_max = z0 + hz / 2.0
        indices: list[tuple[int, int, int]] = []
        mn, _ = tensor.bounds
        cs = tensor.cell_size
        for idx, cell in tensor.cells.items():
            ix, iy, iz = idx
            cx = mn[0] + (ix + 0.5) * cs
            cy = mn[1] + (iy + 0.5) * cs
            cz = mn[2] + (iz + 0.5) * cs
            if x_min <= cx <= x_max and y_min <= cy <= y_max and z_min <= cz <= z_max:
                indices.append(idx)
        return indices
