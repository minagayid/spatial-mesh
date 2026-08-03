from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .space_tensor import SpaceTensor


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
        return list(tensor.iter_indices_in_aabb((x_min, y_min, z_min), (x_max, y_max, z_max)))
