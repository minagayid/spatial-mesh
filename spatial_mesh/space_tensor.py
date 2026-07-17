from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Iterable, Optional, List, Callable, Dict, Any
import math
from .spaxel import Spaxel


@dataclass
class SpaceTensorCell:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    occupancy: float = 0.0
    reflectivity: float = 0.0
    motion: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0
    temperature: float = 0.0
    material_conf: float = 0.0
    confidence: float = 0.0
    material: str = ""
    object_id: int = 0
    timestamp: float = 0.0
    motion_phoniness: float = 0.0  # ponytail: retained for compatibility, unused

    def integrate(self, field_key: str, value: float, weight: float) -> None:
        current = float(getattr(self, field_key, 0.0))
        if current == 0.0:
            setattr(self, field_key, value)
        else:
            blended = (1.0 - weight) * current + weight * value
            setattr(self, field_key, blended)

    def apply_observation(self, observation: Dict[str, Any]) -> None:
        if not observation:
            return
        confidence = float(observation.get("confidence", 1.0))
        if confidence < 0.0:
            return
        for key in ("occupancy", "reflectivity", "motion", "temperature", "confidence"):
            if key in observation:
                self.integrate(key, float(observation[key]), confidence * 0.9)
        for axis in ("x", "y", "z"):
            key = f"velocity_{axis}"
            if key in observation:
                velocity = float(observation[key])
                current = float(getattr(self, key, 0.0))
                setattr(self, key, (1.0 - confidence * 0.9) * current + confidence * 0.9 * velocity)
        if "material" in observation and observation["material"]:
            self.material = observation["material"]
        if "object_id" in observation:
            self.object_id = int(observation["object_id"])
        if "timestamp" in observation:
            self.timestamp = float(observation["timestamp"])

    def channel_vector(self) -> List[float]:
        return [
            self.occupancy,
            self.reflectivity,
            self.motion,
            self.velocity_x,
            self.velocity_y,
            self.velocity_z,
            self.temperature,
            self.material_conf,
            self.confidence,
        ]

    def to_spaxel(self) -> Spaxel:
        s = Spaxel(self.x, self.y, self.z)
        s.occupancy = self.occupancy
        s.reflectivity = self.reflectivity
        s.motion = self.motion
        s.temperature = self.temperature
        s.object_id = self.object_id
        s.timestamp = self.timestamp
        s.material = self.material
        s.channels = self.channel_vector()
        if self.velocity_x or self.velocity_y or self.velocity_z:
            s.velocity = (self.velocity_x, self.velocity_y, self.velocity_z)
        return s


@dataclass
class SpaceTensor:
    bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    cell_size: float
    channels: int = 9
    cells: Dict[Tuple[int, int, int], SpaceTensorCell] = field(default_factory=dict)
    min_bound: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_bound: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    shape: Tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self) -> None:
        mn, mx = self.bounds
        self.min_bound = tuple(float(v) for v in mn)
        self.max_bound = tuple(float(v) for v in mx)
        self.shape = tuple(max(1, int(math.ceil((mx[i] - mn[i]) / self.cell_size))) for i in range(3))
        self.cells = {}
        # initialize grid
        self._init_cells()

    def _init_cells(self) -> None:
        mn = self.min_bound
        shape = self.shape
        size = self.cell_size
        for ix in range(shape[0]):
            for iy in range(shape[1]):
                for iz in range(shape[2]):
                    cx = mn[0] + (ix + 0.5) * size
                    cy = mn[1] + (iy + 0.5) * size
                    cz = mn[2] + (iz + 0.5) * size
                    self.cells[(ix, iy, iz)] = SpaceTensorCell(x=cx, y=cy, z=cz)

    def cell(self, idx: Tuple[int, int, int]) -> Optional[SpaceTensorCell]:
        return self.cells.get(idx)

    def apply_observation(self, idx: Tuple[int, int, int], observation: Dict[str, Any]) -> None:
        cell = self.cells.get(idx)
        if cell:
            cell.apply_observation(observation)

    def occupied_cells(self) -> Iterable[Tuple[Tuple[int, int, int], SpaceTensorCell]]:
        return ((idx, cell) for idx, cell in self.cells.items() if cell.occupancy > 0.01)

    def to_spaxels(self) -> List[Spaxel]:
        return [cell.to_spaxel() for cell in self.cells.values() if cell.occupancy > 0.01]

    def channel_tensor_shape(self) -> Tuple[int, int, int, int]:
        return (self.shape[0], self.shape[1], self.shape[2], self.channels)

    def empty_tensor(self) -> List[List[List[List[float]]]]:
        nx, ny, nz = self.shape
        return [[[[0.0] * self.channels for _ in range(nz)] for _ in range(ny)] for _ in range(nx)]

    def to_tensor(self) -> List[List[List[List[float]]]]:
        nx, ny, nz = self.shape
        tensor = self.empty_tensor()
        for idx, cell in self.cells.items():
            ix, iy, iz = idx
            tensor[ix][iy][iz] = cell.channel_vector()
        return tensor
