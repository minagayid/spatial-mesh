from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Iterable, Optional, List, Dict, Any
import math
import hashlib
import json
from .spaxel import Spaxel
from .fusion import Observation, OccupancyFusion


_OCCUPANCY_FUSION = OccupancyFusion()


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
    object_type: str = ""
    timestamp: float = 0.0
    last_modality: str = "unknown"
    uncertainty: float = 1.0
    observation_count: int = 0
    motion_phoniness: float = 0.0  # ponytail: retained for compatibility, unused

    def integrate(self, field_key: str, value: float, weight: float) -> None:
        if weight <= 0.0:
            return
        current = float(getattr(self, field_key, 0.0))
        if current == 0.0:
            setattr(self, field_key, value)
        else:
            blended = (1.0 - weight) * current + weight * value
            setattr(self, field_key, blended)

    def apply_observation(self, observation: Dict[str, Any]) -> None:
        if not observation:
            return
        prior_observations = self.observation_count
        confidence = _clamp(observation.get("confidence", 1.0))
        uncertainty = _clamp(observation.get("uncertainty", 0.0))
        weight = min(0.9, confidence * (1.0 - uncertainty) * 0.9)
        if weight <= 0.0:
            return
        self.observation_count += 1
        self.uncertainty = (self.uncertainty * (self.observation_count - 1) + uncertainty) / self.observation_count
        self.last_modality = str(observation.get("modality", self.last_modality or "unknown"))
        if "occupancy" in observation and weight > 0.0:
            measurement = _clamp(observation["occupancy"])
            self.occupancy = measurement if prior_observations == 0 else _OCCUPANCY_FUSION.update(
                self.occupancy, measurement, confidence, uncertainty
            )
        for key in ("reflectivity", "motion", "confidence", "material_conf"):
            if key in observation:
                self.integrate(key, _clamp(observation[key]), weight)
        if "temperature" in observation:
            temperature = float(observation["temperature"])
            if math.isfinite(temperature):
                self.integrate("temperature", temperature, weight)
        for axis in ("x", "y", "z"):
            key = f"velocity_{axis}"
            if key in observation:
                velocity = float(observation[key])
                if not math.isfinite(velocity):
                    continue
                current = float(getattr(self, key, 0.0))
                if prior_observations == 0:
                    setattr(self, key, velocity)
                else:
                    setattr(self, key, (1.0 - weight) * current + weight * velocity)
        if "material" in observation and observation["material"]:
            self.material = observation["material"]
        if "object_id" in observation:
            self.object_id = int(observation["object_id"])
        if "object_type" in observation and observation["object_type"]:
            self.object_type = str(observation["object_type"])
        if "timestamp" in observation:
            timestamp = float(observation["timestamp"])
            if math.isfinite(timestamp):
                self.timestamp = max(self.timestamp, timestamp)

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
        s.confidence = self.confidence
        s.temperature = self.temperature
        s.velocity = (self.velocity_x, self.velocity_y, self.velocity_z)
        s.object_id = self.object_id
        s.object_type = self.object_type
        s.timestamp = self.timestamp
        s.material = self.material
        s.channels = self.channel_vector()
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
    eager: bool = True
    seen_event_ids: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        mn, mx = self.bounds
        if not math.isfinite(float(self.cell_size)) or self.cell_size <= 0.0:
            raise ValueError("cell_size must be positive")
        self.min_bound = tuple(float(v) for v in mn)
        self.max_bound = tuple(float(v) for v in mx)
        if not all(math.isfinite(value) for value in (*self.min_bound, *self.max_bound)):
            raise ValueError("bounds must be finite")
        if any(self.max_bound[i] <= self.min_bound[i] for i in range(3)):
            raise ValueError("bounds must have increasing min and max coordinates")
        self.shape = tuple(max(1, int(math.ceil((self.max_bound[i] - self.min_bound[i]) / self.cell_size))) for i in range(3))
        self.cells = {}
        if self.eager:
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

    def _new_cell(self, idx: Tuple[int, int, int]) -> SpaceTensorCell:
        cell = SpaceTensorCell(
            x=self.min_bound[0] + (idx[0] + 0.5) * self.cell_size,
            y=self.min_bound[1] + (idx[1] + 0.5) * self.cell_size,
            z=self.min_bound[2] + (idx[2] + 0.5) * self.cell_size,
        )
        self.cells[idx] = cell
        return cell

    def valid_index(self, idx: Tuple[int, int, int]) -> bool:
        return all(0 <= int(idx[i]) < self.shape[i] for i in range(3))

    def index_of(self, position: Tuple[float, float, float]) -> Optional[Tuple[int, int, int]]:
        if not all(math.isfinite(float(value)) for value in position):
            return None
        idx = tuple(int(math.floor((float(position[i]) - self.min_bound[i]) / self.cell_size)) for i in range(3))
        return idx if self.valid_index(idx) else None

    def iter_indices_in_aabb(self, minimum, maximum) -> Iterable[Tuple[int, int, int]]:
        """Yield valid cell indices intersecting a world-coordinate AABB."""
        starts = tuple(
            max(0, int(math.ceil((float(minimum[i]) - self.min_bound[i]) / self.cell_size - 0.5)))
            for i in range(3)
        )
        stops = tuple(
            min(self.shape[i] - 1, int(math.floor((float(maximum[i]) - self.min_bound[i]) / self.cell_size - 0.5)))
            for i in range(3)
        )
        for ix in range(starts[0], stops[0] + 1):
            for iy in range(starts[1], stops[1] + 1):
                for iz in range(starts[2], stops[2] + 1):
                    yield (ix, iy, iz)

    def apply_observation(self, idx: Tuple[int, int, int], observation: Dict[str, Any]) -> None:
        if self.valid_index(idx):
            cell = self.cells.get(idx)
            if cell is None:
                cell = self._new_cell(idx)
            cell.apply_observation(observation)

    def observe(self, observation: Observation) -> Optional[Tuple[int, int, int]]:
        idx = self.index_of(observation.position)
        if idx is None:
            return None
        if observation.event_id:
            if observation.event_id in self.seen_event_ids:
                return idx
            self.seen_event_ids.add(observation.event_id)
        payload: Dict[str, Any] = {
            "occupancy": observation.occupancy,
            "confidence": observation.confidence,
            "modality": observation.modality,
            "timestamp": observation.timestamp,
            "uncertainty": observation.uncertainty,
            "material": observation.material,
            "object_id": observation.object_id,
            "object_type": observation.object_type,
            "event_id": observation.event_id,
        }
        for key, value in (("reflectivity", observation.reflectivity), ("motion", observation.motion),
                           ("temperature", observation.temperature), ("material_conf", observation.material_conf)):
            if value is not None:
                payload[key] = value
        if observation.velocity is not None:
            for axis, value in zip(("x", "y", "z"), observation.velocity):
                payload[f"velocity_{axis}"] = value
        self.apply_observation(idx, payload)
        return idx

    def reset(self) -> None:
        """Clear observations while retaining the allocated grid geometry."""
        if not self.eager:
            self.cells.clear()
            self.seen_event_ids.clear()
            return
        for cell in self.cells.values():
            cell.occupancy = 0.0
            cell.reflectivity = 0.0
            cell.motion = 0.0
            cell.velocity_x = cell.velocity_y = cell.velocity_z = 0.0
            cell.temperature = 0.0
            cell.material_conf = 0.0
            cell.confidence = 0.0
            cell.material = ""
            cell.object_id = 0
            cell.object_type = ""
            cell.timestamp = 0.0
            cell.last_modality = "unknown"
            cell.uncertainty = 1.0
            cell.observation_count = 0
        self.seen_event_ids.clear()

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable deterministic snapshot of occupied cells."""
        return {
            "schema": "spatial-mesh.tensor.v1",
            "bounds": [list(self.min_bound), list(self.max_bound)],
            "cell_size": self.cell_size,
            "channels": self.channels,
            "event_ids": sorted(self.seen_event_ids),
            "cells": [
                {"index": list(idx), **cell.to_spaxel().as_dict(), "uncertainty": cell.uncertainty,
                 "modality": cell.last_modality, "observation_count": cell.observation_count}
                for idx, cell in sorted(self.occupied_cells())
            ],
        }

    def digest(self) -> str:
        canonical = json.dumps(self.snapshot(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @classmethod
    def from_snapshot(cls, snapshot: Dict[str, Any]) -> "SpaceTensor":
        if snapshot.get("schema") != "spatial-mesh.tensor.v1":
            raise ValueError("unsupported tensor snapshot schema")
        bounds = tuple(tuple(float(v) for v in bound) for bound in snapshot["bounds"])
        tensor = cls(bounds, float(snapshot["cell_size"]), int(snapshot.get("channels", 9)))
        for entry in snapshot.get("cells", []):
            idx = tuple(int(v) for v in entry["index"])
            cell = tensor.cell(idx)
            if cell is None:
                raise ValueError(f"snapshot cell index is outside tensor: {idx}")
            cell.occupancy = _clamp(entry.get("occupancy", 0.0))
            cell.confidence = _clamp(entry.get("confidence", 0.0))
            cell.reflectivity = _clamp(entry.get("reflectivity", 0.0))
            cell.motion = _clamp(entry.get("motion", 0.0))
            cell.temperature = float(entry.get("temperature", 0.0))
            velocities = tuple(float(v) for v in entry.get("velocity", (0.0, 0.0, 0.0)))
            if not all(math.isfinite(value) for value in (cell.temperature, *velocities)):
                raise ValueError("snapshot contains non-finite continuous values")
            cell.velocity_x, cell.velocity_y, cell.velocity_z = velocities
            cell.material = str(entry.get("material", ""))
            cell.object_id = int(entry.get("object_id", 0))
            cell.object_type = str(entry.get("object_type", ""))
            cell.timestamp = float(entry.get("timestamp", 0.0))
            if not math.isfinite(cell.timestamp):
                raise ValueError("snapshot contains a non-finite timestamp")
            cell.uncertainty = _clamp(entry.get("uncertainty", 1.0))
            cell.last_modality = str(entry.get("modality", "unknown"))
            cell.observation_count = int(entry.get("observation_count", 0))
        tensor.seen_event_ids.update(str(event_id) for event_id in snapshot.get("event_ids", []))
        return tensor

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


def _clamp(value: Any) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0 if value != float("inf") else 1.0
    return max(0.0, min(1.0, value))
