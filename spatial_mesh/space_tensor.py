from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Iterable, Optional, List, Dict, Any
import math
import hashlib
import json
from .spaxel import Spaxel
from .fusion import Observation, OccupancyFusion


_OCCUPANCY_FUSION = OccupancyFusion()
_MAX_EAGER_CELLS = 250_000
_CHANNEL_COUNT = 9
_MAX_DENSE_VALUES = _MAX_EAGER_CELLS * 9
_MAX_SEEN_EVENT_IDS = 100_000
_MAX_EVENT_ID_CHARS = 200
_MAX_CELL_OBSERVATIONS = 2**53


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
    occupancy_initialized: bool = False
    initialized_channels: set[str] = field(default_factory=set)
    motion_phoniness: float = 0.0  # ponytail: retained for compatibility, unused

    def integrate(self, field_key: str, value: float, weight: float) -> None:
        if weight <= 0.0:
            return
        if field_key not in self.initialized_channels:
            setattr(self, field_key, value)
            self.initialized_channels.add(field_key)
            return
        current = float(getattr(self, field_key, 0.0))
        blended = (1.0 - weight) * current + weight * value
        setattr(self, field_key, blended)

    def apply_observation(self, observation: Dict[str, Any]) -> bool:
        if not observation:
            return False
        if isinstance(self.observation_count, bool) or not isinstance(self.observation_count, int) or self.observation_count < 0:
            raise ValueError("observation_count must be a non-negative integer")
        if self.observation_count >= _MAX_CELL_OBSERVATIONS:
            raise ValueError("cell observation count exceeds the supported limit")
        confidence = _clamp(observation.get("confidence", 1.0))
        uncertainty = _clamp(observation.get("uncertainty", 0.0))
        weight = min(0.9, confidence * (1.0 - uncertainty) * 0.9)
        if weight <= 0.0:
            return False

        occupancy = _clamp(observation["occupancy"]) if "occupancy" in observation else None
        bounded_channels = {
            key: _clamp(observation[key])
            for key in ("reflectivity", "motion", "confidence", "material_conf")
            if key in observation
        }
        temperature = float(observation["temperature"]) if "temperature" in observation else None
        velocities = {
            f"velocity_{axis}": float(observation[f"velocity_{axis}"])
            for axis in ("x", "y", "z")
            if f"velocity_{axis}" in observation
        }
        object_id = int(observation["object_id"]) if "object_id" in observation else None
        timestamp = float(observation["timestamp"]) if "timestamp" in observation else None
        modality = str(observation.get("modality", self.last_modality or "unknown"))
        material = str(observation["material"]) if observation.get("material") else None
        object_type = str(observation["object_type"]) if observation.get("object_type") else None

        # Validate every value that could raise before mutating this cell. This
        # keeps a failed event retryable without double-applying partial state.
        if temperature is not None and not math.isfinite(temperature):
            temperature = None
        velocities = {key: value for key, value in velocities.items() if math.isfinite(value)}
        if timestamp is not None and not math.isfinite(timestamp):
            timestamp = None

        self.observation_count += 1
        self.uncertainty += (uncertainty - self.uncertainty) / self.observation_count
        self.last_modality = modality
        if occupancy is not None:
            prior = self.occupancy if self.occupancy_initialized else 0.5
            self.occupancy = _OCCUPANCY_FUSION.update(prior, occupancy, confidence, uncertainty)
            self.occupancy_initialized = True
        for key, value in bounded_channels.items():
            self.integrate(key, value, weight)
        if temperature is not None:
            self.integrate("temperature", temperature, weight)
        for key, velocity in velocities.items():
            self.integrate(key, velocity, weight)
        if material is not None:
            self.material = material
        if object_id is not None:
            self.object_id = object_id
        if object_type is not None:
            self.object_type = object_type
        if timestamp is not None:
            self.timestamp = max(self.timestamp, timestamp)
        return True

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
        if not isinstance(self.eager, bool):
            raise ValueError("eager must be a boolean")
        if isinstance(self.channels, bool) or not isinstance(self.channels, int) or self.channels != _CHANNEL_COUNT:
            raise ValueError(f"channels must be {_CHANNEL_COUNT} for the current cell schema")
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
        if self.eager and math.prod(self.shape) > _MAX_EAGER_CELLS:
            raise ValueError(f"eager grid exceeds {_MAX_EAGER_CELLS} cells; use eager=False or larger cells")
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
        if not self.valid_index(idx):
            return None
        return self.cells.get(idx)

    def _create_cell(self, idx: Tuple[int, int, int]) -> SpaceTensorCell:
        return SpaceTensorCell(
            x=self.min_bound[0] + (idx[0] + 0.5) * self.cell_size,
            y=self.min_bound[1] + (idx[1] + 0.5) * self.cell_size,
            z=self.min_bound[2] + (idx[2] + 0.5) * self.cell_size,
        )

    def _new_cell(self, idx: Tuple[int, int, int]) -> SpaceTensorCell:
        cell = self._create_cell(idx)
        self.cells[idx] = cell
        return cell

    def valid_index(self, idx: Tuple[int, int, int]) -> bool:
        return (
            isinstance(idx, tuple)
            and len(idx) == 3
            and all(
                isinstance(value, int) and not isinstance(value, bool) and 0 <= value < self.shape[axis]
                for axis, value in enumerate(idx)
            )
        )

    def index_of(self, position: Tuple[float, float, float]) -> Optional[Tuple[int, int, int]]:
        if not isinstance(position, (tuple, list)) or len(position) != 3:
            return None
        try:
            coordinates = tuple(float(value) for value in position)
        except (OverflowError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in coordinates):
            return None
        if any(coordinates[i] < self.min_bound[i] or coordinates[i] >= self.max_bound[i] for i in range(3)):
            return None
        idx = tuple(int(math.floor((coordinates[i] - self.min_bound[i]) / self.cell_size)) for i in range(3))
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

    def apply_observation(self, idx: Tuple[int, int, int], observation: Dict[str, Any]) -> bool:
        if not self.valid_index(idx):
            return False
        cell = self.cells.get(idx)
        if cell is not None:
            return cell.apply_observation(observation)
        candidate = self._create_cell(idx)
        if not candidate.apply_observation(observation):
            return False
        if len(self.cells) >= _MAX_EAGER_CELLS:
            raise ValueError(f"sparse tensor exceeds {_MAX_EAGER_CELLS} allocated cells")
        self.cells[idx] = candidate
        return True

    def observe(self, observation: Observation) -> Optional[Tuple[int, int, int]]:
        idx = self.index_of(observation.position)
        if idx is None:
            return None
        event_id = observation.event_id
        if event_id:
            if not isinstance(event_id, str) or len(event_id) > _MAX_EVENT_ID_CHARS:
                raise ValueError(f"event_id must be a string of at most {_MAX_EVENT_ID_CHARS} characters")
            if observation.event_id in self.seen_event_ids:
                return idx
            if len(self.seen_event_ids) >= _MAX_SEEN_EVENT_IDS:
                raise ValueError(f"event ID history exceeds {_MAX_SEEN_EVENT_IDS} entries")
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
        applied = self.apply_observation(idx, payload)
        if event_id and applied:
            self.seen_event_ids.add(event_id)
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
            cell.occupancy_initialized = False
            cell.initialized_channels.clear()
        self.seen_event_ids.clear()

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable deterministic snapshot of observed cells."""
        return {
            "schema": "spatial-mesh.tensor.v1",
            "bounds": [list(self.min_bound), list(self.max_bound)],
            "cell_size": self.cell_size,
            "channels": self.channels,
            "eager": self.eager,
            "event_ids": sorted(self.seen_event_ids),
            "cells": [
                {"index": list(idx), **cell.to_spaxel().as_dict(), "uncertainty": cell.uncertainty,
                 "modality": cell.last_modality, "observation_count": cell.observation_count,
                 "occupancy_initialized": cell.occupancy_initialized,
                 "initialized_channels": sorted(cell.initialized_channels)}
                for idx, cell in sorted(
                    (item for item in self.cells.items() if item[1].observation_count > 0),
                    key=lambda item: item[0],
                )
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
        cell_size = float(snapshot["cell_size"])
        channels = snapshot.get("channels", 9)
        if "eager" not in snapshot:
            probe = cls(bounds, cell_size, channels, eager=False)
            eager = math.prod(probe.shape) <= _MAX_EAGER_CELLS
        else:
            raw_eager = snapshot["eager"]
            if not isinstance(raw_eager, bool):
                raise ValueError("snapshot eager must be a boolean")
            eager = raw_eager
        tensor = cls(bounds, cell_size, channels, eager=eager)
        entries = snapshot.get("cells", [])
        if not isinstance(entries, list) or len(entries) > _MAX_EAGER_CELLS:
            raise ValueError(f"snapshot cells exceeds {_MAX_EAGER_CELLS}")
        seen_indices = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("snapshot cell entries must be objects")
            raw_index = entry.get("index")
            if (
                not isinstance(raw_index, (list, tuple))
                or len(raw_index) != 3
                or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_index)
            ):
                raise ValueError("snapshot cell index must contain exactly three integers")
            idx = tuple(raw_index)
            if idx in seen_indices:
                raise ValueError(f"snapshot contains duplicate cell index: {idx}")
            seen_indices.add(idx)
            cell = tensor.cell(idx)
            if cell is None:
                if not tensor.valid_index(idx):
                    raise ValueError(f"snapshot cell index is outside tensor: {idx}")
                if len(tensor.cells) >= _MAX_EAGER_CELLS:
                    raise ValueError(f"snapshot sparse cells exceed {_MAX_EAGER_CELLS}")
                cell = tensor._new_cell(idx)
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
            raw_observation_count = entry.get("observation_count", 0)
            if (
                isinstance(raw_observation_count, bool)
                or not isinstance(raw_observation_count, int)
                or raw_observation_count <= 0
                or raw_observation_count > _MAX_CELL_OBSERVATIONS
            ):
                raise ValueError("snapshot observation_count must be a positive integer")
            cell.observation_count = raw_observation_count
            if "occupancy_initialized" in entry and not isinstance(entry["occupancy_initialized"], bool):
                raise ValueError("snapshot occupancy_initialized must be a boolean")
            cell.occupancy_initialized = entry.get("occupancy_initialized", cell.occupancy != 0.0)
            saved_channels = entry.get("initialized_channels")
            allowed_channels = {
                "reflectivity", "motion", "confidence", "material_conf", "temperature",
                "velocity_x", "velocity_y", "velocity_z",
            }
            if saved_channels is not None:
                if not isinstance(saved_channels, list) or any(
                    not isinstance(channel, str) or channel not in allowed_channels
                    for channel in saved_channels
                ):
                    raise ValueError("snapshot initialized_channels contains invalid values")
                cell.initialized_channels.update(saved_channels)
            elif cell.observation_count > 0:
                # Older snapshots did not record channel initialization state.
                cell.initialized_channels.update({
                    "reflectivity", "motion", "confidence", "material_conf", "temperature",
                    "velocity_x", "velocity_y", "velocity_z",
                })
        event_ids = snapshot.get("event_ids", [])
        if not isinstance(event_ids, list) or len(event_ids) > _MAX_SEEN_EVENT_IDS:
            raise ValueError("snapshot event_ids exceeds the supported limit")
        if any(not isinstance(event_id, str) or len(event_id) > _MAX_EVENT_ID_CHARS for event_id in event_ids):
            raise ValueError("snapshot event_ids contains invalid values")
        tensor.seen_event_ids.update(event_ids)
        return tensor

    def occupied_cells(self) -> Iterable[Tuple[Tuple[int, int, int], SpaceTensorCell]]:
        return ((idx, cell) for idx, cell in self.cells.items() if cell.occupancy > 0.01)

    def to_spaxels(self) -> List[Spaxel]:
        return [cell.to_spaxel() for cell in self.cells.values() if cell.occupancy > 0.01]

    def channel_tensor_shape(self) -> Tuple[int, int, int, int]:
        return (self.shape[0], self.shape[1], self.shape[2], self.channels)

    def empty_tensor(self) -> List[List[List[List[float]]]]:
        if math.prod(self.shape) * self.channels > _MAX_DENSE_VALUES:
            raise ValueError(
                f"dense tensor exceeds {_MAX_DENSE_VALUES} values; use sparse cell access instead"
            )
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
