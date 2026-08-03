"""High-level offline facade for the spatial-mesh core."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

from .fusion import Observation
from .navigation import AStarNavigation
from .semos import SpaceOperatingSystem
from .space_tensor import SpaceTensor


class SpatialMesh:
    """Replayable perception pipeline with no device or network dependency."""

    def __init__(self, bounds=((-3.0, -3.0, 0.0), (3.0, 3.0, 2.5)), cell_size=0.1):
        self.tensor = SpaceTensor(bounds, cell_size)
        self.operating_system = SpaceOperatingSystem()

    @property
    def scene(self):
        return self.operating_system.scene

    def observe(self, observation: Observation):
        return self.tensor.observe(observation)

    def ingest(self, observations: Iterable[Observation], *, reset: bool = True):
        if reset:
            self.tensor.reset()
            self.tensor.seen_event_ids.clear()
        for observation in observations:
            self.observe(observation)
        return self.operating_system.ingest(self.tensor)

    def replay(self, observations: Iterable[Observation]):
        ordered = sorted(observations, key=lambda item: (float(item.timestamp), item.event_id))
        return self.ingest(ordered, reset=True)

    def plan(self, start: Tuple[float, float, float], goal: Tuple[float, float, float]):
        return AStarNavigation(self.tensor).plan(start, goal)

    def snapshot(self):
        return self.tensor.snapshot()

    def digest(self) -> str:
        return self.tensor.digest()

    @classmethod
    def from_snapshot(cls, snapshot):
        mesh = cls.__new__(cls)
        mesh.tensor = SpaceTensor.from_snapshot(snapshot)
        mesh.operating_system = SpaceOperatingSystem()
        mesh.operating_system.ingest(mesh.tensor)
        return mesh
