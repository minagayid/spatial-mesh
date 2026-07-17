from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SensorObservation:
    origin: Tuple[float, float, float]
    direction: Tuple[float, float, float]
    distance: float
    object_kind: str
    material: str
    velocity: Tuple[float, float, float]
    timestamp: float
    confidence: float = 0.9
    reflectivity: float = 0.5
    ray_power: float = 1.0


class SensorSuite:
    def __init__(self, sensor_types=("ultrasound", "wifi", "uwb"), origin=(0.0, 0.0, 0.3)):
        self.sensor_types = list(sensor_types)
        self.origin = tuple(float(v) for v in origin)
        self._directivity = {
            "ultrasound": 80.0,
            "wifi": 60.0,
            "uwb": 40.0,
            "mmwave": 30.0,
            "radar": 35.0,
        }
        self._max_range = {
            "ultrasound": 5.0,
            "wifi": 20.0,
            "uwb": 30.0,
            "mmwave": 25.0,
            "radar": 40.0,
        }

    def scan(self, world, azimuth_steps=4, elevation_steps=4):
        observations = []
        kinds = tuple(self.sensor_types) or ("ultrasound",)
        for kind in kinds:
            max_range = self._max_range.get(kind, 20.0)
            for az in np.linspace(-np.pi / 2, np.pi / 2, azimuth_steps):
                for el in np.linspace(-np.pi / 6, np.pi / 6, elevation_steps):
                    direction = np.array([
                        np.cos(el) * np.cos(az),
                        np.cos(el) * np.sin(az),
                        np.sin(el),
                    ], dtype=float)
                    samples = world.raycast(self.origin, tuple(direction), max_range=max_range)
                    for sample in samples:
                        obs = SensorObservation(
                            origin=self.origin,
                            direction=tuple(direction),
                            distance=sample["distance"],
                            object_kind=sample.get("object_kind", "unknown"),
                            material=sample.get("material", ""),
                            velocity=sample.get("velocity", (0.0, 0.0, 0.0)),
                            timestamp=float(sample.get("distance", 0.0) / 343.0),
                            confidence=self._confidence_for(sample),
                            reflectivity=sample.get("reflectivity", 0.5),
                            ray_power=sample.get("reflectivity", 0.5),
                        )
                        observations.append(obs)
        return observations

    def _confidence_for(self, sample):
        distance = float(sample.get("distance", 0.0))
        reflectivity = float(sample.get("reflectivity", 0.0))
        return max(0.05, min(0.99, 0.6 + 0.3 * reflectivity - 0.01 * distance))
