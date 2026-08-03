"""Small, deterministic uncertainty-aware occupancy fusion primitives.

The implementation deliberately uses only the Python standard library.  It is
intended for replaying simulated or recorded measurements locally; it does not
open devices or transmit RF/acoustic energy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    value = float(value)
    if not math.isfinite(value):
        return high if value == float("inf") else low
    return max(low, min(high, value))


@dataclass(frozen=True)
class Observation:
    """One normalized measurement in world coordinates."""

    position: Tuple[float, float, float]
    occupancy: float
    confidence: float = 1.0
    modality: str = "unknown"
    timestamp: float = 0.0
    uncertainty: float = 0.0
    reflectivity: float | None = None
    motion: float | None = None
    velocity: Tuple[float, float, float] | None = None
    temperature: float | None = None
    material: str = ""
    material_conf: float | None = None
    object_id: int = 0
    object_type: str = ""
    event_id: str = ""


class OccupancyFusion:
    """Bayesian log-odds update with bounded confidence and uncertainty.

    ``update`` accepts a prior probability and returns the posterior.  The
    measurement's influence is reduced by both low confidence and high
    uncertainty, then capped to keep replay numerically stable.
    """

    def __init__(self, prior: float = 0.5, max_log_odds: float = 20.0):
        self.prior = _clamp(prior)
        self.max_log_odds = max(1.0, float(max_log_odds))

    def update(
        self,
        prior: float,
        measurement: float,
        confidence: float = 1.0,
        uncertainty: float = 0.0,
    ) -> float:
        prior = _clamp(prior)
        measurement = _clamp(measurement)
        confidence = _clamp(confidence)
        uncertainty = _clamp(uncertainty)
        influence = min(1.0, confidence * (1.0 - uncertainty))
        if influence <= 0.0:
            return prior
        eps = 1e-6
        prior_logit = math.log(max(eps, prior) / max(eps, 1.0 - prior))
        measurement_logit = math.log(max(eps, measurement) / max(eps, 1.0 - measurement))
        posterior_logit = prior_logit + influence * measurement_logit
        posterior_logit = max(-self.max_log_odds, min(self.max_log_odds, posterior_logit))
        return 1.0 / (1.0 + math.exp(-posterior_logit))
