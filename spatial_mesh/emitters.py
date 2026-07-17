from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, List


class WaveForm:
    __slots__ = ('wave_type',)
    def __init__(self, wave_type='ultrasound'):
        # ultrasound, wifi, radar, uwb
        self.wave_type = wave_type


@dataclass
class ArrayEmitter:
    origin: Tuple[float, float, float]
    orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    frequency: float = 40_000.0   # Hz
    power: float = 1.0
    wave_type: str = 'ultrasound'
    fov_deg: float = 120.0

    def emit(self, tx_ns: float):
        # ponytail: real emission would compute beamforming weights
        return {'origin': self.origin, 'wave_type': self.wave_type, 'tx_ns': tx_ns, 'power': self.power}
