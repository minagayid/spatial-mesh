from __future__ import annotations


class Spaxel:
    __slots__ = ("x", "y", "z", "occupancy", "confidence", "material", "temperature",
                 "motion", "reflectivity", "velocity", "object_id", "timestamp", "channels")

    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.occupancy: float = 0.0       # [0, 1]
        self.confidence: float = 0.0      # [0, 1]
        self.material: str = ""           # e.g. "wood", "metal", "human", "glass", ""
        self.temperature: float = 0.0     # arbitrary units / delta C
        self.motion: float = 0.0          # velocity magnitude proxy in cell
        self.reflectivity: float = 0.0    # [0, 1]
        self.velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.object_id: int = 0
        self.timestamp: float = 0.0
        # channels align with the SpaceTensor channel spec:
        # occupancy, reflectivity, motion, velocity_x, velocity_y, velocity_z,
        # temperature, material_conf, confidence
        self.channels: list[float] = [0.0] * 9

    def set_channel(self, channel: int, value: float) -> None:
        if 0 <= channel < len(self.channels):
            self.channels[channel] = float(value)

    def get_channel(self, channel: int) -> float:
        return self.channels[channel] if 0 <= channel < len(self.channels) else 0.0

    def as_dict(self) -> dict:
        return {
            "position": (self.x, self.y, self.z),
            "occupancy": self.occupancy,
            "confidence": self.confidence,
            "material": self.material,
            "temperature": self.temperature,
            "motion": self.motion,
            "reflectivity": self.reflectivity,
            "velocity": self.velocity,
            "object_id": self.object_id,
            "timestamp": self.timestamp,
            "channels": list(self.channels),
        }

    def __repr__(self) -> str:
        mat = self.material or "empty"
        return (f"Spaxel(({self.x:.2f},{self.y:.2f},{self.z:.2f}) "
                f"occ={self.occupancy:.2f} conf={self.confidence:.2f} mat={mat})")
