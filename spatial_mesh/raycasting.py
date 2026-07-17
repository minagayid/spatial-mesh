from __future__ import annotations

import math
from typing import Optional

from .spaxel import Spaxel


class RayResult:
    def __init__(self, origin, direction, distance, object_kind, material, velocity, timestamp):
        self.origin = origin
        self.direction = direction
        self.distance = float(distance)
        self.object_kind = object_kind
        self.material = material
        self.velocity = velocity
        self.timestamp = float(timestamp)

    def hit_position(self):
        x, y, z = self.origin
        dx, dy, dz = self.direction
        d = self.distance
        return (x + dx * d, y + dy * d, z + dz * d)


def raycast(origin, direction, objects):
    ox, oy, oz = origin
    dx, dy, dz = _normalize(direction)
    best = None
    best_t = float('inf')
    for obj in objects:
        cx, cy, cz = obj.center
        hx, hy, hz = obj.size
        t = _aabb_intersect((ox, oy, oz), (dx, dy, dz), (cx, cy, cz), (hx, hy, hz))
        if t is not None and 0.0 < t < best_t:
            best_t = t
            best = obj
    if best is None:
        return None
    hit = best.center
    mat = best.material or best.kind
    vel = best.velocity
    return RayResult(origin, (dx, dy, dz), best_t, best.kind, mat, vel, best_t)


def _normalize(v):
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    n = math.sqrt(x*x + y*y + z*z)
    if n < 1e-9:
        return (0.0, 0.0, 0.0)
    return (x/n, y/n, z/n)


def _aabb_intersect(origin, direction, center, half):
    ox, oy, oz = origin
    dx, dy, dz = direction
    cx, cy, cz = center
    hx, hy, hz = half
    inv = []
    for a in (dx, dy, dz):
        inv.append(1.0 / a if abs(a) > 1e-9 else float('inf'))
    t1 = ((cx - hx) - ox) * inv[0]
    t2 = ((cx + hx) - ox) * inv[0]
    t3 = ((cy - hy) - oy) * inv[1]
    t4 = ((cy + hy) - oy) * inv[1]
    t5 = ((cz - hz) - oz) * inv[2]
    t6 = ((cz + hz) - oz) * inv[2]
    tmin = max(min(t1, t2), min(t3, t4), min(t5, t6))
    tmax = min(max(t1, t2), max(t3, t4), max(t5, t6))
    if tmax < 0.0 or tmin > tmax:
        return None
    return tmin if tmin > 0.0 else tmax
