from __future__ import annotations

import math


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
    if not all(math.isfinite(value) for value in (x, y, z)):
        raise ValueError("ray direction must contain finite values")
    n = math.sqrt(x*x + y*y + z*z)
    if n < 1e-9:
        raise ValueError("ray direction must be non-zero")
    return (x/n, y/n, z/n)


def _aabb_intersect(origin, direction, center, half):
    ox, oy, oz = origin
    dx, dy, dz = direction
    cx, cy, cz = center
    hx, hy, hz = half
    tmin, tmax = 0.0, float("inf")
    for origin_axis, direction_axis, center_axis, half_axis in zip(
        (ox, oy, oz), (dx, dy, dz), (cx, cy, cz), (hx, hy, hz)
    ):
        lower = center_axis - half_axis
        upper = center_axis + half_axis
        if abs(direction_axis) < 1e-12:
            if origin_axis < lower or origin_axis > upper:
                return None
            continue
        near = (lower - origin_axis) / direction_axis
        far = (upper - origin_axis) / direction_axis
        if near > far:
            near, far = far, near
        tmin = max(tmin, near)
        tmax = min(tmax, far)
        if tmin > tmax:
            return None
    if tmax <= 0.0:
        return None
    return tmin if tmin > 0.0 else tmax
