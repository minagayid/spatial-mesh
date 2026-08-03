from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

from .space_tensor import SpaceTensor


def _passable(idx: Tuple[int, int, int], tensor: SpaceTensor, clearance_cells: int = 0) -> bool:
    if not tensor.valid_index(idx):
        return False
    cell = tensor.cell(idx)
    if cell is not None and cell.occupancy >= 0.5:
        return False
    radius = max(0, int(clearance_cells))
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                neighbor = (idx[0] + dx, idx[1] + dy, idx[2] + dz)
                if tensor.valid_index(neighbor):
                    neighbor_cell = tensor.cell(neighbor)
                    if neighbor_cell is not None and neighbor_cell.occupancy >= 0.5:
                        return False
    return True


def _neighbors(idx: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    x, y, z = idx
    neighbors: List[Tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neighbors.append((x + dx, y + dy, z + dz))
    return neighbors


def _heuristic(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> int:
    distances = sorted((abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2])), reverse=True)
    diagonal_3d, diagonal_2d, straight = distances
    return 17 * diagonal_3d + 14 * (diagonal_2d - diagonal_3d) + 10 * (straight - diagonal_2d)


def a_star(
    start_idx: Tuple[int, int, int],
    goal_idx: Tuple[int, int, int],
    tensor: SpaceTensor,
    clearance_cells: int = 0,
    risk_weight: float = 0.0,
) -> List[Tuple[int, int, int]]:
    if not _passable(start_idx, tensor, clearance_cells) or not _passable(goal_idx, tensor, clearance_cells):
        return []

    open_set: List[Tuple[int, Tuple[int, int, int]]] = []
    heapq.heappush(open_set, (0, start_idx))
    came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    g_score: Dict[Tuple[int, int, int], int] = {start_idx: 0}
    best_g: Dict[Tuple[int, int, int], int] = {start_idx: 0}

    while open_set:
        current_f, current = heapq.heappop(open_set)
        if current == goal_idx:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        current_g = best_g[current]
        for neighbor in _neighbors(current):
            if not _passable(neighbor, tensor, clearance_cells):
                continue
            step_dimensions = sum(abs(a - b) for a, b in zip(neighbor, current))
            step_cost = {1: 10, 2: 14, 3: 17}[step_dimensions]
            if step_dimensions > 1 and not _diagonal_is_clear(current, neighbor, tensor, clearance_cells):
                continue
            neighbor_cell = tensor.cell(neighbor)
            risk_cost = max(0.0, float(risk_weight)) * (neighbor_cell.uncertainty if neighbor_cell else 1.0) * 10.0
            tentative_g = current_g + step_cost + risk_cost
            if tentative_g < best_g.get(neighbor, 10**18):
                came_from[neighbor] = current
                best_g[neighbor] = tentative_g
                g_score[neighbor] = tentative_g
                f = tentative_g + _heuristic(neighbor, goal_idx)
                heapq.heappush(open_set, (f, neighbor))

    return []


class AStarNavigation:
    def __init__(self, tensor: SpaceTensor, robot_radius: float = 0.0, risk_weight: float = 0.0):
        self.tensor = tensor
        if robot_radius < 0.0:
            raise ValueError("robot_radius must be non-negative")
        self.clearance_cells = int(math.ceil(float(robot_radius) / tensor.cell_size))
        self.risk_weight = float(risk_weight)

    def plan(self, start, goal):
        start_idx = self._world_to_grid(start)
        goal_idx = self._world_to_grid(goal)
        if start_idx is None or goal_idx is None:
            return []
        path = a_star(start_idx, goal_idx, self.tensor, self.clearance_cells, self.risk_weight)
        return [self._grid_to_world(idx) for idx in path]

    def _world_to_grid(self, world):
        mn = self.tensor.min_bound
        cs = self.tensor.cell_size
        if not all(math.isfinite(float(value)) for value in world):
            return None
        idx = tuple(int((float(v) - m) // cs) for v, m in zip(world, mn))
        return idx if self.tensor.valid_index(idx) else None

    def _grid_to_world(self, grid):
        mn = self.tensor.min_bound
        cs = self.tensor.cell_size
        return tuple(m + (i + 0.5) * cs for m, i in zip(mn, grid))


def _diagonal_is_clear(current, neighbor, tensor: SpaceTensor, clearance_cells: int) -> bool:
    deltas = [axis for axis in range(3) if neighbor[axis] != current[axis]]
    for mask in range(1, 1 << len(deltas)):
        intermediate = list(current)
        for bit, axis in enumerate(deltas):
            if mask & (1 << bit):
                intermediate[axis] = neighbor[axis]
        if not _passable(tuple(intermediate), tensor, clearance_cells):
            return False
    return True
