from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

from .space_tensor import SpaceTensor


def _passable(idx: Tuple[int, int, int], tensor: SpaceTensor) -> bool:
    nx, ny, nz = tensor.shape
    ix, iy, iz = idx
    if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
        return False
    return tensor.cell(idx).occupancy < 0.5


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
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


def a_star(
    start_idx: Tuple[int, int, int],
    goal_idx: Tuple[int, int, int],
    tensor: SpaceTensor,
) -> List[Tuple[int, int, int]]:
    if not _passable(goal_idx, tensor):
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
            if not _passable(neighbor, tensor):
                continue
            step_cost = 14 if sum(abs(a - b) for a, b in zip(neighbor, current)) > 1 else 10
            tentative_g = current_g + step_cost
            if tentative_g < best_g.get(neighbor, 10**18):
                came_from[neighbor] = current
                best_g[neighbor] = tentative_g
                g_score[neighbor] = tentative_g
                f = tentative_g + 14 * _heuristic(neighbor, goal_idx)
                heapq.heappush(open_set, (f, neighbor))

    return []


class AStarNavigation:
    def __init__(self, tensor: SpaceTensor):
        self.tensor = tensor

    def plan(self, start, goal):
        start_idx = self._world_to_grid(start)
        goal_idx = self._world_to_grid(goal)
        path = a_star(start_idx, goal_idx, self.tensor)
        return [self._grid_to_world(idx) for idx in path]

    def _world_to_grid(self, world):
        mn = self.tensor.min_bound
        cs = self.tensor.cell_size
        return tuple(int((v - m) / cs) for v, m in zip(world, mn))

    def _grid_to_world(self, grid):
        mn = self.tensor.min_bound
        cs = self.tensor.cell_size
        return tuple(m + (i + 0.5) * cs for m, i in zip(mn, grid))
