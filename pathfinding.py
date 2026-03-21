"""A* pathfinding on a hex grid."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from hex_core import HexCoord
from hex_grid import HexGrid, HexTile, Terrain

if TYPE_CHECKING:
    pass

# Default movement costs per terrain type.
# Mountains are expensive to traverse; fertile land moves like plains.
DEFAULT_COST: dict[Terrain, float] = {
    Terrain.PLAINS: 1.0,
    Terrain.MOUNTAIN: 3.0,
    Terrain.FERTILE: 1.0,
}


@dataclass(frozen=True)
class PathResult:
    """Result of an A* pathfinding query."""

    path: tuple[HexCoord, ...]
    cost: float

    @property
    def found(self) -> bool:
        """Whether a valid path was found."""
        return len(self.path) > 0

    def __len__(self) -> int:
        return len(self.path)


_EMPTY_RESULT = PathResult(path=(), cost=float("inf"))


def astar(
    grid: HexGrid,
    start: HexCoord,
    goal: HexCoord,
    cost_fn: Callable[[HexTile], float] | None = None,
    passable_fn: Callable[[HexTile], bool] | None = None,
) -> PathResult:
    """Find the shortest path between two hexes using A*.

    Args:
        grid: The hex grid to pathfind on.
        start: Starting hex coordinate.
        goal: Target hex coordinate.
        cost_fn: Returns the movement cost to enter a tile.
            Defaults to terrain-based cost via DEFAULT_COST.
            Must return values >= 1.0 for the heuristic to remain admissible.
        passable_fn: Returns whether a tile can be traversed.
            Defaults to all tiles passable.

    Returns:
        PathResult with the path (start to goal inclusive) and total cost.
        Empty result if no path exists.
    """
    if start not in grid or goal not in grid:
        return _EMPTY_RESULT

    if start == goal:
        return PathResult(path=(start,), cost=0.0)

    if cost_fn is None:
        cost_fn = lambda tile: DEFAULT_COST[tile.terrain]
    if passable_fn is None:
        passable_fn = lambda tile: True

    if not passable_fn(grid[goal]):
        return _EMPTY_RESULT

    # Priority queue: (f_score, tiebreaker, coord)
    # Integer tiebreaker avoids comparing HexCoord objects.
    counter = 0
    open_set: list[tuple[float, int, HexCoord]] = []
    heapq.heappush(open_set, (start.distance_to(goal), counter, start))
    counter += 1

    came_from: dict[HexCoord, HexCoord] = {}
    g_score: dict[HexCoord, float] = {start: 0.0}

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return PathResult(path=tuple(path), cost=g_score[goal])

        current_g = g_score[current]

        # Skip stale heap entries (we already found a cheaper path to this node)
        if current_g > g_score.get(current, float("inf")):
            continue

        for neighbor_tile in grid.neighbors_of(current):
            neighbor = neighbor_tile.coord

            if not passable_fn(neighbor_tile):
                continue

            tentative_g = current_g + cost_fn(neighbor_tile)

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + neighbor.distance_to(goal)
                heapq.heappush(open_set, (f_score, counter, neighbor))
                counter += 1

    return _EMPTY_RESULT
