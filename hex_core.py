"""Core hex coordinate system using doubled-width coordinates (pointy-top).

This is the lowest-level module in the project — almost every other file
imports from here. Changes to HexCoord, DIRECTIONS, or pixel conversion
functions will ripple across the entire codebase.

Depended on by:
    hex_grid, pathfinding, map_generator, renderer, main, play, replay,
    game/config, game/state, game/actions, game/engine, game/environment,
    game/flat_env, game/game_renderer, game/recorder, game/bots,
    agents (indirectly), tests/test_hex_core, tests/test_hex_grid,
    tests/test_pathfinding

Dependencies: None (standard library only)

In doubled-width coordinates:
- col + row is always even
- East/West neighbors differ by ±2 in col
- Diagonal neighbors differ by ±1 in both col and row

Pixel conversion (pointy-top):
- x = sqrt(3)/2 * size * col
- y = 3/2 * size * row
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# The six neighbor directions in doubled-width coords (pointy-top).
# Order: E, NE, NW, W, SW, SE
DIRECTIONS = [
    (2, 0),    # East
    (1, -1),   # Northeast
    (-1, -1),  # Northwest
    (-2, 0),   # West
    (-1, 1),   # Southwest
    (1, 1),    # Southeast
]


@dataclass(frozen=True, slots=True)
class HexCoord:
    """A hex position in doubled-width coordinates.

    Constraint: col + row must be even.
    """

    col: int
    row: int

    def __post_init__(self) -> None:
        if (self.col + self.row) & 1:
            raise ValueError(
                f"Invalid doubled-width coord: col + row must be even, "
                f"got ({self.col}, {self.row})"
            )

    def neighbor(self, direction: int) -> HexCoord:
        """Return the neighbor in the given direction (0-5, see DIRECTIONS)."""
        dc, dr = DIRECTIONS[direction]
        return HexCoord(self.col + dc, self.row + dr)

    def neighbors(self) -> list[HexCoord]:
        """Return all six neighbors."""
        return [self.neighbor(d) for d in range(6)]

    def distance_to(self, other: HexCoord) -> int:
        """Manhattan distance on the hex grid."""
        dcol = abs(self.col - other.col)
        drow = abs(self.row - other.row)
        return drow + max(0, (dcol - drow) // 2)

    def to_pixel(self, size: float) -> tuple[float, float]:
        """Convert to pixel coordinates (center of hex).

        Args:
            size: Distance from center to vertex.
        """
        x = math.sqrt(3) / 2 * size * self.col
        y = 1.5 * size * self.row
        return (x, y)

    def __repr__(self) -> str:
        return f"Hex({self.col}, {self.row})"


def pixel_to_hex(x: float, y: float, size: float) -> HexCoord:
    """Convert pixel coordinates back to the nearest hex (doubled-width).

    Uses fractional axial coords as an intermediate step, then rounds
    to the nearest valid doubled-width hex.
    """
    # Pixel -> fractional axial (pointy-top)
    frac_q = (math.sqrt(3) / 3 * x - 1 / 3 * y) / size
    frac_r = (2 / 3 * y) / size

    # Round axial
    frac_s = -frac_q - frac_r
    rq = round(frac_q)
    rr = round(frac_r)
    rs = round(frac_s)

    q_diff = abs(rq - frac_q)
    r_diff = abs(rr - frac_r)
    s_diff = abs(rs - frac_s)

    if q_diff > r_diff and q_diff > s_diff:
        rq = -rr - rs
    elif r_diff > s_diff:
        rr = -rq - rs

    # Axial (q, r) -> doubled-width: col = 2*q + r, row = r
    col = 2 * rq + rr
    row = rr
    return HexCoord(col, row)


def hex_vertices(center_x: float, center_y: float, size: float) -> list[tuple[float, float]]:
    """Compute the 6 vertices of a pointy-top hexagon."""
    vertices = []
    for i in range(6):
        angle = math.radians(60 * i - 30)
        vx = center_x + size * math.cos(angle)
        vy = center_y + size * math.sin(angle)
        vertices.append((vx, vy))
    return vertices
