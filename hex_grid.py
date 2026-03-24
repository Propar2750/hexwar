"""HexGrid — manages a rectangular collection of hex tiles.

Defines the Terrain enum (PLAINS, FERTILE, MOUNTAIN) and the HexGrid
container that holds HexTile objects. This is the second foundational
layer after hex_core — most game logic depends on it.

Depended on by:
    pathfinding, map_generator, renderer, main, play, replay,
    game/config, game/state, game/engine, game/game_renderer,
    game/environment, game/flat_env, game/bots,
    tests/test_hex_grid, tests/test_pathfinding, tests/test_main_utils

Dependencies:
    hex_core (HexCoord), pathfinding (lazy import in find_path)

Ripple effects:
    - Adding a new Terrain variant → update game/config.py dicts
      (troop_generation, defense_bonus, movement_cost) and renderer
      TERRAIN_FILL colors.
    - Changing HexGrid API → affects map_generator, game/engine,
      game/state, and all rendering code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from hex_core import HexCoord


class Terrain(str, Enum):
    """Terrain types that affect movement cost and gameplay."""

    PLAINS = "plains"
    MOUNTAIN = "mountain"
    FERTILE = "fertile"


@dataclass
class HexTile:
    """Data stored at a single hex position."""

    coord: HexCoord
    terrain: Terrain = Terrain.PLAINS


class HexGrid:
    """A grid of hex tiles addressed by doubled-width coordinates.

    Generates a rectangular-shaped grid of the given width and height
    (measured in hex rows/columns).
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._tiles: dict[HexCoord, HexTile] = {}
        self._build_grid()

    def _build_grid(self) -> None:
        """Populate the grid with tiles in a rectangular layout."""
        for row in range(self.height):
            for q in range(self.width):
                # In doubled-width: col = 2*q + (row % 2)
                # This staggers odd rows by 1 column, keeping col+row even.
                col = 2 * q + (row % 2)
                coord = HexCoord(col, row)
                self._tiles[coord] = HexTile(coord=coord)

    def __contains__(self, coord: HexCoord) -> bool:
        return coord in self._tiles

    def __getitem__(self, coord: HexCoord) -> HexTile:
        return self._tiles[coord]

    def __iter__(self):
        return iter(self._tiles.values())

    def __len__(self) -> int:
        return len(self._tiles)

    def get(self, coord: HexCoord) -> HexTile | None:
        return self._tiles.get(coord)

    def neighbors_of(self, coord: HexCoord) -> list[HexTile]:
        """Return tiles adjacent to coord that exist in this grid."""
        return [
            self._tiles[n]
            for n in coord.neighbors()
            if n in self._tiles
        ]

    def find_path(
        self,
        start: HexCoord,
        goal: HexCoord,
        cost_fn=None,
        passable_fn=None,
    ):
        """Find shortest path between two hexes using A*."""
        from pathfinding import astar

        return astar(self, start, goal, cost_fn, passable_fn)

    @property
    def tiles(self) -> list[HexTile]:
        return list(self._tiles.values())
