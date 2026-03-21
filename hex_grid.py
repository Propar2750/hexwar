"""HexGrid — manages a rectangular collection of hex tiles."""

from __future__ import annotations

from dataclasses import dataclass, field

from hex_core import HexCoord


@dataclass
class HexTile:
    """Data stored at a single hex position."""

    coord: HexCoord

    # Extend later with terrain, ownership, units, etc.


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

    @property
    def tiles(self) -> list[HexTile]:
        return list(self._tiles.values())
