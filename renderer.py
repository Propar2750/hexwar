"""Pygame rendering for hex grids."""

from __future__ import annotations

import pygame

from hex_core import HexCoord, hex_vertices
from hex_grid import HexGrid

# Colors
BG_COLOR = (30, 30, 30)
HEX_FILL = (50, 70, 90)
HEX_BORDER = (140, 180, 210)
COORD_COLOR = (200, 200, 200)
HIGHLIGHT_FILL = (80, 120, 60)
HIGHLIGHT_BORDER = (160, 220, 100)


class HexRenderer:
    """Draws a HexGrid onto a pygame surface."""

    def __init__(
        self,
        surface: pygame.Surface,
        hex_size: float = 30.0,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.surface = surface
        self.hex_size = hex_size
        # origin is the pixel offset applied to all hex positions
        self.origin_x, self.origin_y = origin
        self._font: pygame.font.Font | None = None

    @property
    def font(self) -> pygame.font.Font:
        if self._font is None:
            self._font = pygame.font.SysFont("consolas", max(10, int(self.hex_size * 0.4)))
        return self._font

    def hex_to_screen(self, coord: HexCoord) -> tuple[float, float]:
        """Convert a hex coord to screen pixel position."""
        px, py = coord.to_pixel(self.hex_size)
        return (px + self.origin_x, py + self.origin_y)

    def draw_grid(
        self,
        grid: HexGrid,
        highlight: HexCoord | None = None,
        show_coords: bool = True,
    ) -> None:
        """Draw all hexes in the grid."""
        # Draw non-highlighted hexes first, then the highlighted one on top
        # so its border isn't overwritten by neighbors.
        for tile in grid:
            if highlight is not None and tile.coord == highlight:
                continue
            self._draw_hex(tile.coord, HEX_FILL, HEX_BORDER)
            if show_coords:
                self._draw_coord_label(tile.coord)

        if highlight is not None and highlight in grid:
            self._draw_hex(highlight, HIGHLIGHT_FILL, HIGHLIGHT_BORDER)
            if show_coords:
                self._draw_coord_label(highlight)

    def _draw_hex(
        self,
        coord: HexCoord,
        fill: tuple[int, int, int],
        border: tuple[int, int, int],
    ) -> None:
        cx, cy = self.hex_to_screen(coord)
        verts = hex_vertices(cx, cy, self.hex_size)
        pygame.draw.polygon(self.surface, fill, verts)
        pygame.draw.polygon(self.surface, border, verts, 2)

    def _draw_coord_label(self, coord: HexCoord) -> None:
        cx, cy = self.hex_to_screen(coord)
        label = self.font.render(f"{coord.col},{coord.row}", True, COORD_COLOR)
        rect = label.get_rect(center=(cx, cy))
        self.surface.blit(label, rect)
