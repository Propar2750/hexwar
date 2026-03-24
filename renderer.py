"""Pygame rendering for hex grids.

Base rendering layer — draws terrain-coloured hexagons, highlights, and
path overlays. Extended by game/game_renderer.py which adds ownership
tinting, troop labels, and HUD elements.

Depended on by:
    main, play, game/game_renderer, replay

Dependencies:
    hex_core (HexCoord, hex_vertices), hex_grid (HexGrid, Terrain)

Ripple effects:
    - Changing TERRAIN_FILL colours or HexRenderer API affects all visual
      output including game_renderer and replay viewer.
    - BG_COLOR is imported by play.py and replay.py for window background.
"""

from __future__ import annotations

import pygame

from hex_core import HexCoord, hex_vertices
from hex_grid import HexGrid, Terrain

# Colors
BG_COLOR = (30, 30, 30)
HEX_FILL = (50, 70, 90)
HEX_BORDER = (140, 180, 210)
TERRAIN_FILL: dict[Terrain, tuple[int, int, int]] = {
    Terrain.PLAINS: (50, 70, 90),
    Terrain.MOUNTAIN: (90, 80, 70),
    Terrain.FERTILE: (50, 90, 55),
}
COORD_COLOR = (200, 200, 200)
HIGHLIGHT_FILL = (80, 120, 60)
HIGHLIGHT_BORDER = (160, 220, 100)
PATH_FILL = (180, 100, 40)
PATH_BORDER = (240, 160, 60)
START_FILL = (40, 160, 80)
START_BORDER = (80, 220, 120)
GOAL_FILL = (160, 40, 40)
GOAL_BORDER = (220, 80, 80)


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
            fill = TERRAIN_FILL.get(tile.terrain, HEX_FILL)
            self._draw_hex(tile.coord, fill, HEX_BORDER)
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

    def draw_path(
        self,
        path: tuple[HexCoord, ...] | list[HexCoord],
        show_coords: bool = True,
    ) -> None:
        """Highlight a sequence of hexes as a path."""
        if not path:
            return
        for coord in path:
            self._draw_hex(coord, PATH_FILL, PATH_BORDER)
            if show_coords:
                self._draw_coord_label(coord)
        # Draw start and goal with distinct colors
        self._draw_hex(path[0], START_FILL, START_BORDER)
        if show_coords:
            self._draw_coord_label(path[0])
        if len(path) > 1:
            self._draw_hex(path[-1], GOAL_FILL, GOAL_BORDER)
            if show_coords:
                self._draw_coord_label(path[-1])

    def _draw_coord_label(self, coord: HexCoord) -> None:
        cx, cy = self.hex_to_screen(coord)
        label = self.font.render(f"{coord.col},{coord.row}", True, COORD_COLOR)
        rect = label.get_rect(center=(cx, cy))
        self.surface.blit(label, rect)
