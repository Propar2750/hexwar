"""Tests for utility functions in main.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hex_grid import HexGrid, Terrain
from main import sprinkle_mountains


class TestSprinkleMountains:
    def test_correct_count(self):
        grid = HexGrid(8, 6)  # 48 tiles
        sprinkle_mountains(grid, 0.25)
        mountain_count = sum(1 for t in grid if t.terrain is Terrain.MOUNTAIN)
        assert mountain_count == 12  # int(48 * 0.25)

    def test_zero_ratio(self):
        grid = HexGrid(8, 6)
        sprinkle_mountains(grid, 0.0)
        assert all(t.terrain is Terrain.PLAINS for t in grid)

    def test_remaining_tiles_unchanged(self):
        grid = HexGrid(8, 6)
        sprinkle_mountains(grid, 0.15)
        for tile in grid:
            assert tile.terrain in (Terrain.PLAINS, Terrain.MOUNTAIN)

    def test_small_grid(self):
        grid = HexGrid(2, 2)  # 4 tiles
        sprinkle_mountains(grid, 0.5)
        mountain_count = sum(1 for t in grid if t.terrain is Terrain.MOUNTAIN)
        assert mountain_count == 2

    def test_does_not_create_other_terrain(self):
        grid = HexGrid(8, 6)
        sprinkle_mountains(grid, 0.3)
        fertile_count = sum(1 for t in grid if t.terrain is Terrain.FERTILE)
        assert fertile_count == 0

    def test_idempotent_count_on_fresh_grid(self):
        """Calling on an all-plains grid gives the expected count."""
        grid = HexGrid(10, 10)  # 100 tiles
        sprinkle_mountains(grid, 0.10)
        assert sum(1 for t in grid if t.terrain is Terrain.MOUNTAIN) == 10
