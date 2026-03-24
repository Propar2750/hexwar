"""Tests for map_generator terrain generation.

Covers: deterministic seeding, mountain placement, fertile distribution,
and edge-case parameter ranges. Run with: pytest
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hex_grid import HexGrid, Terrain
from map_generator import generate_terrain


class TestGenerateTerrain:
    def test_all_tiles_have_valid_terrain(self):
        grid = HexGrid(8, 6)
        generate_terrain(grid, seed=42)
        for tile in grid:
            assert tile.terrain in (Terrain.PLAINS, Terrain.FERTILE, Terrain.MOUNTAIN)

    def test_deterministic_with_seed(self):
        """Same seed should produce identical terrain."""
        grid1 = HexGrid(8, 6)
        generate_terrain(grid1, seed=123)
        grid2 = HexGrid(8, 6)
        generate_terrain(grid2, seed=123)
        for t1, t2 in zip(grid1, grid2):
            assert t1.terrain == t2.terrain

    def test_different_seeds_differ(self):
        """Different seeds should (almost certainly) produce different terrain."""
        grid1 = HexGrid(8, 6)
        generate_terrain(grid1, seed=1)
        grid2 = HexGrid(8, 6)
        generate_terrain(grid2, seed=999)
        terrains1 = [t.terrain for t in grid1]
        terrains2 = [t.terrain for t in grid2]
        assert terrains1 != terrains2

    def test_mountains_present(self):
        """Mountain ranges should produce at least some mountains."""
        grid = HexGrid(10, 10)
        generate_terrain(grid, seed=42, num_ranges=6)
        mountain_count = sum(1 for t in grid if t.terrain is Terrain.MOUNTAIN)
        assert mountain_count > 0

    def test_fertile_present(self):
        """CA seeding should produce some fertile tiles."""
        grid = HexGrid(10, 10)
        generate_terrain(grid, seed=42, fertile_p=0.45)
        fertile_count = sum(1 for t in grid if t.terrain is Terrain.FERTILE)
        assert fertile_count > 0

    def test_no_mountains_when_zero_ranges(self):
        """With zero mountain ranges, no mountains should appear."""
        grid = HexGrid(8, 6)
        generate_terrain(grid, seed=42, num_ranges=0)
        mountain_count = sum(1 for t in grid if t.terrain is Terrain.MOUNTAIN)
        assert mountain_count == 0
