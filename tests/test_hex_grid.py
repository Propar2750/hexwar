"""Tests for hex_grid — Terrain, HexTile, HexGrid.

Covers: terrain enum values, tile creation, grid dimensions, coordinate
validity, and tile lookup. Run with: pytest
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hex_core import HexCoord
from hex_grid import HexGrid, HexTile, Terrain


# ---------------------------------------------------------------------------
# Terrain enum
# ---------------------------------------------------------------------------

class TestTerrain:
    def test_three_members(self):
        assert len(Terrain) == 3

    def test_plains_value(self):
        assert Terrain.PLAINS.value == "plains"

    def test_mountain_value(self):
        assert Terrain.MOUNTAIN.value == "mountain"

    def test_fertile_value(self):
        assert Terrain.FERTILE.value == "fertile"

    def test_is_str_subclass(self):
        assert isinstance(Terrain.PLAINS, str)

    def test_str_comparison(self):
        assert Terrain.PLAINS == "plains"
        assert Terrain.MOUNTAIN == "mountain"

    def test_from_value(self):
        assert Terrain("plains") is Terrain.PLAINS
        assert Terrain("mountain") is Terrain.MOUNTAIN
        assert Terrain("fertile") is Terrain.FERTILE

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            Terrain("swamp")


# ---------------------------------------------------------------------------
# HexTile
# ---------------------------------------------------------------------------

class TestHexTile:
    def test_creation_default_terrain(self):
        tile = HexTile(coord=HexCoord(0, 0))
        assert tile.terrain is Terrain.PLAINS

    def test_creation_with_terrain(self):
        tile = HexTile(coord=HexCoord(0, 0), terrain=Terrain.MOUNTAIN)
        assert tile.terrain is Terrain.MOUNTAIN

    def test_mutable_terrain(self):
        tile = HexTile(coord=HexCoord(0, 0))
        tile.terrain = Terrain.FERTILE
        assert tile.terrain is Terrain.FERTILE

    def test_coord_accessible(self):
        c = HexCoord(4, 2)
        tile = HexTile(coord=c)
        assert tile.coord == c

    def test_coord_identity(self):
        c = HexCoord(2, 0)
        tile = HexTile(coord=c)
        assert tile.coord is c


# ---------------------------------------------------------------------------
# HexGrid construction
# ---------------------------------------------------------------------------

class TestHexGridConstruction:
    def test_1x1_grid_size(self):
        grid = HexGrid(1, 1)
        assert len(grid) == 1

    def test_1x1_contains_origin(self):
        grid = HexGrid(1, 1)
        assert HexCoord(0, 0) in grid

    def test_2x1_grid_size(self):
        grid = HexGrid(2, 1)
        assert len(grid) == 2

    def test_2x1_contents(self):
        grid = HexGrid(2, 1)
        assert HexCoord(0, 0) in grid
        assert HexCoord(2, 0) in grid

    def test_8x6_grid_size(self):
        grid = HexGrid(8, 6)
        assert len(grid) == 48

    def test_width_height_stored(self):
        grid = HexGrid(5, 3)
        assert grid.width == 5
        assert grid.height == 3

    def test_even_row_columns(self):
        """Even rows start at col=0 and go 0, 2, 4, ..."""
        grid = HexGrid(4, 2)
        for q in range(4):
            assert HexCoord(2 * q, 0) in grid

    def test_odd_row_columns(self):
        """Odd rows start at col=1 and go 1, 3, 5, ..."""
        grid = HexGrid(4, 2)
        for q in range(4):
            assert HexCoord(2 * q + 1, 1) in grid

    def test_all_tiles_default_plains(self):
        grid = HexGrid(4, 3)
        for tile in grid:
            assert tile.terrain is Terrain.PLAINS

    def test_all_coords_valid(self):
        """Every tile coordinate must satisfy the col+row even constraint."""
        grid = HexGrid(6, 5)
        for tile in grid:
            assert (tile.coord.col + tile.coord.row) % 2 == 0

    def test_tile_coord_matches_key(self):
        """Each tile's .coord should match how we look it up."""
        grid = HexGrid(4, 3)
        for tile in grid:
            assert grid[tile.coord] is tile


# ---------------------------------------------------------------------------
# HexGrid.__contains__
# ---------------------------------------------------------------------------

class TestHexGridContains:
    def test_contains_valid_coord(self):
        grid = HexGrid(4, 3)
        assert HexCoord(0, 0) in grid

    def test_not_contains_outside(self):
        grid = HexGrid(4, 3)
        assert HexCoord(100, 100) not in grid

    def test_not_contains_negative(self):
        grid = HexGrid(4, 3)
        assert HexCoord(-2, 0) not in grid


# ---------------------------------------------------------------------------
# HexGrid.__getitem__
# ---------------------------------------------------------------------------

class TestHexGridGetItem:
    def test_getitem_returns_tile(self):
        grid = HexGrid(4, 3)
        tile = grid[HexCoord(0, 0)]
        assert isinstance(tile, HexTile)

    def test_getitem_correct_coord(self):
        grid = HexGrid(4, 3)
        c = HexCoord(2, 0)
        assert grid[c].coord == c

    def test_getitem_missing_raises(self):
        grid = HexGrid(4, 3)
        with pytest.raises(KeyError):
            grid[HexCoord(100, 100)]


# ---------------------------------------------------------------------------
# HexGrid.__iter__
# ---------------------------------------------------------------------------

class TestHexGridIter:
    def test_iter_yields_tiles(self):
        grid = HexGrid(3, 2)
        tiles = list(grid)
        assert all(isinstance(t, HexTile) for t in tiles)

    def test_iter_count_matches_len(self):
        grid = HexGrid(5, 4)
        assert len(list(grid)) == len(grid)

    def test_iter_all_unique(self):
        grid = HexGrid(4, 3)
        coords = [t.coord for t in grid]
        assert len(set(coords)) == len(coords)


# ---------------------------------------------------------------------------
# HexGrid.__len__
# ---------------------------------------------------------------------------

class TestHexGridLen:
    def test_len_1x1(self):
        assert len(HexGrid(1, 1)) == 1

    def test_len_matches_width_times_height(self):
        assert len(HexGrid(8, 6)) == 48
        assert len(HexGrid(3, 4)) == 12

    def test_len_single_row(self):
        assert len(HexGrid(5, 1)) == 5

    def test_len_single_col(self):
        assert len(HexGrid(1, 5)) == 5


# ---------------------------------------------------------------------------
# HexGrid.get
# ---------------------------------------------------------------------------

class TestHexGridGet:
    def test_get_existing(self):
        grid = HexGrid(4, 3)
        tile = grid.get(HexCoord(0, 0))
        assert tile is not None
        assert tile.coord == HexCoord(0, 0)

    def test_get_missing_returns_none(self):
        grid = HexGrid(4, 3)
        assert grid.get(HexCoord(100, 100)) is None

    def test_get_matches_getitem(self):
        grid = HexGrid(4, 3)
        c = HexCoord(2, 0)
        assert grid.get(c) is grid[c]


# ---------------------------------------------------------------------------
# HexGrid.neighbors_of
# ---------------------------------------------------------------------------

class TestHexGridNeighborsOf:
    def test_corner_has_fewer_neighbors(self):
        grid = HexGrid(4, 3)
        nbrs = grid.neighbors_of(HexCoord(0, 0))
        # Corner hex won't have all 6 neighbors in the grid
        assert len(nbrs) < 6

    def test_center_has_six_neighbors(self):
        """A hex in the interior should have all 6 neighbors."""
        grid = HexGrid(8, 6)
        # (4,2) is well inside the grid
        nbrs = grid.neighbors_of(HexCoord(4, 2))
        assert len(nbrs) == 6

    def test_returns_hex_tiles(self):
        grid = HexGrid(4, 3)
        nbrs = grid.neighbors_of(HexCoord(2, 0))
        assert all(isinstance(t, HexTile) for t in nbrs)

    def test_neighbor_coords_are_actual_neighbors(self):
        grid = HexGrid(8, 6)
        center = HexCoord(4, 2)
        expected = set(center.neighbors())
        actual = {t.coord for t in grid.neighbors_of(center)}
        assert actual.issubset(expected)

    def test_neighbors_all_in_grid(self):
        grid = HexGrid(4, 3)
        for tile in grid:
            for nbr in grid.neighbors_of(tile.coord):
                assert nbr.coord in grid

    def test_1x1_grid_no_neighbors(self):
        grid = HexGrid(1, 1)
        assert grid.neighbors_of(HexCoord(0, 0)) == []


# ---------------------------------------------------------------------------
# HexGrid.tiles property
# ---------------------------------------------------------------------------

class TestHexGridTilesProperty:
    def test_returns_list(self):
        grid = HexGrid(3, 2)
        assert isinstance(grid.tiles, list)

    def test_length_matches(self):
        grid = HexGrid(4, 3)
        assert len(grid.tiles) == len(grid)

    def test_tiles_are_tile_objects(self):
        grid = HexGrid(3, 2)
        for t in grid.tiles:
            assert isinstance(t, HexTile)

    def test_modifying_returned_list_does_not_affect_grid(self):
        grid = HexGrid(3, 2)
        tiles = grid.tiles
        original_len = len(grid)
        tiles.clear()
        assert len(grid) == original_len

    def test_tile_mutation_reflects_in_grid(self):
        """Tiles in the list are the same objects as in the grid."""
        grid = HexGrid(3, 2)
        tile = grid.tiles[0]
        tile.terrain = Terrain.MOUNTAIN
        assert grid[tile.coord].terrain is Terrain.MOUNTAIN


# ---------------------------------------------------------------------------
# HexGrid.find_path (wrapper)
# ---------------------------------------------------------------------------

class TestHexGridFindPath:
    def test_same_tile(self):
        grid = HexGrid(4, 3)
        result = grid.find_path(HexCoord(0, 0), HexCoord(0, 0))
        assert result.found
        assert result.cost == 0.0

    def test_adjacent(self):
        grid = HexGrid(4, 3)
        result = grid.find_path(HexCoord(0, 0), HexCoord(2, 0))
        assert result.found
        assert result.cost == 1.0

    def test_no_path_outside_grid(self):
        grid = HexGrid(4, 3)
        result = grid.find_path(HexCoord(0, 0), HexCoord(100, 100))
        assert not result.found
