"""Tests for hex_core — HexCoord, pixel_to_hex, hex_vertices."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hex_core import DIRECTIONS, HexCoord, hex_vertices, pixel_to_hex


# ---------------------------------------------------------------------------
# DIRECTIONS constant
# ---------------------------------------------------------------------------

class TestDirections:
    def test_six_directions(self):
        assert len(DIRECTIONS) == 6

    def test_direction_values(self):
        assert DIRECTIONS[0] == (2, 0)    # East
        assert DIRECTIONS[1] == (1, -1)   # NE
        assert DIRECTIONS[2] == (-1, -1)  # NW
        assert DIRECTIONS[3] == (-2, 0)   # West
        assert DIRECTIONS[4] == (-1, 1)   # SW
        assert DIRECTIONS[5] == (1, 1)    # SE

    def test_east_west_opposite(self):
        """East and West should cancel out."""
        dc_e, dr_e = DIRECTIONS[0]
        dc_w, dr_w = DIRECTIONS[3]
        assert dc_e + dc_w == 0
        assert dr_e + dr_w == 0

    def test_ne_sw_opposite(self):
        dc_ne, dr_ne = DIRECTIONS[1]
        dc_sw, dr_sw = DIRECTIONS[4]
        assert dc_ne + dc_sw == 0
        assert dr_ne + dr_sw == 0

    def test_nw_se_opposite(self):
        dc_nw, dr_nw = DIRECTIONS[2]
        dc_se, dr_se = DIRECTIONS[5]
        assert dc_nw + dc_se == 0
        assert dr_nw + dr_se == 0

    def test_all_directions_produce_valid_coords(self):
        """Applying any direction to a valid coord keeps col+row even."""
        for dc, dr in DIRECTIONS:
            # (0,0) is valid; neighbor must also be valid
            assert (0 + dc + 0 + dr) % 2 == 0


# ---------------------------------------------------------------------------
# HexCoord creation and validation
# ---------------------------------------------------------------------------

class TestHexCoordCreation:
    def test_valid_coord_even_even(self):
        h = HexCoord(0, 0)
        assert h.col == 0 and h.row == 0

    def test_valid_coord_odd_odd(self):
        h = HexCoord(1, 1)
        assert h.col == 1 and h.row == 1

    def test_valid_coord_positive(self):
        h = HexCoord(4, 2)
        assert h.col == 4 and h.row == 2

    def test_valid_coord_negative(self):
        h = HexCoord(-3, -1)
        assert h.col == -3 and h.row == -1

    def test_invalid_even_odd(self):
        with pytest.raises(ValueError, match="col \\+ row must be even"):
            HexCoord(0, 1)

    def test_invalid_odd_even(self):
        with pytest.raises(ValueError, match="col \\+ row must be even"):
            HexCoord(1, 0)

    def test_invalid_large_numbers(self):
        with pytest.raises(ValueError):
            HexCoord(3, 2)

    def test_frozen(self):
        h = HexCoord(0, 0)
        with pytest.raises(AttributeError):
            h.col = 2

    def test_hashable(self):
        """Frozen dataclass should be usable as dict key / set member."""
        a = HexCoord(0, 0)
        b = HexCoord(0, 0)
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_equality(self):
        assert HexCoord(2, 0) == HexCoord(2, 0)

    def test_inequality(self):
        assert HexCoord(0, 0) != HexCoord(2, 0)


# ---------------------------------------------------------------------------
# HexCoord.neighbor
# ---------------------------------------------------------------------------

class TestHexCoordNeighbor:
    def test_east(self):
        assert HexCoord(0, 0).neighbor(0) == HexCoord(2, 0)

    def test_northeast(self):
        assert HexCoord(0, 0).neighbor(1) == HexCoord(1, -1)

    def test_northwest(self):
        assert HexCoord(0, 0).neighbor(2) == HexCoord(-1, -1)

    def test_west(self):
        assert HexCoord(0, 0).neighbor(3) == HexCoord(-2, 0)

    def test_southwest(self):
        assert HexCoord(0, 0).neighbor(4) == HexCoord(-1, 1)

    def test_southeast(self):
        assert HexCoord(0, 0).neighbor(5) == HexCoord(1, 1)

    def test_neighbor_from_odd_row(self):
        """Neighbors from an odd-row coord should also be valid."""
        h = HexCoord(1, 1)
        for d in range(6):
            n = h.neighbor(d)
            assert (n.col + n.row) % 2 == 0

    def test_roundtrip_east_west(self):
        origin = HexCoord(4, 2)
        assert origin.neighbor(0).neighbor(3) == origin

    def test_roundtrip_ne_sw(self):
        origin = HexCoord(4, 2)
        assert origin.neighbor(1).neighbor(4) == origin

    def test_roundtrip_nw_se(self):
        origin = HexCoord(4, 2)
        assert origin.neighbor(2).neighbor(5) == origin


# ---------------------------------------------------------------------------
# HexCoord.neighbors
# ---------------------------------------------------------------------------

class TestHexCoordNeighbors:
    def test_returns_six(self):
        assert len(HexCoord(0, 0).neighbors()) == 6

    def test_all_unique(self):
        nbrs = HexCoord(0, 0).neighbors()
        assert len(set(nbrs)) == 6

    def test_matches_individual_calls(self):
        h = HexCoord(2, 2)
        nbrs = h.neighbors()
        for d in range(6):
            assert nbrs[d] == h.neighbor(d)

    def test_all_valid_coords(self):
        for n in HexCoord(4, 2).neighbors():
            assert (n.col + n.row) % 2 == 0

    def test_origin_not_in_own_neighbors(self):
        origin = HexCoord(0, 0)
        assert origin not in origin.neighbors()


# ---------------------------------------------------------------------------
# HexCoord.distance_to
# ---------------------------------------------------------------------------

class TestHexCoordDistance:
    def test_distance_to_self(self):
        assert HexCoord(0, 0).distance_to(HexCoord(0, 0)) == 0

    def test_distance_to_east_neighbor(self):
        assert HexCoord(0, 0).distance_to(HexCoord(2, 0)) == 1

    def test_distance_to_diagonal_neighbor(self):
        assert HexCoord(0, 0).distance_to(HexCoord(1, 1)) == 1

    def test_distance_symmetric(self):
        a = HexCoord(0, 0)
        b = HexCoord(6, 4)
        assert a.distance_to(b) == b.distance_to(a)

    def test_distance_three_east(self):
        assert HexCoord(0, 0).distance_to(HexCoord(6, 0)) == 3

    def test_distance_two_south(self):
        assert HexCoord(0, 0).distance_to(HexCoord(0, 2)) == 2

    def test_distance_diagonal_multi(self):
        # (0,0) to (3,3): each diagonal step covers 1 col + 1 row
        assert HexCoord(0, 0).distance_to(HexCoord(3, 3)) == 3

    def test_distance_triangle_inequality(self):
        a = HexCoord(0, 0)
        b = HexCoord(2, 0)
        c = HexCoord(1, 1)
        assert a.distance_to(c) <= a.distance_to(b) + b.distance_to(c)

    def test_distance_negative_coords(self):
        a = HexCoord(-2, 0)
        b = HexCoord(2, 0)
        assert a.distance_to(b) == 2


# ---------------------------------------------------------------------------
# HexCoord.to_pixel
# ---------------------------------------------------------------------------

class TestHexCoordToPixel:
    def test_origin_at_zero(self):
        x, y = HexCoord(0, 0).to_pixel(10.0)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)

    def test_east_neighbor_pixel(self):
        size = 10.0
        x, y = HexCoord(2, 0).to_pixel(size)
        expected_x = math.sqrt(3) / 2 * size * 2
        assert x == pytest.approx(expected_x)
        assert y == pytest.approx(0.0)

    def test_south_pixel(self):
        size = 10.0
        x, y = HexCoord(0, 2).to_pixel(size)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(1.5 * size * 2)

    def test_diagonal_pixel(self):
        size = 20.0
        x, y = HexCoord(1, 1).to_pixel(size)
        assert x == pytest.approx(math.sqrt(3) / 2 * 20.0 * 1)
        assert y == pytest.approx(1.5 * 20.0 * 1)

    def test_different_sizes_scale_linearly(self):
        coord = HexCoord(4, 2)
        x1, y1 = coord.to_pixel(10.0)
        x2, y2 = coord.to_pixel(20.0)
        assert x2 == pytest.approx(x1 * 2)
        assert y2 == pytest.approx(y1 * 2)


# ---------------------------------------------------------------------------
# HexCoord.__repr__
# ---------------------------------------------------------------------------

class TestHexCoordRepr:
    def test_repr_origin(self):
        assert repr(HexCoord(0, 0)) == "Hex(0, 0)"

    def test_repr_positive(self):
        assert repr(HexCoord(4, 2)) == "Hex(4, 2)"

    def test_repr_negative(self):
        assert repr(HexCoord(-3, -1)) == "Hex(-3, -1)"


# ---------------------------------------------------------------------------
# pixel_to_hex round-trip
# ---------------------------------------------------------------------------

class TestPixelToHex:
    def test_origin_roundtrip(self):
        coord = HexCoord(0, 0)
        x, y = coord.to_pixel(30.0)
        assert pixel_to_hex(x, y, 30.0) == coord

    def test_east_neighbor_roundtrip(self):
        coord = HexCoord(2, 0)
        x, y = coord.to_pixel(30.0)
        assert pixel_to_hex(x, y, 30.0) == coord

    def test_diagonal_roundtrip(self):
        coord = HexCoord(1, 1)
        x, y = coord.to_pixel(30.0)
        assert pixel_to_hex(x, y, 30.0) == coord

    def test_several_coords_roundtrip(self):
        size = 25.0
        coords = [
            HexCoord(0, 0), HexCoord(2, 0), HexCoord(4, 0),
            HexCoord(1, 1), HexCoord(3, 1), HexCoord(5, 1),
            HexCoord(0, 2), HexCoord(2, 2), HexCoord(4, 2),
        ]
        for coord in coords:
            x, y = coord.to_pixel(size)
            assert pixel_to_hex(x, y, size) == coord, f"Failed for {coord}"

    def test_slightly_offset_still_resolves(self):
        """Pixel slightly off-center should still resolve to same hex."""
        coord = HexCoord(4, 2)
        x, y = coord.to_pixel(30.0)
        # Nudge by a small amount
        assert pixel_to_hex(x + 1.0, y + 1.0, 30.0) == coord

    def test_different_sizes(self):
        coord = HexCoord(6, 4)
        for size in [10.0, 20.0, 50.0]:
            x, y = coord.to_pixel(size)
            assert pixel_to_hex(x, y, size) == coord


# ---------------------------------------------------------------------------
# hex_vertices
# ---------------------------------------------------------------------------

class TestHexVertices:
    def test_returns_six_vertices(self):
        assert len(hex_vertices(0, 0, 10)) == 6

    def test_all_at_correct_distance(self):
        """Each vertex should be exactly `size` away from center."""
        cx, cy, size = 100.0, 200.0, 30.0
        for vx, vy in hex_vertices(cx, cy, size):
            dist = math.hypot(vx - cx, vy - cy)
            assert dist == pytest.approx(size)

    def test_vertices_are_distinct(self):
        verts = hex_vertices(0, 0, 10)
        # All 6 should be unique
        unique = set((round(vx, 8), round(vy, 8)) for vx, vy in verts)
        assert len(unique) == 6

    def test_first_vertex_angle(self):
        """First vertex at angle -30 degrees (pointy-top)."""
        size = 10.0
        verts = hex_vertices(0, 0, size)
        angle = math.radians(-30)
        assert verts[0][0] == pytest.approx(size * math.cos(angle))
        assert verts[0][1] == pytest.approx(size * math.sin(angle))

    def test_center_offset(self):
        """Vertices should be shifted by center position."""
        cx, cy = 50.0, 75.0
        verts_origin = hex_vertices(0, 0, 10)
        verts_offset = hex_vertices(cx, cy, 10)
        for (ox, oy), (sx, sy) in zip(verts_origin, verts_offset):
            assert sx == pytest.approx(ox + cx)
            assert sy == pytest.approx(oy + cy)

    def test_vertex_spacing_uniform(self):
        """Adjacent vertices should be equally spaced."""
        verts = hex_vertices(0, 0, 20.0)
        dists = []
        for i in range(6):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % 6]
            dists.append(math.hypot(x2 - x1, y2 - y1))
        for d in dists:
            assert d == pytest.approx(dists[0])
