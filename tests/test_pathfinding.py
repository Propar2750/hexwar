"""Tests for A* pathfinding on the hex grid.

Covers: basic path correctness, terrain cost handling, obstacle avoidance,
unreachable targets, and PathResult fields. Run with: pytest
"""

import sys
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hex_core import HexCoord
from hex_grid import HexGrid, HexTile, Terrain
from pathfinding import DEFAULT_COST, PathResult, astar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_grid(w: int = 8, h: int = 6) -> HexGrid:
    """Create a default all-plains grid."""
    return HexGrid(w, h)


def set_terrain(grid: HexGrid, coord: HexCoord, terrain: Terrain) -> None:
    """Set the terrain of a specific tile."""
    tile = grid.get(coord)
    if tile is not None:
        tile.terrain = terrain


# ---------------------------------------------------------------------------
# Terrain enum
# ---------------------------------------------------------------------------

class TestTerrain:
    def test_three_values(self):
        assert len(Terrain) == 3

    def test_default_is_plains(self):
        tile = HexTile(coord=HexCoord(0, 0))
        assert tile.terrain is Terrain.PLAINS

    def test_str_values(self):
        assert Terrain.PLAINS.value == "plains"
        assert Terrain.MOUNTAIN.value == "mountain"
        assert Terrain.FERTILE.value == "fertile"


# ---------------------------------------------------------------------------
# Core A* correctness
# ---------------------------------------------------------------------------

class TestAstarCorrectness:
    def test_same_tile(self):
        grid = make_grid()
        start = HexCoord(0, 0)
        result = astar(grid, start, start)
        assert result.found
        assert result.path == (start,)
        assert result.cost == 0.0

    def test_adjacent_tiles(self):
        grid = make_grid()
        start = HexCoord(0, 0)
        goal = HexCoord(2, 0)  # East neighbor
        result = astar(grid, start, goal)
        assert result.found
        assert result.path == (start, goal)
        assert result.cost == 1.0

    def test_straight_line_east(self):
        grid = make_grid()
        start = HexCoord(0, 0)
        goal = HexCoord(6, 0)  # 3 steps east
        result = astar(grid, start, goal)
        assert result.found
        assert len(result.path) == 4  # start + 3 steps
        assert result.path[0] == start
        assert result.path[-1] == goal
        assert result.cost == 3.0

    def test_path_includes_both_endpoints(self):
        grid = make_grid()
        start = HexCoord(0, 0)
        goal = HexCoord(4, 2)
        result = astar(grid, start, goal)
        assert result.found
        assert result.path[0] == start
        assert result.path[-1] == goal

    def test_path_is_connected(self):
        """Each step in the path should be a neighbor of the previous one."""
        grid = make_grid()
        start = HexCoord(0, 0)
        goal = HexCoord(10, 4)
        result = astar(grid, start, goal)
        assert result.found
        for i in range(len(result.path) - 1):
            a, b = result.path[i], result.path[i + 1]
            assert b in a.neighbors(), f"{b} not a neighbor of {a}"

    def test_no_path_when_surrounded_by_impassable(self):
        grid = make_grid()
        start = HexCoord(4, 2)
        goal = HexCoord(10, 4)
        # Block all neighbors of start
        blocked = set(start.neighbors())
        result = astar(
            grid, start, goal,
            passable_fn=lambda tile: tile.coord not in blocked,
        )
        assert not result.found
        assert result.cost == float("inf")

    def test_start_not_in_grid(self):
        grid = make_grid()
        fake = HexCoord(100, 100)
        result = astar(grid, fake, HexCoord(0, 0))
        assert not result.found

    def test_goal_not_in_grid(self):
        grid = make_grid()
        fake = HexCoord(100, 100)
        result = astar(grid, HexCoord(0, 0), fake)
        assert not result.found

    def test_goal_is_impassable(self):
        grid = make_grid()
        start = HexCoord(0, 0)
        goal = HexCoord(4, 2)
        result = astar(
            grid, start, goal,
            passable_fn=lambda tile: tile.coord != goal,
        )
        assert not result.found

    def test_path_around_obstacle(self):
        """Wall of impassable tiles forces path to go around."""
        grid = make_grid(w=8, h=6)
        start = HexCoord(0, 2)
        goal = HexCoord(8, 2)

        # Block a vertical column of tiles at col=4 (rows 1-3)
        wall = {HexCoord(4, 2), HexCoord(5, 1), HexCoord(5, 3)}
        result_blocked = astar(
            grid, start, goal,
            passable_fn=lambda tile: tile.coord not in wall,
        )
        result_free = astar(grid, start, goal)

        assert result_free.found
        assert result_blocked.found
        # Blocked path should be longer
        assert len(result_blocked.path) > len(result_free.path)


# ---------------------------------------------------------------------------
# Cost correctness
# ---------------------------------------------------------------------------

class TestAstarCost:
    def test_uniform_cost_equals_steps(self):
        """On all-plains grid, cost == number of steps (len(path) - 1)."""
        grid = make_grid()
        start = HexCoord(0, 0)
        goal = HexCoord(6, 4)
        result = astar(grid, start, goal)
        assert result.found
        assert result.cost == len(result.path) - 1

    def test_mountain_increases_cost(self):
        """Path through mountains costs more than plains-only path."""
        grid = make_grid()
        start = HexCoord(0, 0)
        goal = HexCoord(4, 0)  # 2 steps east

        # All plains
        result_plains = astar(grid, start, goal)
        assert result_plains.cost == 2.0

        # Put mountain in the middle
        set_terrain(grid, HexCoord(2, 0), Terrain.MOUNTAIN)
        result_mountain = astar(grid, start, goal)
        assert result_mountain.cost > result_plains.cost

    def test_prefers_plains_over_mountain_shortcut(self):
        """A* should prefer longer plains route when mountain shortcut is more expensive."""
        grid = make_grid(w=6, h=4)
        start = HexCoord(0, 0)
        goal = HexCoord(4, 0)

        # Make the direct middle tile a mountain (cost 3.0)
        set_terrain(grid, HexCoord(2, 0), Terrain.MOUNTAIN)

        result = astar(grid, start, goal)
        assert result.found
        # Direct route: cost = 1.0 + 3.0 = 4.0 (through mountain)
        # Detour via row 1: 3 steps * 1.0 = 3.0 (all plains)
        # A* should pick the cheaper detour
        assert result.cost <= 4.0

    def test_custom_cost_fn(self):
        """Custom cost function overrides terrain-based costs."""
        grid = make_grid()
        start = HexCoord(0, 0)
        goal = HexCoord(4, 0)

        # Make everything cost 5.0
        result = astar(grid, start, goal, cost_fn=lambda tile: 5.0)
        assert result.found
        assert result.cost == 10.0  # 2 steps * 5.0

    def test_fertile_costs_same_as_plains(self):
        grid = make_grid()
        start = HexCoord(0, 0)
        goal = HexCoord(4, 0)

        result_plains = astar(grid, start, goal)

        set_terrain(grid, HexCoord(2, 0), Terrain.FERTILE)
        result_fertile = astar(grid, start, goal)

        assert result_fertile.cost == result_plains.cost


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_1x1_grid(self):
        grid = HexGrid(1, 1)
        coord = HexCoord(0, 0)
        result = astar(grid, coord, coord)
        assert result.found
        assert result.path == (coord,)
        assert result.cost == 0.0

    def test_1x1_grid_no_path_to_outside(self):
        grid = HexGrid(1, 1)
        result = astar(grid, HexCoord(0, 0), HexCoord(2, 0))
        assert not result.found

    def test_2x1_grid(self):
        grid = HexGrid(2, 1)
        start = HexCoord(0, 0)
        goal = HexCoord(2, 0)
        result = astar(grid, start, goal)
        assert result.found
        assert result.path == (start, goal)

    def test_large_grid(self):
        grid = HexGrid(15, 10)
        start = HexCoord(0, 0)
        # Bottom-right corner: col = 2*14 + (9%2) = 29, row = 9
        goal = HexCoord(29, 9)
        result = astar(grid, start, goal)
        assert result.found
        # Verify cost matches hex distance for uniform grid
        assert result.cost == start.distance_to(goal)


# ---------------------------------------------------------------------------
# PathResult
# ---------------------------------------------------------------------------

class TestPathResult:
    def test_found_property(self):
        assert PathResult(path=(HexCoord(0, 0),), cost=0.0).found
        assert not PathResult(path=(), cost=float("inf")).found

    def test_len(self):
        path = (HexCoord(0, 0), HexCoord(2, 0))
        result = PathResult(path=path, cost=1.0)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# HexGrid.find_path() integration
# ---------------------------------------------------------------------------

class TestFindPathIntegration:
    def test_matches_astar_directly(self):
        grid = make_grid()
        start = HexCoord(0, 0)
        goal = HexCoord(6, 4)

        direct = astar(grid, start, goal)
        wrapper = grid.find_path(start, goal)

        assert direct.path == wrapper.path
        assert direct.cost == wrapper.cost

    def test_with_callbacks(self):
        grid = make_grid()
        start = HexCoord(0, 0)
        goal = HexCoord(4, 0)

        result = grid.find_path(
            start, goal,
            cost_fn=lambda tile: 2.0,
            passable_fn=lambda tile: True,
        )
        assert result.found
        assert result.cost == 4.0  # 2 steps * 2.0
