"""Map generation using cellular automata for hex grids.

Produces organic terrain clusters:
1. Seed tiles as fertile with probability p
2. Run CA iterations (symmetric threshold rule)
3. Sprinkle mountains on top of remaining plains
"""

from __future__ import annotations

import random

from hex_core import DIRECTIONS, HexCoord
from hex_grid import HexGrid, Terrain

# Defaults tuned via test_cellular_automata.py sweep
DEFAULT_FERTILE_SEED_P = 0.45
DEFAULT_CA_ITERATIONS = 6
DEFAULT_CA_THRESHOLD = 4
DEFAULT_NUM_RANGES = 5
DEFAULT_RANGE_STEPS = 8
DEFAULT_TURN_PROB = 0.3


def generate_terrain(
    grid: HexGrid,
    fertile_p: float = DEFAULT_FERTILE_SEED_P,
    ca_iterations: int = DEFAULT_CA_ITERATIONS,
    ca_threshold: int = DEFAULT_CA_THRESHOLD,
    num_ranges: int = DEFAULT_NUM_RANGES,
    range_steps: int = DEFAULT_RANGE_STEPS,
    turn_prob: float = DEFAULT_TURN_PROB,
    seed: int | None = None,
) -> None:
    """Generate terrain on an existing grid in-place.

    Steps:
        1. Randomly seed fertile tiles with probability fertile_p.
        2. Run cellular automata to form organic clusters.
           Rule: a tile flips to type X if >= ca_threshold neighbors are X.
        3. Generate mountain ranges via random walks.
    """
    rng = random.Random(seed)

    # Step 1: seed fertile
    for tile in grid:
        tile.terrain = Terrain.FERTILE if rng.random() < fertile_p else Terrain.PLAINS

    # Step 2: cellular automata
    for _ in range(ca_iterations):
        changed = _ca_step(grid, ca_threshold)
        if changed == 0:
            break

    # Step 3: mountain ranges via random walks
    _generate_mountain_ranges(grid, num_ranges, range_steps, turn_prob, rng)


def _ca_step(grid: HexGrid, threshold: int) -> int:
    """One cellular automata pass. Returns number of tiles changed."""
    changes = []
    for tile in grid:
        neighbors = grid.neighbors_of(tile.coord)
        fertile_count = sum(1 for n in neighbors if n.terrain == Terrain.FERTILE)
        plain_count = len(neighbors) - fertile_count

        if tile.terrain != Terrain.FERTILE and fertile_count >= threshold:
            changes.append((tile, Terrain.FERTILE))
        elif tile.terrain == Terrain.FERTILE and plain_count >= threshold:
            changes.append((tile, Terrain.PLAINS))

    for tile, new_terrain in changes:
        tile.terrain = new_terrain
    return len(changes)


def _generate_mountain_ranges(
    grid: HexGrid,
    num_ranges: int,
    range_steps: int,
    turn_prob: float,
    rng: random.Random,
) -> None:
    """Carve mountain ranges as random walks across the grid.

    For each range: pick a random start tile, pick a random direction (0-5),
    then walk range_steps tiles. At each step, continue in the same direction
    with probability (1 - turn_prob), or turn +-1 direction with turn_prob.
    """
    all_tiles = grid.tiles
    for _ in range(num_ranges):
        # Random starting point
        start_tile = rng.choice(all_tiles)
        coord = start_tile.coord
        direction = rng.randint(0, 5)

        for _ in range(range_steps):
            tile = grid.get(coord)
            if tile is None:
                break
            tile.terrain = Terrain.MOUNTAIN

            # Decide next direction: keep straight or turn +-1
            if rng.random() < turn_prob:
                direction = (direction + rng.choice([-1, 1])) % 6

            # Step in current direction
            dc, dr = DIRECTIONS[direction]
            coord = HexCoord(coord.col + dc, coord.row + dr)
