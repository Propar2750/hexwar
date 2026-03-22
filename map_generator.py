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
DEFAULT_CA_ITERATIONS = 5
DEFAULT_CA_THRESHOLD = 4
DEFAULT_NUM_RANGES = 12
DEFAULT_MIN_RANGE_STEPS = 6
DEFAULT_RANGE_END_PROB = 0.1
DEFAULT_STRAIGHT_PROB = 0.32
DEFAULT_SLIGHT_TURN_PROB = 0.28
DEFAULT_HARD_TURN_PROB = 0.06


def generate_terrain(
    grid: HexGrid,
    fertile_p: float = DEFAULT_FERTILE_SEED_P,
    ca_iterations: int = DEFAULT_CA_ITERATIONS,
    ca_threshold: int = DEFAULT_CA_THRESHOLD,
    num_ranges: int = DEFAULT_NUM_RANGES,
    min_range_steps: int = DEFAULT_MIN_RANGE_STEPS,
    range_end_prob: float = DEFAULT_RANGE_END_PROB,
    straight_prob: float = DEFAULT_STRAIGHT_PROB,
    slight_turn_prob: float = DEFAULT_SLIGHT_TURN_PROB,
    hard_turn_prob: float = DEFAULT_HARD_TURN_PROB,
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
    _generate_mountain_ranges(
        grid, num_ranges, min_range_steps, range_end_prob,
        straight_prob, slight_turn_prob, hard_turn_prob, rng,
    )


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
    min_steps: int,
    end_prob: float,
    straight_prob: float,
    slight_turn_prob: float,
    hard_turn_prob: float,
    rng: random.Random,
) -> None:
    """Carve mountain ranges as random walks across the grid.

    For each range: pick a random start tile, pick a random direction (0-5),
    then walk at least min_steps tiles. After the minimum, each additional
    step has end_prob chance of ending the range. Direction changes are
    weighted:
        straight (0):    straight_prob  (0.32)
        slight +-1:      slight_turn_prob each (0.24)
        hard +-2:        hard_turn_prob each (0.10)
    """
    turn_weights = [hard_turn_prob, slight_turn_prob, straight_prob, slight_turn_prob, hard_turn_prob]
    turn_offsets = [-2, -1, 0, 1, 2]

    all_tiles = grid.tiles
    for _ in range(num_ranges):
        # Random starting point
        start_tile = rng.choice(all_tiles)
        coord = start_tile.coord
        direction = rng.randint(0, 5)

        step = 0
        while True:
            tile = grid.get(coord)
            if tile is None:
                break
            tile.terrain = Terrain.MOUNTAIN
            step += 1

            # After minimum steps, chance to end each step
            if step >= min_steps and rng.random() < end_prob:
                break

            # Weighted direction change
            offset = rng.choices(turn_offsets, weights=turn_weights, k=1)[0]
            direction = (direction + offset) % 6

            # Step in current direction
            dc, dr = DIRECTIONS[direction]
            coord = HexCoord(coord.col + dc, coord.row + dr)
