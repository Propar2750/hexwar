"""Temporary test: cellular automata terrain generation on hex grid.

One-off parameter sweep script for tuning map_generator defaults.
Not imported by any other module — can be deleted once tuning is done.

Dependencies:
    hex_grid (HexGrid, Terrain)

Sweeps seed probability p for fertile tiles, runs CA iterations
(threshold=4: a tile becomes fertile if >=4 of its neighbors are fertile),
and reports the final fertile fraction for each p.
"""

import random
from hex_grid import HexGrid, Terrain

WIDTH = 50
HEIGHT = 50
CA_ITERATIONS = 6
THRESHOLDS = [4]
TRIALS = 20    # more trials for tighter stats


def seed_grid(grid: HexGrid, p: float, rng: random.Random) -> None:
    """Randomly set each tile to FERTILE with probability p, else PLAINS."""
    for tile in grid:
        tile.terrain = Terrain.FERTILE if rng.random() < p else Terrain.PLAINS


def ca_step(grid: HexGrid, threshold: int) -> int:
    """One cellular automata pass. Returns number of tiles that changed."""
    changes: list[tuple] = []
    for tile in grid:
        neighbors = grid.neighbors_of(tile.coord)
        fertile_neighbors = sum(1 for n in neighbors if n.terrain == Terrain.FERTILE)
        plain_neighbors = len(neighbors) - fertile_neighbors
        if tile.terrain != Terrain.FERTILE and fertile_neighbors >= threshold:
            changes.append((tile, Terrain.FERTILE))
        elif tile.terrain == Terrain.FERTILE and plain_neighbors >= threshold:
            changes.append((tile, Terrain.PLAINS))
    for tile, new_terrain in changes:
        tile.terrain = new_terrain
    return len(changes)


def fertile_fraction(grid: HexGrid) -> float:
    total = len(grid)
    fertile = sum(1 for t in grid if t.terrain == Terrain.FERTILE)
    return fertile / total


def run_trial(p: float, threshold: int, seed: int) -> float:
    rng = random.Random(seed)
    grid = HexGrid(WIDTH, HEIGHT)
    seed_grid(grid, p, rng)
    for i in range(CA_ITERATIONS):
        changed = ca_step(grid, threshold)
        if changed == 0:
            break
    return fertile_fraction(grid)


def main():
    p_values = [round(0.40 + 0.01 * i, 2) for i in range(11)]  # 0.40 to 0.50
    for threshold in THRESHOLDS:
        print(f"\n{'='*50}")
        print(f"THRESHOLD = {threshold}  (flip if >= {threshold}/6 neighbors fertile)")
        print(f"Grid: {WIDTH}x{HEIGHT} = {WIDTH*HEIGHT} hexes, "
              f"CA iterations: {CA_ITERATIONS}, trials: {TRIALS}")
        print(f"{'p':>6}  {'avg fertile%':>12}  {'min':>6}  {'max':>6}")
        print("-" * 40)

        for p in p_values:
            results = [run_trial(p, threshold, seed=s) for s in range(TRIALS)]
            avg = sum(results) / len(results)
            lo, hi = min(results), max(results)
            marker = " <-- ~35%" if 0.30 <= avg <= 0.40 else ""
            print(f"{p:>6.2f}  {avg:>11.1%}  {lo:>5.1%}  {hi:>5.1%}{marker}")


if __name__ == "__main__":
    main()
