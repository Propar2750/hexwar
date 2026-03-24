"""Game configuration — all tunable parameters live here.

To add a new terrain type: add it to hex_grid.Terrain, then add entries
to troop_generation, defense_bonus, and movement_cost dicts here.

To add a new troop type: extend this config with per-type stats and
update the engine/combat modules to handle them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hex_grid import Terrain


@dataclass
class GameConfig:
    """Central, immutable-ish configuration for a HexWar game."""

    # --- Players & map ---
    num_players: int = 2
    grid_width: int = 14
    grid_height: int = 10
    max_turns: int = 30
    win_threshold: float = 0.50  # fraction of map to win
    starting_troops: int = 10
    min_start_distance: int = 4  # minimum hex distance between starting positions
    moves_per_turn: int = 6

    # --- Terrain parameters (keyed by Terrain enum) ---
    troop_generation: dict[Terrain, int] = field(default_factory=lambda: {
        Terrain.PLAINS: 2,
        Terrain.FERTILE: 3,
        Terrain.MOUNTAIN: 1,
    })

    defense_bonus: dict[Terrain, float] = field(default_factory=lambda: {
        Terrain.PLAINS: 1.0,
        Terrain.FERTILE: 1.0,
        Terrain.MOUNTAIN: 2.0,
    })

    movement_cost: dict[Terrain, float] = field(default_factory=lambda: {
        Terrain.PLAINS: 1.0,
        Terrain.FERTILE: 1.0,
        Terrain.MOUNTAIN: 3.0,
    })

    # --- Map generation (forwarded to map_generator) ---
    map_seed: int | None = None
    num_mountain_ranges: int = 4
    min_range_steps: int = 3
    range_end_prob: float = 0.2
