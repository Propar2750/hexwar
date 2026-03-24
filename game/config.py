"""Game configuration — all tunable parameters live here.

To add a new terrain type: add it to hex_grid.Terrain, then add entries
to troop_generation, defense_bonus, and movement_cost dicts here.

To add a new troop type: extend this config with per-type stats and
update the engine/combat modules to handle them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hex_core import HexCoord
from hex_grid import Terrain


class MapPreset(str, Enum):
    """Named map-generation presets tuned for different grid sizes."""
    SMALL = "small"
    SMALL_FIXED = "small-fixed"
    MEDIUM = "medium"
    LARGE = "large"


# ── Fixed map for SMALL_FIXED preset ──────────────────────────
# 8×6 grid (48 tiles) with 180° rotational symmetry.
# Mountain ridge through the centre, fertile clusters at each corner.
#
#   row 0:  F  F  P  P | P  P  P  P
#   row 1:    F  F  F  M| P  P  P  P
#   row 2:  P  P  F  M  M  P  P  P
#   row 3:    P  P  P  M  M  F  P  P
#   row 4:  P  P  P  P |M  F  F  F
#   row 5:    P  P  P  P| P  P  F  F
#
_F, _P, _M = Terrain.FERTILE, Terrain.PLAINS, Terrain.MOUNTAIN

SMALL_FIXED_TERRAIN: dict[tuple[int, int], Terrain] = {
    # row 0 (even): cols 0,2,4,6,8,10,12,14
    (0, 0): _F,  (2, 0): _F,  (4, 0): _P,  (6, 0): _P,
    (8, 0): _P,  (10, 0): _P, (12, 0): _P, (14, 0): _P,
    # row 1 (odd): cols 1,3,5,7,9,11,13,15
    (1, 1): _F,  (3, 1): _F,  (5, 1): _F,  (7, 1): _M,
    (9, 1): _P,  (11, 1): _P, (13, 1): _P, (15, 1): _P,
    # row 2 (even)
    (0, 2): _P,  (2, 2): _P,  (4, 2): _F,  (6, 2): _M,
    (8, 2): _M,  (10, 2): _P, (12, 2): _P, (14, 2): _P,
    # row 3 (odd)  — 180° rotation of row 2
    (1, 3): _P,  (3, 3): _P,  (5, 3): _P,  (7, 3): _M,
    (9, 3): _M,  (11, 3): _F, (13, 3): _P, (15, 3): _P,
    # row 4 (even) — 180° rotation of row 1
    (0, 4): _P,  (2, 4): _P,  (4, 4): _P,  (6, 4): _P,
    (8, 4): _M,  (10, 4): _F, (12, 4): _F, (14, 4): _F,
    # row 5 (odd)  — 180° rotation of row 0
    (1, 5): _P,  (3, 5): _P,  (5, 5): _P,  (7, 5): _P,
    (9, 5): _P,  (11, 5): _P, (13, 5): _F, (15, 5): _F,
}

SMALL_FIXED_STARTS: list[tuple[int, int]] = [
    (1, 1),   # Player 1 — centre of left fertile cluster
    (14, 4),  # Player 2 — 180° mirror, centre of right fertile cluster
]

# ── Preset parameter tables ──────────────────────────────────

# Full parameter set for each preset.
# Keys match GameConfig field names so they can be applied directly.
MAP_PRESET_PARAMS: dict[MapPreset, dict[str, Any]] = {
    MapPreset.SMALL: dict(
        grid_width=8,
        grid_height=6,
        ca_threshold=3,
        ca_iterations=4,
        fertile_p=0.45,
        num_mountain_ranges=2,
        min_range_steps=2,
        range_end_prob=0.35,
        min_start_distance=3,
        max_turns=20,
        moves_per_turn=4,
    ),
    MapPreset.SMALL_FIXED: dict(
        grid_width=8,
        grid_height=6,
        ca_threshold=3,      # unused — terrain is fixed
        ca_iterations=0,     # unused
        fertile_p=0.0,       # unused
        num_mountain_ranges=0,  # unused
        min_range_steps=0,   # unused
        range_end_prob=0.0,  # unused
        min_start_distance=3,
        max_turns=20,
        moves_per_turn=4,
    ),
    MapPreset.MEDIUM: dict(
        grid_width=14,
        grid_height=10,
        ca_threshold=4,
        ca_iterations=5,
        fertile_p=0.45,
        num_mountain_ranges=4,
        min_range_steps=3,
        range_end_prob=0.20,
        min_start_distance=4,
        max_turns=30,
        moves_per_turn=6,
    ),
    MapPreset.LARGE: dict(
        grid_width=20,
        grid_height=14,
        ca_threshold=4,
        ca_iterations=6,
        fertile_p=0.45,
        num_mountain_ranges=10,
        min_range_steps=5,
        range_end_prob=0.12,
        min_start_distance=6,
        max_turns=50,
        moves_per_turn=8,
    ),
}

# Sentinel to distinguish "user didn't set this" from an actual value.
_UNSET: Any = object()


@dataclass
class GameConfig:
    """Central, immutable-ish configuration for a HexWar game."""

    # --- Preset (applied in __post_init__) ---
    preset: MapPreset | None = None

    # --- Players & map ---
    num_players: int = 2
    grid_width: int = _UNSET
    grid_height: int = _UNSET
    max_turns: int = _UNSET
    win_threshold: float = 0.50  # fraction of map to win
    starting_troops: int = 10
    min_start_distance: int = _UNSET
    moves_per_turn: int = _UNSET

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
    num_mountain_ranges: int = _UNSET
    min_range_steps: int = _UNSET
    range_end_prob: float = _UNSET
    fertile_p: float = _UNSET
    ca_threshold: int = _UNSET
    ca_iterations: int = _UNSET

    # --- Balanced auto-placement ---
    auto_place_starts: bool = True
    balance_radius: int = 3
    balance_threshold: float = 0.70

    def __post_init__(self) -> None:
        # Use MEDIUM as the fallback preset for any unset values.
        base = MAP_PRESET_PARAMS[self.preset] if self.preset else MAP_PRESET_PARAMS[MapPreset.MEDIUM]
        for key, value in base.items():
            if getattr(self, key) is _UNSET:
                object.__setattr__(self, key, value)
