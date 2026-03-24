"""Gym-compatible environment wrapper for HexWar.

Follows the Gymnasium API (reset / step / render) without requiring
gymnasium as a dependency.  Drop-in compatible if you later want to
register it via gymnasium.make().

Observation: (C, H, W) float32 array
    Channel 0 — terrain type  (0=plains, 1=mountain, 2=fertile)
    Channel 1 — ownership     (-1=unowned, 0..N-1 = player id, normalised to [-1,1])
    Channel 2 — troop count   (raw int, can be normalised externally)
    Channel 3 — current player mask (1.0 for tiles owned by the acting player)

Action: dict with keys
    "source_index" : int   — index into the sorted tile list
    "direction"    : int   — 0..5 hex direction
    "troops"       : int   — troops to send (clamped to valid range)
    Special: source_index == -1 means EndTurn.

To add new observation channels (fog, resources, new troop types),
extend _build_observation().
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hex_core import DIRECTIONS, HexCoord
from hex_grid import Terrain

from game.actions import EndTurnAction, MoveAction, SetupSupplyChainAction
from game.config import GameConfig
from game.engine import GameEngine
from game.state import GamePhase


TERRAIN_INDEX = {Terrain.PLAINS: 0, Terrain.MOUNTAIN: 1, Terrain.FERTILE: 2}


class HexWarEnv:
    """Gymnasium-style environment for HexWar."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config: GameConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        self.config = config or GameConfig()
        self.engine = GameEngine(self.config)
        self.render_mode = render_mode

        # Sorted coordinate list for index ↔ coord mapping
        self._coord_list: list[HexCoord] = []
        self._coord_to_idx: dict[HexCoord, int] = {}

        # Spaces (shapes only — no gymnasium dependency)
        self.n_channels = 4
        self.obs_shape = (self.n_channels, self.config.grid_height, self.config.grid_width)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.config.map_seed = seed
            self.engine = GameEngine(self.config)
        state = self.engine.reset()

        # Build coord list (sorted for deterministic indexing)
        self._coord_list = sorted(state.tiles.keys(), key=lambda c: (c.row, c.col))
        self._coord_to_idx = {c: i for i, c in enumerate(self._coord_list)}

        # Auto-place starting positions in corners for RL
        if options and options.get("auto_place", False):
            corners = self._get_corners()
            for coord in corners[: self.config.num_players]:
                self.engine.place_starting_position(coord)

        obs = self._build_observation()
        return obs, self._info()

    def step(
        self, action: dict[str, int],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one action.

        action keys: source_index, direction, troops.
        source_index == -1 → EndTurn.
        source_index == -2 → SetupSupplyChain (requires dest_index).
        """
        state = self.engine.state
        prev_territory = state.territory_count(state.current_player)

        src_idx = action.get("source_index", -1)
        if src_idx == -2:
            # Supply chain action
            sc_src = action.get("sc_source_index", 0)
            sc_dst = action.get("sc_dest_index", 0)
            source = self._coord_list[sc_src]
            dest = self._coord_list[sc_dst]
            sc_action = SetupSupplyChainAction(source=source, destination=dest)
            err = self.engine.execute_action(sc_action)
            if err:
                obs = self._build_observation()
                return obs, -0.01, False, False, {**self._info(), "error": err}
        elif src_idx == -1:
            self.engine.execute_action(EndTurnAction())
        else:
            coord = self._coord_list[src_idx]
            direction = action.get("direction", 0)
            dc, dr = DIRECTIONS[direction % 6]
            target = HexCoord(coord.col + dc, coord.row + dr)
            troops = action.get("troops", 1)

            # Clamp troops
            ts = state.tiles.get(coord)
            if ts and ts.troops > 1:
                troops = max(1, min(troops, ts.troops - 1))

            move = MoveAction(source=coord, target=target, troops=troops)
            err = self.engine.execute_action(move)
            if err:
                # Invalid action — small penalty, no state change
                obs = self._build_observation()
                return obs, -0.01, False, False, {**self._info(), "error": err}

        # Compute reward
        state = self.engine.state
        new_territory = state.territory_count(state.current_player)
        reward = float(new_territory - prev_territory)

        done = state.phase == GamePhase.GAME_OVER
        truncated = state.turn > self.config.max_turns if not done else False

        if done and state.winner is not None:
            reward += 10.0 if state.winner == state.current_player else -10.0

        obs = self._build_observation()
        return obs, reward, done, truncated, self._info()

    def render(self) -> None:
        """Placeholder — use GameRenderer for visual rendering."""
        pass

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def _build_observation(self) -> np.ndarray:
        """Encode game state as a multi-channel 2D array."""
        state = self.engine.state
        H, W = self.config.grid_height, self.config.grid_width
        obs = np.zeros((self.n_channels, H, W), dtype=np.float32)

        for tile in state.grid:
            coord = tile.coord
            # Map doubled-width col back to grid column index
            c = (coord.col - (coord.row % 2)) // 2
            r = coord.row
            if 0 <= r < H and 0 <= c < W:
                ts = state.tiles[coord]
                obs[0, r, c] = TERRAIN_INDEX.get(tile.terrain, 0)
                obs[1, r, c] = ts.owner if ts.owner is not None else -1
                obs[2, r, c] = ts.troops
                obs[3, r, c] = 1.0 if ts.owner == state.current_player else 0.0

        return obs

    def _info(self) -> dict[str, Any]:
        state = self.engine.state
        return {
            "turn": state.turn,
            "current_player": state.current_player,
            "phase": state.phase.name,
            "territory": {
                p: state.territory_count(p) for p in range(state.num_players)
            },
        }

    def _get_corners(self) -> list[HexCoord]:
        """Return corner-ish tile coordinates for starting placement."""
        coords = self._coord_list
        if not coords:
            return []

        # Find actual tiles closest to each geometric corner
        min_r = min(c.row for c in coords)
        max_r = max(c.row for c in coords)
        min_c = min(c.col for c in coords)
        max_c = max(c.col for c in coords)

        corner_targets = [
            (min_c, min_r),  # top-left
            (max_c, max_r),  # bottom-right
            (max_c, min_r),  # top-right
            (min_c, max_r),  # bottom-left
        ]
        result = []
        for tc, tr in corner_targets:
            best = min(coords, key=lambda c: abs(c.col - tc) + abs(c.row - tr))
            if best not in result:
                result.append(best)
        return result
