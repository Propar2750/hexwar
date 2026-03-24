"""Flat-vector Gym-compatible environment for HexWar (v2).

The primary RL training environment. Wraps GameEngine with a flat
observation/action interface suitable for DQN and PPO agents. Also
provides BotFlatAdapter for using rule-based bots as opponents.

Depended on by:
    game/__init__, tests/test_flat_env, (future training scripts)

Dependencies:
    hex_core, hex_grid, game/config, game/engine, game/actions,
    game/combat (win_probability), game/state

Ripple effects:
    - Changing observation encoding → retrain all RL agents.
    - Changing reward shaping → affects learning dynamics.
    - Action space changes → update agents/dqn_agent action selection.

Five innovations over the initial version:
  1. Ego-centric observation (ownership relative to acting player)
  2. Per-tile neighborhood features (6 neighbors × 4 + 4 derived tactical)
  3. Two-tier masking (hard structural + soft state-based logit bias)
  4. Decomposed reward shaping (territory + combat + border + supply chain)
  5. Autoregressive sub-step mode alongside full-turn mode

Observation  (1-D float32)
-----------
Per tile: self features, 6-neighbor features, 4 derived tactical features.
Appended: ego-rotated global features.

Action
------
**Full-turn mode** (``sub_step=False``):
  Integer vector of length ``moves_per_turn + 2``.
  Slots 0..M-1  — move actions  (0 = no-op)
  Slots M..M+1  — supply chain  (0 = no-op)

**Sub-step mode** (``sub_step=True``):
  Single integer from unified Discrete space.
  0 = end turn, 1..N_MOVE = move, N_MOVE+1..N_MOVE+N_SC = supply chain.

Masks
-----
Returned in ``info["action_masks"]``:
  ``hard``  — bool, structural (static per map)
  ``soft``  — float, state-based logit bias (0.0 valid, -10.0 invalid)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hex_core import DIRECTIONS, HexCoord
from hex_grid import Terrain

from game.actions import EndTurnAction, MoveAction, SetupSupplyChainAction
from game.combat import win_probability
from game.config import GameConfig
from game.engine import GameEngine
from game.state import GamePhase

# ── constants ────────────────────────────────────────────────────────

TERRAIN_INDEX = {Terrain.PLAINS: 0, Terrain.MOUNTAIN: 1, Terrain.FERTILE: 2}
N_TERRAIN = len(TERRAIN_INDEX)

TROOP_FRACTIONS = [0.25, 0.5, 0.75, 1.0]
N_TROOP_BINS = len(TROOP_FRACTIONS)

N_DIRECTIONS = 6
SKIP_PENALTY = -0.01
SOFT_MASK_BIAS = -10.0

# Per-neighbor feature count: exists, relative_owner, troop_diff, terrain_defense
_NEIGHBOR_FEAT = 4
# Derived tactical feature count: is_border, attack_opp, threat, border_ratio
_DERIVED_FEAT = 4


# ── environment ──────────────────────────────────────────────────────

class FlatHexWarEnv:
    """Gymnasium-style environment with flat obs, two-tier masks, and sub-step mode."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config: GameConfig | None = None,
        render_mode: str | None = None,
        sub_step: bool = False,
    ) -> None:
        self.config = config or GameConfig()
        self.engine = GameEngine(self.config)
        self.render_mode = render_mode
        self.sub_step = sub_step

        # Space sizing (final n_tiles set at reset)
        self.n_tiles = self.config.grid_width * self.config.grid_height
        self.moves_per_turn: int = self.config.moves_per_turn
        num_p = self.config.num_players

        # Per-tile feature widths
        self._self_feat_width = N_TERRAIN + (num_p + 1) + 1 + 1 + 1 + 1  # 8 + num_p
        self._tile_feat_width = (
            self._self_feat_width + N_DIRECTIONS * _NEIGHBOR_FEAT + _DERIVED_FEAT
        )  # 36 + num_p
        self._global_feat_width = 5 + 2 * num_p
        self.obs_size = self.n_tiles * self._tile_feat_width + self._global_feat_width

        # Action sizes
        self.move_slot_size = self.n_tiles * N_DIRECTIONS * N_TROOP_BINS + 1
        self.sc_slot_size = self.n_tiles * N_DIRECTIONS + 1
        self.action_vector_length = self.moves_per_turn + 2
        # Unified action size for sub-step: end_turn + moves + SCs
        self._n_move_actions = self.n_tiles * N_DIRECTIONS * N_TROOP_BINS
        self._n_sc_actions = self.n_tiles * N_DIRECTIONS
        self.unified_action_size = 1 + self._n_move_actions + self._n_sc_actions

        # Populated at reset
        self._coord_list: list[HexCoord] = []
        self._coord_to_idx: dict[HexCoord, int] = {}
        self._neighbor_table: np.ndarray = np.empty(0)

        # Hard masks (static, bool)
        self._move_hard: np.ndarray = np.empty(0)
        self._sc_hard: np.ndarray = np.empty(0)
        self._unified_hard: np.ndarray = np.empty(0)

        # Sub-step turn state
        self._in_turn = False
        self._turn_player = 0
        self._pre_territory = 0
        self._pre_border_pressure = 0.5
        self._turn_attacks: dict[str, int] = {"won": 0, "total": 0}
        self._turn_penalty = 0.0

        # Pending game-over reward for the player who DIDN'T trigger game-over.
        # In a 2-player alternating game, only the acting player's step gets
        # the +/-10 bonus.  The other player's trajectory needs the bonus too,
        # so we queue it for delivery on the next step() call.
        self._pending_game_over_reward: float | None = None

    # ── Gym API ──────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.config.map_seed = seed
            self.engine = GameEngine(self.config)

        self.engine.reset()

        # Build deterministic coord ordering
        state = self.engine.state
        self._coord_list = sorted(state.tiles.keys(), key=lambda c: (c.row, c.col))
        self._coord_to_idx = {c: i for i, c in enumerate(self._coord_list)}
        self.n_tiles = len(self._coord_list)

        # Recompute sizes with actual tile count
        self._self_feat_width = N_TERRAIN + (self.config.num_players + 1) + 1 + 1 + 1 + 1
        self._tile_feat_width = (
            self._self_feat_width + N_DIRECTIONS * _NEIGHBOR_FEAT + _DERIVED_FEAT
        )
        self._global_feat_width = 5 + 2 * self.config.num_players
        self.obs_size = self.n_tiles * self._tile_feat_width + self._global_feat_width

        self.move_slot_size = self.n_tiles * N_DIRECTIONS * N_TROOP_BINS + 1
        self.sc_slot_size = self.n_tiles * N_DIRECTIONS + 1
        self._n_move_actions = self.n_tiles * N_DIRECTIONS * N_TROOP_BINS
        self._n_sc_actions = self.n_tiles * N_DIRECTIONS
        self.unified_action_size = 1 + self._n_move_actions + self._n_sc_actions

        # Neighbor lookup table
        self._neighbor_table = np.full((self.n_tiles, N_DIRECTIONS), -1, dtype=np.int32)
        for i, coord in enumerate(self._coord_list):
            for d in range(N_DIRECTIONS):
                dc, dr = DIRECTIONS[d]
                nc = HexCoord(coord.col + dc, coord.row + dr)
                if nc in self._coord_to_idx:
                    self._neighbor_table[i, d] = self._coord_to_idx[nc]

        # Build static hard masks
        self._build_hard_masks()

        # Reset sub-step and game-over state
        self._in_turn = False
        self._pending_game_over_reward = None

        obs = self._build_flat_observation()
        return obs, self._info()

    def step(
        self,
        action: np.ndarray | list[int] | int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.sub_step:
            return self._step_sub(int(action) if not isinstance(action, int) else action)
        return self._step_full(np.asarray(action, dtype=np.int64))

    def render(self) -> None:
        """Placeholder — use GameRenderer for visual rendering."""

    # ── full-turn step ───────────────────────────────────────────────

    def _step_full(
        self, action_vector: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        state = self.engine.state

        # Guard: deliver pending reward for the other player, then 0
        if state.phase == GamePhase.GAME_OVER:
            pending = self._pending_game_over_reward or 0.0
            self._pending_game_over_reward = None
            obs = self._build_flat_observation()
            return obs, pending, True, False, self._info()

        acting_player = state.current_player
        pre_territory = state.territory_count(acting_player)
        pre_border = self._border_pressure(acting_player)
        attacks: dict[str, int] = {"won": 0, "total": 0}
        penalty = 0.0

        # Supply chains first
        for sc_int in action_vector[self.moves_per_turn:]:
            if sc_int == 0:
                continue
            sc_action = self._decode_supply_chain(int(sc_int))
            if sc_action is None:
                penalty += SKIP_PENALTY
                continue
            err = self.engine.execute_action(sc_action)
            if err:
                penalty += SKIP_PENALTY

        # Moves (stop early if game ends mid-turn via elimination)
        for move_int in action_vector[:self.moves_per_turn]:
            if state.phase == GamePhase.GAME_OVER:
                break
            if move_int == 0:
                continue
            move_action = self._decode_move(int(move_int))
            if move_action is None:
                penalty += SKIP_PENALTY
                continue
            self._execute_and_track(move_action, acting_player, attacks)
            if attacks.get("_err"):
                penalty += SKIP_PENALTY
                attacks.pop("_err")

        # End turn (engine rejects if already game-over)
        self.engine.execute_action(EndTurnAction())

        reward = self._compute_reward(
            acting_player, pre_territory, pre_border, attacks, penalty,
        )
        done = state.phase == GamePhase.GAME_OVER
        obs = self._build_flat_observation()
        return obs, reward, done, False, self._info()

    # ── sub-step step ────────────────────────────────────────────────

    def _step_sub(
        self, action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        state = self.engine.state

        # Guard: deliver pending reward for the other player, then 0
        if state.phase == GamePhase.GAME_OVER:
            pending = self._pending_game_over_reward or 0.0
            self._pending_game_over_reward = None
            obs = self._build_flat_observation()
            return obs, pending, True, False, self._info()

        if not self._in_turn:
            self._in_turn = True
            self._turn_player = state.current_player
            self._pre_territory = state.territory_count(self._turn_player)
            self._pre_border_pressure = self._border_pressure(self._turn_player)
            self._turn_attacks = {"won": 0, "total": 0}
            self._turn_penalty = 0.0

        player = self._turn_player

        if action == 0:
            # End turn
            self.engine.execute_action(EndTurnAction())
            self._in_turn = False
            reward = self._compute_reward(
                player, self._pre_territory, self._pre_border_pressure,
                self._turn_attacks, self._turn_penalty,
            )
            done = self.engine.state.phase == GamePhase.GAME_OVER
            obs = self._build_flat_observation()
            return obs, reward, done, False, self._info()

        # Decode unified action
        game_action = self._decode_unified(action)
        if game_action is None:
            self._turn_penalty += SKIP_PENALTY
        elif isinstance(game_action, MoveAction):
            self._execute_and_track(game_action, player, self._turn_attacks)
            if self._turn_attacks.get("_err"):
                self._turn_penalty += SKIP_PENALTY
                self._turn_attacks.pop("_err")
        else:
            err = self.engine.execute_action(game_action)
            if err:
                self._turn_penalty += SKIP_PENALTY

        # Check if game ended mid-turn (e.g., player eliminated)
        if self.engine.state.phase == GamePhase.GAME_OVER:
            self._in_turn = False
            reward = self._compute_reward(
                player, self._pre_territory, self._pre_border_pressure,
                self._turn_attacks, self._turn_penalty,
            )
            obs = self._build_flat_observation()
            return obs, reward, True, False, self._info()

        obs = self._build_flat_observation()
        return obs, 0.0, False, False, self._info()

    # ── observation ──────────────────────────────────────────────────

    def _build_flat_observation(self) -> np.ndarray:
        state = self.engine.state
        perspective = state.current_player
        num_p = self.config.num_players
        obs = np.zeros(self.obs_size, dtype=np.float32)
        w = self._tile_feat_width
        sw = self._self_feat_width
        defense_bonus = self.config.defense_bonus

        # Supply chain lookup
        sc_sources: set[HexCoord] = set()
        sc_dests: set[HexCoord] = set()
        for sc in state.supply_chains:
            sc_sources.add(sc.source)
            sc_dests.add(sc.destination)

        for i, coord in enumerate(self._coord_list):
            base = i * w
            tile = state.grid[coord]
            ts = state.tiles[coord]
            my_troops = ts.troops

            # ── self features ──
            obs[base + TERRAIN_INDEX.get(tile.terrain, 0)] = 1.0

            ego_idx = self._ego_owner(ts.owner, perspective)
            obs[base + N_TERRAIN + ego_idx] = 1.0

            obs[base + N_TERRAIN + num_p + 1] = my_troops / (my_troops + 50.0)
            obs[base + N_TERRAIN + num_p + 2] = 1.0  # visibility placeholder
            obs[base + N_TERRAIN + num_p + 3] = 1.0 if coord in sc_sources else 0.0
            obs[base + N_TERRAIN + num_p + 4] = 1.0 if coord in sc_dests else 0.0

            # ── neighborhood features (6 neighbors × 4) ──
            nb_base = base + sw
            is_border = False
            best_attack = 0.0
            best_threat = 0.0
            enemy_border_troops = 0
            my_defense = defense_bonus.get(tile.terrain, 1.0)

            for d in range(N_DIRECTIONS):
                nb_off = nb_base + d * _NEIGHBOR_FEAT
                ni = self._neighbor_table[i, d]
                if ni == -1:
                    continue  # off-grid: zeros

                n_coord = self._coord_list[ni]
                n_tile = state.grid[n_coord]
                n_ts = state.tiles[n_coord]
                n_defense = defense_bonus.get(n_tile.terrain, 1.0)
                n_ego = self._ego_owner(n_ts.owner, perspective)

                obs[nb_off] = 1.0  # exists
                # relative_owner: +1 mine, 0 neutral, -1 enemy
                if n_ego == 0:
                    obs[nb_off + 1] = 0.0   # neutral
                elif n_ego == 1:
                    obs[nb_off + 1] = 1.0   # mine
                else:
                    obs[nb_off + 1] = -1.0  # enemy

                # troop_diff
                total = my_troops + n_ts.troops + 1.0
                obs[nb_off + 2] = (my_troops - n_ts.troops) / total

                # terrain_defense of neighbor
                obs[nb_off + 3] = n_defense

                # ── accumulate for derived features ──
                if n_ego != 1:  # not mine
                    is_border = True
                if n_ego >= 2:  # enemy
                    enemy_border_troops += n_ts.troops
                    # Attack opportunity: can I beat this neighbor?
                    if my_troops >= 2 and n_ts.troops > 0:
                        prob = win_probability(my_troops - 1, n_ts.troops, n_defense)
                        best_attack = max(best_attack, prob)
                    # Threat: can this enemy beat me?
                    if n_ts.troops >= 2 and my_troops > 0:
                        prob = win_probability(n_ts.troops - 1, my_troops, my_defense)
                        best_threat = max(best_threat, prob)
                elif n_ego == 0 and n_ts.troops > 0:
                    # Neutral tile with garrison — also an attack opportunity
                    if my_troops >= 2:
                        prob = win_probability(my_troops - 1, n_ts.troops, n_defense)
                        best_attack = max(best_attack, prob)

            # ── derived tactical features ──
            der_base = nb_base + N_DIRECTIONS * _NEIGHBOR_FEAT
            obs[der_base] = 1.0 if is_border else 0.0
            obs[der_base + 1] = best_attack
            obs[der_base + 2] = best_threat
            ratio = my_troops / (enemy_border_troops + 1.0)
            obs[der_base + 3] = min(ratio, 2.0) / 2.0

        # ── global features (ego-centric) ──
        g = self.n_tiles * w
        obs[g] = min(state.turn, self.config.max_turns) / max(self.config.max_turns, 1)

        # Moves remaining (meaningful in sub-step mode)
        obs[g + 1] = (
            (self.moves_per_turn - state.moves_made) / max(self.moves_per_turn, 1)
        )

        # Supply chains remaining this turn
        sc_set = state.supply_chains_set_this_turn.get(perspective, 0)
        obs[g + 2] = (2 - sc_set) / 2.0

        # Movable tile count
        movable = sum(
            1 for c in self._coord_list
            if state.tiles[c].owner == perspective and state.tiles[c].troops >= 2
        )
        obs[g + 3] = movable / max(self.n_tiles, 1)

        # Active supply chains
        obs[g + 4] = len(state.supply_chains) / 10.0

        # Ego-rotated player stats: mine first, then enemies
        obs[g + 5] = state.territory_count(perspective) / max(self.n_tiles, 1)
        my_total = state.total_troops(perspective)
        enemy_idx = 0
        for p in range(num_p):
            if p == perspective:
                continue
            obs[g + 6 + enemy_idx] = state.territory_count(p) / max(self.n_tiles, 1)
            enemy_idx += 1

        obs[g + 5 + num_p] = my_total / (my_total + 50.0)
        enemy_idx = 0
        for p in range(num_p):
            if p == perspective:
                continue
            et = state.total_troops(p)
            obs[g + 6 + num_p + enemy_idx] = et / (et + 50.0)
            enemy_idx += 1

        return obs

    # ── ego helper ───────────────────────────────────────────────────

    @staticmethod
    def _ego_owner(owner: int | None, perspective: int) -> int:
        """Map absolute owner to ego index: 0=neutral, 1=mine, 2+=enemies."""
        if owner is None:
            return 0
        if owner == perspective:
            return 1
        return 2 + (owner if owner < perspective else owner - 1)

    # ── action masks ─────────────────────────────────────────────────

    def _build_hard_masks(self) -> None:
        """Build structural hard masks (static per map layout)."""
        # Move hard mask
        self._move_hard = np.zeros(self.move_slot_size, dtype=bool)
        self._move_hard[0] = True  # no-op
        for ti in range(self.n_tiles):
            for d in range(N_DIRECTIONS):
                if self._neighbor_table[ti, d] == -1:
                    continue
                for b in range(N_TROOP_BINS):
                    self._move_hard[ti * N_DIRECTIONS * N_TROOP_BINS + d * N_TROOP_BINS + b + 1] = True

        # SC hard mask
        self._sc_hard = np.zeros(self.sc_slot_size, dtype=bool)
        self._sc_hard[0] = True
        for ti in range(self.n_tiles):
            for d in range(N_DIRECTIONS):
                if self._neighbor_table[ti, d] == -1:
                    continue
                self._sc_hard[ti * N_DIRECTIONS + d + 1] = True

        # Unified hard mask (for sub-step mode)
        self._unified_hard = np.zeros(self.unified_action_size, dtype=bool)
        self._unified_hard[0] = True  # end turn
        self._unified_hard[1: 1 + self._n_move_actions] = self._move_hard[1:]
        self._unified_hard[1 + self._n_move_actions:] = self._sc_hard[1:]

    def _build_soft_masks(self) -> dict[str, np.ndarray]:
        """Build state-based soft masks (logit bias). Called each step."""
        state = self.engine.state
        player = state.current_player

        if self.sub_step:
            return self._build_unified_soft(state, player)

        # Full-turn mode: separate move and SC soft masks
        move_soft = np.full(self.move_slot_size, SOFT_MASK_BIAS, dtype=np.float32)
        move_soft[0] = 0.0  # no-op always valid

        sc_soft = np.full(self.sc_slot_size, SOFT_MASK_BIAS, dtype=np.float32)
        sc_soft[0] = 0.0

        # Move: tile owned + >= 2 troops + neighbor exists + moves remaining
        moves_left = self.moves_per_turn - state.moves_made
        if moves_left > 0:
            for ti in range(self.n_tiles):
                ts = state.tiles[self._coord_list[ti]]
                if ts.owner != player or ts.troops < 2:
                    continue
                for d in range(N_DIRECTIONS):
                    if self._neighbor_table[ti, d] == -1:
                        continue
                    for b in range(N_TROOP_BINS):
                        move_soft[ti * N_DIRECTIONS * N_TROOP_BINS + d * N_TROOP_BINS + b + 1] = 0.0

        # SC: both tiles owned, no outgoing chain, chains_set < 2
        sc_set = state.supply_chains_set_this_turn.get(player, 0)
        outgoing_sources = {
            sc.source for sc in state.supply_chains if sc.owner == player
        }
        if sc_set < 2:
            for ti in range(self.n_tiles):
                coord = self._coord_list[ti]
                ts = state.tiles[coord]
                if ts.owner != player or coord in outgoing_sources:
                    continue
                for d in range(N_DIRECTIONS):
                    ni = self._neighbor_table[ti, d]
                    if ni == -1:
                        continue
                    n_ts = state.tiles[self._coord_list[ni]]
                    if n_ts.owner == player:
                        sc_soft[ti * N_DIRECTIONS + d + 1] = 0.0

        return {"move_soft": move_soft, "sc_soft": sc_soft}

    def _build_unified_soft(
        self, state: Any, player: int,
    ) -> dict[str, np.ndarray]:
        """Unified soft mask for sub-step mode."""
        soft = np.full(self.unified_action_size, SOFT_MASK_BIAS, dtype=np.float32)
        soft[0] = 0.0  # end turn always valid

        # Moves: [1, 1 + _n_move_actions) — only if moves remaining
        moves_left = self.moves_per_turn - state.moves_made
        if moves_left > 0:
            for ti in range(self.n_tiles):
                ts = state.tiles[self._coord_list[ti]]
                if ts.owner != player or ts.troops < 2:
                    continue
                for d in range(N_DIRECTIONS):
                    if self._neighbor_table[ti, d] == -1:
                        continue
                    for b in range(N_TROOP_BINS):
                        idx = 1 + ti * N_DIRECTIONS * N_TROOP_BINS + d * N_TROOP_BINS + b
                        soft[idx] = 0.0

        # SCs: [1 + _n_move_actions, end)
        sc_set = state.supply_chains_set_this_turn.get(player, 0)
        outgoing = {sc.source for sc in state.supply_chains if sc.owner == player}
        if sc_set < 2:
            for ti in range(self.n_tiles):
                coord = self._coord_list[ti]
                ts = state.tiles[coord]
                if ts.owner != player or coord in outgoing:
                    continue
                for d in range(N_DIRECTIONS):
                    ni = self._neighbor_table[ti, d]
                    if ni == -1:
                        continue
                    if state.tiles[self._coord_list[ni]].owner == player:
                        idx = 1 + self._n_move_actions + ti * N_DIRECTIONS + d
                        soft[idx] = 0.0

        return {"soft": soft}

    # ── action decode / encode ───────────────────────────────────────

    def _decode_move(self, action_int: int) -> MoveAction | None:
        if action_int <= 0 or action_int >= self.move_slot_size:
            return None
        val = action_int - 1
        tile_idx = val // (N_DIRECTIONS * N_TROOP_BINS)
        remainder = val % (N_DIRECTIONS * N_TROOP_BINS)
        direction = remainder // N_TROOP_BINS
        troop_bin = remainder % N_TROOP_BINS
        if tile_idx >= self.n_tiles:
            return None
        coord = self._coord_list[tile_idx]
        dc, dr = DIRECTIONS[direction]
        target = HexCoord(coord.col + dc, coord.row + dr)
        ts = self.engine.state.tiles.get(coord)
        if ts is None or ts.troops < 2:
            return MoveAction(source=coord, target=target, troops=1)
        available = ts.troops - 1
        troops = max(1, int(TROOP_FRACTIONS[troop_bin] * available))
        return MoveAction(source=coord, target=target, troops=troops)

    def _decode_supply_chain(self, action_int: int) -> SetupSupplyChainAction | None:
        if action_int <= 0 or action_int >= self.sc_slot_size:
            return None
        val = action_int - 1
        tile_idx = val // N_DIRECTIONS
        direction = val % N_DIRECTIONS
        if tile_idx >= self.n_tiles:
            return None
        coord = self._coord_list[tile_idx]
        dc, dr = DIRECTIONS[direction]
        return SetupSupplyChainAction(
            source=coord, destination=HexCoord(coord.col + dc, coord.row + dr),
        )

    def _decode_unified(self, action: int) -> MoveAction | SetupSupplyChainAction | None:
        """Decode a unified sub-step action integer."""
        if action <= 0:
            return None
        if action <= self._n_move_actions:
            return self._decode_move(action)  # same encoding, 1-indexed
        sc_local = action - self._n_move_actions
        return self._decode_supply_chain(sc_local)

    def encode_move(self, action: MoveAction) -> int:
        tile_idx = self._coord_to_idx.get(action.source)
        if tile_idx is None:
            return 0
        dc = action.target.col - action.source.col
        dr = action.target.row - action.source.row
        try:
            direction = DIRECTIONS.index((dc, dr))
        except ValueError:
            return 0
        ts = self.engine.state.tiles.get(action.source)
        available = (ts.troops - 1) if ts and ts.troops > 1 else 1
        frac = action.troops / max(available, 1)
        best_bin = min(range(N_TROOP_BINS), key=lambda b: abs(TROOP_FRACTIONS[b] - frac))
        return tile_idx * N_DIRECTIONS * N_TROOP_BINS + direction * N_TROOP_BINS + best_bin + 1

    def encode_supply_chain(self, action: SetupSupplyChainAction) -> int:
        tile_idx = self._coord_to_idx.get(action.source)
        if tile_idx is None:
            return 0
        dc = action.destination.col - action.source.col
        dr = action.destination.row - action.source.row
        try:
            direction = DIRECTIONS.index((dc, dr))
        except ValueError:
            return 0
        return tile_idx * N_DIRECTIONS + direction + 1

    def encode_unified(self, action: MoveAction | SetupSupplyChainAction | EndTurnAction) -> int:
        """Encode a game action to a unified sub-step integer."""
        if isinstance(action, EndTurnAction):
            return 0
        if isinstance(action, MoveAction):
            return self.encode_move(action)  # 1-indexed, same range
        if isinstance(action, SetupSupplyChainAction):
            sc_encoded = self.encode_supply_chain(action)
            if sc_encoded == 0:
                return 0  # invalid SC → treat as end turn
            return self._n_move_actions + sc_encoded
        return 0

    # ── reward ───────────────────────────────────────────────────────

    def _execute_and_track(
        self,
        action: MoveAction,
        player: int,
        attacks: dict[str, int],
    ) -> None:
        """Execute a move and track attack outcomes."""
        state = self.engine.state
        target_ts = state.tiles.get(action.target)
        is_attack = target_ts is not None and target_ts.owner != player
        pre_count = state.territory_count(player)

        err = self.engine.execute_action(action)
        if err:
            attacks["_err"] = 1
            return

        if is_attack:
            attacks["total"] += 1
            if state.territory_count(player) > pre_count:
                attacks["won"] += 1

    def _compute_reward(
        self,
        player: int,
        pre_territory: int,
        pre_border: float,
        attacks: dict[str, int],
        penalty: float,
    ) -> float:
        state = self.engine.state
        new_territory = state.territory_count(player)
        territory_delta = float(new_territory - pre_territory)

        combat_eff = (
            attacks["won"] / attacks["total"] if attacks["total"] > 0 else 0.0
        )

        new_border = self._border_pressure(player)
        border_delta = new_border - pre_border

        # Supply chain value: troop generation flowing through my chains
        sc_value = 0.0
        troop_gen = self.config.troop_generation
        for sc in state.supply_chains:
            if sc.owner == player:
                dest_tile = state.grid[sc.destination]
                sc_value += troop_gen.get(dest_tile.terrain, 1)
        sc_value /= 10.0

        # Shaping weights are intentionally small relative to the +/-10
        # win/loss bonus so that cumulative shaping over a full game
        # (~20 turns) stays well under 10 — otherwise the loser
        # accumulates enough positive shaping to end up net-positive.
        reward = (
            territory_delta * 0.1
            + combat_eff * 0.05
            + border_delta * 0.03
            + sc_value * 0.02
            + penalty
        )

        if state.phase == GamePhase.GAME_OVER and state.winner is not None:
            reward += 10.0 if state.winner == player else -10.0
            # Queue the OTHER player's game-over bonus for the next step() call.
            # In a 2-player game, only the acting player's step gets the bonus;
            # the other player needs it delivered via the done guard.
            for p in range(self.config.num_players):
                if p != player:
                    self._pending_game_over_reward = (
                        10.0 if state.winner == p else -10.0
                    )
                    break  # only one pending reward (2-player)

        return reward

    def _border_pressure(self, player: int) -> float:
        """Ratio of my border troops to enemy border troops (0..1 scale)."""
        state = self.engine.state
        my_border = 0
        enemy_border = 0
        for i, coord in enumerate(self._coord_list):
            ts = state.tiles[coord]
            if ts.owner != player:
                continue
            on_border = False
            for d in range(N_DIRECTIONS):
                ni = self._neighbor_table[i, d]
                if ni == -1:
                    continue
                n_ts = state.tiles[self._coord_list[ni]]
                if n_ts.owner is not None and n_ts.owner != player:
                    on_border = True
                    enemy_border += n_ts.troops
            if on_border:
                my_border += ts.troops
        total = my_border + enemy_border
        return my_border / total if total > 0 else 0.5

    # ── info ─────────────────────────────────────────────────────────

    def _info(self) -> dict[str, Any]:
        state = self.engine.state

        # Build soft masks (dynamic)
        soft_masks = self._build_soft_masks()

        if self.sub_step:
            return {
                "turn": state.turn,
                "current_player": state.current_player,
                "phase": state.phase.name,
                "territory": {
                    p: state.territory_count(p) for p in range(state.num_players)
                },
                "action_masks": {
                    "hard": self._unified_hard,
                    **soft_masks,
                },
            }

        return {
            "turn": state.turn,
            "current_player": state.current_player,
            "phase": state.phase.name,
            "territory": {
                p: state.territory_count(p) for p in range(state.num_players)
            },
            "action_masks": {
                "move_hard": self._move_hard,
                "sc_hard": self._sc_hard,
                **soft_masks,
            },
        }


# ── bot adapter ──────────────────────────────────────────────────────

class BotFlatAdapter:
    """Wraps a Bot to produce actions for FlatHexWarEnv in either mode."""

    def __init__(self, bot: Any) -> None:
        self.bot = bot

    @property
    def name(self) -> str:
        return self.bot.name

    def choose_action_vector(self, env: FlatHexWarEnv) -> np.ndarray:
        """Full-turn mode: produce action vector by querying bot repeatedly."""
        state = env.engine.state
        player_id = state.current_player
        config = env.config
        vec = np.zeros(env.action_vector_length, dtype=np.int64)
        move_idx = 0
        sc_idx = 0

        max_calls = env.moves_per_turn + 4
        for _ in range(max_calls):
            action = self.bot.choose_action(state, player_id, config)
            if isinstance(action, EndTurnAction):
                break
            if isinstance(action, MoveAction):
                if move_idx < env.moves_per_turn:
                    vec[move_idx] = env.encode_move(action)
                    move_idx += 1
            elif isinstance(action, SetupSupplyChainAction):
                if sc_idx < 2:
                    vec[env.moves_per_turn + sc_idx] = env.encode_supply_chain(action)
                    sc_idx += 1
        return vec

    def choose_sub_action(self, env: FlatHexWarEnv) -> int:
        """Sub-step mode: produce one unified action integer."""
        state = env.engine.state
        player_id = state.current_player
        action = self.bot.choose_action(state, player_id, env.config)
        return env.encode_unified(action)
