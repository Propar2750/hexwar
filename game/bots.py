"""Rule-based bot strategies for HexWar.

Three baseline bots for game validation and future RL benchmarking:
  - RandomBot: uniformly random legal moves
  - GreedyExpansionBot: maximize territory gain per turn
  - TurtleDefendBot: consolidate, then attack with overwhelming force
"""

from __future__ import annotations

import math
import random
from typing import Protocol

from hex_core import HexCoord
from hex_grid import Terrain

from game.state import GameState
from game.config import GameConfig
from game.actions import (
    MoveAction, EndTurnAction, SetupSupplyChainAction,
    get_valid_targets, validate_supply_chain,
)
from game.combat import win_probability


# ---------------------------------------------------------------------------
# Bot protocol
# ---------------------------------------------------------------------------

class Bot(Protocol):
    """Interface for game-playing agents."""

    name: str

    def choose_action(
        self,
        state: GameState,
        player_id: int,
        config: GameConfig,
    ) -> MoveAction | EndTurnAction | SetupSupplyChainAction:
        """Return the next action for this player.

        Called repeatedly until the bot returns EndTurnAction or the game ends.
        """
        ...


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _movable_tiles(state: GameState, player: int) -> list[HexCoord]:
    """Return owned tiles that have enough troops to move (>= 2)."""
    return [
        c for c in state.owned_coords(player)
        if state.tiles[c].troops >= 2
    ]


def _is_border(state: GameState, coord: HexCoord, player: int) -> bool:
    """True if any neighbor is not owned by player."""
    for n in state.grid.neighbors_of(coord):
        ts = state.tiles.get(n.coord)
        if ts is None or ts.owner != player:
            return True
    return False


def _troops_for_guaranteed_win(defender: int, defense_bonus: float) -> int:
    """Minimum attacker troops for 100% win probability."""
    if defender == 0:
        return 1
    return math.ceil(defense_bonus * (1 + defender + math.sqrt(defender)))


# ---------------------------------------------------------------------------
# RandomBot
# ---------------------------------------------------------------------------

class RandomBot:
    """Picks uniformly random legal moves. No supply chains."""

    name: str = "Random"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose_action(
        self,
        state: GameState,
        player_id: int,
        config: GameConfig,
    ) -> MoveAction | EndTurnAction | SetupSupplyChainAction:
        if config.moves_per_turn > 0 and state.moves_made >= config.moves_per_turn:
            return EndTurnAction()

        sources = _movable_tiles(state, player_id)
        if not sources:
            return EndTurnAction()

        # Try a few random sources until we find one with valid targets
        self.rng.shuffle(sources)
        for src_coord in sources[:10]:
            targets = get_valid_targets(state, src_coord, player_id)
            if not targets:
                continue
            target = self.rng.choice(targets)
            max_troops = state.tiles[src_coord].troops - 1
            troops = self.rng.randint(1, max_troops)
            return MoveAction(src_coord, target, troops)

        return EndTurnAction()


# ---------------------------------------------------------------------------
# GreedyExpansionBot
# ---------------------------------------------------------------------------

class GreedyExpansionBot:
    """Maximize territory gain — prefer high-value unowned tiles."""

    name: str = "Greedy"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose_action(
        self,
        state: GameState,
        player_id: int,
        config: GameConfig,
    ) -> MoveAction | EndTurnAction | SetupSupplyChainAction:
        if config.moves_per_turn > 0 and state.moves_made >= config.moves_per_turn:
            return EndTurnAction()

        # Try supply chains first (before using move slots)
        sc_action = self._try_supply_chain(state, player_id, config)
        if sc_action is not None:
            return sc_action

        best = self._best_attack(state, player_id, config)
        if best is not None:
            return best

        # No good attack — try reinforcing a border tile
        reinforce = self._reinforce(state, player_id, config)
        if reinforce is not None:
            return reinforce

        return EndTurnAction()

    def _best_attack(
        self, state: GameState, player_id: int, config: GameConfig,
    ) -> MoveAction | None:
        candidates: list[tuple[float, HexCoord, HexCoord]] = []

        for src_coord in _movable_tiles(state, player_id):
            src = state.tiles[src_coord]
            attack_troops = src.troops - 1

            for tgt_coord in get_valid_targets(state, src_coord, player_id):
                tgt = state.tiles[tgt_coord]

                # Skip friendly tiles for attack scoring
                if tgt.owner == player_id:
                    continue

                terrain = state.grid[tgt_coord].terrain
                defense = config.defense_bonus[terrain]
                prob = win_probability(attack_troops, tgt.troops, defense)

                if prob < 0.6:
                    continue

                tile_value = config.troop_generation[terrain]
                # Bonus for neutral tiles (cheap to take)
                neutral_bonus = 1.5 if tgt.owner is None else 1.0
                score = tile_value * prob * neutral_bonus

                candidates.append((score, src_coord, tgt_coord))

        if not candidates:
            return None

        # Pick best; break ties randomly
        self.rng.shuffle(candidates)
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, src_coord, tgt_coord = candidates[0]
        troops = state.tiles[src_coord].troops - 1
        return MoveAction(src_coord, tgt_coord, troops)

    def _reinforce(
        self, state: GameState, player_id: int, config: GameConfig,
    ) -> MoveAction | None:
        """Move troops from an interior tile to an adjacent border tile."""
        for src_coord in _movable_tiles(state, player_id):
            if _is_border(state, src_coord, player_id):
                continue  # only move FROM interior tiles
            for n in state.grid.neighbors_of(src_coord):
                if (n.coord in state.tiles
                        and state.tiles[n.coord].owner == player_id
                        and _is_border(state, n.coord, player_id)):
                    troops = state.tiles[src_coord].troops - 1
                    return MoveAction(src_coord, n.coord, troops)
        return None

    def _try_supply_chain(
        self, state: GameState, player_id: int, config: GameConfig,
    ) -> SetupSupplyChainAction | None:
        """Set up a supply chain from an interior tile to an adjacent border tile."""
        if state.supply_chains_set_this_turn.get(player_id, 0) >= 2:
            return None

        # Find interior tiles that don't already have an outgoing chain
        existing_sources = {
            sc.source for sc in state.supply_chains if sc.owner == player_id
        }

        for coord in state.owned_coords(player_id):
            if _is_border(state, coord, player_id):
                continue
            if coord in existing_sources:
                continue

            # Find an adjacent border tile to chain to
            for n in state.grid.neighbors_of(coord):
                if (n.coord in state.tiles
                        and state.tiles[n.coord].owner == player_id
                        and _is_border(state, n.coord, player_id)):
                    action = SetupSupplyChainAction(coord, n.coord)
                    if validate_supply_chain(action, state, player_id) is None:
                        return action

        return None


# ---------------------------------------------------------------------------
# TurtleDefendBot
# ---------------------------------------------------------------------------

class TurtleDefendBot:
    """Consolidate troops, then attack only with overwhelming force."""

    name: str = "Turtle"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose_action(
        self,
        state: GameState,
        player_id: int,
        config: GameConfig,
    ) -> MoveAction | EndTurnAction | SetupSupplyChainAction:
        if config.moves_per_turn > 0 and state.moves_made >= config.moves_per_turn:
            return EndTurnAction()

        # Supply chains — aggressively funnel troops to weakest border
        sc_action = self._try_supply_chain(state, player_id, config)
        if sc_action is not None:
            return sc_action

        # Try a guaranteed-win attack
        attack = self._safe_attack(state, player_id, config)
        if attack is not None:
            return attack

        # Otherwise reinforce: move interior troops toward weakest border
        reinforce = self._reinforce(state, player_id, config)
        if reinforce is not None:
            return reinforce

        return EndTurnAction()

    def _safe_attack(
        self, state: GameState, player_id: int, config: GameConfig,
    ) -> MoveAction | None:
        """Only attack when win probability >= 0.85."""
        candidates: list[tuple[float, int, HexCoord, HexCoord]] = []

        for src_coord in _movable_tiles(state, player_id):
            src = state.tiles[src_coord]

            for tgt_coord in get_valid_targets(state, src_coord, player_id):
                tgt = state.tiles[tgt_coord]
                if tgt.owner == player_id:
                    continue

                terrain = state.grid[tgt_coord].terrain
                defense = config.defense_bonus[terrain]
                prob = win_probability(src.troops - 1, tgt.troops, defense)

                if prob < 0.85:
                    continue

                tile_value = config.troop_generation[terrain]
                # Send only enough for guaranteed win when possible
                needed = _troops_for_guaranteed_win(tgt.troops, defense)
                troops_to_send = min(src.troops - 1, max(needed, 1))

                candidates.append((tile_value * prob, troops_to_send, src_coord, tgt_coord))

        if not candidates:
            return None

        self.rng.shuffle(candidates)
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, troops, src_coord, tgt_coord = candidates[0]
        return MoveAction(src_coord, tgt_coord, troops)

    def _reinforce(
        self, state: GameState, player_id: int, config: GameConfig,
    ) -> MoveAction | None:
        """Move troops from interior tiles toward the weakest border tile."""
        border_tiles = [
            c for c in state.owned_coords(player_id)
            if _is_border(state, c, player_id)
        ]
        if not border_tiles:
            return None

        # Find the weakest border tile
        weakest = min(border_tiles, key=lambda c: state.tiles[c].troops)

        # Find an interior tile adjacent to weakest with troops to spare
        for n in state.grid.neighbors_of(weakest):
            if (n.coord in state.tiles
                    and state.tiles[n.coord].owner == player_id
                    and state.tiles[n.coord].troops >= 2
                    and not _is_border(state, n.coord, player_id)):
                troops = state.tiles[n.coord].troops - 1
                return MoveAction(n.coord, weakest, troops)

        # No interior tile adjacent to weakest — try any interior→border move
        for src_coord in _movable_tiles(state, player_id):
            if _is_border(state, src_coord, player_id):
                continue
            for n in state.grid.neighbors_of(src_coord):
                if (n.coord in state.tiles
                        and state.tiles[n.coord].owner == player_id
                        and _is_border(state, n.coord, player_id)):
                    troops = state.tiles[src_coord].troops - 1
                    return MoveAction(src_coord, n.coord, troops)

        return None

    def _try_supply_chain(
        self, state: GameState, player_id: int, config: GameConfig,
    ) -> SetupSupplyChainAction | None:
        """Chain interior tiles toward the weakest border tile."""
        if state.supply_chains_set_this_turn.get(player_id, 0) >= 2:
            return None

        existing_sources = {
            sc.source for sc in state.supply_chains if sc.owner == player_id
        }

        border_tiles = [
            c for c in state.owned_coords(player_id)
            if _is_border(state, c, player_id)
        ]
        if not border_tiles:
            return None

        weakest = min(border_tiles, key=lambda c: state.tiles[c].troops)

        # Try to chain an interior neighbor of the weakest border tile
        for n in state.grid.neighbors_of(weakest):
            if (n.coord in state.tiles
                    and state.tiles[n.coord].owner == player_id
                    and n.coord not in existing_sources
                    and not _is_border(state, n.coord, player_id)):
                action = SetupSupplyChainAction(n.coord, weakest)
                if validate_supply_chain(action, state, player_id) is None:
                    return action

        # Fallback: any interior → adjacent border
        for coord in state.owned_coords(player_id):
            if _is_border(state, coord, player_id):
                continue
            if coord in existing_sources:
                continue
            for n in state.grid.neighbors_of(coord):
                if (n.coord in state.tiles
                        and state.tiles[n.coord].owner == player_id
                        and _is_border(state, n.coord, player_id)):
                    action = SetupSupplyChainAction(coord, n.coord)
                    if validate_supply_chain(action, state, player_id) is None:
                        return action

        return None


# Bot registry for CLI convenience
BOT_REGISTRY: dict[str, type] = {
    "random": RandomBot,
    "greedy": GreedyExpansionBot,
    "turtle": TurtleDefendBot,
}
