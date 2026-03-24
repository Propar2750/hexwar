"""Action types and validation.

Every player action is a dataclass.  To add a new action kind (e.g.
BuildAction for a new troop type), define it here and teach
GameEngine.execute_action() to handle it.
"""

from __future__ import annotations

from dataclasses import dataclass

from hex_core import HexCoord
from game.state import GameState, GamePhase, SupplyChain


@dataclass(frozen=True)
class MoveAction:
    """Move troops from one tile to an adjacent tile."""

    source: HexCoord
    target: HexCoord
    troops: int  # number of troops to send (source keeps the rest)


@dataclass(frozen=True)
class SetupSupplyChainAction:
    """Create a supply chain between two adjacent owned tiles."""

    source: HexCoord
    destination: HexCoord


@dataclass(frozen=True)
class EndTurnAction:
    """Current player ends their turn."""

    pass


# ----- validation -----

def validate_move(action: MoveAction, state: GameState, player: int) -> str | None:
    """Return an error string, or None if the move is legal."""
    if state.phase != GamePhase.PLAY:
        return "Game is not in PLAY phase"

    if state.current_player != player:
        return "Not your turn"

    src = state.tiles.get(action.source)
    if src is None:
        return "Source tile does not exist"
    if src.owner != player:
        return "You do not own the source tile"
    if action.troops < 1:
        return "Must send at least 1 troop"
    if action.troops >= src.troops:
        return "Must leave at least 1 troop behind"

    # Target must be adjacent
    if action.target not in [n.coord for n in state.grid.neighbors_of(action.source)]:
        return "Target is not adjacent to source"

    if action.target not in state.tiles:
        return "Target tile does not exist"

    return None


def validate_supply_chain(
    action: SetupSupplyChainAction,
    state: GameState,
    player: int,
) -> str | None:
    """Return an error string, or None if the supply chain setup is legal."""
    if state.phase != GamePhase.PLAY:
        return "Game is not in PLAY phase"
    if state.current_player != player:
        return "Not your turn"
    if player in state.supply_chain_set_this_turn:
        return "Already set up a supply chain this turn"

    src = state.tiles.get(action.source)
    if src is None:
        return "Source tile does not exist"
    if src.owner != player:
        return "You do not own the source tile"

    dst = state.tiles.get(action.destination)
    if dst is None:
        return "Destination tile does not exist"
    if dst.owner != player:
        return "You do not own the destination tile"

    if action.source == action.destination:
        return "Source and destination must be different"

    # Must be adjacent
    if action.destination not in [n.coord for n in state.grid.neighbors_of(action.source)]:
        return "Destination must be adjacent to source"

    # Only one outgoing chain per source tile
    if any(sc.source == action.source and sc.owner == player for sc in state.supply_chains):
        return "This tile already has an outgoing supply chain"

    # No cycles — walk outgoing chains from destination back toward source
    if _would_create_cycle(action.source, action.destination, player, state.supply_chains):
        return "Would create a supply chain cycle"

    return None


def _would_create_cycle(
    source: HexCoord,
    destination: HexCoord,
    player: int,
    supply_chains: list[SupplyChain],
) -> bool:
    """Return True if adding source→destination creates a cycle."""
    outgoing = {sc.source: sc.destination for sc in supply_chains if sc.owner == player}
    current = destination
    visited: set[HexCoord] = set()
    while current in outgoing:
        if current == source:
            return True
        if current in visited:
            break
        visited.add(current)
        current = outgoing[current]
    return current == source


def get_valid_targets(state: GameState, source: HexCoord, player: int) -> list[HexCoord]:
    """Return adjacent coords that the player could legally move to from source."""
    src = state.tiles.get(source)
    if src is None or src.owner != player or src.troops < 2:
        return []
    return [
        n.coord
        for n in state.grid.neighbors_of(source)
        if n.coord in state.tiles
    ]
