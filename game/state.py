"""Game state — the complete snapshot of a game at any point in time.

GameState is intentionally a plain data container. All mutation goes
through GameEngine so that rules are enforced in one place.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto

from hex_core import HexCoord
from hex_grid import HexGrid


class GamePhase(Enum):
    """High-level phase of the game."""

    SETUP = auto()       # players choosing starting positions
    PLAY = auto()        # main game loop
    GAME_OVER = auto()   # someone won (or draw)


@dataclass
class TileState:
    """Per-tile ownership and troop data."""

    owner: int | None = None  # player index, or None = unowned
    troops: int = 0


@dataclass
class SupplyChain:
    """A troop supply chain between two adjacent owned tiles.

    Each round, all troops on *source* (except 1) are automatically
    forwarded to *destination*.  The chain is destroyed when either
    endpoint loses ownership.
    """

    source: HexCoord
    destination: HexCoord
    owner: int  # player who created the chain


class GameState:
    """Complete, serialisable game state.

    The hex grid topology and terrain live in ``grid`` (read-only during
    play).  Ownership / troop data is in ``tiles``, keyed by HexCoord.
    """

    def __init__(self, grid: HexGrid, num_players: int) -> None:
        self.grid = grid
        self.num_players = num_players
        self.tiles: dict[HexCoord, TileState] = {
            tile.coord: TileState(owner=None, troops=1) for tile in grid
        }
        self.phase: GamePhase = GamePhase.SETUP
        self.current_player: int = 0
        self.turn: int = 0          # increments after all players act
        self.players_placed: int = 0  # how many have chosen a start
        self.winner: int | None = None
        self.moves_made: int = 0    # moves made this turn by current player
        self.supply_chains: list[SupplyChain] = []
        self.supply_chains_set_this_turn: dict[int, int] = {}  # player -> count of chains set this turn (max 2)

    # ----- queries -----

    def territory_count(self, player: int) -> int:
        return sum(1 for ts in self.tiles.values() if ts.owner == player)

    def total_troops(self, player: int) -> int:
        return sum(
            ts.troops for ts in self.tiles.values() if ts.owner == player
        )

    def owned_coords(self, player: int) -> list[HexCoord]:
        return [c for c, ts in self.tiles.items() if ts.owner == player]

    def is_alive(self, player: int) -> bool:
        return self.territory_count(player) > 0

    # ----- copy -----

    def clone(self) -> GameState:
        return deepcopy(self)
