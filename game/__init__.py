"""HexWar game package — modular strategy game on a hex grid."""

from game.config import GameConfig
from game.state import GamePhase, GameState, TileState
from game.combat import CombatResult, CombatResolver, DefaultCombatResolver
from game.actions import MoveAction, EndTurnAction
from game.engine import GameEngine

__all__ = [
    "GameConfig",
    "GamePhase",
    "GameState",
    "TileState",
    "CombatResult",
    "CombatResolver",
    "DefaultCombatResolver",
    "MoveAction",
    "EndTurnAction",
    "GameEngine",
]
