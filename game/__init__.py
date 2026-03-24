"""HexWar game package — modular strategy game on a hex grid."""

from game.config import GameConfig
from game.state import GamePhase, GameState, TileState
from game.combat import CombatResult, CombatResolver, DefaultCombatResolver
from game.actions import MoveAction, EndTurnAction, SetupSupplyChainAction
from game.combat import win_probability
from game.engine import GameEngine
from game.bots import Bot, RandomBot, GreedyExpansionBot, TurtleDefendBot, BOT_REGISTRY

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
    "SetupSupplyChainAction",
    "win_probability",
    "GameEngine",
    "Bot",
    "RandomBot",
    "GreedyExpansionBot",
    "TurtleDefendBot",
    "BOT_REGISTRY",
]
