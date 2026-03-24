"""HexWar game package — modular strategy game on a hex grid.

Public API re-exports. Import from here (e.g. `from game import GameEngine`)
rather than reaching into submodules directly. When adding a new submodule,
add its key symbols to the imports and __all__ below.

Depended on by:
    play, bot_runner (and any future training scripts)

Dependencies:
    game/config, game/state, game/combat, game/actions, game/engine,
    game/bots, game/flat_env
"""

from game.config import GameConfig
from game.state import GamePhase, GameState, TileState
from game.combat import CombatResult, CombatResolver, DefaultCombatResolver
from game.actions import MoveAction, EndTurnAction, SetupSupplyChainAction
from game.combat import win_probability
from game.engine import GameEngine
from game.bots import Bot, RandomBot, GreedyExpansionBot, TurtleDefendBot, NoOpBot, BOT_REGISTRY
from game.flat_env import FlatHexWarEnv, BotFlatAdapter

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
    "NoOpBot",
    "BOT_REGISTRY",
    "FlatHexWarEnv",
    "BotFlatAdapter",
]
