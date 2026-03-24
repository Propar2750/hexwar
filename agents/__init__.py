"""RL agents package for HexWar.

Public API re-exports for agent classes. Currently contains DQN; PPO
will be added in Phase 3. Agents consume game/flat_env for training.

Depended on by: (future training scripts)

Dependencies:
    agents/dqn_agent
"""

from agents.dqn_agent import DQNAgent

__all__ = ["DQNAgent"]
