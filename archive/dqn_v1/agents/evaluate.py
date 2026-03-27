"""Greedy evaluation harness for DQN agent with game replay recording."""

from __future__ import annotations

import numpy as np

from game.bots import NoOpBot
from game.config import GameConfig
from game.flat_env import BotFlatAdapter, FlatHexWarEnv

from agents.dqn_agent import DQNAgent


# ── Replay data structure ───────────────────────────────────────────

class TurnSnapshot:
    """Lightweight record of one agent turn for text display."""

    __slots__ = ("turn", "actions_taken", "territory_before", "territory_after",
                 "troops_before", "troops_after", "tiles_gained")

    def __init__(self) -> None:
        self.turn: int = 0
        self.actions_taken: list[str] = []
        self.territory_before: int = 0
        self.territory_after: int = 0
        self.troops_before: int = 0
        self.troops_after: int = 0
        self.tiles_gained: int = 0


class GameReplay:
    """Record of a full game for text display."""

    def __init__(self) -> None:
        self.turns: list[TurnSnapshot] = []
        self.final_territory: int = 0
        self.total_tiles: int = 0
        self.winner: int | None = None
        self.total_game_turns: int = 0

    def display(self, label: str = "Game Replay") -> None:
        """Print a compact turn-by-turn summary."""
        print(f"\n  --- {label} ---")
        print(f"  {'Turn':>4}  {'Terr':>8}  {'Troops':>9}  {'Gained':>6}  Actions")
        print(f"  {'----':>4}  {'--------':>8}  {'---------':>9}  {'------':>6}  -------")
        for t in self.turns:
            actions_str = ", ".join(t.actions_taken[:4])
            if len(t.actions_taken) > 4:
                actions_str += f" (+{len(t.actions_taken) - 4} more)"
            print(
                f"  {t.turn:4d}  "
                f"{t.territory_before:3d}->{t.territory_after:<3d}  "
                f"{t.troops_before:4d}->{t.troops_after:<4d}  "
                f"{t.tiles_gained:+5d}  "
                f"{actions_str}"
            )
        print(
            f"  Final: {self.final_territory}/{self.total_tiles} tiles | "
            f"Winner: P{self.winner}\n"
        )


def _describe_action(env: FlatHexWarEnv, action_int: int) -> str:
    """Human-readable description of a unified sub-step action."""
    if action_int == 0:
        return "END"
    game_action = env._decode_unified(action_int)
    if game_action is None:
        return f"?({action_int})"
    from game.actions import MoveAction, SetupSupplyChainAction
    if isinstance(game_action, MoveAction):
        return f"Move({game_action.source.col},{game_action.source.row})->({game_action.target.col},{game_action.target.row}) x{game_action.troops}"
    if isinstance(game_action, SetupSupplyChainAction):
        return f"SC({game_action.source.col},{game_action.source.row})->({game_action.destination.col},{game_action.destination.row})"
    return str(game_action)


# ── Evaluation ──────────────────────────────────────────────────────

def evaluate(
    agent: DQNAgent,
    config: GameConfig,
    n_episodes: int = 10,
    seed: int = 42,
    record_game: bool = False,
    max_steps: int = 500,
) -> dict:
    """Run *n_episodes* with greedy policy (epsilon=0) and return metrics.

    If *record_game* is True, the first episode is recorded and returned
    as ``result["replay"]`` (a GameReplay instance).
    """

    env = FlatHexWarEnv(config, sub_step=True)
    opponent = BotFlatAdapter(NoOpBot())

    wins = 0
    total_reward = 0.0
    total_territory = 0
    total_actions = 0
    total_turns = 0
    replay: GameReplay | None = None

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        ep_actions = 0

        # Recording state for first episode.
        recording = record_game and ep == 0
        if recording:
            replay = GameReplay()
            replay.total_tiles = env.n_tiles
        current_turn_snap: TurnSnapshot | None = None
        last_turn = 0

        while not done:
            if ep_actions >= max_steps:
                done = True
                break

            if info["current_player"] == 0:
                state = env.engine.state

                # Start a new turn snapshot if the game turn advanced.
                if recording and state.turn != last_turn:
                    if current_turn_snap is not None:
                        current_turn_snap.territory_after = state.territory_count(0)
                        current_turn_snap.troops_after = state.total_troops(0)
                        current_turn_snap.tiles_gained = (
                            current_turn_snap.territory_after - current_turn_snap.territory_before
                        )
                        replay.turns.append(current_turn_snap)
                    current_turn_snap = TurnSnapshot()
                    current_turn_snap.turn = state.turn
                    current_turn_snap.territory_before = state.territory_count(0)
                    current_turn_snap.troops_before = state.total_troops(0)
                    last_turn = state.turn

                soft_mask = info["action_masks"]["soft"]
                action = agent.select_action(obs, soft_mask, epsilon=0.0)

                if recording and current_turn_snap is not None:
                    current_turn_snap.actions_taken.append(
                        _describe_action(env, action)
                    )

                next_obs, reward, terminated, truncated, next_info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_actions += 1

                # Fast-forward opponent turns.
                while not done and next_info["current_player"] != 0:
                    opp_act = opponent.choose_sub_action(env)
                    next_obs, _, terminated, truncated, next_info = env.step(opp_act)
                    done = terminated or truncated

                obs, info = next_obs, next_info
            else:
                opp_act = opponent.choose_sub_action(env)
                obs, _, terminated, truncated, info = env.step(opp_act)
                done = terminated or truncated

        state = env.engine.state

        # Flush last turn snapshot.
        if recording and current_turn_snap is not None:
            current_turn_snap.territory_after = state.territory_count(0)
            current_turn_snap.troops_after = state.total_troops(0)
            current_turn_snap.tiles_gained = (
                current_turn_snap.territory_after - current_turn_snap.territory_before
            )
            replay.turns.append(current_turn_snap)
            replay.final_territory = state.territory_count(0)
            replay.winner = state.winner
            replay.total_game_turns = state.turn

        if state.winner == 0:
            wins += 1
        total_reward += ep_reward
        total_territory += state.territory_count(0)
        total_actions += ep_actions
        total_turns += state.turn

    n = n_episodes
    result = {
        "win_rate": wins / n,
        "mean_reward": total_reward / n,
        "mean_territory": total_territory / n,
        "mean_actions": total_actions / n,
        "mean_turns": total_turns / n,
    }
    if replay is not None:
        result["replay"] = replay
    return result
