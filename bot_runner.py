"""Headless bot-vs-bot game runner for HexWar.

Run a round-robin tournament between rule-based bots, print win-rate
statistics, and replay the most interesting match in pygame.

Usage:
    python bot_runner.py                          # round-robin, 100 games, medium map
    python bot_runner.py --bots random greedy     # specific matchup
    python bot_runner.py --preset small --games 50
    python bot_runner.py --no-replay              # skip the replay viewer
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

from game.config import GameConfig, MapPreset
from game.engine import GameEngine
from game.state import GamePhase
from game.actions import EndTurnAction
from game.bots import Bot, BOT_REGISTRY
from game.recorder import (
    GameRecord, FrameSnapshot, capture_snapshot, capture_initial_state,
    score_interestingness, save_record,
)

TOURNAMENTS_DIR = Path("tournaments")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class GameResult:
    """Outcome of a single game."""
    winner: int | None
    turns: int
    final_territory: dict[int, int]
    final_troops: dict[int, int]
    record: GameRecord


@dataclass
class MatchStats:
    """Aggregate stats for a series of games between two bots."""
    bot_a_name: str
    bot_b_name: str
    games_played: int
    bot_a_wins: int
    bot_b_wins: int
    draws: int
    avg_game_length: float
    records: list[GameRecord]

    @property
    def bot_a_win_rate(self) -> float:
        return self.bot_a_wins / self.games_played * 100 if self.games_played else 0

    @property
    def bot_b_win_rate(self) -> float:
        return self.bot_b_wins / self.games_played * 100 if self.games_played else 0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games_played * 100 if self.games_played else 0


# ---------------------------------------------------------------------------
# Game runner (with recording)
# ---------------------------------------------------------------------------

def run_game(
    bots: dict[int, Bot],
    config: GameConfig,
    seed: int | None = None,
) -> GameResult:
    """Run a single game to completion, recording every action."""
    cfg = GameConfig(
        preset=config.preset,
        num_players=config.num_players,
        map_seed=seed,
        auto_place_starts=True,
    )
    engine = GameEngine(cfg)
    engine.reset()
    state = engine.state

    # Start recording
    bot_names = {pid: bot.name for pid, bot in bots.items()}
    initial_tiles, initial_chains, initial_turn = capture_initial_state(state)
    record = GameRecord(
        config=cfg,
        seed=seed,
        bot_names=bot_names,
        initial_tile_states=initial_tiles,
        initial_supply_chains=initial_chains,
        initial_turn=initial_turn,
    )

    # Safety: max actions per turn to prevent infinite loops
    max_actions_per_turn = (config.moves_per_turn or 20) + 10

    while state.phase == GamePhase.PLAY:
        player = state.current_player
        bot = bots[player]
        actions_this_turn = 0

        while state.phase == GamePhase.PLAY and state.current_player == player:
            action = bot.choose_action(state, player, cfg)
            engine.execute_action(action)

            # Record the frame
            record.frames.append(capture_snapshot(action, player, state))

            actions_this_turn += 1
            if actions_this_turn > max_actions_per_turn:
                engine.execute_action(EndTurnAction())
                record.frames.append(capture_snapshot(EndTurnAction(), player, state))
                break

    record.winner = state.winner
    record.total_turns = state.turn

    return GameResult(
        winner=state.winner,
        turns=state.turn,
        final_territory={
            p: state.territory_count(p) for p in range(state.num_players)
        },
        final_troops={
            p: state.total_troops(p) for p in range(state.num_players)
        },
        record=record,
    )


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------

def run_match(
    bot_a: Bot,
    bot_b: Bot,
    config: GameConfig,
    num_games: int = 100,
    base_seed: int = 0,
) -> MatchStats:
    """Run many games, alternating who goes first."""
    a_wins = 0
    b_wins = 0
    draws = 0
    total_turns = 0
    records: list[GameRecord] = []

    for i in range(num_games):
        seed = base_seed + i

        if i % 2 == 0:
            bots = {0: bot_a, 1: bot_b}
            a_player, b_player = 0, 1
        else:
            bots = {0: bot_b, 1: bot_a}
            a_player, b_player = 1, 0

        result = run_game(bots, config, seed)
        total_turns += result.turns
        records.append(result.record)

        if result.winner == a_player:
            a_wins += 1
        elif result.winner == b_player:
            b_wins += 1
        else:
            draws += 1

    return MatchStats(
        bot_a_name=bot_a.name,
        bot_b_name=bot_b.name,
        games_played=num_games,
        bot_a_wins=a_wins,
        bot_b_wins=b_wins,
        draws=draws,
        avg_game_length=total_turns / num_games if num_games else 0,
        records=records,
    )


# ---------------------------------------------------------------------------
# Tournament
# ---------------------------------------------------------------------------

def run_tournament(
    bots: list[Bot],
    config: GameConfig,
    num_games: int = 100,
    base_seed: int = 0,
) -> list[MatchStats]:
    """Round-robin: every pair of bots plays num_games."""
    results: list[MatchStats] = []
    for a, b in itertools.combinations(bots, 2):
        stats = run_match(a, b, config, num_games, base_seed)
        results.append(stats)
    return results


def print_results(results: list[MatchStats], config: GameConfig) -> None:
    """Pretty-print tournament results."""
    preset_name = config.preset.value if config.preset else "medium"
    w, h = config.grid_width, config.grid_height
    games = results[0].games_played if results else 0

    print(f"\n{'=' * 60}")
    print(f"  HexWar Bot Tournament")
    print(f"  Map: {preset_name} ({w}x{h}), {games} games per matchup")
    print(f"{'=' * 60}\n")

    print(f"  {'Matchup':<24} | {'Win A':>6} | {'Win B':>6} | {'Draw':>5} | {'Avg Turns':>9}")
    print(f"  {'-' * 24}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 5}-+-{'-' * 9}")

    for s in results:
        label = f"{s.bot_a_name} vs {s.bot_b_name}"
        print(
            f"  {label:<24} | {s.bot_a_win_rate:5.1f}% | {s.bot_b_win_rate:5.1f}% "
            f"| {s.draw_rate:4.1f}% | {s.avg_game_length:9.1f}"
        )

    print()


def find_best_game(results: list[MatchStats]) -> tuple[GameRecord, float, str]:
    """Find the most interesting game across all matchups."""
    best_record: GameRecord | None = None
    best_score = -1.0
    best_matchup = ""

    for stats in results:
        for record in stats.records:
            score = score_interestingness(record)
            if score > best_score:
                best_score = score
                best_record = record
                best_matchup = f"{stats.bot_a_name} vs {stats.bot_b_name}"

    assert best_record is not None
    return best_record, best_score, best_matchup


# ---------------------------------------------------------------------------
# Tournament persistence
# ---------------------------------------------------------------------------

def _next_tournament_number() -> int:
    """Find the next available tournament number."""
    if not TOURNAMENTS_DIR.exists():
        return 1
    existing = [
        int(d.name.split("_")[1])
        for d in TOURNAMENTS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("tournament_") and d.name.split("_")[1].isdigit()
    ]
    return max(existing, default=0) + 1


def save_tournament(
    results: list[MatchStats],
    config: GameConfig,
    elapsed: float,
    best_record: GameRecord,
    best_score: float,
) -> Path:
    """Save full tournament data to tournaments/tournament_N/."""
    num = _next_tournament_number()
    tournament_dir = TOURNAMENTS_DIR / f"tournament_{num}"
    replays_dir = tournament_dir / "replays"
    replays_dir.mkdir(parents=True, exist_ok=True)

    # --- summary.json ---
    preset_name = config.preset.value if config.preset else "medium"
    matchups = []
    game_index = 0
    for stats in results:
        matchup_data = {
            "bot_a": stats.bot_a_name,
            "bot_b": stats.bot_b_name,
            "games_played": stats.games_played,
            "bot_a_wins": stats.bot_a_wins,
            "bot_b_wins": stats.bot_b_wins,
            "draws": stats.draws,
            "bot_a_win_rate": round(stats.bot_a_win_rate, 1),
            "bot_b_win_rate": round(stats.bot_b_win_rate, 1),
            "avg_game_length": round(stats.avg_game_length, 1),
            "game_files": [],
        }

        # Save each game replay
        for i, record in enumerate(stats.records):
            game_index += 1
            filename = f"game_{game_index:03d}_{stats.bot_a_name}_vs_{stats.bot_b_name}.json"
            save_record(record, replays_dir / filename)
            matchup_data["game_files"].append(filename)

        matchups.append(matchup_data)

    # Best game info
    best_names = " vs ".join(
        best_record.bot_names[i] for i in sorted(best_record.bot_names)
    )
    best_winner = best_record.bot_names.get(best_record.winner, "draw") if best_record.winner is not None else "draw"

    summary = {
        "tournament_number": num,
        "preset": preset_name,
        "grid_size": f"{config.grid_width}x{config.grid_height}",
        "elapsed_seconds": round(elapsed, 1),
        "matchups": matchups,
        "most_interesting_game": {
            "matchup": best_names,
            "score": round(best_score, 1),
            "turns": best_record.total_turns,
            "frames": len(best_record.frames),
            "winner": best_winner,
        },
    }

    summary_path = tournament_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )

    return tournament_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="HexWar bot tournament runner")
    parser.add_argument(
        "--bots", nargs="+", choices=list(BOT_REGISTRY.keys()),
        default=list(BOT_REGISTRY.keys()),
        help="Which bots to include (default: all)",
    )
    parser.add_argument(
        "--preset", choices=[p.value for p in MapPreset],
        default="medium",
        help="Map preset (default: medium)",
    )
    parser.add_argument(
        "--games", type=int, default=100,
        help="Games per matchup (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base random seed (default: 0)",
    )
    parser.add_argument(
        "--no-replay", action="store_true",
        help="Skip the replay viewer",
    )
    args = parser.parse_args()

    config = GameConfig(preset=MapPreset(args.preset))
    bots = [BOT_REGISTRY[name](seed=args.seed + i) for i, name in enumerate(args.bots)]

    print(f"Running tournament: {', '.join(b.name for b in bots)}")
    start = time.time()

    results = run_tournament(bots, config, num_games=args.games, base_seed=args.seed)

    elapsed = time.time() - start
    print_results(results, config)
    print(f"  Completed in {elapsed:.1f}s")

    # Find the most interesting game
    best_record, best_score, best_matchup = find_best_game(results)
    names = " vs ".join(best_record.bot_names[i] for i in sorted(best_record.bot_names))
    print(f"  Most interesting match: {names}")
    print(f"  Interestingness score: {best_score:.1f}")
    print(f"  Turns: {best_record.total_turns}, Frames: {len(best_record.frames)}")
    winner = best_record.winner
    if winner is not None:
        print(f"  Winner: {best_record.bot_names.get(winner, '?')}")

    # Save tournament to disk
    tournament_dir = save_tournament(results, config, elapsed, best_record, best_score)
    print(f"\n  Saved to {tournament_dir}/")

    if not args.no_replay:
        print(f"  Launching replay...")
        from replay import replay_game
        replay_game(best_record)


if __name__ == "__main__":
    main()
