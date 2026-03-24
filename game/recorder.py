"""Game recording, interestingness scoring, and JSON serialization.

Provides GameRecord and FrameSnapshot dataclasses, plus helpers for
capturing state, scoring game interestingness, and save/load to JSON.
Used by bot_runner during tournaments and replay.py for playback.

Depended on by:
    bot_runner, replay

Dependencies:
    hex_core, game/actions, game/config, game/state

Ripple effects:
    - Changing serialization format → update replay.py deserialization.
    - Adding new action types → update action serialization in
      _serialize_action / _deserialize_action.

Records every action + lightweight state snapshot during a game,
scores games by how interesting they'd be to watch, and provides
save/load to JSON for persistent tournament replays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from hex_core import HexCoord

from game.actions import MoveAction, EndTurnAction, SetupSupplyChainAction
from game.config import GameConfig, MapPreset
from game.state import GameState, SupplyChain


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class FrameSnapshot:
    """One frame: an action + the resulting state."""

    action: MoveAction | EndTurnAction | SetupSupplyChainAction
    player: int
    tile_states: dict[HexCoord, tuple[int | None, int]]  # coord -> (owner, troops)
    supply_chains: list[tuple[HexCoord, HexCoord, int]]   # (src, dst, owner)
    turn: int
    current_player: int
    territory_counts: dict[int, int]
    total_troops: dict[int, int]
    phase_name: str  # "PLAY" or "GAME_OVER"
    winner: int | None


@dataclass
class GameRecord:
    """Complete recording of a game."""

    config: GameConfig
    seed: int | None
    bot_names: dict[int, str]
    # Initial state (before any actions)
    initial_tile_states: dict[HexCoord, tuple[int | None, int]] = field(default_factory=dict)
    initial_supply_chains: list[tuple[HexCoord, HexCoord, int]] = field(default_factory=list)
    initial_turn: int = 1
    # Action frames
    frames: list[FrameSnapshot] = field(default_factory=list)
    winner: int | None = None
    total_turns: int = 0


# ---------------------------------------------------------------------------
# Snapshot capture
# ---------------------------------------------------------------------------

def capture_snapshot(
    action: MoveAction | EndTurnAction | SetupSupplyChainAction,
    player: int,
    state: GameState,
) -> FrameSnapshot:
    """Capture a lightweight snapshot of the current state after an action."""
    return FrameSnapshot(
        action=action,
        player=player,
        tile_states={
            coord: (ts.owner, ts.troops)
            for coord, ts in state.tiles.items()
        },
        supply_chains=[
            (sc.source, sc.destination, sc.owner)
            for sc in state.supply_chains
        ],
        turn=state.turn,
        current_player=state.current_player,
        territory_counts={
            p: state.territory_count(p) for p in range(state.num_players)
        },
        total_troops={
            p: state.total_troops(p) for p in range(state.num_players)
        },
        phase_name=state.phase.name,
        winner=state.winner,
    )


def capture_initial_state(state: GameState) -> tuple[dict, list, int]:
    """Capture the initial tile states before any actions."""
    tiles = {
        coord: (ts.owner, ts.troops)
        for coord, ts in state.tiles.items()
    }
    chains = [
        (sc.source, sc.destination, sc.owner)
        for sc in state.supply_chains
    ]
    return tiles, chains, state.turn


# ---------------------------------------------------------------------------
# Interestingness scoring
# ---------------------------------------------------------------------------

def score_interestingness(record: GameRecord) -> float:
    """Score how interesting a game is to watch.

    Higher = more interesting. Factors:
    - Lead changes (territory leader swaps)
    - Final closeness (tight finish)
    - Game length (longer games have more action)
    - Combat density (attacks vs end-turns)
    """
    if not record.frames:
        return 0.0

    # --- Lead changes ---
    lead_changes = 0
    prev_leader: int | None = None
    for frame in record.frames:
        tc = frame.territory_counts
        if not tc:
            continue
        leader = max(tc, key=tc.get)  # type: ignore[arg-type]
        if prev_leader is not None and leader != prev_leader:
            lead_changes += 1
        prev_leader = leader

    # --- Final closeness (0-10 scale) ---
    final_tc = record.frames[-1].territory_counts
    if len(final_tc) >= 2:
        counts = sorted(final_tc.values(), reverse=True)
        total = sum(counts)
        if total > 0:
            gap = (counts[0] - counts[1]) / total
            closeness = max(0, 10 * (1 - gap))  # 10 = tied, 0 = blowout
        else:
            closeness = 0
    else:
        closeness = 0

    # --- Game length (0-10 scale) ---
    max_turns = record.config.max_turns
    if max_turns > 0:
        length_ratio = record.total_turns / max_turns
        # Prefer games at 50-80% of max length (not too short, not timeout)
        if length_ratio > 0.8:
            length_score = 10 * (1 - (length_ratio - 0.8) / 0.2)  # penalize timeouts
        else:
            length_score = 10 * min(1.0, length_ratio / 0.6)
    else:
        length_score = 5

    # --- Combat density ---
    combat_count = sum(
        1 for f in record.frames
        if isinstance(f.action, MoveAction)
    )

    return lead_changes * 3 + closeness + length_score + combat_count * 0.5


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

def _coord_key(coord: HexCoord) -> str:
    """Serialize a HexCoord as a string key for JSON dicts."""
    return f"{coord.col},{coord.row}"


def _parse_coord_key(key: str) -> HexCoord:
    """Deserialize a coord string key back to HexCoord."""
    col, row = key.split(",")
    return HexCoord(int(col), int(row))


def _action_to_dict(action: MoveAction | EndTurnAction | SetupSupplyChainAction) -> dict:
    if isinstance(action, MoveAction):
        return {
            "type": "move",
            "source": [action.source.col, action.source.row],
            "target": [action.target.col, action.target.row],
            "troops": action.troops,
        }
    if isinstance(action, SetupSupplyChainAction):
        return {
            "type": "supply_chain",
            "source": [action.source.col, action.source.row],
            "destination": [action.destination.col, action.destination.row],
        }
    return {"type": "end_turn"}


def _action_from_dict(d: dict) -> MoveAction | EndTurnAction | SetupSupplyChainAction:
    if d["type"] == "move":
        return MoveAction(
            source=HexCoord(d["source"][0], d["source"][1]),
            target=HexCoord(d["target"][0], d["target"][1]),
            troops=d["troops"],
        )
    if d["type"] == "supply_chain":
        return SetupSupplyChainAction(
            source=HexCoord(d["source"][0], d["source"][1]),
            destination=HexCoord(d["destination"][0], d["destination"][1]),
        )
    return EndTurnAction()


def _tile_states_to_dict(
    tiles: dict[HexCoord, tuple[int | None, int]],
) -> dict[str, list]:
    """Serialize tile states as {coord_key: [owner, troops]}."""
    return {_coord_key(c): [owner, troops] for c, (owner, troops) in tiles.items()}


def _tile_states_from_dict(
    d: dict[str, list],
) -> dict[HexCoord, tuple[int | None, int]]:
    return {_parse_coord_key(k): (v[0], v[1]) for k, v in d.items()}


def _chains_to_list(
    chains: list[tuple[HexCoord, HexCoord, int]],
) -> list[list]:
    return [[src.col, src.row, dst.col, dst.row, owner] for src, dst, owner in chains]


def _chains_from_list(
    data: list[list],
) -> list[tuple[HexCoord, HexCoord, int]]:
    return [(HexCoord(r[0], r[1]), HexCoord(r[2], r[3]), r[4]) for r in data]


def _frame_to_dict(frame: FrameSnapshot) -> dict:
    return {
        "action": _action_to_dict(frame.action),
        "player": frame.player,
        "tiles": _tile_states_to_dict(frame.tile_states),
        "chains": _chains_to_list(frame.supply_chains),
        "turn": frame.turn,
        "current_player": frame.current_player,
        "territory": frame.territory_counts,
        "troops": frame.total_troops,
        "phase": frame.phase_name,
        "winner": frame.winner,
    }


def _frame_from_dict(d: dict) -> FrameSnapshot:
    return FrameSnapshot(
        action=_action_from_dict(d["action"]),
        player=d["player"],
        tile_states=_tile_states_from_dict(d["tiles"]),
        supply_chains=_chains_from_list(d["chains"]),
        turn=d["turn"],
        current_player=d["current_player"],
        territory_counts={int(k): v for k, v in d["territory"].items()},
        total_troops={int(k): v for k, v in d["troops"].items()},
        phase_name=d["phase"],
        winner=d["winner"],
    )


def record_to_dict(record: GameRecord) -> dict:
    """Serialize a GameRecord to a JSON-compatible dict."""
    cfg = record.config
    return {
        "config": {
            "preset": cfg.preset.value if cfg.preset else None,
            "num_players": cfg.num_players,
            "grid_width": cfg.grid_width,
            "grid_height": cfg.grid_height,
            "max_turns": cfg.max_turns,
            "moves_per_turn": cfg.moves_per_turn,
        },
        "seed": record.seed,
        "bot_names": {str(k): v for k, v in record.bot_names.items()},
        "initial_tiles": _tile_states_to_dict(record.initial_tile_states),
        "initial_chains": _chains_to_list(record.initial_supply_chains),
        "initial_turn": record.initial_turn,
        "winner": record.winner,
        "total_turns": record.total_turns,
        "frames": [_frame_to_dict(f) for f in record.frames],
    }


def record_from_dict(d: dict) -> GameRecord:
    """Deserialize a GameRecord from a JSON-compatible dict."""
    cfg_d = d["config"]
    preset = MapPreset(cfg_d["preset"]) if cfg_d["preset"] else None
    config = GameConfig(
        preset=preset,
        num_players=cfg_d["num_players"],
        map_seed=d["seed"],
        auto_place_starts=True,
    )

    return GameRecord(
        config=config,
        seed=d["seed"],
        bot_names={int(k): v for k, v in d["bot_names"].items()},
        initial_tile_states=_tile_states_from_dict(d["initial_tiles"]),
        initial_supply_chains=_chains_from_list(d["initial_chains"]),
        initial_turn=d["initial_turn"],
        frames=[_frame_from_dict(f) for f in d["frames"]],
        winner=d["winner"],
        total_turns=d["total_turns"],
    )


def save_record(record: GameRecord, path: Path) -> None:
    """Save a GameRecord to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record_to_dict(record)), encoding="utf-8")


def load_record(path: Path) -> GameRecord:
    """Load a GameRecord from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return record_from_dict(data)
