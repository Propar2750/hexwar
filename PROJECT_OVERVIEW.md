# HexWar — Project Overview

## What it is

A hex-grid territory control game where players expand by generating troops and attacking neighbors. Currently playable as a 2-player hotseat game via pygame.

## Core Infrastructure (root-level modules)

- **`hex_core.py`** — `HexCoord` using doubled-width coordinates. Provides neighbor lookup, hex distance, and pixel↔hex conversions for pointy-top hexes.
- **`hex_grid.py`** — `HexGrid` container mapping `HexCoord → Terrain`. Three terrain types: `PLAINS`, `FERTILE`, `MOUNTAIN`.
- **`map_generator.py`** — Cellular automata to place fertile land + random-walk mountain ranges with weighted direction turns and variable lengths.
- **`pathfinding.py`** — A* search with terrain-aware movement costs.
- **`renderer.py`** — Base pygame hex renderer (terrain colors, grid lines, hover highlights).

## Game Logic (`game/` package)

- **`config.py`** — `GameConfig` dataclass with all tunable params: grid size (14×10), 30 max turns, 50% win threshold, troop gen rates, defense bonuses, movement costs, map gen seeds.
- **`state.py`** — `GameState` (plain data container) + `TileState` (owner, troop count) + `GamePhase` enum (`SETUP → PLAYING → GAME_OVER`).
- **`actions.py`** — `MoveAction` / `EndTurnAction` types, move validation, and valid-target computation (adjacent tiles only).
- **`combat.py`** — Pluggable `CombatResolver` protocol. Default formula: `threshold = defense_bonus × (1 + D + √D)`. Above = guaranteed win, below D = guaranteed loss, in between = linear probability.
- **`engine.py`** — `GameEngine` orchestrates everything: map gen, starting position placement, per-turn troop generation, move execution with combat, turn transitions, and victory detection.
- **`game_renderer.py`** — Game-specific rendering: player colors, troop counts on tiles, HUD (current player, turn counter, territory %), selection/movement highlights.
- **`environment.py`** — Gym-compatible wrapper (in progress).

## Entrypoints

- **`play.py`** — Interactive 2-player game. Click to select tiles, scroll to pick troop count, click target to attack, Space to end turn.
- **`main.py`** — Demo/sandbox: renders a larger grid (30×30) with A* pathfinding visualization.

## Tests

- `tests/` — Pytest suite covering `hex_core`, `hex_grid`, pathfinding, and main utility functions.
- `test_cellular_automata.py` — Tests for the terrain generator.

## Status

Phase 1 (Environment Engineering) is the active phase. The hex engine, terrain gen, game rules, combat system, and a playable 2-player demo are done. Next up from BIGPLAN.md would be completing the Gym-compatible environment wrapper and building rule-based baseline bots before moving into RL training in Phase 2.
