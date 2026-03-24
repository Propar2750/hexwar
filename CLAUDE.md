# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

- **Remote:** https://github.com/Propar2750/hexwar
- **Workflow:** Commit and push frequently — every meaningful change should be published to GitHub promptly.

## Project Overview

HexWar is a multi-agent reinforcement learning project where AI players compete for territory control on a hex grid. It's a college project built in phases over ~14 weeks (see BIGPLAN.md for the full roadmap).

**Core game:** Players start on a hex grid, generate troops each turn based on terrain, and attack adjacent tiles to expand territory. Combat uses a defense-bonus formula with probabilistic outcomes. Win by controlling 50%+ of the map or having the most territory after N turns.

## Current State

Phase 1 (Environment Engineering) is well underway. The hex grid engine, terrain generation, A* pathfinding, game engine, and a playable 2-player interactive demo are implemented.

### Project Structure

```
hex_core.py          — HexCoord (doubled-width), neighbors, distance, pixel conversions
hex_grid.py          — HexGrid container, Terrain enum (PLAINS, FERTILE, MOUNTAIN)
map_generator.py     — Cellular automata terrain gen with mountain range random walks
pathfinding.py       — A* pathfinding with terrain-aware movement costs
renderer.py          — Pygame hex renderer (used by main.py demo)
main.py              — Demo: renders hex grid with A* pathfinding visualization
play.py              — Interactive 2-player game (pygame)

game/                — Game logic package
  config.py          — GameConfig dataclass (all tunable parameters)
  state.py           — GameState, TileState, GamePhase (SETUP/PLAYING/GAME_OVER)
  actions.py         — MoveAction, EndTurnAction, validation, valid-target computation
  combat.py          — CombatResolver protocol + DefaultCombatResolver
  engine.py          — GameEngine (reset, placement, troop gen, move execution, victory)
  game_renderer.py   — Game-specific pygame rendering (HUD, troop counts, highlights)
  environment.py     — Gym-compatible environment wrapper (in progress)

tests/               — Pytest tests for hex_core, hex_grid, pathfinding, main utils
```

### Architecture

The project follows four major phases:

1. **Environment Engineering** *(in progress)* — Hex grid engine, game state machine, action/observation spaces, rule-based baseline bots
2. **First RL Agents** — CNN-based state representation, DQN from scratch, curriculum learning (2→4 players, small→large maps)
3. **Advanced Algorithms** — PPO, multi-agent self-play with league training, emergent diplomacy
4. **Analysis & Polish** — Strategy visualization, fog of war / POMDP, tournament with Elo ratings, ablation studies

## Key Design Decisions

- **Hex coordinates:** Doubled-width coordinate system, pointy-top orientation (reference: Red Blob Games hex grid guide)
- **Terrain types:** Plains (move cost 1, defense 1, generates 2 troops/turn), Mountains (move cost 3, defense 2, generates 1), Fertile (move cost 1, defense 1, generates 3)
- **Combat formula:** `threshold = defense_bonus * (1 + D + sqrt(D))` — attacker above threshold wins guaranteed, between D and threshold wins probabilistically, at or below D loses guaranteed
- **Game engine pattern:** GameState is a plain data container; all mutations go through GameEngine so rules are enforced in one place. CombatResolver is pluggable via protocol.
- **Config-driven:** All tunable parameters (grid size, troop gen rates, defense bonuses, map gen params) live in `GameConfig` dataclass

## Running

```bash
# Interactive 2-player game
python play.py

# Pathfinding / terrain demo
python main.py

# Tests
pytest
```
