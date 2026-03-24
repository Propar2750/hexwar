# Architecture Reference

This document describes how the HexWar codebase is structured, how modules depend on each other, and how changes propagate. Consult this before modifying any module.

## Dependency Layers

The codebase is organized in layers. Lower layers know nothing about higher layers. Always modify bottom-up.

```
Layer 0 — Primitives (no project imports)
    hex_core.py          Hex coordinate math, pixel conversions
    game/combat.py       Combat formula, win_probability()

Layer 1 — Data Structures
    hex_grid.py          HexGrid, Terrain enum, HexTile          [← hex_core]

Layer 2 — Algorithms & Config
    pathfinding.py       A* on hex grids                         [← hex_core, hex_grid]
    map_generator.py     Cellular automata terrain generation     [← hex_core, hex_grid]
    game/config.py       GameConfig, MapPreset, tunable params   [← hex_core, hex_grid]

Layer 3 — Game State & Actions
    game/state.py        GameState, TileState, GamePhase         [← hex_core, hex_grid]
    game/actions.py      Action dataclasses, validation          [← hex_core, game/state]

Layer 4 — Game Engine
    game/engine.py       Full lifecycle orchestrator              [← layers 0-3, map_generator]

Layer 5 — Consumers (UI, bots, environments, recording)
    renderer.py          Base hex renderer (pygame)               [← hex_core, hex_grid]
    game/game_renderer   Game-aware renderer with HUD             [← renderer, game/state, game/config]
    game/bots.py         Rule-based bot strategies                [← game/state, actions, combat, config]
    game/recorder.py     Game recording & serialization           [← game/actions, state, config]
    game/environment.py  Gym env v1 (legacy)                     [← game/engine, actions, config]
    game/flat_env.py     Gym env v2 (primary RL interface)       [← game/engine, actions, combat, config]

Layer 6 — RL Agents (separate dependency tree)
    agents/replay_buffer.py   Experience replay (numpy/torch only)
    agents/networks.py        QNetwork architecture (torch only)
    agents/dqn_agent.py       DQN with two-tier masking          [← networks, replay_buffer]

Layer 7 — Entry Points (not imported by anything)
    main.py              Pathfinding demo
    play.py              Interactive 2-player game
    bot_runner.py        Bot tournament runner
    replay.py            Replay viewer
```

## File Dependency Graph

```
hex_core ──┬── hex_grid ──┬── pathfinding
           │              ├── map_generator
           │              ├── renderer ──── game/game_renderer
           │              ├── game/config
           │              └── game/state ──── game/actions
           │                                      │
           │              game/combat ─────────────┤
           │                                       │
           │              game/engine ◄────────────┘
           │                  │
           │          ┌───────┼───────────┐
           │          │       │           │
           │    game/flat_env │   game/environment
           │          │       │
           │    game/bots  game/recorder
           │
           ├── main.py
           ├── play.py ◄── game/engine, game/game_renderer
           ├── bot_runner.py ◄── game/engine, game/bots, game/recorder
           └── replay.py ◄── game/recorder, game/game_renderer

agents/replay_buffer ─┐
agents/networks ──────┼── agents/dqn_agent
```

## Change Ripple Guide

Use this table to know what else to check/update when you modify a file.

| When you change...          | Also check / update...                                                |
|-----------------------------|-----------------------------------------------------------------------|
| `hex_core.py`               | Nearly everything — this is the foundation                            |
| `hex_grid.Terrain` enum     | `game/config.py` dicts, `renderer.py` TERRAIN_FILL, `pathfinding.py` DEFAULT_COST |
| `hex_grid.HexGrid` API      | `map_generator`, `game/engine`, `game/state`, all renderers           |
| `pathfinding.py` costs      | Movement behavior in `main.py` demo                                   |
| `map_generator.py` params   | Map balance; also check `game/config.py` MapPreset overrides          |
| `game/config.py` defaults   | Game balance across play, bots, and RL training                       |
| `game/state.GameState`      | `game/recorder` serialization, `game/flat_env` obs, `game/game_renderer` |
| `game/state.GamePhase`      | `game/engine` transitions, UI phase checks in `play.py`/`replay.py`  |
| `game/actions` (new action) | `game/engine.execute_action()`, `game/bots`, `game/flat_env`, `game/recorder` |
| `game/combat` formula       | Game balance, bot effectiveness, RL reward signals                    |
| `game/engine` rules         | ALL consumers: play, bots, tournaments, RL training                   |
| `game/flat_env` obs/actions | Retrain all RL agents; update `agents/networks` dimensions            |
| `game/bots` (new bot)       | Add to `BOT_REGISTRY`; `bot_runner` picks it up automatically        |
| `game/recorder` format      | `replay.py` deserialization                                           |
| `renderer.py` colors/API    | `game/game_renderer`, `main.py`, `play.py`, `replay.py`              |
| `agents/networks` dims      | Must match `game/flat_env` observation/action sizes                   |

## Common Tasks — Where to Look

| Task                           | Primary file(s)                          |
|--------------------------------|------------------------------------------|
| Add a new terrain type         | `hex_grid.py`, `game/config.py`, `renderer.py`, `pathfinding.py` |
| Add a new action type          | `game/actions.py`, `game/engine.py`, `game/bots.py`, `game/flat_env.py`, `game/recorder.py` |
| Add a new bot strategy         | `game/bots.py` (add class + register in BOT_REGISTRY) |
| Change combat formula          | `game/combat.py`                         |
| Change victory conditions      | `game/engine.py` (_check_victory)        |
| Change observation encoding    | `game/flat_env.py`                       |
| Change reward shaping          | `game/flat_env.py` (_compute_reward)     |
| Change troop generation rates  | `game/config.py` (troop_generation dict) |
| Change map generation          | `map_generator.py`, `game/config.py` (MapPreset) |
| Add RL algorithm               | `agents/` (new file), wire into `game/flat_env` |
| Fix rendering                  | `renderer.py` (base), `game/game_renderer.py` (game-specific) |

## Test Coverage Map

| Module                | Test file                  |
|-----------------------|----------------------------|
| `hex_core.py`         | `tests/test_hex_core.py`   |
| `hex_grid.py`         | `tests/test_hex_grid.py`   |
| `pathfinding.py`      | `tests/test_pathfinding.py`|
| `map_generator.py`    | `tests/test_main_utils.py` |
| `game/flat_env.py`    | `tests/test_flat_env.py`   |
| `game/engine.py`      | *(needs tests)*            |
| `game/actions.py`     | *(needs tests)*            |
| `game/combat.py`      | *(needs tests)*            |
| `game/bots.py`        | *(needs tests)*            |

## Entry Points

| Command              | What it runs        | Purpose                              |
|----------------------|---------------------|--------------------------------------|
| `python play.py`     | play.py             | Interactive 2-player game            |
| `python main.py`     | main.py             | Pathfinding / terrain demo           |
| `python bot_runner.py` | bot_runner.py     | Bot tournament + optional replay     |
| `pytest`             | tests/              | All unit tests                       |
