# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

- **Remote:** https://github.com/Propar2750/hexwar
- **Workflow:** Commit and push frequently — every meaningful change should be published to GitHub promptly.

## Project Overview

HexWar is a multi-agent reinforcement learning project where AI players compete for territory control on a hex grid. It's a college project built in phases over ~14 weeks (see BIGPLAN.md for the full roadmap).

**Core game:** Players start on a hex grid, generate troops each turn based on terrain, and attack adjacent tiles to expand territory. Combat uses a defense-bonus formula with probabilistic outcomes. Win by controlling 50%+ of the map or having the most territory after N turns.

## Current State

Phase 1 (Environment Engineering) is well underway. The hex grid engine, terrain generation, A* pathfinding, game engine, playable 2-player demo, rule-based bots, bot tournament runner, and the v2 flat RL environment are implemented. DQN agent scaffolding exists in agents/.

## Reference Documents

Before making changes, consult these docs:

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Dependency layers, file dependency graph, change ripple guide, common-task lookup table, test coverage map. **Read this first when modifying any module.**
- **[BIGPLAN.md](BIGPLAN.md)** — Full 14-week project roadmap and phase breakdown.
- **[GAMELOGIC.md](GAMELOGIC.md)** — Game mechanics specification (combat formula, terrain stats, victory conditions).
- **[DESIGN_DECISIONS.md](DESIGN_DECISIONS.md)** — Rationale behind key architectural choices.
- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** — High-level project description.

Every Python file has a header docstring listing its purpose, dependencies, dependents, and ripple effects. Read the file header before editing.

## Project Structure

```
Root — Hex primitives, algorithms, rendering, entry points
    hex_core.py            Hex coordinates (doubled-width), neighbors, distance, pixel math
    hex_grid.py            HexGrid container, Terrain enum (PLAINS, FERTILE, MOUNTAIN)
    pathfinding.py         A* pathfinding with terrain-aware movement costs
    map_generator.py       Cellular automata terrain + random-walk mountain ranges
    renderer.py            Base pygame hex renderer (terrain colours, highlights)
    main.py                Entry point: pathfinding / terrain demo
    play.py                Entry point: interactive 2-player game
    bot_runner.py          Entry point: headless bot tournament runner
    replay.py              Entry point: replay viewer for recorded bot games
    test_cellular_automata.py  One-off parameter sweep (can be deleted)

game/                      Game logic package
    __init__.py            Public API re-exports
    config.py              GameConfig dataclass (all tunable parameters, MapPreset)
    state.py               GameState, TileState, GamePhase, SupplyChain
    actions.py             MoveAction, EndTurnAction, SetupSupplyChainAction, validation
    combat.py              CombatResolver protocol, DefaultCombatResolver, win_probability
    engine.py              GameEngine — full lifecycle orchestrator (the only state mutator)
    game_renderer.py       Game-aware renderer (ownership, troops, HUD, supply chains)
    bots.py                Bot protocol + 4 strategies (Random, Greedy, Turtle, NoOp)
    environment.py         Gym env v1 (legacy — replaced by flat_env)
    flat_env.py            Gym env v2 (primary RL interface: ego-centric, flat, masked)
    recorder.py            Game recording, interestingness scoring, JSON serialization

agents/                    RL agents package
    __init__.py            Public API re-exports
    dqn_agent.py           DQN with two-tier action masking, epsilon-greedy
    networks.py            QNetwork (MLP feature extractor + Q-head)
    replay_buffer.py       Circular replay buffer (pre-allocated numpy arrays)

tests/                     Pytest test suite
    test_hex_core.py       Coordinate math, pixel conversions
    test_hex_grid.py       Grid container, terrain, tiles
    test_pathfinding.py    A* correctness, costs, obstacles
    test_main_utils.py     Map generator terrain distribution
    test_flat_env.py       Flat env obs/action spaces, masking, rewards, BotFlatAdapter
```

## Key Design Decisions

- **Hex coordinates:** Doubled-width coordinate system, pointy-top orientation (reference: Red Blob Games hex grid guide)
- **Terrain types:** Plains (move cost 1, defense 1, generates 2 troops/turn), Mountains (move cost 3, defense 2, generates 1), Fertile (move cost 1, defense 1, generates 3)
- **Combat formula:** `threshold = defense_bonus * (1 + D + sqrt(D))` — attacker above threshold wins guaranteed, between D and threshold wins probabilistically, at or below D loses guaranteed
- **Game engine pattern:** GameState is a plain data container; all mutations go through GameEngine so rules are enforced in one place. CombatResolver is pluggable via protocol.
- **Config-driven:** All tunable parameters (grid size, troop gen rates, defense bonuses, map gen params) live in `GameConfig` dataclass
- **RL environment:** v2 flat_env is the primary training interface — ego-centric obs, flat action space with two-tier masking (hard/soft), decomposed reward shaping

## Architecture Quick Reference

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full version. Summary of dependency layers:

```
Layer 0  hex_core, game/combat           ← no project imports
Layer 1  hex_grid                        ← hex_core
Layer 2  pathfinding, map_generator, game/config
Layer 3  game/state, game/actions
Layer 4  game/engine                     ← central orchestrator
Layer 5  renderer, game_renderer, bots, flat_env, recorder
Layer 6  agents/ (DQN, networks, replay buffer)
Layer 7  Entry points: main, play, bot_runner, replay
```

## Running

```bash
# Interactive 2-player game
python play.py

# Pathfinding / terrain demo
python main.py

# Bot tournament
python bot_runner.py

# Tests
pytest
```
