# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

- **Remote:** https://github.com/Propar2750/hexwar
- **Workflow:** Commit and push frequently — every meaningful change should be published to GitHub promptly.

## Project Overview

HexWar is a multi-agent reinforcement learning project where 4 AI players compete for territory control on a hex grid. It's a college project built in phases over ~14 weeks (see BIGPLAN.md for the full roadmap).

**Core game:** 4 players start in corners of a hex grid (~80-120 hexes), gather resources, build units (scouts and soldiers), and compete for map control. All moves are simultaneous. Win by controlling 60%+ of the map or having the most territory after N turns.

## Architecture (Planned)

The project follows four major phases:

1. **Environment Engineering** — Hex grid engine (axial coordinates, A* pathfinding), game state machine (Gym-compatible with `reset()`/`step()`/`render()`), action/observation spaces, and rule-based baseline bots
2. **First RL Agents** — CNN-based state representation (multi-channel 2D: terrain, ownership, unit counts, fog), DQN from scratch, curriculum learning (2→4 players, small→large maps)
3. **Advanced Algorithms** — PPO (actor-critic, GAE, clipped objective), multi-agent self-play with league training, emergent diplomacy/communication
4. **Analysis & Polish** — Strategy visualization, fog of war / POMDP, tournament with Elo ratings, ablation studies

## Key Design Decisions

- **Hex coordinates:** Doubled-width coordinate system, pointy-top orientation (reference: Red Blob Games hex grid guide)
- **Terrain types:** Plains (easy movement), mountains (defensive bonus), fertile land (extra resources)
- **Observation encoding:** Multi-channel 2D array — channel per feature (terrain, ownership, unit counts, fog)
- **Action space:** Multi-head approach — one output per unit + build action + diplomacy action
- **Simultaneous resolution:** All players submit orders at once; conflicts resolved by unit strength + terrain bonus + small random factor
- **Diplomacy:** Binary signals (peace proposal / threat) — cheap talk, not binding
- **Training stability:** Mix of old checkpoints + self-play to avoid strategy cycling

## Running

```
python main.py
```
