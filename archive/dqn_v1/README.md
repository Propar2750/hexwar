# DQN v1 Archive

Archived 2026-03-26. First DQN attempt against NoOp (passive) opponent.

## What it does
- DQN with Double DQN, two-tier action masking (hard/soft), Huber loss
- Trained 5000 episodes on SMALL_FIXED map (48 tiles) vs NoOp opponent
- Territory-based reward shaping (no win/loss bonus)

## Results
- 97% win rate, ~22/48 avg territory, loss converged to ~0.03
- Agent learns to capture territory but gets stuck in action loops
  (shuffling troops back and forth without ending turn)

## Why archived
The sub-step action design causes the agent to waste actions moving troops
in circles. It never learns to press "end turn" because there's no reward
signal for it. Needs a fundamental rethink of the action space or reward
structure before retrying.

## Key files
- `agents/` — DQNAgent, QNetwork, ReplayBuffer, training loop, evaluation, visualization
- `checkpoints/` — 11 model checkpoints (every 500 eps + final)
- `logs/` — training CSV + dashboard PNG
