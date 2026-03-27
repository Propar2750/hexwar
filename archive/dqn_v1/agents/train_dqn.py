"""DQN training loop for HexWar — Phase 1: passive opponent.

Usage:
    python -m agents.train_dqn [--episodes 5000] [--lr 1e-4] [--batch-size 64] ...

Logs training metrics to ``logs/dqn_training.csv`` and saves checkpoints
to ``checkpoints/``.

Reward override: strips the +/-10 win/loss bonus from the environment
reward and replaces it with a pure territory-based signal. Against a
passive opponent, "winning" is meaningless — what matters is how many
tiles the agent captures.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch

from game.bots import NoOpBot
from game.config import GameConfig, MapPreset
from game.flat_env import BotFlatAdapter, FlatHexWarEnv
from game.state import GamePhase

from agents.dqn_agent import DQNAgent
from agents.evaluate import evaluate
from agents.replay_buffer import ReplayBuffer


# ── CLI ─────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DQN on HexWar (passive opponent)")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=30_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-episodes", type=int, default=2000)
    p.add_argument("--target-update", type=int, default=1000, help="Train steps between hard target copies")
    p.add_argument("--train-freq", type=int, default=4, help="Agent steps between gradient updates")
    p.add_argument("--min-buffer", type=int, default=1000, help="Min buffer size before training starts")
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--eval-every", type=int, default=50, help="Episodes between evaluations")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=500, help="Episodes between checkpoints")
    p.add_argument("--replay-every", type=int, default=250, help="Episodes between recorded game replays")
    p.add_argument("--max-steps-per-ep", type=int, default=500, help="Force-end episode after this many agent steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--log-dir", type=str, default="logs")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    return p.parse_args(argv)


# ── Reward ──────────────────────────────────────────────────────────

def compute_reward(
    territory_before: int,
    territory_after: int,
    total_tiles: int,
    done: bool,
) -> float:
    """Pure territory-based reward (no win/loss bonus).

    - Per-step: tiles gained this step (can be 0 or 1 typically)
    - On game end: bonus proportional to fraction of map controlled
    """
    tiles_gained = territory_after - territory_before
    reward = float(tiles_gained)  # +1 per tile captured

    if done:
        # Terminal bonus: how much of the map you hold (0..total_tiles).
        # Scale so controlling the whole map gives ~5.0 bonus.
        reward += 5.0 * (territory_after / total_tiles)

    return reward


# ── Training ────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    # Seed everything.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device.
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Environment.
    config = GameConfig(preset=MapPreset.SMALL_FIXED)
    env = FlatHexWarEnv(config, sub_step=True)
    opponent = BotFlatAdapter(NoOpBot())
    total_tiles = env.n_tiles

    obs_size = env.obs_size
    action_size = env.unified_action_size
    print(f"Obs size: {obs_size}  |  Action size: {action_size}  |  Tiles: {total_tiles}")

    # Do a throwaway reset to grab the static hard mask.
    _, init_info = env.reset(seed=args.seed)
    hard_mask = init_info["action_masks"]["hard"]
    print(f"Structurally valid actions: {hard_mask.sum()}/{action_size}")

    # Agent + buffer.
    agent = DQNAgent(
        obs_size=obs_size,
        action_size=action_size,
        hard_mask=hard_mask,
        device=device,
        lr=args.lr,
        gamma=args.gamma,
        grad_clip=args.grad_clip,
    )
    buffer = ReplayBuffer(args.buffer_size, obs_size, action_size)

    # Epsilon schedule (linear decay).
    eps = args.eps_start
    eps_step = (args.eps_start - args.eps_end) / max(args.eps_decay_episodes, 1)

    # Logging.
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)

    log_path = log_dir / "dqn_training.csv"
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow([
        "episode", "epsilon", "reward", "territory", "winner",
        "steps", "loss_mean", "q_mean", "time_s",
    ])

    total_agent_steps = 0
    last_target_update = 0

    print(f"\nStarting training for {args.episodes} episodes...\n")
    t0 = time.time()

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + episode)
        done = False
        ep_reward = 0.0
        ep_steps = 0
        ep_losses: list[float] = []
        ep_q_vals: list[float] = []

        while not done:
            if ep_steps >= args.max_steps_per_ep:
                # Force-end: treat as truncated to avoid infinite episodes.
                done = True
                break

            if info["current_player"] == 0:
                # --- Agent acts ---
                state = env.engine.state
                territory_before = state.territory_count(0)

                soft_mask = info["action_masks"]["soft"]
                action = agent.select_action(obs, soft_mask, eps)
                next_obs, _env_reward, terminated, truncated, next_info = env.step(action)
                done = terminated or truncated

                # Fast-forward through opponent turns so next_obs is
                # from the agent's next decision point.
                while not done and next_info["current_player"] != 0:
                    opp_act = opponent.choose_sub_action(env)
                    next_obs, _, terminated, truncated, next_info = env.step(opp_act)
                    done = terminated or truncated

                # Compute territory-based reward (replaces env reward).
                territory_after = env.engine.state.territory_count(0)
                reward = compute_reward(territory_before, territory_after, total_tiles, done)

                # Soft mask for the *next* state (needed for target Q masking).
                if not done:
                    next_soft = next_info["action_masks"]["soft"]
                else:
                    next_soft = np.zeros(action_size, dtype=np.float32)

                buffer.push(obs, action, reward, next_obs, done, next_soft)
                ep_reward += reward
                ep_steps += 1
                total_agent_steps += 1

                # Train.
                if (
                    len(buffer) >= max(args.min_buffer, args.batch_size)
                    and total_agent_steps % args.train_freq == 0
                ):
                    batch = buffer.sample(args.batch_size, device=device)
                    loss = agent.train_step(batch)
                    ep_losses.append(loss)

                    # Track mean Q for diagnostics.
                    with torch.no_grad():
                        q_vals = agent.q_net(batch["obs"])
                        ep_q_vals.append(q_vals.max(dim=1).values.mean().item())

                    # Target network update.
                    if agent.train_steps - last_target_update >= args.target_update:
                        agent.update_target()
                        last_target_update = agent.train_steps

                obs, info = next_obs, next_info
            else:
                # Safety fallback — shouldn't happen after fast-forward.
                opp_act = opponent.choose_sub_action(env)
                obs, _, terminated, truncated, info = env.step(opp_act)
                done = terminated or truncated

        # Epsilon decay.
        eps = max(args.eps_end, eps - eps_step)

        # Log episode.
        state = env.engine.state
        territory = state.territory_count(0)
        winner = state.winner
        loss_mean = np.mean(ep_losses) if ep_losses else 0.0
        q_mean = np.mean(ep_q_vals) if ep_q_vals else 0.0
        elapsed = time.time() - t0

        writer.writerow([
            episode, f"{eps:.4f}", f"{ep_reward:.4f}", territory,
            winner, ep_steps, f"{loss_mean:.6f}", f"{q_mean:.4f}",
            f"{elapsed:.1f}",
        ])
        log_file.flush()

        # Console output.
        if episode % 10 == 0 or episode == 1:
            print(
                f"Ep {episode:5d} | eps {eps:.3f} | R {ep_reward:+7.2f} | "
                f"terr {territory:3d}/{total_tiles} | "
                f"loss {loss_mean:.4f} | Q {q_mean:.2f} | "
                f"buf {len(buffer):6d} | steps {total_agent_steps}"
            )

        # Evaluation (with replay every replay_every episodes).
        if episode % args.eval_every == 0:
            show_replay = (episode % args.replay_every == 0)
            metrics = evaluate(
                agent, config,
                n_episodes=args.eval_episodes,
                seed=99,
                record_game=show_replay,
            )
            print(
                f"  EVAL | "
                f"terr {metrics['mean_territory']:.1f}/{total_tiles} | "
                f"R {metrics['mean_reward']:.2f} | "
                f"acts {metrics['mean_actions']:.1f}"
            )
            if show_replay and "replay" in metrics:
                metrics["replay"].display(label=f"Ep {episode} Sample Game")

        # Checkpoint.
        if episode % args.ckpt_every == 0:
            path = ckpt_dir / f"dqn_ep{episode}.pt"
            agent.save(path)
            print(f"  Saved checkpoint -> {path}")

    # Final save.
    agent.save(ckpt_dir / "dqn_final.pt")
    log_file.close()
    print(f"\nTraining complete. {args.episodes} episodes in {time.time() - t0:.0f}s")
    print(f"Logs: {log_path}  |  Final checkpoint: {ckpt_dir / 'dqn_final.pt'}")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
