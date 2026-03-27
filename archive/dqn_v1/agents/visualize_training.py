"""Visualize DQN training results from logs/dqn_training.csv."""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_training_log(path: str = "logs/dqn_training.csv") -> dict:
    """Load training CSV into arrays."""
    data = {
        "episode": [], "epsilon": [], "reward": [], "territory": [],
        "winner": [], "steps": [], "loss_mean": [], "q_mean": [], "time_s": [],
    }
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["episode"].append(int(row["episode"]))
            data["epsilon"].append(float(row["epsilon"]))
            data["reward"].append(float(row["reward"]))
            data["territory"].append(int(row["territory"]))
            w = row["winner"].strip()
            data["winner"].append(int(w) if w not in ("None", "") else -1)
            data["steps"].append(int(row["steps"]))
            data["loss_mean"].append(float(row["loss_mean"]))
            data["q_mean"].append(float(row["q_mean"]))
            data["time_s"].append(float(row["time_s"]))
    return {k: np.array(v) for k, v in data.items()}


def moving_average(arr: np.ndarray, window: int = 100) -> np.ndarray:
    """Compute moving average with given window."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def main():
    data = load_training_log()
    total_tiles = 48  # SMALL_FIXED map
    n = len(data["episode"])
    window = 100

    # Smoothed series
    reward_ma = moving_average(data["reward"], window)
    territory_ma = moving_average(data["territory"], window)
    loss_ma = moving_average(data["loss_mean"], window)
    q_ma = moving_average(data["q_mean"], window)
    steps_ma = moving_average(data["steps"], window)
    ep_ma = data["episode"][window - 1:]

    # Win rate in rolling windows
    wins = (data["winner"] == 0).astype(float)
    win_rate_ma = moving_average(wins, window)

    # Territory fraction
    terr_frac = data["territory"] / total_tiles
    terr_frac_ma = moving_average(terr_frac, window)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("DQN Training on HexWar — 5000 Episodes (vs NoOp Opponent)",
                 fontsize=16, fontweight="bold", y=0.98)

    # 1. Episode Reward
    ax = axes[0, 0]
    ax.scatter(data["episode"], data["reward"], alpha=0.08, s=4, color="steelblue", label="Per episode")
    ax.plot(ep_ma, reward_ma, color="darkblue", linewidth=2, label=f"{window}-ep moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Reward over Training")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # 2. Territory Captured (fraction of map)
    ax = axes[0, 1]
    ax.scatter(data["episode"], terr_frac * 100, alpha=0.08, s=4, color="forestgreen", label="Per episode")
    ax.plot(ep_ma, terr_frac_ma * 100, color="darkgreen", linewidth=2, label=f"{window}-ep moving avg")
    ax.axhline(y=50, color="red", linestyle="--", linewidth=1, alpha=0.7, label="50% (win threshold)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Territory Captured (%)")
    ax.set_title("Territory Control over Training")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # 3. Training Loss
    ax = axes[1, 0]
    # Filter out zero losses (before training starts)
    mask = data["loss_mean"] > 0
    ax.scatter(data["episode"][mask], data["loss_mean"][mask], alpha=0.08, s=4, color="coral")
    loss_nonzero_ma = moving_average(data["loss_mean"][mask], min(window, mask.sum()))
    loss_ep = data["episode"][mask][min(window, mask.sum()) - 1:]
    ax.plot(loss_ep, loss_nonzero_ma, color="darkred", linewidth=2, label=f"{window}-ep moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Huber Loss")
    ax.set_title("Training Loss (Huber)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Mean Q-Value
    ax = axes[1, 1]
    mask_q = data["q_mean"] > 0
    ax.scatter(data["episode"][mask_q], data["q_mean"][mask_q], alpha=0.08, s=4, color="orchid")
    ax.plot(ep_ma, q_ma, color="purple", linewidth=2, label=f"{window}-ep moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Max Q-Value")
    ax.set_title("Q-Value Estimates over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Win Rate (rolling)
    ax = axes[2, 0]
    ax.plot(ep_ma, win_rate_ma * 100, color="goldenrod", linewidth=2)
    ax.axhline(y=100, color="green", linestyle="--", linewidth=1, alpha=0.5, label="100% wins")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title(f"Win Rate ({window}-Episode Rolling Window)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Epsilon Decay + Steps per Episode
    ax = axes[2, 1]
    ax2 = ax.twinx()
    l1 = ax.plot(data["episode"], data["epsilon"], color="teal", linewidth=2, label="Epsilon")
    l2 = ax2.plot(ep_ma, steps_ma, color="sienna", linewidth=1.5, alpha=0.8, label=f"Steps/ep ({window}-avg)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon", color="teal")
    ax2.set_ylabel("Steps per Episode", color="sienna")
    ax.set_title("Exploration Schedule & Episode Length")
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="center right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = Path("logs/training_dashboard.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved dashboard -> {out_path}")
    plt.close()

    # ── Print summary statistics ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("  DQN TRAINING SUMMARY — 5000 Episodes vs NoOp Opponent")
    print("=" * 65)

    # Phase breakdown
    phases = [
        ("Early (1-500)", 0, 500),
        ("Mid-early (501-1500)", 500, 1500),
        ("Mid (1501-3000)", 1500, 3000),
        ("Late (3001-5000)", 3000, 5000),
    ]

    print(f"\n{'Phase':<22} {'Avg Reward':>11} {'Avg Terr':>10} {'Win %':>7} {'Avg Loss':>10} {'Avg Q':>8}")
    print("-" * 70)
    for name, start, end in phases:
        sl = slice(start, end)
        avg_r = data["reward"][sl].mean()
        avg_t = data["territory"][sl].mean()
        wr = (data["winner"][sl] == 0).mean() * 100
        loss_vals = data["loss_mean"][sl]
        avg_l = loss_vals[loss_vals > 0].mean() if (loss_vals > 0).any() else 0
        avg_q = data["q_mean"][sl][data["q_mean"][sl] > 0].mean() if (data["q_mean"][sl] > 0).any() else 0
        print(f"  {name:<20} {avg_r:>10.2f} {avg_t:>8.1f}/48 {wr:>6.1f}% {avg_l:>10.4f} {avg_q:>7.2f}")

    # Overall
    total_time_min = data["time_s"][-1] / 60
    print(f"\n  Total training time:    {total_time_min:.1f} minutes")
    print(f"  Total agent steps:      {data['steps'].sum():,}")
    print(f"  Final epsilon:          {data['epsilon'][-1]:.4f}")
    print(f"  Peak territory (single): {data['territory'].max()}/48 ({data['territory'].max()/48*100:.0f}%)")
    print(f"  Overall win rate:       {(data['winner'] == 0).mean()*100:.1f}%")

    # Last 100 episodes
    last = slice(-100, None)
    print(f"\n  Last 100 episodes:")
    print(f"    Avg reward:   {data['reward'][last].mean():.2f}")
    print(f"    Avg territory: {data['territory'][last].mean():.1f}/48 ({data['territory'][last].mean()/48*100:.1f}%)")
    print(f"    Win rate:     {(data['winner'][last] == 0).mean()*100:.1f}%")
    print(f"    Avg loss:     {data['loss_mean'][last][data['loss_mean'][last]>0].mean():.4f}")
    print(f"    Avg Q-value:  {data['q_mean'][last][data['q_mean'][last]>0].mean():.2f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
