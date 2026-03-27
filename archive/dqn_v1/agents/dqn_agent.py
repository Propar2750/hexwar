"""DQN agent with two-tier action masking for HexWar.

Core RL agent that trains on game/flat_env. Uses QNetwork from
agents/networks and ReplayBuffer from agents/replay_buffer.

Depended on by:
    agents/__init__, (future training scripts)

Dependencies:
    agents/networks (QNetwork), agents/replay_buffer (ReplayBuffer)

Ripple effects:
    - Changing action selection logic → must stay compatible with
      flat_env's action space and masking conventions.
    - Checkpoint format changes → update save/load methods together.

Implements:
  - Epsilon-greedy action selection respecting hard + soft masks
  - Training step with Huber loss and gradient clipping
  - Hard target-network updates
  - Checkpoint save / load
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from agents.networks import QNetwork
from agents.replay_buffer import ReplayBuffer


class DQNAgent:
    """Deep Q-Network agent with two-tier action masking.

    Parameters
    ----------
    obs_size : int
        Length of the flat observation vector.
    action_size : int
        Number of discrete actions (including end-turn at index 0).
    hard_mask : np.ndarray
        Boolean array of shape ``(action_size,)`` — structural validity
        (static per map).  ``True`` = structurally possible.
    device : str or torch.device
        PyTorch device.
    lr : float
        Adam learning rate.
    gamma : float
        Discount factor.
    grad_clip : float
        Max gradient norm for clipping.
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hard_mask: np.ndarray,
        *,
        device: str | torch.device = "cpu",
        lr: float = 1e-4,
        gamma: float = 0.99,
        grad_clip: float = 10.0,
    ) -> None:
        self.device = torch.device(device)
        self.action_size = action_size
        self.gamma = gamma
        self.grad_clip = grad_clip

        # Networks
        self.q_net = QNetwork(obs_size, action_size).to(self.device)
        self.target_net = QNetwork(obs_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        # Static hard mask — True where action is structurally valid.
        self.hard_mask = torch.as_tensor(hard_mask, dtype=torch.bool, device=self.device)

        self.train_steps = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs: np.ndarray,
        soft_mask: np.ndarray,
        epsilon: float,
    ) -> int:
        """Epsilon-greedy action selection with two-tier masking.

        With probability *epsilon* a uniformly random **valid** action is
        chosen.  Otherwise the action with the highest masked Q-value is
        selected.
        """
        soft_t = torch.as_tensor(soft_mask, dtype=torch.float32, device=self.device)

        if np.random.random() < epsilon:
            # Random among fully-valid actions (hard=True AND soft==0).
            valid = self.hard_mask & (soft_t == 0.0)
            valid_idx = valid.nonzero(as_tuple=True)[0]
            if len(valid_idx) == 0:
                return 0  # fallback: end turn
            return valid_idx[torch.randint(len(valid_idx), (1,))].item()

        # Greedy: pick argmax of masked Q-values.
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_t).squeeze(0)           # (action_size,)
            q_values[~self.hard_mask] = float("-inf")
            q_values[soft_t != 0.0] = float("-inf")            # hard-block state-invalid
            return q_values.argmax().item()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, batch: dict[str, torch.Tensor]) -> float:
        """One gradient step on a sampled batch.  Returns the loss value."""
        obs = batch["obs"]                     # (B, obs_size)
        action = batch["action"]               # (B, 1)
        reward = batch["reward"]               # (B, 1)
        next_obs = batch["next_obs"]           # (B, obs_size)
        done = batch["done"]                   # (B, 1)
        next_soft = batch["next_soft_mask"]    # (B, action_size)

        # Current Q(s, a)
        q_values = self.q_net(obs)                              # (B, A)
        q_sa = q_values.gather(1, action)                       # (B, 1)

        # Double DQN target: Q_target(s', argmax_a Q_online(s', a))
        # Online net selects best action; target net evaluates it.
        # This reduces Q-value overestimation.
        with torch.no_grad():
            # Build validity mask for next state.
            valid = self.hard_mask.unsqueeze(0).expand_as(next_soft).clone()
            valid[next_soft != 0.0] = False

            # Online net picks the best valid action.
            online_q = self.q_net(next_obs)                     # (B, A)
            online_q[~valid] = float("-inf")
            best_actions = online_q.argmax(dim=1, keepdim=True) # (B, 1)

            # Target net evaluates that action.
            target_q = self.target_net(next_obs)                # (B, A)
            max_next_q = target_q.gather(1, best_actions)       # (B, 1)
            target = reward + self.gamma * max_next_q * (1.0 - done)

        loss = self.loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.train_steps += 1
        return loss.item()

    # ------------------------------------------------------------------
    # Target network
    # ------------------------------------------------------------------

    def update_target(self) -> None:
        """Hard-copy online network weights to the target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_steps": self.train_steps,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.train_steps = ckpt["train_steps"]
