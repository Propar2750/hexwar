"""Circular replay buffer backed by pre-allocated numpy arrays.

Used by agents/dqn_agent for experience replay during training.
Self-contained utility with no project dependencies.

Depended on by:
    agents/dqn_agent

Dependencies: None (numpy, torch only)
"""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity circular buffer for DQN experience replay.

    Stores transitions as numpy arrays and converts to PyTorch tensors
    on-demand when sampling. The hard mask is static per map and is NOT
    stored here — it lives on the agent.

    Each transition: (obs, action, reward, next_obs, done, next_soft_mask).
    """

    def __init__(self, capacity: int, obs_size: int, action_size: int) -> None:
        self.capacity = capacity
        self.pos = 0
        self.size = 0

        # Pre-allocate arrays.
        self.obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.bool_)
        self.next_soft_mask = np.zeros((capacity, action_size), dtype=np.float32)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_soft_mask: np.ndarray,
    ) -> None:
        """Store one transition, overwriting the oldest if full."""
        i = self.pos
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = done
        self.next_soft_mask[i] = next_soft_mask

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        """Sample a random batch and return as tensors on *device*."""
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return {
            "obs": torch.as_tensor(self.obs[indices], device=device),
            "action": torch.as_tensor(self.action[indices], device=device).unsqueeze(1),
            "reward": torch.as_tensor(self.reward[indices], device=device).unsqueeze(1),
            "next_obs": torch.as_tensor(self.next_obs[indices], device=device),
            "done": torch.as_tensor(self.done[indices], dtype=torch.float32, device=device).unsqueeze(1),
            "next_soft_mask": torch.as_tensor(self.next_soft_mask[indices], device=device),
        }

    def __len__(self) -> int:
        return self.size
