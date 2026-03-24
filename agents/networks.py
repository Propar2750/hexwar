"""Neural network architectures for RL agents.

Defines QNetwork (MLP with feature extractor + Q-head). The feature
extractor is designed for reuse by PPO's policy/value heads in Phase 3.

Depended on by:
    agents/dqn_agent

Dependencies: None (torch only)

Ripple effects:
    - Changing input dimensions → must match flat_env observation size.
    - Changing output dimensions → must match flat_env action space size.

QNetwork is designed with a split between feature extraction and the Q-head
so that PPO can later reuse the feature layers with its own policy/value heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """MLP Q-value network with separable feature extractor.

    Architecture:
        obs → [Linear → ReLU → LayerNorm] × N → Linear → Q-values

    The ``features()`` method returns the output of the last hidden layer,
    allowing PPO (or other algorithms) to attach their own heads.
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_sizes: tuple[int, ...] = (512, 256, 256),
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev = obs_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.LayerNorm(h)])
            prev = h
        self.feature_layers = nn.Sequential(*layers)

        self.q_head = nn.Linear(hidden_sizes[-1], action_size)
        # Small init for output layer — start Q-values near zero.
        nn.init.uniform_(self.q_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q_head.bias, -3e-3, 3e-3)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the last hidden-layer activations (batch, hidden)."""
        return self.feature_layers(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return raw Q-values (batch, action_size)."""
        return self.q_head(self.features(obs))
