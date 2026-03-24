"""Combat resolution — pluggable via the CombatResolver protocol.

Default formula (from GAMELOGIC.md):
    threshold = defense_bonus * (1 + D + sqrt(D))

    A > threshold   → guaranteed win
    A <= D          → guaranteed loss
    D < A <= thresh → win probability = (A - D) / (threshold - D)

On win:  attacker takes the tile with max(1, A - defense_bonus*D) troops.
On loss: all attacking troops are destroyed; defender unchanged.

To swap in a different combat model, implement CombatResolver and pass
it to GameEngine.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CombatResult:
    """Outcome of a single combat."""

    attacker_won: bool
    attacker_remaining: int  # troops left for the winner on the tile
    defender_remaining: int  # 0 if attacker won, unchanged if lost


class CombatResolver(Protocol):
    """Interface for combat resolution strategies."""

    def resolve(
        self,
        attacker_troops: int,
        defender_troops: int,
        defense_bonus: float,
        rng: random.Random | None = None,
    ) -> CombatResult: ...


def win_probability(attacker: int, defender: int, defense_bonus: float) -> float:
    """Compute attacker's win probability using the threshold formula.

    Returns 1.0 for guaranteed win, 0.0 for guaranteed loss,
    or a linear interpolation in between.
    """
    if defender == 0:
        return 1.0
    threshold = defense_bonus * (1 + defender + math.sqrt(defender))
    if attacker >= threshold:
        return 1.0
    if attacker <= defender:
        return 0.0
    return (attacker - defender) / (threshold - defender)


class DefaultCombatResolver:
    """GAMELOGIC.md threshold-based combat with linear probability band."""

    def resolve(
        self,
        attacker_troops: int,
        defender_troops: int,
        defense_bonus: float,
        rng: random.Random | None = None,
    ) -> CombatResult:
        if rng is None:
            rng = random.Random()

        prob = self._win_probability(attacker_troops, defender_troops, defense_bonus)

        won = rng.random() < prob
        if won:
            cost = int(defense_bonus * defender_troops)
            remaining = max(1, attacker_troops - cost)
            return CombatResult(
                attacker_won=True,
                attacker_remaining=remaining,
                defender_remaining=0,
            )
        else:
            return CombatResult(
                attacker_won=False,
                attacker_remaining=0,
                defender_remaining=defender_troops,
            )

    @staticmethod
    def _win_probability(
        attacker: int, defender: int, defense_bonus: float
    ) -> float:
        return win_probability(attacker, defender, defense_bonus)
