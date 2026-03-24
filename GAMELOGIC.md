# Game Logic

## Terrain & Troop Generation

Each owned tile generates troops at the start of every round (after all players have acted):

| Terrain  | Troops/turn | Movement cost | Defense bonus |
|----------|-------------|---------------|---------------|
| Fertile  | 3           | 1             | 1             |
| Plains   | 2           | 1             | 1             |
| Mountain | 1           | 3             | 2             |

Unowned tiles start with a garrison of 1 troop.

## Setup

Players take turns selecting a starting tile. Each start is placed with 10 troops. Starting positions must be at least 4 hexes apart. Once all players have placed, the game enters the play phase and the first round of troop generation fires immediately.

## Turns

Each turn a player gets up to 6 moves. A move sends troops from an owned tile to an adjacent tile (must leave at least 1 troop behind). After using moves (or choosing to end early), the turn passes to the next alive player. After all players have acted, a new round begins: victory is checked, the turn counter increments, troops are generated, and supply chains are processed.

## Combat

When troops move onto an enemy or neutral tile, combat occurs.

**Formula:** `threshold = defense_bonus * (1 + D + sqrt(D))` where D = defender troops.

| Condition             | Outcome                                                    |
|-----------------------|------------------------------------------------------------|
| A >= threshold        | Guaranteed attacker win                                    |
| D < A < threshold     | Win probability = (A - D) / (threshold - D), linear scale |
| A <= D                | Guaranteed attacker loss                                   |

**On win:** Attacker takes the tile with `max(1, A - defense_bonus * D)` troops remaining. Defender is wiped out.
**On loss:** All attacking troops are destroyed. Defender keeps their troops unchanged.

## Supply Chains

Players can set up supply chains between two adjacent tiles they own (up to 2 per turn). Each round, all troops on the source tile (minus 1) are forwarded to the chain's terminal endpoint. Chains can be linked — relay tiles keep 1 troop and pass everything onward.

**Rules:**
- One outgoing chain per source tile
- No cycles allowed
- Chains break when either endpoint changes ownership (e.g. captured by enemy)
- When a tile is captured, all supply chains touching it are destroyed

## Victory Conditions

The game ends when any of these occur:

1. **Territory control:** A player controls 50%+ of the map at the end of a round
2. **Elimination:** A player loses all territory (opponent wins immediately)
3. **Time limit:** After 30 turns, the player with the most territory wins (tiebreak: most troops)
