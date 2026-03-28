# HexWar — Design Decisions & Learnings

Everything I chose, why I chose it, and what I learned along the way.

---

## Table of Contents

1. [Hex Coordinate System](#1-hex-coordinate-system)
2. [Grid Data Structure](#2-grid-data-structure)
3. [Terrain System](#3-terrain-system)
4. [Map Generation](#4-map-generation)
5. [Pathfinding](#5-pathfinding)
6. [Game State Architecture](#6-game-state-architecture)
7. [Configuration System](#7-configuration-system)
8. [Action Design](#8-action-design)
9. [Combat System](#9-combat-system)
10. [Game Engine Pattern](#10-game-engine-pattern)
11. [Supply Chains](#11-supply-chains)
12. [Victory Conditions](#12-victory-conditions)
13. [Rendering](#13-rendering)
14. [Interactive Game Client](#14-interactive-game-client)
15. [Rule-Based Bots](#15-rule-based-bots)
16. [Gym Environment (v1 — Multi-Channel 2D)](#16-gym-environment-v1--multi-channel-2d)
17. [Flat Environment (v2 — Ego-Centric Flat Vector)](#17-flat-environment-v2--ego-centric-flat-vector)
18. [Action Masking (Two-Tier)](#18-action-masking-two-tier)
19. [Reward Shaping](#19-reward-shaping)
20. [Game Recording & Replay](#20-game-recording--replay)
21. [Testing Strategy](#21-testing-strategy)
22. [Type System & Python Patterns](#22-type-system--python-patterns)
23. [Project Architecture & Separation of Concerns](#23-project-architecture--separation-of-concerns)
24. [What I'd Do Differently](#24-what-id-do-differently)

---

## 1. Hex Coordinate System

**Choice:** Doubled-width coordinates, pointy-top orientation.

**Alternatives considered:**
- Offset coordinates (simpler but asymmetric neighbor logic)
- Cube coordinates (elegant math but 3 values per coordinate is wasteful)
- Axial coordinates (good, but doubled-width felt more natural for a rectangular grid)

**Why doubled-width:** The constraint `col + row must be even` gives a clean rectangular layout while keeping neighbor lookups as simple constant offsets. Every hex has exactly 6 neighbors found by adding from a fixed `DIRECTIONS` array: `[(2,0), (1,-1), (-1,-1), (-2,0), (-1,1), (1,1)]`. No odd/even row branching logic needed.

**Why pointy-top:** Rows align horizontally, which maps better to a rectangular grid and feels natural for a top-down strategy game. Flat-top would have worked too — this was mostly aesthetic.

**What I learned:**
- Red Blob Games' hex guide is genuinely the gold standard. Every hex grid project should start there.
- The `__post_init__` validation (`if (col + row) % 2 != 0: raise`) catches coordinate bugs early. Without it, you get silent wrong-neighbor lookups that are brutal to debug.
- Pixel-to-hex conversion is surprisingly tricky — you convert to axial intermediates, round all three cube coordinates, then convert back. Getting this right required careful reading of the rounding algorithm.
- Making `HexCoord` a frozen dataclass with `slots=True` was worth it for both hashability (dict keys, set membership) and minor performance gains.

---

## 2. Grid Data Structure

**Choice:** `dict[HexCoord, HexTile]` for the grid storage.

**Why not a 2D array:** The doubled-width system means roughly half the array indices would be invalid (odd `col+row`). A dictionary only stores real hexes, so no wasted space and no need to check validity on every access.

**Why not adjacency list / graph:** Hex grids have such regular structure that a graph abstraction adds complexity without benefit. The constant `DIRECTIONS` array gives you neighbors in O(1) — no need for edge lists.

**What I learned:**
- `neighbors_of(coord)` filtering out-of-bounds neighbors by checking dict membership is clean and avoids boundary-special-casing entirely.
- The dictionary approach scales fine for the grid sizes I'm using (48–280 tiles). For thousands of tiles, a numpy array with an index mapping might be faster, but that's premature optimization for this project.

---

## 3. Terrain System

**Choice:** Three terrain types with distinct mechanical roles.

| Terrain  | Move Cost | Defense Bonus | Troops/Turn |
|----------|-----------|---------------|-------------|
| Plains   | 1.0       | 1.0           | 2           |
| Mountain | 3.0       | 2.0           | 1           |
| Fertile  | 1.0       | 1.0           | 3           |

**Why these three:** Each creates a distinct strategic identity:
- **Plains** are the baseline — cheap to cross, cheap to hold, medium income.
- **Mountains** are natural fortifications — hard to reach, hard to take, low income. They shape the map into lanes and chokepoints.
- **Fertile land** is the prize — easy to reach but worth fighting over because of the income boost.

**Why not more types:** Three is enough to create interesting maps without overwhelming the observation space for RL agents. More terrain types would add complexity without proportional strategic depth at this stage.

**What I learned:**
- Storing terrain stats in dictionaries keyed by the `Terrain` enum (inside `GameConfig`) was much cleaner than hardcoding values. When I needed to tweak balance, I changed one dict instead of hunting through code.
- The defense bonus on mountains matters more than I initially expected — it makes mountain tiles surprisingly hard to take, which naturally creates frontlines along ridges.

---

## 4. Map Generation

**Choice:** Two-stage procedural generation — cellular automata for fertile clusters + random walk for mountain ranges.

### Stage 1: Cellular Automata (Fertile Land)

1. Seed random tiles as fertile with probability `fertile_p` (default 0.45)
2. Each iteration: tile flips to the majority terrain of its neighbors (if ≥`ca_threshold` agree)
3. Run for `ca_iterations` (default 5), stop early if stable

**Why CA:** It produces organic, blob-like clusters that look natural. Random scattering would create noise; hand-placed regions wouldn't scale. CA is the sweet spot of "looks designed, is algorithmic."

### Stage 2: Random Walk (Mountain Ranges)

1. Pick a random starting tile and direction
2. Walk forward, placing mountains, with weighted direction changes:
   - Straight: 32%, slight turn: 28% each, hard turn: 6% each
3. Terminate probabilistically after `min_range_steps`

**Why random walks:** Mountains should form ridges and ranges, not scattered boulders. Random walks with directional momentum produce ridge-like formations that divide the map into natural regions — exactly what you want for strategy.

**Why these specific weights:** Heavy straight + slight-turn bias (88% combined) makes ranges that feel like real mountain chains — they meander but have a clear direction. The 12% hard-turn probability adds occasional dramatic bends.

**What I learned:**
- Cellular automata iterations converge fast. 5 iterations is usually enough; I added early-stop detection when no tiles change.
- The `MapPreset` system (SMALL, MEDIUM, LARGE) with different range counts and lengths was essential. A "large" mountain config on a small map just fills everything with mountains.
- The SMALL_FIXED preset (hand-crafted symmetric map) was invaluable for testing. Procedural maps make tests non-deterministic; having one fixed map with known tile positions made debugging much easier.
- Balanced start placement was harder than expected. The engine tries up to 10 different terrain seeds, greedily picks spawn points that are far apart, and scores them by resource balance (min/max ratio of nearby troop generation). Sometimes no "fair" placement exists and you just take the best attempt.

---

## 5. Pathfinding

**Choice:** A* with terrain-aware movement costs.

**Why A* over Dijkstra:** The hex distance heuristic is admissible (all costs ≥ 1.0), so A* is strictly faster. On these grid sizes the difference is small, but it was easy to implement correctly.

**Why terrain costs:** Mountains cost 3× to traverse. This makes pathfinding non-trivial and creates strategic depth — the shortest path isn't always the cheapest. Armies have to decide: go through the mountain pass or take the long way around?

**Implementation details:**
- Priority queue entries: `(f_score, tiebreaker, coord)` — the tiebreaker (incrementing int) avoids comparing `HexCoord` objects, which would crash since they don't implement `<`.
- Stale entry pruning: when we pop a node that's already been visited with a better cost, we skip it rather than re-processing.
- Custom `cost_fn` and `passable_fn` parameters: the same A* code powers both the demo pathfinding visualization and the game's movement validation.

**What I learned:**
- The tiebreaker trick is essential when your node type isn't orderable. Python's `heapq` compares tuples element-by-element, so without a tiebreaker, equal f-scores try to compare `HexCoord` objects.
- Returning `PathResult(path=(), cost=inf)` for "no path" is cleaner than returning `None` — callers can always unpack the result without checking for None first.

---

## 6. Game State Architecture

**Choice:** `GameState` is a plain data container with no mutation methods. All changes go through `GameEngine`.

```
GameState (data)          GameEngine (rules)
├── grid: HexGrid         ├── reset()
├── tiles: dict            ├── execute_action()
├── supply_chains          ├── _handle_move()
├── phase: GamePhase       ├── _generate_troops()
├── current_player         └── _check_victory()
├── turn
└── winner
```

**Why this separation:**
- **Testing:** You can construct any game state directly and test engine methods against it without playing through a full game.
- **Serialization:** Plain data containers are easy to snapshot for recording/replay.
- **Single source of truth:** All rule enforcement lives in one class. No risk of state mutation from two different places disagreeing about the rules.

**What I learned:**
- This is basically the "anemic domain model" pattern, which gets criticized in enterprise Java but works perfectly for game state. Game state really is just data — the interesting logic is in the rules, not the data.
- `GamePhase` as an enum (SETUP → PLAYING → GAME_OVER) with explicit phase checks in every engine method prevents entire classes of bugs. "You can't move during setup" is enforced once, not hoped-for everywhere.
- Tracking `moves_made` and `supply_chains_set_this_turn` as counters on the state (reset each turn) was simpler than computing them from action history.

---

## 7. Configuration System

**Choice:** Single `GameConfig` dataclass with preset-based defaults.

**How it works:**
1. `MapPreset` enum: `SMALL`, `SMALL_FIXED`, `MEDIUM`, `LARGE`
2. `MAP_PRESET_PARAMS` maps each preset to a full parameter dictionary
3. `GameConfig.__post_init__` applies preset defaults to any field left as `None`
4. Fallback to MEDIUM if no preset specified

**Why presets + override:** You want sensible defaults (so you can write `GameConfig()` and get a working game) but also full control for experiments (change one parameter without specifying everything).

**What I learned:**
- Every magic number in the game engine should be in the config. I started with some hardcoded values and kept having to extract them when I needed to tweak balance. Now everything is in one place: grid size, terrain stats, combat bonuses, map gen params, win conditions.
- The SMALL_FIXED preset (8×6 hand-crafted map) became my go-to for development. Known tile positions, known spawn points, 180° rotational symmetry — perfect for testing that both players have equal chances.

---

## 8. Action Design

**Choice:** Frozen dataclasses for actions with separate validation functions.

```python
@dataclass(frozen=True)
class MoveAction:
    source: HexCoord
    target: HexCoord
    troops: int

@dataclass(frozen=True)
class SetupSupplyChainAction:
    source: HexCoord
    destination: HexCoord

@dataclass(frozen=True)
class EndTurnAction:
    pass
```

**Why frozen:** Actions are immutable facts — "player moved 5 troops from A to B." Freezing them makes them hashable (useful for recording) and prevents accidental modification.

**Why separate validation functions (not methods):** Validation needs access to the full game state, which actions don't carry. `validate_move(action, state, config)` keeps actions as pure data and validation logic alongside the action definitions.

**Why validation returns error strings (not exceptions):** The engine can check validity silently and skip invalid actions (important for RL, where agents will try invalid things). Exceptions would require try/catch everywhere. Returning `None` for valid / `str` for invalid is simple and cheap.

**What I learned:**
- The `get_valid_targets()` function (returns all legal targets for a source tile) was essential for the UI — it powers the green highlight showing where you can move. Calculating this once and caching it is much better than validating each target individually.
- Supply chain cycle detection via DFS (`_would_create_cycle()`) was the trickiest validation to get right. You need to walk the chain graph forward and check if the new chain would create a loop. I initially forgot that relay tiles (both source and destination of different chains) exist, which caused a bug.

---

## 9. Combat System

**Choice:** Threshold-based probabilistic combat with a pluggable resolver protocol.

**The formula:**
```
threshold = defense_bonus × (1 + D + √D)

A ≥ threshold        → guaranteed win
D < A < threshold    → win probability = (A - D) / (threshold - D)
A ≤ D                → guaranteed loss
```

**Why this specific formula:**
- The `√D` term makes the defender advantage scale sub-linearly — 4 defenders aren't twice as hard as 2. This prevents "turtling" from being dominant.
- The linear probability band between D and threshold creates meaningful risk/reward decisions. You can attack with borderline numbers and sometimes win.
- The guaranteed-loss floor (A ≤ D) means you can't suicide-rush with 1 troop and get lucky.

**Why defense bonus as a multiplier:** Mountains have `defense_bonus = 2.0`, which doubles the threshold. This makes mountain tiles genuinely hard to take without being unassailable. The attacker needs roughly twice the troops — significant but not impossible.

**Troop cost on win:** `max(1, A - ceil(defense_bonus × D))`. Winners lose troops proportional to the defense, so pyrrhic victories are real. You might take a mountain tile but arrive with only 1 troop, making it easy to recapture.

**Why pluggable (Protocol):**
- Testing with a deterministic resolver (always win / always lose) makes engine tests predictable.
- Future plans include exploring different combat models (e.g., Lanchester's laws, dice-based) without touching the engine.

**What I learned:**
- The `win_probability()` standalone function (outside the resolver class) was a good call. The bots use it to evaluate attacks without needing a resolver instance. The renderer uses it to show combat previews.
- Passing `random.Random` instances (not using global random) through the resolver makes games reproducible with a seed. This was critical for debugging — I could replay exact sequences.

---

## 10. Game Engine Pattern

**Choice:** Stateful controller that owns the game state and enforces all rules.

**Key responsibilities:**
1. **Reset:** Generate map, find balanced starts, initialize state
2. **Action dispatch:** `execute_action()` routes to `_handle_move()`, `_handle_supply_chain()`, or `_handle_end_turn()`
3. **Move execution:** Decrement source → branch (friendly reinforce / neutral garrison / enemy combat) → update ownership → break affected supply chains
4. **Turn management:** Cycle players, increment turn counter, generate troops at round boundaries
5. **Victory detection:** Territory threshold (50%), elimination, or turn limit

**Why centralized mutation:**
- Every state change goes through one class. When I had a bug where supply chains weren't breaking on tile capture, I knew exactly where to look.
- The engine can enforce sequencing (can't move during setup, can't exceed moves_per_turn) without cooperation from callers.

**Balanced auto-placement algorithm:**
1. Try up to 10 different terrain seeds
2. For each: greedily place players ≥ `min_start_distance` apart
3. Score placements by `min/max` ratio of nearby resource value
4. Accept if ratio ≥ `balance_threshold`, else keep the best attempt

**What I learned:**
- The "break supply chains on tile capture" logic was the most bug-prone part. A chain breaks if either endpoint changes ownership, but you need to check this after every combat resolution, not just at end of turn. I initially had chains surviving one move and forwarding troops to enemy tiles.
- Troop generation at round boundaries (after all players act) rather than at each player's turn start prevents timing advantages.
- The 10-attempt loop for balanced placement was a pragmatic solution. Perfect balance is NP-hard on procedural maps; "good enough after 10 tries" works in practice.

---

## 11. Supply Chains

**Choice:** Directed one-to-one logistics links between owned tiles.

**Rules:**
- One outgoing chain per source tile
- Source and destination must be adjacent and owned by the same player
- Max 2 new chains per turn
- Chains break if either endpoint loses ownership
- A tile can be both a destination and a source (relay chains)

**Processing (end of turn):**
1. Find terminal endpoints (tiles that are chain destinations but not sources of another chain)
2. Walk backward from terminals, forwarding troops (each tile keeps 1, sends the rest)
3. Relay tiles forward troops through the chain

**Why this design:**
- One outgoing chain per tile prevents degenerate fan-out patterns.
- Adjacent-only keeps chains physically grounded (no teleportation).
- Breaking on ownership change creates dynamic frontlines — you can disrupt enemy logistics by capturing a key relay tile.
- Relay chains enable long-distance logistics while requiring investment (you need to own every tile in the chain).

**What I learned:**
- Cycle detection was essential. Without it, players could create circular chains that would loop troop forwarding infinitely. The DFS-based `_would_create_cycle()` check prevents this.
- Terminal endpoint memoization in the engine (caching which tile a chain ultimately delivers to) was a performance win for long relay chains.
- Supply chains add surprising strategic depth — cutting an enemy's supply line by capturing one tile in the middle is often more impactful than a direct assault.

---

## 12. Victory Conditions

**Choice:** Three ways to win, creating tension between aggression and caution.

1. **Territory control:** Own ≥ 50% of all tiles → immediate win
2. **Elimination:** Capture all of opponent's tiles → immediate win
3. **Turn limit:** After `max_turns` (default 30), most territory wins. Tiebreak: most total troops.

**Why 50% threshold:** High enough that you can't win by accident, low enough that one dominant push can close the game. 60% (from the original BIGPLAN) felt too grindy in practice.

**Why turn limit:** Without it, two defensive players could stalemate forever. The turn limit creates urgency — you need to be expanding, not just defending.

**What I learned:**
- The tiebreaker (total troops) matters more than I expected. In close games where both players hover near 50%, troop count determines who's actually "winning" the attrition war.
- Checking victory after every action (not just end of turn) is important. A mid-turn capture could push you over 50%, and the game should end immediately.

---

## 13. Rendering

**Choice:** Layered pygame rendering with a base renderer and game-specific subclass.

**Base renderer (`renderer.py`):**
- Hex polygon drawing via vertex calculation
- Terrain-based fill colors (blues/greens for plains, browns for mountains)
- Grid lines, hover highlights, coordinate labels
- Origin offset for centering

**Game renderer (`game_renderer.py`):**
- Three rendering layers: unowned tiles → owned tiles with player borders → highlights/selection on top
- Player colors: blue, red, green, yellow — blended 22% into terrain base for subtle ownership tinting
- Troop count labels on each tile (with shadow for readability)
- Move preview arrows with troop count
- Supply chain dashed lines
- HUD panel: turn, territory %, moves remaining, supply chain slots
- Game-over overlay: translucent dark screen with winner text

**Why layered rendering:**
- Z-order control. Selection highlights must always render on top of everything, player borders on top of terrain. Without explicit layers, you get flickering and rendering order bugs.

**Why subtle color blending (22%):**
- Full player-color tiles make terrain unreadable. Subtle tinting preserves terrain visibility while clearly showing ownership. The 22% blend ratio was tuned by eye.

**What I learned:**
- Lazy font initialization (`pygame.font.init()` only when first needed) avoids crashes when importing the renderer module without a display.
- Shadow text (dark offset behind light text) is a simple trick that makes labels readable on any background color.
- The dark background (30, 30, 30) was chosen after trying white and gray — dark backgrounds make colored tiles pop and reduce eye strain during long play sessions.
- Hex vertex calculation needs to account for pointy-top orientation: vertices start at 30° offset. Getting this wrong gives you flat-top hexes that don't match the coordinate system.

---

## 14. Interactive Game Client

**Choice:** State machine UI with pygame event loop.

**UI state flow:**
```
Nothing selected
  → Click owned tile: SELECT (show valid targets)
    → Click valid target: TARGETING (show troop slider)
      → Scroll wheel: adjust troops
      → Click target: EXECUTE (send move)
        → Back to nothing selected

S key → SUPPLY CHAIN MODE
  → Click source: select source
    → Click adjacent owned tile: create chain
```

**Controls:**
- Left click: select/execute
- Scroll wheel: adjust troop count
- Space: end turn
- S: supply chain mode
- R: restart
- Q/Escape: quit

**Why this specific flow:** It mirrors classic strategy game UX — select unit, see options, confirm action. The scroll wheel for troop count is more ergonomic than typing a number or using a slider widget.

**What I learned:**
- Separating `UIState` from `GameState` was essential. The UI tracks selection, hover, and targeting state that has nothing to do with the game rules. Mixing them would create a mess.
- Message display with frame-limited timers (auto-clear after N frames) gives feedback without cluttering the screen. "Invalid move!" flashes briefly and disappears.
- `pixel_to_hex()` for hover detection works well but has edge cases near hex borders. The rounding algorithm from Red Blob Games handles this correctly.

---

## 15. Rule-Based Bots

**Choice:** Three bots with distinct strategies, all implementing a `Bot` protocol.

### RandomBot
- Picks random movable tiles, random targets, random troop counts
- No supply chains
- **Purpose:** Absolute baseline. If your RL agent can't beat this, something is fundamentally wrong.

### GreedyExpansionBot
- Evaluates attacks by `tile_value × win_probability × neutral_bonus`
- Only attacks with ≥ 60% win probability
- Falls back to reinforcing border tiles from interior
- Sets up supply chains from interior to adjacent border tiles
- **Purpose:** "Reasonable play" baseline. Represents what a simple heuristic achieves.

### TurtleDefendBot
- Consolidates interior troops to weakest border tile
- Only attacks with ≥ 85% win probability (near-guaranteed)
- Aggressively feeds border tiles via supply chains
- **Purpose:** Tests that aggressive strategies can beat passive play. If turtle always wins, combat is too defender-favored.

**Why a Protocol (not abstract base class):**
- Structural typing — any object with a `choose_action(state, player_id, config)` method works. No inheritance required. Easy to drop in a new bot.

**What I learned:**
- The `_is_border()` helper (has any non-owned neighbor) is used by both bots and the flat environment's tactical features. Extracting it as a utility was worthwhile.
- `_troops_for_guaranteed_win()` using `ceil(defense_bonus × (1 + D + √D))` directly from the combat formula ensures bots and combat system stay in sync.
- Running thousands of bot-vs-bot games validated the game balance. Initially, GreedyBot won 95%+ against TurtleBot, which was expected — pure defense shouldn't beat smart aggression. But Random vs Greedy was ~0% Random wins, confirming that strategy matters.

---

## 16. Gym Environment (v1 — Multi-Channel 2D)

**Choice:** 4-channel 2D grid observation with dict-based actions.

**Observation shape:** `(4, grid_height, grid_width)`
- Channel 0: terrain type (0/1/2)
- Channel 1: ownership (normalized to [-1, 1])
- Channel 2: troop count (raw integer)
- Channel 3: current-player mask (1.0 if owned by acting player)

**Action space (Dict):**
- `source_index`: tile index (or -1 for EndTurn, -2 for supply chain)
- `direction`: 0–5 (hex direction)
- `troops`: integer (clamped)

**Why 2D multi-channel first:** Natural fit for CNN-based agents (the grid is spatial data). This was the "obvious" first design.

**Why I moved on to v2:** Several problems emerged:
- The 2D layout has wasted cells (doubled-width means ~half the grid is padding)
- Dict action space is awkward for standard RL algorithms
- No action masking — agents waste training time on invalid actions
- Single-action-per-step means no coordinated multi-move planning
- Ownership encoding isn't ego-centric — agent has to learn "I am player 1" as implicit knowledge

---

## 17. Flat Environment (v2 — Ego-Centric Flat Vector)

**Choice:** Complete redesign with five innovations over v1.

### Innovation 1: Ego-Centric Observation
- Ownership rotated relative to acting player: mine=1, neutral=0, enemies=2+
- Global features reordered: my stats first, then enemies
- **Why:** In multi-agent RL, the agent shouldn't need to learn "which player number am I." Ego-centric encoding means the same network works regardless of player index.

### Innovation 2: Per-Tile Neighborhood Features
Each tile encodes not just itself but its 6 neighbors:
- **Self features (8 + num_players):** terrain one-hot, ownership one-hot, normalized troops, visibility, supply chain flags
- **Per neighbor (6 × 4):** exists, relative_owner (-1 same / 0 neutral / 1 enemy), troop_diff (normalized), terrain_defense
- **Derived tactical (4):** is_border, best_attack_probability, best_threat_probability, border_troop_ratio
- **Total per tile:** 36 + num_players

**Why neighborhood features:** An MLP can't see spatial relationships. By encoding each tile's local context directly, even a flat network understands "this tile has a weak enemy neighbor with low defense."

### Innovation 3: Precomputed Neighbor Table
- `np.ndarray` shape `(n_tiles, 6)`, values are neighbor indices or -1
- Built once at `reset()`, used for O(1) neighbor lookups in mask generation and observation building

**Why:** Observation and mask building happen every step. Looking up neighbors via dict/method calls each time was measurably slow. The numpy table makes it a single array index.

### Innovation 4: Full-Turn Mode
- Agent outputs entire turn as one vector: `[move_0, ..., move_{M-1}, sc_0, sc_1]`
- All actions executed sequentially in one `step()` call
- Invalid actions mid-execution get skipped with -0.01 penalty

**Why:** Single-action-per-step can't learn coordinated strategies (pincer attacks, logistics setup before assault). Full-turn output forces the agent to plan.

### Innovation 5: Sub-Step Mode (Alternative)
- Single unified action integer per step
- 0 = end turn, 1..N = move, N+1..M = supply chain
- More standard RL interface, but loses coordination signal

**Why both:** Different algorithms suit different interfaces. PPO might prefer sub-step, while planning-based methods might prefer full-turn.

### Action Encoding
- **Move:** `tile_idx × 6 × 4 + direction × 4 + troop_bin + 1` (0 = no-op)
- **Supply chain:** `tile_idx × 6 + direction + 1` (0 = no-op)
- **Troop bins:** [25%, 50%, 75%, 100%] of available troops (discretized to keep action space manageable)

**What I learned:**
- Soft normalization `troops / (troops + 50)` is better than raw counts or hard clipping. It compresses the range [0, ∞) into [0, 1) without losing information about relative magnitudes.
- The BotFlatAdapter (wraps rule-based bots to produce flat-env actions) was essential for testing — I could run full games with existing bots in the new environment format.
- Computing observation and mask sizes from config parameters (not hardcoded) was crucial. Different presets have different tile counts, which changes every dimension.
- The pending game-over reward buffer was a subtle but important fix: in 2-player alternating games, when player A's move triggers game-over, player B's trajectory also needs the terminal reward. Buffering it for delivery on the "next" step prevents missing reward signals.

---

## 18. Action Masking (Two-Tier)

**Choice:** Hard structural masks (bool) + soft state-based masks (float logit bias).

### Hard Masks
- Boolean arrays, static per map (computed once at reset)
- Encodes structural impossibility: "tile 5 has no neighbor in direction 3" → always False
- Used to constrain the action space permanently

### Soft Masks
- Float arrays, recomputed every step
- 0.0 = valid, -10.0 = invalid
- Encodes state-dependent invalidity: "tile 5 has only 1 troop, can't send any" → -10.0 bias

**Why two tiers:**
- Hard masks never change, so computing them once saves work every step
- Soft masks as logit biases (not hard zeroing) let the agent learn "this is bad" rather than "this doesn't exist." The gradient still flows through -10.0 biased logits, just heavily discouraged.
- Standard RL libraries (e.g., Stable Baselines3, CleanRL) support logit masking natively

**Why -10.0 specifically:** Large enough to make the action extremely unlikely after softmax, small enough to avoid numerical issues. Empirically standard in masked RL literature.

**What I learned:**
- Computing masks from the pre-turn state (not mid-execution state) was a deliberate tradeoff. Mid-execution state changes as earlier actions resolve, but recomputing masks between every action in full-turn mode would be expensive and complicate the interface. "Mask at turn start, skip if invalid mid-execution" is simpler and works.
- The no-op action (index 0) is always valid in every mask. This gives the agent an escape hatch — if it's confused, it can do nothing rather than being forced into a bad action.

---

## 19. Reward Shaping

**Choice:** Decomposed reward with interpretable components.

### v1 (Multi-Channel Env)
- Simple: `territory_delta + terminal_bonus`
- Territory delta: new territory count − old territory count (per turn)
- Terminal: +10.0 win, -10.0 loss

### v2 (Flat Env) — Decomposed
| Component | Weight | Signal |
|-----------|--------|--------|
| Territory delta | 0.10 | Expand territory |
| Combat efficiency | 0.05 | Won attacks / total attacks |
| Border pressure delta | 0.03 | Improve frontline troop ratio |
| Supply chain value | 0.02 | Reward active logistics |
| Action skip penalty | -0.01 | Avoid wasted actions |
| Terminal bonus | ±10.0 | Win/loss |

**Why decomposed:**
- Raw win/lose is too sparse — agents need intermediate signal to learn.
- But monolithic "territory delta" doesn't tell the agent *why* it gained territory. Was it good combat? Good logistics? Good positioning?
- Decomposed rewards let you diagnose training problems. If combat efficiency reward is high but territory reward is low, the agent is winning battles but not capitalizing.

**Why these specific weights:**
- Territory (0.10) dominates because it's the win condition.
- Combat efficiency (0.05) rewards winning the fights you pick — don't suicide-attack.
- Border pressure (0.03) rewards positioning advantage even without combat.
- Supply chain (0.02) is small because chains are means to an end, not the goal.
- Skip penalty (-0.01) is tiny but accumulates — 4 skipped actions = -0.04, comparable to a small territory gain.

**What I learned:**
- The weights are educated guesses that will need tuning during RL training. The point of decomposing is that each component can be tuned independently.
- Terminal bonus magnitude (±10.0) needs to dominate the sum of all intermediate rewards across a game. With ~30 turns and intermediate rewards totaling ~2-3 per game, ±10.0 ensures the agent prioritizes winning.
- The skip penalty was a late addition that solved a real problem: without it, agents in full-turn mode would sometimes output entirely no-op vectors (all zeros) because doing nothing has zero risk.

---

## 20. Game Recording & Replay

**Choice:** Lightweight per-frame snapshots with JSON serialization.

### Recording
- `FrameSnapshot` captures: action taken, player, all tile states (owner + troops), supply chains, turn, phase, winner
- `GameRecord` wraps: config, seed, bot names, initial state, frame list, interestingness score
- JSON persistence with `HexCoord → (col, row)` tuple serialization

### Replay Viewer
- Load JSON → reconstruct snapshots → step through with pygame
- Play/pause, speed control (0.25× to 8×), frame-by-frame navigation
- Reuses `GameRenderer` for consistent visuals

### Interestingness Score
- Rewards: close games, lead changes, moderate length
- Penalizes: blowouts, draws, very short/long games
- Used to select the "best" replays from tournament runs

**What I learned:**
- Storing full tile state per frame (not just deltas) makes replay reconstruction trivial at the cost of slightly larger files. For ~30 turns × ~100 tiles, this is negligible.
- The interestingness score was a nice-to-have that became essential for filtering thousands of bot tournament games down to the few worth watching.
- Reusing the same `GameRenderer` class for both live play and replay was a huge win — no duplicate rendering code.

---

## 21. Testing Strategy

**What's tested:**
- `hex_core`: Coordinate creation, validation (odd+even should fail), neighbor lookup, pixel conversion round-trips
- `hex_grid`: Grid construction, terrain assignment, neighbor filtering at boundaries
- `pathfinding`: A* correctness on known paths, terrain cost application, no-path-exists cases
- `main_utils`: Pixel-to-hex rounding, hexagon vertex calculation
- `flat_env`: Space dimensions for each preset, mask validity (every True mask → valid execution), encode/decode round-trips, full game rollouts with bot adapters, observation spot-checks

**Patterns used:**
- Pytest fixtures for reusable environment setup
- Parametrized tests across presets (SMALL, MEDIUM, LARGE)
- Round-trip tests: encode an action → decode → compare to original
- "Smoke test" full game rollouts: run entire games to check for crashes
- Spot-check observations against known state (SMALL_FIXED map)

**What's NOT tested (and why):**
- Rendering: visual output is hard to test automatically, validated by eye
- Interactive client (`play.py`): requires pygame display, tested manually
- Combat probabilities: the RNG makes exact outcome testing fragile; tested indirectly through bot games

**What I learned:**
- The SMALL_FIXED preset was built specifically for testing. Known coordinates, known terrain layout, known spawn points — makes assertions concrete.
- Full game rollout tests (bot vs bot to completion) catch integration bugs that unit tests miss. "The bot tried to move to an invalid tile on turn 15" only shows up in full games.
- Testing action masking was the most valuable flat_env test: "for every action index where the mask says True, executing that action should not produce an engine error." This is an invariant that must always hold.

---

## 22. Type System & Python Patterns

**Choices I made consistently:**

### Frozen Dataclasses Everywhere
- `HexCoord`, `HexTile`, `PathResult`, `MoveAction`, `EndTurnAction`, `SetupSupplyChainAction`, `CombatResult`, `SupplyChain`
- **Why:** Immutable data is easier to reason about. Frozen = hashable = usable as dict keys and set members. `slots=True` on `HexCoord` for minor memory/speed benefit.

### Enums for Closed Sets
- `Terrain` (PLAINS, MOUNTAIN, FERTILE)
- `GamePhase` (SETUP, PLAYING, GAME_OVER)
- `MapPreset` (SMALL, SMALL_FIXED, MEDIUM, LARGE)
- **Why:** Type-safe, auto-complete friendly, impossible to misspell.

### Protocols for Interfaces
- `CombatResolver` protocol: any class with `resolve()` method
- `Bot` protocol: any class with `choose_action()` method
- **Why:** Structural typing (duck typing with type checker support). No inheritance hierarchy needed.

### `from __future__ import annotations`
- Used in every file for PEP 604 union syntax (`int | None`) on Python 3.9+
- **Why:** Cleaner type hints without runtime eval overhead.

### Type Hints Throughout
- All function signatures are annotated
- No `Any` except in JSON serialization contexts
- **Why:** Catches bugs at edit time (with Pylance/mypy), serves as documentation.

**What I learned:**
- Protocols are underused in Python. They're perfect for game engines where you want pluggable components without inheritance.
- `slots=True` on frozen dataclasses is free performance. Always use it for small, frequently-created objects.
- The `from __future__ import annotations` import is always worth it — forward references and modern syntax with zero runtime cost.

---

## 23. Project Architecture & Separation of Concerns

**Layering:**
```
hex_core.py          ← Pure math. No game logic. No dependencies.
hex_grid.py          ← Grid container. Depends on hex_core only.
map_generator.py     ← Terrain gen. Depends on hex_grid.
pathfinding.py       ← A*. Depends on hex_core, hex_grid.
renderer.py          ← Base rendering. Depends on hex_core, hex_grid, pygame.

game/
  config.py          ← Configuration. No game logic dependencies.
  state.py           ← Data containers. Depends on hex_core, hex_grid.
  actions.py         ← Action types + validation. Depends on state, hex_core.
  combat.py          ← Combat resolution. Self-contained + math.
  engine.py          ← Orchestrator. Depends on everything above.
  game_renderer.py   ← Game rendering. Depends on renderer, state, config.
  environment.py     ← Gym wrapper v1. Depends on engine.
  flat_env.py        ← Gym wrapper v2. Depends on engine, combat.
  bots.py            ← Rule-based bots. Depends on state, actions, combat.
  recorder.py        ← Recording. Depends on state, config.
```

**Design principles followed:**
1. **Lower layers don't know about higher layers.** hex_core doesn't know games exist. hex_grid doesn't know about combat.
2. **Game state is passive data.** Engine is the active controller.
3. **Rendering is leaf-node.** Nothing depends on the renderer.
4. **Config is root-level.** Everything reads from config, nothing writes to it.
5. **Environments wrap the engine.** They translate between RL interfaces and game actions.

**What I learned:**
- This layering wasn't designed upfront — it emerged from refactoring. Initially hex_core and hex_grid were one file, and game state had mutation methods. Splitting them made each piece independently testable.
- The root-level modules (hex_core, hex_grid, etc.) vs the game/ package split maps to "reusable hex infrastructure" vs "HexWar-specific logic." If I built a different hex game, the root modules could be reused as-is.
- Keeping rendering as a leaf dependency (nothing imports renderer) prevents the classic "UI coupled to logic" problem. I can run full games headlessly (in tests, in RL training) without ever importing pygame.

---

## 24. What I'd Do Differently

Things I learned the hard way that I'd change if starting over:

1. **Start with the fixed map immediately.** I spent time debugging on procedural maps where I couldn't predict tile positions. The SMALL_FIXED preset should have been day-one infrastructure.

2. **Design the flat observation space first.** The 2D multi-channel env (v1) was a natural first thought but turned out to be a dead end for MLP-based RL. If I'd thought about the agent architecture earlier, I'd have gone straight to flat vectors with neighborhood features.

3. **Action masking from the start.** v1 had no masking. Agents wasting training time on invalid actions was predictable and avoidable.

4. **Formalize the Bot protocol earlier.** I wrote GreedyBot before formalizing the protocol, then had to refactor it to match. Protocol-first design would have saved time.

5. **Record everything from day one.** The recording system came late but was invaluable for debugging bot behavior. Earlier recording would have caught bugs faster.

---

*This document captures the design journey of HexWar through Phase 1 (Environment Engineering). As the project progresses through RL training (Phase 2–4), new decisions around network architecture, training curricula, and self-play will build on these foundations.*
