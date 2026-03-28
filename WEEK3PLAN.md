 Plan: Flat Observation + Multi-Discrete Full-Turn Action Space with Action Masking

 Context

 The existing game/environment.py has a 2D multi-channel observation and dict-based single-action-per-step API. For RL training, we need:
 - Flat 1D observation vector
 - Full-turn action output (all moves + supply chains at once) so the agent learns coordinated strategy
 - Action masking to guide exploration

 The agent outputs its entire turn plan as a single vector, forcing it to reason about multi-move coordination (pincer attacks, supply chain setup before pushes, etc.).        

 Key Design Decisions

 1. Full turn per step — step() accepts a vector of [move_0, ..., move_{M-1}, sc_0, sc_1] and executes them sequentially. This captures planning/coordination that
 single-action steps cannot.
 2. Skip invalid + small penalty — actions that become invalid mid-execution (e.g., tile depleted by an earlier move) are skipped with a -0.01 penalty per skip. Agent learns   
 to avoid wasted slots.
 3. Config-dependent vector size — moves_per_turn move slots + 2 supply chain slots. Matches actual game rules per preset.
 4. Multi-Discrete action space — each slot is an independent categorical. Move slots choose from [0..N_TILES*6*5] (0=no-op), SC slots from [0..N_TILES*6] (0=no-op).
 5. Masks based on pre-turn state — since we can't predict how earlier slots affect later ones, all slot masks reflect the state at turn start. Sequential execution + skip     
 handles the rest.

 Space Dimensions

 Observation (1D float32 vector)

 Per tile (canonical sorted order via _coord_list):

 ┌───────────────────┬───────────────┬───────────────────────────────────────────┐
 │      Feature      │     Size      │                 Encoding                  │
 ├───────────────────┼───────────────┼───────────────────────────────────────────┤
 │ terrain           │ 3             │ one-hot [plains, mountain, fertile]       │
 ├───────────────────┼───────────────┼───────────────────────────────────────────┤
 │ ownership         │ num_players+1 │ one-hot [neutral, p0, p1, ...]            │
 ├───────────────────┼───────────────┼───────────────────────────────────────────┤
 │ troop_count       │ 1             │ troops / (troops + 50) soft normalization │
 ├───────────────────┼───────────────┼───────────────────────────────────────────┤
 │ is_current_player │ 1             │ 1.0 if owned by acting player             │
 ├───────────────────┼───────────────┼───────────────────────────────────────────┤
 │ visibility        │ 1             │ 1.0 always (fog placeholder)              │
 ├───────────────────┼───────────────┼───────────────────────────────────────────┤
 │ supply_chain_out  │ 1             │ 1.0 if tile is supply chain source        │
 ├───────────────────┼───────────────┼───────────────────────────────────────────┤
 │ supply_chain_in   │ 1             │ 1.0 if tile is supply chain dest          │
 └───────────────────┴───────────────┴───────────────────────────────────────────┘

 Per-tile width = 9 + num_players

 Global features (appended):

 ┌──────────────────────────────────────┬─────────────┐
 │               Feature                │    Size     │
 ├──────────────────────────────────────┼─────────────┤
 │ current_turn / max_turns             │ 1           │
 ├──────────────────────────────────────┼─────────────┤
 │ moves_remaining / moves_per_turn     │ 1           │
 ├──────────────────────────────────────┼─────────────┤
 │ supply_chains_remaining / 2          │ 1           │
 ├──────────────────────────────────────┼─────────────┤
 │ territory_fraction per player        │ num_players │
 ├──────────────────────────────────────┼─────────────┤
 │ total_troops per player (normalized) │ num_players │
 └──────────────────────────────────────┴─────────────┘

 Total obs = N_TILES * (9 + num_players) + 3 + 2 * num_players
 - SMALL 2p: 48×11 + 7 = 535
 - MEDIUM 2p: 140×11 + 7 = 1547

 Action (integer vector, length = moves_per_turn + 2)

 Each move slot: integer in [0, N_TILES * 6 * 5]
 - 0 = no-op (skip this slot)
 - 1..N = encoded as (tile_idx * 6 * 5) + (dir * 5) + troop_bin + 1
 - 6 directions (hex neighbors), 5 troop bins [25%, 50%, 75%, 100%, all-but-1]

 Each SC slot: integer in [0, N_TILES * 6]
 - 0 = no-op
 - 1..N = encoded as (tile_idx * 6) + neighbor_dir + 1

 Sizes per preset (2 players):

 ┌────────┬────────────────┬──────────────┬───────────────┐
 │ Preset │ Move slot size │ SC slot size │ Vector length │
 ├────────┼────────────────┼──────────────┼───────────────┤
 │ SMALL  │ 1441           │ 289          │ 6 (4+2)       │
 ├────────┼────────────────┼──────────────┼───────────────┤
 │ MEDIUM │ 4201           │ 841          │ 8 (6+2)       │
 ├────────┼────────────────┼──────────────┼───────────────┤
 │ LARGE  │ 8401           │ 1681         │ 10 (8+2)      │
 └────────┴────────────────┴──────────────┴───────────────┘

 Action Mask

 One mask array per slot, returned as info["action_masks"] — a list of bool arrays.
 - Move slot mask: shape (move_slot_size,) — index 0 (no-op) always True
 - SC slot mask: shape (sc_slot_size,) — index 0 (no-op) always True
 - All masks computed from pre-turn state

 Implementation Steps

 Step 1: Constants + FlatHexWarEnv.__init__

 File: game/flat_env.py (new)

 - TROOP_FRACTIONS = [0.25, 0.5, 0.75, 1.0] + all-but-1 flag (index 4)
 - Compute n_tiles, move_slot_size, sc_slot_size, action_vector_length from config
 - Store obs_size
 - Init GameEngine (reuse existing)

 Step 2: reset() with neighbor lookup table

 - Delegate to engine.reset() (handles auto-placement)
 - Build _coord_list, _coord_to_idx (same pattern as existing environment.py)
 - Precompute _neighbor_table: np.ndarray shape (n_tiles, 6) — [tile_idx, dir] -> neighbor_tile_idx or -1
 - Return (flat_obs, info) with info["action_masks"]

 Step 3: _build_flat_observation()

 - Iterate _coord_list, encode per-tile features into pre-allocated array
 - Append global features
 - Return 1D float32 array

 Step 4: _build_action_masks() — returns list of mask arrays

 Move slot mask (same mask used for all move slots):
 - For each (tile_idx, dir, troop_bin) combo, True if:
   - Tile owned by current player AND has ≥ 2 troops
   - _neighbor_table[tile_idx, dir] != -1 (neighbor exists)
 - No-op (index 0) always True
 - Use _neighbor_table for O(1) lookups

 SC slot mask (same mask for both SC slots):
 - For each (tile_idx, neighbor_dir), True if:
   - Tile owned by current player
   - Neighbor exists and owned by current player
   - No existing outgoing chain from source
   - supply_chains_set_this_turn < 2
 - No-op always True
 - Cycle check: call existing _would_create_cycle() from game/actions.py

 Step 5: _decode_move(action_int) and _decode_supply_chain(action_int)

 - Move: action_int - 1 → tile_idx, dir, troop_bin → MoveAction
 - Troop decode: troops = max(1, int(TROOP_FRACTIONS[bin] * (src_troops - 1))), bin 4 → src_troops - 1
 - SC: action_int - 1 → tile_idx, neighbor_dir → SetupSupplyChainAction
 - 0 → None (no-op)

 Step 6: step(action_vector: np.ndarray)

 Core logic:
 penalty = 0.0
 prev_territory = state.territory_count(acting_player)

 # Execute supply chains first (order: SC slots, then move slots)
 for sc_action_int in action_vector[moves_per_turn:]:
     if sc_action_int == 0: continue
     game_action = _decode_supply_chain(sc_action_int)
     err = engine.execute_action(game_action)
     if err: penalty -= 0.01

 # Execute moves
 for move_action_int in action_vector[:moves_per_turn]:
     if move_action_int == 0: continue
     game_action = _decode_move(move_action_int)
     err = engine.execute_action(game_action)
     if err: penalty -= 0.01

 # End turn
 engine.execute_action(EndTurnAction())

 # Reward
 new_territory = state.territory_count(acting_player)
 reward = float(new_territory - prev_territory) + penalty
 if state.phase == GamePhase.GAME_OVER and state.winner is not None:
     reward += 10.0 if state.winner == acting_player else -10.0

 Return (obs, reward, done, truncated, info) with fresh masks.

 Note: Supply chains execute first so the agent can set up logistics before attacking.

 Step 7: encode_action(game_actions) + BotFlatAdapter

 - encode_action() takes a list of game action objects, returns action vector
 - BotFlatAdapter wraps a Bot: calls bot.choose_action() repeatedly (up to moves_per_turn times or until EndTurnAction), collects actions, encodes into vector

 Step 8: Tests (tests/test_flat_env.py)

 1. Space size correctness for each preset
 2. Mask validity: every True-masked action in slot 0 executes without engine error
 3. Encode/decode round-trip for move and SC actions
 4. Full game: BotFlatAdapter(RandomBot) vs BotFlatAdapter(GreedyExpansionBot) completes without crash
 5. Observation spot-check on SMALL_FIXED
 6. Skip penalty: craft a turn where move 1 depletes a tile, move 2 tries the same tile → verify -0.01 penalty

 Step 9 (optional): Gymnasium wrapper

 - If gymnasium installed, use MultiDiscrete action space and Box observation space

 Files

 Create:
 - game/flat_env.py (~350 lines) — FlatHexWarEnv, BotFlatAdapter, constants
 - tests/test_flat_env.py (~150 lines)

 Modify:
 - game/__init__.py — add FlatHexWarEnv, BotFlatAdapter to exports

 Existing Code to Reuse

 - game/engine.py: GameEngine — wrap directly, use execute_action() for sequential execution
 - game/actions.py: _would_create_cycle() for SC mask cycle check, validate_move/validate_supply_chain used internally by engine
 - hex_core.py: DIRECTIONS for neighbor table construction
 - game/environment.py: reference for coord list setup, reward pattern, info dict

 Verification

 1. pytest tests/test_flat_env.py — all pass
 2. Manual: create SMALL_FIXED env, reset(), print obs shape (535,) and mask shapes
 3. Full game rollout with bot adapters, verify completion
 4. Verify penalty accumulates when crafting intentionally conflicting moves
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌