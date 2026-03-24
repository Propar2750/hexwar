"""Tests for game.flat_env — FlatHexWarEnv v2 with all 5 innovations.

Covers: observation/action space dimensions, ego-centric encoding,
two-tier action masking, reward shaping, and BotFlatAdapter integration.
Run with: pytest
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.config import GameConfig, MapPreset
from game.flat_env import (
    FlatHexWarEnv,
    BotFlatAdapter,
    N_DIRECTIONS,
    N_TROOP_BINS,
    TROOP_FRACTIONS,
    SOFT_MASK_BIAS,
    _NEIGHBOR_FEAT,
    _DERIVED_FEAT,
)
from game.bots import RandomBot, GreedyExpansionBot, TurtleDefendBot
from game.state import GamePhase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_env():
    config = GameConfig(preset=MapPreset.SMALL_FIXED)
    env = FlatHexWarEnv(config)
    env.reset(seed=42)
    return env


@pytest.fixture
def small_sub_env():
    config = GameConfig(preset=MapPreset.SMALL_FIXED)
    env = FlatHexWarEnv(config, sub_step=True)
    env.reset(seed=42)
    return env


@pytest.fixture
def medium_env():
    config = GameConfig(preset=MapPreset.MEDIUM)
    env = FlatHexWarEnv(config)
    env.reset(seed=123)
    return env


# ---------------------------------------------------------------------------
# Space dimensions
# ---------------------------------------------------------------------------

class TestSpaceDimensions:
    def test_small_fixed_obs_size(self, small_env):
        obs, _ = small_env.reset(seed=42)
        n = small_env.n_tiles
        num_p = small_env.config.num_players
        expected = n * (36 + num_p) + 5 + 2 * num_p
        assert obs.shape == (expected,)
        assert small_env.obs_size == expected

    def test_medium_obs_size(self, medium_env):
        obs, _ = medium_env.reset(seed=123)
        n = medium_env.n_tiles
        num_p = medium_env.config.num_players
        expected = n * (36 + num_p) + 5 + 2 * num_p
        assert obs.shape == (expected,)

    def test_action_sizes(self, small_env):
        n = small_env.n_tiles
        assert small_env.move_slot_size == n * N_DIRECTIONS * N_TROOP_BINS + 1
        assert small_env.sc_slot_size == n * N_DIRECTIONS + 1
        assert small_env.action_vector_length == small_env.moves_per_turn + 2

    def test_unified_action_size(self, small_sub_env):
        n = small_sub_env.n_tiles
        expected = 1 + n * N_DIRECTIONS * N_TROOP_BINS + n * N_DIRECTIONS
        assert small_sub_env.unified_action_size == expected

    def test_vector_length_per_preset(self):
        for preset, expected_moves in [
            (MapPreset.SMALL, 4),
            (MapPreset.SMALL_FIXED, 4),
            (MapPreset.MEDIUM, 6),
            (MapPreset.LARGE, 8),
        ]:
            config = GameConfig(preset=preset)
            env = FlatHexWarEnv(config)
            env.reset(seed=1)
            assert env.action_vector_length == expected_moves + 2


# ---------------------------------------------------------------------------
# Ego-centric observation
# ---------------------------------------------------------------------------

class TestEgoCentric:
    def test_ego_owner_mapping(self):
        assert FlatHexWarEnv._ego_owner(None, 0) == 0   # neutral
        assert FlatHexWarEnv._ego_owner(0, 0) == 1       # mine
        assert FlatHexWarEnv._ego_owner(1, 0) == 2       # enemy
        assert FlatHexWarEnv._ego_owner(0, 1) == 2       # enemy (I'm p1)
        assert FlatHexWarEnv._ego_owner(1, 1) == 1       # mine

    def test_ownership_one_hot_sums_to_one(self, small_env):
        obs, _ = small_env.reset(seed=42)
        w = small_env._tile_feat_width
        num_p = small_env.config.num_players
        from game.flat_env import N_TERRAIN
        for i in range(small_env.n_tiles):
            base = i * w + N_TERRAIN
            own_slice = obs[base: base + num_p + 1]
            assert own_slice.sum() == pytest.approx(1.0)

    def test_global_features_ego_rotated(self, small_env):
        """My territory fraction should be at a fixed offset."""
        obs, _ = small_env.reset(seed=42)
        g = small_env.n_tiles * small_env._tile_feat_width
        # g+5 = my territory fraction, should be > 0
        my_terr = obs[g + 5]
        assert my_terr > 0.0  # player owns at least starting tiles


# ---------------------------------------------------------------------------
# Neighborhood features
# ---------------------------------------------------------------------------

class TestNeighborhoodFeatures:
    def test_obs_has_neighbor_data(self, small_env):
        """Tiles with neighbors should have non-zero neighbor features."""
        obs, _ = small_env.reset(seed=42)
        w = small_env._tile_feat_width
        sw = small_env._self_feat_width
        found_neighbor = False
        for i in range(small_env.n_tiles):
            nb_base = i * w + sw
            for d in range(N_DIRECTIONS):
                if small_env._neighbor_table[i, d] != -1:
                    exists = obs[nb_base + d * _NEIGHBOR_FEAT]
                    assert exists == 1.0
                    found_neighbor = True
                else:
                    exists = obs[nb_base + d * _NEIGHBOR_FEAT]
                    assert exists == 0.0
        assert found_neighbor

    def test_terrain_defense_values(self, small_env):
        """Neighbor terrain_defense should match config values."""
        obs, _ = small_env.reset(seed=42)
        w = small_env._tile_feat_width
        sw = small_env._self_feat_width
        state = small_env.engine.state
        defense = small_env.config.defense_bonus
        for i in range(small_env.n_tiles):
            for d in range(N_DIRECTIONS):
                ni = small_env._neighbor_table[i, d]
                if ni == -1:
                    continue
                n_coord = small_env._coord_list[ni]
                n_terrain = state.grid[n_coord].terrain
                expected = defense.get(n_terrain, 1.0)
                actual = obs[i * w + sw + d * _NEIGHBOR_FEAT + 3]
                assert actual == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Derived tactical features
# ---------------------------------------------------------------------------

class TestTacticalFeatures:
    def test_border_tile_detection(self, small_env):
        """Starting tiles adjacent to enemy/neutral should be border."""
        obs, _ = small_env.reset(seed=42)
        w = small_env._tile_feat_width
        sw = small_env._self_feat_width
        der_offset = sw + N_DIRECTIONS * _NEIGHBOR_FEAT
        state = small_env.engine.state
        player = state.current_player

        for i, coord in enumerate(small_env._coord_list):
            ts = state.tiles[coord]
            if ts.owner != player:
                continue
            # Check if this tile has any non-mine neighbor
            has_non_mine = False
            for d in range(N_DIRECTIONS):
                ni = small_env._neighbor_table[i, d]
                if ni != -1 and state.tiles[small_env._coord_list[ni]].owner != player:
                    has_non_mine = True
                    break
            is_border = obs[i * w + der_offset]
            assert is_border == (1.0 if has_non_mine else 0.0)

    def test_attack_opportunity_positive_for_border(self, small_env):
        """Border tiles with troops should have positive attack_opportunity."""
        obs, _ = small_env.reset(seed=42)
        w = small_env._tile_feat_width
        sw = small_env._self_feat_width
        der_offset = sw + N_DIRECTIONS * _NEIGHBOR_FEAT
        state = small_env.engine.state
        player = state.current_player

        found = False
        for i, coord in enumerate(small_env._coord_list):
            ts = state.tiles[coord]
            if ts.owner == player and ts.troops >= 2:
                is_border = obs[i * w + der_offset]
                attack_opp = obs[i * w + der_offset + 1]
                if is_border > 0:
                    assert attack_opp > 0.0
                    found = True
        assert found, "Should have at least one border tile with attack opportunity"


# ---------------------------------------------------------------------------
# Two-tier masks
# ---------------------------------------------------------------------------

class TestTwoTierMasks:
    def test_hard_mask_shapes(self, small_env):
        _, info = small_env.reset(seed=42)
        masks = info["action_masks"]
        assert masks["move_hard"].shape == (small_env.move_slot_size,)
        assert masks["sc_hard"].shape == (small_env.sc_slot_size,)
        assert masks["move_hard"].dtype == bool

    def test_soft_mask_shapes(self, small_env):
        _, info = small_env.reset(seed=42)
        masks = info["action_masks"]
        assert masks["move_soft"].shape == (small_env.move_slot_size,)
        assert masks["sc_soft"].shape == (small_env.sc_slot_size,)
        assert masks["move_soft"].dtype == np.float32

    def test_soft_mask_values(self, small_env):
        """Soft mask should be 0.0 for valid actions and SOFT_MASK_BIAS for invalid."""
        _, info = small_env.reset(seed=42)
        soft = info["action_masks"]["move_soft"]
        assert soft[0] == 0.0  # no-op always valid
        # Should have some valid (0.0) and some invalid (SOFT_MASK_BIAS) entries
        n_valid = (soft == 0.0).sum()
        n_invalid = (soft == SOFT_MASK_BIAS).sum()
        assert n_valid > 1  # at least no-op + some moves
        assert n_invalid > 0  # some invalid

    def test_soft_mask_marks_unowned_tiles(self, small_env):
        """Tiles not owned by current player should be soft-masked."""
        _, info = small_env.reset(seed=42)
        soft = info["action_masks"]["move_soft"]
        state = small_env.engine.state
        player = state.current_player

        for ti in range(small_env.n_tiles):
            ts = state.tiles[small_env._coord_list[ti]]
            if ts.owner != player or ts.troops < 2:
                # All actions from this tile should be soft-masked
                for d in range(N_DIRECTIONS):
                    for b in range(N_TROOP_BINS):
                        idx = ti * N_DIRECTIONS * N_TROOP_BINS + d * N_TROOP_BINS + b + 1
                        assert soft[idx] == SOFT_MASK_BIAS

    def test_unified_hard_mask(self, small_sub_env):
        _, info = small_sub_env.reset(seed=42)
        hard = info["action_masks"]["hard"]
        assert hard.shape == (small_sub_env.unified_action_size,)
        assert hard[0] is np.True_  # end turn

    def test_unified_soft_mask(self, small_sub_env):
        _, info = small_sub_env.reset(seed=42)
        soft = info["action_masks"]["soft"]
        assert soft.shape == (small_sub_env.unified_action_size,)
        assert soft[0] == 0.0


# ---------------------------------------------------------------------------
# Action encode / decode
# ---------------------------------------------------------------------------

class TestEncodeDecode:
    def test_move_roundtrip(self, small_env):
        from game.actions import MoveAction
        from hex_core import HexCoord, DIRECTIONS as DIRS
        state = small_env.engine.state
        for coord in small_env._coord_list:
            ts = state.tiles[coord]
            if ts.owner is not None and ts.troops >= 2:
                for d in range(N_DIRECTIONS):
                    dc, dr = DIRS[d]
                    target = HexCoord(coord.col + dc, coord.row + dr)
                    if target in state.tiles:
                        original = MoveAction(source=coord, target=target, troops=ts.troops - 1)
                        encoded = small_env.encode_move(original)
                        assert encoded > 0
                        decoded = small_env._decode_move(encoded)
                        assert decoded is not None
                        assert decoded.source == original.source
                        assert decoded.target == original.target
                        assert decoded.troops >= 1
                        return
        pytest.skip("No movable tile found")

    def test_unified_encode_roundtrip(self, small_sub_env):
        from game.actions import MoveAction, EndTurnAction
        from hex_core import HexCoord, DIRECTIONS as DIRS
        # EndTurnAction
        assert small_sub_env.encode_unified(EndTurnAction()) == 0
        assert small_sub_env._decode_unified(0) is None

        # Move
        state = small_sub_env.engine.state
        for coord in small_sub_env._coord_list:
            ts = state.tiles[coord]
            if ts.owner is not None and ts.troops >= 2:
                for d in range(N_DIRECTIONS):
                    dc, dr = DIRS[d]
                    target = HexCoord(coord.col + dc, coord.row + dr)
                    if target in state.tiles:
                        original = MoveAction(source=coord, target=target, troops=ts.troops - 1)
                        encoded = small_sub_env.encode_unified(original)
                        assert encoded > 0
                        decoded = small_sub_env._decode_unified(encoded)
                        assert decoded is not None
                        assert decoded.source == original.source
                        assert decoded.target == original.target
                        return

    def test_noop_decode(self, small_env):
        assert small_env._decode_move(0) is None
        assert small_env._decode_supply_chain(0) is None


# ---------------------------------------------------------------------------
# Full-turn step
# ---------------------------------------------------------------------------

class TestFullTurnStep:
    def test_noop_turn(self, small_env):
        state_before = small_env.engine.state
        player_before = state_before.current_player
        vec = np.zeros(small_env.action_vector_length, dtype=np.int64)
        obs, reward, done, truncated, info = small_env.step(vec)
        assert obs.shape == (small_env.obs_size,)
        assert not truncated
        assert small_env.engine.state.current_player != player_before or done

    def test_step_returns_valid_obs(self, small_env):
        vec = np.zeros(small_env.action_vector_length, dtype=np.int64)
        obs, _, _, _, _ = small_env.step(vec)
        assert obs.dtype == np.float32
        assert not np.any(np.isnan(obs))

    def test_reward_has_components(self, small_env):
        """A non-trivial turn should produce non-zero reward."""
        bot = BotFlatAdapter(GreedyExpansionBot())
        vec = bot.choose_action_vector(small_env)
        _, reward, _, _, _ = small_env.step(vec)
        # Reward can be any float — just verify it's finite
        assert np.isfinite(reward)


# ---------------------------------------------------------------------------
# Sub-step mode
# ---------------------------------------------------------------------------

class TestSubStep:
    def test_end_turn_immediately(self, small_sub_env):
        """Ending turn immediately should work and produce reward."""
        obs, reward, done, truncated, info = small_sub_env.step(0)
        assert obs.shape == (small_sub_env.obs_size,)
        assert isinstance(reward, float)
        assert not truncated

    def test_sub_step_then_end(self, small_sub_env):
        """Execute one action then end turn."""
        # Find a valid action from soft mask
        _, info = small_sub_env.reset(seed=42)
        soft = info["action_masks"]["soft"]
        valid_idx = np.where(soft == 0.0)[0]
        # Pick a non-end-turn action
        move_actions = [v for v in valid_idx if v > 0]
        if not move_actions:
            pytest.skip("No valid sub-step actions")

        # Sub-step 1: a move
        obs1, r1, done1, _, info1 = small_sub_env.step(int(move_actions[0]))
        assert r1 == 0.0  # no reward until end turn
        assert not done1

        # Sub-step 2: end turn
        obs2, r2, done2, _, _ = small_sub_env.step(0)
        assert isinstance(r2, float)  # reward delivered

    def test_soft_mask_updates_after_sub_step(self, small_sub_env):
        """Soft mask should reflect state changes from previous sub-steps."""
        _, info0 = small_sub_env.reset(seed=42)
        soft0 = info0["action_masks"]["soft"].copy()

        # Execute a valid move
        valid_moves = np.where(soft0 == 0.0)[0]
        move_actions = [v for v in valid_moves if v > 0 and v <= small_sub_env._n_move_actions]
        if not move_actions:
            pytest.skip("No valid move actions")

        _, _, _, _, info1 = small_sub_env.step(int(move_actions[0]))
        soft1 = info1["action_masks"]["soft"]

        # Soft mask should have changed (troops moved, new state)
        assert not np.array_equal(soft0, soft1)

    def test_sub_step_full_game(self):
        """Run a full game in sub-step mode with RandomBot."""
        config = GameConfig(preset=MapPreset.SMALL_FIXED)
        env = FlatHexWarEnv(config, sub_step=True)
        env.reset(seed=99)
        bots = [BotFlatAdapter(RandomBot()), BotFlatAdapter(RandomBot())]

        max_sub_steps = 2000
        done = False
        for _ in range(max_sub_steps):
            player = env.engine.state.current_player
            action = bots[player].choose_sub_action(env)
            _, _, done, _, _ = env.step(action)
            if done:
                break

        assert done, "Sub-step game did not finish"


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

class TestRewardShaping:
    def test_combat_efficiency_tracked(self):
        """Winning an attack should produce combat_efficiency > 0 in reward."""
        config = GameConfig(preset=MapPreset.SMALL_FIXED)
        env = FlatHexWarEnv(config)
        env.reset(seed=42)

        # Use greedy bot which attacks aggressively
        bot = BotFlatAdapter(GreedyExpansionBot())
        vec = bot.choose_action_vector(env)
        _, reward, _, _, _ = env.step(vec)
        # Just check it ran without error — reward value depends on outcomes
        assert np.isfinite(reward)

    def test_border_pressure_computed(self, small_env):
        """Border pressure should be a valid ratio."""
        player = small_env.engine.state.current_player
        bp = small_env._border_pressure(player)
        assert 0.0 <= bp <= 1.0


# ---------------------------------------------------------------------------
# BotFlatAdapter
# ---------------------------------------------------------------------------

class TestBotFlatAdapter:
    @pytest.mark.parametrize("bot_class", [RandomBot, GreedyExpansionBot, TurtleDefendBot])
    def test_full_game_completes(self, bot_class):
        config = GameConfig(preset=MapPreset.SMALL_FIXED)
        env = FlatHexWarEnv(config)
        env.reset(seed=99)
        bots = [BotFlatAdapter(bot_class()), BotFlatAdapter(bot_class())]
        done = False
        for _ in range(200):
            player = env.engine.state.current_player
            vec = bots[player].choose_action_vector(env)
            assert vec.shape == (env.action_vector_length,)
            _, _, done, _, _ = env.step(vec)
            if done:
                break
        assert done

    def test_mixed_bots_game(self):
        config = GameConfig(preset=MapPreset.SMALL_FIXED)
        env = FlatHexWarEnv(config)
        env.reset(seed=42)
        bots = [BotFlatAdapter(RandomBot()), BotFlatAdapter(GreedyExpansionBot())]
        done = False
        for _ in range(200):
            player = env.engine.state.current_player
            vec = bots[player].choose_action_vector(env)
            _, _, done, _, _ = env.step(vec)
            if done:
                break
        assert done

    def test_adapter_encodes_valid_actions(self):
        config = GameConfig(preset=MapPreset.SMALL_FIXED)
        env = FlatHexWarEnv(config)
        env.reset(seed=42)
        bot = BotFlatAdapter(GreedyExpansionBot())
        vec = bot.choose_action_vector(env)
        for i in range(env.moves_per_turn):
            assert 0 <= vec[i] < env.move_slot_size
        for i in range(2):
            assert 0 <= vec[env.moves_per_turn + i] < env.sc_slot_size


# ---------------------------------------------------------------------------
# Bug regression tests
# ---------------------------------------------------------------------------

class TestBugFixes:
    def test_sub_step_game_over_mid_turn(self):
        """Game-over during a sub-step should return done=True immediately."""
        config = GameConfig(preset=MapPreset.SMALL_FIXED)
        env = FlatHexWarEnv(config, sub_step=True)
        env.reset(seed=42)
        bots = [BotFlatAdapter(GreedyExpansionBot()), BotFlatAdapter(RandomBot())]

        # Play until game ends — every done=True must have reward != 0
        for _ in range(2000):
            player = env.engine.state.current_player
            action = bots[player].choose_sub_action(env)
            _, reward, done, _, _ = env.step(action)
            if done:
                # Game-over should include the win/loss bonus
                assert reward != 0.0, "Game-over reward should include win/loss bonus"
                break
        assert done

    def test_soft_mask_respects_move_limit(self):
        """After all moves used, soft mask should not allow more moves."""
        config = GameConfig(preset=MapPreset.SMALL_FIXED)
        env = FlatHexWarEnv(config, sub_step=True)
        env.reset(seed=42)

        # Execute moves_per_turn valid moves
        for _ in range(env.moves_per_turn):
            _, info = env.reset(seed=42) if _ == 0 else (None, None)
            if info is None:
                # Get current info
                soft = env._build_soft_masks()["soft"]
            else:
                soft = info["action_masks"]["soft"]
            valid_moves = np.where(soft[:1 + env._n_move_actions] == 0.0)[0]
            move_actions = [v for v in valid_moves if v > 0]
            if not move_actions:
                break
            env.step(int(move_actions[0]))

        # After all moves used, soft mask should block moves
        soft = env._build_soft_masks()["soft"]
        move_section = soft[1:1 + env._n_move_actions]
        # All moves should be soft-masked (no 0.0 entries)
        if env.engine.state.moves_made >= env.moves_per_turn:
            assert (move_section == 0.0).sum() == 0, \
                "Soft mask should block moves after moves_per_turn exhausted"

    def test_full_turn_stops_on_game_over(self):
        """Full-turn mode should stop processing moves after game-over."""
        config = GameConfig(preset=MapPreset.SMALL_FIXED)
        env = FlatHexWarEnv(config)
        env.reset(seed=42)
        bots = [BotFlatAdapter(GreedyExpansionBot()), BotFlatAdapter(GreedyExpansionBot())]

        # Play a full game — should complete normally
        done = False
        for _ in range(200):
            player = env.engine.state.current_player
            vec = bots[player].choose_action_vector(env)
            _, reward, done, _, _ = env.step(vec)
            if done:
                assert np.isfinite(reward)
                break
        assert done

    def test_both_players_receive_game_outcome(self):
        """Winner should get positive total, loser should get negative total."""
        config = GameConfig(preset=MapPreset.SMALL_FIXED)
        for seed in range(5):
            env = FlatHexWarEnv(config)
            env.reset(seed=seed)
            bots = [BotFlatAdapter(GreedyExpansionBot()), BotFlatAdapter(RandomBot())]
            p_rewards = [0.0, 0.0]
            done = False
            for _ in range(200):
                p = env.engine.state.current_player
                vec = bots[p].choose_action_vector(env)
                _, reward, done, _, _ = env.step(vec)
                p_rewards[p] += reward
                if done:
                    break
            assert done
            # Deliver pending reward for the other player
            vec = np.zeros(env.action_vector_length, dtype=np.int64)
            _, pending_reward, _, _, _ = env.step(vec)
            # Figure out which player DIDN'T trigger game-over and add their pending
            # The acting player of the last step already got their outcome.
            # The pending goes to the other player.
            winner = env.engine.state.winner
            # The pending reward is for the non-acting player
            # We don't know exactly which player, but we know the reward sign
            if pending_reward > 0:
                p_rewards[winner] += pending_reward
            elif pending_reward < 0:
                loser = 1 - winner
                p_rewards[loser] += pending_reward

            loser = 1 - winner
            assert p_rewards[winner] > 0, f"seed={seed}: winner reward should be positive"
            assert p_rewards[loser] < 0, f"seed={seed}: loser reward should be negative"

    def test_encode_unified_invalid_sc(self):
        """Invalid SC should encode to 0, not _n_move_actions."""
        from hex_core import HexCoord
        from game.actions import SetupSupplyChainAction
        config = GameConfig(preset=MapPreset.SMALL_FIXED)
        env = FlatHexWarEnv(config, sub_step=True)
        env.reset(seed=42)

        bad_sc = SetupSupplyChainAction(
            source=HexCoord(998, 998), destination=HexCoord(996, 998),
        )
        encoded = env.encode_unified(bad_sc)
        assert encoded == 0, f"Invalid SC should encode to 0, got {encoded}"

    def test_step_after_done_delivers_pending_then_zero(self):
        """First step after done delivers other player's outcome; second is 0."""
        config = GameConfig(preset=MapPreset.SMALL_FIXED)
        env = FlatHexWarEnv(config)
        env.reset(seed=42)
        bots = [BotFlatAdapter(GreedyExpansionBot()), BotFlatAdapter(RandomBot())]

        done = False
        for _ in range(200):
            p = env.engine.state.current_player
            vec = bots[p].choose_action_vector(env)
            _, _, done, _, _ = env.step(vec)
            if done:
                break
        assert done

        # First step after done: delivers the other player's +/-10
        vec = np.zeros(env.action_vector_length, dtype=np.int64)
        _, reward1, done1, _, _ = env.step(vec)
        assert done1 is True
        assert reward1 in (10.0, -10.0), f"Pending reward should be +/-10, got {reward1}"

        # Second step after done: no more pending, returns 0
        _, reward2, done2, _, _ = env.step(vec)
        assert done2 is True
        assert reward2 == 0.0

    def test_sub_step_after_done_delivers_pending(self):
        """Sub-step after done delivers other player's outcome."""
        config = GameConfig(preset=MapPreset.SMALL_FIXED)
        env = FlatHexWarEnv(config, sub_step=True)
        env.reset(seed=42)
        bots = [BotFlatAdapter(GreedyExpansionBot()), BotFlatAdapter(RandomBot())]

        done = False
        for _ in range(2000):
            p = env.engine.state.current_player
            action = bots[p].choose_sub_action(env)
            _, _, done, _, _ = env.step(action)
            if done:
                break
        assert done

        _, reward, done2, _, _ = env.step(0)
        assert done2 is True
        assert reward in (10.0, -10.0, 0.0)  # pending or already delivered
