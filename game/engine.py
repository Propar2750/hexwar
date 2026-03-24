"""GameEngine — orchestrates the full game lifecycle.

Responsibilities:
    • reset / map generation
    • starting-position placement
    • troop generation each round
    • move execution (with combat)
    • turn / phase transitions
    • victory detection

All state mutations go through this class so rules are enforced once.
"""

from __future__ import annotations

import random

from hex_core import HexCoord
from hex_grid import HexGrid, Terrain
from map_generator import generate_terrain

from game.actions import (
    EndTurnAction, MoveAction, SetupSupplyChainAction,
    validate_move, validate_supply_chain, get_valid_targets,
)
from game.state import SupplyChain
from game.combat import CombatResolver, DefaultCombatResolver
from game.config import GameConfig, MapPreset, SMALL_FIXED_TERRAIN, SMALL_FIXED_STARTS
from game.state import GamePhase, GameState


class GameEngine:
    """Stateful game controller."""

    def __init__(
        self,
        config: GameConfig | None = None,
        combat_resolver: CombatResolver | None = None,
    ) -> None:
        self.config = config or GameConfig()
        self.combat: CombatResolver = combat_resolver or DefaultCombatResolver()
        self.rng = random.Random(self.config.map_seed)
        self.state: GameState = None  # type: ignore[assignment]  — set by reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> GameState:
        """Create a fresh game: build grid, generate terrain, init state.

        When ``auto_place_starts`` is enabled (default), the engine will
        automatically choose resource-balanced starting positions and skip
        the SETUP phase.  If no balanced placement is found after several
        terrain regenerations, the best attempt is used.

        For the SMALL_FIXED preset, a hand-crafted symmetric map with
        predetermined spawn points is used instead of procedural generation.
        """
        if self.config.preset == MapPreset.SMALL_FIXED:
            return self._reset_fixed()

        cfg = self.config
        seed = cfg.map_seed

        best_state: GameState | None = None
        best_ratio: float = -1.0

        for attempt in range(10):
            grid = HexGrid(cfg.grid_width, cfg.grid_height)
            generate_terrain(
                grid,
                seed=seed,
                fertile_p=cfg.fertile_p,
                ca_iterations=cfg.ca_iterations,
                ca_threshold=cfg.ca_threshold,
                num_ranges=cfg.num_mountain_ranges,
                min_range_steps=cfg.min_range_steps,
                range_end_prob=cfg.range_end_prob,
            )
            state = GameState(grid, cfg.num_players)

            if not cfg.auto_place_starts:
                self.state = state
                return self.state

            starts = self._find_balanced_starts(state, grid)
            if starts is not None:
                ratio = self._placement_ratio(state, grid, starts)
                if ratio >= cfg.balance_threshold:
                    self._apply_starts(state, starts)
                    self.state = state
                    return self.state
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_state = state
                    self._apply_starts(best_state, starts)

            # Try a different seed for next attempt
            seed = (seed or 0) + 1

        # Use the best attempt we found, or fall back to unplaced state
        if best_state is not None:
            self.state = best_state
        else:
            self.state = state  # type: ignore[possibly-undefined]
        return self.state

    def _reset_fixed(self) -> GameState:
        """Build the hand-crafted SMALL_FIXED map with predetermined spawns."""
        cfg = self.config
        grid = HexGrid(cfg.grid_width, cfg.grid_height)

        # Paint terrain from the fixed layout
        for tile in grid:
            key = (tile.coord.col, tile.coord.row)
            tile.terrain = SMALL_FIXED_TERRAIN.get(key, Terrain.PLAINS)

        state = GameState(grid, cfg.num_players)

        # Place players at the fixed spawn points
        starts = [HexCoord(c, r) for c, r in SMALL_FIXED_STARTS]
        self._apply_starts(state, starts)
        self.state = state
        return self.state

    # ------------------------------------------------------------------
    # Setup phase
    # ------------------------------------------------------------------

    def place_starting_position(self, coord: HexCoord) -> str | None:
        """Current setup-player claims *coord* as their start.

        Returns an error string or None on success.
        """
        s = self.state
        if s.phase != GamePhase.SETUP:
            return "Not in setup phase"
        if coord not in s.tiles:
            return "Invalid tile"

        # Enforce minimum distance from already-placed players
        for other_coord, ts in s.tiles.items():
            if ts.owner is not None and coord.distance_to(other_coord) < self.config.min_start_distance:
                return f"Too close to player {ts.owner + 1}'s start (min distance: {self.config.min_start_distance})"

        player = s.players_placed
        ts = s.tiles[coord]
        ts.owner = player
        ts.troops = self.config.starting_troops
        s.players_placed += 1

        if s.players_placed >= s.num_players:
            s.phase = GamePhase.PLAY
            s.current_player = 0
            s.turn = 1
            self._generate_troops()  # first round troop gen

        return None

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def execute_action(self, action: MoveAction | EndTurnAction | SetupSupplyChainAction) -> str | None:
        """Execute a player action.  Returns error string or None."""
        s = self.state

        if isinstance(action, EndTurnAction):
            return self._end_turn()

        if isinstance(action, MoveAction):
            return self._execute_move(action)

        if isinstance(action, SetupSupplyChainAction):
            return self._setup_supply_chain(action)

        return f"Unknown action type: {type(action)}"

    def _execute_move(self, action: MoveAction) -> str | None:
        s = self.state
        player = s.current_player

        # Check move limit
        if self.config.moves_per_turn > 0 and s.moves_made >= self.config.moves_per_turn:
            return "No moves remaining this turn"

        err = validate_move(action, s, player)
        if err:
            return err

        src = s.tiles[action.source]
        tgt = s.tiles[action.target]
        terrain = s.grid[action.target].terrain

        # Remove troops from source
        src.troops -= action.troops

        if tgt.owner == player:
            # Friendly — just reinforce
            tgt.troops += action.troops
        elif tgt.owner is None:
            # Neutral tile (has garrison) — fight the garrison
            defense_bonus = self.config.defense_bonus[terrain]
            result = self.combat.resolve(
                action.troops, tgt.troops, defense_bonus, self.rng
            )
            if result.attacker_won:
                tgt.owner = player
                tgt.troops = result.attacker_remaining
            else:
                tgt.troops = result.defender_remaining
        else:
            # Enemy tile — combat!
            defense_bonus = self.config.defense_bonus[terrain]
            result = self.combat.resolve(
                action.troops, tgt.troops, defense_bonus, self.rng
            )
            if result.attacker_won:
                # Tile changes hands — destroy all supply chains touching it
                self._break_supply_chains_at(action.target)
                tgt.owner = player
                tgt.troops = result.attacker_remaining
            else:
                tgt.troops = result.defender_remaining
                # attacking troops are already removed from source

        s.moves_made += 1

        # Check if defender was eliminated
        self._check_elimination()

        return None

    def _setup_supply_chain(self, action: SetupSupplyChainAction) -> str | None:
        s = self.state
        player = s.current_player

        err = validate_supply_chain(action, s, player)
        if err:
            return err

        chain = SupplyChain(
            source=action.source,
            destination=action.destination,
            owner=player,
        )
        s.supply_chains.append(chain)
        s.supply_chains_set_this_turn[player] = s.supply_chains_set_this_turn.get(player, 0) + 1
        return None

    def _end_turn(self) -> str | None:
        s = self.state
        if s.phase != GamePhase.PLAY:
            return "Game is not in play phase"

        s.moves_made = 0

        # Find the next alive player (skip eliminated ones)
        next_player = (s.current_player + 1) % s.num_players
        while not s.is_alive(next_player) and next_player != s.current_player:
            next_player = (next_player + 1) % s.num_players

        if next_player <= s.current_player:
            # All players have acted — new round
            winner = self._check_victory()
            if winner is not None:
                s.winner = winner
                s.phase = GamePhase.GAME_OVER
                return None

            s.turn += 1
            if s.turn > self.config.max_turns:
                # Time's up — most territory wins
                s.winner = self._most_territory()
                s.phase = GamePhase.GAME_OVER
                return None

            self._generate_troops()
            self._process_supply_chains()
            s.supply_chains_set_this_turn.clear()

        s.current_player = next_player
        return None

    # ------------------------------------------------------------------
    # Balanced placement helpers
    # ------------------------------------------------------------------

    def _region_value(self, state: GameState, grid: HexGrid, coord: HexCoord) -> int:
        """Sum troop-generation potential for all tiles within balance_radius."""
        total = 0
        for tile in grid:
            if coord.distance_to(tile.coord) <= self.config.balance_radius:
                total += self.config.troop_generation.get(tile.terrain, 0)
        return total

    def _placement_ratio(
        self, state: GameState, grid: HexGrid, starts: list[HexCoord],
    ) -> float:
        """Return min/max region-value ratio (1.0 = perfectly balanced)."""
        values = [self._region_value(state, grid, c) for c in starts]
        if not values or max(values) == 0:
            return 0.0
        return min(values) / max(values)

    def _find_balanced_starts(
        self, state: GameState, grid: HexGrid, max_attempts: int = 50,
    ) -> list[HexCoord] | None:
        """Try to find starting positions that satisfy distance + balance constraints.

        Returns a list of coords (one per player) or None if no valid set was found.
        """
        from hex_grid import Terrain

        cfg = self.config
        candidates = [t.coord for t in grid if t.terrain != Terrain.MOUNTAIN]
        if len(candidates) < cfg.num_players:
            return None

        best_starts: list[HexCoord] | None = None
        best_ratio: float = -1.0

        for _ in range(max_attempts):
            picks: list[HexCoord] = []
            pool = list(candidates)
            self.rng.shuffle(pool)

            for coord in pool:
                if all(coord.distance_to(p) >= cfg.min_start_distance for p in picks):
                    picks.append(coord)
                    if len(picks) == cfg.num_players:
                        break

            if len(picks) < cfg.num_players:
                continue

            ratio = self._placement_ratio(state, grid, picks)
            if ratio > best_ratio:
                best_ratio = ratio
                best_starts = picks

            if ratio >= cfg.balance_threshold:
                return picks

        return best_starts

    def _apply_starts(self, state: GameState, starts: list[HexCoord]) -> None:
        """Place players at the given starting coords and transition to PLAY."""
        for player, coord in enumerate(starts):
            ts = state.tiles[coord]
            ts.owner = player
            ts.troops = self.config.starting_troops
            state.players_placed += 1

        state.phase = GamePhase.PLAY
        state.current_player = 0
        state.turn = 1
        self._state_backup = self.state  # save in case _generate_troops needs self.state
        self.state = state
        self._generate_troops()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_troops(self) -> None:
        """Add troops to every owned tile based on terrain."""
        s = self.state
        for tile in s.grid:
            ts = s.tiles[tile.coord]
            if ts.owner is not None:
                gen = self.config.troop_generation.get(tile.terrain, 0)
                ts.troops += gen

    def _process_supply_chains(self) -> None:
        """Transfer troops along supply chains to their terminal endpoints.

        Troops flow all the way to the end of a chain path — relay tiles
        (tiles that are both a destination and a source) keep only 1 troop
        and pass everything onward.  Broken chains are removed.
        """
        s = self.state
        surviving: list[SupplyChain] = []

        # First pass: validate chains and build the outgoing lookup
        outgoing: dict[HexCoord, HexCoord] = {}  # source → destination
        for chain in s.supply_chains:
            src_ts = s.tiles.get(chain.source)
            dst_ts = s.tiles.get(chain.destination)
            if (src_ts is None or src_ts.owner != chain.owner
                    or dst_ts is None or dst_ts.owner != chain.owner):
                continue  # chain destroyed
            outgoing[chain.source] = chain.destination
            surviving.append(chain)

        s.supply_chains = surviving

        # Find the terminal endpoint for each source tile
        # (follow outgoing links until we reach a tile with no outgoing chain)
        terminal_cache: dict[HexCoord, HexCoord] = {}

        def find_terminal(coord: HexCoord) -> HexCoord:
            if coord in terminal_cache:
                return terminal_cache[coord]
            current = coord
            path = []
            while current in outgoing:
                path.append(current)
                current = outgoing[current]
            # current is the terminal — cache for every tile in the path
            for node in path:
                terminal_cache[node] = current
            return current

        # Collect all source tiles (tiles that have an outgoing chain)
        # and transfer their troops to the terminal endpoint
        for source_coord in list(outgoing.keys()):
            src_ts = s.tiles[source_coord]
            if src_ts.troops > 1:
                terminal = find_terminal(source_coord)
                transfer = src_ts.troops - 1
                src_ts.troops = 1
                s.tiles[terminal].troops += transfer

    def _break_supply_chains_at(self, coord: HexCoord) -> None:
        """Destroy every supply chain that touches *coord*."""
        s = self.state
        s.supply_chains = [
            sc for sc in s.supply_chains
            if sc.source != coord and sc.destination != coord
        ]

    def _check_victory(self) -> int | None:
        """Return the winning player index, or None."""
        s = self.state
        total = len(s.tiles)
        threshold = int(total * self.config.win_threshold)
        for p in range(s.num_players):
            if s.territory_count(p) >= threshold:
                return p
        return None

    def _check_elimination(self) -> None:
        """If any player has 0 territory, the other player wins."""
        s = self.state
        alive = [p for p in range(s.num_players) if s.is_alive(p)]
        if len(alive) == 1:
            s.winner = alive[0]
            s.phase = GamePhase.GAME_OVER

    def _most_territory(self) -> int:
        """Return the player controlling the most tiles (tiebreak: most troops)."""
        s = self.state
        best = max(
            range(s.num_players),
            key=lambda p: (s.territory_count(p), s.total_troops(p)),
        )
        return best

    # ------------------------------------------------------------------
    # Queries (convenience wrappers)
    # ------------------------------------------------------------------

    def get_valid_targets(self, source: HexCoord) -> list[HexCoord]:
        return get_valid_targets(self.state, source, self.state.current_player)
