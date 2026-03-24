"""HexWar — interactive 2-player game.

Controls:
    Left-click   — select a tile / confirm target
    Right-click  — cancel selection
    Scroll wheel — adjust troop count (when selecting troops)
    Space        — end your turn
    R            — restart game
    Escape       — quit
"""

import pygame

from hex_core import HexCoord, pixel_to_hex
from renderer import BG_COLOR

from game.config import GameConfig
from game.engine import GameEngine
from game.actions import MoveAction, EndTurnAction, SetupSupplyChainAction, get_valid_targets
from game.state import GamePhase, GameState
from game.game_renderer import GameRenderer, PLAYER_NAMES

# ── Display settings (matches main.py) ────────────────────────
SCREEN_W, SCREEN_H = 1200, 900
HEX_SIZE = 30


class UIState:
    """Tracks interactive selection state (separate from game state)."""

    def __init__(self) -> None:
        self.selected: HexCoord | None = None
        self.valid_targets: set[HexCoord] = set()
        self.troop_target: HexCoord | None = None
        self.troops_to_send: int = 0
        self.max_troops: int = 0
        self.message: str = ""
        self.message_timer: int = 0
        # Supply chain placement mode
        self.supply_chain_mode: bool = False
        self.sc_source: HexCoord | None = None
        self.sc_valid_targets: set[HexCoord] = set()

    def clear(self) -> None:
        self.selected = None
        self.valid_targets = set()
        self.troop_target = None
        self.troops_to_send = 0
        self.max_troops = 0

    def clear_supply_chain(self) -> None:
        self.supply_chain_mode = False
        self.sc_source = None
        self.sc_valid_targets = set()

    def set_message(self, msg: str, frames: int = 120) -> None:
        self.message = msg
        self.message_timer = frames

    def tick(self) -> None:
        if self.message_timer > 0:
            self.message_timer -= 1
            if self.message_timer == 0:
                self.message = ""


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("HexWar")
    clock = pygame.time.Clock()

    config = GameConfig()
    engine = GameEngine(config)
    state = engine.reset()

    # Center the grid — same formula as main.py
    origin_x = SCREEN_W / 2 - (config.grid_width - 1) * HEX_SIZE * 0.866
    origin_y = SCREEN_H / 2 - (config.grid_height - 1) * HEX_SIZE * 0.75
    renderer = GameRenderer(screen, hex_size=HEX_SIZE, origin=(origin_x, origin_y))

    ui = UIState()
    hovered: HexCoord | None = None
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    engine = GameEngine(config)
                    state = engine.reset()
                    ui.clear()
                    ui.message = ""

                elif event.key == pygame.K_s:
                    if state.phase == GamePhase.PLAY:
                        if ui.supply_chain_mode:
                            ui.clear_supply_chain()
                            ui.set_message("Supply chain mode cancelled", 60)
                        elif state.current_player in state.supply_chain_set_this_turn:
                            ui.set_message("Already set a supply chain this turn", 90)
                        else:
                            ui.clear()
                            ui.supply_chain_mode = True
                            ui.set_message("SUPPLY CHAIN — click source tile (your territory)", 0)

                elif event.key == pygame.K_SPACE:
                    if state.phase == GamePhase.PLAY:
                        ui.clear()
                        ui.clear_supply_chain()
                        engine.execute_action(EndTurnAction())
                        state = engine.state
                        if state.phase == GamePhase.PLAY:
                            p = state.current_player
                            name = PLAYER_NAMES[p % len(PLAYER_NAMES)]
                            ui.set_message(f"{name}'s turn", 90)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:  # right click — cancel
                    if ui.supply_chain_mode:
                        ui.clear_supply_chain()
                        ui.set_message("", 0)
                    ui.clear()
                    continue

                if event.button == 1 and hovered is not None:
                    if ui.supply_chain_mode:
                        _handle_supply_chain_click(engine, state, ui, hovered, config)
                        state = engine.state
                    else:
                        _handle_click(engine, state, ui, hovered, config)
                        state = engine.state

                # Scroll wheel — adjust troop count
                if ui.troop_target is not None:
                    if event.button == 4:
                        ui.troops_to_send = min(ui.max_troops, ui.troops_to_send + 1)
                    elif event.button == 5:
                        ui.troops_to_send = max(1, ui.troops_to_send - 1)
                    if ui.troop_target is not None:
                        ui.message = f"Sending {ui.troops_to_send}/{ui.max_troops} troops — scroll to adjust, click target to confirm"
                        ui.message_timer = 0

        # Hover detection
        mx, my = pygame.mouse.get_pos()
        candidate = pixel_to_hex(
            mx - renderer.origin_x, my - renderer.origin_y, HEX_SIZE,
        )
        hovered = candidate if candidate in state.grid else None

        ui.tick()

        # ── Draw ──────────────────────────────────────────────
        screen.fill(BG_COLOR)

        # Merge supply chain targets into the highlight set
        all_valid = ui.valid_targets | ui.sc_valid_targets
        sc_selected = ui.sc_source if ui.supply_chain_mode else None

        renderer.draw_game(
            state.grid, state,
            hovered=hovered,
            selected=ui.selected or sc_selected,
            valid_targets=all_valid,
            troop_target=ui.troop_target,
            troops_to_send=ui.troops_to_send,
        )

        # Supply chains: active + preview
        renderer.draw_supply_chains(state)
        if ui.supply_chain_mode and ui.sc_source is not None:
            preview_dst = hovered if hovered in ui.sc_valid_targets else None
            renderer.draw_supply_chain_preview(ui.sc_source, preview_dst)

        renderer.draw_hud(state, config, message=ui.message, hovered=hovered)
        renderer.draw_controls()

        if state.phase == GamePhase.GAME_OVER:
            renderer.draw_game_over_overlay(state)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def _handle_click(
    engine: GameEngine,
    state: GameState,
    ui: UIState,
    clicked: HexCoord,
    config: GameConfig,
) -> None:
    """Process a left-click on a tile based on current game/UI state."""

    # ── SETUP: place starting positions ──
    if state.phase == GamePhase.SETUP:
        err = engine.place_starting_position(clicked)
        if err:
            ui.set_message(err, 120)
        else:
            p = state.players_placed - 1
            name = PLAYER_NAMES[p % len(PLAYER_NAMES)]
            if state.phase == GamePhase.PLAY:
                ui.set_message(f"{name} placed!  Game on — {PLAYER_NAMES[0]}'s turn", 120)
            else:
                next_name = PLAYER_NAMES[state.players_placed % len(PLAYER_NAMES)]
                ui.set_message(f"{name} placed!  {next_name}, choose your base", 0)
        return

    # ── GAME_OVER: ignore ──
    if state.phase == GamePhase.GAME_OVER:
        return

    # ── PLAY ──
    player = state.current_player

    # Confirming troop count
    if ui.troop_target is not None:
        if clicked == ui.troop_target:
            move = MoveAction(
                source=ui.selected,
                target=ui.troop_target,
                troops=ui.troops_to_send,
            )
            err = engine.execute_action(move)
            if err:
                ui.set_message(err, 120)
            else:
                ui.set_message("", 0)
            ui.clear()
            return
        else:
            ui.clear()
            # fall through to re-evaluate this click

    # Selecting target from valid list
    if ui.selected is not None and clicked in ui.valid_targets:
        src_ts = state.tiles[ui.selected]
        max_send = src_ts.troops - 1
        if max_send < 1:
            ui.set_message("Not enough troops", 90)
            ui.clear()
            return
        ui.troop_target = clicked
        ui.troops_to_send = max_send
        ui.max_troops = max_send
        ui.message = f"Sending {max_send}/{max_send} troops — scroll to adjust, click target to confirm"
        ui.message_timer = 0
        return

    # No moves left — block new selections
    if config.moves_per_turn > 0 and state.moves_made >= config.moves_per_turn:
        ui.clear()
        ui.set_message("No moves remaining — press Space to end turn", 90)
        return

    # Selecting a source tile
    ts = state.tiles.get(clicked)
    if ts and ts.owner == player and ts.troops >= 2:
        ui.clear()
        ui.selected = clicked
        ui.valid_targets = set(get_valid_targets(state, clicked, player))
        if not ui.valid_targets:
            ui.set_message("No valid targets", 60)
            ui.clear()
    elif ts and ts.owner == player:
        ui.clear()
        ui.set_message("Need 2+ troops to move", 60)
    else:
        ui.clear()


def _handle_supply_chain_click(
    engine: GameEngine,
    state: GameState,
    ui: UIState,
    clicked: HexCoord,
    config: GameConfig,
) -> None:
    """Process a left-click while in supply chain placement mode.

    Flow: select source → select adjacent owned neighbour → chain created.
    All generated troops are forwarded each round (no amount to configure).
    """
    player = state.current_player
    ts = state.tiles.get(clicked)

    # Step 1: select source tile
    if ui.sc_source is None:
        if ts and ts.owner == player:
            ui.sc_source = clicked
            # Build neighbour targets (only tiles owned by this player)
            neighbors = [
                n.coord for n in state.grid.neighbors_of(clicked)
                if n.coord in state.tiles and state.tiles[n.coord].owner == player
            ]
            if not neighbors:
                ui.set_message("No adjacent owned tiles to connect", 90)
                ui.clear_supply_chain()
                return
            ui.sc_valid_targets = set(neighbors)
            ui.set_message("SUPPLY CHAIN — click adjacent owned tile", 0)
        else:
            ui.set_message("Must select a tile you own", 90)
        return

    # Step 2: select destination from valid neighbours → create chain
    if clicked in ui.sc_valid_targets:
        action = SetupSupplyChainAction(source=ui.sc_source, destination=clicked)
        err = engine.execute_action(action)
        if err:
            ui.set_message(err, 120)
        else:
            ui.set_message("Supply chain established!", 90)
        ui.clear_supply_chain()
        return

    # Clicked an invalid tile — restart
    ui.set_message("Must click an adjacent owned tile", 90)
    ui.sc_source = None
    ui.sc_valid_targets = set()


if __name__ == "__main__":
    main()
