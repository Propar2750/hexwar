"""HexWar Replay Viewer — watch recorded bot games step by step.

Standalone replay app launched by bot_runner.py after a tournament.
Loads a GameRecord JSON and steps through frame snapshots with pygame.
Not imported by any other module — safe to modify freely.

Dependencies:
    hex_core, hex_grid, renderer, game/config, game/engine, game/actions,
    game/state, game/game_renderer, game/recorder

Controls:
    Space        — play / pause auto-advance
    Right arrow  — next frame
    Left arrow   — previous frame
    Up arrow     — speed up
    Down arrow   — slow down
    R            — restart from beginning
    Escape       — quit
"""

from __future__ import annotations

import pygame

from hex_core import HexCoord, pixel_to_hex
from hex_grid import HexGrid
from renderer import BG_COLOR

from game.config import GameConfig, MapPreset
from game.engine import GameEngine
from game.actions import MoveAction, EndTurnAction, SetupSupplyChainAction
from game.state import GamePhase, GameState, TileState, SupplyChain
from game.game_renderer import (
    GameRenderer, PLAYER_NAMES, PLAYER_COLORS,
    HUD_TEXT, HUD_DIM, HUD_ACCENT, HUD_PANEL_BG,
)
from game.recorder import GameRecord, FrameSnapshot


# ── Display settings ─────────────────────────────────────────
SCREEN_W, SCREEN_H = 1200, 900

HEX_SIZE_FOR_PRESET: dict[MapPreset, int] = {
    MapPreset.SMALL: 40,
    MapPreset.SMALL_FIXED: 40,
    MapPreset.MEDIUM: 30,
    MapPreset.LARGE: 22,
}

SPEED_LEVELS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
DEFAULT_SPEED_INDEX = 2  # 1.0x

# Base delay between frames at 1x speed (milliseconds)
BASE_FRAME_DELAY = 600


def _describe_action(
    action: MoveAction | EndTurnAction | SetupSupplyChainAction,
    player: int,
    bot_names: dict[int, str],
) -> str:
    """Human-readable description of an action."""
    name = bot_names.get(player, PLAYER_NAMES[player % len(PLAYER_NAMES)])
    if isinstance(action, MoveAction):
        return (
            f"[{name}] moves {action.troops} troops "
            f"({action.source.col},{action.source.row}) -> "
            f"({action.target.col},{action.target.row})"
        )
    if isinstance(action, SetupSupplyChainAction):
        return (
            f"[{name}] supply chain "
            f"({action.source.col},{action.source.row}) -> "
            f"({action.destination.col},{action.destination.row})"
        )
    if isinstance(action, EndTurnAction):
        return f"[{name}] ends turn"
    return f"[{name}] unknown action"


def _apply_snapshot(state: GameState, frame: FrameSnapshot) -> None:
    """Apply a recorded snapshot directly onto a GameState."""
    for coord, (owner, troops) in frame.tile_states.items():
        ts = state.tiles.get(coord)
        if ts is not None:
            ts.owner = owner
            ts.troops = troops

    state.supply_chains = [
        SupplyChain(source=src, destination=dst, owner=owner)
        for src, dst, owner in frame.supply_chains
    ]
    state.turn = frame.turn
    state.current_player = frame.current_player
    state.winner = frame.winner

    if frame.phase_name == "GAME_OVER":
        state.phase = GamePhase.GAME_OVER
    else:
        state.phase = GamePhase.PLAY


def _apply_initial_state(state: GameState, record: GameRecord) -> None:
    """Apply the initial recorded state."""
    for coord, (owner, troops) in record.initial_tile_states.items():
        ts = state.tiles.get(coord)
        if ts is not None:
            ts.owner = owner
            ts.troops = troops
    state.supply_chains = [
        SupplyChain(source=src, destination=dst, owner=owner)
        for src, dst, owner in record.initial_supply_chains
    ]
    state.turn = record.initial_turn
    state.current_player = 0
    state.phase = GamePhase.PLAY


# ---------------------------------------------------------------------------
# Main replay function
# ---------------------------------------------------------------------------

def replay_game(record: GameRecord) -> None:
    """Launch the pygame replay viewer for a recorded game."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("HexWar Replay")
    clock = pygame.time.Clock()

    config = record.config
    preset = config.preset or MapPreset.MEDIUM
    hex_size = HEX_SIZE_FOR_PRESET.get(preset, 30)

    # Build the grid (just for topology/terrain — we'll overwrite tile states)
    engine = GameEngine(GameConfig(
        preset=preset,
        map_seed=record.seed,
        auto_place_starts=True,
    ))
    engine.reset()
    state = engine.state

    # Apply initial recorded state
    _apply_initial_state(state, record)

    origin_x = SCREEN_W / 2 - (config.grid_width - 1) * hex_size * 0.866
    origin_y = SCREEN_H / 2 - (config.grid_height - 1) * hex_size * 0.75
    renderer = GameRenderer(screen, hex_size=hex_size, origin=(origin_x, origin_y))

    # Replay state
    frame_index = -1  # -1 = showing initial state
    playing = False
    speed_index = DEFAULT_SPEED_INDEX
    last_advance = pygame.time.get_ticks()

    # Fonts for replay HUD
    hud_font = pygame.font.SysFont("consolas", 16)
    hud_font_bold = pygame.font.SysFont("consolas", 16, bold=True)

    hovered: HexCoord | None = None
    running = True

    while running:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    playing = not playing
                    last_advance = now

                elif event.key == pygame.K_RIGHT:
                    if frame_index < len(record.frames) - 1:
                        frame_index += 1
                        _apply_snapshot(state, record.frames[frame_index])
                    playing = False

                elif event.key == pygame.K_LEFT:
                    if frame_index > -1:
                        frame_index -= 1
                        if frame_index == -1:
                            _apply_initial_state(state, record)
                        else:
                            _apply_snapshot(state, record.frames[frame_index])
                    playing = False

                elif event.key == pygame.K_UP:
                    speed_index = min(len(SPEED_LEVELS) - 1, speed_index + 1)

                elif event.key == pygame.K_DOWN:
                    speed_index = max(0, speed_index - 1)

                elif event.key == pygame.K_r:
                    frame_index = -1
                    _apply_initial_state(state, record)
                    playing = False
                    renderer._game_over_overlay = None

        # Auto-advance
        if playing and frame_index < len(record.frames) - 1:
            delay = int(BASE_FRAME_DELAY / SPEED_LEVELS[speed_index])
            if now - last_advance >= delay:
                frame_index += 1
                _apply_snapshot(state, record.frames[frame_index])
                last_advance = now
                # Stop at end
                if frame_index >= len(record.frames) - 1:
                    playing = False

        # Hover
        mx, my = pygame.mouse.get_pos()
        candidate = pixel_to_hex(
            mx - renderer.origin_x, my - renderer.origin_y, hex_size,
        )
        hovered = candidate if candidate in state.grid else None

        # ── Draw ──────────────────────────────────────────────
        screen.fill(BG_COLOR)

        # Determine move arrow for current frame
        move_src: HexCoord | None = None
        move_tgt: HexCoord | None = None
        move_troops = 0
        if 0 <= frame_index < len(record.frames):
            action = record.frames[frame_index].action
            if isinstance(action, MoveAction):
                move_src = action.source
                move_tgt = action.target
                move_troops = action.troops

        renderer.draw_game(
            state.grid, state,
            hovered=hovered,
            selected=move_src,
            valid_targets=set(),
            troop_target=move_tgt,
            troops_to_send=move_troops,
        )
        renderer.draw_supply_chains(state)

        # Build action description message
        if 0 <= frame_index < len(record.frames):
            frame = record.frames[frame_index]
            message = _describe_action(frame.action, frame.player, record.bot_names)
        else:
            message = "Initial state"

        renderer.draw_hud(state, config, message=message, hovered=hovered)

        # ── Replay controls bar ──────────────────────────────
        _draw_replay_bar(
            screen, hud_font, hud_font_bold,
            frame_index=frame_index,
            total_frames=len(record.frames),
            playing=playing,
            speed=SPEED_LEVELS[speed_index],
            bot_names=record.bot_names,
        )

        if state.phase == GamePhase.GAME_OVER:
            renderer.draw_game_over_overlay(state)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def _draw_replay_bar(
    screen: pygame.Surface,
    font: pygame.font.Font,
    font_bold: pygame.font.Font,
    frame_index: int,
    total_frames: int,
    playing: bool,
    speed: float,
    bot_names: dict[int, str],
) -> None:
    """Draw the bottom replay controls bar."""
    sw = screen.get_width()
    sh = screen.get_height()
    bar_h = 28

    bar = pygame.Surface((sw, bar_h), pygame.SRCALPHA)
    bar.fill((*HUD_PANEL_BG, 200))

    x = 14

    # Frame counter
    display_frame = max(0, frame_index + 1) if frame_index >= 0 else 0
    frame_text = f"Frame {display_frame}/{total_frames}"
    surf = font_bold.render(frame_text, True, HUD_TEXT)
    bar.blit(surf, (x, 5))
    x += surf.get_width() + 20

    # Play/pause indicator
    status = "Playing" if playing else "Paused"
    status_color = (100, 255, 100) if playing else HUD_DIM
    surf = font.render(status, True, status_color)
    bar.blit(surf, (x, 5))
    x += surf.get_width() + 20

    # Speed
    speed_text = f"Speed: {speed}x"
    surf = font.render(speed_text, True, HUD_ACCENT)
    bar.blit(surf, (x, 5))
    x += surf.get_width() + 30

    # Controls hint
    controls = "Space:play/pause  Arrows:step  Up/Down:speed  R:restart  Esc:quit"
    surf = font.render(controls, True, HUD_DIM)
    bar.blit(surf, (x, 5))

    # Bot names on right side
    names_text = " vs ".join(
        f"{bot_names.get(i, '?')}" for i in sorted(bot_names)
    )
    surf = font_bold.render(names_text, True, HUD_ACCENT)
    bar.blit(surf, (sw - surf.get_width() - 14, 5))

    screen.blit(bar, (0, sh - bar_h))


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from game.recorder import load_record

    if len(sys.argv) < 2:
        print("Usage: python replay.py <path_to_game.json>")
        print("  e.g. python replay.py tournaments/tournament_1/replays/game_001_Greedy_vs_Turtle.json")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"Loading replay from {path}...")
    record = load_record(path)
    names = " vs ".join(record.bot_names[i] for i in sorted(record.bot_names))
    print(f"  {names} | {record.total_turns} turns | {len(record.frames)} frames")
    replay_game(record)
