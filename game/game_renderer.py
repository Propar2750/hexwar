"""Game-aware hex renderer — draws ownership, troops, and HUD.

Keeps the same dark, organic look as main.py's terrain demo.
Ownership is shown via subtle colour tinting and player-coloured
borders.  Troop counts are rendered as small readable labels.
"""

from __future__ import annotations

import math

import pygame

from hex_core import HexCoord, hex_vertices
from hex_grid import HexGrid, Terrain
from renderer import HexRenderer, TERRAIN_FILL, HEX_BORDER, BG_COLOR

from game.state import GameState, GamePhase, SupplyChain
from game.config import GameConfig


# ── Player palette ────────────────────────────────────────────
# Muted tones that sit well on the dark terrain colours.
PLAYER_COLORS: list[tuple[int, int, int]] = [
    (80, 150, 255),    # Player 0 — soft blue
    (255, 90, 80),     # Player 1 — soft red
    (60, 210, 100),    # Player 2 — green  (future)
    (240, 200, 60),    # Player 3 — yellow (future)
]

PLAYER_BORDER: list[tuple[int, int, int]] = [
    (110, 170, 255),   # lighter blue for borders
    (255, 130, 110),   # lighter red  for borders
    (100, 230, 140),
    (255, 220, 100),
]

PLAYER_NAMES = ["Blue", "Red", "Green", "Yellow"]

# How much player colour mixes into the terrain fill (subtle!)
OWNER_BLEND = 0.22

# ── Selection / highlight colours ─────────────────────────────
SELECTED_BORDER   = (255, 255, 140)
VALID_BORDER      = (120, 255, 120)
TARGET_BORDER     = (255, 200, 80)
HOVER_BORDER      = (200, 210, 220)
SUPPLY_CHAIN_PENDING = (180, 180, 255)  # preview colour while placing

# ── HUD colours ───────────────────────────────────────────────
HUD_TEXT       = (210, 215, 220)
HUD_DIM        = (130, 135, 140)
HUD_ACCENT     = (255, 220, 80)
HUD_PANEL_BG   = (18, 18, 24)


def _blend(c1: tuple, c2: tuple, t: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(a + (b - a) * t))) for a, b in zip(c1, c2))


class GameRenderer(HexRenderer):
    """Renders a HexWar game: board + HUD + controls."""

    def __init__(
        self,
        surface: pygame.Surface,
        hex_size: float = 16.0,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        super().__init__(surface, hex_size, origin)
        self._troop_font: pygame.font.Font | None = None
        self._hud_font: pygame.font.Font | None = None
        self._hud_font_bold: pygame.font.Font | None = None
        self._big_font: pygame.font.Font | None = None

    # ── Lazy fonts ────────────────────────────────────────────

    @property
    def troop_font(self) -> pygame.font.Font:
        if self._troop_font is None:
            size = max(9, int(self.hex_size * 0.55))
            self._troop_font = pygame.font.SysFont("consolas", size, bold=True)
        return self._troop_font

    @property
    def hud_font(self) -> pygame.font.Font:
        if self._hud_font is None:
            self._hud_font = pygame.font.SysFont("consolas", 16)
        return self._hud_font

    @property
    def hud_font_bold(self) -> pygame.font.Font:
        if self._hud_font_bold is None:
            self._hud_font_bold = pygame.font.SysFont("consolas", 16, bold=True)
        return self._hud_font_bold

    @property
    def big_font(self) -> pygame.font.Font:
        if self._big_font is None:
            self._big_font = pygame.font.SysFont("consolas", 36, bold=True)
        return self._big_font

    # ── Board ─────────────────────────────────────────────────

    def draw_game(
        self,
        grid: HexGrid,
        state: GameState,
        hovered: HexCoord | None = None,
        selected: HexCoord | None = None,
        valid_targets: set[HexCoord] | None = None,
        troop_target: HexCoord | None = None,
        troops_to_send: int = 0,
    ) -> None:
        vt = valid_targets or set()

        # Three drawing layers so borders are never occluded by
        # a neighbouring hex's fill:
        #   layer 0 — unowned tiles (drawn first, borders don't matter)
        #   layer 1 — owned tiles   (borders must sit on top of layer 0)
        #   layer 2 — highlighted / hovered (always on top)
        layer_owned: list[tuple[HexCoord, tuple, tuple, int]] = []
        layer_highlight: list[tuple[HexCoord, tuple, tuple, int]] = []

        for tile in grid:
            coord = tile.coord
            ts = state.tiles[coord]
            base_fill = TERRAIN_FILL.get(tile.terrain, (50, 70, 90))

            # ── Fill colour ──
            if ts.owner is not None:
                pcol = PLAYER_COLORS[ts.owner % len(PLAYER_COLORS)]
                fill = _blend(base_fill, pcol, OWNER_BLEND)
            else:
                fill = base_fill

            # ── Border colour + width ──
            if coord == selected:
                border, bw = SELECTED_BORDER, 3
            elif coord == troop_target:
                border, bw = TARGET_BORDER, 3
            elif coord in vt:
                border, bw = VALID_BORDER, 2
            elif coord == hovered:
                border, bw = HOVER_BORDER, 2
            elif ts.owner is not None:
                border = PLAYER_BORDER[ts.owner % len(PLAYER_BORDER)]
                bw = 1
            else:
                border, bw = HEX_BORDER, 1

            # Sort into layers
            if bw >= 2:
                layer_highlight.append((coord, fill, border, bw))
            elif ts.owner is not None:
                layer_owned.append((coord, fill, border, bw))
            else:
                # Layer 0 — draw immediately
                self._draw_hex_styled(coord, fill, border, bw)

        # Layer 1 — owned tiles (borders visible above unowned neighbours)
        for coord, fill, border, bw in layer_owned:
            ts = state.tiles[coord]
            self._draw_hex_styled(coord, fill, border, bw)
            if ts.troops > 0:
                self._draw_troop_label(coord, ts.troops, ts.owner)

        # Layer 2 — selected / valid-target / hovered (always on top)
        for coord, fill, border, bw in layer_highlight:
            ts = state.tiles[coord]
            self._draw_hex_styled(coord, fill, border, bw)
            if ts.owner is not None and ts.troops > 0:
                self._draw_troop_label(coord, ts.troops, ts.owner)

        # ── Move arrow ──
        if selected and troop_target and troops_to_send > 0:
            self._draw_move_arrow(selected, troop_target, troops_to_send)

    # ── Hex primitives ────────────────────────────────────────

    def _draw_hex_styled(
        self, coord: HexCoord,
        fill: tuple, border: tuple, bw: int,
    ) -> None:
        cx, cy = self.hex_to_screen(coord)
        verts = hex_vertices(cx, cy, self.hex_size)
        pygame.draw.polygon(self.surface, fill, verts)
        pygame.draw.aalines(self.surface, border, True, verts)
        if bw >= 2:
            pygame.draw.polygon(self.surface, border, verts, bw)

    def _draw_troop_label(
        self, coord: HexCoord, troops: int, owner: int,
    ) -> None:
        cx, cy = self.hex_to_screen(coord)
        text = str(troops)
        pcol = PLAYER_COLORS[owner % len(PLAYER_COLORS)]
        bright = _blend(pcol, (255, 255, 255), 0.65)

        # Shadow for readability
        shadow = self.troop_font.render(text, True, (0, 0, 0))
        sr = shadow.get_rect(center=(cx + 1, cy + 1))
        self.surface.blit(shadow, sr)

        label = self.troop_font.render(text, True, bright)
        rect = label.get_rect(center=(cx, cy))
        self.surface.blit(label, rect)

    def _draw_move_arrow(
        self, src: HexCoord, tgt: HexCoord, troops: int,
    ) -> None:
        sx, sy = self.hex_to_screen(src)
        tx, ty = self.hex_to_screen(tgt)

        # Arrow line
        pygame.draw.line(self.surface, (255, 230, 100), (sx, sy), (tx, ty), 2)

        # Arrowhead
        dx, dy = tx - sx, ty - sy
        length = math.hypot(dx, dy) or 1
        ux, uy = dx / length, dy / length
        head_len = min(8, length * 0.3)
        px, py = -uy, ux  # perpendicular
        tip_x, tip_y = tx - ux * 2, ty - uy * 2
        left  = (tip_x - ux * head_len + px * head_len * 0.5,
                 tip_y - uy * head_len + py * head_len * 0.5)
        right = (tip_x - ux * head_len - px * head_len * 0.5,
                 tip_y - uy * head_len - py * head_len * 0.5)
        pygame.draw.polygon(self.surface, (255, 230, 100),
                            [(tip_x, tip_y), left, right])

        # Troop count at midpoint
        mx, my = (sx + tx) / 2, (sy + ty) / 2
        label = self.troop_font.render(str(troops), True, (255, 230, 100))
        rect = label.get_rect(center=(mx, my - 8))
        # bg pill
        bg = rect.inflate(6, 2)
        pygame.draw.rect(self.surface, (0, 0, 0, 180), bg, border_radius=3)
        self.surface.blit(label, rect)

    # ── Supply chains ───────────────────────────────────────

    def draw_supply_chains(self, state: GameState) -> None:
        """Draw all active supply chains as arrows between adjacent tiles."""
        for chain in state.supply_chains:
            pcol = PLAYER_COLORS[chain.owner % len(PLAYER_COLORS)]
            bright = _blend(pcol, (255, 255, 255), 0.4)
            self._draw_chain_arrow(chain.source, chain.destination, bright)

    def draw_supply_chain_preview(
        self,
        source: HexCoord | None,
        destination: HexCoord | None,
    ) -> None:
        """Draw a supply chain being placed (preview)."""
        if source is not None and destination is not None:
            self._draw_chain_arrow(source, destination, SUPPLY_CHAIN_PENDING)
        elif source is not None:
            # Just mark the source while waiting for destination
            cx, cy = self.hex_to_screen(source)
            size = max(4, int(self.hex_size * 0.22))
            rect = pygame.Rect(cx - size, cy - size, size * 2, size * 2)
            pygame.draw.rect(self.surface, SUPPLY_CHAIN_PENDING, rect)
            pygame.draw.rect(self.surface, (0, 0, 0), rect, 1)

    def _draw_chain_arrow(
        self,
        src: HexCoord,
        dst: HexCoord,
        color: tuple,
    ) -> None:
        """Draw a dashed arrow from *src* to *dst*."""
        sx, sy = self.hex_to_screen(src)
        tx, ty = self.hex_to_screen(dst)

        dx, dy = tx - sx, ty - sy
        length = math.hypot(dx, dy) or 1
        ux, uy = dx / length, dy / length

        # Shorten so the arrow doesn't overlap the hex centres
        inset = self.hex_size * 0.35
        sx2, sy2 = sx + ux * inset, sy + uy * inset
        tx2, ty2 = tx - ux * inset, ty - uy * inset

        # Dashed line
        seg = math.hypot(tx2 - sx2, ty2 - sy2)
        dash, gap = 5, 4
        drawn = 0.0
        drawing = True
        while drawn < seg:
            step = min(dash if drawing else gap, seg - drawn)
            if drawing:
                x1, y1 = sx2 + ux * drawn, sy2 + uy * drawn
                x2, y2 = sx2 + ux * (drawn + step), sy2 + uy * (drawn + step)
                pygame.draw.line(self.surface, color, (x1, y1), (x2, y2), 2)
            drawn += step
            drawing = not drawing

        # Arrowhead at destination end
        head_len = min(7, seg * 0.3)
        px, py = -uy, ux
        tip = (tx2, ty2)
        left = (tx2 - ux * head_len + px * head_len * 0.45,
                ty2 - uy * head_len + py * head_len * 0.45)
        right = (tx2 - ux * head_len - px * head_len * 0.45,
                 ty2 - uy * head_len - py * head_len * 0.45)
        pygame.draw.polygon(self.surface, color, [tip, left, right])

    # ── HUD (top panel) ──────────────────────────────────────

    def draw_hud(
        self,
        state: GameState,
        config: GameConfig,
        message: str = "",
        hovered: HexCoord | None = None,
    ) -> None:
        sw = self.surface.get_width()

        # Semi-transparent top panel (3 rows: info, messages, hover)
        panel_h = 76
        panel = pygame.Surface((sw, panel_h), pygame.SRCALPHA)
        panel.fill((*HUD_PANEL_BG, 210))
        self.surface.blit(panel, (0, 0))

        # ── Left: phase / turn ──
        if state.phase == GamePhase.SETUP:
            p = state.players_placed
            name = PLAYER_NAMES[p % len(PLAYER_NAMES)]
            pcol = PLAYER_COLORS[p % len(PLAYER_COLORS)]
            self._hud_text(f"SETUP", 12, 8, HUD_DIM, bold=True)
            self._hud_text(f"{name}", 70, 8, pcol, bold=True)
            self._hud_text("click a tile to place your base", 12, 30, HUD_DIM)

        elif state.phase == GamePhase.PLAY:
            p = state.current_player
            name = PLAYER_NAMES[p % len(PLAYER_NAMES)]
            pcol = PLAYER_COLORS[p % len(PLAYER_COLORS)]
            self._hud_text(f"TURN {state.turn}/{config.max_turns}", 12, 8, HUD_DIM, bold=True)

            # Player indicator: coloured square + name
            sq_x = 150
            pygame.draw.rect(self.surface, pcol, (sq_x, 10, 12, 12))
            self._hud_text(f"{name}'s turn", sq_x + 18, 8, pcol, bold=True)

            # Move counter — same row, after player name
            moves_left = config.moves_per_turn - state.moves_made
            moves_str = f"Moves: {moves_left}/{config.moves_per_turn}"
            moves_col = HUD_TEXT if moves_left > 0 else (255, 80, 80)
            moves_x = sq_x + 18 + self.hud_font_bold.size(f"{name}'s turn")[0] + 20
            self._hud_text(moves_str, moves_x, 8, moves_col)

            # Scores
            scores_x = moves_x + self.hud_font.size(moves_str)[0] + 30
            for i in range(state.num_players):
                ic = PLAYER_COLORS[i % len(PLAYER_COLORS)]
                tc = state.territory_count(i)
                tt = state.total_troops(i)
                pygame.draw.rect(self.surface, ic, (scores_x, 10, 10, 10))
                score_text = f"{tc} tiles  {tt} troops"
                self._hud_text(score_text, scores_x + 16, 8, HUD_TEXT)
                scores_x += 16 + self.hud_font.size(score_text)[0] + 24

            # Message line — full second row, no overlap
            if message:
                self._hud_text(message, 12, 32, HUD_ACCENT)

        elif state.phase == GamePhase.GAME_OVER:
            w = state.winner
            if w is not None:
                name = PLAYER_NAMES[w % len(PLAYER_NAMES)]
                pcol = PLAYER_COLORS[w % len(PLAYER_COLORS)]
                self._hud_text(f"{name} wins!", 12, 8, pcol, bold=True)
            else:
                self._hud_text("DRAW", 12, 8, HUD_TEXT, bold=True)
            self._hud_text("Press R to restart", 12, 30, HUD_DIM)

        # ── Row 3: hover info ──
        if hovered and hovered in state.tiles:
            ts = state.tiles[hovered]
            tile = state.grid[hovered]
            terrain = tile.terrain.value
            owner = PLAYER_NAMES[ts.owner % len(PLAYER_NAMES)] if ts.owner is not None else "--"
            db = config.defense_bonus.get(tile.terrain, 1.0)
            tg = config.troop_generation.get(tile.terrain, 0)
            info = f"({hovered.col},{hovered.row})  {terrain}  own:{owner}  troops:{ts.troops}  def:{db:.0f}x  gen:+{tg}"
            self._hud_text(info, 12, 54, HUD_DIM)

    def _hud_text(
        self, text: str, x: int, y: int,
        color: tuple, bold: bool = False,
    ) -> None:
        font = self.hud_font_bold if bold else self.hud_font
        surf = font.render(text, True, color)
        self.surface.blit(surf, (x, y))

    # ── Bottom controls bar ───────────────────────────────────

    def draw_controls(self) -> None:
        sw = self.surface.get_width()
        sh = self.surface.get_height()

        bar_h = 28
        bar = pygame.Surface((sw, bar_h), pygame.SRCALPHA)
        bar.fill((*HUD_PANEL_BG, 180))
        self.surface.blit(bar, (0, sh - bar_h))

        parts = [
            ("LClick", "select / move"),
            ("RClick", "cancel"),
            ("Scroll", "adjust troops"),
            ("S", "supply chain"),
            ("Space", "end turn"),
            ("R", "restart"),
            ("Esc", "quit"),
        ]
        x = 14
        for key, desc in parts:
            ks = self.hud_font_bold.render(key, True, HUD_TEXT)
            ds = self.hud_font.render(f" {desc}", True, HUD_DIM)
            self.surface.blit(ks, (x, sh - bar_h + 5))
            x += ks.get_width()
            self.surface.blit(ds, (x, sh - bar_h + 5))
            x += ds.get_width() + 18

    # ── Game-over overlay ─────────────────────────────────────

    def draw_game_over_overlay(self, state: GameState) -> None:
        if state.phase != GamePhase.GAME_OVER:
            return
        sw = self.surface.get_width()
        sh = self.surface.get_height()

        # Dim overlay
        overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        self.surface.blit(overlay, (0, 0))

        # Winner banner
        w = state.winner
        if w is not None:
            name = PLAYER_NAMES[w % len(PLAYER_NAMES)]
            pcol = PLAYER_COLORS[w % len(PLAYER_COLORS)]
            text = f"{name} Wins!"
        else:
            pcol = HUD_TEXT
            text = "Draw!"

        label = self.big_font.render(text, True, pcol)
        rect = label.get_rect(center=(sw // 2, sh // 2 - 20))
        # shadow
        shadow = self.big_font.render(text, True, (0, 0, 0))
        self.surface.blit(shadow, rect.move(2, 2))
        self.surface.blit(label, rect)

        sub = self.hud_font.render("Press R to restart", True, HUD_DIM)
        sr = sub.get_rect(center=(sw // 2, sh // 2 + 25))
        self.surface.blit(sub, sr)
