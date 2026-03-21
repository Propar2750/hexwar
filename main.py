"""HexWar — demo: render a hex grid with doubled-width coordinates."""

import pygame

from hex_core import HexCoord, pixel_to_hex
from hex_grid import HexGrid
from renderer import BG_COLOR, HexRenderer

SCREEN_W, SCREEN_H = 900, 700
GRID_W, GRID_H = 8, 6
HEX_SIZE = 36


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("HexWar — Doubled Coordinates")
    clock = pygame.time.Clock()

    grid = HexGrid(GRID_W, GRID_H)

    # Center the grid on screen
    origin_x = SCREEN_W / 2 - (GRID_W - 1) * HEX_SIZE * 0.866
    origin_y = SCREEN_H / 2 - (GRID_H - 1) * HEX_SIZE * 0.75
    renderer = HexRenderer(screen, hex_size=HEX_SIZE, origin=(origin_x, origin_y))

    hovered: HexCoord | None = None
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Find which hex the mouse is over
        mx, my = pygame.mouse.get_pos()
        candidate = pixel_to_hex(
            mx - renderer.origin_x,
            my - renderer.origin_y,
            HEX_SIZE,
        )
        hovered = candidate if candidate in grid else None

        # Draw
        screen.fill(BG_COLOR)
        renderer.draw_grid(grid, highlight=hovered, show_coords=True)

        # Show hovered coord in top-left
        if hovered is not None:
            info_font = pygame.font.SysFont("consolas", 18)
            info = info_font.render(
                f"Hover: ({hovered.col}, {hovered.row})  "
                f"Neighbors: {len(grid.neighbors_of(hovered))}",
                True,
                (220, 220, 220),
            )
            screen.blit(info, (12, 12))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
