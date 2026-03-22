"""HexWar — demo: render a hex grid with A* pathfinding."""

import pygame

from hex_core import HexCoord, pixel_to_hex
from hex_grid import HexGrid, Terrain
from map_generator import generate_terrain
from pathfinding import astar
from renderer import BG_COLOR, HexRenderer

SCREEN_W, SCREEN_H = 1200, 900
GRID_W, GRID_H = 30, 30
HEX_SIZE = 16


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("HexWar — A* Pathfinding Demo")
    clock = pygame.time.Clock()

    grid = HexGrid(GRID_W, GRID_H)
    generate_terrain(grid, num_ranges=8, range_steps=12)

    # Center the grid on screen
    origin_x = SCREEN_W / 2 - (GRID_W - 1) * HEX_SIZE * 0.866
    origin_y = SCREEN_H / 2 - (GRID_H - 1) * HEX_SIZE * 0.75
    renderer = HexRenderer(screen, hex_size=HEX_SIZE, origin=(origin_x, origin_y))

    hovered: HexCoord | None = None
    start: HexCoord | None = None
    goal: HexCoord | None = None
    path: tuple[HexCoord, ...] | tuple[()] = ()
    path_cost: float = 0.0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                # Regenerate terrain
                generate_terrain(grid, num_ranges=8, range_steps=12)
                start = goal = None
                path = ()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if hovered is not None:
                    if start is None:
                        start = hovered
                        goal = None
                        path = ()
                    elif goal is None:
                        goal = hovered
                        result = astar(grid, start, goal)
                        path = result.path
                        path_cost = result.cost
                    else:
                        # Reset: start new selection
                        start = hovered
                        goal = None
                        path = ()

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
        renderer.draw_grid(grid, highlight=hovered, show_coords=False)

        # Draw path on top of the grid
        if path:
            renderer.draw_path(path, show_coords=False)

        # HUD info
        info_font = pygame.font.SysFont("consolas", 18)
        lines: list[str] = []

        if hovered is not None:
            tile = grid[hovered]
            lines.append(
                f"Hover: ({hovered.col}, {hovered.row})  "
                f"Terrain: {tile.terrain.value}"
            )

        if start is not None and goal is None:
            lines.append(f"Start: ({start.col}, {start.row})  Click goal...")
        elif start is not None and goal is not None:
            if path:
                lines.append(
                    f"Path: ({start.col},{start.row}) -> ({goal.col},{goal.row})  "
                    f"Steps: {len(path) - 1}  Cost: {path_cost:.1f}"
                )
            else:
                lines.append("No path found!")
            lines.append("Click to select new start")
        else:
            lines.append("Click a hex to set start")

        lines.append("R = regenerate terrain")

        for i, line in enumerate(lines):
            surf = info_font.render(line, True, (220, 220, 220))
            screen.blit(surf, (12, 12 + i * 24))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
