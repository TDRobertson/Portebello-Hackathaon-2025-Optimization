import pygame
import heapq
import math
import sys

pygame.init()

# --- GRID SETTINGS ---
ROWS, COLS = 20, 25
CELL_SIZE = 30
WIDTH, HEIGHT = COLS * CELL_SIZE, ROWS * CELL_SIZE + 36  # extra for HUD

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Warehouse Path Optimization — Polished UI")

# --- COLORS (softer theme) ---
BG = (240, 248, 255)         # background
AISLE = (225, 232, 240)      # aisle cell
SHELF = (70, 75, 90)         # shelf cell (dark)
SHELF_SHADOW = (40, 42, 50)
GRID_LINE = (245, 248, 252)
EXPLORED = (255, 247, 210)   # visited
FRONTIER = (200, 170, 255)   # open set
PATH = (255, 235, 160)       # final path
AGENT = (235, 90, 90)        # agent circle
START_COLOR = (120, 230, 150)
TARGET_COLOR = (110, 160, 255)
HUD_BAR = (250, 250, 250)
HUD_LINE = (230, 230, 230)
TEXT_COLOR = (40, 40, 40)

clock = pygame.time.Clock()
FONT = pygame.font.SysFont("Segoe UI", 18) if pygame.font.get_init() else pygame.font.SysFont(None, 18)

# --- MAP GENERATION ---
warehouse_map = [[0 for _ in range(COLS)] for _ in range(ROWS)]

# Create shelves (1) and aisles (0) with spacing pattern (same as original logic)
for r in range(2, ROWS - 2):
    if r % 4 == 0:
        continue
    for c in range(2, COLS - 2):
        if c % 3 == 0:
            warehouse_map[r][c] = 1

# --- UTILITIES ---
def in_bounds(cell):
    r, c = cell
    return 0 <= r < ROWS and 0 <= c < COLS

def neighbors(cell):
    r, c = cell
    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            yield (nr, nc)

def draw_rounded_rect(surface, color, rect, radius=6):
    pygame.draw.rect(surface, color, rect, border_radius=radius)

# --- DRAW FUNCTION ---
def draw_map(agent_pixel=None, agent_cell=None, start=None, targets=None, path=None, closed=None, openset=None, msg=""):
    # background
    screen.fill(BG)

    # grid area
    grid_surface = pygame.Surface((COLS * CELL_SIZE, ROWS * CELL_SIZE), pygame.SRCALPHA)
    for r in range(ROWS):
        for c in range(COLS):
            x, y = c * CELL_SIZE, r * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

            if warehouse_map[r][c] == 1:
                # shelf with shadow and slight gradient look
                shadow_rect = rect.copy()
                shadow_rect.x += 2
                shadow_rect.y += 2
                pygame.draw.rect(grid_surface, SHELF_SHADOW, shadow_rect, border_radius=5)
                pygame.draw.rect(grid_surface, SHELF, rect, border_radius=5)
            else:
                pygame.draw.rect(grid_surface, AISLE, rect, border_radius=5)

            # subtle grid lines
            pygame.draw.rect(grid_surface, GRID_LINE, rect, 1, border_radius=5)

    # draw explored (closed) set
    if closed:
        for (r, c) in closed:
            pygame.draw.rect(grid_surface, EXPLORED, (c*CELL_SIZE+2, r*CELL_SIZE+2, CELL_SIZE-4, CELL_SIZE-4), border_radius=4)

    # draw frontier (open set)
    if openset:
        for (r, c) in openset:
            pygame.draw.rect(grid_surface, FRONTIER, (c*CELL_SIZE+4, r*CELL_SIZE+4, CELL_SIZE-8, CELL_SIZE-8), border_radius=4)

    # draw path (highlight)
    if path:
        for (r, c) in path:
            pygame.draw.rect(grid_surface, PATH, (c*CELL_SIZE+3, r*CELL_SIZE+3, CELL_SIZE-6, CELL_SIZE-6), border_radius=5)

    # draw start and targets
    if targets:
        for (r, c) in targets:
            pygame.draw.rect(grid_surface, TARGET_COLOR, (c*CELL_SIZE+5, r*CELL_SIZE+5, CELL_SIZE-10, CELL_SIZE-10), border_radius=5)
    if start:
        pygame.draw.rect(grid_surface, START_COLOR, (start[1]*CELL_SIZE+4, start[0]*CELL_SIZE+4, CELL_SIZE-8, CELL_SIZE-8), border_radius=5)

    # blit the prepared grid onto screen (keeps layers tidy)
    screen.blit(grid_surface, (0, 36))

    # draw agent as circle (pixel position preferred for smooth interpolation)
    if agent_pixel:
        ax, ay = agent_pixel
        pygame.draw.circle(screen, AGENT, (int(ax), int(ay)), CELL_SIZE//3)
        # small glow
        glow = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(glow, (AGENT[0], AGENT[1], AGENT[2], 50), (CELL_SIZE//2, CELL_SIZE//2), CELL_SIZE//2)
        screen.blit(glow, (int(ax - CELL_SIZE//2), int(ay - CELL_SIZE//2)), special_flags=pygame.BLEND_RGBA_ADD)
    elif agent_cell:
        # fallback to grid-aligned agent
        r, c = agent_cell
        cx = c * CELL_SIZE + CELL_SIZE // 2
        cy = r * CELL_SIZE + CELL_SIZE // 2 + 36
        pygame.draw.circle(screen, AGENT, (cx, cy), CELL_SIZE//3)

    # HUD (top bar)
    pygame.draw.rect(screen, HUD_BAR, (0, 0, WIDTH, 36))
    pygame.draw.line(screen, HUD_LINE, (0, 36), (WIDTH, 36))
    text = FONT.render(msg, True, TEXT_COLOR)
    screen.blit(text, (10, 8))

    # footer summary (targets count)
    if targets:
        footer = FONT.render(f"Targets: {len(targets)}  |  Press ESC to quit", True, TEXT_COLOR)
        screen.blit(footer, (WIDTH - 260, 8))

    pygame.display.flip()

# --- HEURISTIC ---
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# --- A* WITH VISUALIZATION ---
def a_star_visualized(start, goal, visualize=True, speed_wait=6):
    open_heap = []
    heapq.heappush(open_heap, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    closed_set = set()
    open_set = {start}

    while open_heap:
        # handle events so window remains responsive
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()

        _, current = heapq.heappop(open_heap)
        open_set.discard(current)

        if current in closed_set:
            continue
        closed_set.add(current)

        # visualize current frontier & closed
        if visualize:
            # create a list of open nodes for drawing
            open_list = list(open_set)
            draw_map(
                agent_pixel=None,
                agent_cell=None,
                start=start,
                targets=[goal],
                path=None,
                closed=closed_set,
                openset=open_list,
                msg="🔍 Searching — A* expanding..."
            )
            pygame.time.delay(speed_wait)

        if current == goal:
            # reconstruct path
            path = [goal]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            # return excluding start? keep full path from start to goal
            return path

        for nb in neighbors(current):
            nr, nc = nb
            if warehouse_map[nr][nc] == 1:
                continue
            tentative = g_score[current] + 1
            if tentative < g_score.get(nb, float('inf')):
                came_from[nb] = current
                g_score[nb] = tentative
                f = tentative + heuristic(nb, goal)
                f_score[nb] = f
                if nb not in open_set:
                    heapq.heappush(open_heap, (f, nb))
                    open_set.add(nb)

    return []

# --- TARGET SELECTION (mouse) ---
def select_targets(start):
    targets = []
    selecting = True
    draw_map(agent_cell=None, start=start, targets=targets, msg="🖱 Click free cells to add targets, press ENTER when done")
    while selecting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    selecting = False
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                # ignore clicks on HUD
                if my < 36:
                    continue
                col = mx // CELL_SIZE
                row = (my - 36) // CELL_SIZE
                if in_bounds((row, col)) and warehouse_map[row][col] == 0:
                    if (row, col) not in targets and (row, col) != start:
                        targets.append((row, col))
                    elif (row, col) in targets:
                        targets.remove((row, col))

        draw_map(agent_cell=None, start=start, targets=targets, msg="🖱 Click free cells to add targets, press ENTER when done")
        clock.tick(60)
    return targets

# --- SIMPLE GREEDY ORDERING (using A* distances, quiet) ---
def find_best_order(targets, start):
    remaining = targets[:]
    order = []
    current = start
    while remaining:
        best_target = None
        best_dist = float('inf')
        for t in list(remaining):
            path = a_star_visualized(current, t, visualize=False)
            if path and len(path) < best_dist:
                best_dist = len(path)
                best_target = t
        if best_target is None:
            break
        order.append(best_target)
        remaining.remove(best_target)
        current = best_target
    return order

# --- FADE IN TRANSITION ---
def fade_in(duration_ms=400):
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.fill((0,0,0))
    steps = 16
    wait = max(1, duration_ms // steps)
    for i in range(steps + 1):
        alpha = int(255 * (1 - i / steps))
        overlay.set_alpha(alpha)
        draw_map(msg="Welcome — select targets", agent_cell=None, start=(1,1), targets=[], path=None, closed=None, openset=None)
        screen.blit(overlay, (0, 0))
        pygame.display.flip()
        pygame.time.delay(wait)

# --- SMOOTH ANIMATION: agent moves pixel-by-pixel, handles quit events ---
def animate_path(full_path, start, targets, per_cell_frames=12):
    if not full_path:
        return
    # agent start pixel (center)
    agent_x = start[1] * CELL_SIZE + CELL_SIZE // 2
    agent_y = start[0] * CELL_SIZE + CELL_SIZE // 2 + 36  # account for HUD offset

    # draw final combined path highlight as background as we move
    for idx, cell in enumerate(full_path):
        target_r, target_c = cell
        tx = target_c * CELL_SIZE + CELL_SIZE // 2
        ty = target_r * CELL_SIZE + CELL_SIZE // 2 + 36

        # interpolate with easing for nice motion
        for frame in range(per_cell_frames):
            t = (frame + 1) / per_cell_frames
            # ease in-out (smooth)
            t_ease = -(math.cos(math.pi * t) - 1) / 2
            cur_x = agent_x + (tx - agent_x) * t_ease
            cur_y = agent_y + (ty - agent_y) * t_ease

            # event handling
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

            draw_map(agent_pixel=(cur_x, cur_y), start=start, targets=targets, path=full_path, closed=None, openset=None,
                     msg="🤖 Moving along optimal route...")
            clock.tick(60)

        agent_x, agent_y = tx, ty

    # final state
    draw_map(agent_pixel=(agent_x, agent_y), start=start, targets=targets, path=full_path, closed=None, openset=None,
             msg="✅ Path complete!")
    pygame.time.delay(1000)

# --- MAIN ---
def main():
    start = (1, 1)
    fade_in(450)

    draw_map(agent_cell=None, start=start, targets=[], msg="🖱 Click to select targets, press ENTER to start")
    targets = select_targets(start)

    if not targets:
        print("No targets selected.")
        return

    # order targets with greedy heuristic (quiet)
    draw_map(msg="🔢 Computing visit order...", start=start, targets=targets)
    best_order = find_best_order(targets, start)

    # compute full path by concatenating A* segments with visualization for each segment
    full_path = []
    current = start
    for idx, target in enumerate(best_order):
        # visualize A* search for this segment (slower, visible)
        segment = a_star_visualized(current, target, visualize=True, speed_wait=6)
        if not segment:
            print("A segment was unreachable:", target)
            continue
        # if the segment starts with current, skip duplicated first step to avoid repeating cell
        if full_path and segment and segment[0] == full_path[-1]:
            full_path.extend(segment[1:])
        else:
            full_path.extend(segment)
        current = target

    # animate the agent along the resulting full path
    animate_path(full_path, start, best_order)

    # finished — wait a bit and exit gracefully
    pygame.time.delay(400)
    pygame.quit()

if __name__ == "__main__":
    main()
