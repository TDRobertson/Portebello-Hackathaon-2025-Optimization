"""
warehouse_auto_targets_weighted.py

Same polished Pygame UI, now with:
- Random weights assigned to each location
- Weighted item display on HUD
- Optimal path found using greedy A* search
"""

import pygame
import heapq
import math
import sys
import random
from collections import deque

pygame.init()

# --- GRID SETTINGS ---
ROWS, COLS = 20, 25
CELL_SIZE = 30
WIDTH, HEIGHT = COLS * CELL_SIZE, ROWS * CELL_SIZE + 36  # extra for HUD

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Warehouse Path Optimization — Weighted")

# --- COLORS (same soft theme) ---
BG = (240, 248, 255)
AISLE = (225, 232, 240)
SHELF = (70, 75, 90)
SHELF_SHADOW = (40, 42, 50)
GRID_LINE = (245, 248, 252)
EXPLORED = (255, 247, 210)
FRONTIER = (200, 170, 255)
PATH = (255, 235, 160)
AGENT = (235, 90, 90)
START_COLOR = (120, 230, 150)
TARGET_COLOR = (110, 160, 255)
HUD_BAR = (250, 250, 250)
HUD_LINE = (230, 230, 230)
TEXT_COLOR = (40, 40, 40)

clock = pygame.time.Clock()
FONT = pygame.font.SysFont("Segoe UI", 18) if pygame.font.get_init() else pygame.font.SysFont(None, 18)

# --- MAP GENERATION ---
warehouse_map = [[0 for _ in range(COLS)] for _ in range(ROWS)]
for r in range(2, ROWS - 2):
    if r % 4 == 0:
        continue
    for c in range(2, COLS - 2):
        if c % 3 == 0:
            warehouse_map[r][c] = 1

# --- UTILITIES ---
def neighbors(cell):
    r, c = cell
    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            yield (nr, nc)

def find_nearest_free(start_cell):
    if warehouse_map[start_cell[0]][start_cell[1]] == 0:
        return start_cell
    visited = {start_cell}
    dq = deque([start_cell])
    while dq:
        cell = dq.popleft()
        for nb in neighbors(cell):
            if nb in visited:
                continue
            visited.add(nb)
            r, c = nb
            if warehouse_map[r][c] == 0:
                return nb
            dq.append(nb)
    return (1, 1)

# --- DRAW FUNCTION (unchanged visuals) ---
def draw_map(agent_pixel=None, agent_cell=None, start=None, targets=None, path=None,
              closed=None, openset=None, msg="", show_codes=None):
    screen.fill(BG)
    grid_surface = pygame.Surface((COLS * CELL_SIZE, ROWS * CELL_SIZE), pygame.SRCALPHA)

    for r in range(ROWS):
        for c in range(COLS):
            x, y = c * CELL_SIZE, r * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            if warehouse_map[r][c] == 1:
                shadow_rect = rect.copy(); shadow_rect.x += 2; shadow_rect.y += 2
                pygame.draw.rect(grid_surface, SHELF_SHADOW, shadow_rect, border_radius=5)
                pygame.draw.rect(grid_surface, SHELF, rect, border_radius=5)
            else:
                pygame.draw.rect(grid_surface, AISLE, rect, border_radius=5)
            pygame.draw.rect(grid_surface, GRID_LINE, rect, 1, border_radius=5)

    if closed:
        for (r, c) in closed:
            pygame.draw.rect(grid_surface, EXPLORED, (c*CELL_SIZE+2, r*CELL_SIZE+2, CELL_SIZE-4, CELL_SIZE-4), border_radius=4)
    if openset:
        for (r, c) in openset:
            pygame.draw.rect(grid_surface, FRONTIER, (c*CELL_SIZE+4, r*CELL_SIZE+4, CELL_SIZE-8, CELL_SIZE-8), border_radius=4)
    if path:
        for (r, c) in path:
            pygame.draw.rect(grid_surface, PATH, (c*CELL_SIZE+3, r*CELL_SIZE+3, CELL_SIZE-6, CELL_SIZE-6), border_radius=5)
    if targets:
        for (r, c) in targets:
            pygame.draw.rect(grid_surface, TARGET_COLOR, (c*CELL_SIZE+5, r*CELL_SIZE+5, CELL_SIZE-10, CELL_SIZE-10), border_radius=5)
    if start:
        pygame.draw.rect(grid_surface, START_COLOR, (start[1]*CELL_SIZE+4, start[0]*CELL_SIZE+4, CELL_SIZE-8, CELL_SIZE-8), border_radius=5)

    screen.blit(grid_surface, (0, 36))

    # agent
    if agent_pixel:
        ax, ay = agent_pixel
        pygame.draw.circle(screen, AGENT, (int(ax), int(ay)), CELL_SIZE//3)
    elif agent_cell:
        r, c = agent_cell
        cx = c * CELL_SIZE + CELL_SIZE // 2
        cy = r * CELL_SIZE + CELL_SIZE // 2 + 36
        pygame.draw.circle(screen, AGENT, (cx, cy), CELL_SIZE//3)

    # HUD
    pygame.draw.rect(screen, HUD_BAR, (0, 0, WIDTH, 36))
    pygame.draw.line(screen, HUD_LINE, (0, 36), (WIDTH, 36))
    screen.blit(FONT.render(msg, True, TEXT_COLOR), (10, 8))

    if show_codes:
        # now includes weights
        codes_display = " | ".join([f"{code} ({w:.1f}kg)" for code, w in show_codes])
        if len(codes_display) > 85:
            codes_display = codes_display[:82] + "..."
        screen.blit(FONT.render(codes_display, True, TEXT_COLOR), (10, 40))

    pygame.display.flip()

# --- PATHFINDING ---
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def a_star_visualized(start, goal, visualize=True, speed_wait=6):
    open_heap = []
    heapq.heappush(open_heap, (0, start))
    came_from = {}
    g = {start: 0}
    f = {start: heuristic(start, goal)}
    closed, open_set = set(), {start}

    while open_heap:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.quit(); sys.exit()

        _, cur = heapq.heappop(open_heap)
        if cur in closed: continue
        closed.add(cur)

        if visualize:
            draw_map(start=start, targets=[goal], closed=closed, openset=open_set, msg="🔍 Searching path...")
            pygame.time.delay(speed_wait)

        if cur == goal:
            path = [goal]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            return path

        for nb in neighbors(cur):
            r, c = nb
            if warehouse_map[r][c] == 1: continue
            t = g[cur] + 1
            if t < g.get(nb, float('inf')):
                came_from[nb] = cur
                g[nb] = t
                f[nb] = t + heuristic(nb, goal)
                heapq.heappush(open_heap, (f[nb], nb))
                open_set.add(nb)
    return []

# --- LOCATION & WEIGHT GENERATION ---
def generate_weighted_locations(n=6):
    items = []
    for _ in range(n):
        code = f"B{random.randint(1,9)}.J{random.randint(1,4)}.{random.randint(1,99):02}.{random.randint(1,5):02}"
        weight = random.uniform(5, 100)
        items.append((code, weight))
    items.sort(key=lambda x: x[1], reverse=True)  # sort by weight descending
    return items

def location_to_grid(loc):
    parts = loc.split(".")
    b = int(parts[0][1:]); j = int(parts[1][1:]); s = int(parts[2]); f = int(parts[3])
    row = 2 + (j - 1) * 4 + (s % 3)
    col = 2 + (b - 1) * 2 + (f % 3)
    row, col = min(max(row,0),ROWS-1), min(max(col,0),COLS-1)
    if warehouse_map[row][col] == 1:
        row, col = find_nearest_free((row, col))
    return (row, col)

# --- ORDERING FUNCTION (use A* greedy same as before) ---
def find_best_order(targets, start):
    remaining = targets[:]
    order = []
    current = start
    while remaining:
        best_target, best_dist = None, float('inf')
        for t in remaining:
            path = a_star_visualized(current, t, visualize=False)
            if path and len(path) < best_dist:
                best_target, best_dist = t, len(path)
        if best_target:
            order.append(best_target)
            remaining.remove(best_target)
            current = best_target
        else:
            break
    return order

# --- ANIMATION ---
def animate_path(full_path, start, targets, per_cell_frames=12):
    if not full_path:
        return
    agent_x = start[1]*CELL_SIZE + CELL_SIZE//2
    agent_y = start[0]*CELL_SIZE + CELL_SIZE//2 + 36
    for cell in full_path:
        tx = cell[1]*CELL_SIZE + CELL_SIZE//2
        ty = cell[0]*CELL_SIZE + CELL_SIZE//2 + 36
        for f in range(per_cell_frames):
            t = (f+1)/per_cell_frames
            t_ease = -(math.cos(math.pi*t)-1)/2
            cx = agent_x + (tx-agent_x)*t_ease
            cy = agent_y + (ty-agent_y)*t_ease
            draw_map(agent_pixel=(cx,cy), start=start, targets=targets, path=full_path, msg="🤖 Moving along optimal route...")
            clock.tick(60)
        agent_x, agent_y = tx, ty
    draw_map(agent_pixel=(agent_x, agent_y), start=start, targets=targets, path=full_path, msg="✅ Path complete!")
    pygame.time.delay(800)

# --- MAIN ---
def main():
    start = (1, 1)

    # weighted codes
    items = generate_weighted_locations(6)
    codes = [code for code, _ in items]
    targets = [location_to_grid(code) for code in codes]

    draw_map(start=start, targets=targets, msg="📦 Weighted targets loaded — computing optimal path...", show_codes=items)
    pygame.time.delay(700)

    # compute optimal path (greedy)
    best_order = find_best_order(targets, start)

    # full path
    full_path = []
    current = start
    for t in best_order:
        seg = a_star_visualized(current, t, visualize=True, speed_wait=6)
        if seg:
            if full_path and seg[0] == full_path[-1]:
                full_path.extend(seg[1:])
            else:
                full_path.extend(seg)
            current = t

    animate_path(full_path, start, best_order)
    draw_map(start=start, targets=best_order, path=full_path, msg="✅ Weighted route complete!", show_codes=items)
    pygame.time.delay(1400)
    pygame.quit()

if __name__ == "__main__":
    main()
