import pygame
import heapq
import itertools

pygame.init()

# --- GRID SETTINGS ---
ROWS, COLS = 20, 25
CELL_SIZE = 30
WIDTH, HEIGHT = COLS * CELL_SIZE, ROWS * CELL_SIZE

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Warehouse Path Optimization")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (50, 100, 255)
GREEN = (50, 255, 100)
RED = (255, 60, 60)
YELLOW = (255, 255, 100)
LIGHT_YELLOW = (255, 255, 180)
PURPLE = (180, 50, 255)

clock = pygame.time.Clock()

# --- MAP GENERATION ---
warehouse_map = [[0 for _ in range(COLS)] for _ in range(ROWS)]

# Create shelves (1) and aisles (0)
for r in range(2, ROWS - 2):
    if r % 4 == 0:
        continue  # horizontal spacing every 4 rows
    for c in range(2, COLS - 2):
        if c % 3 == 0:  # vertical shelf every 3rd column
            warehouse_map[r][c] = 1

# --- DRAW FUNCTION ---
def draw_map(agent_pos, start, targets, path=None, explored=None, msg=""):
    screen.fill(WHITE)
    for r in range(ROWS):
        for c in range(COLS):
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = GRAY if warehouse_map[r][c] == 0 else BLACK
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, WHITE, rect, 1)

    if explored:
        for (r, c) in explored:
            pygame.draw.rect(screen, LIGHT_YELLOW, (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    if path:
        for (r, c) in path:
            pygame.draw.rect(screen, YELLOW, (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    if start:
        pygame.draw.rect(screen, GREEN, (start[1] * CELL_SIZE, start[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    if targets:
        for (r, c) in targets:
            pygame.draw.rect(screen, BLUE, (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    if agent_pos:
        pygame.draw.rect(screen, RED, (agent_pos[1] * CELL_SIZE, agent_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    if msg:
        font = pygame.font.SysFont(None, 22)
        text = font.render(msg, True, (0, 0, 0))
        screen.blit(text, (10, 5))

    pygame.display.update()


# --- PATHFINDING (A*) with Visualization ---
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_visualized(start, goal, visualize=True):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    explored = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        explored.add(current)

        # Faster visualization — 10x speedup
        if visualize:
            draw_map(None, start, [goal], None, explored, "🔍 Searching for best path...")
            pygame.time.wait(3)  # much faster

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()

            # Show final path clearly
            draw_map(None, start, [goal], path, explored, "✨ Optimal path found!")
            pygame.time.wait(300)
            return path

        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)
            if 0 <= nr < ROWS and 0 <= nc < COLS and warehouse_map[nr][nc] == 0:
                tentative = g_score[current] + 1
                if tentative < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f_score[neighbor] = tentative + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []


# --- TARGET SELECTION ---
def select_targets(start):
    targets = []
    selecting = True
    while selecting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                row, col = my // CELL_SIZE, mx // CELL_SIZE
                if 0 <= row < ROWS and 0 <= col < COLS and warehouse_map[row][col] == 0:
                    if (row, col) not in targets:
                        targets.append((row, col))
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                selecting = False

        draw_map(None, start, targets, None, None, "🖱 Click to select targets, press ENTER to confirm")
        clock.tick(60)
    return targets

# --- OPTIMIZE VISIT ORDER (Greedy Nearest-Neighbor) ---
def find_best_order(targets, start):
    remaining = targets[:]
    order = []
    current = start

    while remaining:
        # Find nearest target using path length (A* distance)
        best_target = None
        best_dist = float('inf')
        for target in remaining:
            path = a_star_visualized(current, target, visualize=False)
            if path and len(path) < best_dist:
                best_dist = len(path)
                best_target = target

        if best_target is None:
            break  # no reachable targets
        order.append(best_target)
        remaining.remove(best_target)
        current = best_target

    return order



# --- AGENT ANIMATION ---
def animate_path(full_path, start, targets):
    agent_pos = start
    for step in full_path:
        agent_pos = step
        draw_map(agent_pos, start, targets, full_path, None, "🤖 Moving along optimal route...")
        pygame.time.wait(100)  # slower, smoother motion
    draw_map(agent_pos, start, targets, full_path, None, "✅ Path Complete!")
    pygame.time.wait(1200)


# --- MAIN LOOP ---
def main():
    start = (1, 1)
    draw_map(None, start, [], None, None, "🖱 Click to select targets, press ENTER to start")
    targets = select_targets(start)

    if not targets:
        print("No targets selected.")
        return

    best_order = find_best_order(targets, start)
    full_path = []
    current = start
    for target in best_order:
        segment = a_star_visualized(current, target, visualize=True)
        full_path.extend(segment)
        current = target

    animate_path(full_path, start, best_order)
    pygame.quit()


if __name__ == "__main__":
    main()
