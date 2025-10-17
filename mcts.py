import csv
import sys
import time
import random
import math
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# --- Load & sort dims globally ---
def load_dims_from_csv(path: str) -> List[Tuple[int,int,int]]:
    dims: List[Tuple[int,int,int]] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for lineno, row in enumerate(reader, start=1):
            if len(row) < 3:
                print(f"Skipping line {lineno}: not enough values")
                continue
            try:
                w, d, h = map(int, row[:3])
                if w <= 0 or d <= 0 or h <= 0:
                    raise ValueError
            except ValueError:
                print(f"Skipping line {lineno}: invalid dims {row}")
                continue
            dims.append((w, d, h))
    return dims

if len(sys.argv) != 2:
    print("Usage: python mcts_place_boxes.py boxes.csv")
    sys.exit(1)

dims = load_dims_from_csv(sys.argv[1])
if not dims:
    print("No valid boxes to place."); sys.exit(1)

# Sort by descending volume
dims.sort(key=lambda t: t[0]*t[1]*t[2], reverse=True)
TOTAL_BOXES = len(dims)


# --- World & Box definitions ---
class Box:
    __slots__ = ("x","y","z","w","d","h","volume")
    def __init__(self, x:int, y:int, z:int, w:int, d:int, h:int) -> None:
        self.x, self.y, self.z = x, y, z
        self.w, self.d, self.h = w, d, h
        self.volume = w * d * h

    def overlaps(self, other: "Box") -> bool:
        return (
            self.x < other.x + other.w and
            self.x + self.w > other.x and
            self.y < other.y + other.d and
            self.y + self.d > other.y and
            self.z < other.z + other.h and
            self.z + self.h > other.z
        )


class Space:
    def __init__(self, width:int, depth:int, height:int) -> None:
        self.width, self.depth, self.height = width, depth, height
        self.boxes: List[Box] = []

    def find_drop_height(self, x:int, y:int, w:int, d:int) -> int:
        """Return highest z at which a new box of footprint (x..x+w, y..y+d) can rest."""
        drop_z = 0
        for existing_box in self.boxes:
            # if footprints overlap in XY
            if not (x + w <= existing_box.x or
                    existing_box.x + existing_box.w <= x or
                    y + d <= existing_box.y or
                    existing_box.y + existing_box.d <= y):
                drop_z = max(drop_z, existing_box.z + existing_box.h)
        return drop_z

    def legal_xy_positions(self, w:int, d:int, h:int) -> List[Tuple[int,int]]:
        """
        Enumerate all legal (x,y) placements for a w×d×h box:
         - in‐bounds
         - under ceiling
         - no 3D overlap
         - full‐footprint support (no overhang)
         - no larger‐on‐smaller support
        """
        legal_positions: List[Tuple[int,int]] = []
        for x in range(0, self.width - w + 1):
            for y in range(0, self.depth - d + 1):
                z_floor = self.find_drop_height(x, y, w, d)
                # under ceiling?
                if z_floor + h > self.height:
                    continue
                candidate = Box(x, y, z_floor, w, d, h)
                # 3D overlap?
                if any(candidate.overlaps(e) for e in self.boxes):
                    continue
                # full‐footprint support
                if z_floor > 0:
                    fully_supported = True
                    for xi in range(x, x + w):
                        for yi in range(y, y + d):
                            # must have some box whose top == z_floor at (xi,yi)
                            if not any(
                                (eb.z + eb.h == z_floor and
                                 eb.x <= xi < eb.x + eb.w and
                                 eb.y <= yi < eb.y + eb.d)
                                for eb in self.boxes
                            ):
                                fully_supported = False
                                break
                        if not fully_supported:
                            break
                    if not fully_supported:
                        continue
                # no larger‐on‐smaller
                violates_support = False
                for supporting_box in self.boxes:
                    if supporting_box.z + supporting_box.h == z_floor:
                        if w * d * h > supporting_box.volume:
                            violates_support = True
                            break
                if violates_support:
                    continue

                legal_positions.append((x, y))
        return legal_positions

    def place_at(self, x:int, y:int, w:int, d:int, h:int) -> Box:
        """Drop and permanently place a box of size w×d×h at horizontal (x,y)."""
        z_floor = self.find_drop_height(x, y, w, d)
        new_box = Box(x, y, z_floor, w, d, h)
        self.boxes.append(new_box)
        return new_box


# --- MCTS Node ---
class Node:
    __slots__ = ("parent","action","box_index","placed_boxes",
                 "children","visits","total_reward","_untried_actions")

    def __init__(self,
                 parent: Optional["Node"],
                 action: Optional[Tuple[int,int]],
                 box_index: int,
                 placed_boxes: List[Box]) -> None:
        self.parent = parent
        self.action = action               # the (x,y) that led here
        self.box_index = box_index         # next box to place
        self.placed_boxes = placed_boxes   # current board state
        self.children: List["Node"] = []
        self.visits = 0
        self.total_reward = 0.0
        self._untried_actions: Optional[List[Tuple[int,int]]] = None

    def untried_actions(self) -> List[Tuple[int,int]]:
        """Compute (and cache) all legal actions at this node."""
        if self._untried_actions is None:
            if self.box_index >= TOTAL_BOXES:
                self._untried_actions = []
            else:
                w,d,h = dims[self.box_index]
                sim_space = Space(SPACE_W, SPACE_D, SPACE_H)
                sim_space.boxes = list(self.placed_boxes)
                self._untried_actions = sim_space.legal_xy_positions(w,d,h)
        return self._untried_actions

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions()) == 0

    def is_terminal(self) -> bool:
        # terminal if we've placed all boxes
        return self.box_index >= TOTAL_BOXES


def uct_select(children: List[Node]) -> Optional[Node]:
    """
    Select a child with highest UCT value:
       Q_i / N_i  + c * sqrt( ln(N_parent) / N_i )
    Returns None if children is empty.
    """
    if not children:
        return None
    parent_visits = children[0].parent.visits if children[0].parent else 1
    log_parent = math.log(parent_visits)
    C = 1.4
    def uct_value(child: Node) -> float:
        return (child.total_reward / child.visits) + \
               C * math.sqrt(log_parent / child.visits)
    return max(children, key=uct_value)


def expand(parent_node: Node) -> Node:
    """Take one untried action and add a new child node."""
    w,d,h = dims[parent_node.box_index]
    action_list = parent_node.untried_actions()
    chosen_action = action_list.pop(random.randrange(len(action_list)))
    sim_space = Space(SPACE_W, SPACE_D, SPACE_H)
    sim_space.boxes = list(parent_node.placed_boxes)
    placed_box = sim_space.place_at(*chosen_action, w,d,h)
    child_node = Node(parent_node,
                      chosen_action,
                      parent_node.box_index + 1,
                      parent_node.placed_boxes + [placed_box])
    parent_node.children.append(child_node)
    return child_node


def rollout(box_state: Tuple[int,List[Box]]) -> float:
    """
    Simulate random play from given state until no move possible.
    Returns reward = (boxes placed) / TOTAL_BOXES.
    """
    next_index, placed = box_state
    sim_space = Space(SPACE_W, SPACE_D, SPACE_H)
    sim_space.boxes = list(placed)
    placed_count = len(placed)
    for idx in range(next_index, TOTAL_BOXES):
        w,d,h = dims[idx]
        legal = sim_space.legal_xy_positions(w,d,h)
        if not legal:
            break
        sim_space.place_at(*random.choice(legal), w,d,h)
        placed_count += 1
    return placed_count / TOTAL_BOXES


def backpropagate(node: Optional[Node], reward: float) -> None:
    """Propagate reward up to root."""
    current = node
    while current is not None:
        current.visits += 1
        current.total_reward += reward
        current = current.parent


# --- Main MCTS loop ---
SPACE_W, SPACE_D, SPACE_H = 15, 15, 10
root = Node(None, None, 0, [])  # box_index=0, no boxes placed
final_placements: List[Box] = []

print(f"Running MCTS up to {TOTAL_BOXES} boxes...")

while not root.is_terminal():
    start_time = time.time()
    # 0.1s of MCTS
    while time.time() - start_time < 1.0:
        # 1) Selection
        node = root
        while not node.is_terminal() and node.is_fully_expanded():
            selected = uct_select(node.children)
            if selected is None:
                break
            node = selected

        # 2) Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            node = expand(node)

        # 3) Simulation
        reward = rollout((node.box_index, node.placed_boxes))

        # 4) Backpropagation
        backpropagate(node, reward)

    # Choose the most‐visited child of root
    if not root.children:
        print("No legal root actions remain; terminating early.")
        break
    best_child = max(root.children, key=lambda c: c.visits)

    # Commit that action
    next_idx = root.box_index
    w,d,h = dims[next_idx]
    action_x, action_y = best_child.action  # guaranteed not None
    sim_space = Space(SPACE_W, SPACE_D, SPACE_H)
    sim_space.boxes = list(root.placed_boxes)
    placed_box = sim_space.place_at(action_x, action_y, w,d,h)
    final_placements = root.placed_boxes + [placed_box]

    print(f"Step {next_idx+1:02d}: placed {w}×{d}×{h} at "
          f"x={action_x},y={action_y},z={placed_box.z}  "
          f"(visits={best_child.visits}, Q={best_child.total_reward:.2f})")

    # Advance root
    best_child.parent = None
    root = best_child

print(f"\nDone. Placed {len(final_placements)} / {TOTAL_BOXES} boxes.")


# --- Final plot ---
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection="3d")
volumes = [b.volume for b in final_placements]
cmap = plt.cm.viridis
norm = plt.Normalize(min(volumes), max(volumes))
for b in final_placements:
    ax.bar3d(b.x, b.y, b.z, b.w, b.d, b.h,
             color=cmap(norm(b.volume)), edgecolor='k', alpha=0.7)
ax.set_xlim(0, SPACE_W)
ax.set_ylim(0, SPACE_D)
ax.set_zlim(0, SPACE_H)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
plt.title(f"MCTS placed {len(final_placements)}/{TOTAL_BOXES} boxes")
plt.tight_layout()
plt.show()
