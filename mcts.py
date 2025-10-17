"""
mcts_place_boxes.py

Pack a sequence of boxes into a 3D container using Monte Carlo Tree Search (MCTS),
observing gravity, full‐footprint support, and no “larger‐on‐smaller” rules.

Usage:
    python mcts_place_boxes.py boxes.csv
"""

import sys
import csv
import time
import math
import random
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
SPACE_W, SPACE_D, SPACE_H = 15, 15, 10   # Container dimensions
UCT_C = 1.4                              # UCT exploration constant
TIME_LIMIT = 1.0                        # Seconds per MCTS move

# Global list of box dimensions, sorted by descending volume
dims: List[Tuple[int,int,int]] = []


# ------------------------------------------------------------------------------
# Utility: load and sort box dimensions
# ------------------------------------------------------------------------------
def load_dims_from_csv(path: str) -> List[Tuple[int,int,int]]:
    """Read (w,d,h) triples from CSV, skip invalid lines, sort by volume desc."""
    raw: List[Tuple[int,int,int]] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for lineno, row in enumerate(reader, start=1):
            if len(row) < 3:
                print(f"[WARN] line {lineno}: not enough cols, skipping")
                continue
            try:
                w, d, h = map(int, row[:3])
                if w <= 0 or d <= 0 or h <= 0:
                    raise ValueError
            except ValueError:
                print(f"[WARN] line {lineno}: invalid dims {row}, skipping")
                continue
            raw.append((w, d, h))
    # sort largest volume first
    raw.sort(key=lambda t: t[0]*t[1]*t[2], reverse=True)
    return raw


# ------------------------------------------------------------------------------
# Box & Space Classes
# ------------------------------------------------------------------------------
class Box:
    """Axis-aligned rectangular prism placed in the container."""
    __slots__ = ("x","y","z","w","d","h","volume")
    def __init__(self, x:int, y:int, z:int, w:int, d:int, h:int) -> None:
        self.x, self.y, self.z = x, y, z
        self.w, self.d, self.h = w, d, h
        self.volume = w * d * h

    def overlaps(self, other: "Box") -> bool:
        """Check 3D overlap with another box."""
        return (
            self.x < other.x + other.w and
            self.x + self.w > other.x and
            self.y < other.y + other.d and
            self.y + self.d > other.y and
            self.z < other.z + other.h and
            self.z + self.h > other.z
        )

class Space:
    """3D container enforcing gravity and support rules."""
    def __init__(self, W:int, D:int, H:int) -> None:
        self.W, self.D, self.H = W, D, H
        self.boxes: List[Box] = []

    def copy_with(self, placed: List[Box]) -> "Space":
        """Return a copy of this Space with given boxes already placed."""
        s = Space(self.W, self.D, self.H)
        s.boxes = placed.copy()
        return s

    def find_floor_z(self, x:int, y:int, w:int, d:int) -> int:
        """
        Find the z‐height (floor or top of some overlapping box)
        where a w×d footprint at (x,y) would rest.
        """
        zf = 0
        for b in self.boxes:
            # check XY overlap
            if not (x + w <= b.x or b.x + b.w <= x or
                    y + d <= b.y or b.y + b.d <= y):
                zf = max(zf, b.z + b.h)
        return zf

    def is_fully_supported(self, x:int, y:int, w:int, d:int, zf:int) -> bool:
        """
        If zf>0, ensure every cell under the footprint is supported by
        some box whose top is exactly at zf (no overhang).
        """
        if zf == 0:
            return True
        for xi in range(x, x + w):
            for yi in range(y, y + d):
                if not any(
                    (b.z + b.h) == zf and
                    b.x <= xi < b.x + b.w and
                    b.y <= yi < b.y + b.d
                    for b in self.boxes
                ):
                    return False
        return True

    def no_larger_on_smaller(self, w:int, d:int, h:int, zf:int) -> bool:
        """
        Forbid placing a box of volume > vol(b) on top of any b whose top
        is at zf.
        """
        vol = w * d * h
        for b in self.boxes:
            if b.z + b.h == zf and b.volume < vol:
                return False
        return True

    def legal_xy_positions(self, w:int, d:int, h:int) -> List[Tuple[int,int]]:
        """
        Enumerate all (x,y) where a w×d×h box can be legally placed:
          - fully inside W×D×H
          - no 3D overlap
          - full footprint support
          - no larger‐on‐smaller
        """
        out: List[Tuple[int,int]] = []
        for x in range(0, self.W - w + 1):
            for y in range(0, self.D - d + 1):
                zf = self.find_floor_z(x, y, w, d)
                if zf + h > self.H:
                    continue
                cand = Box(x, y, zf, w, d, h)
                if any(cand.overlaps(b) for b in self.boxes):
                    continue
                if not self.is_fully_supported(x, y, w, d, zf):
                    continue
                if not self.no_larger_on_smaller(w, d, h, zf):
                    continue
                out.append((x, y))
        return out

    def place_at(self, x:int, y:int, w:int, d:int, h:int) -> Box:
        """
        Drop and permanently place a box at (x,y) with size w×d×h.
        Returns the new Box.
        """
        zf = self.find_floor_z(x, y, w, d)
        b = Box(x, y, zf, w, d, h)
        self.boxes.append(b)
        return b


# ------------------------------------------------------------------------------
# MCTS Node and operations
# ------------------------------------------------------------------------------
class Node:
    """
    MCTS node storing:
      - parent, action that led here, next box index, placed boxes so far
      - visits, total_reward
      - cached untried actions
    """
    __slots__ = ("parent","action","box_index","placed",
                 "children","visits","total_reward","_untried")

    def __init__(self,
                 parent: Optional["Node"],
                 action: Optional[Tuple[int,int]],
                 box_index: int,
                 placed: List[Box]):
        self.parent       = parent
        self.action       = action
        self.box_index    = box_index
        self.placed       = placed
        self.children     = []  # type: List[Node]
        self.visits       = 0
        self.total_reward = 0.0
        self._untried     = None  # type: Optional[List[Tuple[int,int]]]

    def untried_actions(self) -> List[Tuple[int,int]]:
        """Compute and cache legal placements for the next box."""
        if self._untried is None:
            if self.box_index >= len(dims):
                self._untried = []
            else:
                w, d, h = dims[self.box_index]
                space = Space(SPACE_W, SPACE_D, SPACE_H).copy_with(self.placed)
                self._untried = space.legal_xy_positions(w, d, h)
        return self._untried

    def is_fully_expanded(self) -> bool:
        """True if there are no untried actions left."""
        return len(self.untried_actions()) == 0

    def is_terminal(self) -> bool:
        """True if we have placed all boxes."""
        return self.box_index >= len(dims)


def uct_select(children: List[Node]) -> Optional[Node]:
    """
    Return the child with highest UCT score, or None if no children.
    UCT = (Q/N) + C * sqrt(ln(N_parent)/N)
    """
    if not children:
        return None
    parent_visits = children[0].parent.visits if children[0].parent else 1
    log_parent = math.log(parent_visits)
    def score(n: Node) -> float:
        return (n.total_reward / n.visits) + UCT_C * math.sqrt(log_parent / n.visits)
    return max(children, key=score)


def expand(node: Node) -> Node:
    """
    Pick one untried action, place that box, and return the new child node.
    """
    w, d, h = dims[node.box_index]
    actions = node.untried_actions()
    action = actions.pop(random.randrange(len(actions)))
    space = Space(SPACE_W, SPACE_D, SPACE_H).copy_with(node.placed)
    new_box = space.place_at(*action, w, d, h)
    child = Node(parent=node,
                 action=action,
                 box_index=node.box_index+1,
                 placed=node.placed + [new_box])
    node.children.append(child)
    return child


def rollout(box_index: int, placed: List[Box]) -> float:
    """
    From this partial state, randomly place boxes until stuck.
    Return reward = (# boxes placed in end) / total_boxes.
    """
    space = Space(SPACE_W, SPACE_D, SPACE_H).copy_with(placed)
    count = len(placed)
    for idx in range(box_index, len(dims)):
        w, d, h = dims[idx]
        legal = space.legal_xy_positions(w, d, h)
        if not legal:
            break
        space.place_at(*random.choice(legal), w, d, h)
        count += 1
    return count / len(dims)


def backpropagate(node: Optional[Node], reward: float):
    """Propagate reward up to the root, updating visits & total_reward."""
    while node is not None:
        node.visits += 1
        node.total_reward += reward
        node = node.parent


# ------------------------------------------------------------------------------
# MCTS main routine
# ------------------------------------------------------------------------------
def run_mcts() -> List[Box]:
    """Perform MCTS, return the final list of placed boxes."""
    root = Node(parent=None, action=None, box_index=0, placed=[])
    total = len(dims)
    print(f"Running MCTS on {total} boxes (each move ~{TIME_LIMIT:.1f}s)...")

    while not root.is_terminal():
        start = time.time()
        # Run as many MCTS iterations as we can in TIME_LIMIT
        while time.time() - start < TIME_LIMIT:
            # SELECTION
            node = root
            while not node.is_terminal() and node.is_fully_expanded() and node.children:
                sel = uct_select(node.children)
                if sel is None:
                    break
                node = sel

            # EXPANSION
            if not node.is_terminal() and not node.is_fully_expanded():
                node = expand(node)

            # SIMULATION
            reward = rollout(node.box_index, node.placed)

            # BACKPROPAGATION
            backpropagate(node, reward)

        # Choose best child of root by visits
        if not root.children:
            print("[INFO] No moves left; stopping early.")
            break
        best = max(root.children, key=lambda c: c.visits)

        # Commit that move
        idx = root.box_index
        w, d, h = dims[idx]
        x, y = best.action  # guaranteed not None
        space = Space(SPACE_W, SPACE_D, SPACE_H).copy_with(root.placed)
        placed_box = space.place_at(x, y, w, d, h)

        print(f"Step {idx+1:02d}: placed {w}×{d}×{h} at (x={x},y={y},z={placed_box.z}) "
              f"[visits={best.visits}, reward={best.total_reward:.2f}]")

        # Advance root
        best.parent = None
        root = best

    return root.placed


# ------------------------------------------------------------------------------
# Plot final placement
# ------------------------------------------------------------------------------
def plot_placement(boxes: List[Box]):
    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111, projection="3d")
    vols = [b.volume for b in boxes]
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(vols), max(vols)) if vols else None

    for b in boxes:
        color = cmap(norm(b.volume)) if norm else "skyblue"
        ax.bar3d(b.x, b.y, b.z, b.w, b.d, b.h,
                 color=color, edgecolor="k", alpha=0.7)

    ax.set_xlim(0, SPACE_W)
    ax.set_ylim(0, SPACE_D)
    ax.set_zlim(0, SPACE_H)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.title(f"Placed {len(boxes)}/{len(dims)} boxes")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python mcts_place_boxes.py boxes.csv")
        sys.exit(1)

    global dims
    dims = load_dims_from_csv(sys.argv[1])
    if not dims:
        print("[ERROR] No valid boxes to place.")
        sys.exit(1)

    random.seed(0)
    final = run_mcts()
    plot_placement(final)


if __name__ == "__main__":
    main()
