"""
mcts_with_nn.py

Use MCTS to pack boxes into a 24×24×10 container.  Instead of random rollouts,
we use a pretrained CNN to predict how many more boxes can fit.

Usage:
    python mcts_with_nn.py boxes.csv model.pth
"""

import sys
import csv
import time
import math
import random
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SPACE_W, SPACE_D, SPACE_H = 24, 24, 10    # grid dims
UCT_C      = 1.0                         # exploration constant
TIME_LIMIT = 1.0                         # sec per MCTS move

# This will hold our sorted list of (w,d,h)
dims: List[Tuple[int,int,int]] = []


# -----------------------------------------------------------------------------
# Load and sort dims
# -----------------------------------------------------------------------------
def load_dims(path: str) -> List[Tuple[int,int,int]]:
    raw: List[Tuple[int,int,int]] = []
    with open(path, newline="") as f:
        for lineno, row in enumerate(csv.reader(f), 1):
            if len(row) < 3:
                print(f"[WARN] line {lineno}: need 3 values, skipping")
                continue
            try:
                w, d, h = map(int, row[:3])
                if w<=0 or d<=0 or h<=0:
                    raise ValueError
            except ValueError:
                print(f"[WARN] line {lineno}: bad dims {row}, skipping")
                continue
            raw.append((w, d, h))
    # sort largest volume first
    raw.sort(key=lambda t: t[0]*t[1]*t[2], reverse=True)
    return raw


# -----------------------------------------------------------------------------
# Box and Space
# -----------------------------------------------------------------------------
class Box:
    __slots__ = ("x","y","z","w","d","h","volume")
    def __init__(self, x:int, y:int, z:int, w:int, d:int, h:int):
        self.x, self.y, self.z = x, y, z
        self.w, self.d, self.h = w, d, h
        self.volume = w * d * h

    def overlaps(self, o: "Box") -> bool:
        return (
            self.x < o.x+o.w and self.x+self.w > o.x and
            self.y < o.y+o.d and self.y+self.d > o.y and
            self.z < o.z+o.h and self.z+self.h > o.z
        )

class Space:
    def __init__(self, W:int, D:int, H:int):
        self.W, self.D, self.H = W, D, H
        self.boxes: List[Box] = []

    def copy_with(self, placed: List[Box]) -> "Space":
        s = Space(self.W, self.D, self.H)
        s.boxes = placed.copy()
        return s

    def find_floor(self, x:int, y:int, w:int, d:int) -> int:
        zf = 0
        for b in self.boxes:
            if not (x+w <= b.x or b.x+b.w <= x or y+d <= b.y or b.y+b.d <= y):
                zf = max(zf, b.z + b.h)
        return zf

    def is_supported(self, x:int, y:int, w:int, d:int, zf:int) -> bool:
        if zf == 0: return True
        for xi in range(x, x+w):
            for yi in range(y, y+d):
                if not any(
                    b.z + b.h == zf and
                    b.x <= xi < b.x + b.w and
                    b.y <= yi < b.y + b.d
                    for b in self.boxes
                ):
                    return False
        return True

    def no_larger_on_smaller(self, w:int, d:int, h:int, zf:int) -> bool:
        vol = w * d * h
        for b in self.boxes:
            if b.z + b.h == zf and b.volume < vol:
                return False
        return True

    def legal_xy(self, w:int, d:int, h:int) -> List[Tuple[int,int]]:
        out = []
        for x in range(self.W - w + 1):
            for y in range(self.D - d + 1):
                zf = self.find_floor(x, y, w, d)
                if zf + h > self.H:
                    continue
                cand = Box(x, y, zf, w, d, h)
                if any(cand.overlaps(b) for b in self.boxes):
                    continue
                if not self.is_supported(x, y, w, d, zf):
                    continue
                if not self.no_larger_on_smaller(w, d, h, zf):
                    continue
                out.append((x, y))
        return out

    def place(self, x:int, y:int, w:int, d:int, h:int) -> Box:
        zf = self.find_floor(x, y, w, d)
        b = Box(x, y, zf, w, d, h)
        self.boxes.append(b)
        return b

    def height_map(self) -> List[List[int]]:
        hm = [[0]*self.D for _ in range(self.W)]
        for b in self.boxes:
            top = b.z + b.h
            for xi in range(b.x, b.x+b.w):
                for yi in range(b.y, b.y+b.d):
                    hm[xi][yi] = max(hm[xi][yi], top)
        return hm


# -----------------------------------------------------------------------------
# CNN evaluator
# -----------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),             # 16×12×12
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),             # 32×6×6
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),             # 64×3×3
            nn.Flatten(),                # 576
            nn.Linear(576,256), nn.ReLU(),
            nn.Linear(256,1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # (N,)


# -----------------------------------------------------------------------------
# MCTS node
# -----------------------------------------------------------------------------
class Node:
    __slots__ = ("parent","action","idx","placed","children","visits","Q","_untried")

    def __init__(self,
                 parent: Optional["Node"],
                 action: Optional[Tuple[int,int]],
                 idx: int,
                 placed: List[Box]):
        self.parent   = parent
        self.action   = action
        self.idx      = idx
        self.placed   = placed
        self.children = []    # type: List[Node]
        self.visits   = 0
        self.Q        = 0.0
        self._untried = None  # type: Optional[List[Tuple[int,int]]]

    def untried(self) -> List[Tuple[int,int]]:
        if self._untried is None:
            if self.idx >= len(dims):
                self._untried = []
            else:
                w, d, h = dims[self.idx]
                sp = Space(SPACE_W,SPACE_D,SPACE_H).copy_with(self.placed)
                self._untried = sp.legal_xy(w, d, h)
        return self._untried

    def fully_expanded(self) -> bool:
        return not self.untried()

    def terminal(self) -> bool:
        return self.idx >= len(dims)


# -----------------------------------------------------------------------------
# UCT, expand, backprop
# -----------------------------------------------------------------------------
def uct_select(node: Node) -> Optional[Node]:
    if not node.children:
        return None
    logN = math.log(node.visits)
    def score(c: Node) -> float:
        return (c.Q/c.visits) + UCT_C * math.sqrt(logN/c.visits)
    return max(node.children, key=score)

def expand(node: Node) -> Node:
    w,d,h = dims[node.idx]
    act   = node.untried().pop(random.randrange(len(node._untried)))
    sp    = Space(SPACE_W,SPACE_D,SPACE_H).copy_with(node.placed)
    newb  = sp.place(*act, w, d, h)
    child = Node(node, act, node.idx+1, node.placed + [newb])
    node.children.append(child)
    return child

def backprop(node: Optional[Node], reward: float):
    while node:
        node.visits += 1
        node.Q      += reward
        node = node.parent


# -----------------------------------------------------------------------------
# Rollout via CNN
# -----------------------------------------------------------------------------
def rollout_nn(node: Node,
               model: SimpleCNN,
               device: torch.device,
               max_box_dim: float) -> float:
    # If terminal, all boxes are placed → max reward
    if node.idx >= len(dims):
        return 1.0

    sp  = Space(SPACE_W,SPACE_D,SPACE_H).copy_with(node.placed)
    hm  = sp.height_map()      # 24×24 grid
    w,d,h = dims[node.idx]

    # Channel 0: height_map
    hm_t = torch.tensor(hm, dtype=torch.float32, device=device) / SPACE_H
    hm_t = hm_t.unsqueeze(0)   # (1,24,24)

    # Channels 1-3: box dims
    bd = torch.tensor([w,d,h], dtype=torch.float32, device=device)/max_box_dim
    bd = bd.view(3,1,1).expand(3,SPACE_W,SPACE_D)  # (3,24,24)

    x = torch.cat([hm_t, bd], dim=0)    # (4,24,24)
    x = x.unsqueeze(0)                  # (1,4,24,24)

    # Sanity check
    assert x.shape == (1,4,SPACE_W,SPACE_D), f"Got {x.shape}, expected (1,4,24,24)"

    model.eval()
    with torch.no_grad():
        pred = model(x).item()  # expected # future boxes

    placed = len(node.placed)
    total  = placed + pred
    return total / len(dims)


# -----------------------------------------------------------------------------
# MCTS main loop
# -----------------------------------------------------------------------------
def run_mcts(model: SimpleCNN, device: torch.device, max_box_dim: float) -> List[Box]:
    root = Node(None, None, 0, [])
    total = len(dims)
    print(f"MCTS+NN on {total} boxes (TIME_LIMIT={TIME_LIMIT}s)")

    while not root.terminal():
        t0 = time.time()
        while time.time() - t0 < TIME_LIMIT:
            # SELECTION
            node = root
            while (not node.terminal() and node.fully_expanded() and node.children):
                nxt = uct_select(node)
                if nxt is None: break
                node = nxt

            # EXPANSION
            if not node.terminal() and not node.fully_expanded():
                node = expand(node)

            # ROLLOUT via CNN
            reward = rollout_nn(node, model, device, max_box_dim)

            # BACKPROP
            backprop(node, reward)

        # pick best child
        if not root.children:
            print("[INFO] no legal moves; stopping early.")
            break
        best = max(root.children, key=lambda c: c.visits)

        # commit the best action
        w,d,h = dims[root.idx]
        x,y   = best.action  # guaranteed not None
        sp    = Space(SPACE_W,SPACE_D,SPACE_H).copy_with(root.placed)
        placed= sp.place(x,y,w,d,h)
        print(f"Step {root.idx+1:02d}: placed {w}×{d}×{h} @({x},{y},{placed.z}) "
              f"[visits={best.visits}]")

        best.parent = None
        root = best

    return root.placed


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_placement(boxes: List[Box]):
    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111, projection="3d")
    vols = [b.volume for b in boxes]
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(vols), max(vols)) if vols else None

    for b in boxes:
        col = cmap(norm(b.volume)) if norm else "skyblue"
        ax.bar3d(b.x, b.y, b.z, b.w, b.d, b.h,
                 color=col, edgecolor="k", alpha=0.7)

    ax.set_xlim(0,SPACE_W)
    ax.set_ylim(0,SPACE_D)
    ax.set_zlim(0,SPACE_H)
    plt.title(f"Placed {len(boxes)}/{len(dims)} boxes")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def main():
    if len(sys.argv) != 3:
        print("Usage: python mcts_with_nn.py boxes.csv model.pth")
        sys.exit(1)

    global dims
    dims = load_dims(sys.argv[1])
    if not dims:
        print("[ERROR] No valid boxes."); sys.exit(1)

    # Load trained CNN
    model = SimpleCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(sys.argv[2], map_location=device)
    model.load_state_dict(state)
    model.to(device)

    max_box_dim = float(max(max(w,d,h) for w,d,h in dims))
    random.seed(0)

    final = run_mcts(model, device, max_box_dim)
    plot_placement(final)


if __name__ == "__main__":
    main()
