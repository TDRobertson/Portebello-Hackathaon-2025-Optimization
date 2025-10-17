"""
mcts_with_nn_leaf_parallel_zero.py

MCTS packing in a 24×24×10 container, using a pretrained CNN for rollouts,
leaf‐parallelized, and assigning 0.0 reward to any nonterminal node with no legal moves.
"""

import sys, csv, time, math, random
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
SPACE_W, SPACE_D, SPACE_H = 24, 24, 10   # container dims
UCT_C      = 1.4                        # exploration constant
TIME_LIMIT = 1.0                        # seconds per MCTS move
ROLLOUTS_PER_LEAF = 8                   # parallel rollouts per expansion

# Global list of box dims, sorted by descending volume
dims: List[Tuple[int,int,int]] = []


# -----------------------------------------------------------------------------
# LOAD & SORT BOX DIMENSIONS
# -----------------------------------------------------------------------------
def load_dims(path: str) -> List[Tuple[int,int,int]]:
    raw = []
    with open(path, newline="") as f:
        for ln,row in enumerate(csv.reader(f),1):
            if len(row)<3:
                print(f"[WARN] line {ln}: skipping, need 3 cols")
                continue
            try:
                w,d,h = map(int, row[:3])
                assert w>0 and d>0 and h>0
            except:
                print(f"[WARN] line {ln}: bad dims {row}, skipping")
                continue
            raw.append((w,d,h))
    raw.sort(key=lambda t: t[0]*t[1]*t[2], reverse=True)
    return raw


# -----------------------------------------------------------------------------
# BOX & SPACE
# -----------------------------------------------------------------------------
class Box:
    __slots__ = ("x","y","z","w","d","h","volume")
    def __init__(self,x,y,z,w,d,h):
        self.x,self.y,self.z = x,y,z
        self.w,self.d,self.h = w,d,h
        self.volume = w*d*h

    def overlaps(self, o:"Box") -> bool:
        return (
            self.x < o.x+o.w and self.x+self.w > o.x and
            self.y < o.y+o.d and self.y+self.d > o.y and
            self.z < o.z+o.h and self.z+self.h > o.z
        )

class Space:
    def __init__(self,W,D,H):
        self.W,self.D,self.H = W,D,H
        self.boxes: List[Box] = []

    def copy_with(self, placed:List[Box]) -> "Space":
        s = Space(self.W, self.D, self.H)
        s.boxes = placed.copy()
        return s

    def find_floor(self,x,y,w,d) -> int:
        zf=0
        for b in self.boxes:
            if not (x+w<=b.x or b.x+b.w<=x or y+d<=b.y or b.y+b.d<=y):
                zf = max(zf, b.z+b.h)
        return zf

    def is_supported(self,x,y,w,d,zf) -> bool:
        if zf==0: return True
        for xi in range(x,x+w):
            for yi in range(y,y+d):
                if not any(
                    (b.z+b.h)==zf and b.x<=xi<b.x+b.w and b.y<=yi<b.y+b.d
                    for b in self.boxes
                ):
                    return False
        return True

    def no_larger_on_smaller(self,w,d,h,zf) -> bool:
        vol = w*d*h
        for b in self.boxes:
            if b.z+b.h==zf and b.volume<vol:
                return False
        return True

    def legal_xy(self,w,d,h) -> List[Tuple[int,int]]:
        out=[]
        for x in range(self.W-w+1):
            for y in range(self.D-d+1):
                zf = self.find_floor(x,y,w,d)
                if zf+h>self.H: continue
                b = Box(x,y,zf,w,d,h)
                if any(b.overlaps(o) for o in self.boxes): continue
                if not self.is_supported(x,y,w,d,zf): continue
                if not self.no_larger_on_smaller(w,d,h,zf): continue
                out.append((x,y))
        return out

    def place(self,x,y,w,d,h) -> Box:
        zf = self.find_floor(x,y,w,d)
        b  = Box(x,y,zf,w,d,h)
        self.boxes.append(b)
        return b

    def height_map(self) -> List[List[int]]:
        hm = [[0]*self.D for _ in range(self.W)]
        for b in self.boxes:
            top = b.z+b.h
            for xi in range(b.x, b.x+b.w):
                for yi in range(b.y, b.y+b.d):
                    hm[xi][yi] = max(hm[xi][yi], top)
        return hm


# -----------------------------------------------------------------------------
# CNN EVALUATOR
# -----------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),            # 16×12×12
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),            # 32×6×6
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),            # 64×3×3
            nn.Flatten(),               # 576
            nn.Linear(576,256), nn.ReLU(),
            nn.Linear(256,1)            # scalar
        )

    def forward(self,x):
        return self.net(x).squeeze(1)  # (N,)


# -----------------------------------------------------------------------------
# MCTS NODE
# -----------------------------------------------------------------------------
class Node:
    __slots__ = ("parent","action","idx","placed","children","visits","Q","_untried")
    def __init__(self, parent, action, idx, placed):
        self.parent, self.action, self.idx, self.placed = parent, action, idx, placed
        self.children: List[Node] = []
        self.visits = 0
        self.Q      = 0.0
        self._untried = None

    def untried(self) -> List[Tuple[int,int]]:
        if self._untried is None:
            if self.idx >= len(dims):
                self._untried = []
            else:
                w,d,h = dims[self.idx]
                sp = Space(SPACE_W,SPACE_D,SPACE_H).copy_with(self.placed)
                self._untried = sp.legal_xy(w,d,h)
        return self._untried

    def fully_expanded(self) -> bool:
        return not self.untried()

    def terminal(self) -> bool:
        return self.idx >= len(dims)


# -----------------------------------------------------------------------------
# MCTS HELPERS
# -----------------------------------------------------------------------------
def uct_select(node:Node) -> Optional[Node]:
    if not node.children:
        return None
    ln = math.log(node.visits)
    def score(c:Node)->float:
        return (c.Q/c.visits) + UCT_C * math.sqrt(ln/c.visits)
    return max(node.children, key=score)

def expand(node:Node) -> Node:
    w,d,h = dims[node.idx]
    act   = node.untried().pop(random.randrange(len(node._untried)))
    sp    = Space(SPACE_W,SPACE_D,SPACE_H).copy_with(node.placed)
    newb  = sp.place(*act, w,d,h)
    child = Node(node, act, node.idx+1, node.placed + [newb])
    node.children.append(child)
    return child

def backprop(node:Optional[Node], reward:float):
    while node:
        node.visits += 1
        node.Q      += reward
        node = node.parent


# -----------------------------------------------------------------------------
# ROLLOUT VIA CNN
# -----------------------------------------------------------------------------
def rollout_nn(node:Node,
               model:SimpleCNN,
               device:torch.device,
               max_bd:float) -> float:
    # 1) If fully terminal, perfect reward
    if node.idx >= len(dims):
        return 1.0

    # 2) If no legal moves at this non-terminal node, zero reward
    if node.untried() == [] and node.children == []:
        return 0.0

    # 3) Otherwise build input and infer
    sp  = Space(SPACE_W,SPACE_D,SPACE_H).copy_with(node.placed)
    hm  = sp.height_map()
    w,d,h = dims[node.idx]

    # channel0: height_map
    hm_t = torch.tensor(hm, dtype=torch.float32, device=device) / SPACE_H
    hm_t = hm_t.unsqueeze(0)               # (1,24,24)

    # channels1–3: box dims
    bd = torch.tensor([w,d,h],dtype=torch.float32,device=device)/max_bd
    bd = bd.view(3,1,1).expand(3,SPACE_W,SPACE_D)  # (3,24,24)

    x = torch.cat([hm_t, bd], dim=0).unsqueeze(0)  # (1,4,24,24)
    assert x.shape == (1,4,SPACE_W,SPACE_D)

    model.eval()
    with torch.no_grad():
        pred = model(x).item()  # predicted # of future placements

    placed = len(node.placed)
    return (placed + pred) / len(dims)


# -----------------------------------------------------------------------------
# MCTS MAIN LOOP WITH LEAF PARALLELIZATION
# -----------------------------------------------------------------------------
def run_mcts(model:SimpleCNN,
             device:torch.device,
             max_bd:float,
             executor:ThreadPoolExecutor) -> List[Box]:

    root = Node(None, None, 0, [])
    print(f"MCTS+NN (R={ROLLOUTS_PER_LEAF}) on {len(dims)} boxes")

    while not root.terminal():
        t0 = time.time()
        while time.time() - t0 < TIME_LIMIT:
            # SELECTION
            node = root
            while (not node.terminal()
                   and node.fully_expanded()
                   and node.children):
                nxt = uct_select(node)
                if not nxt: break
                node = nxt

            # EXPANSION
            if not node.terminal() and not node.fully_expanded():
                node = expand(node)

            # LEAF PARALLEL ROLLOUTS
            futures = [
                executor.submit(rollout_nn, node, model, device, max_bd)
                for _ in range(ROLLOUTS_PER_LEAF)
            ]
            for f in futures:
                backprop(node, f.result())

        # pick best child to commit
        if not root.children:
            print("[INFO] no moves left; done")
            break
        best = max(root.children, key=lambda c: c.visits)

        # commit
        w,d,h = dims[root.idx]
        x,y   = best.action
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
def plot_placement(boxes:List[Box]):
    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111, projection="3d")
    vols = [b.volume for b in boxes]
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(vols), max(vols)) if vols else None

    for b in boxes:
        c = cmap(norm(b.volume)) if norm else "skyblue"
        ax.bar3d(b.x,b.y,b.z,b.w,b.d,b.h, color=c, edgecolor="k", alpha=0.7)

    ax.set_xlim(0,SPACE_W); ax.set_ylim(0,SPACE_D); ax.set_zlim(0,SPACE_H)
    plt.title(f"Placed {len(boxes)}/{len(dims)} boxes")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def main():
    if len(sys.argv) != 3:
        print("Usage: python mcts_with_nn_leaf_parallel_zero.py boxes.csv model.pth")
        sys.exit(1)

    global dims
    dims = load_dims(sys.argv[1])
    if not dims:
        print("[ERROR] No valid boxes."); sys.exit(1)

    # load CNN
    model = SimpleCNN()
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(sys.argv[2], map_location=device)
    model.load_state_dict(state); model.to(device)

    max_bd = float(max(max(w,d,h) for w,d,h in dims))
    random.seed(0)

    with ThreadPoolExecutor(max_workers=ROLLOUTS_PER_LEAF) as exe:
        final = run_mcts(model, device, max_bd, exe)

    plot_placement(final)


if __name__=="__main__":
    main()
