"""
collect_mcts_policy_random.py

Self-play using MCTS+ValueNet to pack randomly generated boxes each episode,
and online-train a PolicyNet on the last 10 episodes (1 epoch each).

Every episode:
  1) Generate a fresh list of box dimensions
  2) Run MCTS where rollouts use the pretrained ValueNet
  3) Collect (state, policy) at each step, and final placed count
  4) Keep a sliding window of the last 10 episodes’ (states, policies)
  5) Train the PolicyNet for one epoch on that window
  6) Print the rolling average #boxes placed and the policy-network loss
"""

import argparse
import random
import time
import math
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SPACE_W, SPACE_D, SPACE_H = 24, 24, 24   # Container dimensions
UCT_C      = 1.0                         # UCT exploration constant
WINDOW_EP  = 10                          # Episodes window for policy training

# -----------------------------------------------------------------------------
# Generate random box dimensions per episode
# -----------------------------------------------------------------------------
def generate_random_dims(num_boxes:int,
                         min_size:int=1,
                         max_size:int=6,
                         num_types:int=10
                        ) -> List[Tuple[int,int,int]]:
    """
    Randomly pick num_types distinct (w,d,h) in [min_size..max_size],
    then assign random counts summing to num_boxes, and shuffle.
    """
    # partition num_boxes into num_types counts
    cuts = sorted(random.sample(range(num_boxes+1), num_types-1))
    cuts = [0] + cuts + [num_boxes]
    counts = [cuts[i+1]-cuts[i] for i in range(num_types)]
    # random types
    types = [
        (random.randint(min_size, max_size),
         random.randint(min_size, max_size),
         random.randint(min_size, max_size))
        for _ in range(num_types)
    ]
    dims = []
    for dims_type, cnt in zip(types, counts):
        dims += [dims_type]*cnt
    random.shuffle(dims)
    return dims

# Global dims list (updated each episode)
dims: List[Tuple[int,int,int]] = []

# -----------------------------------------------------------------------------
# Box & Space classes
# -----------------------------------------------------------------------------
class Box:
    """Axis-aligned box in 3D."""
    __slots__ = ("x","y","z","w","d","h","volume")
    def __init__(self, x,y,z,w,d,h):
        self.x, self.y, self.z = x,y,z
        self.w, self.d, self.h = w,d,h
        self.volume = w*d*h
    def overlaps(self, o:"Box") -> bool:
        return not (
            self.x+self.w <= o.x or o.x+o.w <= self.x or
            self.y+self.d <= o.y or o.y+o.d <= self.y or
            self.z+self.h <= o.z or o.z+o.h <= self.z
        )

class Space:
    """Container enforcing gravity, full support, no larger-on-smaller."""
    def __init__(self, W,D,H):
        self.W, self.D, self.H = W,D,H
        self.boxes: List[Box] = []

    def copy_with(self, placed:List[Box]) -> "Space":
        s = Space(self.W,self.D,self.H)
        s.boxes = placed.copy()
        return s

    def find_floor(self, x,y,w,d) -> int:
        """Return z-floor for footprint (x,y,w,d)."""
        zf = 0
        for b in self.boxes:
            if not (x+w<=b.x or b.x+b.w<=x or y+d<=b.y or b.y+b.d<=y):
                zf = max(zf, b.z + b.h)
        return zf

    def is_supported(self, x,y,w,d,zf) -> bool:
        """Full‐footprint support (no overhang)."""
        if zf==0: return True
        for xi in range(x,x+w):
            for yi in range(y,y+d):
                if not any(
                    (b.z+b.h)==zf and b.x<=xi<b.x+b.w and b.y<=yi<b.y+b.d
                    for b in self.boxes
                ):
                    return False
        return True

    def no_larger(self, w,d,h,zf) -> bool:
        """No larger‐on‐smaller rule."""
        vol = w*d*h
        for b in self.boxes:
            if b.z+b.h==zf and b.volume<vol:
                return False
        return True

    def legal_xy(self, w,d,h) -> List[Tuple[int,int]]:
        """All legal (x,y) for placing w×d×h."""
        out = []
        for x in range(self.W-w+1):
            for y in range(self.D-d+1):
                zf = self.find_floor(x,y,w,d)
                if zf+h > self.H: continue
                c = Box(x,y,zf,w,d,h)
                if any(c.overlaps(b) for b in self.boxes): continue
                if not self.is_supported(x,y,w,d,zf): continue
                if not self.no_larger(w,d,h,zf): continue
                out.append((x,y))
        return out

    def place(self, x,y,w,d,h) -> Box:
        """Drop and commit a new box."""
        zf = self.find_floor(x,y,w,d)
        b = Box(x,y,zf,w,d,h)
        self.boxes.append(b)
        return b

    def height_map(self) -> List[List[int]]:
        """2D height‐map of existing packing."""
        hm = [[0]*self.D for _ in range(self.W)]
        for b in self.boxes:
            top = b.z + b.h
            for xi in range(b.x, b.x+b.w):
                for yi in range(b.y, b.y+b.d):
                    hm[xi][yi] = max(hm[xi][yi], top)
        return hm

# -----------------------------------------------------------------------------
# ValueNet for rollouts (must match the checkpoint architecture!)
# -----------------------------------------------------------------------------
class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Conv block 1
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),              # → 16×12×12

            # Conv block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),              # → 32×6×6

            # Conv block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),              # → 64×3×3

            # Flatten + Dense head
            nn.Flatten(),                    # → 64*3*3 = 576
            nn.Linear(576, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)                # → scalar
        )

    def forward(self, x):
        return self.net(x).squeeze(1)         # (B,)

# -----------------------------------------------------------------------------
# PolicyNet: backbone + upsampler to 2×24×24
# -----------------------------------------------------------------------------
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # shared conv‐backbone to 64×3×3
        self.backbone = nn.Sequential(
            nn.Conv2d(4,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # up‐projection to 2 channels
        self.head = nn.Sequential(
            nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32,16,3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16,2,3,stride=2,padding=1,output_padding=1),
            # yields 2×24×24
        )
    def forward(self,x):
        f = self.backbone(x)
        return self.head(f)  # (B,2,24,24)

# -----------------------------------------------------------------------------
# MCTS Node & functions
# -----------------------------------------------------------------------------
class Node:
    __slots__ = ("parent","action","idx","placed","children","visits","Q","_untried")
    def __init__(self, parent, action, idx, placed):
        self.parent, self.action, self.idx, self.placed = parent, action, idx, placed
        self.children = []; self.visits=0; self.Q=0.0; self._untried=None
    def untried(self):
        if self._untried is None:
            if self.idx >= len(dims):
                self._untried = []
            else:
                w,d,h = dims[self.idx]
                sp = Space(SPACE_W,SPACE_D,SPACE_H).copy_with(self.placed)
                self._untried = sp.legal_xy(w,d,h)
        return self._untried
    def fully_expanded(self): return not self.untried()
    def terminal(self): return self.idx >= len(dims)

def uct_select(node:Node):
    if not node.children: return None
    logN = math.log(node.visits)
    def score(c:Node):
        return (c.Q/c.visits) + UCT_C*math.sqrt(logN/c.visits)
    return max(node.children, key=score)

def expand(node:Node):
    w,d,h = dims[node.idx]
    act   = node.untried().pop(random.randrange(len(node._untried)))
    sp    = Space(SPACE_W,SPACE_D,SPACE_H).copy_with(node.placed)
    newb  = sp.place(*act, w,d,h)
    child = Node(node, act, node.idx+1, node.placed+[newb])
    node.children.append(child)
    return child

def backprop(node:Node, r:float):
    while node:
        node.visits += 1
        node.Q      += r
        node = node.parent

# -----------------------------------------------------------------------------
# Rollout via ValueNet
# -----------------------------------------------------------------------------
def rollout_nn(node:Node, vnet:ValueNet, device, max_bd:float) -> float:
    if node.idx >= len(dims):
        return 1.0
    sp  = Space(SPACE_W,SPACE_D,SPACE_H).copy_with(node.placed)
    hm  = sp.height_map()  # W×D list
    w,d,h = dims[node.idx]

    # build input tensor 1×4×24×24
    hm_t = torch.tensor(hm, device=device).float() / SPACE_H
    hm_t = hm_t.unsqueeze(0)               # (1,24,24)
    bd   = torch.tensor([w,d,h], device=device).float()/max_bd
    bd   = bd.view(3,1,1).expand(3,SPACE_W,SPACE_D)
    x    = torch.cat([hm_t, bd], dim=0).unsqueeze(0)

    assert x.shape == (1,4,SPACE_W,SPACE_D)
    vnet.eval()
    with torch.no_grad():
        future = vnet(x).item()
    return (len(node.placed) + future) / len(dims)

# -----------------------------------------------------------------------------
# Self-play episode: collect (state,policy,value) and placed-count
# -----------------------------------------------------------------------------
def self_play(vnet:ValueNet, device, max_bd:float, time_limit:float
             ) -> Tuple[List[np.ndarray],List[np.ndarray],List[float],int]:
    root = Node(None,None,0,[])
    states, policies, values = [], [], []

    while not root.terminal():
        # MCTS for time_limit secs
        t0 = time.time()
        while time.time() - t0 < time_limit:
            node = root
            # selection
            while not node.terminal() and node.fully_expanded() and node.children:
                nxt = uct_select(node)
                if nxt is None: break
                node = nxt
            # expansion
            if not node.terminal() and not node.fully_expanded():
                node = expand(node)
            # rollout
            r = rollout_nn(node, vnet, device, max_bd)
            # backprop
            backprop(node, r)

        # extract root policy (visit distribution)
        pol = np.zeros((2,SPACE_W,SPACE_D), dtype=np.float32)
        tot = sum(c.visits for c in root.children)
        if tot>0:
            for c in root.children:
                x,y = c.action
                pol[0,x,y] = c.visits / tot
        # channel 1 (rotation) left as zeros for now

        # extract root value
        val = (root.Q / root.visits) if root.visits>0 else 0.0

        # build state tensor
        sp  = Space(SPACE_W,SPACE_D,SPACE_H).copy_with(root.placed)
        hm  = np.array(sp.height_map(),dtype=np.float32) / SPACE_H
        w,d,h = dims[root.idx]
        st  = np.zeros((4,SPACE_W,SPACE_D),dtype=np.float32)
        st[0] = hm
        st[1].fill(w/max_bd)
        st[2].fill(d/max_bd)
        st[3].fill(h/max_bd)

        states.append(st)
        policies.append(pol)
        values.append(val)

        # commit best move
        if not root.children:
            break
        best = max(root.children, key=lambda c:c.visits)
        w,d,h = dims[root.idx]
        sp = Space(SPACE_W,SPACE_D,SPACE_H).copy_with(root.placed)
        sp.place(*best.action, w,d,h)
        best.parent = None
        root = best

    placed_count = len(root.placed)
    return states, policies, values, placed_count

# -----------------------------------------------------------------------------
# Policy dataset & train fn
# -----------------------------------------------------------------------------
class PolicyDataset(Dataset):
    def __init__(self, states, policies):
        # states: List[np(4,24,24)]   policies: List[np(2,24,24)]
        self.states  = torch.from_numpy(np.stack(states))
        self.policies= torch.from_numpy(np.stack(policies))
    def __len__(self): return len(self.states)
    def __getitem__(self,i):
        return self.states[i], self.policies[i]

def train_policy(pnet:PolicyNet, opt, data:PolicyDataset, device) -> float:
    loader = DataLoader(data, batch_size=32, shuffle=True)
    kl = nn.KLDivLoss(reduction='batchmean')
    pnet.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = pnet(xb)                            # (B,2,24,24)
        logp   = torch.log_softmax(logits.view(xb.size(0),-1), dim=1)
        target = yb.view(xb.size(0),-1)  # assume sum=1 over channel0
        loss   = kl(logp, target)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(data)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",   type=int,   default=100)
    parser.add_argument("--time-limit", type=float, default=1.0)
    parser.add_argument("--num-boxes",  type=int,   default=100)
    parser.add_argument("--min-size",   type=int,   default=1)
    parser.add_argument("--max-size",   type=int,   default=6)
    parser.add_argument("--num-types",  type=int,   default=10)
    parser.add_argument("--value-model", required=True,
                        help="path to pretrained ValueNet .pth")
    args = parser.parse_args()

    # load dims per episode
    # load ValueNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vnet   = ValueNet().to(device)
    ckpt   = torch.load(args.value_model, map_location=device)
    vnet.load_state_dict(ckpt)

    max_bd = float(args.max_size)  # safe bound for normalization

    # PolicyNet + optimizer
    pnet = PolicyNet().to(device)
    optimizer = torch.optim.Adam(pnet.parameters(), lr=1e-3)

    # sliding windows
    recent_eps = deque(maxlen=WINDOW_EP)
    placed_hist= deque(maxlen=WINDOW_EP)

    random.seed(0)
    for ep in range(1, args.episodes+1):
        # fresh dims
        global dims
        dims = generate_random_dims(
            num_boxes=args.num_boxes,
            min_size=args.min_size,
            max_size=args.max_size,
            num_types=args.num_types
        )

        # self-play + collect
        states, policies, values, placed = self_play(
            vnet, device, max_bd, args.time_limit
        )

        # append to window
        recent_eps.append((states, policies))
        placed_hist.append(placed)

        # build training data from last WINDOW_EP eps
        all_st, all_pol = [], []
        for st,pol in recent_eps:
            all_st.extend(st)
            all_pol.extend(pol)
        dataset = PolicyDataset(all_st, all_pol)

        # train policy net 1 epoch
        loss = train_policy(pnet, optimizer, dataset, device)

        # report
        avg_placed = sum(placed_hist)/len(placed_hist)
        print(f"Episode {ep}/{args.episodes}  "
              f"avg_placed(last{len(placed_hist)})={avg_placed:.2f}  "
              f"policy_loss={loss:.4f}")

if __name__=="__main__":
    main()
