"""
collect_data.py

Run many episodes of a simple box‐packing simulator, record at each step:
  - the 24×24 height‐map of placed boxes
  - the (w,d,h) of the current box/orientation
  - the list of remaining boxes to place
  - the chosen action (place/rotate/skip)
  - how many boxes are already placed (to compute future “return”)
Write all records to a JSON Lines file for later training.
"""

import sys
import json
import random
from itertools import permutations
from typing import List, Tuple, Dict, Any

# Try to import tqdm for a progress bar; if missing, define a dummy
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x

# Type aliases
Dims       = Tuple[int,int,int]           # (w, d, h)
Action     = Dict[str,Any]                # {"type":str, "payload":...}
StepRecord = Dict[str,Any]                # see below

# ----------------------------------------------------------------------
# BOX & SPACE CLASSES
# ----------------------------------------------------------------------

class Box:
    """A single placed rectangular prism in 3D space."""
    def __init__(self, x:int, y:int, z:int, w:int, d:int, h:int) -> None:
        self.x, self.y, self.z = x, y, z
        self.w, self.d, self.h = w, d, h

    def overlaps(self, other: "Box") -> bool:
        """Return True if this box overlaps `other` in 3D."""
        return (
            self.x < other.x + other.w and
            self.x + self.w > other.x and
            self.y < other.y + other.d and
            self.y + self.d > other.y and
            self.z < other.z + other.h and
            self.z + self.h > other.z
        )


class Space:
    """
    The container of size W×D×H, keeps track of placed boxes,
    computes legal placements under gravity and full‐support rules,
    and can produce a 2D height‐map.
    """
    def __init__(self, W:int, D:int, H:int) -> None:
        self.W, self.D, self.H = W, D, H
        self.boxes: List[Box] = []

    def find_floor_z(self, x:int, y:int, w:int, d:int) -> int:
        """
        For a footprint at (x,y) of size w×d, return the maximum z
        such that the new box sits at z (floor or top of existing boxes).
        """
        zf = 0
        for b in self.boxes:
            # if footprints overlap in XY, candidate floor = b.z + b.h
            if not (x+w <= b.x or b.x+b.w <= x or
                    y+d <= b.y or b.y+b.d <= y):
                zf = max(zf, b.z + b.h)
        return zf

    def is_supported(self, x:int, y:int, w:int, d:int, zf:int) -> bool:
        """
        If zf>0, ensure *every* cell under the w×d footprint
        is supported by some box top exactly at zf (no overhangs).
        """
        if zf == 0:
            return True  # the floor supports everything
        for xi in range(x, x + w):
            for yi in range(y, y + d):
                # must find a box whose top is zf covering (xi,yi)
                if not any(
                    (b.z + b.h) == zf and
                    b.x <= xi < b.x + b.w and
                    b.y <= yi < b.y + b.d
                    for b in self.boxes
                ):
                    return False
        return True

    def legal_positions(self, w:int, d:int, h:int) -> List[Tuple[int,int,int]]:
        """
        Return all legal (x,y,zf) placements for a w×d×h box:
          - fits within [0..W]×[0..D]×[0..H]
          - does not overlap existing boxes
          - is fully supported
        """
        positions: List[Tuple[int,int,int]] = []
        for x in range(0, self.W - w + 1):
            for y in range(0, self.D - d + 1):
                zf = self.find_floor_z(x, y, w, d)
                # must fit under the ceiling
                if zf + h > self.H:
                    continue
                # no 3D overlap
                cand = Box(x, y, zf, w, d, h)
                if any(cand.overlaps(b) for b in self.boxes):
                    continue
                # full‐footprint support
                if not self.is_supported(x, y, w, d, zf):
                    continue
                positions.append((x, y, zf))
        return positions

    def place(self, x:int, y:int, z:int, w:int, d:int, h:int) -> None:
        """Place a new box at (x,y,z) of size w×d×h."""
        self.boxes.append(Box(x, y, z, w, d, h))

    def compute_height_map(self) -> List[List[int]]:
        """
        Build and return a 2D height‐map HMap[x][y] = max(z+h)
        over all placed boxes covering (x,y).  Shape is W×D.
        """
        hm = [[0 for _ in range(self.D)] for _ in range(self.W)]
        for b in self.boxes:
            top_z = b.z + b.h
            for xi in range(b.x, b.x + b.w):
                for yi in range(b.y, b.y + b.d):
                    if top_z > hm[xi][yi]:
                        hm[xi][yi] = top_z
        return hm

# ----------------------------------------------------------------------
# ORIENTATION & DIMENSION GENERATORS
# ----------------------------------------------------------------------

def get_allowed_orients(dims: Dims) -> Tuple[List[Dims], int]:
    """
    Given (w,d,h), compute:
      - max_face_area = max(w*d, w*h, d*h)
      - allowed_orients = all unique permutations whose base‐area != max_face_area
    Returns (allowed_orients, max_face_area).
    """
    w, d, h = dims
    face_areas = [w*d, w*h, d*h]
    max_area = max(face_areas)
    all_orients = set(permutations(dims))
    allowed = [o for o in all_orients if o[0] * o[1] != max_area]
    return allowed, max_area

def generate_random_dims(num_boxes:int,
                         min_size:int=1,
                         max_size:int=5,
                         num_types:int=10) -> List[Dims]:
    """
    Build exactly `num_boxes` dims by:
      1) sampling `num_types` random (w,d,h)
      2) assigning random counts summing to num_boxes
      3) shuffling the final list
    """
    # random cut‐points to split num_boxes into num_types counts
    cuts = sorted(random.sample(range(num_boxes+1), num_types-1))
    cuts = [0] + cuts + [num_boxes]
    counts = [cuts[i+1] - cuts[i] for i in range(num_types)]

    types: List[Dims] = [
        (random.randint(min_size, max_size),
         random.randint(min_size, max_size),
         random.randint(min_size, max_size))
        for _ in range(num_types)
    ]

    out: List[Dims] = []
    for t, cnt in zip(types, counts):
        out.extend([t] * cnt)
    random.shuffle(out)
    return out

# ----------------------------------------------------------------------
# EPISODE SIMULATION
# ----------------------------------------------------------------------

def run_episode(dims_list: List[Dims],
                W:int, D:int, H:int
               ) -> Tuple[List[StepRecord], int]:
    """
    Simulate one episode of placing/skipping each box in dims_list.
    Returns:
      - trajectory: a list of StepRecord dicts, one per decision step
      - final_count: total # boxes successfully placed
    Each StepRecord contains:
      {
        "height_map":   List[List[int]]   # shape W×D
        "box_dims":     [w, d, h]         # current orientation
        "remaining":    List[Dims]        # dims_list from current idx onward
        "action":       {"type":str,      # "place","rotate","skip"
                         "payload":...}
        "placed_count": int               # #boxes placed so far
      }
    """
    space = Space(W, D, H)
    trajectory: List[StepRecord] = []

    for idx, orig_dims in enumerate(dims_list):
        allowed_orients, max_area = get_allowed_orients(orig_dims)
        current = orig_dims
        rotations_used = 0

        while True:
            w0, d0, h0 = current
            base_area = w0 * d0
            on_largest_face = (base_area == max_area)

            # 1) Build place‐actions if not lying flat on largest face
            place_actions: List[Action] = []
            if not on_largest_face:
                for x, y, zf in space.legal_positions(w0, d0, h0):
                    place_actions.append({
                        "type": "place",
                        "payload": (x, y, w0, d0, h0)
                    })

            # 2) Build rotate‐actions if we still have budget
            rotate_actions: List[Action] = []
            if rotations_used < 3:
                for orient in allowed_orients:
                    if orient != current:
                        rotate_actions.append({
                            "type": "rotate",
                            "payload": orient
                        })

            all_actions = place_actions + rotate_actions

            # 3) If no legal actions, we skip
            if not all_actions:
                action = {"type": "skip", "payload": None}
            else:
                action = random.choice(all_actions)

            # 4) Record the *state* and chosen *action*
            step: StepRecord = {
                "height_map":   space.compute_height_map(),
                "box_dims":     list(current),
                "remaining":    dims_list[idx:],    # includes this box
                "action":       action,
                "placed_count": len(space.boxes)
            }
            trajectory.append(step)

            # 5) Apply the action
            if action["type"] == "rotate":
                current = tuple(action["payload"])  # new (w,d,h)
                rotations_used += 1
                continue
            elif action["type"] == "place":
                x, y, w, d, h = action["payload"]
                zf = space.find_floor_z(x, y, w, d)
                space.place(x, y, zf, w, d, h)
            # else "skip": do nothing
            break

    final_count = len(space.boxes)
    return trajectory, final_count

# ----------------------------------------------------------------------
# DATA COLLECTION LOOP
# ----------------------------------------------------------------------

def collect_data(outfile: str,
                 episodes: int = 5_000,
                 num_boxes: int = 100,
                 min_size: int = 1,
                 max_size: int = 6,
                 num_types: int = 10,
                 space_dims: Tuple[int,int,int] = (24,24,24)
                ) -> None:
    """
    Run `episodes` simulations, write JSON Lines to `outfile`.
    Each line is one StepRecord with an added "return" = final_count - placed_count.
    """
    W, D, H = space_dims
    with open(outfile, "w") as fp:
        for _ in tqdm(range(episodes), desc="Episodes"):
            dims_list = generate_random_dims(num_boxes, min_size, max_size, num_types)
            traj, final_count = run_episode(dims_list, W, D, H)

            for step in traj:
                record = {
                    "height_map":   step["height_map"],    # 24×24 grid
                    "box_dims":     step["box_dims"],      # [w,d,h]
                    "remaining":    step["remaining"],     # list of dims
                    "action":       step["action"],        # place/rotate/skip
                    "placed_count": step["placed_count"],  # # already placed
                    "return":       final_count - step["placed_count"]
                }
                fp.write(json.dumps(record) + "\n")

# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python collect_data.py out.jsonl")
        sys.exit(1)

    random.seed(0)  # for reproducibility; change/remove as desired
    out_path = sys.argv[1]
    collect_data(out_path)
    print(f"Done — data written to {out_path}")

if __name__ == "__main__":
    main()
