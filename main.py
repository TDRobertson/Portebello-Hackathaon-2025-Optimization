import csv
import sys
import random
from typing import List, Tuple, Optional
from itertools import permutations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

class Box:
    def __init__(self, x: int, y: int, z: int,
                 w: int, d: int, h: int) -> None:
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
    def __init__(self, width: int, depth: int, height: int) -> None:
        self.width, self.depth, self.height = width, depth, height
        self.boxes: List[Box] = []

    def find_z_floor(self, x: int, y: int, w: int, d: int) -> int:
        zf = 0
        for e in self.boxes:
            if not (x + w <= e.x or e.x + e.w <= x or
                    y + d <= e.y or e.y + e.d <= y):
                zf = max(zf, e.z + e.h)
        return zf

    def legal_xy_positions(self, w: int, d: int, h: int) -> List[Tuple[int, int]]:
        legal = []
        for x in range(0, self.width - w + 1):
            for y in range(0, self.depth - d + 1):
                zf = self.find_z_floor(x, y, w, d)
                if zf + h > self.height:
                    continue
                candidate = Box(x, y, zf, w, d, h)
                # 3D overlap?
                if any(candidate.overlaps(e) for e in self.boxes):
                    continue
                # full‐footprint support
                if zf > 0:
                    ok = True
                    for xi in range(x, x + w):
                        for yi in range(y, y + d):
                            if not any(
                                e.z + e.h == zf and
                                e.x <= xi < e.x + e.w and
                                e.y <= yi < e.y + e.d
                                for e in self.boxes
                            ):
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                legal.append((x, y))
        return legal

    def place_at(self, x: int, y: int, w: int, d: int, h: int) -> Box:
        zf = self.find_z_floor(x, y, w, d)
        b = Box(x, y, zf, w, d, h)
        self.boxes.append(b)
        return b

def load_dims_from_csv(path: str) -> List[Tuple[int,int,int]]:
    dims: List[Tuple[int,int,int]] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader,1):
            if len(row)<3:
                print(f"Skipping line {i}: not enough cols")
                continue
            try:
                w,d,h = map(int, row[:3])
                if w<=0 or d<=0 or h<=0:
                    raise ValueError
            except ValueError:
                print(f"Skipping line {i}: invalid dims {row}")
                continue
            dims.append((w,d,h))
    return dims

def plot_space(space: Space) -> None:
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection="3d")
    vols = [b.volume for b in space.boxes]
    cmap = plt.cm.viridis if vols else None
    norm = plt.Normalize(min(vols), max(vols)) if vols else None
    for b in space.boxes:
        color = cmap(norm(b.volume)) if vols else "skyblue"
        ax.bar3d(b.x, b.y, b.z, b.w, b.d, b.h,
                 color=color, edgecolor="k", alpha=1)
    ax.set_xlim(0, space.width)
    ax.set_ylim(0, space.depth)
    ax.set_zlim(0, space.height)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()

def main() -> None:
    if len(sys.argv)!=2:
        print("Usage: python place_boxes.py boxes.csv")
        sys.exit(1)
    dims_list = load_dims_from_csv(sys.argv[1])
    if not dims_list:
        print("No valid boxes.")
        sys.exit(1)

    # Sort by descending volume
    dims_list.sort(key=lambda t: t[0]*t[1]*t[2], reverse=True)

    SPACE_W,SPACE_D,SPACE_H = 15,15,10
    space = Space(SPACE_W, SPACE_D, SPACE_H)

    for idx, dims in enumerate(dims_list, start=1):
        w0,d0,h0 = dims
        # original box’s largest face‐area:
        max_area = max(w0*d0, w0*h0, d0*h0)
        # all unique orientations:
        orients = set(permutations(dims))
        # only keep those that do NOT put largest face horizontal
        allowed_orients = [o for o in orients if o[0]*o[1]!=max_area]

        current = dims  # start in default orientation
        rot_count = 0

        print(f"Box #{idx:02d} orig={dims} vol={w0*d0*h0}")
        while True:
            # build legal rotation actions
            rots = []
            if rot_count<3:
                for o in allowed_orients:
                    if o!=current:
                        rots.append(o)

            # is current orientation “illegal” (flat on largest face)?
            base_area = current[0]*current[1]
            largest_flat = (base_area==max_area)

            # build place actions only if not lying on largest face
            places: List[Tuple[int,int]] = []
            if not largest_flat:
                places = space.legal_xy_positions(current[0],
                                                  current[1],
                                                  current[2])

            # assemble action space
            actions: List[Tuple[str, Tuple[int,int,int]]] = []
            # place actions
            for (x,y) in places:
                actions.append(("place", (x,y)+current))
            # rotation actions
            for o in rots:
                actions.append(("rotate", o))

            print(f"  Orientation={current}  rotations_used={rot_count}")
            print(f"    → {len(places)} place‐actions, {len(rots)} rotations")

            if not actions:
                print("    ! No legal actions → skipping this box\n")
                break

            # pick one action at random
            act, payload = random.choice(actions)
            if act=="rotate":
                current = payload
                rot_count += 1
                print(f"    * ROTATE → new orient={current}")
                continue
            else:  # "place"
                x,y,w,d,h = payload
                placed = space.place_at(x,y,w,d,h)
                print(f"    ✔ PLACE at x={placed.x},y={placed.y},z={placed.z}\n")
                break

    print(f"Summary: placed {len(space.boxes)}/{len(dims_list)} boxes.")
    plot_space(space)

if __name__ == "__main__":
    main()
