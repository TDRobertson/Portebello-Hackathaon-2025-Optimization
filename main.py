import csv
import sys
import random
import math
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


class Box:
    # Bottom Left Corner?
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

    def place_with_gravity(self,
                           w: int, d: int, h: int,
                           max_attempts: int
                           ) -> Optional[Box]:
        """
        Try up to max_attempts to place a box of size (w,d,h):
         1) pick random (x,y)
         2) let it fall to z_floor = max(0, tops of overlapping footprints)
         3) ensure z_floor+h <= height
         4) ensure no 3D overlap
         5) ensure full‐footprint support (no overhang)
        On success, appends the Box and returns it; otherwise returns None.
        """
        distances = []
        cur_candidate = 0
        if self.boxes == []:
            x, y = 0, 0
            z_floor = 0

            candidate = Box(x, y, z_floor, w, d, h)

            if z_floor + h <= self.height:
                self.boxes.append(candidate)
                return candidate
            else: return None

        else:
            for b in self.boxes:
                for x, y in [(b.x+b.w, b.y), (b.x, b.y+b.d), (b.x+b.w, b.y+b.d)]:
                    # find resting height
                    z_floor = 0
                    for e in self.boxes:
                        if not (x + w <= e.x or e.x + e.w <= x or
                                y + d <= e.y or e.y + e.d <= y):
                            z_floor = max(z_floor, e.z + e.h)

                    # too tall?
                    if z_floor + h > self.height:
                        continue

                    candidate = Box(x, y, z_floor, w, d, h)

                    # 3D overlap?
                    if any(candidate.overlaps(e) for e in self.boxes):
                        continue

                    # full‐footprint support
                    if z_floor > 0:
                        supported = True
                        for xi in range(x, x + w):
                            for yi in range(y, y + d):
                                # must find a box whose top == z_floor covering (xi,yi)
                                if not any(
                                    e.z + e.h == z_floor
                                    and e.x <= xi < e.x + e.w
                                    and e.y <= yi < e.y + e.d
                                    for e in self.boxes
                                ):
                                    supported = False
                                    break
                            if not supported:
                                break
                        if not supported:
                            continue
                    if x + w > self.width or y + d > self.depth: continue
                    
                    dist = math.dist([0,0], [x,y])
                    if distances == [] or dist < min(distances): 
                        cur_candidate = candidate
                        distances.append(dist)

            # place it
            if distances == []:
                return None
            else:   
                self.boxes.append(cur_candidate)
                return cur_candidate

        return None


def load_dims_from_csv(path: str) -> List[Tuple[int, int, int]]:
    dims: List[Tuple[int, int, int]] = []
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


def plot_space(space: Space) -> None:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    vols = [b.volume for b in space.boxes]
    if vols:
        cmap = plt.cm.viridis
        norm = plt.Normalize(min(vols), max(vols))
    for b in space.boxes:
        color = cmap(norm(b.volume)) if vols else "skyblue"
        ax.bar3d(b.x, b.y, b.z, b.w, b.d, b.h,
                 color=color, edgecolor="k", alpha=0.7)
    ax.set_xlim(0, space.width)
    ax.set_ylim(0, space.depth)
    ax.set_zlim(0, space.height)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.title(f"Placed {len(space.boxes)} boxes")
    plt.tight_layout()
    plt.show()


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python place_boxes.py boxes.csv")
        sys.exit(1)
    csv_path = sys.argv[1]

    dims = load_dims_from_csv(csv_path)
    if not dims:
        print("No valid boxes to place.")
        sys.exit(1)

    # sort by descending volume
    dims.sort(key=lambda t: t[0] * t[1] * t[2], reverse=True)

    # Space parameters
    SPACE_W, SPACE_D, SPACE_H = 15, 15, 10
    MAX_ATTEMPTS = 10_000

    space = Space(SPACE_W, SPACE_D, SPACE_H)

    # Place in sequence, stop at first failure
    for idx, (w, d, h) in enumerate(dims, start=1):
        placed = space.place_with_gravity(w, d, h, MAX_ATTEMPTS)
        if placed:
            print(f"#{idx:02d} Placed size={w}×{d}×{h} at x={placed.x},"
                  f" y={placed.y}, z={placed.z} (vol={placed.volume})")
        else:
            print(f"#{idx:02d} FAILED to place size={w}×{d}×{h} (vol={w*d*h}) "
                  f"after {MAX_ATTEMPTS} attempts. Stopping.")
            break

    print(f"\nSummary: placed {len(space.boxes)} / {len(dims)} boxes.")
    plot_space(space)


if __name__ == "__main__":
    main()
