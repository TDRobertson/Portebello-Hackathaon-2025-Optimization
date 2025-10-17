import csv
import sys
import random
from itertools import permutations
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


Dims = Tuple[int, int, int]
Action = Tuple[str, Tuple[int, ...]]  # ("place", (x, y, w, d, h)) or ("rotate", (w,d,h))


class Box:
    """A single rectangular prism placed in the space."""
    def __init__(self, x: int, y: int, z: int, w: int, d: int, h: int) -> None:
        self.x, self.y, self.z = x, y, z
        self.w, self.d, self.h = w, d, h
        self.volume = w * d * h

    def overlaps(self, other: "Box") -> bool:
        """Check if this box overlaps another in 3D."""
        return (
            self.x < other.x + other.w and
            self.x + self.w > other.x and
            self.y < other.y + other.d and
            self.y + self.d > other.y and
            self.z < other.z + other.h and
            self.z + self.h > other.z
        )


class Space:
    """Container in which we place boxes under gravity and support rules."""
    def __init__(self, width: int, depth: int, height: int) -> None:
        self.width, self.depth, self.height = width, depth, height
        self.boxes: List[Box] = []

    def find_floor_z(self, x: int, y: int, w: int, d: int) -> int:
        """
        Given a footprint at (x,y) of size w×d, return the z-coordinate
        where this box would come to rest (on floor or on top of existing boxes).
        """
        z_floor = 0
        for b in self.boxes:
            # Check if footprints overlap in XY
            if not (x + w <= b.x or b.x + b.w <= x or
                    y + d <= b.y or b.y + b.d <= y):
                z_floor = max(z_floor, b.z + b.h)
        return z_floor

    def is_fully_supported(self, x: int, y: int, w: int, d: int, zf: int) -> bool:
        """
        If zf > 0, ensure every cell under the w×d footprint is
        supported by some existing box whose top is exactly zf.
        """
        if zf == 0:
            return True  # floor support

        # For each integer coordinate in the footprint
        for xi in range(x, x + w):
            for yi in range(y, y + d):
                # Must find at least one box top exactly at zf covering (xi, yi)
                if not any(
                    b.z + b.h == zf and
                    b.x <= xi < b.x + b.w and
                    b.y <= yi < b.y + b.d
                    for b in self.boxes
                ):
                    return False
        return True

    def legal_xy_positions(self, w: int, d: int, h: int) -> List[Tuple[int, int, int]]:
        """
        Return a list of (x, y, zf) positions where a w×d×h box
        could legally be placed under current packing state.
        """
        legal_positions = []
        max_x = self.width - w
        max_y = self.depth - d

        for x in range(0, max_x + 1):
            for y in range(0, max_y + 1):
                zf = self.find_floor_z(x, y, w, d)

                # 1) Must fit below the top of the container
                if zf + h > self.height:
                    continue

                # 2) Must not overlap any existing box in 3D
                candidate = Box(x, y, zf, w, d, h)
                if any(candidate.overlaps(b) for b in self.boxes):
                    continue

                # 3) Must be fully supported (no overhangs)
                if not self.is_fully_supported(x, y, w, d, zf):
                    continue

                legal_positions.append((x, y, zf))

        return legal_positions

    def place_box(self, x: int, y: int, z: int, w: int, d: int, h: int) -> Box:
        """Instantiate and record a new box at the given coords."""
        new_box = Box(x, y, z, w, d, h)
        self.boxes.append(new_box)
        return new_box


def load_dims_from_csv(path: str) -> List[Dims]:
    """Read triples of positive ints from a CSV, skipping invalid lines."""
    dims_list: List[Dims] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader, start=1):
            if len(row) < 3:
                print(f"Skipping line {idx}: not enough columns")
                continue
            try:
                w, d, h = map(int, row[:3])
                if w <= 0 or d <= 0 or h <= 0:
                    raise ValueError
            except ValueError:
                print(f"Skipping line {idx}: invalid dims {row}")
                continue
            dims_list.append((w, d, h))
    return dims_list


def get_allowed_orientations(dims: Dims) -> Tuple[List[Dims], int]:
    """
    Given (w,d,h), return:
      - allowed_orients: all unique permutations whose base-area != max_face_area
      - max_face_area    : the area of the largest face (used to detect 'flat-on-largest' state)
    """
    w, d, h = dims
    face_areas = [w*d, w*h, d*h]
    max_face_area = max(face_areas)
    # All unique orientations
    all_orients = set(permutations(dims))
    # Only those whose base (first two) area != max_face_area
    allowed = [o for o in all_orients if o[0] * o[1] != max_face_area]
    return allowed, max_face_area


def plot_space(space: Space) -> None:
    """3D bar‐plot of all placed boxes, colored by volume."""
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    volumes = [b.volume for b in space.boxes]
    cmap = plt.cm.viridis if volumes else None
    norm = plt.Normalize(min(volumes), max(volumes)) if volumes else None

    for b in space.boxes:
        color = cmap(norm(b.volume)) if cmap else "skyblue"
        ax.bar3d(b.x, b.y, b.z, b.w, b.d, b.h,
                 color=color, edgecolor="k", alpha=0.7)

    ax.set_xlim(0, space.width)
    ax.set_ylim(0, space.depth)
    ax.set_zlim(0, space.height)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python place_boxes.py boxes.csv")
        sys.exit(1)

    dims_list = load_dims_from_csv(sys.argv[1])
    if not dims_list:
        print("No valid boxes found.")
        sys.exit(1)

    # Sort boxes by descending volume
    dims_list.sort(key=lambda t: t[0] * t[1] * t[2], reverse=True)

    SPACE_W, SPACE_D, SPACE_H = 15, 15, 10
    space = Space(SPACE_W, SPACE_D, SPACE_H)

    for idx, dims in enumerate(dims_list, start=1):
        w0, d0, h0 = dims
        allowed_orients, max_face_area = get_allowed_orientations(dims)
        current = dims
        rotations_used = 0

        print(f"Box #{idx:02d}: orig={dims}, volume={w0*d0*h0}")

        while True:
            # Are we lying flat on the largest face?
            base_area = current[0] * current[1]
            on_largest_face = (base_area == max_face_area)

            # 1) Collect legal place actions (only if not on largest face)
            place_actions: List[Action] = []
            if not on_largest_face:
                for x, y, zf in space.legal_xy_positions(*current):
                    place_actions.append(("place", (x, y, *current)))

            # 2) Collect legal rotate actions (as long as we have budget)
            rotate_actions: List[Action] = []
            if rotations_used < 3:
                for orient in allowed_orients:
                    if orient != current:
                        rotate_actions.append(("rotate", orient))

            # Combine and report
            actions = place_actions + rotate_actions
            print(f"  Orientation={current}  rotations_used={rotations_used}")
            print(f"    → {len(place_actions)} place-actions, {len(rotate_actions)} rotate-actions")

            if not actions:
                print("    ! No legal actions available; skipping box.\n")
                break

            action, payload = random.choice(actions)
            if action == "rotate":
                current = payload  # payload is a new (w,d,h)
                rotations_used += 1
                print(f"    * ROTATE → new orientation={current}")
            else:
                # payload = (x, y, w, d, h)
                x, y, w, d, h = payload
                zf = space.find_floor_z(x, y, w, d)
                placed = space.place_box(x, y, zf, w, d, h)
                print(f"    ✔ PLACE at x={placed.x}, y={placed.y}, z={placed.z}\n")
                break

    print(f"Summary: placed {len(space.boxes)}/{len(dims_list)} boxes.")
    plot_space(space)


if __name__ == "__main__":
    main()
