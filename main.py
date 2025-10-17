import csv
import sys
import random
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import math


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
                 color=color, edgecolor="k", alpha=1)
    ax.set_xlim(0, space.width)
    ax.set_ylim(0, space.depth)
    ax.set_zlim(0, space.height)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.title(f"Placed {len(space.boxes)} boxes")
    plt.tight_layout()
    plt.show()

def plot_space_interactive(space: Space, update_interval: int = 2000) -> None:
    from matplotlib.animation import FuncAnimation
    from collections import defaultdict
    """
    Create an interactive 3D plot that adds boxes layer by layer based on z-coordinate.
    
    Args:
        space: Space object containing boxes
        update_interval: Time between layer additions in milliseconds (default: 2000ms = 2 seconds)
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # Group boxes by layers (z coordinate)
    groups = defaultdict(list)
    for b in space.boxes:
        groups[b.z].append(b)
    
    # Sort layers by z coordinate
    sorted_z_values = sorted(groups.keys())
    box_layers = [groups[z] for z in sorted_z_values]
    
    # Setup colormap
    vols = [b.volume for b in space.boxes]
    if vols:
        cmap = plt.cm.viridis
        norm = plt.Normalize(min(vols), max(vols))
    
    def update_plot(frame):
        """Update function called at each animation frame"""
        ax.clear()
        
        # Draw boxes up to current layer (frame)
        layers_to_show = min(frame + 1, len(box_layers))
        boxes_shown = 0
        
        for layer_idx in range(layers_to_show):
            for b in box_layers[layer_idx]:
                color = cmap(norm(b.volume)) if vols else "skyblue"
                ax.bar3d(b.x, b.y, b.z, b.w, b.d, b.h,
                        color=color, edgecolor="k", alpha=0.7)
                boxes_shown += 1
        
        # Set axis properties
        ax.set_xlim(0, space.width)
        ax.set_ylim(0, space.depth)
        ax.set_zlim(0, space.height)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Placed {boxes_shown}/{len(space.boxes)} boxes | Z-Layer {layers_to_show}/{len(box_layers)}")
        
        return ax,
    
    # Create animation - frames = number of layers
    anim = FuncAnimation(fig, update_plot, frames=len(box_layers), 
                        interval=update_interval, blit=False, 
                        repeat=False, cache_frame_data=False)
    
    plt.tight_layout()
    plt.show()
    
    return anim  # Return animation object to keep it alive


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print("Usage: python place_boxes.py boxes.csv [output.csv]")
        sys.exit(1)
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) == 3 else "placements.csv"

    dims = load_dims_from_csv(input_csv)
    if not dims:
        print("No valid boxes to place.")
        sys.exit(1)

    # sort by descending volume
    dims.sort(key=lambda t: t[0] * t[1] * t[2], reverse=True)

    # Space parameters
    SPACE_W, SPACE_D, SPACE_H = 15, 15, 10
    MAX_ATTEMPTS = 10_000

    space = Space(SPACE_W, SPACE_D, SPACE_H)

    # open output CSV and write header
    with open(output_csv, "w", newline="") as outf:
        writer = csv.writer(outf)
        writer.writerow(["id", "w", "d", "h", "x", "y", "z"])

        # Place in sequence, stop at first failure
        for idx, (w, d, h) in enumerate(dims, start=1):
            placed = space.place_with_gravity(w, d, h, MAX_ATTEMPTS)
            if placed:
                print(f"#{idx:02d} Placed size={w}×{d}×{h} "
                      f"at x={placed.x}, y={placed.y}, z={placed.z}")
                writer.writerow([idx, w, d, h,
                                 placed.x, placed.y, placed.z])
            else:
                print(f"#{idx:02d} FAILED to place size={w}×{d}×{h} "
                      f"after {MAX_ATTEMPTS} attempts. Stopping.")
                break

    print(f"\nSummary: placed {len(space.boxes)} / {len(dims)} boxes.")
    print(f"Placements written to '{output_csv}'")
    anim = plot_space_interactive(space)


if __name__ == "__main__":
    main()
