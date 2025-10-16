import random
from typing import List, Tuple
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

    def place_with_gravity(self,
                           w: int, d: int, h: int,
                           max_attempts: int = 1000) -> bool:
        """
        Try up to max_attempts to place a w×d×h box by:
         1) choosing random (x,y)
         2) letting it fall to z_floor = max(0, all supporting tops)
         3) verifying no 3D overlap
         4) verifying full support under its footprint (no overhang)
         5) verifying it stays under the ceiling
        """
        for _ in range(max_attempts):
            x = random.randint(0, self.width  - w)
            y = random.randint(0, self.depth  - d)

            # find the highest top‐face among boxes whose XY‐footprint overlaps
            z_floor = 0
            for e in self.boxes:
                if not (x + w <= e.x or e.x + e.w <= x or
                        y + d <= e.y or e.y + e.d <= y):
                    z_floor = max(z_floor, e.z + e.h)

            # check ceiling
            if z_floor + h > self.height:
                continue

            candidate = Box(x, y, z_floor, w, d, h)

            # 3D‐overlap check
            if any(candidate.overlaps(e) for e in self.boxes):
                continue

            # support check: if not on floor, every integer cell beneath
            # must be covered by some box top at z_floor
            if z_floor > 0:
                supported = True
                for xi in range(x, x + w):
                    for yi in range(y, y + d):
                        # is there a box e with top == z_floor covering (xi,yi)?
                        found = False
                        for e in self.boxes:
                            if e.z + e.h == z_floor:
                                if e.x <= xi < e.x + e.w and e.y <= yi < e.y + e.d:
                                    found = True
                                    break
                        if not found:
                            supported = False
                            break
                    if not supported:
                        break
                if not supported:
                    continue

            # OK—place it
            self.boxes.append(candidate)
            return True

        return False


def plot_space(space: Space) -> None:
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    vols = [b.volume for b in space.boxes]
    if vols:
        cmap = plt.cm.viridis
        norm = plt.Normalize(min(vols), max(vols))
    for b in space.boxes:
        color = cmap(norm(b.volume)) if vols else 'skyblue'
        ax.bar3d(b.x, b.y, b.z, b.w, b.d, b.h,
                 color=color, edgecolor='k', alpha=0.7)
    ax.set_xlim(0, space.width)
    ax.set_ylim(0, space.depth)
    ax.set_zlim(0, space.height)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.title(f'Placed {len(space.boxes)} boxes')
    plt.tight_layout()
    plt.show()


def main() -> None:
    # PARAMETERS
    SPACE_W, SPACE_D, SPACE_H = 15, 15, 10
    NUM_BOXES = 50
    MIN_SIZE, MAX_SIZE = 1, 5
    MAX_ATTEMPTS = 500

    # 1) generate dims & sort large→small
    dims: List[Tuple[int,int,int]] = [
        (random.randint(MIN_SIZE, MAX_SIZE),
         random.randint(MIN_SIZE, MAX_SIZE),
         random.randint(MIN_SIZE, MAX_SIZE))
        for _ in range(NUM_BOXES)
    ]
    dims.sort(key=lambda t: t[0]*t[1]*t[2], reverse=True)

    # 2) place them
    space = Space(SPACE_W, SPACE_D, SPACE_H)
    for i, (w,d,h) in enumerate(dims, 1):
        if not space.place_with_gravity(w,d,h, MAX_ATTEMPTS):
            print(f"✗ Failed to place #{i} size={w}×{d}×{h}")
    print(f"✓ Placed {len(space.boxes)} / {NUM_BOXES} boxes")

    # 3) visualize
    plot_space(space)


if __name__ == "__main__":
    main()
