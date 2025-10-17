import matplotlib.pyplot as plt
import numpy as np

plt.ion()
plt.style.use('_mpl-gallery')

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
input()