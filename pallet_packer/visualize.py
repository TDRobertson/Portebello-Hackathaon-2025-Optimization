from __future__ import annotations

from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .models import Placement, Pallet


def _cuboid_data(o, size=(1,1,1)):
    X = [[0,0,1,1,0],[0,0,1,1,0],[0,0,0,0,0],[1,1,1,1,1],[0,0,1,1,0],[0,0,1,1,0]]
    Y = [[0,1,1,0,0],[0,0,0,0,0],[0,0,1,1,0],[0,0,1,1,0],[1,1,1,1,1],[0,0,0,0,0]]
    Z = [[0,0,0,0,0],[1,1,1,1,1],[0,1,1,0,0],[0,1,1,0,0],[0,0,0,0,0],[1,1,1,1,1]]
    X = [[o[0] + xi*size[0] for xi in x] for x in X]
    Y = [[o[1] + yi*size[1] for yi in y] for y in Y]
    Z = [[o[2] + zi*size[2] for zi in z] for z in Z]
    return X, Y, Z


def render(pallet: Pallet, placements: List[Placement], path: str) -> None:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, pallet.length)
    ax.set_ylim(0, pallet.width)
    ax.set_zlim(0, pallet.height_limit)
    ax.set_xlabel('X (length)')
    ax.set_ylabel('Y (width)')
    ax.set_zlabel('Z (height)')

    colors = {}
    palette = plt.cm.get_cmap('tab20', 20)
    def color_for(box_id: str):
        if box_id not in colors:
            colors[box_id] = palette(len(colors) % 20)
        return colors[box_id]

    for p in placements:
        X, Y, Z = _cuboid_data((p.x, p.y, p.z), (p.length, p.width, p.height))
        faces = []
        for i in range(len(X)):
            verts = list(zip(X[i], Y[i], Z[i]))
            faces.append(verts)
        poly = Poly3DCollection(faces, facecolors=color_for(p.box_id), linewidths=0.5, edgecolors='k', alpha=0.7)
        ax.add_collection3d(poly)

    ax.view_init(elev=20, azim=35)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


