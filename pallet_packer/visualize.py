from __future__ import annotations

from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from .models import Placement, Pallet


def _cuboid_data(o, size=(1,1,1)):
    X = [[0,0,1,1,0],[0,0,1,1,0],[0,0,0,0,0],[1,1,1,1,1],[0,0,1,1,0],[0,0,1,1,0]]
    Y = [[0,1,1,0,0],[0,0,0,0,0],[0,0,1,1,0],[0,0,1,1,0],[1,1,1,1,1],[0,0,0,0,0]]
    Z = [[0,0,0,0,0],[1,1,1,1,1],[0,1,1,0,0],[0,1,1,0,0],[0,0,0,0,0],[1,1,1,1,1]]
    X = [[o[0] + xi*size[0] for xi in x] for x in X]
    Y = [[o[1] + yi*size[1] for yi in y] for y in Y]
    Z = [[o[2] + zi*size[2] for zi in z] for z in Z]
    return X, Y, Z


def render_interactive(pallet: Pallet, placements: List[Placement]) -> None:
    """Create an interactive 3D visualization of the pallet layout"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up the plot
    ax.set_xlim(0, pallet.length)
    ax.set_ylim(0, pallet.width)
    ax.set_zlim(0, pallet.height_limit)
    ax.set_xlabel('X (length)', fontsize=12)
    ax.set_ylabel('Y (width)', fontsize=12)
    ax.set_zlabel('Z (height)', fontsize=12)
    ax.set_title('Interactive 3D Pallet Layout', fontsize=14, fontweight='bold')

    # Create color mapping for different box types
    colors = {}
    palette = plt.cm.get_cmap('tab20', 20)
    def color_for(box_id: str):
        if box_id not in colors:
            colors[box_id] = palette(len(colors) % 20)
        return colors[box_id]

    # Add pallet outline
    pallet_vertices = [
        [0, 0, 0], [pallet.length, 0, 0], [pallet.length, pallet.width, 0], [0, pallet.width, 0],
        [0, 0, 0], [0, 0, pallet.height_limit], [pallet.length, 0, pallet.height_limit], 
        [pallet.length, pallet.width, pallet.height_limit], [0, pallet.width, pallet.height_limit]
    ]
    
    # Draw pallet base
    base_vertices = [[0, 0, 0], [pallet.length, 0, 0], [pallet.length, pallet.width, 0], [0, pallet.width, 0]]
    base_faces = [base_vertices]
    base_poly = Poly3DCollection(base_faces, facecolors='lightgray', alpha=0.3, linewidths=1, edgecolors='black')
    ax.add_collection3d(base_poly)

    # Add boxes
    box_polys = []
    for i, p in enumerate(placements):
        X, Y, Z = _cuboid_data((p.x, p.y, p.z), (p.length, p.width, p.height))
        faces = []
        for j in range(len(X)):
            verts = list(zip(X[j], Y[j], Z[j]))
            faces.append(verts)
        
        color = color_for(p.box_id)
        # Make boxes solid but with transparency for depth perception
        # Use slightly different alpha for different faces to create depth effect
        poly = Poly3DCollection(faces, facecolors=color, linewidths=1.2, edgecolors='black', alpha=0.7)
        ax.add_collection3d(poly)
        box_polys.append((poly, p))

    # Add legend
    legend_elements = []
    for box_id, color in colors.items():
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=box_id))
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

    # Set initial view
    ax.view_init(elev=25, azim=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add lighting effect by setting background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Set pane colors to white for better contrast
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    
    # Add subtle lighting
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Add statistics text
    total_boxes = len(placements)
    used_volume = sum(p.length * p.width * p.height for p in placements)
    pallet_volume = pallet.length * pallet.width * pallet.height_limit
    utilization = (used_volume / pallet_volume * 100) if pallet_volume > 0 else 0
    
    stats_text = f'Boxes: {total_boxes}\nUtilization: {utilization:.1f}%\nVolume: {used_volume:.0f}/{pallet_volume:.0f}'
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Enable interactive features
    plt.tight_layout()
    
    # Add mouse interaction info
    info_text = "Interactive Controls:\n• Mouse: Rotate view\n• Scroll: Zoom in/out\n• Right-click: Pan"
    ax.text2D(0.98, 0.02, info_text, transform=ax.transAxes, fontsize=9, 
              verticalalignment='bottom', horizontalalignment='right',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Show the interactive plot
    plt.show()


def render(pallet: Pallet, placements: List[Placement], path: str) -> None:
    """Create a static image and save it to file"""
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
        # Make boxes solid but with transparency for depth perception
        poly = Poly3DCollection(faces, facecolors=color_for(p.box_id), linewidths=1.0, edgecolors='black', alpha=0.6)
        ax.add_collection3d(poly)

    ax.view_init(elev=20, azim=35)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


