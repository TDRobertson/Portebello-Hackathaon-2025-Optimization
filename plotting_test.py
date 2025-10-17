import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.ion()
plt.style.use('_mpl-gallery')

import csv
import sys
import random
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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


import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # Needed even if unused directly
import time

def plot_space(boxes: pd.DataFrame, pause_time: float = 0) -> None:
    # Compute volume if not already in DataFrame
    if 'volume' not in boxes.columns:
        boxes['volume'] = boxes['w'] * boxes['d'] * boxes['h']
    
    vols = boxes['volume'].tolist()
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(vols), max(vols)) if vols else None

    # Precompute axis limits
    xlim = boxes['x'].max() + boxes['w'].max()
    ylim = boxes['y'].max() + boxes['d'].max()
    zlim = boxes['z'].max() + boxes['h'].max()

    # Enable interactive mode
    plt.ion()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Clear previous data before each draw
    ax.clear()

    for i in range(len(boxes)):
        # Clear axes to redraw all boxes
        ax.clear()

        # Draw all placed boxes so far
        for j in range(i + 1):
            b = boxes.iloc[j]
            color = cmap(norm(b['volume'])) if norm else "skyblue"
            ax.bar3d(b['x'], b['y'], b['z'], b['w'], b['d'], b['h'],
                     color=color, edgecolor="k", alpha=0.7)

        # Set fixed limits
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, ylim)
        ax.set_zlim(0, zlim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Placed {i + 1} of {len(boxes)} boxes")

        # Pause to update the figure
        plt.draw()
        if pause_time > 0:
            plt.pause(pause_time)
        else:
            input("Press Enter to place the next box...")

    # Turn off interactive mode and show final plot
    plt.ioff()
    plt.show()





# Read CSV into a DataFrame
df = pd.read_csv('placements.csv')  # replace with your actual file name

# Combine the last 6 columns into a tuple and store in a new column
df['dims_and_pos'] = list(zip(df['w'], df['d'], df['h'], df['x'], df['y'], df['z']))

plot_space(df, 1.0)