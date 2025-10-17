"""
visualize_height_map.py

Load a JSON Lines file of box‐packing records (as produced by collect_data.py),
extract a single 24×24 height_map, and render it either as a 2D heatmap or a 3D surface.
"""

import json
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_height_maps(jsonl_path: str):
    """
    Read through the JSON Lines file, extract the 'height_map' field
    from each record, return a list of 2D Python lists.
    """
    maps = []
    with open(jsonl_path, 'r') as f:
        for lineno, line in enumerate(f, 1):
            try:
                rec = json.loads(line)
                hm = rec.get("height_map", None)
                if hm is None:
                    continue
                # we expect hm to be a 2D list of size 24×24
                maps.append(hm)
            except json.JSONDecodeError:
                print(f"Warning: skipping invalid JSON on line {lineno}")
    return maps


def plot_heatmap(hm: np.ndarray, cmap="viridis"):
    """
    Plot the 2D height‐map as a colored heatmap.
    """
    plt.figure(figsize=(6,6))
    plt.imshow(hm.T, origin='lower', cmap=cmap, interpolation='nearest')
    plt.colorbar(label="stack height")
    plt.title("Height map (top-down view)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()


def plot_surface(hm: np.ndarray, cmap="terrain"):
    """
    Plot the 2D height‐map as a 3D surface.
    """
    W, D = hm.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(D), indexing='ij')

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        X, Y, hm,
        rstride=1, cstride=1,
        cmap=cmap, edgecolor='k', linewidth=0.3, antialiased=True
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, label="stack height")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")
    ax.set_title("Height map (3D surface)")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a single 24×24 height_map from a data.jsonl file"
    )
    parser.add_argument("jsonl_file",
        help="path to the JSON Lines file (e.g. out.jsonl)")
    parser.add_argument("--index", "-i", type=int, default=0,
        help="which record index to plot (0-based); -1 for a random record")
    parser.add_argument("--mode", "-m", choices=["heatmap","surface"],
        default="heatmap",
        help="2D heatmap or 3D surface")
    args = parser.parse_args()

    # Load all height maps
    print(f"Loading height maps from '{args.jsonl_file}' …")
    maps = load_height_maps(args.jsonl_file)
    if not maps:
        print("No height_map entries found. Exiting.")
        return

    idx = args.index
    if idx < 0:
        idx = random.randrange(len(maps))
    elif idx >= len(maps):
        print(f"Index {idx} out of range; there are only {len(maps)} records.")
        return

    hm = np.array(maps[idx], dtype=int)
    if hm.shape != (24,24):
        print(f"Warning: record {idx} has shape {hm.shape}, not (24,24)")

    print(f"Plotting record {idx} / {len(maps)} as a {args.mode} …")
    if args.mode == "heatmap":
        plot_heatmap(hm)
    else:
        plot_surface(hm)


if __name__ == "__main__":
    main()
