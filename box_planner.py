import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # type: ignore

class SegmentTree:
    def __init__(self, data):
        n = len(data)
        size = 1
        while size < n:
            size <<= 1
        self.size = size
        self.tree = [(float('inf'), float('inf'))] * (2 * size)
        self.tree = [None] * (2 * size)
        for i in range(n):
            self.tree[size + i] = data[i]
        for i in range(n, size):
            self.tree[size + i] = (float('inf'), float('inf'))
        for i in range(size - 1, 0, -1):
            self.tree[i] = min(self.tree[2 * i], self.tree[2 * i + 1])

    def update(self, idx, val):
        pos = self.size + idx
        self.tree[pos] = val
        pos //= 2
        while pos >= 1:
            self.tree[pos] = min(self.tree[2 * pos], self.tree[2 * pos + 1])
            pos //= 2

    def query(self, l, r):
        res = (float('inf'), float('inf'))
        l += self.size
        r += self.size
        while l < r:
            if l & 1:
                res = min(res, self.tree[l])
                l += 1
            if r & 1:
                r -= 1
                res = min(res, self.tree[r])
            l >>= 1
            r >>= 1
        return res


def merge_skylines(skylines):
    merged = []
    for seg in sorted(skylines):
        if merged and merged[-1][2] == seg[2] and merged[-1][1] == seg[0]:
            merged[-1] = (merged[-1][0], seg[1], seg[2])
        else:
            merged.append(seg)
    return merged


def update_skyline(skylines, x, w, y, h):
    new_segment = (x, x + w, y + h)
    filtered = []
    for seg in skylines:
        if seg[1] <= x or seg[0] >= x + w:
            filtered.append(seg)
    filtered.append(new_segment)
    return merge_skylines(filtered)


def fitness(rect_w, rect_h, gap_w, gap_h):
    fit = 0
    if rect_w == gap_w:
        fit += 1
    if rect_h == gap_h:
        fit += 1
    return fit


def select_gap(skylines, rect):
    rect_w, rect_h = rect
    candidates = []
    for seg in skylines:
        x0, x1, y = seg
        gap_w = x1 - x0
        # consider original orientation
        if gap_w >= rect_w:
            fit = fitness(rect_w, rect_h, gap_w, float('inf'))  # height unbounded here
            candidates.append((y, x0, seg, fit))
        # consider rotated orientation
        if gap_w >= rect_h:
            fit = fitness(rect_h, rect_w, gap_w, float('inf'))
            candidates.append((y, x0, seg, fit))
    if not candidates:
        return None
    # Sort by (height, x position) ascending, largest fitness last for bottom-left heuristic
    candidates.sort(key=lambda x: (x[0], x[1], -x[3]))
    return candidates[0][2]  # return best gap segments


def packing(rectangles, container_width):
    skylines = [(0, container_width, 0)]  # initial flat skyline at height 0
    n = len(rectangles)
    # initialize fitness segment tree; high fitness preferred (negate for min tree)
    data = [((float('inf'), i)) for i in range(n)]
    seg_tree = SegmentTree(data)

    placed = [False] * n
    positions = [None] * n

    for _ in range(n):
        # For simplicity: pick first unplaced rectangle (you can use seg_tree and fitness update for advanced)
        idx = next(i for i, p in enumerate(placed) if not p)
        rect = rectangles[idx]
        gap = select_gap(skylines, rect)
        if gap is None:
            # cannot place any more rectangles
            break
        x0, x1, y0 = gap
        w, h = rect
        positions[idx] = (x0, y0)
        placed[idx] = True
        skylines = update_skyline(skylines, x0, w, y0, h)

    return positions, skylines


if __name__ == "__main__":
    df = pd.read_csv('UTT.csv')
    unconsidered_df = df[(df["Mosaic"] == True) | (df["Width"] == "48")]
    df = df[~((df["Mosaic"] == True) | (df["Width"] == 48))]
    boxes = []
    # Getting length and width for each box
    for index, row in df.iterrows():
        boxes.append((row["Length"], row["Width"]))

    container_w = 48
    positions, final_skylines = packing(boxes, container_w)
    for i, pos in enumerate(positions):
        if pos:
            print(f"Rectangle {i} placed at {pos}")
        else:
            print(f"Rectangle {i} could not be placed.")
    print("Final skylines:", final_skylines)