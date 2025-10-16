from __future__ import annotations

from typing import List, Tuple
from .models import InputSpec, Placement, Result, BoxType


def _boxes_expanded(boxes: List[BoxType]) -> List[Tuple[BoxType, int]]:
    expanded: List[Tuple[BoxType, int]] = []
    for b in boxes:
        for i in range(b.quantity):
            expanded.append((b, i))
    return expanded


def _sort_for_stability(expanded: List[Tuple[BoxType, int]]) -> List[Tuple[BoxType, int]]:
    # Larger footprint and taller first to improve base stability
    return sorted(
        expanded,
        key=lambda t: (
            max(t[0].length, t[0].width) * min(t[0].length, t[0].width),  # footprint area
            t[0].height,
        ),
        reverse=True,
    )


def _does_overlap(a: Placement, b: Placement) -> bool:
    ax1, ay1, az1, ax2, ay2, az2 = a.bbox()
    bx1, by1, bz1, bx2, by2, bz2 = b.bbox()
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1 or az2 <= bz1 or bz2 <= az1)


def _fits_within(pallet_l: float, pallet_w: float, height_limit: float, p: Placement) -> bool:
    return (
        p.x >= 0
        and p.y >= 0
        and p.z >= 0
        and p.x + p.length <= pallet_l
        and p.y + p.width <= pallet_w
        and p.z + p.height <= height_limit
    )


def _can_orient_label_on_perimeter(pallet_l: float, pallet_w: float, p: Placement) -> bool:
    # If label specified, try to align on outer faces when possible.
    if p.label_side is None:
        return True
    if p.label_side == "+x":
        return abs((p.x + p.length) - pallet_l) < 1e-6
    if p.label_side == "-x":
        return abs(p.x - 0) < 1e-6
    if p.label_side == "+y":
        return abs((p.y + p.width) - float(pallet_w)) < 1e-6
    if p.label_side == "-y":
        return abs(p.y - 0) < 1e-6
    # For +z/-z we ensure top or pallet floor contact
    if p.label_side == "+z":
        return True  # will be visible at top at final height
    if p.label_side == "-z":
        return abs(p.z - 0) < 1e-6
    return True


def _support_coverage(placements: List[Placement], p: Placement, eps: float = 1e-6) -> float:
    # Compute fraction of p's footprint covered by boxes whose top z equals p.z
    if abs(p.z) < eps:
        return 1.0
    target = (p.x, p.y, p.x + p.length, p.y + p.width)
    # Collect supporting rectangles clipped to target
    rects: List[Tuple[float, float, float, float]] = []
    for q in placements:
        top_z = q.z + q.height
        if abs(top_z - p.z) > eps:
            continue
        qrect = (q.x, q.y, q.x + q.length, q.y + q.width)
        ix1 = max(target[0], qrect[0])
        iy1 = max(target[1], qrect[1])
        ix2 = min(target[2], qrect[2])
        iy2 = min(target[3], qrect[3])
        if ix2 - ix1 > eps and iy2 - iy1 > eps:
            rects.append((ix1, iy1, ix2, iy2))
    if not rects:
        return 0.0
    # Plane sweep on x to compute union area within target
    xs = sorted({target[0], target[2]} | {x for r in rects for x in (r[0], r[2])})
    area = 0.0
    for i in range(len(xs) - 1):
        x1, x2 = xs[i], xs[i + 1]
        if x2 - x1 <= eps:
            continue
        intervals: List[Tuple[float, float]] = []
        for rx1, ry1, rx2, ry2 in rects:
            if rx1 < x2 - eps and rx2 > x1 + eps:
                intervals.append((max(ry1, target[1]), min(ry2, target[3])))
        if not intervals:
            continue
        # Union length of y-intervals
        intervals.sort()
        covered = 0.0
        cy1, cy2 = intervals[0]
        for y1, y2 in intervals[1:]:
            if y1 <= cy2 + eps:
                cy2 = max(cy2, y2)
            else:
                covered += max(0.0, cy2 - cy1)
                cy1, cy2 = y1, y2
        covered += max(0.0, cy2 - cy1)
        area += covered * (x2 - x1)
    target_area = (target[2] - target[0]) * (target[3] - target[1])
    return area / target_area if target_area > eps else 0.0


def pack(spec: InputSpec) -> Result:
    pallet = spec.pallet
    candidates = _sort_for_stability(_boxes_expanded(spec.boxes))

    placements: List[Placement] = []
    unplaced: List[Tuple[str, int]] = []

    # Layered shelf heuristic
    current_z = 0.0
    used_height = 0.0
    layer_height = 0.0

    # Within a layer, maintain shelves along y; along each shelf, fill x.
    shelf_y = 0.0
    shelf_height = 0.0
    cursor_x = 0.0

    for box, idx in candidates:
        placed = False
        # Try existing layer and shelves; if not, create new shelf or new layer.
        for attempt in range(2):  # 0: try new shelves in same layer; 1: open new layer
            if attempt == 1:
                # open a new layer on top of current_z
                current_z += max(layer_height, shelf_height)
                if current_z >= pallet.height_limit - 1e-9:
                    break
                used_height = max(used_height, current_z)
                # reset shelves
                shelf_y = 0.0
                shelf_height = 0.0
                cursor_x = 0.0
                layer_height = 0.0

            # Try orientations (prefer ones increasing footprint stability)
            for L, W, H, tag in box.orientations():
                # If label wants +z/-z, respect orientation height mapping minimally by not rotating that face away
                if box.label_side in {"+z", "-z"} and H not in {box.height}:
                    pass  # still allow; visibility is managed by position

                # Try to place within current shelf
                def try_place_at(x0: float, y0: float, z0: float) -> bool:
                    nonlocal shelf_height, layer_height, cursor_x
                    p = Placement(
                        box_id=box.id,
                        index=idx,
                        x=x0,
                        y=y0,
                        z=z0,
                        length=L,
                        width=W,
                        height=H,
                        orientation=tag,
                        label_side=box.label_side,
                    )
                    if not _fits_within(pallet.length, pallet.width, pallet.height_limit, p):
                        return False
                    # overlap check
                    for q in placements:
                        if _does_overlap(p, q):
                            return False
                    # support check for upper layers: require full support coverage
                    if z0 > 0.0:
                        coverage = _support_coverage(placements, p)
                        if coverage < 0.999:  # effectively 100%
                            return False
                    # label exposure heuristic: prefer perimeter alignment; if not possible, allow but deprioritize
                    if not _can_orient_label_on_perimeter(pallet.length, pallet.width, p):
                        return False

                    placements.append(p)
                    cursor_x = x0 + L
                    shelf_height = max(shelf_height, H)
                    layer_height = max(layer_height, shelf_height)
                    return True

                # 1) Place in current shelf if fits in x
                if cursor_x + L <= pallet.length + 1e-9 and shelf_y + W <= pallet.width + 1e-9:
                    if try_place_at(cursor_x, shelf_y, current_z):
                        placed = True
                        break

                # 2) Start new shelf in same layer, if width allows
                if shelf_y + shelf_height + W <= pallet.width + 1e-9:
                    if try_place_at(0.0, shelf_y + shelf_height, current_z):
                        # moved to new shelf
                        shelf_y = shelf_y + shelf_height
                        cursor_x = L
                        shelf_height = max(H, shelf_height)
                        placed = True
                        break

            if placed:
                break

        if not placed:
            unplaced.append((box.id, idx))

    # Compute metrics
    used_height = max(used_height, current_z + max(layer_height, shelf_height))
    pallet_volume = pallet.length * pallet.width * pallet.height_limit
    used_volume = sum(p.length * p.width * p.height for p in placements)
    utilization = used_volume / pallet_volume if pallet_volume > 0 else 0.0

    # Sequences: pick larger first, stack in placement z-then-y-then-x order
    pick_sequence = [(b.id, i) for b, i in _sort_for_stability(_boxes_expanded(spec.boxes))]
    stack_sequence = [
        (p.box_id, p.index)
        for p in sorted(placements, key=lambda r: (r.z, r.y, r.x))
    ]

    return Result(
        placements=placements,
        unplaced=unplaced,
        utilization=utilization,
        used_height=used_height,
        pick_sequence=pick_sequence,
        stack_sequence=stack_sequence,
    )


