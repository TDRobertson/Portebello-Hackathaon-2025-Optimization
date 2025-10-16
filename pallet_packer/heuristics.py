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


def _has_gap_below(placements: List[Placement], p: Placement, eps: float = 1e-6) -> bool:
    """Check if there's empty space below this placement"""
    if p.z <= eps:  # On the ground
        return False
    
    # Check if there are any boxes directly below this one
    for q in placements:
        if q.z + q.height <= p.z + eps:  # q is at or below p's bottom
            # Check if q's top face overlaps with p's bottom face
            if (q.x < p.x + p.length and q.x + q.length > p.x and 
                q.y < p.y + p.width and q.y + q.width > p.y and
                abs(q.z + q.height - p.z) < eps):
                return False  # Found support directly below
    return True  # No support found, there's a gap


def pack(spec: InputSpec) -> Result:
    pallet = spec.pallet
    candidates = _sort_for_stability(_boxes_expanded(spec.boxes))

    placements: List[Placement] = []
    unplaced: List[Tuple[str, int]] = []

    for box, idx in candidates:
        placed = False
        best_placement = None
        best_score = float('inf')
        
        # Try all possible placement positions and orientations
        # Get all possible z-levels (base + tops of existing boxes)
        z_levels = [0.0] + [p.z + p.height for p in placements]
        z_levels = sorted(set(z_levels))
        
        # Try each z-level from bottom to top
        for z_level in z_levels:
            if z_level >= pallet.height_limit - 1e-9:
                continue
                
            # Try all orientations
            for L, W, H, tag in box.orientations():
                if z_level + H > pallet.height_limit + 1e-9:
                    continue
                    
                # Generate candidate positions:
                # 1. Corner positions (0,0)
                # 2. Positions aligned with existing box edges at this z-level
                x_coords = {0.0}
                y_coords = {0.0}
                
                for p in placements:
                    x_coords.add(p.x)
                    x_coords.add(p.x + p.length)
                    y_coords.add(p.y)
                    y_coords.add(p.y + p.width)
                
                # Try each position
                for x in sorted(x_coords):
                    for y in sorted(y_coords):
                        if x + L > pallet.length + 1e-9 or y + W > pallet.width + 1e-9:
                            continue
                            
                        p = Placement(
                            box_id=box.id,
                            index=idx,
                            x=x,
                            y=y,
                            z=z_level,
                            length=L,
                            width=W,
                            height=H,
                            orientation=tag,
                            label_side=box.label_side,
                        )
                        
                        # Check if placement is valid
                        if not _fits_within(pallet.length, pallet.width, pallet.height_limit, p):
                            continue
                            
                        # Check for overlaps
                        overlap = False
                        for q in placements:
                            if _does_overlap(p, q):
                                overlap = True
                                break
                        if overlap:
                            continue
                            
                        # Support check: require full coverage for z > 0
                        if z_level > 0.0:
                            coverage = _support_coverage(placements, p)
                            if coverage < 0.999:  # Require nearly full support
                                continue
                        
                        # Calculate placement score (lower is better)
                        # Strongly prioritize lower z-levels
                        score = z_level * 1000
                        
                        # Add position in x,y to break ties
                        score += (x + y) * 0.01
                        
                        # Label exposure: small bonus for good label placement
                        label_ok = _can_orient_label_on_perimeter(pallet.length, pallet.width, p)
                        if label_ok and box.label_side is not None:
                            score -= 100  # Bonus for good label placement
                        
                        # Keep track of best placement
                        if score < best_score:
                            best_score = score
                            best_placement = p
        
        # Place the best placement found
        if best_placement is not None:
            placements.append(best_placement)
            placed = True

        if not placed:
            unplaced.append((box.id, idx))

    # Compute metrics
    used_height = max([p.z + p.height for p in placements], default=0.0)
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



