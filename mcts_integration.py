"""
MCTS Integration Module for PyQt GUI
Modified version of mcts.py to work better with the GUI integration
"""

import csv
import time
import random
import math
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


class Box:
    """Represents a 3D box in the warehouse"""
    __slots__ = ("x", "y", "z", "w", "d", "h", "volume")
    
    def __init__(self, x: int, y: int, z: int, w: int, d: int, h: int) -> None:
        self.x, self.y, self.z = x, y, z
        self.w, self.d, self.h = w, d, h
        self.volume = w * d * h

    def overlaps(self, other: "Box") -> bool:
        """Check if this box overlaps with another box"""
        return (
            self.x < other.x + other.w and
            self.x + self.w > other.x and
            self.y < other.y + other.d and
            self.y + self.d > other.y and
            self.z < other.z + other.h and
            self.z + self.h > other.z
        )


class Space:
    """Represents the 3D warehouse space"""
    
    def __init__(self, width: int, depth: int, height: int) -> None:
        self.width, self.depth, self.height = width, depth, height
        self.boxes: List[Box] = []

    def find_drop_height(self, x: int, y: int, w: int, d: int) -> int:
        """Return highest z at which a new box of footprint (x..x+w, y..y+d) can rest."""
        drop_z = 0
        for existing_box in self.boxes:
            # if footprints overlap in XY
            if not (x + w <= existing_box.x or
                    existing_box.x + existing_box.w <= x or
                    y + d <= existing_box.y or
                    existing_box.y + existing_box.d <= y):
                drop_z = max(drop_z, existing_box.z + existing_box.h)
        return drop_z

    def legal_xy_positions(self, w: int, d: int, h: int) -> List[Tuple[int, int]]:
        """
        Enumerate all legal (x,y) placements for a w×d×h box:
         - in‐bounds
         - under ceiling
         - no 3D overlap
         - full‐footprint support (no overhang)
         - no larger‐on‐smaller support
        """
        legal_positions: List[Tuple[int, int]] = []
        for x in range(0, self.width - w + 1):
            for y in range(0, self.depth - d + 1):
                z_floor = self.find_drop_height(x, y, w, d)
                # under ceiling?
                if z_floor + h > self.height:
                    continue
                candidate = Box(x, y, z_floor, w, d, h)
                # 3D overlap?
                if any(candidate.overlaps(e) for e in self.boxes):
                    continue
                # full‐footprint support
                if z_floor > 0:
                    fully_supported = True
                    for xi in range(x, x + w):
                        for yi in range(y, y + d):
                            # must have some box whose top == z_floor at (xi,yi)
                            if not any(
                                (eb.z + eb.h == z_floor and
                                 eb.x <= xi < eb.x + eb.w and
                                 eb.y <= yi < eb.y + eb.d)
                                for eb in self.boxes
                            ):
                                fully_supported = False
                                break
                        if not fully_supported:
                            break
                    if not fully_supported:
                        continue
                # no larger‐on‐smaller
                violates_support = False
                for supporting_box in self.boxes:
                    if supporting_box.z + supporting_box.h == z_floor:
                        if w * d * h > supporting_box.volume:
                            violates_support = True
                            break
                if violates_support:
                    continue

                legal_positions.append((x, y))
        return legal_positions

    def place_at(self, x: int, y: int, w: int, d: int, h: int) -> Box:
        """Drop and permanently place a box of size w×d×h at horizontal (x,y)."""
        z_floor = self.find_drop_height(x, y, w, d)
        new_box = Box(x, y, z_floor, w, d, h)
        self.boxes.append(new_box)
        return new_box


class Node:
    """MCTS Node representing a state in the search tree"""
    __slots__ = ("parent", "action", "box_index", "placed_boxes",
                 "children", "visits", "total_reward", "_untried_actions")

    def __init__(self,
                 parent: Optional["Node"],
                 action: Optional[Tuple[int, int]],
                 box_index: int,
                 placed_boxes: List[Box]) -> None:
        self.parent = parent
        self.action = action               # the (x,y) that led here
        self.box_index = box_index         # next box to place
        self.placed_boxes = placed_boxes   # current board state
        self.children: List["Node"] = []
        self.visits = 0
        self.total_reward = 0.0
        self._untried_actions: Optional[List[Tuple[int, int]]] = None

    def untried_actions(self, dims: List[Tuple[int, int, int]], 
                       space_dims: Tuple[int, int, int]) -> List[Tuple[int, int]]:
        """Compute (and cache) all legal actions at this node."""
        if self._untried_actions is None:
            if self.box_index >= len(dims):
                self._untried_actions = []
            else:
                w, d, h = dims[self.box_index]
                sim_space = Space(space_dims[0], space_dims[1], space_dims[2])
                sim_space.boxes = list(self.placed_boxes)
                self._untried_actions = sim_space.legal_xy_positions(w, d, h)
        return self._untried_actions

    def is_fully_expanded(self, dims: List[Tuple[int, int, int]], 
                         space_dims: Tuple[int, int, int]) -> bool:
        """Check if this node is fully expanded (no untried actions)"""
        if self._untried_actions is None:
            self._untried_actions = self.untried_actions(dims, space_dims)
        return len(self._untried_actions) == 0

    def is_terminal(self, total_boxes: int) -> bool:
        # terminal if we've placed all boxes
        return self.box_index >= total_boxes


def uct_select(children: List[Node]) -> Optional[Node]:
    """
    Select a child with highest UCT value:
       Q_i / N_i  + c * sqrt( ln(N_parent) / N_i )
    Returns None if children is empty.
    """
    if not children:
        return None
    parent_visits = children[0].parent.visits if children[0].parent else 1
    log_parent = math.log(parent_visits)
    C = 1.4
    def uct_value(child: Node) -> float:
        if child.visits == 0:
            return float('inf')
        return (child.total_reward / child.visits) + \
               C * math.sqrt(log_parent / child.visits)
    return max(children, key=uct_value)


def expand(parent_node: Node, dims: List[Tuple[int, int, int]], 
          space_dims: Tuple[int, int, int]) -> Node:
    """Take one untried action and add a new child node."""
    w, d, h = dims[parent_node.box_index]
    action_list = parent_node.untried_actions(dims, space_dims)
    if not action_list:
        return parent_node
    chosen_action = action_list.pop(random.randrange(len(action_list)))
    sim_space = Space(space_dims[0], space_dims[1], space_dims[2])
    sim_space.boxes = list(parent_node.placed_boxes)
    placed_box = sim_space.place_at(*chosen_action, w, d, h)
    child_node = Node(parent_node,
                      chosen_action,
                      parent_node.box_index + 1,
                      parent_node.placed_boxes + [placed_box])
    parent_node.children.append(child_node)
    return child_node


def rollout(box_state: Tuple[int, List[Box]], dims: List[Tuple[int, int, int]], 
           space_dims: Tuple[int, int, int]) -> float:
    """
    Simulate random play from given state until no move possible.
    Returns reward = (boxes placed) / TOTAL_BOXES.
    """
    next_index, placed = box_state
    sim_space = Space(space_dims[0], space_dims[1], space_dims[2])
    sim_space.boxes = list(placed)
    placed_count = len(placed)
    for idx in range(next_index, len(dims)):
        w, d, h = dims[idx]
        legal = sim_space.legal_xy_positions(w, d, h)
        if not legal:
            break
        sim_space.place_at(*random.choice(legal), w, d, h)
        placed_count += 1
    return placed_count / len(dims)


def backpropagate(node: Optional[Node], reward: float) -> None:
    """Propagate reward up to root."""
    current = node
    while current is not None:
        current.visits += 1
        current.total_reward += reward
        current = current.parent


def run_mcts_optimization(box_items: List[Dict], space_dims: Tuple[int, int, int], 
                         max_time_per_step: float = 1.0, 
                         progress_callback=None) -> Tuple[List[Box], Dict[str, Any]]:
    """
    Run MCTS optimization on a list of box items.
    
    Args:
        box_items: List of box item dictionaries with 'dimensions' and 'quantity' keys
        space_dims: Tuple of (width, depth, height) for the warehouse space
        max_time_per_step: Maximum time to spend on each box placement step
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Tuple of (placed_boxes, statistics_dict)
    """
    # Convert box items to dimensions for MCTS
    dims = []
    for item in box_items:
        for _ in range(item.get('quantity', 1)):
            # Convert to integers for MCTS (assuming dimensions are in inches)
            w, d, h = [int(round(x)) for x in item['dimensions']]
            # Check if box fits in space
            if w > space_dims[0] or d > space_dims[1] or h > space_dims[2]:
                print(f"Warning: Box {w}x{d}x{h} is too large for space {space_dims}, skipping")
                continue
            dims.append((w, d, h))
    
    if not dims:
        print("Error: No valid boxes that fit in the given space")
        return [], {'total_boxes': 0, 'placed_boxes': 0, 'efficiency': 0, 'space_utilization': 0}
        
    # Sort by descending volume
    dims.sort(key=lambda t: t[0]*t[1]*t[2], reverse=True)
    total_boxes = len(dims)
    
    # Initialize MCTS
    root = Node(None, None, 0, [])
    final_placements: List[Box] = []
    
    step = 0
    while not root.is_terminal(total_boxes):
        start_time = time.time()
        # Run MCTS for specified time per step
        while time.time() - start_time < max_time_per_step:
            # 1) Selection
            node = root
            while not node.is_terminal(total_boxes) and node.is_fully_expanded(dims, space_dims):
                selected = uct_select(node.children)
                if selected is None:
                    break
                node = selected

            # 2) Expansion
            if not node.is_terminal(total_boxes) and not node.is_fully_expanded(dims, space_dims):
                node = expand(node, dims, space_dims)

            # 3) Simulation
            reward = rollout((node.box_index, node.placed_boxes), dims, space_dims)

            # 4) Backpropagation
            backpropagate(node, reward)

        # Choose the most‐visited child of root
        if not root.children:
            print(f"No legal positions found for box {dims[root.box_index]} at step {step}")
            break
            
        best_child = max(root.children, key=lambda c: c.visits)

        # Commit that action
        next_idx = root.box_index
        w, d, h = dims[next_idx]
        action_x, action_y = best_child.action  # guaranteed not None
        sim_space = Space(space_dims[0], space_dims[1], space_dims[2])
        sim_space.boxes = list(root.placed_boxes)
        placed_box = sim_space.place_at(action_x, action_y, w, d, h)
        final_placements = root.placed_boxes + [placed_box]

        step += 1
        if progress_callback:
            progress_callback(step, total_boxes)

        # Advance root
        best_child.parent = None
        root = best_child

    # Calculate statistics
    stats = {
        'total_boxes': total_boxes,
        'placed_boxes': len(final_placements),
        'efficiency': len(final_placements) / total_boxes if total_boxes > 0 else 0,
        'space_utilization': sum(b.volume for b in final_placements) / (space_dims[0] * space_dims[1] * space_dims[2])
    }
    
    return final_placements, stats
