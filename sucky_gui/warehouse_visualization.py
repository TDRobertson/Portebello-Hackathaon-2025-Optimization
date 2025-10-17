"""
Warehouse Visualization Module for PyQt Integration
Provides 2D and 3D visualization capabilities for warehouse layouts
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional, Dict, Any
import heapq
import math


class WarehouseVisualization2D(FigureCanvas):
    """2D top-down view of warehouse with path visualization"""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111)
        self.warehouse_map = None
        self.rows, self.cols = 20, 25
        self.cell_size = 1
        
        # Pathfinding state
        self.start_pos = None
        self.targets = []
        self.current_path = []
        self.pathfinder = None
        
        # Colors
        self.colors = {
            'aisle': '#E1E8F0',
            'shelf': '#464B5A',
            'shelf_shadow': '#282A32',
            'grid_line': '#F5F8FC',
            'explored': '#FFF7D2',
            'frontier': '#C8AAFF',
            'path': '#FFEBA0',
            'agent': '#EB5A5A',
            'start': '#78E696',
            'target': '#6EA0FF'
        }
        
    def generate_warehouse_map(self, rows: int = 20, cols: int = 25):
        """Generate a warehouse map with shelves and aisles"""
        self.rows, self.cols = rows, cols
        self.warehouse_map = [[0 for _ in range(cols)] for _ in range(rows)]
        
        # Create shelves (1) and aisles (0) with spacing pattern
        for r in range(2, rows - 2):
            if r % 4 == 0:
                continue
            for c in range(2, cols - 2):
                if c % 3 == 0:
                    self.warehouse_map[r][c] = 1
    
    def draw_warehouse(self, agent_pos: Optional[Tuple[int, int]] = None, 
                      start: Optional[Tuple[int, int]] = None,
                      targets: Optional[List[Tuple[int, int]]] = None,
                      path: Optional[List[Tuple[int, int]]] = None,
                      closed: Optional[set] = None,
                      openset: Optional[set] = None,
                      title: str = "Warehouse Layout"):
        """Draw the warehouse with optional path visualization"""
        self.ax.clear()
        
        if self.warehouse_map is None:
            self.generate_warehouse_map()
        
        # Draw warehouse cells
        for r in range(self.rows):
            for c in range(self.cols):
                x, y = c, r
                rect = plt.Rectangle((x, y), 1, 1, 
                                   facecolor=self.colors['aisle'] if self.warehouse_map[r][c] == 0 else self.colors['shelf'],
                                   edgecolor=self.colors['grid_line'],
                                   linewidth=0.5)
                self.ax.add_patch(rect)
                
                # Add shelf shadow effect
                if self.warehouse_map[r][c] == 1:
                    shadow_rect = plt.Rectangle((x + 0.1, y + 0.1), 1, 1,
                                              facecolor=self.colors['shelf_shadow'],
                                              alpha=0.3)
                    self.ax.add_patch(shadow_rect)
        
        # Draw explored cells
        if closed:
            for (r, c) in closed:
                rect = plt.Rectangle((c, r), 1, 1, 
                                   facecolor=self.colors['explored'],
                                   alpha=0.7)
                self.ax.add_patch(rect)
        
        # Draw frontier cells
        if openset:
            for (r, c) in openset:
                rect = plt.Rectangle((c, r), 1, 1, 
                                   facecolor=self.colors['frontier'],
                                   alpha=0.7)
                self.ax.add_patch(rect)
        
        # Draw path
        if path:
            for (r, c) in path:
                rect = plt.Rectangle((c, r), 1, 1, 
                                   facecolor=self.colors['path'],
                                   alpha=0.8)
                self.ax.add_patch(rect)
        
        # Draw targets
        if targets:
            for (r, c) in targets:
                circle = plt.Circle((c + 0.5, r + 0.5), 0.3, 
                                  color=self.colors['target'], alpha=0.8)
                self.ax.add_patch(circle)
        
        # Draw start position
        if start:
            circle = plt.Circle((start[1] + 0.5, start[0] + 0.5), 0.3, 
                              color=self.colors['start'], alpha=0.8)
            self.ax.add_patch(circle)
        
        # Draw agent
        if agent_pos:
            circle = plt.Circle((agent_pos[1] + 0.5, agent_pos[0] + 0.5), 0.2, 
                              color=self.colors['agent'], alpha=0.9)
            self.ax.add_patch(circle)
        
        # Set axis properties
        self.ax.set_xlim(0, self.cols)
        self.ax.set_ylim(0, self.rows)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (Width)')
        self.ax.set_ylabel('Y (Depth)')
        self.ax.set_title(title)
        self.ax.invert_yaxis()  # Invert Y axis so (0,0) is top-left
        
        # Connect mouse click events for interactive target selection
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        self.draw()
    
    def on_click(self, event):
        """Handle mouse clicks for setting start and adding targets"""
        if event.inaxes != self.ax:
            return
        if event.button == 1:  # Left click
            col = int(event.xdata)
            row = int(event.ydata)
            if 0 <= row < self.rows and 0 <= col < self.cols:
                pos = (row, col)
                if self.start_pos is None:
                    self.set_start_position(pos)
                    self.draw_warehouse(start=self.start_pos, targets=self.targets, path=self.current_path)
                else:
                    self.add_target_interactive(pos)
    
    def set_start_position(self, start: Tuple[int, int]):
        """Set the starting position for pathfinding"""
        self.start_pos = start
        self.pathfinder = PathfindingVisualization(self.warehouse_map) if self.warehouse_map else None
    
    def set_targets(self, targets: List[Tuple[int, int]]):
        """Set target positions for pathfinding"""
        self.targets = targets
        if self.pathfinder and self.start_pos:
            self.calculate_optimal_route()
    
    def calculate_optimal_route(self):
        """Calculate the optimal route visiting all targets"""
        if not self.pathfinder or not self.start_pos or not self.targets:
            return
        
        # Find optimal order to visit all targets
        optimal_order = self.pathfinder.find_optimal_order(self.start_pos, self.targets)
        
        # Calculate full path
        self.current_path = self.pathfinder.get_full_path(self.start_pos, optimal_order)
        
        # Update visualization
        self.draw_warehouse(
            start=self.start_pos,
            targets=self.targets,
            path=self.current_path,
            title=f"Optimal Route - {len(self.current_path)} steps"
        )
    
    def add_target_interactive(self, pos: Tuple[int, int]):
        """Add a target position interactively"""
        if self.warehouse_map and self.warehouse_map[pos[0]][pos[1]] == 0:  # Only on aisles
            if pos not in self.targets and pos != self.start_pos:
                self.targets.append(pos)
                if self.start_pos:
                    self.calculate_optimal_route()
                else:
                    self.draw_warehouse(
                        start=self.start_pos,
                        targets=self.targets,
                        path=self.current_path,
                        title="Click to set start position, then add targets"
                    )
    
    def clear_targets(self):
        """Clear all targets and reset path"""
        self.targets = []
        self.current_path = []
        self.draw_warehouse(
            start=self.start_pos,
            targets=self.targets,
            path=self.current_path,
            title="Warehouse Layout - Click to add targets"
        )
    
    def get_route_info(self) -> Dict[str, Any]:
        """Get information about the current route"""
        if not self.current_path:
            return {"steps": 0, "targets_visited": 0, "efficiency": 0}
        
        return {
            "steps": len(self.current_path),
            "targets_visited": len(self.targets),
            "efficiency": len(self.targets) / len(self.current_path) if self.current_path else 0,
            "path": self.current_path
        }


class WarehouseVisualization3D(FigureCanvas):
    """3D visualization of warehouse with placed boxes"""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.space_dims = (15, 15, 10)  # Default space dimensions
        self.placements = []
        
    def update_visualization(self, placements: List[Any], space_dims: Tuple[int, int, int]):
        """Update the 3D visualization with new placements"""
        self.placements = placements
        self.space_dims = space_dims
        
        self.ax.clear()
        
        if not placements:
            self.ax.text(0.5, 0.5, 0.5, 'No boxes placed', 
                        transform=self.ax.transAxes, ha='center', va='center')
            self.draw()
            return
        
        # Draw placed boxes
        volumes = [b.volume for b in placements]
        if volumes:
            cmap = plt.cm.viridis
            norm = plt.Normalize(min(volumes), max(volumes))
            
            for b in placements:
                self.ax.bar3d(b.x, b.y, b.z, b.w, b.d, b.h,
                            color=cmap(norm(b.volume)), 
                            edgecolor='k', alpha=0.7)
        
        # Set axis limits and labels
        self.ax.set_xlim(0, space_dims[0])
        self.ax.set_ylim(0, space_dims[1])
        self.ax.set_zlim(0, space_dims[2])
        self.ax.set_xlabel('X (Width)')
        self.ax.set_ylabel('Y (Depth)')
        self.ax.set_zlabel('Z (Height)')
        self.ax.set_title(f'Warehouse Layout - {len(placements)} boxes placed')
        
        self.draw()


class PathfindingVisualization:
    """Pathfinding visualization using A* algorithm"""
    
    def __init__(self, warehouse_map: List[List[int]]):
        self.warehouse_map = warehouse_map
        self.rows = len(warehouse_map)
        self.cols = len(warehouse_map[0]) if warehouse_map else 0
    
    def neighbors(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighbors of a cell"""
        r, c = cell
        neighbors = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors.append((nr, nc))
        return neighbors
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* pathfinding algorithm"""
        open_heap = []
        heapq.heappush(open_heap, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        closed_set = set()
        open_set = {start}
        
        while open_heap:
            _, current = heapq.heappop(open_heap)
            open_set.discard(current)
            
            if current in closed_set:
                continue
            closed_set.add(current)
            
            if current == goal:
                # Reconstruct path
                path = [goal]
                while path[-1] in came_from:
                    path.append(came_from[path[-1]])
                path.reverse()
                return path
            
            for nb in self.neighbors(current):
                nr, nc = nb
                if self.warehouse_map[nr][nc] == 1:  # Skip shelves
                    continue
                
                tentative = g_score[current] + 1
                if tentative < g_score.get(nb, float('inf')):
                    came_from[nb] = current
                    g_score[nb] = tentative
                    f = tentative + self.heuristic(nb, goal)
                    f_score[nb] = f
                    if nb not in open_set:
                        heapq.heappush(open_heap, (f, nb))
                        open_set.add(nb)
        
        return []  # No path found
    
    def find_optimal_order(self, start: Tuple[int, int], targets: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Find optimal order to visit all targets using greedy approach"""
        remaining = targets[:]
        order = []
        current = start
        
        while remaining:
            best_target = None
            best_dist = float('inf')
            
            for t in list(remaining):
                path = self.a_star(current, t)
                if path and len(path) < best_dist:
                    best_dist = len(path)
                    best_target = t
            
            if best_target is None:
                break
                
            order.append(best_target)
            remaining.remove(best_target)
            current = best_target
        
        return order
    
    def get_full_path(self, start: Tuple[int, int], targets: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Get full path visiting all targets in optimal order"""
        order = self.find_optimal_order(start, targets)
        full_path = []
        current = start
        
        for target in order:
            segment = self.a_star(current, target)
            if not segment:
                continue
            
            # Avoid duplicate cells
            if full_path and segment and segment[0] == full_path[-1]:
                full_path.extend(segment[1:])
            else:
                full_path.extend(segment)
            current = target
        
        return full_path