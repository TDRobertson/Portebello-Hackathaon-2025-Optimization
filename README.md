# Portobello America Hackathon 2025 — Pallet Optimization System

**Best Innovation Award — Portobello America Hackathon 2025, Baxter, TN**

A multi-algorithm pallet optimization and warehouse logistics system built in 30 hours for Portobello America's "Perfect Pallet, Perfect Delivery" challenge. The system automates pallet assembly decisions using AI-driven 3D bin packing algorithms, enforces real-world physics and label compliance constraints, and provides warehouse workers with optimized traversal routes — replacing operator-dependent institutional knowledge with a replicable, data-driven solution.

---

## About The Project

Portobello America is a Brazilian specialty tile manufacturer operating a production facility in Baxter, TN. Their hackathon challenge posed a real operations problem: pallet assembly strategy — the sequencing, orientation, and placement of boxes on a pallet — currently lives in the heads of a small number of experienced operators. When those operators leave, the knowledge walks out with them.

The challenge asked teams to build AI-driven tools that could automate or suggest the best pallet assembly strategy, optimizing for space utilization, structural stability, label visibility for logistics scanning, and transportation cost. Teams were encouraged to explore computer vision, 3D packing algorithms, and digital training tooling.

### The Problem We Solve

- **Knowledge retention risk:** Optimal pallet configuration relies on tacit operator expertise that is difficult to document or transfer in high-turnover manufacturing environments
- **Spatial inefficiency:** Without algorithmic guidance, pallets are commonly under-packed, increasing per-shipment transportation cost
- **Stability failures:** Improperly stacked pallets — larger boxes on smaller, unsupported overhangs — result in damaged goods in transit
- **Label compliance:** Shipping labels and scan-facing sides must be visible on the pallet perimeter; manual placement frequently misses this requirement
- **Worker routing inefficiency:** Once a pallet spec is determined, warehouse workers need optimal traversal routes to pick and stage materials — an unsolved secondary problem

### Award

This system was awarded **Best Innovation** at the Portobello America Hackathon 2025.

---

## Team

| Name | Role | GitHub |
|---|---|---|
| Thomas Robertson | Team Lead, Architecture, MCTS v1/v3, GUI Integration | [@TDRobertson](https://github.com/TDRobertson) |
| Maximus Jessey | MCTS v2 (CNN-based heuristic) | [@mjessey](https://github.com/mjessey) |
| Nol Patterson | Mathematical algorithm, CSV pipeline, packaging | [@NullPatterson](https://github.com/NullPatterson) |
| Jun Han | Warehouse pathfinding v2, TSP optimization, animation | [@JunHan10](https://github.com/JunHan10) |

---

## Core Features

### 1. Monte Carlo Tree Search Pallet Optimizer

Three MCTS variants implement progressively improved search strategies for box placement:

- **v1** (`mcts.py`) — Foundation MCTS with gravity simulation. UCT (Upper Confidence Bound applied to Trees) selection with exploration constant `C = 1.0`, random rollouts to estimate remaining placement count, time-bounded search at 1.0 second per placement decision. Boxes sorted by volume descending before search begins.

- **v2** (`mcts_v2.py`) — CNN-enhanced MCTS. Replaces random rollouts with a pretrained convolutional neural network that predicts remaining placement capacity from the current 3D grid state, improving value estimation accuracy and reducing search time. Outperforms v1 on standard test sets.

- **v3** (`mcts_v3.py`) — Further refinements to the search strategy and placement heuristics building on v2.

All MCTS variants operate on a voxel grid representation of the pallet space and test all 6 box orientations at each node.

### 2. Heuristic Pallet Packer

A deterministic, physics-aware packing algorithm optimized for real-world constraint satisfaction (`pallet_packer/`):

**Algorithm overview:**
1. Expand box types into individual instances by quantity
2. Sort by stability priority: larger footprint area + greater height first
3. For each box, evaluate all candidate placements across:
   - All current z-levels (ground plane + tops of placed boxes)
   - All 6 possible orientations (LWH, WLH, LHW, WHL, HLW, HWL)
   - All x,y coordinates aligned to existing box corners and edges
4. Score each candidate placement: z-level is the dominant factor (lower is better), then x,y position, with a bonus for label-side perimeter exposure
5. Accept placement only if full footprint support is confirmed (≥99.9% coverage by underlying boxes), no overlaps exist, and the box fits within pallet dimensions

**Key constraints enforced:**
- Full footprint support — no unsupported overhangs at any box corner
- Stability sort — larger, heavier base boxes placed before smaller ones
- Label-side alignment — `+x`, `-x`, `+y`, `-y`, `+z`, `-z` constraints route the specified face to the nearest pallet perimeter when possible

**Outputs:** Placement coordinates with orientations, pick sequence (largest-first retrieval order), stack sequence (z→y→x assembly order), 3D visualization, utilization summary.

### 3. Warehouse Pathfinding Simulator

An interactive simulation for optimizing worker traversal routes across a warehouse floor (`warehouse/warehouse_sim_v2.py`):

- **Grid layout:** 20×25 cell warehouse with realistic shelf placement, aisle spacing, and walkable path generation
- **A* pathfinding:** Computes the optimal path from worker position to any single target location, avoiding shelf cells
- **Greedy TSP:** For multi-target pick lists, locations are sorted by weight (priority) in descending order and visited in that sequence — a practical approximation of the Traveling Salesman Problem for warehouse routing
- **Visualization:** Pygame-based interactive display with color-coded path rendering, frontier/closed-set visualization, and HUD showing weighted item list and route metrics
- **Animation:** Smooth agent movement with easing functions; configurable speed via `per_cell_frames`

### 4. Integrated PyQt5 GUI

A unified desktop application combining all three system components (`pyqt_integration.py`, `run_gui.py`):

- Box selection and picklist management loaded from `db.json`
- Configurable pallet dimensions
- MCTS optimization running in a background `QThread` with real-time progress reporting
- Interactive 3D pallet visualization (Matplotlib) and 2D warehouse map view side-by-side
- Load/save picklist configurations

### 5. Data Pipeline

- **Input:** JSON configuration (`sample_input.json`) or CSV box data (`boxes.csv`)
- **Output:** Six files in the `out/` directory (see Output Specification below)
- **Oracle compatibility:** Pydantic models and structured CSV output designed for Oracle Database integration
- **Test data generation:** `gen_csv.py` generates synthetic box datasets with configurable count and dimension ranges

---

## Algorithm Details

### MCTS with UCT Selection

The MCTS implementation uses the UCT formula to balance exploration and exploitation when selecting which tree node to expand:

```
UCT(node) = wins/visits + C * sqrt(ln(parent.visits) / visits)
```

Where `C = 1.0` is the exploration constant. Each node represents a partial packing state. The CNN in v2 replaces the rollout phase — instead of randomly placing remaining boxes to estimate a terminal value, the network predicts placement count from the current 3D voxel grid, providing a faster and more accurate value estimate.

### Heuristic Layered Shelf Packer

The heuristic uses a guillotine-like shelf model where boxes are placed in layers. The scoring function prioritizes:

1. **z-level** (dominant) — lower placements are always preferred to minimize height waste
2. **x, y position** — secondary tie-breaking
3. **Label side perimeter bonus** — added when the specified face aligns with a pallet edge

Support coverage is computed by projecting the box footprint onto the layer below and measuring intersection area. Placements below 99.9% coverage are rejected.

### A\* Pathfinding + Greedy TSP

A* computes shortest paths using Manhattan distance as the admissible heuristic. Shelf cells are obstacles; aisle cells are walkable. For multi-target routing, `find_nearest_free()` maps location codes to walkable grid cells using BFS when the direct cell is occupied. The greedy TSP orders targets by weight descending — higher-priority items are retrieved first, reducing backtracking for typical pick list distributions.

---

## Technical Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.10+ |
| Visualization | Matplotlib | 3.9.2 |
| Numerical computing | NumPy | 2.1.2 |
| Data validation | Pydantic | 2.9.2 |
| CLI framework | Typer | 0.12.5 |
| Terminal output | Rich | 13.9.2 |
| Desktop GUI | PyQt5 | 5.15.10 |
| Warehouse simulation | Pygame | 2.6.1 |
| Neural network (MCTS v2) | PyTorch | latest |

---

## Getting Started

### Requirements

- Python 3.10 or higher
- Dependencies listed in `requirements.txt`

### Installation

```bash
pip install -r requirements.txt
```

### Usage

**Heuristic pallet packer (recommended for production-like runs):**
```bash
python -m pallet_packer.cli --input sample_input.json --out out
```

**MCTS optimization (v1 — no neural network required):**
```bash
python mcts.py boxes.csv
```

**MCTS with CNN heuristic (v2 — requires trained model):**
```bash
python mcts_v2.py boxes.csv nn.pth
```

**Gravity-based placement:**
```bash
python main.py boxes.csv
```

**Interactive warehouse pathfinding simulation:**
```bash
python warehouse/warehouse_sim_v2.py
```

**Integrated PyQt5 GUI:**
```bash
python run_gui.py
```

**Generate synthetic test data:**
```bash
# boxes.csv = output file, 50 = box count, 1/5 = min/max dimension range multiplier
python gen_csv.py boxes.csv 50 1 5
```

---

## Input Specification

### JSON Configuration

```json
{
  "pallet": {
    "length": 1000,
    "width": 1000,
    "height_limit": 1000
  },
  "boxes": [
    {
      "id": "SKU-A",
      "length": 400,
      "width": 300,
      "height": 200,
      "quantity": 8,
      "label_side": "+y"
    }
  ]
}
```

### Label Side Options

| Value | Meaning |
|---|---|
| `+x` | Label faces the positive X direction (right wall of pallet) |
| `-x` | Label faces the negative X direction (left wall of pallet) |
| `+y` | Label faces the positive Y direction (front of pallet) |
| `-y` | Label faces the negative Y direction (back of pallet) |
| `+z` | Label faces upward |
| `-z` | Label faces downward (floor contact) |

If `label_side` is omitted, the algorithm will place the box in whatever orientation maximizes packing quality. When specified, the heuristic will prefer placements that expose that face on the pallet perimeter.

---

## Output Specification

All outputs are written to the `out/` directory:

| File | Format | Contents |
|---|---|---|
| `placements.json` | JSON | Full placement data: box ID, coordinates (x, y, z), dimensions (length, width, height), label side |
| `placements.csv` | CSV | Tabular placement data for database import |
| `pick_sequence.csv` | CSV | Optimized box retrieval order (largest-first) for warehouse workers |
| `stack_sequence.csv` | CSV | Pallet assembly order (z→y→x) for pallet builders |
| `visualization.png` | PNG | 3D rendered view of the completed pallet layout |
| `summary.txt` | Text | Total boxes, placed count, unplaced count, used height, volume utilization % |

### Sample Summary Output

```
Total boxes: 90
Placed boxes: 23
Unplaced boxes: 67
Used height: 980.00
Utilization (by volume): 71.29%
```

---

## Project Structure

```
pallet_packer/
├── __init__.py
├── models.py          # Pydantic data models: InputSpec, Pallet, BoxType, Placement, Result
├── heuristics.py      # Heuristic packing algorithm
├── visualize.py       # 3D Matplotlib rendering
└── cli.py             # Typer CLI entry point

warehouse/
├── warehouse_sim_v2.py    # A* pathfinding + greedy TSP Pygame simulation
└── food_for_thought.md    # Design notes and extension guidance

mcts.py                # MCTS v1 — random rollouts
mcts_v2.py             # MCTS v2 — CNN-based value estimation
mcts_v3.py             # MCTS v3 — further refinements
mcts_integration.py    # MCTS integration layer for PyQt GUI
main.py                # Gravity-based placement (standalone)
math_based.py          # Mathematical positioning algorithm
train.py               # CNN training script for MCTS v2
pyqt_integration.py    # Unified PyQt5 desktop application
run_gui.py             # GUI launcher
warehouse_visualization.py  # 2D/3D visualization components
gen_csv.py             # Synthetic test data generator
sample_input.json      # Example input configuration
db.json                # Box type definitions for GUI
out/                   # Generated output files
screenshots/           # Visualization output images
nn.pth                 # Trained CNN weights for MCTS v2
```

---

## Notes

- The heuristic packer is deterministic and fast but not guaranteed optimal — it is designed for good utilization under real-world stability and label constraints, not pure theoretical bin-packing optimality.
- MCTS v2 requires the trained `nn.pth` model file. To retrain from scratch, use `train.py`.
- The warehouse simulation uses a heuristic grid mapping. For production use with a real warehouse layout, replace the `location_to_grid()` mapping with a deterministic lookup against actual aisle geometry (see `warehouse/food_for_thought.md`).
- All visualizations use approximate box dimensions and do not model fine manufacturing tolerances.
