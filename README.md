# Portebello Hackathon 2025 - Pallet Optimization System

## Team: Pallet Packers
### Members:
- [@Thomas Robertson](https://github.com/TDRobertson)
- [@Maximum Jessey](https://github.com/mjessey)
- [@Nol Patterson](https://github.com/NullPatterson)
- [@Jun Han](https://github.com/JunHan10)

A comprehensive pallet optimization and warehouse management system designed for the Portebello Hackathon 2025. This system combines advanced 3D pallet packing algorithms with Monte Carlo Tree Search (MCTS) optimization, warehouse pathfinding simulation, and Oracle Database integration capabilities to solve complex logistics optimization problems.

## 🚀 System Overview

This project addresses the **Traveling Salesman Problem (TSP)** in warehouse logistics by optimizing both pallet space utilization and worker traversal paths. The system balances weight/size distribution while providing efficient traversal routes for warehouse workers to access materials in different zones.

## 🏗️ Architecture & Components

### Core Optimization Engines

1. **Monte Carlo Tree Search (MCTS) Algorithm** (`mcts.py`)
   - Advanced tree search optimization for complex pallet packing scenarios
   - UCT (Upper Confidence Bound applied to Trees) selection strategy
   - Handles up to 50+ boxes with intelligent placement decisions
   - Real-time visualization of optimization process

2. **Heuristic Pallet Packer** (`pallet_packer/`)
   - Layered shelf (guillotine-like) packing algorithm
   - Stability-aware stacking with gravity-based placement
   - Label-side constraint optimization
   - Support coverage analysis to prevent overhangs

3. **Warehouse Pathfinding Simulator** (`warehouse/warehouse_sim_v2.py`)
   - Interactive A* pathfinding algorithm
   - Real-time warehouse grid simulation (20x25 cells)
   - Greedy TSP optimization for multi-target traversal
   - Smooth agent animation with easing functions

### Data Management & Integration

4. **Oracle Database Integration Ready**
   - Structured data models using Pydantic
   - CSV/JSON input/output compatibility
   - Ready for Oracle Python Application integration
   - Scalable data pipeline architecture

5. **Advanced Visualization System**
   - Interactive 3D pallet visualization with Matplotlib
   - Layer-by-layer packing animation
   - Real-time warehouse pathfinding visualization
   - Color-coded box identification and volume mapping

## 🎯 Key Features

### Pallet Optimization
- **Multi-Algorithm Approach**: Combines MCTS and heuristic algorithms
- **3D Space Utilization**: Maximizes pallet volume usage with stability constraints
- **Label-Side Constraints**: Ensures specified label faces are exposed on perimeter
- **Support Analysis**: Prevents overhangs and ensures structural integrity
- **Orientation Optimization**: Tests all 6 possible box orientations

### Warehouse Management
- **Pathfinding Optimization**: A* algorithm for optimal worker routes
- **TSP Integration**: Greedy heuristic for multi-target traversal
- **Interactive Simulation**: Real-time warehouse grid visualization
- **Zone-Based Organization**: Efficient material location management

### Data Processing
- **CSV Generation**: Automated test data creation (`gen_csv.py`)
- **Multiple Input Formats**: JSON configuration and CSV data files
- **Comprehensive Output**: Placements, sequences, visualizations, and summaries
- **Oracle Compatibility**: Structured for database integration

## 📋 Requirements

- Python 3.10+
- Oracle Database (for production deployment)
- Required packages: `matplotlib`, `numpy`, `pydantic`, `typer`, `rich`, `pygame`

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Basic Pallet Packing
```bash
# Using the heuristic packer
python -m pallet_packer.cli --input sample_input.json --out out

# Using MCTS optimization
python mcts.py boxes.csv

# Using gravity-based placement
python main.py boxes.csv
```

### Warehouse Simulation
```bash
# Interactive warehouse pathfinding
python warehouse/warehouse_sim_v2.py
```

### Data Generation
```bash
# Generate test data
python gen_csv.py boxes.csv 50 1 5
```

## 📊 Output Files

The system generates comprehensive outputs in the `out` directory:

- **`placements.json`** - Detailed placement data with coordinates and orientations
- **`placements.csv`** - Tabular format for database import
- **`pick_sequence.csv`** - Optimized picking order for workers
- **`stack_sequence.csv`** - Stacking sequence for pallet assembly
- **`visualization.png`** - 3D rendering of the final pallet layout
- **`summary.txt`** - Utilization statistics and performance metrics

## 🔧 Input Format

### JSON Configuration (`sample_input.json`)
```json
{
  "pallet": { "length": 1000, "width": 1000, "height_limit": 1000 },
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

Label side options: "+x", "-x", "+y", "-y", "+z", "-z". If omitted, any orientation is allowed; the algorithm will still try to place label sides on an exterior face when specified.

### Notes
- This is a heuristic (not guaranteed optimal) but designed for good utilization with practical constraints.
- The visualization uses approximate boxes and does not depict fine tolerances.
- The system is designed for warehouse logistics and may not be suitable for other applications.
This project was developed for the Portebello Hackathon 2025. For questions or collaboration opportunities, please contact the development team.


