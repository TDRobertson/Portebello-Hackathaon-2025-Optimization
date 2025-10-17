## Pallet Packing Optimizer

This tool generates an efficient 3D pallet layout for a set of boxes with quantities, orientations, and label-side constraints, given pallet dimensions and a height limit.

### Features
- Places boxes within pallet boundaries without overlap
- Maximizes space utilization using a layered shelf (guillotine-like) heuristic
- Stability-aware stacking (larger/heavier boxes on lower layers)
- Label-side constraint: ensures specified label face is exposed on the outer perimeter where possible
- Outputs placements (JSON/CSV), pick and stack sequences, and a 3D visualization PNG

### Requirements
- Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

**Basic usage (static output):**
```bash
python -m pallet_packer.cli sample_input.json out
```

**Interactive 3D visualization (matplotlib):**
```bash
python -m pallet_packer.cli sample_input.json out --interactive
```

**High-quality Plotly visualization:**
```bash
python -m pallet_packer.cli sample_input.json out --plotly
```

**Interactive Plotly visualization:**
```bash
python -m pallet_packer.cli sample_input.json out --plotly --interactive
```

**Standalone viewers:**
```bash
# Matplotlib viewer
python interactive_viewer.py sample_input.json out

# Plotly viewer (recommended)
python plotly_viewer.py sample_input.json out
```

Outputs will be written to the `out` directory:
- `placements.json` and `placements.csv`
- `pick_sequence.csv` (picking order)
- `stack_sequence.csv` (stacking order)
- `visualization.png` (static 3D rendering)
- `summary.txt` (utilization and stats)

### Visualization Options

#### Matplotlib (Default)
- **Interactive 3D viewer**: Rotate, zoom, pan with mouse
- **Static images**: High-quality PNG output
- **Features**: Color-coded boxes, legend, statistics overlay

#### Plotly (Recommended)
- **Web-based interactive plots**: Opens in browser
- **Superior 3D rendering**: Realistic lighting and materials
- **Advanced features**: 
  - Smooth animations and transitions
  - Click legend to toggle box types
  - Hover for detailed box information
  - Export options (PNG, HTML, SVG, PDF)
  - Better performance with large datasets
  - Professional appearance

#### Comparison
| Feature | Matplotlib | Plotly |
|---------|------------|--------|
| 3D Quality | Good | Excellent |
| Interactivity | Basic | Advanced |
| Performance | Good | Better |
| Export Options | PNG only | Multiple formats |
| Browser Required | No | Yes (for interactive) |
| File Size | Small | Larger |

### Input Format

See `sample_input.json` for an example. Schema (units are arbitrary but consistent):

```json
{
  "pallet": { "length": 1200, "width": 1000, "height_limit": 1500 },
  "boxes": [
    {
      "id": "SKU-A",
      "length": 400,
      "width": 300,
      "height": 200,
      "quantity": 10,
      "label_side": "+y"  
    }
  ]
}
```

Label side options: "+x", "-x", "+y", "-y", "+z", "-z". If omitted, any orientation is allowed; the algorithm will still try to place label sides on an exterior face when specified.

### Notes
- This is a heuristic (not guaranteed optimal) but designed for good utilization with practical constraints.
- The visualization uses approximate boxes and does not depict fine tolerances.


