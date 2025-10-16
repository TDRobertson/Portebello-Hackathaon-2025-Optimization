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

```bash
python -m pallet_packer.cli --input sample_input.json --out out
```

Outputs will be written to the `out` directory:
- `placements.json` and `placements.csv`
- `pick_sequence.csv` (picking order)
- `stack_sequence.csv` (stacking order)
- `visualization.png` (3D rendering of the layout)
- `summary.txt` (utilization and stats)

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


