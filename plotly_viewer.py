#!/usr/bin/env python3
"""
Standalone Plotly interactive 3D pallet viewer
Usage: python plotly_viewer.py [input_file] [output_dir]
"""

import sys
from pathlib import Path
from pallet_packer.models import InputSpec
from pallet_packer.heuristics import pack
from pallet_packer.plotly_viz import render_plotly_interactive

def main():
    if len(sys.argv) < 2:
        print("Usage: python plotly_viewer.py <input_file> [output_dir]")
        print("Example: python plotly_viewer.py sample_input.json out")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("out")
    
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    # Load and process the input
    spec = InputSpec.model_validate_json(input_file.read_text())
    result = pack(spec)
    
    print(f"Loaded {len(spec.boxes)} box types with {sum(b.quantity for b in spec.boxes)} total boxes")
    print(f"Placed {len(result.placements)} boxes ({len(result.unplaced)} unplaced)")
    print(f"Utilization: {result.utilization*100:.1f}%")
    print("\nOpening Plotly interactive 3D visualization...")
    print("Features:")
    print("  • Web-based interactive 3D plot")
    print("  • Smooth mouse controls (rotate, zoom, pan)")
    print("  • Realistic lighting and materials")
    print("  • Click legend to toggle box types")
    print("  • Hover for box details")
    print("  • Export options (PNG, HTML, etc.)")
    
    # Show interactive visualization
    render_plotly_interactive(spec.pallet, result.placements)

if __name__ == "__main__":
    main()
