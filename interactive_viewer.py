#!/usr/bin/env python3
"""
Standalone interactive 3D pallet viewer
Usage: python interactive_viewer.py [input_file] [output_dir]
"""

import sys
from pathlib import Path
from pallet_packer.models import InputSpec
from pallet_packer.heuristics import pack
from pallet_packer.visualize import render_interactive

def main():
    if len(sys.argv) < 2:
        print("Usage: python interactive_viewer.py <input_file> [output_dir]")
        print("Example: python interactive_viewer.py sample_input.json out")
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
    print("\nOpening interactive 3D visualization...")
    print("Controls:")
    print("  • Mouse drag: Rotate view")
    print("  • Mouse scroll: Zoom in/out") 
    print("  • Right-click drag: Pan view")
    print("  • Close window to exit")
    
    # Show interactive visualization
    render_interactive(spec.pallet, result.placements)

if __name__ == "__main__":
    main()
