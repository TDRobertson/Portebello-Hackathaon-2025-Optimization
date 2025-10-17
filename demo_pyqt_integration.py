#!/usr/bin/env python3
"""
Demo script for the PyQt Integration
Shows how to use the MCTS optimization programmatically
"""

import json
from mcts_integration import run_mcts_optimization

def demo_mcts_optimization():
    """Demonstrate MCTS optimization with sample data"""
    
    print("=== PyQt Integration Demo ===\n")
    
    # Load box data
    try:
        with open('db.json', 'r') as f:
            box_data = json.load(f)
        print("+ Loaded box data from db.json")
    except FileNotFoundError:
        print("- Error: db.json not found")
        return
    
    # Create sample picklist
    sample_items = [
        {
            'dimensions': box_data['boxes']['PRESSED']['12x24']['size'],
            'quantity': 3
        },
        {
            'dimensions': box_data['boxes']['RECT']['3x10']['size'],
            'quantity': 5
        },
        {
            'dimensions': box_data['boxes']['MOS']['2x2']['size'],
            'quantity': 2
        }
    ]
    
    print(f"Sample picklist:")
    for i, item in enumerate(sample_items):
        print(f"  {i+1}. {item['dimensions']} (qty: {item['quantity']})")
    
    # Set warehouse space (large enough for all boxes)
    space_dims = (60, 60, 25)
    print(f"\nWarehouse space: {space_dims[0]}×{space_dims[1]}×{space_dims[2]}")
    
    # Run optimization
    print("\nRunning MCTS optimization...")
    
    def progress_callback(current, total):
        print(f"  Progress: {current}/{total} boxes placed")
    
    try:
        placements, stats = run_mcts_optimization(
            sample_items,
            space_dims,
            max_time_per_step=0.5,  # Fast for demo
            progress_callback=progress_callback
        )
        
        print(f"\n=== Results ===")
        print(f"Total boxes: {stats['total_boxes']}")
        print(f"Placed boxes: {stats['placed_boxes']}")
        print(f"Efficiency: {stats['efficiency']:.1%}")
        print(f"Space utilization: {stats['space_utilization']:.1%}")
        
        if placements:
            print(f"\nBox placements:")
            for i, box in enumerate(placements):
                print(f"  Box {i+1}: {box.w}×{box.d}×{box.h} at ({box.x}, {box.y}, {box.z})")
        else:
            print("No boxes were placed!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n=== Next Steps ===")
    print("1. Run the PyQt GUI: python run_gui.py")
    print("2. Select boxes from the picklist interface")
    print("3. Adjust warehouse space dimensions if needed")
    print("4. Click 'Run MCTS Optimization' to see the 3D visualization")

if __name__ == "__main__":
    demo_mcts_optimization()
