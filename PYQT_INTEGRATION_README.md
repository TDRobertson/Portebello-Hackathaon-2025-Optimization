# PyQt Integration for Portobello Optimization Suite

This PyQt GUI application integrates the picklist functionality from `index.html`, the MCTS optimization algorithm from `mcts.py`, and the warehouse simulation from `warehouse_sim_v2.py` into a unified desktop application.

## Features

### 1. Picklist Management
- **Box Selection**: Choose from predefined box types and sizes loaded from `db.json`
- **Quantity Control**: Specify how many of each box type to include
- **Visual List**: See all selected items in a clear, manageable list
- **Data Persistence**: Load and save picklist configurations

### 2. MCTS Optimization
- **Monte Carlo Tree Search**: Advanced algorithm for optimal box placement
- **Real-time Progress**: Visual progress bar and status updates
- **Configurable Space**: Set warehouse dimensions (width, depth, height)
- **Performance Metrics**: Efficiency and space utilization statistics

### 3. Warehouse Visualization
- **3D Layout View**: Interactive 3D visualization of placed boxes
- **2D Path View**: Top-down warehouse layout with path visualization
- **Color-coded Boxes**: Different colors for different box volumes
- **Interactive Controls**: Zoom, pan, and rotate the 3D view

### 4. Integration Features
- **Seamless Data Flow**: Picklist → MCTS → Visualization
- **Threaded Processing**: Non-blocking optimization runs
- **Error Handling**: Comprehensive error messages and recovery
- **Professional UI**: Clean, modern interface following Portobello branding

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Required Files**:
   - `db.json` - Box type definitions
   - `boxes.csv` - Sample box data (optional)

## Usage

### Running the Application

```bash
python run_gui.py
```

Or directly:
```bash
python pyqt_integration.py
```

### Basic Workflow

1. **Select Boxes**:
   - Choose box type from dropdown
   - Select size from available options
   - Set quantity
   - Click "Add to Picklist"

2. **Configure Space**:
   - Set warehouse dimensions (width, depth, height)
   - Default is 15×15×10 units

3. **Run Optimization**:
   - Click "Run MCTS Optimization"
   - Monitor progress in real-time
   - Stop if needed using "Stop Optimization"

4. **View Results**:
   - Check 3D layout in "3D Layout" tab
   - View 2D warehouse map in "2D Path" tab
   - Review statistics in results panel

### Advanced Features

- **Load/Save Picklists**: Use File menu to persist configurations
- **Multiple Visualizations**: Switch between 3D and 2D views
- **Real-time Updates**: Progress and results update automatically
- **Error Recovery**: Graceful handling of optimization failures

## File Structure

```
pyqt_integration.py          # Main GUI application
mcts_integration.py          # MCTS algorithm integration
warehouse_visualization.py   # 2D/3D visualization components
run_gui.py                   # Application launcher
requirements.txt             # Python dependencies
db.json                      # Box type definitions
```

## Technical Details

### Architecture
- **Main Window**: `MainWindow` class manages the overall application
- **Picklist Widget**: `PicklistWidget` handles box selection and management
- **MCTS Worker**: `MCTSWorker` runs optimization in separate thread
- **Visualization**: Separate classes for 2D and 3D warehouse views

### Data Flow
1. User selects boxes in picklist widget
2. Box data converted to MCTS format
3. MCTS algorithm runs in background thread
4. Results passed to visualization components
5. 3D and 2D views updated with placement results

### Threading
- MCTS optimization runs in `QThread` to prevent UI freezing
- Progress updates sent via Qt signals
- Safe thread communication using Qt's signal/slot system

## Customization

### Adding New Box Types
Edit `db.json` to add new box types and sizes:
```json
{
  "boxes": {
    "NEW_TYPE": {
      "size_name": {
        "size": [width, depth, height],
        "weight": weight_value
      }
    }
  }
}
```

### Modifying Warehouse Layout
Edit `warehouse_visualization.py` to change:
- Shelf placement patterns
- Aisle configurations
- Color schemes
- Grid dimensions

### Adjusting MCTS Parameters
Modify `mcts_integration.py` to change:
- UCT exploration constant
- Simulation time per step
- Heuristic functions
- Termination conditions

## Troubleshooting

### Common Issues

1. **PyQt5 Import Error**:
   ```bash
   pip install PyQt5
   ```

2. **Missing db.json**:
   - Ensure `db.json` exists in the same directory
   - Check file permissions

3. **Optimization Hangs**:
   - Use "Stop Optimization" button
   - Check for invalid box dimensions
   - Verify space dimensions are reasonable

4. **Visualization Not Updating**:
   - Check console for error messages
   - Ensure matplotlib backend is properly configured
   - Try restarting the application

### Performance Tips

- **Large Picklists**: Consider reducing quantity for very large lists
- **Complex Spaces**: Increase MCTS simulation time for better results
- **Memory Usage**: Close other applications if running out of memory
- **Real-time Updates**: Disable progress updates for faster processing

## Future Enhancements

- **Path Planning**: Integration with warehouse pathfinding
- **Multiple Algorithms**: Support for different optimization methods
- **Export Features**: Save results to CSV/JSON formats
- **Batch Processing**: Process multiple picklists automatically
- **Advanced Visualization**: Interactive 3D controls and animations

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review console output for error messages
3. Ensure all dependencies are properly installed
4. Verify input data format matches expected structure
