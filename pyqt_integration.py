#!/usr/bin/env python3
"""
PyQt Integration for Portobello Hackathon Optimization
Integrates picklist functionality, MCTS algorithm, and warehouse simulation
"""

import sys
import json
import csv
import math
import random
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QComboBox, QPushButton, QListWidget, 
    QListWidgetItem, QTabWidget, QTextEdit, QProgressBar, QSplitter,
    QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox, QMessageBox,
    QFileDialog, QStatusBar, QMenuBar, QAction
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor

# Import the integration modules
from mcts_integration import run_mcts_optimization, Box
from warehouse_visualization import WarehouseVisualization2D, WarehouseVisualization3D, PathfindingVisualization


@dataclass
class BoxItem:
    """Represents a box item from the picklist"""
    type: str
    size: str
    dimensions: List[float]
    weight: float
    quantity: int = 1


class MCTSWorker(QThread):
    """Worker thread for running MCTS algorithm"""
    progress_updated = pyqtSignal(int, int)  # current, total
    result_ready = pyqtSignal(list, dict)  # placements, stats
    error_occurred = pyqtSignal(str)
    
    def __init__(self, box_items: List[BoxItem], space_dims: Tuple[int, int, int]):
        super().__init__()
        self.box_items = box_items
        self.space_dims = space_dims
        self.running = True
        
    def run(self):
        try:
            # Convert box items to format expected by MCTS
            box_data = []
            for item in self.box_items:
                for _ in range(item.quantity):
                    box_data.append({
                        'dimensions': item.dimensions,
                        'quantity': 1
                    })
            
            if not box_data:
                self.error_occurred.emit("No boxes to place")
                return
            
            # Run MCTS optimization
            def progress_callback(current, total):
                if self.running:
                    self.progress_updated.emit(current, total)
            
            placements, stats = run_mcts_optimization(
                box_data, 
                self.space_dims, 
                max_time_per_step=1.0,
                progress_callback=progress_callback
            )
            
            if self.running:
                self.result_ready.emit(placements, stats)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def stop(self):
        self.running = False


# WarehouseVisualization class is now imported from warehouse_visualization module


class PicklistWidget(QWidget):
    """Widget for managing the picklist functionality"""
    
    def __init__(self):
        super().__init__()
        self.box_data = {}
        self.selected_items = []
        self.init_ui()
        self.load_box_data()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Box selection group
        selection_group = QGroupBox("Box Selection")
        selection_layout = QGridLayout()
        
        # Box type selection
        selection_layout.addWidget(QLabel("Box Type:"), 0, 0)
        self.box_type_combo = QComboBox()
        self.box_type_combo.currentTextChanged.connect(self.update_size_options)
        selection_layout.addWidget(self.box_type_combo, 0, 1)
        
        # Box size selection
        selection_layout.addWidget(QLabel("Size:"), 1, 0)
        self.box_size_combo = QComboBox()
        selection_layout.addWidget(self.box_size_combo, 1, 1)
        
        # Quantity selection
        selection_layout.addWidget(QLabel("Quantity:"), 2, 0)
        self.quantity_spin = QSpinBox()
        self.quantity_spin.setMinimum(1)
        self.quantity_spin.setMaximum(100)
        self.quantity_spin.setValue(1)
        selection_layout.addWidget(self.quantity_spin, 2, 1)
        
        # Add button
        self.add_button = QPushButton("Add to Picklist")
        self.add_button.clicked.connect(self.add_to_picklist)
        selection_layout.addWidget(self.add_button, 3, 0, 1, 2)
        
        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)
        
        # Selected items list
        items_group = QGroupBox("Selected Items")
        items_layout = QVBoxLayout()
        
        self.items_list = QListWidget()
        items_layout.addWidget(self.items_list)
        
        # Clear button
        clear_button = QPushButton("Clear All")
        clear_button.clicked.connect(self.clear_picklist)
        items_layout.addWidget(clear_button)
        
        items_group.setLayout(items_layout)
        layout.addWidget(items_group)
        
        self.setLayout(layout)
        
    def load_box_data(self):
        """Load box data from db.json"""
        try:
            with open('db.json', 'r') as f:
                self.box_data = json.load(f)
            
            # Populate box type combo
            self.box_type_combo.clear()
            self.box_type_combo.addItem("-- Select a box type --")
            for box_type in self.box_data.get('boxes', {}).keys():
                self.box_type_combo.addItem(box_type)
                
        except FileNotFoundError:
            QMessageBox.warning(self, "Error", "db.json file not found")
        except json.JSONDecodeError:
            QMessageBox.warning(self, "Error", "Invalid JSON in db.json")
    
    def update_size_options(self):
        """Update size options based on selected box type"""
        self.box_size_combo.clear()
        self.box_size_combo.addItem("-- Select a size --")
        
        box_type = self.box_type_combo.currentText()
        if box_type in self.box_data.get('boxes', {}):
            for size in self.box_data['boxes'][box_type].keys():
                self.box_size_combo.addItem(size)
    
    def add_to_picklist(self):
        """Add selected box to picklist"""
        box_type = self.box_type_combo.currentText()
        size = self.box_size_combo.currentText()
        quantity = self.quantity_spin.value()
        
        if box_type == "-- Select a box type --" or size == "-- Select a size --":
            QMessageBox.warning(self, "Error", "Please select both box type and size")
            return
        
        # Get box details
        box_info = self.box_data['boxes'][box_type][size]
        dimensions = box_info['size']
        weight = box_info['weight']
        
        # Create box item
        box_item = BoxItem(
            type=box_type,
            size=size,
            dimensions=dimensions,
            weight=weight,
            quantity=quantity
        )
        
        # Add to list
        self.selected_items.append(box_item)
        self.update_items_display()
    
    def update_items_display(self):
        """Update the items list display"""
        self.items_list.clear()
        for i, item in enumerate(self.selected_items):
            item_text = f"{item.type} - {item.size} (Qty: {item.quantity})"
            list_item = QListWidgetItem(item_text)
            list_item.setData(Qt.UserRole, i)
            self.items_list.addItem(list_item)
    
    def clear_picklist(self):
        """Clear all items from picklist"""
        self.selected_items.clear()
        self.update_items_display()
    
    def get_selected_items(self) -> List[BoxItem]:
        """Get list of selected items"""
        return self.selected_items.copy()


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.mcts_worker = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Portobello America - Optimization Suite")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Picklist and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        # Picklist widget
        self.picklist_widget = PicklistWidget()
        left_layout.addWidget(self.picklist_widget)
        
        # Space configuration
        space_group = QGroupBox("Warehouse Space Configuration")
        space_layout = QGridLayout()
        
        space_layout.addWidget(QLabel("Width:"), 0, 0)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(10, 100)
        self.width_spin.setValue(50)
        space_layout.addWidget(self.width_spin, 0, 1)
        
        space_layout.addWidget(QLabel("Depth:"), 1, 0)
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(10, 100)
        self.depth_spin.setValue(50)
        space_layout.addWidget(self.depth_spin, 1, 1)
        
        space_layout.addWidget(QLabel("Height:"), 2, 0)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(5, 50)
        self.height_spin.setValue(20)
        space_layout.addWidget(self.height_spin, 2, 1)
        
        space_group.setLayout(space_layout)
        left_layout.addWidget(space_group)
        
        # Optimization controls
        opt_group = QGroupBox("Optimization Controls")
        opt_layout = QVBoxLayout()
        
        self.optimize_button = QPushButton("Run MCTS Optimization")
        self.optimize_button.clicked.connect(self.run_optimization)
        opt_layout.addWidget(self.optimize_button)
        
        self.stop_button = QPushButton("Stop Optimization")
        self.stop_button.clicked.connect(self.stop_optimization)
        self.stop_button.setEnabled(False)
        opt_layout.addWidget(self.stop_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        opt_layout.addWidget(self.progress_bar)
        
        opt_group.setLayout(opt_layout)
        left_layout.addWidget(opt_group)
        
        # Results display
        results_group = QGroupBox("Optimization Results")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        results_group.setLayout(results_layout)
        left_layout.addWidget(results_group)
        
        left_widget.setLayout(left_layout)
        main_splitter.addWidget(left_widget)
        
        # Right side - Visualization
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        # Create tab widget for different visualizations
        viz_tabs = QTabWidget()
        
        # 3D visualization tab
        self.visualization_3d = WarehouseVisualization3D()
        viz_tabs.addTab(self.visualization_3d, "3D Layout")
        
        # 2D path visualization tab
        self.visualization_2d = WarehouseVisualization2D()
        viz_tabs.addTab(self.visualization_2d, "2D Path")
        
        right_layout.addWidget(viz_tabs)
        right_widget.setLayout(right_layout)
        main_splitter.addWidget(right_widget)
        
        # Set splitter proportions
        main_splitter.setSizes([500, 900])
        
        # Set main layout
        central_layout = QHBoxLayout()
        central_layout.addWidget(main_splitter)
        central_widget.setLayout(central_layout)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        load_action = QAction('Load Picklist', self)
        load_action.triggered.connect(self.load_picklist)
        file_menu.addAction(load_action)
        
        save_action = QAction('Save Picklist', self)
        save_action.triggered.connect(self.save_picklist)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def run_optimization(self):
        """Run MCTS optimization on selected items"""
        selected_items = self.picklist_widget.get_selected_items()
        
        if not selected_items:
            QMessageBox.warning(self, "Error", "Please add items to the picklist first")
            return
        
        # Get space dimensions
        space_dims = (self.width_spin.value(), self.depth_spin.value(), self.height_spin.value())
        
        # Validate that boxes can fit in the space
        max_w = max(item.dimensions[0] for item in selected_items)
        max_d = max(item.dimensions[1] for item in selected_items)
        max_h = max(item.dimensions[2] for item in selected_items)
        
        if max_w > space_dims[0] or max_d > space_dims[1] or max_h > space_dims[2]:
            QMessageBox.warning(self, "Space Too Small", 
                              f"Some boxes are too large for the current space:\n"
                              f"Largest box: {max_w}×{max_d}×{max_h}\n"
                              f"Current space: {space_dims[0]}×{space_dims[1]}×{space_dims[2]}\n\n"
                              f"Please increase the space dimensions or select smaller boxes.")
            return
        
        # Disable controls
        self.optimize_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # Start MCTS worker
        self.mcts_worker = MCTSWorker(selected_items, space_dims)
        self.mcts_worker.progress_updated.connect(self.update_progress)
        self.mcts_worker.result_ready.connect(self.optimization_complete)
        self.mcts_worker.error_occurred.connect(self.optimization_error)
        self.mcts_worker.start()
        
        self.status_bar.showMessage("Running optimization...")
    
    def stop_optimization(self):
        """Stop the running optimization"""
        if self.mcts_worker and self.mcts_worker.isRunning():
            self.mcts_worker.stop()
            self.mcts_worker.wait()
        
        self.optimize_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_bar.showMessage("Optimization stopped")
    
    def update_progress(self, current, total):
        """Update progress bar"""
        progress = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)
        self.status_bar.showMessage(f"Optimizing... {current}/{total} boxes placed")
    
    def optimization_complete(self, placements, stats):
        """Handle optimization completion"""
        self.optimize_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)
        
        # Update visualizations
        space_dims = (self.width_spin.value(), self.depth_spin.value(), self.height_spin.value())
        self.visualization_3d.update_visualization(placements, space_dims)
        
        # Update 2D visualization with warehouse map
        self.visualization_2d.generate_warehouse_map(20, 25)
        self.visualization_2d.draw_warehouse(title="Warehouse Layout - Optimization Complete")
        
        # Update results text
        results = f"""Optimization Complete!

Statistics:
- Total boxes: {stats['total_boxes']}
- Placed boxes: {stats['placed_boxes']}
- Efficiency: {stats['efficiency']:.2%}
- Space utilization: {stats['space_utilization']:.2%}

Placement details:
"""
        for i, box in enumerate(placements):
            results += f"Box {i+1}: {box.w}×{box.d}×{box.h} at ({box.x}, {box.y}, {box.z})\n"
        
        self.results_text.setPlainText(results)
        self.status_bar.showMessage(f"Optimization complete - {stats['placed_boxes']}/{stats['total_boxes']} boxes placed")
    
    def optimization_error(self, error_msg):
        """Handle optimization error"""
        self.optimize_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Show detailed error message
        error_dialog = QMessageBox(self)
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle("Optimization Error")
        error_dialog.setText("An error occurred during optimization:")
        error_dialog.setDetailedText(error_msg)
        error_dialog.exec_()
        
        self.status_bar.showMessage("Optimization failed")
    
    def load_picklist(self):
        """Load picklist from file"""
        # Implementation for loading picklist from file
        pass
    
    def save_picklist(self):
        """Save picklist to file"""
        # Implementation for saving picklist to file
        pass
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About", 
                         "Portobello America Optimization Suite\n\n"
                         "Integrates picklist management, MCTS optimization, "
                         "and warehouse visualization for efficient box placement.")


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
