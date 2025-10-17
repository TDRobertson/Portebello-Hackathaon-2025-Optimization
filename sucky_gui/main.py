import json
import csv
import io
import random
import subprocess
from PyQt6 import QtWidgets, QtGui, QtCore
from interface import Ui_MainWindow
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
import math
from warehouse_visualization import WarehouseVisualization2D


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        # Load SKU database
        with open('sku_box_index.json', 'r') as f:
            self.sku_database = json.load(f)
        
        # Initialize order dictionary
        self.order_dict = {}
        
        # Create and set up list widgets
        self.order_list = QtWidgets.QListWidget()
        self.sorted_list = QtWidgets.QListWidget()
        
        # Add list widgets to scroll areas
        self.input_data.setWidget(self.order_list)
        self.sorted_data.setWidget(self.sorted_list)
        # Connect signals
        self.button_run.clicked.connect(self.run_clicked)
        self.add_item.clicked.connect(self.add_sku)
        self.render_visual.setScaledContents(True)
        self.render_progress.setValue(0)
        self.pallet_size.setCurrentIndex(0)
        self.comboBox.currentIndexChanged.connect(self.update_visualization_mode)
    def update_visualization_mode(self):
        mode = self.comboBox.currentText().lower()
        if mode == "pallet":
            self.render_visual.clear()
        elif mode == "route":
            self.render_warehouse()

    def render_warehouse(self):
        # Ensure render_visual has a layout
        if self.render_visual.layout() is None:
            layout = QtWidgets.QVBoxLayout(self.render_visual)
            self.render_visual.setLayout(layout)
        else:
            layout = self.render_visual.layout()
        # Remove any previous widget from render_visual
        for i in reversed(range(layout.count())):
            widget_to_remove = layout.itemAt(i).widget()
            if widget_to_remove:
                widget_to_remove.setParent(None)
        # Create and add the warehouse visualization (persistently)
        self.warehouse_canvas = WarehouseVisualization2D(parent=self.render_visual)
        self.warehouse_canvas.generate_warehouse_map()
        self.warehouse_canvas.draw_warehouse()
        layout.addWidget(self.warehouse_canvas)
        
        # Set up ammount spinner
        self.ammount_spin.setMinimum(1)
        self.ammount_spin.setMaximum(999)
        
        # Set up search bar with clear button
        self.search_bar.setClearButtonEnabled(True)
        
        # Set up the order list widget
        self.order_list.setSpacing(2)
        
    def add_sku(self):
        sku = self.search_bar.text().strip()
        if not sku:
            return
            
        if sku not in self.sku_database:
            print(f"Error: SKU {sku} not found in database")
            return
            
        # Add to order dictionary with amount from spinner
        amount = self.ammount_spin.value()
        self.order_dict[sku] = self.order_dict.get(sku, 0) + amount
        
        # Update displays
        self.update_displays()
        
    def remove_sku(self, sku):
        if sku in self.order_dict:
            self.order_dict[sku] -= 1
            if self.order_dict[sku] <= 0:
                del self.order_dict[sku]
            self.update_displays()
            
    def update_displays(self):
        # Clear current displays
        self.order_list.clear()
        self.sorted_list.clear()
        
        # Update order list with quantities
        for sku, quantity in self.order_dict.items():
            for _ in range(quantity):
                item = QtWidgets.QListWidgetItem()
                widget = QtWidgets.QWidget()
                layout = QtWidgets.QHBoxLayout()
                layout.setContentsMargins(5, 2, 5, 2)
                
                # SKU label
                label = QtWidgets.QLabel(sku)
                layout.addWidget(label)
                
                # Remove button
                remove_btn = QtWidgets.QPushButton("✕")
                remove_btn.setStyleSheet("color: red;")
                remove_btn.setMaximumWidth(30)
                remove_btn.clicked.connect(lambda checked, s=sku: self.remove_sku(s))
                layout.addWidget(remove_btn)
                
                widget.setLayout(layout)
                item.setSizeHint(widget.sizeHint())
                self.order_list.addItem(item)
                self.order_list.setItemWidget(item, widget)
        
        # Update sorted list
        sorted_items = []
        for sku, qty in self.order_dict.items():
            dims = self.sku_database[sku]
            w, d, h = map(math.ceil, dims)
            volume = w * d * h
            sorted_items.append((sku, volume, qty, w, d, h))
        sorted_items.sort(key=lambda x: x[1], reverse=True)
        for sku, volume, qty, w, d, h in sorted_items:
            self.sorted_list.addItem(f"{sku} ({qty}) {w}x{d}x{h}")
            
    def convert_fig_to_pixmap(self, fig):
        # Convert matplotlib figure to QPixmap
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)

        image = QtGui.QImage.fromData(buf.getvalue())
        return QtGui.QPixmap.fromImage(image)

    def run_clicked(self):
        mode = self.comboBox.currentText().lower()
        if mode == "route":
            # Only create warehouse_canvas if it doesn't exist
            if not hasattr(self, 'warehouse_canvas') or self.warehouse_canvas is None:
                self.render_warehouse()
            # Check if start and targets are set
            if not self.warehouse_canvas.start_pos or not self.warehouse_canvas.targets:
                QtWidgets.QMessageBox.warning(self, "Warehouse Simulation", "Please select a start position and at least one target in the warehouse visualization before running the simulation.")
                return
            # Simulate ENTER key: draw path
            self.warehouse_canvas.calculate_optimal_route()
            # Explicitly redraw the warehouse with the updated path
            self.warehouse_canvas.draw_warehouse(
                start=self.warehouse_canvas.start_pos,
                targets=self.warehouse_canvas.targets,
                path=self.warehouse_canvas.current_path,
                title=f"Optimal Route - {len(self.warehouse_canvas.current_path)} steps"
            )
            # Force matplotlib canvas redraw in PyQt
            self.warehouse_canvas.fig.canvas.draw()
            return
        elif mode == "pallet":
            # Remove any previous widget from render_visual
            if self.render_visual.layout() is None:
                layout = QtWidgets.QVBoxLayout(self.render_visual)
                self.render_visual.setLayout(layout)
            else:
                layout = self.render_visual.layout()
            for i in reversed(range(layout.count())):
                widget_to_remove = layout.itemAt(i).widget()
                if widget_to_remove:
                    widget_to_remove.setParent(None)
            # Run the pallet visualization logic using mcts_v2.py and nn.pth
            if not self.order_dict:
                print("Error: No items in order")
                return
            selected_size = self.pallet_size.currentText()
            if not selected_size:
                print("Error: No pallet size selected")
                return
            # Write order to CSV
            with open('order_boxes.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                for sku, quantity in self.order_dict.items():
                    dimensions = self.sku_database[sku]
                    # Reorganize so d is largest, w is smallest, h is middle
                    sorted_dims = sorted(dimensions)
                    w = sorted_dims[0]  # smallest
                    h = sorted_dims[1]  # middle
                    d = sorted_dims[2]  # largest
                    # Scale down by 4 before rounding
                    w, d, h = map(lambda x: math.ceil(x / 2), (w, d, h))
                    for _ in range(quantity):
                        writer.writerow((w, d, h))
            # Show progress bar at start
            self.render_progress.setValue(10)
            QtWidgets.QApplication.processEvents()
            try:
                import mcts_v2
                import torch
                # Load dims
                self.render_progress.setValue(20)
                QtWidgets.QApplication.processEvents()
                mcts_v2.dims = mcts_v2.load_dims('order_boxes.csv')
                if not mcts_v2.dims:
                    print("No valid boxes.")
                    self.render_progress.setValue(0)
                    return
                # Load model
                self.render_progress.setValue(40)
                QtWidgets.QApplication.processEvents()
                model = mcts_v2.SimpleCNN()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                state = torch.load("nn.pth", map_location=device)
                model.load_state_dict(state)
                model.to(device)
                self.render_progress.setValue(60)
                QtWidgets.QApplication.processEvents()
                max_box_dim = float(max(max(w,d,h) for w,d,h in mcts_v2.dims))
                # Run MCTS (long step)
                final = mcts_v2.run_mcts(model, device, max_box_dim)
                self.render_progress.setValue(90)
                QtWidgets.QApplication.processEvents()
                # Plot placement
                fig = plt.figure(figsize=(8,8))
                ax  = fig.add_subplot(111, projection="3d")
                vols = [b.volume for b in final]
                cmap = plt.cm.viridis
                norm = plt.Normalize(min(vols), max(vols)) if vols else None
                for b in final:
                    col = cmap(norm(b.volume)) if norm else "skyblue"
                    ax.bar3d(b.x, b.y, b.z, b.w, b.d, b.h,
                             color=col, edgecolor="k", alpha=0.7)
                ax.set_xlim(0, mcts_v2.SPACE_W)
                ax.set_ylim(0, mcts_v2.SPACE_D)
                ax.set_zlim(0, mcts_v2.SPACE_H)
                plt.title(f"Placed {len(final)}/{len(mcts_v2.dims)} boxes")
                plt.tight_layout()
                # Embed interactive Matplotlib canvas so the user can rotate the 3D plot with the mouse
                canvas = FigureCanvas(fig)
                canvas.setParent(self.render_visual)
                canvas.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
                canvas.setFocus()
                toolbar = NavigationToolbar(canvas, self.render_visual)
                # Add toolbar and canvas to layout
                layout.addWidget(toolbar)
                layout.addWidget(canvas)
                plt.close(fig)
                self.render_progress.setValue(100)
                QtWidgets.QApplication.processEvents()
                print(f"Summary: placed {len(final)}/{len(mcts_v2.dims)} boxes.")
            except Exception as e:
                print(f"Error: {str(e)}")
                self.render_progress.setValue(0)
            return


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
