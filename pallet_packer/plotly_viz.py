from __future__ import annotations

from typing import List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

from .models import Placement, Pallet


def create_solid_box(x: float, y: float, z: float, length: float, width: float, height: float, color: str, name: str = ""):
    """Create a solid 3D box using efficient mesh faces"""
    
    # Define the 8 vertices of the box
    vertices = np.array([
        [x, y, z],                    # 0: bottom-left-back
        [x + length, y, z],           # 1: bottom-right-back
        [x + length, y + width, z],   # 2: bottom-right-front
        [x, y + width, z],            # 3: bottom-left-front
        [x, y, z + height],           # 4: top-left-back
        [x + length, y, z + height],  # 5: top-right-back
        [x + length, y + width, z + height],  # 6: top-right-front
        [x, y + width, z + height],   # 7: top-left-front
    ])
    
    # Define the 12 triangular faces (2 triangles per box face)
    # Each face is defined by 3 vertices in counter-clockwise order
    faces = np.array([
        # Bottom face (z=0)
        [0, 1, 2], [0, 2, 3],
        # Top face (z=height) 
        [4, 7, 6], [4, 6, 5],
        # Front face (y=width)
        [3, 2, 6], [3, 6, 7],
        # Back face (y=0)
        [0, 4, 5], [0, 5, 1],
        # Left face (x=0)
        [0, 3, 7], [0, 7, 4],
        # Right face (x=length)
        [1, 5, 6], [1, 6, 2],
    ])
    
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=0.8,
        name=name,
        showlegend=True,
        lighting=dict(
            ambient=0.3,
            diffuse=0.8,
            specular=0.2,
            roughness=0.4,
            fresnel=0.1
        ),
        lightposition=dict(
            x=1000,
            y=1000,
            z=1000
        ),
        flatshading=True,
        alphahull=0
    )


def create_box_mesh(x: float, y: float, z: float, length: float, width: float, height: float, color: str, name: str = ""):
    """Create a solid 3D box using proper mesh faces"""
    # Define the 8 vertices of the box
    vertices = np.array([
        [x, y, z],                    # 0: bottom-left-back
        [x + length, y, z],           # 1: bottom-right-back
        [x + length, y + width, z],   # 2: bottom-right-front
        [x, y + width, z],            # 3: bottom-left-front
        [x, y, z + height],           # 4: top-left-back
        [x + length, y, z + height],  # 5: top-right-back
        [x + length, y + width, z + height],  # 6: top-right-front
        [x, y + width, z + height],   # 7: top-left-front
    ])
    
    # Define the 12 triangular faces (2 triangles per box face)
    # Note: Order matters for proper face orientation
    faces = np.array([
        # Bottom face (z=0)
        [0, 1, 2], [0, 2, 3],
        # Top face (z=height)
        [4, 7, 6], [4, 6, 5],
        # Front face (y=width)
        [3, 2, 6], [3, 6, 7],
        # Back face (y=0)
        [0, 4, 5], [0, 5, 1],
        # Left face (x=0)
        [0, 3, 7], [0, 7, 4],
        # Right face (x=length)
        [1, 5, 6], [1, 6, 2],
    ])
    
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=0.9,
        name=name,
        showlegend=True,
        lighting=dict(
            ambient=0.3,
            diffuse=0.8,
            specular=0.2,
            roughness=0.4,
            fresnel=0.1
        ),
        lightposition=dict(
            x=1000,
            y=1000,
            z=1000
        ),
        flatshading=True,  # This helps with solid appearance
        alphahull=0  # This ensures all faces are rendered
    )


def render_plotly_interactive(pallet: Pallet, placements: List[Placement]) -> None:
    """Create an interactive 3D visualization using Plotly"""
    
    # Create color mapping
    colors = {}
    color_palette = px.colors.qualitative.Set3
    def color_for(box_id: str):
        if box_id not in colors:
            colors[box_id] = color_palette[len(colors) % len(color_palette)]
        return colors[box_id]
    
    # Create the figure
    fig = go.Figure()
    
    # Add solid pallet base
    pallet_base = go.Mesh3d(
        x=[0, pallet.length, pallet.length, 0, 0, pallet.length, pallet.length, 0],
        y=[0, 0, pallet.width, pallet.width, 0, 0, pallet.width, pallet.width],
        z=[0, 0, 0, 0, 0, 0, 0, 0],
        i=[0, 1, 2, 3],
        j=[1, 2, 3, 0],
        k=[4, 5, 6, 7],
        color='lightgray',
        opacity=0.5,
        name='Pallet Base',
        showlegend=True,
        flatshading=True
    )
    fig.add_trace(pallet_base)
    
    # Add solid boxes
    for i, p in enumerate(placements):
        color = color_for(p.box_id)
        solid_box = create_solid_box(
            p.x, p.y, p.z, p.length, p.width, p.height, 
            color, f"{p.box_id} #{p.index}"
        )
        fig.add_trace(solid_box)
    
    # Calculate statistics
    total_boxes = len(placements)
    used_volume = sum(p.length * p.width * p.height for p in placements)
    pallet_volume = pallet.length * pallet.width * pallet.height_limit
    utilization = (used_volume / pallet_volume * 100) if pallet_volume > 0 else 0
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'Interactive 3D Pallet Layout<br><sub>Boxes: {total_boxes} | Utilization: {utilization:.1f}% | Volume: {used_volume:.0f}/{pallet_volume:.0f}</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        scene=dict(
            xaxis=dict(
                title='X (Length)',
                range=[0, pallet.length],
                backgroundcolor='rgba(0,0,0,0)',
                gridcolor='lightgray',
                showbackground=True,
                showgrid=True,
                gridwidth=1
            ),
            yaxis=dict(
                title='Y (Width)',
                range=[0, pallet.width],
                backgroundcolor='rgba(0,0,0,0)',
                gridcolor='lightgray',
                showbackground=True,
                showgrid=True,
                gridwidth=1
            ),
            zaxis=dict(
                title='Z (Height)',
                range=[0, pallet.height_limit],
                backgroundcolor='rgba(0,0,0,0)',
                gridcolor='lightgray',
                showbackground=True,
                showgrid=True,
                gridwidth=1
            ),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=1.2)
            ),
            bgcolor='white'
        ),
        width=800,
        height=600,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    # Add annotations for controls
    fig.add_annotation(
        text="Controls: Mouse drag to rotate • Scroll to zoom • Right-click to pan",
        xref="paper", yref="paper",
        x=0.98, y=0.02,
        showarrow=False,
        font=dict(size=10, color="gray"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
    # Show the interactive plot
    fig.show()


def render_plotly_static(pallet: Pallet, placements: List[Placement], path: str) -> None:
    """Create a static image using Plotly and save it"""
    
    # Create color mapping
    colors = {}
    color_palette = px.colors.qualitative.Set3
    def color_for(box_id: str):
        if box_id not in colors:
            colors[box_id] = color_palette[len(colors) % len(color_palette)]
        return colors[box_id]
    
    # Create the figure
    fig = go.Figure()
    
    # Add solid pallet base
    pallet_base = go.Mesh3d(
        x=[0, pallet.length, pallet.length, 0, 0, pallet.length, pallet.length, 0],
        y=[0, 0, pallet.width, pallet.width, 0, 0, pallet.width, pallet.width],
        z=[0, 0, 0, 0, 0, 0, 0, 0],
        i=[0, 1, 2, 3],
        j=[1, 2, 3, 0],
        k=[4, 5, 6, 7],
        color='lightgray',
        opacity=0.5,
        name='Pallet Base',
        showlegend=True,
        flatshading=True
    )
    fig.add_trace(pallet_base)
    
    # Add solid boxes
    for i, p in enumerate(placements):
        color = color_for(p.box_id)
        solid_box = create_solid_box(
            p.x, p.y, p.z, p.length, p.width, p.height, 
            color, f"{p.box_id} #{p.index}"
        )
        fig.add_trace(solid_box)
    
    # Calculate statistics
    total_boxes = len(placements)
    used_volume = sum(p.length * p.width * p.height for p in placements)
    pallet_volume = pallet.length * pallet.width * pallet.height_limit
    utilization = (used_volume / pallet_volume * 100) if pallet_volume > 0 else 0
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'3D Pallet Layout<br><sub>Boxes: {total_boxes} | Utilization: {utilization:.1f}% | Volume: {used_volume:.0f}/{pallet_volume:.0f}</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        scene=dict(
            xaxis=dict(
                title='X (Length)',
                range=[0, pallet.length],
                backgroundcolor='rgba(0,0,0,0)',
                gridcolor='lightgray',
                showbackground=True,
                showgrid=True,
                gridwidth=1
            ),
            yaxis=dict(
                title='Y (Width)',
                range=[0, pallet.width],
                backgroundcolor='rgba(0,0,0,0)',
                gridcolor='lightgray',
                showbackground=True,
                showgrid=True,
                gridwidth=1
            ),
            zaxis=dict(
                title='Z (Height)',
                range=[0, pallet.height_limit],
                backgroundcolor='rgba(0,0,0,0)',
                gridcolor='lightgray',
                showbackground=True,
                showgrid=True,
                gridwidth=1
            ),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=1.2)
            ),
            bgcolor='white'
        ),
        width=800,
        height=600,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    # Save as static image
    fig.write_image(path, width=1000, height=800, scale=2)
