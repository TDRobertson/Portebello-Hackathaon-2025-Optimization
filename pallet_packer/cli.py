from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print
from .models import InputSpec
from .heuristics import pack
from .visualize import render, render_interactive
from .plotly_viz import render_plotly_interactive, render_plotly_static


app = typer.Typer(add_completion=False)


@app.command()
def main(
    input: Path = typer.Argument(..., exists=True, readable=True, help="Path to input JSON"),
    out: Path = typer.Argument(Path("out"), help="Output directory"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Show interactive 3D visualization"),
    plotly: bool = typer.Option(False, "--plotly", "-p", help="Use Plotly for better 3D visualization"),
):
    out.mkdir(parents=True, exist_ok=True)
    spec = InputSpec.model_validate_json(input.read_text())
    result = pack(spec)

    # Write placements JSON/CSV
    placements_json = out / "placements.json"
    with placements_json.open("w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in result.placements], f, indent=2)

    placements_csv = out / "placements.csv"
    with placements_csv.open("w", encoding="utf-8") as f:
        f.write("box_id,index,x,y,z,length,width,height,orientation,label_side\n")
        for p in result.placements:
            f.write(
                f"{p.box_id},{p.index},{p.x:.2f},{p.y:.2f},{p.z:.2f},{p.length:.2f},{p.width:.2f},{p.height:.2f},{p.orientation},{p.label_side or ''}\n"
            )

    # Sequences
    pick_csv = out / "pick_sequence.csv"
    with pick_csv.open("w", encoding="utf-8") as f:
        f.write("box_id,index\n")
        for bid, idx in result.pick_sequence:
            f.write(f"{bid},{idx}\n")

    stack_csv = out / "stack_sequence.csv"
    with stack_csv.open("w", encoding="utf-8") as f:
        f.write("box_id,index\n")
        for bid, idx in result.stack_sequence:
            f.write(f"{bid},{idx}\n")

    # Visualization
    viz_path = out / "visualization.png"
    if plotly:
        render_plotly_static(spec.pallet, result.placements, str(viz_path))
    else:
        render(spec.pallet, result.placements, str(viz_path))

    # Summary
    summary = out / "summary.txt"
    total_boxes = sum(b.quantity for b in spec.boxes)
    placed_boxes = len(result.placements)
    with summary.open("w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    f"Total boxes: {total_boxes}",
                    f"Placed boxes: {placed_boxes}",
                    f"Unplaced boxes: {len(result.unplaced)}",
                    f"Used height: {result.used_height:.2f}",
                    f"Utilization (by volume): {result.utilization*100:.2f}%",
                ]
            )
        )

    print(f"[green]Wrote outputs to[/green] {out}")
    
    # Show interactive visualization if requested
    if interactive:
        if plotly:
            print("[blue]Opening Plotly interactive 3D visualization...[/blue]")
            render_plotly_interactive(spec.pallet, result.placements)
        else:
            print("[blue]Opening matplotlib interactive 3D visualization...[/blue]")
            render_interactive(spec.pallet, result.placements)


if __name__ == "__main__":
    app()


