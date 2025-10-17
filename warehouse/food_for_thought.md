Notes & tuning ideas

The mapping in location_to_grid() is heuristic. When you have a concrete warehouse layout (actual aisle spacing and shelf geometry), replace the formula with a deterministic mapping that matches real distances.

If a mapped cell falls on a shelf, find_nearest_free() uses BFS to find the closest free aisle cell — so destination markers will always be on walkable cells.

Change n_targets to simulate longer runs; increase per_cell_frames in animate_path() for smoother slower movement.

When you move to real external input, replace the generate_random_locations() call with your input source, e.g. reading from a socket, file, or API:

external_codes = ["B5.J2.43.01", "B1.J3.21.02"]
targets = [location_to_grid(c) for c in external_codes]