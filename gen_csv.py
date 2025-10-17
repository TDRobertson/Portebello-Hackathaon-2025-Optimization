import csv
import random
import sys
from typing import List, Tuple

def generate_csv(path: str,
                 num_boxes: int = 75,
                 min_size: int = 1,
                 max_size: int = 5,
                 num_types: int = 6) -> None:
    """
    Generates a CSV at `path` with `num_boxes` lines, drawn from
    `num_types` distinct box‐types.  Each type is a random (w,d,h)
    in [min_size..max_size], and the counts of each type are random
    but sum to num_boxes.
    """
    box_dict = {'t1': (17, 5, 7), 't2': (10, 6, 9), 't3': (11, 4, 7),
                't4': (15, 7, 7), 't5': (11, 5, 7), 't6': (12, 5, 12)}
    
    boxes: List[Tuple[int, int, int]] = []

    # 2) Randomly partition num_boxes into num_types non-negative counts
    #    by sampling (num_types-1) cut points in [0..num_boxes]
    cuts = sorted(random.sample(range(num_boxes+1), num_types-1))
    cuts = [0] + cuts + [num_boxes]
    counts = [cuts[i+1] - cuts[i] for i in range(num_types)]

    # 2) Randomly select box types
    selected_types = random.sample(list(box_dict.items()), num_types)

    # 3) Write boxes to CSV
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for (type_name, (w, d, h)), count in zip(selected_types, counts):
            for _ in range(count):
                writer.writerow([w, d, h])

    # 4) Report
    print(f"Wrote {num_boxes} boxes to '{path}' in {num_types} types:")
    for i, ((type_name, (w, d, h)), cnt) in enumerate(zip(selected_types, counts), start=1):
        print(f"  Type #{i} ({type_name}): size={w}×{d}×{h} → count={cnt}")

def main() -> None:
    if len(sys.argv) not in (2,5):
        print("Usage:")
        print("  python gen_boxes.py out.csv")
        print("  python gen_boxes.py out.csv NUM MIN_SIZE MAX_SIZE")
        sys.exit(1)

    path = sys.argv[1]
    if len(sys.argv) == 2:
        num, mn, mx = 75, 1, 5
    else:
        num = int(sys.argv[2])
        mn  = int(sys.argv[3])
        mx  = int(sys.argv[4])
        if not (1 <= mn <= mx):
            print("Require 1 <= MIN_SIZE <= MAX_SIZE")
            sys.exit(1)

    random.seed()  # or set a fixed seed for reproducibility
    generate_csv(path, num, mn, mx, num_types=5)

if __name__ == "__main__":
    main()