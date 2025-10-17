import csv
import random
import sys
from typing import List, Tuple

def generate_csv(path: str,
                 num_boxes: int = 50,
                 min_size: int = 1,
                 max_size: int = 5,
                 num_types: int = 5) -> None:
    """
    Generates a CSV at `path` with `num_boxes` lines, drawn from
    `num_types` distinct box‐types.  Each type is a random (w,d,h)
    in [min_size..max_size], and the counts of each type are random
    but sum to num_boxes.
    """
    # 1) Generate num_types distinct random (w,d,h) triples
    types: List[Tuple[int,int,int]] = []
    seen = set()
    while len(types) < num_types:
        w = random.randint(min_size, max_size)
        d = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        tpl = (w,d,h)
        if tpl not in seen:
            seen.add(tpl)
            types.append(tpl)

    # 2) Randomly partition num_boxes into num_types non-negative counts
    #    by sampling (num_types-1) cut points in [0..num_boxes]
    cuts = sorted(random.sample(range(num_boxes+1), num_types-1))
    cuts = [0] + cuts + [num_boxes]
    counts = [cuts[i+1] - cuts[i] for i in range(num_types)]

    # 3) Write out CSV: each line is one box from one of the types
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for box_type, count in zip(types, counts):
            w,d,h = box_type
            for _ in range(count):
                writer.writerow([w, d, h])

    # 4) Report
    print(f"Wrote {num_boxes} boxes to '{path}' in {num_types} types:")
    for i, ((w,d,h), cnt) in enumerate(zip(types, counts), start=1):
        print(f"  Type #{i}: size={w}×{d}×{h}  →  count={cnt}")

def main() -> None:
    if len(sys.argv) not in (2,5):
        print("Usage:")
        print("  python gen_boxes.py out.csv")
        print("  python gen_boxes.py out.csv NUM MIN_SIZE MAX_SIZE")
        sys.exit(1)

    path = sys.argv[1]
    if len(sys.argv) == 2:
        num, mn, mx = 50, 1, 5
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
