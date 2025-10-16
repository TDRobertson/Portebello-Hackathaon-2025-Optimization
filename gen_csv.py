import csv
import random
import sys

def generate_csv(path: str,
                 num_boxes: int = 50,
                 min_size: int = 1,
                 max_size: int = 5) -> None:
    """
    Generates a CSV at `path` with `num_boxes` lines.
    Each line has three integers w,d,h in [min_size..max_size].
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for _ in range(num_boxes):
            w = random.randint(min_size, max_size)
            d = random.randint(min_size, max_size)
            h = random.randint(min_size, max_size)
            writer.writerow([w, d, h])
    print(f"Wrote {num_boxes} boxes to '{path}'")

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

    random.seed()  # or fixed seed for reproducibility
    generate_csv(path, num, mn, mx)

if __name__ == "__main__":
    main()
