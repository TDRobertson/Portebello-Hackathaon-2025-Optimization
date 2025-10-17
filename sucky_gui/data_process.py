import pandas as pd
import json

# Load your Excel file
# (change "tiles.xlsx" to your actual file name)
df = pd.read_excel("sku_box_index.xlsx")

# Function to convert box size like "11x6.5x3.5" → [11, 6.5, 3.5]
def parse_box_size(size_str):
    return [float(x) for x in str(size_str).lower().split('x')]

# Build dictionary: SKU → [Box Size as numbers]
data = {
    row["SKU"]: parse_box_size(row["Box Size"])
    for _, row in df.iterrows()
}

# Save result to JSON file
with open("sku_box_index.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ JSON file created successfully!")
