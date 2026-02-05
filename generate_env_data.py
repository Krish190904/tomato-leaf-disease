import os
import csv
import random

DATASET_ROOT = "dataset"
OUTPUT_CSV = "env_data.csv"

# Disease → Environmental ranges
ENV_RANGES = {
    "Early_blight": (24, 30, 60, 80, 2, 8),
    "Late_blight": (18, 24, 80, 95, 8, 20),
    "Leaf_Mold": (20, 26, 75, 90, 5, 15),
    "Septoria_leaf_spot": (20, 28, 70, 90, 5, 15),
    "Yellow_Leaf_Curl_Virus": (28, 35, 40, 60, 0, 5)
}

rows = []

# We explicitly define the splits we use
splits = [
    "lab/train",
    "field/train",
    "field/test"
]

for split in splits:
    split_path = os.path.join(DATASET_ROOT, split)

    for cls in os.listdir(split_path):
        cls_path = os.path.join(split_path, cls)

        if cls not in ENV_RANGES or not os.path.isdir(cls_path):
            continue

        tmin, tmax, hmin, hmax, rmin, rmax = ENV_RANGES[cls]

        for img in os.listdir(cls_path):
            if not img.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            # Store RELATIVE PATH (THIS IS THE KEY FIX)
            relative_path = os.path.join(split, cls, img)
            relative_path = relative_path.replace("\\", "/")

            rows.append([
                relative_path,
                round(random.uniform(tmin, tmax), 2),
                round(random.uniform(hmin, hmax), 2),
                round(random.uniform(rmin, rmax), 2)
            ])

# Write CSV safely (UTF-8)
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "temperature", "humidity", "rainfall"])
    writer.writerows(rows)

print(f"Environment data generated successfully → {OUTPUT_CSV}")
