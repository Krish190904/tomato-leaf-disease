import os
import random
import shutil

LAB_PATH = "dataset/lab/train"
TARGET_COUNTS = {
    "Early_blight": 176,
    "Late_blight": 222,
    "Leaf_Mold": 182,
    "Septoria_leaf_spot": 300,
    "Yellow_Leaf_Curl_Virus": 152
}

for class_name, target in TARGET_COUNTS.items():
    class_path = os.path.join(LAB_PATH, class_name)
    images = [f for f in os.listdir(class_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"\n{class_name}: {len(images)} images found")

    if len(images) > target:
        remove = random.sample(images, len(images) - target)
        for img in remove:
            os.remove(os.path.join(class_path, img))
        print(f"  → Reduced to {target}")
    else:
        print("  → No change needed")