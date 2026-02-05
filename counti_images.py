import os

LAB_PATH = "dataset/lab/train"

def count_lab_images():
    print("\nCounting images in lab/train:\n")

    for class_name in sorted(os.listdir(LAB_PATH)):
        class_path = os.path.join(LAB_PATH, class_name)
        if os.path.isdir(class_path):
            count = len([
                f for f in os.listdir(class_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            print(f"{class_name}: {count}")

count_lab_images()