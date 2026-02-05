import os
from collections import defaultdict

DATASET_ROOT = "dataset"

def count_images(base_path):
    stats = defaultdict(lambda: defaultdict(int))

    for domain in ["lab", "field"]:
        domain_path = os.path.join(base_path, domain)
        if not os.path.exists(domain_path):
            continue

        for split in ["train", "test"]:
            split_path = os.path.join(domain_path, split)
            if not os.path.exists(split_path):
                continue

            for cls in os.listdir(split_path):
                cls_path = os.path.join(split_path, cls)
                if not os.path.isdir(cls_path):
                    continue

                count = len([
                    f for f in os.listdir(cls_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])

                stats[domain][(split, cls)] += count

    return stats


def print_dataset_description(stats):
    print("\n========== DATASET DESCRIPTION ==========\n")

    total_images = 0

    for domain in stats:
        domain_total = sum(stats[domain].values())
        total_images += domain_total

        print(f"{domain.upper()} DATASET")
        print("-" * 40)

        split_totals = defaultdict(int)

        for (split, cls), count in sorted(stats[domain].items()):
            print(f"{split:5} | {cls:25} : {count}")
            split_totals[split] += count

        print("\nSplit Summary:")
        for split, count in split_totals.items():
            print(f"{split.capitalize():5} Images : {count}")

        print(f"Total {domain.capitalize()} Images : {domain_total}")
        print("\n")

    print("=" * 40)
    print(f"TOTAL IMAGES USED IN PROJECT : {total_images}")
    print("=" * 40)


if __name__ == "__main__":
    stats = count_images(DATASET_ROOT)
    print_dataset_description(stats)