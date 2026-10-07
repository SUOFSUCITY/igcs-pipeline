"""Generate data splits from image filenames."""

import argparse
import json
from pathlib import Path


def generate_splits(images_dir, output_path, num_folds, seed):
    from sklearn.model_selection import KFold

    images_dir = Path(images_dir)
    output_path = Path(output_path)
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)
    suffix = "_0000.nii.gz"
    case_ids = sorted(
        p.name[: -len(suffix)] for p in images_dir.glob("*" + suffix) if p.is_file()
    )
    if not 2 <= num_folds <= len(case_ids):
        raise ValueError("Fold count must be between two and the number of cases.")
    splitter = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = [
        {"train": [case_ids[i] for i in train], "val": [case_ids[i] for i in val]}
        for train, val in splitter.split(case_ids)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-folds", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    splits = generate_splits(args.images_dir, args.output, args.num_folds, args.seed)
    print(f"Saved {len(splits)} folds.")


if __name__ == "__main__":
    main()
