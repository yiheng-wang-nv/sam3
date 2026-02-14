"""
Merge a new single-prompt SAM3 mask into existing multi-prompt masks.

The new prompt run (in new_dir) produces masks with label=1 (single prompt).
This script remaps that label to new_label (e.g., 5) and merges it into
the existing masks in existing_dir where the existing mask is background (0).

Usage:
    python merge_prompt_masks.py \
        --existing_dir /path/to/sam3_output \
        --new_dir /path/to/sam3_output_new_prompt \
        --new_label 5 \
        --cameras cam1 cam2 ... \
        --num_workers 32
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing


def merge_single(args_tuple):
    """Merge a single new-prompt mask into the existing mask file."""
    existing_path, new_path, new_label, skip_if_label_exists = args_tuple
    existing_path = Path(existing_path)
    new_path = Path(new_path)

    if not existing_path.exists():
        return existing_path.name, "no_existing"
    if not new_path.exists():
        return existing_path.name, "no_new"

    existing = np.load(existing_path)["arr_0"]  # (T, H, W)

    # Check if new_label already present -> skip
    if skip_if_label_exists and np.any(existing == new_label):
        return existing_path.name, "skipped"

    new = np.load(new_path)["arr_0"]  # (T, H, W)

    # Sanity check shapes
    if existing.shape != new.shape:
        return existing_path.name, f"shape_mismatch({existing.shape} vs {new.shape})"

    # New mask has label=1 for the single new prompt.
    # Merge: where existing==0 and new==1, set to new_label
    merge_mask = (existing == 0) & (new == 1)
    result = existing.copy()
    result[merge_mask] = new_label

    np.savez_compressed(existing_path, arr_0=result)
    return existing_path.name, "merged"


def main():
    parser = argparse.ArgumentParser(
        description="Merge new single-prompt masks into existing multi-prompt masks"
    )
    parser.add_argument(
        "--existing_dir",
        type=str,
        required=True,
        help="Existing sam3_output directory with *_masks.npz (labels 1..N)",
    )
    parser.add_argument(
        "--new_dir",
        type=str,
        required=True,
        help="New sam3_output directory with single-prompt *_masks.npz (label=1)",
    )
    parser.add_argument(
        "--new_label",
        type=int,
        required=True,
        help="Label value for the new prompt (e.g. 5)",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Camera folder names. If not specified, process all camera dirs.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of parallel workers (default: 32)",
    )
    parser.add_argument(
        "--skip_if_label_exists",
        action="store_true",
        default=True,
        help="Skip merging if new_label already exists in existing mask (default: True)",
    )
    parser.add_argument(
        "--no_skip_if_label_exists",
        action="store_true",
        help="Force merge even if new_label already exists (overwrites background only)",
    )

    args = parser.parse_args()

    existing_dir = Path(args.existing_dir)
    new_dir = Path(args.new_dir)
    skip_flag = not args.no_skip_if_label_exists

    if not existing_dir.exists():
        print(f"ERROR: existing_dir does not exist: {existing_dir}")
        return
    if not new_dir.exists():
        print(f"ERROR: new_dir does not exist: {new_dir}")
        return

    # Discover camera directories
    if args.cameras:
        camera_names = args.cameras
    else:
        camera_names = [
            d.name
            for d in existing_dir.iterdir()
            if d.is_dir() and "observation.images" in d.name
        ]

    if not camera_names:
        print(f"No camera directories found in {existing_dir}")
        return

    print("=" * 60)
    print("Merge New Prompt Masks")
    print("=" * 60)
    print(f"Existing dir: {existing_dir}")
    print(f"New dir:      {new_dir}")
    print(f"New label:    {args.new_label}")
    print(f"Cameras:      {camera_names}")
    print(f"Workers:      {args.num_workers}")
    print(f"Skip if exists: {skip_flag}")
    print("=" * 60)

    # Build task list
    tasks = []
    for cam in camera_names:
        existing_cam_dir = existing_dir / cam
        new_cam_dir = new_dir / cam
        if not existing_cam_dir.exists():
            print(f"  WARN: existing camera dir missing: {existing_cam_dir}")
            continue
        if not new_cam_dir.exists():
            print(f"  WARN: new camera dir missing: {new_cam_dir}")
            continue

        for existing_file in sorted(existing_cam_dir.glob("*_masks.npz")):
            # Skip _masks_post.npz
            if "_masks_post.npz" in existing_file.name:
                continue
            new_file = new_cam_dir / existing_file.name
            tasks.append(
                (str(existing_file), str(new_file), args.new_label, skip_flag)
            )

    if not tasks:
        print("No files to merge.")
        return

    print(f"Total files to merge: {len(tasks)}")

    merged = 0
    skipped = 0
    errors = 0

    if args.num_workers <= 1:
        for task in tqdm(tasks, desc="Merging"):
            name, status = merge_single(task)
            if status == "merged":
                merged += 1
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
                print(f"  {name}: {status}")
    else:
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            for name, status in tqdm(
                pool.imap_unordered(merge_single, tasks),
                total=len(tasks),
                desc="Merging",
            ):
                if status == "merged":
                    merged += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    errors += 1
                    print(f"  {name}: {status}")

    print("=" * 60)
    print(f"Done! merged={merged}, skipped={skipped}, errors={errors}")
    print("=" * 60)


if __name__ == "__main__":
    main()
