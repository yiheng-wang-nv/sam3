"""
Postprocess segmentation masks to fill small holes and remove noise.

Usage:
python /localhome/local-vennw/code/sam3/postprocess_masks.py \
  --input_dir /localhome/local-vennw/code/orca_dataset/galbot_lerobot_dataset/task3_01210122_merged/sam3_output \
  --fill_interior_class 1,3 \
  --fill_interior_target 4 \
  --overwrite \
  --copy_to_dataset_root /localhome/local-vennw/code/orca-template1-dev/task3_01210122_merged_with_mask

"""

import argparse
import numpy as np
import re
import shutil
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
from skimage.morphology import remove_small_objects, remove_small_holes


def postprocess_mask_2d(mask, 
                        fill_holes=True,
                        min_hole_size=64,
                        min_object_size=50,
                        closing_iterations=1):
    """
    Postprocess a single 2D binary mask.
    
    Args:
        mask: 2D binary mask (H, W)
        fill_holes: Whether to fill holes
        min_hole_size: Fill holes smaller than this (pixels)
        min_object_size: Remove objects smaller than this (pixels)
        closing_iterations: Morphological closing iterations
    
    Returns:
        Processed binary mask
    """
    if mask.sum() == 0:
        return mask
    
    result = mask.astype(bool)
    
    # 1. Morphological closing (fill small gaps)
    if closing_iterations > 0:
        struct = ndimage.generate_binary_structure(2, 1)  # 2D 4-connectivity
        result = ndimage.binary_dilation(result, struct, iterations=closing_iterations)
        result = ndimage.binary_erosion(result, struct, iterations=closing_iterations)
    
    # 2. Fill small holes
    if fill_holes and min_hole_size > 0:
        result = remove_small_holes(result, area_threshold=min_hole_size)
    
    # 3. Remove small noise objects
    if min_object_size > 0:
        result = remove_small_objects(result, min_size=min_object_size)
    
    return result.astype(np.uint8)


def fill_interior_with_class(mask, target_classes, fill_class=4):
    """
    Fill background (0) inside the contour of target_classes with fill_class.
    
    Rule: Within the outermost boundary of any target_class region,
          all background pixels (0) become fill_class.
    
    Args:
        mask: Multi-class mask (H, W) with labels 0,1,2,3,...
        target_classes: List of classes whose interior to fill (e.g., [1, 3] for red and blue)
                       Can also be a single int for backwards compatibility.
        fill_class: The new class label for interior background (default: 4)
    
    Returns:
        Modified mask with interior filled
    """
    # Handle single int input
    if isinstance(target_classes, int):
        target_classes = [target_classes]
    
    result = mask.copy()
    
    for target_class in target_classes:
        binary = (mask == target_class)
        
        if binary.sum() == 0:
            continue
        
        # Fill holes to get the interior region (everything inside the contour)
        filled = ndimage.binary_fill_holes(binary)
        
        # Within the filled region, change background (0) to fill_class
        interior_background = filled & (result == 0)
        result[interior_background] = fill_class
    
    return result


def postprocess_video_masks(masks, 
                            num_classes=4,
                            fill_holes=True,
                            min_hole_size=64,
                            min_object_size=50,
                            closing_iterations=1,
                            fill_interior_class=None,
                            fill_interior_target=4):
    """
    Postprocess masks for all frames in a video.
    
    Args:
        masks: (T, H, W) array with class labels 0,1,2,...
        num_classes: Number of classes (including background 0)
        fill_holes, min_hole_size, min_object_size, closing_iterations: 
            same as postprocess_mask_2d
        fill_interior_class: If set, fill background inside this class's contour
                            (e.g., 1 for red). None to disable.
        fill_interior_target: The new class label for filled interior (default: 4)
    
    Returns:
        Processed masks (T, H, W)
    """
    T, H, W = masks.shape
    result = np.zeros_like(masks)
    
    for t in range(T):
        frame_mask = masks[t]
        processed_frame = np.zeros((H, W), dtype=np.uint8)
        
        # Process each class separately (skip background 0)
        for cls in range(1, num_classes):
            binary_mask = (frame_mask == cls).astype(np.uint8)
            
            if binary_mask.sum() > 0:
                processed = postprocess_mask_2d(
                    binary_mask,
                    fill_holes=fill_holes,
                    min_hole_size=min_hole_size,
                    min_object_size=min_object_size,
                    closing_iterations=closing_iterations,
                )
                # Only write where currently background (avoid overwriting)
                processed_frame = np.where(
                    (processed > 0) & (processed_frame == 0),
                    cls,
                    processed_frame
                )
        
        # Apply interior filling rule: background inside target_class -> fill_class
        if fill_interior_class is not None:
            processed_frame = fill_interior_with_class(
                processed_frame, 
                target_classes=fill_interior_class, 
                fill_class=fill_interior_target
            )
        
        result[t] = processed_frame
    
    return result


def process_directory(input_dir: Path,
                      num_classes=4,
                      fill_holes=True,
                      min_hole_size=64,
                      min_object_size=50,
                      closing_iterations=1,
                      fill_interior_class=None,
                      fill_interior_target=4,
                      overwrite=False):
    """
    Process all *_masks.npz files in a directory.
    Output: *_masks_post.npz
    """
    mask_files = sorted(input_dir.glob("*_masks.npz"))
    
    if not mask_files:
        print(f"No mask files found in {input_dir}")
        return
    
    print(f"Processing {len(mask_files)} mask files in {input_dir.name}...")
    
    for mask_file in tqdm(mask_files, desc=input_dir.name):
        # Output path: episode_000000_masks.npz -> episode_000000_masks_post.npz
        out_path = mask_file.parent / mask_file.name.replace("_masks.npz", "_masks_post.npz")
        
        if out_path.exists() and not overwrite:
            continue
        
        # Load masks
        data = np.load(mask_file)
        masks = data['arr_0']  # (T, H, W)
        
        # Detect number of classes
        detected_classes = len(np.unique(masks))
        use_num_classes = max(num_classes, detected_classes)
        
        # Process
        processed = postprocess_video_masks(
            masks,
            num_classes=use_num_classes,
            fill_holes=fill_holes,
            min_hole_size=min_hole_size,
            min_object_size=min_object_size,
            closing_iterations=closing_iterations,
            fill_interior_class=fill_interior_class,
            fill_interior_target=fill_interior_target,
        )
        
        # Save
        np.savez_compressed(out_path, processed)
    
    print(f"Done! Processed masks saved as *_masks_post.npz")


def _build_episode_to_chunk_map(dataset_root: Path) -> dict[str, str]:
    data_root = dataset_root / "data"
    if not data_root.exists():
        return {}
    mapping: dict[str, str] = {}
    for chunk_dir in sorted(data_root.glob("chunk-*")):
        for parquet_file in chunk_dir.glob("episode_*.parquet"):
            episode_id = parquet_file.stem.replace("episode_", "")
            if episode_id in mapping and mapping[episode_id] != chunk_dir.name:
                raise ValueError(
                    f"Episode {episode_id} appears in multiple chunks: "
                    f"{mapping[episode_id]} and {chunk_dir.name}"
                )
            mapping[episode_id] = chunk_dir.name
    return mapping


def copy_postprocessed_masks(
    sam3_output_dir: Path,
    dataset_root: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Copy *_masks_post.npz from sam3_output into dataset_root/masks/<chunk>/<camera>/.
    Renames *_masks_post.npz -> *_masks.npz.
    """
    if not sam3_output_dir.exists():
        raise FileNotFoundError(f"Missing sam3_output: {sam3_output_dir}")
    masks_root = dataset_root / "masks"
    masks_root.mkdir(parents=True, exist_ok=True)

    episode_to_chunk = _build_episode_to_chunk_map(dataset_root)
    camera_dirs = [d for d in sam3_output_dir.iterdir() if d.is_dir() and "observation.images" in d.name]
    if not camera_dirs:
        raise ValueError(f"No camera directories found in {sam3_output_dir}")

    copied = 0
    skipped = 0
    missing_episode = 0
    pattern = re.compile(r"episode_(\d+)_masks_post\.npz$")

    for camera_dir in sorted(camera_dirs):
        for mask_file in sorted(camera_dir.glob("*_masks_post.npz")):
            match = pattern.match(mask_file.name)
            if not match:
                continue
            episode_id = match.group(1)
            chunk = episode_to_chunk.get(episode_id)
            if chunk is None and episode_to_chunk:
                missing_episode += 1
                continue
            if chunk is None:
                chunk = "chunk-000"
            dest_dir = masks_root / chunk / camera_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_name = mask_file.name.replace("_masks_post.npz", "_masks.npz")
            dest_path = dest_dir / dest_name
            if dest_path.exists() and not overwrite:
                skipped += 1
                continue
            if not dry_run:
                shutil.copy2(mask_file, dest_path)
            copied += 1

    print("=" * 50)
    print("Copy Postprocessed Masks")
    print("=" * 50)
    print(f"sam3_output: {sam3_output_dir}")
    print(f"dataset_root: {dataset_root}")
    print(f"dry_run: {dry_run}")
    print(f"copied: {copied}")
    print(f"skipped: {skipped}")
    print(f"missing_episode: {missing_episode}")
    print("=" * 50)


def copy_dataset_folders(
    src_root: Path,
    dest_root: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    folders = ["data", "meta", "videos"]
    print("=" * 50)
    print("Copy Dataset Folders")
    print("=" * 50)
    print(f"source_root: {src_root}")
    print(f"dest_root: {dest_root}")
    print(f"dry_run: {dry_run}")
    for folder in folders:
        src_dir = src_root / folder
        dest_dir = dest_root / folder
        if not src_dir.exists():
            print(f"skip (missing): {src_dir}")
            continue
        if dest_dir.exists() and not overwrite:
            print(f"skip (exists): {dest_dir}")
            continue
        if not dry_run:
            if dest_dir.exists() and overwrite:
                shutil.rmtree(dest_dir)
            shutil.copytree(src_dir, dest_dir, dirs_exist_ok=overwrite)
        print(f"copied: {folder}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='Postprocess segmentation masks')
    parser.add_argument('--input_dir', type=str, 
                        default='/localhome/local-vennw/code/orca_dataset/galbot_lerobot_dataset/task3_01210122_merged/sam3_output',
                        help='Input directory containing camera subdirectories')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of classes including background (default: 4)')
    parser.add_argument('--min_hole_size', type=int, default=0,
                        help='Fill holes smaller than this (pixels, default: 64)')
    parser.add_argument('--min_object_size', type=int, default=0,
                        help='Remove objects smaller than this (pixels, default: 50)')
    parser.add_argument('--closing_iterations', type=int, default=1,
                        help='Morphological closing iterations (default: 1)')
    parser.add_argument('--no_fill_holes', action='store_true',
                        help='Disable hole filling')
    parser.add_argument('--fill_interior_class', type=str, default=None,
                        help='Fill background inside these class contours. Comma-separated (e.g., "1,3" for red and blue). None to disable.')
    parser.add_argument('--fill_interior_target', type=int, default=4,
                        help='New class label for filled interior (default: 4)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing *_masks_post.npz files')
    parser.add_argument('--copy_to_dataset_root', type=str, default=None,
                        help='After postprocess, copy *_masks_post.npz to <dataset_root>/masks/')
    parser.add_argument('--copy_only', action='store_true',
                        help='Only copy postprocessed masks, skip postprocess')
    parser.add_argument('--copy_dataset_folders', action='store_true',
                        help='Copy data/meta/videos from dataset root (input_dir parent)')
    parser.add_argument('--source_dataset_root', type=str, default=None,
                        help='Explicit source dataset root for data/meta/videos')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print copy summary without copying files')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    # Parse fill_interior_class (comma-separated string to list of ints)
    fill_interior_classes = None
    if args.fill_interior_class is not None:
        fill_interior_classes = [int(x.strip()) for x in args.fill_interior_class.split(',')]
    
    if not args.copy_only:
        print("="*50)
        print("Mask Postprocessing")
        print("="*50)
        print(f"Input: {input_dir}")
        print(f"Parameters:")
        print(f"  - fill_holes: {not args.no_fill_holes}")
        print(f"  - min_hole_size: {args.min_hole_size}")
        print(f"  - min_object_size: {args.min_object_size}")
        print(f"  - closing_iterations: {args.closing_iterations}")
        if fill_interior_classes is not None:
            class_names = {1: 'red', 2: 'green', 3: 'blue'}
            class_str = ', '.join([f"{c}({class_names.get(c, '?')})" for c in fill_interior_classes])
            print(f"  - fill_interior: classes [{class_str}] interior -> {args.fill_interior_target}")
        print("="*50)

        # Find all camera subdirectories
        camera_dirs = [d for d in input_dir.iterdir() if d.is_dir() and 'observation.images' in d.name]

        if not camera_dirs:
            print(f"No camera directories found in {input_dir}")
            return

        print(f"Found {len(camera_dirs)} camera directories")

        for camera_dir in camera_dirs:
            process_directory(
                camera_dir,
                num_classes=args.num_classes,
                fill_holes=not args.no_fill_holes,
                min_hole_size=args.min_hole_size,
                min_object_size=args.min_object_size,
                closing_iterations=args.closing_iterations,
                fill_interior_class=fill_interior_classes,
                fill_interior_target=args.fill_interior_target,
                overwrite=args.overwrite,
            )

        print("\nAll done!")

    if args.copy_to_dataset_root:
        copy_postprocessed_masks(
            sam3_output_dir=input_dir,
            dataset_root=Path(args.copy_to_dataset_root),
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        if args.copy_dataset_folders:
            source_root = Path(args.source_dataset_root) if args.source_dataset_root else input_dir.parent
            copy_dataset_folders(
                src_root=source_root,
                dest_root=Path(args.copy_to_dataset_root),
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()

