"""
Convert segmentation view videos (colored) to integer mask npz files.

Each stage may have different color mappings. Use --stage to specify which mapping to use.

Supported stages:
- stage1_3: 6 classes (black, blue, green, red, cyan, dark_red)
- stage1_7: Same as stage1_3
- stage1_5: 6 classes (black, blue, green, red, yellow_green, cyan)
- stage1_8: 6 classes (black, blue, green, red, yellow_green, dark_green)

Usage - Convert all stages:
    # Stage 1_3
    python convert_seg_video_to_mask.py \
        --dataset_root /localhome/local-vennw/code/orca-sim-pick-and-place-mimic/stage1_3_cosmos/lerobot \
        --stage stage1_3 \
        --output_dir /localhome/local-vennw/code/orca-sim-pick-and-place-mimic/stage1_3_cosmos/lerobot/masks/chunk-000/observation.images.ego_view \
        --overwrite --verify
    
    # Stage 1_5
    python convert_seg_video_to_mask.py \
        --dataset_root /localhome/local-vennw/code/orca-sim-pick-and-place-mimic/stage1_5_cosmos/lerobot \
        --stage stage1_5 \
        --output_dir /localhome/local-vennw/code/orca-sim-pick-and-place-mimic/stage1_5_cosmos/lerobot/masks/chunk-000/observation.images.ego_view \
        --overwrite --verify
    
    # Stage 1_7
    python convert_seg_video_to_mask.py \
        --dataset_root /localhome/local-vennw/code/orca-sim-pick-and-place-mimic/stage1_7_cosmos/lerobot \
        --stage stage1_7 \
        --output_dir /localhome/local-vennw/code/orca-sim-pick-and-place-mimic/stage1_7_cosmos/lerobot/masks/chunk-000/observation.images.ego_view \
        --overwrite --verify
    
    # Stage 1_8
    python convert_seg_video_to_mask.py \
        --dataset_root /localhome/local-vennw/code/orca-sim-pick-and-place-mimic/stage1_8_cosmos/lerobot \
        --stage stage1_8 \
        --output_dir /localhome/local-vennw/code/orca-sim-pick-and-place-mimic/stage1_8_cosmos/lerobot/masks/chunk-000/observation.images.ego_view \
        --overwrite --verify

Usage - Generate debug videos:
    python convert_seg_video_to_mask.py --debug_videos
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# =============================================================================
# Stage-specific color classifiers
# =============================================================================

def classify_stage1_3_7(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Classify pixels for stage1_3_cosmos and stage1_7_cosmos datasets.
    
    Verified color mapping (0% unmatched):
        0: black      - R<20, G<20, B<20 (ground/floor)
        1: blue       - B>200, R<50, G<50 (background)
        2: green      - G>200, R<50, B<50 (target object)
        3: red        - R>200, G<50, B<50 (unused in this stage)
        4: cyan       - G in [100,180], B>180, R<100 (table/surface)
        5: dark_red   - R>180, G<50, B in [30,80] (robot arm)
    
    Args:
        frame_rgb: (H, W, 3) RGB image
    
    Returns:
        labels: (H, W) uint8 array with labels 0-5
    """
    r = frame_rgb[:, :, 0].astype(np.int16)
    g = frame_rgb[:, :, 1].astype(np.int16)
    b = frame_rgb[:, :, 2].astype(np.int16)
    
    labels = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=np.uint8)
    
    # 0: black - all channels low
    labels[(r < 20) & (g < 20) & (b < 20)] = 0
    
    # 1: blue - B channel high, R and G low
    labels[(b > 200) & (r < 50) & (g < 50)] = 1
    
    # 2: green - G channel high, R and B low
    labels[(g > 200) & (r < 50) & (b < 50)] = 2
    
    # 3: red (pure) - R channel high, G and B low (may not appear in this stage)
    labels[(r > 200) & (g < 50) & (b < 50)] = 3
    
    # 4: cyan - G in mid-range, B high, R low
    labels[(g > 100) & (g < 180) & (b > 180) & (r < 100)] = 4
    
    # 5: dark_red (robot) - R high, G low, B in specific range
    labels[(r > 180) & (g < 50) & (b > 30) & (b < 80)] = 5
    
    return labels


def classify_stage1_8(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Classify pixels for stage1_8_cosmos dataset.
    
    Verified color mapping (0% unmatched):
        0: black           - R<20, G<20, B<20 (background)
        1: blue            - B>200, R<50, G<50 (box/container)
        2: green (cart)    - G>240, R<20, B<20 (cart/trolley)
        3: red             - R>200, G<50, B<50 (marker)
        4: yellow_green    - R>100, G>200, B<100 (shelf)
        5: dark_green      - G in [180,240], R<30, B<30 (floor)
    
    Args:
        frame_rgb: (H, W, 3) RGB image
    
    Returns:
        labels: (H, W) uint8 array with labels 0-5
    """
    r = frame_rgb[:, :, 0].astype(np.int16)
    g = frame_rgb[:, :, 1].astype(np.int16)
    b = frame_rgb[:, :, 2].astype(np.int16)
    
    labels = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=np.uint8)
    
    # 0: black - all channels low
    labels[(r < 20) & (g < 20) & (b < 20)] = 0
    
    # 1: blue - B channel high, R and G low
    labels[(b > 200) & (r < 50) & (g < 50)] = 1
    
    # 2: pure_green (cart) - G very high, R and B very low
    labels[(g > 240) & (r < 20) & (b < 20)] = 2
    
    # 3: red - R channel high, G and B low
    labels[(r > 200) & (g < 50) & (b < 50)] = 3
    
    # 4: yellow_green (shelf) - R>100, G high, B low
    labels[(r > 100) & (g > 200) & (b < 100)] = 4
    
    # 5: dark_green (floor) - G in mid-high range, R and B low but not pure green
    labels[(g > 180) & (g <= 240) & (r < 30) & (b < 30)] = 5
    
    return labels


def classify_stage1_5(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Classify pixels for stage1_5_cosmos dataset.
    
    Verified color mapping (0% unmatched):
        0: black        - R<20, G<20, B<20 (ground/floor)
        1: blue         - B>200, R<50, G<50 (background)
        2: green        - G>200, R<50, B<50 (target object)
        3: red          - R>200, G<50, B<50 (marker/indicator)
        4: yellow_green - R>100, G>200, B<100 (another target)
        5: cyan         - G in [100,180], B>180, R<100 (shelf)
    
    Args:
        frame_rgb: (H, W, 3) RGB image
    
    Returns:
        labels: (H, W) uint8 array with labels 0-5
    """
    r = frame_rgb[:, :, 0].astype(np.int16)
    g = frame_rgb[:, :, 1].astype(np.int16)
    b = frame_rgb[:, :, 2].astype(np.int16)
    
    labels = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=np.uint8)
    
    # 0: black - all channels low
    labels[(r < 20) & (g < 20) & (b < 20)] = 0
    
    # 1: blue - B channel high, R and G low
    labels[(b > 200) & (r < 50) & (g < 50)] = 1
    
    # 2: green - G channel high, R and B low
    labels[(g > 200) & (r < 50) & (b < 50)] = 2
    
    # 3: red (pure) - R channel high, G and B low
    labels[(r > 200) & (g < 50) & (b < 50)] = 3
    
    # 4: yellow_green - R>100, G high, B low
    labels[(r > 100) & (g > 200) & (b < 100)] = 4
    
    # 5: cyan (shelf) - G in mid-range, B high, R low
    labels[(g > 100) & (g < 180) & (b > 180) & (r < 100)] = 5
    
    return labels


# Registry of stage classifiers
STAGE_CLASSIFIERS = {
    'stage1_3': classify_stage1_3_7,
    'stage1_7': classify_stage1_3_7,  # Same color mapping as stage1_3
    'stage1_5': classify_stage1_5,
    'stage1_8': classify_stage1_8,
}

# Label names per stage
STAGE_LABELS = {
    'stage1_3': {
        0: 'black',
        1: 'blue',
        2: 'green',
        3: 'red',
        4: 'cyan',
        5: 'dark_red',
    },
    'stage1_7': {  # Same as stage1_3
        0: 'black',
        1: 'blue',
        2: 'green',
        3: 'red',
        4: 'cyan',
        5: 'dark_red',
    },
    'stage1_5': {
        0: 'black',
        1: 'blue',
        2: 'green',
        3: 'red',
        4: 'yellow_green',
        5: 'cyan',
    },
    'stage1_8': {
        0: 'black',
        1: 'blue',
        2: 'green',
        3: 'red',
        4: 'yellow_green',
        5: 'dark_green',
    },
}


# =============================================================================
# Core conversion functions
# =============================================================================

def convert_seg_video_to_mask(
    video_path: Path,
    output_path: Path,
    classifier_fn,
    label_names: dict,
) -> dict:
    """
    Convert a segmentation video to integer mask npz.
    
    Args:
        video_path: Path to input video
        output_path: Path to output npz file
        classifier_fn: Function to classify pixels
        label_names: Dict mapping label int to name
    
    Returns:
        stats: dict with label percentages
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Pre-allocate mask array
    masks = np.zeros((frame_count, height, width), dtype=np.uint8)
    
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        masks[frame_idx] = classifier_fn(frame_rgb)
    
    cap.release()
    
    # Save with key 'arr_0' for GR00T compatibility
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, arr_0=masks)
    
    # Compute stats
    stats = {}
    total = masks.size
    for label, name in label_names.items():
        count = (masks == label).sum()
        stats[name] = count / total * 100
    
    return stats


def process_dataset(
    dataset_root: Path,
    stage: str,
    output_dir: Path = None,
    overwrite: bool = False,
) -> None:
    """Process all episodes in a dataset."""
    
    if stage not in STAGE_CLASSIFIERS:
        raise ValueError(f"Unknown stage: {stage}. Available: {list(STAGE_CLASSIFIERS.keys())}")
    
    classifier_fn = STAGE_CLASSIFIERS[stage]
    label_names = STAGE_LABELS.get(stage, {})
    
    seg_view_dir = dataset_root / "videos" / "chunk-000" / "observation.images.seg_view"
    
    if not seg_view_dir.exists():
        print(f"Warning: seg_view directory not found: {seg_view_dir}")
        return
    
    if output_dir is None:
        output_dir = dataset_root / "masks" / "chunk-000" / "ego_view"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_files = sorted(seg_view_dir.glob("episode_*.mp4"))
    if not video_files:
        print(f"No episode videos found in {seg_view_dir}")
        return
    
    print(f"\nProcessing {len(video_files)} episodes from {dataset_root.name}")
    print(f"Stage: {stage}")
    print(f"Output: {output_dir}")
    
    stats = None
    for video_path in tqdm(video_files, desc=dataset_root.name):
        episode_id = video_path.stem  # episode_000000
        output_path = output_dir / f"{episode_id}_masks.npz"
        
        if output_path.exists() and not overwrite:
            continue
        
        try:
            stats = convert_seg_video_to_mask(
                video_path, output_path, classifier_fn, label_names
            )
        except Exception as e:
            print(f"\nError processing {video_path}: {e}")
            continue
    
    # Print stats for last episode
    if stats:
        stats_str = ", ".join([f"{k}={v:.1f}%" for k, v in stats.items()])
        print(f"  Sample stats (last episode): {stats_str}")


def verify_mask(npz_path: Path, stage: str) -> None:
    """Verify a generated mask file."""
    data = np.load(npz_path)
    arr = data['arr_0']
    
    label_names = STAGE_LABELS.get(stage, {})
    
    print(f"\nVerifying {npz_path.name}:")
    print(f"  Shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"  Unique values: {np.unique(arr)}")
    
    total = arr.size
    for label in sorted(label_names.keys()):
        name = label_names[label]
        count = (arr == label).sum()
        pct = count / total * 100
        print(f"  {name} (label {label}): {pct:.2f}%")


# Display colors for debug visualization (consistent across all stages, high contrast)
DISPLAY_COLORS = {
    0: (50, 50, 50),      # label 0: dark gray (easy to see against black bg)
    1: (0, 100, 255),     # label 1: bright blue
    2: (0, 255, 100),     # label 2: bright green
    3: (255, 50, 50),     # label 3: bright red
    4: (255, 255, 0),     # label 4: yellow (high contrast)
    5: (255, 0, 255),     # label 5: magenta (high contrast)
}


def create_debug_video(
    video_path: Path,
    output_path: Path,
    classifier_fn,
    fps: int = 10,
) -> None:
    """
    Create a side-by-side debug video: original | colored mask.
    
    Args:
        video_path: Path to input seg_view video
        output_path: Path to output debug video
        classifier_fn: Function to classify pixels
        fps: Output video fps (default 10 for smaller file)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output is side-by-side (2x width)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width * 2, height))
    
    # Sample every N frames to reduce file size
    sample_rate = max(1, int(cap.get(cv2.CAP_PROP_FPS) / fps))
    
    for frame_idx in range(0, frame_count, sample_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        labels = classifier_fn(frame_rgb)
        
        # Create colored mask
        colored_mask = np.zeros_like(frame_rgb)
        for label, color in DISPLAY_COLORS.items():
            colored_mask[labels == label] = color
        
        # Side-by-side: original | mask
        combined = np.hstack([frame_rgb, colored_mask])
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        out.write(combined_bgr)
    
    cap.release()
    out.release()
    print(f"  Saved: {output_path}")


def generate_debug_videos(
    base_dir: Path,
    output_dir: Path,
    stages: list[str] = None,
) -> None:
    """
    Generate debug videos for random samples from each stage.
    
    Args:
        base_dir: Base directory containing stage folders
        output_dir: Output directory for debug videos
        stages: List of stages to process (default: all)
    """
    if stages is None:
        stages = list(STAGE_CLASSIFIERS.keys())
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import random
    
    for stage in stages:
        stage_dir = base_dir / f"{stage}_cosmos" / "lerobot" / "videos" / "chunk-000" / "observation.images.seg_view"
        
        if not stage_dir.exists():
            print(f"Skipping {stage}: directory not found")
            continue
        
        videos = list(stage_dir.glob("episode_*.mp4"))
        if not videos:
            print(f"Skipping {stage}: no videos found")
            continue
        
        # Random sample
        video_path = random.choice(videos)
        output_path = output_dir / f"{stage}_{video_path.stem}_debug.mp4"
        
        print(f"\n{stage}: {video_path.name}")
        
        classifier_fn = STAGE_CLASSIFIERS[stage]
        create_debug_video(video_path, output_path, classifier_fn)
    
    print(f"\nDebug videos saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Convert seg_view videos to mask npz')
    parser.add_argument('--dataset_root', type=str, nargs='+', default=None,
                        help='Dataset root path(s)')
    parser.add_argument('--stage', type=str, default=None,
                        choices=list(STAGE_CLASSIFIERS.keys()),
                        help='Stage name for color mapping')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: <dataset_root>/masks/chunk-000/ego_view)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing mask files')
    parser.add_argument('--verify', action='store_true',
                        help='Verify first generated mask file')
    parser.add_argument('--debug_videos', action='store_true',
                        help='Generate debug videos (side-by-side original + mask)')
    parser.add_argument('--debug_output', type=str, default=None,
                        help='Output directory for debug videos')
    parser.add_argument('--debug_base_dir', type=str, 
                        default='/localhome/local-vennw/code/orca-sim-pick-and-place-mimic',
                        help='Base directory for debug video generation')
    
    args = parser.parse_args()
    
    # Handle debug video mode
    if args.debug_videos:
        base_dir = Path(args.debug_base_dir)
        output_dir = Path(args.debug_output) if args.debug_output else base_dir / "debug_videos"
        generate_debug_videos(base_dir, output_dir)
        return
    
    # Validate required args for conversion mode
    if not args.dataset_root or not args.stage:
        parser.error("--dataset_root and --stage are required for conversion mode")

    print("=" * 60)
    print("Convert Segmentation Videos to Mask NPZ")
    print("=" * 60)
    print(f"Stage: {args.stage}")
    print(f"Label mapping:")
    for label, name in STAGE_LABELS.get(args.stage, {}).items():
        print(f"  {label} = {name}")
    print("=" * 60)
    
    for dataset_root_str in args.dataset_root:
        dataset_root = Path(dataset_root_str)
        output_dir = Path(args.output_dir) if args.output_dir else None
        
        process_dataset(
            dataset_root=dataset_root,
            stage=args.stage,
            output_dir=output_dir,
            overwrite=args.overwrite,
        )
        
        if args.verify:
            # Verify first mask file
            if output_dir is None:
                output_dir = dataset_root / "masks" / "chunk-000" / "ego_view"
            first_mask = sorted(output_dir.glob("*_masks.npz"))
            if first_mask:
                verify_mask(first_mask[0], args.stage)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
