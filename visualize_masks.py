"""
Visualize original frames vs masks for multiple cameras.

Example:
python /localhome/local-vennw/code/sam3/visualize_masks.py \
  --dataset_root /localhome/local-vennw/code/orca_dataset/galbot_lerobot_dataset/task3_01210122_merged \
  --mask_root /localhome/local-vennw/code/orca_dataset/galbot_lerobot_dataset/task3_01210122_merged/sam3_output \
  --use_post_masks \
  --frames 0,10,20 \
  --cameras observation.images.head_left_camera_color_optical_frame,observation.images.right_arm_camera_color_optical_frame \
  --output_dir /localhome/local-vennw/code/orca_dataset/galbot_lerobot_dataset/task3_01210122_merged/viz_masks
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def _parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _load_video_frames(video_path: Path, frame_indices: list[int]) -> dict[int, np.ndarray]:
    if not frame_indices:
        return {}
    frame_indices = sorted(set(frame_indices))
    max_index = frame_indices[-1]
    frames = {}
    reader = imageio.get_reader(str(video_path))
    try:
        for idx, frame in enumerate(reader):
            if idx in frame_indices:
                frames[idx] = frame
                if len(frames) == len(frame_indices):
                    break
            if idx > max_index:
                break
    finally:
        reader.close()
    return frames


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    palette = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (255, 255, 0),
    }
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in palette.items():
        colored[mask == label] = color
    return colored


def _find_chunk_for_episode(dataset_root: Path, camera: str, episode_id: str) -> str | None:
    for chunk_dir in sorted((dataset_root / "videos").glob("chunk-*")):
        candidate = chunk_dir / camera / f"episode_{episode_id}.mp4"
        if candidate.exists():
            return chunk_dir.name
    return None


def visualize_masks(
    dataset_root: Path,
    mask_root: Path,
    output_dir: Path,
    cameras: list[str],
    frame_indices: list[int],
    episode_id: str | None = None,
    use_post_masks: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_dirs = [d for d in mask_root.iterdir() if d.is_dir() and "observation.images" in d.name]
    if cameras:
        camera_dirs = [d for d in camera_dirs if d.name in cameras]
    if not camera_dirs:
        raise ValueError(f"No camera directories found in {mask_root}")

    for camera_dir in sorted(camera_dirs):
        mask_pattern = "*_masks_post.npz" if use_post_masks else "*_masks.npz"
        mask_files = sorted(camera_dir.glob(mask_pattern))
        if not mask_files:
            continue
        if episode_id:
            mask_files = [m for m in mask_files if f"episode_{episode_id}" in m.name]
            if not mask_files:
                continue
        mask_file = mask_files[0]
        match = re.match(r"episode_(\d+)_masks", mask_file.name)
        if not match:
            continue
        current_episode = match.group(1)
        chunk = _find_chunk_for_episode(dataset_root, camera_dir.name, current_episode)
        if chunk is None:
            chunk = "chunk-000"
        video_path = (
            dataset_root
            / "videos"
            / chunk
            / camera_dir.name
            / f"episode_{current_episode}.mp4"
        )
        if not video_path.exists():
            continue

        masks = np.load(mask_file)["arr_0"]
        frames = _load_video_frames(video_path, frame_indices)

        for frame_idx, frame in frames.items():
            if frame_idx >= masks.shape[0]:
                continue
            mask = masks[frame_idx]
            colored_mask = _colorize_mask(mask)
            overlay = (0.6 * frame + 0.4 * colored_mask).astype(np.uint8)

            canvas = np.concatenate([frame, colored_mask, overlay], axis=1)
            out_name = f"{camera_dir.name}_episode_{current_episode}_frame_{frame_idx}.png"
            Image.fromarray(canvas).save(output_dir / out_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize original frames vs masks.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root with videos/")
    parser.add_argument("--mask_root", type=str, required=True, help="Mask root with camera subdirs")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for images")
    parser.add_argument("--cameras", type=str, default=None, help="Comma-separated camera names")
    parser.add_argument("--frames", type=str, default="0,10,20", help="Comma-separated frame indices")
    parser.add_argument("--episode_id", type=str, default=None, help="Episode id, e.g. 000000")
    parser.add_argument("--use_post_masks", action="store_true", help="Use *_masks_post.npz")
    args = parser.parse_args()

    cameras = [c.strip() for c in args.cameras.split(",")] if args.cameras else []
    frame_indices = _parse_int_list(args.frames)

    visualize_masks(
        dataset_root=Path(args.dataset_root),
        mask_root=Path(args.mask_root),
        output_dir=Path(args.output_dir),
        cameras=cameras,
        frame_indices=frame_indices,
        episode_id=args.episode_id,
        use_post_masks=args.use_post_masks,
    )


if __name__ == "__main__":
    main()
