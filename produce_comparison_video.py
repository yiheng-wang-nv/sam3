#!/usr/bin/env python3
"""
Generate 3-panel comparison video: original | raw mask overlay | post-processed mask overlay.

Reads pre-existing _masks.npz and _masks_post.npz (no postprocessing is run).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from sam3.visualization_utils import load_frame, render_masklet_frame


def label_masks_to_outputs(label_masks):
    """Convert (T,H,W) label masks to per-frame output dicts for rendering."""
    outputs = {}
    if label_masks.size == 0:
        return outputs
    t_count, h, w = label_masks.shape
    for t in range(t_count):
        frame_mask = label_masks[t]
        labels = np.unique(frame_mask)
        labels = labels[labels > 0]
        if labels.size == 0:
            outputs[t] = {
                "out_obj_ids": np.array([], dtype=np.int64),
                "out_probs": np.array([], dtype=np.float32),
                "out_boxes_xywh": np.zeros((0, 4), dtype=np.float32),
                "out_binary_masks": np.zeros((0, h, w), dtype=bool),
            }
            continue
        masks = []
        obj_ids = []
        for label in labels:
            masks.append(frame_mask == label)
            obj_ids.append(int(label) - 1)
        outputs[t] = {
            "out_obj_ids": np.array(obj_ids, dtype=np.int64),
            "out_probs": np.ones(len(obj_ids), dtype=np.float32),
            "out_boxes_xywh": np.zeros((len(obj_ids), 4), dtype=np.float32),
            "out_binary_masks": np.stack(masks, axis=0),
        }
    return outputs


def render_overlay(frame, output):
    """Render mask overlay on frame, with black background where no mask."""
    overlay = render_masklet_frame(frame, output, frame_idx=None)
    masks = output["out_binary_masks"]
    if masks.size > 0:
        combined = np.any(masks, axis=0)
    else:
        combined = np.zeros(frame.shape[:2], dtype=bool)
    overlay[~combined] = 0
    return overlay


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3-panel comparison video: original | raw mask | post mask."
    )
    parser.add_argument("--mask_dir", required=True, help="Directory containing *_masks.npz and *_masks_post.npz")
    parser.add_argument("--video_path", required=True, help="Path to original video .mp4")
    parser.add_argument("--episode", required=True, help="Episode id, e.g. episode_000075")
    parser.add_argument("--output_path", default=None, help="Output video path (default: <mask_dir>/<episode>_comparison.mp4)")
    parser.add_argument("--fps", type=int, default=30, help="Output video fps")
    args = parser.parse_args()

    mask_dir = Path(args.mask_dir)
    raw_path = mask_dir / f"{args.episode}_masks.npz"
    post_path = mask_dir / f"{args.episode}_masks_post.npz"
    video_path = Path(args.video_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw mask not found: {raw_path}")
    if not post_path.exists():
        raise FileNotFoundError(f"Post mask not found: {post_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    raw_masks = np.load(raw_path)["arr_0"]
    post_masks = np.load(post_path)["arr_0"]

    raw_outputs = label_masks_to_outputs(raw_masks)
    post_outputs = label_masks_to_outputs(post_masks)

    # Read video frames
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")

    height, width = frames[0].shape[:2]
    output_path = Path(args.output_path) if args.output_path else mask_dir / f"{args.episode}_comparison.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (width * 3, height))

    n_frames = min(len(frames), len(raw_outputs), len(post_outputs))
    for t in range(n_frames):
        frame = load_frame(frames[t])
        raw_overlay = render_overlay(frame, raw_outputs.get(t, raw_outputs.get(0)))
        post_overlay = render_overlay(frame, post_outputs.get(t, post_outputs.get(0)))
        panel = np.concatenate([frame, raw_overlay, post_overlay], axis=1)
        writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Saved: {output_path} ({n_frames} frames)")


if __name__ == "__main__":
    main()
