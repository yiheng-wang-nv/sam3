#!/usr/bin/env python3
"""
Generate 2-panel comparison video: original | mask overlay.

Reads a single _masks.npz file and the corresponding video.
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
        description="Generate 2-panel comparison video: original | mask overlay."
    )
    parser.add_argument("--mask_path", required=True, help="Path to _masks.npz file")
    parser.add_argument("--video_path", required=True, help="Path to original video .mp4")
    parser.add_argument("--output_path", default=None, help="Output video path")
    parser.add_argument("--fps", type=int, default=30, help="Output video fps")
    args = parser.parse_args()

    mask_path = Path(args.mask_path)
    video_path = Path(args.video_path)

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    masks = np.load(mask_path)["arr_0"]
    mask_outputs = label_masks_to_outputs(masks)

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
    output_path = Path(args.output_path) if args.output_path else mask_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (width * 2, height))

    n_frames = min(len(frames), len(mask_outputs))
    for t in range(n_frames):
        frame = load_frame(frames[t])
        overlay = render_overlay(frame, mask_outputs.get(t, mask_outputs.get(0)))
        panel = np.concatenate([frame, overlay], axis=1)
        writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Saved: {output_path} ({n_frames} frames)")


if __name__ == "__main__":
    main()
