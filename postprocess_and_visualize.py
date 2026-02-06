#!/usr/bin/env python3
"""
Postprocess existing mask npz and generate side-by-side video.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from sam3.visualization_utils import load_frame, render_masklet_frame
from postprocess_masks import postprocess_video_masks, parse_fill_bg_roi


def _parse_class_list(value):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def label_masks_to_outputs(label_masks):
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


def build_paths(args):
    camera_dir = Path(args.input_dir) / args.camera
    mask_path = camera_dir / f"{args.episode}_masks.npz"
    if args.video_path:
        video_path = Path(args.video_path)
    else:
        if not args.videos_dir:
            raise ValueError("Provide --video_path or --videos_dir.")
        video_path = Path(args.videos_dir) / args.camera / f"{args.episode}.mp4"
    return mask_path, video_path


def main():
    parser = argparse.ArgumentParser(description="Postprocess masks and generate compare video.")
    parser.add_argument("--input_dir", required=True, help="sam3_output_debug directory")
    parser.add_argument("--camera", required=True, help="camera folder name")
    parser.add_argument("--episode", required=True, help="episode id, e.g. episode_000075")
    parser.add_argument("--video_path", default=None, help="override video path")
    parser.add_argument("--videos_dir", default=None, help="base videos dir, e.g. .../videos/chunk-000")
    parser.add_argument("--output_path", default=None, help="output compare video path")
    parser.add_argument("--fps", type=int, default=30, help="output video fps")

    parser.add_argument("--min_hole_size", type=int, default=64, help="fill holes smaller than this")
    parser.add_argument("--min_object_size", type=int, default=50, help="remove objects smaller than this")
    parser.add_argument("--closing_iterations", type=int, default=1, help="closing iterations")
    parser.add_argument("--no_fill_holes", action="store_true", help="disable hole filling")
    parser.add_argument("--no_remove_small_objects", action="store_true", help="disable removing small objects")
    parser.add_argument("--union_hole_fill", action="store_true", help="fill holes based on union of all >0 classes")
    parser.add_argument("--union_gap_fill", action="store_true", help="fill thin background gaps using union closing")
    parser.add_argument("--union_gap_closing_iterations", type=int, default=1, help="union gap closing iterations")
    parser.add_argument("--fill_blue_table_quadrant", action="store_true", help="fill black region in blue table quadrant")
    parser.add_argument("--blue_table_label", type=int, default=1, help="blue table label id")
    parser.add_argument("--blue_table_target", type=int, default=4, help="target label for blue table quadrant fill")
    parser.add_argument("--blue_table_quadrant_mode", type=str, default="right_down", help="quadrant fill mode")
    parser.add_argument("--blue_table_y_pad_top", type=int, default=60, help="top padding rows to exclude")
    parser.add_argument("--blue_table_y_pad_bottom", type=int, default=60, help="bottom padding rows to exclude")
    parser.add_argument("--blue_table_skip_if_label_above", type=int, default=None, help="skip fill if label above blue table")
    parser.add_argument("--blue_table_skip_if_label_area_gt", type=int, default=None, help="skip fill if label area exceeds this size")
    parser.add_argument("--fill_interior_class", type=str, default=None, help='e.g. "1,3"')
    parser.add_argument("--fill_interior_target", type=int, default=4, help="target label for interior fill")
    parser.add_argument("--fill_bg_roi", type=str, action="append", default=None,
                        help='Fill background in ROI. Format: "frame_start,frame_end_ratio,y_min,y_max,x_min,x_max,target". '
                             'Use -1 for full range. Can be specified multiple times.')

    args = parser.parse_args()

    mask_path, video_path = build_paths(args)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    data = np.load(mask_path)
    label_masks = data["arr_0"]
    if label_masks.ndim != 3:
        raise ValueError(f"Expected (T,H,W) masks, got {label_masks.shape}")

    max_label = int(label_masks.max()) if label_masks.size > 0 else 0
    num_classes = max_label + 1
    fill_classes = _parse_class_list(args.fill_interior_class)
    fill_bg_roi_list = None
    if args.fill_bg_roi:
        fill_bg_roi_list = [parse_fill_bg_roi(s) for s in args.fill_bg_roi]

    processed = postprocess_video_masks(
        label_masks,
        num_classes=num_classes,
        fill_holes=not args.no_fill_holes,
        min_hole_size=args.min_hole_size,
        min_object_size=args.min_object_size,
        closing_iterations=args.closing_iterations,
        fill_interior_class=fill_classes,
        fill_interior_target=args.fill_interior_target,
        union_hole_fill=args.union_hole_fill,
        remove_small_objects_enabled=not args.no_remove_small_objects,
        union_gap_fill=args.union_gap_fill,
        union_gap_closing_iterations=args.union_gap_closing_iterations,
        fill_blue_table_quadrant_enabled=args.fill_blue_table_quadrant,
        blue_table_label=args.blue_table_label,
        blue_table_target=args.blue_table_target,
        blue_table_quadrant_mode=args.blue_table_quadrant_mode,
        blue_table_y_pad_top=args.blue_table_y_pad_top,
        blue_table_y_pad_bottom=args.blue_table_y_pad_bottom,
        blue_table_skip_if_label_above=args.blue_table_skip_if_label_above,
        blue_table_skip_if_label_area_gt=args.blue_table_skip_if_label_area_gt,
        fill_bg_roi_list=fill_bg_roi_list,
    )

    outputs = label_masks_to_outputs(processed)

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = mask_path.parent / f"{args.episode}_compare_post.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    first_frame = load_frame(frames[0])
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (width * 2, height))

    for frame_idx in sorted(outputs.keys()):
        if frame_idx >= len(frames):
            break
        frame = load_frame(frames[frame_idx])
        overlay = render_masklet_frame(frame, outputs[frame_idx], frame_idx=None)
        masks = outputs[frame_idx]["out_binary_masks"]
        if masks.size > 0:
            combined = np.any(masks, axis=0)
        else:
            combined = np.zeros(frame.shape[:2], dtype=bool)
        overlay[~combined] = 0
        compare = np.concatenate([frame, overlay], axis=1)
        writer.write(cv2.cvtColor(compare, cv2.COLOR_RGB2BGR))
    writer.release()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
