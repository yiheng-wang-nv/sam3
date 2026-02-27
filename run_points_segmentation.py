#!/usr/bin/env python3
"""
Run SAM3 segmentation with text prompts and/or point clicks.

Supports three modes:
  1. Points only  (--annotations / --labelme_dir)
  2. Prompts only (--prompts)
  3. Combined      (--prompts + --annotations / --labelme_dir)

In combined mode, text prompts run first via the SAM3 detector, then
point clicks supplement/override via the SAM3 tracker. All results are
merged into a single label mask.

Usage:
    # Points only
    python run_points_segmentation.py \
        --video_path video.mp4 --checkpoint_path sam3.pt \
        --output_path out.npz --annotations 65:frame_65.json

    # Prompts + points
    python run_points_segmentation.py \
        --video_path video.mp4 --checkpoint_path sam3.pt \
        --output_path out.npz \
        --prompts "blue table" "robotic arm(s)" \
        --annotations 65:frame_65.json
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import cv2
import numpy as np
import torch

from sam3.model_builder import build_sam3_video_predictor


def parse_labelme_json(json_path):
    """Parse a labelme JSON, return {class_id: [(x,y), ...]}."""
    with open(json_path) as f:
        data = json.load(f)
    points_by_class = defaultdict(list)
    for shape in data.get("shapes", []):
        if shape["shape_type"] != "point":
            continue
        label = int(shape["label"])
        x, y = shape["points"][0]
        points_by_class[label].append((round(x), round(y)))
    return points_by_class


def collect_annotations(annotations_list=None, labelme_dir=None):
    """
    Collect all frame annotations.
    Returns: {class_id: {frame_idx: [(x,y), ...]}}
    """
    frames_by_class = defaultdict(lambda: defaultdict(list))

    if annotations_list:
        for entry in annotations_list:
            frame_str, json_path = entry.split(":", 1)
            frame_idx = int(frame_str)
            for cls, pts in parse_labelme_json(json_path).items():
                frames_by_class[cls][frame_idx].extend(pts)

    if labelme_dir:
        for jf in sorted(glob.glob(os.path.join(labelme_dir, "*.json"))):
            basename = os.path.splitext(os.path.basename(jf))[0]
            nums = re.findall(r"\d+", basename)
            if not nums:
                print(f"Warning: cannot extract frame index from {jf}, skipping")
                continue
            frame_idx = int(nums[-1])
            for cls, pts in parse_labelme_json(jf).items():
                frames_by_class[cls][frame_idx].extend(pts)

    return dict(frames_by_class)


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 segmentation with text prompts and/or point clicks."
    )
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_path", required=True, help="Output npz path")
    parser.add_argument(
        "--prompts", nargs="+", default=None,
        help="Text prompts for SAM3 detector (run before point clicks). "
             "Each prompt gets mask label = prompt_index + 1.",
    )
    parser.add_argument(
        "--annotations", nargs="+", default=None,
        help="Frame-to-labelme-JSON mappings: 'frame_idx:path.json' ...",
    )
    parser.add_argument(
        "--labelme_dir", default=None,
        help="Directory of labelme JSONs named frame_NNNN.json or NNNN.json",
    )
    parser.add_argument("--save_video", action="store_true", help="Save comparison video")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--binary", action="store_true",
                        help="Merge all classes into label 1 (binary foreground/background)")
    parser.add_argument("--postprocess", action="store_true", help="Run postprocessing on masks")
    parser.add_argument("--pp_fill_interior_class", type=str, default=None,
                        help='Fill bg inside these class contours, e.g. "1,2,3,4,5"')
    parser.add_argument("--pp_fill_interior_target", type=int, default=6)
    parser.add_argument("--pp_min_hole_size", type=int, default=64)
    parser.add_argument("--pp_min_object_size", type=int, default=50)
    parser.add_argument("--pp_closing_iterations", type=int, default=1)
    parser.add_argument("--pp_union_hole_fill", action="store_true")
    parser.add_argument("--pp_union_gap_fill", action="store_true")
    parser.add_argument("--pp_union_gap_closing_iterations", type=int, default=1)
    parser.add_argument("--pp_no_remove_small_objects", action="store_true")
    args = parser.parse_args()

    has_points = args.annotations or args.labelme_dir
    has_prompts = args.prompts and len(args.prompts) > 0
    if not has_points and not has_prompts:
        parser.error("Must provide --prompts and/or --annotations/--labelme_dir")

    # ── Collect point annotations (if any) ──
    frames_by_class = {}
    if has_points:
        frames_by_class = collect_annotations(args.annotations, args.labelme_dir)

    if args.binary and frames_by_class:
        merged = defaultdict(list)
        for cls, frames in frames_by_class.items():
            for fidx, pts in frames.items():
                merged[fidx].extend(pts)
        frames_by_class = {1: dict(merged)}
        args.pp_fill_interior_class = "1"
        args.pp_fill_interior_target = 1
        print("Binary mode: all classes merged into label 1")

    if frames_by_class:
        all_point_classes = sorted(frames_by_class.keys())
        print(f"Point classes: {all_point_classes}")
        for cls in all_point_classes:
            frames = frames_by_class[cls]
            total_pts = sum(len(pts) for pts in frames.values())
            print(f"  Class {cls}: {total_pts} points across {len(frames)} frame(s)")

    prompts_list = args.prompts or []
    if prompts_list:
        print(f"Text prompts ({len(prompts_list)}): {prompts_list}")

    # ── Video info ──
    cap = cv2.VideoCapture(args.video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Video: {args.video_path} ({n_frames} frames, {W}x{H})")

    print(f"Loading model from {args.checkpoint_path}...")
    video_predictor = build_sam3_video_predictor(args.checkpoint_path)

    label_masks = np.zeros((n_frames, H, W), dtype=np.uint8)

    # ── Phase 1: Text prompts (detector) ──
    if prompts_list:
        response = video_predictor.handle_request(
            request=dict(type="start_session", resource_path=args.video_path)
        )
        session_id = response["session_id"]

        for prompt_idx, prompt_text in enumerate(prompts_list):
            mask_label = prompt_idx + 1
            print(f"Prompt [{prompt_idx}] '{prompt_text}' → mask label {mask_label}")

            video_predictor.handle_request(
                request=dict(type="reset_session", session_id=session_id)
            )
            video_predictor.handle_request(
                request=dict(
                    type="add_prompt", session_id=session_id,
                    frame_index=0, text=prompt_text,
                )
            )

            for resp in video_predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video", session_id=session_id,
                    max_frame_num_to_track=n_frames,
                )
            ):
                fidx = resp["frame_index"]
                if fidx >= n_frames:
                    continue
                out = resp["outputs"]
                n_objs = len(out["out_obj_ids"])
                if n_objs == 0:
                    continue
                bmasks = np.array(out["out_binary_masks"], dtype=bool)
                if bmasks.ndim == 4:
                    bmasks = bmasks[:, 0]
                combined = np.any(bmasks, axis=0)
                label_masks[fidx][combined] = mask_label

        prompt_labels = np.unique(label_masks).tolist()
        print(f"After text prompts: labels present = {prompt_labels}")

    # ── Phase 2: Point clicks (tracker) ──
    if frames_by_class:
        tracker = video_predictor.model.tracker
        tracker.backbone = video_predictor.model.detector.backbone

        inference_state = tracker.init_state(video_path=args.video_path)
        tracker.clear_all_points_in_video(inference_state)

        all_point_classes = sorted(frames_by_class.keys())
        for cls in all_point_classes:
            for frame_idx, pts in sorted(frames_by_class[cls].items()):
                rel_points = [[x / W, y / H] for x, y in pts]
                points_tensor = torch.tensor(rel_points, dtype=torch.float32)
                labels_tensor = torch.ones(len(pts), dtype=torch.int32)
                tracker.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=cls,
                    points=points_tensor,
                    labels=labels_tensor,
                    clear_old_points=False,
                    rel_coordinates=True,
                )
                print(f"  Added {len(pts)} points for class {cls} at frame {frame_idx}")

        start_frame = min(
            min(frames.keys()) for frames in frames_by_class.values()
        )
        print(f"Propagating points from frame {start_frame}...")

        for (
            frame_idx, obj_ids, _, video_res_masks, _
        ) in tracker.propagate_in_video(
            inference_state,
            start_frame_idx=start_frame,
            max_frame_num_to_track=n_frames,
            reverse=False,
            propagate_preflight=True,
        ):
            if frame_idx >= n_frames:
                continue
            masks = (video_res_masks > 0.0).to(torch.bool).cpu().numpy()
            if masks.ndim == 4:
                masks = masks[:, 0]
            for i, oid in enumerate(obj_ids):
                label_masks[frame_idx][masks[i]] = int(oid)

    final_labels = np.unique(label_masks).tolist()
    print(f"Final labels present: {final_labels}")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    np.savez_compressed(args.output_path, arr_0=label_masks)
    labels_present = np.unique(label_masks).tolist()
    print(f"Saved raw: {args.output_path} ({n_frames} frames, labels: {labels_present})")

    post_masks = None
    if args.postprocess:
        from postprocess_masks import postprocess_video_masks

        fill_classes = None
        if args.pp_fill_interior_class:
            fill_classes = [int(x) for x in args.pp_fill_interior_class.split(",")]

        max_label = int(label_masks.max())
        post_masks = postprocess_video_masks(
            label_masks,
            num_classes=max_label + 1,
            fill_holes=True,
            min_hole_size=args.pp_min_hole_size,
            min_object_size=args.pp_min_object_size,
            closing_iterations=args.pp_closing_iterations,
            fill_interior_class=fill_classes,
            fill_interior_target=args.pp_fill_interior_target,
            union_hole_fill=args.pp_union_hole_fill,
            remove_small_objects_enabled=not args.pp_no_remove_small_objects,
            union_gap_fill=args.pp_union_gap_fill,
            union_gap_closing_iterations=args.pp_union_gap_closing_iterations,
        )
        post_path = args.output_path.replace("_masks.npz", "_masks_post.npz")
        if post_path == args.output_path:
            post_path = args.output_path.replace(".npz", "_post.npz")
        np.savez_compressed(post_path, arr_0=post_masks)
        post_labels = np.unique(post_masks).tolist()
        print(f"Saved post: {post_path} (labels: {post_labels})")

    if args.save_video:
        from sam3.visualization_utils import load_frame, render_masklet_frame
        from produce_mask_comparison_video import label_masks_to_outputs

        raw_outputs = label_masks_to_outputs(label_masks)
        post_outputs = label_masks_to_outputs(post_masks) if post_masks is not None else None

        n_panels = 3 if post_outputs else 2
        video_out = args.output_path.replace(".npz", "_comparison.mp4")

        cap = cv2.VideoCapture(args.video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_out, fourcc, args.fps, (W * n_panels, H))

        def render_overlay(frame, output):
            overlay = render_masklet_frame(frame, output, frame_idx=None)
            masks_arr = output["out_binary_masks"]
            if masks_arr.size > 0:
                combined = np.any(masks_arr, axis=0)
            else:
                combined = np.zeros(frame.shape[:2], dtype=bool)
            overlay[~combined] = 0
            return overlay

        t = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rendered = load_frame(frame_rgb)

            raw_out = raw_outputs.get(t)
            if raw_out:
                raw_panel = render_overlay(rendered, raw_out)
            else:
                raw_panel = np.zeros_like(rendered)

            panels = [rendered, raw_panel]

            if post_outputs is not None:
                post_out = post_outputs.get(t)
                if post_out:
                    post_panel = render_overlay(rendered, post_out)
                else:
                    post_panel = np.zeros_like(rendered)
                panels.append(post_panel)

            combined = np.concatenate(panels, axis=1)
            writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            t += 1

        cap.release()
        writer.release()
        print(f"Saved video: {video_out} ({n_panels} panels)")


if __name__ == "__main__":
    main()
