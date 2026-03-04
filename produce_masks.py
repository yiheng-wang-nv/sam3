import numpy as np
import torch
import cv2
import glob
import os
import shutil
import argparse
import pickle
from sam3.visualization_utils import load_frame, render_masklet_frame, save_masklet_video
from sam3.model_builder import build_sam3_video_predictor
from postprocess_masks import postprocess_video_masks, parse_fill_bg_roi


def parse_points(points_str):
    if not points_str:
        return []
    points = []
    for item in points_str.split(";"):
        item = item.strip()
        if not item:
            continue
        x_str, y_str = item.split(",")
        points.append((float(x_str), float(y_str)))
    return points


def parse_labels(labels_str, num_points):
    if not labels_str:
        return [1] * num_points
    labels = [int(x.strip()) for x in labels_str.split(",") if x.strip()]
    if len(labels) != num_points:
        raise ValueError(
            f"labels length {len(labels)} does not match points length {num_points}"
        )
    return labels


def parse_points_by_frame(points_by_frame_str):
    if not points_by_frame_str:
        return {}
    points_by_frame = {}
    for entry in points_by_frame_str.split("|"):
        entry = entry.strip()
        if not entry:
            continue
        frame_str, points_str = entry.split(":", 1)
        frame_idx = int(frame_str)
        points_by_frame[frame_idx] = parse_points(points_str)
    return points_by_frame


def parse_labels_by_frame(labels_by_frame_str):
    if not labels_by_frame_str:
        return {}
    labels_by_frame = {}
    for entry in labels_by_frame_str.split("|"):
        entry = entry.strip()
        if not entry:
            continue
        frame_str, labels_str = entry.split(":", 1)
        frame_idx = int(frame_str)
        labels_by_frame[frame_idx] = [
            int(x.strip()) for x in labels_str.split(",") if x.strip()
        ]
    return labels_by_frame


def parse_assist_points(assist_points_str):
    """Parse assist points: 'prompt_idx:frame_idx:x1,y1;x2,y2[:direction]|...'
    Returns {prompt_idx: {frame_idx: {"points": [(x,y),...], "direction": str|None}}}
    """
    if not assist_points_str:
        return {}
    result = {}
    for entry in assist_points_str.split("|"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        prompt_idx = int(parts[0])
        frame_idx = int(parts[1])
        points_str = parts[2]
        direction = parts[3] if len(parts) > 3 else None
        points = parse_points(points_str)
        result.setdefault(prompt_idx, {})[frame_idx] = {
            "points": points,
            "direction": direction,
        }
    return result


def parse_prompt_extra_frames(s):
    """Parse 'prompt_idx:frame_idx|...' → {prompt_idx: [frame_idx, ...]}
    frame_idx=-1 means last frame (resolved at runtime).
    """
    if not s:
        return {}
    result = {}
    for entry in s.split("|"):
        entry = entry.strip()
        if not entry:
            continue
        prompt_str, frame_str = entry.split(":")
        prompt_idx = int(prompt_str)
        frame_idx = int(frame_str)
        result.setdefault(prompt_idx, []).append(frame_idx)
    return result


def outputs_to_npz(outputs, output_path, merge_objects, default_hw):
    frame_indices = sorted(outputs.keys())
    if not frame_indices:
        print("No outputs available for npz conversion.")
        return None

    first_frame = outputs[frame_indices[0]]
    if first_frame["out_binary_masks"].size > 0:
        _, h, w = first_frame["out_binary_masks"].shape
    else:
        h, w = default_hw

    if merge_objects:
        all_masks = np.zeros((len(frame_indices), h, w), dtype=np.uint8)
        for t, frame_idx in enumerate(frame_indices):
            frame_data = outputs[frame_idx]
            binary_masks = frame_data["out_binary_masks"]
            obj_ids = frame_data["out_obj_ids"]
            for mask, obj_id in zip(binary_masks, obj_ids):
                label = int(obj_id) + 1
                all_masks[t][mask] = label
        output_data = all_masks
    else:
        max_objects = max(len(outputs[idx]["out_obj_ids"]) for idx in frame_indices)
        all_masks = np.zeros((max_objects, len(frame_indices), h, w), dtype=np.uint8)
        for t, frame_idx in enumerate(frame_indices):
            frame_data = outputs[frame_idx]
            binary_masks = frame_data["out_binary_masks"]
            for obj_idx, mask in enumerate(binary_masks):
                all_masks[obj_idx, t] = mask.astype(np.uint8) * 255
        output_data = all_masks

    np.savez_compressed(output_path, arr_0=output_data)
    print(f"Saved npz: {output_path}")
    return output_path


def outputs_to_label_masks(outputs, default_hw):
    frame_indices = sorted(outputs.keys())
    if not frame_indices:
        return np.zeros((0, *default_hw), dtype=np.uint8), []

    first_frame = outputs[frame_indices[0]]
    if first_frame["out_binary_masks"].size > 0:
        _, h, w = first_frame["out_binary_masks"].shape
    else:
        h, w = default_hw

    label_masks = np.zeros((len(frame_indices), h, w), dtype=np.uint8)
    for t, frame_idx in enumerate(frame_indices):
        frame_data = outputs[frame_idx]
        binary_masks = frame_data["out_binary_masks"]
        obj_ids = frame_data["out_obj_ids"]
        for mask, obj_id in zip(binary_masks, obj_ids):
            label = int(obj_id) + 1
            label_masks[t][mask] = label
    return label_masks, frame_indices


def label_masks_to_outputs(label_masks, frame_indices):
    outputs = {}
    if len(frame_indices) == 0:
        return outputs
    _, h, w = label_masks.shape
    for t, frame_idx in enumerate(frame_indices):
        frame_mask = label_masks[t]
        labels = np.unique(frame_mask)
        labels = labels[labels > 0]
        if labels.size == 0:
            outputs[frame_idx] = {
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
        outputs[frame_idx] = {
            "out_obj_ids": np.array(obj_ids, dtype=np.int64),
            "out_probs": np.ones(len(obj_ids), dtype=np.float32),
            "out_boxes_xywh": np.zeros((len(obj_ids), 4), dtype=np.float32),
            "out_binary_masks": np.stack(masks, axis=0),
        }
    return outputs


def _parse_class_list(value):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]

def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 Video Segmentation")
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        default="/localhome/local-vennw/code/3rd_sam3/sam3.pt",
        help="Path to SAM3 checkpoint"
    )
    parser.add_argument(
        "--video_path", 
        type=str, 
        required=True,
        help="Path to input video file or directory of frames"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--prompts", 
        type=str, 
        nargs='+', 
        default=["blue cloth", "robotic arms"],
        help="List of text prompts to segment"
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=30,
        help="FPS for output video"
    )
    parser.add_argument(
        "--postprocess_for_vis",
        action="store_true",
        help="Apply postprocess to masks for visualization only."
    )
    parser.add_argument(
        "--pp_min_hole_size",
        type=int,
        default=64,
        help="Postprocess: fill holes smaller than this."
    )
    parser.add_argument(
        "--pp_min_object_size",
        type=int,
        default=50,
        help="Postprocess: remove objects smaller than this."
    )
    parser.add_argument(
        "--pp_closing_iterations",
        type=int,
        default=1,
        help="Postprocess: closing iterations."
    )
    parser.add_argument(
        "--pp_no_fill_holes",
        action="store_true",
        help="Postprocess: disable hole filling."
    )
    parser.add_argument(
        "--pp_no_remove_small_objects",
        action="store_true",
        help="Postprocess: disable removing small objects."
    )
    parser.add_argument(
        "--pp_union_hole_fill",
        action="store_true",
        help="Postprocess: fill holes based on union of all >0 classes."
    )
    parser.add_argument(
        "--pp_union_gap_fill",
        action="store_true",
        help="Postprocess: fill thin background gaps using union closing."
    )
    parser.add_argument(
        "--pp_union_gap_closing_iterations",
        type=int,
        default=1,
        help="Postprocess: union gap fill closing iterations."
    )
    parser.add_argument(
        "--pp_fill_blue_table_quadrant",
        action="store_true",
        help="Postprocess: fill black region in blue table quadrant."
    )
    parser.add_argument(
        "--pp_blue_table_label",
        type=int,
        default=1,
        help="Postprocess: blue table label id."
    )
    parser.add_argument(
        "--pp_blue_table_target",
        type=int,
        default=4,
        help="Postprocess: target label for blue table quadrant fill."
    )
    parser.add_argument(
        "--pp_blue_table_quadrant_mode",
        type=str,
        default="right_down",
        help="Postprocess: quadrant fill mode (right_down or left_down)."
    )
    parser.add_argument(
        "--pp_blue_table_y_pad_top",
        type=int,
        default=60,
        help="Postprocess: top padding rows to exclude."
    )
    parser.add_argument(
        "--pp_blue_table_y_pad_bottom",
        type=int,
        default=60,
        help="Postprocess: bottom padding rows to exclude."
    )
    parser.add_argument(
        "--pp_blue_table_skip_if_label_above",
        type=int,
        default=None,
        help="Postprocess: skip quadrant fill if this label is higher than blue table."
    )
    parser.add_argument(
        "--pp_blue_table_skip_if_label_area_gt",
        type=int,
        default=None,
        help="Postprocess: skip quadrant fill if skip label area exceeds this pixel count."
    )
    parser.add_argument(
        "--pp_fill_interior_class",
        type=str,
        default=None,
        help='Postprocess: fill background inside these class contours, e.g. "1,3".'
    )
    parser.add_argument(
        "--pp_fill_interior_target",
        type=int,
        default=4,
        help="Postprocess: target class for interior fill."
    )
    parser.add_argument(
        "--pp_scanline_fill",
        action="store_true",
        help="Postprocess: row-based fill between first/last source_label pixels.",
    )
    parser.add_argument(
        "--pp_scanline_source_label",
        type=int,
        default=1,
        help="Postprocess: label to scan for in scanline fill (default: 1).",
    )
    parser.add_argument(
        "--pp_scanline_fill_value",
        type=int,
        default=3,
        help="Postprocess: value to fill background with in scanline fill (default: 3).",
    )
    parser.add_argument(
        "--save_video", 
        action="store_true",
        help="Whether to save the visualization video"
    )
    parser.add_argument(
        "--save_side_by_side",
        action="store_true",
        help="Save side-by-side video (original | mask overlay).",
    )
    parser.add_argument(
        "--points",
        type=str,
        default="",
        help="Extra point prompts as 'x1,y1;x2,y2;...'. Treated as one extra category.",
    )
    parser.add_argument(
        "--point_labels",
        type=str,
        default="",
        help="Point labels as '1,0,1,...' (1=positive, 0=negative).",
    )
    parser.add_argument(
        "--points_frame_idx",
        type=int,
        default=0,
        help="Frame index to apply the point prompts.",
    )
    parser.add_argument(
        "--points_by_frame",
        type=str,
        default="",
        help="Multiple frames: 'frame: x1,y1;...|frame: x1,y1;...'.",
    )
    parser.add_argument(
        "--point_labels_by_frame",
        type=str,
        default="",
        help="Labels per frame: 'frame:1,0|frame:1,1'.",
    )
    parser.add_argument(
        "--points_prompt_idx",
        type=int,
        default=None,
        help="Assign point-click masks to this prompt index (0-based). "
             "If not set, points become an extra category after all prompts.",
    )
    parser.add_argument(
        "--assist_points",
        type=str,
        default="",
        help="Points that assist text prompts (added in the same SAM3 session). "
             'Format: "prompt_idx:frame_idx:x1,y1;x2,y2|..." '
             "e.g. '0:0:35.5,224.2|3:0:100,200;150,250'",
    )
    parser.add_argument(
        "--prompt_extra_frames",
        type=str,
        default="",
        help="Re-add text prompt on extra frames for bidirectional propagation. "
             'Format: "prompt_idx:frame_idx|..." e.g. "2:-1" (-1 = last frame). '
             "Useful for objects that appear later in the video.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Only process the first N frames (for quick debug).",
    )
    parser.add_argument(
        "--save_npz",
        action="store_true",
        help="Also save Cosmos-compatible npz from the pkl output.",
    )
    parser.add_argument(
        "--npz_separate",
        action="store_true",
        help="Keep objects separate in npz (N,T,H,W). Default merges objects.",
    )
    parser.add_argument(
        "--no_pkl",
        action="store_true",
        help="Do not save pkl output.",
    )
    parser.add_argument(
        "--invert_mask",
        action="store_true",
        help="Invert masks: all predicted labels -> 0, background -> 1.",
    )
    parser.add_argument(
        "--static_prompts",
        type=str,
        nargs="*",
        default=None,
        help="Prompts whose mask is static (segment frame 0 only, replicate to all frames).",
    )
    parser.add_argument(
        "--postprocess_only",
        action="store_true",
        help="Skip inference. Load existing *_masks.npz, run postprocessing, save *_masks_post.npz.",
    )
    parser.add_argument(
        "--pp_fill_bg_roi",
        type=str,
        action="append",
        default=None,
        help='Fill background in ROI. Format: "frame_start,frame_end_ratio,y_min,y_max,x_min,x_max,target". '
             'Use -1 for full range. Can be specified multiple times.',
    )
    parser.add_argument(
        "--pp_overwrite",
        action="store_true",
        help="Overwrite existing *_masks_post.npz (used with --postprocess_only).",
    )
    parser.add_argument(
        "--pp_topleft_rect",
        action="store_true",
        help="Fill bg in rect left-below the top-left corner of a label.",
    )
    parser.add_argument("--pp_topleft_rect_label", type=int, default=1)
    parser.add_argument("--pp_topleft_rect_fill", type=int, default=5)
    parser.add_argument("--pp_topleft_rect_y_max", type=int, default=420)
    parser.add_argument("--pp_topleft_rect_frame_start", type=int, default=10)
    parser.add_argument("--pp_topleft_rect_frame_end_ratio", type=float, default=0.667)
    parser.add_argument("--pp_topright_rect", action="store_true",
                        help="Fill bg rect left-below top-right corner of a label (last 1/3).")
    parser.add_argument("--pp_topright_rect_label", type=int, default=1)
    parser.add_argument("--pp_topright_rect_fill", type=int, default=5)
    parser.add_argument("--pp_topright_rect_y_max", type=int, default=420)
    parser.add_argument("--pp_topright_rect_y_threshold", type=int, default=200)
    parser.add_argument("--pp_topright_rect_frame_start_ratio", type=float, default=0.667)
    parser.add_argument("--pp_leftmost_rect", action="store_true",
                        help="Fill bg below-left of the leftmost label-1 pixel. "
                             "First half: always. Second half: only if skip_label absent.")
    parser.add_argument("--pp_leftmost_rect_label", type=int, default=1)
    parser.add_argument("--pp_leftmost_rect_fill", type=int, default=5)
    parser.add_argument("--pp_leftmost_rect_y_max", type=int, default=420)
    parser.add_argument("--pp_leftmost_rect_skip_label", type=int, default=3)
    return parser.parse_args()

def propagate_in_video(predictor, session_id, max_frames=None, direction="both"):
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            max_frame_num_to_track=max_frames,
            propagation_direction=direction,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame

def _run_postprocess_only(args):
    """Load existing *_masks.npz, apply postprocessing, save *_masks_post.npz and optional vis."""
    video_path = args.video_path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if not video_name and os.path.isdir(video_path):
        video_name = os.path.basename(os.path.normpath(video_path))

    raw_npz_path = os.path.join(args.output_dir, f"{video_name}_masks.npz")
    post_npz_path = os.path.join(args.output_dir, f"{video_name}_masks_post.npz")

    if not os.path.exists(raw_npz_path):
        print(f"Error: Raw masks not found: {raw_npz_path}")
        return

    if os.path.exists(post_npz_path) and not args.pp_overwrite:
        print(f"Skipping (already exists): {post_npz_path}  (use --pp_overwrite to redo)")
        return

    print(f"[postprocess_only] Loading raw masks from {raw_npz_path}")
    data = np.load(raw_npz_path)
    label_masks = data["arr_0"]  # (T, H, W)
    T, H, W = label_masks.shape
    frame_indices = list(range(T))
    print(f"  Loaded masks: {label_masks.shape}, labels present: {np.unique(label_masks).tolist()}")

    max_label = int(label_masks.max())
    num_classes = max_label + 1

    fill_classes = _parse_class_list(args.pp_fill_interior_class)
    fill_bg_roi_list = None
    if args.pp_fill_bg_roi:
        fill_bg_roi_list = [parse_fill_bg_roi(s) for s in args.pp_fill_bg_roi]

    processed = postprocess_video_masks(
        label_masks,
        num_classes=num_classes,
        fill_holes=not args.pp_no_fill_holes,
        min_hole_size=args.pp_min_hole_size,
        min_object_size=args.pp_min_object_size,
        closing_iterations=args.pp_closing_iterations,
        fill_interior_class=fill_classes,
        fill_interior_target=args.pp_fill_interior_target,
        union_hole_fill=args.pp_union_hole_fill,
        remove_small_objects_enabled=not args.pp_no_remove_small_objects,
        union_gap_fill=args.pp_union_gap_fill,
        union_gap_closing_iterations=args.pp_union_gap_closing_iterations,
        fill_blue_table_quadrant_enabled=args.pp_fill_blue_table_quadrant,
        blue_table_label=args.pp_blue_table_label,
        blue_table_target=args.pp_blue_table_target,
        blue_table_quadrant_mode=args.pp_blue_table_quadrant_mode,
        blue_table_y_pad_top=args.pp_blue_table_y_pad_top,
        blue_table_y_pad_bottom=args.pp_blue_table_y_pad_bottom,
        blue_table_skip_if_label_above=args.pp_blue_table_skip_if_label_above,
        blue_table_skip_if_label_area_gt=args.pp_blue_table_skip_if_label_area_gt,
        fill_bg_roi_list=fill_bg_roi_list,
        scanline_fill_enabled=args.pp_scanline_fill,
        scanline_source_label=args.pp_scanline_source_label,
        scanline_fill_value=args.pp_scanline_fill_value,
        topleft_rect_enabled=args.pp_topleft_rect,
        topleft_rect_label=args.pp_topleft_rect_label,
        topleft_rect_fill=args.pp_topleft_rect_fill,
        topleft_rect_y_max=args.pp_topleft_rect_y_max,
        topleft_rect_frame_start=args.pp_topleft_rect_frame_start,
        topleft_rect_frame_end_ratio=args.pp_topleft_rect_frame_end_ratio,
        topright_rect_enabled=args.pp_topright_rect,
        topright_rect_label=args.pp_topright_rect_label,
        topright_rect_fill=args.pp_topright_rect_fill,
        topright_rect_y_max=args.pp_topright_rect_y_max,
        topright_rect_y_threshold=args.pp_topright_rect_y_threshold,
        topright_rect_frame_start_ratio=args.pp_topright_rect_frame_start_ratio,
        leftmost_rect_enabled=args.pp_leftmost_rect,
        leftmost_rect_label=args.pp_leftmost_rect_label,
        leftmost_rect_fill=args.pp_leftmost_rect_fill,
        leftmost_rect_y_max=args.pp_leftmost_rect_y_max,
        leftmost_rect_skip_label=args.pp_leftmost_rect_skip_label,
    )

    np.savez_compressed(post_npz_path, arr_0=processed)
    print(f"  Saved postprocessed masks: {post_npz_path}")
    print(f"  Labels after postprocess: {np.unique(processed).tolist()}")

    if args.save_side_by_side:
        vis_outputs = label_masks_to_outputs(processed, frame_indices)
        if isinstance(video_path, str) and video_path.endswith(".mp4"):
            cap = cv2.VideoCapture(video_path)
            video_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        else:
            video_frames = sorted(glob.glob(os.path.join(video_path, "*.jpg")))

        if video_frames:
            output_compare_path = os.path.join(args.output_dir, f"{video_name}_compare.mp4")
            print(f"  Saving side-by-side video to {output_compare_path}")
            first_frame = load_frame(video_frames[0])
            height, width = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_compare_path, fourcc, args.fps, (width * 2, height))
            for frame_idx in sorted(vis_outputs.keys()):
                frame = load_frame(video_frames[frame_idx])
                overlay = render_masklet_frame(frame, vis_outputs[frame_idx], frame_idx=None)
                masks = vis_outputs[frame_idx]["out_binary_masks"]
                combined = np.any(masks, axis=0) if masks.size > 0 else np.zeros(frame.shape[:2], dtype=bool)
                overlay[~combined] = 0
                compare = np.concatenate([frame, overlay], axis=1)
                writer.write(cv2.cvtColor(compare, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"  Side-by-side video saved to {output_compare_path}")

    print("[postprocess_only] Done.")


def process_single_video(video_predictor, args):
    """Process a single video with a pre-loaded SAM3 model. Handles session lifecycle."""
    video_path = args.video_path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if not video_name and os.path.isdir(video_path):
        video_name = os.path.basename(os.path.normpath(video_path))
        
    print(f"Processing video: {video_path}")

    # Load video frames for visualization
    # Even if we don't save the video, we might need image dimensions.
    # But usually we need the frames to properly visualize/save video.
    # If not saving video, we can skip loading all frames to memory to save RAM,
    # unless SAM3 requires them for initialization (it does need 'resource_path' to be valid).
    
    total_video_frames = 0

    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if args.max_frames is not None:
            total_video_frames = min(total_video_frames, args.max_frames)
        video_frames_for_vis = []
        if args.save_video or args.save_side_by_side:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if args.max_frames is not None and len(video_frames_for_vis) >= args.max_frames:
                    break
        else:
            # Just read one frame to get dimensions
            ret, frame = cap.read()
            if ret:
                video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
        try:
            video_frames_for_vis.sort(
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
            )
        except ValueError:
            print(f"Falling back to lexicographic sort for frames.")
            video_frames_for_vis.sort()

        total_video_frames = len(video_frames_for_vis)
        if args.max_frames is not None:
            total_video_frames = min(total_video_frames, args.max_frames)

        if not (args.save_video or args.save_side_by_side) and video_frames_for_vis:
             # Just keep one for dimension check if needed
             # Actually glob just returns paths, so memory is fine.
             pass
        elif (args.save_video or args.save_side_by_side) and args.max_frames is not None:
            video_frames_for_vis = video_frames_for_vis[: args.max_frames]

    if not video_frames_for_vis:
        print("Error: No frames found!")
        return

    # Get frame dimensions for point conversion
    if isinstance(video_frames_for_vis[0], np.ndarray):
        H, W = video_frames_for_vis[0].shape[:2]
    elif isinstance(video_frames_for_vis[0], str):
        img = cv2.imread(video_frames_for_vis[0])
        H, W = img.shape[:2]
    else:
        H, W = 480, 640

    # Initialize SAM3 session
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]

    prompts_list = args.prompts
    merged_outputs = {}
    static_prompts_set = set(args.static_prompts) if args.static_prompts else set()
    static_prompt_frame0 = {}
    static_prompt_merged_frames = {}

    assist_points_map = parse_assist_points(args.assist_points)
    extra_frames_map = parse_prompt_extra_frames(args.prompt_extra_frames)

    print(f"Start processing {len(prompts_list)} prompts: {prompts_list}")
    if assist_points_map:
        print(f"Assist points for prompt indices: {sorted(assist_points_map.keys())}")
    if extra_frames_map:
        print(f"Extra prompt frames: {extra_frames_map}")
    if static_prompts_set:
        print(f"Static prompts (frame-0 only, replicate to all frames): {static_prompts_set}")

    for prompt_idx, p in enumerate(prompts_list):
        is_static = p in static_prompts_set
        print(f"Processing prompt [{prompt_idx}]: '{p}'" + (" [STATIC]" if is_static else ""))
        
        # Reset session for new prompt
        _ = video_predictor.handle_request(
            request=dict(type="reset_session", session_id=session_id)
        )

        # Check if any assist entry specifies backward-only direction
        assist_direction = None
        if prompt_idx in assist_points_map:
            for _, a_entry in assist_points_map[prompt_idx].items():
                if a_entry.get("direction"):
                    assist_direction = a_entry["direction"]

        # Collect frames to add text prompt on
        prompt_frames = []
        if assist_direction != "backward":
            prompt_frames.append(0)
        if prompt_idx in extra_frames_map:
            for f in extra_frames_map[prompt_idx]:
                resolved = f if f >= 0 else total_video_frames + f
                if resolved not in prompt_frames:
                    prompt_frames.append(resolved)

        # Add text prompt on each frame
        for pf in prompt_frames:
            _ = video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=pf,
                    text=p,
                )
            )
        if prompt_frames:
            print(f"  Text prompt on frames: {prompt_frames}")

        # Assist points require: full propagation first (text only), then add points
        # and propagate again (partial, lightweight — reuses cached VG predictions).
        prompt_direction = assist_direction or "both"
        has_assist = prompt_idx in assist_points_map

        if has_assist and prompt_frames:
            # Step 1: full propagation with text prompt only (populates cache)
            print(f"  Running initial propagation (text only) to populate cache...")
            propagate_in_video(
                video_predictor, session_id, max_frames=args.max_frames,
                direction=prompt_direction
            )
            # Step 2: add assist point clicks (triggers partial propagation next)
            for a_frame_idx, a_entry in assist_points_map[prompt_idx].items():
                a_points = a_entry["points"]
                if a_frame_idx < 0:
                    a_frame_idx = total_video_frames + a_frame_idx
                rel_pts = [[x / W, y / H] for x, y in a_points]
                pts_tensor = torch.tensor(rel_pts, dtype=torch.float32)
                labels_tensor = torch.tensor([1] * len(a_points), dtype=torch.int32)
                _ = video_predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=a_frame_idx,
                        points=pts_tensor,
                        point_labels=labels_tensor,
                        obj_id=900 + prompt_idx,
                    )
                )
                print(f"  + Assist points on frame {a_frame_idx}: {len(a_points)} point(s)")
            # Step 3: partial propagation (merges point refinement with cached VG)
            print(f"  Running refinement propagation (partial)...")

        if has_assist and not prompt_frames:
            # backward-only: add text prompt on assist frame, propagate to fill
            # cache, then add point refinement.
            # Step 1: find the assist frame and add text prompt there
            assist_frame_for_text = None
            for a_frame_idx_raw, a_entry in assist_points_map[prompt_idx].items():
                resolved = a_frame_idx_raw if a_frame_idx_raw >= 0 else total_video_frames + a_frame_idx_raw
                assist_frame_for_text = resolved
                break
            if assist_frame_for_text is not None:
                _ = video_predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=assist_frame_for_text,
                        text=p,
                    )
                )
                print(f"  Text prompt on frame {assist_frame_for_text} (backward anchor)")
                # Step 2: full backward propagation to populate cache
                print(f"  Running initial propagation (text only, {prompt_direction}) to populate cache...")
                propagate_in_video(
                    video_predictor, session_id, max_frames=args.max_frames,
                    direction=prompt_direction
                )
            # Step 3: add assist point clicks (now cache exists)
            for a_frame_idx, a_entry in assist_points_map[prompt_idx].items():
                a_points = a_entry["points"]
                if a_frame_idx < 0:
                    a_frame_idx = total_video_frames + a_frame_idx
                rel_pts = [[x / W, y / H] for x, y in a_points]
                pts_tensor = torch.tensor(rel_pts, dtype=torch.float32)
                labels_tensor = torch.tensor([1] * len(a_points), dtype=torch.int32)
                _ = video_predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=a_frame_idx,
                        points=pts_tensor,
                        point_labels=labels_tensor,
                        obj_id=900 + prompt_idx,
                    )
                )
                print(f"  + Assist points on frame {a_frame_idx}: {len(a_points)} point(s)"
                      f" [direction={prompt_direction}]")
            # Step 4: partial refinement propagation
            print(f"  Running refinement propagation (partial)...")

        # Final propagation
        if is_static:
            outputs_per_frame = propagate_in_video(
                video_predictor, session_id, max_frames=1
            )
            if outputs_per_frame:
                ref_key = min(outputs_per_frame.keys())
                static_prompt_frame0[prompt_idx] = outputs_per_frame[ref_key]
                static_prompt_merged_frames[prompt_idx] = set(outputs_per_frame.keys())
        else:
            outputs_per_frame = propagate_in_video(
                video_predictor, session_id, max_frames=args.max_frames,
                direction=prompt_direction
            )
        
        # Merge this prompt's results (for static prompts, only the propagated frames)
        for frame_idx, frame_data in outputs_per_frame.items():
            if frame_idx not in merged_outputs:
                merged_outputs[frame_idx] = {
                    'out_obj_ids': [],
                    'out_probs': [],
                    'out_boxes_xywh': [],
                    'out_binary_masks': []
                }
            
            num_objs = len(frame_data['out_obj_ids'])
            if num_objs > 0:
                local_ids = np.full(num_objs, prompt_idx, dtype=np.int64)
                
                merged_outputs[frame_idx]['out_obj_ids'].append(local_ids)
                merged_outputs[frame_idx]['out_probs'].append(
                    np.array(frame_data['out_probs'], dtype=np.float32).reshape(-1)
                )
                merged_outputs[frame_idx]['out_boxes_xywh'].append(frame_data['out_boxes_xywh'])
                merged_outputs[frame_idx]['out_binary_masks'].append(frame_data['out_binary_masks'])

    # Optional: extra category from points (SAM3 tracker, no pre-propagation needed)
    points_frames = {}
    labels_frames = {}
    points_by_frame = parse_points_by_frame(args.points_by_frame)
    labels_by_frame = parse_labels_by_frame(args.point_labels_by_frame)

    for frame_idx, pts in points_by_frame.items():
        if not pts:
            continue
        lbls = labels_by_frame.get(frame_idx, [1] * len(pts))
        if len(lbls) != len(pts):
            raise ValueError(
                f"labels length {len(lbls)} does not match points length {len(pts)} "
                f"for frame {frame_idx}"
            )
        points_frames[frame_idx] = list(pts)
        labels_frames[frame_idx] = list(lbls)

    if args.points:
        points_abs = parse_points(args.points)
        labels = parse_labels(args.point_labels, len(points_abs))
        if points_abs:
            if args.points_frame_idx in points_frames:
                points_frames[args.points_frame_idx].extend(points_abs)
                labels_frames[args.points_frame_idx].extend(labels)
            else:
                points_frames[args.points_frame_idx] = list(points_abs)
                labels_frames[args.points_frame_idx] = list(labels)

    if points_frames:
        print(f"Processing points as extra category: {points_frames}")
        tracker = video_predictor.model.tracker
        tracker.backbone = video_predictor.model.detector.backbone

        inference_state_points = tracker.init_state(video_path=video_path)
        tracker.clear_all_points_in_video(inference_state_points)

        for frame_idx, points_abs in points_frames.items():
            if args.max_frames is not None and frame_idx >= args.max_frames:
                print(
                    f"Skipping points for frame {frame_idx} >= max_frames {args.max_frames}"
                )
                continue
            labels = labels_frames[frame_idx]
            rel_points = [[x / W, y / H] for x, y in points_abs]
            points_tensor = torch.tensor(rel_points, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int32)

            tracker.add_new_points_or_box(
                inference_state=inference_state_points,
                frame_idx=frame_idx,
                obj_id=1,
                points=points_tensor,
                labels=labels_tensor,
                clear_old_points=False,
                rel_coordinates=True,
            )

        points_outputs = {}
        start_frame_idx = min(points_frames.keys())
        max_track = (
            args.max_frames - start_frame_idx
            if args.max_frames is not None
            else inference_state_points["num_frames"]
        )
        if max_track is not None and max_track <= 0:
            print(
                f"Skipping points propagation: start_frame_idx={start_frame_idx} "
                f">= max_frames={args.max_frames}"
            )
            points_outputs = {}
        else:
            points_outputs = {}
            for (
                frame_idx,
                obj_ids,
                _,
                video_res_masks,
                obj_scores,
            ) in tracker.propagate_in_video(
                inference_state_points,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_track,
                reverse=False,
                propagate_preflight=True,
            ):
                if args.max_frames is not None and frame_idx >= args.max_frames:
                    continue
                if len(obj_ids) == 0:
                    continue
                masks = (video_res_masks > 0.0).to(torch.bool).cpu().numpy()
                if masks.ndim == 4:
                    masks = masks[:, 0]
                if torch.is_tensor(obj_scores):
                    scores = obj_scores.detach().float().cpu().numpy().astype(np.float32)
                else:
                    scores = np.array(obj_scores, dtype=np.float32).reshape(-1)
                points_outputs[frame_idx] = {
                    "out_obj_ids": np.array(obj_ids, dtype=np.int64),
                    "out_probs": scores,
                    "out_boxes_xywh": np.zeros((len(obj_ids), 4), dtype=np.float32),
                    "out_binary_masks": masks,
                }

        prompt_idx = args.points_prompt_idx if args.points_prompt_idx is not None else len(prompts_list)
        print(f"Merging point-click masks as prompt_idx={prompt_idx} (mask label {prompt_idx + 1})")
        for frame_idx, frame_data in points_outputs.items():
            if frame_idx not in merged_outputs:
                merged_outputs[frame_idx] = {
                    "out_obj_ids": [],
                    "out_probs": [],
                    "out_boxes_xywh": [],
                    "out_binary_masks": [],
                }
            num_objs = len(frame_data["out_obj_ids"])
            if num_objs > 0:
                local_ids = np.full(num_objs, prompt_idx, dtype=np.int64)
                merged_outputs[frame_idx]["out_obj_ids"].append(local_ids)
                merged_outputs[frame_idx]["out_probs"].append(
                    np.array(frame_data["out_probs"], dtype=np.float32).reshape(-1)
                )
                merged_outputs[frame_idx]["out_boxes_xywh"].append(
                    frame_data["out_boxes_xywh"]
                )
                merged_outputs[frame_idx]["out_binary_masks"].append(
                    frame_data["out_binary_masks"]
                )

    # Replicate static prompt masks to all frames
    if static_prompt_frame0:
        all_frame_indices = sorted(merged_outputs.keys())
        if not all_frame_indices and total_video_frames > 0:
            all_frame_indices = list(range(total_video_frames))
        for s_prompt_idx, frame0_data in static_prompt_frame0.items():
            num_objs = len(frame0_data['out_obj_ids'])
            if num_objs == 0:
                continue
            already_merged = static_prompt_merged_frames.get(s_prompt_idx, set())
            frames_to_add = [f for f in all_frame_indices if f not in already_merged]
            print(f"Replicating static prompt [{s_prompt_idx}] ('{prompts_list[s_prompt_idx]}') "
                  f"to {len(frames_to_add)} additional frames")
            for frame_idx in frames_to_add:
                if frame_idx not in merged_outputs:
                    merged_outputs[frame_idx] = {
                        'out_obj_ids': [], 'out_probs': [],
                        'out_boxes_xywh': [], 'out_binary_masks': []
                    }
                local_ids = np.full(num_objs, s_prompt_idx, dtype=np.int64)
                merged_outputs[frame_idx]['out_obj_ids'].append(local_ids)
                merged_outputs[frame_idx]['out_probs'].append(
                    np.array(frame0_data['out_probs'], dtype=np.float32).reshape(-1)
                )
                merged_outputs[frame_idx]['out_boxes_xywh'].append(
                    frame0_data['out_boxes_xywh']
                )
                merged_outputs[frame_idx]['out_binary_masks'].append(
                    frame0_data['out_binary_masks']
                )

    # Format merged outputs
    final_formatted_outputs = {}

    print("Merging results...")
    for frame_idx in merged_outputs.keys():
        if args.max_frames is not None and frame_idx >= args.max_frames:
            continue
        data_lists = merged_outputs[frame_idx]
        if len(data_lists['out_obj_ids']) > 0:
            final_formatted_outputs[frame_idx] = {
                'out_obj_ids': np.concatenate(data_lists['out_obj_ids']),
                'out_probs': np.concatenate(data_lists['out_probs']),
                'out_boxes_xywh': np.concatenate(data_lists['out_boxes_xywh'], axis=0),
                'out_binary_masks': np.concatenate(data_lists['out_binary_masks'], axis=0)
            }
        else:
            final_formatted_outputs[frame_idx] = {
                'out_obj_ids': np.array([], dtype=np.int64),
                'out_probs': np.array([], dtype=np.float32),
                'out_boxes_xywh': np.zeros((0, 4), dtype=np.float32),
                'out_binary_masks': np.zeros((0, H, W), dtype=bool)
            }

    # Disable bounding box rendering by zeroing boxes
    for frame_idx, frame_data in final_formatted_outputs.items():
        num_objs = len(frame_data["out_obj_ids"])
        frame_data["out_boxes_xywh"] = np.zeros((num_objs, 4), dtype=np.float32)

    # Optional: invert masks to keep background as label 1
    if args.invert_mask:
        print("Inverting masks: predicted -> 0, background -> 1")
        for frame_idx, frame_data in final_formatted_outputs.items():
            if frame_data["out_binary_masks"].size > 0:
                combined = np.any(frame_data["out_binary_masks"], axis=0)
            else:
                combined = np.zeros((H, W), dtype=bool)
            inverted = ~combined
            frame_data["out_binary_masks"] = inverted.reshape(1, H, W)
            frame_data["out_obj_ids"] = np.array([0], dtype=np.int64)
            frame_data["out_probs"] = np.array([1.0], dtype=np.float32)
            frame_data["out_boxes_xywh"] = np.zeros((1, 4), dtype=np.float32)
            
    raw_output_path = os.path.join(args.output_dir, f"{video_name}_segmentation_results.pkl")
    if not args.no_pkl:
        print(f"Saving raw segmentation results to {raw_output_path} ...")
        with open(raw_output_path, 'wb') as f:
            pickle.dump(final_formatted_outputs, f)

    # Optional: save Cosmos-compatible npz directly from outputs
    if args.save_npz:
        npz_output_path = os.path.join(args.output_dir, f"{video_name}_masks.npz")
        outputs_to_npz(
            final_formatted_outputs,
            npz_output_path,
            merge_objects=not args.npz_separate,
            default_hw=(H, W),
        )

    vis_outputs = final_formatted_outputs
    if args.postprocess_for_vis:
        label_masks, frame_indices = outputs_to_label_masks(
            final_formatted_outputs, default_hw=(H, W)
        )
        if label_masks.size > 0:
            max_label = int(label_masks.max())
            num_classes = max_label + 1
        else:
            num_classes = 1
        fill_classes = _parse_class_list(args.pp_fill_interior_class)
        processed = postprocess_video_masks(
            label_masks,
            num_classes=num_classes,
            fill_holes=not args.pp_no_fill_holes,
            min_hole_size=args.pp_min_hole_size,
            min_object_size=args.pp_min_object_size,
            closing_iterations=args.pp_closing_iterations,
            fill_interior_class=fill_classes,
            fill_interior_target=args.pp_fill_interior_target,
            union_hole_fill=args.pp_union_hole_fill,
            remove_small_objects_enabled=not args.pp_no_remove_small_objects,
            union_gap_fill=args.pp_union_gap_fill,
            union_gap_closing_iterations=args.pp_union_gap_closing_iterations,
            fill_blue_table_quadrant_enabled=args.pp_fill_blue_table_quadrant,
            blue_table_label=args.pp_blue_table_label,
            blue_table_target=args.pp_blue_table_target,
            blue_table_quadrant_mode=args.pp_blue_table_quadrant_mode,
            blue_table_y_pad_top=args.pp_blue_table_y_pad_top,
            blue_table_y_pad_bottom=args.pp_blue_table_y_pad_bottom,
            blue_table_skip_if_label_above=args.pp_blue_table_skip_if_label_above,
            blue_table_skip_if_label_area_gt=args.pp_blue_table_skip_if_label_area_gt,
            scanline_fill_enabled=args.pp_scanline_fill,
            scanline_source_label=args.pp_scanline_source_label,
            scanline_fill_value=args.pp_scanline_fill_value,
            topleft_rect_enabled=args.pp_topleft_rect,
            topleft_rect_label=args.pp_topleft_rect_label,
            topleft_rect_fill=args.pp_topleft_rect_fill,
            topleft_rect_y_max=args.pp_topleft_rect_y_max,
            topleft_rect_frame_start=args.pp_topleft_rect_frame_start,
            topleft_rect_frame_end_ratio=args.pp_topleft_rect_frame_end_ratio,
            topright_rect_enabled=args.pp_topright_rect,
            topright_rect_label=args.pp_topright_rect_label,
            topright_rect_fill=args.pp_topright_rect_fill,
            topright_rect_y_max=args.pp_topright_rect_y_max,
            topright_rect_y_threshold=args.pp_topright_rect_y_threshold,
            topright_rect_frame_start_ratio=args.pp_topright_rect_frame_start_ratio,
            leftmost_rect_enabled=args.pp_leftmost_rect,
            leftmost_rect_label=args.pp_leftmost_rect_label,
            leftmost_rect_fill=args.pp_leftmost_rect_fill,
            leftmost_rect_y_max=args.pp_leftmost_rect_y_max,
            leftmost_rect_skip_label=args.pp_leftmost_rect_skip_label,
        )
        vis_outputs = label_masks_to_outputs(processed, frame_indices)

    # Save visualization video ONLY if requested
    if args.save_video or args.save_side_by_side:
        # If we didn't load all frames earlier, we need to reload them now
        # Check if video_frames_for_vis contains enough frames
        # This simple check assumes if we loaded > 1 frame, we loaded the video.
        # If it was an mp4 and we loaded only 1, we need to reload.
        
        need_reload = False
        if isinstance(video_path, str) and video_path.endswith(".mp4"):
             if len(video_frames_for_vis) <= 1:
                 need_reload = True
        
        if need_reload:
            print("Reloading video frames for visualization...")
            cap = cv2.VideoCapture(video_path)
            video_frames_for_vis = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

        if args.save_video:
            output_video_path = os.path.join(args.output_dir, f"{video_name}_vis.mp4")
            print(f"Saving video to {output_video_path} ...")

            if shutil.which("ffmpeg") is not None:
                save_masklet_video(
                    video_frames=video_frames_for_vis,
                    outputs=vis_outputs,
                    out_path=output_video_path,
                    fps=args.fps,
                    show_frame_idx=False,
                )
            else:
                print("ffmpeg not found. Falling back to OpenCV video writer.")
                if len(video_frames_for_vis) == 0:
                    print("No frames available for video writing.")
                else:
                    first_frame = load_frame(video_frames_for_vis[0])
                    height, width = first_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        output_video_path, fourcc, args.fps, (width, height)
                    )
                    for frame_idx in sorted(vis_outputs.keys()):
                        if args.max_frames is not None and frame_idx >= args.max_frames:
                            continue
                        frame = load_frame(video_frames_for_vis[frame_idx])
                        overlay = render_masklet_frame(
                            frame,
                            vis_outputs[frame_idx],
                            frame_idx=None,
                        )
                        writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    writer.release()
                    print(f"Video saved (OpenCV) to {output_video_path}")

        if args.save_side_by_side:
            output_compare_path = os.path.join(
                args.output_dir, f"{video_name}_compare.mp4"
            )
            print(f"Saving side-by-side video to {output_compare_path} ...")
            if len(video_frames_for_vis) == 0:
                print("No frames available for side-by-side video.")
            else:
                first_frame = load_frame(video_frames_for_vis[0])
                height, width = first_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    output_compare_path, fourcc, args.fps, (width * 2, height)
                )
                for frame_idx in sorted(vis_outputs.keys()):
                    if args.max_frames is not None and frame_idx >= args.max_frames:
                        continue
                    frame = load_frame(video_frames_for_vis[frame_idx])
                    overlay = render_masklet_frame(
                        frame,
                        vis_outputs[frame_idx],
                        frame_idx=None,
                    )
                    masks = vis_outputs[frame_idx]["out_binary_masks"]
                    if masks.size > 0:
                        combined = np.any(masks, axis=0)
                    else:
                        combined = np.zeros(frame.shape[:2], dtype=bool)
                    overlay[~combined] = 0
                    compare = np.concatenate([frame, overlay], axis=1)
                    writer.write(cv2.cvtColor(compare, cv2.COLOR_RGB2BGR))
                writer.release()
                print(f"Side-by-side video saved to {output_compare_path}")
    else:
        print("Skipping video generation (--save_video not set).")

    video_predictor.handle_request(
        request=dict(type="close_session", session_id=session_id)
    )


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.postprocess_only:
        _run_postprocess_only(args)
        return

    print(f"Loading model from {args.checkpoint_path}...")
    video_predictor = build_sam3_video_predictor(args.checkpoint_path)
    process_single_video(video_predictor, args)


if __name__ == "__main__":
    main()
