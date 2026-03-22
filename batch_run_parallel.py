import os
import glob
import subprocess
import argparse
import multiprocessing

def get_available_gpus():
    """Get list of available GPU IDs"""
    try:
        import torch
        return list(range(torch.cuda.device_count()))
    except:
        return [0]

def chunk_list(data, num_chunks):
    """Split list into N roughly equal chunks"""
    if num_chunks <= 0:
        return [data]
    n = len(data)
    out = []
    for i in range(num_chunks):
        start = i * n // num_chunks
        end = (i + 1) * n // num_chunks
        out.append(data[start:end])
    return out


def _parse_class_list(value):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_pp_per_camera(value):
    """
    Parse per-camera postprocess mapping.
    Format: "camera=1,3:1;camera2=2:2"
    Returns: {camera: (classes_or_None, target_or_None)}
    """
    mapping = {}
    if not value:
        return mapping
    entries = [e for e in value.split(";") if e.strip()]
    for entry in entries:
        if "=" not in entry:
            continue
        cam, rest = entry.split("=", 1)
        cam = cam.strip()
        parts = rest.split(":")
        classes_part = parts[0].strip() if parts else ""
        target_part = parts[1].strip() if len(parts) > 1 else ""
        classes = _parse_class_list(classes_part) if classes_part else None
        target = int(target_part) if target_part else None
        if cam:
            mapping[cam] = (classes, target)
    return mapping

def run_worker(
    gpu_id,
    video_list,
    checkpoint,
    prompts,
    base_output_dir,
    script_dir,
    points=None,
    point_labels=None,
    points_frame_idx=None,
    points_by_frame=None,
    point_labels_by_frame=None,
    points_prompt_idx=None,
    save_video=False,
    save_side_by_side=False,
    max_frames=None,
    save_npz=False,
    npz_separate=False,
    no_pkl=False,
    debug_one=False,
    invert_mask=False,
    postprocess_for_vis=False,
    pp_min_hole_size=64,
    pp_min_object_size=50,
    pp_closing_iterations=1,
    pp_no_fill_holes=False,
    pp_no_remove_small_objects=False,
    pp_union_hole_fill=False,
    pp_union_gap_fill=False,
    pp_union_gap_closing_iterations=1,
    pp_fill_blue_table_quadrant=False,
    pp_blue_table_label=1,
    pp_blue_table_target=4,
    pp_blue_table_quadrant_mode="right_down",
    pp_blue_table_y_pad_top=60,
    pp_blue_table_y_pad_bottom=60,
    pp_blue_table_skip_if_label_above=None,
    pp_blue_table_skip_if_label_area_gt=None,
    pp_fill_interior_class=None,
    pp_fill_interior_target=4,
    pp_per_camera=None,
    skip_if_exists=False,
    skip_if_masks_dir=None,
    static_prompts=None,
    pp_topleft_rect=False,
    pp_topleft_rect_label=1,
    pp_topleft_rect_fill=5,
    pp_topleft_rect_y_max=420,
    pp_topleft_rect_frame_start=10,
    pp_topleft_rect_frame_end_ratio=0.667,
    pp_topright_rect=False,
    pp_topright_rect_label=1,
    pp_topright_rect_fill=5,
    pp_topright_rect_y_max=420,
    pp_topright_rect_y_threshold=200,
    pp_topright_rect_frame_start_ratio=0.667,
    pp_leftmost_rect=False,
    pp_leftmost_rect_label=1,
    pp_leftmost_rect_fill=5,
    pp_leftmost_rect_y_max=420,
    pp_leftmost_rect_skip_label=3,
    pp_halves_rect=False,
    pp_halves_rect_label=1,
    pp_halves_rect_fill=5,
    pp_halves_rect_y_max=420,
    point_clicks=None,
    prompt_extra_frames_str="",
):
    """Worker function for GPU-based segmentation inference (PKL only, no video)"""
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"[Worker GPU {gpu_id}] Starting processing {len(video_list)} videos...")
    
    produce_masks_script = os.path.join(script_dir, "produce_masks.py")
    
    for video_path in video_list:
        # Determine output directory structure
        # video_path: .../videos/chunk-000/{camera_name}/{video_file}.mp4
        
        parts = video_path.split(os.sep)
        try:
            camera_name = parts[-2]
            output_dir = os.path.join(base_output_dir, camera_name)
        except IndexError:
            print(f"[Worker GPU {gpu_id}] Could not parse path structure for {video_path}, using flat output.")
            output_dir = base_output_dir
            camera_name = ""

        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Check if output already exists
        if skip_if_exists:
            expected_output = os.path.join(output_dir, f"{video_name}_masks.npz")
            expected_post_output = os.path.join(output_dir, f"{video_name}_masks_post.npz")
            if os.path.exists(expected_output) or os.path.exists(expected_post_output):
                print(f"[Worker GPU {gpu_id}] Skipping {video_path} (Output exists: {expected_output} or {expected_post_output})")
                continue
            if skip_if_masks_dir and camera_name:
                masks_path = os.path.join(skip_if_masks_dir, "chunk-000", camera_name, f"{video_name}_masks.npz")
                if os.path.exists(masks_path):
                    print(f"[Worker GPU {gpu_id}] Skipping {video_path} (Masks exist: {masks_path})")
                    continue

        print(f"[Worker GPU {gpu_id}] Processing: {video_path} -> {output_dir}")
        
        # Build command: inference only, no video generation
        cmd = [
            "python", produce_masks_script,
            "--video_path", video_path,
            "--checkpoint_path", checkpoint,
            "--output_dir", output_dir,
            "--prompts"
        ] + prompts

        # Per-episode point clicks override global points
        # Supports "_default" key as fallback for all episodes in a camera
        # Format: list of {assist_prompt_idx, frame_idx, points} → --assist_points
        #     or: dict {frame_idx: {points, labels}}              → --points_by_frame (legacy)
        ep_clicks = None
        if point_clicks and camera_name:
            cam_clicks = point_clicks.get(camera_name, {})
            ep_clicks = cam_clicks.get(video_name, cam_clicks.get("_default", None))

        ep_prompt_idx = points_prompt_idx
        if isinstance(ep_clicks, list):
            # Assist mode: points are injected into text prompt sessions
            assist_parts = []
            for entry in ep_clicks:
                pidx = entry["assist_prompt_idx"]
                fidx = entry.get("frame_idx", 0)
                pts = entry["points"]
                d = entry.get("direction", "")
                assist_parts.append(f"{pidx}:{fidx}:{pts}:{d}" if d else f"{pidx}:{fidx}:{pts}")
            cmd += ["--assist_points", "|".join(assist_parts)]
            print(f"[Worker GPU {gpu_id}]   Assist points for {camera_name}/{video_name}: {len(ep_clicks)} group(s)")
        elif isinstance(ep_clicks, dict) and ep_clicks:
            # Legacy mode: separate SAM2 tracker pass
            pbf_parts = []
            plbf_parts = []
            frame_count = 0
            mask_label = None
            for frame_idx, entry in ep_clicks.items():
                if not isinstance(entry, dict) or "points" not in entry:
                    continue
                pbf_parts.append(f"{frame_idx}:{entry['points']}")
                n_pts = len(entry['points'].split(';'))
                plbf_parts.append(f"{frame_idx}:{','.join(['1'] * n_pts)}")
                if mask_label is None:
                    first_label = int(entry['labels'].split(',')[0])
                    mask_label = first_label
                frame_count += 1
            if mask_label is not None:
                ep_prompt_idx = mask_label - 1
            cmd += ["--points_by_frame", "|".join(pbf_parts)]
            cmd += ["--point_labels_by_frame", "|".join(plbf_parts)]
            print(f"[Worker GPU {gpu_id}]   Using point_clicks for {camera_name}/{video_name} ({frame_count} frames, mask_label={mask_label}, prompt_idx={ep_prompt_idx})")
        else:
            if points:
                cmd += ["--points", points]
            if point_labels:
                cmd += ["--point_labels", point_labels]
            if points_frame_idx is not None:
                cmd += ["--points_frame_idx", str(points_frame_idx)]
            if points_by_frame:
                cmd += ["--points_by_frame", points_by_frame]
            if point_labels_by_frame:
                cmd += ["--point_labels_by_frame", point_labels_by_frame]
        if ep_prompt_idx is not None:
            cmd += ["--points_prompt_idx", str(ep_prompt_idx)]
        if prompt_extra_frames_str:
            cmd += ["--prompt_extra_frames", prompt_extra_frames_str]
        if save_video:
            cmd += ["--save_video"]
        if save_side_by_side:
            cmd += ["--save_side_by_side"]
        if max_frames is not None:
            cmd += ["--max_frames", str(max_frames)]
        if save_npz:
            cmd += ["--save_npz"]
        if npz_separate:
            cmd += ["--npz_separate"]
        if no_pkl:
            cmd += ["--no_pkl"]
        if invert_mask:
            cmd += ["--invert_mask"]
        if static_prompts:
            cmd += ["--static_prompts"] + list(static_prompts)
        if postprocess_for_vis:
            cam_classes = pp_fill_interior_class
            cam_target = pp_fill_interior_target
            if pp_per_camera and camera_name in pp_per_camera:
                cam_classes_override, cam_target_override = pp_per_camera[camera_name]
                if cam_classes_override is not None:
                    cam_classes = cam_classes_override
                if cam_target_override is not None:
                    cam_target = cam_target_override

            cmd += [
                "--postprocess_for_vis",
                "--pp_min_hole_size",
                str(pp_min_hole_size),
                "--pp_min_object_size",
                str(pp_min_object_size),
                "--pp_closing_iterations",
                str(pp_closing_iterations),
            ]
            if pp_no_fill_holes:
                cmd += ["--pp_no_fill_holes"]
            if pp_no_remove_small_objects:
                cmd += ["--pp_no_remove_small_objects"]
            if pp_union_hole_fill:
                cmd += ["--pp_union_hole_fill"]
            if pp_union_gap_fill:
                cmd += ["--pp_union_gap_fill"]
                cmd += ["--pp_union_gap_closing_iterations", str(pp_union_gap_closing_iterations)]
            if pp_fill_blue_table_quadrant:
                cmd += ["--pp_fill_blue_table_quadrant"]
                cmd += ["--pp_blue_table_label", str(pp_blue_table_label)]
                cmd += ["--pp_blue_table_target", str(pp_blue_table_target)]
                cmd += ["--pp_blue_table_quadrant_mode", str(pp_blue_table_quadrant_mode)]
                cmd += ["--pp_blue_table_y_pad_top", str(pp_blue_table_y_pad_top)]
                cmd += ["--pp_blue_table_y_pad_bottom", str(pp_blue_table_y_pad_bottom)]
                if pp_blue_table_skip_if_label_above is not None:
                    cmd += ["--pp_blue_table_skip_if_label_above", str(pp_blue_table_skip_if_label_above)]
                if pp_blue_table_skip_if_label_area_gt is not None:
                    cmd += ["--pp_blue_table_skip_if_label_area_gt", str(pp_blue_table_skip_if_label_area_gt)]
            if cam_classes:
                cmd += ["--pp_fill_interior_class", ",".join(str(x) for x in cam_classes)]
                cmd += ["--pp_fill_interior_target", str(cam_target)]
            if pp_topleft_rect:
                cmd += [
                    "--pp_topleft_rect",
                    "--pp_topleft_rect_label", str(pp_topleft_rect_label),
                    "--pp_topleft_rect_fill", str(pp_topleft_rect_fill),
                    "--pp_topleft_rect_y_max", str(pp_topleft_rect_y_max),
                    "--pp_topleft_rect_frame_start", str(pp_topleft_rect_frame_start),
                    "--pp_topleft_rect_frame_end_ratio", str(pp_topleft_rect_frame_end_ratio),
                ]
            if pp_topright_rect:
                cmd += [
                    "--pp_topright_rect",
                    "--pp_topright_rect_label", str(pp_topright_rect_label),
                    "--pp_topright_rect_fill", str(pp_topright_rect_fill),
                    "--pp_topright_rect_y_max", str(pp_topright_rect_y_max),
                    "--pp_topright_rect_y_threshold", str(pp_topright_rect_y_threshold),
                    "--pp_topright_rect_frame_start_ratio", str(pp_topright_rect_frame_start_ratio),
                ]
            if pp_leftmost_rect:
                cmd += [
                    "--pp_leftmost_rect",
                    "--pp_leftmost_rect_label", str(pp_leftmost_rect_label),
                    "--pp_leftmost_rect_fill", str(pp_leftmost_rect_fill),
                    "--pp_leftmost_rect_y_max", str(pp_leftmost_rect_y_max),
                    "--pp_leftmost_rect_skip_label", str(pp_leftmost_rect_skip_label),
                ]
            if pp_halves_rect:
                cmd += [
                    "--pp_halves_rect",
                    "--pp_halves_rect_label", str(pp_halves_rect_label),
                    "--pp_halves_rect_fill", str(pp_halves_rect_fill),
                    "--pp_halves_rect_y_max", str(pp_halves_rect_y_max),
                ]

        try:
            subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[Worker GPU {gpu_id}] ERROR processing {video_path}: {e}")

    print(f"[Worker GPU {gpu_id}] Finished.")


def _check_single_npz(filepath):
    """Check a single npz file. Returns (filepath, error_msg) if corrupt, else (filepath, None)."""
    import numpy as np
    try:
        data = np.load(filepath)
        _ = data['arr_0'].shape
        data.close()
        return filepath, None
    except Exception as e:
        return filepath, str(e)


def scan_and_remove_corrupt_npz(output_dir, cameras, num_workers=32):
    """Scan existing *_masks.npz files and remove corrupt ones before distributing work."""
    all_files = []
    for cam in cameras:
        cam_dir = os.path.join(output_dir, cam)
        if not os.path.isdir(cam_dir):
            continue
        all_files.extend(glob.glob(os.path.join(cam_dir, "*_masks.npz")))

    if not all_files:
        print("Integrity scan: no npz files to check.")
        return 0

    removed = 0
    n = min(num_workers, len(all_files))
    with multiprocessing.Pool(n) as pool:
        for filepath, error in pool.imap_unordered(_check_single_npz, all_files, chunksize=16):
            if error is not None:
                print(f"  [CORRUPT] Removing {filepath}  ({error})")
                os.remove(filepath)
                removed += 1

    print(f"Integrity scan: checked {len(all_files)} npz files, removed {removed} corrupt.")
    return removed


def filter_videos_skip_existing(all_videos, base_output_dir, skip_if_masks_dir=None):
    """Pre-filter videos: only return those that still need processing."""
    needed = []
    skipped = 0
    for video_path in all_videos:
        parts = video_path.split(os.sep)
        try:
            camera_name = parts[-2]
            output_dir = os.path.join(base_output_dir, camera_name)
        except IndexError:
            output_dir = base_output_dir
            camera_name = ""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        expected_output = os.path.join(output_dir, f"{video_name}_masks.npz")
        expected_post = os.path.join(output_dir, f"{video_name}_masks_post.npz")
        if os.path.exists(expected_output) or os.path.exists(expected_post):
            skipped += 1
            continue
        if skip_if_masks_dir and camera_name:
            masks_path = os.path.join(skip_if_masks_dir, "chunk-000", camera_name, f"{video_name}_masks.npz")
            if os.path.exists(masks_path):
                skipped += 1
                continue
        needed.append(video_path)
    if skipped:
        print(f"Skipped {skipped} videos (outputs already exist), {len(needed)} remaining.")
    return needed


def run_worker_batch(
    gpu_id,
    video_list,
    checkpoint,
    prompts,
    base_output_dir,
    point_clicks,
    prompt_extra_frames_str,
    save_video,
    save_side_by_side,
    max_frames,
    save_npz,
    npz_separate,
    no_pkl,
    invert_mask,
    postprocess_for_vis,
    pp_min_hole_size,
    pp_min_object_size,
    pp_closing_iterations,
    pp_no_fill_holes,
    pp_no_remove_small_objects,
    pp_union_hole_fill,
    pp_union_gap_fill,
    pp_union_gap_closing_iterations,
    pp_fill_blue_table_quadrant,
    pp_blue_table_label,
    pp_blue_table_target,
    pp_blue_table_quadrant_mode,
    pp_blue_table_y_pad_top,
    pp_blue_table_y_pad_bottom,
    pp_blue_table_skip_if_label_above,
    pp_blue_table_skip_if_label_area_gt,
    pp_fill_interior_class,
    pp_fill_interior_target,
    pp_per_camera,
    static_prompts,
    pp_topleft_rect,
    pp_topleft_rect_label,
    pp_topleft_rect_fill,
    pp_topleft_rect_y_max,
    pp_topleft_rect_frame_start,
    pp_topleft_rect_frame_end_ratio,
    pp_topright_rect,
    pp_topright_rect_label,
    pp_topright_rect_fill,
    pp_topright_rect_y_max,
    pp_topright_rect_y_threshold,
    pp_topright_rect_frame_start_ratio,
    pp_leftmost_rect,
    pp_leftmost_rect_label,
    pp_leftmost_rect_fill,
    pp_leftmost_rect_y_max,
    pp_leftmost_rect_skip_label,
    pp_halves_rect,
    pp_halves_rect_label,
    pp_halves_rect_fill,
    pp_halves_rect_y_max,
    points=None,
    point_labels=None,
    points_frame_idx=None,
    points_by_frame=None,
    point_labels_by_frame=None,
    points_prompt_idx=None,
):
    """Batch worker: load model ONCE, then process all assigned videos in-process."""
    import argparse as _argparse

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from produce_masks import process_single_video
    from sam3.model_builder import build_sam3_video_predictor

    print(f"[Worker GPU {gpu_id}] Loading SAM3 model (once for {len(video_list)} videos)...")
    video_predictor = build_sam3_video_predictor(checkpoint)
    print(f"[Worker GPU {gpu_id}] Model loaded. Starting batch processing...")

    for vid_idx, video_path in enumerate(video_list):
        parts = video_path.split(os.sep)
        try:
            camera_name = parts[-2]
            output_dir = os.path.join(base_output_dir, camera_name)
        except IndexError:
            output_dir = base_output_dir
            camera_name = ""
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Resolve per-camera fill classes
        fill_class = pp_fill_interior_class
        fill_target = pp_fill_interior_target
        if pp_per_camera and camera_name in pp_per_camera:
            cam_classes, cam_target = pp_per_camera[camera_name]
            if cam_classes is not None:
                fill_class = cam_classes
            if cam_target is not None:
                fill_target = cam_target
        fill_class_str = ",".join(str(c) for c in fill_class) if fill_class else None

        # Resolve per-episode point clicks → assist_points string
        assist_points_str = ""
        ep_local_points = points
        ep_local_labels = point_labels
        ep_local_frame_idx = points_frame_idx
        ep_local_by_frame = points_by_frame
        ep_local_labels_by_frame = point_labels_by_frame
        ep_local_prompt_idx = points_prompt_idx

        if point_clicks and camera_name:
            cam_clicks = point_clicks.get(camera_name, {})
            ep_clicks = cam_clicks.get(video_name, cam_clicks.get("_default", None))
            if isinstance(ep_clicks, list):
                assist_parts = []
                for entry in ep_clicks:
                    pidx = entry["assist_prompt_idx"]
                    fidx = entry.get("frame_idx", 0)
                    pts = entry["points"]
                    d = entry.get("direction", "")
                    assist_parts.append(f"{pidx}:{fidx}:{pts}:{d}" if d else f"{pidx}:{fidx}:{pts}")
                assist_points_str = "|".join(assist_parts)
                ep_local_points = None
                ep_local_labels = None
            elif isinstance(ep_clicks, dict) and ep_clicks:
                pbf_parts = []
                plbf_parts = []
                for fkey, fval in sorted(ep_clicks.items(), key=lambda kv: int(kv[0])):
                    pbf_parts.append(f"{fkey}: {fval['points']}")
                    plbf_parts.append(f"{fkey}: {fval['labels']}")
                ep_local_by_frame = "|".join(pbf_parts)
                ep_local_labels_by_frame = "|".join(plbf_parts)
                ep_local_prompt_idx = ep_clicks.get(list(ep_clicks.keys())[0], {}).get("prompt_idx", points_prompt_idx)
                ep_local_points = None
                ep_local_labels = None

        os.makedirs(output_dir, exist_ok=True)

        args = _argparse.Namespace(
            checkpoint_path=checkpoint,
            video_path=video_path,
            output_dir=output_dir,
            prompts=prompts,
            fps=30,
            save_video=save_video,
            save_side_by_side=save_side_by_side,
            max_frames=max_frames,
            save_npz=save_npz,
            npz_separate=npz_separate,
            no_pkl=no_pkl,
            invert_mask=invert_mask,
            static_prompts=static_prompts,
            points=ep_local_points or "",
            point_labels=ep_local_labels or "",
            points_frame_idx=ep_local_frame_idx if ep_local_frame_idx is not None else 0,
            points_by_frame=ep_local_by_frame or "",
            point_labels_by_frame=ep_local_labels_by_frame or "",
            points_prompt_idx=ep_local_prompt_idx,
            assist_points=assist_points_str,
            prompt_extra_frames=prompt_extra_frames_str,
            postprocess_for_vis=postprocess_for_vis,
            postprocess_only=False,
            pp_min_hole_size=pp_min_hole_size,
            pp_min_object_size=pp_min_object_size,
            pp_closing_iterations=pp_closing_iterations,
            pp_no_fill_holes=pp_no_fill_holes,
            pp_no_remove_small_objects=pp_no_remove_small_objects,
            pp_union_hole_fill=pp_union_hole_fill,
            pp_union_gap_fill=pp_union_gap_fill,
            pp_union_gap_closing_iterations=pp_union_gap_closing_iterations,
            pp_fill_blue_table_quadrant=pp_fill_blue_table_quadrant,
            pp_blue_table_label=pp_blue_table_label,
            pp_blue_table_target=pp_blue_table_target,
            pp_blue_table_quadrant_mode=pp_blue_table_quadrant_mode,
            pp_blue_table_y_pad_top=pp_blue_table_y_pad_top,
            pp_blue_table_y_pad_bottom=pp_blue_table_y_pad_bottom,
            pp_blue_table_skip_if_label_above=pp_blue_table_skip_if_label_above,
            pp_blue_table_skip_if_label_area_gt=pp_blue_table_skip_if_label_area_gt,
            pp_fill_interior_class=fill_class_str,
            pp_fill_interior_target=fill_target,
            pp_overwrite=False,
            pp_fill_bg_roi=None,
            pp_scanline_fill=False,
            pp_scanline_source_label=1,
            pp_scanline_fill_value=3,
            pp_topleft_rect=pp_topleft_rect,
            pp_topleft_rect_label=pp_topleft_rect_label,
            pp_topleft_rect_fill=pp_topleft_rect_fill,
            pp_topleft_rect_y_max=pp_topleft_rect_y_max,
            pp_topleft_rect_frame_start=pp_topleft_rect_frame_start,
            pp_topleft_rect_frame_end_ratio=pp_topleft_rect_frame_end_ratio,
            pp_topright_rect=pp_topright_rect,
            pp_topright_rect_label=pp_topright_rect_label,
            pp_topright_rect_fill=pp_topright_rect_fill,
            pp_topright_rect_y_max=pp_topright_rect_y_max,
            pp_topright_rect_y_threshold=pp_topright_rect_y_threshold,
            pp_topright_rect_frame_start_ratio=pp_topright_rect_frame_start_ratio,
            pp_leftmost_rect=pp_leftmost_rect,
            pp_leftmost_rect_label=pp_leftmost_rect_label,
            pp_leftmost_rect_fill=pp_leftmost_rect_fill,
            pp_leftmost_rect_y_max=pp_leftmost_rect_y_max,
            pp_leftmost_rect_skip_label=pp_leftmost_rect_skip_label,
            pp_halves_rect=pp_halves_rect,
            pp_halves_rect_label=pp_halves_rect_label,
            pp_halves_rect_fill=pp_halves_rect_fill,
            pp_halves_rect_y_max=pp_halves_rect_y_max,
        )

        print(f"[Worker GPU {gpu_id}] ({vid_idx+1}/{len(video_list)}) {video_path}")
        try:
            process_single_video(video_predictor, args)
        except Exception as e:
            print(f"[Worker GPU {gpu_id}] ERROR processing {video_path}: {e}")
            import traceback
            traceback.print_exc()

    print(f"[Worker GPU {gpu_id}] Finished all {len(video_list)} videos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel SAM3 segmentation across multiple GPUs (PKL output only)")
    parser.add_argument("--base_dir", required=True, help="Base directory containing camera folders")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM3 checkpoint")
    parser.add_argument("--output_dir", required=True, help="Base output directory")
    parser.add_argument("--prompts", nargs="+", required=True, help="Prompts list")
    parser.add_argument("--cameras", nargs="+", required=True, help="List of camera folder names to scan")
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=None, help="Specific GPU IDs to use. If None, uses all available.")
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="Number of parallel workers per GPU (default: 1).")
    parser.add_argument("--points", type=str, default=None, help="Extra points as 'x1,y1;x2,y2;...'.")
    parser.add_argument("--point_labels", type=str, default=None, help="Point labels as '1,0,1,...'.")
    parser.add_argument("--points_frame_idx", type=int, default=None, help="Frame index for points.")
    parser.add_argument("--points_by_frame", type=str, default=None, help="Multiple frames: 'frame: x1,y1;...|frame: x1,y1;...'.")
    parser.add_argument("--point_labels_by_frame", type=str, default=None, help="Labels per frame: 'frame:1,0|frame:1,1'.")
    parser.add_argument("--points_prompt_idx", type=int, default=None, help="Assign point-click masks to this prompt index (0-based).")
    parser.add_argument("--save_video", action="store_true", help="Save visualization videos.")
    parser.add_argument("--save_side_by_side", action="store_true", help="Save side-by-side videos.")
    parser.add_argument("--max_frames", type=int, default=None, help="Only process first N frames.")
    parser.add_argument("--save_npz", action="store_true", help="Also save Cosmos npz outputs.")
    parser.add_argument("--npz_separate", action="store_true", help="Keep objects separate in npz.")
    parser.add_argument("--no_pkl", action="store_true", help="Do not save pkl outputs.")
    parser.add_argument("--debug_one", action="store_true", help="Randomly pick one video and run once.")
    parser.add_argument("--debug_n", type=int, default=None, help="Randomly pick N videos and run (overrides debug_one).")
    parser.add_argument("--debug_seed", type=int, default=None, help="Random seed for debug_one/debug_n.")
    parser.add_argument("--invert_mask", action="store_true", help="Invert masks in outputs.")
    parser.add_argument("--postprocess", action="store_true", help="Run postprocess on npz masks after segmentation.")
    parser.add_argument("--pp_min_hole_size", type=int, default=64, help="Postprocess: fill holes smaller than this.")
    parser.add_argument("--pp_min_object_size", type=int, default=50, help="Postprocess: remove objects smaller than this.")
    parser.add_argument("--pp_closing_iterations", type=int, default=1, help="Postprocess: closing iterations.")
    parser.add_argument("--pp_no_fill_holes", action="store_true", help="Postprocess: disable hole filling.")
    parser.add_argument("--pp_no_remove_small_objects", action="store_true", help="Postprocess: disable removing small objects.")
    parser.add_argument("--pp_union_hole_fill", action="store_true", help="Postprocess: fill holes based on union of all >0 classes.")
    parser.add_argument("--pp_union_gap_fill", action="store_true", help="Postprocess: fill thin background gaps using union closing.")
    parser.add_argument("--pp_union_gap_closing_iterations", type=int, default=1, help="Postprocess: union gap fill closing iterations.")
    parser.add_argument("--pp_fill_blue_table_quadrant", action="store_true", help="Postprocess: fill black region in blue table quadrant.")
    parser.add_argument("--pp_blue_table_label", type=int, default=1, help="Postprocess: blue table label id.")
    parser.add_argument("--pp_blue_table_target", type=int, default=4, help="Postprocess: target label for blue table quadrant fill.")
    parser.add_argument("--pp_blue_table_quadrant_mode", type=str, default="right_down", help="Postprocess: quadrant fill mode.")
    parser.add_argument("--pp_blue_table_y_pad_top", type=int, default=60, help="Postprocess: top padding rows to exclude.")
    parser.add_argument("--pp_blue_table_y_pad_bottom", type=int, default=60, help="Postprocess: bottom padding rows to exclude.")
    parser.add_argument("--pp_blue_table_skip_if_label_above", type=int, default=None, help="Postprocess: skip fill if label above blue table.")
    parser.add_argument("--pp_blue_table_skip_if_label_area_gt", type=int, default=None, help="Postprocess: skip fill if label area exceeds this size.")
    parser.add_argument("--pp_fill_interior_class", type=str, default=None, help='Postprocess: fill background inside these class contours, e.g. "1,3".')
    parser.add_argument("--pp_fill_interior_target", type=int, default=4, help="Postprocess: target class for interior fill.")
    parser.add_argument("--pp_fill_table_top_line", action="store_true", help="Postprocess: fill background inside table top-line closed region.")
    parser.add_argument("--pp_table_top_label", type=int, default=1, help="Postprocess: table label for fill_table_top_line (default: 1).")
    parser.add_argument("--pp_table_top_fill_target", type=int, default=6, help="Postprocess: fill target for fill_table_top_line (default: 6).")
    parser.add_argument("--pp_table_top_corner_ranges", type=str, default=None,
                        help='Corner ROI ranges for fill_table_top_line. Format: '
                             '"tl_x0,tl_x1,tl_y0,tl_y1;tr_x0,tr_x1,tr_y0,tr_y1;bl_x0,bl_x1,bl_y0,bl_y1;br_x0,br_x1,br_y0,br_y1"')
    parser.add_argument("--pp_scanline_fill", action="store_true", help="Postprocess: row-based fill between first/last source_label pixels.")
    parser.add_argument("--pp_scanline_source_label", type=int, default=1, help="Postprocess: label to scan for in scanline fill (default: 1).")
    parser.add_argument("--pp_scanline_fill_value", type=int, default=3, help="Postprocess: value to fill background with in scanline fill (default: 3).")
    parser.add_argument("--pp_topleft_rect", action="store_true", help="Postprocess: fill bg rect left-below top-left corner of a label.")
    parser.add_argument("--pp_topleft_rect_label", type=int, default=1)
    parser.add_argument("--pp_topleft_rect_fill", type=int, default=5)
    parser.add_argument("--pp_topleft_rect_y_max", type=int, default=420)
    parser.add_argument("--pp_topleft_rect_frame_start", type=int, default=10)
    parser.add_argument("--pp_topleft_rect_frame_end_ratio", type=float, default=0.667)
    parser.add_argument("--pp_topright_rect", action="store_true", help="Postprocess: fill bg rect left-below top-right corner (last 1/3).")
    parser.add_argument("--pp_topright_rect_label", type=int, default=1)
    parser.add_argument("--pp_topright_rect_fill", type=int, default=5)
    parser.add_argument("--pp_topright_rect_y_max", type=int, default=420)
    parser.add_argument("--pp_topright_rect_y_threshold", type=int, default=200)
    parser.add_argument("--pp_topright_rect_frame_start_ratio", type=float, default=0.667)
    parser.add_argument("--pp_leftmost_rect", action="store_true",
                        help="Fill bg below-left of leftmost label pixel. First half: always. Second half: only if skip_label absent.")
    parser.add_argument("--pp_leftmost_rect_label", type=int, default=1)
    parser.add_argument("--pp_leftmost_rect_fill", type=int, default=5)
    parser.add_argument("--pp_leftmost_rect_y_max", type=int, default=420)
    parser.add_argument("--pp_leftmost_rect_skip_label", type=int, default=3)
    parser.add_argument("--pp_halves_rect", action="store_true",
                        help="First half: topleft rect fill. Second half: topleft-topright rect fill.")
    parser.add_argument("--pp_halves_rect_label", type=int, default=1)
    parser.add_argument("--pp_halves_rect_fill", type=int, default=5)
    parser.add_argument("--pp_halves_rect_y_max", type=int, default=420)
    parser.add_argument("--pp_overwrite", action="store_true", help="Postprocess: overwrite existing *_masks_post.npz.")
    parser.add_argument(
        "--pp_per_camera",
        type=str,
        default=None,
        help='Postprocess per camera. Format: "camera=1,3:1;camera2=2:2".',
    )
    parser.add_argument(
        "--postprocess_for_vis",
        action="store_true",
        help="Apply postprocess to masks for visualization only.",
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip processing if output mask file already exists.",
    )
    parser.add_argument(
        "--skip_if_masks_dir",
        type=str,
        default=None,
        help="Additional masks directory to check when --skip_if_exists is set. "
             "Structure: {dir}/chunk-000/{camera}/{episode}_masks.npz",
    )
    parser.add_argument(
        "--static_prompts",
        nargs="*",
        default=None,
        help="Prompts whose mask is static (segment frame 0 only, replicate to all frames).",
    )
    parser.add_argument(
        "--pp_num_workers",
        type=int,
        default=8,
        help="Number of parallel workers for postprocessing (default: 8).",
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
        "--point_clicks_json",
        type=str,
        default=None,
        help='JSON file with per-episode point clicks. '
             'Format: {"camera": {"episode": {"frame_idx": {"points": "x,y;...", "labels": "l,..."}}}}',
    )
    parser.add_argument("--pp_corner_rect", action="store_true",
                        help="Fill bg rect from top-corner of label(s).")
    parser.add_argument("--pp_corner_rect_mode", type=str, default="topright",
                        help='Corner mode: "topright" or "topleft" (default: topright).')
    parser.add_argument("--pp_corner_rect_labels", type=str, default=None,
                        help='Comma-separated source labels, e.g. "1,3".')
    parser.add_argument("--pp_corner_rect_fill", type=int, default=4,
                        help="Fill value for corner rect (default: 4).")
    parser.add_argument("--pp_corner_rect_y_max", type=int, default=420,
                        help="Bottom boundary of fill rect (default: 420).")
    parser.add_argument("--pp_corner_rect_x_first_labels", type=str, default=None,
                        help='Comma-separated labels using x-first anchor, e.g. "3".')
    parser.add_argument("--pp_temporal_fill", action="store_true",
                        help="Forward temporal fill: propagate labels to next frame bg pixels.")
    parser.add_argument("--pp_temporal_fill_labels", type=str, default=None,
                        help='Comma-separated labels for forward/backward propagation, e.g. "1,5".')
    parser.add_argument("--pp_temporal_fill_union_labels", type=str, default=None,
                        help='Comma-separated labels for union fill, e.g. "3,4".')
    parser.add_argument("--pp_temporal_fill_value", type=int, default=5,
                        help="Fill value for temporal fill (default: 5).")
    parser.add_argument("--legacy_subprocess", action="store_true",
                        help="Use legacy subprocess-per-video mode instead of batch worker.")

    args = parser.parse_args()
    
    # Load per-episode point clicks if provided
    point_clicks = None
    prompt_extra_frames_str = ""
    if args.point_clicks_json:
        import json
        with open(args.point_clicks_json) as f:
            point_clicks = json.load(f)
        # Extract prompt_extra_frames (global, not per-camera)
        pef = point_clicks.pop("prompt_extra_frames", None)
        if pef:
            # Convert {"2": [-1], "0": [0, 100]} → "2:-1|0:0|0:100"
            parts = []
            for pidx, frames in pef.items():
                for f in frames:
                    parts.append(f"{pidx}:{f}")
            prompt_extra_frames_str = "|".join(parts)
            print(f"Prompt extra frames: {pef}")
        total_eps = sum(
            len(eps) for k, eps in point_clicks.items()
            if isinstance(eps, dict)
        )
        print(f"Loaded point_clicks_json: {total_eps} episode(s) across {len(point_clicks)} camera(s)")
    
    # Get script directory (where this python file is located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Collect all videos
    all_videos = []
    for cam in args.cameras:
        search_path = os.path.join(args.base_dir, cam, "*.mp4")
        videos = glob.glob(search_path)
        all_videos.extend(videos)
        print(f"Found {len(videos)} videos in {cam}")
    
    if not all_videos:
        print("No videos found. Exiting.")
        exit(0)
        
    print(f"Total videos to process: {len(all_videos)}")
    
    # Debug mode: pick random video(s)
    debug_sample = args.debug_n if args.debug_n else (1 if args.debug_one else 0)
    if debug_sample > 0:
        import random
        if args.debug_seed is not None:
            random.seed(args.debug_seed)
        n = min(debug_sample, len(all_videos))
        all_videos = random.sample(all_videos, n)
        print(f"Debug mode: sampled {n} video(s)")

    # Integrity scan: detect and remove corrupt npz files before filtering
    if args.skip_if_exists:
        print("Running integrity scan on existing outputs...")
        scan_and_remove_corrupt_npz(args.output_dir, args.cameras)

    # Upfront skip-if-exists filtering (before distributing work)
    if args.skip_if_exists:
        all_videos = filter_videos_skip_existing(
            all_videos, args.output_dir, args.skip_if_masks_dir
        )
        if not all_videos:
            print("All videos already processed. Nothing to do.")
            # Still run postprocess if requested
            if not args.postprocess:
                exit(0)

    # 2. Assign GPUs
    if args.gpu_ids:
        gpu_ids = args.gpu_ids
    else:
        gpu_ids = get_available_gpus()
    
    print(f"Using GPUs: {gpu_ids} (workers_per_gpu={args.workers_per_gpu})")
    
    # 3. Distribute work across GPUs (repeat each GPU for multiple workers)
    if debug_sample == 1:
        gpu_ids = [gpu_ids[0]]
    worker_gpu_ids = [gid for gid in gpu_ids for _ in range(args.workers_per_gpu)]
    chunks = chunk_list(all_videos, len(worker_gpu_ids))
    
    per_camera = _parse_pp_per_camera(args.pp_per_camera)
    default_classes = _parse_class_list(args.pp_fill_interior_class)
    processes = []

    if args.legacy_subprocess:
        for i, gpu_id in enumerate(worker_gpu_ids):
            if i < len(chunks) and chunks[i]:
                p = multiprocessing.Process(
                    target=run_worker,
                    args=(
                        gpu_id,
                        chunks[i],
                        args.checkpoint,
                        args.prompts,
                        args.output_dir,
                        script_dir,
                        args.points,
                        args.point_labels,
                        args.points_frame_idx,
                        args.points_by_frame,
                        args.point_labels_by_frame,
                        args.points_prompt_idx,
                        args.save_video,
                        args.save_side_by_side,
                        args.max_frames,
                        args.save_npz,
                        args.npz_separate,
                        args.no_pkl,
                        args.debug_one,
                        args.invert_mask,
                        args.postprocess_for_vis,
                        args.pp_min_hole_size,
                        args.pp_min_object_size,
                        args.pp_closing_iterations,
                        args.pp_no_fill_holes,
                        args.pp_no_remove_small_objects,
                        args.pp_union_hole_fill,
                        args.pp_union_gap_fill,
                        args.pp_union_gap_closing_iterations,
                        args.pp_fill_blue_table_quadrant,
                        args.pp_blue_table_label,
                        args.pp_blue_table_target,
                        args.pp_blue_table_quadrant_mode,
                        args.pp_blue_table_y_pad_top,
                        args.pp_blue_table_y_pad_bottom,
                        args.pp_blue_table_skip_if_label_above,
                        args.pp_blue_table_skip_if_label_area_gt,
                        default_classes,
                        args.pp_fill_interior_target,
                        per_camera,
                        False,  # skip_if_exists already done upfront
                        None,   # skip_if_masks_dir already done upfront
                        args.static_prompts,
                        args.pp_topleft_rect,
                        args.pp_topleft_rect_label,
                        args.pp_topleft_rect_fill,
                        args.pp_topleft_rect_y_max,
                        args.pp_topleft_rect_frame_start,
                        args.pp_topleft_rect_frame_end_ratio,
                        args.pp_topright_rect,
                        args.pp_topright_rect_label,
                        args.pp_topright_rect_fill,
                        args.pp_topright_rect_y_max,
                        args.pp_topright_rect_y_threshold,
                        args.pp_topright_rect_frame_start_ratio,
                        args.pp_leftmost_rect,
                        args.pp_leftmost_rect_label,
                        args.pp_leftmost_rect_fill,
                        args.pp_leftmost_rect_y_max,
                        args.pp_leftmost_rect_skip_label,
                        args.pp_halves_rect,
                        args.pp_halves_rect_label,
                        args.pp_halves_rect_fill,
                        args.pp_halves_rect_y_max,
                        point_clicks,
                        prompt_extra_frames_str,
                    )
                )
                processes.append(p)
                p.start()
    else:
        for i, gpu_id in enumerate(worker_gpu_ids):
            if i < len(chunks) and chunks[i]:
                p = multiprocessing.Process(
                    target=run_worker_batch,
                    args=(
                        gpu_id,
                        chunks[i],
                        args.checkpoint,
                        args.prompts,
                        args.output_dir,
                        point_clicks,
                        prompt_extra_frames_str,
                        args.save_video,
                        args.save_side_by_side,
                        args.max_frames,
                        args.save_npz,
                        args.npz_separate,
                        args.no_pkl,
                        args.invert_mask,
                        args.postprocess_for_vis,
                        args.pp_min_hole_size,
                        args.pp_min_object_size,
                        args.pp_closing_iterations,
                        args.pp_no_fill_holes,
                        args.pp_no_remove_small_objects,
                        args.pp_union_hole_fill,
                        args.pp_union_gap_fill,
                        args.pp_union_gap_closing_iterations,
                        args.pp_fill_blue_table_quadrant,
                        args.pp_blue_table_label,
                        args.pp_blue_table_target,
                        args.pp_blue_table_quadrant_mode,
                        args.pp_blue_table_y_pad_top,
                        args.pp_blue_table_y_pad_bottom,
                        args.pp_blue_table_skip_if_label_above,
                        args.pp_blue_table_skip_if_label_area_gt,
                        default_classes,
                        args.pp_fill_interior_target,
                        per_camera,
                        args.static_prompts,
                        args.pp_topleft_rect,
                        args.pp_topleft_rect_label,
                        args.pp_topleft_rect_fill,
                        args.pp_topleft_rect_y_max,
                        args.pp_topleft_rect_frame_start,
                        args.pp_topleft_rect_frame_end_ratio,
                        args.pp_topright_rect,
                        args.pp_topright_rect_label,
                        args.pp_topright_rect_fill,
                        args.pp_topright_rect_y_max,
                        args.pp_topright_rect_y_threshold,
                        args.pp_topright_rect_frame_start_ratio,
                        args.pp_leftmost_rect,
                        args.pp_leftmost_rect_label,
                        args.pp_leftmost_rect_fill,
                        args.pp_leftmost_rect_y_max,
                        args.pp_leftmost_rect_skip_label,
                        args.pp_halves_rect,
                        args.pp_halves_rect_label,
                        args.pp_halves_rect_fill,
                        args.pp_halves_rect_y_max,
                        args.points,
                        args.point_labels,
                        args.points_frame_idx,
                        args.points_by_frame,
                        args.point_labels_by_frame,
                        args.points_prompt_idx,
                    )
                )
                processes.append(p)
                p.start()
    
    for p in processes:
        p.join()

    failed_workers = [(p.name, p.exitcode) for p in processes if p.exitcode != 0]
    if failed_workers:
        print(f"WARNING: {len(failed_workers)} worker(s) exited with errors:")
        for name, code in failed_workers:
            print(f"  {name}: exit code {code}")

    if args.postprocess:
        if args.npz_separate:
            print("Postprocess skipped: --npz_separate outputs are not supported.")
        else:
            from pathlib import Path
            from postprocess_masks import process_directory, parse_fill_bg_roi

            input_dir = Path(args.output_dir)
            if not input_dir.exists():
                print(f"Postprocess skipped: missing output dir {input_dir}")
            else:
                print("Postprocessing masks...")
                per_camera = _parse_pp_per_camera(args.pp_per_camera)
                default_classes = _parse_class_list(args.pp_fill_interior_class)
                default_target = args.pp_fill_interior_target
                # Parse table top corner ranges
                table_top_corner_ranges = {}
                if args.pp_table_top_corner_ranges:
                    parts = args.pp_table_top_corner_ranges.split(";")
                    names = [
                        ("table_top_tl_x_range", "table_top_tl_y_range"),
                        ("table_top_tr_x_range", "table_top_tr_y_range"),
                        ("table_top_bl_x_range", "table_top_bl_y_range"),
                        ("table_top_br_x_range", "table_top_br_y_range"),
                    ]
                    for part, (xname, yname) in zip(parts, names):
                        vals = [int(v) for v in part.split(",")]
                        table_top_corner_ranges[xname] = (vals[0], vals[1])
                        table_top_corner_ranges[yname] = (vals[2], vals[3])
                fill_bg_roi_list = None
                if args.pp_fill_bg_roi:
                    fill_bg_roi_list = [parse_fill_bg_roi(s) for s in args.pp_fill_bg_roi]

                camera_dirs = [
                    d for d in input_dir.iterdir()
                    if d.is_dir()
                    and "observation.images" in d.name
                    and d.name in set(args.cameras)
                ]
                if not camera_dirs:
                    print(f"No camera directories found in {input_dir}")
                for camera_dir in camera_dirs:
                    cam_name = camera_dir.name
                    classes = default_classes
                    target = default_target
                    if cam_name in per_camera:
                        cam_classes, cam_target = per_camera[cam_name]
                        if cam_classes is not None:
                            classes = cam_classes
                        if cam_target is not None:
                            target = cam_target

                    process_directory(
                        camera_dir,
                        fill_holes=not args.pp_no_fill_holes,
                        min_hole_size=args.pp_min_hole_size,
                        min_object_size=args.pp_min_object_size,
                        closing_iterations=args.pp_closing_iterations,
                        fill_interior_class=classes,
                        fill_interior_target=target,
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
                        overwrite=args.pp_overwrite,
                        num_workers=args.pp_num_workers,
                        fill_bg_roi_list=fill_bg_roi_list,
                        fill_table_top_line_enabled=args.pp_fill_table_top_line,
                        table_top_label=args.pp_table_top_label,
                        table_top_fill_target=args.pp_table_top_fill_target,
                        table_top_corner_ranges=table_top_corner_ranges,
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
                        halves_rect_enabled=args.pp_halves_rect,
                        halves_rect_label=args.pp_halves_rect_label,
                        halves_rect_fill=args.pp_halves_rect_fill,
                        halves_rect_y_max=args.pp_halves_rect_y_max,
                        corner_rect_enabled=args.pp_corner_rect,
                        corner_rect_mode=args.pp_corner_rect_mode,
                        corner_rect_labels=[int(x) for x in args.pp_corner_rect_labels.split(",")] if args.pp_corner_rect_labels else None,
                        corner_rect_fill=args.pp_corner_rect_fill,
                        corner_rect_y_max=args.pp_corner_rect_y_max,
                        corner_rect_x_first_labels=[int(x) for x in args.pp_corner_rect_x_first_labels.split(",")] if args.pp_corner_rect_x_first_labels else None,
                        temporal_fill_enabled=args.pp_temporal_fill,
                        temporal_fill_labels=[int(x) for x in args.pp_temporal_fill_labels.split(",")] if args.pp_temporal_fill_labels else None,
                        temporal_fill_union_labels=[int(x) for x in args.pp_temporal_fill_union_labels.split(",")] if args.pp_temporal_fill_union_labels else None,
                        temporal_fill_value=args.pp_temporal_fill_value,
                    )

    print("All segmentation tasks completed.")
