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
    avg = len(data) / float(num_chunks)
    out = []
    last = 0.0
    while last < len(data):
        out.append(data[int(last):int(last + avg)])
        last += avg
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

        # Check if output already exists
        if skip_if_exists:
            video_filename = os.path.basename(video_path)
            # Assuming output filename format: {video_name}_masks.npz
            # video_name usually is video_filename without extension
            video_name = os.path.splitext(video_filename)[0]
            expected_output = os.path.join(output_dir, f"{video_name}_masks.npz")
            expected_post_output = os.path.join(output_dir, f"{video_name}_masks_post.npz")
            if os.path.exists(expected_output) or os.path.exists(expected_post_output):
                print(f"[Worker GPU {gpu_id}] Skipping {video_path} (Output exists: {expected_output} or {expected_post_output})")
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
        
        try:
            subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[Worker GPU {gpu_id}] ERROR processing {video_path}: {e}")

    print(f"[Worker GPU {gpu_id}] Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel SAM3 segmentation across multiple GPUs (PKL output only)")
    parser.add_argument("--base_dir", required=True, help="Base directory containing camera folders")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM3 checkpoint")
    parser.add_argument("--output_dir", required=True, help="Base output directory")
    parser.add_argument("--prompts", nargs="+", required=True, help="Prompts list")
    parser.add_argument("--cameras", nargs="+", required=True, help="List of camera folder names to scan")
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=None, help="Specific GPU IDs to use. If None, uses all available.")
    parser.add_argument("--points", type=str, default=None, help="Extra points as 'x1,y1;x2,y2;...'.")
    parser.add_argument("--point_labels", type=str, default=None, help="Point labels as '1,0,1,...'.")
    parser.add_argument("--points_frame_idx", type=int, default=None, help="Frame index for points.")
    parser.add_argument("--points_by_frame", type=str, default=None, help="Multiple frames: 'frame: x1,y1;...|frame: x1,y1;...'.")
    parser.add_argument("--point_labels_by_frame", type=str, default=None, help="Labels per frame: 'frame:1,0|frame:1,1'.")
    parser.add_argument("--save_video", action="store_true", help="Save visualization videos.")
    parser.add_argument("--save_side_by_side", action="store_true", help="Save side-by-side videos.")
    parser.add_argument("--max_frames", type=int, default=None, help="Only process first N frames.")
    parser.add_argument("--save_npz", action="store_true", help="Also save Cosmos npz outputs.")
    parser.add_argument("--npz_separate", action="store_true", help="Keep objects separate in npz.")
    parser.add_argument("--no_pkl", action="store_true", help="Do not save pkl outputs.")
    parser.add_argument("--debug_one", action="store_true", help="Randomly pick one video and run once.")
    parser.add_argument("--debug_seed", type=int, default=None, help="Random seed for debug_one.")
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
    
    args = parser.parse_args()
    
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
    
    # Debug mode: pick one random video
    if args.debug_one:
        import random

        if args.debug_seed is not None:
            random.seed(args.debug_seed)
        all_videos = [random.choice(all_videos)]

    # 2. Assign GPUs
    if args.gpu_ids:
        gpu_ids = args.gpu_ids
    else:
        gpu_ids = get_available_gpus()
    
    print(f"Using GPUs: {gpu_ids}")
    
    # 3. Distribute work across GPUs
    if args.debug_one:
        gpu_ids = [gpu_ids[0]]
    chunks = chunk_list(all_videos, len(gpu_ids))
    
    per_camera = _parse_pp_per_camera(args.pp_per_camera)
    default_classes = _parse_class_list(args.pp_fill_interior_class)
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
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
                    args.skip_if_exists,
                )
            )
            processes.append(p)
            p.start()
    
    for p in processes:
        p.join()
        
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
                    )

    print("All segmentation tasks completed.")
