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
                )
            )
            processes.append(p)
            p.start()
    
    for p in processes:
        p.join()
        
    print("All segmentation tasks completed.")
