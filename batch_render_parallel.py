import os
import glob
import subprocess
import argparse
import multiprocessing

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

def run_render_worker(worker_id, pkl_list, base_dir, script_dir):
    """Worker function for rendering videos from PKL files (CPU only)"""
    
    print(f"[Render Worker {worker_id}] Starting processing {len(pkl_list)} files...")
    
    render_script = os.path.join(script_dir, "render_video.py")
    
    for pkl_path in pkl_list:
        try:
            # Reconstruct paths
            # pkl_path: output/task7_segmentation/{camera}/{video_name}_segmentation_results.pkl
            # video_path: base_dir/{camera}/{video_name}.mp4
            # output_video: output/task7_segmentation/{camera}/{video_name}_vis.mp4
            
            parts = pkl_path.split(os.sep)
            camera_name = parts[-2]
            filename = os.path.basename(pkl_path)
            
            # {video_name}_segmentation_results.pkl -> {video_name}
            video_stem = filename.replace("_segmentation_results.pkl", "")
            video_name = video_stem + ".mp4"
            
            # Construct original video path
            video_path = os.path.join(base_dir, camera_name, video_name)
            
            # Construct output video path (same dir as pkl)
            output_path = os.path.join(os.path.dirname(pkl_path), f"{video_stem}_vis.mp4")
            
            if not os.path.exists(video_path):
                print(f"[Render Worker {worker_id}] Skipping, original video not found: {video_path}")
                continue
                
            print(f"[Render Worker {worker_id}] Rendering: {video_stem}")
            
            cmd = [
                "python", render_script,
                "--pkl_path", pkl_path,
                "--video_path", video_path,
                "--output_path", output_path,
                "--fps", "30"
            ]
            
            subprocess.run(cmd, check=True)
            
        except Exception as e:
            print(f"[Render Worker {worker_id}] Error rendering {pkl_path}: {e}")

    print(f"[Render Worker {worker_id}] Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel video rendering from PKL files (CPU only)")
    parser.add_argument("--base_dir", required=True, help="Base directory containing original videos")
    parser.add_argument("--pkl_dir", required=True, help="Base directory containing pkl files")
    parser.add_argument("--cameras", nargs="+", required=True, help="List of camera folder names")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers. Defaults to CPU count.")
    parser.add_argument("--script_dir", type=str, default=None, help="Directory containing the scripts")
    
    args = parser.parse_args()
    
    # Get script directory
    if args.script_dir:
        script_dir = args.script_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Collect all PKL files
    all_pkls = []
    for cam in args.cameras:
        search_path = os.path.join(args.pkl_dir, cam, "*_segmentation_results.pkl")
        pkls = glob.glob(search_path)
        all_pkls.extend(pkls)
        print(f"Found {len(pkls)} PKL files in {cam}")
    
    if not all_pkls:
        print("No PKL files found to render.")
        exit(0)
        
    print(f"Total PKL files to render: {len(all_pkls)}")
    
    # 2. Determine number of workers (CPU-based parallelism)
    if args.num_workers:
        num_workers = args.num_workers
    else:
        num_workers = multiprocessing.cpu_count()
    
    # Limit workers to avoid excessive IO thrashing
    num_workers = min(num_workers, len(all_pkls), 32)
    
    print(f"Using {num_workers} parallel workers")
    
    # 3. Distribute work
    chunks = chunk_list(all_pkls, num_workers)
    
    processes = []
    for i in range(len(chunks)):
        if chunks[i]:
            p = multiprocessing.Process(
                target=run_render_worker,
                args=(i, chunks[i], args.base_dir, script_dir)
            )
            processes.append(p)
            p.start()
    
    for p in processes:
        p.join()
        
    print("All render tasks completed.")
