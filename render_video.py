import argparse
import os
import pickle
import cv2
import glob
import numpy as np
from sam3.visualization_utils import save_masklet_video

def parse_args():
    parser = argparse.ArgumentParser(description="Render video from SAM3 pickle results")
    parser.add_argument(
        "--pkl_path", 
        type=str, 
        required=True,
        help="Path to the segmentation results .pkl file"
    )
    parser.add_argument(
        "--video_path", 
        type=str, 
        required=True,
        help="Path to the original video file or frames directory (must match the pkl data)"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=None,
        help="Path to save the output video. If None, defaults to pkl name + _vis.mp4"
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=30,
        help="FPS for output video"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Load pickle data
    print(f"Loading results from {args.pkl_path}...")
    if not os.path.exists(args.pkl_path):
        print(f"Error: pkl file not found at {args.pkl_path}")
        return

    with open(args.pkl_path, 'rb') as f:
        outputs = pickle.load(f)
    
    # 2. Load video frames
    print(f"Loading video frames from {args.video_path}...")
    if isinstance(args.video_path, str) and args.video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(args.video_path)
        video_frames_for_vis = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        video_frames_for_vis = glob.glob(os.path.join(args.video_path, "*.jpg"))
        try:
            video_frames_for_vis.sort(
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
            )
        except ValueError:
            video_frames_for_vis.sort()
            
    if not video_frames_for_vis:
        print("Error: No frames found!")
        return

    # 3. Determine output path
    if args.output_path is None:
        base_name = os.path.splitext(args.pkl_path)[0]
        output_path = f"{base_name}_vis.mp4"
    else:
        output_path = args.output_path

    # 4. Save video
    print(f"Rendering video to {output_path} (FPS: {args.fps})...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    try:
        save_masklet_video(
            video_frames=video_frames_for_vis, 
            outputs=outputs, 
            out_path=output_path, 
            fps=args.fps
        )
        print("Done!")
    except Exception as e:
        print(f"Error saving video: {e}")

if __name__ == "__main__":
    main()

