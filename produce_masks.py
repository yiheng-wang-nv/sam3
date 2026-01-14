import numpy as np
import torch
import cv2
import glob
import os
import argparse
import pickle
from sam3.visualization_utils import save_masklet_video
from sam3.model_builder import build_sam3_video_predictor

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
        "--save_video", 
        action="store_true",
        help="Whether to save the visualization video"
    )
    return parser.parse_args()

def propagate_in_video(predictor, session_id):
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame

def main():
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading model from {args.checkpoint_path}...")
    video_predictor = build_sam3_video_predictor(args.checkpoint_path)
    
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
    
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        video_frames_for_vis = []
        if args.save_video:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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
        
        if not args.save_video and video_frames_for_vis:
             # Just keep one for dimension check if needed
             # Actually glob just returns paths, so memory is fine.
             pass

    if not video_frames_for_vis:
        print("Error: No frames found!")
        return

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
    
    print(f"Start processing {len(prompts_list)} prompts: {prompts_list}")

    for prompt_idx, p in enumerate(prompts_list):
        print(f"Processing prompt [{prompt_idx}]: '{p}'")
        
        # Reset session for new prompt
        _ = video_predictor.handle_request(
            request=dict(type="reset_session", session_id=session_id)
        )

        # Add prompt
        _ = video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=p,
            )
        )

        # Propagate
        outputs_per_frame = propagate_in_video(video_predictor, session_id)
        
        # Merge this prompt's results
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
                merged_outputs[frame_idx]['out_probs'].append(frame_data['out_probs'])
                merged_outputs[frame_idx]['out_boxes_xywh'].append(frame_data['out_boxes_xywh'])
                merged_outputs[frame_idx]['out_binary_masks'].append(frame_data['out_binary_masks'])

    # Format merged outputs
    final_formatted_outputs = {}
    
    # Get frame dimensions
    if len(video_frames_for_vis) > 0:
        if isinstance(video_frames_for_vis[0], np.ndarray):
            H, W = video_frames_for_vis[0].shape[:2]
        elif isinstance(video_frames_for_vis[0], str):
             img = cv2.imread(video_frames_for_vis[0])
             H, W = img.shape[:2]
        else:
             H, W = 480, 640
    else:
        H, W = 480, 640 

    print("Merging results...")
    for frame_idx in merged_outputs.keys():
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
            
    # Save raw results (pickle)
    raw_output_path = os.path.join(args.output_dir, f"{video_name}_segmentation_results.pkl")
    print(f"Saving raw segmentation results to {raw_output_path} ...")
    with open(raw_output_path, 'wb') as f:
        pickle.dump(final_formatted_outputs, f)

    # Save visualization video ONLY if requested
    if args.save_video:
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

        output_video_path = os.path.join(args.output_dir, f"{video_name}_vis.mp4")
        print(f"Saving video to {output_video_path} ...")

        save_masklet_video(
            video_frames=video_frames_for_vis, 
            outputs=final_formatted_outputs, 
            out_path=output_video_path, 
            fps=args.fps
        )
    else:
        print("Skipping video generation (--save_video not set).")

if __name__ == "__main__":
    main()
