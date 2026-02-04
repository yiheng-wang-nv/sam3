python produce_masks.py \
  --video_path /localhome/local-vennw/code/task6_01120119_merged/videos/chunk-000/observation.images.head_left_camera_color_optical_frame/episode_000025.mp4 \
  --prompts "blue table" "robotic arm(s)" \
  --points "300,214;349,274;287.3,276.6;347.4,294.5" \
  --point_labels "1,1,1,1" \
  --points_frame_idx 0 \
  --save_video \
  --checkpoint_path /localhome/local-vennw/code/sam3/sam3.pt \
  --output_dir .


# points for right: "307,220;318,280;289.6,274.8;345.6,273.6"
# points for left: "300,214;349,274;287.3,276.6;347.4,294.5"
# points for left arm frame 380: "493,356;571,344" frame 50: "520.6,404"
# points for right arm frame 380: "96,328;189,329"

python /localhome/local-vennw/code/sam3/scripts/save_first_frames_with_points.py \
  --video-dir "/localhome/local-vennw/code/task6_01120119_merged/videos/chunk-000/observation.images.left_arm_camera_color_optical_frame" \
  --points "520.6,404.6" \
  --labels "1" \
  --frame-idx 50 \
  --output-dir frames/