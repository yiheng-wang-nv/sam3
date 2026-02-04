python /localhome/local-vennw/code/sam3/visualize_masks.py \
  --dataset_root /localhome/local-vennw/code/task7_20260122_trimmed \
  --mask_root /localhome/local-vennw/code/task7_20260122_trimmed/sam3_output \
  --output_dir /localhome/local-vennw/code/task7_20260122_trimmed/viz_videos \
  --cameras observation.images.head_left_camera_color_optical_frame,observation.images.head_right_camera_color_optical_frame,observation.images.left_arm_camera_color_optical_frame,observation.images.right_arm_camera_color_optical_frame \
  --use_post_masks \
  --save_video \
  --videos_per_camera 3 \
  --fps 30 \
  --seed 0
