
# # head left prompt "floor"
# points_abs = np.array(
#     [
#         [30.1, 190.3], [262.9, 90.3], [453.3, 84.3], [552, 239], [320.6, 259.9], [363, 226], [221.8, 313.5]
#     ]
# )
# # positive clicks have label 1, while negative clicks have label 0
# labels = np.array([1, 1, 1, 1, 0, 0, 0])

# # head right prompt "floor"
# points_abs = np.array(
#     [
#         [30.1, 190.3], [262.9, 90.3], [453.3, 84.3], [552, 239], [320.6, 259.9], [363, 226], [221.8, 313.5]
#     ]
# )
# # positive clicks have label 1, while negative clicks have label 0
# labels = np.array([1, 1, 1, 1, 0, 0, 0])

# # left arm: prompt "background"


python /localhome/local-vennw/code/sam3/scripts/save_first_frames_with_points.py \
  --video-dir "/localhome/local-vennw/code/task3_01210122_merged/videos/chunk-000/observation.images.head_left_camera_color_optical_frame" \
  --points "30.1,190.3;262.9,90.3;453.3,84.3;552,239;320.6,259.9;363,226;221.8,313.5" \
  --labels "1,1,1,1,0,0,0" \
  --random-frame \
  --output-dir frames/head_left

python /localhome/local-vennw/code/sam3/scripts/save_first_frames_with_points.py \
  --video-dir "/localhome/local-vennw/code/task3_01210122_merged/videos/chunk-000/observation.images.head_right_camera_color_optical_frame" \
  --points "30.1,190.3;262.9,90.3;453.3,84.3;552,239;320.6,259.9;363,226;221.8,313.5" \
  --labels "1,1,1,1,0,0,0" \
  --random-frame \
  --output-dir frames/head_right