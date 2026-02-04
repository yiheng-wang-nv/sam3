#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Configuration
CHECKPOINT="${SAM3_DIR}/sam3.pt"
BASE_DIR="/localhome/local-vennw/code/task3_01210122_merged/videos/chunk-000"
BASE_OUTPUT_DIR="/localhome/local-vennw/code/task3_01210122_merged/sam3_output"

# Prompts (shared)
PROMPTS=("floor")

# Head left camera + points
HEAD_LEFT_CAMERA="observation.images.head_left_camera_color_optical_frame"
HEAD_LEFT_POINTS="30.1,190.3;262.9,90.3;453.3,84.3;552,239;320.6,259.9;363,226;221.8,313.5"
HEAD_LEFT_POINT_LABELS="1,1,1,1,0,0,0"
HEAD_LEFT_POINTS_FRAME_IDX=0

# Head right camera + points
HEAD_RIGHT_CAMERA="observation.images.head_right_camera_color_optical_frame"
HEAD_RIGHT_POINTS="30.1,190.3;262.9,90.3;453.3,84.3;552,239;320.6,259.9;363,226;221.8,313.5"
HEAD_RIGHT_POINT_LABELS="1,1,1,1,0,0,0"
HEAD_RIGHT_POINTS_FRAME_IDX=0

# Use 8 GPUs (0-7)
GPU_IDS="1 2 3 4 5 6 7"

echo "🚀 Starting Parallel Segmentation Job (task3 head cams, 8 GPUs)"
echo "--------------------------------------------------------------"
echo "SAM3 Dir:  $SAM3_DIR"
echo "Base Dir:  $BASE_DIR"
echo "Output:    $BASE_OUTPUT_DIR"
echo "Camera:    ${HEAD_LEFT_CAMERA}, ${HEAD_RIGHT_CAMERA}"
echo "Prompts:   ${PROMPTS[*]}"
echo "GPUs:      ${GPU_IDS}"
echo "--------------------------------------------------------------"

echo "-> Running head left camera"
python "${SAM3_DIR}/batch_run_parallel.py" \
  --base_dir "$BASE_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$BASE_OUTPUT_DIR" \
  --cameras "$HEAD_LEFT_CAMERA" \
  --prompts "${PROMPTS[@]}" \
  --points "$HEAD_LEFT_POINTS" \
  --point_labels "$HEAD_LEFT_POINT_LABELS" \
  --points_frame_idx "$HEAD_LEFT_POINTS_FRAME_IDX" \
  --save_npz \
  --no_pkl \
  --invert_mask \
  --save_video \
  --gpu_ids $GPU_IDS

echo "-> Running head right camera"
python "${SAM3_DIR}/batch_run_parallel.py" \
  --base_dir "$BASE_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$BASE_OUTPUT_DIR" \
  --cameras "$HEAD_RIGHT_CAMERA" \
  --prompts "${PROMPTS[@]}" \
  --points "$HEAD_RIGHT_POINTS" \
  --point_labels "$HEAD_RIGHT_POINT_LABELS" \
  --points_frame_idx "$HEAD_RIGHT_POINTS_FRAME_IDX" \
  --save_npz \
  --no_pkl \
  --invert_mask \
  --save_video \
  --gpu_ids $GPU_IDS

echo "🎉 Batch segmentation job finished!"
