#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Configuration
CHECKPOINT="${SAM3_DIR}/sam3.pt"
BASE_DIR="/localhome/local-vennw/code/task6_01120119_merged/videos/chunk-000"
BASE_OUTPUT_DIR="/localhome/local-vennw/code/task6_01120119_merged/sam3_output"

# Single camera
CAMERAS=(
  "observation.images.head_left_camera_color_optical_frame"
)

# Prompts and points
PROMPTS=("blue table" "robotic arm(s)")
POINTS="300,214;349,274;287.3,276.6;347.4,294.5"
POINT_LABELS="1,1,1,1"
POINTS_FRAME_IDX=0

# Use 8 GPUs (0-7)
GPU_IDS="0 1 2 3 4 5 6 7"

echo "🚀 Starting Parallel Segmentation Job (task6 head_right, 8 GPUs)"
echo "--------------------------------------------------------------"
echo "SAM3 Dir:  $SAM3_DIR"
echo "Base Dir:  $BASE_DIR"
echo "Output:    $BASE_OUTPUT_DIR"
echo "Camera:    ${CAMERAS[*]}"
echo "Prompts:   ${PROMPTS[*]}"
echo "GPUs:      ${GPU_IDS}"
echo "--------------------------------------------------------------"

python "${SAM3_DIR}/batch_run_parallel.py" \
  --base_dir "$BASE_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$BASE_OUTPUT_DIR" \
  --cameras "${CAMERAS[@]}" \
  --prompts "${PROMPTS[@]}" \
  --points "$POINTS" \
  --point_labels "$POINT_LABELS" \
  --points_frame_idx "$POINTS_FRAME_IDX" \
  --save_video \
  --gpu_ids $GPU_IDS

echo "🎉 Batch segmentation job finished!"
