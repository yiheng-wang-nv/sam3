#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configuration
CHECKPOINT="/localhome/local-vennw/code/3rd_sam3/sam3.pt"
BASE_DIR="/localhome/local-vennw/code/orca-dev-galbot-1/galbot_lerobot_dataset/task7_20260106_no_rnd_lerobot/videos/chunk-000"
BASE_OUTPUT_DIR="${SCRIPT_DIR}/output/task7_segmentation"

# Define cameras
CAMERAS=(
    "observation.images.head_left_camera_color_optical_frame"
    "observation.images.head_right_camera_color_optical_frame"
    "observation.images.left_arm_camera_color_optical_frame"
    "observation.images.right_arm_camera_color_optical_frame"
)

# Prompts
PROMPTS=("blue table" "robotic arm(s)" "silver box")

# GPU Selection (Optional: specific IDs like "0 1 2 3")
# Leave empty to let the script detect all available GPUs
GPU_IDS="0 1 2 3" 

echo "🚀 Starting Parallel Segmentation Job"
echo "-------------------------------------"
echo "Script Dir: $SCRIPT_DIR"
echo "Base Dir:   $BASE_DIR"
echo "Output:     $BASE_OUTPUT_DIR"
echo "GPUs:       ${GPU_IDS:-Auto-detect}"
echo "-------------------------------------"

python "${SCRIPT_DIR}/batch_run_parallel.py" \
    --base_dir "$BASE_DIR" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$BASE_OUTPUT_DIR" \
    --cameras "${CAMERAS[@]}" \
    --prompts "${PROMPTS[@]}" \
    --gpu_ids $GPU_IDS

echo "🎉 Batch segmentation job finished!"
