#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configuration
# This script is for rendering videos from existing PKL files only.
# It uses CPU parallelism.

BASE_DIR="/localhome/local-vennw/code/orca-dev-galbot-1/galbot_lerobot_dataset/task7_20260106_no_rnd_lerobot/videos/chunk-000"
BASE_PKL_DIR="${SCRIPT_DIR}/output/task7_segmentation"

# Define cameras
CAMERAS=(
    "observation.images.head_left_camera_color_optical_frame"
    "observation.images.head_right_camera_color_optical_frame"
    "observation.images.left_arm_camera_color_optical_frame"
    "observation.images.right_arm_camera_color_optical_frame"
)

# Number of parallel workers
# Adjust based on your CPU cores. Rendering 16 videos in parallel is usually fine on modern servers.
NUM_WORKERS=8

echo "🎬 Starting Parallel Video Rendering Job"
echo "-------------------------------------"
echo "Script Dir:  $SCRIPT_DIR"
echo "Base Dir:    $BASE_DIR"
echo "PKL Dir:     $BASE_PKL_DIR"
echo "Num Workers: $NUM_WORKERS"
echo "-------------------------------------"

python "${SCRIPT_DIR}/batch_render_parallel.py" \
    --base_dir "$BASE_DIR" \
    --pkl_dir "$BASE_PKL_DIR" \
    --cameras "${CAMERAS[@]}" \
    --num_workers $NUM_WORKERS \
    --script_dir "$SCRIPT_DIR"

echo "🎉 Batch rendering finished!"
