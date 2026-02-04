#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Configuration
CHECKPOINT="${SAM3_DIR}/sam3.pt"
BASE_DIR="/localhome/local-vennw/code/task7_20260122_trimmed/videos/chunk-000"
BASE_OUTPUT_DIR="/localhome/local-vennw/code/task7_20260122_trimmed/sam3_output"

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
GPU_IDS="1 2 3 4 5 6 7" 

echo "🚀 Starting Parallel Segmentation Job"
echo "-------------------------------------"
echo "Script Dir: $SCRIPT_DIR"
echo "Base Dir:   $BASE_DIR"
echo "Output:     $BASE_OUTPUT_DIR"
echo "GPUs:       ${GPU_IDS:-Auto-detect}"
echo "-------------------------------------"

python "${SAM3_DIR}/batch_run_parallel.py" \
    --base_dir "$BASE_DIR" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$BASE_OUTPUT_DIR" \
    --cameras "${CAMERAS[@]}" \
    --prompts "${PROMPTS[@]}" \
    --save_npz \
    --no_pkl \
    --gpu_ids $GPU_IDS

echo "🎉 Batch segmentation job finished!"

# upload to hf
#  hf upload-large-folder --repo-type dataset nvidia/orca-template1-dev task7_20260106_no_rnd_lerobot_with_mask/

# then run process_sam3_mask_to_cosmos.sh and postprocess_masks.py

# then run

# python /localhome/local-vennw/code/sam3/update_orca_meta_masks_only.py \
#   --target_root /localhome/local-vennw/code/orca-template1-dev/task7_20260122_trimmed