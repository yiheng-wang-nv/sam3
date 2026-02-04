#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Configuration
CHECKPOINT="${SAM3_DIR}/sam3.pt"
BASE_DIR="/localhome/local-vennw/code/task3_01210122_merged/videos/chunk-000"
BASE_OUTPUT_DIR="/localhome/local-vennw/code/task3_01210122_merged/sam3_output_debug"

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

# Use 1 GPU for debug
GPU_IDS="0"

# Set to "false" to run full (non-debug) instead of one random video
DEBUG_ONE=true

DEBUG_FLAGS=()
if [ "$DEBUG_ONE" = true ]; then
  DEBUG_FLAGS+=(--debug_one --debug_seed 0)
fi

echo "🧪 Starting Debug Segmentation (task3 head cams)"
echo "------------------------------------------------"
echo "SAM3 Dir:  $SAM3_DIR"
echo "Base Dir:  $BASE_DIR"
echo "Output:    $BASE_OUTPUT_DIR"
echo "Prompts:   ${PROMPTS[*]}"
echo "GPU:       ${GPU_IDS}"
echo "------------------------------------------------"

echo "-> Debug head left camera (one random video)"
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
  --save_side_by_side \
  --invert_mask \
  "${DEBUG_FLAGS[@]}" \
  --gpu_ids $GPU_IDS

echo "-> Debug head right camera (one random video)"
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
  --save_side_by_side \
  --invert_mask \
  "${DEBUG_FLAGS[@]}" \
  --gpu_ids $GPU_IDS

echo "✅ Debug segmentation finished!"
