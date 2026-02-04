#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Configuration
CHECKPOINT="${SAM3_DIR}/sam3.pt"
BASE_DIR="/localhome/local-vennw/code/task6_01120119_merged/videos/chunk-000"
BASE_OUTPUT_DIR="/localhome/local-vennw/code/task6_01120119_merged/sam3_output"

# Prompts (shared)
PROMPTS=("blue table" "robotic arm(s)")

# Left arm camera + points
LEFT_ARM_CAMERA="observation.images.left_arm_camera_color_optical_frame"
LEFT_ARM_POINTS_BY_FRAME="380:493,356;571,344|50:520.6,404.6"
LEFT_ARM_POINT_LABELS_BY_FRAME="380:1,1|50:1"

# Right arm camera + points
RIGHT_ARM_CAMERA="observation.images.right_arm_camera_color_optical_frame"
RIGHT_ARM_POINTS="96,328;189,329"
RIGHT_ARM_POINT_LABELS="1,1"
RIGHT_ARM_POINTS_FRAME_IDX=380

# Use 8 GPUs (0-7)
GPU_IDS="0 1 2 3 4 5 6 7"

echo "🚀 Starting Parallel Segmentation Job (task6 arms, 8 GPUs)"
echo "--------------------------------------------------------------"
echo "SAM3 Dir:  $SAM3_DIR"
echo "Base Dir:  $BASE_DIR"
echo "Output:    $BASE_OUTPUT_DIR"
echo "Camera:    ${LEFT_ARM_CAMERA}, ${RIGHT_ARM_CAMERA}"
echo "Prompts:   ${PROMPTS[*]}"
echo "GPUs:      ${GPU_IDS}"
echo "--------------------------------------------------------------"

echo "-> Running left arm camera"
python "${SAM3_DIR}/batch_run_parallel.py" \
  --base_dir "$BASE_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$BASE_OUTPUT_DIR" \
  --cameras "$LEFT_ARM_CAMERA" \
  --prompts "${PROMPTS[@]}" \
  --points_by_frame "$LEFT_ARM_POINTS_BY_FRAME" \
  --point_labels_by_frame "$LEFT_ARM_POINT_LABELS_BY_FRAME" \
  --save_npz \
  --no_pkl \
  --save_video \
  --gpu_ids $GPU_IDS

echo "-> Running right arm camera"
python "${SAM3_DIR}/batch_run_parallel.py" \
  --base_dir "$BASE_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$BASE_OUTPUT_DIR" \
  --cameras "$RIGHT_ARM_CAMERA" \
  --prompts "${PROMPTS[@]}" \
  --points "$RIGHT_ARM_POINTS" \
  --point_labels "$RIGHT_ARM_POINT_LABELS" \
  --points_frame_idx "$RIGHT_ARM_POINTS_FRAME_IDX" \
  --save_npz \
  --no_pkl \
  --save_video \
  --gpu_ids $GPU_IDS

echo "🎉 Batch segmentation job finished!"
