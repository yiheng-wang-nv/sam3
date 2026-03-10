#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Configuration
CHECKPOINT="${SAM3_DIR}/sam3.pt"
BASE_DIR="/localhome/local-vennw/code/task7_01220206022402280306_merged/videos/chunk-000"
BASE_OUTPUT_DIR="/localhome/local-vennw/code/task7_01220206022402280306_merged/sam3_output"

# Prompts (same for all cameras)
PROMPTS=("blue table" "robotic arm(s)" "silver box")

# Camera names
HEAD_LEFT_CAMERA="observation.images.head_left_camera_color_optical_frame"
HEAD_RIGHT_CAMERA="observation.images.head_right_camera_color_optical_frame"
LEFT_ARM_CAMERA="observation.images.left_arm_camera_color_optical_frame"
RIGHT_ARM_CAMERA="observation.images.right_arm_camera_color_optical_frame"

# GPU Selection
GPU_IDS="0 1 2 3 5 6 7"

# Postprocess settings (shared)
PP_NUM_WORKERS=96
PP_MIN_HOLE_SIZE=64
PP_MIN_OBJECT_SIZE=50
PP_CLOSING_ITERATIONS=1
PP_NO_REMOVE_SMALL_OBJECTS=true
PP_UNION_HOLE_FILL=true
PP_UNION_GAP_FILL=true
PP_UNION_GAP_CLOSING_ITERATIONS=2

# Interior fill (all cameras)
PP_FILL_CLASS="1,2,3"
PP_FILL_TARGET=4

CAMERAS=(
    "$HEAD_LEFT_CAMERA"
    "$HEAD_RIGHT_CAMERA"
    "$LEFT_ARM_CAMERA"
    "$RIGHT_ARM_CAMERA"
)

echo "🚀 Starting Parallel Segmentation Job (task7, enhanced postprocess)"
echo "--------------------------------------------------------------------"
echo "SAM3 Dir:  $SAM3_DIR"
echo "Base Dir:  $BASE_DIR"
echo "Output:    $BASE_OUTPUT_DIR"
echo "GPUs:      ${GPU_IDS}"
echo "Postprocess: union_hole_fill, union_gap_fill(iter=$PP_UNION_GAP_CLOSING_ITERATIONS), fill_interior($PP_FILL_CLASS->$PP_FILL_TARGET)"
echo "--------------------------------------------------------------------"

python "${SAM3_DIR}/batch_run_parallel.py" \
  --base_dir "$BASE_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$BASE_OUTPUT_DIR" \
  --cameras "${CAMERAS[@]}" \
  --prompts "${PROMPTS[@]}" \
  --save_npz \
  --no_pkl \
  --skip_if_exists \
  --skip_if_masks_dir "/localhome/local-vennw/code/task7_01220206022402280306_merged/masks" \
  --postprocess \
  --pp_num_workers "$PP_NUM_WORKERS" \
  --pp_min_hole_size "$PP_MIN_HOLE_SIZE" \
  --pp_min_object_size "$PP_MIN_OBJECT_SIZE" \
  --pp_closing_iterations "$PP_CLOSING_ITERATIONS" \
  $( [ "$PP_NO_REMOVE_SMALL_OBJECTS" = true ] && echo "--pp_no_remove_small_objects" ) \
  $( [ "$PP_UNION_HOLE_FILL" = true ] && echo "--pp_union_hole_fill" ) \
  $( [ "$PP_UNION_GAP_FILL" = true ] && echo "--pp_union_gap_fill" ) \
  $( [ "$PP_UNION_GAP_FILL" = true ] && echo "--pp_union_gap_closing_iterations $PP_UNION_GAP_CLOSING_ITERATIONS" ) \
  --pp_fill_interior_class "$PP_FILL_CLASS" \
  --pp_fill_interior_target "$PP_FILL_TARGET" \
  --pp_overwrite \
  --gpu_ids $GPU_IDS

echo "🎉 Batch segmentation job finished!"

# upload to hf
#  hf upload-large-folder --repo-type dataset nvidia/orca-template1-dev task7_20260106_no_rnd_lerobot_with_mask/

# then run process_sam3_mask_to_cosmos.sh and postprocess_masks.py

# then run

# python /localhome/local-vennw/code/sam3/update_orca_meta_masks_only.py \
#   --target_root /localhome/local-vennw/code/orca-template1-dev/task7_20260122_trimmed
