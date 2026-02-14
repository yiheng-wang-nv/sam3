#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Configuration
CHECKPOINT="${SAM3_DIR}/sam3.pt"
BASE_DIR="/localhome/local-vennw/code/task4-2_02020205_merged/videos/chunk-000"
BASE_OUTPUT_DIR="/localhome/local-vennw/code/task4-2_02020205_merged/sam3_output"

# Prompts (same for all cameras)
PROMPTS=("blue table" "robotic arm(s)" "tools" "trash can" "silver box")

# Camera names
HEAD_RIGHT_CAMERA="observation.images.head_right_camera_color_optical_frame"
RIGHT_ARM_CAMERA="observation.images.right_arm_camera_color_optical_frame"

# GPU Selection
GPU_IDS="0 1 2"

# Postprocess settings (shared)
PP_NUM_WORKERS=32
PP_MIN_HOLE_SIZE=64
PP_MIN_OBJECT_SIZE=50
PP_CLOSING_ITERATIONS=1
PP_NO_REMOVE_SMALL_OBJECTS=true
PP_UNION_HOLE_FILL=true
PP_UNION_GAP_FILL=true
PP_UNION_GAP_CLOSING_ITERATIONS=1

# Interior fill (all cameras)
PP_FILL_CLASS="1,2,3,4,5"
PP_FILL_TARGET=6

# Table top-line fill (fill background inside table top-line closed region)
PP_FILL_TABLE_TOP_LINE=true
PP_TABLE_TOP_LABEL=1
PP_TABLE_TOP_FILL_TARGET=6
# Corner ROI ranges: "tl_x0,tl_x1,tl_y0,tl_y1;tr;bl;br"
PP_TABLE_TOP_CORNER_RANGES="100,230,150,200;300,430,150,200;50,170,300,370;350,450,260,350"

CAMERAS=(
    "$HEAD_RIGHT_CAMERA"
    "$RIGHT_ARM_CAMERA"
)

echo "🚀 Starting Parallel Segmentation Job (task7, enhanced postprocess)"
echo "--------------------------------------------------------------------"
echo "SAM3 Dir:  $SAM3_DIR"
echo "Base Dir:  $BASE_DIR"
echo "Output:    $BASE_OUTPUT_DIR"
echo "GPUs:      ${GPU_IDS}"
echo "Postprocess: union_hole_fill, union_gap_fill(iter=$PP_UNION_GAP_CLOSING_ITERATIONS), fill_interior($PP_FILL_CLASS->$PP_FILL_TARGET), fill_table_top_line=$PP_FILL_TABLE_TOP_LINE"
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
  $( [ "$PP_FILL_TABLE_TOP_LINE" = true ] && echo "--pp_fill_table_top_line" ) \
  $( [ "$PP_FILL_TABLE_TOP_LINE" = true ] && echo "--pp_table_top_label $PP_TABLE_TOP_LABEL" ) \
  $( [ "$PP_FILL_TABLE_TOP_LINE" = true ] && echo "--pp_table_top_fill_target $PP_TABLE_TOP_FILL_TARGET" ) \
  $( [ "$PP_FILL_TABLE_TOP_LINE" = true ] && echo "--pp_table_top_corner_ranges $PP_TABLE_TOP_CORNER_RANGES" ) \
  --pp_overwrite \
  --gpu_ids $GPU_IDS

echo "🎉 Batch segmentation job finished!"

# upload to hf
#  hf upload-large-folder --repo-type dataset nvidia/orca-template1-dev task7_20260106_no_rnd_lerobot_with_mask/

# then run process_sam3_mask_to_cosmos.sh and postprocess_masks.py

# then run

# python /localhome/local-vennw/code/sam3/update_orca_meta_masks_only.py \
#   --target_root /localhome/local-vennw/code/orca-template1-dev/task7_20260122_trimmed