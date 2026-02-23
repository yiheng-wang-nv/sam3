#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Configuration
CHECKPOINT="${SAM3_DIR}/sam3.pt"
BASE_DIR="/localhome/local-vennw/code/task5-1_02030204021002130214_merged/videos/chunk-000"
BASE_OUTPUT_DIR="/localhome/local-vennw/code/task5-1_02030204021002130214_merged/sam3_output"

# Prompts (same for all cameras)
PROMPTS=("blue table" "robotic arm(s)")

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
PP_FILL_CLASS="1,2"
PP_FILL_TARGET=3

# Scanline fill: fill background between first/last table pixels per row (head right only)
PP_SCANLINE_FILL=true
PP_SCANLINE_SOURCE_LABEL=1
PP_SCANLINE_FILL_VALUE=3

echo "🚀 Starting Parallel Segmentation Job (task5)"
echo "--------------------------------------------------------------------"
echo "SAM3 Dir:  $SAM3_DIR"
echo "Base Dir:  $BASE_DIR"
echo "Output:    $BASE_OUTPUT_DIR"
echo "GPUs:      ${GPU_IDS}"
echo "Scanline fill: head right only ($PP_SCANLINE_SOURCE_LABEL->$PP_SCANLINE_FILL_VALUE)"
echo "Postprocess: union_hole_fill, union_gap_fill(iter=$PP_UNION_GAP_CLOSING_ITERATIONS), fill_interior($PP_FILL_CLASS->$PP_FILL_TARGET)"
echo "--------------------------------------------------------------------"

PP_COMMON_ARGS=(
  --save_npz
  --no_pkl
  --skip_if_exists
  --postprocess
  --pp_num_workers "$PP_NUM_WORKERS"
  --pp_min_hole_size "$PP_MIN_HOLE_SIZE"
  --pp_min_object_size "$PP_MIN_OBJECT_SIZE"
  --pp_closing_iterations "$PP_CLOSING_ITERATIONS"
  $( [ "$PP_NO_REMOVE_SMALL_OBJECTS" = true ] && echo "--pp_no_remove_small_objects" )
  $( [ "$PP_UNION_HOLE_FILL" = true ] && echo "--pp_union_hole_fill" )
  $( [ "$PP_UNION_GAP_FILL" = true ] && echo "--pp_union_gap_fill" )
  $( [ "$PP_UNION_GAP_FILL" = true ] && echo "--pp_union_gap_closing_iterations $PP_UNION_GAP_CLOSING_ITERATIONS" )
  --pp_fill_interior_class "$PP_FILL_CLASS"
  --pp_fill_interior_target "$PP_FILL_TARGET"
  --pp_overwrite
  --gpu_ids $GPU_IDS
)

echo "-> Running head right camera (with scanline fill)"
python "${SAM3_DIR}/batch_run_parallel.py" \
  --base_dir "$BASE_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$BASE_OUTPUT_DIR" \
  --cameras "$HEAD_RIGHT_CAMERA" \
  --prompts "${PROMPTS[@]}" \
  $( [ "$PP_SCANLINE_FILL" = true ] && echo "--pp_scanline_fill" ) \
  $( [ "$PP_SCANLINE_FILL" = true ] && echo "--pp_scanline_source_label $PP_SCANLINE_SOURCE_LABEL" ) \
  $( [ "$PP_SCANLINE_FILL" = true ] && echo "--pp_scanline_fill_value $PP_SCANLINE_FILL_VALUE" ) \
  "${PP_COMMON_ARGS[@]}"

echo "-> Running right arm camera"
python "${SAM3_DIR}/batch_run_parallel.py" \
  --base_dir "$BASE_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$BASE_OUTPUT_DIR" \
  --cameras "$RIGHT_ARM_CAMERA" \
  --prompts "${PROMPTS[@]}" \
  "${PP_COMMON_ARGS[@]}"

echo "🎉 Batch segmentation job finished!"
