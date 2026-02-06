#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Configuration
CHECKPOINT="${SAM3_DIR}/sam3.pt"
BASE_DIR="/localhome/local-vennw/code/task3_01210122_merged/videos/chunk-000"
BASE_OUTPUT_DIR="/localhome/local-vennw/code/task3_01210122_merged/sam3_output"

# Prompts (per camera)
HEAD_LEFT_PROMPTS=("blue table" "robotic arm(s)" "silver box")
HEAD_RIGHT_PROMPTS=("blue table" "robotic arm(s)" "silver box")
LEFT_ARM_PROMPTS=("robotic arm(s)")
RIGHT_ARM_PROMPTS=("blue table" "robotic arm(s)" "silver box" "metal tool")

# Postprocess settings (per camera; left arm disabled below)
PP_NUM_WORKERS=64
PP_MIN_HOLE_SIZE=64
PP_MIN_OBJECT_SIZE=50
PP_CLOSING_ITERATIONS=1
PP_NO_REMOVE_SMALL_OBJECTS=true
PP_UNION_HOLE_FILL=true
PP_UNION_GAP_FILL=true
PP_UNION_GAP_CLOSING_ITERATIONS=2
HEAD_LEFT_PP_FILL_CLASS="1,2,3"
HEAD_LEFT_PP_FILL_TARGET=4
HEAD_RIGHT_PP_FILL_CLASS="1,2,3"
HEAD_RIGHT_PP_FILL_TARGET=4
RIGHT_ARM_PP_FILL_CLASS="1,2,3,4"
RIGHT_ARM_PP_FILL_TARGET=5

# ROI background fill rules (format: frame_start,frame_end_ratio,y_min,y_max,x_min,x_max,target)
HEAD_LEFT_FILL_BG_ROI="0,0.9,180,420,160,450,${HEAD_LEFT_PP_FILL_TARGET}"
HEAD_RIGHT_FILL_BG_ROI="0,0.9,180,420,160,450,${HEAD_RIGHT_PP_FILL_TARGET}"
RIGHT_ARM_FILL_BG_ROI="10,0.5,60,420,-1,-1,${RIGHT_ARM_PP_FILL_TARGET}"

# Head left camera
HEAD_LEFT_CAMERA="observation.images.head_left_camera_color_optical_frame"

# Head right camera
HEAD_RIGHT_CAMERA="observation.images.head_right_camera_color_optical_frame"

# Left arm camera
LEFT_ARM_CAMERA="observation.images.left_arm_camera_color_optical_frame"

# Right arm camera
RIGHT_ARM_CAMERA="observation.images.right_arm_camera_color_optical_frame"

# Use 8 GPUs (0-7)
GPU_IDS="0"

echo "🚀 Starting Parallel Segmentation Job (task3 4 cams, 8 GPUs)"
echo "--------------------------------------------------------------"
echo "SAM3 Dir:  $SAM3_DIR"
echo "Base Dir:  $BASE_DIR"
echo "Output:    $BASE_OUTPUT_DIR"
echo "Cameras:   ${HEAD_LEFT_CAMERA}, ${HEAD_RIGHT_CAMERA}, ${LEFT_ARM_CAMERA}, ${RIGHT_ARM_CAMERA}"
echo "Head Left Prompts:  ${HEAD_LEFT_PROMPTS[*]}"
echo "Head Right Prompts: ${HEAD_RIGHT_PROMPTS[*]}"
echo "Left Arm Prompts:   ${LEFT_ARM_PROMPTS[*]}"
echo "Right Arm Prompts:  ${RIGHT_ARM_PROMPTS[*]}"
echo "GPUs:      ${GPU_IDS}"
echo "--------------------------------------------------------------"

echo "-> Running head left camera"
python "${SAM3_DIR}/batch_run_parallel.py" \
  --base_dir "$BASE_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$BASE_OUTPUT_DIR" \
  --cameras "$HEAD_LEFT_CAMERA" \
  --prompts "${HEAD_LEFT_PROMPTS[@]}" \
  --save_npz \
  --no_pkl \
  --save_side_by_side \
  --postprocess \
  --pp_min_hole_size "$PP_MIN_HOLE_SIZE" \
  --pp_min_object_size "$PP_MIN_OBJECT_SIZE" \
  --pp_closing_iterations "$PP_CLOSING_ITERATIONS" \
  $( [ "$PP_NO_REMOVE_SMALL_OBJECTS" = true ] && echo "--pp_no_remove_small_objects" ) \
  $( [ "$PP_UNION_HOLE_FILL" = true ] && echo "--pp_union_hole_fill" ) \
  $( [ "$PP_UNION_GAP_FILL" = true ] && echo "--pp_union_gap_fill" ) \
  $( [ "$PP_UNION_GAP_FILL" = true ] && echo "--pp_union_gap_closing_iterations" "$PP_UNION_GAP_CLOSING_ITERATIONS" ) \
  --pp_fill_interior_class "$HEAD_LEFT_PP_FILL_CLASS" \
  --pp_fill_interior_target "$HEAD_LEFT_PP_FILL_TARGET" \
  --pp_fill_bg_roi "$HEAD_LEFT_FILL_BG_ROI" \
  --pp_overwrite \
  --skip_if_exists \
  --pp_num_workers "$PP_NUM_WORKERS" \
  --gpu_ids $GPU_IDS

echo "-> Running head right camera"
python "${SAM3_DIR}/batch_run_parallel.py" \
  --base_dir "$BASE_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$BASE_OUTPUT_DIR" \
  --cameras "$HEAD_RIGHT_CAMERA" \
  --prompts "${HEAD_RIGHT_PROMPTS[@]}" \
  --save_npz \
  --no_pkl \
  --save_side_by_side \
  --postprocess \
  --pp_min_hole_size "$PP_MIN_HOLE_SIZE" \
  --pp_min_object_size "$PP_MIN_OBJECT_SIZE" \
  --pp_closing_iterations "$PP_CLOSING_ITERATIONS" \
  $( [ "$PP_NO_REMOVE_SMALL_OBJECTS" = true ] && echo "--pp_no_remove_small_objects" ) \
  $( [ "$PP_UNION_HOLE_FILL" = true ] && echo "--pp_union_hole_fill" ) \
  $( [ "$PP_UNION_GAP_FILL" = true ] && echo "--pp_union_gap_fill" ) \
  $( [ "$PP_UNION_GAP_FILL" = true ] && echo "--pp_union_gap_closing_iterations" "$PP_UNION_GAP_CLOSING_ITERATIONS" ) \
  --pp_fill_interior_class "$HEAD_RIGHT_PP_FILL_CLASS" \
  --pp_fill_interior_target "$HEAD_RIGHT_PP_FILL_TARGET" \
  --pp_fill_bg_roi "$HEAD_RIGHT_FILL_BG_ROI" \
  --pp_overwrite \
  --skip_if_exists \
  --pp_num_workers "$PP_NUM_WORKERS" \
  --gpu_ids $GPU_IDS

echo "-> Running left arm camera"
python "${SAM3_DIR}/batch_run_parallel.py" \
  --base_dir "$BASE_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$BASE_OUTPUT_DIR" \
  --cameras "$LEFT_ARM_CAMERA" \
  --prompts "${LEFT_ARM_PROMPTS[@]}" \
  --save_npz \
  --no_pkl \
  --save_side_by_side \
  --skip_if_exists \
  --gpu_ids $GPU_IDS

echo "-> Running right arm camera"
python "${SAM3_DIR}/batch_run_parallel.py" \
  --base_dir "$BASE_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$BASE_OUTPUT_DIR" \
  --cameras "$RIGHT_ARM_CAMERA" \
  --prompts "${RIGHT_ARM_PROMPTS[@]}" \
  --save_npz \
  --no_pkl \
  --save_side_by_side \
  --postprocess \
  --pp_min_hole_size "$PP_MIN_HOLE_SIZE" \
  --pp_min_object_size "$PP_MIN_OBJECT_SIZE" \
  --pp_closing_iterations "$PP_CLOSING_ITERATIONS" \
  $( [ "$PP_NO_REMOVE_SMALL_OBJECTS" = true ] && echo "--pp_no_remove_small_objects" ) \
  $( [ "$PP_UNION_HOLE_FILL" = true ] && echo "--pp_union_hole_fill" ) \
  $( [ "$PP_UNION_GAP_FILL" = true ] && echo "--pp_union_gap_fill" ) \
  $( [ "$PP_UNION_GAP_FILL" = true ] && echo "--pp_union_gap_closing_iterations" "$PP_UNION_GAP_CLOSING_ITERATIONS" ) \
  --pp_fill_interior_class "$RIGHT_ARM_PP_FILL_CLASS" \
  --pp_fill_interior_target "$RIGHT_ARM_PP_FILL_TARGET" \
  --pp_fill_bg_roi "$RIGHT_ARM_FILL_BG_ROI" \
  --pp_overwrite \
  --skip_if_exists \
  --pp_num_workers "$PP_NUM_WORKERS" \
  --gpu_ids $GPU_IDS

echo "🎉 Batch segmentation job finished!"
