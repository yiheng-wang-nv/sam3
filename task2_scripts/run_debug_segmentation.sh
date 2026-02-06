#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Configuration
CHECKPOINT="${SAM3_DIR}/sam3.pt"
BASE_DIR="/localhome/local-vennw/code/task2_01210123012601280130_600merged/videos/chunk-000"
BASE_OUTPUT_DIR="/localhome/local-vennw/code/task2_01210123012601280130_600merged/sam3_output_debug"

# Prompts (per camera)
HEAD_LEFT_PROMPTS=("blue table" "robotic arm(s)" "silver box")
HEAD_RIGHT_PROMPTS=("blue table" "robotic arm(s)" "silver box")
LEFT_ARM_PROMPTS=("blue table" "robotic arm(s)" "silver box")
RIGHT_ARM_PROMPTS=("blue table" "robotic arm(s)" "silver box")

# Postprocess settings (per camera)
PP_MIN_HOLE_SIZE=64
PP_MIN_OBJECT_SIZE=50
PP_CLOSING_ITERATIONS=1
PP_NO_REMOVE_SMALL_OBJECTS=true
PP_UNION_HOLE_FILL=true
PP_UNION_GAP_FILL=true
PP_UNION_GAP_CLOSING_ITERATIONS=2
LEFT_ARM_BLUE_TABLE_QUADRANT=true
LEFT_ARM_BLUE_TABLE_LABEL=1
LEFT_ARM_BLUE_TABLE_TARGET=4
BLUE_TABLE_Y_PAD_TOP=60
BLUE_TABLE_Y_PAD_BOTTOM=60
BLUE_TABLE_SKIP_IF_LABEL_ABOVE=3
BLUE_TABLE_SKIP_IF_LABEL_AREA_GT=2500
RIGHT_ARM_BLUE_TABLE_QUADRANT=true
RIGHT_ARM_BLUE_TABLE_LABEL=1
RIGHT_ARM_BLUE_TABLE_TARGET=4
RIGHT_ARM_BLUE_TABLE_MODE="left_down"
HEAD_LEFT_PP_FILL_CLASS="1,2,3"
HEAD_LEFT_PP_FILL_TARGET=4
HEAD_RIGHT_PP_FILL_CLASS="1,2,3"
HEAD_RIGHT_PP_FILL_TARGET=4
RIGHT_ARM_PP_FILL_CLASS="1,2,3"
RIGHT_ARM_PP_FILL_TARGET=4

# Head left camera
HEAD_LEFT_CAMERA="observation.images.head_left_camera_color_optical_frame"

# Head right camera
HEAD_RIGHT_CAMERA="observation.images.head_right_camera_color_optical_frame"

# Left arm camera
LEFT_ARM_CAMERA="observation.images.left_arm_camera_color_optical_frame"

# Right arm camera
RIGHT_ARM_CAMERA="observation.images.right_arm_camera_color_optical_frame"

# Use 1 GPU for debug
GPU_IDS="0"

# Set to "false" to run full (non-debug) instead of one random video
DEBUG_ONE=true
# Base seed for per-camera random selection
DEBUG_SEED_BASE=10

DEBUG_FLAGS_HEAD_LEFT=()
DEBUG_FLAGS_HEAD_RIGHT=()
DEBUG_FLAGS_LEFT_ARM=()
DEBUG_FLAGS_RIGHT_ARM=()
if [ "$DEBUG_ONE" = true ]; then
  DEBUG_FLAGS_HEAD_LEFT+=(--debug_one --debug_seed $((DEBUG_SEED_BASE + 1)))
  DEBUG_FLAGS_HEAD_RIGHT+=(--debug_one --debug_seed $((DEBUG_SEED_BASE + 2)))
  DEBUG_FLAGS_LEFT_ARM+=(--debug_one --debug_seed $((DEBUG_SEED_BASE + 3)))
  DEBUG_FLAGS_RIGHT_ARM+=(--debug_one --debug_seed $((DEBUG_SEED_BASE + 4)))
fi

get_latest_episode() {
  local camera_dir="$1"
  local latest
  latest=$(ls -t "${camera_dir}"/*_compare.mp4 2>/dev/null | head -n 1)
  if [ -z "$latest" ]; then
    echo ""
    return
  fi
  basename "$latest" | sed "s/_compare.mp4$//"
}

postprocess_and_merge() {
  local camera="$1"
  local fill_class="$2"
  local fill_target="$3"

  local camera_dir="${BASE_OUTPUT_DIR}/${camera}"
  local episode
  episode=$(get_latest_episode "$camera_dir")
  if [ -z "$episode" ]; then
    echo "No compare video found for ${camera}, skipping post/merge."
    return
  fi

  EXTRA_ARGS=()
  if [ "$camera" = "$LEFT_ARM_CAMERA" ] && [ "$LEFT_ARM_BLUE_TABLE_QUADRANT" = true ]; then
    EXTRA_ARGS+=(--fill_blue_table_quadrant)
    EXTRA_ARGS+=(--blue_table_label "$LEFT_ARM_BLUE_TABLE_LABEL")
    EXTRA_ARGS+=(--blue_table_target "$LEFT_ARM_BLUE_TABLE_TARGET")
    EXTRA_ARGS+=(--blue_table_quadrant_mode "right_down")
    EXTRA_ARGS+=(--blue_table_y_pad_top "$BLUE_TABLE_Y_PAD_TOP")
    EXTRA_ARGS+=(--blue_table_y_pad_bottom "$BLUE_TABLE_Y_PAD_BOTTOM")
    EXTRA_ARGS+=(--blue_table_skip_if_label_above "$BLUE_TABLE_SKIP_IF_LABEL_ABOVE")
    EXTRA_ARGS+=(--blue_table_skip_if_label_area_gt "$BLUE_TABLE_SKIP_IF_LABEL_AREA_GT")
  fi
  if [ "$camera" = "$RIGHT_ARM_CAMERA" ] && [ "$RIGHT_ARM_BLUE_TABLE_QUADRANT" = true ]; then
    EXTRA_ARGS+=(--fill_blue_table_quadrant)
    EXTRA_ARGS+=(--blue_table_label "$RIGHT_ARM_BLUE_TABLE_LABEL")
    EXTRA_ARGS+=(--blue_table_target "$RIGHT_ARM_BLUE_TABLE_TARGET")
    EXTRA_ARGS+=(--blue_table_quadrant_mode "$RIGHT_ARM_BLUE_TABLE_MODE")
    EXTRA_ARGS+=(--blue_table_y_pad_top "$BLUE_TABLE_Y_PAD_TOP")
    EXTRA_ARGS+=(--blue_table_y_pad_bottom "$BLUE_TABLE_Y_PAD_BOTTOM")
    EXTRA_ARGS+=(--blue_table_skip_if_label_above "$BLUE_TABLE_SKIP_IF_LABEL_ABOVE")
    EXTRA_ARGS+=(--blue_table_skip_if_label_area_gt "$BLUE_TABLE_SKIP_IF_LABEL_AREA_GT")
  fi

  python "${SAM3_DIR}/postprocess_and_visualize.py" \
    --input_dir "$BASE_OUTPUT_DIR" \
    --camera "$camera" \
    --episode "$episode" \
    --videos_dir "$BASE_DIR" \
    --min_hole_size "$PP_MIN_HOLE_SIZE" \
    --min_object_size "$PP_MIN_OBJECT_SIZE" \
    --closing_iterations "$PP_CLOSING_ITERATIONS" \
    $( [ "$PP_NO_REMOVE_SMALL_OBJECTS" = true ] && echo "--no_remove_small_objects" ) \
    $( [ "$PP_UNION_HOLE_FILL" = true ] && echo "--union_hole_fill" ) \
    $( [ "$PP_UNION_GAP_FILL" = true ] && echo "--union_gap_fill" ) \
    $( [ "$PP_UNION_GAP_FILL" = true ] && echo "--union_gap_closing_iterations" "$PP_UNION_GAP_CLOSING_ITERATIONS" ) \
    --fill_interior_class "$fill_class" \
    --fill_interior_target "$fill_target" \
    "${EXTRA_ARGS[@]}"

  python "${SAM3_DIR}/combine_compare_and_post.py" \
    --compare_video "${camera_dir}/${episode}_compare.mp4" \
    --post_video "${camera_dir}/${episode}_compare_post.mp4" \
    --output_path "${camera_dir}/${episode}_compare_merged.mp4"
}

echo "🧪 Starting Debug Segmentation (task2 4 cams)"
echo "------------------------------------------------"
echo "SAM3 Dir:  $SAM3_DIR"
echo "Base Dir:  $BASE_DIR"
echo "Output:    $BASE_OUTPUT_DIR"
echo "Head Left Prompts:  ${HEAD_LEFT_PROMPTS[*]}"
echo "Head Right Prompts: ${HEAD_RIGHT_PROMPTS[*]}"
echo "Left Arm Prompts:   ${LEFT_ARM_PROMPTS[*]}"
echo "Right Arm Prompts:  ${RIGHT_ARM_PROMPTS[*]}"
echo "GPU:       ${GPU_IDS}"
echo "------------------------------------------------"

echo "-> Debug head left camera (one random video)"
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
  --pp_overwrite \
  "${DEBUG_FLAGS_HEAD_LEFT[@]}" \
  --gpu_ids $GPU_IDS
postprocess_and_merge "$HEAD_LEFT_CAMERA" "$HEAD_LEFT_PP_FILL_CLASS" "$HEAD_LEFT_PP_FILL_TARGET"

echo "-> Debug head right camera (one random video)"
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
  --pp_overwrite \
  "${DEBUG_FLAGS_HEAD_RIGHT[@]}" \
  --gpu_ids $GPU_IDS
postprocess_and_merge "$HEAD_RIGHT_CAMERA" "$HEAD_RIGHT_PP_FILL_CLASS" "$HEAD_RIGHT_PP_FILL_TARGET"

echo "-> Debug left arm camera (one random video)"
python "${SAM3_DIR}/batch_run_parallel.py" \
  --base_dir "$BASE_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$BASE_OUTPUT_DIR" \
  --cameras "$LEFT_ARM_CAMERA" \
  --prompts "${LEFT_ARM_PROMPTS[@]}" \
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
  $( [ "$LEFT_ARM_BLUE_TABLE_QUADRANT" = true ] && echo "--pp_fill_blue_table_quadrant" ) \
  $( [ "$LEFT_ARM_BLUE_TABLE_QUADRANT" = true ] && echo "--pp_blue_table_label" "$LEFT_ARM_BLUE_TABLE_LABEL" ) \
  $( [ "$LEFT_ARM_BLUE_TABLE_QUADRANT" = true ] && echo "--pp_blue_table_target" "$LEFT_ARM_BLUE_TABLE_TARGET" ) \
  $( [ "$LEFT_ARM_BLUE_TABLE_QUADRANT" = true ] && echo "--pp_blue_table_quadrant_mode" "right_down" ) \
  $( [ "$LEFT_ARM_BLUE_TABLE_QUADRANT" = true ] && echo "--pp_blue_table_y_pad_top" "$BLUE_TABLE_Y_PAD_TOP" ) \
  $( [ "$LEFT_ARM_BLUE_TABLE_QUADRANT" = true ] && echo "--pp_blue_table_y_pad_bottom" "$BLUE_TABLE_Y_PAD_BOTTOM" ) \
  $( [ "$LEFT_ARM_BLUE_TABLE_QUADRANT" = true ] && echo "--pp_blue_table_skip_if_label_above" "$BLUE_TABLE_SKIP_IF_LABEL_ABOVE" ) \
  $( [ "$LEFT_ARM_BLUE_TABLE_QUADRANT" = true ] && echo "--pp_blue_table_skip_if_label_area_gt" "$BLUE_TABLE_SKIP_IF_LABEL_AREA_GT" ) \
  --pp_fill_interior_class "$HEAD_LEFT_PP_FILL_CLASS" \
  --pp_fill_interior_target "$HEAD_LEFT_PP_FILL_TARGET" \
  --pp_overwrite \
  --skip_if_exists \
  "${DEBUG_FLAGS_LEFT_ARM[@]}" \
  --gpu_ids $GPU_IDS
postprocess_and_merge "$LEFT_ARM_CAMERA" "$HEAD_LEFT_PP_FILL_CLASS" "$HEAD_LEFT_PP_FILL_TARGET"

echo "-> Debug right arm camera (one random video)"
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
  $( [ "$RIGHT_ARM_BLUE_TABLE_QUADRANT" = true ] && echo "--pp_fill_blue_table_quadrant" ) \
  $( [ "$RIGHT_ARM_BLUE_TABLE_QUADRANT" = true ] && echo "--pp_blue_table_label" "$RIGHT_ARM_BLUE_TABLE_LABEL" ) \
  $( [ "$RIGHT_ARM_BLUE_TABLE_QUADRANT" = true ] && echo "--pp_blue_table_target" "$RIGHT_ARM_BLUE_TABLE_TARGET" ) \
  $( [ "$RIGHT_ARM_BLUE_TABLE_QUADRANT" = true ] && echo "--pp_blue_table_quadrant_mode" "$RIGHT_ARM_BLUE_TABLE_MODE" ) \
  $( [ "$RIGHT_ARM_BLUE_TABLE_QUADRANT" = true ] && echo "--pp_blue_table_y_pad_top" "$BLUE_TABLE_Y_PAD_TOP" ) \
  $( [ "$RIGHT_ARM_BLUE_TABLE_QUADRANT" = true ] && echo "--pp_blue_table_y_pad_bottom" "$BLUE_TABLE_Y_PAD_BOTTOM" ) \
  $( [ "$RIGHT_ARM_BLUE_TABLE_QUADRANT" = true ] && echo "--pp_blue_table_skip_if_label_above" "$BLUE_TABLE_SKIP_IF_LABEL_ABOVE" ) \
  $( [ "$RIGHT_ARM_BLUE_TABLE_QUADRANT" = true ] && echo "--pp_blue_table_skip_if_label_area_gt" "$BLUE_TABLE_SKIP_IF_LABEL_AREA_GT" ) \
  --pp_fill_interior_class "$RIGHT_ARM_PP_FILL_CLASS" \
  --pp_fill_interior_target "$RIGHT_ARM_PP_FILL_TARGET" \
  --pp_overwrite \
  --skip_if_exists \
  "${DEBUG_FLAGS_RIGHT_ARM[@]}" \
  --gpu_ids $GPU_IDS
postprocess_and_merge "$RIGHT_ARM_CAMERA" "$RIGHT_ARM_PP_FILL_CLASS" "$RIGHT_ARM_PP_FILL_TARGET"

echo "✅ Debug segmentation finished!"
