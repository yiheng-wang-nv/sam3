#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Configuration
CHECKPOINT="${SAM3_DIR}/sam3.pt"
BASE_DIR="/localhome/local-vennw/code/task4-1_022602270304_merged/videos/chunk-000"
BASE_OUTPUT_DIR="/localhome/local-vennw/code/task4-1_022602270304_merged/sam3_output"

# Camera names
HEAD_RIGHT_CAMERA="observation.images.head_right_camera_color_optical_frame"
LEFT_ARM_CAMERA="observation.images.left_arm_camera_color_optical_frame"
RIGHT_ARM_CAMERA="observation.images.right_arm_camera_color_optical_frame"

# ── Per-camera configuration ──
CAMERAS=(
    "$HEAD_RIGHT_CAMERA"
    "$LEFT_ARM_CAMERA"
    "$RIGHT_ARM_CAMERA"
)

GPU_IDS="1 2 3 5 6 7"

get_camera_prompts() {
  local cam="$1"
  case "$cam" in
    "$RIGHT_ARM_CAMERA")
      REPLY_PROMPTS=("blue table" "metal items" "red item" "robotic arm(s)" "blue item")
      ;;
    "$LEFT_ARM_CAMERA")
      REPLY_PROMPTS=("blue table" "metal items" "robotic arm(s)" "tools")
      ;;
    "$HEAD_RIGHT_CAMERA")
      REPLY_PROMPTS=("blue table" "metal items" "red item" "robot" "blue item")
      ;;
    *)
      REPLY_PROMPTS=("blue table" "metal items" "robotic arm(s)")
      ;;
  esac
}

get_camera_point_clicks() {
  local cam="$1"
  case "$cam" in
    "$RIGHT_ARM_CAMERA")
      echo "${SCRIPT_DIR}/point_clicks_right_arm.json"
      ;;
    *)
      echo ""
      ;;
  esac
}

get_camera_pp_extra() {
  local cam="$1"
  REPLY_PP_EXTRA=()
  case "$cam" in
    "$RIGHT_ARM_CAMERA")
      REPLY_PP_EXTRA=(--pp_leftmost_rect --pp_leftmost_rect_fill 6)
      ;;
  esac
}

# Postprocess settings (shared)
PP_NUM_WORKERS=96
PP_MIN_HOLE_SIZE=64
PP_MIN_OBJECT_SIZE=50
PP_CLOSING_ITERATIONS=1
PP_NO_REMOVE_SMALL_OBJECTS=true
PP_UNION_HOLE_FILL=true
PP_UNION_GAP_FILL=true
PP_UNION_GAP_CLOSING_ITERATIONS=1
PP_FILL_CLASS="1,2,3,4,5"
PP_FILL_TARGET=6

# Build common postprocess flags
PP_FLAGS=(
  --pp_num_workers "$PP_NUM_WORKERS"
  --pp_min_hole_size "$PP_MIN_HOLE_SIZE"
  --pp_min_object_size "$PP_MIN_OBJECT_SIZE"
  --pp_closing_iterations "$PP_CLOSING_ITERATIONS"
  --pp_fill_interior_class "$PP_FILL_CLASS"
  --pp_fill_interior_target "$PP_FILL_TARGET"
  --pp_overwrite
)
[ "$PP_NO_REMOVE_SMALL_OBJECTS" = true ] && PP_FLAGS+=(--pp_no_remove_small_objects)
[ "$PP_UNION_HOLE_FILL" = true ] && PP_FLAGS+=(--pp_union_hole_fill)
if [ "$PP_UNION_GAP_FILL" = true ]; then
  PP_FLAGS+=(--pp_union_gap_fill --pp_union_gap_closing_iterations "$PP_UNION_GAP_CLOSING_ITERATIONS")
fi

echo "🚀 Starting Parallel Segmentation Job (task4-1_20260226_trimmed)"
echo "--------------------------------------------------------------------"
echo "SAM3 Dir:  $SAM3_DIR"
echo "Base Dir:  $BASE_DIR"
echo "Output:    $BASE_OUTPUT_DIR"
echo "Cameras:   ${CAMERAS[*]}"
echo "GPUs:      ${GPU_IDS}"
echo "--------------------------------------------------------------------"

for cam in "${CAMERAS[@]}"; do
  get_camera_prompts "$cam"
  local_point_clicks=$(get_camera_point_clicks "$cam")
  get_camera_pp_extra "$cam"

  CAM_FLAGS=()
  if [ -n "$local_point_clicks" ] && [ -f "$local_point_clicks" ]; then
    CAM_FLAGS+=(--point_clicks_json "$local_point_clicks")
    echo "  [$cam] Point clicks: $local_point_clicks"
  fi

  echo "  [$cam] Prompts: ${REPLY_PROMPTS[*]}"

  python "${SAM3_DIR}/batch_run_parallel.py" \
    --base_dir "$BASE_DIR" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$BASE_OUTPUT_DIR" \
    --cameras "$cam" \
    --prompts "${REPLY_PROMPTS[@]}" \
    --save_npz \
    --no_pkl \
    --skip_if_exists \
    --postprocess \
    "${PP_FLAGS[@]}" \
    "${REPLY_PP_EXTRA[@]}" \
    "${CAM_FLAGS[@]}" \
    --gpu_ids $GPU_IDS
done

echo "🎉 Batch segmentation job finished!"
