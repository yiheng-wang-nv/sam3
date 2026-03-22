#!/bin/bash

# Randomly sample N videos per camera, run inference + postprocess, and generate
# side-by-side comparison videos to visually inspect the results.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Configuration
CHECKPOINT="${SAM3_DIR}/sam3.pt"
DATASET_DIR="/localhome/local-vennw/code/task5-2_030203090311_newinitpose_merged"
BASE_DIR="${DATASET_DIR}/videos/chunk-000"
BASE_OUTPUT_DIR="${DATASET_DIR}/sam3_output"

HEAD_RIGHT_CAMERA="observation.images.head_right_camera_color_optical_frame"
LEFT_ARM_CAMERA="observation.images.left_arm_camera_color_optical_frame"
RIGHT_ARM_CAMERA="observation.images.right_arm_camera_color_optical_frame"

# Sampling
SAMPLE_N=4
SAMPLE_SEED=42
GPU_IDS="0 1 2 3"

# ── Per-camera prompts ──
get_camera_prompts() {
  local cam="$1"
  case "$cam" in
    "$RIGHT_ARM_CAMERA")
      REPLY_PROMPTS=("blue table" "metal items" "robotic arm(s)" "blue item")
      ;;
    "$LEFT_ARM_CAMERA")
      REPLY_PROMPTS=("blue table" "metal items" "robotic arm(s)" "tools")
      ;;
    "$HEAD_RIGHT_CAMERA")
      REPLY_PROMPTS=("blue table" "metal items" "robot" "blue item")
      ;;
    *)
      REPLY_PROMPTS=("blue table" "metal items" "robotic arm(s)")
      ;;
  esac
}

# ── Per-camera point clicks ──
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

# ── Per-camera fill interior ──
get_camera_fill_interior() {
  local cam="$1"
  case "$cam" in
    "$HEAD_RIGHT_CAMERA") REPLY_FILL_CLASS="1,2,3,4"; REPLY_FILL_TARGET=5 ;;
    "$LEFT_ARM_CAMERA")   REPLY_FILL_CLASS="1,2,3,4"; REPLY_FILL_TARGET=5 ;;
    "$RIGHT_ARM_CAMERA")  REPLY_FILL_CLASS="1,2,3,4"; REPLY_FILL_TARGET=5 ;;
    *) REPLY_FILL_CLASS="1,2,3,4"; REPLY_FILL_TARGET=5 ;;
  esac
}

# ── Per-camera extra PP flags ──
get_camera_pp_extra() {
  local cam="$1"
  REPLY_PP_EXTRA=()
  case "$cam" in
    # "$RIGHT_ARM_CAMERA")
    #   REPLY_PP_EXTRA=(--pp_halves_rect --pp_halves_rect_label 1 --pp_halves_rect_fill 5 --pp_halves_rect_y_max 420)
    #   ;;
  esac
}

# Postprocess settings (shared)
PP_NUM_WORKERS=32
PP_MIN_HOLE_SIZE=64
PP_MIN_OBJECT_SIZE=50
PP_CLOSING_ITERATIONS=1
PP_NO_REMOVE_SMALL_OBJECTS=true
PP_UNION_HOLE_FILL=true
PP_UNION_GAP_FILL=true
PP_UNION_GAP_CLOSING_ITERATIONS=1

CAMERAS=(
    # "$HEAD_RIGHT_CAMERA"
    # "$LEFT_ARM_CAMERA"
    "$RIGHT_ARM_CAMERA"
)

echo "🔍 Sample Segmentation Test (task5-2) - ${SAMPLE_N} random videos per camera"
echo "--------------------------------------------------------------------"
echo "SAM3 Dir:  $SAM3_DIR"
echo "Base Dir:  $BASE_DIR"
echo "Output:    $BASE_OUTPUT_DIR"
echo "GPUs:      ${GPU_IDS}"
echo "--------------------------------------------------------------------"

build_pp_flags() {
  local fill_class="$1"
  local fill_target="$2"
  PP_FLAGS=(
    --pp_num_workers "$PP_NUM_WORKERS"
    --pp_min_hole_size "$PP_MIN_HOLE_SIZE"
    --pp_min_object_size "$PP_MIN_OBJECT_SIZE"
    --pp_closing_iterations "$PP_CLOSING_ITERATIONS"
    --pp_overwrite
  )
  [ "$PP_NO_REMOVE_SMALL_OBJECTS" = true ] && PP_FLAGS+=(--pp_no_remove_small_objects)
  [ "$PP_UNION_HOLE_FILL" = true ] && PP_FLAGS+=(--pp_union_hole_fill)
  if [ "$PP_UNION_GAP_FILL" = true ]; then
    PP_FLAGS+=(--pp_union_gap_fill --pp_union_gap_closing_iterations "$PP_UNION_GAP_CLOSING_ITERATIONS")
  fi
  if [ -n "$fill_class" ]; then
    PP_FLAGS+=(--pp_fill_interior_class "$fill_class" --pp_fill_interior_target "$fill_target")
  fi
}

for cam in "${CAMERAS[@]}"; do
  get_camera_prompts "$cam"
  get_camera_fill_interior "$cam"
  local_point_clicks=$(get_camera_point_clicks "$cam")
  get_camera_pp_extra "$cam"
  build_pp_flags "$REPLY_FILL_CLASS" "$REPLY_FILL_TARGET"

  CAM_FLAGS=()
  if [ -n "$local_point_clicks" ] && [ -f "$local_point_clicks" ]; then
    CAM_FLAGS+=(--point_clicks_json "$local_point_clicks")
    echo "  [$cam] Point clicks: $local_point_clicks"
  fi

  echo "-> Running ${cam} (${SAMPLE_N} samples)"
  echo "   Prompts: ${REPLY_PROMPTS[*]}"
  echo "   Fill: ${REPLY_FILL_CLASS} -> ${REPLY_FILL_TARGET}"

  python "${SAM3_DIR}/batch_run_parallel.py" \
    --base_dir "$BASE_DIR" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$BASE_OUTPUT_DIR" \
    --cameras "$cam" \
    --prompts "${REPLY_PROMPTS[@]}" \
    --save_npz \
    --no_pkl \
    --save_side_by_side \
    --postprocess_for_vis \
    --postprocess \
    --skip_if_exists \
    --debug_n "$SAMPLE_N" \
    --debug_seed "$SAMPLE_SEED" \
    "${PP_FLAGS[@]}" \
    "${REPLY_PP_EXTRA[@]}" \
    "${CAM_FLAGS[@]}" \
    --gpu_ids $GPU_IDS
done

echo "🎉 Sample segmentation test finished!"
echo "Compare videos saved to: $BASE_OUTPUT_DIR/*/episode_*_compare.mp4"
