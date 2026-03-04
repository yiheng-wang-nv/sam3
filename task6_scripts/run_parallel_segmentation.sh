#!/bin/bash

# Run full parallel segmentation on task6 (all episodes, all cameras).

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# ──────────────── Configuration ────────────────
CHECKPOINT="${SAM3_DIR}/sam3.pt"
DATASET_DIR="/localhome/local-vennw/code/sztask6_020902100211021202260227_merged"
CHUNKS=("chunk-000" "chunk-001")

HEAD_LEFT_CAMERA="observation.images.head_left_camera_color_optical_frame"
HEAD_RIGHT_CAMERA="observation.images.head_right_camera_color_optical_frame"
LEFT_ARM_CAMERA="observation.images.left_arm_camera_color_optical_frame"
RIGHT_ARM_CAMERA="observation.images.right_arm_camera_color_optical_frame"

CAMERAS=(
    "$LEFT_ARM_CAMERA"
    "$RIGHT_ARM_CAMERA"
    "$HEAD_LEFT_CAMERA"
    "$HEAD_RIGHT_CAMERA"
)

GPU_IDS="0 1 2 3 5 6 7"
WORKERS_PER_GPU=2
PP_OVERWRITE=false

# ── Per-camera prompts ──
get_camera_prompts() {
  local cam="$1"
  case "$cam" in
    "$HEAD_LEFT_CAMERA")
      REPLY_PROMPTS=("blue table" "robotic arm(s)" "silver box" "metal items")
      ;;
    "$HEAD_RIGHT_CAMERA")
      REPLY_PROMPTS=("blue table" "robotic arm(s)" "silver box" "metal items")
      ;;
    "$LEFT_ARM_CAMERA")
      REPLY_PROMPTS=("blue table" "robotic arm(s)" "silver box" "metal items")
      ;;
    "$RIGHT_ARM_CAMERA")
      REPLY_PROMPTS=("blue table" "robotic arm(s)" "silver box" "metal items")
      ;;
    *)
      REPLY_PROMPTS=("blue table" "robotic arm(s)" "silver box")
      ;;
  esac
}

# ── Per-camera point clicks JSON (empty = disabled) ──
get_camera_point_clicks() {
  local cam="$1"
  case "$cam" in
    "$HEAD_LEFT_CAMERA")
      echo "${SCRIPT_DIR}/point_clicks_head_left.json"
      ;;
    "$HEAD_RIGHT_CAMERA")
      echo "${SCRIPT_DIR}/point_clicks_head_right.json"
      ;;
    "$LEFT_ARM_CAMERA")
      echo "${SCRIPT_DIR}/point_clicks_left_arm.json"
      ;;
    "$RIGHT_ARM_CAMERA")
      echo "${SCRIPT_DIR}/point_clicks_right_arm.json"
      ;;
    *)
      echo ""
      ;;
  esac
}

# ── Per-camera extra PP flags ──
get_camera_pp_extra() {
  local cam="$1"
  REPLY_PP_EXTRA=()
  case "$cam" in
    "$HEAD_LEFT_CAMERA"|"$HEAD_RIGHT_CAMERA")
      REPLY_PP_EXTRA=(
        --pp_temporal_fill
        --pp_temporal_fill_labels "5" --pp_temporal_fill_union_labels "1,3,4"
        --pp_temporal_fill_value 5
      )
      ;;
    "$LEFT_ARM_CAMERA")
      REPLY_PP_EXTRA=(
        --pp_corner_rect --pp_corner_rect_mode topleft
        --pp_corner_rect_labels "1,3,4" --pp_corner_rect_fill 5
        --pp_corner_rect_y_max 420
        --pp_corner_rect_x_first_labels "3,4"
      )
      ;;
    "$RIGHT_ARM_CAMERA")
      REPLY_PP_EXTRA=(
        --pp_corner_rect --pp_corner_rect_mode topright
        --pp_corner_rect_labels "1,3,4" --pp_corner_rect_fill 5
        --pp_corner_rect_y_max 420
        --pp_corner_rect_x_first_labels "3,4"
      )
      ;;
  esac
}

# ── Per-camera fill interior ──
get_camera_fill_interior() {
  local cam="$1"
  case "$cam" in
    "$HEAD_LEFT_CAMERA"|"$HEAD_RIGHT_CAMERA")
      REPLY_FILL_CLASS="1,3,4"; REPLY_FILL_TARGET=5 ;;
    "$LEFT_ARM_CAMERA"|"$RIGHT_ARM_CAMERA")
      REPLY_FILL_CLASS="1,2,3,4"; REPLY_FILL_TARGET=5 ;;
    *)
      REPLY_FILL_CLASS=""; REPLY_FILL_TARGET="" ;;
  esac
}

# ── Postprocess settings (shared) ──
PP_NUM_WORKERS=96
PP_MIN_HOLE_SIZE=64
PP_MIN_OBJECT_SIZE=50
PP_CLOSING_ITERATIONS=1
PP_NO_REMOVE_SMALL_OBJECTS=true
PP_UNION_HOLE_FILL=true
PP_UNION_GAP_FILL=true
PP_UNION_GAP_CLOSING_ITERATIONS=2

# ──────────────── Build common PP flags ────────────────
build_pp_flags() {
  local fill_class="$1"
  local fill_target="$2"
  PP_FLAGS=(
    --pp_num_workers "$PP_NUM_WORKERS"
    --pp_min_hole_size "$PP_MIN_HOLE_SIZE"
    --pp_min_object_size "$PP_MIN_OBJECT_SIZE"
    --pp_closing_iterations "$PP_CLOSING_ITERATIONS"
  )
  [ "$PP_OVERWRITE" = true ] && PP_FLAGS+=(--pp_overwrite)
  [ "$PP_NO_REMOVE_SMALL_OBJECTS" = true ] && PP_FLAGS+=(--pp_no_remove_small_objects)
  [ "$PP_UNION_HOLE_FILL" = true ] && PP_FLAGS+=(--pp_union_hole_fill)
  if [ "$PP_UNION_GAP_FILL" = true ]; then
    PP_FLAGS+=(--pp_union_gap_fill --pp_union_gap_closing_iterations "$PP_UNION_GAP_CLOSING_ITERATIONS")
  fi
  if [ -n "$fill_class" ]; then
    PP_FLAGS+=(--pp_fill_interior_class "$fill_class" --pp_fill_interior_target "$fill_target")
  fi
}

# ──────────────── Header ────────────────
echo "🚀 Starting Parallel Segmentation Job (task6, 4 cameras)"
echo "--------------------------------------------------------------"
echo "SAM3 Dir:  $SAM3_DIR"
echo "Dataset:   $DATASET_DIR"
echo "Chunks:    ${CHUNKS[*]}"
echo "Cameras:   ${CAMERAS[*]}"
echo "GPUs:      ${GPU_IDS}"
echo "--------------------------------------------------------------"

# ──────────────── Run each chunk × camera ────────────────
for chunk in "${CHUNKS[@]}"; do
  BASE_DIR="${DATASET_DIR}/videos/${chunk}"
  BASE_OUTPUT_DIR="${DATASET_DIR}/sam3_output"

  if [ ! -d "$BASE_DIR" ]; then
    echo "⚠️  Skipping ${chunk}: ${BASE_DIR} not found"
    continue
  fi

  echo "============================================================"
  echo "  Processing ${chunk}"
  echo "============================================================"

  for cam in "${CAMERAS[@]}"; do
    get_camera_prompts "$cam"
    local_point_clicks=$(get_camera_point_clicks "$cam")
    get_camera_pp_extra "$cam"
    get_camera_fill_interior "$cam"
    build_pp_flags "$REPLY_FILL_CLASS" "$REPLY_FILL_TARGET"

    CAM_FLAGS=()
    if [ -n "$local_point_clicks" ] && [ -f "$local_point_clicks" ]; then
      CAM_FLAGS+=(--point_clicks_json "$local_point_clicks")
      echo "  [$cam] Point clicks: $local_point_clicks"
    fi

    echo "-> [${chunk}] Running ${cam}"
    echo "   Prompts: ${REPLY_PROMPTS[*]}"

    python "${SAM3_DIR}/batch_run_parallel.py" \
      --base_dir "$BASE_DIR" \
      --checkpoint "$CHECKPOINT" \
      --output_dir "$BASE_OUTPUT_DIR" \
      --cameras "$cam" \
      --prompts "${REPLY_PROMPTS[@]}" \
      --save_npz \
      --no_pkl \
      --postprocess \
      "${PP_FLAGS[@]}" \
      "${REPLY_PP_EXTRA[@]}" \
      "${CAM_FLAGS[@]}" \
      --skip_if_exists \
      --gpu_ids $GPU_IDS \
      --workers_per_gpu "$WORKERS_PER_GPU"
  done
done

echo "🎉 Batch segmentation job finished!"
