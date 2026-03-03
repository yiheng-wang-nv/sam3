#!/bin/bash

# Run segmentation on task6 — debug (one random), specific episodes, or random N.
#
# Usage:
#   # Debug mode (1 random video per camera, with side-by-side visualization):
#   bash run_debug_segmentation.sh
#   bash run_debug_segmentation.sh --debug
#
#   # Specific episodes (comma-separated):
#   bash run_debug_segmentation.sh --episodes 50,500,900
#
#   # Random N episodes:
#   bash run_debug_segmentation.sh --random 5
#   bash run_debug_segmentation.sh --random 5 --seed 123
#
#   # Override GPU list:
#   bash run_debug_segmentation.sh --random 3 --gpus "0 1"

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# ──────────────── Configuration ────────────────
CHECKPOINT="${SAM3_DIR}/sam3.pt"
BASE_DIR="/localhome/local-vennw/code/sztask6_020902100211021202260227_merged/videos/chunk-000"
BASE_OUTPUT_DIR="/localhome/local-vennw/code/sztask6_020902100211021202260227_merged/sam3_output"

HEAD_LEFT_CAMERA="observation.images.head_left_camera_color_optical_frame"
HEAD_RIGHT_CAMERA="observation.images.head_right_camera_color_optical_frame"
LEFT_ARM_CAMERA="observation.images.left_arm_camera_color_optical_frame"
RIGHT_ARM_CAMERA="observation.images.right_arm_camera_color_optical_frame"

CAMERAS=(
    "$HEAD_LEFT_CAMERA"
    "$HEAD_RIGHT_CAMERA"
    # "$LEFT_ARM_CAMERA"
    # "$RIGHT_ARM_CAMERA"
)

GPU_IDS="1 2 3 5 6 7"
WORKERS_PER_GPU=4

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

# ──────────────── Parse CLI args ────────────────
MODE="debug"
EPISODES_CSV=""
NUM_RANDOM=0
SEED=42
DEBUG_SEED_BASE=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug)     MODE="debug"; shift ;;
    --episodes)  MODE="specific"; EPISODES_CSV="$2"; shift 2 ;;
    --random)    MODE="random"; NUM_RANDOM="$2"; shift 2 ;;
    --seed)      SEED="$2"; shift 2 ;;
    --gpus)      GPU_IDS="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--debug] | --episodes 5,10,15 | --random 5 [--seed 42] [--gpus '0 1']"
      exit 1
      ;;
  esac
done

# ──────────────── Build common PP flags ────────────────
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

# ──────────────── Debug-mode visualization helpers ────────────────
get_latest_episode() {
  local camera_dir="$1"
  local latest
  latest=$(ls -t "${camera_dir}"/*_compare.mp4 2>/dev/null | head -n 1)
  if [ -z "$latest" ]; then echo ""; return; fi
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
  case "$camera" in
    "$HEAD_LEFT_CAMERA"|"$HEAD_RIGHT_CAMERA")
      EXTRA_ARGS+=(--temporal_fill --temporal_fill_labels "5" --temporal_fill_union_labels "1,3,4" --temporal_fill_value 5)
      ;;
    "$LEFT_ARM_CAMERA")
      EXTRA_ARGS+=(--corner_rect --corner_rect_mode topleft)
      EXTRA_ARGS+=(--corner_rect_labels "1,3,4" --corner_rect_fill 5 --corner_rect_y_max 420)
      EXTRA_ARGS+=(--corner_rect_x_first_labels "3,4")
      ;;
    "$RIGHT_ARM_CAMERA")
      EXTRA_ARGS+=(--corner_rect --corner_rect_mode topright)
      EXTRA_ARGS+=(--corner_rect_labels "1,3,4" --corner_rect_fill 5 --corner_rect_y_max 420)
      EXTRA_ARGS+=(--corner_rect_x_first_labels "3,4")
      ;;
  esac

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
    $( [ "$PP_UNION_GAP_FILL" = true ] && echo "--union_gap_closing_iterations $PP_UNION_GAP_CLOSING_ITERATIONS" ) \
    --fill_interior_class "$fill_class" \
    --fill_interior_target "$fill_target" \
    "${EXTRA_ARGS[@]}"

  python "${SAM3_DIR}/combine_compare_and_post.py" \
    --compare_video "${camera_dir}/${episode}_compare.mp4" \
    --post_video "${camera_dir}/${episode}_compare_post.mp4" \
    --output_path "${camera_dir}/${episode}_compare_merged.mp4"
}

# ──────────────── Header ────────────────
echo "🧪 Starting Segmentation (task6, mode=${MODE})"
echo "------------------------------------------------"
echo "SAM3 Dir:  $SAM3_DIR"
echo "Base Dir:  $BASE_DIR"
echo "Output:    $BASE_OUTPUT_DIR"
echo "Cameras:   ${CAMERAS[*]}"
echo "GPU:       ${GPU_IDS}"
echo "------------------------------------------------"

# ──────────────── Mode: debug (1 random per camera + visualization) ────────────────
if [ "$MODE" = "debug" ]; then
  CAM_IDX=0
  for cam in "${CAMERAS[@]}"; do
    CAM_IDX=$((CAM_IDX + 1))
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

    echo "-> Debug ${cam} (one random video, seed=$((DEBUG_SEED_BASE + CAM_IDX)))"
    echo "   Prompts: ${REPLY_PROMPTS[*]}"

    python "${SAM3_DIR}/batch_run_parallel.py" \
      --base_dir "$BASE_DIR" \
      --checkpoint "$CHECKPOINT" \
      --output_dir "$BASE_OUTPUT_DIR" \
      --cameras "$cam" \
      --prompts "${REPLY_PROMPTS[@]}" \
      --save_npz \
      --no_pkl \
      --save_side_by_side \
      --postprocess \
      "${PP_FLAGS[@]}" \
      "${REPLY_PP_EXTRA[@]}" \
      "${CAM_FLAGS[@]}" \
      --skip_if_exists \
      --debug_one \
      --debug_seed $((DEBUG_SEED_BASE + CAM_IDX)) \
      --gpu_ids $GPU_IDS

    postprocess_and_merge "$cam" "$REPLY_FILL_CLASS" "$REPLY_FILL_TARGET"
  done

  echo "✅ Debug segmentation finished!"
  exit 0
fi

# ──────────────── Mode: random N ────────────────
if [ "$MODE" = "random" ]; then
  echo "🚀 Running segmentation on ${NUM_RANDOM} random episodes (seed=${SEED})"

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

    echo "  [$cam] Prompts: ${REPLY_PROMPTS[*]}"

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
      --debug_n "$NUM_RANDOM" \
      --debug_seed "$SEED" \
      --gpu_ids $GPU_IDS \
      --workers_per_gpu "$WORKERS_PER_GPU"
  done

  echo "🎉 Done (random ${NUM_RANDOM} episodes)!"
  exit 0
fi

# ──────────────── Mode: specific episodes ────────────────
IFS=',' read -ra EP_INDICES <<< "$EPISODES_CSV"

echo "🚀 Running segmentation on ${#EP_INDICES[@]} specific episodes: ${EPISODES_CSV}"

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

for cam in "${CAMERAS[@]}"; do
  mkdir -p "${TMPDIR}/${cam}"
  for idx in "${EP_INDICES[@]}"; do
    idx=$(echo "$idx" | tr -d ' ')
    ep_name=$(printf "episode_%06d" "$idx")
    src="${BASE_DIR}/${cam}/${ep_name}.mp4"
    if [ ! -f "$src" ]; then
      echo "Warning: ${cam}/${ep_name}.mp4 not found, skipping"
      continue
    fi
    ln -s "$src" "${TMPDIR}/${cam}/${ep_name}.mp4"
  done
done

TOTAL=$(find "$TMPDIR" -name '*.mp4' | wc -l)
if [ "$TOTAL" -eq 0 ]; then
  echo "Error: no matching episode videos found."
  exit 1
fi
echo "Found ${TOTAL} video files to process."

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

  echo "  [$cam] Prompts: ${REPLY_PROMPTS[*]}"

  python "${SAM3_DIR}/batch_run_parallel.py" \
    --base_dir "$TMPDIR" \
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
    --gpu_ids $GPU_IDS \
    --workers_per_gpu "$WORKERS_PER_GPU"
done

echo "🎉 Done (episodes: ${EPISODES_CSV})!"
