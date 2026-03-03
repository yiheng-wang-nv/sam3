#!/bin/bash

# Run segmentation on a subset of episodes — either specific indices or random N.
#
# Usage:
#   # Specific episodes (comma-separated):
#   bash run_segmentation.sh --episodes 5,10,15,20
#
#   # Random N episodes:
#   bash run_segmentation.sh --random 5
#   bash run_segmentation.sh --random 5 --seed 123
#
#   # Override GPU list:
#   bash run_segmentation.sh --random 3 --gpus "0 1"

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# ──────────────── Configuration ────────────────
CHECKPOINT="${SAM3_DIR}/sam3.pt"
BASE_DIR="/localhome/local-vennw/code/task4-2_20260226_trimmed/videos/chunk-000"
BASE_OUTPUT_DIR="/localhome/local-vennw/code/task4-2_20260226_trimmed/sam3_output"

HEAD_RIGHT_CAMERA="observation.images.head_right_camera_color_optical_frame"
LEFT_ARM_CAMERA="observation.images.left_arm_camera_color_optical_frame"
RIGHT_ARM_CAMERA="observation.images.right_arm_camera_color_optical_frame"

# ── Per-camera configuration ──
# Camera list to process (comment/uncomment to toggle)
CAMERAS=(
    "$HEAD_RIGHT_CAMERA"
    "$LEFT_ARM_CAMERA"
    "$RIGHT_ARM_CAMERA"
)

GPU_IDS="1 2 3 5 6 7"

# Returns the prompt list for a given camera (via global REPLY_PROMPTS array)
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

# Returns the point-clicks JSON path for a given camera (empty = disabled)
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

# Returns extra per-camera postprocess flags (via global REPLY_PP_EXTRA array)
get_camera_pp_extra() {
  local cam="$1"
  REPLY_PP_EXTRA=()
  case "$cam" in
    "$RIGHT_ARM_CAMERA")
      REPLY_PP_EXTRA=(--pp_leftmost_rect --pp_leftmost_rect_fill 6)
      ;;
  esac
}

# Postprocess settings
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

# ──────────────── Parse CLI args ────────────────
MODE=""
EPISODES_CSV=""
NUM_RANDOM=0
SEED=42

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episodes)
      MODE="specific"
      EPISODES_CSV="$2"
      shift 2
      ;;
    --random)
      MODE="random"
      NUM_RANDOM="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --gpus)
      GPU_IDS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 --episodes 5,10,15  OR  $0 --random 5 [--seed 42] [--gpus '0 1']"
      exit 1
      ;;
  esac
done

if [ -z "$MODE" ]; then
  echo "Error: must specify --episodes <indices> or --random <N>"
  echo "Usage:"
  echo "  $0 --episodes 5,10,15,20"
  echo "  $0 --random 5 [--seed 42] [--gpus '0 1']"
  exit 1
fi

# ──────────────── Build common postprocess flags ────────────────
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

# Per-camera point-clicks and extra PP flags are resolved in the run loop below.

# ──────────────── Mode: random N ────────────────
if [ "$MODE" = "random" ]; then
  echo "🚀 Running segmentation on ${NUM_RANDOM} random episodes (seed=${SEED})"
  echo "--------------------------------------------------------------------"
  echo "Cameras: ${CAMERAS[*]}"
  echo "GPUs: ${GPU_IDS}"
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
      --postprocess \
      "${PP_FLAGS[@]}" \
      "${REPLY_PP_EXTRA[@]}" \
      --debug_n "$NUM_RANDOM" \
      --debug_seed "$SEED" \
      "${CAM_FLAGS[@]}" \
      --gpu_ids $GPU_IDS
  done

  echo "🎉 Done (random ${NUM_RANDOM} episodes)!"
  exit 0
fi

# ──────────────── Mode: specific episodes ────────────────
IFS=',' read -ra EP_INDICES <<< "$EPISODES_CSV"

echo "🚀 Running segmentation on ${#EP_INDICES[@]} specific episodes: ${EPISODES_CSV}"
echo "--------------------------------------------------------------------"
echo "Cameras: ${CAMERAS[*]}"
echo "GPUs: ${GPU_IDS}"
echo "--------------------------------------------------------------------"

# Build a temp directory with symlinks to the requested episode videos
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

# Count total linked videos
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
    --gpu_ids $GPU_IDS
done

echo "🎉 Done (episodes: ${EPISODES_CSV})!"
