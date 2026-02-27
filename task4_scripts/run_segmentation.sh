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
BASE_DIR="/localhome/local-vennw/code/task4-1_020202050212_merged/videos/chunk-000"
BASE_OUTPUT_DIR="/localhome/local-vennw/code/task4-1_020202050212_merged/sam3_output"

PROMPTS=("blue table" "robotic arm(s)" "tools" "trash can")

HEAD_RIGHT_CAMERA="observation.images.head_right_camera_color_optical_frame"
RIGHT_ARM_CAMERA="observation.images.right_arm_camera_color_optical_frame"

CAMERAS=(
    "$HEAD_RIGHT_CAMERA"
    "$RIGHT_ARM_CAMERA"
)

GPU_IDS="0 1 2 3"

# Per-episode point clicks JSON (set to "" to disable)
POINT_CLICKS_JSON="/localhome/local-vennw/code/task4-1_020202050212_merged/point_clicks.json"

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
    --point-clicks)
      POINT_CLICKS_JSON="$2"
      shift 2
      ;;
    --no-point-clicks)
      POINT_CLICKS_JSON=""
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 --episodes 5,10,15  OR  $0 --random 5 [--seed 42] [--gpus '0 1'] [--point-clicks path.json] [--no-point-clicks]"
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

POINT_CLICKS_FLAG=()
if [ -n "$POINT_CLICKS_JSON" ] && [ -f "$POINT_CLICKS_JSON" ]; then
  POINT_CLICKS_FLAG=(--point_clicks_json "$POINT_CLICKS_JSON")
  echo "Point clicks: $POINT_CLICKS_JSON"
fi

# ──────────────── Mode: random N ────────────────
if [ "$MODE" = "random" ]; then
  echo "🚀 Running segmentation on ${NUM_RANDOM} random episodes (seed=${SEED})"
  echo "--------------------------------------------------------------------"
  echo "GPUs: ${GPU_IDS}"
  echo "--------------------------------------------------------------------"

  python "${SAM3_DIR}/batch_run_parallel.py" \
    --base_dir "$BASE_DIR" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$BASE_OUTPUT_DIR" \
    --cameras "${CAMERAS[@]}" \
    --prompts "${PROMPTS[@]}" \
    --save_npz \
    --no_pkl \
    --postprocess \
    "${PP_FLAGS[@]}" \
    --debug_n "$NUM_RANDOM" \
    --debug_seed "$SEED" \
    "${POINT_CLICKS_FLAG[@]}" \
    --gpu_ids $GPU_IDS

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

python "${SAM3_DIR}/batch_run_parallel.py" \
  --base_dir "$TMPDIR" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$BASE_OUTPUT_DIR" \
  --cameras "${CAMERAS[@]}" \
  --prompts "${PROMPTS[@]}" \
  --save_npz \
  --no_pkl \
  --postprocess \
  "${PP_FLAGS[@]}" \
  "${POINT_CLICKS_FLAG[@]}" \
  --gpu_ids $GPU_IDS

echo "🎉 Done (episodes: ${EPISODES_CSV})!"
