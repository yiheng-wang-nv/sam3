#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Configuration
VIDEO_DIR="/localhome/local-vennw/code/task4-1_022602270304_merged/videos/chunk-000"
MASK_DIR="/localhome/local-vennw/code/task4-1_022602270304_merged/masks/chunk-000"
OUTPUT_DIR="/localhome/local-vennw/code/task4-1_022602270304_merged/mask_comparison_videos"

CAMERAS=(
    "observation.images.head_right_camera_color_optical_frame"
    "observation.images.right_arm_camera_color_optical_frame"
    "observation.images.left_arm_camera_color_optical_frame"
)

NUM_SAMPLES=2
N_PARALLEL=10
SEED=42
MIN_EPISODE=1
MAX_EPISODE=199

mkdir -p "$OUTPUT_DIR"

echo "🎬 Producing comparison videos (original | mask overlay)"
echo "----------------------------------------------------------------"
echo "Cameras: ${#CAMERAS[@]}"
echo "Samples per camera: $NUM_SAMPLES"
echo "Episode range: ($MIN_EPISODE, $MAX_EPISODE)"
echo "Seed: $SEED"
echo "Output: $OUTPUT_DIR"
echo "----------------------------------------------------------------"

produce_random_comparisons() {
  local camera="$1"
  local seed="$2"

  local cam_mask_dir="${MASK_DIR}/${camera}"
  local cam_video_dir="${VIDEO_DIR}/${camera}"

  local episodes=()
  for f in "${cam_mask_dir}"/*_masks.npz; do
    [ -f "$f" ] || continue
    local ep
    ep=$(basename "$f" | sed 's/_masks\.npz$//')
    local ep_num
    ep_num=$(echo "$ep" | grep -oE '[0-9]+$')
    if [ -z "$ep_num" ]; then
      continue
    fi
    ep_num=$((10#$ep_num))
    if [ "$ep_num" -le "$MIN_EPISODE" ] || [ "$ep_num" -ge "$MAX_EPISODE" ]; then
      continue
    fi
    if [ -f "${cam_video_dir}/${ep}.mp4" ]; then
      episodes+=("$ep")
    fi
  done

  if [ ${#episodes[@]} -eq 0 ]; then
    echo "No valid episodes in (${MIN_EPISODE}, ${MAX_EPISODE}) found for ${camera}, skipping."
    return
  fi

  echo "-> ${camera}: found ${#episodes[@]} episodes in (${MIN_EPISODE}, ${MAX_EPISODE}), sampling ${NUM_SAMPLES}"

  local selected
  selected=$(python3 -c "
import random
episodes = '''${episodes[*]}'''.split()
random.seed(${seed})
selected = random.sample(episodes, min(${NUM_SAMPLES}, len(episodes)))
for e in selected:
    print(e)
")

  local short_cam
  short_cam=$(echo "$camera" | sed 's/observation\.images\.\(.*\)_camera_color_optical_frame/\1/')

  local pids=()
  for ep in $selected; do
    echo "   Launching ${short_cam}/${ep}..."
    python "${SAM3_DIR}/produce_mask_comparison_video.py" \
      --mask_path "${cam_mask_dir}/${ep}_masks.npz" \
      --video_path "${cam_video_dir}/${ep}.mp4" \
      --output_path "${OUTPUT_DIR}/${short_cam}_${ep}_comparison.mp4" &
    pids+=($!)

    if [ ${#pids[@]} -ge "$N_PARALLEL" ]; then
      wait "${pids[0]}"
      pids=("${pids[@]:1}")
    fi
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

for i in "${!CAMERAS[@]}"; do
  produce_random_comparisons "${CAMERAS[$i]}" "$((SEED + i))"
done

echo "✅ Done! Comparison videos saved to ${OUTPUT_DIR}"
