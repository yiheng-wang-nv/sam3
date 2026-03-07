#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Configuration
BASE_DIR="/localhome/local-vennw/code/task7_20260303_nocover/videos/chunk-000"
SAM3_OUTPUT="/localhome/local-vennw/code/task7_20260303_nocover/sam3_output"

# All 4 cameras
CAMERAS=(
    "observation.images.head_left_camera_color_optical_frame"
    "observation.images.head_right_camera_color_optical_frame"
    "observation.images.left_arm_camera_color_optical_frame"
    "observation.images.right_arm_camera_color_optical_frame"
)

OUTPUT_DIR="/localhome/local-vennw/code/task7_20260303_nocover/comparison_videos"

NUM_SAMPLES=20
N_PARALLEL=10
SEED=42
MIN_EPISODE=0  # Only select episodes with index > this value
MAX_EPISODE=199  # Only select episodes with index < this value

mkdir -p "$OUTPUT_DIR"

echo "🎬 Producing comparison videos (original | raw mask | post mask)"
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

  local mask_dir="${SAM3_OUTPUT}/${camera}"
  local video_dir="${BASE_DIR}/${camera}"

  # Find episodes that have both _masks.npz and _masks_post.npz, filtered by MIN_EPISODE
  local episodes=()
  for f in "${mask_dir}"/*_masks_post.npz; do
    [ -f "$f" ] || continue
    local ep
    ep=$(basename "$f" | sed 's/_masks_post\.npz$//')
    # Extract episode number (last numeric part of the episode name, e.g. episode_000123 -> 123)
    local ep_num
    ep_num=$(echo "$ep" | grep -oE '[0-9]+$')
    if [ -z "$ep_num" ]; then
      continue
    fi
    # Remove leading zeros for numeric comparison
    ep_num=$((10#$ep_num))
    if [ "$ep_num" -le "$MIN_EPISODE" ] || [ "$ep_num" -ge "$MAX_EPISODE" ]; then
      continue
    fi
    if [ -f "${mask_dir}/${ep}_masks.npz" ] && [ -f "${video_dir}/${ep}.mp4" ]; then
      episodes+=("$ep")
    fi
  done

  if [ ${#episodes[@]} -eq 0 ]; then
    echo "No valid episodes in (${MIN_EPISODE}, ${MAX_EPISODE}) found for ${camera}, skipping."
    return
  fi

  echo "-> ${camera}: found ${#episodes[@]} episodes in (${MIN_EPISODE}, ${MAX_EPISODE}), sampling ${NUM_SAMPLES}"

  # Randomly sample using Python
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
    python "${SAM3_DIR}/produce_comparison_video.py" \
      --mask_dir "${mask_dir}" \
      --video_path "${video_dir}/${ep}.mp4" \
      --episode "${ep}" \
      --output_path "${OUTPUT_DIR}/${short_cam}_${ep}_comparison.mp4" &
    pids+=($!)

    # Limit to N_PARALLEL concurrent jobs
    if [ ${#pids[@]} -ge "$N_PARALLEL" ]; then
      wait "${pids[0]}"
      pids=("${pids[@]:1}")
    fi
  done
  # Wait for remaining jobs
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

for i in "${!CAMERAS[@]}"; do
  produce_random_comparisons "${CAMERAS[$i]}" "$((SEED + i))"
done

echo "✅ Done! Comparison videos saved as *_comparison.mp4"
