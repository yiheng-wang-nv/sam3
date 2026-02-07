#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Configuration
BASE_DIR="/localhome/local-vennw/code/task2_01210123012601280130_600merged/videos/chunk-000"
SAM3_OUTPUT="/localhome/local-vennw/code/task2_01210123012601280130_600merged/sam3_output"

HEAD_LEFT_CAMERA="observation.images.head_left_camera_color_optical_frame"
HEAD_RIGHT_CAMERA="observation.images.head_right_camera_color_optical_frame"

OUTPUT_DIR="/localhome/local-vennw/code/task2_01210123012601280130_600merged/comparison_videos"

NUM_SAMPLES=10
N_PARALLEL=10
SEED=42

mkdir -p "$OUTPUT_DIR"

echo "🎬 Producing comparison videos (original | raw mask | post mask)"
echo "----------------------------------------------------------------"
echo "Samples per camera: $NUM_SAMPLES"
echo "Seed: $SEED"
echo "Output: $OUTPUT_DIR"
echo "----------------------------------------------------------------"

produce_random_comparisons() {
  local camera="$1"
  local seed="$2"

  local mask_dir="${SAM3_OUTPUT}/${camera}"
  local video_dir="${BASE_DIR}/${camera}"

  # Find episodes that have both _masks.npz and _masks_post.npz
  local episodes=()
  for f in "${mask_dir}"/*_masks_post.npz; do
    [ -f "$f" ] || continue
    local ep
    ep=$(basename "$f" | sed 's/_masks_post\.npz$//')
    if [ -f "${mask_dir}/${ep}_masks.npz" ] && [ -f "${video_dir}/${ep}.mp4" ]; then
      episodes+=("$ep")
    fi
  done

  if [ ${#episodes[@]} -eq 0 ]; then
    echo "No valid episodes found for ${camera}, skipping."
    return
  fi

  echo "-> ${camera}: found ${#episodes[@]} episodes, sampling ${NUM_SAMPLES}"

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
    echo "   Launching ${ep}..."
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

produce_random_comparisons "$HEAD_LEFT_CAMERA" "$SEED"
produce_random_comparisons "$HEAD_RIGHT_CAMERA" "$((SEED + 1))"

echo "✅ Done! Comparison videos saved as *_comparison.mp4"
