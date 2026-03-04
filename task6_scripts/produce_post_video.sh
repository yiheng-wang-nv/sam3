#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Configuration
DATASET_DIR="/localhome/local-vennw/code/sztask6_020902100211021202260227_merged"
CHUNKS=("chunk-000" "chunk-001")
SAM3_OUTPUT="${DATASET_DIR}/sam3_output"

HEAD_LEFT_CAMERA="observation.images.head_left_camera_color_optical_frame"
HEAD_RIGHT_CAMERA="observation.images.head_right_camera_color_optical_frame"
LEFT_ARM_CAMERA="observation.images.left_arm_camera_color_optical_frame"
RIGHT_ARM_CAMERA="observation.images.right_arm_camera_color_optical_frame"

OUTPUT_DIR="${DATASET_DIR}/comparison_videos"

NUM_SAMPLES=20
N_PARALLEL=10
SEED=42

mkdir -p "$OUTPUT_DIR"

echo "🎬 Producing comparison videos (original | raw mask | post mask)"
echo "----------------------------------------------------------------"
echo "Samples per camera: $NUM_SAMPLES"
echo "Seed: $SEED"
echo "Chunks: ${CHUNKS[*]}"
echo "Output: $OUTPUT_DIR"
echo "----------------------------------------------------------------"

produce_random_comparisons() {
  local camera="$1"
  local seed="$2"

  local mask_dir="${SAM3_OUTPUT}/${camera}"

  # Collect episodes across all chunks
  local episodes=()
  declare -A ep_video_map
  for chunk in "${CHUNKS[@]}"; do
    local video_dir="${DATASET_DIR}/videos/${chunk}/${camera}"
    [ -d "$video_dir" ] || continue
    for f in "${mask_dir}"/*_masks_post.npz; do
      [ -f "$f" ] || continue
      local ep
      ep=$(basename "$f" | sed 's/_masks_post\.npz$//')
      [ -n "${ep_video_map[$ep]+x}" ] && continue
      if [ -f "${mask_dir}/${ep}_masks.npz" ] && [ -f "${video_dir}/${ep}.mp4" ]; then
        episodes+=("$ep")
        ep_video_map[$ep]="${video_dir}/${ep}.mp4"
      fi
    done
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
    local video_path="${ep_video_map[$ep]}"
    echo "   Launching ${ep} ($(basename "$(dirname "$(dirname "$video_path")")"))"
    python "${SAM3_DIR}/produce_comparison_video.py" \
      --mask_dir "${mask_dir}" \
      --video_path "${video_path}" \
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
produce_random_comparisons "$LEFT_ARM_CAMERA" "$((SEED + 2))"
produce_random_comparisons "$RIGHT_ARM_CAMERA" "$((SEED + 3))"

echo "✅ Done! Comparison videos saved as *_comparison.mp4"
