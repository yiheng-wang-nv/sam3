#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Configuration
BASE_DIR="/localhome/local-vennw/code/task7_01220206022402280306_merged/videos/chunk-000"
MASK_DIR="/localhome/local-vennw/code/task7_01220206022402280306_merged/masks/chunk-000"

# All 4 cameras
CAMERAS=(
    "observation.images.head_left_camera_color_optical_frame"
    # "observation.images.head_right_camera_color_optical_frame"
    # "observation.images.left_arm_camera_color_optical_frame"
    # "observation.images.right_arm_camera_color_optical_frame"
)

OUTPUT_DIR="/localhome/local-vennw/code/task7_01220206022402280306_merged/comparison_videos"

BUCKET_SIZE=100
N_PARALLEL=10
SEED=42

mkdir -p "$OUTPUT_DIR"

echo "🎬 Producing comparison videos (original | mask)"
echo "----------------------------------------------------------------"
echo "Cameras: ${#CAMERAS[@]}"
echo "Mask dir: $MASK_DIR"
echo "Sampling: 1 per ${BUCKET_SIZE} episodes"
echo "Seed: $SEED"
echo "Output: $OUTPUT_DIR"
echo "----------------------------------------------------------------"

produce_bucket_comparisons() {
  local camera="$1"
  local seed="$2"

  local mask_cam_dir="${MASK_DIR}/${camera}"
  local video_dir="${BASE_DIR}/${camera}"

  # Collect all episodes that have masks
  local all_episodes=()
  for f in "${mask_cam_dir}"/*_masks.npz; do
    [ -f "$f" ] || continue
    local ep
    ep=$(basename "$f" | sed 's/_masks\.npz$//')
    # skip _masks_post files
    [[ "$ep" == *_post ]] && continue
    if [ -f "${video_dir}/${ep}.mp4" ]; then
      all_episodes+=("$ep")
    fi
  done

  if [ ${#all_episodes[@]} -eq 0 ]; then
    echo "No valid episodes found for ${camera}, skipping."
    return
  fi

  # Use Python to bucket by every BUCKET_SIZE and pick 1 random per bucket
  local selected
  selected=$(python3 -c "
import random, re
episodes = '''${all_episodes[*]}'''.split()
buckets = {}
for ep in episodes:
    m = re.search(r'(\d+)$', ep)
    if not m: continue
    num = int(m.group(1))
    b = num // ${BUCKET_SIZE}
    buckets.setdefault(b, []).append(ep)
random.seed(${seed})
for b in sorted(buckets):
    print(random.choice(buckets[b]))
")

  local n_selected
  n_selected=$(echo "$selected" | wc -l)

  local short_cam
  short_cam=$(echo "$camera" | sed 's/observation\.images\.\(.*\)_camera_color_optical_frame/\1/')

  echo "-> ${short_cam}: ${#all_episodes[@]} episodes, ${n_selected} buckets sampled"

  local pids=()
  for ep in $selected; do
    echo "   Launching ${short_cam}/${ep}..."
    python "${SAM3_DIR}/produce_comparison_video.py" \
      --mask_dir "${mask_cam_dir}" \
      --video_path "${video_dir}/${ep}.mp4" \
      --episode "${ep}" \
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
  produce_bucket_comparisons "${CAMERAS[$i]}" "$((SEED + i))"
done

echo "✅ Done! Comparison videos saved to $OUTPUT_DIR"
