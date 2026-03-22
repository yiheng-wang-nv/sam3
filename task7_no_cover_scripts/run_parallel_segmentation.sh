#!/bin/bash

# Run full parallel segmentation on task7_nocover (all episodes, all cameras).

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# ──────────────── Configuration ────────────────
CHECKPOINT="${SAM3_DIR}/sam3.pt"
DATASET_DIR="/localhome/local-vennw/code/task7_03030306031203130315_nocover_merged"
CHUNKS=("chunk-000")

HEAD_LEFT_CAMERA="observation.images.head_left_camera_color_optical_frame"
HEAD_RIGHT_CAMERA="observation.images.head_right_camera_color_optical_frame"
LEFT_ARM_CAMERA="observation.images.left_arm_camera_color_optical_frame"
RIGHT_ARM_CAMERA="observation.images.right_arm_camera_color_optical_frame"

CAMERAS=(
    "$HEAD_LEFT_CAMERA"
    "$HEAD_RIGHT_CAMERA"
    "$LEFT_ARM_CAMERA"
    "$RIGHT_ARM_CAMERA"
)

PROMPTS=("blue table" "robotic arm(s)" "silver box" "metal items")

GPU_IDS="1 2 3 4 5 6 7"
WORKERS_PER_GPU=2
PP_OVERWRITE=false

# ── Postprocess settings (shared) ──
PP_NUM_WORKERS=96
PP_MIN_HOLE_SIZE=64
PP_MIN_OBJECT_SIZE=50
PP_CLOSING_ITERATIONS=1
PP_NO_REMOVE_SMALL_OBJECTS=true
PP_UNION_HOLE_FILL=true
PP_UNION_GAP_FILL=true
PP_UNION_GAP_CLOSING_ITERATIONS=2
PP_FILL_CLASS="1,2,3"
PP_FILL_TARGET=4

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
echo "🚀 Starting Parallel Segmentation Job (task7_nocover, 4 cameras)"
echo "--------------------------------------------------------------"
echo "SAM3 Dir:    $SAM3_DIR"
echo "Dataset:     $DATASET_DIR"
echo "Chunks:      ${CHUNKS[*]}"
echo "Cameras:     ${CAMERAS[*]}"
echo "GPUs:        ${GPU_IDS}"
echo "Workers/GPU: ${WORKERS_PER_GPU}"
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
    build_pp_flags "$PP_FILL_CLASS" "$PP_FILL_TARGET"

    echo "-> [${chunk}] Running ${cam}"
    echo "   Prompts: ${PROMPTS[*]}"

    python "${SAM3_DIR}/batch_run_parallel.py" \
      --base_dir "$BASE_DIR" \
      --checkpoint "$CHECKPOINT" \
      --output_dir "$BASE_OUTPUT_DIR" \
      --cameras "$cam" \
      --prompts "${PROMPTS[@]}" \
      --save_npz \
      --no_pkl \
      --postprocess \
      "${PP_FLAGS[@]}" \
      --skip_if_exists \
      --gpu_ids $GPU_IDS \
      --workers_per_gpu "$WORKERS_PER_GPU"
  done
done

echo "🎉 Batch segmentation job finished!"
