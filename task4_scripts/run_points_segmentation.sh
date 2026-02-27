#!/bin/bash

# Run SAM3 segmentation with text prompts and/or point clicks.
#
# Usage:
#   # Points only
#   bash run_points_segmentation.sh --episode 2 --camera right_arm \
#       --annotations "65:frame_65.json" "40:frame_40.json" --save-video
#
#   # Prompts only
#   bash run_points_segmentation.sh --episode 2 --camera right_arm \
#       --prompts "blue table" "robotic arm(s)" --save-video
#
#   # Prompts + points (prompts first, then points supplement/override)
  # bash run_points_segmentation.sh --episode 2 --camera right_arm \
  #     --prompts "blue table" "robotic arm(s)" "trash can" "tools" \
  #     --annotations "65:frame_65.json" "19:frame_19.json" --save-video


set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# ──────────────── Configuration ────────────────
CHECKPOINT="${SAM3_DIR}/sam3.pt"
BASE_DIR="/localhome/local-vennw/code/task4-1_020202050212_merged"
BASE_VIDEO_DIR="${BASE_DIR}/videos/chunk-000"
OUTPUT_DIR="${BASE_DIR}/sam3_output"
JSON_DIR="${BASE_DIR}"

# Postprocess settings
PP_FILL_CLASS="1,2,3,4,5"
PP_FILL_TARGET=6
PP_MIN_HOLE_SIZE=64
PP_MIN_OBJECT_SIZE=50
PP_CLOSING_ITERATIONS=1
PP_UNION_HOLE_FILL=true
PP_UNION_GAP_FILL=true
PP_UNION_GAP_CLOSING_ITERATIONS=1
PP_NO_REMOVE_SMALL_OBJECTS=true

# Camera short name → full name mapping
declare -A CAMERA_MAP=(
    [head_left]="observation.images.head_left_camera_color_optical_frame"
    [head_right]="observation.images.head_right_camera_color_optical_frame"
    [left_arm]="observation.images.left_arm_camera_color_optical_frame"
    [right_arm]="observation.images.right_arm_camera_color_optical_frame"
)

# ──────────────── Parse CLI args ────────────────
EPISODE=""
CAMERA_SHORT=""
LABELME_DIR=""
ANNOTATIONS=()
PROMPTS=()
SAVE_VIDEO=false
BINARY=false
GPU_ID=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episode)
      EPISODE="$2"
      shift 2
      ;;
    --camera)
      CAMERA_SHORT="$2"
      shift 2
      ;;
    --json-dir)
      JSON_DIR="$2"
      shift 2
      ;;
    --labelme-dir)
      LABELME_DIR="$2"
      shift 2
      ;;
    --annotations)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        ANNOTATIONS+=("$1")
        shift
      done
      ;;
    --prompts)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        PROMPTS+=("$1")
        shift
      done
      ;;
    --save-video)
      SAVE_VIDEO=true
      shift
      ;;
    --binary)
      BINARY=true
      shift
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [ -z "$EPISODE" ] || [ -z "$CAMERA_SHORT" ]; then
  echo "Usage: $0 --episode <N> --camera <name> [--annotations frame:path ...] [--labelme-dir dir] [--save-video] [--gpu N]"
  echo "Camera names: ${!CAMERA_MAP[*]}"
  exit 1
fi

if [ -z "$LABELME_DIR" ] && [ ${#ANNOTATIONS[@]} -eq 0 ] && [ ${#PROMPTS[@]} -eq 0 ]; then
  echo "Error: must provide --prompts and/or --annotations/--labelme-dir"
  exit 1
fi

CAMERA_FULL="${CAMERA_MAP[$CAMERA_SHORT]}"
if [ -z "$CAMERA_FULL" ]; then
  echo "Error: unknown camera '$CAMERA_SHORT'. Available: ${!CAMERA_MAP[*]}"
  exit 1
fi

EP_NAME=$(printf "episode_%06d" "$EPISODE")
VIDEO_PATH="${BASE_VIDEO_DIR}/${CAMERA_FULL}/${EP_NAME}.mp4"
OUTPUT_PATH="${OUTPUT_DIR}/${CAMERA_FULL}/${EP_NAME}_masks.npz"

if [ ! -f "$VIDEO_PATH" ]; then
  echo "Error: video not found: $VIDEO_PATH"
  exit 1
fi

echo "🚀 Running segmentation (prompts + points)"
echo "--------------------------------------------------------------------"
echo "Episode:    ${EP_NAME}"
echo "Camera:     ${CAMERA_SHORT} (${CAMERA_FULL})"
echo "Video:      ${VIDEO_PATH}"
echo "Output:     ${OUTPUT_PATH}"
echo "Prompts:    ${PROMPTS[*]:-(none)}"
echo "GPU:        ${GPU_ID}"
echo "--------------------------------------------------------------------"

PP_FLAGS=(
  --postprocess
  --pp_fill_interior_class "$PP_FILL_CLASS"
  --pp_fill_interior_target "$PP_FILL_TARGET"
  --pp_min_hole_size "$PP_MIN_HOLE_SIZE"
  --pp_min_object_size "$PP_MIN_OBJECT_SIZE"
  --pp_closing_iterations "$PP_CLOSING_ITERATIONS"
)
[ "$PP_UNION_HOLE_FILL" = true ] && PP_FLAGS+=(--pp_union_hole_fill)
[ "$PP_UNION_GAP_FILL" = true ] && PP_FLAGS+=(--pp_union_gap_fill --pp_union_gap_closing_iterations "$PP_UNION_GAP_CLOSING_ITERATIONS")
[ "$PP_NO_REMOVE_SMALL_OBJECTS" = true ] && PP_FLAGS+=(--pp_no_remove_small_objects)

CMD=(
  python "${SAM3_DIR}/run_points_segmentation.py"
  --video_path "$VIDEO_PATH"
  --checkpoint_path "$CHECKPOINT"
  --output_path "$OUTPUT_PATH"
  "${PP_FLAGS[@]}"
)

if [ ${#PROMPTS[@]} -gt 0 ]; then
  CMD+=(--prompts "${PROMPTS[@]}")
fi

if [ -n "$LABELME_DIR" ]; then
  CMD+=(--labelme_dir "$LABELME_DIR")
fi

if [ ${#ANNOTATIONS[@]} -gt 0 ]; then
  RESOLVED=()
  for entry in "${ANNOTATIONS[@]}"; do
    frame_idx="${entry%%:*}"
    json_file="${entry#*:}"
    if [[ "$json_file" != /* ]]; then
      json_file="${JSON_DIR}/${json_file}"
    fi
    RESOLVED+=("${frame_idx}:${json_file}")
  done
  CMD+=(--annotations "${RESOLVED[@]}")
fi

if [ "$SAVE_VIDEO" = true ]; then
  CMD+=(--save_video)
fi

if [ "$BINARY" = true ]; then
  CMD+=(--binary)
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" "${CMD[@]}"

echo "🎉 Done!"
