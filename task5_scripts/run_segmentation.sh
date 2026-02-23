#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Configuration
CHECKPOINT="${SAM3_DIR}/sam3.pt"
BASE_DIR="/localhome/local-vennw/code/task5-1_02030204021002130214_merged/videos/chunk-000"
BASE_OUTPUT_DIR="/localhome/local-vennw/code/task5-1_02030204021002130214_merged/sam3_output"

# Prompts (same for all cameras)
PROMPTS=("blue table" "robotic arm(s)")

# Camera names
HEAD_RIGHT_CAMERA="observation.images.head_right_camera_color_optical_frame"
RIGHT_ARM_CAMERA="observation.images.right_arm_camera_color_optical_frame"

# Episode to test
EPISODE="episode_000000"

# Postprocess settings (shared)
PP_MIN_HOLE_SIZE=64
PP_MIN_OBJECT_SIZE=50
PP_CLOSING_ITERATIONS=1
PP_NO_REMOVE_SMALL_OBJECTS=true
PP_UNION_HOLE_FILL=true
PP_UNION_GAP_FILL=true
PP_UNION_GAP_CLOSING_ITERATIONS=1

# Interior fill (all cameras)
PP_FILL_CLASS="1,2"
PP_FILL_TARGET=3

# Scanline fill: fill background between first/last table pixels per row
PP_SCANLINE_FILL=true
PP_SCANLINE_SOURCE_LABEL=1
PP_SCANLINE_FILL_VALUE=3

echo "🚀 Single-episode test (task5)"
echo "--------------------------------------------------------------------"
echo "SAM3 Dir:  $SAM3_DIR"
echo "Base Dir:  $BASE_DIR"
echo "Output:    $BASE_OUTPUT_DIR"
echo "Episode:   $EPISODE"
echo "Postprocess: union_hole_fill, union_gap_fill(iter=$PP_UNION_GAP_CLOSING_ITERATIONS), scanline_fill($PP_SCANLINE_SOURCE_LABEL->$PP_SCANLINE_FILL_VALUE), fill_interior($PP_FILL_CLASS->$PP_FILL_TARGET)"
echo "--------------------------------------------------------------------"

PP_COMMON_ARGS=(
  --save_npz
  --no_pkl
  --save_side_by_side
  --postprocess_for_vis
  --pp_min_hole_size "$PP_MIN_HOLE_SIZE"
  --pp_min_object_size "$PP_MIN_OBJECT_SIZE"
  --pp_closing_iterations "$PP_CLOSING_ITERATIONS"
  $( [ "$PP_NO_REMOVE_SMALL_OBJECTS" = true ] && echo "--pp_no_remove_small_objects" )
  $( [ "$PP_UNION_HOLE_FILL" = true ] && echo "--pp_union_hole_fill" )
  $( [ "$PP_UNION_GAP_FILL" = true ] && echo "--pp_union_gap_fill" )
  $( [ "$PP_UNION_GAP_FILL" = true ] && echo "--pp_union_gap_closing_iterations $PP_UNION_GAP_CLOSING_ITERATIONS" )
  --pp_fill_interior_class "$PP_FILL_CLASS"
  --pp_fill_interior_target "$PP_FILL_TARGET"
)

SCANLINE_ARGS=(
  $( [ "$PP_SCANLINE_FILL" = true ] && echo "--pp_scanline_fill" )
  $( [ "$PP_SCANLINE_FILL" = true ] && echo "--pp_scanline_source_label $PP_SCANLINE_SOURCE_LABEL" )
  $( [ "$PP_SCANLINE_FILL" = true ] && echo "--pp_scanline_fill_value $PP_SCANLINE_FILL_VALUE" )
)

# --- Head right camera (with scanline fill) ---
HEAD_RIGHT_VIDEO="${BASE_DIR}/${HEAD_RIGHT_CAMERA}/${EPISODE}.mp4"
HEAD_RIGHT_OUTPUT="${BASE_OUTPUT_DIR}/${HEAD_RIGHT_CAMERA}"

echo "-> Running head right camera (with scanline fill)"
python "${SAM3_DIR}/produce_masks.py" \
  --video_path "$HEAD_RIGHT_VIDEO" \
  --checkpoint_path "$CHECKPOINT" \
  --output_dir "$HEAD_RIGHT_OUTPUT" \
  --prompts "${PROMPTS[@]}" \
  "${PP_COMMON_ARGS[@]}" \
  "${SCANLINE_ARGS[@]}"

# --- Right arm camera (no scanline fill) ---
RIGHT_ARM_VIDEO="${BASE_DIR}/${RIGHT_ARM_CAMERA}/${EPISODE}.mp4"
RIGHT_ARM_OUTPUT="${BASE_OUTPUT_DIR}/${RIGHT_ARM_CAMERA}"

echo "-> Running right arm camera"
python "${SAM3_DIR}/produce_masks.py" \
  --video_path "$RIGHT_ARM_VIDEO" \
  --checkpoint_path "$CHECKPOINT" \
  --output_dir "$RIGHT_ARM_OUTPUT" \
  --prompts "${PROMPTS[@]}" \
  "${PP_COMMON_ARGS[@]}"

echo "🎉 Single-episode test finished!"
