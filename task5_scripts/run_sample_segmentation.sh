#!/bin/bash

# Randomly sample N videos per camera, run inference + postprocess, and generate
# side-by-side comparison videos to visually inspect the results.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Configuration
CHECKPOINT="${SAM3_DIR}/sam3.pt"
BASE_DIR="/localhome/local-vennw/code/task5-1_02030204021002130214_merged/videos/chunk-000"
BASE_OUTPUT_DIR="/localhome/local-vennw/code/task5-1_02030204021002130214_merged/sam3_output"

# Prompts (per camera)
HEAD_RIGHT_PROMPTS=("blue table" "robotic arm(s)")
RIGHT_ARM_PROMPTS=("blue table" "robotic arm(s)" "silver box" "tools")

# Camera names
HEAD_RIGHT_CAMERA="observation.images.head_right_camera_color_optical_frame"
RIGHT_ARM_CAMERA="observation.images.right_arm_camera_color_optical_frame"

# Sampling
SAMPLE_N=2
SAMPLE_SEED=42
GPU_IDS="0 1 2 3 4 5 6 7"

# Postprocess settings (shared)
PP_NUM_WORKERS=32
PP_MIN_HOLE_SIZE=64
PP_MIN_OBJECT_SIZE=50
PP_CLOSING_ITERATIONS=1
PP_NO_REMOVE_SMALL_OBJECTS=true
PP_UNION_HOLE_FILL=true
PP_UNION_GAP_FILL=true
PP_UNION_GAP_CLOSING_ITERATIONS=1

# Interior fill (per camera)
HEAD_RIGHT_FILL_CLASS="1,2"
HEAD_RIGHT_FILL_TARGET=3
RIGHT_ARM_FILL_CLASS="1,2,3,4"
RIGHT_ARM_FILL_TARGET=5

# Scanline fill (head right only)
PP_SCANLINE_FILL=true
PP_SCANLINE_SOURCE_LABEL=1
PP_SCANLINE_FILL_VALUE=3

# Topleft rect fill (right arm only, frame 10 to 2/3)
RIGHT_ARM_TOPLEFT_RECT=true
RIGHT_ARM_TOPLEFT_RECT_LABEL=1
RIGHT_ARM_TOPLEFT_RECT_FILL=5
RIGHT_ARM_TOPLEFT_RECT_Y_MAX=420
RIGHT_ARM_TOPLEFT_RECT_FRAME_START=10
RIGHT_ARM_TOPLEFT_RECT_FRAME_END_RATIO=0.667

# Topright rect fill (right arm only, 2/3 to end)
RIGHT_ARM_TOPRIGHT_RECT=true
RIGHT_ARM_TOPRIGHT_RECT_LABEL=2
RIGHT_ARM_TOPRIGHT_RECT_FILL=5
RIGHT_ARM_TOPRIGHT_RECT_Y_MAX=420
RIGHT_ARM_TOPRIGHT_RECT_Y_THRESHOLD=200
RIGHT_ARM_TOPRIGHT_RECT_FRAME_START_RATIO=0.667

echo "🔍 Sample Segmentation Test (task5) - ${SAMPLE_N} random videos per camera"
echo "--------------------------------------------------------------------"
echo "SAM3 Dir:  $SAM3_DIR"
echo "Base Dir:  $BASE_DIR"
echo "Output:    $BASE_OUTPUT_DIR"
echo "GPUs:      ${GPU_IDS}"
echo "  head_right prompts: ${HEAD_RIGHT_PROMPTS[*]}"
echo "  right_arm  prompts: ${RIGHT_ARM_PROMPTS[*]}"
echo "  head_right fill_interior: $HEAD_RIGHT_FILL_CLASS->$HEAD_RIGHT_FILL_TARGET"
echo "  right_arm  fill_interior: $RIGHT_ARM_FILL_CLASS->$RIGHT_ARM_FILL_TARGET"
echo "--------------------------------------------------------------------"

PP_COMMON_ARGS=(
  --save_npz
  --no_pkl
  --save_side_by_side
  --postprocess_for_vis
  --postprocess
  --skip_if_exists
  --debug_n "$SAMPLE_N"
  --debug_seed "$SAMPLE_SEED"
  --pp_num_workers "$PP_NUM_WORKERS"
  --pp_min_hole_size "$PP_MIN_HOLE_SIZE"
  --pp_min_object_size "$PP_MIN_OBJECT_SIZE"
  --pp_closing_iterations "$PP_CLOSING_ITERATIONS"
  $( [ "$PP_NO_REMOVE_SMALL_OBJECTS" = true ] && echo "--pp_no_remove_small_objects" )
  $( [ "$PP_UNION_HOLE_FILL" = true ] && echo "--pp_union_hole_fill" )
  $( [ "$PP_UNION_GAP_FILL" = true ] && echo "--pp_union_gap_fill" )
  $( [ "$PP_UNION_GAP_FILL" = true ] && echo "--pp_union_gap_closing_iterations $PP_UNION_GAP_CLOSING_ITERATIONS" )
  --pp_overwrite
  --gpu_ids $GPU_IDS
)

echo "-> Running head right camera (${SAMPLE_N} samples, with scanline fill)"
python "${SAM3_DIR}/batch_run_parallel.py" \
  --base_dir "$BASE_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$BASE_OUTPUT_DIR" \
  --cameras "$HEAD_RIGHT_CAMERA" \
  --prompts "${HEAD_RIGHT_PROMPTS[@]}" \
  $( [ "$PP_SCANLINE_FILL" = true ] && echo "--pp_scanline_fill" ) \
  $( [ "$PP_SCANLINE_FILL" = true ] && echo "--pp_scanline_source_label $PP_SCANLINE_SOURCE_LABEL" ) \
  $( [ "$PP_SCANLINE_FILL" = true ] && echo "--pp_scanline_fill_value $PP_SCANLINE_FILL_VALUE" ) \
  --pp_fill_interior_class "$HEAD_RIGHT_FILL_CLASS" \
  --pp_fill_interior_target "$HEAD_RIGHT_FILL_TARGET" \
  "${PP_COMMON_ARGS[@]}"

echo "-> Running right arm camera (${SAMPLE_N} samples)"
python "${SAM3_DIR}/batch_run_parallel.py" \
  --base_dir "$BASE_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$BASE_OUTPUT_DIR" \
  --cameras "$RIGHT_ARM_CAMERA" \
  --prompts "${RIGHT_ARM_PROMPTS[@]}" \
  --pp_fill_interior_class "$RIGHT_ARM_FILL_CLASS" \
  --pp_fill_interior_target "$RIGHT_ARM_FILL_TARGET" \
  $( [ "$RIGHT_ARM_TOPLEFT_RECT" = true ] && echo "--pp_topleft_rect" ) \
  $( [ "$RIGHT_ARM_TOPLEFT_RECT" = true ] && echo "--pp_topleft_rect_label $RIGHT_ARM_TOPLEFT_RECT_LABEL" ) \
  $( [ "$RIGHT_ARM_TOPLEFT_RECT" = true ] && echo "--pp_topleft_rect_fill $RIGHT_ARM_TOPLEFT_RECT_FILL" ) \
  $( [ "$RIGHT_ARM_TOPLEFT_RECT" = true ] && echo "--pp_topleft_rect_y_max $RIGHT_ARM_TOPLEFT_RECT_Y_MAX" ) \
  $( [ "$RIGHT_ARM_TOPLEFT_RECT" = true ] && echo "--pp_topleft_rect_frame_start $RIGHT_ARM_TOPLEFT_RECT_FRAME_START" ) \
  $( [ "$RIGHT_ARM_TOPLEFT_RECT" = true ] && echo "--pp_topleft_rect_frame_end_ratio $RIGHT_ARM_TOPLEFT_RECT_FRAME_END_RATIO" ) \
  $( [ "$RIGHT_ARM_TOPRIGHT_RECT" = true ] && echo "--pp_topright_rect" ) \
  $( [ "$RIGHT_ARM_TOPRIGHT_RECT" = true ] && echo "--pp_topright_rect_label $RIGHT_ARM_TOPRIGHT_RECT_LABEL" ) \
  $( [ "$RIGHT_ARM_TOPRIGHT_RECT" = true ] && echo "--pp_topright_rect_fill $RIGHT_ARM_TOPRIGHT_RECT_FILL" ) \
  $( [ "$RIGHT_ARM_TOPRIGHT_RECT" = true ] && echo "--pp_topright_rect_y_max $RIGHT_ARM_TOPRIGHT_RECT_Y_MAX" ) \
  $( [ "$RIGHT_ARM_TOPRIGHT_RECT" = true ] && echo "--pp_topright_rect_y_threshold $RIGHT_ARM_TOPRIGHT_RECT_Y_THRESHOLD" ) \
  $( [ "$RIGHT_ARM_TOPRIGHT_RECT" = true ] && echo "--pp_topright_rect_frame_start_ratio $RIGHT_ARM_TOPRIGHT_RECT_FRAME_START_RATIO" ) \
  "${PP_COMMON_ARGS[@]}"

echo "🎉 Sample segmentation test finished!"
echo "Compare videos saved to: $BASE_OUTPUT_DIR/*/episode_*_compare.mp4"
