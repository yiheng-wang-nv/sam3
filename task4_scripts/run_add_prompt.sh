#!/bin/bash
#
# Add a missing prompt to existing SAM3 segmentation results and re-run postprocessing.
#
# Workflow:
#   Step 1: Run SAM3 for ONLY the new prompt -> save to a temp output dir
#   Step 2: Merge new masks (label 1 -> label 5) into existing _masks.npz
#   Step 3: Delete old _masks_post.npz, re-run postprocessing with updated fill settings
#
# NOTE: Already-processed videos (existing new prompt output) are skipped in Step 1.
#       Files where label 5 already exists are skipped in Step 2.
#

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# ============================================================
# Configuration - EDIT THESE
# ============================================================

CHECKPOINT="${SAM3_DIR}/sam3.pt"
BASE_DIR="/localhome/local-vennw/code/task4-1_02020205_merged/videos/chunk-000"
BASE_OUTPUT_DIR="/localhome/local-vennw/code/task4-1_02020205_merged/sam3_output"

# >>> FILL IN the missing prompt here <<<
NEW_PROMPT="silver box"

# Label value for the new prompt (5th prompt -> label 5)
NEW_LABEL=5

# Camera names
HEAD_RIGHT_CAMERA="observation.images.head_right_camera_color_optical_frame"
RIGHT_ARM_CAMERA="observation.images.right_arm_camera_color_optical_frame"

# GPU Selection
GPU_IDS="0 1 2"

# Updated postprocess settings
PP_NUM_WORKERS=32
PP_MIN_HOLE_SIZE=64
PP_MIN_OBJECT_SIZE=50
PP_CLOSING_ITERATIONS=1
PP_NO_REMOVE_SMALL_OBJECTS=true
PP_UNION_HOLE_FILL=true
PP_UNION_GAP_FILL=true
PP_UNION_GAP_CLOSING_ITERATIONS=1

# Updated interior fill (now includes the new label 5)
PP_FILL_CLASS="1,2,3,4,5"
PP_FILL_TARGET=6

CAMERAS=(
    "$HEAD_RIGHT_CAMERA"
    "$RIGHT_ARM_CAMERA"
)

# Temp output dir for the new prompt's segmentation
NEW_PROMPT_OUTPUT_DIR="${BASE_OUTPUT_DIR}_add_prompt_tmp"

# Merge workers
MERGE_WORKERS=32

echo "======================================================================"
echo "  Add Prompt to Existing SAM3 Masks"
echo "======================================================================"
echo "SAM3 Dir:       $SAM3_DIR"
echo "Base Dir:       $BASE_DIR"
echo "Existing Output:$BASE_OUTPUT_DIR"
echo "Temp Output:    $NEW_PROMPT_OUTPUT_DIR"
echo "New Prompt:     $NEW_PROMPT"
echo "New Label:      $NEW_LABEL"
echo "GPUs:           ${GPU_IDS}"
echo "PP Fill Class:  $PP_FILL_CLASS -> $PP_FILL_TARGET"
echo "======================================================================"

# Safety check
if [ "$NEW_PROMPT" = "FILL_IN_YOUR_MISSING_PROMPT" ]; then
    echo "ERROR: Please edit this script and set NEW_PROMPT to the missing prompt text!"
    exit 1
fi

# ============================================================
# Step 1: Run SAM3 for ONLY the new prompt
# ============================================================
echo ""
echo ">>> Step 1/3: Running SAM3 segmentation for new prompt: '$NEW_PROMPT'"
echo "--------------------------------------------------------------------"

python "${SAM3_DIR}/batch_run_parallel.py" \
  --base_dir "$BASE_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$NEW_PROMPT_OUTPUT_DIR" \
  --cameras "${CAMERAS[@]}" \
  --prompts "$NEW_PROMPT" \
  --save_npz \
  --no_pkl \
  --skip_if_exists \
  --gpu_ids $GPU_IDS

echo ">>> Step 1 done: New prompt segmentation complete."

# ============================================================
# Step 2: Merge new masks into existing masks
# ============================================================
echo ""
echo ">>> Step 2/3: Merging new prompt masks (label 1 -> label $NEW_LABEL) into existing masks"
echo "--------------------------------------------------------------------"

python "${SAM3_DIR}/merge_prompt_masks.py" \
  --existing_dir "$BASE_OUTPUT_DIR" \
  --new_dir "$NEW_PROMPT_OUTPUT_DIR" \
  --new_label "$NEW_LABEL" \
  --cameras "${CAMERAS[@]}" \
  --num_workers "$MERGE_WORKERS"

echo ">>> Step 2 done: Masks merged."

echo ""
echo "======================================================================"
echo "  All done! New prompt '$NEW_PROMPT' added as label $NEW_LABEL."
echo "  NOTE: Postprocessing was NOT run. Please run it separately via"
echo "        run_parallel_segmentation.sh with updated PP_FILL_CLASS/TARGET."
echo "======================================================================"
