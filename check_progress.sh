#!/bin/bash
# Usage: bash check_progress.sh [dataset_dir]
#   dataset_dir: root dataset directory (default: task7_03030306_nocover_merged)
#   Checks both dataset_dir/sam3_output/ and dataset_dir/masks/chunk-*/

DATASET_DIR="${1:-/localhome/local-vennw/code/task4-2_0226022703050309_merged}"
OUTPUT_DIR="${DATASET_DIR}/sam3_output"
MASKS_DIR="${DATASET_DIR}/masks"

echo "========================================"
echo "  Progress Report  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo "Dataset:    $DATASET_DIR"
echo "sam3_output: $OUTPUT_DIR"
echo "masks:       $MASKS_DIR"
echo "----------------------------------------"

for chunk_dir in "$DATASET_DIR"/videos/chunk-*/; do
    [ -d "$chunk_dir" ] || continue
    chunk=$(basename "$chunk_dir")

    echo ""
    echo "  ── ${chunk} ──"
    printf "  %-55s %8s %8s %8s %8s\n" "Camera" "sam3" "post" "masks/" "Total"
    printf "  %-55s %8s %8s %8s %8s\n" "------" "----" "----" "------" "-----"

    for cam_dir in "$chunk_dir"/*/; do
        [ -d "$cam_dir" ] || continue
        cam=$(basename "$cam_dir")
        total=$(ls "$cam_dir"/*.mp4 2>/dev/null | wc -l)

        # Count in sam3_output/
        out_dir="$OUTPUT_DIR/$cam"
        sam3_masks=0; post=0
        if [ -d "$out_dir" ]; then
            for vid in "$cam_dir"/*.mp4; do
                name=$(basename "$vid" .mp4)
                [ -f "$out_dir/${name}_masks.npz" ] && sam3_masks=$((sam3_masks + 1))
                [ -f "$out_dir/${name}_masks_post.npz" ] && post=$((post + 1))
            done
        fi

        # Count in masks/chunk-*/
        masks_cam_dir="$MASKS_DIR/$chunk/$cam"
        masks_count=0
        if [ -d "$masks_cam_dir" ]; then
            for vid in "$cam_dir"/*.mp4; do
                name=$(basename "$vid" .mp4)
                [ -f "$masks_cam_dir/${name}_masks.npz" ] && masks_count=$((masks_count + 1))
            done
        fi

        printf "  %-55s %8s %8s %8s %8s\n" "$cam" "$sam3_masks" "$post" "$masks_count" "$total"
    done
done

echo ""
echo "----------------------------------------"
