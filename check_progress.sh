#!/bin/bash
# Usage: bash check_progress.sh [dataset_dir]
#   dataset_dir: root dataset directory (default: sztask6 merged)
#   Expects: dataset_dir/videos/chunk-*/  and  dataset_dir/sam3_output/

DATASET_DIR="${1:-/localhome/local-vennw/code/task4-2_022602270305_merged}"
OUTPUT_DIR="${DATASET_DIR}/sam3_output"

echo "========================================"
echo "  Progress Report  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo "Dataset: $DATASET_DIR"
echo "Output:  $OUTPUT_DIR"
echo "----------------------------------------"

for chunk_dir in "$DATASET_DIR"/videos/chunk-*/; do
    [ -d "$chunk_dir" ] || continue
    chunk=$(basename "$chunk_dir")

    echo ""
    echo "  ── ${chunk} ──"
    printf "  %-55s %8s %8s %8s\n" "Camera" "Masks" "PostProc" "Total"
    printf "  %-55s %8s %8s %8s\n" "------" "-----" "--------" "-----"

    for cam_dir in "$chunk_dir"/*/; do
        [ -d "$cam_dir" ] || continue
        cam=$(basename "$cam_dir")
        total=$(ls "$cam_dir"/*.mp4 2>/dev/null | wc -l)

        out_dir="$OUTPUT_DIR/$cam"
        if [ ! -d "$out_dir" ]; then
            masks=0; post=0
        else
            # Count masks matching this chunk's episodes
            masks=0; post=0
            for vid in "$cam_dir"/*.mp4; do
                name=$(basename "$vid" .mp4)
                [ -f "$out_dir/${name}_masks.npz" ] && masks=$((masks + 1))
                [ -f "$out_dir/${name}_masks_post.npz" ] && post=$((post + 1))
            done
        fi

        printf "  %-55s %8s %8s %8s\n" "$cam" "$masks" "$post" "$total"
    done
done

echo ""
echo "----------------------------------------"
