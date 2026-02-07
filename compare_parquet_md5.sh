#!/bin/bash
# Compare MD5 checksums of overlapping parquet files between two datasets
# Usage: bash compare_parquet_md5.sh <old_dir> <new_dir>
#   e.g. bash compare_parquet_md5.sh \
#          /localhome/local-vennw/code/task7_20260122_trimmed/data/chunk-000 \
#          /localhome/local-vennw/code/task7_01220206_merged/data/chunk-000

OLD_DIR="${1:?Usage: $0 <old_dir> <new_dir>}"
NEW_DIR="${2:?Usage: $0 <old_dir> <new_dir>}"

if [ ! -d "$OLD_DIR" ]; then
    echo "Error: Old directory not found: $OLD_DIR"
    exit 1
fi
if [ ! -d "$NEW_DIR" ]; then
    echo "Error: New directory not found: $NEW_DIR"
    exit 1
fi

OLD_COUNT=$(ls "$OLD_DIR"/episode_*.parquet 2>/dev/null | wc -l)
NEW_COUNT=$(ls "$NEW_DIR"/episode_*.parquet 2>/dev/null | wc -l)

echo "Old dataset: $OLD_DIR  ($OLD_COUNT episodes)"
echo "New dataset: $NEW_DIR  ($NEW_COUNT episodes)"
echo ""
echo "Comparing MD5 of overlapping parquet files..."
echo ""

MATCH=0
DIFF=0
MISSING=0

for f in "$OLD_DIR"/episode_*.parquet; do
    fname=$(basename "$f")
    new_f="$NEW_DIR/$fname"
    if [ -f "$new_f" ]; then
        old_md5=$(md5sum "$f" | awk '{print $1}')
        new_md5=$(md5sum "$new_f" | awk '{print $1}')
        if [ "$old_md5" = "$new_md5" ]; then
            MATCH=$((MATCH + 1))
        else
            DIFF=$((DIFF + 1))
            echo "DIFF: $fname  old=$old_md5  new=$new_md5"
        fi
    else
        MISSING=$((MISSING + 1))
        echo "MISSING in new: $fname"
    fi
done

echo ""
echo "=== Summary ==="
echo "Old episodes:     $OLD_COUNT"
echo "New episodes:     $NEW_COUNT"
echo "Matching:         $MATCH"
echo "Different:        $DIFF"
echo "Missing in new:   $MISSING"

if [ "$DIFF" -eq 0 ] && [ "$MISSING" -eq 0 ]; then
    echo ""
    echo "✓ All overlapping files are identical. New dataset appends $((NEW_COUNT - OLD_COUNT)) new episodes."
fi
