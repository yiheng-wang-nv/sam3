#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAM3_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Paths
SAM3_OUTPUT_DIR="/localhome/local-vennw/code/task7_03030306031203130315_nocover_merged/sam3_output"
DATASET_ROOT="/localhome/local-vennw/code/task7_03030306031203130315_nocover_merged"

# Set to true to overwrite existing masks in dataset_root/masks
OVERWRITE=true

ARGS=(--input_dir "$SAM3_OUTPUT_DIR" --copy_only --copy_to_dataset_root "$DATASET_ROOT")
if [ "$OVERWRITE" = true ]; then
  ARGS+=(--overwrite)
fi

python "${SAM3_DIR}/postprocess_masks.py" "${ARGS[@]}"
