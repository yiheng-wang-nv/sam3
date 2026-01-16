#!/bin/bash
cd /localhome/local-vennw/code
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dev

BASE_DIR="/localhome/local-vennw/code/galbot_lerobot_dataset/task7_20260106_merged_lerobot/sam3_output"

# 处理所有四个子目录
for subdir in \
    "observation.images.head_left_camera_color_optical_frame" \
    "observation.images.head_right_camera_color_optical_frame" \
    "observation.images.left_arm_camera_color_optical_frame" \
    "observation.images.right_arm_camera_color_optical_frame"
do
    echo "========================================"
    echo "处理目录: $subdir"
    echo "========================================"
    python convert_pkl_to_cosmos_mask.py "${BASE_DIR}/${subdir}/" --verify
    echo ""
done

echo "全部完成!"
