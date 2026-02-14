#!/bin/bash

# SAM3 checkpoint path
CHECKPOINT="/localhome/local-vennw/code/3rd_sam3/sam3.pt"

# base directory
BASE_DIR="/localhome/local-vennw/code/orca-dev-galbot-1/galbot_lerobot_dataset/task7_20260106_no_rnd_lerobot/videos/chunk-000"

# define the 4 camera views to process
CAMERAS=(
    "observation.images.head_left_camera_color_optical_frame"
    "observation.images.head_right_camera_color_optical_frame"
    "observation.images.left_arm_camera_color_optical_frame"
    "observation.images.right_arm_camera_color_optical_frame"
)

# define prompts
PROMPTS=("blue table" "robotic arm(s)" "silver box")

# output directory
BASE_OUTPUT_DIR="output/task7_segmentation"

# flag to control video generation (true/false)
GENERATE_VIDEO=true

# process each video
for camera in "${CAMERAS[@]}"; do
    VIDEO_PATH="${BASE_DIR}/${camera}/episode_000000.mp4"
    CAMERA_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${camera}"
    
    echo "========================================"
    echo "Processing: $camera"
    echo "Video path: $VIDEO_PATH"
    echo "Output dir: $CAMERA_OUTPUT_DIR"
    echo "========================================"

    # 1. Run inference to generate PKL only
    # Note: removed --save_video flag from here since we handle it separately
    python produce_masks.py \
        --video_path "$VIDEO_PATH" \
        --checkpoint_path "$CHECKPOINT" \
        --output_dir "$CAMERA_OUTPUT_DIR" \
        --prompts "${PROMPTS[@]}" \
        --fps 30

    if [ $? -eq 0 ]; then
        echo "✅ Segmentation successful: $camera"
        
        # 2. Optionally generate video using the separate script
        if [ "$GENERATE_VIDEO" = true ]; then
            echo "🎬 Generating video visualization..."
            
            # Construct paths
            # Note: produce_masks.py names the pkl as {video_name}_segmentation_results.pkl
            # Here video_name is 'episode_000000'
            PKL_PATH="${CAMERA_OUTPUT_DIR}/episode_000000_segmentation_results.pkl"
            OUTPUT_VIDEO_PATH="${CAMERA_OUTPUT_DIR}/episode_000000_vis.mp4"
            
            python render_video.py \
                --pkl_path "$PKL_PATH" \
                --video_path "$VIDEO_PATH" \
                --output_path "$OUTPUT_VIDEO_PATH" \
                --fps 30
                
            if [ $? -eq 0 ]; then
                echo "✅ Video generation successful"
            else
                echo "❌ Video generation failed"
            fi
        else
            echo "⏭️  Skipping video generation"
        fi
        
    else
        echo "❌ Segmentation failed: $camera"
    fi
    echo ""
done

echo "🎉 All videos processed! Results saved in: $BASE_OUTPUT_DIR"
