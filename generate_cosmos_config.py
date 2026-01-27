#!/usr/bin/env python3
"""
生成 Cosmos-Transfer2.5 配置文件 (JSONL 格式)

用法:
    python generate_cosmos_config.py --video_dir <原始视频目录> --output <输出jsonl文件>
"""

import os
import glob
import json
import argparse
from pathlib import Path


# 默认 prompt 模板 - 根据你的实际场景修改
DEFAULT_PROMPT = """A robotic manipulation scene in an indoor environment. The robot arm is performing a task, interacting with objects on the table. The scene has realistic lighting and textures."""


def generate_config_for_camera(
    camera_name: str,
    mask_dir: str,
    video_dir: str = None,
    prompt: str = DEFAULT_PROMPT,
    output_file: str = None,
    guidance: int = 3,
    control_weight: float = 1.0,
):
    """
    为单个摄像头目录生成 Cosmos 配置
    
    Args:
        camera_name: 摄像头名称 (如 head_left_camera)
        mask_dir: mask npz 文件所在目录
        video_dir: 原始视频目录 (如果为 None，使用 mask_dir 中的 _vis.mp4)
        prompt: 文本提示
        output_file: 输出 JSONL 文件路径
        guidance: guidance scale
        control_weight: segmentation control weight
    """
    
    # 查找所有 mask 文件
    mask_files = sorted(glob.glob(os.path.join(mask_dir, '*_masks.npz')))
    
    if not mask_files:
        print(f"警告: 在 {mask_dir} 中没有找到 *_masks.npz 文件")
        return []
    
    configs = []
    
    for mask_path in mask_files:
        # 提取 episode 名称
        basename = os.path.basename(mask_path)
        episode_name = basename.replace('_masks.npz', '')  # e.g., episode_000000
        
        # 确定视频路径
        if video_dir:
            # 用户指定了视频目录
            video_path = os.path.join(video_dir, f"{episode_name}.mp4")
            if not os.path.exists(video_path):
                # 尝试其他命名格式
                video_path = os.path.join(video_dir, camera_name, f"{episode_name}.mp4")
        else:
            # 优先使用同目录下的原始视频 episode_XXXXXX.mp4
            video_path = os.path.join(os.path.dirname(mask_path), f"{episode_name}.mp4")
            if not os.path.exists(video_path):
                # Fallback: 使用 _vis.mp4
                video_path = mask_path.replace('_masks.npz', '_vis.mp4')
        
        # 生成配置
        config = {
            "name": f"{camera_name}_{episode_name}",
            "prompt": prompt,
            "video_path": video_path,
            "guided_generation_mask": mask_path,
            "guidance": guidance,
            "seg": {
                "control_weight": control_weight
            }
        }
        
        configs.append(config)
    
    return configs


def main():
    parser = argparse.ArgumentParser(description="生成 Cosmos-Transfer2.5 配置文件")
    parser.add_argument("--mask_base_dir", 
                        default="/localhome/local-vennw/code/sam3_dataset/task7_segmentation",
                        help="mask 文件的基础目录")
    parser.add_argument("--video_dir", 
                        default=None,
                        help="原始视频目录 (如果不指定，使用 _vis.mp4)")
    parser.add_argument("--output", "-o",
                        default="/localhome/local-vennw/code/sam3_dataset/cosmos_config.jsonl",
                        help="输出 JSONL 文件路径")
    parser.add_argument("--prompt", "-p",
                        default=DEFAULT_PROMPT,
                        help="文本提示")
    parser.add_argument("--prompt_file",
                        default=None,
                        help="从文件读取 prompt")
    parser.add_argument("--guidance", type=int, default=3,
                        help="Guidance scale")
    parser.add_argument("--control_weight", type=float, default=1.0,
                        help="Segmentation control weight")
    parser.add_argument("--cameras", nargs='+',
                        default=["head_left", "head_right", "left_arm", "right_arm"],
                        help="要处理的摄像头列表")
    
    args = parser.parse_args()
    
    # 读取 prompt
    if args.prompt_file and os.path.exists(args.prompt_file):
        with open(args.prompt_file, 'r') as f:
            prompt = f.read().strip()
    else:
        prompt = args.prompt
    
    # 摄像头目录映射
    camera_dirs = {
        "head_left": "observation.images.head_left_camera_color_optical_frame",
        "head_right": "observation.images.head_right_camera_color_optical_frame",
        "left_arm": "observation.images.left_arm_camera_color_optical_frame",
        "right_arm": "observation.images.right_arm_camera_color_optical_frame",
    }
    
    all_configs = []
    
    for camera in args.cameras:
        if camera not in camera_dirs:
            print(f"警告: 未知摄像头 {camera}")
            continue
            
        camera_dir = camera_dirs[camera]
        mask_dir = os.path.join(args.mask_base_dir, camera_dir)
        
        if not os.path.exists(mask_dir):
            print(f"警告: 目录不存在 {mask_dir}")
            continue
        
        print(f"处理摄像头: {camera}")
        configs = generate_config_for_camera(
            camera_name=camera,
            mask_dir=mask_dir,
            video_dir=args.video_dir,
            prompt=prompt,
            guidance=args.guidance,
            control_weight=args.control_weight,
        )
        all_configs.extend(configs)
        print(f"  生成 {len(configs)} 个配置")
    
    # 写入 JSONL 文件
    with open(args.output, 'w') as f:
        for config in all_configs:
            f.write(json.dumps(config, ensure_ascii=False) + '\n')
    
    print(f"\n总共生成 {len(all_configs)} 个配置")
    print(f"配置文件已保存到: {args.output}")
    
    # 显示示例配置
    if all_configs:
        print("\n示例配置 (第一条):")
        print(json.dumps(all_configs[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

