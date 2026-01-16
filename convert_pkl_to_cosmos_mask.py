#!/usr/bin/env python3
"""
将 SAM3 分割结果 (pkl) 转换为 Cosmos-Transfer2.5 需要的 mask 格式 (npz)

输入: episode_XXXXXX_segmentation_results.pkl
输出: episode_XXXXXX_masks.npz

Cosmos 期望的格式:
- npz 文件，包含键 "arr_0"
- 数据维度: (T, H, W) 或 (N, T, H, W)
- 数据值: 整数标签 (0=背景, >0=前景)
"""

import pickle
import numpy as np
import os
import glob
from pathlib import Path
import argparse


def convert_single_pkl_to_npz(pkl_path: str, output_path: str = None, merge_objects: bool = True):
    """
    转换单个 pkl 文件为 Cosmos 兼容的 npz 格式
    
    Args:
        pkl_path: 输入的 pkl 文件路径
        output_path: 输出的 npz 文件路径，默认为同目录下同名 .npz 文件
        merge_objects: 是否合并所有对象的 mask，True 则输出 (T, H, W)，False 则输出 (N, T, H, W)
    
    Returns:
        输出文件路径
    """
    print(f"正在处理: {pkl_path}")
    
    # 读取 pkl 文件
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # 获取帧数和分辨率
    num_frames = len(data)
    first_frame = data[0]
    num_objects, h, w = first_frame['out_binary_masks'].shape
    
    print(f"  帧数: {num_frames}")
    print(f"  对象数: {num_objects}")
    print(f"  分辨率: {h}x{w}")
    
    if merge_objects:
        # 方案1: 合并所有对象的 mask -> (T, H, W)
        # 每个对象用不同的标签值 (1, 2, 3, ...)
        all_masks = np.zeros((num_frames, h, w), dtype=np.uint8)
        
        for frame_idx in range(num_frames):
            frame_data = data[frame_idx]
            binary_masks = frame_data['out_binary_masks']  # (num_objects, H, W)
            obj_ids = frame_data['out_obj_ids']  # (num_objects,)
            
            # 为每个对象分配标签
            for obj_idx, (mask, obj_id) in enumerate(zip(binary_masks, obj_ids)):
                # 使用 obj_id + 1 作为标签 (避免与背景0冲突)
                label = obj_id + 1
                all_masks[frame_idx][mask] = label
        
        output_data = all_masks
        print(f"  输出形状 (合并): {output_data.shape}")
        print(f"  唯一标签: {np.unique(output_data)}")
    
    else:
        # 方案2: 保持分离 -> (N, T, H, W)
        all_masks = np.zeros((num_objects, num_frames, h, w), dtype=np.uint8)
        
        for frame_idx in range(num_frames):
            frame_data = data[frame_idx]
            binary_masks = frame_data['out_binary_masks']  # (num_objects, H, W)
            
            for obj_idx, mask in enumerate(binary_masks):
                all_masks[obj_idx, frame_idx] = mask.astype(np.uint8) * 255
        
        output_data = all_masks
        print(f"  输出形状 (分离): {output_data.shape}")
    
    # 确定输出路径
    if output_path is None:
        output_path = pkl_path.replace('_segmentation_results.pkl', '_masks.npz')
    
    # 保存为 npz
    np.savez_compressed(output_path, arr_0=output_data)
    print(f"  已保存: {output_path}")
    print(f"  文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    return output_path


def convert_directory(input_dir: str, output_dir: str = None, merge_objects: bool = True):
    """
    批量转换目录下所有 pkl 文件
    """
    pkl_files = sorted(glob.glob(os.path.join(input_dir, '*_segmentation_results.pkl')))
    
    if not pkl_files:
        print(f"在 {input_dir} 中没有找到 *_segmentation_results.pkl 文件")
        return []
    
    print(f"找到 {len(pkl_files)} 个 pkl 文件")
    
    if output_dir is None:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    output_files = []
    for pkl_path in pkl_files:
        filename = os.path.basename(pkl_path)
        output_filename = filename.replace('_segmentation_results.pkl', '_masks.npz')
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            result = convert_single_pkl_to_npz(pkl_path, output_path, merge_objects)
            output_files.append(result)
        except Exception as e:
            print(f"  错误: {e}")
    
    print(f"\n转换完成! 共 {len(output_files)} 个文件")
    return output_files


def verify_npz_file(npz_path: str):
    """
    验证生成的 npz 文件是否符合 Cosmos 要求
    """
    print(f"\n验证文件: {npz_path}")
    
    data = np.load(npz_path)
    print(f"  包含的键: {list(data.keys())}")
    
    if 'arr_0' in data:
        arr = data['arr_0']
        print(f"  数据形状: {arr.shape}")
        print(f"  数据类型: {arr.dtype}")
        print(f"  值范围: [{arr.min()}, {arr.max()}]")
        print(f"  唯一值数量: {len(np.unique(arr))}")
        
        if arr.ndim == 3:
            print(f"  ✅ 格式正确: (T, H, W) = {arr.shape}")
        elif arr.ndim == 4:
            print(f"  ✅ 格式正确: (N, T, H, W) = {arr.shape}")
        else:
            print(f"  ⚠️ 维度异常: {arr.ndim}D")
        
        return True
    else:
        print("  ❌ 缺少 'arr_0' 键")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 SAM3 分割结果转换为 Cosmos 格式")
    parser.add_argument("input", help="输入 pkl 文件或目录")
    parser.add_argument("-o", "--output", help="输出路径 (文件或目录)")
    parser.add_argument("--separate", action="store_true", 
                        help="保持对象分离 (输出 N,T,H,W 而非 T,H,W)")
    parser.add_argument("--verify", action="store_true", help="验证生成的文件")
    
    args = parser.parse_args()
    
    merge_objects = not args.separate
    
    if os.path.isfile(args.input):
        # 单文件转换
        output_path = convert_single_pkl_to_npz(args.input, args.output, merge_objects)
        if args.verify:
            verify_npz_file(output_path)
    else:
        # 目录批量转换
        output_files = convert_directory(args.input, args.output, merge_objects)
        if args.verify and output_files:
            verify_npz_file(output_files[0])

