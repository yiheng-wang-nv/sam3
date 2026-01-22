#!/usr/bin/env python3
"""
将 sim 数据的 mask 文件重新组织成 galbot_lerobot_dataset 的格式

原始路径: stage_xxx/lerobot/videos/chunk-000/mask/episode_000000.npz
目标路径: stage_xxx/lerobot/masks/chunk-000/observation.images.ego_view/episode_000000_masks.npz
"""

import shutil
from pathlib import Path
import argparse


def reorganize_masks(base_dir: Path, dry_run: bool = False):
    """重新组织 mask 文件结构"""
    
    # 查找所有 stage 文件夹
    stage_dirs = sorted(base_dir.glob("stage*"))
    
    if not stage_dirs:
        print(f"No stage directories found in {base_dir}")
        return
    
    print(f"Found {len(stage_dirs)} stage directories:")
    for stage_dir in stage_dirs:
        print(f"  - {stage_dir.name}")
    print()
    
    for stage_dir in stage_dirs:
        lerobot_dir = stage_dir / "lerobot"
        
        if not lerobot_dir.exists():
            print(f"[SKIP] {stage_dir.name}: no lerobot folder")
            continue
        
        # 查找所有 chunk 文件夹中的 mask
        videos_dir = lerobot_dir / "videos"
        if not videos_dir.exists():
            print(f"[SKIP] {stage_dir.name}: no videos folder")
            continue
        
        # 遍历所有 chunk
        for chunk_dir in sorted(videos_dir.glob("chunk-*")):
            mask_src_dir = chunk_dir / "mask"
            
            if not mask_src_dir.exists():
                print(f"[SKIP] {stage_dir.name}/{chunk_dir.name}: no mask folder")
                continue
            
            # 创建目标目录
            masks_dst_dir = lerobot_dir / "masks" / chunk_dir.name / "observation.images.ego_view"
            
            # 获取所有 npz 文件
            npz_files = sorted(mask_src_dir.glob("episode_*.npz"))
            
            if not npz_files:
                print(f"[SKIP] {stage_dir.name}/{chunk_dir.name}: no npz files in mask folder")
                continue
            
            print(f"\n[PROCESS] {stage_dir.name}/{chunk_dir.name}: {len(npz_files)} mask files")
            print(f"  Source: {mask_src_dir}")
            print(f"  Target: {masks_dst_dir}")
            
            if not dry_run:
                masks_dst_dir.mkdir(parents=True, exist_ok=True)
            
            for npz_file in npz_files:
                # episode_000000.npz -> episode_000000_masks.npz
                src_name = npz_file.stem  # episode_000000
                dst_name = f"{src_name}_masks.npz"
                dst_path = masks_dst_dir / dst_name
                
                if dry_run:
                    print(f"    [DRY-RUN] {npz_file.name} -> {dst_name}")
                else:
                    shutil.copy2(npz_file, dst_path)
                    print(f"    [COPIED] {npz_file.name} -> {dst_name}")
            
            print(f"  Done: {len(npz_files)} files")
    
    print("\n" + "="*50)
    if dry_run:
        print("DRY RUN completed. No files were modified.")
        print("Run without --dry-run to actually copy files.")
    else:
        print("Reorganization completed!")


def main():
    parser = argparse.ArgumentParser(description="Reorganize sim mask files")
    parser.add_argument(
        "base_dir",
        type=Path,
        help="Base directory containing stage_xxx folders"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually copying files"
    )
    args = parser.parse_args()
    
    if not args.base_dir.exists():
        print(f"Error: {args.base_dir} does not exist")
        return 1
    
    reorganize_masks(args.base_dir, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    exit(main())

