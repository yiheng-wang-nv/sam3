#!/usr/bin/env python3
"""
Add mask metadata only (no template):
- meta/modality.json: add "mask" mapping inferred from target
- meta/info.json: add/update "mask_path"

Usage:
  python update_orca_meta_masks_only.py \
    --target_root /path/to/target_dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _discover_mask_keys(target_root: Path) -> list[str]:
    masks_root = target_root / "masks"
    if not masks_root.exists():
        return []
    mask_keys = set()
    for chunk_dir in masks_root.glob("chunk-*"):
        for camera_dir in chunk_dir.iterdir():
            if camera_dir.is_dir():
                mask_keys.add(camera_dir.name)
    return sorted(mask_keys)


def update_masks_only(target_root: Path) -> None:
    target_meta = target_root / "meta"
    target_meta.mkdir(parents=True, exist_ok=True)

    target_modality_path = target_meta / "modality.json"
    target_modality = _read_json(target_modality_path)

    mask_keys = _discover_mask_keys(target_root)
    if not mask_keys:
        # fallback to video keys if masks folder is missing
        mask_keys = list(target_modality.get("video", {}).keys())

    mask_mapping = {}
    video_mapping = target_modality.get("video", {})
    for key in mask_keys:
        if key.startswith("observation.images."):
            short_key = key.replace("observation.images.", "", 1)
            mask_mapping[short_key] = {"original_key": key}
            continue
        if key in video_mapping:
            mask_mapping[key] = {"original_key": video_mapping[key]["original_key"]}
        else:
            # fallback to a common naming convention
            mask_mapping[key] = {"original_key": f"observation.images.{key}"}

    target_modality["mask"] = mask_mapping
    _write_json(target_modality_path, target_modality)

    target_info_path = target_meta / "info.json"
    target_info = _read_json(target_info_path)

    target_info["mask_path"] = (
        "masks/chunk-{episode_chunk:03d}/{mask_key}/"
        "episode_{episode_index:06d}_masks.npz"
    )
    _write_json(target_info_path, target_info)

    print(f"Updated mask metadata in {target_meta}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update ORCA mask metadata only")
    parser.add_argument("--target_root", required=True, help="Target dataset root")
    args = parser.parse_args()

    update_masks_only(Path(args.target_root))


if __name__ == "__main__":
    main()
