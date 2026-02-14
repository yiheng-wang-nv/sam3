#!/usr/bin/env python3
"""
Debug script: draw detected table corners on frame 0 of multiple episodes.

Usage:
    conda run -n sam3 python sam3/task4_scripts/debug_table_top_line.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
from pathlib import Path
from postprocess_masks import postprocess_video_masks, fill_table_top_line

# ── Config ──────────────────────────────────────────────────────────────
SAM3_OUTPUT = Path(
    "/localhome/local-vennw/code/task4-2_02020205_merged/sam3_output/"
    "observation.images.head_right_camera_color_optical_frame"
)
OUT_DIR = Path("/localhome/local-vennw/code/sam3/task4_scripts/debug_table_top_line_ep030")

NUM_EPISODES = 100  # first 100 episodes

TABLE_LABEL = 1
FILL_TARGET = 6

# Corner search ROIs
TL_X_RANGE = (100, 230)
TL_Y_RANGE = (150, 200)
TR_X_RANGE = (300, 430)
TR_Y_RANGE = (150, 200)
BL_X_RANGE = (50, 170)
BL_Y_RANGE = (300, 370)
BR_X_RANGE = (350, 450)
BR_Y_RANGE = (260, 350)

# Postprocess settings (same as run_parallel_segmentation.sh, WITHOUT fill_table_top_line)
PP_KWARGS = dict(
    fill_holes=True,
    min_hole_size=64,
    min_object_size=50,
    closing_iterations=1,
    remove_small_objects_enabled=False,
    union_hole_fill=True,
    union_gap_fill=True,
    union_gap_closing_iterations=1,
    fill_interior_class=[1, 2, 3, 4, 5],
    fill_interior_target=6,
    fill_table_top_line_enabled=False,
)

# ── Color map ───────────────────────────────────────────────────────────
COLORMAP = {
    0: (0, 0, 0),
    1: (0, 100, 255),
    2: (0, 255, 0),
    3: (255, 255, 0),
    4: (255, 0, 0),
    5: (192, 192, 192),
    6: (255, 128, 0),
}

def colorize(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in COLORMAP.items():
        rgb[mask == label] = color
    for label in np.unique(mask):
        if label not in COLORMAP:
            rgb[mask == label] = (128, 128, 128)
    return rgb


def draw_roi_box(img, x_range, y_range, color, label):
    """Draw ROI rectangle and label."""
    x0, x1 = x_range
    y0, y1 = y_range
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)
    cv2.putText(img, label, (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)


def find_corner_in_roi(table_pixels, x_range, y_range, mode):
    """Find corner point in ROI. Returns (x, y) or None."""
    x0, x1 = x_range
    y0, y1 = y_range
    roi = table_pixels[y0:y1+1, x0:x1+1]
    coords = np.argwhere(roi)
    if coords.shape[0] == 0:
        return None
    ry, rx = coords[:, 0], coords[:, 1]
    if mode in ('tl', 'tr'):
        ty = int(ry.min())
        cx = rx[ry == ty]
        tx = int(cx.min()) if mode == 'tl' else int(cx.max())
    else:
        ty = int(ry.max())
        cx = rx[ry == ty]
        tx = int(cx.min()) if mode == 'bl' else int(cx.max())
    return (x0 + tx, y0 + ty)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(SAM3_OUTPUT.glob("*_masks.npz"))
    mask_files = [f for f in mask_files if "_post" not in f.name]
    mask_files = mask_files[:NUM_EPISODES]

    print(f"Processing {len(mask_files)} episodes, frame 0 only ...")

    for mask_file in mask_files:
        ep_name = mask_file.stem.replace("_masks", "")

        raw = np.load(mask_file)["arr_0"]  # (T, H, W)
        if raw.shape[0] == 0:
            continue
        frame0 = raw[0:1]  # (1, H, W)

        # Postprocess frame 0
        num_classes = int(frame0.max()) + 1 if frame0.size > 0 else 1
        pp = postprocess_video_masks(frame0, num_classes=num_classes, **PP_KWARGS)
        mask = pp[0]  # (H, W)

        table_pixels = (mask == TABLE_LABEL)

        # Find corners
        tl = find_corner_in_roi(table_pixels, TL_X_RANGE, TL_Y_RANGE, 'tl')
        tr = find_corner_in_roi(table_pixels, TR_X_RANGE, TR_Y_RANGE, 'tr')
        bl = find_corner_in_roi(table_pixels, BL_X_RANGE, BL_Y_RANGE, 'bl')
        br = find_corner_in_roi(table_pixels, BR_X_RANGE, BR_Y_RANGE, 'br')

        # Colorize
        rgb = colorize(mask)

        # Draw ROI boxes (dim cyan)
        roi_color = (100, 180, 180)
        draw_roi_box(rgb, TL_X_RANGE, TL_Y_RANGE, roi_color, "TL_roi")
        draw_roi_box(rgb, TR_X_RANGE, TR_Y_RANGE, roi_color, "TR_roi")
        draw_roi_box(rgb, BL_X_RANGE, BL_Y_RANGE, roi_color, "BL_roi")
        draw_roi_box(rgb, BR_X_RANGE, BR_Y_RANGE, roi_color, "BR_roi")

        # Draw quadrilateral and corners
        corners = [tl, tr, bl, br]
        names = ["TL", "TR", "BL", "BR"]
        found = all(c is not None for c in corners)

        if found:
            # Draw quad edges (red)
            cv2.line(rgb, tl, tr, (255, 0, 0), 2)
            cv2.line(rgb, tr, br, (255, 0, 0), 2)
            cv2.line(rgb, br, bl, (255, 0, 0), 2)
            cv2.line(rgb, bl, tl, (255, 0, 0), 2)

        for name, pt in zip(names, corners):
            if pt is not None:
                cv2.circle(rgb, pt, 5, (0, 0, 0), -1)
                cv2.circle(rgb, pt, 3, (255, 255, 255), -1)
                cv2.putText(rgb, f"{name}{pt}", (pt[0] + 6, pt[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            else:
                cv2.putText(rgb, f"{name}=NONE", (10, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        status = "OK" if found else "MISSING"
        cv2.putText(rgb, f"{ep_name} [{status}]", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_path = OUT_DIR / f"{ep_name}_corners.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        corner_str = " ".join(
            f"{n}={c}" if c else f"{n}=NONE"
            for n, c in zip(names, corners)
        )
        print(f"  {ep_name}: {corner_str}")

    print(f"Done! {len(mask_files)} images saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
