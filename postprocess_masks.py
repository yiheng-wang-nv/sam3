"""
Postprocess segmentation masks to fill small holes and remove noise.

Usage:
python /localhome/local-vennw/code/sam3/postprocess_masks.py \
  --input_dir /localhome/local-vennw/code/task7_20260122_trimmed/sam3_output \
  --fill_interior_class 1,3 \
  --fill_interior_target 4 \
  --overwrite \
  --copy_to_dataset_root /localhome/local-vennw/code/task7_20260122_trimmed

"""

import argparse
import numpy as np
import re
import shutil
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
from skimage.morphology import remove_small_objects, remove_small_holes
import inspect
import cv2


def postprocess_mask_2d(mask, 
                        fill_holes=True,
                        min_hole_size=64,
                        min_object_size=50,
                        closing_iterations=1,
                        remove_small_objects_enabled=True):
    """
    Postprocess a single 2D binary mask.
    
    Args:
        mask: 2D binary mask (H, W)
        fill_holes: Whether to fill holes
        min_hole_size: Fill holes smaller than this (pixels)
        min_object_size: Remove objects smaller than this (pixels)
        closing_iterations: Morphological closing iterations
    
    Returns:
        Processed binary mask
    """
    if mask.sum() == 0:
        return mask
    
    result = mask.astype(bool)
    
    # 1. Morphological closing (fill small gaps)
    if closing_iterations > 0:
        struct = ndimage.generate_binary_structure(2, 1)  # 2D 4-connectivity
        result = ndimage.binary_dilation(result, struct, iterations=closing_iterations)
        result = ndimage.binary_erosion(result, struct, iterations=closing_iterations)
    
    # 2. Fill small holes
    if fill_holes and min_hole_size > 0:
        holes_kwargs = {}
        if "max_size" in inspect.signature(remove_small_holes).parameters:
            holes_kwargs["max_size"] = min_hole_size
        else:
            holes_kwargs["area_threshold"] = min_hole_size
        result = remove_small_holes(result, **holes_kwargs)
    
    # 3. Remove small noise objects
    if remove_small_objects_enabled and min_object_size > 0:
        objects_kwargs = {}
        if "max_size" in inspect.signature(remove_small_objects).parameters:
            objects_kwargs["max_size"] = min_object_size
        else:
            objects_kwargs["min_size"] = min_object_size
        result = remove_small_objects(result, **objects_kwargs)
    
    return result.astype(np.uint8)


def fill_interior_with_class(mask, target_classes, fill_class=4):
    """
    Fill background (0) inside the contour of target_classes with fill_class.

    Rule:
      1) Fill interiors for each target_class independently.
      2) Take the UNION of all target_classes plus fill_class, then fill its
         interior background to fill_class.

    Args:
        mask: Multi-class mask (H, W) with labels 0,1,2,3,...
        target_classes: List of classes whose interior to fill (e.g., [1, 3] for red and blue)
                       Can also be a single int for backwards compatibility.
        fill_class: The new class label for interior background (default: 4)

    Returns:
        Modified mask with interior filled
    """
    # Handle single int input
    if isinstance(target_classes, int):
        target_classes = [target_classes]

    result = mask.copy()

    # Step 1: fill each target class independently
    for target_class in target_classes:
        binary = (result == target_class)
        if binary.sum() == 0:
            continue
        filled = ndimage.binary_fill_holes(binary)
        interior_background = filled & (result == 0)
        result[interior_background] = fill_class

    # Step 2: union of target classes + fill_class, then fill interior once more
    union_binary = (result == fill_class)
    for target_class in target_classes:
        union_binary |= (result == target_class)
    if union_binary.sum() == 0:
        return result

    filled_union = ndimage.binary_fill_holes(union_binary)
    interior_background = filled_union & (result == 0)
    result[interior_background] = fill_class

    return result


def fill_blue_table_quadrant(
    mask,
    blue_label=1,
    fill_label=4,
    mode="right_down",
    y_pad_top=60,
    y_pad_bottom=60,
    skip_if_label_above=None,
    skip_if_label_area_gt=None,
    margin=1,
):
    """
    Fill the quadrant region below the blue table's top edge.
    
    For mode="right_down" (left arm camera): fills bottom-right region
    For mode="left_down" (right arm camera): fills bottom-left region
    
    Uses two-point strategy with margin.
    """
    h, w = mask.shape
    y_top = int(np.clip(y_pad_top, 0, h - 1))
    y_bottom = int(np.clip(h - 1 - y_pad_bottom, 0, h - 1))
    if y_bottom <= y_top:
        return mask

    # optional skip if another label is higher
    if skip_if_label_above is not None:
        other_coords = np.argwhere(mask == skip_if_label_above)
        if other_coords.size > 0:
            other_y_min = int(other_coords[:, 0].min())
            if other_y_min < y_top:
                return mask
            if skip_if_label_area_gt is not None:
                if other_coords.shape[0] > skip_if_label_area_gt:
                    return mask

    # Two-Point Strategy (with margin & slope check)
    coords = np.argwhere(mask == blue_label)
    if coords.size == 0:
        return mask
    
    ys = coords[:, 0]
    xs = coords[:, 1]
    
    # filter blue pixels to valid region
    valid = (ys >= y_top) & (ys <= y_bottom)
    ys = ys[valid]
    xs = xs[valid]
    if ys.size == 0:
        return mask

    min_x = int(xs.min())
    max_x = int(xs.max())

    # Shrink range by margin to avoid edge noise
    target_x_left = min_x + margin
    target_x_right = max_x - margin

    if target_x_left >= target_x_right:
        target_x_left = min_x
        target_x_right = max_x

    # Find left points
    valid_left = xs >= target_x_left
    if not np.any(valid_left): valid_left = xs >= min_x
    xs_left_subset = xs[valid_left]
    ys_left_subset = ys[valid_left]
    current_min_x = int(xs_left_subset.min())
    left_p_ys = ys_left_subset[xs_left_subset == current_min_x]
    left_up = (current_min_x, int(left_p_ys.min()))
    left_down = (current_min_x, int(left_p_ys.max()))

    # Find right points
    valid_right = xs <= target_x_right
    if not np.any(valid_right): valid_right = xs <= max_x
    xs_right_subset = xs[valid_right]
    ys_right_subset = ys[valid_right]
    current_max_x = int(xs_right_subset.max())
    right_p_ys = ys_right_subset[xs_right_subset == current_max_x]
    right_up = (current_max_x, int(right_p_ys.min()))
    right_down = (current_max_x, int(right_p_ys.max()))

    def y_on_line_at_x(p1, p2, x):
        x1, y1 = p1
        x2, y2 = p2
        if x2 == x1: return y1
        t = (x - x1) / float(x2 - x1)
        return y1 + t * (y2 - y1)

    def x_on_line_at_y(p1, p2, y):
        x1, y1 = p1
        x2, y2 = p2
        if y2 == y1: return x1
        t = (y - y1) / float(y2 - y1)
        return x1 + t * (x2 - x1)

    if mode == "right_down":
        y_right = y_on_line_at_x(left_up, right_up, w - 1)
        y_right = int(np.clip(round(y_right), y_top, y_bottom))

        x_bottom = x_on_line_at_y(left_up, left_down, y_bottom)
        x_bottom = int(np.clip(round(x_bottom), 0, w - 1))

        poly = np.array([
            [left_up[0], left_up[1]],
            [w - 1, y_right],
            [w - 1, y_bottom],
            [x_bottom, y_bottom],
        ], dtype=np.int32)
        
    elif mode == "left_down":
        y_left = y_on_line_at_x(left_up, right_up, 0)
        y_left = int(np.clip(round(y_left), y_top, y_bottom))

        x_bottom = x_on_line_at_y(right_up, right_down, y_bottom)
        x_bottom = int(np.clip(round(x_bottom), 0, w - 1))

        poly = np.array([
            [right_up[0], right_up[1]],
            [0, y_left],
            [0, y_bottom],
            [x_bottom, y_bottom],
        ], dtype=np.int32)
    else:
        raise ValueError(f"Unknown quadrant mode: {mode}")

    fill_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(fill_mask, [poly], 1)

    result = mask.copy()
    to_fill = (fill_mask == 1) & (result == 0)
    result[to_fill] = fill_label
    return result


def fill_table_top_line(mask, table_label=1, fill_label=6,
                        tl_x_range=(100, 230), tl_y_range=(150, 200),
                        tr_x_range=(300, 430), tr_y_range=(150, 200),
                        bl_x_range=(50, 170), bl_y_range=(300, 370),
                        br_x_range=(350, 450), br_y_range=(260, 350)):
    """
    Fill background inside the quadrilateral formed by four corner points
    of table_label pixels, each searched within a constrained ROI to avoid
    being misled by scattered noise pixels.

    Corner search ROIs (x_min, x_max, y_min, y_max):
      TL: top-left      -- min y, then min x within ROI
      TR: top-right     -- min y, then max x within ROI
      BL: bottom-left   -- max y, then min x within ROI
      BR: bottom-right  -- max y, then max x within ROI

    Args:
        mask: Multi-class mask (H, W) with labels 0, 1, 2, ...
        table_label: Label for the table class (default: 1)
        fill_label: Value to fill background with (default: 6)
        tl_x_range, tl_y_range: ROI for top-left corner search
        tr_x_range, tr_y_range: ROI for top-right corner search
        bl_x_range, bl_y_range: ROI for bottom-left corner search
        br_x_range, br_y_range: ROI for bottom-right corner search

    Returns:
        Modified mask with filled background inside the table quadrilateral.
    """
    h, w = mask.shape

    table_pixels = (mask == table_label)
    if table_pixels.sum() == 0:
        return mask

    def _find_corner(mask2d, x_min, x_max, y_min, y_max, mode):
        """Find a corner point within the given ROI.
        mode: 'tl' (min y then min x), 'tr' (min y then max x),
              'bl' (max y then min x), 'br' (max y then max x)
        Returns (x, y) or None.
        """
        roi = mask2d[y_min:y_max+1, x_min:x_max+1]
        coords_roi = np.argwhere(roi)  # [y_local, x_local]
        if coords_roi.shape[0] == 0:
            return None
        ry, rx = coords_roi[:, 0], coords_roi[:, 1]
        if mode in ('tl', 'tr'):
            target_y = int(ry.min())
            cand_x = rx[ry == target_y]
            target_x = int(cand_x.min()) if mode == 'tl' else int(cand_x.max())
        else:  # bl, br
            target_y = int(ry.max())
            cand_x = rx[ry == target_y]
            target_x = int(cand_x.min()) if mode == 'bl' else int(cand_x.max())
        return (x_min + target_x, y_min + target_y)

    tl = _find_corner(table_pixels, tl_x_range[0], tl_x_range[1], tl_y_range[0], tl_y_range[1], 'tl')
    tr = _find_corner(table_pixels, tr_x_range[0], tr_x_range[1], tr_y_range[0], tr_y_range[1], 'tr')
    bl = _find_corner(table_pixels, bl_x_range[0], bl_x_range[1], bl_y_range[0], bl_y_range[1], 'bl')
    br = _find_corner(table_pixels, br_x_range[0], br_x_range[1], br_y_range[0], br_y_range[1], 'br')

    if any(p is None for p in (tl, tr, bl, br)):
        return mask

    # Quadrilateral: TL -> TR -> BR -> BL  (cv2 uses (x, y) order)
    poly = np.array([
        [tl[0], tl[1]],
        [tr[0], tr[1]],
        [br[0], br[1]],
        [bl[0], bl[1]],
    ], dtype=np.int32)

    fill_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(fill_mask, [poly], 1)

    # Only fill where original mask is background (0)
    result = mask.copy()
    to_fill = (fill_mask == 1) & (mask == 0)
    result[to_fill] = fill_label

    return result


def fill_scanline_between(mask, source_label=1, fill_value=3):
    """
    Row-based scanline fill: for each row, fill background (0) pixels between
    the first and last occurrence of source_label with fill_value.
    """
    result = mask.copy()
    H, W = mask.shape
    for y in range(H):
        cols = np.where(mask[y] == source_label)[0]
        if cols.size == 0:
            continue
        start, end = cols[0], cols[-1]
        bg = result[y, start:end + 1] == 0
        result[y, start:end + 1][bg] = fill_value
    return result


def parse_fill_bg_roi(spec_str):
    """
    Parse a fill_bg_roi spec string.
    Format: "frame_start,frame_end_ratio,y_min,y_max,x_min,x_max,target"
    
    frame_start: int, absolute frame index (0-based)
    frame_end_ratio: float, fraction of total frames (e.g. 0.5 = first half)
    y_min, y_max: int, row range (inclusive). -1 = full range.
    x_min, x_max: int, col range (inclusive). -1 = full range.
    target: int, fill value for background pixels.
    
    Returns: tuple (frame_start, frame_end_ratio, y_min, y_max, x_min, x_max, target)
    """
    parts = [p.strip() for p in spec_str.split(",")]
    if len(parts) != 7:
        raise ValueError(
            f"fill_bg_roi spec must have 7 comma-separated values "
            f"(frame_start,frame_end_ratio,y_min,y_max,x_min,x_max,target), got {len(parts)}: {spec_str}"
        )
    frame_start = int(parts[0])
    frame_end_ratio = float(parts[1])
    y_min = int(parts[2])
    y_max = int(parts[3])
    x_min = int(parts[4])
    x_max = int(parts[5])
    target = int(parts[6])
    return (frame_start, frame_end_ratio, y_min, y_max, x_min, x_max, target)


def fill_bg_topleft_rect(mask, source_label, fill_value, y_max):
    """
    Find the top-left corner of source_label (min y, then min x),
    fill all background (0) in the rectangle (0, y_corner) to (x_corner, y_max) with fill_value.
    """
    coords = np.argwhere(mask == source_label)
    if coords.size == 0:
        return mask
    ys = coords[:, 0]
    xs = coords[:, 1]
    min_y = int(ys.min())
    min_x = int(xs[ys == min_y].min())
    h = mask.shape[0]
    y_end = min(y_max + 1, h)
    if min_y >= y_end or min_x <= 0:
        return mask
    result = mask.copy()
    roi = result[min_y:y_end, 0:min_x]
    roi[roi == 0] = fill_value
    return result


def fill_bg_topright_rect(mask, source_label, fill_value, y_max, y_threshold=200):
    """
    For all source_label pixels with y < y_threshold, fill the union of
    their left-below rectangles: for each such pixel (x_i, y_i), the rect
    [0:x_i, y_i:y_max] has bg filled with fill_value.

    Efficiently computed via cumulative max x scanning top-to-bottom.
    """
    coords = np.argwhere(mask == source_label)
    if coords.size == 0:
        return mask
    ys = coords[:, 0]
    xs = coords[:, 1]
    valid = ys < y_threshold
    if not np.any(valid):
        return mask
    ys_v = ys[valid]
    xs_v = xs[valid]

    min_y = int(ys_v.min())
    h = mask.shape[0]
    y_end = min(y_max + 1, h)
    if min_y >= y_end:
        return mask

    max_x_by_row = {}
    for y_val, x_val in zip(ys_v.tolist(), xs_v.tolist()):
        if y_val not in max_x_by_row or x_val > max_x_by_row[y_val]:
            max_x_by_row[y_val] = x_val

    result = mask.copy()
    cum_max_x = 0
    for row in range(min_y, y_end):
        if row in max_x_by_row:
            cum_max_x = max(cum_max_x, max_x_by_row[row])
        if cum_max_x > 0:
            roi = result[row, 0:cum_max_x]
            roi[roi == 0] = fill_value
    return result


def fill_bg_leftmost_corner_rect(mask, source_label, fill_value, y_max, skip_label=None):
    """
    Find the leftmost pixel of source_label (min x, then min y among those).
    Fill background (0) in the rectangle [0:x, y:y_max] with fill_value.
    If skip_label is set and present in the frame, skip entirely.
    """
    if skip_label is not None and np.any(mask == skip_label):
        return mask
    coords = np.argwhere(mask == source_label)
    if coords.size == 0:
        return mask
    ys = coords[:, 0]
    xs = coords[:, 1]
    min_x = int(xs.min())
    min_y = int(ys[xs == min_x].min())
    h = mask.shape[0]
    y_end = min(y_max + 1, h)
    if min_y >= y_end or min_x <= 0:
        return mask
    result = mask.copy()
    roi = result[min_y:y_end, 0:min_x]
    roi[roi == 0] = fill_value
    return result


def fill_bg_corner_rect(mask, source_label, fill_value, y_max, mode="topright", x_first=False):
    """
    Find the top corner point of source_label and fill background in a rectangle.

    mode="topright": fill [y:y_max, 0:x+1]
    mode="topleft":  fill [y:y_max, x:W]

    x_first=False (default): min y first, then extremal x among that row.
    x_first=True:  extremal x first, then min y among that column.
    """
    coords = np.argwhere(mask == source_label)
    if coords.size == 0:
        return mask
    ys = coords[:, 0]
    xs = coords[:, 1]

    if mode == "topright":
        if x_first:
            anchor_x = int(xs.max())
            anchor_y = int(ys[xs == anchor_x].min())
        else:
            anchor_y = int(ys.min())
            anchor_x = int(xs[ys == anchor_y].max())
    elif mode == "topleft":
        if x_first:
            anchor_x = int(xs.min())
            anchor_y = int(ys[xs == anchor_x].min())
        else:
            anchor_y = int(ys.min())
            anchor_x = int(xs[ys == anchor_y].min())
    else:
        raise ValueError(f"Unknown corner_rect mode: {mode}")

    h, w = mask.shape
    y_end = min(y_max + 1, h)
    if anchor_y >= y_end:
        return mask

    result = mask.copy()
    if mode == "topright":
        if anchor_x <= 0:
            return mask
        roi = result[anchor_y:y_end, 0:anchor_x + 1]
    else:
        if anchor_x >= w - 1:
            return mask
        roi = result[anchor_y:y_end, anchor_x:w]

    roi[roi == 0] = fill_value
    return result


def postprocess_video_masks(masks, 
                            num_classes=4,
                            fill_holes=True,
                            min_hole_size=64,
                            min_object_size=50,
                            closing_iterations=1,
                            fill_interior_class=None,
                            fill_interior_target=4,
                            union_hole_fill=False,
                            remove_small_objects_enabled=True,
                            union_gap_fill=False,
                            union_gap_closing_iterations=1,
                            fill_blue_table_quadrant_enabled=False,
                            blue_table_label=1,
                            blue_table_target=4,
                            blue_table_quadrant_mode="right_down",
                            blue_table_y_pad_top=60,
                            blue_table_y_pad_bottom=60,
                            blue_table_skip_if_label_above=None,
                            blue_table_skip_if_label_area_gt=None,
                            fill_table_top_line_enabled=False,
                            table_top_label=1,
                            table_top_fill_target=6,
                            table_top_tl_x_range=(100, 230),
                            table_top_tl_y_range=(150, 200),
                            table_top_tr_x_range=(300, 430),
                            table_top_tr_y_range=(150, 200),
                            table_top_bl_x_range=(50, 170),
                            table_top_bl_y_range=(300, 370),
                            table_top_br_x_range=(350, 450),
                            table_top_br_y_range=(260, 350),
                            fill_bg_roi_list=None,
                            scanline_fill_enabled=False,
                            scanline_source_label=1,
                            scanline_fill_value=3,
                            topleft_rect_enabled=False,
                            topleft_rect_label=1,
                            topleft_rect_fill=5,
                            topleft_rect_y_max=420,
                            topleft_rect_frame_start=10,
                            topleft_rect_frame_end_ratio=0.667,
                            topright_rect_enabled=False,
                            topright_rect_label=1,
                            topright_rect_fill=5,
                            topright_rect_y_max=420,
                            topright_rect_y_threshold=200,
                            topright_rect_frame_start_ratio=0.667,
                            leftmost_rect_enabled=False,
                            leftmost_rect_label=1,
                            leftmost_rect_fill=5,
                            leftmost_rect_y_max=420,
                            leftmost_rect_skip_label=3,
                            corner_rect_enabled=False,
                            corner_rect_mode="topright",
                            corner_rect_labels=None,
                            corner_rect_fill=4,
                            corner_rect_y_max=420,
                            corner_rect_x_first_labels=None,
                            temporal_fill_enabled=False,
                            temporal_fill_labels=None,
                            temporal_fill_union_labels=None,
                            temporal_fill_value=5):
    """
    Postprocess masks for all frames in a video.
    
    Args:
        masks: (T, H, W) array with class labels 0,1,2,...
        num_classes: Number of classes (including background 0)
        fill_holes, min_hole_size, min_object_size, closing_iterations: 
            same as postprocess_mask_2d
        fill_interior_class: If set, fill background inside this class's contour
                            (e.g., 1 for red). None to disable.
        fill_interior_target: The new class label for filled interior (default: 4)
        fill_bg_roi_list: List of ROI fill specs. Each is a tuple:
            (frame_start, frame_end_ratio, y_min, y_max, x_min, x_max, target)
            Fill all background (0) pixels within the ROI with target value
            for the specified frame range.
        scanline_fill_enabled: Row-based fill between first/last source_label pixels.
        scanline_source_label: Label to scan for (default: 1).
        scanline_fill_value: Value to fill background with (default: 3).
    
    Returns:
        Processed masks (T, H, W)
    """
    T, H, W = masks.shape
    result = np.zeros_like(masks)
    
    # Precompute topleft_rect frame range
    _tl_frame_start = topleft_rect_frame_start if topleft_rect_enabled else T
    _tl_frame_end = int(T * topleft_rect_frame_end_ratio) if topleft_rect_enabled else 0

    # Precompute topright_rect frame range
    _tr_frame_start = int(T * topright_rect_frame_start_ratio) if topright_rect_enabled else T
    _tr_frame_end = T if topright_rect_enabled else 0

    # Precompute ROI frame ranges
    _roi_ranges = []
    if fill_bg_roi_list:
        for roi in fill_bg_roi_list:
            fs, fer, y0, y1, x0, x1, tgt = roi
            frame_end = int(T * fer)
            ry0 = 0 if y0 < 0 else y0
            ry1 = H if y1 < 0 else min(y1 + 1, H)
            rx0 = 0 if x0 < 0 else x0
            rx1 = W if x1 < 0 else min(x1 + 1, W)
            _roi_ranges.append((fs, frame_end, ry0, ry1, rx0, rx1, tgt))
    
    for t in range(T):
        frame_mask = masks[t]
        processed_frame = np.zeros((H, W), dtype=np.uint8)
        
        # Process each class separately (skip background 0)
        for cls in range(1, num_classes):
            binary_mask = (frame_mask == cls).astype(np.uint8)
            
            if binary_mask.sum() > 0:
                processed = postprocess_mask_2d(
                    binary_mask,
                    fill_holes=False if union_hole_fill else fill_holes,
                    min_hole_size=min_hole_size,
                    min_object_size=min_object_size,
                    closing_iterations=closing_iterations,
                    remove_small_objects_enabled=remove_small_objects_enabled,
                )
                # Only write where currently background (avoid overwriting)
                processed_frame = np.where(
                    (processed > 0) & (processed_frame == 0),
                    cls,
                    processed_frame
                )
        
        # Optional union-based hole filling: any holes inside union of >0 become target
        if union_hole_fill and fill_holes and min_hole_size > 0:
            union_binary = processed_frame > 0
            if union_binary.any():
                holes_kwargs = {}
                if "max_size" in inspect.signature(remove_small_holes).parameters:
                    holes_kwargs["max_size"] = min_hole_size
                else:
                    holes_kwargs["area_threshold"] = min_hole_size
                filled_union = remove_small_holes(union_binary, **holes_kwargs)
                interior_background = filled_union & (processed_frame == 0)
                processed_frame[interior_background] = fill_interior_target

        # Optional union gap fill: close thin background lines between classes
        if union_gap_fill and union_gap_closing_iterations > 0:
            union_binary = processed_frame > 0
            if union_binary.any():
                struct = ndimage.generate_binary_structure(2, 1)
                closed_union = ndimage.binary_closing(
                    union_binary, structure=struct, iterations=union_gap_closing_iterations
                )
                gap = closed_union & (processed_frame == 0)
                if gap.any():
                    _, indices = ndimage.distance_transform_edt(
                        processed_frame == 0, return_indices=True
                    )
                    nearest = processed_frame[indices[0], indices[1]]
                    fill_mask = gap & (nearest > 0)
                    processed_frame[fill_mask] = nearest[fill_mask]

        if fill_blue_table_quadrant_enabled:
            processed_frame = fill_blue_table_quadrant(
                processed_frame,
                blue_label=blue_table_label,
                fill_label=blue_table_target,
                mode=blue_table_quadrant_mode,
                y_pad_top=blue_table_y_pad_top,
                y_pad_bottom=blue_table_y_pad_bottom,
                skip_if_label_above=blue_table_skip_if_label_above,
                skip_if_label_area_gt=blue_table_skip_if_label_area_gt,
            )

        if fill_table_top_line_enabled:
            processed_frame = fill_table_top_line(
                processed_frame,
                table_label=table_top_label,
                fill_label=table_top_fill_target,
                tl_x_range=table_top_tl_x_range,
                tl_y_range=table_top_tl_y_range,
                tr_x_range=table_top_tr_x_range,
                tr_y_range=table_top_tr_y_range,
                bl_x_range=table_top_bl_x_range,
                bl_y_range=table_top_bl_y_range,
                br_x_range=table_top_br_x_range,
                br_y_range=table_top_br_y_range,
            )

        if scanline_fill_enabled:
            processed_frame = fill_scanline_between(
                processed_frame,
                source_label=scanline_source_label,
                fill_value=scanline_fill_value,
            )

        # Apply interior filling rule: background inside target_class -> fill_class
        if fill_interior_class is not None:
            processed_frame = fill_interior_with_class(
                processed_frame, 
                target_classes=fill_interior_class, 
                fill_class=fill_interior_target
            )

        # Fill background in ROI regions
        for fs, fe, ry0, ry1, rx0, rx1, tgt in _roi_ranges:
            if fs <= t < fe:
                roi = processed_frame[ry0:ry1, rx0:rx1]
                roi[roi == 0] = tgt

        # Fill bg in rect left-below the top-left corner of a label
        if topleft_rect_enabled and _tl_frame_start <= t < _tl_frame_end:
            processed_frame = fill_bg_topleft_rect(
                processed_frame, topleft_rect_label, topleft_rect_fill, topleft_rect_y_max
            )

        # Fill bg in rect left-below the top-right corner of a label
        if topright_rect_enabled and _tr_frame_start <= t < _tr_frame_end:
            processed_frame = fill_bg_topright_rect(
                processed_frame, topright_rect_label, topright_rect_fill,
                topright_rect_y_max, topright_rect_y_threshold
            )

        # Fill bg below-left of leftmost label pixel.
        # First half: always apply. Second half: only if skip_label absent.
        if leftmost_rect_enabled:
            in_first_half = t < T // 2
            skip = None if in_first_half else leftmost_rect_skip_label
            processed_frame = fill_bg_leftmost_corner_rect(
                processed_frame, leftmost_rect_label, leftmost_rect_fill,
                leftmost_rect_y_max, skip_label=skip
            )

        if corner_rect_enabled and corner_rect_labels:
            for lbl in corner_rect_labels:
                use_x_first = corner_rect_x_first_labels and lbl in corner_rect_x_first_labels
                processed_frame = fill_bg_corner_rect(
                    processed_frame, lbl, corner_rect_fill,
                    corner_rect_y_max, corner_rect_mode, x_first=use_x_first
                )

        result[t] = processed_frame

    if temporal_fill_enabled:
        all_relevant = list(set(
            (temporal_fill_labels or []) +
            (temporal_fill_union_labels or []) +
            [temporal_fill_value]
        ))
        first_has = np.isin(result[0], all_relevant)
        last_has = np.isin(result[T - 1], all_relevant)
        skip_mask = ~first_has & ~last_has

        if temporal_fill_union_labels:
            union_mask = np.zeros((result.shape[1], result.shape[2]), dtype=bool)
            for t in range(T):
                union_mask |= np.isin(result[t], temporal_fill_union_labels)
            union_mask &= ~skip_mask
            for t in range(T):
                result[t][(result[t] == 0) & union_mask] = temporal_fill_value

        if temporal_fill_labels:
            propagate_labels = list(set(temporal_fill_labels) | {temporal_fill_value})
            for t in range(T - 1):
                source_mask = np.isin(result[t], propagate_labels) & ~skip_mask
                result[t + 1][(result[t + 1] == 0) & source_mask] = temporal_fill_value
            for t in range(T - 1, 0, -1):
                source_mask = np.isin(result[t], propagate_labels) & ~skip_mask
                result[t - 1][(result[t - 1] == 0) & source_mask] = temporal_fill_value

    return result


def _process_single_file(args_tuple):
    """
    Process a single mask file. Designed to be called from multiprocessing.Pool.
    Takes a single tuple argument for compatibility with Pool.imap_unordered.
    """
    (mask_file, num_classes, fill_holes, min_hole_size, min_object_size,
     closing_iterations, fill_interior_class, fill_interior_target,
     union_hole_fill, remove_small_objects_enabled, union_gap_fill,
     union_gap_closing_iterations, fill_blue_table_quadrant_enabled,
     blue_table_label, blue_table_target, blue_table_quadrant_mode,
     blue_table_y_pad_top, blue_table_y_pad_bottom,
     blue_table_skip_if_label_above, blue_table_skip_if_label_area_gt,
     overwrite, fill_bg_roi_list,
     fill_table_top_line_enabled, table_top_label, table_top_fill_target,
     table_top_corner_ranges,
     scanline_fill_enabled, scanline_source_label, scanline_fill_value,
     topleft_rect_enabled, topleft_rect_label, topleft_rect_fill,
     topleft_rect_y_max, topleft_rect_frame_start, topleft_rect_frame_end_ratio,
     topright_rect_enabled, topright_rect_label, topright_rect_fill,
     topright_rect_y_max, topright_rect_y_threshold, topright_rect_frame_start_ratio,
     leftmost_rect_enabled, leftmost_rect_label, leftmost_rect_fill,
     leftmost_rect_y_max, leftmost_rect_skip_label,
     corner_rect_enabled, corner_rect_mode, corner_rect_labels,
     corner_rect_fill, corner_rect_y_max, corner_rect_x_first_labels,
     temporal_fill_enabled, temporal_fill_labels, temporal_fill_union_labels,
     temporal_fill_value) = args_tuple

    mask_file = Path(mask_file)
    out_path = mask_file.parent / mask_file.name.replace("_masks.npz", "_masks_post.npz")

    if out_path.exists() and not overwrite:
        return mask_file.name, "skipped"

    # Load masks
    data = np.load(mask_file)
    masks = data['arr_0']  # (T, H, W)

    # Detect number of classes (use max label + 1, not count of unique values,
    # to handle non-contiguous labels like [0,1,2,3,5] where label 4 is absent)
    detected_classes = int(masks.max()) + 1 if masks.size > 0 else 1
    use_num_classes = max(num_classes, detected_classes)

    # Process
    processed = postprocess_video_masks(
        masks,
        num_classes=use_num_classes,
        fill_holes=fill_holes,
        min_hole_size=min_hole_size,
        min_object_size=min_object_size,
        closing_iterations=closing_iterations,
        fill_interior_class=fill_interior_class,
        fill_interior_target=fill_interior_target,
        union_hole_fill=union_hole_fill,
        remove_small_objects_enabled=remove_small_objects_enabled,
        union_gap_fill=union_gap_fill,
        union_gap_closing_iterations=union_gap_closing_iterations,
        fill_blue_table_quadrant_enabled=fill_blue_table_quadrant_enabled,
        blue_table_label=blue_table_label,
        blue_table_target=blue_table_target,
        blue_table_quadrant_mode=blue_table_quadrant_mode,
        blue_table_y_pad_top=blue_table_y_pad_top,
        blue_table_y_pad_bottom=blue_table_y_pad_bottom,
        blue_table_skip_if_label_above=blue_table_skip_if_label_above,
        blue_table_skip_if_label_area_gt=blue_table_skip_if_label_area_gt,
        fill_table_top_line_enabled=fill_table_top_line_enabled,
        table_top_label=table_top_label,
        table_top_fill_target=table_top_fill_target,
        **table_top_corner_ranges,
        fill_bg_roi_list=fill_bg_roi_list,
        scanline_fill_enabled=scanline_fill_enabled,
        scanline_source_label=scanline_source_label,
        scanline_fill_value=scanline_fill_value,
        topleft_rect_enabled=topleft_rect_enabled,
        topleft_rect_label=topleft_rect_label,
        topleft_rect_fill=topleft_rect_fill,
        topleft_rect_y_max=topleft_rect_y_max,
        topleft_rect_frame_start=topleft_rect_frame_start,
        topleft_rect_frame_end_ratio=topleft_rect_frame_end_ratio,
        topright_rect_enabled=topright_rect_enabled,
        topright_rect_label=topright_rect_label,
        topright_rect_fill=topright_rect_fill,
        topright_rect_y_max=topright_rect_y_max,
        topright_rect_y_threshold=topright_rect_y_threshold,
        topright_rect_frame_start_ratio=topright_rect_frame_start_ratio,
        leftmost_rect_enabled=leftmost_rect_enabled,
        leftmost_rect_label=leftmost_rect_label,
        leftmost_rect_fill=leftmost_rect_fill,
        leftmost_rect_y_max=leftmost_rect_y_max,
        leftmost_rect_skip_label=leftmost_rect_skip_label,
        corner_rect_enabled=corner_rect_enabled,
        corner_rect_mode=corner_rect_mode,
        corner_rect_labels=corner_rect_labels,
        corner_rect_fill=corner_rect_fill,
        corner_rect_y_max=corner_rect_y_max,
        corner_rect_x_first_labels=corner_rect_x_first_labels,
        temporal_fill_enabled=temporal_fill_enabled,
        temporal_fill_labels=temporal_fill_labels,
        temporal_fill_union_labels=temporal_fill_union_labels,
        temporal_fill_value=temporal_fill_value,
    )

    # Save
    np.savez_compressed(out_path, processed)
    return mask_file.name, "done"


def process_directory(input_dir: Path,
                      num_classes=4,
                      fill_holes=True,
                      min_hole_size=64,
                      min_object_size=50,
                      closing_iterations=1,
                      fill_interior_class=None,
                      fill_interior_target=4,
                      union_hole_fill=False,
                      remove_small_objects_enabled=True,
                      union_gap_fill=False,
                      union_gap_closing_iterations=1,
                      fill_blue_table_quadrant_enabled=False,
                      blue_table_label=1,
                      blue_table_target=4,
                      blue_table_quadrant_mode="right_down",
                      blue_table_y_pad_top=60,
                      blue_table_y_pad_bottom=60,
                      blue_table_skip_if_label_above=None,
                      blue_table_skip_if_label_area_gt=None,
                      overwrite=False,
                      num_workers=1,
                      fill_bg_roi_list=None,
                      fill_table_top_line_enabled=False,
                      table_top_label=1,
                      table_top_fill_target=6,
                      table_top_corner_ranges=None,
                      scanline_fill_enabled=False,
                      scanline_source_label=1,
                      scanline_fill_value=3,
                      topleft_rect_enabled=False,
                      topleft_rect_label=1,
                      topleft_rect_fill=5,
                      topleft_rect_y_max=420,
                      topleft_rect_frame_start=10,
                      topleft_rect_frame_end_ratio=0.667,
                      topright_rect_enabled=False,
                      topright_rect_label=1,
                      topright_rect_fill=5,
                      topright_rect_y_max=420,
                      topright_rect_y_threshold=200,
                      topright_rect_frame_start_ratio=0.667,
                      leftmost_rect_enabled=False,
                      leftmost_rect_label=1,
                      leftmost_rect_fill=5,
                      leftmost_rect_y_max=420,
                      leftmost_rect_skip_label=3,
                      corner_rect_enabled=False,
                      corner_rect_mode="topright",
                      corner_rect_labels=None,
                      corner_rect_fill=4,
                      corner_rect_y_max=420,
                      corner_rect_x_first_labels=None,
                      temporal_fill_enabled=False,
                      temporal_fill_labels=None,
                      temporal_fill_union_labels=None,
                      temporal_fill_value=5):
    """
    Process all *_masks.npz files in a directory.
    Output: *_masks_post.npz
    
    Args:
        num_workers: Number of parallel workers. 1 = serial (default).
                     Set to > 1 for multiprocessing parallelism.
        fill_bg_roi_list: List of ROI fill specs (see parse_fill_bg_roi).
        table_top_corner_ranges: dict of ROI ranges for fill_table_top_line corners.
    """
    if table_top_corner_ranges is None:
        table_top_corner_ranges = {}
    import multiprocessing

    mask_files = sorted(input_dir.glob("*_masks.npz"))
    
    if not mask_files:
        print(f"No mask files found in {input_dir}")
        return
    
    print(f"Processing {len(mask_files)} mask files in {input_dir.name} (workers={num_workers})...")

    # Build argument tuples for each file
    task_args = [
        (str(mask_file), num_classes, fill_holes, min_hole_size, min_object_size,
         closing_iterations, fill_interior_class, fill_interior_target,
         union_hole_fill, remove_small_objects_enabled, union_gap_fill,
         union_gap_closing_iterations, fill_blue_table_quadrant_enabled,
         blue_table_label, blue_table_target, blue_table_quadrant_mode,
         blue_table_y_pad_top, blue_table_y_pad_bottom,
         blue_table_skip_if_label_above, blue_table_skip_if_label_area_gt,
         overwrite, fill_bg_roi_list,
         fill_table_top_line_enabled, table_top_label, table_top_fill_target,
         table_top_corner_ranges,
         scanline_fill_enabled, scanline_source_label, scanline_fill_value,
         topleft_rect_enabled, topleft_rect_label, topleft_rect_fill,
         topleft_rect_y_max, topleft_rect_frame_start, topleft_rect_frame_end_ratio,
         topright_rect_enabled, topright_rect_label, topright_rect_fill,
         topright_rect_y_max, topright_rect_y_threshold, topright_rect_frame_start_ratio,
         leftmost_rect_enabled, leftmost_rect_label, leftmost_rect_fill,
         leftmost_rect_y_max, leftmost_rect_skip_label,
         corner_rect_enabled, corner_rect_mode, corner_rect_labels,
         corner_rect_fill, corner_rect_y_max, corner_rect_x_first_labels,
         temporal_fill_enabled, temporal_fill_labels, temporal_fill_union_labels,
         temporal_fill_value)
        for mask_file in mask_files
    ]

    done_count = 0
    skipped_count = 0

    if num_workers <= 1:
        # Serial fallback
        for args_tuple in tqdm(task_args, desc=input_dir.name):
            name, status = _process_single_file(args_tuple)
            if status == "skipped":
                skipped_count += 1
            else:
                done_count += 1
    else:
        # Parallel processing
        with multiprocessing.Pool(processes=num_workers) as pool:
            for name, status in tqdm(
                pool.imap_unordered(_process_single_file, task_args),
                total=len(task_args),
                desc=input_dir.name,
            ):
                if status == "skipped":
                    skipped_count += 1
                else:
                    done_count += 1

    print(f"Done! {input_dir.name}: {done_count} processed, {skipped_count} skipped.")


def _build_episode_to_chunk_map(dataset_root: Path) -> dict[str, str]:
    data_root = dataset_root / "data"
    if not data_root.exists():
        return {}
    mapping: dict[str, str] = {}
    for chunk_dir in sorted(data_root.glob("chunk-*")):
        for parquet_file in chunk_dir.glob("episode_*.parquet"):
            episode_id = parquet_file.stem.replace("episode_", "")
            if episode_id in mapping and mapping[episode_id] != chunk_dir.name:
                raise ValueError(
                    f"Episode {episode_id} appears in multiple chunks: "
                    f"{mapping[episode_id]} and {chunk_dir.name}"
                )
            mapping[episode_id] = chunk_dir.name
    return mapping


def copy_postprocessed_masks(
    sam3_output_dir: Path,
    dataset_root: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Copy *_masks_post.npz from sam3_output into dataset_root/masks/<chunk>/<camera>/.
    Renames *_masks_post.npz -> *_masks.npz.
    """
    if not sam3_output_dir.exists():
        raise FileNotFoundError(f"Missing sam3_output: {sam3_output_dir}")
    masks_root = dataset_root / "masks"
    masks_root.mkdir(parents=True, exist_ok=True)

    episode_to_chunk = _build_episode_to_chunk_map(dataset_root)
    camera_dirs = [d for d in sam3_output_dir.iterdir() if d.is_dir() and "observation.images" in d.name]
    if not camera_dirs:
        raise ValueError(f"No camera directories found in {sam3_output_dir}")

    copied = 0
    skipped = 0
    missing_episode = 0
    pattern = re.compile(r"episode_(\d+)_masks_post\.npz$")

    for camera_dir in sorted(camera_dirs):
        for mask_file in sorted(camera_dir.glob("*_masks_post.npz")):
            match = pattern.match(mask_file.name)
            if not match:
                continue
            episode_id = match.group(1)
            chunk = episode_to_chunk.get(episode_id)
            if chunk is None and episode_to_chunk:
                missing_episode += 1
                continue
            if chunk is None:
                chunk = "chunk-000"
            dest_dir = masks_root / chunk / camera_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_name = mask_file.name.replace("_masks_post.npz", "_masks.npz")
            dest_path = dest_dir / dest_name
            if dest_path.exists() and not overwrite:
                skipped += 1
                continue
            if not dry_run:
                shutil.copy2(mask_file, dest_path)
            copied += 1

    print("=" * 50)
    print("Copy Postprocessed Masks")
    print("=" * 50)
    print(f"sam3_output: {sam3_output_dir}")
    print(f"dataset_root: {dataset_root}")
    print(f"dry_run: {dry_run}")
    print(f"copied: {copied}")
    print(f"skipped: {skipped}")
    print(f"missing_episode: {missing_episode}")
    print("=" * 50)


def copy_dataset_folders(
    src_root: Path,
    dest_root: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    folders = ["data", "meta", "videos"]
    print("=" * 50)
    print("Copy Dataset Folders")
    print("=" * 50)
    print(f"source_root: {src_root}")
    print(f"dest_root: {dest_root}")
    print(f"dry_run: {dry_run}")
    for folder in folders:
        src_dir = src_root / folder
        dest_dir = dest_root / folder
        if not src_dir.exists():
            print(f"skip (missing): {src_dir}")
            continue
        if dest_dir.exists() and not overwrite:
            print(f"skip (exists): {dest_dir}")
            continue
        if not dry_run:
            if dest_dir.exists() and overwrite:
                shutil.rmtree(dest_dir)
            shutil.copytree(src_dir, dest_dir, dirs_exist_ok=overwrite)
        print(f"copied: {folder}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='Postprocess segmentation masks')
    parser.add_argument('--input_dir', type=str, 
                        default='/localhome/local-vennw/code/orca_dataset/galbot_lerobot_dataset/task3_01210122_merged/sam3_output',
                        help='Input directory containing camera subdirectories')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of classes including background (default: 4)')
    parser.add_argument('--min_hole_size', type=int, default=0,
                        help='Fill holes smaller than this (pixels, default: 64)')
    parser.add_argument('--min_object_size', type=int, default=0,
                        help='Remove objects smaller than this (pixels, default: 50)')
    parser.add_argument('--closing_iterations', type=int, default=1,
                        help='Morphological closing iterations (default: 1)')
    parser.add_argument('--no_fill_holes', action='store_true',
                        help='Disable hole filling')
    parser.add_argument('--no_remove_small_objects', action='store_true',
                        help='Disable removing small objects')
    parser.add_argument('--union_hole_fill', action='store_true',
                        help='Fill holes based on union of all >0 classes')
    parser.add_argument('--union_gap_fill', action='store_true',
                        help='Fill thin background gaps between classes using union closing')
    parser.add_argument('--union_gap_closing_iterations', type=int, default=1,
                        help='Closing iterations for union gap fill (default: 1)')
    parser.add_argument('--fill_blue_table_quadrant', action='store_true',
                        help='Fill black region in blue table quadrant')
    parser.add_argument('--blue_table_label', type=int, default=1,
                        help='Label id for blue table (default: 1)')
    parser.add_argument('--blue_table_target', type=int, default=4,
                        help='Target label for blue table quadrant fill (default: 4)')
    parser.add_argument('--blue_table_quadrant_mode', type=str, default="right_down",
                        help='Quadrant fill mode: right_down or left_down')
    parser.add_argument('--blue_table_y_pad_top', type=int, default=60,
                        help='Top padding rows to exclude from quadrant fill')
    parser.add_argument('--blue_table_y_pad_bottom', type=int, default=60,
                        help='Bottom padding rows to exclude from quadrant fill')
    parser.add_argument('--blue_table_skip_if_label_above', type=int, default=None,
                        help='Skip quadrant fill if this label is higher than blue table')
    parser.add_argument('--blue_table_skip_if_label_area_gt', type=int, default=None,
                        help='Skip quadrant fill if skip label area exceeds this pixel count')
    parser.add_argument('--fill_interior_class', type=str, default=None,
                        help='Fill background inside these class contours. Comma-separated (e.g., "1,3" for red and blue). None to disable.')
    parser.add_argument('--fill_interior_target', type=int, default=4,
                        help='New class label for filled interior (default: 4)')
    parser.add_argument('--fill_table_top_line', action='store_true',
                        help='Fill background inside the closed region formed by the largest CC of table label and a line connecting its top-left and top-right corners.')
    parser.add_argument('--table_top_label', type=int, default=1,
                        help='Label for the table class used by fill_table_top_line (default: 1)')
    parser.add_argument('--table_top_fill_target', type=int, default=6,
                        help='Fill target for fill_table_top_line (default: 6)')
    parser.add_argument('--scanline_fill', action='store_true',
                        help='Row-based fill: fill background between first/last source_label pixels per row.')
    parser.add_argument('--scanline_source_label', type=int, default=1,
                        help='Label to scan for in scanline fill (default: 1).')
    parser.add_argument('--scanline_fill_value', type=int, default=3,
                        help='Value to fill background with in scanline fill (default: 3).')
    parser.add_argument('--topleft_rect', action='store_true',
                        help='Fill bg in rect left-below the top-left corner of a label (first 2/3).')
    parser.add_argument('--topright_rect', action='store_true',
                        help='Fill bg in rect left-below the top-right corner of a label (last 1/3).')
    parser.add_argument('--topleft_rect_label', type=int, default=1,
                        help='Source label to find top-left corner (default: 1).')
    parser.add_argument('--topleft_rect_fill', type=int, default=5,
                        help='Fill value for topleft rect (default: 5).')
    parser.add_argument('--topleft_rect_y_max', type=int, default=420,
                        help='Bottom boundary of fill rect (default: 420).')
    parser.add_argument('--topleft_rect_frame_start', type=int, default=10,
                        help='First frame to apply (default: 10).')
    parser.add_argument('--topleft_rect_frame_end_ratio', type=float, default=0.667,
                        help='Fraction of video to apply (default: 0.667).')
    parser.add_argument('--topright_rect_label', type=int, default=1)
    parser.add_argument('--topright_rect_fill', type=int, default=5)
    parser.add_argument('--topright_rect_y_max', type=int, default=420)
    parser.add_argument('--topright_rect_y_threshold', type=int, default=200,
                        help='Only consider label pixels with y < this (default: 200).')
    parser.add_argument('--topright_rect_frame_start_ratio', type=float, default=0.667,
                        help='Start ratio of video (default: 0.667 = last 1/3).')
    parser.add_argument('--corner_rect', action='store_true',
                        help='Fill bg rect from top-corner of label(s). '
                             'topright: fill left-below; topleft: fill right-below.')
    parser.add_argument('--corner_rect_mode', type=str, default="topright",
                        help='Corner mode: "topright" or "topleft" (default: topright).')
    parser.add_argument('--corner_rect_labels', type=str, default=None,
                        help='Comma-separated source labels, e.g. "1,3".')
    parser.add_argument('--corner_rect_fill', type=int, default=4,
                        help='Fill value for corner rect (default: 4).')
    parser.add_argument('--corner_rect_y_max', type=int, default=420,
                        help='Bottom boundary of fill rect (default: 420).')
    parser.add_argument('--corner_rect_x_first_labels', type=str, default=None,
                        help='Comma-separated labels that use x-first anchor (extremal x, then min y). '
                             'Other labels use y-first (min y, then extremal x). E.g. "3".')
    parser.add_argument('--temporal_fill', action='store_true',
                        help='Forward temporal fill: if pixel has source label in frame N and is 0 in frame N+1, fill with target.')
    parser.add_argument('--temporal_fill_labels', type=str, default=None,
                        help='Comma-separated labels for forward/backward propagation, e.g. "1,5".')
    parser.add_argument('--temporal_fill_union_labels', type=str, default=None,
                        help='Comma-separated labels for union fill (fill all frames where pixel ever had label), e.g. "3,4".')
    parser.add_argument('--temporal_fill_value', type=int, default=5,
                        help='Fill value for temporal fill (default: 5).')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing *_masks_post.npz files')
    parser.add_argument('--copy_to_dataset_root', type=str, default=None,
                        help='After postprocess, copy *_masks_post.npz to <dataset_root>/masks/')
    parser.add_argument('--copy_only', action='store_true',
                        help='Only copy postprocessed masks, skip postprocess')
    parser.add_argument('--copy_dataset_folders', action='store_true',
                        help='Copy data/meta/videos from dataset root (input_dir parent)')
    parser.add_argument('--source_dataset_root', type=str, default=None,
                        help='Explicit source dataset root for data/meta/videos')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print copy summary without copying files')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers for postprocessing (default: 8)')
    parser.add_argument('--fill_bg_roi', type=str, action='append', default=None,
                        help='Fill background in ROI. Format: "frame_start,frame_end_ratio,y_min,y_max,x_min,x_max,target". '
                             'Use -1 for full range. Can be specified multiple times.')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    # Parse fill_interior_class (comma-separated string to list of ints)
    fill_interior_classes = None
    if args.fill_interior_class is not None:
        fill_interior_classes = [int(x.strip()) for x in args.fill_interior_class.split(',')]

    # Parse fill_bg_roi specs
    fill_bg_roi_list = None
    if args.fill_bg_roi:
        fill_bg_roi_list = [parse_fill_bg_roi(s) for s in args.fill_bg_roi]
    
    if not args.copy_only:
        print("="*50)
        print("Mask Postprocessing")
        print("="*50)
        print(f"Input: {input_dir}")
        print(f"Parameters:")
        print(f"  - fill_holes: {not args.no_fill_holes}")
        print(f"  - min_hole_size: {args.min_hole_size}")
        print(f"  - min_object_size: {args.min_object_size}")
        print(f"  - closing_iterations: {args.closing_iterations}")
        print(f"  - remove_small_objects: {not args.no_remove_small_objects}")
        print(f"  - union_hole_fill: {args.union_hole_fill}")
        print(f"  - union_gap_fill: {args.union_gap_fill}")
        print(f"  - union_gap_closing_iterations: {args.union_gap_closing_iterations}")
        print(f"  - fill_blue_table_quadrant: {args.fill_blue_table_quadrant}")
        if args.fill_blue_table_quadrant:
            print(f"  - blue_table_label: {args.blue_table_label}")
            print(f"  - blue_table_target: {args.blue_table_target}")
            print(f"  - blue_table_quadrant_mode: {args.blue_table_quadrant_mode}")
            print(f"  - blue_table_y_pad_top: {args.blue_table_y_pad_top}")
            print(f"  - blue_table_y_pad_bottom: {args.blue_table_y_pad_bottom}")
            print(f"  - blue_table_skip_if_label_above: {args.blue_table_skip_if_label_above}")
            print(f"  - blue_table_skip_if_label_area_gt: {args.blue_table_skip_if_label_area_gt}")
        if fill_interior_classes is not None:
            class_names = {1: 'red', 2: 'green', 3: 'blue'}
            class_str = ', '.join([f"{c}({class_names.get(c, '?')})" for c in fill_interior_classes])
            print(f"  - fill_interior: classes [{class_str}] interior -> {args.fill_interior_target}")
        print(f"  - fill_table_top_line: {args.fill_table_top_line}")
        if args.fill_table_top_line:
            print(f"  - table_top_label: {args.table_top_label}")
            print(f"  - table_top_fill_target: {args.table_top_fill_target}")
        print("="*50)

        # Find all camera subdirectories
        camera_dirs = [d for d in input_dir.iterdir() if d.is_dir() and 'observation.images' in d.name]

        if not camera_dirs:
            print(f"No camera directories found in {input_dir}")
            return

        print(f"Found {len(camera_dirs)} camera directories")

        for camera_dir in camera_dirs:
            process_directory(
                camera_dir,
                num_classes=args.num_classes,
                fill_holes=not args.no_fill_holes,
                min_hole_size=args.min_hole_size,
                min_object_size=args.min_object_size,
                closing_iterations=args.closing_iterations,
                fill_interior_class=fill_interior_classes,
                fill_interior_target=args.fill_interior_target,
                union_hole_fill=args.union_hole_fill,
                remove_small_objects_enabled=not args.no_remove_small_objects,
                union_gap_fill=args.union_gap_fill,
                union_gap_closing_iterations=args.union_gap_closing_iterations,
                fill_blue_table_quadrant_enabled=args.fill_blue_table_quadrant,
                blue_table_label=args.blue_table_label,
                blue_table_target=args.blue_table_target,
                blue_table_quadrant_mode=args.blue_table_quadrant_mode,
                blue_table_y_pad_top=args.blue_table_y_pad_top,
                blue_table_y_pad_bottom=args.blue_table_y_pad_bottom,
                blue_table_skip_if_label_above=args.blue_table_skip_if_label_above,
                blue_table_skip_if_label_area_gt=args.blue_table_skip_if_label_area_gt,
                overwrite=args.overwrite,
                num_workers=args.num_workers,
                fill_bg_roi_list=fill_bg_roi_list,
                fill_table_top_line_enabled=args.fill_table_top_line,
                table_top_label=args.table_top_label,
                table_top_fill_target=args.table_top_fill_target,
                scanline_fill_enabled=args.scanline_fill,
                scanline_source_label=args.scanline_source_label,
                scanline_fill_value=args.scanline_fill_value,
                topleft_rect_enabled=args.topleft_rect,
                topleft_rect_label=args.topleft_rect_label,
                topleft_rect_fill=args.topleft_rect_fill,
                topleft_rect_y_max=args.topleft_rect_y_max,
                topleft_rect_frame_start=args.topleft_rect_frame_start,
                topleft_rect_frame_end_ratio=args.topleft_rect_frame_end_ratio,
                topright_rect_enabled=args.topright_rect,
                topright_rect_label=args.topright_rect_label,
                topright_rect_fill=args.topright_rect_fill,
                topright_rect_y_max=args.topright_rect_y_max,
                topright_rect_y_threshold=args.topright_rect_y_threshold,
                topright_rect_frame_start_ratio=args.topright_rect_frame_start_ratio,
                corner_rect_enabled=args.corner_rect,
                corner_rect_mode=args.corner_rect_mode,
                corner_rect_labels=[int(x) for x in args.corner_rect_labels.split(",")] if args.corner_rect_labels else None,
                corner_rect_fill=args.corner_rect_fill,
                corner_rect_y_max=args.corner_rect_y_max,
                corner_rect_x_first_labels=[int(x) for x in args.corner_rect_x_first_labels.split(",")] if args.corner_rect_x_first_labels else None,
                temporal_fill_enabled=args.temporal_fill,
                temporal_fill_labels=[int(x) for x in args.temporal_fill_labels.split(",")] if args.temporal_fill_labels else None,
                temporal_fill_union_labels=[int(x) for x in args.temporal_fill_union_labels.split(",")] if args.temporal_fill_union_labels else None,
                temporal_fill_value=args.temporal_fill_value,
            )

        print("\nAll done!")

    if args.copy_to_dataset_root:
        copy_postprocessed_masks(
            sam3_output_dir=input_dir,
            dataset_root=Path(args.copy_to_dataset_root),
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        if args.copy_dataset_folders:
            source_root = Path(args.source_dataset_root) if args.source_dataset_root else input_dir.parent
            copy_dataset_folders(
                src_root=source_root,
                dest_root=Path(args.copy_to_dataset_root),
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()

