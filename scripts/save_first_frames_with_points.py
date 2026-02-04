import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np


def parse_points(points_str: str) -> List[Tuple[float, float]]:
    if not points_str:
        return []
    points = []
    for item in points_str.split(";"):
        item = item.strip()
        if not item:
            continue
        x_str, y_str = item.split(",")
        points.append((float(x_str), float(y_str)))
    return points


def parse_labels(labels_str: str, num_points: int) -> List[int]:
    if not labels_str:
        return [1] * num_points
    labels = [int(x.strip()) for x in labels_str.split(",") if x.strip()]
    if len(labels) != num_points:
        raise ValueError(
            f"labels length {len(labels)} does not match points length {num_points}"
        )
    return labels


def draw_points_bgr(
    image_bgr, points: List[Tuple[float, float]], labels: List[int]
):
    for (x, y), label in zip(points, labels):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)  # green/red in BGR
        center = (int(round(x)), int(round(y)))
        cv2.circle(image_bgr, center, 6, color, 2, lineType=cv2.LINE_AA)
        cv2.drawMarker(
            image_bgr,
            center,
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=2,
            line_type=cv2.LINE_AA,
        )


def save_first_frame_with_points(
    video_path: str,
    output_dir: str,
    points: List[Tuple[float, float]],
    labels: List[int],
    save_raw: bool,
    frame_idx: int,
):
    cap = cv2.VideoCapture(video_path)
    if frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        print(f"Warning: failed to read frame {frame_idx} from {video_path}")
        return

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    if save_raw:
        raw_path = os.path.join(output_dir, f"{base_name}_frame{frame_idx}.png")
        cv2.imwrite(raw_path, frame_bgr)

    annotated = frame_bgr.copy()
    if points:
        draw_points_bgr(annotated, points, labels)

    annotated_path = os.path.join(
        output_dir, f"{base_name}_frame{frame_idx}_points.png"
    )
    cv2.imwrite(annotated_path, annotated)


def main():
    parser = argparse.ArgumentParser(
        description="Save the first frame from each video with point annotations."
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Directory containing video files (e.g., .mp4).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for saved frames (default: <video-dir>/first_frames).",
    )
    parser.add_argument(
        "--points",
        default="300,214;349,274;287.3,276.6;347.4,294.5",
        help="Points as 'x1,y1;x2,y2;...'.",
    )
    parser.add_argument(
        "--labels",
        default="1,1,1,1",
        help="Labels as '1,0,1,...' (1=positive, 0=negative).",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Also save the raw first frame without annotations.",
    )
    parser.add_argument(
        "--frame-idx",
        type=int,
        default=0,
        help="Frame index to extract (0-based).",
    )
    parser.add_argument(
        "--random-frame",
        action="store_true",
        help="Randomly pick a frame index per video.",
    )
    args = parser.parse_args()

    video_dir = args.video_dir
    output_dir = args.output_dir or os.path.join(video_dir, "first_frames")
    os.makedirs(output_dir, exist_ok=True)

    points = parse_points(args.points)
    labels = parse_labels(args.labels, len(points))

    video_files = [
        f
        for f in os.listdir(video_dir)
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    ]
    video_files.sort()

    if not video_files:
        raise RuntimeError(f"No video files found in {video_dir}")

    for fname in video_files:
        video_path = os.path.join(video_dir, fname)
        frame_idx = args.frame_idx
        if args.random_frame:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total_frames > 0:
                frame_idx = int(np.random.randint(0, total_frames))
            else:
                frame_idx = 0
        save_first_frame_with_points(
            video_path=video_path,
            output_dir=output_dir,
            points=points,
            labels=labels,
            save_raw=args.save_raw,
            frame_idx=frame_idx,
        )

    print(f"Saved {len(video_files)} frames to {output_dir}")


if __name__ == "__main__":
    main()
