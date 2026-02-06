#!/usr/bin/env python3
"""
Combine compare.mp4 (orig|mask) with post compare (orig|post-mask).
Output: orig | mask | post-mask (3 panels).
"""

import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="Combine compare and post videos.")
    parser.add_argument("--compare_video", required=True, help="Path to *_compare.mp4")
    parser.add_argument("--post_video", required=True, help="Path to *_compare_post.mp4")
    parser.add_argument("--output_path", required=True, help="Output video path")
    parser.add_argument("--fps", type=int, default=None, help="Override fps")
    args = parser.parse_args()

    compare_cap = cv2.VideoCapture(args.compare_video)
    post_cap = cv2.VideoCapture(args.post_video)

    if not compare_cap.isOpened():
        raise RuntimeError(f"Cannot open compare video: {args.compare_video}")
    if not post_cap.isOpened():
        raise RuntimeError(f"Cannot open post video: {args.post_video}")

    compare_fps = compare_cap.get(cv2.CAP_PROP_FPS) or 30
    fps = args.fps or compare_fps

    ret_c, frame_c = compare_cap.read()
    ret_p, frame_p = post_cap.read()
    if not ret_c or not ret_p:
        raise RuntimeError("Failed to read first frame from inputs.")

    h, w2 = frame_c.shape[:2]
    if frame_p.shape[0] != h or frame_p.shape[1] != w2:
        raise ValueError("Input videos must have the same frame size.")
    if w2 % 2 != 0:
        raise ValueError("Input videos must have even width (orig|mask).")
    w = w2 // 2

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w * 3, h))

    def process_frame(frame_compare, frame_post):
        orig = frame_compare[:, :w]
        mask = frame_compare[:, w:]
        post_mask = frame_post[:, w:]
        return cv2.hconcat([orig, mask, post_mask])

    writer.write(process_frame(frame_c, frame_p))

    while True:
        ret_c, frame_c = compare_cap.read()
        ret_p, frame_p = post_cap.read()
        if not ret_c or not ret_p:
            break
        writer.write(process_frame(frame_c, frame_p))

    compare_cap.release()
    post_cap.release()
    writer.release()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
