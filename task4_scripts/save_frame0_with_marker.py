"""Extract a specific frame from all videos in a camera dir and draw marker points."""

import argparse
import glob
import os

import cv2

LABEL_COLORS_BGR = {
    0: (0, 0, 255),
    1: (255, 0, 0),
    2: (0, 200, 0),
    3: (0, 165, 255),
    4: (255, 0, 255),
    5: (255, 255, 0),
    6: (0, 255, 255),
    7: (128, 0, 128),
}


def parse_points(points_str):
    """Parse points string. Supports two formats:
    - Simple: "x1,y1;x2,y2" (all label 0)
    - With labels: "1:x1,y1;2:x2,y2;2:x3,y3"
    """
    results = []
    for pt in points_str.split(";"):
        pt = pt.strip()
        if not pt:
            continue
        if ":" in pt.split(",")[0]:
            label_str, coords = pt.split(":", 1)
            label = int(label_str)
        else:
            label = 0
            coords = pt
        x, y = coords.split(",")
        results.append((label, float(x), float(y)))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True, help="Video base directory (chunk-000)")
    parser.add_argument(
        "--camera",
        default="observation.images.right_arm_camera_color_optical_frame",
    )
    parser.add_argument("--output_dir", required=True, help="Where to save annotated frame images")
    parser.add_argument(
        "--points",
        type=str,
        default="35.5,224.2",
        help='Points: "x1,y1;x2,y2" or with labels "1:x1,y1;2:x2,y2"',
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to extract. Use -1 for last frame.",
    )
    parser.add_argument("--radius", type=int, default=4)
    args = parser.parse_args()

    points = parse_points(args.points)

    video_dir = os.path.join(args.base_dir, args.camera)
    videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not videos:
        print(f"No .mp4 files found in {video_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Found {len(videos)} videos in {video_dir}")

    frame_idx = args.frame
    tag = f"frame{frame_idx}" if frame_idx >= 0 else "last_frame"

    for vpath in videos:
        ep_name = os.path.splitext(os.path.basename(vpath))[0]
        cap = cv2.VideoCapture(vpath)
        if frame_idx < 0:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            target = total + frame_idx
        else:
            target = frame_idx
        if target > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"  SKIP {ep_name}: cannot read frame {target}")
            continue

        for label, px, py in points:
            ix, iy = int(round(px)), int(round(py))
            color = LABEL_COLORS_BGR.get(label, (0, 0, 255))
            cv2.circle(frame, (ix, iy), args.radius, color, -1)
            cv2.circle(frame, (ix, iy), args.radius + 1, (255, 255, 255), 1)

        out_path = os.path.join(args.output_dir, f"{ep_name}_{tag}.png")
        cv2.imwrite(out_path, frame)

    print(f"Done. Saved {len(videos)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
