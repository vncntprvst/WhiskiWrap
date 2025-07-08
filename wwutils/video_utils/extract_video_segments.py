"""
This script extracts specified segments from a video file and optionally concatenates them into a single output file.
Segments can be specified directly as pairs of frame indices or read from a file.

Usage:
    python extract_video_segments.py <video_path> --segments <start1> <end1> [<start2> <end2> ...]
    python extract_video_segments.py <video_path> --segments-file <file_path>
    python extract_video_segments.py <video_path> --segments-file <file_path> --no-concat
"""
import argparse
from pathlib import Path
import cv2
import os


def parse_segments_from_file(file_path):
    segments = []
    with open(file_path, 'r') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Line {line_no} in {file_path} must have two integers: 'start end'")
            segments.append((int(parts[0]), int(parts[1])))
    return segments


def extract_segments(video_path, segments, output_dir, concat=True):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    outputs = []

    if concat:
        combined_name = "_".join(f"{s}_{e}" for s, e in segments)
        output_path = output_dir / f"{video_path.stem}_{combined_name}.mp4"
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for i, (start, end) in enumerate(segments):
        if not concat:
            output_path = output_dir / f"{video_path.stem}_seg{i+1}_{start}_{end}.mp4"
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for fid in range(start, end):
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {fid} from {video_path}")
                break
            writer.write(frame)

        if not concat:
            writer.release()
            outputs.append(output_path)

    cap.release()
    if concat:
        writer.release()
        outputs.append(output_path)

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Extract (and optionally concatenate) video segments by frame range.")
    parser.add_argument("video", help="Path to input video file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--segments", nargs="+", type=int, help="List of start/end frame pairs")
    group.add_argument("--segments-file", type=str, help="File with 'start end' pairs per line")
    parser.add_argument("--output-dir", default="extracted_segments", help="Directory to save extracted clips")
    parser.add_argument("--no-concat", action="store_true", help="Do not concatenate; output individual clips")
    args = parser.parse_args()

    if args.segments:
        if len(args.segments) % 2 != 0:
            parser.error("Provide an even number of frames (start/end pairs) with --segments")
        segments = [(args.segments[i], args.segments[i + 1]) for i in range(0, len(args.segments), 2)]
    else:
        segments = parse_segments_from_file(args.segments_file)

    outputs = extract_segments(args.video, segments, args.output_dir, concat=not args.no_concat)
    print("\nExtracted segments:")
    for out in outputs:
        print(f"  → {out}")


if __name__ == "__main__":
    main()
