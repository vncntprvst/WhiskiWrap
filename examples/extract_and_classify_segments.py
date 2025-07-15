#!/usr/bin/env python3
"""Extract video segments and classify whiskers.

This utility uses :mod:`wwutils.video_utils.extract_video_segments` to grab one or
more frame ranges from a video. The extracted segments are concatenated into a
new clip which is processed with the first two steps of the main pipeline
(`trace` and `combine`). Finally a whisker ID classifier (U-Net or GNN) is
applied to the resulting parquet file.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import pandas as pd

from wwutils.video_utils.extract_video_segments import extract_segments
from whisker_tracking_pipeline import trace_measure
from wwutils.data_manip import combine_sides as cs
from wwutils.classifiers.unet_classifier import assign_whisker_ids
from wwutils.classifiers.gnn_classifier import reassign_whisker_ids_gnn


def run_pipeline(video: str, output_dir: str, nproc: int) -> str:
    """Run tracing and combining on *video* and return the combined parquet."""
    base = Path(video).stem
    log_path = Path(output_dir) / f"{base}_pipeline.log"
    with open(log_path, "w") as log:
        out_files, wp_file = trace_measure(video, base, output_dir, nproc, False, log)
        combined = cs.combine_to_file(out_files, wp_file)
    return combined


def parse_segments(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.segments:
        if len(args.segments) % 2 != 0:
            raise SystemExit("--segments requires start/end frame pairs")
        return [
            (args.segments[i], args.segments[i + 1])
            for i in range(0, len(args.segments), 2)
        ]
    segments: list[tuple[int, int]] = []
    with open(args.segments_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            start, end = map(int, line.split())
            segments.append((start, end))
    return segments


def main() -> None:
    p = argparse.ArgumentParser(description="Extract segments and classify whiskers")
    p.add_argument("video", help="Input video file")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--segments", nargs="+", type=int, help="start1 end1 [start2 end2 ...]")
    g.add_argument("--segments-file", help="Text file with start/end pairs")
    p.add_argument("--output-dir", default="segment_classify", help="Output directory")
    p.add_argument("--classifier", choices=["unet", "gnn"], default="gnn", help="Classifier to run")
    p.add_argument("--model", help="Path to pretrained model")
    p.add_argument("--nproc", type=int, default=4, help="Number of trace processes")
    args = p.parse_args()

    segments = parse_segments(args)
    os.makedirs(args.output_dir, exist_ok=True)

    composite = extract_segments(args.video, segments, args.output_dir, concat=True)[0]
    print(f"Composite video created: {composite}")

    combined = run_pipeline(str(composite), args.output_dir, args.nproc)
    print(f"Tracking results saved to: {combined}")

    if args.classifier == "unet":
        if not args.model:
            raise SystemExit("U-Net classifier requires --model")
        out = assign_whisker_ids(str(composite), combined, args.model)
    else:
        df = pd.read_parquet(combined)
        df_new, _ = reassign_whisker_ids_gnn(df, model_path=args.model)
        out = combined.replace(".parquet", "_gnn.parquet")
        df_new.to_parquet(out, index=False)
    print(f"Classifier output: {out}")


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
