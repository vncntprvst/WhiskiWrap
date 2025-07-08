"""Utility to track whiskers on short portions of a video.

This script extracts one or more ranges of frames from an input video, saves them as new
clips and runs the standard tracking pipeline on those clips. The pipeline mirrors
the steps found in
``scripts/whisker_tracking/whisker_tracking_container_dynamic_directives.sh``:

``trace_measure`` → ``combine_to_file`` → ``reclassify``.

Examples
--------

Single segment:
```bash
python -m whisker_tracking.python.utils.extract_segment_and_track \
    input.mp4 100 200 --output-dir output --nproc 20
```

Multiple segments:
```bash
python -m whisker_tracking.python.utils.extract_segment_and_track \
    input.mp4 --segments 100 200 300 400 500 600 --output-dir output --nproc 20
```

Multiple segments from file:
```bash
python -m whisker_tracking.python.utils.extract_segment_and_track \
    input.mp4 --segments-file segments.txt --output-dir output --nproc 20
```

Where segments.txt contains:
```
100 200
300 400
500 600
```

After running, ``output`` will contain the tracked parquet files that can be
opened in the labeling application for refining training data.
"""

import argparse
import os
import sys
from pathlib import Path
import cv2

# Add the parent directory to sys.path to allow imports
script_dir = Path(__file__).parent
python_dir = script_dir.parent
sys.path.insert(0, str(python_dir))

from whisker_tracking_pipeline import trace_measure
from wwutils.data_manip import combine_sides as cs
from wwutils.classifiers import reclassify as rc


def extract_video_segment(video_path, start_frame, end_frame, output_path):
    """Extract frames [start_frame, end_frame) from video_path into new video using ffmpeg."""
    import subprocess
    
    # Get video properties first
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Calculate time range from frame numbers
    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps
    
    # Use ffmpeg to extract the segment
    cmd = [
        'ffmpeg', '-y',  # -y to overwrite output file
        '-ss', str(start_time),  # start time in seconds
        '-i', str(video_path),   # input file
        '-t', str(duration),     # duration in seconds
        '-c', 'copy',            # copy streams without re-encoding (faster)
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr}")
    
    # Verify the output file was created
    if not Path(output_path).exists():
        raise RuntimeError(f"Output file was not created: {output_path}")

def extract_video_segments(video_path, segments, output_dir, concat=True):
    """Extract one or more segments from *video_path*.

    Parameters
    ----------
    video_path : Path-like
        Path to the source video.
    segments : list of tuple
        ``(start_frame, end_frame)`` pairs defining the frames to keep.
    output_dir : Path
        Directory where extracted segments are written.
    concat : bool, optional
        If True, concatenate all segments into a single clip and return its path.

    Returns
    -------
    list[Path]
        Paths to the extracted video files.  When *concat* is True a single path
        is returned.
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    output_paths = []
    if concat:
        combined_name = "_".join(f"{s}_{e}" for s, e in segments)
        out_path = Path(output_dir) / f"{Path(video_path).stem}_{combined_name}.mp4"
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        for start, end in segments:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for fid in range(start, end):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
        writer.release()
        output_paths.append(out_path)
    else:
        for start, end in segments:
            out_path = Path(output_dir) / f"{Path(video_path).stem}_{start}_{end}.mp4"
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for fid in range(start, end):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
            writer.release()
            output_paths.append(out_path)

    cap.release()
    return output_paths

def run_tracking(segment_path, base_name, output_dir, nproc, original_whiskerpad_file=None):
    """Run trace_measure, combine_to_file and reclassify on the segment."""
    log_file_path = Path(output_dir) / f"track_{base_name}.log"
    
    # If we have an original whiskerpad file, copy it to the segment directory
    if original_whiskerpad_file and os.path.exists(original_whiskerpad_file):
        segment_whiskerpad_file = Path(output_dir) / f"whiskerpad_{base_name}.json"
        import shutil
        shutil.copy2(original_whiskerpad_file, segment_whiskerpad_file)
        print(f"  Copied whiskerpad file: {original_whiskerpad_file} -> {segment_whiskerpad_file}")
    
    with open(log_file_path, "w") as log_file:
        output_fns, whiskerpad_file = trace_measure(
            str(segment_path), base_name, str(output_dir), nproc, True, log_file
        )
        combined = cs.combine_to_file(output_fns, whiskerpad_file)
        rc.reclassify(combined, whiskerpad_file)
    return combined


def main():
    p = argparse.ArgumentParser(
        description="Extract video segment(s) and run whisker tracking on them"
    )
    p.add_argument("video", help="Input video path")
    
    # Create mutually exclusive group for segment specification
    segment_group = p.add_mutually_exclusive_group(required=True)
    segment_group.add_argument("start_frame", nargs="?", type=int, help="Start frame (for single segment)")
    segment_group.add_argument("--segments", nargs="+", type=int, metavar="FRAME",
                              help="Multiple segments as: start1 end1 start2 end2 ... (pairs of frames)")
    segment_group.add_argument("--segments-file", type=str, metavar="FILE",
                              help="File containing segment pairs, one per line: 'start end'")
    
    p.add_argument("end_frame", nargs="?", type=int, help="End frame (exclusive, for single segment)")
    p.add_argument("--output-dir", default="segment_output", help="Output directory")
    p.add_argument("--nproc", type=int, default=40, help="Number of trace processes")
    p.add_argument(
        "--no-concat",
        action="store_true",
        help="Do not concatenate segments; track each individually",
    )
    args = p.parse_args()

    # Parse segments
    segments = []
    
    if args.start_frame is not None:
        # Single segment mode (backward compatibility)
        if args.end_frame is None:
            p.error("end_frame is required when using positional arguments")
        segments = [(args.start_frame, args.end_frame)]
    elif args.segments:
        # Multiple segments from command line
        if len(args.segments) % 2 != 0:
            p.error("--segments requires an even number of arguments (start-end pairs)")
        segments = [(args.segments[i], args.segments[i+1]) for i in range(0, len(args.segments), 2)]
    elif args.segments_file:
        # Multiple segments from file
        with open(args.segments_file, 'r') as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue  # Skip empty lines and comments
                parts = line.split()
                if len(parts) != 2:
                    p.error(f"Invalid format in {args.segments_file} line {line_no}: expected 'start end'")
                try:
                    start, end = int(parts[0]), int(parts[1])
                    segments.append((start, end))
                except ValueError:
                    p.error(f"Invalid numbers in {args.segments_file} line {line_no}: {line}")

    os.makedirs(args.output_dir, exist_ok=True)
    
    # print(f"Processing {len(segments)} segment(s) from {args.video}")
    
    # Create whiskerpad parameters once for the original video
    original_base_name = Path(args.video).stem
    original_whiskerpad_file = Path(args.output_dir) / f"whiskerpad_{original_base_name}.json"
    
    if not original_whiskerpad_file.exists():
        print(f"Creating whiskerpad parameters for original video: {args.video}")
        print(f"Whiskerpad file will be saved as: {original_whiskerpad_file}")
        
        # Import whiskerpad here to avoid import issues
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import whiskerpad as wp
        import shutil
        
        # Create whiskerpad parameters for the original video
        whiskerpad = wp.Params(args.video, True, original_base_name)  # splitUp=True by default
        whiskerpadParams, splitUp = wp.WhiskerPad.get_whiskerpad_params(whiskerpad)
        wp.WhiskerPad.save_whiskerpad_params(whiskerpad, whiskerpadParams)
        
        # The save_whiskerpad_params saves to the video directory, so move it to output directory
        source_file = Path(args.video).parent / f"whiskerpad_{original_base_name}.json"
        if source_file.exists():
            shutil.move(str(source_file), str(original_whiskerpad_file))
            print(f"✅ Whiskerpad file moved to: {original_whiskerpad_file}")
        else:
            print(f"❌ Warning: Expected whiskerpad file not found at: {source_file}")
        
        print(f"✅ Whiskerpad parameters created: {original_whiskerpad_file}")
    else:
        print(f"✅ Using existing whiskerpad file: {original_whiskerpad_file}")
    
    # all_results = []
    
    # for i, (start_frame, end_frame) in enumerate(segments, 1):
    #     print(f"\n=== Processing segment {i}/{len(segments)}: frames {start_frame}-{end_frame} ===")
        
    #     base_name = Path(args.video).stem + f"_seg{i:02d}_{start_frame}_{end_frame}"
    #     segment_path = Path(args.output_dir) / f"{base_name}.mp4"
        
    #     print(f"Extracting frames {start_frame}-{end_frame} to {segment_path}")
    #     extract_video_segment(args.video, start_frame, end_frame, segment_path)
        
    #     print(f"Running tracking on {segment_path}")
    #     result_file = run_tracking(segment_path, base_name, args.output_dir, args.nproc, str(original_whiskerpad_file))
    #     all_results.append(result_file)
        
    #     print(f"Segment {i} completed. Results: {result_file}")
    
    # print(f"\n=== All segments completed ===")
    # print(f"Output directory: {args.output_dir}")
    # print("Tracking results:")
    # for i, result in enumerate(all_results, 1):
    #     print(f"  Segment {i}: {result}")


    segment_paths = extract_video_segments(
        args.video, segments, args.output_dir, concat=not args.no_concat
    )

    all_results = []
    for seg_path in segment_paths:
        base = Path(seg_path).stem
        print(f"Running tracking on {seg_path}")
        result_file = run_tracking(seg_path, base, args.output_dir, args.nproc, str(original_whiskerpad_file))
        all_results.append(result_file)
        print(f"Completed tracking: {result_file}")

    print(f"\n=== All segments completed ===")
    print(f"Output directory: {args.output_dir}")
    print("Tracking results:")
    for i, result in enumerate(all_results, 1):
        print(f"  Segment {i}: {result}")

if __name__ == "__main__":
    main()
