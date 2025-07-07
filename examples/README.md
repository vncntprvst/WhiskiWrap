# WhiskiWrap Examples

This folder contains examples demonstrating how to use WhiskiWrap for whisker tracking.

## Installation

```bash
# Install WhiskiWrap and its dependencies
pip install -e .

# Or if using uv
uv pip install -e .
```

## Command-line Usage

The `trace_whiskers.py` script is a command-line tool for tracking whiskers in videos
using the WhiskiWrap library.

Examples:

### Process a sample video from the test_videos directory

```bash
python trace_whiskers.py -v test_videos/test_video_10s.mp4
```

### Process with custom settings

```bash
python trace_whiskers.py -v test_videos/test_video_10s.mp4 -p 8 -c 200
```

### Use sensitive detection for faint whiskers

```bash
python trace_whiskers.py -v test_videos/test_video_10s.mp4 -s
```

### Use a file dialog to select the video file

```bash
python trace_whiskers.py -u
```

## Programmatic Usage

See `run_whisker_tracking.py` for an example of how to use WhiskiWrap programmatically
in your own Python code.
