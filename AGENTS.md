# WhiskiWrap - Agent Guide

This guide provides essential information for AI agents working with the WhiskiWrap project.

## Environment Setup

The project uses a virtual environment for dependency management. To activate it:

```bash
source .venv/bin/activate
```

## Main Scripts

The project contains two primary scripts for whisker tracking: `trace_whiskers.py`, and `whisker_tracking_pipeline.py`.
The first script is a basic interface for whisker tracking operations, while the second orchestrates the complete workflow.
Both scripts are designed to work with video files as input. The main features of these scripts include:

- Handles video format conversion as needed via ffmpeg
- Splits videos into epochs and chunks
- Runs parallel tracking processes
- Combines results into Parquet or HDF5 files

## Test Files

The project includes test files for validation:

### test_videos/

- `test_video_165s.mp4` - 165-second test video
- `test_video_325s.mp4` - 325-second test video
- `test_video_10s.mp4` - 10-second test video
- `test_video_50s.mp4` - 50-second test video
- `test_video2.mp4` - Alternative test video
- `test_bilateral_view.mp4` - Bilateral view test video

### tests/

- `tests/test_cli_scripts.py` - CLI script tests
- `tests/tests.py` - Main test suite containing benchmark functions
- `tests/__init__.py` - Package initialization for proper imports

Key test function:

- `run_standard_benchmarks()` - Runs comprehensive performance benchmarks

## Usage Example

```python
import WhiskiWrap

# Basic usage
input_video = 'test_videos/test_video_165s.mp4'
output_file = 'output.hdf5'
WhiskiWrap.pipeline_trace(input_video, output_file, n_trace_processes=4)

# Running tests
from tests.tests import run_standard_benchmarks
test_results, test_df = run_standard_benchmarks(force=True)
```

## Key Dependencies

- `ffmpeg` - Video processing
- `whisk-janelia` - Whisker tracking algorithms
- `pyarrow` - Parquet file handling
- `pytables` - HDF5 file handling
- `tifffile` - TIFF stack creation
- `pandas` - Data manipulation
- `numpy` - Numerical operations

## Project Structure

- Main modules: `WhiskiWrap/` and `wwutils/`
- Configuration files: `*.parameters`, `*.detectorbank`
- Example notebooks: `notebooks/`
- Test data: `test_videos/`
- Build configuration: `pyproject.toml`, `setup.py`

## Important Notes

- The project requires both Python dependencies and external binaries (ffmpeg, whisk)
- Test files create temporary directories in `tests/whiski_wrap_test/`
- Parquet and HDF5 output files can be large and should be handled carefully
- Parallel processing is optimized for multi-core systems
