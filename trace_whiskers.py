"""
trace_whiskers.py - A command-line tool for automated whisker tracking using WhiskiWrap

This script processes video files to track whiskers using the WhiskiWrap library,
which is a Python wrapper around the Janelia Whisker Tracking software.

Usage:
    python trace_whiskers.py -v path/to/video.mp4
    python trace_whiskers.py -u  # Opens a file selection dialog
    
Example:
    # Process a sample video (output in same directory as video)
    python trace_whiskers.py -v test_videos/test_video_10s.mp4
    
    # Process with custom settings and HDF5 format
    python trace_whiskers.py -v test_videos/test_video_10s.mp4 -f hdf5 -p 8 -c 200
    
    # Process with custom output directory
    python trace_whiskers.py -v test_videos/test_video_10s.mp4 -o /path/to/output
    
    # Process with Parquet format (default) in specific directory, copying video
    python trace_whiskers.py -v test_videos/test_video_10s.mp4 -f parquet -o ./results --copy-video

The script will:
1. Create a directory named 'whisker_tracking_[videoname]' in the input video's directory (or specified output directory)
2. Process the video with WhiskiWrap's interleaved_reading_and_tracing function
3. Generate a Parquet file (default) or HDF5 file with the tracked whisker data

Parameters:
    -v, --video_path : Path to the video file to process
    -u, --select-folder-ui : Use a file dialog to select the video file
    -o, --output-dir : Output directory (default: same directory as input video)
    -f, --format : Output format - 'parquet' (default) or 'hdf5'
    -p, --processes : Number of trace processes to run in parallel (default: 4)
    -c, --chunk-size : Number of frames per chunk (default: 100)
                       Reduce to 50 or 25 if you encounter "Out of memory" errors
    -s, --sensitive : Use sensitive detection parameters (default: False)
    --copy-video : Copy the input video to the output directory (default: False)

Output:
    The script creates a directory structure:
    whisker_tracking_[videoname]/
    ├── [videoname].mp4     # Copy of input video (only if --copy-video is used)
    ├── [videoname].parquet # Parquet file with whisker tracking data (default)
    ├── [videoname].hdf5    # HDF5 file with whisker tracking data (if --format hdf5)
    └── tracking_files/     # Temporary directory for TIFF image chunks

After processing, the HDF5 file can be analyzed with the provided notebooks:
- notebooks/plot_angle_trace.ipynb
- notebooks/plot_overlay_2sides.ipynb

Requirements:
    - WhiskiWrap and its dependencies must be installed
    - FFmpeg must be installed on the system
    
Updated by: Vincent Prevosto (vncntprvst) - 2025

Copyright (c) 2009 HHMI. Free downloads and distribution are allowed for any
non-profit research and educational purposes as long as proper credit is given
to the author. All other rights reserved.
"""
from warnings import warn

import WhiskiWrap
import os
import tempfile
from multiprocessing import freeze_support
import argparse
import shutil
import sys
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate whisk output from video in Parquet (default) or HDF5 format.')
    parser.add_argument('-v','--video_path',type=str, help='Path to the video (video must have an extension e.g. video.mp4).', default=None)
    parser.add_argument('-u','--select-folder-ui',action='store_true', help='Use file dialog to select video file.')
    parser.add_argument('-o','--output-dir',type=str, help='Output directory (default: same directory as input video).', default=None)
    parser.add_argument('-f','--format',type=str, choices=['parquet', 'hdf5'], help='Output format: parquet (default) or hdf5.', default='parquet')
    parser.add_argument('-p','--processes',type=int, help='Number of trace processes to run in parallel.', default=4)
    parser.add_argument('-c','--chunk-size',type=int, help='Number of frames per chunk (default: 100). Reduce to 50 or 25 if you get "Out of memory" errors.', default=100)
    parser.add_argument('-s','--sensitive',action='store_true', help='Use sensitive detection parameters for faint whiskers.')
    parser.add_argument('--copy-video',action='store_true', help='Copy the input video to the output directory (default: False).')
    args = parser.parse_args()
    
    # if select folder with ui
    if args.select_folder_ui:
        try:
            import easygui
            video_path = easygui.fileopenbox(title="Select video file to process")
            if not video_path:
                print("No file selected. Exiting.")
                sys.exit(0)
        except ImportError:
            print("Error: easygui module not found. Install with 'pip install easygui'")
            print("Or specify a video path with -v option")
            sys.exit(1)
    else:
        if args.video_path is None:
            parser.print_help()
            print("\nError: Video path must be specified when not using UI selection.")
            print("Example: python trace_whiskers.py -v test_videos/test_video_10s.mp4")
            sys.exit(1)
        video_path = args.video_path
        
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)


    try:
        # Determine working directory
        if args.output_dir:
            wdir = os.path.expanduser(args.output_dir)
            if not os.path.exists(wdir):
                print(f"Creating output directory: {wdir}")
                os.makedirs(wdir)
        else:
            # Default: same directory as input video
            wdir = os.path.dirname(os.path.abspath(video_path))
        
        # get video name
        video_fname = os.path.basename(video_path)
        video_name = ''.join(video_fname.split('.')[:-1])
        # output_path has the same name of the video name plus whisker_tracking_
        output_path = os.path.join(wdir,'whisker_tracking_'+video_name)
        # creates output path if it doesn't exists
        if not os.path.exists(output_path):
            print(f"Creating output directory: {output_path}")
            os.mkdir(output_path)
        
        # copies video if requested (optional)
        input_video = video_path  # Use original video path by default
        if args.copy_video:
            copied_video = os.path.join(output_path, video_fname)
            if not os.path.exists(copied_video):
                print(f"Copying video to working directory: {copied_video}")
                shutil.copy(video_path, copied_video)
            input_video = copied_video
        
        # Set up output file path based on format
        if args.format == 'parquet':
            output_file = os.path.join(output_path, video_name+'.parquet')
            file_format = 'parquet'
        else:  # hdf5
            output_file = os.path.join(output_path, video_name+'.hdf5')
            file_format = 'hdf5'
            
        freeze_support()
        input_video = os.path.expanduser(input_video)
        output_file = os.path.expanduser(output_file)
        print(f"Input video: {input_video}")
        print(f"Output file: {output_file}")
        print(f"Output format: {file_format}")

        # Create temporary directory for tiff stacks
        tiffs_directory = os.path.join(output_path, 'tracking_files')
        if not os.path.exists(tiffs_directory):
            os.makedirs(tiffs_directory)

        # Create FFmpegReader object for the input video
        try:
            input_reader = WhiskiWrap.FFmpegReader(input_video)
            
            print(f"Processing video with the following settings:")
            print(f"- Output directory: {output_path}")
            print(f"- Output format: {file_format}")
            print(f"- Parallel processes: {args.processes}")
            print(f"- Chunk size: {args.chunk_size} frames")
            print(f"- Sensitive detection: {'Yes' if args.sensitive else 'No'}")
            print(f"- Video dimensions: {input_reader.frame_width}x{input_reader.frame_height}")
            print(f"- Frame rate: {input_reader.frame_rate} fps")
            
            # Process video using interleaved_reading_and_tracing
            if args.format == 'parquet':
                WhiskiWrap.interleaved_reading_and_tracing(
                    input_reader=input_reader, 
                    tiffs_to_trace_directory=tiffs_directory,
                    parquet_filename=output_file,
                    n_trace_processes=args.processes,
                    chunk_size=args.chunk_size,
                    delete_tiffs=True,
                    sensitive=args.sensitive,
                    verbose=True
                )
            else:  # hdf5
                WhiskiWrap.interleaved_reading_and_tracing(
                    input_reader=input_reader, 
                    tiffs_to_trace_directory=tiffs_directory,
                    h5_filename=output_file,
                    n_trace_processes=args.processes,
                    chunk_size=args.chunk_size,
                    delete_tiffs=True,
                    sensitive=args.sensitive,
                    verbose=True
                )
        except Exception as e:
            error_message = str(e)
            print(f"Error during video processing: {error_message}")
            
            # Provide context-specific error messages
            if "FFmpeg" in error_message or "codec" in error_message.lower():
                print("Please check if FFmpeg is installed and accessible.")
            elif "whisk" in error_message.lower() or "trace" in error_message.lower():
                print("Please check if the whisk package is properly installed and the trace executable is available.")
            else:
                print("This may be due to video format issues, missing dependencies, or system resources.")
            
            sys.exit(1)
        
        print(f"\nWhisker tracking complete!")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        error_message = str(e)
        print(f"Error processing video: {error_message}")
        
        # Check for specific "Out of memory" error
        if "Out of memory" in error_message or "out of memory" in error_message.lower():
            print("\n" + "="*60)
            print("MEMORY ERROR DETECTED")
            print("="*60)
            print("The whisker tracking process ran out of memory.")
            print("This typically happens when processing large videos or using large chunk sizes.")
            print()
            print("SUGGESTED SOLUTIONS:")
            print(f"1. Try reducing the chunk size (current: {args.chunk_size}):")
            print(f"   python trace_whiskers.py -v {video_path} -c 50")
            print(f"   python trace_whiskers.py -v {video_path} -c 25")
            print()
            print("2. Reduce the number of parallel processes:")
            print(f"   python trace_whiskers.py -v {video_path} -p 2 -c 50")
            print()
            print("3. For very large videos, try both:")
            print(f"   python trace_whiskers.py -v {video_path} -p 2 -c 25")
            if args.format != 'parquet':
                print()
                print("4. Try using Parquet format (smaller memory footprint):")
                print(f"   python trace_whiskers.py -v {video_path} -f parquet -c 50")
            print("="*60)
        else:
            import traceback
            traceback.print_exc()
        
        sys.exit(1)
