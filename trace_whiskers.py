"""
trace_whiskers.py - A command-line tool for automated whisker tracking using WhiskiWrap

This script processes video files to track whiskers using the WhiskiWrap library,
which is a Python wrapper around the Janelia Whisker Tracking software.

Usage:
    python trace_whiskers.py -v path/to/video.mp4
    python trace_whiskers.py -u  # Opens a file selection dialog
    
Example:
    # Process a sample video from the test_videos directory
    python trace_whiskers.py -v test_videos/test_video_10s.mp4
    
    # Process with custom settings
    python trace_whiskers.py -v test_videos/test_video_10s.mp4 -p 8 -c 200

The script will:
1. Create a directory named 'whiski_[videoname]' in the current working directory
2. Copy the input video to this directory
3. Process the video with WhiskiWrap's interleaved_reading_and_tracing function
4. Generate an HDF5 file with the tracked whisker data

Parameters:
    -v, --video_path : Path to the video file to process
    -u, --select-folder-ui : Use a file dialog to select the video file
    -p, --processes : Number of trace processes to run in parallel (default: 4)
    -c, --chunk-size : Number of frames per chunk (default: 100)
    -s, --sensitive : Use sensitive detection parameters (default: False)

Output:
    The script creates a directory structure:
    whiski_[videoname]/
    ├── [videoname].mp4     # Copy of input video
    ├── [videoname].hdf5    # HDF5 file with whisker tracking data
    └── tiff_stacks/        # Temporary directory for TIFF image chunks

After processing, the HDF5 file can be analyzed with the provided notebooks:
- notebooks/plot_angle_trace.ipynb
- notebooks/plot_overlay_2sides.ipynb

Requirements:
    - WhiskiWrap and its dependencies must be installed
    - FFmpeg must be installed on the system

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
    parser = argparse.ArgumentParser(description='Generate whisk output from video.')
    parser.add_argument('-v','--video_path',type=str, help='Path to the video (video must have an extension e.g. video.mp4).', default=None)
    parser.add_argument('-u','--select-folder-ui',action='store_true', help='Use file dialog to select video file.')
    parser.add_argument('-p','--processes',type=int, help='Number of trace processes to run in parallel.', default=4)
    parser.add_argument('-c','--chunk-size',type=int, help='Number of frames per chunk.', default=100)
    parser.add_argument('-s','--sensitive',action='store_true', help='Use sensitive detection parameters for faint whiskers.')
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
        # working directory is always the script directory
        wdir = os.getcwd()
        # get video name
        video_fname = os.path.basename(video_path)
        video_name = ''.join(video_fname.split('.')[:-1])
        # output_path has the same name of the video name plus whiski_
        output_path = os.path.join(wdir,'whiski_'+video_name)
        # creates output path if it doesn't exists
        if not os.path.exists(output_path):
            print(f"Creating output directory: {output_path}")
            os.mkdir(output_path)
        
        # copies video if it is not there (in the output path)
        input_video = os.path.join(output_path, video_fname)
        if not os.path.exists(input_video):
            print(f"Copying video to working directory: {input_video}")
            shutil.copy(video_path, input_video)
        
        # Set up output file path
        output_file = os.path.join(output_path, video_name+'.hdf5')
        freeze_support()
        input_video = os.path.expanduser(input_video)
        output_file = os.path.expanduser(output_file)
        print(f"Input video: {input_video}")
        print(f"Output file: {output_file}")

        # Create temporary directory for tiff stacks
        tiffs_directory = os.path.join(output_path, 'tiff_stacks')
        if not os.path.exists(tiffs_directory):
            os.makedirs(tiffs_directory)

        # Create FFmpegReader object for the input video
        try:
            input_reader = WhiskiWrap.FFmpegReader(input_video)
            
            # Print processing information
            print(f"Processing video with the following settings:")
            print(f"- Parallel processes: {args.processes}")
            print(f"- Chunk size: {args.chunk_size} frames")
            print(f"- Sensitive detection: {'Yes' if args.sensitive else 'No'}")
            print(f"- Video dimensions: {input_reader.frame_width}x{input_reader.frame_height}")
            print(f"- Frame rate: {input_reader.frame_rate} fps")
            
            # Process video using interleaved_reading_and_tracing
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
            print(f"Error initializing video reader: {str(e)}")
            print(f"Please check if FFmpeg is installed and accessible.")
            sys.exit(1)
        
        print(f"\nWhisker tracking complete!")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
