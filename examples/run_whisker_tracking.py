#!/usr/bin/env python
"""
Example script showing how to run whisker tracking programmatically

This demonstrates how to use WhiskiWrap's interleaved_reading_and_tracing function
directly in your Python code.
"""
import os
import sys
import WhiskiWrap
from multiprocessing import freeze_support

# Add parent directory to path to ensure WhiskiWrap is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def track_whiskers(video_path, output_dir=None, n_processes=4, chunk_size=100, sensitive=False):
    """
    Track whiskers in a video using WhiskiWrap
    
    Parameters:
    -----------
    video_path : str
        Path to the video file
    output_dir : str, optional
        Directory to save results. If None, uses 'whiski_[videoname]'
    n_processes : int, optional
        Number of trace processes to run in parallel
    chunk_size : int, optional
        Number of frames per chunk
    sensitive : bool, optional
        Whether to use sensitive detection parameters
    
    Returns:
    --------
    str
        Path to the output HDF5 file
    """
    # Ensure video path is valid
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    # Set up paths
    video_name = os.path.basename(video_path).rsplit('.', 1)[0]
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), f'whiski_{video_name}')
        
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Create tiffs directory
    tiffs_dir = os.path.join(output_dir, 'tiff_stacks')
    if not os.path.exists(tiffs_dir):
        os.makedirs(tiffs_dir)
        
    # Output HDF5 file path
    output_file = os.path.join(output_dir, f"{video_name}.hdf5")
    
    # Create FFmpegReader
    input_reader = WhiskiWrap.FFmpegReader(video_path)
    
    # Run whisker tracking
    results = WhiskiWrap.interleaved_reading_and_tracing(
        input_reader=input_reader,
        tiffs_to_trace_directory=tiffs_dir,
        h5_filename=output_file,
        n_trace_processes=n_processes,
        chunk_size=chunk_size,
        delete_tiffs=True,
        sensitive=sensitive,
        verbose=True
    )
    
    return output_file

if __name__ == "__main__":
    freeze_support()  # Needed for Windows
    
    # Example using a test video
    test_video = "../test_videos/test_video_10s.mp4"
    
    if len(sys.argv) > 1:
        test_video = sys.argv[1]
    
    try:
        output_file = track_whiskers(
            test_video,
            n_processes=4,
            chunk_size=100,
            sensitive=False
        )
        print(f"Whisker tracking complete! Output saved to: {output_file}")
    except Exception as e:
        print(f"Error tracking whiskers: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
