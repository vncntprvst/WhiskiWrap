"""WhiskiWrap provides tools for efficiently running whisk.

This package contains the following modules:
    base - Core functions for whisker tracing, measurement, and data processing
    pipeline - High-level pipeline functions for processing video data
    io - Input/output classes for reading/writing video and data files
    wfile_io - Functions for reading and writing whisker (.whiskers) files
    mfile_io - Functions for reading and writing measurement (.measurements) files

Related packages:
    wwutils - Utility functions including video processing, data loading, 
              classification, plotting, and system utilities (separate package)

To read Photonfocus double-rate files, you need to install libpfdoublerate
This requires libboost_thread 1.50 to be installed to /usr/local/lib
And the libpfDoubleRate.so in this module's directory

Example workflow:

import WhiskiWrap
import pandas as pd

# Run whisker tracing on video data
WhiskiWrap.interleaved_split_trace_and_measure(
    input_reader=WhiskiWrap.FFmpegReader('my_input_video.mp4'),
    tiffs_to_trace_directory='./trace_output',
    output_filename='traced_whiskers.parquet',
    chunk_size=200, n_trace_processes=4)

# Load the results of that analysis
traced_data = pd.read_parquet('traced_whiskers.parquet') 
"""

# Import all modules
from . import base, pipeline, io, wfile_io, mfile_io

# Import key functions and classes into the main namespace
from .base import *
from .io import *
from .pipeline import *
