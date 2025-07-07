"""WhiskiWrap provides tools for efficiently running whisk.

This module contains the following sub-modules:
    base - The core functions for whisker tracing, measurement, and data processing.
        Everything is imported from base into the main WhiskiWrap namespace.
    wfile_io - Functions for reading and writing whisker (.whiskers) files
    mfile_io - Functions for reading and writing measurement (.measurements) files

Related packages:
    wwutils - Utility functions including video processing, data loading, 
              classification, plotting, and system utilities (separate package)

To read Photonfocus double-rate files, you need to install libpfdoublerate
This requires libboost_thread 1.50 to be installed to /usr/local/lib
And the libpfDoubleRate.so in this module's directory

Here is an example workflow:

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

from . import base
#import video_utils
# Note: wwutils is a separate package - import it directly when needed
import importlib
importlib.reload(base)
from .base import *
from . import pipeline, io
importlib.reload(pipeline)
importlib.reload(io)
from .io import *
from .pipeline import *
