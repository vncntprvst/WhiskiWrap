"""Main functions for running input videos through trace.

The overall algorithm is contained in `interleaved_reading_and_tracing`.
* The input can be a video file or a directory of PF files.
* Chunks of ~200 frames are read using ffmpeg, and then written to disk
  as uncompressed tiff stacks.
* Trace is called in parallel on each tiff stack
* Additional chunks are read as trace completes.
* At the end, all of the HDF5 files are stitched together.

The previous function `pipeline_trace` is now deprecated.
"""
import sys
try:
    import tifffile
except ImportError:
    pass
import os
import re
import numpy as np
import pandas as pd
import subprocess
import multiprocessing as mp
import tables
import scipy.io
import ctypes
import time
import shutil
import tempfile
import itertools
import json
import zarr
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
# from filelock import FileLock
# from threading import Lock
# Initialize a lock
# lock = Lock()

import pickle
import logging
# Set up logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# from . import wfile_io
# from .mfile_io import MeasurementsTable
# for debugging
from WhiskiWrap import wfile_io
from WhiskiWrap.mfile_io import MeasurementsTable

# try:
#     from whisk import trace
#     from whisk.traj import MeasurementsTable
# except ImportError as e:
    # import traceback
#     traceback.print_tb(e.__traceback__)
#     print("cannot import whisk")
if sys.platform == 'win32':
    whisk_path = ''
else:
    whisk_path = os.environ['WHISKPATH']
    if not whisk_path.endswith('/'):
        whisk_path += '/'
    print('WHISKPATH detected: ', whisk_path)

import WhiskiWrap
from WhiskiWrap import video_utils
import wwutils

# Find the repo directory and the default param files
# The banks don't differe with sensitive or default
DIRECTORY = os.path.split(__file__)[0]
PARAMETERS_FILE = os.path.join(DIRECTORY, 'default.parameters')
SENSITIVE_PARAMETERS_FILE = os.path.join(DIRECTORY, 'sensitive.parameters')
HALFSPACE_DB_FILE = os.path.join(DIRECTORY, 'halfspace.detectorbank')
LINE_DB_FILE = os.path.join(DIRECTORY, 'line.detectorbank')

# libpfDoubleRate library, needed for PFReader
LIB_DOUBLERATE = os.path.join(DIRECTORY, 'libpfDoubleRate.so')

def copy_parameters_files(target_directory, sensitive=False):
    """Copies in parameters and banks"""
    if sensitive:
        shutil.copyfile(SENSITIVE_PARAMETERS_FILE, os.path.join(target_directory,
            'default.parameters'))
    else:
        shutil.copyfile(PARAMETERS_FILE, os.path.join(target_directory,
            'default.parameters'))

    # Banks are the same regardless
    shutil.copyfile(HALFSPACE_DB_FILE, os.path.join(target_directory,
        'halfspace.detectorbank'))
    shutil.copyfile(LINE_DB_FILE, os.path.join(target_directory,
        'line.detectorbank'))

class WhiskerSeg(tables.IsDescription):
    fid = tables.UInt32Col()
    wid = tables.UInt16Col()
    tip_x = tables.Float32Col()
    tip_y = tables.Float32Col()
    follicle_x = tables.Float32Col()
    follicle_y = tables.Float32Col()
    pixel_length = tables.UInt16Col()
    chunk_start = tables.UInt32Col()

class WhiskerSeg_measure(tables.IsDescription):
    fid = tables.UInt32Col()
    wid = tables.UInt16Col()
    tip_x = tables.Float32Col()
    tip_y = tables.Float32Col()
    follicle_x = tables.Float32Col()
    follicle_y = tables.Float32Col()
    pixel_length = tables.UInt16Col()
    length = tables.Float32Col()
    score = tables.Float32Col()
    angle = tables.Float32Col()
    curvature = tables.Float32Col()
    chunk_start = tables.UInt32Col()
    # Adding previous missing fields label, face_x, face_y:
    label = tables.UInt16Col()
    # face_x and face_y are signed integers as the face location assigned by measure may be outside the frame
    face_x = tables.Int32Col()
    face_y = tables.Int32Col()
    # Adding a new face_side field
    face_side = tables.StringCol(itemsize=6)
    # Face side is a string, either 'left', 'right', 'top' or 'bottom'. Default is 'NA'.

def define_parquet_schema(class_definition):
    """
    Define the Parquet schema based on a PyTables IsDescription class.
    
    Parameters:
    - class_definition: The PyTables IsDescription class (e.g., WhiskerSeg or WhiskerSeg_measure)
    
    Returns:
    - schema: A PyArrow schema object
    """
    fields = []
    for field_name, field_type in class_definition.columns.items():
        if isinstance(field_type, tables.UInt32Col):
            fields.append((field_name, pa.uint32()))
        elif isinstance(field_type, tables.UInt16Col):
            fields.append((field_name, pa.uint16()))
        elif isinstance(field_type, tables.Float32Col):
            fields.append((field_name, pa.float32()))
        elif isinstance(field_type, tables.Int32Col):
            fields.append((field_name, pa.int32()))
        elif isinstance(field_type, tables.StringCol):
            fields.append((field_name, pa.string()))
        else:
            raise TypeError(f"Unsupported PyTables data type for field '{field_name}'")

    schema = pa.schema(fields)
    return schema

def write_chunk(chunk, chunkname, directory='.'):
    tifffile.imsave(os.path.join(directory, chunkname), chunk, compress=0)

def trace_chunk(video_filename, delete_when_done=False):
    """Run trace on an input file

    First we create a whiskers filename from `video_filename`, which is
    the same file with '.whiskers' replacing the extension. Then we run
    trace using subprocess.

    Care is taken to move into the working directory during trace, and then
    back to the original directory.

    Returns:
        stdout, stderr
    """
    print(("Starting", video_filename))
    orig_dir = os.getcwd()
    run_dir, raw_video_filename = os.path.split(os.path.abspath(video_filename))
    whiskers_file = WhiskiWrap.utils.FileNamer.from_video(video_filename).whiskers
    command = [whisk_path + 'trace', raw_video_filename, whiskers_file]

    os.chdir(run_dir)
    try:
        pipe = subprocess.Popen(command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            )
        stdout, stderr = pipe.communicate()
    except:
        raise
    finally:
        os.chdir(orig_dir)
    print(("Done", video_filename))

    if not os.path.exists(whiskers_file):
        print(raw_video_filename)
        raise IOError("tracing seems to have failed. STDOUT: " + stdout.decode('ascii') + '. STDERR: ' + stderr.decode('ascii')  )

    if delete_when_done:
        os.remove(video_filename)

    return {'video_filename': video_filename, 'stdout': stdout, 'stderr': stderr}

def measure_chunk(whiskers_filename, face, delete_when_done=False):
    """Run measure on an input file

    First we create a measurement filename from `whiskers_filename`, which is
    the same file with '.measurements' replacing the extension. Then we run
    trace using subprocess.

    Care is taken to move into the working directory during trace, and then
    back to the original directory.

    Returns:
        stdout, stderr
    """
    print(("Starting", whiskers_filename))
    orig_dir = os.getcwd()
    run_dir, raw_whiskers_filename = os.path.split(os.path.abspath(whiskers_filename))
    measurements_file = WhiskiWrap.utils.FileNamer.from_whiskers(whiskers_filename).measurements
    command = ['measure', '--face', face, raw_whiskers_filename, measurements_file]

    os.chdir(run_dir)
    try:
        pipe = subprocess.Popen(command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            )
        stdout, stderr = pipe.communicate()
    except:
        raise
    finally:
        os.chdir(orig_dir)
    print(("Done", whiskers_filename))

    if not os.path.exists(measurements_file):
        print(raw_whiskers_filename)
        raise IOError("measurement seems to have failed")

    if delete_when_done:
        os.remove(whiskers_filename)

    return {'whiskers_filename': whiskers_filename, 'stdout': stdout, 'stderr': stderr}

def trace_and_measure_chunk(video_filename, delete_when_done=False, face='right', classify=None, temp_dir=None, convert_chunk_to=None):
    """Run trace on an input file

    First we create a whiskers filename from `video_filename`, which is
    the same file with '.whiskers' replacing the extension. Then we run
    trace using subprocess.

    Care is taken to move into the working directory during trace, and then
    back to the original directory.

    Returns:
        stdout, stderr
    """
    print("Starting", video_filename)

    orig_dir = os.getcwd()
    run_dir, raw_video_filename = os.path.split(os.path.abspath(video_filename))

    # Run trace:
    whiskers_file = WhiskiWrap.utils.FileNamer.from_video(video_filename).whiskers
    trace_command = [whisk_path + 'trace', raw_video_filename, whiskers_file]

    os.chdir(run_dir)
    try:
        trace_pipe = subprocess.Popen(trace_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            )
        stdout, stderr = trace_pipe.communicate()
    except Exception as e:
        print(e)
        raise
    finally:
        os.chdir(orig_dir)
    print("Done", video_filename)

    if not os.path.exists(whiskers_file):
        print(raw_video_filename)
        raise IOError("tracing seems to have failed")

    # Run measure:
    measurements_file = WhiskiWrap.utils.FileNamer.from_video(video_filename).measurements
    measure_command = [whisk_path + 'measure', '--face', face, whiskers_file, measurements_file]

    os.chdir(run_dir)
    try:
        measure_pipe = subprocess.Popen(measure_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            )
        stdout, stderr = measure_pipe.communicate()
    except Exception as e:
        print(e)
        raise
    finally:
        os.chdir(orig_dir)
    print("Done", whiskers_file)

    if not os.path.exists(measurements_file):
        print(whiskers_file)
        raise IOError("measuring seems to have failed")
    
    # Run classify / re-classify
    # Typical classify arguments are {'px2mm': '0.04', 'n_whiskers': '3'}
    # Reference page: https://wikis.janelia.org/display/WT/Whisker+Tracking+Command+Line+Reference#WhiskerTrackingCommandLineReference-classify
    # Re-classify reference page: https://wikis.janelia.org/display/WT/Whisker+Tracking+Command+Line+Reference#WhiskerTrackingCommandLineReference-reclassify
    if classify is not None:
        measurements_file = WhiskiWrap.utils.FileNamer.from_video(video_filename).measurements
        
        classify_command = [whisk_path + 'classify', measurements_file, measurements_file, face, '--px2mm', classify['px2mm'], '-n', classify['n_whiskers']]
        if 'limit' in classify and classify['limit'] is not None:
            classify_command.append(f"--limit{classify['limit']}")
        if 'follicle' in classify and classify['follicle'] is not None:
            classify_command.extend(['--follicle', classify['follicle']])
            
        reclassify_command = [whisk_path + 'reclassify', measurements_file, measurements_file, '-n', classify['n_whiskers']]

        os.chdir(run_dir)
        try:
            classify_pipe = subprocess.Popen(classify_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                )
            stdout, stderr = classify_pipe.communicate()
            reclassify_pipe = subprocess.Popen(reclassify_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                )
            stdout, stderr = reclassify_pipe.communicate()
        except Exception as e:
            print(e)
            raise
        finally:
            os.chdir(orig_dir)
        print("Done", measurements_file)

        if not os.path.exists(measurements_file):
            print(measurements_file)
            raise IOError("classify seems to have failed")
        
    # Convert chunk to a different format
    if convert_chunk_to is not None:
        # Get chunk number from whiskers filename
        chunk_start = extract_chunk_number(whiskers_file)
        if convert_chunk_to == 'parquet':
            process_and_write_parquet(whiskers_file, chunk_start=chunk_start, face_side=face, temp_dir=temp_dir)
        elif convert_chunk_to == 'zarr':
            print("Converting chunk to Zarr is untested")
            initialize_zarr(temp_dir, chunk_size=(1000,))
            process_and_write_zarr(whiskers_file, chunk_start=chunk_start, output_dir=temp_dir)
        elif convert_chunk_to == 'hdf5':
            print("Converting chunk to HDF5 is untested")
            # Initialize HDF5 file
            h5_filename = os.path.join(temp_dir, f'chunk_{chunk_start}.h5')
            setup_hdf5(h5_filename, expected_rows=1000, measure=True)
            # Append whiskers to HDF5 file
            append_whiskers_to_hdf5(whiskers_file, h5_filename, chunk_start=chunk_start, measurements_filename=measurements_file, summary_only=False, face_side=face)
        else:
            raise ValueError(f"Unsupported conversion format: {convert_chunk_to}")

    # Clean up:
    if delete_when_done:
        os.remove(video_filename)

    return {'video_filename': video_filename,'stdout': stdout, 'stderr': stderr}

def sham_trace_chunk(video_filename):
    print(("sham tracing", video_filename))
    time.sleep(2)
    return video_filename

def setup_hdf5(h5_filename, expected_rows, measure=False):

    # Open file
    h5file = tables.open_file(h5_filename, mode="w")

    if not measure:
        WhiskerDescription = WhiskerSeg
    elif measure:
        WhiskerDescription = WhiskerSeg_measure

    # A group for the normal data
    table = h5file.create_table(h5file.root, "summary", WhiskerDescription,
        "Summary data about each whisker segment",
        expectedrows=expected_rows)

    # Put the contour here
    xpixels_vlarray = h5file.create_vlarray(
        h5file.root, 'pixels_x',
        tables.Float32Atom(shape=()),
        title='Every pixel of each whisker (x-coordinate)',
        expectedrows=expected_rows)
    ypixels_vlarray = h5file.create_vlarray(
        h5file.root, 'pixels_y',
        tables.Float32Atom(shape=()),
        title='Every pixel of each whisker (y-coordinate)',
        expectedrows=expected_rows)

    h5file.close()

def index_measurements(whiskers, measurements):
    """
    Re-indexing the measurements array according to whisker segment IDs and frame IDs in whiskers
    """    
    # We get all the segment ids from the whiskers dictionary. Since ids are not unique (they repeat for each frame), we also get the frame number, to get a unique set of identifiers. 
    
    # Create a list of (frame, segment_id) tuples from the whiskers dictionary
    segment_id_list = [(frame, wseg.id) for frame, frame_whiskers in whiskers.items() for wseg in frame_whiskers.values()]

    # Create a dictionary to map (frame, segment_id) to the corresponding index in the measurements array
    measurement_ids_dict = {(int(m[1]), int(m[2])): idx for idx, m in enumerate(measurements)}

    # Re-index the measurements array based on the order of segment_id_list
    # This is done by matching the frame number and the segment id. The result is a list of indices that can be used to index the measurements array
    measurements_reidx = []
    for segment_id in segment_id_list:
        # Find the corresponding index in the measurements array
        idx = measurement_ids_dict.get(segment_id, -1)
        measurements_reidx.append(idx)

    # Report how many whiskers were removed
    removed_count = measurements_reidx.count(-1)
    print(f"Number of whiskers removed by classify: {removed_count}")

    # Handle missing whiskers (where idx is -1)
    # valid_indices = [idx for idx in measurements_reidx if idx != -1]
    # measurements_reindexed = measurements[valid_indices]
    
    measurements_reindexed = measurements[measurements_reidx]   

    return measurements_reindexed

              
def append_whiskers_to_hdf5(whisk_filename, h5_filename, chunk_start, measurements_filename=None, summary_only=False, face_side='NA'):
    """Load data from whisk_file and put it into an hdf5 file

    The HDF5 file will have two basic components:
        /summary : A table with the following columns:
            If only .whiskers file given - fid, wid, follicle_x, follicle_y, tip_x, tip_y, pixel_length and chunk_start
            With measurements - fields above + length, score, angle, curvature, face_x, face_y, label, and face_side
        /pixels_x : A vlarray of the same length as summary but with the
            entire array of x-coordinates of each segment.
        /pixels_y : Same but for y-coordinates
    """

    ## Load it, so we know what expected rows is
    # This loads all whisker info into C data types
    # wv is like an array of trace.LP_cWhisker_Seg
    # Each entry is a trace.cWhisker_Seg and can be converted to
    # a python object via: wseg = trace.Whisker_Seg(wv[idx])
    # The python object responds to .time and .id (integers) and .x and .y (numpy
    # float arrays).
    #wv, nwhisk = trace.Debug_Load_Whiskers(whisk_filename)
    # try:
    
    # logging.debug(f'Starting append_whiskers_to_hdf5 with whisk_filename={whisk_filename}, h5_filename={h5_filename}, chunk_start={chunk_start}')
    print(whisk_filename)

    whiskers = wfile_io.Load_Whiskers(whisk_filename)
    # whiskers is a list of dictionaries, each dictionary is a frame, each frame has a dictionary of whiskers
    # len(whiskers) is = to chunk_size (which is a number of frames)

    nwhisk = np.sum(list(map(len, list(whiskers.values()))))

    if measurements_filename is not None:
        # logging.debug(f'Loading measurements from {measurements_filename}')
        print(measurements_filename)
        M = MeasurementsTable(str(measurements_filename))
        measurements = M.asarray()
        # len(measurements) is = to number of whiskers in those frames = chunk_size * n whiskers per frame

        measurements_idx = 0

        # If `classify` was run on the measurements file, then the index of the individual whiskers will have changed
        # Basically, the order of measurements is scrambled with respect to wseg.
        # Also, some whiskers may have been removed.
        # So we need to match wseg.id to measurements[2]

        # First check whether classify was run on the measurements file, by comparing whisker ids in the first frame
        # of the whiskers dictionary to the whisker ids in the measurements array

        wid_from_trace = np.array(list(whiskers[0].keys())).astype(int)
        initial_frame_measurements = measurements[:len(wid_from_trace)]
        wid_from_measure = initial_frame_measurements[:, 2].astype(int)

        if not np.array_equal(wid_from_trace, wid_from_measure):
            measurements=index_measurements(whiskers,measurements)
            
    # # If this function is called in parallel (or other situations with concurrent access
    # lock the file to ensure exclusive access:
    # lock = FileLock(h5_filename + ".lock")
    # with lock:
    # Open file
    h5file = tables.open_file(h5_filename, mode="a")

    # See setup_hdf5 for creation of the table
    # 05/11/23: changed WhiskerSeg_measure to add face_x / face_y / label fields

    ## Iterate over rows and store
    table = h5file.get_node('/summary')
    h5seg = table.row
    if not summary_only:
        xpixels_vlarray = h5file.get_node('/pixels_x')
        ypixels_vlarray = h5file.get_node('/pixels_y')
    for frame, frame_whiskers in list(whiskers.items()):
        for whisker_id, wseg in list(frame_whiskers.items()):
            # Write to the table
            h5seg['chunk_start'] = chunk_start
            h5seg['fid'] = wseg.time + chunk_start
            h5seg['wid'] = wseg.id

            if measurements_filename is not None:
                h5seg['length'] = measurements[measurements_idx][3]
                h5seg['score'] = measurements[measurements_idx][4]
                h5seg['angle'] = measurements[measurements_idx][5]
                h5seg['curvature'] = measurements[measurements_idx][6]
                h5seg['pixel_length'] = len(wseg.x)
                h5seg['follicle_x'] = measurements[measurements_idx][7]
                h5seg['follicle_y'] = measurements[measurements_idx][8]
                h5seg['tip_x'] = measurements[measurements_idx][9]
                h5seg['tip_y'] = measurements[measurements_idx][10]
                h5seg['label'] = 0
                h5seg['face_x'] = M._measurements.contents.face_x
                h5seg['face_y'] = M._measurements.contents.face_y
                h5seg['face_side'] = face_side

                measurements_idx += 1

            else:
                h5seg['follicle_x'] = wseg.x[-1]
                h5seg['follicle_y'] = wseg.y[-1]
                h5seg['tip_x'] = wseg.x[0]
                h5seg['tip_y'] = wseg.y[0]
   
            assert len(wseg.x) == len(wseg.y)
            h5seg.append()

            if not summary_only:
            # Write whisker contour x and y pixel values
                xpixels_vlarray.append(wseg.x)
                ypixels_vlarray.append(wseg.y)

    table.flush()
    h5file.close()
    # logging.debug('Finished append_whiskers_to_hdf5 successfully')
    
    # except Exception as e:
        # logging.error(f'Error in append_whiskers_to_hdf5: {e}', exc_info=True)

def process_and_write_parquet(whiskers_filename, chunk_start, face_side, temp_dir):
    # Generate the .whiskers filename - whiskers_filename may be a .tif file at this point
    if not whiskers_filename.endswith('.whiskers'):
        whiskers_filename = whiskers_filename.replace('.tif', '.whiskers')
    
    # Generate the corresponding .measurements filename
    measurements_filename = whiskers_filename.replace('.whiskers', '.measurements')
    # Check that that file exists. If not, set to None
    if not os.path.exists(measurements_filename):
        measurements_filename = None
    
    # Generate the output Parquet filename
    # temp_file = os.path.join(temp_dir, f'chunk_{chunk_start}.parquet')
    parquet_filename = os.path.join(temp_dir, os.path.basename(whiskers_filename).replace('.whiskers', '.parquet'))
        
    # Call the function to append whiskers to the parquet file
    append_whiskers_to_parquet(
        whisk_filename=whiskers_filename,
        measurements_filename=measurements_filename,
        parquet_filename=parquet_filename,
        chunk_start=chunk_start,
        summary_only=False,
        face_side=face_side
    )

def append_whiskers_to_parquet(whisk_filename, measurements_filename, parquet_filename, chunk_start, summary_only=False, face_side='NA'):
    """
    Append whiskers and measurements data to a Parquet file.

    Parameters:
    - whisk_filename: The .whiskers file containing whisker trace data.
    - measurements_filename: The .measurements file containing additional measurement data.
    - parquet_filename: The output Parquet file to append to.
    - chunk_start: The starting frame index for this chunk.
    """
    # Load whiskers data
    whiskers = wfile_io.Load_Whiskers(whisk_filename)
    nwhisk = np.sum(list(map(len, list(whiskers.values()))))

    # Load measurements data if available
    if measurements_filename is not None:
        M = MeasurementsTable(str(measurements_filename))
        measurements = M.asarray()
        measurements_idx = 0

        # Check if measurements need to be reindexed
        wid_from_trace = np.array(list(whiskers[0].keys())).astype(int)
        initial_frame_measurements = measurements[:len(wid_from_trace)]
        wid_from_measure = initial_frame_measurements[:, 2].astype(int)

        if not np.array_equal(wid_from_trace, wid_from_measure):
            measurements = index_measurements(whiskers, measurements)
    else:
        measurements = None

    # Prepare data for Parquet
    summary_data = []
    if not summary_only:
        pixels_x_data = []
        pixels_y_data = []

    for _, frame_whiskers in whiskers.items():
        for _, wseg in frame_whiskers.items():
            whisker_data = {
                'chunk_start': chunk_start,
                'fid': wseg.time + chunk_start,
                'wid': wseg.id                
            }

            if measurements is not None:
                whisker_data.update({
                    'length': measurements[measurements_idx][3],
                    'score': measurements[measurements_idx][4],
                    'angle': measurements[measurements_idx][5],
                    'curvature': measurements[measurements_idx][6],
                    'pixel_length': len(wseg.x),
                    'follicle_x': measurements[measurements_idx][7],
                    'follicle_y': measurements[measurements_idx][8],
                    'tip_x': measurements[measurements_idx][9],
                    'tip_y': measurements[measurements_idx][10],
                    'label': 0,
                    'face_x': M._measurements.contents.face_x,
                    'face_y': M._measurements.contents.face_y,
                    'face_side': face_side
                })
                measurements_idx += 1
            else:
                whisker_data.update({
                    'follicle_x': wseg.x[-1],
                    'follicle_y': wseg.y[-1],
                    'tip_x': wseg.x[0],
                    'tip_y': wseg.y[0]
                })

            summary_data.append(whisker_data)
            if not summary_only:
                pixels_x_data.append(wseg.x.tolist())
                pixels_y_data.append(wseg.y.tolist())

    # # Convert to Pandas DataFrames
    # summary_df = pd.DataFrame(summary_data)
    # if not summary_only:
    #     pixels_x_df = pd.DataFrame({'pixels_x': pixels_x_data})
    #     pixels_y_df = pd.DataFrame({'pixels_y': pixels_y_data})
    
    # # Define the schema for WhiskerSeg_measure
    # whisker_seg_measure_schema = define_parquet_schema(WhiskerSeg_measure)

    # # Write to Parquet using PyArrow
    # with pq.ParquetWriter(parquet_filename, schema=whisker_seg_measure_schema) as writer:
    #     if not summary_df.empty:
    #         table_summary = pa.Table.from_pandas(summary_df, schema=whisker_seg_measure_schema)
    #         writer.write_table(table_summary)
    #     if not summary_only:
    #         if not pixels_x_df.empty:
    #             table_pixels_x = pa.Table.from_pandas(pixels_x_df)
    #             writer.write_table(table_pixels_x)

    #         if not pixels_y_df.empty:
    #             table_pixels_y = pa.Table.from_pandas(pixels_y_df)
    #             writer.write_table(table_pixels_y)
            
    # Convert to Pandas DataFrame
    summary_df = pd.DataFrame(summary_data)

    if not summary_only:
        # Ensure pixels_x_df and pixels_y_df are combined with summary_df
        summary_df['pixels_x'] = pixels_x_data
        summary_df['pixels_y'] = pixels_y_data

    # Convert the combined DataFrame to a PyArrow Table
    combined_table = pa.Table.from_pandas(summary_df)

    # Write the combined table to the Parquet file
    pq.write_table(combined_table, parquet_filename)
    
def extract_chunk_number(filename):
    """
    Extract the chunk number from a filename.
    """
    # Assuming the filename pattern is '<filename>_<number>.<file_extension>'
    return int(re.search(r'\d+', filename.split('_')[-1]).group())    

def merge_parquet_files(temp_dir, output_filename):
    """
    Merge all Parquet files in the temporary directory into a single Parquet file.
    """
    # List all Parquet files in the temporary directory
    parquet_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.parquet')]
    
    # Sort the files based on the extracted chunk number
    parquet_files_sorted = sorted(parquet_files, key=extract_chunk_number)

    # Read all Parquet files into a list of tables
    tables = [pq.read_table(f) for f in parquet_files_sorted]

    # Concatenate all tables
    combined_table = pa.concat_tables(tables)

    # Write the combined table to the final Parquet file
    pq.write_table(combined_table, output_filename)
    
def process_and_write_zarr(fn, chunk_start, output_dir):
    # Example function for direct Zarr writing
    whisk_filename, measurements_filename = fn.whiskers, fn.measurements
    append_whiskers_to_zarr(
        whisk_filename=whisk_filename,
        measurements_filename=measurements_filename,
        zarr_filename=output_dir,
        chunk_start=chunk_start
    )

def consolidate_zarr_metadata(zarr_dir):
    # Function to consolidate Zarr metadata
    zarr.consolidate_metadata(zarr_dir)
    
def initialize_zarr(zarr_filename, chunk_size=(1000,)):
    zarr_file = zarr.open(zarr_filename, mode='a')

    if 'summary' not in zarr_file:
        zarr_file.create_dataset('summary', shape=(0,), dtype=[
            ('chunk_start', 'i4'), ('fid', 'i4'), ('wid', 'i4'),
            ('follicle_x', 'f4'), ('follicle_y', 'f4'), ('tip_x', 'f4'),
            ('tip_y', 'f4'), ('pixel_length', 'i4'), ('length', 'f4'),
            ('score', 'f4'), ('angle', 'f4'), ('curvature', 'f4'),
            ('face_x', 'i4'), ('face_y', 'i4'), ('face_side', 'S6'), ('label', 'i4')
        ], chunks=chunk_size)

    if 'pixels_x' not in zarr_file:
        zarr_file.create_dataset('pixels_x', shape=(0,), dtype='f8', chunks=chunk_size)
        zarr_file.create_dataset('pixels_x_indices', shape=(0, 5), dtype='i4', chunks=chunk_size)
    if 'pixels_y' not in zarr_file:
        zarr_file.create_dataset('pixels_y', shape=(0,), dtype='f8', chunks=chunk_size)
        zarr_file.create_dataset('pixels_y_indices', shape=(0, 5), dtype='i4', chunks=chunk_size)

    return zarr_file

def face_side_to_bool(face_side):

    if face_side == 'left':
        face_side_bool = 0
    elif face_side == 'right':
        face_side_bool = 1
    else:
        face_side_bool = 2
        
    return face_side_bool

def append_whiskers_to_zarr(whisk_filename, zarr_filename, chunk_start, measurements_filename=None, 
                            face_side='NA', chunk_size=(1000,), add_to_queue=False, summary_only=False):
    """
    Fast function to append whiskers to a Zarr file. 
    Note that this function does not support parallel writing to the same Zarr file (at least for the pixels_x and pixels_y arrays).
    """
    # logging.debug(f'Starting append_whiskers_to_zarr with whisk_filename={whisk_filename}, zarr_filename={zarr_filename}, chunk_start={chunk_start}')
    
    face_side_bool = face_side_to_bool(face_side)
    
    try:
        whiskers = wfile_io.Load_Whiskers(whisk_filename)
        nwhisk = np.sum(list(map(len, list(whiskers.values()))))

        if measurements_filename is not None:
            # logging.debug(f'Loading measurements from {measurements_filename}')
            M = MeasurementsTable(str(measurements_filename))
            measurements = M.asarray()
            measurements_idx = 0

            wid_from_trace = np.array(list(whiskers[0].keys())).astype(int)
            initial_frame_measurements = measurements[:len(wid_from_trace)]
            wid_from_measure = initial_frame_measurements[:, 2].astype(int)

            if not np.array_equal(wid_from_trace, wid_from_measure):
                measurements = index_measurements(whiskers, measurements)

        # Initialize or open Zarr file
        if not add_to_queue:
            zarr_file = initialize_zarr(zarr_filename, chunk_size)

        summary_data_list = []
        pixels_x_list = []
        pixels_y_list = []
        pixels_x_indices_list = []
        pixels_y_indices_list = []

        for frame, frame_whiskers in list(whiskers.items()):
            for whisker_id, wseg in list(frame_whiskers.items()):
                summary_data = {
                    'chunk_start': chunk_start,
                    'fid': wseg.time + chunk_start,
                    'wid': wseg.id,
                    'follicle_x': wseg.x[-1],
                    'follicle_y': wseg.y[-1],
                    'tip_x': wseg.x[0],
                    'tip_y': wseg.y[0],
                    'pixel_length': len(wseg.x)
                }

                if measurements_filename is not None:
                    # logging.debug(f'Loading measurements from {measurements_filename}')
                    summary_data.update({
                        'follicle_x': measurements[measurements_idx][7],
                        'follicle_y': measurements[measurements_idx][8],
                        'tip_x': measurements[measurements_idx][9],
                        'tip_y': measurements[measurements_idx][10],
                        'length': measurements[measurements_idx][3],
                        'score': measurements[measurements_idx][4],
                        'angle': measurements[measurements_idx][5],
                        'curvature': measurements[measurements_idx][6],
                        'face_x': M._measurements.contents.face_x,
                        'face_y': M._measurements.contents.face_y,
                        'face_side': face_side,
                        'label': 0
                    })
                    measurements_idx += 1

                summary_data_list.append(summary_data)
                if not summary_only:
                    # Simultaneous retrieval of data
                    pixels_x_list_temp, pixels_y_list_temp = wseg.x.tolist(), wseg.y.tolist()
                
                    # Calculate start and end indices
                    # start_x, start_y = len(pixels_x_list), len(pixels_y_list)
                    # end_x, end_y = start_x + len(pixels_x_list_temp), start_y + len(pixels_y_list_temp)
                    start_x, start_y, end_x, end_y = len(pixels_x_list), len(pixels_y_list), len(pixels_x_list) + len(pixels_x_list_temp), len(pixels_y_list) + len(pixels_y_list_temp)

                    # Extend pixels lists
                    pixels_x_list.extend(pixels_x_list_temp)
                    pixels_y_list.extend(pixels_y_list_temp)

                    # Append frame id, face side, and whisker id along with start and end indices
                    pixels_x_indices_list.append([wseg.time + chunk_start, face_side_bool, wseg.id, start_x, end_x])
                    pixels_y_indices_list.append([wseg.time + chunk_start, face_side_bool, wseg.id, start_y, end_y])
                    
        # Append collected data to Zarr arrays
        if add_to_queue:
            result = (summary_data_list, pixels_x_list, pixels_x_indices_list, pixels_y_list, pixels_y_indices_list)
            # logging.debug(f"Prepared result: {len(summary_data_list)} summary records, {len(pixels_x_list)} pixels_x, {len(pixels_y_list)} pixels_y")
            return result
        else:
            if summary_data_list:
                summary_array = np.fromiter((tuple(d.values()) for d in summary_data_list), dtype=zarr_file['summary'].dtype)
                zarr_file['summary'].append(summary_array)
            if pixels_x_list:
                zarr_file['pixels_x'].append(pixels_x_list)
                zarr_file['pixels_x_indices'].append(pixels_x_indices_list)
            #     for pixels_x in pixels_x_list:
            #         pixels_x_str = json.dumps(pixels_x)
            #         zarr_file['pixels_x'].append([pixels_x_str])
            if pixels_y_list:
                zarr_file['pixels_y'].append(pixels_y_list)
                zarr_file['pixels_y_indices'].append(pixels_y_indices_list)
            #     for pixels_y in pixels_y_list:
            #         pixels_y_str = json.dumps(pixels_y)
            #         zarr_file['pixels_y'].append([pixels_y_str])                
                
        # pd.DataFrame(summary_array).to_csv(f"{whisk_filename.split('.')[0]}_summary.csv", index=False)
        # logging.debug('Finished append_whiskers_to_zarr successfully')

    except Exception as e:
        logging.error(f'Error in append_whiskers_to_zarr: {e}', exc_info=True)

def write_whiskers_to_tmp(whisk_filename, measurements_filename, chunk_start, face_side):
    """
    Function to process whisker data and write it to a temporary file.
    
    Example usage:
    whiskers_files = [...]  # List of whiskers files
    measurement_files = [...]  # List of measurement files
    output_file = "output.zarr"
    chunk_starts = [...]  # List of chunk start positions
    sides = [...]  # List of face sides

    with ProcessPoolExecutor() as executor:
        temp_files = list(executor.map(process_data, whiskers_files, 
                            measurement_files, chunk_starts, sides))  
    """
    try:
        whiskers = wfile_io.Load_Whiskers(whisk_filename)
        processed_data = {
            'summary_data': [],
            'pixels_x': [],
            'pixels_y': []
        }

        if measurements_filename is not None:
            M = MeasurementsTable(str(measurements_filename))
            measurements = M.asarray()
            measurements_idx = 0

            wid_from_trace = np.array(list(whiskers[0].keys())).astype(int)
            initial_frame_measurements = measurements[:len(wid_from_trace)]
            wid_from_measure = initial_frame_measurements[:, 2].astype(int)

            if not np.array_equal(wid_from_trace, wid_from_measure):
                measurements = index_measurements(whiskers, measurements)

        for frame, frame_whiskers in list(whiskers.items()):
            for whisker_id, wseg in list(frame_whiskers.items()):
                summary_data = {
                    'chunk_start': chunk_start,
                    'fid': wseg.time + chunk_start,
                    'wid': wseg.id,
                    'follicle_x': wseg.x[-1],
                    'follicle_y': wseg.y[-1],
                    'tip_x': wseg.x[0],
                    'tip_y': wseg.y[0],
                    'pixel_length': len(wseg.x)
                }

                if measurements_filename is not None:
                    summary_data.update({
                        'follicle_x': measurements[measurements_idx][7],
                        'follicle_y': measurements[measurements_idx][8],
                        'tip_x': measurements[measurements_idx][9],
                        'tip_y': measurements[measurements_idx][10],
                        'length': measurements[measurements_idx][3],
                        'score': measurements[measurements_idx][4],
                        'angle': measurements[measurements_idx][5],
                        'curvature': measurements[measurements_idx][6],
                        'face_x': M._measurements.contents.face_x,
                        'face_y': M._measurements.contents.face_y,
                        'face_side': face_side,
                        'label': 0
                    })
                    measurements_idx += 1

                processed_data['summary_data'].append(summary_data)
                processed_data['pixels_x'].append(wseg.x.tolist())
                processed_data['pixels_y'].append(wseg.y.tolist())

        temp_filename = f"temp_{os.path.basename(whisk_filename)}.pkl"
        with open(temp_filename, 'wb') as temp_file:
            pickle.dump(processed_data, temp_file)

        return temp_filename

    except Exception as e:
        logging.error(f"Error processing {whisk_filename}: {e}")
        return None

def write_tmp_to_zarr(zarr_filename, temp_files):
    """
    Function to write temporary whisker data to a Zarr file. 
    Use it when writing whisker data in parallel to temporary files 
    and then combining them serially into a single Zarr file.
    
    Example usage:
    zarr_filename = 'test.zarr'
    temp_files = ['temp_whiskers1.pkl', 'temp_whiskers2.pkl']
    write_tmp_to_zarr(zarr_filename, temp_files)
    """
    zarr_file = initialize_zarr(zarr_filename)
    current_x_index = 0
    current_y_index = 0

    for temp_file in temp_files:
        if temp_file is None:
            continue
        with open(temp_file, 'rb') as f:
            processed_data = pickle.load(f)

        summary_data = processed_data['summary_data']
        pixels_x = processed_data['pixels_x']
        pixels_y = processed_data['pixels_y']

        summary_array = np.fromiter((tuple(d.values()) for d in summary_data), dtype=zarr_file['summary'].dtype)
        zarr_file['summary'].append(summary_array)

        for px in pixels_x:
            start_idx = current_x_index
            end_idx = current_x_index + len(px)
            zarr_file['pixels_x'].append(px)
            zarr_file['pixels_x_indices'].append([(start_idx, end_idx)])
            current_x_index = end_idx

        for py in pixels_y:
            start_idx = current_y_index
            end_idx = current_y_index + len(py)
            zarr_file['pixels_y'].append(py)
            zarr_file['pixels_y_indices'].append([(start_idx, end_idx)])
            current_y_index = end_idx

        os.remove(temp_file)
        
def initialize_whisker_measurement_table():
    """Initialize tabular data to enter into WhiskerMeasurementTable, using Pandas DataFrames"""
    # Define column types
    column_types = {
        'frame_id': 'uint32',
        'whisker_id': 'uint16',
        'label': 'int16',
        'face_x': 'int32',
        'face_y': 'int32',
        'length': 'float32',
        'pixel_length': 'uint16',
        'score': 'float32',
        'angle': 'float32',
        'curvature': 'float32',
        'follicle_x': 'float32',
        'follicle_y': 'float32',
        'tip_x': 'float32',
        'tip_y': 'float32',
        'chunk_start': 'uint32'
    }
    
    measurement_data = pd.DataFrame(columns=column_types.keys()).astype(column_types)

    return measurement_data, column_types

def read_whisker_data(filename, output_format='dict'):
    """
    Parameter: filename (str) - path to whisker file.
    Accepts hdf5 and .whiskers files. If the latter, it will look for a .measurements file in the same directory.
    Returns a dictionary with the following keys:
        'whisker_id' - list of whisker segment ids
        'label' - list of labels (segment id, or if re-classified, initial guess of segment identities)
        'frame_id' - list of frame ids
        'follicle_x' - list of follicle x coordinates
        'follicle_y' - list of follicle y coordinates
        'tip_x' - list of tip x coordinates
        'tip_y' - list of tip y coordinates
        'pixel_length' - list of pixel lengths
        'length' - list of lengths
        'score' - list of scores
        'angle' - list of angles
        'curvature' - list of curvatures
        'face_x' - list of face x coordinates
        'face_y' - list of face y coordinates
        'chunk_start' - list of chunk start indices
    """    
    if filename is None:
        print("No measurements file provided")
        return None
    
    if filename.endswith('.whiskers'):
        # Read whisker file
        whiskers = wfile_io.Load_Whiskers(filename)
        
        # check if measurements file exists
        measurements_filename = filename.replace('.whiskers', '.measurements')
        
        if os.path.isfile(measurements_filename):
            # print(measurements_filename)
            M = MeasurementsTable(str(measurements_filename))
            measurements = M.asarray()
            measurements_idx = 0
        else:
            # return in error
            print("No measurements file found")
            return None
            
        # First check whether classify was run on the measurements file, by comparing whisker ids in the first frame of the whiskers dictionary to the whisker ids in the measurements array

        wid_from_trace = np.array(list(whiskers[0].keys())).astype(int)
        initial_frame_measurements = measurements[:len(wid_from_trace)]
        wid_from_measure = initial_frame_measurements[:, 2].astype(int)

        if not np.array_equal(wid_from_trace, wid_from_measure):
            print("classify was run on the measurements file")
            measurements=index_measurements(whiskers,measurements)

        meas_rows = []
        chunk_start = 0
        
        ## Iterate over rows and append to table
        for frame, frame_whiskers in list(whiskers.items()):
            for whisker_id, wseg in list(frame_whiskers.items()):
                # Write to the table, using Pandas DataFrame (more efficient for adding multiple rows):
                # Create a new row as a dictionary
                new_meas_row = {
                    'frame_id': wseg.time + chunk_start,
                    'whisker_id': wseg.id,
                    'label': measurements[measurements_idx][0].astype('int16'),
                    'face_x': M._measurements.contents.face_x,
                    'face_y': M._measurements.contents.face_y,
                    'length': measurements[measurements_idx][3],
                    'pixel_length': len(wseg.x),
                    'score': measurements[measurements_idx][4],
                    'angle': measurements[measurements_idx][5],
                    'curvature': measurements[measurements_idx][6],
                    'follicle_x': measurements[measurements_idx][7],
                    'follicle_y': measurements[measurements_idx][8],
                    'tip_x': measurements[measurements_idx][9],
                    'tip_y': measurements[measurements_idx][10],
                    'chunk_start': chunk_start,
                }
                
                measurements_idx += 1
                assert len(wseg.x) == len(wseg.y)
                
                # Append the new row to the DataFrame
                # meas_table = meas_table.append(new_meas_row, ignore_index=True)
                meas_rows.append(new_meas_row)
                
                # Write whisker contour x and y pixel values
                # %TODO: save whisker contours
                # xpixels_vlarray.append(wseg.x)
                # ypixels_vlarray.append(wseg.y)
                
        # no need to initilize the table, just create a new one
        _, column_types = initialize_whisker_measurement_table()
        meas_table = pd.DataFrame(meas_rows).astype(column_types)
        
    elif filename.endswith('.hdf5'):   
        # Read hdf5 file and returns a pandas DataFrame
        meas_table = read_whiskers_hdf5_summary(filename)
        # print(meas_table.head())

        # Convert non-matching table labels
        meas_table.rename(columns={'fid': 'frame_id', 'wid': 'whisker_id'}, inplace=True)
                
    # Convert to dictionary
    if output_format == 'dict':
        data_dict = {k: meas_table[k].values for k in meas_table.columns}
        return data_dict
    elif output_format == 'df':
        return meas_table

def pipeline_trace(input_vfile, h5_filename,
    epoch_sz_frames=3200, chunk_sz_frames=200,
    frame_start=0, frame_stop=None,
    n_trace_processes=4, expectedrows=1000000, flush_interval=100000,
    measure=False,face='right'):
    """Trace a video file using a chunked strategy.

    This is now deprecated in favor of interleaved_reading_and_tracing.
    The issue with this function is that it has to write out all of the tiffs
    first, before tracing, which is a wasteful use of disk space.

    input_vfile : input video filename
    h5_filename : output HDF5 file
    epoch_sz_frames : Video is first broken into epochs of this length
    chunk_sz_frames : Each epoch is broken into chunks of this length
    frame_start, frame_stop : where to start and stop processing
    n_trace_processes : how many simultaneous processes to use for tracing
    expectedrows, flush_interval : used to set up hdf5 file

    TODO: combine the reading and writing stages using frame_func so that
    we don't have to load the whole epoch in at once. In fact then we don't
    even need epochs at all.
    """
    WhiskiWrap.utils.probe_needed_commands()

    # Figure out where to store temporary data
    input_vfile = os.path.abspath(input_vfile)
    input_dir = os.path.split(input_vfile)[0]

    # Setup the result file
    setup_hdf5(h5_filename, expectedrows, measure=measure)

    # Figure out how many frames and epochs
    duration = wwutils.video.get_video_duration(input_vfile)
    frame_rate = wwutils.video.get_video_params(input_vfile)[2]
    total_frames = int(np.rint(duration * frame_rate))
    if frame_stop is None:
        frame_stop = total_frames
    if frame_stop > total_frames:
        print("too many frames requested, truncating")
        frame_stop = total_frames

    # Iterate over epochs
    for start_epoch in range(frame_start, frame_stop, epoch_sz_frames):
        # How many frames in this epoch
        stop_epoch = np.min([frame_stop, start_epoch + epoch_sz_frames])
        print(("Epoch %d - %d" % (start_epoch, stop_epoch)))

        # Chunks
        chunk_starts = np.arange(start_epoch, stop_epoch, chunk_sz_frames)
        chunk_names = ['chunk%08d.tif' % nframe for nframe in chunk_starts]
        whisk_names = ['chunk%08d.whiskers' % nframe for nframe in chunk_starts]

        # read everything
        # need to be able to crop here
        print("Reading")
        frames = video_utils.process_chunks_of_video(input_vfile,
            frame_start=start_epoch, frame_stop=stop_epoch,
            frames_per_chunk=chunk_sz_frames, # only necessary for chunk_func
            frame_func=None, chunk_func=None,
            verbose=False, finalize='listcomp')

        # Dump frames into tiffs or lossless
        print("Writing")
        for n_whiski_chunk, chunk_name in enumerate(chunk_names):
            print(n_whiski_chunk)
            chunkstart = n_whiski_chunk * chunk_sz_frames
            chunkstop = (n_whiski_chunk + 1) * chunk_sz_frames
            chunk = frames[chunkstart:chunkstop]
            if len(chunk) in [3, 4]:
                print("WARNING: trace will fail on tiff stacks of length 3 or 4")
            write_chunk(chunk, chunk_name, input_dir)

        # Also write lossless and/or lossy monitor video here?
        # would really only be useful if cropping applied

        # trace each
        print("Tracing")
        pool = mp.Pool(n_trace_processes)
        trace_res = pool.map(trace_chunk,
            [os.path.join(input_dir, chunk_name)
                for chunk_name in chunk_names])
        pool.close()

        # take measurements:
        if measure:
            print("Measuring")
            pool = mp.Pool(n_trace_processes)
            meas_res = pool.map(measure_chunk_star,
                list(zip([os.path.join(input_dir, whisk_name)
                    for whisk_name in whisk_names],itertools.repeat(face))))
            pool.close()

        # stitch
        print("Stitching")
        for chunk_start, chunk_name in zip(chunk_starts, chunk_names):
            # Append each chunk to the hdf5 file
            fn = WhiskiWrap.utils.FileNamer.from_tiff_stack(
                os.path.join(input_dir, chunk_name))

            if not measure:
                append_whiskers_to_hdf5(
                    whisk_filename=fn.whiskers,
                    h5_filename=h5_filename,
                    chunk_start=chunk_start)
            elif measure:
                append_whiskers_to_hdf5(
                    whisk_filename=fn.whiskers,
                    measurements_filename=fn.measurements,
                    h5_filename=h5_filename,
                    chunk_start=chunk_start)

def write_video_as_chunked_tiffs(input_reader, tiffs_to_trace_directory,
    chunk_size=200, chunk_name_pattern='chunk%08d.tif',
    stop_after_frame=None, monitor_video=None, timestamps_filename=None,
    monitor_video_kwargs=None):
    """Write frames to disk as tiff stacks

    input_reader : object providing .iter_frames() method and perhaps
        also a .timestamps attribute. For instance, PFReader, or some
        FFmpegReader object.
    tiffs_to_trace_directory : where to store the chunked tiffs
    stop_after_frame : to stop early
    monitor_video : if not None, should be a filename to write a movie to
    timestamps_filename : if not None, should be the name to write timestamps
    monitor_video_kwargs : ffmpeg params

    Returns: ChunkedTiffWriter object
    """
    # Tiff writer
    ctw = WhiskiWrap.ChunkedTiffWriter(tiffs_to_trace_directory,
        chunk_size=chunk_size, chunk_name_pattern=chunk_name_pattern)

    # FFmpeg writer is initalized after first frame
    ffw = None

    # Iterate over frames
    for nframe, frame in enumerate(input_reader.iter_frames()):
        # Stop early?
        if stop_after_frame is not None and nframe >= stop_after_frame:
            break

        # Write to chunked tiff
        ctw.write(frame)

        # Optionally write to monitor video
        if monitor_video is not None:
            # Initialize ffw after first frame so we know the size
            if ffw is None:
                ffw = WhiskiWrap.FFmpegWriter(monitor_video,
                    frame_width=frame.shape[1], frame_height=frame.shape[0],
                    **monitor_video_kwargs)
            ffw.write(frame)

    # Finalize writers
    ctw.close()
    if ffw is not None:
        ff_stdout, ff_stderr = ffw.close()

    # Also write timestamps as numpy file
    if hasattr(input_reader, 'timestamps') and timestamps_filename is not None:
        timestamps = np.concatenate(input_reader.timestamps)
        assert len(timestamps) >= ctw.frames_written
        np.save(timestamps_filename, timestamps[:ctw.frames_written])

    return ctw

def trace_chunked_tiffs(input_tiff_directory, h5_filename,
    n_trace_processes=4, expectedrows=1000000,
    ):
    """Trace tiffs that have been written to disk in parallel and stitch.

    input_tiff_directory : directory containing tiffs
    h5_filename : output HDF5 file
    n_trace_processes : how many simultaneous processes to use for tracing
    expectedrows : used to set up hdf5 file
    """
    WhiskiWrap.utils.probe_needed_commands()

    # Setup the result file
    setup_hdf5(h5_filename, expectedrows)

    # The tiffs have been written, figure out which they are
    tif_file_number_strings = wwutils.misc.apply_and_filter_by_regex(
        '^chunk(\d+).tif$', os.listdir(input_tiff_directory), sort=False)
    tif_full_filenames = [
        os.path.join(input_tiff_directory, 'chunk%s.tif' % fns)
        for fns in tif_file_number_strings]
    tif_file_numbers = list(map(int, tif_file_number_strings))
    tif_ordering = np.argsort(tif_file_numbers)
    tif_sorted_filenames = np.array(tif_full_filenames)[
        tif_ordering]
    tif_sorted_file_numbers = np.array(tif_file_numbers)[
        tif_ordering]

    # trace each
    print("Tracing")
    pool = mp.Pool(n_trace_processes)
    trace_res = pool.map(trace_chunk, tif_sorted_filenames)
    pool.close()

    # stitch
    print("Stitching")
    for chunk_start, chunk_name in zip(tif_sorted_file_numbers, tif_sorted_filenames):
        # Append each chunk to the hdf5 file
        fn = WhiskiWrap.utils.FileNamer.from_tiff_stack(chunk_name)
        append_whiskers_to_hdf5(
            whisk_filename=fn.whiskers,
            h5_filename=h5_filename,
            chunk_start=chunk_start)

def interleaved_read_trace_and_measure(input_reader, tiffs_to_trace_directory,
    sensitive=False,
    chunk_size=200, chunk_name_pattern='chunk%08d.tif',
    stop_after_frame=None, delete_tiffs=True,
    timestamps_filename=None, monitor_video=None,
    monitor_video_kwargs=None, write_monitor_ffmpeg_stderr_to_screen=False,
    h5_filename=None, frame_func=None,
    n_trace_processes=4, expectedrows=1000000,
    verbose=True, skip_stitch=False, face='right'
    ):
    """Read, write, and trace each chunk, one at a time.

    This is an alternative to first calling:
        write_video_as_chunked_tiffs
    And then calling
        trace_chunked_tiffs

    input_reader : Typically a PFReader or FFmpegReader
    tiffs_to_trace_directory : Location to write the tiffs
    sensitive: if False, use default. If True, lower MIN_SIGNAL
    chunk_size : frames per chunk
    chunk_name_pattern : how to name them
    stop_after_frame : break early, for debugging
    delete_tiffs : whether to delete tiffs after done tracing
    timestamps_filename : Where to store the timestamps
        Only vallid for PFReader input_reader
    monitor_video : filename for a monitor video
        If None, no monitor video will be written
    monitor_video_kwargs : kwargs to pass to FFmpegWriter for monitor
    write_monitor_ffmpeg_stderr_to_screen : whether to display
        output from ffmpeg writing instance
    h5_filename : hdf5 file to stitch whiskers information into
    frame_func : function to apply to each frame
        If 'invert', will apply 255 - frame
    n_trace_processes : number of simultaneous trace processes
    expectedrows : how to set up hdf5 file
    verbose : verbose
    skip_stitch : skip the stitching phase

    Returns: dict
        trace_pool_results : result of each call to trace
        monitor_ff_stderr, monitor_ff_stdout : results from monitor
            video ffmpeg instance
    """
    ## Set up kwargs
    if monitor_video_kwargs is None:
        monitor_video_kwargs = {}

    if frame_func == 'invert':
        frame_func = lambda frame: 255 - frame

    # Check commands
    WhiskiWrap.utils.probe_needed_commands()

    ## Initialize readers and writers
    if verbose:
        print("initalizing readers and writers")
    # Tiff writer
    ctw = WhiskiWrap.ChunkedTiffWriter(tiffs_to_trace_directory,
        chunk_size=chunk_size, chunk_name_pattern=chunk_name_pattern)

    # FFmpeg writer is initalized after first frame
    ffw = None

    # Setup the result file
    if not skip_stitch:
        setup_hdf5(h5_filename, expectedrows, measure=True)

    # Copy the parameters files
    copy_parameters_files(tiffs_to_trace_directory, sensitive=sensitive)

    ## Set up the worker pool
    # Pool of trace workers
    trace_pool = mp.Pool(n_trace_processes)

    # Keep track of results
    trace_pool_results = []
    # deleted_tiffs = []
    def log_result(result):
        print("Result logged:", result)  # Verify the callback
        trace_pool_results.append(result)

    ## Iterate over chunks
    out_of_frames = False
    nframe = 0

    # Init the iterator outside of the loop so that it persists
    iter_obj = input_reader.iter_frames()

    while not out_of_frames:
        # Get a chunk of frames
        if verbose:
            print("loading chunk of frames starting with", nframe)
        chunk_of_frames = []
        for frame in iter_obj:
            if frame_func is not None:
                frame = frame_func(frame)
            chunk_of_frames.append(frame)
            nframe = nframe + 1
            if stop_after_frame is not None and nframe >= stop_after_frame:
                break
            if len(chunk_of_frames) == chunk_size:
                break

        # Check if we ran out
        if len(chunk_of_frames) != chunk_size:
            out_of_frames = True

        ## Write tiffs
        # We do this synchronously to ensure that it happens before
        # the trace starts
        for frame in chunk_of_frames:
            ctw.write(frame)

        # Make sure the chunk was written, in case this is the last one
        # and we didn't reach chunk_size yet
        if len(chunk_of_frames) != chunk_size:
            ctw._write_chunk()
        assert ctw.count_unwritten_frames() == 0

        # Figure out which tiff file was just generated
        tif_filename = ctw.chunknames_written[-1]

        ## Start trace
        trace_pool.apply_async(trace_and_measure_chunk, args=(tif_filename, delete_tiffs, face),
            callback=log_result)

        ## Determine whether we can delete any tiffs
        #~ if delete_tiffs:
            #~ tiffs_to_delete = [
                #~ tpres['video_filename'] for tpres in trace_pool_results
                #~ if tpres['video_filename'] not in deleted_tiffs]
            #~ for filename in tiffs_to_delete:
                #~ if verbose:
                    #~ print "deleting", filename
                #~ os.remove(filename)

        ## Start monitor encode
        # This is also synchronous, otherwise the input buffer might fill up
        if monitor_video is not None:
            if ffw is None:
                ffw = WhiskiWrap.FFmpegWriter(monitor_video,
                    frame_width=frame.shape[1], frame_height=frame.shape[0],
                    write_stderr_to_screen=write_monitor_ffmpeg_stderr_to_screen,
                    **monitor_video_kwargs)
            for frame in chunk_of_frames:
                ffw.write(frame)

        ## Determine if we should pause
        while len(ctw.chunknames_written) > len(trace_pool_results) + 2 * n_trace_processes:
            print("waiting for tracing to catch up")
            time.sleep(10)

    ## Wait for trace to complete
    if verbose:
        print("done with reading and writing, just waiting for tracing")
    # Tell it no more jobs, so close when done
    trace_pool.close()

    # Wait for everything to finish
    trace_pool.join()

    ## Error check the tifs that were processed
    # Get the tifs we wrote, and the tifs we trace
    written_chunks = sorted(ctw.chunknames_written)
    traced_filenames = sorted([
        res['video_filename'] for res in trace_pool_results])

    # Check that they are the same
    if not np.all(np.array(written_chunks) == np.array(traced_filenames)):
        raise ValueError("not all chunks were traced")

    ## Extract the chunk numbers from the filenames
    # The tiffs have been written, figure out which they are
    split_traced_filenames = [os.path.split(fn)[1] for fn in traced_filenames]
    # tif_file_number_strings = wwutils.misc.apply_and_filter_by_regex(
    #     '^chunk(\d+).tif$', split_traced_filenames, sort=False)
    
    # Replace the format specifier with (\d+) in the pattern
    mod_chunk_name_pattern = re.sub(r'%\d+d', r'(\\d+)', chunk_name_pattern)
    mod_chunk_name_pattern = '^' + mod_chunk_name_pattern + '$'
    tif_file_number_strings = wwutils.misc.apply_and_filter_by_regex(
        mod_chunk_name_pattern, split_traced_filenames, sort=False)

    # Replace the format specifier with %s in the pattern
    tif_chunk_name_pattern = re.sub(r'%\d+d', r'%s', chunk_name_pattern)
    tif_full_filenames = [
        os.path.join(tiffs_to_trace_directory, tif_chunk_name_pattern % fns)
        for fns in tif_file_number_strings]
    
    tif_file_numbers = list(map(int, tif_file_number_strings))
    tif_ordering = np.argsort(tif_file_numbers)
    tif_sorted_filenames = np.array(tif_full_filenames)[
        tif_ordering]
    tif_sorted_file_numbers = np.array(tif_file_numbers)[
        tif_ordering]

    # stitch
    if not skip_stitch:
        print("Stitching")
        zobj = list(zip(tif_sorted_file_numbers, tif_sorted_filenames))
        for chunk_start, chunk_name in zobj:
            # Append each chunk to the hdf5 file
            fn = WhiskiWrap.utils.FileNamer.from_tiff_stack(chunk_name)
            append_whiskers_to_hdf5(
                whisk_filename=fn.whiskers,
		        measurements_filename = fn.measurements,
                h5_filename=h5_filename,
                chunk_start=chunk_start)

    # Finalize writers
    ctw.close()
    if ffw is not None:
        ff_stdout, ff_stderr = ffw.close()
    else:
        ff_stdout, ff_stderr = None, None

    # Also write timestamps as numpy file
    if hasattr(input_reader, 'timestamps') and timestamps_filename is not None:
        timestamps = np.concatenate(input_reader.timestamps)
        assert len(timestamps) >= ctw.frames_written
        np.save(timestamps_filename, timestamps[:ctw.frames_written])

    return {'trace_pool_results': trace_pool_results,
        'monitor_ff_stdout': ff_stdout,
        'monitor_ff_stderr': ff_stderr,
        'tif_sorted_file_numbers': tif_sorted_file_numbers,
        'tif_sorted_filenames': tif_sorted_filenames,
        }

def interleaved_split_trace_and_measure(input_reader, tiffs_to_trace_directory,
    sensitive=False,
    chunk_size=200, chunk_name_pattern='chunk%08d.tif',
    stop_after_frame=None, delete_tiffs=True,
    timestamps_filename=None, monitor_video=None,
    monitor_video_kwargs=None, write_monitor_ffmpeg_stderr_to_screen=False,
    output_filename=None, frame_func=None,
    n_trace_processes=4, expected_rows=1000000,
    verbose=True, skip_stitch=False, face='NA', classify=None, 
    summary_only=False, skip_existing=False, convert_chunks=False
    ):
    """Read, write, and trace each chunk, one at a time.

    This is an extension of interleaved_read_trace_and_measure for bilateral whisker tracking.
    The function needs to be called twice, once for each side of the face. 

    input_reader : Typically a PFReader or FFmpegReader
    tiffs_to_trace_directory : Location to write the tiffs
    sensitive: if False, use default. If True, lower MIN_SIGNAL
    chunk_size : frames per chunk
    chunk_name_pattern : how to name them
    stop_after_frame : break early, for debugging
    delete_tiffs : whether to delete tiffs after done tracing
    timestamps_filename : Where to store the timestamps
        Only valid for PFReader input_reader
    monitor_video : filename for a monitor video
        If None, no monitor video will be written
    monitor_video_kwargs : kwargs to pass to FFmpegWriter for monitor
    write_monitor_ffmpeg_stderr_to_screen : whether to display
        output from ffmpeg writing instance
    output_filename : hdf5, zarr or parquet file to stitch whiskers information into
    frame_func : function to apply to each frame
        If 'invert', will apply 255 - frame
    n_trace_processes : number of simultaneous trace processes
    expected_rows : how to set up hdf5 file
    verbose : verbose
    skip_stitch : skip the stitching phase
    face : 'left','right','top','bottom'
    classify : if not None, classify whiskers using passed arguments

    Returns: dict
        trace_pool_results : result of each call to trace
        monitor_ff_stderr, monitor_ff_stdout : results from monitor
            video ffmpeg instance
    """
    ## Set up kwargs
    if monitor_video_kwargs is None:
        monitor_video_kwargs = {}

    if frame_func == 'invert':
        frame_func = lambda frame: 255 - frame

    if frame_func == 'fliph':
        frame_func = lambda frame: np.fliplr(frame)
    elif frame_func == 'flipv':
        frame_func = lambda frame: np.flipud(frame)

    if frame_func == 'crop':
        crop_coord = input_reader.crop
        # crop_coord format is width:height:x:y
        frame_func = lambda frame: frame[crop_coord[3]:crop_coord[3] + crop_coord[1], crop_coord[2]:crop_coord[2] + crop_coord[0]]
        # reset crop field to None
        input_reader.crop = None

    # Check commands
    WhiskiWrap.utils.probe_needed_commands(paths=[whisk_path])

    ## Initialize readers and writers
    if verbose:
        print("Initializing readers and writers")

    # Tiff writer
    ctw = WhiskiWrap.ChunkedTiffWriter(tiffs_to_trace_directory,
        chunk_size=chunk_size, chunk_name_pattern=chunk_name_pattern)

    # FFmpeg writer is initialized after first frame
    ffw = None

    # Setup the result file for some formats
    if not skip_stitch:
        if output_filename.endswith('.hdf5'):
            setup_hdf5(output_filename, expected_rows, measure=True)
        elif output_filename.endswith('.zarr'):
            initialize_zarr(output_filename, (chunk_size,))
    
    # Copy the parameters files
    copy_parameters_files(tiffs_to_trace_directory, sensitive=sensitive)

    trace_pool_results = []

    if skip_existing:
    # Check if .whiskers or .measurements files already exist
        # Remove the extension from the chunk_name_pattern and replace %08d with a regular expression
        pattern = chunk_name_pattern.replace('.tif', r'(.whiskers|.measurements)').replace('%08d', r'\d{8}')
        existing_files = [f for f in os.listdir(tiffs_to_trace_directory) if re.match(pattern, f)]
        # Any whiskers or measurements files (less precise than the pattern):
        # existing_files = [f for f in os.listdir(tiffs_to_trace_directory) if f.endswith('.whiskers') or f.endswith('.measurements')]

        if len(existing_files) > 0:    
            print("Existing files found, skipping to stitching")
            # create sorted_file_numbers, sorted_filenames
            # Keep only the .whiskers files in the list
            split_traced_filenames = [os.path.split(fn)[1] for fn in existing_files]
            split_traced_filenames = [fn for fn in split_traced_filenames if fn.endswith('.whiskers')]
            full_filenames = [os.path.join(tiffs_to_trace_directory, fn) for fn in split_traced_filenames]
            # Get the file numbers
            chunk_name_pattern = r'_(\d+)\.(whiskers)$'
            file_number_strings = [re.search(chunk_name_pattern, fn).group(1) 
                                   for fn in split_traced_filenames if re.search(chunk_name_pattern, fn)]
            # Sort the filenames by the file numbers
            file_numbers = list(map(int, file_number_strings))
            ordering = np.argsort(file_numbers)
            sorted_filenames = np.array(full_filenames)[ordering]
            sorted_file_numbers = np.array(file_numbers)[ordering]
            
        else: 
            sorted_filenames = []
            sorted_file_numbers = []
            skip_existing = False

    if not skip_existing:
        ## Set up the worker pool
        # Pool of trace workers
        trace_pool = mp.Pool(n_trace_processes)

        # Keep track of results
        deleted_tiffs = []
        def log_result(result):
            print("Result logged:", result)
            trace_pool_results.append(result)

        ## Iterate over chunks
        out_of_frames = False
        nframe = 0

        # Init the iterator outside of the loop so that it persists
        iter_obj = input_reader.iter_frames()

        # Create temporary directory if output_filename ends with .zarr or. parquet
        if output_filename.endswith('.zarr') or output_filename.endswith('.parquet'):
            temp_dir = tempfile.mkdtemp()
        else:
            temp_dir = None

        if convert_chunks:
            #  Check what format to the chunks to based on the output file extension
            convert_chunks_to = output_filename.split('.')[-1]
                    
        while not out_of_frames:
            # Get a chunk of frames
            if verbose:
                print("Loading chunk of frames starting with", nframe)
            chunk_of_frames = []
            for frame in iter_obj:
                if frame_func is not None:
                    frame = frame_func(frame)
                chunk_of_frames.append(frame)
                nframe = nframe + 1
                if stop_after_frame is not None and nframe >= stop_after_frame:
                    break
                if len(chunk_of_frames) == chunk_size:
                    break

            # Check if we ran out
            if len(chunk_of_frames) != chunk_size:
                out_of_frames = True

            ## Write tiffs
            # We do this synchronously to ensure that it happens before
            # the trace starts
            for frame in chunk_of_frames:
                ctw.write(frame)

            # Make sure the chunk was written, in case this is the last one
            # and we didn't reach chunk_size yet
            if len(chunk_of_frames) != chunk_size:
                ctw._write_chunk()
            assert ctw.count_unwritten_frames() == 0

            # Figure out which tiff file was just generated
            tif_filename = ctw.chunknames_written[-1]

            # Start trace
            try:
                trace_pool.apply_async(trace_and_measure_chunk, args=(tif_filename, delete_tiffs, face, classify, temp_dir, convert_chunks_to), callback=log_result)
            except Exception as e:
                print("Error in apply_async:", e)

            ## Determine whether we can delete any tiffs
            #~ if delete_tiffs:
                #~ tiffs_to_delete = [
                    #~ tpres['video_filename'] for tpres in trace_pool_results
                    #~ if tpres['video_filename'] not in deleted_tiffs]
                #~ for filename in tiffs_to_delete:
                    #~ if verbose:
                        #~ print "deleting", filename
                    #~ os.remove(filename)

            ## Start monitor encode
            # This is also synchronous, otherwise the input buffer might fill up
            if monitor_video is not None:
                if ffw is None:
                    ffw = WhiskiWrap.FFmpegWriter(monitor_video,
                        frame_width=frame.shape[1], frame_height=frame.shape[0],
                        write_stderr_to_screen=write_monitor_ffmpeg_stderr_to_screen,
                        **monitor_video_kwargs)
                for frame in chunk_of_frames:
                    ffw.write(frame)

            ## Determine if we should pause
            # if len(ctw.chunknames_written) > len(trace_pool_results) + 2 * n_trace_processes:
            #     print("Waiting for tracing to catch up")
            #     #initialize time counter
            #     wait_t0 = time.time()
            #     while len(ctw.chunknames_written) > len(trace_pool_results) + 2 * n_trace_processes:
            #         # print dot every second
            #         if time.time() - wait_t0 > 1:
            #             print('.', end='')
                # print("")
            while len(ctw.chunknames_written) > len(trace_pool_results) + 2 * n_trace_processes:
                print("waiting for tracing to catch up")
                time.sleep(5)

        ## Wait for trace to complete
        if verbose:
            print("Done with reading and writing, just waiting for tracing")
        # Tell it no more jobs, so close when done
        trace_pool.close()

        # Wait for everything to finish
        trace_pool.join()

        ## Error check the tifs that were processed
        # Get the tifs we wrote, and the tifs we trace
        written_chunks = sorted(ctw.chunknames_written)
        traced_filenames = sorted([
            res['video_filename'] for res in trace_pool_results])

        # Check that they are the same
        if not np.all(np.array(written_chunks) == np.array(traced_filenames)):
            raise ValueError("Not all chunks were traced")

        ## Extract the chunk numbers from the filenames
        # The tiffs have been written, figure out which they are
        split_traced_filenames = [os.path.split(fn)[1] for fn in traced_filenames]
        
        # Replace the format specifier with (\d+) in the pattern
        mod_chunk_name_pattern = re.sub(r'%\d+d', r'(\\d+)', chunk_name_pattern)
        mod_chunk_name_pattern = '^' + mod_chunk_name_pattern + '$'
        tif_file_number_strings = wwutils.misc.apply_and_filter_by_regex(
            mod_chunk_name_pattern, split_traced_filenames, sort=False)

        # Replace the format specifier with %s in the pattern
        tif_chunk_name_pattern = re.sub(r'%\d+d', r'%s', chunk_name_pattern)
        tif_full_filenames = [
            os.path.join(tiffs_to_trace_directory, tif_chunk_name_pattern % fns)
            for fns in tif_file_number_strings]
        
        tif_file_numbers = list(map(int, tif_file_number_strings))
        tif_ordering = np.argsort(tif_file_numbers)
        tif_sorted_filenames = np.array(tif_full_filenames)[
            tif_ordering]
        tif_sorted_file_numbers = np.array(tif_file_numbers)[
            tif_ordering]
        
        sorted_file_numbers = tif_sorted_file_numbers
        sorted_filenames = tif_sorted_filenames

    # stitch
    if not skip_stitch:
        print("Stitching")
        zobj = list(zip(sorted_file_numbers, sorted_filenames))
        # if output_filename contains h5
        if output_filename.endswith('.hdf5'):
            for chunk_start, chunk_name in zobj:
                # Append each chunk to the hdf5 file
                if chunk_name.endswith('.tif'):
                    fn = WhiskiWrap.utils.FileNamer.from_tiff_stack(chunk_name)
                elif chunk_name.endswith('.whiskers'):
                    fn = WhiskiWrap.utils.FileNamer.from_whiskers(chunk_name)
                append_whiskers_to_hdf5(
                    whisk_filename=fn.whiskers,
                    measurements_filename = fn.measurements,
                    h5_filename=output_filename,
                    chunk_start=chunk_start,
                    summary_only=summary_only,
                    face_side=face)
                
        elif output_filename.endswith('.parquet'):
            try:
                # Check whether the stitched file has been successfully created previously
                if not os.path.exists(output_filename):
                    # Then check if the chunk files need to be created first
                    if 'temp_dir' not in locals() or \
                        temp_dir is None or \
                        not os.path.exists(temp_dir) or \
                        len(os.listdir(temp_dir)) != len(sorted_filenames):
                            # Need to create the chunk files first before merging
                            temp_dir = tempfile.mkdtemp()
                            # Prepare the arguments for parallel processing
                            args_list = [(whiskers_file, extract_chunk_number(whiskers_file), face, temp_dir) for whiskers_file in sorted_filenames]
                            with mp.Pool(n_trace_processes) as parquet_pool:
                                parquet_pool.starmap(
                                    process_and_write_parquet, 
                                    args_list
                                )
                            
                            # for whiskers_file in sorted_filenames:
                            #     chunk_start = extract_chunk_number(whiskers_file)
                            #     process_and_write_parquet(whiskers_file, chunk_start=chunk_start, face_side=face, temp_dir=temp_dir)
                            
                    # Merge all chunk Parquet files into the final output Parquet file
                    merge_parquet_files(temp_dir, output_filename)
            finally:
                # Clean up the temporary directory if it exists
                if 'temp_dir' in locals() and temp_dir is not None:
                    shutil.rmtree(temp_dir)
                
        elif output_filename.endswith('.zarr'):
            # Direct parallel writing to Zarr with consolidated metadata
            with mp.Pool() as pool:
                pool.starmap(
                    process_and_write_zarr, 
                    [(fn, chunk_start, output_filename) for chunk_start, fn in zobj]
                )
            # Consolidate Zarr metadata
            consolidate_zarr_metadata(output_filename)
        # elif output_filename.endswith('.zarr'): 
        #     append_whiskers_to_zarr(
        #         whisk_filename=fn.whiskers,
        #         measurements_filename=fn.measurements,
        #         zarr_filename=output_filename,
        #         chunk_start=chunk_start,
        #         summary_only=summary_only)

    # Finalize writers
    ctw.close()
    if ffw is not None:
        ff_stdout, ff_stderr = ffw.close()
    else:
        ff_stdout, ff_stderr = None, None

    # Also write timestamps as numpy file
    if hasattr(input_reader, 'timestamps') and timestamps_filename is not None:
        timestamps = np.concatenate(input_reader.timestamps)
        assert len(timestamps) >= ctw.frames_written
        np.save(timestamps_filename, timestamps[:ctw.frames_written])

    return {'trace_pool_results': trace_pool_results,
        'monitor_ff_stdout': ff_stdout,
        'monitor_ff_stderr': ff_stderr,
        'tif_sorted_file_numbers': sorted_file_numbers,
        'tif_sorted_filenames': sorted_filenames,
        }

def interleaved_reading_and_tracing(input_reader, tiffs_to_trace_directory,
    sensitive=False,
    chunk_size=200, chunk_name_pattern='chunk%08d.tif',
    stop_after_frame=None, delete_tiffs=True,
    timestamps_filename=None, monitor_video=None,
    monitor_video_kwargs=None, write_monitor_ffmpeg_stderr_to_screen=False,
    h5_filename=None, frame_func=None,
    n_trace_processes=4, expectedrows=1000000,
    verbose=True, skip_stitch=False,
    ):
    """Read, write, and trace each chunk, one at a time.

    This is an alternative to first calling:
        write_video_as_chunked_tiffs
    And then calling
        trace_chunked_tiffs

    input_reader : Typically a PFReader or FFmpegReader
    tiffs_to_trace_directory : Location to write the tiffs
    sensitive: if False, use default. If True, lower MIN_SIGNAL
    chunk_size : frames per chunk
    chunk_name_pattern : how to name them
    stop_after_frame : break early, for debugging
    delete_tiffs : whether to delete tiffs after done tracing
    timestamps_filename : Where to store the timestamps
        Only vallid for PFReader input_reader
    monitor_video : filename for a monitor video
        If None, no monitor video will be written
    monitor_video_kwargs : kwargs to pass to FFmpegWriter for monitor
    write_monitor_ffmpeg_stderr_to_screen : whether to display
        output from ffmpeg writing instance
    h5_filename : hdf5 file to stitch whiskers information into
    frame_func : function to apply to each frame
        If 'invert', will apply 255 - frame
    n_trace_processes : number of simultaneous trace processes
    expectedrows : how to set up hdf5 file
    verbose : verbose
    skip_stitch : skip the stitching phase

    Returns: dict
        trace_pool_results : result of each call to trace
        monitor_ff_stderr, monitor_ff_stdout : results from monitor
            video ffmpeg instance
    """
    ## Set up kwargs
    if monitor_video_kwargs is None:
        monitor_video_kwargs = {}

    if frame_func == 'invert':
        frame_func = lambda frame: 255 - frame

    # Check commands
    WhiskiWrap.utils.probe_needed_commands()

    ## Initialize readers and writers
    if verbose:
        print("initalizing readers and writers")
    # Tiff writer
    ctw = WhiskiWrap.ChunkedTiffWriter(tiffs_to_trace_directory,
        chunk_size=chunk_size, chunk_name_pattern=chunk_name_pattern)

    # FFmpeg writer is initalized after first frame
    ffw = None

    # Setup the result file
    if not skip_stitch:
        setup_hdf5(h5_filename, expectedrows)

    # Copy the parameters files
    copy_parameters_files(tiffs_to_trace_directory, sensitive=sensitive)

    ## Set up the worker pool
    # Pool of trace workers
    trace_pool = mp.Pool(n_trace_processes)

    # Keep track of results
    trace_pool_results = []
    deleted_tiffs = []
    def log_result(result):
        trace_pool_results.append(result)

    ## Iterate over chunks
    out_of_frames = False
    nframe = 0

    # Init the iterator outside of the loop so that it persists
    iter_obj = input_reader.iter_frames()

    while not out_of_frames:
        # Get a chunk of frames
        if verbose:
            print("loading chunk of frames starting with", nframe)
        chunk_of_frames = []
        for frame in iter_obj:
            if frame_func is not None:
                frame = frame_func(frame)
            chunk_of_frames.append(frame)
            nframe = nframe + 1
            if stop_after_frame is not None and nframe >= stop_after_frame:
                break
            if len(chunk_of_frames) == chunk_size:
                break

        # Check if we ran out
        if len(chunk_of_frames) != chunk_size:
            out_of_frames = True

        ## Write tiffs
        # We do this synchronously to ensure that it happens before
        # the trace starts
        for frame in chunk_of_frames:
            ctw.write(frame)

        # Make sure the chunk was written, in case this is the last one
        # and we didn't reach chunk_size yet
        if len(chunk_of_frames) != chunk_size:
            ctw._write_chunk()
        assert ctw.count_unwritten_frames() == 0

        # Figure out which tiff file was just generated
        tif_filename = ctw.chunknames_written[-1]

        ## Start trace
        trace_pool.apply_async(trace_chunk, args=(tif_filename, delete_tiffs),
            callback=log_result)

        ## Determine whether we can delete any tiffs
        #~ if delete_tiffs:
            #~ tiffs_to_delete = [
                #~ tpres['video_filename'] for tpres in trace_pool_results
                #~ if tpres['video_filename'] not in deleted_tiffs]
            #~ for filename in tiffs_to_delete:
                #~ if verbose:
                    #~ print "deleting", filename
                #~ os.remove(filename)

        ## Start monitor encode
        # This is also synchronous, otherwise the input buffer might fill up
        if monitor_video is not None:
            if ffw is None:
                ffw = WhiskiWrap.FFmpegWriter(monitor_video,
                    frame_width=frame.shape[1], frame_height=frame.shape[0],
                    write_stderr_to_screen=write_monitor_ffmpeg_stderr_to_screen,
                    **monitor_video_kwargs)
            for frame in chunk_of_frames:
                ffw.write(frame)

        ## Determine if we should pause
        while len(ctw.chunknames_written) > len(trace_pool_results) + 2 * n_trace_processes:
            print("waiting for tracing to catch up")
            time.sleep(10)

    ## Wait for trace to complete
    if verbose:
        print("done with reading and writing, just waiting for tracing")
    # Tell it no more jobs, so close when done
    trace_pool.close()

    # Wait for everything to finish
    trace_pool.join()

    ## Error check the tifs that were processed
    # Get the tifs we wrote, and the tifs we trace
    written_chunks = sorted(ctw.chunknames_written)
    traced_filenames = sorted([
        res['video_filename'] for res in trace_pool_results])

    # Check that they are the same
    if not np.all(np.array(written_chunks) == np.array(traced_filenames)):
        raise ValueError("not all chunks were traced")

    ## Extract the chunk numbers from the filenames
    # The tiffs have been written, figure out which they are
    split_traced_filenames = [os.path.split(fn)[1] for fn in traced_filenames]
    tif_file_number_strings = wwutils.misc.apply_and_filter_by_regex(
        '^chunk(\d+).tif$', split_traced_filenames, sort=False)
    tif_full_filenames = [
        os.path.join(tiffs_to_trace_directory, 'chunk%s.tif' % fns)
        for fns in tif_file_number_strings]
    tif_file_numbers = list(map(int, tif_file_number_strings))
    tif_ordering = np.argsort(tif_file_numbers)
    tif_sorted_filenames = np.array(tif_full_filenames)[
        tif_ordering]
    tif_sorted_file_numbers = np.array(tif_file_numbers)[
        tif_ordering]

    # stitch
    if not skip_stitch:
        print("Stitching")
        zobj = list(zip(tif_sorted_file_numbers, tif_sorted_filenames))
        for chunk_start, chunk_name in zobj:
            # Append each chunk to the hdf5 file
            fn = WhiskiWrap.utils.FileNamer.from_tiff_stack(chunk_name)
            append_whiskers_to_hdf5(
                whisk_filename=fn.whiskers,
                h5_filename=h5_filename,
                chunk_start=chunk_start)

    # Finalize writers
    ctw.close()
    if ffw is not None:
        ff_stdout, ff_stderr = ffw.close()
    else:
        ff_stdout, ff_stderr = None, None

    # Also write timestamps as numpy file
    if hasattr(input_reader, 'timestamps') and timestamps_filename is not None:
        timestamps = np.concatenate(input_reader.timestamps)
        assert len(timestamps) >= ctw.frames_written
        np.save(timestamps_filename, timestamps[:ctw.frames_written])

    return {'trace_pool_results': trace_pool_results,
        'monitor_ff_stdout': ff_stdout,
        'monitor_ff_stderr': ff_stderr,
        'tif_sorted_file_numbers': tif_sorted_file_numbers,
        'tif_sorted_filenames': tif_sorted_filenames,
        }

def compress_pf_to_video(input_reader, chunk_size=200, stop_after_frame=None,
    timestamps_filename=None, monitor_video=None, monitor_video_kwargs=None,
    write_monitor_ffmpeg_stderr_to_screen=False, frame_func=None, verbose=True,
    ):
    """Read modulated data and compress to video

    Adapted from interleaved_reading_and_tracing

    input_reader : typically a PFReader
    chunk_size : frames per chunk
    stop_after_frame : break early, for debugging
    timestamps_filename : Where to store the timestamps
        Only valid for PFReader input_reader
    monitor_video : filename for a monitor video
        If None, no monitor video will be written
    monitor_video_kwargs : kwargs to pass to FFmpegWriter for monitor
        If None, the default is {'qp': 15} for a high-fidelity compression
        that is still ~6x smaller than lossless.
    write_monitor_ffmpeg_stderr_to_screen : whether to display
        output from ffmpeg writing instance
    frame_func : function to apply to each frame
        If 'invert', will apply 255 - frame
    verbose : verbose

    Returns: dict
        monitor_ff_stderr, monitor_ff_stdout : results from monitor
            video ffmpeg instance
    """
    ## Set up kwargs
    if monitor_video_kwargs is None:
        monitor_video_kwargs = {'qp': 15}

    if frame_func == 'invert':
        frame_func = lambda frame: 255 - frame

    ## Initialize readers and writers
    if verbose:
        print("initalizing readers and writers")

    # FFmpeg writer is initalized after first frame
    ffw = None

    ## Iterate over chunks
    out_of_frames = False
    nframe = 0
    nframes_written = 0

    # Init the iterator outside of the loop so that it persists
    iter_obj = input_reader.iter_frames()

    while not out_of_frames:
        # Get a chunk of frames
        if verbose:
            print("loading chunk of frames starting with", nframe)
        chunk_of_frames = []
        for frame in iter_obj:
            if frame_func is not None:
                frame = frame_func(frame)
            chunk_of_frames.append(frame)
            nframe = nframe + 1
            if stop_after_frame is not None and nframe >= stop_after_frame:
                break
            if len(chunk_of_frames) == chunk_size:
                break

        # Check if we ran out
        if len(chunk_of_frames) != chunk_size:
            out_of_frames = True

        ## Start monitor encode
        # This is also synchronous, otherwise the input buffer might fill up
        if monitor_video is not None:
            if ffw is None:
                ffw = WhiskiWrap.FFmpegWriter(monitor_video,
                    frame_width=frame.shape[1], frame_height=frame.shape[0],
                    write_stderr_to_screen=write_monitor_ffmpeg_stderr_to_screen,
                    **monitor_video_kwargs)
            for frame in chunk_of_frames:
                ffw.write(frame)
                nframes_written = nframes_written + 1

    # Finalize writers
    if ffw is not None:
        ff_stdout, ff_stderr = ffw.close()
    else:
        ff_stdout, ff_stderr = None, None

    # Also write timestamps as numpy file
    if hasattr(input_reader, 'timestamps') and timestamps_filename is not None:
        timestamps = np.concatenate(input_reader.timestamps)

        # These assertions only make sense if we wrote the whole file
        if stop_after_frame is None:
            assert len(timestamps) == nframes_written
            assert nframes_written == nframe

        # Save timestamps
        np.save(timestamps_filename, timestamps)

    return {
        'monitor_ff_stdout': ff_stdout,
        'monitor_ff_stderr': ff_stderr,
    }

def measure_chunk_star(args):
    return measure_chunk(*args)

def read_whiskers_hdf5_summary(filename):
    """Reads and returns the `summary` table in an HDF5 file"""
    with tables.open_file(filename) as fi:
        # Check whether the summary table exists, or updated_summary
        if '/summary' in fi:
            summary = pd.DataFrame.from_records(fi.root.summary.read())
        elif '/updated_summary' in fi:
            try:
                # Assuming a group
                summary = pd.DataFrame.from_records(fi.root.updated_summary.read())
            except:
                # Then assuming a table
                table_node = fi.get_node('/updated_summary/table')
                summary = pd.DataFrame.from_records(table_node.read())
        else:
            raise ValueError("no summary table found")

    return summary

def read_whiskers_measurements(filenames):
    # Loads all the whiskers/measurements files and returns an aggregated dataframe

    # Check that it is a list
    if not isinstance(filenames, list):
        filenames = [filenames]

    # Load the measurements
    all_measurements = []
    for filename in filenames:
        #  strip the extension and replace with .whiskers
        if not filename.endswith('.whiskers'):
            filename = filename.replace(filename.split('.')[-1], 'whiskers')
        # get the whisker data
        measurements = WhiskiWrap.read_whisker_data(filename, 'df')
        # append the dataframe to the list
        all_measurements.append(measurements)

    # Concatenate the dataframes
    all_measurements = pd.concat(all_measurements)

    return all_measurements

class PFReader:
    """Reads photonfocus modulated data stored in matlab files"""
    def __init__(self, input_directory, n_threads=4, verbose=True,
        error_on_unsorted_filetimes=True):
        """Initialize a new reader.

        input_directory : where the mat files are
            They are assumed to have a format like img10.mat, etc.
            They should contain variables called 'img' (a 4d array of
            modulated frames) and 't' (timestamps of each frame).
        n_threads : sent to pfDoubleRate_SetNrOfThreads

        error_on_unsorted_filetimes : bool
            Whether to raise an error if the modification times of the
            matfiles are not in sorted order, which typically happens if
            something has gone wrong (but could just be that the file times
            weren't preserved)
        """
        self.input_directory = input_directory
        self.verbose = verbose

        ## Load the libraries
        # boost_thread needs boost_system
        # I think it used to be able to find boost_system without this line
        libboost_system = ctypes.cdll.LoadLibrary(
            '/usr/local/lib/libboost_system.so.1.50.0')

        # Load boost_thread
        libboost_thread = ctypes.cdll.LoadLibrary(
            '/usr/local/lib/libboost_thread.so')

        # Load the pf_lib (which requires boost_thread)
        self.pf_lib = ctypes.cdll.LoadLibrary(LIB_DOUBLERATE)
        self.demod_func = self.pf_lib['pfDoubleRate_DeModulateImage']

        # Set the number of threads
        self.pf_lib['pfDoubleRate_SetNrOfThreads'](n_threads)

        # Find all the imgN.mat files in the input directory
        self.matfile_number_strings = wwutils.misc.apply_and_filter_by_regex(
            '^img(\d+)\.mat$', os.listdir(self.input_directory), sort=False)
        self.matfile_names = [
            os.path.join(self.input_directory, 'img%s.mat' % fns)
            for fns in self.matfile_number_strings]
        self.matfile_numbers = list(map(int, self.matfile_number_strings))
        self.matfile_ordering = np.argsort(self.matfile_numbers)

        # Sort the names and numbers
        self.sorted_matfile_names = np.array(self.matfile_names)[
            self.matfile_ordering]
        self.sorted_matfile_numbers = np.array(self.matfile_numbers)[
            self.matfile_ordering]

        # Error check the file times
        filetimes = np.array([
            wwutils.misc.get_file_time(filename)
            for filename in self.sorted_matfile_names])
        if (np.diff(filetimes) < 0).any():
            if error_on_unsorted_filetimes:
                raise IOError("unsorted matfiles")
            else:
                print("warning: unsorted matfiles")

        # Create variable to store timestamps
        self.timestamps = []
        self.n_frames_read = 0

        # Set these values once they are read from the file
        self.frame_height = None
        self.frame_width = None

    def iter_frames(self):
        """Yields frames as they are read and demodulated.

        Iterates through the matfiles in order, demodulates each frame,
        and yields them one at a time.

        The chunk of timestamps from each matfile is appended to the list
        self.timestamps, so that self.timestamps is a list of arrays. There
        will be more timestamps than read frames until the end of the chunk.

        Also sets self.frame_height and self.frame_width and checks that
        they are consistent over the session.
        """
        # Iterate through matfiles and load
        for matfile_name in self.sorted_matfile_names:
            if self.verbose:
                print(("loading %s" % matfile_name))

            # Load the raw data
            # This is the slowest step
            matfile_load = scipy.io.loadmat(matfile_name)
            matfile_t = matfile_load['t'].flatten()
            matfile_modulated_data = matfile_load['img'].squeeze()
            assert matfile_modulated_data.ndim == 3 # height, width, time

            # Append the timestamps
            self.timestamps.append(matfile_t)

            # Extract shape
            n_frames = len(matfile_t)
            assert matfile_modulated_data.shape[-1] == n_frames
            modulated_frame_width = matfile_modulated_data.shape[1]
            frame_height = matfile_modulated_data.shape[0]

            if self.verbose:
                print(("loaded %d modulated frames @ %dx%d" % (n_frames,
                    modulated_frame_width, frame_height)))

            # Find the demodulated width
            # This can be done by pfDoubleRate_GetDeModulatedWidth
            # but I don't understand what it's doing.
            if modulated_frame_width == 416:
                demodulated_frame_width = 800
            elif modulated_frame_width == 332:
                demodulated_frame_width = 640
            else:
                raise ValueError("unknown modulated width: %d" %
                    modulated_frame_width)

            # Store the frame sizes as necessary
            if self.frame_width is None:
                self.frame_width = demodulated_frame_width
            if self.frame_width != demodulated_frame_width:
                raise ValueError("inconsistent frame widths")
            if self.frame_height is None:
                self.frame_height = frame_height
            if self.frame_height != frame_height:
                raise ValueError("inconsistent frame heights")

            # Create a buffer for the result of each frame
            demodulated_frame_buffer = ctypes.create_string_buffer(
                frame_height * demodulated_frame_width)

            # Iterate over frames
            for n_frame in range(n_frames):
                if self.verbose and np.mod(n_frame, 200) == 0:
                    print(("iterator has reached frame %d" % n_frame))

                # Convert image to char array
                # Ideally we would use a buffer here instead of a copy in order
                # to save time. But Matlab data comes back in Fortran order
                # instead of C order, so this is not possible.
                frame_charry = ctypes.c_char_p(
                    matfile_modulated_data[:, :, n_frame].tobytes())
                #~ frame_charry = ctypes.c_char_p(
                    #~ bytes(matfile_modulated_data[:, :, n_frame].data))

                # Run
                self.demod_func(demodulated_frame_buffer, frame_charry,
                    demodulated_frame_width, frame_height,
                    modulated_frame_width)

                # Extract the result from the buffer
                demodulated_frame = np.fromstring(demodulated_frame_buffer,
                    dtype=np.uint8).reshape(
                    frame_height, demodulated_frame_width)

                self.n_frames_read = self.n_frames_read + 1

                yield demodulated_frame

        if self.verbose:
            print("iterator is empty")

    def close(self):
        """Currently does nothing"""
        pass

    def isclosed(self):
        return True

class ChunkedTiffWriter:
    """Writes frames to a series of tiff stacks"""
    def __init__(self, output_directory, chunk_size=200, chunk_name_pattern='chunk%08d.tif', compress=False):
        """Initialize a new chunked tiff writer.

        output_directory : where to write the chunks
        chunk_size : frames per chunk
        chunk_name_pattern : how to name the chunk, using the number of
            the first frame in it
        """
        self.output_directory = output_directory
        self.chunk_size = chunk_size
        self.chunk_name_pattern = chunk_name_pattern
        self.compress = compress

        # Initialize counters so we know what frame and chunk we're on
        self.frames_written = 0
        self.frame_buffer = []
        self.chunknames_written = []

    def write(self, frame):
        """Buffered write frame to tiff stacks"""
        # Append to buffer
        self.frame_buffer.append(frame)

        # Write chunk if buffer is full
        if len(self.frame_buffer) == self.chunk_size:
            self._write_chunk()

    def _write_chunk(self):
        if len(self.frame_buffer) != 0:
            # Form the chunk
            chunk = np.array(self.frame_buffer)

            # Name it
            chunkname = os.path.join(self.output_directory,
                self.chunk_name_pattern % self.frames_written)

            # Write it
            if self.compress:
                tifffile.imsave(chunkname, chunk, compression='lzw')
            else:
                tifffile.imsave(chunkname, chunk, compression=False)

            # Update the counter
            self.frames_written += len(self.frame_buffer)
            self.frame_buffer = []

            # Update the list of written chunks
            self.chunknames_written.append(chunkname)

    def count_unwritten_frames(self):
        """Returns the number of buffered, unwritten frames"""
        return len(self.frame_buffer)

    def close(self):
        """Finish writing any final unfinished chunk"""
        self._write_chunk()

class FFmpegReader:
    """Reads frames from a video file using ffmpeg process"""
    def __init__(self, input_filename, pix_fmt='gray', bufsize=10**9,
        duration=None, start_frame_time=None, start_frame_number=None, crop=None,
        write_stderr_to_screen=False, vsync='drop'):
        """Initialize a new reader

        input_filename : name of file
        pix_fmt : used to format the raw data coming from ffmpeg into
            a numpy array
        bufsize : probably not necessary because we read one frame at a time
        duration : duration of video to read (-t parameter)
        start_frame_time, start_frame_number : -ss parameter
            Parsed using wwutils.video.ffmpeg_frame_string
        crop : crop the video using the ffmpeg crop filter. Format is width:height:x:y
        write_stderr_to_screen : if True, writes to screen, otherwise to
            /dev/null
        """
        self.input_filename = input_filename

        # Get params
        self.frame_width, self.frame_height, self.frame_rate = \
            wwutils.video.get_video_params(input_filename, crop=crop)

        # Set up pix_fmt
        if pix_fmt == 'gray':
            self.bytes_per_pixel = 1
        elif pix_fmt == 'rgb24':
            self.bytes_per_pixel = 3
        else:
            raise ValueError("can't handle pix_fmt:", pix_fmt)
        self.read_size_per_frame = self.bytes_per_pixel * \
            self.frame_width * self.frame_height

        # Create the command
        command = ['ffmpeg']

        # Add ss string
        if start_frame_time is not None or start_frame_number is not None:
            ss_string = wwutils.video.ffmpeg_frame_string(input_filename,
                frame_time=start_frame_time, frame_number=start_frame_number)
            command += [
                '-ss', ss_string]

        # Add input file
        command += [
            '-i', input_filename,
            '-vsync', vsync,
            '-f', 'image2pipe',
            '-pix_fmt', pix_fmt]
        
        # Add crop string. The crop argument is a list
        if crop is not None:
            self.crop = crop
        else:
            self.crop = None
        #     command += [
        #         '-vf', 'crop=%d:%d:%d:%d' % tuple(crop)]

        # Add duration string
        if duration is not None:
            command += [
                '-t', str(duration),]

        # Add vcodec for pipe
        command += [
            '-vcodec', 'rawvideo', '-']

        # To store result
        self.n_frames_read = 0

        # stderr
        if write_stderr_to_screen:
            stderr = None
        else:
            stderr = open(os.devnull, 'w')

        # Init the pipe
        # We set stderr to null so it doesn't fill up screen or buffers
        # And we set stdin to PIPE to keep it from breaking our STDIN
        self.ffmpeg_proc = subprocess.Popen(command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=stderr,
            bufsize=bufsize)

    def iter_frames(self):
        """Yields one frame at a time

        When done: terminates ffmpeg process, and stores any remaining
        results in self.leftover_bytes and self.stdout and self.stderr

        It might be worth writing this as a chunked reader if this is too
        slow. Also we need to be able to seek through the file.
        """
        # Read this_chunk, or as much as we can
        while(True):
            raw_image = self.ffmpeg_proc.stdout.read(self.read_size_per_frame)

            # if self.crop is not None:
                # import matplotlib.pyplot as plt
                # # plot raw image, full and cropped (cropp coordinates format is width:height:x:y)
                # raw_image_full = np.frombuffer(raw_image, dtype='uint8').reshape((self.frame_height, self.frame_width, self.bytes_per_pixel))
                # raw_image_cropped = raw_image_full[self.crop[3]:self.crop[3]+self.crop[1], self.crop[2]:self.crop[2]+self.crop[0], :]
                # plt.figure()
                # plt.subplot(1,2,1)
                # plt.imshow(raw_image_full)
                # plt.subplot(1,2,2)
                # plt.imshow(raw_image_cropped)
                # plt.show()

            # check if we ran out of frames
            if len(raw_image) != self.read_size_per_frame:
                self.leftover_bytes = raw_image
                self.close()
                return

            # Convert to array, flatten and squeeze (and crop if required, although it is slightly faster to crop in the calling function with frame_func)
            if self.crop is None:
                frame = np.frombuffer(raw_image, dtype='uint8').reshape((self.frame_height, self.frame_width, self.bytes_per_pixel)).squeeze()
            else:
                frame = np.frombuffer(raw_image, dtype='uint8').reshape((self.frame_height, self.frame_width, self.bytes_per_pixel)).squeeze()[self.crop[3]:self.crop[3]+self.crop[1], self.crop[2]:self.crop[2]+self.crop[0]]

            # Update
            self.n_frames_read = self.n_frames_read + 1

            # Yield
            yield frame

    def close(self):
        """Closes the process"""
        # Need to terminate in case there is more data but we don't
        # care about it
        # But if it's already terminated, don't try to terminate again
        if self.ffmpeg_proc.returncode is None:
            self.ffmpeg_proc.terminate()

            # Extract the leftover bits
            self.stdout, self.stderr = self.ffmpeg_proc.communicate()

        return self.ffmpeg_proc.returncode

    def isclosed(self):
        if hasattr(self.ffmpeg_proc, 'returncode'):
            return self.ffmpeg_proc.returncode is not None
        else:
            # Never even ran? I guess this counts as closed.
            return True

class FFmpegWriter:
    """Writes frames to an ffmpeg compression process"""
    def __init__(self, output_filename, frame_width, frame_height,
        output_fps=30, vcodec='libx264', qp=15, preset='medium',
        input_pix_fmt='gray', output_pix_fmt='yuv420p',
        write_stderr_to_screen=False):
        """Initialize the ffmpeg writer

        output_filename : name of output file
        frame_width, frame_height : Used to inform ffmpeg how to interpret
            the data coming in the stdin pipe
        output_fps : frame rate
        input_pix_fmt : Tell ffmpeg how to interpret the raw data on the pipe
            This should match the output generated by frame.tostring()
        output_pix_fmt : pix_fmt of the output
        crf : quality. 0 means lossless
        preset : speed/compression tradeoff
        write_stderr_to_screen :
            If True, writes ffmpeg's updates to screen
            If False, writes to /dev/null

        With old versions of ffmpeg (jon-severinsson) I was not able to get
        truly lossless encoding with libx264. It was clamping the luminances to
        16..235. Some weird YUV conversion?
        '-vf', 'scale=in_range=full:out_range=full' seems to help with this
        In any case it works with new ffmpeg. Also the codec ffv1 will work
        but is slightly larger filesize.
        """
        # Open an ffmpeg process
        cmdstring = ('ffmpeg',
            '-y', '-r', '%d' % output_fps,
            '-s', '%dx%d' % (frame_width, frame_height), # size of image string
            '-pix_fmt', input_pix_fmt,
            '-f', 'rawvideo',  '-i', '-', # raw video from the pipe
            '-pix_fmt', output_pix_fmt,
            '-vcodec', vcodec,
            '-qp', str(qp),
            '-preset', preset,
            output_filename) # output encoding

        if write_stderr_to_screen:
            self.ffmpeg_proc = subprocess.Popen(cmdstring, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE)
        else:
            self.ffmpeg_proc = subprocess.Popen(cmdstring, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=open('/dev/null', 'w'))

    def write(self, frame):
        """Write a frame to the ffmpeg process"""
        self.ffmpeg_proc.stdin.write(frame.tostring())

    def write_bytes(self, bytestring):
        self.ffmpeg_proc.stdin.write(bytestring)

    def close(self):
        """Closes the ffmpeg process and returns stdout, stderr"""
        return self.ffmpeg_proc.communicate()

