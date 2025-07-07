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
# import json
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
from wwutils import video
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
    # Use imwrite for newer tifffile versions, fallback to imsave for older versions
    if hasattr(tifffile, 'imwrite'):
        tifffile.imwrite(os.path.join(directory, chunkname), chunk, compression=None)
    else:
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
    whiskers_file = wwutils.utils.FileNamer.from_video(video_filename).whiskers
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
    measurements_file = wwutils.utils.FileNamer.from_whiskers(whiskers_filename).measurements
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
    whiskers_file = wwutils.utils.FileNamer.from_video(video_filename).whiskers
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
    measurements_file = wwutils.utils.FileNamer.from_video(video_filename).measurements
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
        measurements_file = wwutils.utils.FileNamer.from_video(video_filename).measurements
        
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

            if pixels_y_list:
                zarr_file['pixels_y'].append(pixels_y_list)
                zarr_file['pixels_y_indices'].append(pixels_y_indices_list)            
                
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

\nfrom .io import PFReader, ChunkedTiffWriter, FFmpegReader, FFmpegWriter
from .pipeline import pipeline_trace, write_video_as_chunked_tiffs, trace_chunked_tiffs, interleaved_read_trace_and_measure, interleaved_split_trace_and_measure, interleaved_trace_and_measure, compress_pf_to_video, measure_chunk_star, read_whiskers_hdf5_summary, read_whiskers_measurements, stitch_h5_to_parquet
