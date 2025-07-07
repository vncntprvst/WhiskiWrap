"""
This script combines whiskers and measurement files into a single file (formats: csv, parquet, hdf5, zarr).

Example usage:
python combine_sides.py /path/to/input_dir -b base_name -f csv -od /path/to/output_dir
python combine_sides.py /home/wanglab/data/whisker_asym/sc012/test/WT -b sc012_0119_001 -f zarr -od /home/wanglab/data/whisker_asym/sc012/test
"""
    
import os
import glob
import re
import argparse
import pandas as pd
import numpy as np
import tables
import pyarrow.parquet as pq
import h5py
import json
import time
import logging
from typing import List
import multiprocessing as mp
import tempfile

# Try to import WhiskiWrap for .whiskers file support
try:
    import WhiskiWrap as ww
    from WhiskiWrap.load_whisker_data import get_summary
    WHISKIWRAP_AVAILABLE = True
except ImportError:
    WHISKIWRAP_AVAILABLE = False
    ww = None
    get_summary = None
    logging.warning("WhiskiWrap not available. .whiskers file support will be limited.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define sides
default_sides = ['left', 'right', 'top', 'bottom']

def get_files(input_dir: str):
    """
    Get whisker tracking files from the input directory.
    Return the whiskers and measurement files with identified sides.
    """
    wt_formats = ['whiskers', 'hdf5', 'parquet']
    wt_files = []

    # Search for files in the available formats
    for wt_format in wt_formats:
        wt_files = glob.glob(os.path.join(input_dir, f'*.{wt_format}'))
        if wt_files:
            break

    if not wt_files:
        logging.error("No whiskers files found in input directory")
        return [], [], []

    # Identify the sides from the file names
    sides = [side for side in default_sides if any(side in f for f in wt_files)]

    if wt_format == 'whiskers':
        # Initialize list of measurement files
        measurement_files = []

        # Loop through sides to gather measurements
        for side in sides:
            measurement_files += sorted(glob.glob(os.path.join(input_dir, f'*{side}*.measurements')))

        # Get base names and filter matching whiskers and measurement files
        whiskers_base_names = {os.path.splitext(os.path.basename(f))[0] for f in wt_files}
        measurement_base_names = {os.path.splitext(os.path.basename(f))[0] for f in measurement_files}
        matching_base_names = whiskers_base_names.intersection(measurement_base_names)

        # Filter and sort whiskers and measurement files
        filtered_whiskers_files = sorted(f for f in wt_files if os.path.splitext(os.path.basename(f))[0] in matching_base_names)
        filtered_measurement_files = sorted(f for f in measurement_files if os.path.splitext(os.path.basename(f))[0] in matching_base_names)

        return filtered_whiskers_files, sides, filtered_measurement_files

    # For 'hdf5' or 'parquet', return the file list and sides
    return wt_files, sides

def combine_measurement_files(whiskers_files: List[str], measurement_files: List[str], sides: List[str], output_file: str):       
    """ 
    Combine whiskers and measurement files and save to output file.
    """                            

    if output_file.endswith('.hdf5'):
        if os.path.exists(output_file):
            os.remove(output_file)
        ww.setup_hdf5(output_file, 1000000, measure=True)
        
        for whiskers_file, measurement_file in zip(whiskers_files, measurement_files):
            # Get which side the whiskers file is from
            side = [side for side in sides if side in whiskers_file][0]
            # Get chunk start
            chunk_start = get_chunk_start_from_filename(whiskers_file)
            # Call append_whiskers_to_hdf5 function
            ww.base.append_whiskers_to_hdf5(
                whisk_filename=whiskers_file,
                measurements_filename=measurement_file,
                h5_filename=output_file,
                chunk_start=chunk_start,
                face_side=side)
    
        # Saving to hdf5 file using parallel processing:
        # hdf5 is not thread safe, but data can be processed to temporary files
        # in parallel and then written to the hdf5 file in a single thread.
        # See ww.base.write_whiskers_to_tmp
    
    elif output_file.endswith('.zarr'):
        
        # Get the chunk_size from the whiskers_files file name pattern
        chunk_size = get_chunk_start_from_filename(whiskers_files[1]) - get_chunk_start_from_filename(whiskers_files[0])
          
        # Save to zarr file, with regular loop
        # for whiskers_file, measurement_file in zip(whiskers_files, measurement_files):
        #     # Get which side the whiskers file is from
        #     side = [side for side in sides if side in whiskers_file][0]
        #     # Get chunk start
        #     chunk_start = get_chunk_start(whiskers_file)
        #     # Call append_whiskers_to_zarr function
        #     ww.base.append_whiskers_to_zarr(
        #         whiskers_file, output_file, chunk_start, measurement_file, side, (chunk_size,))
                        
        # Save to zarr file, using parallel processing.
        # Warning: for a small number of files, this is much slower than a regular loop. 
        # TODO: find threshold for when to use parallel processing
        # with ProcessPoolExecutor() as executor:
        #     executor.map(lambda params: process_whiskers_files(params, output_file, sides, chunk_size),
        #         zip(whiskers_files, measurement_files)
        #     )
        
        logging.debug(f"Creating writer process")
        queue = mp.Queue()
        
        # Start the writer process
        writer = mp.Process(target=writer_process, args=(queue, output_file, chunk_size))
        writer.start()
        
        # # Process files in parallel
        # with ProcessPoolExecutor() as executor:
        #     executor.map(lambda params: process_whiskers_files(params, output_file, sides, chunk_size, queue),
        #                  zip(whiskers_files, measurement_files))
            
        # Process files sequentially
        for params in zip(whiskers_files, measurement_files):
            process_whiskers_files(params, output_file, sides, chunk_size, queue)
            
        # Signal the writer process to finish
        # logging.debug(f"Final state of the queue: {inspect_queue(queue)}")
        queue.put('DONE')
        logging.debug(f"Signalling writer process to finish")
        writer.join()
                

def combine_hdf5(h5_files: List[str], output_file: str = 'combined.csv') -> None:
    """ 
    Combine hdf5 files into a single hdf5 or csv file.
    """

    # Initialize table to concatenate tables
    combined_table = pd.DataFrame()
    num_wids = 0

    # Loop through h5_files
    for h5_file in h5_files:
        table = ww.base.read_whiskers_hdf5_summary(h5_file)
        # print(table.head())
        # size = table.shape[0]

        # Add num_wids to wid column
        table['wid'] = table['wid'] + num_wids

        # Add table to combined table
        combined_table = pd.concat([combined_table, table], ignore_index=True)

        # Find number of unique whisker ids
        unique_wids = combined_table['wid'].unique() 
        num_wids = len(unique_wids)
        print(f"Number of unique whisker ids: {num_wids}")

    # Display unique times
    unique_times = combined_table['fid'].unique()
    num_times = len(unique_times)
    print(f"Number of unique times: {num_times}")

    # Sort combined table by frame id and whisker id
    combined_table = combined_table.sort_values(by=['fid', 'wid'])

    # If output file is hdf5 format, save combined table to hdf5 file
    if output_file.endswith('.hdf5'):
        # Open output hdf5 file
        output_hdf5 = tables.open_file(output_file, mode='w')
        # Create table
        output_hdf5.create_table('/', 'summary', obj=combined_table.to_records(index=False))
        # Close output hdf5 file
        output_hdf5.close()
    elif output_file.endswith('.csv'):
        # Save combined table to csv file
        combined_table.to_csv(output_file, index=False)
        
def read_parquet_file(file):
    """Helper function to read a Parquet file into a Pandas DataFrame"""
    return pq.read_table(file).to_pandas()

def read_hdf5_file(file):
    """Helper function to read an HDF5 file into a Pandas DataFrame"""
    with h5py.File(file, 'r') as f:
        data = f['data'][:]
        df = pd.DataFrame(data)
    return df

def read_whiskers_file(file, side):
    """Helper function to read a .whiskers file into a Pandas DataFrame with full pixel data using WhiskiWrap"""
    if not WHISKIWRAP_AVAILABLE:
        raise ImportError("WhiskiWrap is required to read .whiskers files. Please install it or convert files to .parquet/.hdf5 format.")
    
    # Find the corresponding .measurements file
    measurements_file = file.replace('.whiskers', '.measurements')
    if not os.path.exists(measurements_file):
        raise FileNotFoundError(f"Corresponding measurements file not found: {measurements_file}")
    
    try:
        # Load whiskers data using WhiskiWrap wfile_io
        from WhiskiWrap import wfile_io
        from WhiskiWrap.mfile_io import MeasurementsTable
        
        whiskers = wfile_io.Load_Whiskers(file)
        
        # Load measurements data
        M = MeasurementsTable(str(measurements_file))
        measurements = M.asarray()
        measurements_idx = 0
        
        # Check if measurements need to be reindexed (similar to append_whiskers_to_parquet)
        if len(whiskers) > 0:
            wid_from_trace = np.array(list(whiskers[0].keys())).astype(int)
            initial_frame_measurements = measurements[:len(wid_from_trace)]
            wid_from_measure = initial_frame_measurements[:, 2].astype(int)
            
            if not np.array_equal(wid_from_trace, wid_from_measure):
                measurements = ww.base.index_measurements(whiskers, measurements)
        
        # Prepare data (similar to append_whiskers_to_parquet)
        summary_data = []
        pixels_x_data = []
        pixels_y_data = []
        
        # Get chunk_start from filename
        chunk_start = get_chunk_start_from_filename(file)
        
        for _, frame_whiskers in whiskers.items():
            for _, wseg in frame_whiskers.items():
                whisker_data = {
                    'chunk_start': chunk_start,
                    'fid': wseg.time + chunk_start,
                    'wid': wseg.id,
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
                    'face_side': side
                }
                
                summary_data.append(whisker_data)
                pixels_x_data.append(wseg.x.tolist())
                pixels_y_data.append(wseg.y.tolist())
                measurements_idx += 1
        
        # Convert to Pandas DataFrame
        df = pd.DataFrame(summary_data)
        df['pixels_x'] = pixels_x_data
        df['pixels_y'] = pixels_y_data
        
        return df
        
    except Exception as e:
        logging.error(f"Error reading whiskers file {file}: {str(e)}")
        raise

def get_chunk_start_from_filename(filename):
    """Extract chunk start (frame number) from whiskers filename."""
    import re
    basename = os.path.basename(filename)
    # Look for pattern like _00000000 at the end of the filename
    match = re.search(r'_(\d{8})\.whiskers$', basename)
    if match:
        return int(match.group(1))
    else:
        # If no frame number found, assume it starts at 0
        logging.warning(f"Could not extract frame number from {basename}, assuming chunk_start=0")
        return 0

def adjust_coordinates(summary, whiskerpad_params):
    """Adjust x and y coordinates using the whiskerpad params."""
    for side, df in summary.items():
        whiskerpad_info = next((pad for pad in whiskerpad_params['whiskerpads'] if pad['FaceSide'].lower() == side), None)
        if whiskerpad_info:
            image_coord = whiskerpad_info['ImageCoordinates']
            
            # Adjust pixel coordinates if they exist (for HDF5/parquet files)
            if 'pixels_x' in df.columns and 'pixels_y' in df.columns:
                df['pixels_x'] = df['pixels_x'].apply(lambda x: np.array(x) + image_coord[0])
                df['pixels_y'] = df['pixels_y'].apply(lambda y: np.array(y) + image_coord[1])
            
            # Adjust coordinate fields that are present in the dataframe
            x_columns = [col for col in ['face_x', 'follicle_x', 'tip_x'] if col in df.columns]
            y_columns = [col for col in ['face_y', 'follicle_y', 'tip_y'] if col in df.columns]
            
            if x_columns:
                df[x_columns] += image_coord[0]
            if y_columns:
                df[y_columns] += image_coord[1]
                
    return summary

def combine_sides(wt_files, whiskerpad_file, filter_short=True, resort_frequency=True):
    """Combine left and right whisker tracking data by adjusting whisker IDs."""
    summary = {}
    whiskerpad_params = None
    
    if whiskerpad_file:
        with open(whiskerpad_file, 'r') as f:
            whiskerpad_params = json.load(f)

    # Get sides from the whisker tracking files
    sides = [side for file in wt_files for side in default_sides if side in file]
    
    logging.info(f"Processing {len(wt_files)} files with sides: {sides}")
    
    for file, side in zip(wt_files, sides):
        logging.info(f"Reading {side} side from: {os.path.basename(file)}")
        if file.endswith('.parquet'):
            summary[side] = read_parquet_file(file)
        elif file.endswith('.hdf5'):
            summary[side] = read_hdf5_file(file)
        elif file.endswith('.whiskers'):
            # For .whiskers files, we need to use WhiskiWrap to read them
            if not WHISKIWRAP_AVAILABLE:
                raise ImportError("WhiskiWrap is required to read .whiskers files. Please install it or convert files to .parquet/.hdf5 format.")
            summary[side] = read_whiskers_file(file, side)
        else:
            raise ValueError(f"Unsupported file format: {file}")
    
    if whiskerpad_params:
        summary = adjust_coordinates(summary, whiskerpad_params)

    # Adjust the whisker IDs for the second side
    max_wid_first_side = summary[sides[0]]['wid'].max()
    if len(sides) > 1:
        summary[sides[1]]['wid'] += max_wid_first_side + 1    # Concatenate all sides
    combined_summary = pd.concat(summary.values(), ignore_index=True)
    
    # Apply filtering if requested
    if filter_short:
        logging.info("Applying short whisker filtering...")
        combined_summary = filter_whiskers_by_length(combined_summary)
    
    # Apply frequency-based resorting if requested
    if resort_frequency:
        logging.info("Applying frequency-based resorting...")
        combined_summary = auto_assign_by_frequency(combined_summary)
    
    # Sort combined table by frame id (fid) and whisker id (wid)
    combined_summary = combined_summary.sort_values(by=['fid', 'wid'])

    return combined_summary

def combine_to_file(wt_files, whiskerpad_file, output_file=None, keep_wt_files=False, filter_short=True, resort_frequency=True):
    """Combine whisker tracking files and save to the specified format."""
            
    # If output_file is None, use common prefix of the whisker tracking files, and file format of the whisker tracking files
    if output_file is None:
        base_name = os.path.commonprefix([os.path.basename(f) for f in wt_files]).rstrip('_')     
        output_file = os.path.join(os.path.dirname(wt_files[0]), base_name + '.' + wt_files[0].split('.')[-1])
        
    file_format = output_file.split('.')[-1]
          # Combine whisker tracking files
    start_time = time.time()
    combined_summary = combine_sides(wt_files, whiskerpad_file, filter_short, resort_frequency)
    
    # Save the combined summary to the output file
    if file_format == 'csv':
        combined_summary.to_csv(output_file, index=False)
    elif file_format == 'parquet':
        combined_summary.to_parquet(output_file, index=False)
    elif file_format == 'hdf5':
        with tables.open_file(output_file, mode='w') as f:
            f.create_table('/', 'summary', obj=combined_summary.to_records(index=False))
    elif file_format == 'zarr':
        combined_summary.to_zarr(output_file)

    # Remove whisker tracking files if keep_wt_files is False
    if not keep_wt_files:
        for f in wt_files:
            os.remove(f)

    logging.info(f"File saved to {output_file}")
    logging.info(f"Time taken: {time.time() - start_time} seconds")
    
    return output_file

def filter_whiskers_by_length(df, threshold_factor=0.5):
    """Filter out whiskers that are shorter than 1/2 of the mean median length."""
    # Calculate median length per whisker ID (wid)
    median_length = df.groupby('wid')['length'].median().reset_index(name='median_length')
    
    # Set threshold as ratio of the mean of median lengths (default: 0.5)
    length_threshold = median_length['median_length'].mean() * threshold_factor
    
    logging.info(f"Mean of median whisker lengths: {median_length['median_length'].mean():.2f}")
    logging.info(f"Length threshold ({threshold_factor} * mean median): {length_threshold:.2f}")
    
    # Count whiskers before filtering
    initial_count = len(df)
    
    # Filter out short whiskers
    df_filtered = df[df['length'] >= length_threshold].copy()
    
    # Count whiskers after filtering
    final_count = len(df_filtered)
    removed_count = initial_count - final_count
    
    logging.info(f"Removed {removed_count} whiskers shorter than threshold")
    logging.info(f"Remaining whiskers: {final_count} (from {initial_count})")
    
    return df_filtered

def determine_head_orientation(face_side_value):
    """Determine if head orientation is horizontal or vertical based on face_side value."""
    face_side_str = str(face_side_value).lower()
    if any(side in face_side_str for side in ['left', 'right']):
        return 'vertical'  # sort by y-axis (follicle_y)
    elif any(side in face_side_str for side in ['top', 'bottom']):
        return 'horizontal'  # sort by x-axis (follicle_x)
    else:
        # Default fallback
        return 'vertical'

def auto_assign_by_frequency(df):
    """Automatically reassign whisker IDs based on frequency and antero-posterior axis."""
    if 'face_side' not in df.columns:
        logging.warning("No face_side column found - skipping frequency-based assignment")
        return df
    
    logging.info("Auto-assigning whisker IDs based on frequency and antero-posterior axis...")
    
    # First, calculate frequency of each ID per face_side across all frames
    frequency_per_face_side = {}
    for face_side in df['face_side'].unique():
        face_side_data = df[df['face_side'] == face_side]
        id_counts = face_side_data['label'].value_counts() if 'label' in df.columns else face_side_data['wid'].value_counts()
        # Sort IDs by frequency (descending), then by ID value for ties
        sorted_ids = id_counts.sort_values(ascending=False).index.tolist()
        frequency_per_face_side[face_side] = sorted_ids
        logging.info(f"Face side '{face_side}' - IDs by frequency: {sorted_ids}")
    
    # Now assign IDs for each frame and face_side
    for (fid, face_side), group in df.groupby(['fid', 'face_side']):
        if len(group) == 0:
            continue
            
        # Get the frequency-sorted IDs for this face_side
        frequent_ids = frequency_per_face_side[face_side]
        
        # Get existing labels for this group
        label_col = 'label' if 'label' in df.columns else 'wid'
        existing_labels = group[label_col].tolist()
        
        # Take only the most frequent IDs that we need (same count as existing whiskers)
        num_whiskers = len(existing_labels)
        if len(frequent_ids) >= num_whiskers:
            assigned_ids = frequent_ids[:num_whiskers]
        else:
            # If we don't have enough frequent IDs, pad with sequential numbers
            assigned_ids = frequent_ids + list(range(len(frequent_ids), num_whiskers))
            logging.warning(f"Frame {fid}, face_side {face_side}: Only {len(frequent_ids)} frequent IDs available for {num_whiskers} whiskers. Padding with sequential IDs.")
        
        # Determine sorting axis based on head orientation
        orientation = determine_head_orientation(face_side)
        
        if orientation == 'horizontal':
            # Sort by x-axis (follicle_x) for top/bottom face sides
            sorted_indices = group.sort_values('follicle_x').index
        else:
            # Sort by y-axis (follicle_y) for left/right face sides
            sorted_indices = group.sort_values('follicle_y').index
        
        # Assign frequency-based IDs in spatial order
        for i, idx in enumerate(sorted_indices):
            if i < len(assigned_ids):
                new_label = assigned_ids[i]
                df.loc[idx, label_col] = new_label
            else:
                # Fallback: keep original label if we run out of assigned IDs
                logging.warning(f"No assigned ID for whisker {i} in frame {fid}, face_side {face_side}. Keeping original label.")
    
    logging.info("Frequency-based auto-assignment complete!")
    return df

def main():
    """Main function to parse arguments and combine whisker tracking files."""
    parser = argparse.ArgumentParser(description="Combine whiskers files and measurement files into a single file.")
    parser.add_argument("input_dir", help="Path to the directory containing the whiskers and measurement files.")
    parser.add_argument("-b", "--base", help="Base name for output files", type=str)
    parser.add_argument("-f", "--format", help="Output format: 'csv', 'parquet', 'hdf5', 'zarr'.")
    parser.add_argument("-od", "--output_dir", help="Path to save the output file.")
    parser.add_argument("-k", "--keep", help="Keep the whisker tracking files after combining.", action="store_true")
    parser.add_argument("--filter_short", help="Filter out short whiskers (default: enabled).", action="store_true")
    parser.add_argument("--no_filter_short", help="Disable filtering of short whiskers.", action="store_true")
    parser.add_argument("--resort_frequency", help="Resort whiskers by frequency and AP axis (default: enabled).", action="store_true")
    parser.add_argument("--no_resort", help="Disable frequency-based resorting.", action="store_true")
    
    args = parser.parse_args()

    # Handle filter and resort logic - both are True by default unless disabled
    filter_short = not args.no_filter_short
    resort_frequency = not args.no_resort

    input_dir = args.input_dir
    output_dir = args.output_dir if args.output_dir else input_dir
    
    wt_files, _ = get_files(input_dir)
    if not wt_files:
        raise ValueError("No valid whisker tracking files found.")
    
    base_name = args.base if args.base else os.path.commonprefix([os.path.basename(f) for f in wt_files]).rstrip('_')
    whiskerpad_file = glob.glob(os.path.join(input_dir, f"whiskerpad_{base_name}.json")) + \
                      glob.glob(os.path.join(os.path.dirname(input_dir), f"whiskerpad_{base_name}.json"))
    whiskerpad_file = whiskerpad_file[0] if whiskerpad_file else None
    
    if not args.format:
        args.format = wt_files[0].split('.')[-1]

    output_file = os.path.join(output_dir, f"{base_name}.{args.format}")
    logging.info(f"Output file: {output_file}")

    combine_to_file(wt_files, whiskerpad_file, output_file, args.keep, filter_short, resort_frequency)

if __name__ == "__main__":
    main()
