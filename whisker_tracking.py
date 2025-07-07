"""
Script to trace and measure whiskers in a video file.
The script traces whiskers in the video file, combines left and right whisker data, reclassifies whiskers, and plots an overlay of the whiskers on the video's first frame.

Usage:
python whisker_tracking.py <input_video> [-b <base_name>] [-s] [-p <nproc>] [-o <output_dir>]

Arguments:
    input: Path to the input video file.
    -b, --base: Base name for output files. Default is the input file name without the extension.
    -s, --splitUp: Flag to split the video. If set, the video will be split into left and right sides.
    -p, --nproc: Number of trace processes. Default is 40.
    -o, --output_dir: Output directory. Default is the same directory as the input file + WT.
    
Example:
python whisker_tracking.py /path/to/video.mp4 -b video_name -s -p 40 -o /path/to/output_dir    
    
"""

import argparse
import os
import sys
import json
import numpy as np
import time
import gc
import WhiskiWrap as ww
from WhiskiWrap import load_whisker_data as lwd
import whiskerpad as wp
import combine_sides as cs
import reclassify as rc
import unet_classifier as uc
import plot_overlay as po

def trace_measure(input_file, base_name, output_dir, nproc, splitUp, log_file):
    """Trace and measure whiskers."""
    # if output directory doesn't exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_dir = os.path.dirname(input_file)

    # Load whiskerpad json file
    whiskerpad_file = os.path.join(input_dir, f'whiskerpad_{base_name}.json')
    
    if not os.path.exists(whiskerpad_file):
        # If whiskerpad file does not exist, create it
        log_file.write(f"Creating whiskerpad parameters file {whiskerpad_file}\n")
        log_file.flush()
        whiskerpad = wp.Params(input_file, splitUp, base_name)
        # Get whiskerpad parameters
        whiskerpadParams, splitUp = wp.WhiskerPad.get_whiskerpad_params(whiskerpad)
        # Save whisking parameters to json file
        wp.WhiskerPad.save_whiskerpad_params(whiskerpad, whiskerpadParams)

    with open(whiskerpad_file, 'r') as f:
        whiskerpad_params = json.load(f)

    # Check that left and right whiskerpad parameters are defined
    if np.size(whiskerpad_params['whiskerpads']) < 2:
        raise Exception('Missing whiskerpad parameters in whiskerpad json file.')

    # Get side types (left / right or top / bottom)
    side_types = [whiskerpad['FaceSide'].lower() for whiskerpad in whiskerpad_params['whiskerpads']]

    ########################
    ### Run whisker tracking
    ########################
    
    chunk_size = 200
    
    # Define classify arguments
    # See reference for classify arguments: https://wikis.janelia.org/display/WT/Whisker+Tracking+Command+Line+Reference#WhiskerTrackingCommandLineReference-classify
    px2mm = 0.06            # Pixel size in millimeters (mm per pixel).
    num_whiskers = -1       # Expected number of segments longer than the length threshold.
    size_limit = '2.0:40.0' # Low and high length threshold (mm).
    follicle = 150          # Only count follicles that lie on one side of the line specified by this threshold (px). 
    
    classify_args = {'px2mm': str(px2mm), 'n_whiskers': str(num_whiskers)}
    if size_limit is not None:
        classify_args['limit'] = size_limit
    if follicle is not None:
        classify_args['follicle'] = str(follicle)

    output_filenames = []

    for side in side_types:
        log_file.write(f'Running whisker tracking for {side} face side video\n')
        log_file.flush()
        start_time_track = time.time()

        output_filename = os.path.join(os.path.dirname(input_file), f'{base_name}_{side}.parquet')
        chunk_name_pattern = f'{base_name}_{side}_%08d.tif'
        # im_side is the side of the video frame where the face is located. 
        # It is passed to the `face` argument below to tell `measure` which side of traced objects should be considered the follicle.
        im_side = next((whiskerpad['ImageBorderAxis'] for whiskerpad in whiskerpad_params['whiskerpads'] if whiskerpad['FaceSide'].lower() == side), None)

        if im_side is None:
            raise ValueError(f'Could not find {side} whiskerpad ImageBorderAxis in whiskerpad_params')

        # Get the image coordinates
        side_im_coord = next((whiskerpad['ImageCoordinates'] for whiskerpad in whiskerpad_params['whiskerpads'] if whiskerpad['FaceSide'].lower() == side), None)
        # reorder side_im_coord to fit -vf crop format width:height:x:y
        side_im_coord = [side_im_coord[2], side_im_coord[3], side_im_coord[0], side_im_coord[1]]

        log_file.write(f'Number of trace processes: {nproc}\n')
        log_file.write(f'Output directory: {output_dir}\n')
        log_file.write(f'Chunk size: {chunk_size}\n')
        log_file.write(f'Output filename: {output_filename}\n')
        log_file.write(f'Chunk name pattern: {chunk_name_pattern}\n')
        log_file.flush()

        result_dict = ww.interleaved_split_trace_and_measure(
            ww.FFmpegReader(input_file, crop=side_im_coord),
            output_dir,
            chunk_name_pattern=chunk_name_pattern,
            chunk_size=chunk_size,
            output_filename=output_filename,
            n_trace_processes=nproc,
            frame_func='crop',
            face=im_side,
            classify=classify_args,
            summary_only=True,
            skip_existing=True,
            convert_chunks=True,
        )

        time_track = time.time() - start_time_track
        log_file.write(f'Tracking for {side} took {time_track} seconds.\n')
        log_file.flush()

        output_filenames.append(output_filename)

    return output_filenames, whiskerpad_file


def main():
    """Main function to trace, combine, and plot whiskers."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Path to input video file')
    parser.add_argument('-b', '--base', type=str, help='Base name for output files')
    parser.add_argument('-s', '--splitUp', action="store_true", help="Flag to split the video")
    parser.add_argument('-p', '--nproc', type=int, default=40, help='Number of trace processes')
    parser.add_argument('-o', '--output_dir', type=str, help='Output directory. Default is the same directory as the input file + WT')
    args = parser.parse_args()

    # Set input and output file paths
    input_file = args.input
    output_dir = args.output_dir if args.output_dir else os.path.join(os.path.dirname(input_file), 'WT')
    base_name = args.base if args.base else os.path.basename(input_file).split('.')[0]
    splitUp = args.splitUp
    nproc = args.nproc

    # Set up logging
    log_file_path = os.path.join(os.path.dirname(input_file), f'whisker_tracking_{base_name}_log.txt')
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file

        start_time = time.time()
        log_file.write(f'Start time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}\n')

        # Trace and measure whiskers
        log_file.write(f"Tracing and measuring whiskers for {input_file}...\n")
        log_file.flush()
        output_filenames, whiskerpad_file = trace_measure(input_file, base_name, output_dir, nproc, splitUp, log_file)
        log_file.write(f'Tracing and measuring whiskers took {time.time() - start_time} seconds.\n')

        # Force garbage collection
        gc.collect()

        # Combine left and right whisker data
        log_file.write("Combining whisker tracking files...\n")
        log_file.flush()
        start_time_combine = time.time()
        output_file = cs.combine_to_file(output_filenames, whiskerpad_file)
        log_file.write(f'Combining whiskers took {time.time() - start_time_combine} seconds.\n')
        log_file.flush()

        # Force garbage collection
        gc.collect()

        # Attempt automatic whisker labelling using a U-Net model
        log_file.write("Assigning whisker IDs with U-Net...\n")
        log_file.flush()
        start_time_unet = time.time()
        unet_output_file = uc.assign_whisker_ids(input_file, output_file)
        if unet_output_file is not None:
            output_file = unet_output_file
            base_name = os.path.basename(output_file).split('.')[0]
            log_file.write(f'Automatic labelling took {time.time() - start_time_unet} seconds.\n')
        else:
            log_file.write('U-Net labelling failed, falling back to reclassification.\n')
            start_time_reclassify = time.time()
            updated_output_file = rc.reclassify(output_file, whiskerpad_file)
            if updated_output_file is not None:
                output_file = updated_output_file
                base_name = os.path.basename(output_file).split('.')[0]
                log_file.write(f'Reclassifying whiskers took {time.time() - start_time_reclassify} seconds.\n')
            else:
                log_file.write('Reclassification failed.\n')
        log_file.flush()
        
        # Force garbage collection
        gc.collect()

        # Plot overlay
        log_file.write("Creating overlay plot...\n")
        log_file.flush()
        start_time_plot = time.time()
        po.plot_frame_overlay(input_file, base_name)
        log_file.write(f'Plotting overlay took {time.time() - start_time_plot} seconds.\n')
        log_file.flush()
        
        # Force garbage collection
        gc.collect()

        total_time = time.time() - start_time
        log_file.write(f'Total time for the script: {total_time} seconds\n')
        log_file.flush()

        # Close log and restore stdout
        sys.stdout = sys.__stdout__

    print(f"Log file saved at: {log_file_path}")


if __name__ == '__main__':
    main()
