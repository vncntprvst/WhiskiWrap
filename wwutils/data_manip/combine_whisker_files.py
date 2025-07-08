#!/usr/bin/env python3
"""
WHISKER TRACKING DATA COMBINER

This script combines whisker tracking analysis results from multiple files/sides into 
a single parquet file. Used AFTER whisker tracking analysis is complete.

Purpose: Data postprocessing - merges whisker tracking results from different camera 
         views (left/right sides) into one unified dataset with coordinate adjustments

Workflow position: Step 3 in the analysis pipeline
  1. Extract video segments (concat_segments.sh)  
  2. Run whisker tracking analysis
  3. Combine tracking results (THIS SCRIPT)

Features:
- Automatically detects whisker tracking file formats (.parquet, .hdf5, .whiskers)
- Adjusts coordinates using whiskerpad parameters
- Combines left/right side data with proper whisker ID management
- Optional filtering of short whiskers and frequency-based resorting
- Validates input files and provides detailed progress information

Usage:
    combine_whisker_files.py <tracking_folder> <whiskerpad_file> [<output_file>]

Arguments:
    tracking_folder      Path to the folder containing whisker tracking files
    whiskerpad_file      Path to the whiskerpad JSON file containing coordinate adjustments
    output_file          (Optional) Path to the output parquet file

Example:
    python combine_whisker_files.py /path/to/tracking_files /path/to/whiskerpad_file.json output_combined.parquet
"""

import os
import sys
import glob
import json
import logging
import argparse
import time
from typing import List

# Add the parent directory to sys.path to import combine_sides
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import combine_sides as cs

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verify_whiskerpad_file(whiskerpad_file: str) -> bool:
    """
    Verify that the whiskerpad file exists and has the correct format.
    
    Args:
        whiskerpad_file: Path to the whiskerpad JSON file
        
    Returns:
        True if the file exists and has valid content, False otherwise
    """
    if not os.path.isfile(whiskerpad_file):
        logging.error(f"Whiskerpad file not found: {whiskerpad_file}")
        return False
    
    try:
        with open(whiskerpad_file, 'r') as f:
            whiskerpad_params = json.load(f)
        
        # Check if the whiskerpad file has the required structure
        if 'whiskerpads' not in whiskerpad_params:
            logging.error(f"Invalid whiskerpad file format: 'whiskerpads' key not found")
            return False
        
        # Check if at least one whiskerpad has the required fields
        valid = False
        for whiskerpad in whiskerpad_params['whiskerpads']:
            if all(key in whiskerpad for key in ['FaceSide', 'ImageCoordinates']):
                valid = True
                break
        
        if not valid:
            logging.error(f"Invalid whiskerpad file: at least one whiskerpad must have 'FaceSide' and 'ImageCoordinates'")
            return False
        
        return True
    
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in whiskerpad file: {whiskerpad_file}")
        return False
    except Exception as e:
        logging.error(f"Error reading whiskerpad file: {str(e)}")
        return False

def determine_output_file(tracking_folder: str, wt_files: List[str]) -> str:
    """
    Determine the output file path based on the folder name and whisker tracking files.
    
    Args:
        tracking_folder: Path to the folder containing whisker tracking files
        wt_files: List of whisker tracking files
        
    Returns:
        Path to the output parquet file
    """
    # Try to find a common prefix among the whisker tracking files
    base_name = os.path.commonprefix([os.path.basename(f) for f in wt_files]).rstrip('_')
    
    # If common prefix is too short, use the folder name
    if len(base_name) < 3:
        base_name = os.path.basename(tracking_folder)
    
    # Create output file path
    output_file = os.path.join(tracking_folder, f"{base_name}_combined.parquet")
    return output_file

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Combine whisker tracking files into a parquet file.")
    parser.add_argument("tracking_folder", help="Path to the folder containing whisker tracking files")
    parser.add_argument("output_file", nargs="?", default=None, help="Path to the output parquet file (optional)")
    parser.add_argument("--output", "-o", dest="output_file_flag", help="Path to the output parquet file")
    parser.add_argument("--whiskerpad", "-w", help="Path to the whiskerpad JSON file")
    parser.add_argument("--keep", "-k", action="store_true", help="Keep original whisker tracking files after combining")
    parser.add_argument("--no-filter", action="store_true", help="Disable filtering of short whiskers")
    parser.add_argument("--no-resort", action="store_true", help="Disable frequency-based resorting")
    
    args = parser.parse_args()
    
    # Handle alternative flag formats for output and whiskerpad
    output_file = args.output_file
    whiskerpad_file = args.whiskerpad
    
    if not whiskerpad_file:
        print("Error: whiskerpad file is required. Use the positional argument or --whiskerpad/-w flag.")
        sys.exit(1)
    
    # Get absolute paths
    tracking_folder = os.path.abspath(args.tracking_folder)
    whiskerpad_file = os.path.abspath(whiskerpad_file)
    
    # Check if the tracking folder exists
    if not os.path.isdir(tracking_folder):
        logging.error(f"Tracking folder not found: {tracking_folder}")
        sys.exit(1)
    
    # Verify the whiskerpad file
    if not verify_whiskerpad_file(whiskerpad_file):
        sys.exit(1)
    
    # Get whisker tracking files using combine_sides logic
    try:
        wt_files_result = cs.get_files(tracking_folder)
        if len(wt_files_result) == 3:
            # whiskers format returns (whiskers_files, sides, measurement_files)
            wt_files, sides, measurement_files = wt_files_result
            file_format = 'whiskers'
        else:
            # hdf5/parquet format returns (files, sides)
            wt_files, sides = wt_files_result
            measurement_files = None
            # Determine format from file extension
            if wt_files and wt_files[0].endswith('.hdf5'):
                file_format = 'hdf5'
            elif wt_files and wt_files[0].endswith('.parquet'):
                file_format = 'parquet'
            else:
                file_format = 'unknown'
    except Exception as e:
        logging.error(f"Error getting whisker files: {str(e)}")
        sys.exit(1)
    
    if not wt_files:
        logging.error("No valid whisker tracking files found.")
        sys.exit(1)
    
    logging.info(f"Found {len(wt_files)} {file_format} files")
    if sides:
        logging.info(f"Detected sides: {', '.join(sides)}")
    
    # Determine output file if not provided
    if not output_file:
        output_file = determine_output_file(tracking_folder, wt_files)
    
    # Ensure output file has .parquet extension
    if not output_file.endswith('.parquet'):
        output_file += '.parquet'
    
    logging.info(f"Output file: {output_file}")
    
    # Combine whisker files
    start_time = time.time()
    try:
        # Use the updated combine_to_file function for all formats
        # The function now handles .whiskers files directly through combine_sides
        result_file = cs.combine_to_file(
            wt_files=wt_files,
            whiskerpad_file=whiskerpad_file,
            output_file=output_file,
            keep_wt_files=args.keep,
            filter_short=not args.no_filter,
            resort_frequency=not args.no_resort
        )
        
        logging.info(f"Successfully combined whisker files into: {result_file}")
        logging.info(f"Finished in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Error combining whisker files: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
