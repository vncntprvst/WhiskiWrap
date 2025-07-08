"""
whisker_tracking_pipeline.py - Complete pipeline for automated whisker tracking and analysis

This script provides a comprehensive end-to-end pipeline for whisker tracking that includes:
    1. Whisker tracing and measurement from video input
    2. Combining bilateral (left/right) whisker data  
    3. Automatic whisker labeling with U-Net classifier
    4. Reclassification fallback for improved accuracy
    5. Overlay plot generation for visualization

The pipeline can be run in full automation mode or with fine-grained control over
which steps to execute, making it suitable for both complete analysis workflows
and iterative development/debugging.

Usage:
    # Single side tracking (no whiskerpad detection)
    python whisker_tracking_pipeline.py <input_video> [options]

    # Bilateral tracking (with whiskerpad detection)
    python whisker_tracking_pipeline.py <input_video> -s [options]
    
    # Example for bilateral tracking with specific steps
    python whisker_tracking_pipeline.py test_videos/test_bilateral_view.mp4 -b excerpt_video -s -p 40 -o ./test_videos/whisker_tracking --steps trace

    # Example for single-side tracking
    python whisker_tracking_pipeline.py test_videos/test_video.mp4 -b single_test -p 40

Arguments:
    input_video         Path to the input video file (required)

Core Options:
    -b, --base          Base name for output files (default: input filename stem)
    -s, --splitUp       Split video into left and right sides for bilateral tracking
                        (enables whiskerpad detection; omit for single-side tracking)
    -p, --nproc         Number of parallel trace processes (default: 40)
    -o, --output_dir    Output directory (default: input_dir/whisker_tracking)
    
Pipeline Control:
    --steps             Comma-separated list of pipeline steps to execute.
                        Available: trace, combine, label, reclassify, plot
                        (default: all steps in sequence)
    --skip-trace        Skip whisker tracing (assumes trace files exist)
    --skip-combine      Skip combining bilateral data
    --skip-label        Skip automatic whisker labeling
    --skip-reclassify   Skip reclassification step
    --skip-plot         Skip overlay plot generation

Examples:
    # Run complete pipeline with bilateral tracking
    python whisker_tracking_pipeline.py video.mp4 -b experiment_01 -s -p 40
    
    # Run complete pipeline with single-side tracking
    python whisker_tracking_pipeline.py video.mp4 -b experiment_01 -p 40
    
    # Run only tracing and combining steps (bilateral)
    python whisker_tracking_pipeline.py video.mp4 --steps trace,combine -s
    
    # Skip tracing (use existing files) and run analysis only
    python whisker_tracking_pipeline.py video.mp4 --skip-trace
    
    # Run only labeling and visualization steps
    python whisker_tracking_pipeline.py video.mp4 --steps label,plot -b custom_name

Output Files:
    Single-side tracking (-s flag absent):
    - {base_name}.parquet                  : Single-side whisker traces
    
    Bilateral tracking (-s flag present):
    - {base_name}_left.parquet             : Left side whisker traces
    - {base_name}_right.parquet            : Right side whisker traces  
    - {base_name}_combined.parquet         : Combined bilateral data
    - whiskerpad_{base_name}.json          : Whiskerpad detection parameters
    
    Both modes:
    - {base_name}_labeled.parquet          : Data with whisker IDs (after labeling)
    - whisker_tracking_{base_name}_log.txt : Detailed execution log

Written by Vincent Prevosto
Licensed under the MIT License.
"""

import argparse
import os
import sys
import json
import numpy as np
import time
import gc
import WhiskiWrap as ww
# from wwutils.data_manip import load_data as ld
from wwutils import whiskerpad as wp
from wwutils.data_manip import combine_sides as cs
from wwutils.classifiers import reclassify as rc
from wwutils import unet_classifier as uc
from wwutils import plots as po

def trace_measure(input_file, base_name, output_dir, nproc, splitUp, log_file):
    """
    Trace and measure whiskers from video input.
    
    This function handles the core whisker tracking pipeline including:
    - Loading or creating whiskerpad configuration parameters (for bilateral tracking)
    - Setting up tracking parameters for bilateral (left/right) or single-side processing
    - Running parallel whisker tracing with specified process count
    - Generating output files in Parquet format for downstream analysis
    
    Args:
        input_file (str): Path to input video file
        base_name (str): Base name for output files
        output_dir (str): Directory for output files
        nproc (int): Number of parallel trace processes
        splitUp (bool): Whether to split video into left/right sides
        log_file (file): Open file handle for logging output
        
    Returns:
        tuple: (output_filenames, whiskerpad_file)
            - output_filenames (list): Paths to generated trace files
            - whiskerpad_file (str): Path to whiskerpad configuration file (None for single-side)
            
    Raises:
        Exception: If whiskerpad parameters are missing or invalid (bilateral tracking only)
        ValueError: If side-specific image coordinates cannot be found (bilateral tracking only)
    """
    # if output directory doesn't exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_dir = os.path.dirname(input_file)

    # Handle single-side tracking (no whiskerpad detection needed)
    if not splitUp:
        log_file.write("Single-side tracking mode - skipping whiskerpad detection\n")
        log_file.flush()
        
        # Define side types for single-side tracking
        side_types = ['single']
        whiskerpad_file = None
        whiskerpad_params = None
        
    else:
        # Load whiskerpad json file for bilateral tracking
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

        if splitUp:
            # Bilateral tracking - use side-specific parameters
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
            
            # Create FFmpeg reader with cropping
            reader = ww.FFmpegReader(input_file, crop=side_im_coord)
        else:
            # Single-side tracking - use full video frame
            output_filename = os.path.join(os.path.dirname(input_file), f'{base_name}.parquet')
            chunk_name_pattern = f'{base_name}_%08d.tif'
            
            # Create FFmpeg reader without cropping
            reader = ww.FFmpegReader(input_file)

        log_file.write(f'Number of trace processes: {nproc}\n')
        log_file.write(f'Output directory: {output_dir}\n')
        log_file.write(f'Chunk size: {chunk_size}\n')
        log_file.write(f'Output filename: {output_filename}\n')
        log_file.write(f'Chunk name pattern: {chunk_name_pattern}\n')
        log_file.flush()

        if splitUp:
            # Bilateral tracking - use interleaved_split_trace_and_measure
            result_dict = ww.interleaved_split_trace_and_measure(
                reader,
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
        else:
            # Single-side tracking - use interleaved_trace_and_measure
            result_dict = ww.interleaved_trace_and_measure(
                reader,
                output_dir,
                chunk_name_pattern=chunk_name_pattern,
                chunk_size=chunk_size,
                output_filename=output_filename,
                n_trace_processes=nproc,
                frame_func='crop',
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
    """
    Main function to execute the whisker tracking pipeline.
    
    Orchestrates the complete whisker tracking workflow with user-configurable
    step selection and comprehensive error handling. Provides detailed logging
    and validation of file dependencies between pipeline steps.
    
    The function:
    1. Parses command-line arguments and validates inputs
    2. Determines which pipeline steps to execute based on user preferences
    3. Validates file dependencies between steps
    4. Executes each step with timing and error reporting
    5. Generates comprehensive logs for debugging and analysis
    
    Raises:
        SystemExit: On invalid command-line arguments or missing dependencies
        FileNotFoundError: If required input files are not found
        Various: Depending on individual pipeline step failures
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Complete pipeline for automated whisker tracking and analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4 -b experiment_01 -s -p 40
      Run complete pipeline with bilateral tracking using 40 processes
  
  %(prog)s video.mp4 --steps trace,combine
      Run only tracing and combining steps
  
  %(prog)s video.mp4 --skip-trace --steps combine,label,plot
      Skip tracing and run analysis steps only
  
  %(prog)s video.mp4 --steps label,plot -b custom_name
      Run only labeling and visualization with custom base name

For more information, see the documentation in the script header.
        """
    )
    
    # Required arguments
    parser.add_argument('input', help='Path to input video file')
    
    # Basic options
    parser.add_argument('-b', '--base', type=str, 
                       help='Base name for output files (default: input filename stem)')
    parser.add_argument('-s', '--splitUp', action="store_true", 
                       help="Split video into left and right sides for bilateral tracking")
    parser.add_argument('-p', '--nproc', type=int, default=40, 
                       help='Number of parallel trace processes (default: 40, min: 1)')
    parser.add_argument('-o', '--output_dir', type=str, 
                       help='Output directory (default: input_dir/whisker_tracking)')
    
    # Pipeline control options
    parser.add_argument('--steps', type=str, 
                       help='Comma-separated list of steps: trace,combine,label,reclassify,plot (default: all)')
    parser.add_argument('--skip-trace', action='store_true', 
                       help='Skip whisker tracing (assumes trace files exist)')
    parser.add_argument('--skip-combine', action='store_true', 
                       help='Skip combining bilateral data')
    parser.add_argument('--skip-label', action='store_true', 
                       help='Skip automatic whisker labeling')
    parser.add_argument('--skip-reclassify', action='store_true', 
                       help='Skip reclassification step')
    parser.add_argument('--skip-plot', action='store_true', 
                       help='Skip overlay plot generation')
    
    # Version and help
    parser.add_argument('--version', action='version', version='WhiskiWrap Pipeline v2.0')
    
    args = parser.parse_args()

    # Validate input file exists
    input_file = args.input
    if not os.path.exists(input_file):
        print(f"Error: Input video file not found: {input_file}")
        sys.exit(1)
    
    if not os.path.isfile(input_file):
        print(f"Error: Input path is not a file: {input_file}")
        sys.exit(1)

    # Set up paths and parameters
    output_dir = args.output_dir if args.output_dir else os.path.join(os.path.dirname(input_file), 'whisker_tracking')
    
    # Extract base name more robustly
    if args.base:
        base_name = args.base
    else:
        # Get filename without path and extension
        filename = os.path.basename(input_file)
        if '.' in filename:
            base_name = filename.rsplit('.', 1)[0]  # Remove last extension
        else:
            base_name = filename
    
    splitUp = args.splitUp
    nproc = max(1, args.nproc)  # Ensure at least 1 process
    
    # Validate output directory can be created
    try:
        os.makedirs(output_dir, exist_ok=True)
    except PermissionError:
        print(f"Error: Cannot create output directory (permission denied): {output_dir}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Cannot create output directory: {output_dir} - {e}")
        sys.exit(1)

    # Determine which pipeline steps to execute
    all_steps = ['trace', 'combine', 'label', 'reclassify', 'plot']
    
    if args.steps:
        # User specified specific steps
        requested_steps = [step.strip() for step in args.steps.split(',')]
        # Validate step names
        invalid_steps = [step for step in requested_steps if step not in all_steps]
        if invalid_steps:
            print(f"Error: Invalid step(s): {', '.join(invalid_steps)}")
            print(f"Valid steps are: {', '.join(all_steps)}")
            sys.exit(1)
        steps_to_run = requested_steps
    else:
        # Default: run all steps, but respect skip flags
        steps_to_run = all_steps.copy()
        if args.skip_trace:
            steps_to_run.remove('trace')
        if args.skip_combine:
            steps_to_run.remove('combine')
        if args.skip_label:
            steps_to_run.remove('label')
        if args.skip_reclassify:
            steps_to_run.remove('reclassify')
        if args.skip_plot:
            steps_to_run.remove('plot')
    
    print(f"Pipeline steps to execute: {', '.join(steps_to_run)}")
    
    # Validate dependencies
    if 'combine' in steps_to_run and 'trace' not in steps_to_run:
        # Check if trace files exist
        if splitUp:
            expected_trace_files = [
                os.path.join(os.path.dirname(input_file), f'{base_name}_left.parquet'),
                os.path.join(os.path.dirname(input_file), f'{base_name}_right.parquet')
            ]
        else:
            expected_trace_files = [
                os.path.join(os.path.dirname(input_file), f'{base_name}.parquet')
            ]
        missing_files = [f for f in expected_trace_files if not os.path.exists(f)]
        if missing_files:
            print(f"Warning: Combine step requested but trace files missing: {missing_files}")
            print("Consider running trace step first or using --skip-combine")
    
    if any(step in steps_to_run for step in ['label', 'reclassify', 'plot']) and 'combine' not in steps_to_run:
        # Check if combined file exists
        if splitUp:
            expected_combined_file = os.path.join(os.path.dirname(input_file), f'{base_name}_combined.parquet')
        else:
            expected_combined_file = os.path.join(os.path.dirname(input_file), f'{base_name}.parquet')
        if not os.path.exists(expected_combined_file):
            print(f"Warning: Analysis steps requested but file missing: {expected_combined_file}")
            print("Consider running combine step first or using appropriate skip flags")

    # Set up logging with error handling
    log_file_path = os.path.join(os.path.dirname(input_file), f'whisker_tracking_{base_name}_log.txt')
    
    try:
        with open(log_file_path, 'w') as log_file:
            # Redirect stdout to log file for comprehensive logging
            original_stdout = sys.stdout
            sys.stdout = log_file

            start_time = time.time()
            log_file.write(f'=== WHISKER TRACKING PIPELINE LOG ===\n')
            log_file.write(f'Start time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}\n')
            log_file.write(f'Input file: {input_file}\n')
            log_file.write(f'Output directory: {output_dir}\n')
            log_file.write(f'Base name: {base_name}\n')
            log_file.write(f'Number of processes: {nproc}\n')
            log_file.write(f'Split video: {splitUp}\n')
            log_file.write(f'Pipeline steps to execute: {", ".join(steps_to_run)}\n')
            log_file.write(f'{"="*50}\n\n')
            log_file.flush()

            # Initialize variables for pipeline steps
            output_filenames = None
            whiskerpad_file = None
            output_file = None
            
            try:
                # Step 1: Trace and measure whiskers
                if 'trace' in steps_to_run:
                    log_file.write(f"=== STEP 1: Tracing and measuring whiskers for {input_file} ===\n")
                    log_file.flush()
                    step_start_time = time.time()
                    
                    output_filenames, whiskerpad_file = trace_measure(input_file, base_name, output_dir, nproc, splitUp, log_file)
                    
                    step_time = time.time() - step_start_time
                    log_file.write(f'Tracing and measuring whiskers took {step_time:.2f} seconds.\n')
                    log_file.flush()
                    gc.collect()
                else:
                    log_file.write("=== STEP 1: Skipping whisker tracing ===\n")
                    # Look for existing trace files
                    if splitUp:
                        output_filenames = [
                            os.path.join(os.path.dirname(input_file), f'{base_name}_left.parquet'),
                            os.path.join(os.path.dirname(input_file), f'{base_name}_right.parquet')
                        ]
                        whiskerpad_file = os.path.join(os.path.dirname(input_file), f'whiskerpad_{base_name}.json')
                    else:
                        output_filenames = [
                            os.path.join(os.path.dirname(input_file), f'{base_name}.parquet')
                        ]
                        whiskerpad_file = None
                    log_file.write(f"Using existing trace files: {output_filenames}\n")
                    log_file.flush()

                # Step 2: Combine left and right whisker data
                if 'combine' in steps_to_run:
                    if splitUp and output_filenames and whiskerpad_file:
                        # Bilateral tracking - combine left and right data
                        log_file.write("=== STEP 2: Combining whisker tracking files ===\n")
                        log_file.flush()
                        step_start_time = time.time()
                        
                        output_file = cs.combine_to_file(output_filenames, whiskerpad_file)
                        
                        step_time = time.time() - step_start_time
                        log_file.write(f'Combining whiskers took {step_time:.2f} seconds.\n')
                        log_file.flush()
                        gc.collect()
                    elif not splitUp and output_filenames:
                        # Single-side tracking - use the single output file directly
                        log_file.write("=== STEP 2: Single-side tracking - using single output file ===\n")
                        output_file = output_filenames[0]  # Use the single parquet file
                        log_file.write(f'Using single-side file: {output_file}\n')
                        log_file.flush()
                    else:
                        log_file.write("=== STEP 2: Cannot combine - missing trace files or whiskerpad file ===\n")
                        log_file.flush()
                else:
                    log_file.write("=== STEP 2: Skipping combining whisker data ===\n")
                    # Look for existing combined file
                    if splitUp:
                        output_file = os.path.join(os.path.dirname(input_file), f'{base_name}_combined.parquet')
                    else:
                        output_file = os.path.join(os.path.dirname(input_file), f'{base_name}.parquet')
                    
                    if os.path.exists(output_file):
                        log_file.write(f"Using existing file: {output_file}\n")
                    else:
                        log_file.write(f"Warning: Expected file not found: {output_file}\n")
                    log_file.flush()

                # Step 3: Automatic whisker labelling using U-Net
                if 'label' in steps_to_run:
                    if output_file and os.path.exists(output_file):
                        log_file.write("=== STEP 3: Assigning whisker IDs with U-Net ===\n")
                        log_file.flush()
                        step_start_time = time.time()
                        
                        unet_output_file = uc.assign_whisker_ids(input_file, output_file)
                        if unet_output_file is not None:
                            output_file = unet_output_file
                            # Extract base name from output file more robustly
                            output_filename = os.path.basename(output_file)
                            if '.' in output_filename:
                                base_name = output_filename.rsplit('.', 1)[0]
                            else:
                                base_name = output_filename
                            step_time = time.time() - step_start_time
                            log_file.write(f'Automatic labelling took {step_time:.2f} seconds.\n')
                        else:
                            log_file.write('U-Net labelling failed.\n')
                        
                        log_file.flush()
                        gc.collect()
                    else:
                        log_file.write("=== STEP 3: Cannot label - missing tracking file ===\n")
                        log_file.flush()
                else:
                    log_file.write("=== STEP 3: Skipping automatic whisker labeling ===\n")
                    log_file.flush()

                # Step 4: Reclassification fallback
                if 'reclassify' in steps_to_run:
                    if output_file and os.path.exists(output_file):
                        log_file.write("=== STEP 4: Reclassifying whiskers ===\n")
                        log_file.flush()
                        step_start_time = time.time()
                        
                        # Only use whiskerpad_file if it exists (bilateral tracking)
                        if whiskerpad_file and os.path.exists(whiskerpad_file):
                            updated_output_file = rc.reclassify(output_file, whiskerpad_file)
                        else:
                            # For single-side tracking, pass None for whiskerpad_file
                            updated_output_file = rc.reclassify(output_file, None)
                            
                        if updated_output_file is not None:
                            output_file = updated_output_file
                            # Extract base name from output file more robustly
                            output_filename = os.path.basename(output_file)
                            if '.' in output_filename:
                                base_name = output_filename.rsplit('.', 1)[0]
                            else:
                                base_name = output_filename
                            step_time = time.time() - step_start_time
                            log_file.write(f'Reclassifying whiskers took {step_time:.2f} seconds.\n')
                        else:
                            log_file.write('Reclassification failed.\n')
                        
                        log_file.flush()
                        gc.collect()
                    else:
                        log_file.write("=== STEP 4: Cannot reclassify - missing output file ===\n")
                        log_file.flush()
                else:
                    log_file.write("=== STEP 4: Skipping reclassification ===\n")
                    log_file.flush()

                # Step 5: Plot overlay
                if 'plot' in steps_to_run:
                    if base_name:
                        log_file.write("=== STEP 5: Creating overlay plot ===\n")
                        log_file.flush()
                        step_start_time = time.time()
                        
                        po.plot_frame_overlay(input_file, base_name)
                        
                        step_time = time.time() - step_start_time
                        log_file.write(f'Plotting overlay took {step_time:.2f} seconds.\n')
                        log_file.flush()
                        gc.collect()
                    else:
                        log_file.write("=== STEP 5: Cannot plot - missing base name ===\n")
                        log_file.flush()
                else:
                    log_file.write("=== STEP 5: Skipping overlay plot generation ===\n")
                    log_file.flush()

                # Log completion
                total_time = time.time() - start_time
                log_file.write(f'\n{"="*50}\n')
                log_file.write(f'Pipeline completed successfully!\n')
                log_file.write(f'Total execution time: {total_time:.2f} seconds\n')
                log_file.write(f'End time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n')
                log_file.flush()
                
            except Exception as e:
                # Log any pipeline errors
                log_file.write(f'\n{"="*50}\n')
                log_file.write(f'PIPELINE ERROR: {str(e)}\n')
                log_file.write(f'Error occurred at: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n')
                log_file.flush()
                raise  # Re-raise to ensure error is visible to user
                
            finally:
                # Always restore stdout
                sys.stdout = original_stdout

        print(f"Pipeline execution completed. Log file saved at: {log_file_path}")
        
    except PermissionError:
        print(f"Error: Cannot write to log file (permission denied): {log_file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Cannot create log file: {log_file_path} - {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
