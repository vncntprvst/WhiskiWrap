import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import tempfile
import shutil
import re
from wwutils import video
import wwutils
from .base import copy_parameters_files, setup_hdf5, write_chunk, trace_chunk, measure_chunk, trace_and_measure_chunk, append_whiskers_to_hdf5, append_whiskers_to_parquet, merge_parquet_files, process_and_write_zarr, consolidate_zarr_metadata, initialize_zarr, read_whisker_data, whisk_path
from .io import ChunkedTiffWriter, FFmpegWriter

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
    wwutils.utils.probe_needed_commands()

    # Figure out where to store temporary data
    input_vfile = os.path.abspath(input_vfile)
    input_dir = os.path.split(input_vfile)[0]

    # Setup the result file
    setup_hdf5(h5_filename, expectedrows, measure=measure)

    # Figure out how many frames and epochs
    duration = video.get_video_duration(input_vfile)
    frame_rate = video.get_video_params(input_vfile)[2]
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
        frames = video.process_chunks_of_video(input_vfile,
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
            fn = wwutils.utils.FileNamer.from_tiff_stack(
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
    ctw = ChunkedTiffWriter(tiffs_to_trace_directory,
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
                ffw = FFmpegWriter(monitor_video,
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
    wwutils.utils.probe_needed_commands()

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
        fn = wwutils.utils.FileNamer.from_tiff_stack(chunk_name)
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
    wwutils.utils.probe_needed_commands()

    ## Initialize readers and writers
    if verbose:
        print("initalizing readers and writers")
    # Tiff writer
    ctw = ChunkedTiffWriter(tiffs_to_trace_directory,
        chunk_size=chunk_size, chunk_name_pattern=chunk_name_pattern)

    # FFmpeg writer is initalized after first frame
    ffw = None

    # Setup the result file
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
                ffw = FFmpegWriter(monitor_video,
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
            fn = wwutils.utils.FileNamer.from_tiff_stack(chunk_name)
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
    wwutils.utils.probe_needed_commands(paths=[whisk_path])

    ## Initialize readers and writers
    if verbose:
        print("Initializing readers and writers")

    # Tiff writer
    ctw = ChunkedTiffWriter(tiffs_to_trace_directory,
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
        # existing_files = [f for f in os.listdir(tiffs_to_trace_directory) if f.endswith('.whiskers') or f.endsWith('.measurements')]

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
                    ffw = FFmpegWriter(monitor_video,
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
                    fn = wwutils.utils.FileNamer.from_tiff_stack(chunk_name)
                elif chunk_name.endswith('.whiskers'):
                    fn = wwutils.utils.FileNamer.from_whiskers(chunk_name)
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
                ffw = FFmpegWriter(monitor_video,
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
        measurements = read_whisker_data(filename, 'df')
        # append the dataframe to the list
        all_measurements.append(measurements)

    # Concatenate the dataframes
    all_measurements = pd.concat(all_measurements)

    return all_measurements


def interleaved_trace_and_measure(input_reader, tiffs_to_trace_directory,
    sensitive=False,
    chunk_size=200, chunk_name_pattern='chunk%08d.tif',
    stop_after_frame=None, delete_tiffs=True,
    timestamps_filename=None, monitor_video=None,
    monitor_video_kwargs=None, write_monitor_ffmpeg_stderr_to_screen=False,
    output_filename=None, frame_func=None,
    n_trace_processes=4, expectedrows=1000000,
    verbose=True, skip_stitch=False, classify=None,
    summary_only=False, skip_existing=False, convert_chunks=False
    ):
    """Read, write, trace and measure each chunk, one at a time.

    This function is designed for single-side whisker tracking where the full video
    frame is processed without splitting. It combines tracing and measurement in
    a single pipeline with Parquet output support.

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
    output_filename : Parquet file to write whisker data to
    frame_func : function to apply to each frame
        If 'invert', will apply 255 - frame
    n_trace_processes : number of simultaneous trace processes
    expectedrows : expected number of whisker segments (for optimization)
    verbose : whether to print progress information
    skip_stitch : if True, don't stitch results together
    classify : dictionary of classify arguments for whisker classification
    summary_only : if True, only save summary data (not pixel coordinates)
    skip_existing : if True, skip processing if output file already exists
    convert_chunks : if True, convert individual chunks to desired format
    """
    if skip_existing and output_filename and os.path.exists(output_filename):
        if verbose:
            print(f"Output file {output_filename} already exists, skipping...")
        return {'output_filename': output_filename}

    # Set up parameters file copying
    copy_parameters_files(tiffs_to_trace_directory, sensitive=sensitive)
    
    # Initialize output tracking
    result_dict = {}
    
    # Create temporary directory for chunk processing
    temp_dir = os.path.join(tiffs_to_trace_directory, 'temp_chunks')
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Write video as chunked tiffs and process them
        ctw = write_video_as_chunked_tiffs(
            input_reader, 
            tiffs_to_trace_directory,
            chunk_size=chunk_size, 
            chunk_name_pattern=chunk_name_pattern,
            stop_after_frame=stop_after_frame, 
            monitor_video=monitor_video,
            timestamps_filename=timestamps_filename,
            monitor_video_kwargs=monitor_video_kwargs
        )
        
        # Get list of tiff files to process
        tif_filenames = ctw.chunknames_written
        
        if verbose:
            print(f"Processing {len(tif_filenames)} chunks with {n_trace_processes} processes")
        
        # Process chunks in parallel
        pool = mp.Pool(n_trace_processes)
        
        # Keep track of results
        trace_results = []
        def log_result(result):
            print("Result logged:", result)
            trace_results.append(result)
        
        # Process all chunks
        if verbose:
            print("Tracing and measuring chunks...")
            
        # Start async jobs for each chunk
        convert_chunks_to = 'parquet' if convert_chunks else None
        for tif_filename in tif_filenames:
            pool.apply_async(trace_and_measure_chunk, 
                           args=(tif_filename, delete_tiffs, 'right', classify, 
                                temp_dir if convert_chunks else None, convert_chunks_to),
                           callback=log_result)
        
        # Wait for all jobs to complete
        pool.close()
        pool.join()
        
        if not skip_stitch and output_filename:
            if verbose:
                print("Stitching results together...")
            
            if convert_chunks:
                # Merge parquet files from temp directory
                merge_parquet_files(temp_dir, output_filename)
            else:
                # Convert and stitch HDF5 files to Parquet
                stitch_h5_to_parquet(tif_filenames, output_filename, face_side='single')
        
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
        
        result_dict['output_filename'] = output_filename
        result_dict['trace_results'] = trace_results
        result_dict['n_chunks'] = len(tif_filenames)
        
        if verbose:
            print(f"Completed processing. Output saved to {output_filename}")
        
        return result_dict
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
        raise e

def stitch_h5_to_parquet(tif_filenames, output_filename, face_side='single'):
    """
    Stitch together HDF5 files from individual chunks into a single Parquet file.
    
    This is a helper function for interleaved_trace_and_measure when convert_chunks=False.
    """
    all_data = []
    
    for chunk_idx, tif_filename in enumerate(tif_filenames):
        h5_filename = tif_filename.replace('.tif', '.hdf5')
        if os.path.exists(h5_filename):
            chunk_start = chunk_idx * 200  # Default chunk size
            append_whiskers_to_parquet(
                whisk_filename=tif_filename.replace('.tif', '.whiskers'),
                measurements_filename=tif_filename.replace('.tif', '.measurements'),
                parquet_filename=output_filename,
                chunk_start=chunk_start,
                summary_only=False,
                face_side=face_side
            )
            # Clean up individual HDF5 file
            os.remove(h5_filename)
