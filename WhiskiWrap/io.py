"""Input/output classes for reading and writing video and data files.

This module provides classes for reading various input formats and writing output:

Reader classes:
* FFmpegReader - Read video files using ffmpeg subprocess
* PFReader - Read Photonfocus modulated data from MATLAB files
* ChunkedTiffWriter - Write video frames as chunked TIFF stacks
* FFmpegWriter - Write compressed video using ffmpeg subprocess

The readers provide a common interface with iter_frames() method for frame-by-frame
processing. Writers handle buffering and format conversion.

These classes are designed to work with the pipeline functions for efficient
video processing workflows.
"""

import os
import numpy as np
import scipy.io
import ctypes
import wwutils
import tifffile
from wwutils import video
import subprocess
import time

# libpfDoubleRate library, needed for PFReader
LIB_DOUBLERATE = os.path.join(os.path.split(__file__)[0], 'libpfDoubleRate.so')

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
                demodulated_frame = np.fromstring(demulated_frame_buffer,
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
                if hasattr(tifffile, 'imwrite'):
                    tifffile.imwrite(chunkname, chunk, compression='lzw')
                else:
                    tifffile.imsave(chunkname, chunk, compression='lzw')
            else:
                if hasattr(tifffile, 'imwrite'):
                    tifffile.imwrite(chunkname, chunk, compression=None)
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
            Parsed using video.ffmpeg_frame_string
        crop : crop the video using the ffmpeg crop filter. Format is width:height:x:y
        write_stderr_to_screen : if True, writes to screen, otherwise to
            /dev/null
        """
        self.input_filename = input_filename

        # Get params
        self.frame_width, self.frame_height, self.frame_rate = \
            video.get_video_params(input_filename, crop=crop)

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
            ss_string = video.ffmpeg_frame_string(input_filename,
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
