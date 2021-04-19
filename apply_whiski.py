"""
Copyright (c) 2009 HHMI. Free downloads and distribution are allowed for any
non-profit research and educational purposes as long as proper credit is given
to the author. All other rights reserved.
"""
from warnings import warn

import WhiskiWrap
import os
from multiprocessing import freeze_support
import argparse
import shutil
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate whiski output from video.')
    parser.add_argument('video_path',type=str, help='path to the video (video must have an extension e.g. video.avi).')

    args = parser.parse_args()
    # working directory is always the script directory
    wdir = os.getcwd()
    # get video name
    video_fname = os.path.basename(args.video_path)
    video_name = ''.join(video_fname.split('.')[:-1])
    # output_path has the same name of the video name plus whiki_
    output_path = os.path.join(wdir,'whiski_'+video_name)
    # creates output path if it doesn't exists
    if not os.path.exists(output_path):
        warn('out path didn\'t exist creating output path ' + output_path)
        os.mkdir(output_path)
    # copies video if it is not there (in the output path)
    input_video = os.path.join(output_path,video_fname)
    if not os.path.exists(input_video):
        warn('input video didn\'t exist coping from source ' + output_path)
        shutil.copy(args.video_path, input_video)
    output_file = os.path.join(output_path,video_name+'.hdf5')
    freeze_support()
    input_video = os.path.expanduser(input_video)
    output_file = os.path.expanduser(output_file)
    print('input_video ', input_video)
    print('output_file', output_file)

    WhiskiWrap.pipeline_trace(input_video, output_file, n_trace_processes=4, chunk_sz_frames=100)
