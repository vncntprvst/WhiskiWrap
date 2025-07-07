"""
This script reads the whisker tracking data from an HDF5 or Parquet file and overlays the whisker tracking on a video frame.

Usage:
    python plot_overlay.py <video_file> [--base_name <base_name>] [--fid_num <fid_num>]
    
Arguments:
    video_file: Path to the video file
    base_name: Base name of the video file (default: None)
    fid_num: Frame number to overlay whiskers on (default: 0)
    
Example:
    python plot_overlay.py /path/to/video.mp4 --base_name behavior_video --fid_num 100
"""

import os
import cv2
from pathlib import Path
import tables
# import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from cmcrameri import cm
import multiprocessing
from functools import partial
# from concurrent.futures import ProcessPoolExecutor
import tempfile

def load_whisker_data(base_name, data_dir):
    """Load whisker data from HDF5 or Parquet."""
    print(f"Loading whisker data from {data_dir}/{base_name}")
    if os.path.exists(f'{data_dir}/{base_name}.hdf5'):
        with tables.open_file(f'{data_dir}/{base_name}.hdf5', mode='r') as h5file:
            pixels_x = h5file.get_node('/pixels_x')
            pixels_y = h5file.get_node('/pixels_y')
            summary = h5file.get_node('/summary')
            df = pd.DataFrame(summary[:])
            df['pixels_x'] = [pixels_x[i] for i in df['wid']]
            df['pixels_y'] = [pixels_y[i] for i in df['wid']]
        return df
    
    elif os.path.exists(f'{data_dir}/{base_name}.parquet'):
        parquet_file = f'{base_name}.parquet'
        table = pq.read_table(f'{data_dir}/{parquet_file}')
        df = table.to_pandas()
        return df

    else:
        raise FileNotFoundError("No valid whisker data file found")

def get_longest_whiskers_data(df, fid_num):
    """Get the longest whiskers for the specified frame."""
    frame_df = df[df['fid'] == fid_num]
    sides = frame_df['face_side'].unique()

    longest_whiskers = []
    for side in sides:
        side_df = frame_df[frame_df['face_side'] == side]
        longest_whiskers.append(side_df.nlargest(3, 'pixel_length'))

    return longest_whiskers

def get_whiskers_data(df, fid_num, wids):
    """Get whisker data for the specified frame and whisker IDs."""
    frame_df = df[df['fid'] == fid_num]
    whisker_data = []
    for wid in wids:
        whisker_data.append(frame_df[frame_df['wid'] == wid])
    return whisker_data

def get_colors(numb_colors=10):
    """Get a list of colors for plotting whiskers."""
    
    # Ensure numb_colors is even
    if numb_colors % 2 != 0:
        # Add 1 to make it even
        numb_colors += 1
        print(f"Number of colors must be even. Using {numb_colors} colors.")
    
    # Generate colors from the colormap
    cmap=cm.managua # Divergent colormaps: coolwarm, managua, vanimo
    colors = cmap(np.linspace(0, 1, numb_colors))
    
    # Convert colors to RGB tuples
    colors = [tuple(int(c * 255) for c in color[:3]) for color in colors]
    
    # Split colors into two halves and combine them
    half = numb_colors // 2
    combined_colors = colors[:half] + colors[-half:]
    
    return combined_colors

def get_whiskerpad_params(whiskerpad_file):
    if os.path.exists(whiskerpad_file):
        with open(whiskerpad_file, 'r') as f:
            whiskerpad_params = json.load(f)
            
        # Get the face side(s) and whiskerpad location
        face_sides = [whiskerpad['FaceSide'].lower() for whiskerpad in whiskerpad_params['whiskerpads']]
        
        whiskerpad_location = {}
        
        if len(face_sides) == 1:
            return np.array([0, 0]), face_sides
        else:
            image_coord = np.zeros((len(face_sides), 2))
            for i in range(len(whiskerpad_params['whiskerpads'])):
                whiskerpad_location[face_sides[i]] = whiskerpad_params['whiskerpads'][i]['Location']
                image_coord[i] = whiskerpad_params['whiskerpads'][i]['Location']
            return image_coord, face_sides, whiskerpad_location
    else:
        raise FileNotFoundError("Whiskerpad file provided but not found") 

def plot_whiskers_on_frame(whisker_data, frame, colors):
    """Plot the whiskers on a video frame."""

    for whiskers_side in whisker_data:
        for index, whisker_data in whiskers_side.iterrows():
            if isinstance(colors, dict):
                # Get the color based on the whisker ID
                color = colors[whisker_data['wid']]
            else:
                # Get the color based on the index
                color_index = index % len(colors)
                color = colors[color_index]
            # Plot the whisker points
            for j in range(whisker_data['pixels_x'].shape[0]):
                x = int(whisker_data['pixels_x'][j])
                y = int(whisker_data['pixels_y'][j])
                cv2.circle(frame, (x, y), 2, color, -1)
    
    return frame

def save_frame_with_overlay(frame, data_dir, base_name, fid_num):
    """Save the video frame with whisker overlay."""
    Path(f'{data_dir}/plots').mkdir(parents=True, exist_ok=True)
    plt.imshow(frame)
    plt.savefig(f'{data_dir}/plots/{base_name}_WhiskerOverlay_Frame_{fid_num}.png')
    plt.close()

def plot_frame_overlay(video_file, base_name, fid_num=0):
    """Main function to overlay whisker tracking on video."""
    data_dir = os.path.dirname(video_file)
                
    # Load whisker data
    print(f"Loading whisker data for video: {video_file}")
    df = load_whisker_data(base_name, data_dir)

    # Get the longest whiskers for the specified frame
    longest_whiskers = get_longest_whiskers_data(df, fid_num)

    # Read the corresponding video frame
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame {fid_num} from video")
    
    colors = get_colors()

    # Plot whiskers on the frame
    frame_with_overlay = plot_whiskers_on_frame(longest_whiskers, frame, colors)

    # Save the frame with the whisker overlay
    save_frame_with_overlay(frame_with_overlay, data_dir, base_name, fid_num)
    
def process_frame_with_data(video_file, whisker_data, base_name, fid_num, colors):
    """Process a single frame for whisker overlay with preloaded whisker data."""
    # Get the longest whiskers for the specified frame
    longest_whiskers = get_longest_whiskers_data(whisker_data, fid_num)

    # Read the corresponding video frame
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame {fid_num} from video")

    # Plot whiskers on the frame
    frame_with_overlay = plot_whiskers_on_frame(longest_whiskers, frame, colors)

    # Save the frame with overlay to a temporary file
    temp_file = f"{tempfile.gettempdir()}/{base_name}_frame_{fid_num}.png"
    cv2.imwrite(temp_file, frame_with_overlay)
    return temp_file

def read_video_frames(video_file):
    """Load all video frames into memory."""
    cap = cv2.VideoCapture(video_file)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def single_frame_overlay(frame, whisker_data, fid_num, wids, colors):
    """Process a single frame with preloaded whisker data."""
    # Get whisker data for the specified frame and whisker IDs
    whisker_data = get_whiskers_data(whisker_data, fid_num, wids)

    # Plot whiskers on the frame
    frame_with_overlay = plot_whiskers_on_frame(whisker_data, frame, colors)

    # Save the frame with overlay to a temporary file
    temp_file = f"{tempfile.gettempdir()}/frame_{fid_num}.png"
    cv2.imwrite(temp_file, frame_with_overlay)
    return temp_file

# Function to lazy-load and process a frame
def process_frame_lazy(fid_num, video_file, whisker_data, wids, colors):
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Failed to read frame {fid_num} from {video_file}")

    temp_file = single_frame_overlay(frame, whisker_data, fid_num, wids, colors)
    return temp_file

def get_video_properties(video_file):
    # Open the video file to get frame properties
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_file}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    return frame_count, frame_width, frame_height, fps

def plot_video_overlay(video_file, base_name, wids=None, output_video_file=None, num_workers=40):
    """Create a video with overlaid whiskers using preloaded data and video frames."""
    data_dir = os.path.dirname(video_file)
    
    if output_video_file is None:
        output_video_file = f"{data_dir}/{base_name}_whisker_overlay.mp4"

    # Load whisker data once
    print(f"Loading whisker data for video: {video_file}")
    whisker_data = load_whisker_data(base_name, data_dir)
    
    if wids is None:
        # Get wids of the 3 longest whiskers on each side
        wids = []
        face_sides = whisker_data['face_side'].unique()
        for side in face_sides:
            side_df = whisker_data[whisker_data['face_side'] == side]
            longest_whiskers = side_df.groupby('wid')['pixel_length'].sum().nlargest(3).index.tolist()
            wids.extend(longest_whiskers)

    # Get the video properties 
    frame_count, frame_width, frame_height, fps = get_video_properties(video_file)

    # Prepare colors for whisker overlay
    colors = get_colors(len(wids))
    
    # Assign colors to whisker IDs by side (left/right)
    color_mapping = {}
    for i, wid in enumerate(wids):
        face_side = whisker_data.loc[whisker_data['wid'] == wid, 'face_side'].iloc[0]
        if len(face_side) == 1:
            color_mapping[wid] = colors[i]
        else:
            side_index = 0 if face_side == 'left' else 1
            color_index = i % (len(colors) // 2) + (side_index * (len(colors) // 2))
            color_mapping[wid] = colors[color_index]
    
    # Use multiprocessing to process frames in parallel with lazy loading
    with multiprocessing.Pool(num_workers) as pool:
        process_partial = partial(
            process_frame_lazy, video_file=video_file, whisker_data=whisker_data, wids = wids, colors=color_mapping
        )
        temp_files = pool.map(process_partial, range(frame_count))

    # Create the output video with overlayed frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    for temp_file in temp_files:
        frame = cv2.imread(temp_file)
        out.write(frame)
        os.remove(temp_file)  # Clean up temporary files

    out.release()
    print(f"Video with overlay saved to {output_video_file}")


if __name__ == "__main__":
    # get the path to the video file from the arguments using argparse
    parser = argparse.ArgumentParser(description='Overlay whisker tracking on video.')
    parser.add_argument('video_file', type=str, help='Path to the video file')
    parser.add_argument('--base_name', type=str, default=None, help='Base name of the video file')
    parser.add_argument('--fid_num', type=int, default=0, help='Frame number to overlay whiskers on')
    parser.add_argument('--video', action='store_true', help='Flag to process the entire video')
    parser.add_argument('--whiskerpad_file', type=str, default=None, help='Path to the whiskerpad file')
    args = parser.parse_args()    
    
    if args.base_name is None:
        args.base_name = os.path.basename(args.video_file).split(".")[0]
        
    if args.whiskerpad_file is not None:
        # get_whiskerpad_params(args.whiskerpad_file)
        print(f"Obsolete format. Whiskerpad parameters are not needed.")
    
    # call the main function with the video file path
    print(f"Overlaying whisker tracking on video: {args.video_file}")
    
    if args.video:
        plot_video_overlay(args.video_file, args.base_name)
    else:
        plot_frame_overlay(args.video_file, args.base_name, args.fid_num)
