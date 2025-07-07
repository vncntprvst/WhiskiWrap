"""
Script to track and plot whiskers on the first frame of a video.

Example usage:
input_dir=Path('/data')
video_name='sc016_0630_001_30sWhisking.mp4'
input_file = input_dir / video_name
base_name='sc016_0630_001'
splitUp=True
nproc=40

python wt_single_frame.py input_file --splitUp True --base_name sc016_0630_001

import wt_single_frame as wtsf
wtsf.track_whiskers(input_file, whiskerpad_params, splitUp)

"""
import argparse
# import os, sys
# import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import WhiskiWrap as ww
import whiskerpad as wp

def save_image_halves(image_halves, image_sides, face_sides, base_name, input_dir, direction, output_dir=None):
    """
    Save image halves as tif files
    """
    if output_dir is None:
        output_dir = Path(input_dir) / "frame_0_whiskers"
    output_dir.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
     
    for i, (image_half, face_side, image_side) in enumerate(zip(image_halves, face_sides, image_sides)):
        # print each image halves side by side
        ax[i].imshow(image_half, cmap='gray')
        ax[i].axis('off')
        ax[i].set_title(f"Image side: {image_side}, Face side: {face_side}")
        
        # Save each side as tif
        output_file = output_dir / f'{base_name}_first_frame_face_{face_side}.tif'
        cv2.imwrite(str(output_file), image_halves[i])

    plt.show()

    side_types = face_sides

    print(f"Side types: {side_types}, direction: {direction}")
    
    return output_dir


def run_whisker_tracking(image_halves, face_sides, face_im_sides, base_name, output_dir):

    for _, (face_side, face_im_side) in enumerate(zip(face_sides, face_im_sides)):
        image_filename = output_dir / f'{base_name}_first_frame_face_{face_side}.tif' 
        tracking_results = ww.trace_and_measure_chunk(image_filename,
                                                delete_when_done=False,
                                                face=face_im_side)
                                                # classify={'px2mm': '0.04', 'n_whiskers': '3'})
    
    return tracking_results

def load_whisker_data(side_types, base_name, output_dir, save_to_csv=False):
    """
    Load whisker data from whisker files.
    "whiskers" is a list of dictionaries, each dictionary is a frame, each frame has a dictionary of whiskers.
    """
    # Initialize the dictionary to store the whisker data for each side
    whisker_data = {}

    for face_side in side_types:
        print(f'Loading whiskers for {face_side} face side video')
        whisk_filename = output_dir / f'{base_name}_first_frame_face_{face_side}.whiskers'
        # Load whiskers
        whiskers = ww.wfile_io.Load_Whiskers(str(whisk_filename))
        whisker_data[face_side] = whiskers

    print(f"Whisker data for {len(whisker_data)} sides loaded.")
                
    # Load whisker measurements from measurements file
    wmeas = load_whisker_measurements(output_dir, base_name, side_types)

    # Initialize the dictionaries to store the whisker pixel values for each side
    xpixels, ypixels = {}, {}
    # whisker_ids = {}

    for face_side, whiskers in whisker_data.items():
        # Initialize lists for this side if they don't exist yet
        if face_side not in xpixels:
            xpixels[face_side] = []
        if face_side not in ypixels:
            ypixels[face_side] = []
        # if side not in whisker_ids:
        #     whisker_ids[side] = []
        
        for frame, frame_whiskers in list(whiskers.items()):
            for whisker_id, wseg in list(frame_whiskers.items()):
                # Write whisker contour x and y pixel values
                xpixels[face_side].append(wseg.x)
                ypixels[face_side].append(wseg.y)
                # whisker_ids[side].append(wseg.id)

    # Check how many whiskers were detected for each side
    n_whiskers = {side: len(xpixels[side]) for side in side_types}
    print(f"Number of whiskers detected: {n_whiskers}")
    # print(f"Whisker IDs for each sides: {whisker_ids}")

    # Get unique whisker IDs
    whisker_ids = {side: np.unique([wseg.id for frame_whiskers in whisker_data[side].values() for wseg in frame_whiskers.values()]) for side in side_types}
    print(f"Unique whisker IDs: {whisker_ids}")
    
    # Get whisker lengths
    whisker_lengths, whisker_lengths_meas = compare_whisker_lengths(whisker_data, wmeas, side_types)
    
    # Get whisker scores
    whisker_scores = get_whisker_scores(wmeas, side_types, whisker_lengths)
       
    # Get follicles
    im_orientation = define_image_orientation(side_types)
    follicle_x, follicle_y = get_follicles(wmeas, side_types, im_orientation)
    
    # Create set of colors for each whisker ID
    colors = {side: plt.cm.viridis(np.linspace(0, 1, len(whisker_ids[side]))) for side in side_types}

    for whisker_id, color in zip(whisker_ids[face_side], colors[face_side]):
        print(f"Whisker ID: {whisker_id}, color: {color}")
        
    # Save whisker lengths and scores to csv
    if save_to_csv:
        save_whisker_info(output_dir, base_name, side_types, whisker_lengths, whisker_scores)
        
    return xpixels, ypixels, whisker_ids, colors, whisker_lengths, whisker_scores, follicle_x, follicle_y

def load_whisker_measurements(output_dir, base_name, side_types):
    """
    Load whisker measurements from measurements file
    """
    # Initialize the dictionary to store the whisker measurements for each side
    wmeas = {}

    for face_side in side_types:
        #  Check if measurement file exists
        measurement_file = output_dir / f'{base_name}_first_frame_face_{face_side}.measurements'
        if measurement_file.exists():
            print(f'Loading whisker measurements for {face_side} image side video')
            whisk_filename = output_dir / f'{base_name}_first_frame_face_{face_side}.whiskers'
            # Load whiskers
            wmeas[face_side] = ww.read_whisker_data(str(whisk_filename))
        else:
            wmeas[face_side] = None

        print(f"Whisker measurements for {face_side} image side loaded.")
    
    return wmeas

def plot_whiskers_on_half_images(image_halves, side_types, whisker_ids, xpixels, ypixels, colors):
    """
    Plot whiskers on each image side
    """
    
    # Plot whiskers on each image

    # define image orientation
    if 'top' in side_types or 'bottom' in side_types:
        im_orientation = 'horizontal'
    elif 'left' in side_types or 'right' in side_types:
        im_orientation = 'vertical'

    if im_orientation == 'horizontal':
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        # Remove space between subplots
        plt.subplots_adjust(wspace=0, hspace=-0.05)
    elif im_orientation == 'vertical':
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # Remove space between subplots
        plt.subplots_adjust(wspace=-0.51, hspace=0)

    for i, face_side in enumerate(side_types):
        if im_orientation == 'vertical':
            ax[-(i+1)].imshow(image_halves[i], cmap='gray')
        elif im_orientation == 'horizontal':
            # ax[i].imshow(image_halves_rotated[i], cmap='gray')
            ax[i].imshow(image_halves[i], cmap='gray')
        # for whisker_id, color in zip(whisker_ids[face_side], colors[face_side]):
        for idx, (whisker_id, color) in enumerate(zip(whisker_ids[face_side], colors[face_side])):
            # Get the whisker pixel values for this whisker ID from xpixels and ypixels
            whisker_x = xpixels[face_side][idx] # whisker_id
            whisker_y = ypixels[face_side][idx] # whisker_id
            if im_orientation == 'vertical':
                ax[-(i+1)].plot(whisker_x, whisker_y, color=color)
            elif im_orientation == 'horizontal':
                ax[i].plot(whisker_x, whisker_y, color=color)
        if im_orientation == 'vertical':
            ax[-(i+1)].axis('off')
            ax[-(i+1)].set_title(f"Face image side: {face_side}")
        elif im_orientation == 'horizontal':    
            ax[i].axis('off')
            if i == 0:
                ax[i].set_title(f"Face image side: {face_side}")
            elif i == 1:
                ax[i].text(0.5, -0.1, f"Face image side: {face_side}", size=12, ha="center", transform=ax[i].transAxes)
                        
    plt.show()

def compare_whisker_lengths(whisker_data, wmeas, side_types):
    """
    Compare whisker lengths from whisker data and measurements file
    """
    # as sanity check, for each whisker, (on each side), compute whisker length and compare to the length in the measurements file
    # Initialize the dictionaries to store the whisker lengths for each side
    whisker_lengths, whisker_lengths_meas = {}, {}
    for face_side in side_types:
        # Initialize lists for this side if they don't exist yet
        if face_side not in whisker_lengths:
            whisker_lengths[face_side] = []
        
        for frame, frame_whiskers in list(whisker_data[face_side].items()):
            for whisker_id, wseg in list(frame_whiskers.items()):
                # Compute whisker length
                whisker_length = np.sqrt((wseg.x[-1] - wseg.x[0])**2 + (wseg.y[-1] - wseg.y[0])**2)
                whisker_lengths[face_side].append(whisker_length)

    # Compare whisker lengths to lengths in measurements file
    whisker_lengths_diff={}
    for face_side in side_types:
        if face_side not in whisker_lengths_meas:
            whisker_lengths_meas[face_side] = []
        # Get whisker lengths from measurements file 
        # Sort the whisker lengths according to the sorted indices
        sorted_indices = np.argsort(wmeas[face_side]['label'])
        whisker_lengths_meas[face_side].append(np.array(wmeas[face_side]['length'])[sorted_indices])
        
        # Compare whisker lengths
        whisker_lengths_diff[face_side] = np.array(whisker_lengths[face_side]) - np.array(whisker_lengths_meas[face_side])
                
        # print(f"Mean difference in whisker lengths for {im_side} face side: {np.mean(whisker_lengths_diff)}")
        # print(f"Max difference in whisker lengths for {im_side} face side: {np.max(np.abs(whisker_lengths_diff))}")
        
    # if mean diff is >1 for either side, raise an error
    for face_side in side_types:
        if np.median(np.abs(whisker_lengths_diff[face_side])) > 1:
            raise ValueError(f"Mean difference between .whisker and .measurement files' whisker lengths for {face_side} face side is greater than 1 pixel.")
        
    # Print arrays
    # print(np.array(whisker_lengths[face_side]))
    # print( np.array(whisker_lengths_meas[face_side]))
    
    return whisker_lengths, whisker_lengths_meas

def get_whisker_scores(wmeas, side_types, whisker_lengths):
    """
    Get whisker scores from whisker measurements
    """
    #  print 'score' for each whisker from measurements data (wmeas), for each side. again, resort by labels
    # Initialize the dictionaries to store the whisker scores for each side
    whisker_scores = {}
    for face_side in side_types:
        # Initialize lists for this side if they don't exist yet
        if face_side not in whisker_scores:
            whisker_scores[face_side] = []
        
        sorted_indices = np.argsort(wmeas[face_side]['label'])
        whisker_scores[face_side].append(np.array(wmeas[face_side]['score'])[sorted_indices])

    # Print whisker scores
    for face_side in side_types:
        print(f"Whisker lengths and scores for {face_side} image side:")
        
        for length, score in zip(whisker_lengths[face_side], whisker_scores[face_side][0]):
            print(f"{length}, {score}")

    # Normalize the scores to be between 0 and 1
    for face_side in side_types:
        whisker_scores[face_side] = np.array(whisker_scores[face_side][0])
        whisker_scores[face_side] = (whisker_scores[face_side] - np.min(whisker_scores[face_side])) / (np.max(whisker_scores[face_side]) - np.min(whisker_scores[face_side]))
        
    return whisker_scores

def define_image_orientation(side_types):
    """
    Define the image orientation based on the side types
    """
    # define image orientation
    if 'top' in side_types or 'bottom' in side_types:
        im_orientation = 'horizontal'
    elif 'left' in side_types or 'right' in side_types:
        im_orientation = 'vertical'
        
    return im_orientation

def get_follicles(wmeas, side_types, im_orientation):
    """
    Get follicles from whisker measurements
    """
    # Get follicles

    follicle_x, follicle_y = {}, {}
    for face_side in side_types:
        # Initialize lists for this side if they don't exist yet
        if face_side not in follicle_x:
            follicle_x[face_side] = []
            follicle_y[face_side] = []
        
        sorted_indices = np.argsort(wmeas[face_side]['label'])
        if im_orientation == 'horizontal':
            follicle_x[face_side].append(np.array(wmeas[face_side]['follicle_x'])[sorted_indices]) #tip_x
            follicle_y[face_side].append(np.array(wmeas[face_side]['follicle_y'])[sorted_indices]) #tip_y
        elif im_orientation == 'vertical':
            follicle_x[face_side].append(np.array(wmeas[face_side]['follicle_x'])[sorted_indices])
            follicle_y[face_side].append(np.array(wmeas[face_side]['follicle_y'])[sorted_indices])

    # Flatten the lists
    for face_side in side_types:
        follicle_x[face_side] = np.concatenate(follicle_x[face_side])
        follicle_y[face_side] = np.concatenate(follicle_y[face_side])
        
    return follicle_x, follicle_y

def save_whisker_info(output_dir, base_name, side_types, whisker_lengths, whisker_scores, save_format='csv'):

    if save_format == 'csv':
        # Save whisker lengths and scores to csv
        for face_side in side_types:
            output_file = output_dir / f'{base_name}_first_frame_face_{face_side}_whisker_lengths_scores.csv'
            # Convert whisker lengths and scores to dataframe and save
            data = {'length': whisker_lengths[face_side], 'score': whisker_scores[face_side]}
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
    else:
        # Save whisker lengths and scores to npz
        data = {}
        for face_side in side_types:
            data[f'whisker_lengths_{face_side}'] = whisker_lengths[face_side]
            data[f'whisker_scores_{face_side}'] = whisker_scores[face_side]

        output_file = output_dir / f'{base_name}_first_frame_whisker_lengths_scores.npz'
        np.savez(output_file, **data)
        
def plot_whiskers_on_image(image_halves, side_types, direction, follicle_x, follicle_y, whisker_ids, whisker_scores, xpixels, ypixels, colors, base_name, output_dir):
    """
    Plot whiskers for each side on the whole image
    """
    # define image orientation
    if 'top' in side_types or 'bottom' in side_types:
        im_orientation = 'horizontal'
    elif 'left' in side_types or 'right' in side_types:
        im_orientation = 'vertical'

    # Plot whiskers on each image
    if im_orientation == 'vertical':
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # Remove space between subplots
        plt.subplots_adjust(wspace=-0.51, hspace=0)
    elif im_orientation == 'horizontal':
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        # Remove space between subplots
        plt.subplots_adjust(wspace=0, hspace=-0.05)

    for side_idx, (face_side, image_half) in enumerate(zip(side_types, image_halves)):
        if im_orientation == 'vertical':
            if direction == 'north':
                ax[side_idx].imshow(image_half, cmap='gray')
            elif direction == 'south':
                ax[-(side_idx+1)].imshow(image_half, cmap='gray')
        elif im_orientation == 'horizontal':
            # ax[i].imshow(image_halves_rotated[i], cmap='gray')
            if direction == 'east':
                ax[side_idx].imshow(image_half, cmap='gray')
            elif direction == 'west':
                ax[-(side_idx+1)].imshow(image_half, cmap='gray')

        # Plot the follicles (follicle_x, follicle_y) as circles of the same color as 
        # the corresponding whisker, with intensity defined by whisker scores
        for fx, fy, score, color in zip(follicle_x[face_side],
                                        follicle_y[face_side],
                                        whisker_scores[face_side],
                                        colors[face_side]):
            # If the score is below 0.5, set the color to red
            if score < 0.5:
                color = 'red'
                alpha_level = 0.1
            else:
                alpha_level = score
            if im_orientation == 'vertical':
                if direction == 'north':
                    ax[side_idx].scatter(fx, fy, s=100, c=[color], alpha=alpha_level)
                elif direction == 'south':
                    ax[-(side_idx+1)].scatter(fx, fy, s=100, c=[color], alpha=alpha_level)
            elif im_orientation == 'horizontal':
                if direction == 'east':
                    ax[side_idx].scatter(fx, fy, s=100, c=[color], alpha=alpha_level)
                if direction == 'west':
                    ax[-(side_idx+1)].scatter(fx, fy, s=100, c=[color], alpha=alpha_level)

        # Plot the whiskers
        # for whisker_id, score, color in zip(whisker_ids[face_side],
        #                              whisker_scores[face_side],
        #                              colors[face_side]):
        for idx, (whisker_id, score, color) in enumerate(zip(whisker_ids[face_side],
                                                                whisker_scores[face_side],
                                                                colors[face_side])):
            # Get the whisker pixel values for this whisker ID from xpixels and ypixels
            whisker_x = xpixels[face_side][idx] #whisker_id
            whisker_y = ypixels[face_side][idx] #whisker_id
            # If the score is below 0.5, set the color to red
            if whisker_scores[face_side][idx] < 0.5:
                color = 'red'
                alpha_level = 0.1
                # print(f"Whisker ID {whisker_id} for {side} face side has a score of {whisker_scores[side][whisker_id]}.")
            else:
                alpha_level = score
            # alpha_level = 1
            if im_orientation == 'vertical':
                if direction == 'north':
                    ax[side_idx].plot(whisker_x, whisker_y, color=color, alpha=alpha_level)
                elif direction == 'south':
                    ax[-(side_idx+1)].plot(whisker_x, whisker_y, color=color, alpha=alpha_level)
            elif im_orientation == 'horizontal':
                if direction == 'east':
                    ax[side_idx].plot(whisker_x, whisker_y, color=color, alpha=alpha_level)
                elif direction == 'west':
                    ax[-(side_idx+1)].plot(whisker_x, whisker_y, color=color, alpha=alpha_level)

        if im_orientation == 'vertical':
            ax[-(side_idx+1)].axis('off')
            ax[-(side_idx+1)].set_title(f"Face side: {face_side}")
        elif im_orientation == 'horizontal':
            # ax[i].axis('off')
            ax[-(side_idx+1)].axis('off')
            if side_idx == 1:
                # ax[i].set_title(f"Face image side: {im_side}")
                ax[-(side_idx+1)].set_title(f"Face side: {face_side}")
            elif side_idx == 0:
                # ax[i].text(0.5, -0.1, f"Face side: {im_side}", size=12, ha="center", transform=ax[i].transAxes)
                ax[-(side_idx+1)].text(0.5, -0.1, f"Face side: {face_side}", size=12, ha="center", transform=ax[-(side_idx+1)].transAxes)
            
    plt.show()

    # save the figure
    output_file = output_dir / f'{base_name}_first_frame_whiskers_scores.png'
    fig.savefig(output_file, bbox_inches='tight', dpi=300)


def track_whiskers(input_file, whiskerpad_params, splitUp, base_name=None, output_dir=None):
    """
    Track whiskers on the first frame of a video
    """

    input_dir = Path(input_file).parent
    output_dir = Path(output_dir)

    if base_name is None:
        base_name = Path(input_file).stem

    # Save image halves as tif files
    image_halves, image_sides, face_sides, fp = wp.get_side_image(str(input_file), splitUp)
    if fp.FaceOrientation == 'down':
        direction='south'
    elif fp.FaceOrientation == 'up':
        direction='north'
    elif fp.FaceOrientation == 'left':
        direction='west'
    elif fp.FaceOrientation == 'right':
        direction='east'

    face_im_sides = [whiskerpads['ImageBorderAxis'] for whiskerpads in whiskerpad_params['whiskerpads']]

    output_dir = save_image_halves(image_halves, image_sides, face_im_sides, base_name, input_dir, direction)
    
    # Run the whisker tracking
    run_whisker_tracking(image_halves, face_sides, face_im_sides, base_name, output_dir)

    # Load the whisker data
    xpix, ypix, whisker_ids, colors, whisker_lengths, whisker_scores, follicle_x, follicle_y = load_whisker_data(face_sides, base_name, output_dir, save_to_csv=True)
    
    # Plot whiskers over the image for each side 
    plot_whiskers_on_image(image_halves, face_sides, direction, follicle_x, follicle_y, whisker_ids, whisker_scores, xpix, ypix, colors, base_name, output_dir)
    
    return whisker_ids, whisker_lengths, whisker_scores, follicle_x, follicle_y
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Track whiskers on first frame of video')
    parser.add_argument('input_file', type=str, help='Input video file')
    parser.add_argument('whiskerpad_params', type=str, help='Whiskerpad parameters json file')
    parser.add_argument('-s','--splitUp', type=bool, default=False, help='Whether to split up the video')
    parser.add_argument('-b','--base_name', type=str, default=None, help='Base name for the output files')
    parser.add_argument('-o', '--output_dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    
    if args.base_name is None:
        args.base_name = Path(args.input_file).stem
            
    if args.output_dir is None:
        args.output_dir = Path(args.input_file).parent / f'{Path(args.input_file).stem}_first_frame'
    else:
        args.output_dir = Path(args.output_dir)

    track_whiskers(args.input_file, args.whiskerpad_params, args.splitUp, args.base_name, args.output_dir)