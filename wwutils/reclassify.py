"""
Script to re-classify whiskers based on features such as curvature and follicle x/y.

Example:
python reclassify.py ../example_files/excerpt_data.parquet ../example_files/whiskerpad_TopCam.json

Or simply pass the protraction direction as an argument:
python reclassify.py ../example_files/excerpt_data.parquet "downward" --plot
"""
# %%
############# Import Libraries ############
###########################################
import pandas as pd
import numpy as np
from collections import deque
from multiprocessing import Pool
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.spatial import ConvexHull
import seaborn as sns
import cv2
from itertools import groupby
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
import psutil
import sys
import json
import os
import argparse
import wwutils.plot_overlay as po

# %%
############# Define Functions ############
###########################################
def plot_whisker_data(df, length_threshold=20, score_threshold=100):
    """
    Plots average score and length per whisker ID with horizontal threshold lines.

    Parameters:
    df (DataFrame): The input data frame containing 'face_side', 'wid', 'score', and 'length' columns.
    length_threshold (int): The threshold value for length.
    score_threshold (int): The threshold value for score.
    """
    # Group data for the first plot
    grouped_score = df.groupby(['face_side', 'wid'])['score'].mean().reset_index()

    # Group data for the second plot
    grouped_length = df.groupby(['face_side', 'wid'])['length'].mean().reset_index()

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # First plot: Average Score per Whisker ID (wid)
    sns.scatterplot(ax=axes[0], x='wid', y='score', hue='face_side', data=grouped_score)
    axes[0].axhline(score_threshold, color='red', linestyle='--', label=f'Score Threshold ({score_threshold})')
    axes[0].set_title('Average Score per Whisker ID (wid)')
    axes[0].set_xlabel('Whisker ID (wid)')
    axes[0].set_ylabel('Average Score')
    axes[0].legend(title='Face Side')

    # Second plot: Average Length per Whisker ID (wid)
    sns.scatterplot(ax=axes[1], x='wid', y='length', hue='face_side', data=grouped_length)
    axes[1].axhline(length_threshold, color='red', linestyle='--', label=f'Length Threshold ({length_threshold})')
    axes[1].set_title('Average Length per Whisker ID (wid)')
    axes[1].set_xlabel('Whisker ID (wid)')
    axes[1].set_ylabel('Average Length')
    axes[1].legend(title='Face Side')

    # Show the plots
    plt.tight_layout()
    plt.show()

def plot_whisker_angle(w_times, w_angles, wids):
    fig = go.Figure()

    # Add data for each whisker
    for w_time, w_angle, wid in zip(w_times, w_angles, wids):
        fig.add_trace(go.Scatter(
            x=w_time,
            y=w_angle,
            mode='lines',
            name=f'wid {wid}'
        ))

    fig.update_layout(
        title='Angle Over Time for Multiple Whiskers',
        xaxis_title='Frame ID',
        yaxis_title='Angle (degrees)',
        legend_title='Legend',
        template='plotly_white'
    )

    fig.show()

def plot_whisker_curvature(w_times, w_curvatures, wids, pairing=False):
    fig = go.Figure()

    # Add data for each whisker
    for i, (w_time, w_curvature, wid) in enumerate(zip(w_times, w_curvatures, wids)):
        line_style = 'lines'
        # if pairing and i % 2 == 1:
        #     line_style = 'lines+markers'
        
        fig.add_trace(go.Scatter(
            x=w_time,
            y=w_curvature,
            mode='lines',
            name=f'wid {wid}',
            line=dict(dash='dash' if pairing and i % 2 == 1 else 'solid')
        ))

    fig.update_layout(
        title='Whisker Curvature Over Time for Multiple Whiskers',
        xaxis_title='Frame ID',
        yaxis_title='Curvature',
        legend_title='Legend',
        template='plotly_white'
    )

    fig.show()
    
def plot_whisker_follicle_loc(w_fol_xs, w_fol_ys, wids):
    fig = go.Figure()

    # Add data for each whisker
    for w_fol_x, w_fol_y, wid in zip(w_fol_xs, w_fol_ys, wids):
        fig.add_trace(go.Scatter(
            x=w_fol_x,
            y=w_fol_y,
            mode='lines',
            name=f'wid {wid}'
        ))

    fig.update_layout(
        title='Whisker Follicle Location for Multiple Whiskers',
        xaxis_title='Frame X Position',
        yaxis_title='Follicle Y Position',
        legend_title='Legend',
        template='plotly_white'
    )

    fig.show()
 
def plot_overlay(frame_data, frame_num):
    data_dir = '/home/wanglab/data/whisker_asym/WA003/WA003_082824'
    video_file = 'WA003_082824_01_TopCam.mp4'
    cap = cv2.VideoCapture(f'{data_dir}/{video_file}')

    # Read frames sequentially until the desired frame
    # Setting the frame position with CAP_PROP_POS_FRAMES does not work for all video formats
    current_frame = 0
    while current_frame < frame_num:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame at position {current_frame}")
            break
        current_frame += 1

    # Verify the frame position
    print(f"Current frame position: {current_frame}")

    # Display the frame
    plt.figure(figsize=(8, 6))    
    
    # Create set of colors for up to 20 whiskers, starting with red, green, blue
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255),
                (0,255,255), (128,0,0), (0,128,0), (0,0,128), (128,128,0),
                (128,0,128), (0,128,128), (64,0,0), (0,64,0), (0,0,64),
                (64,64,0), (64,0,64), (0,64,64), (192,0,0), (0,192,0)]
    
    for index, whisker_data in frame_data.iterrows():
        color_index = index % len(colors)
        color = colors[color_index]
        print(f"Whisker ID: {whisker_data['wid']}, color: {color}")

        print(whisker_data['pixels_x'][0], whisker_data['pixels_y'][0])
        for j in range(whisker_data['pixels_x'].shape[0]):
            x = int(whisker_data['pixels_x'][j])
            y = int(whisker_data['pixels_y'][j])
            cv2.circle(frame, (x, y), 2, color, -1)
            
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Save the plot
    plt.savefig(f'{data_dir}/plots/overlay_frame_{frame_num}.png')

    cap.release()      

# Define the cost function
def compute_cost_matrix(tracks, detections, max_cost=50, position_weight=0.4, angle_weight=0.1, length_weight=0.2, wid_weight=0.1, curvature_weight=0.5, normalize=True):
    """
    Compute the cost matrix between tracks and detections for assignment.

    Handles both cases:
    1. Tracks with a nested 'detection' key (as in the tracking loop).
    2. Flattened track/detection dictionaries.

    Parameters:
        tracks (list[dict]): List of tracks.
        detections (list[dict]): List of detections.
        max_cost (float): Maximum cost for assignment.

    Returns:
        scipy.sparse.csr_matrix: Sparse cost matrix.
    """
    num_tracks = len(tracks)
    num_detections = len(detections)
    cost_matrix = np.full((num_tracks, num_detections), max_cost + 1, dtype=np.float32)  # Initialize with max cost
    
    # Handle both nested and flat formats
    if tracks and 'detection' in tracks[0]:
        tracks = [track['detection'] for track in tracks]

    # Compute ranges once, outside the loop
    if normalize:
        follicle_x_range = np.ptp([det['follicle_x'] for det in detections]) or 1
        follicle_y_range = np.ptp([det['follicle_y'] for det in detections]) or 1
        angle_range = np.ptp([det['angle'] for det in detections]) or 1
        length_range = np.ptp([det['length'] for det in detections]) or 1
        curvature_range = np.ptp([det['curvature'] for det in detections]) or 1
    else:
        follicle_x_range, follicle_y_range, angle_range, length_range, curvature_range = 1, 1, 1, 1, 1

    # Predefine weights
    # position_weight, angle_weight, length_weight, wid_weight, curvature_weight = 0.4, 0.1, 0.2, 0.1, 0.5

    # Calculate cost matrix in a vectorized way
    for i, track in enumerate(tracks):
        track_face_side = track['face_side']

        # Skip calculation for NaN values
        if np.isnan(track['follicle_x']):
            continue

        for j, detection in enumerate(detections):
            # Face side and NaN check
            if track_face_side != detection['face_side'] or np.isnan(detection['follicle_x']):
                continue
            
            # Vectorized cost calculation
            position_cost = (
                np.hypot(
                    (track['follicle_x'] - detection['follicle_x']) / follicle_x_range,
                    (track['follicle_y'] - detection['follicle_y']) / follicle_y_range
                ) * 100 * position_weight
            )

            angle_cost = (abs(track['angle'] - detection['angle']) / angle_range) * 100 * angle_weight
            length_cost = (abs(track['length'] - detection['length']) / length_range) * 100 * length_weight
            wid_diff = abs(track['wid'] - detection['wid'])
            wid_cost = wid_weight / (1 + wid_diff)
            curvature_cost = (abs(track['curvature'] - detection['curvature']) / curvature_range) * 100 * curvature_weight
            
            total_cost = position_cost + angle_cost + length_cost + wid_cost + curvature_cost
            
            # Set cost in matrix
            cost_matrix[i, j] = total_cost if total_cost < max_cost else max_cost + 1
    
    # Convert to sparse matrix for faster operations
    return csr_matrix(cost_matrix, dtype=np.float32)

# Tracking loop
def tracking_loop(whisker_df, max_cost=50, max_missed_frames=20):
    tracks = []
    next_track_id = 0
    frame_ids = whisker_df['fid'].unique()
    total_frames = len(frame_ids)
    track_histories = []
    
    # Define the frequency of progress updates
    update_frequency = 1000

    for idx, frame_id in enumerate(frame_ids):
        if idx % update_frequency == 0 or idx == len(frame_ids) - 1:
            progress = int(20 * (idx + 1) / total_frames)
            sys.stdout.write(f"\rProcessing frame {idx + 1}/{total_frames} [{'#' * progress}{'.' * (20 - progress)}]")
            sys.stdout.flush()

        frame_data = whisker_df[whisker_df['fid'] == frame_id]
        
        # plot_overlay(frame_data, frame_id)
        
        detections = frame_data.to_dict('records')
        
        if tracks:
            # tracks are existing objects being tracked from previous frames
            # detections are objects detected in the current frame
            cost_matrix = compute_cost_matrix(tracks, detections, max_cost=max_cost)
            # The row_ind and col_ind arrays represent the optimal assignment found by the algorithm, where:
            # row_ind[i] is the index of the track.
            # col_ind[i] is the index of the detection assigned to that track.
            row_ind, col_ind = linear_sum_assignment(cost_matrix.toarray())
            # For example, if row_ind has 7 elements [0, 1, 2, 3, 6, 8, 9], and col_ind has 7 elements [5, 4, 3, 1, 2, 6, 0]. This indicates that:
            # Track 0 is assigned to detection 5.
            # Track 1 is assigned to detection 4.
            # Track 2 is assigned to detection 3. 
            # Track 3 is assigned to detection 1.
            # Track 6 is assigned to detection 2.
            # Track 8 is assigned to detection 6.
            # Track 9 is assigned to detection 0.
            
            # Assignment loop
            assigned_tracks, assigned_detections = set(), set()
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] <= max_cost:
                    tracks[i]['detection'] = detections[j]
                    tracks[i]['missed_frames'] = 0
                    assigned_tracks.add(i)
                    assigned_detections.add(j)
                else:
                    tracks[i]['missed_frames'] += 1
            
            # Remove tracks that exceed missed frames threshold
            tracks = [track for i, track in enumerate(tracks) if track['missed_frames'] <= max_missed_frames]
            
            # Add new tracks
            for j, detection in enumerate(detections):
                if j not in assigned_detections:
                    new_track = {
                        'track_id': next_track_id,
                        'original_wid': detection['wid'],
                        'face_side': detection['face_side'],
                        'detection': detection,
                        'missed_frames': 0
                    }
                    tracks.append(new_track)
                    next_track_id += 1
        else:
            for detection in detections:
                new_track = {
                    'track_id': next_track_id,
                    'original_wid': detection['wid'],
                    'face_side': detection['face_side'],
                    'detection': detection,
                    'missed_frames': 0
                }
                tracks.append(new_track)
                next_track_id += 1
        
        # Collect track history
        track_histories.extend({
            'fid': frame_id,
            'track_id': track['track_id'],
            'original_wid': track['original_wid'],
            **track['detection']
        } for track in tracks)

    # Ensure the final progress is printed
    progress = 20
    sys.stdout.write(f"\rProcessing frame {total_frames}/{total_frames} [{'#' * progress}{'.' * (20 - progress)}]\n")
    sys.stdout.flush()

    return track_histories

# Tracking loop with profiling
def optimized_tracking_loop(whisker_df, max_cost=50, max_missed_frames=20):
    tracks = deque()
    next_track_id = 0
    frame_ids = whisker_df['fid'].unique()
    total_frames = len(frame_ids)
    track_histories = []

    # Profiling data lists
    frame_durations = []
    num_active_tracks = []
    memory_usage = []
    update_frequency = 1000

    for idx, frame_id in enumerate(frame_ids):
        frame_start_time = time.time()

        # Update progress occasionally
        if idx % update_frequency == 0 or idx == len(frame_ids) - 1:
            progress = int(20 * (idx + 1) / total_frames)
            sys.stdout.write(f"\rProcessing frame {idx + 1}/{total_frames} [{'#' * progress}{'.' * (20 - progress)}]")
            sys.stdout.flush()

        frame_data = whisker_df[whisker_df['fid'] == frame_id]
        detections = frame_data.to_dict('records')

        if tracks:
            cost_matrix = compute_cost_matrix(tracks, detections, max_cost=max_cost)
            row_ind, col_ind = linear_sum_assignment(cost_matrix.toarray())
            
            assigned_tracks, assigned_detections = set(), set()
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] <= max_cost:
                    tracks[i]['detection'] = detections[j]
                    tracks[i]['missed_frames'] = 0
                    assigned_tracks.add(i)
                    assigned_detections.add(j)
                else:
                    tracks[i]['missed_frames'] += 1

            # Remove expired tracks
            for _ in range(len(tracks)):
                track = tracks.popleft()
                if track['missed_frames'] <= max_missed_frames:
                    tracks.append(track)
                else:
                    del track

            # Add unmatched detections as new tracks
            for j, detection in enumerate(detections):
                if j not in assigned_detections:
                    new_track = {
                        'track_id': next_track_id,
                        'original_wid': detection['wid'],
                        'face_side': detection['face_side'],
                        'detection': detection,
                        'missed_frames': 0
                    }
                    tracks.append(new_track)
                    next_track_id += 1
        else:
            # Initialize tracks on first frame
            for detection in detections:
                new_track = {
                    'track_id': next_track_id,
                    'original_wid': detection['wid'],
                    'face_side': detection['face_side'],
                    'detection': detection,
                    'missed_frames': 0
                }
                tracks.append(new_track)
                next_track_id += 1

        # Collect track history, ensuring no duplicate entries
        frame_history = [{
            'fid': frame_id,
            'track_id': track['track_id'],
            'original_wid': track['original_wid'],
            **track['detection']
        } for track in tracks]
        
        # Append only unique entries for this frame
        track_histories.extend(pd.DataFrame(frame_history).drop_duplicates(
            subset=['fid', 'track_id', 'original_wid']).to_dict('records'))

        # Profiling data
        frame_end_time = time.time()
        frame_durations.append(frame_end_time - frame_start_time)
        num_active_tracks.append(len(tracks))
        memory_usage.append(psutil.Process().memory_info().rss / (1024 ** 2))

    # Final progress print
    sys.stdout.write(f"\rProcessing frame {total_frames}/{total_frames} [{'#' * 20}]\n")
    sys.stdout.flush()

    # Plot profiling results
    # plt.figure(figsize=(14, 5))

    # plt.subplot(1, 3, 1)
    # plt.plot(frame_durations, label="Frame Duration (s)")
    # plt.xlabel("Frame")
    # plt.ylabel("Duration (s)")
    # plt.title("Frame Processing Time")
    # plt.legend()

    # plt.subplot(1, 3, 2)
    # plt.plot(num_active_tracks, label="Number of Active Tracks")
    # plt.xlabel("Frame")
    # plt.ylabel("Number of Tracks")
    # plt.title("Number of Active Tracks Over Time")
    # plt.legend()

    # plt.subplot(1, 3, 3)
    # plt.plot(memory_usage, label="Memory Usage (MB)")
    # plt.xlabel("Frame")
    # plt.ylabel("Memory (MB)")
    # plt.title("Memory Usage Over Time")
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

    return track_histories, frame_durations, num_active_tracks, memory_usage

def filter_by_min_curvature_deviation(df):
    # Sort the DataFrame to ensure 'fid' and 'original_wid' are in order
    df = df.sort_values(by=['original_wid', 'fid']).reset_index(drop=True)
    
    # Group by 'original_wid' to operate on each whisker track independently
    groups = df.groupby('original_wid')
    
    # Calculate previous curvature using group shift to avoid manual previous frame search
    df['previous_curvature'] = groups['curvature'].shift(1)
    
    # Calculate the absolute curvature deviation from the previous frame
    df['curvature_deviation'] = (df['curvature'] - df['previous_curvature']).abs()
    
    # For the first frame of each 'original_wid', keep it by default (as there's no previous curvature)
    df['keep'] = groups['fid'].transform(lambda x: x == x.min())
    
    # Filter out groups where all values in 'curvature_deviation' are NaN
    valid_deviation_idx = df[~df['curvature_deviation'].isna()].groupby(['fid', 'original_wid'])['curvature_deviation'].idxmin()
    
    # Mark rows with minimum valid deviation to keep them
    df.loc[valid_deviation_idx, 'keep'] = True
    
    # Filter the DataFrame to keep only marked rows
    return df[df['keep']].drop(columns=['keep', 'previous_curvature', 'curvature_deviation']).reset_index(drop=True)
    
def detect_and_swap_crossings(df):
    # Sort the DataFrame by 'face_side', 'original_wid', and 'fid' for consistent processing
    df = df.sort_values(by=['face_side', 'original_wid', 'fid']).reset_index(drop=True)
    
    # Initialize list to store rows that need to be swapped
    swap_pairs = []

    # Process each face side separately
    for face_side, side_df in df.groupby('face_side'):
        # Get the unique 'original_wid' values for this face side, sorted, and restrict to the first 3
        unique_wids = sorted(side_df['original_wid'].unique())[:3]
        
        # Iterate over pairs of neighboring 'original_wid'
        for wid1, wid2 in zip(unique_wids, unique_wids[1:]):
            # Filter data for these two whiskers
            wid1_df = side_df[side_df['original_wid'] == wid1]
            wid2_df = side_df[side_df['original_wid'] == wid2]
            
            # Merge on 'fid' to detect crossings for matching frames
            merged_df = wid1_df.merge(wid2_df, on='fid', suffixes=('_1', '_2'))

            # Verify that columns with suffixes exist after the merge
            required_columns = ['follicle_x_1', 'follicle_x_2', 'curvature_1', 'curvature_2']
            if not all(col in merged_df.columns for col in required_columns):
                print(f"Expected columns not found in merged DataFrame: {required_columns}")
                continue  # Skip this pair if merge didn't produce expected columns
            
            # Calculate the curvature difference
            curvature_diff = merged_df['curvature_1'] - merged_df['curvature_2']
            
            # Determine if there’s a general trend for curvature ordering between the whiskers
            if curvature_diff.mean() > 0:
                crossing_frames = merged_df[curvature_diff < 0]['fid']
            else:
                crossing_frames = merged_df[curvature_diff > 0]['fid']
            
            # Find continuous crossing sequences for these frames
            crossing_sequences = [
                list(group) for _, group in groupby(crossing_frames, key=lambda x, c=iter(range(len(crossing_frames))): next(c) - x)
            ]
            
            # Evaluate and collect swaps for each sequence
            for crossing in crossing_sequences:
                first_frame = crossing[0]
                prev_frame = first_frame - 1
                
                # Get previous frame data for continuity comparison
                prev_frame_row = df[(df['fid'] == prev_frame) & (df['original_wid'] == wid1)]
                
                # Skip if no data for previous frame
                if prev_frame_row.empty:
                    continue
                
                # Calculate continuity costs for both whiskers in the crossing frame
                track_detection = prev_frame_row.iloc[0]
                wid1_detection = merged_df.loc[merged_df['fid'] == first_frame, ['follicle_x_1', 'follicle_y_1', 'curvature_1']].iloc[0]
                wid2_detection = merged_df.loc[merged_df['fid'] == first_frame, ['follicle_x_2', 'follicle_y_2', 'curvature_2']].iloc[0]
                
                # Continuity costs based on position and curvature deviation
                position_cost_1 = np.hypot(track_detection['follicle_x'] - wid1_detection['follicle_x_1'],
                                           track_detection['follicle_y'] - wid1_detection['follicle_y_1'])
                curvature_cost_1 = abs(track_detection['curvature'] - wid1_detection['curvature_1'])
                position_cost_2 = np.hypot(track_detection['follicle_x'] - wid2_detection['follicle_x_2'],
                                           track_detection['follicle_y'] - wid2_detection['follicle_y_2'])
                curvature_cost_2 = abs(track_detection['curvature'] - wid2_detection['curvature_2'])
                
                # Total continuity cost for each whisker
                total_cost_1 = position_cost_1 + curvature_cost_1 * 10000
                total_cost_2 = position_cost_2 + curvature_cost_2 * 10000
                
                # Collect pairs to swap if wid1 has higher continuity cost than wid2
                if total_cost_1 > total_cost_2:
                    for fid in crossing:
                        swap_pairs.append((fid, wid1, wid2))

    # Perform swaps in one go to reduce access time
    for fid, wid1, wid2 in swap_pairs:
        # Locate the indices of the rows for wid1 and wid2 in the original dataframe
        row1_index = df[(df['fid'] == fid) & (df['original_wid'] == wid1)].index[0]
        row2_index = df[(df['fid'] == fid) & (df['original_wid'] == wid2)].index[0]

        # Columns to swap (excluding identifiers)
        columns_to_swap = [col for col in df.columns if col not in ['track_id', 'original_wid', 'wid']]

        # Swap the rows
        df.loc[row1_index, columns_to_swap], df.loc[row2_index, columns_to_swap] = (
            df.loc[row2_index, columns_to_swap].values,
            df.loc[row1_index, columns_to_swap].values,
        )
    
    return df

# Function to apply temporal smoothing to tracking data
def apply_temporal_smoothing(tracks_df, window_size=3):
    # Define features to smooth
    smoothing_features = ['angle', 'curvature', 'follicle_x', 'follicle_y']
    smoothed_df = tracks_df.copy()

    for feature in smoothing_features:
        smoothed_df[feature] = (
            smoothed_df.groupby('track_id')[feature]
            .transform(lambda x: x.rolling(window=window_size, min_periods=1, center=True).mean())
        )
    return smoothed_df

# Function to process a single segment
def process_segment(segment_data, segment_id, max_cost=50, max_missed_frames=20):
    # Call the optimized tracking loop
    track_histories, _, _, _ = optimized_tracking_loop(
        segment_data, max_cost=max_cost, max_missed_frames=max_missed_frames
    )
    # Tag the results with the segment ID for debugging
    tracks_df = pd.DataFrame(track_histories)
    tracks_df['segment_id'] = segment_id
    
    # Check for duplicates in tracks_df
    duplicates = tracks_df[tracks_df.duplicated(subset=['fid', 'track_id', 'original_wid', 'wid'], keep=False)]
    # print(f"Number of duplicate entries: {len(duplicates)}")
    # Remove duplicates from tracks_df
    tracks_df = tracks_df.drop_duplicates(subset=['fid', 'track_id', 'original_wid', 'wid'])
    # print("Duplicates removed.")
    
    # Sort by 'fid' and 'original_wid' 
    tracks_df = tracks_df.sort_values(by=['fid', 'original_wid']).reset_index(drop=True)
    
    # Filter by Minimum Curvature Deviation
    filtered_tracks_df = filter_by_min_curvature_deviation(tracks_df)

    # Detect and Swap Crossings
    updated_tracks_df = detect_and_swap_crossings(filtered_tracks_df)
    updated_tracks_df = updated_tracks_df.sort_values(by=['fid', 'original_wid']).reset_index(drop=True)
    
    duplicates = detect_duplicates(updated_tracks_df)
    if not duplicates.empty:
        print(f"Duplicate entries found in Segment {segment_id}")
        print(duplicates)
        
    # Call order_whiskers to reorder the whiskers
    
    # Get the most frequent whisker ids 
    # frequent_wid_idx = find_frequent_whiskers(tracks_df, wid_field='original_wid', min_frame_ratio=0.9)
    # unique_wids = tracks_df[frequent_wid_idx]['original_wid'].unique()

    # ordered_tracks_df = order_whiskers(updated_tracks_df, whiskerpad=None, wids=None)
        
    # Apply smoothing to tracking data
    tracks_df_smoothed = apply_temporal_smoothing(updated_tracks_df, window_size=3)

    return tracks_df_smoothed

# Parallelized processing
def process_data_in_segments(whisker_df, segment_size=1000, max_cost=50, max_missed_frames=20, num_workers=4):
    """
    Processes whisker data in segments using parallel processing.
    
    Args:
        whisker_df (pd.DataFrame): Input DataFrame containing whisker data.
        segment_size (int): Number of frames per segment.
        max_cost (float): Maximum cost for processing segments.
        max_missed_frames (int): Maximum number of missed frames allowed.
    
    Returns:
        pd.DataFrame: Concatenated processed DataFrame.
    """
    # Split data into segments
    frame_ids = whisker_df['fid'].unique()
    segments = [
        whisker_df[whisker_df['fid'].isin(frame_ids[i:i + segment_size])]
        for i in range(0, len(frame_ids), segment_size)
    ]
    
    # Prepare arguments for parallel processing
    args = [
        (segment, idx, max_cost, max_missed_frames)
        for idx, segment in enumerate(segments)
    ]
    
    # Process segments in parallel
    with Pool(num_workers) as pool:
        segment_results = pool.starmap(process_segment, args)
    
    # # Process segments sequentially in a loop
    # segment_results = []
    # for idx, segment in enumerate(segments):
    #     print(f"Processing segment {idx+1}/{len(segments)}")
    #     result = process_segment(segment, idx, max_cost, max_missed_frames)
    #     segment_results.append(result)
    
    # Concatenate all segment results
    processed_data = pd.concat(segment_results, ignore_index=True)
    
    # Reconcile tracks across segment boundaries
    # processed_data = reconcile_segment_boundaries(processed_data)
    
    return processed_data

def detect_duplicates(whisker_df, wid_field='original_wid'):
    duplicates = (
        whisker_df.groupby(['fid', wid_field])
        .size()
        .reset_index(name='count')
        .query('count > 1')
    )
    return duplicates

def fix_duplicates(next_segment, current_tracks = None, max_cost=50):

    next_duplicates = detect_duplicates(next_segment)
    
    # Resolve duplicates using a cost matrix
    for _, dup_row in next_duplicates.iterrows():
        dup_wid = dup_row['original_wid']   
        # Get the duplicate rows
        dup_row_data = next_segment[(next_segment['fid'] == dup_row['fid']) & (next_segment['original_wid'] == dup_wid)]
        face_side = dup_row_data['face_side'].iloc[0]
        # Get the previous row
        if dup_row['fid'] == next_segment['fid'].min():
            prev_row_target_index = current_tracks[(current_tracks['fid'] == current_tracks['fid'].max()) & (current_tracks['original_wid'] == dup_wid)].index.tolist()
            prev_row = current_tracks[(current_tracks['fid'] == current_tracks['fid'].max()) & (current_tracks['face_side'] == face_side)]
        else:
            prev_row_target_index = next_segment[(next_segment['fid'] == dup_row['fid'] - 1) & (next_segment['original_wid'] == dup_wid)].index.tolist()
            prev_row = next_segment[(next_segment['fid'] == dup_row['fid'] - 1) & (next_segment['face_side'] == face_side)]
                
        # Calculate cost matrix for the duplicates against the previous row
        cost_matrix = compute_cost_matrix(
            prev_row.to_dict('records'),
            dup_row_data.to_dict('records'),
            max_cost=max_cost * 100, # Increase the max_cost to compensate for limited ranges
            position_weight=0.8,
            angle_weight=0.05,
            length_weight=0.2,
            wid_weight=0.05,
            curvature_weight=0.25,
            normalize=False
        )
        
        # Perform assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix.toarray())
        
        if len(row_ind) == 0:
            # Identify the index of the duplicate with the highest score
            highest_score_index = dup_row_data['score'].idxmax()
            
            # Assign new IDs to all duplicates except the one with the highest score
            new_wid = next_segment['original_wid'].max() + 1
            for idx in dup_row_data.index:
                if idx != highest_score_index:
                    next_segment.loc[idx, 'original_wid'] = new_wid
                    new_wid += 1
        else:
            # Create a list of matches with their costs
            assignments = [(r, c, cost_matrix[r, c]) for r, c in zip(row_ind, col_ind)]

            # Sort the assignments by cost (ascending order)
            assignments.sort(key=lambda x: x[2])

            # Process assignments starting with the best matches
            for r, c, cost in assignments:
                if cost <= max_cost * 100:
            # for r, c in zip(row_ind, col_ind):
            #     if cost_matrix[r, c] <= max_cost * 100:
                    if prev_row.iloc[r].name in prev_row_target_index:
                        # If the previous row is the target, we should just be able to skip, but check its original_wid. It should hopefully match the duplicate's original_wid
                        if prev_row.iloc[r]['original_wid'] == dup_row['original_wid']:
                        # Don't change the original_wid then
                            continue
                            
                    # Assign the `original_wid` from the `prev_row` to the appropriate duplicate in `dup_row_data`                         

                    resolved_og_wid = prev_row.iloc[r]['original_wid']
                    dup_row_index = dup_row_data.iloc[c].name
                    # print(f"Resolved duplicate {dup_wid} in frame {dup_row['fid']} with index {dup_row_index} and cost {cost_matrix[r, c]}. Assigned original_wid: {resolved_og_wid}")
                    
                    if resolved_og_wid not in next_segment[(next_segment['fid'] == dup_row['fid'])]['original_wid'].values:
                        # Assign the resolved `original_wid` (from the previous row) to the duplicate
                        next_segment.at[dup_row_index, 'original_wid'] = resolved_og_wid
                    # Fall back 1    
                    elif next_segment.loc[dup_row_index, 'wid'] not in next_segment[(next_segment['fid'] == dup_row['fid'])]['original_wid'].values:
                        # Assign the selected duplicate's wid as the duplicate's original_wid
                        next_segment.at[dup_row_index, 'original_wid'] = next_segment.loc[dup_row_index, 'wid']
                    # Fall back 2
                    elif prev_row.iloc[r]['wid'] not in next_segment[(next_segment['fid'] == dup_row['fid'])]['original_wid'].values:
                        # Assign the matching duplicate's wid as its original_wid
                        next_segment.at[dup_row_index, 'original_wid'] = prev_row.iloc[r]['wid']
                    # Fall back to assigning a unique new ID and label it for later review    
                    else:
                        # Maybe the job is done, check the current original_wid of the duplicate
                        # if next_segment.loc[dup_row_index, 'original_wid'] == dup_row['original_wid']:
                        #     continue
                        face_side = dup_row_data.iloc[c]['face_side']
                        new_wid = next_segment.loc[next_segment['face_side'] == face_side, 'original_wid'].max() + 1
                        next_segment.at[dup_row_index, 'original_wid'] = new_wid
                        next_segment.at[dup_row_index, 'label'] = -1

                    # # Update `next_segment` with the resolved `original_wid`
                    # Not needed as the `original_wid` is already that value
                    # next_segment.at[dup_row_index, 'original_wid'] = resolved_og_wid
                    
                    # Try to assign a wid to original_wid. 
                    # indices_to_drop = []
                    # for dup_index, dup_row in dup_row_data.iterrows():
                    #     if dup_index != dup_row_index:
                            # if dup_row['wid'] not in next_segment[(next_segment['fid'] == dup_row['fid'])]['original_wid'].values:
                            #     # Assign the "dropped" duplicate's wid as its original_wid
                            #     next_segment.at[dup_index, 'original_wid'] = dup_row['wid']
                            # elif next_segment.loc[dup_row_index, 'wid'] not in next_segment[(next_segment['fid'] == dup_row['fid'])]['original_wid'].values:
                            #     # Assign the selected duplicate's wid as the duplicate's original_wid
                            #     next_segment.at[dup_index, 'original_wid'] = next_segment.loc[dup_row_index, 'wid']
                            # else:
                                # Bar that, drop it
                                # indices_to_drop.append(dup_index)
                                # Assign a unique new ID and label it for later review
                                
                    # Drop duplicates after the loop
                    # next_segment = next_segment.drop(indices_to_drop).reset_index(drop=True)

                else:
                    # If no valid match is found (cost exceeds max_cost), handle it
                    # if there is a row in dup_row_data with the same wid as the prev_row, keep that one
                    # if prev_row['original_wid'].values[0] in dup_row_data['original_wid'].values:
                    # if any(value in dup_row_data['original_wid'].values for value in prev_row['original_wid'].values):
                    #     # Get the index of the row with the same wid as the prev_row
                    #     dup_row_index = dup_row_data[dup_row_data['original_wid'] == prev_row['original_wid'].values[0]].index[0]
                    #     # Drop all other rows with the same wid
                    #     next_segment = next_segment.drop(dup_row_data.index.difference([dup_row_index]))
                    # else:
                    print(f"No valid assignment found for duplicate {dup_wid} in frame {dup_row['fid']}, with cost {cost_matrix[r, c]}")
                    # Assign a unique new ID and label it for later review
                    face_side = dup_row_data.iloc[c]['face_side']
                    new_wid = next_segment.loc[next_segment['face_side'] == face_side, 'original_wid'].max() + 1
                    next_segment.loc[dup_row_data.iloc[c].name, 'original_wid'] = new_wid
                    next_segment.loc[dup_row_data.iloc[c].name, 'label'] = -1
                    
    # After resolving duplicates, ensure no duplicates remain in `next_segment`
    if detect_duplicates(next_segment).shape[0] > 0:
        # raise ValueError("Duplicates remain after resolution!")
        if detect_duplicates(next_segment)['fid'].nunique() < 0.01 * next_segment['fid'].nunique():  
            # if number of fid with duplicates is < 1% of the total number of fids, drop the duplicates
            next_segment = next_segment.drop_duplicates(subset=['fid', 'original_wid'], keep='first').reset_index(drop=True)
        else:
            # call fix_duplicates recursively
            next_segment, _ = fix_duplicates(next_segment)
            # detect_duplicates(next_segment)
            
        print(f"Resolved duplicates for Segment {next_segment['segment_id'].iloc[0]}")
        
    # Update next_tracks
    next_tracks = next_segment[next_segment['fid'] == next_segment['fid'].min()]
    
    return next_segment, next_tracks

# Reconciliation function
def reconcile_segment_boundaries(whisker_df, discarded_df=None, max_cost=50):
    # Identify unique segment IDs
    segment_ids = sorted(whisker_df['segment_id'].unique())
    
    for segment_index in range(len(segment_ids) - 1):
        # Get current and next segments
        current_segment = whisker_df[whisker_df['segment_id'] == segment_ids[segment_index]].sort_values(by=['fid', 'original_wid']).copy()
        next_segment = whisker_df[whisker_df['segment_id'] == segment_ids[segment_index + 1]].sort_values(by=['fid', 'original_wid']).copy()
        
        # Make sure the segments are not empty
        if current_segment.empty or next_segment.empty:
            print(f"Segment {segment_ids[segment_index]} or Segment {segment_ids[segment_index + 1]} is empty.")
            continue
        
        # Check that the last frame of the current segment matches the first frame of the next segment
        if current_segment['fid'].max() != next_segment['fid'].min() - 1:
            print(f"Segment {segment_ids[segment_index]} and Segment {segment_ids[segment_index + 1]} are not contiguous.")
            continue
        
        print(f"Reconciling Segment {segment_ids[segment_index]} with last frame {current_segment['fid'].max()} and Segment {segment_ids[segment_index + 1]} with first frame {next_segment['fid'].min()}")
        
        # Get tracks for the boundary frame in both segments
        current_tracks = current_segment[current_segment['fid'] == current_segment['fid'].max()]
        next_tracks = next_segment[next_segment['fid'] == next_segment['fid'].min()]
        
        # Cost assignment works best when there are as many or more tracks in current_tracks as in next_tracks.
        # For each face side, check here whether there are less tracks in current_tracks than in next_tracks.
        face_sides = next_tracks['face_side'].unique()
                
        for face_side in face_sides:
            current_face_tracks = current_tracks[current_tracks['face_side'] == face_side]
            next_face_tracks = next_tracks[next_tracks['face_side'] == face_side]
            
            if len(current_face_tracks) < len(next_face_tracks):
                print(f"Low count tracks in current_tracks for face side '{face_side}'. Looking up discarded_df.")
                
                # Check if discarded_df is available and has relevant tracks
                if discarded_df is not None and not discarded_df.empty:
                    discarded_candidates = discarded_df[
                        (discarded_df['face_side'] == face_side) &
                        (discarded_df['segment_id'] == segment_ids[segment_index]) &
                        (discarded_df['fid'] == current_face_tracks['fid'].max())
                    ]
                    
                    if not discarded_candidates.empty:
                        # Compute cost matrix between discarded candidates and next_face_tracks
                        cost_matrix = compute_cost_matrix(
                            discarded_candidates.to_dict('records'),
                            next_face_tracks.to_dict('records'),
                            max_cost=max_cost * 10,
                            position_weight=1,
                            angle_weight=0.1,
                            length_weight=0.4,
                            normalize=False
                        )
                        
                        # Perform assignment
                        row_ind, col_ind = linear_sum_assignment(cost_matrix.toarray())
                        
                        # Create a list of matches with their costs
                        assignments = [(r, c, cost_matrix[r, c]) for r, c in zip(row_ind, col_ind)]
                        
                        # Sort the assignments by cost (ascending order)
                        assignments.sort(key=lambda x: x[2])

                        # Determine the number of missing tracks
                        missing_count = len(next_face_tracks) - len(current_face_tracks)

                        # Process the assignments, starting with the best matches
                        for r, c, cost in assignments[:missing_count]:
                            if cost <= max_cost * 10:
                                matched_track = discarded_candidates.iloc[r].copy()
                                matched_wid = matched_track['original_wid']
                                next_track = next_face_tracks.iloc[c].copy()
                                
                                # Retrieve the full track from discarded_df going backward
                                full_track = discarded_df[
                                    (discarded_df['face_side'] == face_side) &
                                    (discarded_df['segment_id'] == segment_ids[segment_index]) &
                                    (discarded_df['track_id'] == matched_track['track_id'])
                                    ].copy()
                                  
                                # Make sure that the next_track's original_wid is not already in current_segment
                                if next_track['original_wid'] in current_face_tracks['original_wid'].values:
                                    # Keep the matched_track's original_wid, unless it's also in current_segment
                                    if matched_wid in current_face_tracks['original_wid'].values:
                                        # Assign a new original_wid to the matched track
                                        full_track.loc[:, 'original_wid'] = current_face_tracks['original_wid'].max() + 1      
                                else:
                                    # Assign the next_track's original_wid to the matched track
                                    full_track.loc[:, 'original_wid'] = next_track['original_wid']      
                                    
                                # update matched_wid
                                matched_wid = full_track['original_wid'].iloc[0]
                                if matched_wid in current_segment['original_wid'].values:
                                    stopping_frame = current_segment[current_segment['original_wid'] == matched_wid]['fid'].max()
                                    # Filter the full track to include only rows up to the stopping frame, going backward
                                    full_track = full_track[full_track['fid'] > stopping_frame]                                                   
                                    
                                # Add the matched track to current_segment
                                current_segment = pd.concat([current_segment, full_track], ignore_index=True).sort_values(by=['fid', 'original_wid'])
                                # if current_segment['original_wid'].dtype != 'int64':
                                #     print(f"original_wid data type changed: {current_segment['original_wid'].dtype}")
                                
                                # # Update whisker_df to include the newly added track
                                # whisker_df = pd.concat([whisker_df, full_track], ignore_index=True).sort_values(by=['fid', 'original_wid'])
                                
                                # # Reset index for alignment
                                # current_segment = current_segment.reset_index(drop=True)
                                # whisker_df = whisker_df.reset_index(drop=True)
                                # next_segment = next_segment.reset_index(drop=True)
                                
                                # Store original data types of whisker_df
                                original_dtypes = whisker_df.dtypes.to_dict()
                                
                                # Ensure `full_track` has the same structure as `whisker_df`
                                if not set(full_track.columns).issubset(whisker_df.columns):
                                    raise ValueError("Columns in `full_track` do not align with `whisker_df`.")

                                # Perform the merge operation
                                merged_df = whisker_df.merge(
                                    full_track,
                                    on=['fid', 'original_wid'],  # Use common identifiers
                                    how='outer',                 # Include all rows
                                    suffixes=('', '_new')        # Resolve column name conflicts
                                )

                                # Combine overlapping rows, preferring the new data
                                for col in full_track.columns:
                                    if f"{col}_new" in merged_df.columns:
                                        merged_df[col] = merged_df[f"{col}_new"].combine_first(merged_df[col])
                                        merged_df.drop(columns=[f"{col}_new"], inplace=True)
                                # List unique columns in merged_df
                                # print(merged_df.columns)
                                        
                                # Restore original data types for all columns
                                for col, dtype in original_dtypes.items():
                                    if col in merged_df.columns:
                                        merged_df[col] = merged_df[col].astype(dtype)

                                # Drop duplicates (if any) and reset index
                                # whisker_df = whisker_df.drop_duplicates(subset=['fid', 'original_wid']).reset_index(drop=True)
                                whisker_df = merged_df.reset_index(drop=True)

                                # Ensure indices in `current_segment` and `next_segment` remain consistent
                                current_segment_id = current_segment['segment_id'].iloc[0]
                                next_segment_id = next_segment['segment_id'].iloc[0]

                                # Use `.loc` for explicit indexing and avoid slicing ambiguities
                                current_segment = whisker_df.loc[whisker_df['segment_id'] == current_segment_id].copy()
                                next_segment = whisker_df.loc[whisker_df['segment_id'] == next_segment_id].copy()
                                                         
                                # Remove the added track from discarded_df
                                # discarded_df.drop(full_track.index, inplace=True, errors='ignore')
                
                # Ensure original_wid remains consistent
                current_segment['original_wid'] = current_segment['original_wid'].fillna(-1).astype('int64')
                whisker_df['original_wid'] = whisker_df['original_wid'].fillna(-1).astype('int64')
                                                
                # Remove duplicates if any
                if detect_duplicates(current_segment).shape[0] > 0:
                    current_segment = current_segment.drop_duplicates(subset=['fid', 'original_wid'], keep='first').reset_index(drop=True)
                                            
                # Update current_tracks with the newly added tracks
                current_tracks = current_segment[current_segment['fid'] == current_segment['fid'].max()]        
        
        # Check for duplicates in the next segment
        current_duplicates = detect_duplicates(current_segment)
        next_duplicates = detect_duplicates(next_segment)
        
        # If there duplicate entries in either segment, remove them
        if not current_duplicates.empty:
            # Return an error. This should not happen.
            print(f"Duplicate entries found in Segment {segment_ids[segment_index]}")
            break
            
        if not next_duplicates.empty:
                # Do cost matrix calculation specifically on those rows, starting with a comparison of the current_tracks vs next_segment rows and loop through the duplicates
                print(f"Resolving duplicates in Segment {segment_ids[segment_index + 1]} starting at frame {next_tracks['fid'].iloc[0]}")
                
                next_segment, next_tracks = fix_duplicates(next_segment, current_tracks)

        # Calculate cost matrix for the overlapping frame
        cost_matrix = compute_cost_matrix(
            current_tracks.to_dict('records'),
            next_tracks.to_dict('records'),
            max_cost=max_cost * 100, 
            position_weight=1,
            angle_weight=0.1,
            length_weight=0.4,
            normalize=False
        )

        # Perform linear sum assignment to find optimal track assignments
        row_ind, col_ind = linear_sum_assignment(cost_matrix.toarray())
        
        # Update the original_wids in the next segment to match the current segment
        next_segment['temp_wid'] = next_segment['original_wid']  # Create a temporary column for swapping

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= max_cost * 100:
                # Get original_wid values from both tracks
                current_og_wid = current_tracks.iloc[r]['original_wid']
                next_og_wid = next_tracks.iloc[c]['original_wid']
                # print(f"Current original_wid: {current_og_wid}, Next original_wid: {next_og_wid}")
                # If they are not the same, swap them in the next segment
                if next_og_wid != current_og_wid:
                    # Temporarily store current_og_wid where next_og_wid matches
                    next_segment.loc[
                        next_segment['original_wid'] == next_og_wid,
                        'temp_wid'
                    ] = current_og_wid
                                        
        # if current_og_wid in next_segment['original_wid'].values and not detect_duplicates(next_segment, wid_field='temp_wid').empty:
        #     # swap the current_og_wid with the next_og_wid
        #     next_segment.loc[
        #         next_segment['original_wid'] == current_og_wid,
        #         'temp_wid'
        #     ] = next_og_wid                        
                    
        # Plot the segment boundaries before finalizing the swap
        # plot_segment_boundaries(current_segment, next_segment)                    
                    
        # Finalize the swap by copying temp_wid back into original_wid
        next_segment['original_wid'] = next_segment['temp_wid']
        next_segment.drop(columns=['temp_wid'], inplace=True)
        
        # Plot the segment boundaries after finalizing the swap
        # plot_segment_boundaries(current_segment, next_segment)
        
        # Now there could be duplicates in the next segment
        next_duplicates = detect_duplicates(next_segment)
        if not next_duplicates.empty:
            print(f"Duplicate entries found in Segment {segment_ids[segment_index + 1]} after reconciliation.")
            # Resolve duplicates in the next segment
            # >> Currently goes to infinite iteration if duplicates are not resolved
            next_segment, _ = fix_duplicates(next_segment, current_tracks)
            
            # Plot again after resolving duplicates
            # plot_segment_boundaries(current_segment, next_segment)

        # Ensure alignment between the mask and next_segment
        mask = whisker_df['segment_id'] == segment_ids[segment_index + 1]

        if mask.sum() == len(next_segment):
            whisker_df.loc[mask, 'original_wid'] = next_segment['original_wid'].values
        else:
            print(f"Mismatch: {mask.sum()} rows in whisker_df, {len(next_segment)} rows in next_segment.")

        # # Update whisker_df with the modified next segment
        # whisker_df.loc[whisker_df['segment_id'] == segment_ids[segment_index + 1], 'original_wid'] = next_segment['original_wid']
        
        # Plot the concatenated segments 
        plot_whisker_traces(pd.concat([current_segment, next_segment]), feature={'curvature'})        

    return whisker_df

def order_whiskers(whisker_df, whiskerpad, wids=None):
    
    #  Get info the first whiskerpads in whiskerpad
    whiskerpad = whiskerpad['whiskerpads'][0]    
 
    sorting_order = {
        "downward": 'bottom_to_top',
        "upward": 'top_to_bottom',
        "rightward": 'right_to_left',
        "leftward": 'left_to_right'
    }.get(whiskerpad['ProtractionDirection'], None)
    
    if sorting_order is None:
        raise ValueError("Invalid ProtractionDirection")   
    
    # Identify unique segment IDs
    segment_ids = sorted(whisker_df['segment_id'].unique())
    
    for segment_id in segment_ids:
        current_segment = whisker_df[whisker_df['segment_id'] == segment_id].copy()
        
        plot_whisker_traces(current_segment, feature={'curvature'}) 
        
        if current_segment.empty:
            print(f"Segment {segment_id} is empty.")
            continue
        
        print(f"Reordering Segment {segment_id}")
        
        for face_side in current_segment['face_side'].unique():
            face_side_data = current_segment[current_segment['face_side'] == face_side]
            
            if face_side_data.empty or (wids is not None and face_side_data['original_wid'].isin(wids).sum() == 0):
                continue
            
            if wids is not None:
                face_side_data = face_side_data[face_side_data['original_wid'].isin(wids)]
            
            if face_side_data['original_wid'].nunique() == 1:
                continue
            
            # # Save the original indices before sorting
            # original_indices = face_side_data.index

            # # Sort `face_side_data` by fid and follicle information
            # sorted_data = face_side_data.sort_values(
            #     by=['fid', 'follicle_y' if 'y' in sorting_order else 'follicle_x'],
            #     ascending=[True, sorting_order in ['top_to_bottom', 'left_to_right']]
            # )
            
            # # Assign the sorted `original_wid` back to the original indices
            # current_segment.loc[original_indices, 'original_wid'] = sorted_data['original_wid'].values
            
            # Calculate average follicle values for each `original_wid`
            avg_follicles = face_side_data.groupby('original_wid')[['follicle_x', 'follicle_y']].mean()
            
            original_wids = avg_follicles.index.tolist()
            
            # Sort by the chosen follicle axis (x or y) based on the sorting order
            if sorting_order in ['bottom_to_top', 'top_to_bottom']:
                avg_follicles = avg_follicles.sort_values('follicle_y', ascending=sorting_order == 'top_to_bottom')
            elif sorting_order in ['right_to_left', 'left_to_right']:
                avg_follicles = avg_follicles.sort_values('follicle_x', ascending=sorting_order == 'left_to_right')
            
            # Create a mapping from old WIDs to sorted WIDs
            sorted_wids = avg_follicles.index.tolist()
            # wid_mapping = {old_wid: new_wid for old_wid, new_wid in zip(original_wids, sorted_wids)}
            wid_mapping = {new_wid: old_wid for old_wid, new_wid in zip(original_wids, sorted_wids)}
            
            # Map the sorted `original_wid` back to face_side_data
            face_side_data['original_wid'] = face_side_data['original_wid'].map(wid_mapping)
    
            # Update `current_segment` with the updated `original_wid`
            current_segment.loc[face_side_data.index, 'original_wid'] = face_side_data['original_wid']    
      
        print(f"Reordered Segment {segment_id}")
            
        plot_whisker_traces(current_segment, feature={'curvature'}) 
        
        # Update whisker_df with the modified current segment
        whisker_df.loc[whisker_df['segment_id'] == segment_id, 'original_wid'] = current_segment['original_wid']
            
    return whisker_df  

def order_whiskers_for_segment(args):
    segment_id, segment_data, protraction_direction, wids = args
    
    # Determine sorting order based on ProtractionDirection
    sorting_order = {
        "downward": 'bottom_to_top',
        "upward": 'top_to_bottom',
        "rightward": 'right_to_left',
        "leftward": 'left_to_right'
    }.get(protraction_direction, None)
    
    if sorting_order is None:
        raise ValueError(f"Invalid ProtractionDirection for segment {segment_id}")
    
    print(f"Reordering Segment {segment_id}")
    
    for face_side in segment_data['face_side'].unique():
        face_side_data = segment_data[segment_data['face_side'] == face_side]
        
        if face_side_data.empty or (wids is not None and face_side_data['original_wid'].isin(wids).sum() == 0):
            continue
        
        if wids is not None:
            face_side_data = face_side_data[face_side_data['original_wid'].isin(wids)]
        
        if face_side_data['original_wid'].nunique() == 1:
            continue
        
        # Calculate average follicle values for each `original_wid`
        avg_follicles = face_side_data.groupby('original_wid')[['follicle_x', 'follicle_y']].mean()
        
        original_wids = avg_follicles.index.tolist()
        
        # Sort by the chosen follicle axis (x or y) based on the sorting order
        if sorting_order in ['bottom_to_top', 'top_to_bottom']:
            avg_follicles = avg_follicles.sort_values('follicle_y', ascending=sorting_order == 'top_to_bottom')
        elif sorting_order in ['right_to_left', 'left_to_right']:
            avg_follicles = avg_follicles.sort_values('follicle_x', ascending=sorting_order == 'left_to_right')
        
        # Create a mapping from sorted WIDs back to original WIDs
        sorted_wids = avg_follicles.index.tolist()
        wid_mapping = {new_wid: old_wid for old_wid, new_wid in zip(original_wids, sorted_wids)}
        
        # Map the sorted `original_wid` back to face_side_data
        segment_data.loc[face_side_data.index, 'original_wid'] = face_side_data['original_wid'].map(wid_mapping)
    
    print(f"Reordered Segment {segment_id}")
    return segment_data

def order_whiskers_per_frame(whisker_df, protraction_direction, wids=None):
    """
    Reorders whiskers within each frame and face_side according to follicle positions.

    Parameters:
    - whisker_df (pd.DataFrame): The DataFrame containing whisker tracking data.
    - protraction_direction (str): The protraction direction (e.g., "downward", "upward").
    - wids (list or None): List of whisker IDs to include in the ordering.

    Returns:
    - pd.DataFrame: The reordered whisker DataFrame.
    """
    # Determine sorting order based on ProtractionDirection
    sorting_order = {
        "downward": 'bottom_to_top',
        "upward": 'top_to_bottom',
        "rightward": 'right_to_left',
        "leftward": 'left_to_right'
    }.get(protraction_direction, None)

    if sorting_order is None:
        raise ValueError(f"Invalid ProtractionDirection: {protraction_direction}")

    print(f"Reordering whiskers for each frame using direction: {protraction_direction}")

    # Group by frame ID (fid)
    for fid, frame_data in whisker_df.groupby('fid'):
        for face_side in frame_data['face_side'].unique():
            face_side_data = frame_data[frame_data['face_side'] == face_side]

            if face_side_data.empty or (wids is not None and face_side_data['original_wid'].isin(wids).sum() == 0):
                continue

            if wids is not None:
                face_side_data = face_side_data[face_side_data['original_wid'].isin(wids)]

            if face_side_data['original_wid'].nunique() == 1:
                continue

            # Sort by the chosen follicle axis (x or y) based on the sorting order
            if sorting_order in ['bottom_to_top', 'top_to_bottom']:
                sorted_data = face_side_data.sort_values('follicle_y', ascending=sorting_order == 'top_to_bottom')
            elif sorting_order in ['right_to_left', 'left_to_right']:
                sorted_data = face_side_data.sort_values('follicle_x', ascending=sorting_order == 'left_to_right')

            # Map the sorted `original_wid` back to the DataFrame
            whisker_df.loc[sorted_data.index, 'original_wid'] = sorted_data['original_wid'].values

    print("Reordering complete for all frames.")
    return whisker_df

def parallel_order_whiskers(whisker_df, protraction_direction, wids=None, num_workers=4):
    # Split data by segment
    segment_ids = whisker_df['segment_id'].unique()
    segments = [
        (segment_id, whisker_df[whisker_df['segment_id'] == segment_id].copy(), protraction_direction, wids)
        for segment_id in segment_ids
    ]
    
    # Use multiprocessing to reorder whiskers in parallel
    with Pool(num_workers) as pool:
        reordered_segments = pool.map(order_whiskers_for_segment, segments)
    
    # Combine all reordered segments into a single DataFrame
    reordered_whisker_df = pd.concat(reordered_segments, ignore_index=True)
    return reordered_whisker_df

def sequential_order_whiskers(whisker_df, protraction_direction, wids=None):
    return order_whiskers_for_segment(('all', whisker_df, protraction_direction, wids))

def ensure_track_continuity(whisker_df, position_weight=0.8, max_position_deviation=20):
    """
    Ensure continuity of tracks by reassigning original_wid based on follicle position when breaks occur.

    Parameters:
    - whisker_df (pd.DataFrame): DataFrame containing whisker tracking data.
    - position_weight (float): Weight to prioritize position when resolving continuity.
    - max_position_deviation (float): Maximum allowable deviation in follicle position for continuity.

    Returns:
    - pd.DataFrame: Updated DataFrame with reassigned original_wid to ensure continuity.
    """
    # Sort data by frame ID and original_wid for consistent processing
    whisker_df = whisker_df.sort_values(by=['fid', 'original_wid']).reset_index(drop=True)

    # Find unique `original_wid` for processing
    unique_wids = whisker_df['original_wid'].unique()

    # Create a dictionary to map new original_wids
    wid_mapping = {}
    next_wid = whisker_df['original_wid'].max() + 1

    # Iterate over each unique track
    for wid in unique_wids:
        # Filter data for this track
        track_data = whisker_df[whisker_df['original_wid'] == wid]
        track_data = track_data.sort_values(by='fid').reset_index(drop=True)

        # Skip if there is no break in this track
        if (track_data['fid'].diff() > 1).sum() == 0:
            continue

        # Identify frames where the track breaks
        break_indices = track_data['fid'].diff() > 1
        break_points = track_data.loc[break_indices].index

        # Iterate over break points and assign new original_wids to segments
        for i, break_idx in enumerate(break_points):
            start_idx = break_idx if i == 0 else break_points[i - 1]
            end_idx = break_idx

            # Define the segment before and after the break
            previous_segment = track_data.iloc[start_idx:end_idx]
            
            ###### This code below is wrong: here we should be looking at other tracks and find the closest match based on follicle position ######
            print("Code needs to be fixed")
            
    #         next_segment = track_data.iloc[end_idx:]

    #         # If there is no next segment, skip
    #         if next_segment.empty:
    #             continue

    #         # Find the closest match based on follicle position
    #         previous_fol_x, previous_fol_y = (
    #             previous_segment['follicle_x'].iloc[-1],
    #             previous_segment['follicle_y'].iloc[-1],
    #         )
    #         next_fol_x, next_fol_y = (
    #             next_segment['follicle_x'].iloc[0],
    #             next_segment['follicle_y'].iloc[0],
    #         )

    #         position_deviation = (
    #             (previous_fol_x - next_fol_x) ** 2 + (previous_fol_y - next_fol_y) ** 2
    #         ) ** 0.5

    #         # If the position deviation is within the allowable threshold, assign continuity
    #         if position_deviation <= max_position_deviation:
    #             wid_mapping[next_segment.index[0]] = wid
    #         else:
    #             # Assign a new original_wid for the discontinuous segment
    #             wid_mapping[next_segment.index[0]] = next_wid
    #             next_wid += 1

    # # Apply the mapping to the DataFrame
    # whisker_df.loc[:, 'original_wid'] = whisker_df.index.map(
    #     lambda idx: wid_mapping.get(idx, whisker_df.loc[idx, 'original_wid'])
    # )

    return whisker_df

def filter_whiskers_in_whiskerpad(dff, margin=0.1):
    """
    Filters whiskers that do not have their follicle positions within the whiskerpad area,
    modeled as an oblong region using a convex hull for each face side.

    Parameters:
    - dff (pd.DataFrame): DataFrame containing whisker data after length threshold filtering.
                          Must include 'follicle_x', 'follicle_y', 'wid', and 'face_side' columns.
    - margin (float): Fractional margin added around the convex hull (default is 10%).

    Returns:
    - pd.DataFrame: Filtered DataFrame containing only whiskers within the whiskerpad area.
    """
    # Initialize filtered DataFrame
    filtered_dff = pd.DataFrame()

    # Determine the unique face sides
    face_sides = dff['face_side'].unique()
    print(f"Found face sides: {face_sides}")

    for side in face_sides:
        print(f"Processing face side: {side}")

        # Extract data for the current face side
        side_data = dff[dff['face_side'] == side]

        # Calculate "average extrema" positions for each whisker
        extrema_follicles = side_data.groupby('wid').agg(
            min_follicle_x=('follicle_x', 'min'),
            max_follicle_x=('follicle_x', 'max'),
            min_follicle_y=('follicle_y', 'min'),
            max_follicle_y=('follicle_y', 'max')
        ).reset_index()

        extrema_follicles['avg_extrema_follicle_x'] = (
            extrema_follicles['min_follicle_x'] + extrema_follicles['max_follicle_x']
        ) / 2
        extrema_follicles['avg_extrema_follicle_y'] = (
            extrema_follicles['min_follicle_y'] + extrema_follicles['max_follicle_y']
        ) / 2

        follicle_positions = extrema_follicles[['avg_extrema_follicle_x', 'avg_extrema_follicle_y']].values

        # Check if there are enough points to create a convex hull
        if len(follicle_positions) < 3:
            print(f"Not enough whiskers to form a convex hull on face side {side}. Skipping.")
            continue

        # Compute the convex hull for the whiskerpad
        hull = ConvexHull(follicle_positions)
        hull_vertices = follicle_positions[hull.vertices]

        # Expand the hull by adding a margin around it
        centroid = np.mean(hull_vertices, axis=0)
        expanded_hull_vertices = centroid + (hull_vertices - centroid) * (1 + margin)

        # Define a function to check if points are inside the expanded hull
        def is_inside_hull(point, hull_vertices):
            from matplotlib.path import Path
            hull_path = Path(hull_vertices)
            return hull_path.contains_point(point)

        # Filter whiskers whose "average extrema" positions fall within the expanded hull
        extrema_follicles['inside_hull'] = extrema_follicles[['avg_extrema_follicle_x', 'avg_extrema_follicle_y']].apply(
            lambda row: is_inside_hull(row.values, expanded_hull_vertices), axis=1
        )

        # Keep only whiskers inside the hull
        valid_wids = extrema_follicles[extrema_follicles['inside_hull']]['wid']
        filtered_side_data = side_data[side_data['wid'].isin(valid_wids)]

        # Append the filtered data for this face side
        filtered_dff = pd.concat([filtered_dff, filtered_side_data], ignore_index=True)

    print(f"Remaining unique whisker IDs after whiskerpad filtering: {filtered_dff['wid'].unique()}")
    return filtered_dff

def find_frequent_whiskers(tracks_df, wid_field='wid', min_frame_ratio=0.9):
    # Calculate the number of frames each whisker is found in
    frame_counts = tracks_df.groupby(wid_field).size().reset_index(name='frame_count')
    frame_ratio = frame_counts['frame_count'] / len(tracks_df['fid'].unique())
    # print(f"Frame Ratios:\n{frame_ratio}")
    
    # Filter out whiskers with low frame ratios
    valid_wids = frame_counts[frame_ratio >= min_frame_ratio][wid_field]
    # print(f"Valid Whiskers:\n{valid_wids}")
    
    # tracks_df[tracks_df[wid_field].isin(valid_wids)].sort_values(['fid', wid_field])
    
    return tracks_df[wid_field].isin(valid_wids)
   
def plot_whisker_traces(whisker_df, wids=None, wid_type='original_wid', feature={'angle', 'curvature'}, labels=None):
    """
    Plot the Angle and / or Curvature Traces for Multiple Whiskers
    """
    # Select whiskers to plot
    if wids is None:
        unique_wids = whisker_df[wid_type].unique()
        wids = unique_wids  # Select all whiskers
    
    w_times = []
    w_angles = []
    w_curvatures = []

    for wid in wids:
        # # Find original wid
        tracked_wid_data = whisker_df[whisker_df[wid_type] == wid].sort_values('fid')
        w_times.append(tracked_wid_data['fid'])
        w_angles.append(tracked_wid_data['angle'])
        w_curvatures.append(tracked_wid_data['curvature'])

    # Plot the angle traces
    # labels = [f'{wid} ("original")' if i % 2 == 0 else f'{wid} (detected)' for wid in wids for i in range(2)]
    
    if 'angle' in feature:
        # plot_whisker_angle(w_times, w_angles, [f'{wid} (original)' for wid in wids] + [f'{wid} (tracked)' for wid in wids])
        plot_whisker_angle(w_times, w_angles, [f'{wid} (tracked)' for wid in wids])
        # plot_whisker_angle(w_times, w_angles, [f'{wid} (original)' for wid in wids])
        # plot_whisker_angle(w_times, w_angles, [f'{wid} (tracked)' for wid in track_ids])

    if 'curvature' in feature:
        # plot_whisker_curvature(w_times, w_curvatures, labels, pairing=True)
        # plot_whisker_curvature(w_times, w_curvatures, [f'{wid} (original)' for wid in wids])
        plot_whisker_curvature(w_times, w_curvatures, [f'{wid} (tracked)' for wid in wids]) 
    
def plot_segment_boundaries(current_segment, next_segment, wid=None):
    # Concatenate the last 10 frames of the current segment and the first 10 frames of the next segment
    concat_df = pd.concat([current_segment[current_segment['fid'] > current_segment['fid'].max() - 10], next_segment[next_segment['fid'] < next_segment['fid'].min() + 10]])    
    # Plot the curvature traces for all whiskers
    plot_whisker_traces(concat_df, wid, feature={'curvature'})    
    # plot_whisker_traces(pd.concat([current_segment, next_segment]), [0, 28, 29, 30], feature={'curvature'})     
     
# %%
################ Main function #############
#################################################

def reclassify(file_path, protraction_direction, plot=False):
        
    # If file is not a parquet file, exit
    if not file_path.endswith('.parquet'):
        print(f"File {file_path} is not a Parquet file. Exiting reclassification.")
        return None
    
    # Load the Parquet file
    df = pd.read_parquet(file_path)

    # Get index of most frequent whiskers
    frequent_wid_idx = find_frequent_whiskers(df, 'wid', min_frame_ratio=0.8)
    # Create a list of mopst frequent unique whisker ids for plotting purposes
    most_frequent_wid_idx = find_frequent_whiskers(df, 'wid', min_frame_ratio=0.996)
    unique_frequent_wids = df[most_frequent_wid_idx]['wid'].unique()
    # Keep only up to four whiskers for plotting otherwise plotting will be too crowded
    unique_frequent_wids = unique_frequent_wids[:4]

    # Get the median length and score for each whisker, then average them
    median_length = df.groupby('wid')['length'].median().reset_index(name='median_length')
    overall_median_length = median_length['median_length'].mean()
    median_score = df.groupby('wid')['score'].median().reset_index(name='median_score')
    # overall_median_score = median_score['median_score'].mean()
    
    # Set length threshold to 1/4 of the longest whisker
    length_threshold = median_length['median_length'].max() / 4
    # length_threshold = 60 #overall_median_length #80
    # length_threshold = overall_median_length
    # score_threshold = overall_median_score #350 

    # Filter keeping only whiskers that meet the frequency or the length threshold 
    # frequent_wid_idx & 
    dff = df[(df['length'] > length_threshold)].sort_values(['fid', 'wid']).copy()
    unique_wids = dff['wid'].unique()
    print(f"Unique whisker ids: {unique_wids}")
    
    frame_num = 0
    longest_whiskers = po.get_longest_whiskers_data(dff, frame_num)
    # print some of longest_whiskers values 
    combined_df = pd.concat(longest_whiskers, ignore_index=True)
    print("Longest whiskers in the first frame after filtering:")
    print(combined_df[['fid', 'wid', 'length', 'angle', 'face_side']])
    
    print("After filtering:")
    print(dff[dff['fid'] == 0])

    # %%
    ############# First plot #############
    ######################################
    if plot:
        plot_whisker_traces(dff, wids=unique_frequent_wids, wid_type='wid', feature={'angle', 'curvature'})
        
    # %%
    ################ Track the whiskers #############
    #################################################
    # Initialize Variables
    max_missed_frames = 15
    max_cost = 50

    # Start time for tracking loop
    start_time = time.time()

    # Run the parallelized tracking loop
    tracks_df = process_data_in_segments(dff, segment_size=200, max_cost=max_cost, max_missed_frames=max_missed_frames, num_workers=40)

    # End time for tracking loop
    end_time = time.time()
    print(f"Total Parallel Tracking Loop Time: {end_time - start_time}")

    # %% 
    ############ Second plot #############
    ######################################
    if plot:
        plot_whisker_traces(tracks_df, wids=unique_frequent_wids, feature={'angle', 'curvature'})

    # %%
    # Order the whiskers
    if os.path.isfile(protraction_direction):
        # Use the whiskerpad json file to assess orientation
        with open(protraction_direction, 'r') as f:
            whiskerpad = json.load(f)

        # Extract data from the first whiskerpad and get ProtractionDirection
        protraction_direction = whiskerpad['whiskerpads'][0]['ProtractionDirection']
        
    # Get the most frequent whisker ids 
    frequent_wid_idx = find_frequent_whiskers(tracks_df, wid_field='original_wid', min_frame_ratio=0.9)
    unique_wids = tracks_df[frequent_wid_idx]['original_wid'].unique()
    print(f"Unique whisker ids after tracking: {unique_wids}")
    
    print("After tracking:")
    print(tracks_df[tracks_df['fid'] == 0])
        
    start_time = time.time()
    # whisker_df = order_whiskers(tracks_df.copy(), whiskerpad, unique_wids)
    # whisker_df = parallel_order_whiskers(tracks_df.copy(), protraction_direction, unique_wids, num_workers=40)
    # print("Using sequential reconciliation")
    # whisker_df = sequential_order_whiskers(tracks_df.copy(), protraction_direction, unique_wids) -> not needed. 
    
    whisker_df = order_whiskers_per_frame(tracks_df.copy(), protraction_direction=protraction_direction, wids=unique_wids)
    
    end_time = time.time()
    print(f"Total Reconciliation Time: {end_time - start_time}")
    
    frame_num = 0
    longest_whiskers = po.get_longest_whiskers_data(whisker_df, frame_num)
    # print some of longest_whiskers values 
    combined_df = pd.concat(longest_whiskers, ignore_index=True)
    print("Longest whiskers in the first frame after reclassification:")
    print(combined_df[['fid', 'wid', 'original_wid', 'length', 'angle', 'face_side']])
    print("After reclassification:")
    print(whisker_df[whisker_df['fid'] == 0])
        
    # %% 
    ############ Third plot #############
    ######################################
    if plot:
        plot_whisker_traces(whisker_df, wids=unique_frequent_wids, feature={'angle', 'curvature'})

    # %% 
    # Remove whiskers that are not found in most frames
    frequent_wid_idx = find_frequent_whiskers(whisker_df, wid_field='original_wid', min_frame_ratio=0.5)
    whisker_df = whisker_df[frequent_wid_idx]
    
    frame_num = 0
    longest_whiskers = po.get_longest_whiskers_data(whisker_df, frame_num)
    # print some of longest_whiskers values 
    combined_df = pd.concat(longest_whiskers, ignore_index=True)
    print("Longest whiskers in the first frame after removing infrequent whiskers:")
    print(combined_df[['fid', 'wid', 'original_wid', 'length', 'angle', 'face_side']])
    print("After removing infrequent whiskers:")
    print(whisker_df[whisker_df['fid'] == 0])
    
    # 
    
    # %% 
    # Final adjustments
    # Copy values from 'wid' to 'label'
    whisker_df.loc[:, 'label'] = whisker_df['wid']
    # Copy values from 'original_wid' to 'wid'
    whisker_df.loc[:, 'wid'] = whisker_df['original_wid']
    # Drop 'original_wid' column
    whisker_df = whisker_df.drop(columns=['original_wid'])
    
    frame_num = 0
    longest_whiskers = po.get_longest_whiskers_data(whisker_df, frame_num)
    # print some of longest_whiskers values 
    combined_df = pd.concat(longest_whiskers, ignore_index=True)
    print("Longest whiskers in the first frame after final adjustments:")
    print(combined_df[['fid', 'wid', 'length', 'angle', 'face_side']])
    print("After final adjustments:")
    print(whisker_df[whisker_df['fid'] == 0])

    # %% 
    ############ Fourth plot #############
    ######################################
    if plot:
        most_frequent_wid_idx = find_frequent_whiskers(whisker_df, 'wid', min_frame_ratio=0.9)
        unique_frequent_wids = whisker_df[most_frequent_wid_idx]['wid'].unique()
        unique_frequent_wids = unique_frequent_wids[:4]
        plot_whisker_traces(whisker_df, wids=unique_frequent_wids, wid_type='wid', feature={'angle'})

    # %%
    # Update the Dataset
    # Save the updated DataFrame to a Parquet file
    updated_parquet_file = file_path.replace('.parquet', '_updated.parquet')
    whisker_df.to_parquet(updated_parquet_file)

    # %%
    print("Reclassification complete. File saved to", updated_parquet_file)
    
    return updated_parquet_file
    
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Reclassify whisker tracking data")
    parser.add_argument("file_path", type=str, help="Path to the input Parquet file")
    parser.add_argument("protraction_direction", type=str, help="Path to the whiskerpad JSON file, or the protraction direction itself")
    parser.add_argument("--plot", action="store_true", help="Plot the whisker traces before and after reclassification")
    args = parser.parse_args()
    
    # Run the main function
    reclassify(args.file_path, args.protraction_direction, args.plot)