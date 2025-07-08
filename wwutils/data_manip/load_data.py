"""
Data loading utilities for whisk / WhiskiWrap.

This module provides functions to load various datasets:

- get_summary(filename, filter=True): Load and filter whisker summary data
- load_bwid(params): Load big waveform info DataFrame
- load_session_metadata(params): Load session, task, and mouse metadata
- load_big_tm(params, ...): Load and filter big trial matrix
- load_data_from_patterns(params, filename, ...): Load pattern data files
- load_data_from_logreg(params, filename, ...): Load logreg dataset files


Usage:
    from wwutils.data_manip.load_data import load_bwid, load_big_tm, get_summary

Parameters:
    params (dict): Dictionary of directory paths and settings for file locations.
"""
import os, re, json
import wwutils
"""Delay imports to avoid circular dependencies between wwutils and WhiskiWrap"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tables

from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde


def load_bwid(params, drop_1_and_6b=True):
    """Load big_waveform_info_df
    
    Loads from params['unit_db_dir']
    Adds stratum
    Drops 1 and 6b
    Load recording_locations_table from params['unit_db_dir']
    Add location_is_strict
    Joins recording_location, crow_recording_location, location_is_strict
    on big_waveform_info_df
    
    Returns: DataFrame
        big_waveform_info_df
    """
    
    ## Load waveform info stuff
    big_waveform_info_df = pd.read_pickle(
        os.path.join(params['unit_db_dir'], 'big_waveform_info_df'))
    big_waveform_info_df['stratum'] = 'deep'
    big_waveform_info_df.loc[
        big_waveform_info_df.layer.isin(['2/3', '4']), 'stratum'
        ] = 'superficial'

    # Drop 1 and 6b
    if drop_1_and_6b:
        big_waveform_info_df = big_waveform_info_df.loc[
            ~big_waveform_info_df['layer'].isin(['1', '6b'])
            ].copy()
    
    # Remove those levels
    big_waveform_info_df.index = (
        big_waveform_info_df.index.remove_unused_levels())


    ## Join recording location
    # Load and rename
    recording_locations_table = pd.read_csv(
        os.path.join(params['unit_db_dir'], 
        '20191007 electrode locations - Sheet1.csv')).rename(columns={
        'Session': 'session', 'Closest column': 'recording_location', 
        'Closest C-row column': 'crow_recording_location', 
        }).set_index('session').sort_index()

    # fillna the really off-target ones
    recording_locations_table['crow_recording_location'] = (
        recording_locations_table['crow_recording_location'].fillna('off'))

    # Add a "strict" column where the recording was bona fide C-row
    recording_locations_table['location_is_strict'] = (
        recording_locations_table['recording_location'] ==
        recording_locations_table['crow_recording_location'])

    # Join onto bwid
    big_waveform_info_df = big_waveform_info_df.join(recording_locations_table[
        ['recording_location', 'crow_recording_location', 'location_is_strict']
        ], on='session')
    
    # Error check
    assert not big_waveform_info_df.isnull().any().any()
    
    return big_waveform_info_df
    
def load_session_metadata(params):
    """Load metadata about sessions, tasks, and mice.
    
    Returns: tuple
        session_df, task2mouse, mouse2task
    """
    session_df = pd.read_pickle(
        os.path.join(params['pipeline_dir'], 'session_df'))
    task2mouse = session_df.groupby('task')['mouse'].unique()
    mouse2task = session_df[
        ['task', 'mouse']].drop_duplicates().set_index('mouse')['task']
    
    return session_df, task2mouse, mouse2task

def load_big_tm(params, dataset='no_opto', mouse2task=None):
    """Load big_tm, the big trial matrix, and optionally filters.
    
    params : parameters from json file
    
    dataset : string or None
        If string, loads corresponding dataset, and includes only those
        trials in the result.
        If None, returns original big_tm.
    
    mouse2task : Series, or None
        If Series (from load_session_metadat), then adds mouse and task
        levels to big_tm index.
        If None, does nothing.
    
    Returns: DataFrame
        big_tm    
    """
    # Load original big_tm with all trials
    big_tm = pd.read_pickle(
        os.path.join(params['patterns_dir'], 'big_tm'))

    # Slice out the trials of this dataset (no_opto) from big_tm
    if dataset is not None:
        included_trials = pd.read_pickle(
            os.path.join(params['logreg_dir'], 'datasets', dataset, 'labels')
            ).index
        
        # Apply mask
        big_tm = big_tm.loc[included_trials]
        big_tm.index = big_tm.index.remove_unused_levels()

    # Insert mouse and task levels
    if mouse2task is not None:
        big_tm = wwutils.misc.insert_mouse_and_task_levels(
            big_tm, mouse2task)
    
    return big_tm


def load_data_from_patterns(params, filename, dataset='no_opto', 
    mouse2task=None):
    """Common loader function from patterns dir
    
    filename : string
        These are the valid options:
            big_tm
            big_C2_tip_whisk_cycles
            big_cycle_features
            big_touching_df
            big_tip_pos
            big_grasp_df

        These are unsupported, because they aren't indexed the same:
            big_ccs_df
            kappa_parameterized
            peri_contact_kappa
    
    params : parameters from json file

    dataset : string or None
        If string, loads corresponding dataset, and includes only those
        trials in the result.
        If None, returns original big_tm.
    
    mouse2task : Series, or None
        If Series (from load_session_metadat), then adds mouse and task
        levels to big_tm index.
        If None, does nothing.
    
    Returns: DataFrame
        The requested data.
    """
    # Load from patterns directory
    full_filename = os.path.join(params['patterns_dir'], filename)
    
    # Special case loading
    if filename == 'big_tip_pos':
        res = pd.read_hdf(full_filename)
    else:
        res = pd.read_pickle(full_filename)
    
    # Slice out the trials of this dataset (no_opto)
    if dataset is not None:
        # Load trials
        included_trials = pd.read_pickle(
            os.path.join(params['logreg_dir'], 'datasets', dataset, 'labels')
            ).index
        
        # Apply mask
        res = wwutils.misc.slice_df_by_some_levels(res, included_trials)
        res.index = res.index.remove_unused_levels()

    # Insert mouse and task levels
    if mouse2task is not None:
        res = wwutils.misc.insert_mouse_and_task_levels(res, mouse2task)
    
    return res
    

def load_data_from_logreg(params, filename, dataset='no_opto', mouse2task=None):
    """Load data from logreg directory
    
    filename : string
        These are the valid options:
            unobliviated_unaggregated_features
            unobliviated_unaggregated_features_with_bin
            obliviated_aggregated_features
            obliviated_unaggregated_features_with_bin
        
        These are unsupported:
            BINS
        
    params : parameters from json file

    dataset : string or None
        If string, loads corresponding dataset, and includes only those
        trials in the result.
        If None, returns without filtering.
        
        If filename == 'obliviated_aggregated_features' and dataset is not None,
        then the pre-sliced version is loaded from the dataset directory.
    
    mouse2task : Series or None
        If Series (from load_session_metadat), then adds mouse and task
        levels to index.
        If None, does nothing.
    
    Returns: DataFrame
        The requested data.    
    """
    # Load, depending on filename
    if filename == 'oblivated_aggregated_features' and dataset is not None:
        # Special case: this was already sliced and dumped in the dataset dir
        full_filename = os.path.join(
            params['logreg_dir'], 'datasets', dataset, 'features')
        res = pd.read_pickle(full_filename)
    
    else:
        # Load
        full_filename = os.path.join(params['logreg_dir'], filename)
        res = pd.read_pickle(full_filename)

        # Slice out the trials of this dataset (no_opto)
        if dataset is not None:
            # Load trials
            included_trials = pd.read_pickle(os.path.join(
                params['logreg_dir'], 'datasets', dataset, 'labels')
                ).index
            
            # Apply mask
            res = wwutils.misc.slice_df_by_some_levels(res, included_trials)
            res.index = res.index.remove_unused_levels()

    # Insert mouse and task levels
    if mouse2task is not None:
        res = wwutils.misc.insert_mouse_and_task_levels(res, mouse2task)
        return res
    

def get_summary(filename, filter = True):
    # Check if it's a list of whiskers/measurements files
    from wwutils.classifiers import reclassify as rc
    import WhiskiWrap as ww
    if isinstance(filename, list):
        if filename[0].endswith(('.whiskers', '.measurements')):
        # if file is a list of whiskers/measurements files, load all of them
            summary = ww.read_whiskers_measurements(filename)
            # update nomenclature
            if 'whisker_id' in summary:
                # replace whisker_id with wid, and frame_id with fid
                summary.rename(columns={'whisker_id': 'wid', 'frame_id': 'fid'}, inplace=True)
            filename = filename[0]
            base_filename = os.path.basename(filename.replace('_right', '').replace('_left', '')).split(".")[0]
            # also remove the trailing numbers at the end of the filename
            base_filename = re.sub(r'_\d+$', '', base_filename)
            # whiskerpad_file is in the directory above the input file
            whiskerpad_file = os.path.join(os.path.dirname(os.path.dirname(filename)), f'whiskerpad_{base_filename}.json')     

    elif isinstance(filename, str) and filename.endswith('.hdf5'):
    # if file is a hdf5 file, load it
        summary = ww.read_whiskers_hdf5_summary(filename)
        # print(summary.head())   

        base_filename = os.path.basename(filename.replace('_right', '').replace('_left', '')).split(".")[0]
        whiskerpad_file = os.path.join(os.path.dirname(filename), f'whiskerpad_{base_filename}.json')

    # Load whiskerpad json file
    with open(whiskerpad_file, 'r') as f:
        whiskerpad_params = json.load(f)

    # get the whiskerpad params for the correct side         
    if '_right' in str(filename):
        whiskerpad = next((whiskerpad for whiskerpad in whiskerpad_params['whiskerpads'] if whiskerpad['FaceSide'].lower() == 'right'), None)
    else:
        whiskerpad = next((whiskerpad for whiskerpad in whiskerpad_params['whiskerpads'] if whiskerpad['FaceSide'].lower() == 'left'), None)

    # get face axis and orientation 
    face_axis = whiskerpad['FaceAxis']
    face_orientation = whiskerpad['FaceOrientation']

    # Determine a threshold for whisker length
    length_threshold = rc.determine_length_threshold(summary)
    # score_threshold = determine_score_threshold(summary)

    if filter:
        summary = rc.filter_whiskers(summary, length_threshold)

    return summary

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

def is_ascending_order(frame_whiskers, column):
    # Check if the values in the specified column are in ascending order
    y_values = frame_whiskers[column].tolist()
    return y_values == sorted(y_values)

def is_descending_order(frame_whiskers, column):
    # Check if the values in the specified column are in descending order
    y_values = frame_whiskers[column].tolist()
    return y_values == sorted(y_values, reverse=True)

