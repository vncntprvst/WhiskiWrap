import WhiskiWrap as ww
import sys, os, json
import numpy as np
import pandas as pd
# import time
# from sklearn.cluster import KMeans

def reassess_whisker_ids(h5_filename):
    summary = ww.read_whiskers_hdf5_summary(h5_filename)
    # print(summary.head())    

    # Load whiskerpad json file
    base_filename = os.path.basename(h5_filename.replace('_right', '').replace('_left', '')).split(".")[0]
    whiskerpad_file = os.path.join(os.path.dirname(h5_filename), f'whiskerpad_{base_filename}.json')
    with open(whiskerpad_file, 'r') as f:
        whiskerpad_params = json.load(f)

    # get the whiskerpad params for the correct side         
    if '_right' in str(h5_filename):
        whiskerpad = next((whiskerpad for whiskerpad in whiskerpad_params['whiskerpads'] if whiskerpad['FaceSide'].lower() == 'right'), None)
    else:
        whiskerpad = next((whiskerpad for whiskerpad in whiskerpad_params['whiskerpads'] if whiskerpad['FaceSide'].lower() == 'left'), None)

    # get face axis and orientation 
    face_axis = whiskerpad['FaceAxis']
    face_orientation = whiskerpad['FaceOrientation']

    # Determine a threshold for whisker length
    length_threshold = determine_length_threshold(summary)
    # score_threshold = determine_score_threshold(summary)
    filtered_summary = filter_whiskers(summary, length_threshold)

    grouped_by_fid = group_summary_by_fid(filtered_summary)
    for fid in range(1, max(grouped_by_fid.keys())):
        current_frame_whiskers = grouped_by_fid[fid]
        # next_frame_whiskers = grouped_by_fid[fid + 1]
        previous_frame_whiskers = grouped_by_fid[fid - 1]
        
        # Change wids only if number and order varies between frames 
        if is_reassignment_needed(current_frame_whiskers, previous_frame_whiskers, face_axis, face_orientation):
            for whisker_index, whisker in current_frame_whiskers.iterrows():
                closest_whisker, distance =  find_closest_whisker(whisker, previous_frame_whiskers)

                #  Reassign only if wid is different and within limit
                if is_within_threshold(distance) and whisker['wid'] != closest_whisker['wid']:
                    reassign_wid(grouped_by_fid, fid, whisker_index, closest_whisker['wid'])
        else:
            continue

    return update_summary_with_new_ids(grouped_by_fid), filtered_summary

def is_reassignment_needed(current_frame_whiskers, previous_frame_whiskers, face_axis, face_orientation):
    # Check if the number of whiskers is the same in both frames
    # if len(current_frame_whiskers) != len(previous_frame_whiskers):
    #     if len(current_frame_whiskers) > len(previous_frame_whiskers):
    #         return True

    # Order check based on face axis and orientation
    if face_axis.lower() == 'vertical' and face_orientation.lower() == 'down':
        if not is_ascending_order(current_frame_whiskers, 'follicle_y'):
            return True
    elif face_axis.lower() == 'vertical' and face_orientation.lower() == 'up':
        if not is_descending_order(current_frame_whiskers, 'follicle_y'):
            return True
    elif face_axis.lower() == 'horizontal' and face_orientation.lower() == 'left':
        if not is_ascending_order(current_frame_whiskers, 'follicle_x'):
            return True
    elif face_axis.lower() == 'horizontal' and face_orientation.lower() == 'right':
        if not is_descending_order(current_frame_whiskers, 'follicle_x'):
            return True
    return False

def is_ascending_order(frame_whiskers, column):
    # Check if the values in the specified column are in ascending order
    y_values = frame_whiskers[column].tolist()
    return y_values == sorted(y_values)

def is_descending_order(frame_whiskers, column):
    # Check if the values in the specified column are in descending order
    y_values = frame_whiskers[column].tolist()
    return y_values == sorted(y_values, reverse=True)

def is_within_threshold(distance, threshold=10.0):
    return distance <= threshold

def reassign_wid(grouped_by_fid, current_fid, whisker_index, new_wid):
    current_frame = grouped_by_fid[current_fid]
    
    # Find the index of the whisker in the current frame that has the new_wid
    target_index = current_frame[current_frame['wid'] == new_wid].index

    # If a whisker with new_wid is found, swap the wids
    if not target_index.empty:
        # Get the current wid of the whisker at whisker_index
        current_wid = current_frame.loc[whisker_index, 'wid']
        # Assign it 
        grouped_by_fid[current_fid].loc[target_index[0], 'wid'] = current_wid

    # Then assign the new_wid to the whisker_index
    grouped_by_fid[current_fid].loc[whisker_index, 'wid'] = new_wid

def update_summary_with_new_ids(grouped_by_fid):
    updated_summary = pd.DataFrame()
    for fid, group in grouped_by_fid.items():
        updated_summary = pd.concat([updated_summary, group], ignore_index=True)
    return updated_summary

def find_closest_whisker(whisker, previous_frame_whiskers):
    # Method 1
    # time it
    # time1 = time.time()
    min_distance = float('inf')
    closest_whisker = None

    for _, next_whisker in previous_frame_whiskers.iterrows():
        distance = euclidean_distance(whisker, next_whisker)
        if distance < min_distance:
            min_distance = distance
            closest_whisker = next_whisker

    # print(f"Method 1 took {time.time() - time1} seconds")

    # time2 = time.time()
    # #  Method 2
    # closest_whisker, min_distance = closest_whisker_distance(whisker, previous_frame_whiskers)     
    # print(f"Method 2 took {time.time() - time2} seconds")

    return closest_whisker, min_distance

def closest_whisker_distance(whisker, previous_frame_whiskers):
    # Convert whisker to a NumPy array
    whisker_array = np.array([whisker['follicle_x'], whisker['follicle_y'], whisker['angle']])

    # Convert previous_frame_whiskers to a NumPy array
    previous_frame_whiskers_array = previous_frame_whiskers[['follicle_x', 'follicle_y', 'angle']].to_numpy()

    # Calculate the Euclidean distance to all whiskers in the previous frame
    distances = np.sqrt(np.sum((previous_frame_whiskers_array - whisker_array)**2, axis=1))

    # Find the index of the closest whisker
    min_index = np.argmin(distances)

    # Get the closest whisker and minimum distance
    closest_whisker = previous_frame_whiskers.iloc[min_index]
    min_distance = distances[min_index]

    return closest_whisker, min_distance

def euclidean_distance(whisker1, whisker2, method='follicle_angle'):
    if method == 'follicle_tip':
    # Using follicle and tip coordinates
        distance = np.sqrt((whisker1['follicle_x'] - whisker2['follicle_x'])**2 + 
                        (whisker1['follicle_y'] - whisker2['follicle_y'])**2 +
                        (whisker1['tip_x'] - whisker2['tip_x'])**2 +
                        (whisker1['tip_y'] - whisker2['tip_y'])**2)
    elif method == 'follicle_angle':
    # Using follicle coordinates and angle
        distance = np.sqrt((whisker1['follicle_x'] - whisker2['follicle_x'])**2 + 
                        (whisker1['follicle_y'] - whisker2['follicle_y'])**2 +
                        (angle_difference(whisker1['angle'], whisker2['angle']))**2)
    return distance

def angle_difference(angle1, angle2):
    # Calculate the minimum difference between two angles
    return min(abs(angle1 - angle2), 360 - abs(angle1 - angle2))

def extract_features(summary):
    # Extract and possibly create new features
    features = summary[['tip_x', 'tip_y', 'follicle_x', 'follicle_y']]
    return features

def cluster_whiskers(features):
    # Normalize features if necessary
    # features_normalized = normalize(features)

    # Apply clustering algorithm
    kmeans = KMeans(n_clusters=num_whiskers)  # num_whiskers is an estimated number of unique whiskers
    clusters = kmeans.fit_predict(features)

    return clusters

def reassess_whisker_ids_with_clustering(summary):
    features = extract_features(summary)
    clusters = cluster_whiskers(features)

    # Map clusters to new whisker IDs
    summary['new_wid'] = clusters

    return summary

def filter_whiskers(summary, length_threshold, score_threshold=None):
    if score_threshold is not None:
        return summary[(summary['pixel_length'] > length_threshold) & (summary['score'] > score_threshold)]
    else:
        return summary[summary['pixel_length'] > length_threshold]

def determine_length_threshold(summary):
    #  Cutoff at 75th percentile, since most whiskers are actually hair
    length_threshold = np.percentile(summary['pixel_length'], 75)

    # mean_length = np.mean(summary['pixel_length'])
    # std_dev_length = np.std(summary['pixel_length'])
    # length_threshold = mean_length - std_dev_length

    return length_threshold

def determine_score_threshold(summary):

    mean_score = np.mean(summary['score'])
    std_dev_score = np.std(summary['score'])
    score_threshold = mean_score + std_dev_score

    return score_threshold

def group_summary_by_fid(summary):
    """
    Groups whisker segments by frame ID (fid).

    Args:
        summary (DataFrame): The DataFrame containing whisker segment data.

    Returns:
        dict: A dictionary where each key is a frame ID and each value is a DataFrame of whisker segments for that frame.
    """
    # Group the DataFrame by 'fid' and create a dictionary
    grouped = summary.groupby('fid')
    grouped_by_fid = {fid: group for fid, group in grouped}
    
    return grouped_by_fid

def plot_summary_distributions(summary, h5_filename):
    #  plot the distrbution of pixel_length, and score. 
    #  Save the plot to the same directory as the input file
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].hist(summary['pixel_length'], bins=50)
    ax[0].set_title('Pixel Length Distribution')
    ax[0].set_xlabel('Pixel Length')
    ax[0].set_ylabel('Frequency')
    ax[1].hist(summary['score'], bins=50)
    ax[1].set_title('Score Distribution')
    ax[1].set_xlabel('Score')
    ax[1].set_ylabel('Frequency')
    plt.savefig(h5_filename.replace('.hdf5', '_summary.png'))
    plt.close()

def plot_angle_traces(loaded_data, output_filename):
    # plot wid 0, 1 and 2 angle traces
    import matplotlib.pyplot as plt
    # angle = loaded_data[loaded_data['wid'] == 0]['angle']
    # plt.plot(angle)
    # plt.title('Angle trace for wid 0')
    for wid in range(3):
        angle = loaded_data[loaded_data['wid'] == wid]['angle']
        plt.plot(angle)
    plt.title('Angle traces for wids 0, 1, and 2')
    plt.xlabel('Frame')
    plt.ylabel('Angle')
    plt.savefig(output_filename.replace('.hdf5', '_angle_traces.png'))
    plt.close()

def update_wids(h5_filename):
    # Call the reassess function
    updated_summary, filtered_summary = reassess_whisker_ids(h5_filename)

    # Save the updated summary to a new hdf5 file
    output_filename = h5_filename.replace('.hdf5', '_updated_wids.hdf5')

    # Create an HDF5 file and store the DataFrame
    with pd.HDFStore(output_filename, 'w') as store:
        store.put('updated_summary', updated_summary, format='table', data_columns=True)

    print(f"Updated summary saved to {output_filename}")

    #  Load the updated summary and plot the distributions
    # with pd.HDFStore(output_filename, 'r') as store:
    #     loaded_data = store.get('updated_summary')
    plot_angle_traces(filtered_summary, h5_filename)
    plot_angle_traces(updated_summary, output_filename)

if __name__ == "__main__":
    #  get argument from command line
    h5_filename = sys.argv[1]
    update_wids(h5_filename)
