import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_whisker_ids_over_time(file_path, output_prefix, max_frames=100):
    """
    Plot whisker IDs over time to visualize temporal consistency
    
    Parameters:
    -----------
    file_path : str
        Path to the parquet file with whisker data
    output_prefix : str
        Prefix for the output file name
    max_frames : int
        Maximum number of frames to plot
    """
    # Load data
    df = pd.read_parquet(file_path)
    
    # Get unique frame IDs and whisker IDs
    frames = sorted(df['fid'].unique())[:max_frames]
    whisker_ids = sorted(df['wid'].unique())
    
    # Create a matrix to represent whisker presence in each frame
    # Each row is a whisker ID, each column is a frame
    presence_matrix = np.zeros((len(whisker_ids), len(frames)))
    
    # Fill in the matrix
    for i, frame in enumerate(frames):
        frame_data = df[df['fid'] == frame]
        for _, row in frame_data.iterrows():
            wid = row['wid']
            wid_idx = whisker_ids.index(wid)
            presence_matrix[wid_idx, i] = 1
    
    # Plot
    plt.figure(figsize=(15, 8))
    plt.imshow(presence_matrix, aspect='auto', cmap='viridis')
    plt.colorbar(label='Presence (1=present, 0=absent)')
    plt.xlabel('Frame Index')
    plt.ylabel('Whisker ID')
    plt.title(f'Whisker ID Presence Over Time - {os.path.basename(file_path)}')
    plt.yticks(range(len(whisker_ids)), whisker_ids)
    plt.xticks(range(0, len(frames), 10), [frames[i] for i in range(0, len(frames), 10)])
    plt.grid(False)
    
    # Save the figure
    output_file = f"{output_prefix}_whisker_presence.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")
    plt.close()
    
    # Plot number of whiskers per frame
    whiskers_per_frame = df.groupby('fid')['wid'].nunique()
    plt.figure(figsize=(15, 5))
    plt.plot(whiskers_per_frame.index[:max_frames], whiskers_per_frame.values[:max_frames], marker='o')
    plt.xlabel('Frame ID')
    plt.ylabel('Number of Whiskers')
    plt.title(f'Number of Whiskers per Frame - {os.path.basename(file_path)}')
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    output_file = f"{output_prefix}_whiskers_per_frame.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize whisker IDs over time")
    parser.add_argument("file_path", help="Path to parquet file with whisker data")
    parser.add_argument("--output_prefix", default="whisker_vis", help="Output file prefix")
    parser.add_argument("--max_frames", type=int, default=100, help="Maximum number of frames to plot")
    
    args = parser.parse_args()
    plot_whisker_ids_over_time(args.file_path, args.output_prefix, args.max_frames)
