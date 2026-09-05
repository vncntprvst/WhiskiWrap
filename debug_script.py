"""Debug script for combine_to_file function"""
from wwutils.data_manip.combine_sides import combine_to_file

def main():
    """Call combine_to_file with the specified parameters"""
    result = combine_to_file(
        wt_files=['test_videos/excerpt_video_left.parquet', 'test_videos/excerpt_video_right.parquet'],
        whiskerpad_file='test_videos/whiskerpad_excerpt_video.json',
        output_file='test_videos/excerpt_video.parquet',
        keep_wt_files=True,
        filter_short=True,
        resort_frequency=True
    )
    print(f"Combined file saved to: {result}")

if __name__ == "__main__":
    main()