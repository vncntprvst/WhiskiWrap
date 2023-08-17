import os
import sys
from ctypes import CDLL

def get_current_conda_env_name():
    """
    Returns the name of the current conda environment.
    """
    conda_prefix = os.getenv('CONDA_PREFIX')
    if conda_prefix:
        # The last directory in the CONDA_PREFIX path is the name of the current environment
        return os.path.basename(conda_prefix)
    return None

def load_ffmpeg_dlls_from_conda_env():
    # Get the Conda prefix for the environment
    conda_prefix = os.getenv('CONDA_PREFIX')
    
    # If CONDA_PREFIX is not set, try to derive it from sys.prefix
    if not conda_prefix:
        conda_prefix = sys.prefix
        
    # If still not found, return an error
    if not conda_prefix:
        raise EnvironmentError("Conda environment not found")
        
    # Construct the path to the Library\bin directory inside the Conda environment
    library_bin_path = os.path.join(conda_prefix, 'Library', 'bin')
    
    # List of FFmpeg DLLs to be loaded
    ffmpeg_dll_names = [
        "avcodec-60.dll",
        "avdevice-60.dll",
        "avformat-60.dll",
        "avutil-58.dll",
        "swscale-7.dll"
    ]
    
    for dll_name in ffmpeg_dll_names:
        dll_path = os.path.join(library_bin_path, dll_name)
        if os.path.exists(dll_path):
            CDLL(dll_path)
            print(f"Loaded {dll_path} successfully!")
        else:
            raise FileNotFoundError(f"Could not find {dll_name} at {dll_path}")

# Example usage
current_env = get_current_conda_env_name()
if current_env:
    print(f"Current Conda environment: {current_env}")
    load_ffmpeg_dlls_from_conda_env()
else:
    print("Not running in a Conda environment")
