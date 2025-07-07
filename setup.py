
from setuptools import setup

setup(
   name='WhiskiWrap',
   version='1.2.2',
   author='Chris Rodgers',
   author_email='',
   maintainer=', '.join(['cxrodgers','aiporre','vncntprvst']),
   packages=['WhiskiWrap','wwutils'],
   # scripts=['bin/script1','bin/script2'],
   url='http://pypi.python.org/pypi/WhiskiWrap/',
   license='LICENSE.txt',
   description='Whisk package wrapper created by cxrodgers',
   long_description=open('README.md').read(),
   long_description_content_type='text/markdown',
   install_requires=[
      'tables>=3.5.1',
      'pandas',
      'MediaInfo',
      'future',
      'tifffile',
      'imagecodecs',
      'statsmodels',
      'ffmpeg-python>=0.2.0',
      'whisk-janelia[ffmpeg]',
      'zarr',
      'pyarrow',
      'matplotlib',
      'easygui',
      'opencv-python',  # Required for whiskerpad module
      'h5py',  # Required for combine_sides module
      'seaborn',  # Optional, for plotting
      ],
   python_requires='>=3.10',
   include_package_data=True,
)