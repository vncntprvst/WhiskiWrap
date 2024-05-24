""" wfile_io.py

ctypes interface to libwhisk.so
The functionality described here corresponds with that in trace.h.

File I/O
  Load_Whiskers
  Save_Whiskers

Author: Nathan Clack <clackn@janelia.hhmi.org>
Date  : 25 January 2009

Copyright 2010 Howard Hughes Medical Institute.
All rights reserved.
Use is subject to Janelia Farm Research Campus Software Copyright 1.1
license terms (http://license.janelia.org/license/jfrc_copyright_1_1.html).

Copied to WhiskiWrap repository by Vincent Prevosto, 05/2023
"""
import sys,os
from ctypes import *
from ctypes.util import find_library
from numpy import zeros, float32, uint8, array, hypot, arctan2, pi, concatenate, float64, ndarray, int32
from numpy import where, cos, sin, sum
from warnings import warn

import whisk
# import pdb

def get_all_subdirectories(path):
    return [d for d, _, _ in os.walk(path)]
    # return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    
def ensure_ffmpeg_dlls_are_present(whisk_bin_dir):
    ffmpeg_dll_dir = os.path.join(whisk_bin_dir, 'ffmpeg_win64_lgpl_shared')
    
    if not os.path.exists(ffmpeg_dll_dir):
        import requests, zipfile
        print("First-time setup: downloading necessary ffmpeg DLLs. This might take a few minutes...")
        try:
            from whisk import whisk_utils
            whisk_utils.download_and_extract_ffmpeg_dlls()
            
        except requests.exceptions.RequestException as e:
            # Handle exceptions caused by requests library during the download
            print(f"Error downloading ffmpeg DLLs: {str(e)}")
            print("Please check your internet connection and try again.")
        
        except zipfile.BadZipFile:
            # Handle issues with extraction due to a corrupted download
            print("Error extracting ffmpeg DLLs. The downloaded file might be corrupted.")
            print("Please try running the script again.")
        
        except PermissionError:
            # Handle potential write permission errors gracefully
            print("Permission error while trying to save ffmpeg DLLs. Please ensure you have write permissions to the target directory and try again.")
        
        except Exception as e:
            # Catch all other unexpected exceptions
            print(f"An unexpected error occurred: {str(e)}")
            print("Please report this issue to the software maintainers.")

def load_ffmpeg_dlls(whisk_bin_dir):
    """
    Load ffmpeg DLLs required for cWhisk.
    
    Args:
    - whisk_bin_dir: the imported whisk module's bin directory
    """
    ensure_ffmpeg_dlls_are_present(whisk_bin_dir)

    ffmpeg_dll_dir = os.path.join(whisk_bin_dir, 'ffmpeg_win64_lgpl_shared')

    ffmpeg_dll_names = [
        "avcodec-60.dll",
        "avdevice-60.dll",
        "avformat-60.dll",
        "avutil-58.dll",
        "swscale-7.dll",
    ]

    ffmpeg_dlls = [os.path.join(ffmpeg_dll_dir, dll_name) for dll_name in ffmpeg_dll_names]

    # Load each ffmpeg DLL
    for dll in ffmpeg_dlls:
        CDLL(dll)
        # print(f"Loaded {dll} successfully!")

# Find the base directory of the whisk package
whisk_base_dir = os.path.dirname(whisk.__file__)
whisk_bin_dir = os.path.join(whisk_base_dir, 'bin')
# Set WHISKPATH environment variable to whisk/bin
os.environ['WHISKPATH'] = whisk_bin_dir
# Get all subdirectories of whisk/bin
all_directories = get_all_subdirectories(whisk_bin_dir)

if sys.platform == 'win32':
    lib = os.path.join(whisk_bin_dir, 'whisk.dll')
else:
    lib = os.path.join(whisk_bin_dir, 'libwhisk.so')
    
# Append both whisk base directory and bin directory to PATH
os.environ['PATH'] += os.pathsep + os.pathsep.join(['.', '..', whisk_base_dir, whisk_bin_dir]) 
# Append all bin sub_directories to PATH
os.environ['PATH'] = os.environ['PATH'] + ';' + ';'.join(all_directories)

name = lib
if not name:
  name = find_library('whisk')

try:
    cWhisk = CDLL(name)
except OSError as e:
  print(f"Error loading {name}: {str(e)}")
  # If loading cWhisk failed, attempt to load ffmpeg DLLs and then try again
  load_ffmpeg_dlls(whisk_bin_dir)
  cWhisk = CDLL(name)

_param_file = "default.parameters"
if cWhisk.Load_Params_File(_param_file)==1: #returns 0 on success, 1 on failure
  cWhisk.Print_Params_File(_param_file)
  if cWhisk.Load_Params_File(_param_file)==1: #returns 0 on success, 1 on failure
    raise Exception("Could not load tracing parameters from file: %s"%_param_file)

#
# DATA STRUCTURE TRANSLATIONS
#

class cContour(Structure):                       #typedef struct
  _fields_ = [( "length"   , c_int            ), #  { int  length;
              ( "boundary" , c_int            ), #    int  boundary;
              ( "width"    , c_int            ), #    int  width;
              ( "iscon4"   , c_int            ), #    int  iscon4;
              ( "tour"     , POINTER( c_int ) )] #    int *tour;
                                                 #  } Contour;

  def asarray(self):
    a = zeros( (self.length,2) )
    for i in range(self.length):
      a[i,0] = self.tour[i]%self.width
      a[i,1] = self.tour[i]/self.width
    return a

  def plot(self,*args,**kwargs):
    from pylab import plot
    a = self.asarray()
    plot(a[:,0],a[:,1],*args,**kwargs)

  def draw(self, surface, color, scale, drawfunc):
    a = self.asarray()
    drawfunc( surface, color, 1, a*scale )

class cObject_Map(Structure):
  _fields_ = [( "num_objects",  c_int ),
              ( "objects",      POINTER(POINTER( cContour ))) ]

  def plot(self,*args,**kwargs):
    for i in range( self.num_objects ):
      self.objects[i].contents.plot(*args,**kwargs)

  def plot_with_seeds( self, image, *args, **kwargs ):
    from pylab import imshow, cm, axis, subplots_adjust, show
    imshow( image, cmap = cm.gray, hold = 0, interpolation = 'nearest' )
    self.plot( *args, **kwargs )
    for i in range(self.num_objects):
      sds = find_seeds( self.objects[i], image )
      if sds:
        sds.plot( linewidths = (1,),
                  facecolors = ('w',),
                  edgecolors = ('k',) )
    axis('image')
    axis('off')
    subplots_adjust(0,0,1,1,0,0)
    show()
    return gcf()

  def draw(self, surface, color, scale, drawfunc ):
    for i in range( self.num_objects ):
      self.objects[i].contents.draw(surface,color,scale,drawfunc)

class cWhisker_Seg_Old(Structure):                 #typedef struct
  _fields_ = [( "id"       , c_int   ),            #  { int    id;
              ( "width"    , c_double),            #    double width;
              ( "beg"      , c_int   ),            #    int    beg;
              ( "end"      , c_int   ),            #    int    end;
              ( "time"     , c_int   ),            #    int    time;
              ( "track"    , POINTER( c_float ) ), #    float *track;
              ( "scores"   , POINTER( c_float ) )] #    float *scores;
                                                  #  } Whisker_Seg_Old;

class cWhisker_Seg(Structure):                      #typedef struct
  _fields_ = [( "id"       , c_int   ),            #{ int id;
              ( "time"     , c_int   ),            #  int time;
              ( "len"      , c_int   ),            #  int len;
              ( "x"        , POINTER( c_float ) ), #  float *x;
              ( "y"        , POINTER( c_float ) ), #  float *y;
              ( "thick"    , POINTER( c_float ) ), #  float *thick;
              ( "scores"   , POINTER( c_float ) )] #  float *scores;
                                                   #} Whisker_Seg;
  @staticmethod
  def CastFromWhiskerSeg( w ):
    return cWhisker_Seg( w.id,
                         w.time,
                         len(w.x),
                         w.x.ctypes.data_as( POINTER( c_float ) ),
                         w.y.ctypes.data_as( POINTER( c_float ) ),
                         w.thick.ctypes.data_as( POINTER( c_float ) ),
                         w.scores.ctypes.data_as( POINTER( c_float ) ) )
  @staticmethod
  def CastDictToArray( wvd ):
    """
    This function creates a ctypes array of cWhisker_Seg from an input
    whiskers dictionary.  The resulting array can be passed to functions
    in the C library.  Don't use Free_Whisker_Seg_Vec on this array; it
    doesn't "own" its data.  Instead, it passes most of the data as pointers
    to numpy arrays.

    Arguments:
    ---------
    <wvd> is a dictionary.  The keys are "frame ids" (integers indicating
          which time point in the movie).  The value's are another set of
          dictionaries.  The keys of these dictionaries are "whisker ids,"
          and the values are of <class `Whisker_Seg`>.

    Example:
    -------
    >>> import trace
    >>> wvd = trace.Load_Whiskers(r"proc\whisker_data_0140.whiskers")
    >>> wv = trace.cWhisker_Seg.CastDictToArray(wvd)
    >>> len(wv)
    40408
    >>> wv[0]
    <trace.cWhisker_Seg object at 0x03DACC60>
    """
    #first count the number of segments
    nseg = sum( map(len, list(wvd.values())) )

    #create the constructor
    type_wv = cWhisker_Seg * nseg;

    #alloc and fill the array
    def itersegs(wvd):
      for v in list(wvd.values()):
        for w in list(v.values()):
          yield w
    wv = type_wv( *list(map(cWhisker_Seg.CastFromWhiskerSeg,list(itersegs(wvd)))) )

    return wv

class cWhiskerIndex(Structure):
  _fields_ = [( 'index' ,   POINTER(POINTER( cWhisker_Seg )) ),
              ( 'sz'    ,   POINTER( c_int  ) ),
              ( 'n'     ,   c_int           ) ]

class cSeed(Structure):
  _fields_ = [( 'xpnt',   c_int ),
              ( 'ypnt',   c_int ),
              ( 'xdir',   c_int ),
              ( 'ydir',   c_int )]

  def asarray(self):
    return array([self.xpnt,self.ypnt,self.xdir,self.ydir])

  def __repr__(self):
    return "<%s.%s (%d,%d) slope: %5.3g>"%(
                str(self.__module__),
                "cSeed",
                self.xpnt,
                self.ypnt,
                arctan2(self.ydir, self.xdir)*180/pi)

class cSeedVector(Structure):                  # typedef struct
  _fields_ = [('nseeds', c_int            ),   #   { int   nseeds;
              ('seeds' , POINTER( cSeed ) )]   #     Seed *seeds;
                                               #   } Seed_Vector;
  def asarray(self):
    a = zeros(( self.nseeds, 4))
    for i in range( self.nseeds ):
      a[i] = self.seeds[i].asarray()
    return a

  def plot(self, *args, **kwargs):
    from pylab import quiver
    a = self.asarray()
    if not 'pivot' in kwargs:
      kwargs['pivot'] = 'middle'
    if not 'scale' in kwargs:
      kwargs['scale'] = 10
    if a.any():
      norm = hypot( a[:,2], a[:,3] )
      a[:,2] /= norm
      a[:,3] /= norm
      quiver( a[:,0], a[:,1], a[:,2], -a[:,3], **kwargs )

class cImage(Structure):
  _fields_ = [( "kind",     c_int   ),
              ( "width",    c_int   ),
              ( "height",   c_int   ),
              ( "text",     POINTER(c_char_p) ),
              ( "array",    POINTER(c_uint8))]
  @staticmethod
  def fromarray(im):
    return cImage(  c_int(  im.dtype.itemsize ),
                    c_int(  im.shape[1] ),
                    c_int(  im.shape[0] ),
                    pointer( c_char_p("") ),
                    im.ctypes.data_as( POINTER( c_uint8 ) ) )

class Whisker_Seg(object):
  def __init__(self, source=None):
    """
    If source is:
      None        : initialize an empty segment
      cWhisker_Seg: copy contents of cWhisker_Seg into a Whisker_Seg
      tuple       : must be (Whisker_Seg, start_index, stop_index)
                    Create a Whisker_Seg referencng the indicated subsegment

    """
    if source is None:
      self.id   = None
      self.time = None
      self.x    = array([])
      self.y    = array([])
      self.scores = array([])
      self.thick  = array([])
    elif isinstance( source, cWhisker_Seg ):
      self.id = source.id
      self.time = source.time

      self.x      = zeros( source.len, dtype=float32 )
      self.y      = zeros( source.len, dtype=float32)
      self.scores = zeros( source.len, dtype=float32 )
      self.thick  = zeros( source.len, dtype=float32 )

      for i in range( source.len ):
        self.x[i]      = source.x[i]
        self.y[i]      = source.y[i]
        self.thick[i]  = source.thick[i]
        self.scores[i] = source.scores[i]
    elif isinstance( source, tuple ):
      arg,a,b = source
      self.id   = arg.id
      self.time = arg.time
      self.x      = arg.x[a:b]
      self.y      = arg.y[a:b]
      self.thick  = arg.thick[a:b]
      self.scores = arg.scores[a:b]

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx ):
    return [self.x[idx],self.y[idx],self.thick[idx],self.scores[idx]]

  def asarray( self ):
    return concatenate((self.x,self.y,self.thick,self.scores)).reshape(4,len(self.x)).T

  def split( self, idx ):
    a,b = Whisker_Seg(), Whisker_Seg()
    a.id   = b.id   = self.id
    a.time = b.time = self.time
    a.x      = self.x     [:idx]
    a.y      = self.y     [:idx]
    a.thick  = self.thick [:idx]
    a.scores = self.scores[:idx]
    b.x      = self.x     [idx:]
    b.y      = self.y     [idx:]
    b.thick  = self.thick [idx:]
    b.scores = self.scores[idx:]
    if len(a) == 0:
      a = None
    if len(b) == 0:
      b = None
    return a,b

  def join_right( self, w ):
    self.x      = concatenate( (self.x     , w.x     ) )
    self.y      = concatenate( (self.y     , w.y     ) )
    self.thick  = concatenate( (self.thick , w.thick ) )
    self.scores = concatenate( (self.scores, w.scores) )
    return self

  @staticmethod
  def join( left, right ):
    """ Like `join_right` but the joined result is a new whisker. """
    w = left.copy()
    w.join_right( right )
    return w

  def copy(self):
    w = Whisker_Seg()
    w.id = self.id
    w.time = self.time

    w.x      = self.x.copy()
    w.y      = self.y.copy()
    w.scores = self.scores.copy()
    w.thick  = self.thick.copy()
    return w

#
# FILE I/O
#

cWhisk.Load_Whiskers.restype = POINTER( cWhisker_Seg )
cWhisk.Load_Whiskers.argtypes = [
  POINTER( c_char ),
  POINTER( c_char ),
  POINTER( c_int)]

cWhisk.Save_Whiskers.restype = c_int
cWhisk.Save_Whiskers.argtypes = [
  POINTER( c_char ),
  POINTER( c_char ),
  POINTER( cWhisker_Seg ),
  c_int]

def Load_Whiskers( filename ):
  """ Reads whisker segments from a file.

  Returns a dict of dict with Whisker_Seg elements.  The organization looks
  like:

  >>> from trace import Load_Whiskers
  >>> frameid = 0;
  >>> segmentid = 1;
  >>> wv = Load_Whiskers('test.whiskers')
  >>> w = wv[frameid][segmentid]
  >>> w.id == segmentid
  True
  >>> w.time == frameid
  True

  The file is read using a ctypes call to a libwhisk(trace.h) function.  The
  data is then copied into numpy containers and freed.

  It's not clear to me how this effects the part of the heap hidden from
  python's memmory management.  This is done for convenience, since later no
  one needs to remember to free anything and one gets to use numpy.array's for
  plotting, etc...

  For the kinds of things I imaging doing in python this function will only get
  called once per application instance.
  """
  if not os.path.exists(filename):
    raise IOError("File not found.")
  nwhiskers = c_int(0)
  filenameb = bytes(filename, 'utf-8')
  wv = cWhisk.Load_Whiskers( filenameb, None, byref(nwhiskers) );
  # organize into dictionary for ui.py {frameid}{segid}
  whiskers = {}
  for idx in range( nwhiskers.value ):
    w = wv[idx]
    whiskers[ w.time ] = {}
  for idx in range( nwhiskers.value ):
    w = Whisker_Seg(wv[idx])
    whiskers[ w.time ][ w.id ] = w;
  cWhisk.Free_Whisker_Seg_Vec( wv, nwhiskers )
  return whiskers

def Save_Whiskers( filename, whiskers ):
  #count the whiskers
  n = 0
  for v in list(whiskers.values()):
    n += len(v)
  #alloc the c whisker array
  wv = (cWhisker_Seg * n)()
  #copy into c whisker array
  i = 0
  for fid,v in list(whiskers.items()):
    for wid,t in list(v.items()):
      if not t:
        continue;
      wv[i].id   = wid
      wv[i].time = fid
      wv[i].len  = len(t.x)
      wv[i].x       = t.x.ctypes.data_as( POINTER( c_float ) )
      wv[i].y       = t.y.ctypes.data_as( POINTER( c_float ) )
      wv[i].thick   = t.thick.ctypes.data_as( POINTER( c_float ) )
      wv[i].scores  = t.scores.ctypes.data_as( POINTER( c_float ) )
      i += 1
  #prep for save and save
  #print "Saving %d"%i
  #pdb.set_trace()
  if not cWhisk.Save_Whiskers( filename, None, wv, i ):
    warn("Save Whiskers may have failed.")

