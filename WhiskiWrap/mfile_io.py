""" mfile_io.py

Author: Nathan Clack <clackn@janelia.hhmi.org>  
Date  : 25 January 2009

Copyright 2010 Howard Hughes Medical Institute.
All rights reserved.
Use is subject to Janelia Farm Research Campus Software Copyright 1.1
license terms (http://license.janelia.org/license/jfrc_copyright_1_1.html).

Added to WhiskiWrap repository by Vincent Prevosto, 05/2023
"""
import os,sys
from ctypes import *
from ctypes.util import find_library
import numpy
from numpy import zeros, double, fabs, ndarray, array
from . import wfile_io
from .wfile_io import cWhisker_Seg

import warnings

import pdb
from functools import reduce

dllpath = os.path.split(os.path.abspath(__file__))[0]
if sys.platform == 'win32':
  lib = os.path.join(dllpath,'whisk.dll')
else:
  lib = os.path.join(dllpath,'libwhisk.so')
os.environ['PATH']+=os.pathsep + os.pathsep.join(['.','..',dllpath])
name = find_library('whisk')
if not name:
  name=lib
try:
  cWhisk = cdll.LoadLibrary( name )
except:
  raise ImportError("Can not load whisk shared library"); 
if cWhisk._name==None:
  raise ImportError("Can not load whisk shared library");

_param_file = "default.parameters"
if cWhisk.Load_Params_File(_param_file)==1: #returns 0 on success, 1 on failure
  raise Exception("Could not load tracing parameters from file: %s"%_param_file)

class cMeasurements(Structure):
  """ Proxy for Measurements struct. 
  >>> from numpy.random import rand
  >>> data = rand(20,10)
  >>> table = cWhisk.Measurements_Table_From_Doubles( data.ctypes.data_as(POINTER(c_double)), 20, 10 )
  >>> table[0].n
  7
  >>> table   # doctest:+ELLIPSIS
  <ctypes.LP_cMeasurements object at ...>
  """
  _fields_ = [("row",            c_int               ),
              ("fid",            c_int               ),                                                           
              ("wid",            c_int               ),                                                           
              ("state",          c_int               ),                                                           
              ("face_x",         c_int               ),             #// used in ordering whiskers on the face...roughly, the center of the face                                              
              ("face_y",         c_int               ),             #//                                      ...does not need to be in image                                                 
              ("col_follicle_x", c_int               ),             #// index of the column corresponding to the folicle x position                                                          
              ("col_follicle_y", c_int               ),             #// index of the column corresponding to the folicle y position                                                          
              ("valid_velocity", c_int               ),                                                           
              ("n",              c_int               ),                                                           
              ("face_axis",      c_char              ),
              ("data",           POINTER( c_double ) ),             # // array of n elements                      
              ("velocity",       POINTER( c_double ) )]             # // array of n elements - change in data/time

class cDistributions(Structure):
  """
  >>> this = cWhisk.Alloc_Distributions( 32, 8, 4 )
  >>> this                                          # doctest:+ELLIPSIS 
  <ctypes.LP_cDistributions object at ...>
  >>> cWhisk.Free_Distributions( this )
  """
  _fields_ = [("n_measures",   c_int               ),
              ("n_states",     c_int               ),
              ("n_bins",       c_int               ),
              ("bin_min",      POINTER( c_double ) ),   # // array of n_measures elements                                                          
              ("bin_delta",    POINTER( c_double ) ),   # // array of n_measures elements                                                          
              ("data",         POINTER( c_double ) )]   # // array of holding histogram information with dimensions (n_bins,n_measures,n_states)
  def asarray(self):
    d = zeros( (self.n_states, self.n_measures, self.n_bins) )
    cWhisk.Copy_Distribution_To_Doubles( byref(self), d.ctypes.data_as( POINTER(c_double) ) )
    return d
  
  def bins_as_array(self):
    b = zeros( (self.n_measures, self.n_bins) )
    cWhisk.Distributions_Bins_To_Doubles( byref(self), b.ctypes.data_as( POINTER(c_double) ) )
    return b

class MeasurementsTable(object):
  """
  >>> data = numpy.load('data/testing/seq140[autotraj].npy')
  >>> table = MeasurementsTable(data)
  >>> table._measurements[0].n
  8
  """
  def __init__(self, datasource):
    """
    Load table from numpy array or from a file.

    >>> table = MeasurementsTable( zeros((500,5)) )

    >>> table = MeasurementsTable( "data/testing/seq140[autotraj].measurements" )

    >>> import wfile_io
    >>> wvd = wfile_io.Load_Whiskers( "data/testing/seq140.whiskers" )
    >>> table = MeasurementsTable( {'whiskers':wvd, 'facehint':'left'} )

    """
    object.__init__(self)
    self._measurements = None
    self._nrows = 0
    self._sort_state = None
    self._free_measurements = cWhisk.Free_Measurements_Table
    if isinstance(datasource,str):
      self._load(datasource)
    elif isinstance(datasource,dict):
      wvd      = datasource['whiskers']
      facehint = datasource['facehint']
      self._measurements, self._nrows = MeasurementsTable._fromWhiskerDictWithFacehint( wvd, facehint )
    else:
      self._measurements = cWhisk.Measurements_Table_From_Doubles( 
                              datasource.ctypes.data_as( POINTER(c_double) ),   # data buffer       
                              datasource.shape[0],                              # number of rows    
                              datasource.shape[1] )                             # number of columns 
      self._nrows = datasource.shape[0]

  def __del__(self):
    """
    >>> table = MeasurementsTable( zeros((500,5)) )
    >>> del table
    """
    self._free_measurements(self._measurements)

  @staticmethod
  def _fromWhiskerDict(wvd, face_xy, faceaxis ):
    """
    Returns: LP_cMeasurements, int
      
    Warning: the returned cMeasurements object needs to be properly deallocated
             when finished.  Potential memory leak.  For this reason, use the 
             MeasurementsTable constructor (__init__) instead.
    """
    facex,facey = face_xy
    wv = wfile_io.cWhisker_Seg.CastDictToArray(wvd)
    return cWhisk.Whisker_Segments_Measure(wv,len(wv), facex, facey, faceaxis), len(wv)
  
  @staticmethod
  def _fromWhiskerDictWithFacehint(wvd, facehint ):
    """
    Returns: LP_cMeasurements, int
      
    Warning: the returned cMeasurements object needs to be properly deallocated
             when finished.  Potential memory leak.  For this reason, use the 
             MeasurementsTable constructor (__init__) instead.
    """
    x,y,ax = c_int(),c_int(),c_char()
    wv = wfile_io.cWhisker_Seg.CastDictToArray(wvd)
    cWhisk.face_point_from_hint( wv, len(wv), facehint, byref(x), byref(y), byref(ax))
    return cWhisk.Whisker_Segments_Measure(wv,len(wv), x.value, y.value, ax.value), len(wv)


  def asarray(self):
    """  
    >>> from numpy.random import rand
    >>> data = rand(200,10)
    >>> table = MeasurementsTable(data)
    >>> shape = table.asarray()
    >>> print shape.shape
    (200, 10)
    >>> print (shape[:,3:]==data[:,3:]).all()
    True
    """
    if self._nrows==0:
      return []
    data = zeros( (self._nrows, self._measurements[0].n+3), dtype=double )
    cWhisk.Measurements_Table_Data_To_Doubles(self._measurements, 
                                             self._nrows, 
                                             data.ctypes.data_as( POINTER( c_double ))
                                            );
    return data

  def get_trajectories(self):
    """
    >>> table = MeasurementsTable( "data/testing/seq140[autotraj].measurements" )
    >>> traj  = table.get_trajectories()
    >>> max(traj.keys())
    3
    >>> traj.has_key(-1)
    False
    """
    data = self.asarray()
    t = {}
    for row in data:
      r = list(map(int,row[:3]))
      t.setdefault( r[0],{} ).setdefault( r[1], r[2] ) 
    if -1 in list(t.keys()):
      del t[-1]
    return t

  def save_trajectories(self, filename, excludes=[]):
    """  Saves to a trajectories file.

    >>> table = MeasurementsTable( "data/testing/seq140[autotraj].measurements" )
    >>> table.save_trajectories( "data/testing/trash.trajectories" ) # doctest:+ELLIPSIS 
    <...MeasurementsTable object at ...>
    """
    trajectories = self.get_trajectories()
    f = open( filename, 'w' )
    for k,v in trajectories.items():
      if not k in excludes:
        for s,t in v.items():
          print('%d,%d,%d'%(k,s,t), file=f)
    return self

  def load_trajectories(self,filename ):
    """  Loads trajectories and saves them to the table.
    Trajectory id's correspond to the `state` label.

    >>> table = MeasurementsTable( "data/testing/seq140[autotraj].measurements" )
    >>> table.save_trajectories( "data/testing/trash.trajectories" )  # doctest:+ELLIPSIS 
    <...MeasurementsTable object at ...>
    >>> table.load_trajectories( "data/testing/trash.trajectories" )  # doctest:+ELLIPSIS 
    <...MeasurementsTable object at ...>
    """
    trajectories = {}
    f = open( filename, 'r' )

    cur = 0;
    for line in f:
      t = [int(x) for x in line.split(',')[:3]]
      if not t[0] in trajectories:
        trajectories[t[0]] = {}
      trajectories[ t[0] ][ t[1] ] = t[2];

    self.commit_trajectories( trajectories )
    return self

  def commit_trajectories(self,traj):
    """
    >>> traj = {0: {0:0,1:0}, 1: {0:1,1:1} }
    >>> table = MeasurementsTable( "data/testing/seq140[autotraj].measurements" )
    >>> mn,mx = table.commit_trajectories(traj).get_state_range()
    >>> mn
    0
    >>> mx
    1
    >>> lentraj = lambda x: len(table.get_shape_data(x))
    >>> lentraj(0)
    2
    >>> add = lambda a,b:a+b
    >>> table._nrows == reduce(add, map(lentraj,xrange(mn-1,mx+1)))
    True
    """
    inv = {}
    for tid,t in traj.items():
      for k in t.items():
        inv[k] = tid  

    for i in range(self._nrows):  #update new
      row = self._measurements[i]
      s = inv.get( (row.fid,row.wid) )
      row.state = s if (not s is None) else -1 

    return self

  def get_state_range(self):
    """
    >>> data = numpy.load('data/testing/seq140[autotraj].npy')
    >>> mn,mx = MeasurementsTable(data).update_velocities().get_state_range()
    >>> mn
    0
    >>> mx
    3
    """ 
    mn,mx = c_int(),c_int()
    sorted = (not self._sort_state is None ) and \
             ("state" in self._sort_state  )
    n = cWhisk._count_n_states(self._measurements,
                              self._nrows,
                              sorted,
                              byref(mn),
                              byref(mx))
    f = lambda x: x.value if x.value >=0 else 0
    return list(map(f,[mn,mx]))

  def iter_state(self):
    """
    >>> table = MeasurementsTable( "data/testing/seq140[autotraj].measurements" )
    >>> for i in table.update_velocities().iter_state():
    ...     print i
    ...     
    0
    1
    2
    3
    """
    mn,mx = self.get_state_range()
    return range(mn,mx+1)

  def get_shape_table(self):
    """  
    >>> from numpy.random import rand
    >>> data = rand(200,10)
    >>> table = MeasurementsTable(data)
    >>> shape = table.get_shape_table()
    """
    shape = zeros( (self._nrows, self._measurements[0].n), dtype=double )
    cWhisk.Measurements_Table_Copy_Shape_Data( self._measurements, 
                                              self._nrows, 
                                              shape.ctypes.data_as( POINTER(c_double) ) )
    return shape

  def get_time_and_mask(self, state, rows = None):
    """
    Returns `time` and `valid velocity` mask for selected state.  
    Order of results is determined by the table's sort order.

    >>> data = numpy.load('data/testing/seq140[autotraj].npy')
    >>> table = MeasurementsTable(data).update_velocities()
    >>> time,mask = table.get_time_and_mask(1)
    """
    if rows is None:
      rows = cWhisk.Measurements_Table_Size_Select_State( self._measurements, self._nrows, int(state) )
    time = zeros( rows, dtype = double )
    mask = zeros( rows, dtype = int    )
    cWhisk.Measurements_Table_Select_Time_And_Mask_By_State( self._measurements, 
                                                            self._nrows,
                                                            int(state),
                                                            time.ctypes.data_as (POINTER( c_double )), 
                                                            mask.ctypes.data_as (POINTER( c_int    )) )
    return time,mask

  def get_velocities(self, state, rows = None):
    """
    Returns velocity for selected state.  
    Order of results is determined by the table's sort order.

    >>> data = numpy.load('data/testing/seq140[autotraj].npy')
    >>> table = MeasurementsTable(data)
    >>> table.update_velocities() # doctest:+ELLIPSIS 
    <...MeasurementsTable object at ...>
    >>> velocities = table.get_velocities(1)
    """
    if rows is None:
      rows = cWhisk.Measurements_Table_Size_Select_State( self._measurements, self._nrows, int(state) )
    vel  = zeros( (rows, self._measurements[0].n ), dtype = double )
    cWhisk.Measurements_Table_Select_Velocities_By_State( self._measurements, 
                                                         self._nrows,
                                                         int(state),
                                                         vel.ctypes.data_as (POINTER( c_double )) )
    return vel
  
  def get_shape_data(self, state, rows = None):
    """
    Returns shape data for selected state.  
    Order of results is determined by the table's sort order.

    >>> data = numpy.load('data/testing/seq140[autotraj].npy')
    >>> table = MeasurementsTable(data).update_velocities()
    >>> shape = table.get_shape_data(1)
    
    >>> table = MeasurementsTable('data/testing/seq140[autotraj].measurements').update_velocities()
    >>> shape = table.get_shape_data(1)
    """
    if rows is None:
      rows = cWhisk.Measurements_Table_Size_Select_State( self._measurements, self._nrows, int(state) )
    data  = zeros( (rows, self._measurements[0].n ), dtype = double )
    cWhisk.Measurements_Table_Select_Shape_By_State( self._measurements, 
                                                    self._nrows,
                                                    int(state),
                                                    data.ctypes.data_as (POINTER( c_double )) )
    return data

  def get_data(self, state, rows = None ):
    """
    Returns time, shape, velocity and velocity_valid  data for selected state.  
    Order of results is determined by the table's sort order.

    >>> data = numpy.load('data/testing/seq140[autotraj].npy')
    >>> table = MeasurementsTable(data).update_velocities()
    >>> time,shp,vel,mask = table.get_data(1)
    """
    if rows is None:
      time,mask = self.get_time_and_mask(state) 
    else:
      time,mask = self.get_time_and_mask(state, rows = rows) 
    vel = self.get_velocities(state, rows = time.shape[0] )
    shp = self.get_shape_data(state, rows = time.shape[0] )
    return time, shp, vel, mask

  def get_velocities_table(self):
    """
    >>> data = numpy.load('data/testing/seq140[autotraj].npy')
    >>> table = MeasurementsTable(data).update_velocities()
    >>> vel = table.get_velocities_table()
    """
    vel = zeros( (self._nrows, self._measurements[0].n), dtype=double )
    cWhisk.Measurements_Table_Copy_Velocities( self._measurements, 
                                              self._nrows, 
                                              vel.ctypes.data_as( POINTER(c_double) ) )
    return vel

  def set_constant_face_position(self, x, y):
    """
    >>> table = MeasurementsTable( "data/testing/seq140[autotraj].measurements" )
    >>> table = table.set_constant_face_position( -100, 100 )
    >>> table._measurements[0].face_x
    -100
    >>> table._measurements[0].face_y
    100
    """
    cWhisk.Measurements_Table_Set_Constant_Face_Position( self._measurements, self._nrows, x, y )
    return self

  def set_follicle_position_column(self, ix, iy):
    """
    >>> table = MeasurementsTable( "data/testing/seq140[autotraj].measurements" )
    >>> table = table.set_follicle_position_column( 7, 8 ) 
    >>> table._measurements[0].col_follicle_x
    7
    >>> table._measurements[0].col_follicle_y
    8
    """
    cWhisk.Measurements_Table_Set_Follicle_Position_Indices( self._measurements, self._nrows, ix, iy )
    return self

  def sort_by_state_time(self):
    """
    >>> data = numpy.load('data/testing/seq140[autotraj].npy')
    >>> table = MeasurementsTable(data).sort_by_state_time() 
    >>> table._measurements[0].state
    -1
    >>> table._measurements[0].fid
    0
    >>> table._measurements[table._nrows].state
    0
    >>> table._measurements[table._nrows-1].state
    3
    >>> table._measurements[table._nrows-1].fid
    4598
    >>> table._sort_state
    'state,time'
    """
    sortstate = "state,time"
    if self._sort_state != sortstate:
      cWhisk.Sort_Measurements_Table_State_Time( self._measurements, self._nrows )
      self._sort_state = sortstate
    return self

  def sort_by_time(self):
    """
    >>> data = numpy.load('data/testing/seq140[autotraj].npy')
    >>> table = MeasurementsTable(data).sort_by_time() 
    >>> table._measurements[0].fid
    0
    >>> table._measurements[table._nrows-1].fid
    4598
    >>> table._sort_state
    'time'
    """
    sortstate = "time"
    if self._sort_state != sortstate:
      cWhisk.Sort_Measurements_Table_Time( self._measurements, self._nrows )
      self._sort_state = sortstate
    return self

  def sort_by_time_face(self):
    """
    >>> table = MeasurementsTable( "data/testing/seq140[autotraj].measurements" )
    >>> table = table.set_constant_face_position( -100, 100 ).set_follicle_position_column( 7, 8 )
    >>> table = table.sort_by_time_face()
    >>> table._measurements[0].fid
    0
    >>> table._measurements[table._nrows-1].fid
    4598
    >>> table._sort_state
    'time,face'
    """
    sortstate = "time,face"
    if(self._sort_state != sortstate):
      cWhisk.Sort_Measurements_Table_Time_Face( self._measurements, self._nrows )
      self._sort_state = sortstate
    return self

  def update_velocities(self):
    """
    >>> data = numpy.load('data/testing/seq140[autotraj].npy')
    >>> table = MeasurementsTable(data).update_velocities() 
    >>> vel = table.get_velocities_table()
    """
    self.sort_by_state_time()
    cWhisk.Measurements_Table_Compute_Velocities( self._measurements, self._nrows )
    return self

  def save(self, filename):
    """
    >>> data = numpy.load('data/testing/seq140[autotraj].npy')
    >>> table = MeasurementsTable(data).update_velocities()
    >>> table.save( "data/testing/trash.measurements" )    # doctest:+ELLIPSIS 
    <...MeasurementsTable object at ...>
    """
    cWhisk.Measurements_Table_To_Filename( filename, None, self._measurements, self._nrows )
    return self

  def save_to_matlab_file(self, filename, format = '5'):
    """
    Saves shape measurements to Matlab's .mat format.

    This uses the `scipy.io.matlab.savemat` function.  See that functions documentation for
    details on input options.

    >>> table = MeasurementsTable( "data/testing/seq140[autotraj].measurements" )
    >>> table.save_to_matlab_file( "data/testing/trash.mat" )    # doctest:+ELLIPSIS 
    <...MeasurementsTable object at ...>
    """
    from scipy.io.matlab import savemat
    kwargs = locals().copy()
    for k in [ 'self', 'savemat', 'filename' ]:
      del kwargs[k]
    savemat( filename, 
            { 'measurements': self.asarray() }, 
            **kwargs)
    return self

  def _load(self, filename):
    """
    Loads table from a saved file.

    >>> table = MeasurementsTable( "data/testing/seq140[autotraj].measurements" )
    """
    if not os.path.exists(filename):
      raise IOError("Could not find file %s"%filename)
    nrows = c_int()
    if self._measurements:
      cWhisk.Free_Measurements_Table( self._measurements )

    # import ctypes

    # Define the argument types for Measurements_Table_From_Filename
    # cWhisk.Measurements_Table_From_Filename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]

    # Prepare the filename and other variables
    bfilename = bytes(filename, 'utf-8') # Convert the string to a bytes object
    formats = [bytes(f, 'utf-8') for f in ['v0', 'v1', 'v2', 'v3']]
    # See Measurements_File_Formats in measurements_io.c for more information. v3 is default and will be 
    # automatically detected anyway. Older formats are for 32bit systems. 

    self._measurements = cWhisk.Measurements_Table_From_Filename(bfilename, formats[-1], byref(nrows))
                                                    #  ctypes.byref(nrows))
    self._nrows = nrows.value
    self._sort_state = None #unknown
    return self
  
  def diff_identity(self, table):
    """
    Searches two tables for different identity assignments and returns
    a list of frames where a difference was found.  Ideally, the two
    tables would have been derived from the same movie.

    If the tables are identical, an empty list is returned:

    >>> A = MeasurementsTable( "data/testing/seq140[autotraj].measurements" )
    >>> B = MeasurementsTable( "data/testing/seq140[autotraj].measurements" )
    >>> A.diff_identity(B)
    []

    The differences are not symmetric.  This is because "junk" states in the
    table on the left-hand side (`self`) are ignored.

    >>> B = MeasurementsTable( "data/testing/seq140[solve].measurements" )
    >>> len(B.diff_identity(A))
    69
    >>> len(A.diff_identity(B))
    25
    """
    nframes = c_int(0)
    frames = cWhisk.Measurements_Tables_Get_Diff_Frames( self._measurements, self._nrows, 
                                                        table._measurements, table._nrows, 
                                                        byref(nframes) )
    return [frames[i] for i in range(nframes.value)]

  def est_length_threshold(self,lowpx=1.0/0.04,highpx=50.0/0.04):
    ncount = c_int(0)
    thresh = cWhisk.Measurements_Table_Estimate_Best_Threshold(
        self._measurements,
        self._nrows,
        c_int(0),           # length column
        lowpx,highpx,
        1,                  # use greater than
        byref(ncount))      # estimated number of whiskers
    return thresh,ncount
        
class Distributions(object):
  def __init__(self, table = None, nbins = 32):
    """
    Create an empty Distributions object:
    
    >>> dists = Distributions()
    
    Initialize Distributions using a MeasurementTable:
    
    >>> import numpy
    >>> data = numpy.load( "data/testing/seq140[autotraj].npy" )
    >>> table = MeasurementsTable(data)

    >>> table = MeasurementsTable('data/testing/seq140[autotraj].measurements')
    >>> dists = Distributions(table.update_velocities())         # doctest:+ELLIPSIS  
    ...
    """
    object.__init__(self)
    self._free_distributions = cWhisk.Free_Distributions
    self._shp = None
    self._vel = None
    if not table is None:
      self.build(table, nbins)  
  
  def __del__(self):
    if self._shp:
      self._free_distributions( self._shp )
    if self._vel:
      self._free_distributions( self._vel )
  
  def build(self, table, nbins = 32):
    """
    >>> dists = Distributions()
    >>> table = MeasurementsTable('data/testing/seq140[autotraj].measurements')
    >>> dists.build(table)                   # doctest:+ELLIPSIS  
    <...Distributions object at ...>
    """
    assert isinstance(table,MeasurementsTable), "Wrong type for table."
    table.sort_by_state_time()
    self._shp = cWhisk.Build_Distributions         ( table._measurements, table._nrows, nbins )
    cWhisk.Distributions_Normalize( self._shp )
    cWhisk.Distributions_Apply_Log2( self._shp )
    table.update_velocities()
    self._vel = cWhisk.Build_Velocity_Distributions( table._measurements, table._nrows, nbins ) #changes table's sort order
    cWhisk.Distributions_Normalize( self._vel )
    cWhisk.Distributions_Apply_Log2( self._vel )
    table._sort_state = 'time'
    return self
  
  def velocities(self):
    """
    >>> dists = Distributions( MeasurementsTable('data/testing/seq140[autotraj].measurements') )
    >>> vbins, v = dists.velocities()
    """
    return self._vel[0].bins_as_array(), self._vel[0].asarray()  
  
  def shapes(self):
    """
    >>> dists = Distributions( MeasurementsTable('data/testing/seq140[autotraj].measurements') )
    >>> sbins, s = dists.shapes()
    """
    return self._shp[0].bins_as_array(), self._shp[0].asarray()  

# def solve( table ):
#   cWhisk.Solve( table._measurements, table._nrows, 32, 8096 )
#   table._sort_state = "time"
#   return table

def batch_make_measurements(sourcepath, ext = '*.seq', label = 'curated'):
  """
  To update/remake a measurements table, delete the *.npy and *.measurements
  files in the `sourcepath`.
  """
  warnings.simplefilter("ignore")
  from glob import glob
  from ui.whiskerdata import load_trajectories
  from .wfile_io import Load_Whiskers
  from . import summary
  warnings.simplefilter("default")

  def get_summary_data( filename, whiskers, trajectories ):
    if os.path.exists(filename):
      data = numpy.load(filename)
    else:
      data = array(list( summary.features(whiskers) ))
      numpy.save( filename, data )
      return data
    return summary.commit_traj_to_data_table( trajectories, data )

  for name in glob( os.path.join( sourcepath, ext ) ):
    root,ext = os.path.splitext( name )
    prefix = root + '[%s]'%label
    if not os.path.exists( prefix + '.measurements' ):
      t,tid = load_trajectories( prefix + '.trajectories' )
      print(prefix)
      print(list(t.keys()))
      w = Load_Whiskers( prefix + '.whiskers' ) 
      data = get_summary_data( prefix + '.npy', w, t )
      MeasurementsTable( data ).update_velocities().save( prefix + '.measurements' )


#
# Declarations 
#
cWhisk.Whisker_Segments_Measure.restype = POINTER( cMeasurements )
cWhisk.Whisker_Segments_Measure.argtypes = [
  POINTER( cWhisker_Seg ), # array of whisker segments
  c_int,                   # number of whisker segments
  c_int,                   # face x position (px)
  c_int,                   # face y position (px)
  c_char ]                 # face orientation ( one of: 'h','v','x' or 'y' )

# cWhisk.Whisker_Segments_Measure_With_Bar.restype = POINTER( cMeasurements )
# cWhisk.Whisker_Segments_Measure_With_Bar.argtypes = [
#   POINTER( cWhisker_Seg ), # array of whisker segments
#   c_int,                   # number of whisker segments
#   POINTER( cBar ),         # array of bar locations
#   c_int,                   # number of bar positions
#   c_int,                   # face x position (px)
#   c_int,                   # face y position (px)
#   c_char ]                 # face orientation ( one of: 'h','v','x' or 'y' )

cWhisk.face_point_from_hint.restype = None
cWhisk.face_point_from_hint.argtypes = [
  POINTER( cWhisker_Seg ), # array of whisker segments
  c_int,                   # number of whisker segments
  POINTER( c_char ),       # face hint
  POINTER( c_int  ),       # (out) face x position (px)
  POINTER( c_int  ),       # (out) face y position (px)
  POINTER( c_char ) ]      # (out) face orientation ( one of: 'h','v','x' or 'y' )
  

cWhisk.Measurements_Table_From_Doubles.restype = POINTER(cMeasurements)
cWhisk.Measurements_Table_From_Doubles.argtypes = [
  POINTER( c_double ), # data buffer
  c_int,               # number of rows
  c_int ]              # number of columns

cWhisk.Measurements_Table_Copy_Shape_Data.restype = None
cWhisk.Measurements_Table_Copy_Shape_Data.argtypes = [
  POINTER( cMeasurements ), # the table (the source)
  c_int,                    # number of rows
  POINTER( c_double ) ]     # destination

cWhisk.Measurements_Table_Copy_Velocities.restype = None
cWhisk.Measurements_Table_Copy_Velocities.argtypes = [
  POINTER( cMeasurements ), # the table (the source)
  c_int,                    # number of rows
  POINTER( c_double ) ]     # destination

cWhisk.Measurements_Table_From_Filename.restype = POINTER(cMeasurements)
cWhisk.Measurements_Table_From_Filename.argtypes = [
  POINTER( c_char ),
  POINTER( c_char ),
  POINTER( c_int  ) ]

cWhisk.Alloc_Distributions.restype = POINTER(cDistributions)
cWhisk.Alloc_Distributions.argtypes = [
  c_int,  # n_bins
  c_int,  # n_measures
  c_int ] # n_states

cWhisk.Free_Distributions.restype = None
cWhisk.Free_Distributions.argtypes = [ POINTER(cDistributions) ]

cWhisk.Build_Distributions.restype = POINTER( cDistributions )
cWhisk.Build_Distributions.argtype = [
  POINTER( cMeasurements ), # measurements table
  c_int,                    # number of rows
  c_int ]                   # number of bins

cWhisk.Build_Velocity_Distributions.restype = POINTER( cDistributions )
cWhisk.Build_Velocity_Distributions.argtype = [
  POINTER( cMeasurements ), # measurements table
  c_int,                    # number of rows
  c_int ]                   # number of bins

cWhisk.Solve.restype = None
cWhisk.argtypes = [
  POINTER( cMeasurements ), # table
  c_int,                    # number of rows
  c_int ]                   # number of bins

cWhisk.Measurements_Tables_Get_Diff_Frames.restype = POINTER( c_int )
cWhisk.Measurements_Tables_Get_Diff_Frames.argtypes = [
  POINTER( cMeasurements ), #table A
  c_int,                    #number of rows for table A
  POINTER( cMeasurements ), #table B                   
  c_int,                    #number of rows for table B
  POINTER( c_int ) ]        #size of returned static array

cWhisk.Measurements_Table_Estimate_Best_Threshold.restype = c_double
cWhisk.Measurements_Table_Estimate_Best_Threshold.argtypes = [
  POINTER( cMeasurements ), # table
  c_int,                    # n_rows
  c_int,                    # column index of the feature to use
  c_double,                 # low (px)
  c_double,                 # high (px)
  c_int,                    # is_gt
  POINTER(c_int)            # (output) target count
]


# if __name__=='__main__':
#   testcases = [ 
#                 Tests_MeasurementsTable_FromDoubles,
#                 Tests_MeasurementsTable_FromFile ,
#                 Tests_Distributions 
#                 ]
#   suite = reduce( lambda a,b: a if a.addTest(b) else a, 
#                   list(map( unittest.defaultTestLoader.loadTestsFromTestCase, testcases )) 
#                 )
#   suite.addTest( doctest.DocTestSuite() )
#   runner = unittest.TextTestRunner(verbosity=2,descriptions=1).run(suite)
