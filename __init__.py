###################################################################################################
##### Python module for Hyper-EMG fitting of TOF mass spectra
##### Fitting routines taken from lmfit package
##### Code by Stefan Paul, 2019-12-28

##### Import required subpackages
from emgfit.config import *
from emgfit.spectrum import *

##### Define package version
__version__ = '0.0.1'

##### Check Python version
import sys
min_py_version = (3, 0)
assert sys.version_info >= min_py_version, "emgfit package requires Python versions >= "+str(min_py_version)

##### Import following modules immediately if package is loaded with import *
__all__ = [
        #'emg_funcs',
        'config',
        'emg22_fit',
        'peak_detect',
	    'spectrum'
        ]


##################################################################################################
