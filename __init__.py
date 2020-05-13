###################################################################################################
"""Python package for Hyper-EMG fitting of TOF mass spectra

.. moduleauthor:: Stefan Paul <stefan.paul@triumf.ca>
"""
##### Code by Stefan Paul

##### Import required modules
from .config import *
from .spectrum import *

##### Define package version
__version__ = '0.0.7'

##### Check Python version
import sys
min_py_version = (3, 0)
assert sys.version_info >= min_py_version, "emgfit package requires Python versions >= "+str(min_py_version)

##### Import following modules immediately if package is loaded with import *
__all__ = [
        #'emg_funcs',
        'config',
	    'spectrum'
        ]


##################################################################################################
