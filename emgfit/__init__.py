################################################################################
"""Python package for Hyper-EMG fitting of TOF mass spectra

.. moduleauthor:: Stefan Paul <stefan.paul@triumf.ca>
"""
##### Code by Stefan Paul

##### Import required modules
from .config import *
from .spectrum import *

##### Get package version
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

##### Check Python version
import sys
min_py_version = (3, 7)
if sys.version_info < min_py_version:
    msg = "emgfit only supports Python versions >= "+str(min_py_version)
    raise Exception(msg)

##### Import following modules immediately if package is loaded with import *
__all__ = [
        #'emg_funcs',
        'config',
	    'spectrum'
        ]


################################################################################
