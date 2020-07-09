# Setup file for compiling .pyx module with cython
import distutils.core
import Cython.Build
import numpy

distutils.core.setup(
    ext_modules = Cython.Build.cythonize("emg_funcs_cython.pyx"),
    include_dirs = numpy.get_include())
