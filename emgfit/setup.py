# Setup file for compiling .pyx module with cython
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

setup(
    name = "emg_funcs_cython",
    ext_modules = cythonize("emg_funcs_cython.pyx", language_level = "3"),
    include_dirs = numpy.get_include(),
    )
