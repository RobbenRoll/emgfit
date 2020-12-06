======
emgfit
======

.. image:: https://github.com/RobbenRoll/emgfit/workflows/CI%20tests/badge.svg?branch=master
        :target: https://github.com/RobbenRoll/emgfit/actions?query=workflow%3A%22CI+tests%22

.. image:: https://img.shields.io/pypi/v/emgfit.svg
        :target: https://pypi.python.org/pypi/emgfit


Fitting of time-of-flight mass spectra with Hyper-EMG models

* Free software: 3-clause BSD license
* Online documentation: https://RobbenRoll.github.io/emgfit.
* Source code: https://github.com/RobbenRoll/emgfit

`emgfit` is a Python package for peak fitting of time-of-flight (TOF) mass
spectra with hyper-exponentially modified Gaussian (Hyper-EMG_ [1]_) model
functions. `emgfit` is a wrapper around the `lmfit`_ [2]_ curve fitting package
and uses many of lmfit's user-friendly high-level features. Experience with
`lmfit` can be helpful but is not an essential prerequisite for using `emgfit`
since the `lmfit` features stay largely 'hidden under the hood'. `emgfit` is
designed to be user-friendly and offers automation features whenever reasonable
while also supporting a large amount of flexibility and control for the user.
Depending on the user's preferences an entire spectrum can be rapidly analyzed
with only a few lines of code. Alternatively, various optional features are
available to aid the user in a more rigorous analysis. The model functions and
statistical methods provided by emgfit could be useful for analyses of
spectroscopic data from a variety of other fields.

Amongst other features, the `emgfit` toolbox includes:

* Automatic and sensitive peak detection
* Automatic import of relevant literature values from the AME2016_ [3]_ database
* Automatic selection of the best suited peak-shape model
* Fitting of low-statistics peaks with a binned maximum likelihood method
* Simultaneous fitting of an entire spectrum with a large number of peaks
* Export of all relevant fit results including fit statistics and plots to an
  EXCEL output file for convenient post-processing

`emgfit` is designed to be used within Jupyter Notebooks which have become a
standard tool in the data science community. The usage and capabilities of
`emgfit` are best explored by looking at the tutorial. The tutorial and more
details can be found in the `documentation of emgfit`_.

.. _`lmfit`: https://lmfit.github.io/lmfit-py/
.. _AME2016: http://amdc.in2p3.fr/web/masseval.html
.. _Hyper-EMG: https://www.sciencedirect.com/science/article/abs/pii/S1387380616302913
.. _documentation of emgfit: https://RobbenRoll.github.io/emgfit

References
----------
.. [1] Purushothaman, S., et al. "Hyper-EMG: A new probability distribution
   function composed of Exponentially Modified Gaussian distributions to analyze
   asymmetric peak shapes in high-resolution time-of-flight mass spectrometry."
   International Journal of Mass Spectrometry 421 (2017): 245-254.
.. [2] Newville, M., et al. "LMFIT: Non-linear least-square minimization and
   curve-fitting for Python." Astrophysics Source Code Library (2016):
   ascl-1606.
.. [3] Wang, M., et al. "The AME2016 atomic mass evaluation (II). Tables, graphs
   and references." Chinese Physics C 41.3 (2017): 030003.
