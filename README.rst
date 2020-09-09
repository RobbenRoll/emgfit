======
emgfit
======

.. image:: https://travis-ci.org/RobbenRoll/emgfit.svg
        :target: https://travis-ci.org/RobbenRoll/emgfit

.. image:: https://img.shields.io/pypi/v/emgfit.svg
        :target: https://pypi.python.org/pypi/emgfit


Fitting of MR-TOF mass spectra with Hyper-EMG models

* Free software: 3-clause BSD license
* Online documentation: https://RobbenRoll.github.io/emgfit.

`emgfit` is a Python package for peak fitting of MR-TOF mass spectra with
hyper-exponentially modified Gaussian (Hyper-EMG) model functions. `emgfit` is a
wrapper around the `lmfit` curve fitting package and uses many of lmfit's
user-friendly high-level features. Experience with `lmfit` can be helpful but is
not an essential prerequisite for using `emgfit` since the `lmfit` features stay
largely 'hidden under the hood'. `emgfit` is designed to be user-friendly and
offers automization features whenever reasonable while also supporting a
large amount of flexibility and control for the user. Depending on the user's
preferences an entire spectrum can be rapidly analyzed with only a few lines of
code. Alternatively, various optional features are available to aid the user in
a more rigorous analysis process.

Amongst other features, the `emgfit` toolbox includes:

* Automatic and sensitive peak detection
* Automatic import of relevant literature values from the AME2016 database
* Automatic selection of the best suited peak-shape model
* Fitting of low-statistics peaks with a binned maximum likelihood method
* Simultaneous fitting of an entire spectrum with a large number of peaks
* Export of all relevant fit results including fit statistics and plots to an
  EXCEL output file for convenient post-processing

`emgfit` is designed to be used within Jupyter Notebooks which have become a
standard tool in the data science community. The usage and capabilities of
`emgfit` are best explored by looking at the tutorial. The tutorial and more
details can be found in the `documentation of emgfit`_.

.. _documentation of emgfit: https://RobbenRoll.github.io/emgfit