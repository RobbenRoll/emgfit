===============
Release History
===============

An index with links to the documentation of every emgfit version can be found `here`_.

.. _here: https://RobbenRoll.github.io/emgfit

v0.2.2 (2020-09-16)
-------------------
* Fixed bug in mass re-scaling in peak-shape error evaluation.
* Relevant for developers only: Further automatized the deployment of new
  releases.

v0.2.1
------
* Version number skipped due to administrative reasons. 

v0.2.0 (2020-09-09):
--------------------
* Improved numerical robustness and speed of Hyper-EMG functions in emg_funcs
  module. The improved routines avoid arithmetic overflow of exp() or underflow
  of erfc().
* Improved calculation of effective mass shifts in peak-shape error
  determination (_eval_peak_shape_errors() method). The IOI mass shifts are now
  corrected for shifts of the calibrant mass using shifted mass re-calibration
  factors instead of taking the simple mass difference between shifted IOI and
  calibrant centroids.
* Added remove_peaks() method to spectrum class to allow removing multiple peaks
  at once, the remove_peak() method is deprecated but still supported.
* Added upper bound of 1 to Pearson weights for increased numerical stability in
  fits with 'chi-square' cost function. Now Pearson_weights =
  1./np.maximum(1.,np.sqrt(y_m)) where y_m is the model y-value in the foregoing
  fit iteration.
* Improved handling of NaN values in calculation of negative log-likelihood
  ratio for MLE fit residuals.
* Made determine_A_stat_emg() method more robust (better handling of ValueErrors
  due to NaNs in fit model y-values).

v0.1.0 (2020-06-08)
-------------------
Initial Release
