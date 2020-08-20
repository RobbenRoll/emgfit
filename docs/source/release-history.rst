===============
Release History
===============

Change to current develop branch:
---------------------------------
* Improved calculation of effective mass shifts in peak-shape error
  determination (_eval_peak_shape_errors() method). The IOI mass shifts are now
  corrected for shifts of the calibrant mass using shifted mass re-calibration
  factors instead of taking the simple mass difference between shifted IOI and
  calibrant centroids.
* Added remove_peaks() method to spectrum class to allow removing multiple peaks
  at once, added deprecation warning to remove_peak() method.
* Added upper bound of 1 to Pearson weights for increased numerical stability in
  fits with 'chi-square' cost function. Now Pearson_weights = 1./np.maximum(1.,np.sqrt(y_m))
  where y_m is the model y-value in the foregoing fit iteration (2020-08-06)
* Improved handling of NaN values in calculation of negative log-likelihood
  ratio for MLE fit residuals (2020-07-11).
* Made determine_A_stat_emg() method more robust (better handling of ValueErrors
  due to NaNs in fit model y-values). (2020-07-11)
* Optimize and clean up emg_funcs, add support for Numba njit. Turn on all
  runtime warnings.

v0.1.0 (2020-06-08)
-------------------
Initial Release
