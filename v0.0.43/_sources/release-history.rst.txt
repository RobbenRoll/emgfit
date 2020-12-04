===============
Release History
===============

An index with links to the documentation of the latest and former emgfit
versions can be found `here`_.

.. _here: https://RobbenRoll.github.io/emgfit

v0.3.1 (2020-11-27)
-------------------

Changed
^^^^^^^
* Optimized plot appearance.

Fixed
^^^^^
* Fixed a bug causing crashes of parallelized fitting with
  :meth:`~emgfit.spectrum.spectrum.get_errors_from_resampling` &
  :meth:`~emgfit.spectrum.spectrum.get_MC_peakshape_errors` in Python3.8.
* Fixed some deprecation warnings.
* Added additional wait time to prevent Travis CI build from timing out
  prematurely.


v0.3.0 (2020-11-25)
-------------------

Added
^^^^^
* Added :mod:`emgfit.sample` module for easy generation of simulated spectra
  with Gaussian and hyper-EMG line shapes.
* Incorporated the option to perform blind analysis via the new
  :meth:`~emgfit.spectrum.spectrum.set_blinded_peaks` method. The latter hides
  the obtained mass values and positions of user-defined peaks-of-interest.
* Implemented :meth:`~emgfit.spectrum.spectrum.get_errors_from_resampling`
  method which can yield refined estimates of the statistical and peak area
  errors by performing a parametric bootstrap for each fitted peak.
* Added a Markov-Chain Monte Carlo sampling method
  (:meth:`~emgfit.spectrum.spectrum._get_MCMC_par_samples`) for mapping out the
  posterior distributions and correlations of model parameters. This method can
  be called with the `map_par_covar` option in the peak-shape determination.
* Added a method (
  :meth:`~emgfit.spectrum.spectrum.get_MC_peakshape_errors`) for
  obtaining refined peak-shape error estimates that account for correlations and
  non-normal posterior distributions of shape parameters. This method relies on
  shape parameter sets obtained via Markov-Chain Monte Carlo sampling.
* Added `peak_indeces` argument to :meth:`~emgfit.spectrum.spectrum.fit_peaks`
  to enable automatic fit range selection from the specified indeces of
  interest.
* Added `fit_kws` argument to peakfit method to enable more control over the
  underlying scipy optimization algorithms.
* Updated `emgfit` tutorial with new uncertainty estimation methods.
* Add concept articles and apply various edits to the documentation.

Changed
^^^^^^^
* Changed bounding of Pearson weights to addition of small number eps = 1e-10 in
  the denominator of the Pearson chi-square residuals. This ensures that the
  cost function asymptotically converges to a chi-squared distribution while
  still avoiding convergence issues due to overweighting of bins whose predicted
  number of counts approach zero.
* Changed automatic tail order determination in
  :meth:`~emgfit.spectrum.spectrum.determine_peak_shape` method. Now tail orders
  are excluded if either the corresponding eta *or tau* parameter agrees with
  zero within 1-sigma confidence.
* Extended peak-shape error evaluation methods to also estimate the
  corresponding peak area uncertainties and automatically add them in quadrature
  to the statistical peak area uncertainties.
* Updated formatting of peak properties table for more clarity including color
  coding to indicate the way uncertainties have been estimated.

Fixed
^^^^^
* Fixed bug in :meth:`~emgfit.spectrum.spectrum.remove_peaks` method.
* Fixed minor bug in :meth:`~emgfit.spectrum.spectrum._eval_peakshape_errors`
  method.


v0.2.3 (2020-09-18)
-------------------
* Updated docs and README.

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
