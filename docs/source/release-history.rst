===============
Release History
===============

An index with links to the documentation of the latest and former emgfit
versions can be found `here`_.

.. _here: https://RobbenRoll.github.io/emgfit


v0.5.0 (2024-01-21)
-------------------

Added 
^^^^^
* Add the :mod:`~emgfit.hypothesis_tests` module to allow for likelihood ratio 
  hypothesis testing as a means to quantify the experimental evidence for the 
  presence of a peak. 
* Add `vary_shape` option to :meth:`~emgfit.spectrum.spectrum.fit_calibrant` 
  method of spectrum class. 
* Add :attr:`~emgfit.spectrum.spectrum.resolving_power` attribute to spectrum 
  class to enable the convenient initialization of the peak width parameter 
  `sigma` based on the typical resolving power of the instrument.
* Add :mod:`~emgfit.model` module with custom model interface, thus providing
  a cleaner way to override lmfit's default residual with emgfit's custom cost 
  functions. 
* Introduce :attr:`~emgfit.spectrum.peak._index` and :attr:`~emgfit.spectrum.peak._blinded` 
  attributes, and :attr:`~emgfit.spectrum.peak._prefix` property to 
  :class:`~emgfit.spectrum.peak` class.
  
Changed
^^^^^^^
* Change initialization of MCMC walkers in
  :meth:`~emgfit.spectrum.spectrum._get_MCMC_par_samples` to sampling from 
  truncated normal distributions to prevent walkers from starting outside the 
  allowed parameter ranges.
* Update functions in the :mod:`~emgfit.emg_funcs` module to return NaN if any 
  `tau` is negative or if the `theta` parameter falls outside the 
  interval [0,1].
* Rename the static :meth:`~emgfit.spectrum.spectrum.bootstrap_spectrum` method 
  to :func:`~emgfit.sample.resample_events` and moved it to the 
  :mod:`~emgfit.sample` module. 
* Ensure a mass calibrant is available when running 
  :meth:`~emgfit.spectrum.spectrum.fit_peaks`.

Fixed
^^^^^
* Fix bug in color coding of uncertainties listed in peak properties table in
  XLSX output files.
* Fix bug in calculation of statistical mass uncertainties of multiply charged
  peaks with :meth:`~emgfit.spectrum.spectrum.get_errors_from_resampling`.
* Fix peak shape error evalution with 
  :meth:`~emgfit.spectrum.spectrum._eval_peakshape_errors` failing due to 
  varied parameters falling outside of bounds or due to precision loss in mu0 
  calculation. 
* Fix bug in preparation of MCMC shape parameters samples in 
  :meth:`~emgfit.spectrum.spectrum.get_MC_peakshape_errors` for cases where 
  the shape calibrant is not the leftmost peak fitted in the shape calibration. 


v0.4.1 (2022-05-31)
-------------------

Added
^^^^^
* Allow for a simple scaling of shape parameters from the shape-calibrant peak
  to peaks of interest through the new `scale_shape_pars`
  :class:`~emgfit.spectrum.spectrum` attribute and `scl_coeff`
  :class:`~emgfit.spectrum.peak` attribute.
* Add the convenience method :meth:`~emgfit.spectrum.spectrum.save_fit_trace`
  for exporting the trace data of a fit to a TXT file.
* Add `method` :class:`~emgfit.spectrum.peak` attribute to store which 
  minimization algorithm was used for a fit.
* Enable flexible adding and modification of a fit model's parameter constraints
  via the new `par_hint_args` option of :meth:`~emgfit.spectrum.spectrum.peakfit`
  and related spectrum methods.

Changed
^^^^^^^
* Define bounds of scale-dependent parameters as multiples of the standard
  deviation of the underlying Gaussian.
* Improve initialization of peak centroids and amplitudes.
* Adapt initialization of uniform-baseline parameter to support larger
  backgrounds.

Fixed
^^^^^
* Fix bug in recognition of tail order for sampling events with
  :func:`~emgfit.sample.simulate_events`.


v0.4.0 (2021-03-09)
-------------------

Added
^^^^^
* Added the newly published AME2020 mass database.
* Added warning for failed convergence whenever best-fit centroids agree with
  initial values within 1e-09 u.

Changed
^^^^^^^
* By default literature mass values are now taken from AME2020 but AME2016 can
  still optionally be accessed.
* Removed the upper bound for the uniform background parameter.
* Cleaned up definition of peak marker heights.

Fixed
^^^^^
* Correct bug in calculation of literature mass values for molecular isomers.
* Fix bug in calculation of literature values for doubly-charged mass
  calibrants.
* Avoid fails of parallelized MCMC sampling with
  :meth:`~emgfit.spectrum.spectrum._get_MCMC_par_samples` due to PicklingError.
  Parallelized sampling can (for now) only be run using all CPU cores.
* Fix arguments of :meth:`~emgfit.spectrum.spectrum.set_lit_values`.


v0.3.7 (2021-02-02)
-------------------

Added
^^^^^
* Add support for multiply charged ions.
* Expand unit tests with fitting accuracy check and validation of literature
  value fetching for molecular and doubly charged species.

Changed
^^^^^^^
* Rearrange columns in peak properties table.
* Update documentation to reflect support of multiply charged ions and fix some
  minor bugs.

Fixed
^^^^^
* Handle :meth:`~emgfit.spectrum.spectrum._eval_peakshapes_errors` failing with
  non-finite residuals in initial point or parameters running out of bounds.
* Fix bug causing :meth:`~emgfit.spectrum.spectrum._eval_MC_peakshape_errors`
  to fail when first peak is mass calibrant.


v0.3.6 (2020-12-17)
-------------------

Added
^^^^^
* Support marking of isomers in `species` labels and enable quick calculation of
  literature mass values for isomers via the new `Ex` and `Ex_error` options of
  :meth:`~emgfit.spectrum.spectrum.assign_species` and
  :meth:`~emgfit.spectrum.spectrum.add_peak`.
* Enable easy manual definition of literature values via new
  :meth:`~emgfit.spectrum.spectrum.set_lit_values` spectrum method.

Changed
^^^^^^^
* Optimize speed of :meth:`~emgfit.spectrum.spectrum.detect_peaks`.
* Updated docs of :meth:`~emgfit.spectrum.spectrum.get_MC_peakshape_errors`.


Fixed
^^^^^
* Resolved bug in `rerun_MCMC_sampling` option of
  :meth:`~emgfit.spectrum.spectrum._eval_MC_peakshape_errors`.
* Fixed bug in calculation of third order eta parameters in peak-shape error
  evaluations for models with 3 positive or 3 negative tails.


v0.3.5 (2020-12-08)
-------------------

Added
^^^^^
* Plotting of subsample of all error bars with the new `error_every` option.

Fixed
^^^^^
* Fixed bug causing `chi-square` fits in
  :meth:`~emgfit.spectrum.spectrum.parametric_bootstrap` method to fail.
* Fixed broken crosslinks and other minor bugs in docs.


v0.3.4 (2020-12-06)
-------------------

Added
^^^^^
* Added optional saving of plot images to PNG files and improved formatting of
  output files of :meth:`~emgfit.spectrum.spectrum.save_results`.

Fixed
^^^^^
* Fix bug in parallelized fits with `chi-square` cost function.


v0.3.3 (2020-12-05)
-------------------

Fixed
^^^^^
* Resolve CPU-parallelized fits failing with PickleErrors in Python 3.7.
* Improve filtering of user warnings, thus avoiding printing of unnecessary
  deprecation warnings.


v0.3.2 (2020-12-04)
-------------------

Fixed
^^^^^
* Resolved some incompatibility issues observed in notebooks for certain ipython
  and ipykernel versions.
* Fixed failing notebook start-up due to pywin32 ImportError.


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
