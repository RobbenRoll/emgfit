Central concepts
================
Note that `emgfit` is tailored to the needs of analyzing high-precision
time-of-flight mass spectra. However, both the available selection of
hyper-exponentially-modified Gaussian (Hyper-EMG) line shapes as well as the
implemented statistical techniques could be used as powerful tools for
analyzing spectroscopic data sets from a variety of fields.


The peak and spectrum classes
-----------------------------
The data analysis approach of `emgfit` is highly object oriented. The central
objects that users interact with are instances of the
:class:`~emgfit.spectrum.spectrum` and :class:`~emgfit.spectrum.peak` classes.
A spectrum object is instantiated by importing a data set. All relevant
information about this data set is stored in attributes of the spectrum object.
Initially, the user adds a number of peak objects to the spectrum which are then
stored as a list in the spectrum's :attr:`~emgfit.spectrum.spectrum.peaks`
attribute. Each peak object holds specific information about a given peak (e.g.
peak position, peak area, ...). A table with all relevant information about the
peaks in the spectrum can be viewed with the
:meth:`~emgfit.spectrum.spectrum.show_peak_properties` spectrum method.

The different peaks are fitted and more and more information is obtained by
progressively calling different methods on the spectrum object. An outline of a
typical analysis with `emgfit` is given in the tutorial. Once a peak has been
fitted the obtained :class:`~lmfit.model.ModelResult` (or "fit result") is
stored at the corresponding position in the spectrum's
:attr:`~emgfit.spectrum.spectrum.fit_results` list. Comprehensive lists of the
available methods and attributes of the spectrum and peak classes are compiled
along with detailed usage information in the :class:`~emgfit.spectrum.spectrum`
and :class:`~emgfit.spectrum.peak` sections of the API docs. The
spectrum and peak classes are tailored to the analysis needs of multi-reflection
time-of-flight mass spectrometry. However, the statistical techniques
incorporated in the spectrum class could be applied to analyses of spectroscopic
data from various other fields. If specific needs emerge other specialized
classes could be derived from the above.


Adding peaks to a spectrum
--------------------------
The easiest way to add peaks to a spectrum is to use `emgfit's` automatic
peak detection method :meth:`~emgfit.spectrum.spectrum.detect_peaks`. This
method applies some smoothing to the spectrum and then detects peaks via minima
in the second derivative of the smoothed data. This approach yields a high
sensitivity in the identification of overlapping or low-intensity peaks.
Increased sensitivity can be achieved by adapting the method's tuning parameters
such as the minimal threshold for peak detection to the specifics of the
given data set - see the method docs for the available options.

Alternatively or additionally, peaks can be added manually with the
:meth:`~emgfit.spectrum.spectrum.add_peak` method. By default, markers of the
associated peaks are added to plots of spectrum data. Peaks can be removed from
a spectrum object using the :meth:`~emgfit.spectrum.spectrum.remove_peaks`
method. **To avoid ambiguities peaks should only be added or removed in the
initial analysis stage, i.e. before the shape calibration or any other fits have
been performed.**


Assigning species to peaks and fetching literature mass values
--------------------------------------------------------------
The following attributes can be used to select a peak:

* Peak index (i.e. index in the :attr:`~emgfit.spectrum.spectrum.peaks` list)
* Peak marker position `x_pos`
* Ionic `species` label (if assigned)

The peak index and `x_pos` are always defined as soon as a peak is added to a
spectrum. The optional `species` attribute can either be set in
:meth:`~emgfit.spectrum.spectrum.add_peak` or with
:meth:`~emgfit.spectrum.spectrum.assign_species`. The `species` labels must
follow the :ref:`:-notation`. As soon as a `species` is assigned to a peak the
corresponding literature mass and its uncertainty are automatically fetched
from the atomic mass evaluation (AME) mass database. When an AME mass value is
not purely based on experimental data the peak's `extrapolated` attribute is set
to `True`.
By default, literature values are grabbed from AME2020_. Optionally, you can
switch to values from AME2016_. To revert to AME2016 for all peaks, set
`default_lit_src='AME2016'` when instantiating the spectrum object. To only use
AME2016 values for certain peaks, use the `lit_src` option when defining the
peak species with the :meth:`~emgfit.spectrum.spectrum.add_peak` or
:meth:`~emgfit.spectrum.spectrum.assign_species`. Whenever literature values are
fetched from AME2016 this is automatically indicated by adding
`'lit_src: AME2016'` to the respective peak `comment`. User-defined literature
values can be specified with the :meth:`~emgfit.spectrum.spectrum.add_peak` and
:meth:`~emgfit.spectrum.spectrum.set_lit_values` methods.

.. _AME2020: https://www-nds.iaea.org/amdc/
.. _AME2016: http://amdc.in2p3.fr/web/masseval.html


Hyper-EMG distributions
-----------------------
A core feature of `emgfit` is its numerically robust implementation of
hyper-exponentially-modified Gaussian (hyper-EMG) probability density functions.
Exponentially-modified Gaussian distributions have been demonstrated to be a
powerful tool for fitting spectroscopic data from various fields including mass
spectrometry [1]_, alpha-particle spectrometry [2]_ and chromatography [3]_.
Hyper-EMG distributions :math:`h_\mathrm{emg}(x)` as introduced in [1]_ are
mixture models that allow the convolution of a Gaussian with an arbitrary number
of left-hand and right-hand exponential tails, respectively:

.. math::

   h_\mathrm{emg}(x; \mu, \sigma, \Theta, \eta_-, \tau_-, \eta_+, \tau_+) =
   \Theta h_\mathrm{-emg}(x; \mu, \sigma, \eta_-, \tau_-) +
   (1-\Theta) h_\mathrm{+emg}(x; \mu, \sigma, \eta_+, \tau_+).

where :math:`0 \leq \Theta \leq 1` is the mixing weight that determines the
relative contribution of the negative and positive skewed EMG distributions,
:math:`h_\mathrm{-emg}` and :math:`h_\mathrm{+emg}`, respectively. The latter
are defined as:

.. math::

   h_\mathrm{-emg}(x; \mu, \sigma, \eta_-, \tau_-)
   = \sum_{i=1}^{N_-}{\frac{\eta_{-i}}{2\tau_{-i}}
     \exp{\left( \left(\frac{\sigma}{\sqrt{2}\tau_{-i}}\right)^2 +
     \frac{x-\mu}{\sqrt{2}\tau_{-i}}\right)}
     \mathrm{erfc}\left(\frac{\sigma}{\sqrt{2}\tau_{-i}} +
     \frac{x-\mu}{\sqrt{2}\sigma}\right)},

   h_\mathrm{+emg}(x; \mu, \sigma, \eta_+, \tau_+)
   = \sum_{i=1}^{N_+}{\frac{\eta_{+i}}{2\tau_{+i}}
     \exp{\left(\left(\frac{\sigma}{\sqrt{2}\tau_{+i}}\right)^2 -
     \frac{x-\mu}{\sqrt{2}\tau_{+i}}\right)}
     \mathrm{erfc}\left(\frac{\sigma}{\sqrt{2}\tau_{+i}} -
     \frac{x-\mu}{\sqrt{2}\sigma}\right)}.

:math:`N_{-}` and :math:`N_{+}` are referred to as the negative and positive tail
order. :math:`\mu=\mu_G` denotes the mean and :math:`\sigma=\sigma_G` the
standard deviation of the underlying Gaussian distribution. The decay constants
of the left- and right-handed exponential tails are given by
:math:`\tau_-=(\tau_{-1},\tau_{-2},...,\tau_{-N_-})`
& :math:`\tau_+=(\tau_{+1},\tau_{+2},...,\tau_{+N_+})`, respectively. The negative
and positive tail weights are denoted by
:math:`\eta_-=(\eta_{-1},\eta_{-2},...,\eta_{-N_-})`
& :math:`\eta_+=(\eta_{+1},\eta_{+2},...,\eta_{+N_+})`, respectively, and obey
the following normalizations:

.. math::

   \sum_{i=1}^{N_-}{\eta_\mathrm{-i}} = 1,

   \sum_{i=1}^{N_+}{\eta_\mathrm{+i}} = 1.

For information on the numerical implementation of hyper-EMG distributions see
:mod:`emgfit.emg_funcs`.

For fits of actual count data, the normalized :math:`h_\mathrm{emg}(x)`
distribution is multiplied by an amplitude parameter (:math:`A` or `amp`) and
optionally a uniform baseline parameter (:math:`c_\mathrm{bkg} or `bkg_c`) is
added:

.. math::

   H_\mathrm{emg} = A \cdot h_\mathrm{emg}(x) + c_\mathrm{bkg}


.. _fit_model_list:

Available fit models
--------------------
All supported (single peak) fit models or peak shapes are defined in the
:mod:`emgfit.fit_models` module. Currently, the following models are available:

* Gaussian:     :func:`~emgfit.fit_models.Gaussian`
* Hyper-EMG(0,1): :func:`~emgfit.fit_models.emg01`
* Hyper-EMG(1,0): :func:`~emgfit.fit_models.emg10`
* Hyper-EMG(1,1): :func:`~emgfit.fit_models.emg11`
* Hyper-EMG(1,2): :func:`~emgfit.fit_models.emg12`
* Hyper-EMG(2,1): :func:`~emgfit.fit_models.emg21`
* Hyper-EMG(2,2): :func:`~emgfit.fit_models.emg22`
* Hyper-EMG(2,3): :func:`~emgfit.fit_models.emg23`
* Hyper-EMG(3,3): :func:`~emgfit.fit_models.emg33`

where the numbers in brackets indicate the negative and positive tail orders,
i.e. the number of exponential tails added to the left and right side of the
peak, respectively. All fit models in `emgfit` are expressed using `lmfit's`
:class:`~lmfit.model.Model` class. This interface is used to define appropriate
parameter bounds and ensure the normalization of the negative and positive tail
weights (`eta_p` and `eta_m` parameters) of Hyper-EMG models. For more details
on the above fit models see the API docs of the :mod:`emgfit.fit_models` module.

Multi-peak fits
---------------
If multiple peaks are to be fitted at once a suitable multi-peak model is
automatically created within the :class:`~emgfit.spectrum.spectrum` class by
adding a suitable number of single-peak models. In multi-peak fits, the values
of the shape (or scale) parameters of all peaks are enforced to be identical,
only the amplitude and position parameters are allowed to differ. In
multi-reflection time-of-flight mass spectrometry the width of peaks acquired
with a given instrumental resolution scales linearly with the peak's centroid
mass. Simultaneously fitting peaks with significantly different mass centroids
therefore requires a mass-dependent rescaling of the shape parameters to the
respective peak's mass. So far analysis practice has shown that the required
scaling corrections for isobaric peaks are significantly smaller than the
typical relative errors of the corresponding shape parameters. Since `emgfit`
(currently) only supports fits of isobaric species such a mass-rescaling has not
been implemented in the package. Support for fitting non-isobaric mass peaks in
the same spectrum might be added in the future.

.. _`peak-fitting approach`:

Peak fitting approach
---------------------
Peak fits with `emgfit` are executed by the internal
:meth:`~emgfit.spectrum.spectrum.peakfit` method which builds on `lmfit's`
:class:`~lmfit.model.Model` interface. However, usually the user only interacts
with higher level methods (e.g. :meth:`~emgfit.spectrum.spectrum.determine_peak_shape`
or :meth:`~emgfit.spectrum.spectrum.fit_peaks`) that internally call
:meth:`~emgfit.spectrum.spectrum.peakfit`. Initial parameter values are
determined as follows:

* The initial value of the peak amplitude (`amp` parameter) is estimated from
  the product of the number of counts in the bin at the peak's marker position
  :attr:`x_pos` and the initial value of the standard deviation of the
  Gaussian peak component (`sigma`). This product is multiplied by an empirical
  proportionality factor. The factor is somewhat peak-shape dependent but has
  been found to work well for a variety of hyper-EMG peaks. If user intervention
  still becomes necessary, the `par_hint_args` option of the :meth:`peakfit`
  method can be used to overwrite the initial value of the peak amplitude.
* In the case of a Gaussian, the peak centroid is initialized at the peak marker
  position :attr:`x_pos`. For a hyper-EMG fit, the initial centroid of the
  underlying Gaussian (denoted `mu` or :math:`\mu`) is calculated by rearranging
  the equation for the mode (i.e. the x-position with maximum probability):

  .. math::

     \mu = x_{m}
             - \theta\sum_{i=1}^{N_-}\eta_{-i}\left(\sqrt{2}\sigma
               \cdot\mathrm{erfcxinv}\left( \frac{\tau_{-i}}{\sigma}
               \sqrt{\frac{2}{\pi}}\right) - \frac{\sigma^2}{\tau_{-i}}
               \right) \\
             + (1-\theta)\sum_{i=1}^{N_-}\eta_{+i}\left(\sqrt{2}\sigma
               \cdot\mathrm{erfcxinv}\left( \frac{\tau_{+i}}{\sigma}
               \sqrt{\frac{2}{\pi}}\right) - \frac{\sigma^2}{\tau_{-i}}
               \right),

  where the mode :math:`x_{m}` is estimated by the peak marker position
  :attr:`x_pos` and :math:`{N_-}` and :math:`{N_+}` denote the order of negative
  and positive exponential tails, respectively.
* If the shape parameters have not already been determined in a preceding
  peak-shape calibration there is two possibilities for their initialization.
  By default, a set of suitable initial values is then derived by re-scaling the
  shape parameters for a representative peak at mass unit 100 to the mass of the
  given spectrum. The default parameters at mass 100 u are defined in the
  :func:`emgfit.fit_models.create_default_init_pars` function. Alternatively,
  the shape parameter values can be user-defined with the `init_pars` argument.
* If fitting of a uniform background has been activated with the `vary_baseline`
  argument, the baseline amplitude parameter :attr:`bkg_c` is initialized at the
  minimal number of counts found in any bin + 0.1 counts. The slight offset of
  0.1 counts circumvents possible convergence problems due to the parameter
  starting right on its lower bound of zero counts.

Fits are performed by minimizing either of the following cost functions:

* `chi-square`: This variance weighted cost function is commonly known as
  `Pearson's chi squared statistic` and defined as:

  .. math::

     \chi^2_P = \sum_i \frac{(f(x_i) - y_i)^2}{f(x_i)+\epsilon},

  where :math:`x_i` and :math:`y_i` denote the center and contained counts of
  the i-th bin, respectively. On each iteration the variances of the residuals
  are estimated using the latest model predictions:
  :math:`\sigma_i^2 \approx f(x_i)`. The inclusion of the small constant
  :math:`\epsilon = 1e-10` ensures numerical robustness as :math:`f(x_i)`
  approaches zero and only causes a negligibly small bias in the parameter
  estimates. The iteratively re-calculated weights result in improved behavior
  in low-count situations.

* `MLE`: With this cost function a binned maximum likelihood estimation is
  performed by minimizing the (doubled) negative log-likelihood ratio, also
  known as `Cash-statistic` [4]_:

  .. math::

     L = 2\sum_i \left[ f(x_i) - y_i + y_i \ln{\left(\frac{y_i}{f(x_i)}\right)} \right].

  The assumption that the counts in each bin follow a Poisson (instead of a
  normal) distribution makes this method applicable to count data with very
  low statistics. When a non-scalar minimization algorithm is used (e.g.
  `least_squares`) the above optimization problem is rephrased into a
  least-squares problem by minimizing the square roots of the (positive
  semidefinite) summands in the above equation. See the notes section of the
  docs of :meth:`~emgfit.spectrum.spectrum.peakfit` for more details.

A number of different optimization algorithms are available to perform the
minimization. In principle, any of the algorithms listed under `lmfit's`
`fitting methods`_ can be used by passing the respective method name to the
`method` option if `emgfit's` fitting routines. By default, the `least_squares`
minimizer is used.

.. _`fitting methods`: https://lmfit.github.io/lmfit-py/fitting.html#choosing-different-fitting-methods


.. _`peak-shape calibration`:

Peak-shape calibration
----------------------
The peak-shape calibration is performed with the
:meth:`~emgfit.spectrum.spectrum.determine_peak_shape` method and offers a way
to reduce the number of parameters varied in the peak-of-interest fit(s). This
not only increases the robustness and computational speed of multi-peak fits but
can also enhance the sensitivity for detecting unidentified overlapping peaks.

In the peak-shape calibration an ideally well separated, high-statistics peak is
fitted to obtain a suitable peak shape to describe the data. We refer to all
parameters that determine the shape of a single peak in the absence of background
as *shape parameters*. In the case of a Gaussian peak model the only shape
parameter is given by the standard deviation :math:`\sigma`. The **shape
parameters of a hyper-EMG model function**
are given by:

* the standard deviation :math:`\sigma` of the underlying Gaussian,
* the left-right mixture weight :math:`\Theta`,
* the weights for the positive and negative exponential tails, :math:`\eta_{-i}`
  & :math:`\eta_{+i}`, respectively,
* and their corresponding decay constants :math:`\tau_{-i}` & :math:`\tau_{+i}`,
  respectively,

where i = 1, 2, 3, ... indicates the tail order. `emgfit` assumes
that all peaks in a spectrum have been acquired with a fixed instrumental
resolution and exhibit the same theoretical peak shape. In multi-reflection
time-of-flight mass spectrometry this assumption is not strictly satisfied since
at a given resolving power the peak widths exhibit a linear scaling with mass.
However, since `emgfit` is currently only intended for isobaric peaks the
required scale corrections of shape parameters are usually only on the
sub-percent level and hence negligible compared to the typical uncertainties in
determining these parameters in the shape calibration fit. Therefore, an
**identical peak shape is enforced for all simultaneously fitted peaks**. A
mass-dependent re-scaling of the scale parameters might be added in the future.

Before the peak-shape calibration the user must decide which of the
:ref:`fit_model_list` best describes the data. To aid in this process the
:meth:`~emgfit.spectrum.spectrum.determine_peak_shape` method comes with an
**automatic model selection** feature. Therein, `chi-square` fits with
increasingly complicated model functions are performed on the shape calibration
peak, starting from a regular Gaussian up to Hyper-EMG functions of successively
increasing tail order. To avoid overfitting, models with any best-fit shape
parameters agreeing with zero within 1:math:`\sigma` confidence are excluded
from selection. Amongst the remaining models, the one yielding the lowest
chi-square per degree of freedom is selected. Alternatively, this feature can be
skipped by setting the `vary_tail_order` option to `False` and a peak shape can
be defined manually with the `fit_model` option of
:meth:`~emgfit.spectrum.spectrum.determine_peak_shape`.

Once a peak-shape calibration has been established, all subsequent fits will,
by default, be performed with this fixed peak-shape, only varying the peak
amplitudes, peak positions and (if applicable) the amplitude of the uniform
background. If fits with a varying peak shape are desired the `vary_shape`
option of the :meth:`~emgfit.spectrum.spectrum.peakfit` method must be set to
`True`. The imperfect knowledge of the exact peak shape can be associated with
an additional uncertainty in the determination of the peak's mass centroid and
peak area. To include these contributions in the uncertainty budget, `emgfit`
provides specialized methods to quantify the `Peak-shape uncertainties`_.


.. _recalibration:

Mass recalibration and calculation of final mass values
--------------------------------------------------------
Before being imported into `emgfit` mass spectra must have undergone a
preliminary mass calibration. This initial mass scale will persist
throughout the entire analysis process and will be used as the x-axis for
all plots of spectrum data. In multi-reflection time-of-flight mass spectrometry
the initial mass scale is usually established using the following calibration
equation [5]_:

.. math::

   \frac{m}{z} = c \frac{(t-t_0)^2}{(1+Nb)^2},

where :math:`\frac{m}{z}` denotes the mass-to-charge ratio of the ion, t is
the measured time of flight of the ion :math:`t_0` marks a small time offset due
to electronic delays and N is the number of revolutions the ion has undergone.
Since N is easy to infer, the factors c and b and the time offset :math:`t_0`
remain as the calibration constants to be determined.

There is a number of ways to determine the above calibration constants. To
ensure high precision in the final mass values a second mass calibration - the
so-called `mass re-calibration` - must be performed in `emgfit`. This removes
any systematics that could arise when different procedures are used to determine
the calibrant peak position in the initial calibration and the positions of
peaks of interest in the final fitting [5]_. Further, it renders the specific
choice of the peak position parameter irrelevant as long as the same convention
is followed for all peaks. In fact, instead of using the mean or mode of the
full hyper-EMG distribution (:math:`\mu_\mathrm{emg}`) `emgfit` uses the mean of
the underlying Gaussian (:math:`\mu`) to establish peak positions.

In the mass recalibration a calibrant peak with a well-known (ionic) literature
mass :math:`m_{cal, lit}` is fitted and the obtained peak position
:math:`(m/z)_{cal, fit}` is used to calculate the spectrum's mass recalibration
factor defined as:

.. math::

   \gamma_\mathrm{recal} = \frac{(m/z)_\mathrm{cal,lit}}{(m/z)_\mathrm{cal,fit}}
                         = \frac{m_\mathrm{cal,lit}}{m_\mathrm{cal,fit}},

The calibrant peak can either be fitted individually upfront via the
:meth:`~emgfit.spectrum.spectrum.fit_calibrant` method or the calibrant fit can
be performed simultaneous with the ion-of-interest fits using the
`index_mass_calib` or `species_mass_calib` options of the
:meth:`~emgfit.spectrum.spectrum.fit_peaks` method. The relative uncertainty of
the recalibration factor ("recalibration uncertainty") is given by the
literature mass uncertainty :math:`\Delta m_\mathrm{cal, lit}` and the
statistical uncertainty of the calibrant fit result
:math:`\Delta m_\mathrm{cal, fit}`:

.. math::

   \frac{\Delta \gamma_\mathrm{recal}}{\gamma_\mathrm{recal}} =
     \sqrt{ \left(\frac{\Delta m_\mathrm{cal, lit}}{m_\mathrm{cal, lit}} \right)^2
          + \left(\frac{\Delta m_\mathrm{cal, fit}}{m_\mathrm{cal, fit}} \right)^2}.

The final ionic masses :attr:`m_ion` are calculated as:

.. math::

   m_\mathrm{ion} = \frac{(m/z)_\mathrm{cal, lit}}{(m/z)_\mathrm{cal, fit}}
                    \cdot (m/z)_\mathrm{fit} \cdot z
                  = \gamma_\mathrm{recal} \cdot (m/z)_\mathrm{fit} \cdot z.

The relative uncertainty of the final mass values is given by adding the
statistical mass uncertainty, the recalibration uncertainty and the peak-shape
mass uncertainty in quadrature:

.. math::

   \frac{\Delta m_\mathrm{ion}}{m_\mathrm{ion}} =
          \sqrt{ \left(\left(\frac{\Delta m}{m}\right)_\mathrm{stat} \right)^2
          + \left(\frac{\Delta \gamma_\mathrm{recal}}{\gamma_\mathrm{recal}} \right)^2
          + \left( \left(\frac{\Delta m}{m}\right)_\mathrm{PS} \right)^2 }.

Note that in the above, :math:`m` refers to ionic rather than atomic masses.
The atomic mass excess (:attr:`atomic_ME_keV` peak attribute) and its
uncertainty are calculated from :math:`m_\mathrm{ion}` from the following
relations:

.. math::

  \mathrm{ME}= m_\mathrm{ion} + z\cdot m_e  - A \cdot u

  \Delta\mathrm{ME} = \mathrm{ME} \cdot
                      \frac{\Delta m_\mathrm{ion}}{m_\mathrm{ion}},

where A denotes the atomic mass number. Note that the above neglects the atomic
binding energy of the stripped electrons, as well as the uncertainties of the
electron mass and the atomic mass unit :math:`u`. Assuming singly or doubly
charged ions, these contributions lie well below 1 keV.


Fitting peaks of interest
-------------------------
Peaks of interest are fitted with the :meth:`~emgfit.spectrum.spectrum.fit_peaks`
method of the spectrum class. By default, this method fits all defined peaks in
the spectrum. Alternatively, a specific mass range or specific neighboring peaks
to fit can be selected. It is the user's choice whether all peaks are treated at
once or whether :meth:`~emgfit.spectrum.spectrum.fit_peaks` is run multiple
times on single peaks or subgroups of peaks.


Estimation of statistical uncertainties
---------------------------------------
With `emgfit` the statistical uncertainties of peak centroids can be estimated
in two different ways:

1. The default approach exploits the scaling of the statistical uncertainty of
   the mean of a Gaussian or hyper-EMG distribution with the number of counts in
   the peak :math:`N_\mathrm{counts}`:

   .. math::

      \Delta \left(m/z\right)_\mathrm{stat} = A_\mathrm{stat} \frac{\mathrm{FWHM}}{\sqrt{N_\mathrm{counts}}}.

   In the case of a Gaussian :math:`A_\mathrm{stat}` is simply given by
   :math:`A_\mathrm{stat,G} = 1/(2\sqrt{2\ln{2}}) = 0.425`. For hyper-EMG
   distributions the respective constant of proportionality :math:`A_\mathrm{stat,emg}`
   is typically larger and depends on the specific peak shape [5]_. `emgfit's`
   :meth:`~emgfit.spectrum.spectrum.determine_A_stat_emg` method can be used to
   estimate :math:`A_\mathrm{stat,emg}` for a specific peak shape via
   non-parametric bootstrapping of a reference peak with decent statistics (see
   method docs for details). The updated :math:`A_\mathrm{stat,emg}` factor will
   be used in subsequent fits to calculate the stat. mass errors with the above
   equation. If :meth:`~emgfit.spectrum.spectrum.determine_A_stat_emg` is not
   run a default value of :math:`A_\mathrm{stat,emg} = 0.52` [5]_ is used.

2. Alternatively, the statistical uncertainty can be estimated after the peak
   fitting with the :meth:`~emgfit.spectrum.spectrum.get_errors_from_resampling`
   method. This routine follows the approach outlined in [5]_ and does not use a
   reference peak but determines the statistical mass uncertainty for each peak
   of interest individually. This is done by re-performing the fit on many
   synthetic spectra obtained by resampling from the best-fit model curve
   (`parametric bootstrap`). Assuming that the data is well-described by the
   chosen fit model  this method yields refined estimates of the statistical
   uncertainties that account for departures from the simple scaling law above
   (as possible e.g. for strongly overlapping peaks).

Since the computational overhead of the second approach is usually rather
small this method is oftentimes preferable. Note that the second method also
yields estimates of the statistical uncertainty of the peak areas, whereas the
first approach only yields stat. mass errors and requires the area errors to be
independently estimated from the covariance matrix provided by lmfit (which can
be problematic for `MLE` fits).


.. _`Peak-shape uncertainties`:

Peak-shape uncertainties
------------------------
Peak-shape uncertainties quantify the effect of shape parameter uncertainties
obtained in a preceding peak-shape calibration on the final mass values and peak
areas obtained in ion-of-interest fits.

When ion-of-interest fits are performed with a fixed peak-shape, the
uncertainties of shape parameters obtained in the peak-shape calibration can
cause additional uncertainties in the final mass and peak area values.
Consequently, these so-called `peak-shape uncertainties` must be carefully
estimated and propagated into the final mass and area uncertainties. `emgfit`
provides two ways to estimate the peak-shape uncertainties of the
peak areas and the mass values `m_ion`:

  1. A quick peak-shape (PS) estimation is automatically performed in fits with
  :meth:`~emgfit.spectrum.spectrum.fit_peaks` and
  :meth:`~emgfit.spectrum.spectrum.fit_calibrant` by calling the internal
  :meth:`~emgfit.spectrum.spectrum._eval_peakshape_errors` method. This routine
  adapts the approach of [5]_ and re-performs a given fit a number of times,
  each time changing a different shape parameter by plus and minus its
  1:math:`\sigma` confidence interval, respectively, while keeping all other
  shape parameters fixed. For each shape parameter, the larger of the two shifts
  in a peak's mass and area is recorded and the peak-shape uncertainty is
  estimated for each peak by summing those values in quadrature. Mind that the
  considered mass shifts are corrected for the respective shifts of the
  calibrant peak position.

  2. The above approach implicitly assumes that the shape parameters follow
  normal posterior distributions and neglects any correlations between shape
  parameters. Since these assumptions are oftentimes violated, refined PS error
  estimates can be obtained with `emgfit's`
  :meth:`~emgfit.spectrum.spectrum.get_MC_peakshape_errors` method. This
  re-performs a given fit many times with a variety of different peak-shapes.
  For the used peak shapes to be representative of all line shapes supported by
  the data the full shape parameter posterior distributions are upfront
  estimated by Markov-Chain Monte Carlo (MCMC) sampling. Assuming a sufficiently
  large subset of these MCMC parameter sets is used to refit the data, the
  resulting PS errors account for non-normal parameter distributions (typically
  found when a parameter is near its bounds) and parameter correlations. Since
  this approach is computationally expensive it makes heavy use of parallel
  processing. If appropriate MCMC sampling has already been performed in the
  peak-shape calibration (with the `map_par_covar` option) those samples will be
  re-used in the Monte Carlo PS uncertainty estimation. If
  :meth:`~emgfit.spectrum.spectrum.get_MC_peakshape_errors` is run the peak
  properties table is updated with the refined uncertainties and the new
  values are marked in color to clearly indicate the way they were estimated.

Saving fit traces
-----------------
Obtained fit curves and the underlying count data can be exported to a TXT file
using the :meth:`~emgfit.spectrum.spectrum.save_fit_trace` method of the
:class:`spectrum` class.

Saving fit results
------------------
All critical results obtained in the analysis of a spectrum can be saved with
the :meth:`~emgfit.spectrum.spectrum.save_results` spectrum method. This routine
saves the analysis results to an XLSX-file with three worksheets containing:

1. General properties of the spectrum, such as the input filename, the fit model
   used in the peak-shape calibration and the obtained mass recalibration
   factor. For details on what the respective parameters refer to see the
   attribute list of the :class:`~emgfit.spectrum.spectrum` class.
2. The peak properties table with the attributes of all peaks as well as
   images of all best-fit curves. Check the attribute list of the
   :class:`~emgfit.spectrum.peak` class for short descriptions of what the
   different columns contain.
3. The :attr:`eff_mass_shifts` dictionary holding for each peak the larger of
   the two effective mass shifts obtained when varying each shape parameter by
   +-1:math:`\sigma` in the default peak-shape error estimation. These shifts
   are irrelevant for peaks whose peak-shape uncertainties have been estimated
   with the :meth:`~emgfit.spectrum.spectrum.get_MC_peakshape_errors` routine.

Additionally, the spectrum's peak-shape calibration parameters and their
uncertainties are saved to a separate TXT-file and plots with the obtained fit
curves are saved to PNG-files (optional).

.. _:-notation:

:-notation of chemical substances
---------------------------------

`emgfit` follows the :-notation of chemical compounds. The chemical composition
of an ion is denoted as a single string in which the constituting isotopes are
separated by a colon (``:``). Each isotope is denoted as ``'n(El)A'`` where `El`
is the corresponding element symbol, `n` denotes the number of atoms of the
given isotope and `A` is the respective atomic mass number. In the case
`n = 1`, the number indication `n` can be omitted. The charge state of the ion
is indicated by subtracting the desired number of electrons from the atomic
species (i.e. ``':-1e'`` for singly charged cations, ``':-2e'`` for doubly
charged cations etc.). Once the ionic species of a peak is assigned `emgfit`
automatically fetches the respective literature value from the AME2020_ [6]_ (or
the AME2016_ [7]_) mass database. The subtraction of the electron is important
since otherwise the atomic instead of the ionic mass is used for subsequent
calculations. The calculated literature mass values do not account for electron
binding energies which can in most applications safely be neglected for singly
and doubly charged ions. `emgfit` does currently not interface with an isomer
database. However, isomers can be marked by appending an ``'m'`` or ``'m0'`` up
to ``'m9'`` to the end of an isotope substring (see last example below). The
literature mass (and mass error) of an isomer are automatically calculated from
the respective ground-state AME mass when the excitation energy is passed to the
`Ex` (and `Ex_error`) option of the relevant spectrum methods (e.g.
:meth:`~emgfit.spectrum.spectrum.add_peak` &
:meth:`~emgfit.spectrum.spectrum.assign_species`). Further, literature values
can be fully user-defined with the
:meth:`~emgfit.spectrum.spectrum.set_lit_values` & method.

Examples:

- The most abundant isotope of the hydronium cation :math:`H_{3}O^{+}` can be
  denoted as ``'3H1:1O16:-1e'`` or ``'3H1:O16:-e'`` or ``'1O16:3H1:-1e'`` or ...
- The most abundant isotope of the ammonium cation :math:`N H_{4}^{+}` can be
  denoted as ``'4H1:1N14:-1e'`` or ``'4H1:N14:-e'`` or ``'N14:4H1:-1e'`` or ...
- The proton is denoted as ``'1H1:-1e'`` or ``'H1:-1e'`` or ``'H1:-e'``.
- A Indium-127 ion in the second isomeric state is denoted as ``'1In127m1:-1e'``


Creating simulated spectra
--------------------------
The functions in the :mod:`emgfit.sample` module allow the fast creation of
synthetic spectrum data by extending random variate sampling with `Scipy's`
:class:`~scipy.stats._continuous_distns.exponnorm` class to hyper-EMG
distributions. The functions in this module can serve as a valuable tool for
Monte Carlo studies with count data.


Blind analysis
--------------
Premature comparison of fit results to literature values can lead to biased
results. To avoid that user bias (consciously or unconsciously) enters the final
mass values `emgfit` incorporates the option to blind the obtained mass values
and peak positions during the analysis process. Blindfolding is activated with
the :meth:`~emgfit.spectrum.spectrum.set_blinded_peaks` method. The option to
only blind specific peaks of interest leaves the possibility to use less
interesting peaks with well-known literature masses as accuracy checks. The
blinding is automatically lifted once the processing of the spectrum is
finalized and the results are exported.


.. _AME2016: http://amdc.in2p3.fr/web/masseval.html

References
----------
.. [1] Purushothaman, S., et al. "Hyper-EMG: A new probability distribution
   function composed of Exponentially Modified Gaussian distributions to analyze
   asymmetric peak shapes in high-resolution time-of-flight mass spectrometry."
   International Journal of Mass Spectrometry 421 (2017): 245-254.

.. [2] Pommé, S., and B. Caro Marroyo. "Improved peak shape fitting in alpha
   spectra." Applied Radiation and Isotopes 96 (2015): 148-153.

.. [3] Naish, Pamela J., and S. Hartwell. "Exponentially modified Gaussian
   functions — a good model for chromatographic peaks in isocratic HPLC?."
   Chromatographia 26.1 (1988): 285-296.

.. [4] Cash, Webster. "Parameter estimation in astronomy through application of
   the likelihood ratio." The Astrophysical Journal 228 (1979): 939-947.

.. [5] San Andrés, Samuel Ayet, et al. "High-resolution, accurate
  multiple-reflection time-of-flight mass spectrometry for short-lived,
  exotic nuclei of a few events in their ground and low-lying isomeric
  states." Physical Review C 99.6 (2019): 064313.

.. [6] Wang, M., et al. "The AME2020 atomic mass evaluation (II). Tables, graphs
   and references." Chinese Physics C 45 (2021): 030003.

.. [7] Wang, M., et al. "The AME2016 atomic mass evaluation (II). Tables, graphs
   and references." Chinese Physics C 41.3 (2017): 030003.
