################################################################################
##### Python module for creating simulated time-of-flight mass spectra with
##### Gaussian and hyper-exponentially-modified Gaussian lines shapes
##### Author: Stefan Paul

import numpy as np
import pandas as pd
import warnings
from scipy.stats import exponnorm, uniform, norm

################################################################################
##### Define functions for drawing random variates from Gaussian and hyper-EMG
##### PDFs
norm_precision = 1e-06 # required precision for normalization of eta parameters


def Gaussian_rvs(mu, sigma , N_samples=None):
    """Draw random samples from a Gaussian probability density function

    Parameters
    ----------
    mu : float
        Nominal position of simulated peak (mean of Gaussian).
    sigma : float
        Nominal standard deviation of the simulated Gaussian peak.
    N_samples : int, optional, default: 1
        Number of random events to sample.

    Returns
    -------
    :class:`numpy.ndarray` of floats
        Array with simulated events.

    """
    rvs = norm.rvs(loc=mu, scale=sigma, size=N_samples)
    return rvs


def _h_m_i_rvs(mu, sigma, tau_m, N_i):
    """Helper function for definition of h_m_emg_rvs """
    rvs = mu - exponnorm.rvs(loc=0,scale=sigma,K=tau_m/sigma,size=N_i)
    return rvs


def h_m_emg_rvs(mu, sigma, *t_args,N_samples=None):
    """Draw random samples from negative skewed hyper-EMG probability density

    Parameters
    ----------
    mu : float
        Nominal position of simulated peak (mean of underlying Gaussian).
    sigma : float
        Nominal standard deviation (of the underlying Gaussian) of the simulated
        hyper-EMG peak.
    theta : float
        Mixing weight of pos. & neg. skewed EMG distributions.
    t_args : list of lists of float
        List containing lists of the EMG tail parameters with the signature:
        [[eta_m1, eta_m2, ...], [tau_m1, tau_m2, ...]]
    N_samples : int, optional, default: 1
        Number of random events to sample.

    Returns
    -------
    :class:`numpy.ndarray` of floats
        Array with simulated events.

    """
    if not isinstance(N_samples,int):
        raise TypeError("N_samples must be of type int")
    li_eta_m = t_args[0]
    li_tau_m = t_args[1]
    t_order_m = len(li_eta_m) # order of negative tail exponentials
    if abs(sum(li_eta_m) - 1) > norm_precision:
        raise Exception("eta_m's don't add up to 1.")
    if len(li_tau_m) != t_order_m:  # check if all arguments match tail order
        raise Exception("orders of eta_m and tau_m do not match!")

    # randomly distribute ions between tails according to eta_m weights
    tail_nos = np.random.choice(range(t_order_m),size=N_samples,p = li_eta_m)
    rvs = np.array([])
    for i in range(t_order_m):
        N_i = np.count_nonzero(tail_nos == i)
        tau_m = li_tau_m[i]
        rvs_i = _h_m_i_rvs(mu,sigma,tau_m,N_i)
        rvs = np.append(rvs,rvs_i)
    return rvs


def _h_p_i_rvs(mu, sigma, tau_p, N_i):
    """Helper function for definition of h_p_emg_rvs """
    rvs = exponnorm.rvs(loc=mu,scale=sigma,K=tau_p/sigma,size=N_i)
    return rvs


def h_p_emg_rvs(mu, sigma, *t_args, N_samples=None):
    """Draw random samples from pos. skewed hyper-EMG probability density

    Parameters
    ----------
    mu : float
        Nominal position of simulated peak (mean of underlying Gaussian).
    sigma : float
        Nominal standard deviation (of the underlying Gaussian) of the simulated
        hyper-EMG peak.
    t_args : list of lists of float
        List containing lists of the EMG tail parameters with the signature:
        [[eta_p1, eta_p2, ...], [tau_p1, tau_p2, ...]]
    N_samples : int, optional, default: 1
        Number of random events to sample.

    Returns
    -------
    :class:`numpy.ndarray` of floats
        Array with simulated events.

    """
    if not isinstance(N_samples,int):
        raise TypeError("N_samples must be of type int")
    li_eta_p = t_args[0]
    li_tau_p = t_args[1]
    t_order_p = len(li_eta_p) # order of negative tail exponentials
    if abs(sum(li_eta_p) - 1) > norm_precision:
        raise Exception("eta_p's don't add up to 1.")
    if len(li_tau_p) != t_order_p:  # check if all arguments match tail order
        raise Exception("orders of eta_p and tau_p do not match!")

    # randomly distribute ions between tails according to eta_p weights
    tail_nos = np.random.choice(range(t_order_p),size=N_samples,p = li_eta_p)
    rvs = np.array([])
    for i in range(t_order_p):
        N_i = np.count_nonzero(tail_nos == i)
        tau_p = li_tau_p[i]
        rvs_i = _h_p_i_rvs(mu,sigma,tau_p,N_i)
        rvs = np.append(rvs,rvs_i)
    return rvs


def h_emg_rvs(mu, sigma , theta, *t_args, N_samples=None):
    """Draw random samples from a hyper-EMG probability density function

    Parameters
    ----------
    mu : float
        Nominal position of simulated peak (mean of underlying Gaussian).
    sigma : float
        Nominal standard deviation (of the underlying Gaussian) of the simulated
        hyper-EMG peak.
    theta : float
        Mixing weight of pos. & neg. skewed EMG distributions.
    t_args : list of lists of float
        List containing lists of the EMG tail parameters with the signature:
        [[eta_m1, eta_m2, ...], [tau_m1, tau_m2, ...], [eta_p1, eta_p2, ...],
        [tau_p1, tau_p2, ...]]
    N_samples : int, optional, default: 1
        Number of random events to sample.

    Returns
    -------
    :class:`numpy.ndarray` of floats
        Array with simulated events.

    """
    if not isinstance(N_samples,int):
        raise TypeError("N_samples must be of type int")
    li_eta_m = t_args[0]
    li_tau_m = t_args[1]
    li_eta_p = t_args[2]
    li_tau_p = t_args[3]
    if theta == 1:
        rvs = h_m_emg_rvs(mu, sigma, li_eta_m, li_tau_m, N_samples=N_samples)
    elif theta == 0:
        rvs = h_p_emg_rvs(mu, sigma, li_eta_p, li_tau_p, N_samples=N_samples)
    else:
        # randomly distribute ions between h_m_emg and h_p_emg according to
        # left-right-weight theta:
        neg = np.random.choice([1,0],size=N_samples,p = [theta,1-theta])
        N_m = int(np.sum(neg)) #int(np.round(theta*N_samples)) # np.rint(theta*N_samples,dtype=int)
        N_p = N_samples - N_m # np.rint((1-theta)*N_samples,dtype=int)
        rvs_m = h_m_emg_rvs(mu, sigma, li_eta_m, li_tau_m, N_samples=N_m)
        rvs_p = h_p_emg_rvs(mu, sigma, li_eta_p, li_tau_p, N_samples=N_p)
        rvs = np.append(rvs_m,rvs_p)
    return rvs


################################################################################
##### Define functions for simulating events or spectra through random sampling
##### from a reference distribution

def simulate_events(shape_pars, mus, amps, bkg_c, N_events, x_min, x_max,
                    out='hist', scl_facs=None, N_bins=None, bin_cens=None):
    """Create simulated detector events drawn from a user-defined probability
    density function (PDF)

    Events can either be output as a list of single events (mass stamps) or as a
    histogram. In histogram output mode, uniform binning is easily realized by
    specifying the `N_bins` argument. More control over the binning can be
    achieved by parsing the desired bin centers to the `bin_cens` argument (e.g.
    for non-uniform binning).

    Parameters
    ----------
    shape_pars : dict
        Peak-shape parameters to use for sampling. The dictionary must follow
        the structure of the :attr:`~spectrum.shape_cal_pars` attribute of the
        :class:`~emgfit.spectrum.spectrum` class.
    mus : float or list of float
        Nominal peak positions of peaks in simulated spectrum.
    amps : float or list of float [(counts in peak)*(bin width in u)]
        Nominal amplitudes of peaks in simulated spectrum.
    bkg_c : float [counts per bin], optional, default: 0.0
        Nominal amplitude of uniform background in simulated spectrum.
    x_min : float
        Beginning of sampling x-range.
    x_max : float
        End of sampling x-range.
    scl_facs : float or list of float, optional
        Scale factors to use for scaling the scale-dependent shape parameters in
        `shape_pars` to a given peak before sampling events. If `None`, no
        shape-parameter scaling is applied.
    N_events : int, optional, default: 1000
        Total number of events to simulate (signal and background events).
    out : str, optional
        Output format of sampled data. Options:

        - ``'hist'`` for binned mass spectrum (default). The centres of the mass
          bins must be specified with the `bin_cens` argument.
        - ``'array'`` for unbinned array of single ion and background events.

    N_bins : int, optional
        Number of uniform bins to use in ``'hist'`` output mode. The **outer**
        edges of the first and last bin are fixed to the start and end of the
        sampling range respectively (i.e. `x_min` and `x_max`). In between, bins
        are distributed with a fixed spacing of (`x_max`-`x_min`)/`N_bins`.
    bin_cens : :class:`numpy.ndarray`
        Centres of bins to use in ``'hist'`` output mode. This argument
        allows the realization of non-uniform binning. Bin edges are centred
        between neighboring bins. Note: Bins outside the sampling range defined
        with `x_min` and `x_max` will be empty.

    Returns
    -------
    :class:`pandas.Dataframe` or :class:`numpy.ndarray`
       If out='hist' a dataframe with a histogram of the format
       [bin centre, counts in bin] is returned. If out='array' an unbinned
       array with the x-values of single ion or background events is returned.

    Notes
    -----
    Random events are created via custom hyper-EMG extensions of Scipy's
    :meth:`scipy.stats.exponnorm.rvs` method.

    Currently, all simulated peaks have identical width and shape (no re-scaling
    of mass-dependent shape parameters to a peak's mass centroid).

    **Mind the different units for peak amplitudes `amps`
    (<counts in peak> * <bin width in x-axis units>) and the background level
    `bkg_c` (counts per bin).** When spectrum data is simulated counts are
    distributed between the different peaks and the background with probability
    weights `amps` / <bin width in u> and `bkg_c` * <number of bins>,
    respectively. As a consequence, simply changing `N_events` (while keeping
    all other arguments constant), will cause `amps` and `bkg_c` to deviate from
    their nominal units.

    """
    mus = np.atleast_1d(mus)
    amps = np.atleast_1d(amps)
    if len(mus) != len(amps):
        raise Exception("Lengths of `mus` and `amps` arrays must match.")
    if type(N_events) != int:
        raise Exception("`N_events` must be of type int.")
    if (mus < x_min).any() or (mus > x_max).any():
        msg = str("At least one peak centroid in `mus` is outside the sampling range.")
        warnings.warn(msg, UserWarning)
    if scl_facs is None:
        scl_facs = np.ones_like(mus)
    else:
        scl_facs = np.atleast_1d(scl_facs)
        if len(mus) != len(scl_facs):
            raise Exception("Lengths of `mus` and `scl_facs` must match.")

    sample_range = x_max - x_min

    # Get bin parameters
    if N_bins is not None and bin_cens is not None:
        msg =  "Either specify the `N_bins` OR the `bin_cens` argument."
        raise Exception(msg)
    elif bin_cens is not None: # user-defined bins
        N_bins = len(bin_cens)
        bin_edges = np.empty((N_bins+1,))
        spacings = bin_cens[1:] - bin_cens[:-1]
        inner_edges = bin_cens[:-1] + spacings/2
        bin_edges[1:-1] = inner_edges # set inner bin edges
        # Get outer edges
        width_start = bin_cens[1] - bin_cens[0]
        bin_edges[0] = bin_cens[0] - width_start/2 # set first bin edge
        width_end = bin_cens[-1] - bin_cens[-2]
        bin_edges[-1] = bin_cens[-1] + width_end/2 # set last bin edge
        bin_width = (bin_edges[-1] - bin_edges[0])/N_bins # AVERAGE bin width
    elif N_bins is not None: # automatic uniform binning
        bin_edges = np.linspace(x_min, x_max, num=N_bins+1, endpoint=True)
        bin_width = sample_range/N_bins
        bin_cens = bin_edges[:-1] + bin_width/2
    elif out == "array":
        bin_width = 1.
        N_bins = 1.
        pass
    else:
        raise Exception("`N_bins` or `bin_cens` argument must be specified!")

    # Prepare shape parameters
    sigma = shape_pars['sigma']
    li_eta_m = []
    li_tau_m = []
    li_eta_p = []
    li_tau_p = []
    for key, val in shape_pars.items():
        if key.startswith('eta_m'):
            li_eta_m.append(val)
        if key.startswith('tau_m'):
            li_tau_m.append(val)
        if key.startswith('eta_p'):
            li_eta_p.append(val)
        if key.startswith('tau_p'):
            li_tau_p.append(val)
    if len(li_tau_m) == 0 and len(li_tau_p) == 0: # Gaussian
        theta = -1 # flag for below
    else:
        try:
            theta = shape_pars['theta']
        except KeyError:
            pass
        if len(li_eta_m) == 0 and len(li_tau_m) == 1: # emg1X
            li_eta_m = [1]
        elif len(li_eta_m) == 0 and len(li_tau_m) == 0: # emg0X
            theta = 0
        if len(li_eta_p) == 0 and len(li_tau_p) == 1: # emgX1
            li_eta_p = [1]
        elif len(li_eta_p) == 0 and len(li_tau_p) == 0: # emgX0
            theta = 1
    if len(li_eta_m) > 0 and abs(sum(li_eta_m) - 1) > norm_precision:
        msg = "Sum of elements in li_eta_m is not normalized to within {}.".format(norm_precision)
        raise Exception(msg)
    if len(li_eta_m) != len(li_tau_m):
        raise Exception("Lengths of li_eta_m and li_tau_m do not match.")
    if len(li_eta_p) > 0 and abs(sum(li_eta_p) - 1) > norm_precision:
        msg = "Sum of elements in li_eta_p is not normalized to within {}.".format(norm_precision)
        raise Exception(msg)
    if len(li_eta_p) != len(li_tau_p):
        raise Exception("Lengths of li_eta_p and li_tau_p do not match.")

    # Distribute counts over different peaks and background (bkgd)
    # randomly distribute ions using amps and c_bkg as prob. weights
    N_peaks = len(amps)
    counts = np.append(amps/bin_width,bkg_c*N_bins) # cts in each peak & bkgd
    weights = counts/np.sum(counts) # normalized probability weights

    peak_dist = np.random.choice(range(N_peaks+1), size=N_events, p=weights)
    N_bkg = np.count_nonzero(peak_dist == N_peaks) # calc. number of bkgd counts

    events = np.array([])
    # Create & add random samples from each individual peak
    for i in range(N_peaks):
        N_i = np.count_nonzero(peak_dist == i) # get no. of ions in peak
        if theta == -1: # Gaussian
            events_i = Gaussian_rvs(mus[i], sigma*scl_facs[i], N_samples=N_i)
        else: # hyper-EMG
            events_i = h_emg_rvs(mus[i], sigma*scl_facs[i], theta, li_eta_m,
                                 np.array(li_tau_m)*scl_facs[i], li_eta_p,
                                 np.array(li_tau_p)*scl_facs[i], N_samples=N_i)
        events = np.append(events, events_i)

    # Create & add background events
    bkg = uniform.rvs(size=N_bkg, loc=x_min, scale=sample_range)
    events = np.append(events, bkg)

    # Discard events outside of specified sampling range
    events = events[np.logical_and(events >= x_min, events <= x_max)]
    N_discarded = N_events - events.size
    if N_discarded > 0:
        msg = str("{:.0f} simulated events fell outside the specified sampling "
                  "range and were discarded. Peak areas and area ratios might "
                  "deviate from expectation.").format(N_discarded)
        warnings.warn(msg, UserWarning)

    # Return unbinned array of events or dataframe with histogram
    if out == 'array':
        np.random.shuffle(events) # randomize event ordering
        return events
    elif out == 'hist':
        y = np.histogram(events, bins=bin_edges)[0]
        df = pd.DataFrame(data=y, index=bin_cens, columns = ['Counts'])
        df.index.rename('m/z [u]', inplace=True)
        return df


def simulate_spectrum(spec, x_cen=None, x_range=None, mus=None, amps=None,
                      scl_facs=None, bkg_c=None, N_events=None,
                      copy_spec=False):
    """Create a simulated spectrum using the attributes of a reference spectrum

    The peak shape of the sampling probability density function (PDF)
    follows the shape calibration of the reference spectrum (`spec`). By
    default, all other parameters of the sampling PDF are identical to the
    best-fit parameters of the reference spectrum. If desired, the positions,
    amplitudes and number of peaks in the sampling PDF as well as the background
    level can be changed with the `mus`, `amps` and `bkg_c` arguments.

    Parameters
    ----------
    spec : :class:`~emgfit.spectrum.spectrum`
        Reference spectrum object whose best-fit parameters will be used to
        sample from.
    mus : float or list of float, optional
        Nominal peak centres of peaks in simulated spectrum. Defaults to the
        mus of the reference spectrum fit.
    amps : float or list of float [(counts in peak)*(bin width in u)], optional
        Nominal amplitudes of peaks in simulated spectrum. Defaults to the
        amplitudes of the reference spectrum fit.
    scl_facs : float or list of float, optional
        Scale factors to use for scaling the scale-dependent shape parameters in
        `shape_pars` to a given peak before sampling events. Defaults to the
        scale factors asociated with the fit results stored in the reference
        spectrum's :attr:`spectrum.fit_results` attribute.
    bkg_c : float [counts per bin], optional
        Nominal amplitude of uniform background in simulated spectrum. Defaults
        to the c_bkg obtained in the fit of the first peak in the reference
        spectrum.
    x_cen : float, optional
        Center of simulated x-range. Defaults to `x_cen` of `spec`.
    x_range : float, optional
        Covered x-range of simulated spectrum. Defaults to `x_range` of
        `spectrum`.
    N_events : int, optional
        Number of ion events to simulate (including background events). Defaults
        to total number of events in `spec`.
    copy_spec : bool, optional, default: False
        If `False` (default), this function returns a fresh
        :class:`~emgfit.spectrum.spectrum` object created from the simulated
        mass data. If `True`, this function returns an exact copy of `spec` with
        only the :attr:`data` attribute replaced by the new simulated mass data.

    Returns
    -------
    :class:`~emgfit.spectrum.spectrum`
       If `copy_spec = False` (default) a fresh spectrum object holding the
       simulated mass data is returned. If `copy_spec = True`, a copy of the
       reference spectrum `spec` is returned with only the :attr:`data`
       attribute replaced by the new simulated mass data.

    Notes
    -----
    Random events are created via custom Hyper-EMG extensions of Scipy's
    :meth:`scipy.stats.exponnorm.rvs` method.

    The returned spectrum follows the binning of the reference spectrum.

    Mind the different units for peak amplitudes `amps`
    (<counts in peak> * <bin width in x-axis units>) and the background level
    `bkg_c` (counts per bin). When spectrum data is simulated counts are
    distributed between the different peaks and the background with probability
    weights `amps` / <bin width in x-axis units> and `bkg_c` * <number of bins>,
    respectively. As a consequence, simply changing `N_events` (while keeping
    all other arguments constant), will cause `amps` and `bkg_c` to deviate from
    their nominal units.

    """
    if spec.fit_results is [] or None:
        raise Exception("No fit results found in reference spectrum `spec`.")
    if x_cen is None and x_range is None:
        x = spec.data.index.values
        bin_width_start = x[1] - x[0]
        bin_width_end = x[-1] - x[-2]
        x_min = x[0] - bin_width_start/2
        x_max = x[-1] + bin_width_end/2
        indeces = range(len(spec.peaks)) # get peak indeces in sampling range
    else:
        x_min = x_cen - x_range
        x_max = x_cen + x_range
        # Get peak indeces in sampling range:
        peaks = spec.peaks
        indeces = [i for i in range(len(peaks)) if x_min <= peaks[i].x_pos <= x_max]
    if mus is None:
        if len(indeces) == 0:
            msg = str("No peaks in sampling range.")
            warnings.warn(msg, UserWarning)
        mus = []
        for i in indeces:
            result = spec.fit_results[i]
            pref = 'p{0}_'.format(i)
            mus.append(result.best_values[pref+'mu'])
    if amps is None:
        amps = []
        for i in indeces:
            result = spec.fit_results[i]
            pref = 'p{0}_'.format(i)
            amps.append(result.best_values[pref+'amp'])
    if scl_facs is None:
        scl_facs = []
        for i in indeces:
            result = spec.fit_results[i]
            pref = 'p{0}_'.format(i)
            try:
                scl_facs.append(result.params[pref+'scl_fac'])
            except KeyError:
                scl_facs.append(1.0)
    if not (len(mus)==len(amps)==len(scl_facs)):
        msg = "`mus`, `amps` and `scl_facs` argments must have the same length!"
        raise Exception(msg)
    if bkg_c is None:
        if len(indeces) == 0:
            raise Exception("Zero background and no peaks in sampling range.")
        result = spec.fit_results[indeces[0]]
        bkg_c = result.best_values['bkg_c']
    if N_events is None:
        # Get total number of counts in simulated region of original spectrum:
        N_events = int(spec.data[x_min:x_max]["Counts"].sum())

    # Create histogram with Monte Carlo events
    x = spec.data[x_min:x_max].index.values
    df = simulate_events(spec.shape_cal_pars, mus, amps, bkg_c, N_events,
                         x_min, x_max, out='hist', scl_facs=scl_facs,
                         N_bins=None, bin_cens=x)

    # Copy original spectrum and overwrite data
    # This copies all results such as peak assignments, PS calibration,
    # fit_results etc.
    if copy_spec:
        from copy import deepcopy
        new_spec = deepcopy(spec)
        new_spec.data = df
    else: # Define a fresh spectrum with sampled data
        from emgfit.spectrum import spectrum
        new_spec = spectrum(df=df, show_plot=False)

    return new_spec


def fit_simulated_spectra(spec, fit_result, alt_result=None, N_spectra=1000,
                          seed=None, randomize_ref_mus_and_amps=False,
                          MC_shape_par_samples=None, n_cores=-1):
    """Fit spectra simulated via sampling from a reference distribution

    This function performs fits of many simulated spectra. The simulated spectra
    are created by sampling events from the best-fit PDF asociated with
    `fit_result` (as e.g. needed for a parametric bootstrap). The `alt_model`
    can be used to perform the fits with a different distribution than the
    reference PDF used for the event sampling.

    Parameters
    ----------
    spec : :class:`emgfit.spectrum.spectrum`
        Spectrum object to perform bootstrap on.
    fit_result : :class:`lmfit.model.ModelResult`
        Fit result object holding the best-fit distribution to sample from.
    alt_result : :class:`lmfit.model.ModelResult`, optional
        Fit result object holding a prepared fit model to be used for the
        fitting. Defaults to the fit model stored in `fit_result`.
    N_spectra : int, optional
        Number of simulated spectra to fit. Defaults to 1000, which
        typically yields statistical uncertainty estimates with a Monte Carlo
        uncertainty of a few percent.
    randomize_ref_mus_and_amps : bool, default: False
        If `True`, the peak and background amplitudes and the peak centroids of
        the reference spectrum to sample from will be varied assuming normal
        distributions around the best-fit values with standard deviations given
        by the respective standard errors stored in `fit_result`.
    MC_shape_par_samples : :class:`pandas.DataFrame` 
        Monte Carlo shape parameter samples to use in the fitting. 
    seed : int, optional
        Random seed to use for reproducible sampling.
    n_cores : int, optional
        Number of CPU cores to use for parallelized fitting of simulated
        spectra. When set to `-1` (default) all available cores are used.

    Returns
    -------
    :class:`numpy.ndarray` of :class:`lmfit.minimizer.MinimizerResult`
        MinimizerResults obtained in the fits of the simulated spectra.

    Note
    ----
    The `randomize_ref_mus_and_amps` option allows one to propagate systematic
    uncertainties in the determination of the reference parameters into the
    Monte Carlo results. If varying the centroid and amplitude parameters of the
    reference spectrum, the standard deviations of the parameter distributions
    will be taken as the respective standard errors determined by lmfit (see
    lmfit fit report table) and might not be consistent with the area and mass
    uncertainty estimates shown in the peak properties table.

    See also
    --------
    :meth:`emgfit.spectrum.spectrum.get_errors_from_resampling`
    :func:`emgfit.sample.simulate_events` for details on the event sampling.

    """
    bkg_c = fit_result.best_values['bkg_c']
    bkg_c_err = fit_result.params['bkg_c'].stderr
    method = fit_result.method
    shape_pars = spec.shape_cal_pars
    x_cen = fit_result.x_fit_cen
    x_range = fit_result.x_fit_range
    x = fit_result.x
    y = fit_result.y
    x_min = x_cen - 0.5*x_range
    x_max = x_cen + 0.5*x_range

    if alt_result is None:
        fit_model = fit_result.fit_model
        model = fit_result.model
        init_pars = fit_result.init_params
    else:
        fit_model = alt_result.fit_model
        model = alt_result.model
        init_pars = alt_result.init_params

    # Determine tail order of fit model for normalization of initial etas
    if fit_result.fit_model.startswith('emg'):
        n_ltails = int(fit_result.fit_model.lstrip('emg')[0])
        n_rtails = int(fit_result.fit_model.lstrip('emg')[1])
    else:
        n_ltails = 0
        n_rtails = 0

    # Collect aLL peaks, peak centroids and amplitudes of fit_result
    fitted_peaks = [idx for idx, p in enumerate(spec.peaks)
                    if x_min < p.x_pos < x_max] # indeces of all fitted peaks
    mus, mu_errs = [], []
    amps, amp_errs = [], []
    scl_facs = []
    for idx in fitted_peaks:
        pref = 'p{0}_'.format(idx)
        mus.append(fit_result.best_values[pref+'mu'])
        mu_errs.append(fit_result.params[pref+'mu'].stderr)
        amps.append(fit_result.best_values[pref+'amp'])
        amp_errs.append(fit_result.params[pref+'amp'].stderr)
        try:
            scl_facs.append(fit_result.params[pref+'scl_fac'])
        except KeyError:
            scl_facs.append(1)

    import emgfit.fit_models as fit_models
    from .model import save_model, load_model
    from lmfit.minimizer import minimize
    import lmfit
    import time
    datetime = time.localtime() # get current date and time
    datetime_str = time.strftime("%Y-%m-%d_%H-%M-%S", datetime)
    if spec.input_filename is not None:
        data_fname = spec.input_filename.rsplit('.', 1)[0] # del. extension
    else:
        data_fname = ''
    modelfname = data_fname+datetime_str+"_resampl_model.sav"
    save_model(model, modelfname)
    N_events = int(np.sum(y))
    funcdefs = {'constant': lmfit.models.ConstantModel,
                str(fit_model): getattr(fit_models,fit_model)}
    def refit(seed, shape_pars_i):
        # create simulated spectrum data by sampling from fit-result PDF
        np.random.seed(seed)
        if randomize_ref_mus_and_amps:
            bkg_c_i = np.random.normal(loc=bkg_c, scale=bkg_c_err, size=1)
            amps_i = np.random.normal(loc=amps, scale=amp_errs)
            mus_i = np.random.normal(loc=mus, scale=mu_errs)
        else:
            bkg_c_i = bkg_c
            amps_i = amps
            mus_i = mus

        if MC_shape_par_samples is not None:
            # Calculate missing parameters from normalization
            if n_ltails == 2:
                shape_pars_i['eta_m2'] = 1 - shape_pars_i['eta_m1']
            elif n_ltails == 3:
                eta_m2 = shape_pars_i['delta_m'] - shape_pars_i['eta_m1']
                shape_pars_i['eta_m3'] = 1 - shape_pars_i['eta_m1'] - eta_m2
            if n_rtails == 2:
                shape_pars_i['eta_p2'] = 1 - shape_pars_i['eta_p1']
            elif n_rtails == 3:
                eta_p2 = shape_pars_i['delta_p'] - shape_pars_i['eta_p1']
                shape_pars_i['eta_p3'] = 1 - shape_pars_i['eta_p1'] - eta_p2
        df = simulate_events(shape_pars_i, mus_i, amps_i, bkg_c_i,
                             N_events, x_min, x_max,
                             out='hist', scl_facs=scl_facs, bin_cens=x)
        new_x = df.index.values
        new_y = df['Counts'].values
        new_y_err = np.maximum(1, np.sqrt(new_y)) # Poisson (counting) stats
        # Weights for residuals: residual = (fit_model - y) * weights
        new_weights = 1./new_y_err

        model = load_model(modelfname, funcdefs=funcdefs)

        # re-perform fit on simulated spectrum - for performance, use only the
        # underlying Minimizer object instead of full lmfit model interface
        try:
            min_res = minimize(model._residual, init_pars, method=method,
                               args=(new_y, new_weights), kws={'x': new_x},
                               scale_covar=False, nan_policy='propagate',
                               reduce_fcn=None, calc_covar=False)
            return min_res

        except ValueError:
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                msg = str("Fit failed with ValueError (likely NaNs in "
                           "y-model array) and will be excluded.")
                warnings.warn(msg, UserWarning)
            return None

    # For reproducible sampling with joblib parallel, generate `N_spectra`
    # random seeds for refit() - only works with the default Loky backend
    np.random.seed(seed=seed) # for reproducible sampling
    joblib_seeds = np.random.randint(2**31, size=N_spectra)
    from tqdm.auto import tqdm # add progress bar with tqdm
    from joblib import Parallel, delayed
    try:
        if MC_shape_par_samples is None:
            min_results = np.array(Parallel(n_jobs=n_cores)
                                    (delayed(refit)(s, shape_pars) for s in tqdm(joblib_seeds)))
        else:
            from .spectrum import _strip_prefs
            min_results = np.array(Parallel(n_jobs=n_cores)
                                    (delayed(refit)(s, _strip_prefs(dict(MC_shape_par_samples.iloc[i]))) for i, s in tqdm(enumerate(joblib_seeds))))
    finally:
        # Force workers to shut down and clean up temp SAV file
        from joblib.externals.loky import get_reusable_executor
        import os
        get_reusable_executor().shutdown(wait=True)
        os.remove(modelfname)

    return min_results

################################################################################
##### Define functions for non-parametric resampling
def resample_events(df, N_events=None, x_cen=None, x_range=0.02, out='hist'):
    """Create simulated spectrum via non-parametric resampling from `df`.

    The simulated data is obtained through resampling from the specified dataset
    with replacement.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Original histogrammed spectrum data to re-sample from.
    N_events : int, optional
        Number of events to create via non-parametric re-sampling, defaults to
        number of events in original DataFrame `df`.
    x_cen : float [u/z], optional
        Center of mass range to re-sample from. If ``None``, re-sample from
        full mass range of input data `df`.
    x_range : float [u/z], optional
        Width of mass range to re-sample from. Defaults to 0.02 u. `x_range`
        is only relevant if a `x_cen` argument is specified.
    out : str, optional
        Output format of sampled data. Options:

        - ``'hist'`` for binned mass spectrum (default). The centres of the mass
          bins must be specified with the `bin_cens` argument.
        - ``'array'`` for unbinned array of single ion and background events.

    Returns
    -------
    :class:`pandas.Dataframe` or :class:`numpy.ndarray`
       If out='hist' a dataframe with a histogram of the format
       [bin centre, counts in bin] is returned. If out='array' an unbinned
       array with the x-values of single ion or background events is returned.

    """
    if x_cen:
        x_min = x_cen - 0.5*x_range
        x_max = x_cen + 0.5*x_range
        df = df[x_min:x_max]
    mass_bins = df.index.values
    counts = df['Counts'].values.astype(int)

    # Convert histogrammed spectrum (equivalent to MAc HIST export mode) to
    # list of events (equivalent to MAc LIST export mode)
    orig_events =  np.repeat(mass_bins, counts, axis=0)

    # Create new DataFrame of events by bootstrapping from `orig_events`
    if N_events == None:
        N_events = len(orig_events)
    random_indeces = np.random.randint(0, len(orig_events), N_events)
    events = pd.DataFrame(orig_events[random_indeces])

    # Convert list of events back to a DataFrame with histogram data
    bin_cens = df.index.values
    bin_width = df.index.values[1] - df.index.values[0]
    bin_edges = np.append(bin_cens-0.5*bin_width,
                          bin_cens[-1]+0.5*bin_width)

    # Return unbinned array of events or dataframe with histogram
    if out == 'array':
        np.random.shuffle(events) # randomize event ordering
        return events
    elif out == 'hist':
        hist = np.histogram(events, bins=bin_edges)
        df_new = pd.DataFrame(data=hist[0], index=bin_cens, dtype=float,
                            columns=["Counts"])
        df_new.index.name = "m/z [u]"
        return df_new
    