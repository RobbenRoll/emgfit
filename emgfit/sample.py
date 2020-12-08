################################################################################
##### Python module for creating simulated time-of-flight mass spectra with
##### Gaussian and hyper-exponentially-modified Gaussian lines shapes
##### Author: Stefan Paul

import numpy as np
import pandas as pd
from scipy.stats import exponnorm, uniform, poisson

################################################################################
##### Define Functions for drawing random variates from Hyper-EMG PDFs
norm_precision = 1e-09

def _h_m_i_rvs(mu,sigma,tau_m,N_i):
    """Helper function for definition of h_m_emg_rvs """
    rvs = mu - exponnorm.rvs(loc=0,scale=sigma,K=tau_m/sigma,size=N_i)
    return rvs


def h_m_emg_rvs(mu, sigma, *t_args,N_samples=None):
    """Draw random samples from neg. skewed Hyper-EMG PDF

    Parameters
    ----------
    mu : float [u]
        Nominal position of simulated peak (mean of underlying Gaussian).
    sigma : float [u]
        Nominal standard deviation (of the underlying Gaussian) of the simulated
        hyper-EMG peak.
    theta : float
        Mixing weight of pos. & neg. skewed EMG distributions.
    t_args : list of lists of float
        List containing lists of the EMG tail parameters with the signature:
        [[eta_m1, eta_m2, ...], [tau_m1, tau_m2, ...]]
    N_samples : int
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
    assert abs(sum(li_eta_m) - 1) < norm_precision, "eta_m's don't add up to 1."
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


def _h_p_i_rvs(mu,sigma,tau_p,N_i):
    """Helper function for definition of h_p_emg_rvs """
    rvs = exponnorm.rvs(loc=mu,scale=sigma,K=tau_p/sigma,size=N_i)
    return rvs


def h_p_emg_rvs(mu, sigma, *t_args,N_samples=None):
    """Draw random samples from pos. skewed Hyper-EMG PDF

    Parameters
    ----------
    mu : float [u]
        Nominal position of simulated peak (mean of underlying Gaussian).
    sigma : float [u]
        Nominal standard deviation (of the underlying Gaussian) of the simulated
        hyper-EMG peak.
    t_args : list of lists of float
        List containing lists of the EMG tail parameters with the signature:
        [[eta_p1, eta_p2, ...], [tau_p1, tau_p2, ...]]
    N_samples : int
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
    assert abs(sum(li_eta_p) - 1) < norm_precision, "eta_p's don't add up to 1."
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
    """Draw random samples from Hyper-EMG PDF

    Parameters
    ----------
    mu : float [u]
        Nominal position of simulated peak (mean of underlying Gaussian).
    sigma : float [u]
        Nominal standard deviation (of the underlying Gaussian) of the simulated
        hyper-EMG peak.
    theta : float
        Mixing weight of pos. & neg. skewed EMG distributions.
    t_args : list of lists of float
        List containing lists of the EMG tail parameters with the signature:
        [[eta_m1, eta_m2, ...], [tau_m1, tau_m2, ...],
        [eta_p1, eta_p2, ...], [tau_p1, tau_p2, ...]]
    N_samples : int
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
        neg = np.random.choice([1,0],size=N_samples,p = [theta,1-theta]) # randomly distribute ions between h_m_emg and h_p_emg according to weight theta
        N_m = int(np.sum(neg)) #int(np.round(theta*N_samples)) # np.rint(theta*N_samples,dtype=int)
        N_p = N_samples - N_m # np.rint((1-theta)*N_samples,dtype=int)
        rvs_m = h_m_emg_rvs(mu, sigma, li_eta_m, li_tau_m, N_samples=N_m)
        rvs_p = h_p_emg_rvs(mu, sigma, li_eta_p, li_tau_p, N_samples=N_p)
        rvs = np.append(rvs_m,rvs_p)
    return rvs

################################################################################
##### Define functions for creating simulated spectra

def simulate_events(shape_pars, mus, amps, bkg_c, N_events, x_min,
                    x_max, out='hist', N_bins=None, bin_cens=None):
    """Create simulated ion events drawn from a user-defined probability
    distribution function (PDF)

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
    mus : float or list of float [u]
        Nominal peak positions of peaks in simulated spectrum.
    amps : float or list of float [counts per u]
        Nominal amplitudes of peaks in simulated spectrum.
    bkg_c : float [counts per bin], optional, default: 0.0
        Nominal amplitude of uniform background in simulated spectrum.
    x_min : float [u]
        Beginning of sampling mass range.
    x_max : float [u]
        End of sampling mass range.
    N_events : int, optional, default: 1000
        Total number of events to simulate (signal and background events).
    out : str, optional
        Output format of sampled data. Options:

        - ``'hist'`` for binned mass spectrum (default). The centres of the mass
          bins must be specified with the `bin_cens` argument.
        - ``'list'`` for unbinned list of single ion and background events.

    N_bins : int, optional
        Number of uniform bins to use in ``'hist'`` output mode. The **outer**
        edges of the first and last bin are fixed to the start and end of the
        sampling range respectively (i.e. `x_min` and `x_max`). In between, bins
        are distributed with a fixed spacing of (`x_max`-`x_min`)/`N_bins`.
    bin_cens : :class:`numpy.ndarray` [u]
        Centres of mass bins to use in ``'hist'`` output mode. This argument
        allows the realization of non-uniform binning. Bin edges are centred
        between neighboring bins. Note: Bins outside the sampling range defined
        with `x_min` and `x_max` will be empty.

    Returns
    -------
    :class:`numpy.ndarray` or :class:`pandas.Dataframe`
       If out='hist' a dataframe with a histogram of the format
       [bin centre [u], counts in bin] is returned. If out='list' an unbinned
       array with mass values [u] of single ion or background events is returned.

    Notes
    -----
    Random events are created via custom Hyper-EMG extensions of Scipy's
    :meth:`scipy.stats.exponnorm.rvs` method.

    Currently, all simulated peaks have identical width and shape (no re-scaling
    of mass-dependent shape parameters to a peak's mass centroid).

    Routine requires tail arguments in shape_cal_pars dict to be ordered
    (eta_m1,eta_m2,...) etc..

    **Mind the different units for peak amplitudes `amps` (counts per u) and the
    background level `bkg_c` (counts per bin)**. When spectrum data is
    simulated counts are randomly distributed between the different peaks and
    the background with probability weights `amps` * <bin width in> and
    `bkg_c` * <number of bins>, respectively. As a consequence, simply changing
    `N_events` (while keeping all other arguments constant), will cause `amps`
    and `bkg_c` to deviate from their nominal units.

    """
    # TODO: Implement rescaling of peak-shape parameters with mass
    mus = np.atleast_1d(mus)
    amps = np.atleast_1d(amps)
    assert len(mus) == len(amps), "Lengths of `mus` and `amps` arrays must match."
    if (mus < x_min).any() or (mus > x_max).any():
        import warnings
        msg = str("At least one peak centroid in `mus` is outside the sampling range.")
        warnings.warn(msg, UserWarning)

    sample_range = x_max - x_min

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
    else:
        raise Exception("`N_bins` or `bin_cens` argument must be specified!")

    # Prepare shape parameters
    sigma = shape_pars['sigma']
    theta = shape_pars['theta']
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
    if len(li_eta_m) == 0 and len(li_tau_m) != 0:
        li_eta_m = [1]
    if len(li_eta_p) == 0 and len(li_tau_p) != 0:
        li_eta_p = [1]

    # Distribute counts over different peaks and background
    # randomly distribute ions using amps and c_bkg as prob. weights
    N_peaks = len(amps)
    counts = np.append(amps/bin_width,bkg_c*N_bins) # cts in each peak & background
    weights = counts/np.sum(counts) # normalized probability weights

    peak_dist = np.random.choice(range(N_peaks+1),size=N_events,p = weights)
    N_bkg = np.count_nonzero(peak_dist == N_peaks) # calc. number of background counts

    events = np.array([])
    # Loop over peaks and create random samples from each peak
    for i in range(N_peaks):
        N_i = np.count_nonzero(peak_dist == i) # get no. of ions in peak
        mu = mus[i]
        events_i = h_emg_rvs(mu,sigma,theta,li_eta_m, li_tau_m, li_eta_p,li_tau_p, N_samples=N_i)
        events = np.append(events,events_i)

    # Create background events
    bkg = uniform.rvs(size=N_bkg,loc=x_min,scale=sample_range)
    events = np.append(events,bkg)

    if out == 'list':  # return unbinned list of events
        return events
    elif out == 'hist':  # return histogram
        y = np.histogram(events,bins=bin_edges)[0]
        df = pd.DataFrame(data=y,index=bin_cens,columns = ['Counts'])
        df.index.rename('Mass [u]',inplace=True)
        return df


def simulate_spectrum(spec, x_cen=None, x_range=None, mus=None, amps=None,
                      bkg_c=None, N_events=None, copy_spec=False):
    """Create a simulated spectrum using the attributes of a reference spectrum

    The peak shape of the sampling probability distribution function (PDF)
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
    mus : float or list of float [u], optional
        Nominal peak centres of peaks in simulated spectrum. Defaults to the
        mus of the reference spectrum fit.
    amps : float or list of float [counts per u], optional
        Nominal amplitudes of peaks in simulated spectrum. Defaults to the
        amplitudes of the reference spectrum fit.
    bkg_c : float [counts per bin], optional
        Nominal amplitude of uniform background in simulated spectrum. Defaults
        to the c_bkg obtained in the fit of the first peak in the reference
        spectrum.
    x_cen : float [u], optional
        Mass center of simulated spectrum. Defaults to `x_cen` of `spec`.
    x_range : float [u], optional
        Covered mass range of simulated spectrum. Defaults to `x_range` of
        `spectrum`.
    N_events : int, optional
        Number of ion events to simulate (including background events). Defaults
        to total number of events in `spec`.
    copy_spec : bool, optional, default: False
        If `False` (default), this function returns a fresh
        :class:`~emgfit.spectrum.spectrum` object created from the simulated
        mass data. If `True`, this function returns an exact copy of `spec` with
        only the :attr`data` attribute replaced by the new simulated mass data.

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

    Currently, all simulated peaks have identical width and shape (no re-scaling
    of mass-dependent shape parameters to a peak's mass centroid).

    The returned spectrum follows the binning of the reference spectrum.

    Mind the different units for peak amplitudes `amps` (counts per u) and the
    background level `bkg_c` (counts per bin). When spectrum data is
    simulated counts are distributed between the different peaks and the
    background with probability weights `amps` * <bin width in> and
    `bkg_c` * <number of bins>, respectively. As a consequence, simply changing
    `N_events` (while keeping all other arguments constant), will render `amps`
    and `bkg_c` away from their nominal units.

    """
    if spec.fit_results is [] or None:
        raise Exception("No fit results found in reference spectrum `spec`.")
    if x_cen is None and x_range is None:
        x_min = spec.data.index.values[0]
        x_max = spec.data.index.values[-1]
        indeces = range(len(spec.peaks)) # get peak indeces in sampling range
    else:
        x_min = x_cen - x_range
        x_max = x_cen + x_range
        # Get peak indeces in sampling range:
        peaks = spec.peaks
        indeces = [i for i in range(len(peaks)) if x_min <= peaks[i].x_pos <= x_max]
    if mus is None:
        if len(indeces) == 0:
            import warnings
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
    if bkg_c is None:
        assert len(indeces) != 0, "Zero background and no peaks in sampling range."
        result = spec.fit_results[indeces[0]]
        bkg_c = result.best_values['bkg_c']
    if N_events is None:
        N_events = int(np.sum(spec.data['Counts'])) # total number of counts in spectrum

    # Create histogram with Monte Carlo events
    x = spec.data[x_min:x_max].index.values
    df = simulate_events(spec.shape_cal_pars,mus,amps,bkg_c,N_events,x_min,
                         x_max,out='hist',N_bins=None,bin_cens=x)

    # Copy original spectrum and overwrite data
    # This copies all results such as peak assignments, PS calibration,
    # fit_results etc.
    if copy_spec:
        from copy import deepcopy
        new_spec = deepcopy(spec)
        new_spec.data = df
    else: # Define a fresh spectrum with sampled data
        from emgfit import spectrum
        new_spec = spectrum.spectrum(df=df)

    return new_spec
