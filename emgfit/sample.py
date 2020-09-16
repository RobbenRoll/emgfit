################################################################################
##### Python module for creating simulated TOF mass spectra via inverse
##### transform sampling
##### Author: Stefan Paul

from scipy.stats import exponnorm, uniform, poisson
# N_samples = 50000 # total no of counts in spectrum
# # Random samples drawn from h_m EMG PDF
# x_min = 99.99
# x_max = 100.01
# bin_width = 30e-06
# c_bkg = 0 # mean no of background counts per bin
# x_range = x_max - x_min
# N_bins = int(x_range/bin_width)
# N_bkg = int(c_bkg * N_bins)
# print(N_bkg)

################################################################################
##### Define Functions for drawing random variates from Hyper-EMG PDFs
norm_precision = 1e-09

def h_m_i_rvs(mu,sigma,tau_m,N_i):
    """Helper function for definition of h_m_emg_rvs """
    rvs = mu - exponnorm.rvs(loc=0,scale=sigma,K=tau_m/sigma,size=N_i)
    return rvs


def h_m_emg_rvs(mu, sigma, *t_args,N_samples=None):
    """Draw random samples from neg. skewed Hyper-EMG PDF
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
        rvs_i = h_m_i_rvs(mu,sigma,tau_m,N_i)
        rvs = np.append(rvs,rvs_i)
    return rvs


def h_p_i_rvs(mu,sigma,tau_p,N_i):
    """Helper function for definition of h_p_emg_rvs """
    rvs = exponnorm.rvs(loc=mu,scale=sigma,K=tau_p/sigma,size=N_i)
    return rvs


def h_p_emg_rvs(mu, sigma, *t_args,N_samples=None):
    """Draw random samples from pos. skewed Hyper-EMG PDF
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
        rvs_i = h_p_i_rvs(mu,sigma,tau_p,N_i)
        rvs = np.append(rvs,rvs_i)
    return rvs


def h_emg_rvs(mu, sigma , theta, *t_args, N_samples=None):
    """Draw random samples from Hyper-EMG PDF
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

def sample_spectrum(spectrum,mus,amps,bkg_c=0.0,x_cen=None,x_range=None,
                    N_samples=None,out='hist'):
    """ Create simulated ion events using the best-fit peak shape of a reference
    spectrum

    Parameters
    ----------
    spectrum : :class:`~emgfit.spectrum.spectrum`
        Reference spectrum object whose best-fit peak-shape parameters will be
        used to sample from.
    mus : float or list of float [u]
        Nominal peak centres of peaks in simulated spectrum.
    amps : float or list of float
        Nominal amplitudes of peaks in simulated spectrum.
    bkg_c : float, optional, default: 0.0
        Nominal amplitude of constant background in simulated spectrum.
    x_cen : float [u], optional
        Mass center of simulated spectrum. Defaults to `x_cen` of `spectrum`.
    x_range : float [u], optional
        Covered mass range of simulated spectrum. Defaults to `x_range` of
        `spectrum`.
    N_samples : int, optional
        Number of ion events to simulate (excluding background events). Defaults
        to total number of events in `spectrum`.
    out : str
        Output format of sampled data. Options:

        - ``'hist'`` for binned mass spectrum (default).
        - ``'list'`` for unbinned list of single ion and background events.

   Returns
   -------
   class:`numpy.array` or :class:`pandas.Dataframe`
       If out='hist' a dataframe with a histogram of the format
       [bin centre [u], counts in bin] is returned. If out='list' an unbinned
       array with mass values [u] of single ion or background events is returned.

    Notes
    -----

    Routine requires tail arguments in shape_cal_pars dict to be ordered (eta_m1,eta_m2,...).

    Spectra sampled in ``'hist'`` mode follow the binning of the reference
    spectrum.

    """
    if isinstance(mus,float):
        mus = [mus]
    if isinstance(amps,float):
        amps = [amps]
    else:
        assert len(mus) == len(amps), "Lengths of `mus` and `amps` arrays must match."

    #  Get xmin and xmax
    if x_cen is None and x_range is None:
        x_min = spec.data.index[0]
        x_max = spec.data.index[-1]
    else:
        x_min = x_cen - x_range
        x_max = x_cen + x_range
    x_range = x_max -x_min
    bin_width = spec.data.index[1] - spec.data.index[0]

    # Prepare number of samples to draw
    if N_samples is None:
        N_samples = int(np.sum(spec.data['Counts'])) # total number of counts in spectrum

    # Prepare shape parameters
    pars = spectrum.shape_cal_pars
    sigma = pars['sigma']
    theta = pars['theta']
    li_eta_m = []
    li_tau_m = []
    li_eta_p = []
    li_tau_p = []
    for key, val in pars.items():
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
    weights = np.append(amps,bkg_c) # weights of each peak and background
    weights = weights/np.sum(weights) # normalize weights

    peak_dist = np.random.choice(range(N_peaks+1),size=N_samples,p = weights)
    N_bkg = np.count_nonzero(peak_dist == N_peaks) # calc. number of background counts

    events = np.array([])
    # Loop over peaks and create random samples from each peak
    for i in range(N_peaks):
        N_i = np.count_nonzero(peak_dist == i) # get no. of ions in peak
        mu = mus[i]
        events_i = h_emg_rvs(mu,sigma,theta,li_eta_m, li_tau_m, li_eta_p,li_tau_p, N_samples=N_i)
        events = np.append(events,events_i)

    # Create background events
    bkg = uniform.rvs(size=N_bkg,loc=x_min,scale=x_range)
    events = np.append(events,bkg)

    if out is 'list':  # return unbinned list of events
        return events
    elif out is 'hist':  # return histogram with identical binning as `spectrum`
        x = spec.data.loc[x_min:x_max].index.values
        bin_edges = np.append(x,x[-1] + bin_width) - bin_width/2
        y = np.histogram(events,bins=bin_edges)[0]
        df = pd.DataFrame(data=y,index=x,columns = ['Counts'])
        df.index.rename('Mass [u]',inplace=True)
        return df
