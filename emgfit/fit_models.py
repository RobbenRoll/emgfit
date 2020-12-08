################################################################################
##### Module with lmfit models for Gaussian and Hyper-EMG distributions
##### Author: Stefan Paul

##### Import dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lmfit as fit
from .config import *
from .emg_funcs import *
upper_bound_taus = 5e-02 # keeps minimizer from running towards virtually flat tails #TODO: COnsider moving to config for user control
rel_var_mus = 1e-05 # allows centroids of nderlying Gaussians (`mu`) to vary within x_pos +- rel_var_mus*x_pos

def create_default_init_pars(mass_number=100): #TODO: COnsider moving to config for user control
    """
    Scale default parameters to mass of interest and return parameter dictionary.

    Parameters
    ----------
    mass_number : int, optional
        Atomic mass number of peaks of interest, defaults to 100.

    Returns
    -------
    dict
        Dictionary with default initial parameters (scaled to `mass_number`).

    Notes
    -----
    **The default parameters were defined for mass 100**, to obtain suitable
    parameters at other masses all mass-dependent parameters (i.e. shape
    parameters & `amp`) are multiplied by the scaling factor `mass_number`/100.

    """
    # Default initial parameters for peaks around mass 100 (with
    # mass scaling factor):
    scl_factor = mass_number/100
    amp = 0.45*scl_factor
    mu = None
    sigma = 0.00014*scl_factor # [u]
    theta = 0.5
    eta_m1 = 0.85
    eta_m2 = 0.10
    eta_m3 = 0.05
    tau_m1 = 50e-06*scl_factor # [u]
    tau_m2 = 500e-06*scl_factor # [u]
    tau_m3 = 1000e-06*scl_factor # [u]
    eta_p1 = 0.85
    eta_p2 = 0.10
    eta_p3 = 0.05
    tau_p1 = 50e-06*scl_factor # [u]
    tau_p2 = 600e-06*scl_factor # [u]
    tau_p3 = 1000e-06*scl_factor # [u]
    pars_dict = {'amp': amp, 'mu': mu, 'sigma': sigma, 'theta': theta, 'eta_m1': eta_m1, 'eta_m2': eta_m2, 'eta_m3': eta_m3, 'tau_m1': tau_m1, 'tau_m2': tau_m2, 'tau_m3': tau_m3, 'eta_p1': eta_p1, 'eta_p2': eta_p2, 'eta_p3': eta_p3, 'tau_p1': tau_p1, 'tau_p2': tau_p2, 'tau_p3': tau_p3}
    return pars_dict


pars_dict = create_default_init_pars()

################################################################################
##### Define emgfit fit models
def Gaussian(peak_index, x_pos, amp, init_pars=pars_dict,
             vary_shape_pars=True, index_first_peak=None):
    """
    Gaussian lmfit model (single-peak Gaussian fit model)

    Parameters
    ----------
    peak_index :  int
        Index of peak to fit.
    x_pos : float
       Initial guess of peak centroid.
    amp : float
        Initial guess of peak amplitude.
    init_pars : dict
        Initial parameters for fit ('amp' and 'mu' parameters in `init_pars`
        dictionary are overwritten by the given `amp` and `x_pos` arguments)
    vary_shape_pars : bool
        Whether to vary or fix peak shape parameters (i.e. sigma, theta,
        eta's and tau's).
    index_first_peak : int
        Index of the first peak to be fit in a multi-peak-fit. Only use this
        during peak shape determination to enforce common shape parameters
        for all peaks to be fitted. (For a regular fit with
        ``vary_shape_pars = False`` this is irrelevant.)

    Returns
    -------
    :class:`lmfit.model.Model`
        `lmfit` model object

    """
    # Define model function
    def Gaussian(x, amp, mu, sigma):
        return  amp/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*sigma**2))
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak
    model = fit.Model(Gaussian, prefix = pref, nan_policy='propagate')

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp, min=0)
    model.set_param_hint(pref+'mu', value=x_pos, min=x_pos*(1-rel_var_mus), max=x_pos*(1+rel_var_mus))
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, vary=vary_shape_pars)

    # Enfore common shape parameters for all peaks
    # (only needed during peak shape calibration)
    if index_first_peak != None and (peak_index != index_first_peak):
        first_pref = 'p{0}_'.format(index_first_peak)
        model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, expr=first_pref+'sigma')

    return model


def emg01(peak_index, x_pos, amp, init_pars=pars_dict,
          vary_shape_pars=True, index_first_peak=None):
    """
    Hyper-EMG(0,1) lmfit model (single-peak fit model with one exponential tail
    on the right)

    Parameters
    ----------
    peak_index :  int
        Index of peak to fit.
    x_pos : float
       Initial guess of peak centroid.
    amp : float
        Initial guess of peak amplitude.
    init_pars : dict
        Initial parameters for fit ('amp' and 'mu' parameters in `init_pars`
        dictionary are overwritten by the given `amp` and `x_pos` arguments)
    vary_shape_pars : bool
        Whether to vary or fix peak shape parameters (i.e. sigma, theta,
        eta's and tau's).
    index_first_peak : int
        Index of the first peak to be fit in a multi-peak-fit. Only use this
        during peak shape determination to enforce common shape parameters
        for all peaks to be fitted. (For a regular fit with
        ``vary_shape_pars = False`` this is irrelevant.)

    Returns
    -------
    :class:`lmfit.model.Model`
        `lmfit` model object

    """
    # Define model function
    def emg01(x, amp, mu, sigma, tau_p1):
        return amp*h_emg(x, mu, sigma, 0, (0,),(0,),(1,),(tau_p1,))
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak
    model = fit.Model(emg01, prefix = pref, nan_policy='propagate')

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp, min=1e-20)
    model.set_param_hint(pref+'mu', value=x_pos, min=x_pos*(1-rel_var_mus), max=x_pos*(1+rel_var_mus))
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)

    # Enfore common shape parameters for all peaks
    # (only needed during peak shape calibration)
    if index_first_peak != None and (peak_index != index_first_peak):
        first_pref = 'p{0}_'.format(index_first_peak)
        model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, expr=first_pref+'sigma')
        model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p1')

    return model


def emg10(peak_index, x_pos, amp, init_pars=pars_dict,
          vary_shape_pars=True, index_first_peak=None):
    """
    Hyper-EMG(1,0) lmfit model (single-peak fit model with one exponential tail
    on the left)

    Parameters
    ----------
    peak_index :  int
        Index of peak to fit.
    x_pos : float
       Initial guess of peak centroid.
    amp : float
        Initial guess of peak amplitude.
    init_pars : dict
        Initial parameters for fit ('amp' and 'mu' parameters in `init_pars`
        dictionary are overwritten by the given `amp` and `x_pos` arguments)
    vary_shape_pars : bool
        Whether to vary or fix peak shape parameters (i.e. sigma, theta,
        eta's and tau's).
    index_first_peak : int
        Index of the first peak to be fit in a multi-peak-fit. Only use this
        during peak shape determination to enforce common shape parameters
        for all peaks to be fitted. (For a regular fit with
        ``vary_shape_pars = False`` this is irrelevant.)


    Returns
    -------
    :class:`lmfit.model.Model`
        `lmfit` model object

    """
    # Define model function
    def emg10(x, amp, mu, sigma, tau_m1):
        return amp*h_emg(x, mu, sigma, 1, (1,),(tau_m1,),(0,),(0,))
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak
    model = fit.Model(emg10, prefix = pref, nan_policy='propagate')

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp, min=1e-20)
    model.set_param_hint(pref+'mu', value=x_pos, min=x_pos*(1-rel_var_mus), max=x_pos*(1+rel_var_mus))
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)

    # Enfore common shape parameters for all peaks
    # (only needed during peak shape calibration)
    if index_first_peak != None and (peak_index != index_first_peak):
        first_pref = 'p{0}_'.format(index_first_peak)
        model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, expr=first_pref+'sigma')
        model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m1')

    return model


def emg11(peak_index, x_pos, amp, init_pars=pars_dict,
          vary_shape_pars=True, index_first_peak=None):
    """
    Hyper-EMG(1,1) lmfit model (single-peak fit model with one exponential tail
    on the left and one exponential tail on the right)

    Parameters
    ----------
    peak_index :  int
        Index of peak to fit.
    x_pos : float
       Initial guess of peak centroid.
    amp : float
        Initial guess of peak amplitude.
    init_pars : dict
        Initial parameters for fit ('amp' and 'mu' parameters in `init_pars`
        dictionary are overwritten by the given `amp` and `x_pos` arguments)
    vary_shape_pars : bool
        Whether to vary or fix peak shape parameters (i.e. sigma, theta,
        eta's and tau's).
    index_first_peak : int
        Index of the first peak to be fit in a multi-peak-fit. Only use this
        during peak shape determination to enforce common shape parameters
        for all peaks to be fitted. (For a regular fit with
        ``vary_shape_pars = False`` this is irrelevant.)

    Returns
    -------
    :class:`lmfit.model.Model`
        `lmfit` model object

    """
    # Define model function
    def emg11(x, amp, mu, sigma, theta, tau_m1, tau_p1):
        return amp*h_emg(x, mu, sigma, theta, (1,),(tau_m1,),(1,),(tau_p1,)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak
    model = fit.Model(emg11, prefix = pref, nan_policy='propagate')

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp, min=1e-20)
    model.set_param_hint(pref+'mu', value=x_pos, min=x_pos*(1-rel_var_mus), max=x_pos*(1+rel_var_mus))
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, vary=vary_shape_pars)
    model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)

    # Enfore common shape parameters for all peaks
    # (only needed during peak shape calibration)
    if index_first_peak != None and (peak_index != index_first_peak):
        first_pref = 'p{0}_'.format(index_first_peak)
        model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, expr=first_pref+'sigma')
        model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, expr=first_pref+'theta')
        model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m1')
        model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p1')

    return model


def emg12(peak_index, x_pos, amp, init_pars=pars_dict,
          vary_shape_pars=True, index_first_peak=None):
    """
    Hyper-EMG(1,2) lmfit model (single-peak fit model with one exponential tail
    on the left and two exponential tails on the right)

    Parameters
    ----------
    peak_index :  int
        Index of peak to fit.
    x_pos : float
       Initial guess of peak centroid.
    amp : float
        Initial guess of peak amplitude.
    init_pars : dict
        Initial parameters for fit ('amp' and 'mu' parameters in `init_pars`
        dictionary are overwritten by the given `amp` and `x_pos` arguments)
    vary_shape_pars : bool
        Whether to vary or fix peak shape parameters (i.e. sigma, theta,
        eta's and tau's).
    index_first_peak : int
        Index of the first peak to be fit in a multi-peak-fit. Only use this
        during peak shape determination to enforce common shape parameters
        for all peaks to be fitted. (For a regular fit with
        ``vary_shape_pars = False`` this is irrelevant.)

    Returns
    -------
    :class:`lmfit.model.Model`
        `lmfit` model object

    """
    # Define model function
    def emg12(x, amp, mu, sigma, theta, tau_m1,eta_p1,eta_p2,tau_p1,tau_p2):
        return amp*h_emg(x, mu, sigma, theta, (1,),(tau_m1,),(eta_p1,eta_p2),(tau_p1,tau_p2)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = fit.Model(emg12, prefix = pref, nan_policy='propagate')

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp, min=1e-20)
    model.set_param_hint(pref+'mu', value=x_pos, min=x_pos*(1-rel_var_mus), max=x_pos*(1+rel_var_mus))
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, vary=vary_shape_pars)
    model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, max=1, expr='1-'+pref+'eta_p1') # ensures normalization of eta_p's
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)

    # Enfore common shape parameters for all peaks
    # (only needed during peak shape calibration)
    if index_first_peak != None and (peak_index != index_first_peak):
        first_pref = 'p{0}_'.format(index_first_peak)
        model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, expr=first_pref+'sigma')
        model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, expr=first_pref+'theta')
        model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m1')
        model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, expr=first_pref+'eta_p1')
        model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, max=1, expr='1-'+pref+'eta_p1') # ensures normalization of eta_p's
        model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p1')
        model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p2')

    return model


def emg21(peak_index, x_pos, amp, init_pars=pars_dict,
          vary_shape_pars=True, index_first_peak=None):
    """
    Hyper-EMG(2,1) lmfit model (single-peak fit model with two exponential tails
    on the left and one exponential tail on the right)

    Parameters
    ----------
    peak_index :  int
        Index of peak to fit.
    x_pos : float
       Initial guess of peak centroid.
    amp : float
        Initial guess of peak amplitude.
    init_pars : dict
        Initial parameters for fit ('amp' and 'mu' parameters in `init_pars`
        dictionary are overwritten by the given `amp` and `x_pos` arguments)
    vary_shape_pars : bool
        Whether to vary or fix peak shape parameters (i.e. sigma, theta,
        eta's and tau's).
    index_first_peak : int
        Index of the first peak to be fit in a multi-peak-fit. Only use this
        during peak shape determination to enforce common shape parameters
        for all peaks to be fitted. (For a regular fit with
        ``vary_shape_pars = False`` this is irrelevant.)

    Returns
    -------
    :class:`lmfit.model.Model`
        `lmfit` model object

    """
    # Define model function
    def emg21(x, amp, mu, sigma, theta, eta_m1,eta_m2,tau_m1,tau_m2,tau_p1):
        return amp*h_emg(x, mu, sigma, theta, (eta_m1,eta_m2),(tau_m1,tau_m2),(1,),(tau_p1,)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = fit.Model(emg21, prefix = pref, nan_policy='propagate')

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp, min=1e-20)
    model.set_param_hint(pref+'mu', value=x_pos, min=x_pos*(1-rel_var_mus), max=x_pos*(1+rel_var_mus))
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, vary=vary_shape_pars)
    model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1') # ensures normalization of eta_m's
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)

    # Enfore common shape parameters for all peaks
    # (only needed during peak shape calibration)
    if index_first_peak != None and (peak_index != index_first_peak):
        first_pref = 'p{0}_'.format(index_first_peak)
        model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, expr=first_pref+'sigma')
        model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, expr=first_pref+'theta')
        model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, expr=first_pref+'eta_m1' )
        model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1') # ensures normalization of eta_m's
        model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m1')
        model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m2')
        model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p1')

    return model


def emg22(peak_index, x_pos, amp, init_pars=pars_dict,
          vary_shape_pars=True, index_first_peak=None):
    """
    Hyper-EMG(2,2) lmfit model (single-peak fit model with two exponential tails
    on the left and two exponential tails on the right)

    Parameters
    ----------
    peak_index :  int
        Index of peak to fit.
    x_pos : float
       Initial guess of peak centroid.
    amp : float
        Initial guess of peak amplitude.
    init_pars : dict
        Initial parameters for fit ('amp' and 'mu' parameters in `init_pars`
        dictionary are overwritten by the given `amp` and `x_pos` arguments)
    vary_shape_pars : bool
        Whether to vary or fix peak shape parameters (i.e. sigma, theta,
        eta's and tau's).
    index_first_peak : int
        Index of the first peak to be fit in a multi-peak-fit. Only use this
        during peak shape determination to enforce common shape parameters
        for all peaks to be fitted. (For a regular fit with
        ``vary_shape_pars = False`` this is irrelevant.)

    Returns
    -------
    :class:`lmfit.model.Model`
        `lmfit` model object

    """
    # Define model function
    def emg22(x, amp, mu, sigma, theta, eta_m1,eta_m2,tau_m1,tau_m2,eta_p1,eta_p2,tau_p1,tau_p2):
        return amp*h_emg(x, mu, sigma, theta, (eta_m1,eta_m2),(tau_m1,tau_m2),(eta_p1,eta_p2),(tau_p1,tau_p2)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = fit.Model(emg22, prefix = pref, nan_policy='propagate')

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp, min=1e-20)
    model.set_param_hint(pref+'mu', value=x_pos, min=x_pos*(1-rel_var_mus), max=x_pos*(1+rel_var_mus))
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, vary=vary_shape_pars)
    model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1') # ensures normalization of eta_m's
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, max=1, expr='1-'+pref+'eta_p1') # ensures normalization of eta_p's
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)

    # Enfore common shape parameters for all peaks
    # (only needed during peak shape calibration)
    if index_first_peak != None and (peak_index != index_first_peak):
        first_pref = 'p{0}_'.format(index_first_peak)
        model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, expr=first_pref+'sigma')
        model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, expr=first_pref+'theta')
        model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, expr=first_pref+'eta_m1' )
        model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1') # ensures normalization of eta_m's
        model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m1')
        model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m2')
        model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, expr=first_pref+'eta_p1')
        model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, max=1, expr='1-'+pref+'eta_p1') # ensures normalization of eta_p's
        model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p1')
        model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p2')

    return model


def emg23(peak_index, x_pos, amp, init_pars=pars_dict,
          vary_shape_pars=True, index_first_peak=None):
    """
    Hyper-EMG(2,3) lmfit model (single-peak fit model with two exponential tails
    on the left and three exponential tails on the right)

    Parameters
    ----------
    peak_index :  int
        Index of peak to fit.
    x_pos : float
       Initial guess of peak centroid.
    amp : float
        Initial guess of peak amplitude.
    init_pars : dict
        Initial parameters for fit ('amp' and 'mu' parameters in `init_pars`
        dictionary are overwritten by the given `amp` and `x_pos` arguments)
    vary_shape_pars : bool
        Whether to vary or fix peak shape parameters (i.e. sigma, theta,
        eta's and tau's).
    index_first_peak : int
        Index of the first peak to be fit in a multi-peak-fit. Only use this
        during peak shape determination to enforce common shape parameters
        for all peaks to be fitted. (For a regular fit with
        ``vary_shape_pars = False`` this is irrelevant.)

    Returns
    -------
    :class:`lmfit.model.Model`
        `lmfit` model object

    """
    # Define model function
    def emg23(x, amp, mu, sigma, theta, eta_m1,eta_m2,tau_m1,tau_m2,eta_p1,eta_p2,eta_p3,delta_p,tau_p1,tau_p2,tau_p3):
        return amp*h_emg(x, mu, sigma, theta, (eta_m1,eta_m2),(tau_m1,tau_m2),(eta_p1,eta_p2,eta_p3),(tau_p1,tau_p2,tau_p3)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = fit.Model(emg23, prefix = pref, nan_policy='propagate')

    if init_pars['eta_p2'] < 0: # avoids math errors in peak shape uncertainty estimation
        init_pars['eta_p2'] = 0
    if init_pars['eta_p2'] > 1:
        init_pars['eta_p2'] = 1

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp, min=1e-20)
    model.set_param_hint(pref+'mu', value=x_pos, min=x_pos*(1-rel_var_mus), max=x_pos*(1+rel_var_mus))
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, vary=vary_shape_pars)
    model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1') # ensures normalization of eta_m's
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'delta_p', value= init_pars['eta_p1']+init_pars['eta_p2'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, max=1, vary=vary_shape_pars, expr= pref+'delta_p-'+pref+'eta_p1')
    model.set_param_hint(pref+'eta_p3', value= 1-init_pars['eta_p1']-init_pars['eta_p2'], min=0, max=1, expr='1-'+pref+'eta_p1-'+pref+'eta_p2') # ensures normalization of eta_p's
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p3', value= init_pars['tau_p3'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)

    # Enfore common shape parameters for all peaks
    # (only needed during peak shape calibration)
    if index_first_peak != None and (peak_index != index_first_peak):
        first_pref = 'p{0}_'.format(index_first_peak)
        model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, expr=first_pref+'sigma')
        model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, expr=first_pref+'theta')
        model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, expr=first_pref+'eta_m1' )
        model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1') # ensures normalization of eta_m's
        model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m1')
        model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m2')
        model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, expr=first_pref+'eta_p1')
        model.set_param_hint(pref+'delta_p', value= init_pars['eta_p1']+init_pars['eta_p2'], min=0, max=1, expr=first_pref+'delta_p')
        model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, max=1, expr= pref+'delta_p-'+pref+'eta_p1')
        model.set_param_hint(pref+'eta_p3', value= 1-init_pars['eta_p1']-init_pars['eta_p2'], min=0, max=1, expr='1-'+pref+'eta_p1-'+pref+'eta_p2') # ensures norm. of eta_p's
        model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p1')
        model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p2')
        model.set_param_hint(pref+'tau_p3', value= init_pars['tau_p3'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p3')

    return model


def emg32(peak_index, x_pos, amp, init_pars=pars_dict, vary_shape_pars=True,
          index_first_peak=None):
    """
    Hyper-EMG(3,2) lmfit model (single-peak fit model with three exponential
    tails on the left and two exponential tails on the right)

    Parameters
    ----------
    peak_index :  int
        Index of peak to fit.
    x_pos : float
       Initial guess of peak centroid.
    amp : float
        Initial guess of peak amplitude.
    init_pars : dict
        Initial parameters for fit ('amp' and 'mu' parameters in `init_pars`
        dictionary are overwritten by the given `amp` and `x_pos` arguments)
    vary_shape_pars : bool
        Whether to vary or fix peak shape parameters (i.e. sigma, theta,
        eta's and tau's).
    index_first_peak : int
        Index of the first peak to be fit in a multi-peak-fit. Only use this
        during peak shape determination to enforce common shape parameters
        for all peaks to be fitted. (For a regular fit with
        ``vary_shape_pars = False`` this is irrelevant.)

    Returns
    -------
    :class:`lmfit.model.Model`
        `lmfit` model object

    """
    # Define model function
    def emg32(x, amp, mu, sigma, theta, eta_m1,eta_m2,eta_m3,delta_m,tau_m1,tau_m2,tau_m3,eta_p1,eta_p2,tau_p1,tau_p2):
        return amp*h_emg(x, mu, sigma, theta, (eta_m1,eta_m2,eta_m3),(tau_m1,tau_m2,tau_m3),(eta_p1,eta_p2),(tau_p1,tau_p2)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = fit.Model(emg32, prefix = pref, nan_policy='propagate')

    if init_pars['eta_m2'] < 0: # avoids math errors in peak shape uncertainty estimation
        init_pars['eta_m2'] = 0
    if init_pars['eta_m2'] > 1:
        init_pars['eta_m2'] = 1

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp, min=1e-20)
    model.set_param_hint(pref+'mu', value=x_pos, min=x_pos*(1-rel_var_mus), max=x_pos*(1+rel_var_mus))
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, vary=vary_shape_pars)
    model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'delta_m', value= init_pars['eta_m1']+init_pars['eta_m2'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, vary=vary_shape_pars, expr= pref+'delta_m-'+pref+'eta_m1')
    model.set_param_hint(pref+'eta_m3', value= 1-init_pars['eta_m1']-init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1-'+pref+'eta_m2')
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m3', value= init_pars['tau_m3'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p2', value= 1-init_pars['eta_p1'], min=0, max=1, vary=vary_shape_pars, expr= '1-'+pref+'eta_p1') # ensures normalization of eta_p's
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)

    # Enfore common shape parameters for all peaks
    # (only needed during peak shape calibration)
    if index_first_peak != None and (peak_index != index_first_peak):
        first_pref = 'p{0}_'.format(index_first_peak)
        model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, expr=first_pref+'sigma')
        model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, expr=first_pref+'theta')
        model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, expr=first_pref+'eta_m1' )
        model.set_param_hint(pref+'delta_m', value= init_pars['eta_m1']+init_pars['eta_m2'], min=0, max=1, expr= first_pref+'delta_m')
        model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, vary=vary_shape_pars, expr= pref+'delta_m-'+pref+'eta_m1')
        model.set_param_hint(pref+'eta_m3', value= 1-init_pars['eta_m1']-init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1-'+pref+'eta_m2') # ensures normalization of eta_m's
        model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m1')
        model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m2')
        model.set_param_hint(pref+'tau_m3', value= init_pars['tau_m3'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m3')
        model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, expr=first_pref+'eta_p1')
        model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, max=1, expr= '1-'+pref+'eta_p1') # ensures norm. of eta_p's
        model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p1')
        model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p2')

    return model


def emg33(peak_index, x_pos, amp, init_pars=pars_dict,
          vary_shape_pars=True, index_first_peak=None):
    """
    Hyper-EMG(3,3) lmfit model (single-peak fit model with three exponential
    tails on the left and three exponential tails on the right)

    Parameters
    ----------
    peak_index :  int
        Index of peak to fit.
    x_pos : float
       Initial guess of peak centroid.
    amp : float
        Initial guess of peak amplitude.
    init_pars : dict
        Initial parameters for fit ('amp' and 'mu' parameters in `init_pars`
        dictionary are overwritten by the given `amp` and `x_pos` arguments)
    vary_shape_pars : bool
        Whether to vary or fix peak shape parameters (i.e. sigma, theta,
        eta's and tau's).
    index_first_peak : int
        Index of the first peak to be fit in a multi-peak-fit. Only use this
        during peak shape determination to enforce common shape parameters
        for all peaks to be fitted. (For a regular fit with
        ``vary_shape_pars = False`` this is irrelevant.)

    Returns
    -------
    :class:`lmfit.model.Model`
        `lmfit` model object

    """
    # Define model function
    def emg33(x, amp, mu, sigma, theta, eta_m1,eta_m2,eta_m3,delta_m,tau_m1,tau_m2,tau_m3,eta_p1,eta_p2,eta_p3,delta_p,tau_p1,tau_p2,tau_p3):
        return amp*h_emg(x, mu, sigma, theta, (eta_m1,eta_m2,eta_m3),(tau_m1,tau_m2,tau_m3),(eta_p1,eta_p2,eta_p3),(tau_p1,tau_p2,tau_p3)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = fit.Model(emg33, prefix = pref, nan_policy='propagate')

    if init_pars['eta_m2'] < 0: # avoids math errors in peak shape uncertainty estimation
        init_pars['eta_m2'] = 0
    if init_pars['eta_m2'] > 1:
        init_pars['eta_m2'] = 1
    if init_pars['eta_p2'] < 0:
        init_pars['eta_p2'] = 0
    if init_pars['eta_p2'] > 1:
        init_pars['eta_p2'] = 1

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp, min=1e-20)
    model.set_param_hint(pref+'mu', value=x_pos, min=x_pos*(1-rel_var_mus), max=x_pos*(1+rel_var_mus))
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, vary=vary_shape_pars)
    model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'delta_m', value= init_pars['eta_m1']+init_pars['eta_m2'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, vary=vary_shape_pars, expr= pref+'delta_m-'+pref+'eta_m1')
    model.set_param_hint(pref+'eta_m3', value= 1-init_pars['eta_m1']-init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1-'+pref+'eta_m2') # ensures normalization of eta_m's
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m3', value= init_pars['tau_m3'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'delta_p', value= init_pars['eta_p1']+init_pars['eta_p2'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, max=1, vary=vary_shape_pars, expr= pref+'delta_p-'+pref+'eta_p1')
    model.set_param_hint(pref+'eta_p3', value= 1-init_pars['eta_p1']-init_pars['eta_p2'], min=0, max=1, expr='1-'+pref+'eta_p1-'+pref+'eta_p2') # ensures normalization of eta_p's
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p3', value= init_pars['tau_p3'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)

    # Enfore common shape parameters for all peaks
    # (only needed during peak shape calibration)
    if index_first_peak != None and (peak_index != index_first_peak):
        first_pref = 'p{0}_'.format(index_first_peak)
        model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, expr=first_pref+'sigma')
        model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, expr=first_pref+'theta')
        model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, expr=first_pref+'eta_m1' )
        model.set_param_hint(pref+'delta_m', value= init_pars['eta_m1']+init_pars['eta_m2'], min=0, max=1, expr= first_pref+'delta_m')
        model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, expr= pref+'delta_m-'+pref+'eta_m1')
        model.set_param_hint(pref+'eta_m3', value= 1-init_pars['eta_m1']-init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1-'+pref+'eta_m2') # ensures normalization of eta_m's
        model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m1')
        model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m2')
        model.set_param_hint(pref+'tau_m3', value= init_pars['tau_m3'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m3')
        model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, expr=first_pref+'eta_p1')
        model.set_param_hint(pref+'delta_p', value= init_pars['eta_p1']+init_pars['eta_p2'], min=0, max=1, expr= first_pref+'delta_p')
        model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, max=1, expr= pref+'delta_p-'+pref+'eta_p1')
        model.set_param_hint(pref+'eta_p3', value= 1-init_pars['eta_p1']-init_pars['eta_p2'], min=0, max=1, expr='1-'+pref+'eta_p1-'+pref+'eta_p2') # ensures normalization of eta_p's
        model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p1')
        model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p2')
        model.set_param_hint(pref+'tau_p3', value= init_pars['tau_p3'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p3')

    return model
