###############################################################################
##### Module with emgfit models for Gaussian and Hyper-EMG distributions
##### Author: Stefan Paul

##### Import dependencies
import numpy as np
import warnings
from .config import *
from .emg_funcs import *
from .model import EMGModel
from scipy.special import erfcx
from numpy import sqrt, pi

    
###### Define parameter bounds
# Variables ending with _nsigma define bounds in multiples of the initial value
# for the sigma parameter of the underlying Gaussian ('<initial_sigma>')
MU_VAR_NSIGMA = 5 # bound mus to <mu0> +- mu_var_sigma*<initial_sigma>
SIGMA_MIN = 1e-15 # lower bound of sigmas
SIGMA_MAX_NSIGMA = 10 # upper bound of sigmas = SIGMA_MAX_NSIGMA*<initial_sigma>
TAU_MIN = 1e-15 # lower bound of taus
TAU_MAX_NSIGMA = 500 # upper bound of taus = TAU_MAX_NSIGMA*<initial_sigma>
AMP_MIN = 1e-20

def create_default_init_pars(mass_number=100, resolving_power=3e05): 
    """
    Scale default parameters to mass of interest and return parameter dictionary.

    Parameters
    ----------
    mass_number : int, optional
        Atomic mass number of peaks of interest, defaults to 100.
    resolving_power : float, optional
        Typical resolving power of the spectrometer  at FWHM level. 
        Defaults to 3e05.

    Returns
    -------
    dict
        Dictionary with default initial parameters (scaled to `mass_number`).

    Notes
    -----
    **The default parameters were defined for mass 100**, to obtain suitable
    parameters at other masses all mass-dependent parameters (i.e. shape
    parameters & `amp`) are multiplied by the scaling factor `mass_number`/100.

    The standard deviation of the underlying Gaussian :math:`\sigma` is 
    calculated as :math::`\\sigma = A / (R 2 \\sqrt(2 \\ln(2))`, where 
    :math:`A` denotes the specified `mass_number` and :math:`R` is the given 
    `resolving_power`.  

    """
    # Default initial parameters for peaks around mass 100 (with
    # mass scaling factor):
    scl_factor = mass_number/100
    sigma_to_FWHM = 2*np.sqrt(2*np.log(2))  # for Gaussian peak
    amp = 0.45*scl_factor
    mu = None  # flag for below that this is a generic shape parameter set
    sigma = mass_number/(resolving_power*sigma_to_FWHM) # [u]
    theta = 0.5
    eta_m1 = 0.85
    eta_m2 = 0.10
    eta_m3 = 0.05
    tau_m1 =   50e-06*scl_factor # [u]
    tau_m2 =  500e-06*scl_factor # [u]
    tau_m3 = 1000e-06*scl_factor # [u]
    eta_p1 = 0.85
    eta_p2 = 0.10
    eta_p3 = 0.05
    tau_p1 =   50e-06*scl_factor # [u]
    tau_p2 =  500e-06*scl_factor # [u]
    tau_p3 = 1000e-06*scl_factor # [u]
    pars_dict = {'amp': amp, 'mu': mu, 'sigma': sigma, 'theta': theta,
                 'eta_m1': eta_m1, 'eta_m2': eta_m2, 'eta_m3': eta_m3,
                 'tau_m1': tau_m1, 'tau_m2': tau_m2, 'tau_m3': tau_m3,
                 'eta_p1': eta_p1, 'eta_p2': eta_p2, 'eta_p3': eta_p3,
                 'tau_p1': tau_p1, 'tau_p2': tau_p2, 'tau_p3': tau_p3}
    return pars_dict

pars_dict = create_default_init_pars()


def scl_init_pars(init_pars, mu0=None, mu_ref=None, scl_coeff=1.0, decimals=9):
    """Scale initial shape parameters of reference peak to peak at `mu0`.

    If `scl_coeff` is None the original `init_pars` dictionary is returned.

    """
    from copy import deepcopy
    scaled_pars = deepcopy(init_pars)
    for pname, pval in scaled_pars.items():
        if pname.startswith(('sigma','tau')):
            mu_ratio = mu0/mu_ref if mu0 and mu_ref else 1
            scaled_pars[pname] *= scl_coeff*np.round(mu_ratio, decimals)
    return scaled_pars


def _enforce_shared_shape_pars(model, peak_index, index_ref_peak,
                               scale_shape_pars, scl_coeff, mu_ref):
    """Enfore shared peak-shapes and optionally scale the shape parameters

    Parameters
    ----------
    model : :class:`emgfit.model.EMGModel`
        Fit model to impose parameter constraints on.
    peak_index : int
        Index of the :class:`~emgfit.spectrum.peak` corresponding to `model`.
    index_ref_peak : int
        Index of the :class:`~emgfit.spectrum.peak` to be used as
        shape-reference peak.
    scale_shape_pars : bool
        Whether to scale the scale-dependent shape parameters adopted from the
        shape-reference peak.
    scl_coeff : float, optional
        Constant coefficient used to scale the scale-dependent shape parameters.
    mu_ref : float or str, optional
        Centroid of the underlying Gaussian of the shape-reference peak. This
        argument is only relevant when ``scale_shape_pars=True``. If `mu_ref` is
        set to a float, this number is used as the fixed (Gaussian) reference
        centroid for calculating the scale factor. If `mu_ref="varying"`,
        `mu_ref` is set to the varying (Gaussian) centroid shape-reference peak.

    Notes
    -----
    If `scale_shape_pars` is True, the model's scale-dependent shape parameters
    are multiplied with the scale factor ``scl_fac = scl_coeff``. If further
    `mu_ref` is not None, the scale-dependent shape parameters are multiplied
    with the scale factor ``scl_fac = mu/mu_ref * scl_coeff``, where `mu` is the
    centroid of the Gaussian underlying the specified `model`.  When the
    reference peak is among the peaks to be fitted, ``scl_fac`` will then be
    dynamically re-calculated from the `mu` and `mu_ref` values of a given
    iteration in a fit.

    """
    pref = 'p{0}_'.format(peak_index)
    ref_pref = 'p{0}_'.format(index_ref_peak)
    if scale_shape_pars is True:
        if mu_ref is None:
            model.set_param_hint(pref+'scl_fac', expr=str(scl_coeff))
        else:
            model.set_param_hint(pref+'scl_fac',
                                 expr=pref+'mu/'+ref_pref+'mu * '+str(scl_coeff))

        for ppname in model.param_names: # Set parameter constraints
            pname = model._strip_prefix(ppname)
            if pname.startswith(('theta','eta','delta')):
                model.set_param_hint(pref+pname, expr=ref_pref+pname)
            elif pname.startswith(('sigma','tau')):
                model.set_param_hint(pref+pname,
                                     expr=pref+'scl_fac * '+ref_pref+pname)
    else:
        for ppname in model.param_names: # Set parameter constraints
            pname = model._strip_prefix(ppname)
            if pname.startswith(('sigma','theta','eta','tau','delta')):
                model.set_param_hint(pref+pname, expr=ref_pref+pname)

    return model


def _calc_mu_emg(fit_model, pars, pref=""):
    """Calculate the mean of a Hyper-EMG peak from the best-fit parameters.

    Parameters
    ----------
    fit_model :
        Name of fit model used to obtain `pars`.
    pars : :class:`emgfit.parameter.Parameters`, optional
        Parameters object to obtain shape parameters for calculation from.
        This argument can be used when no fit result is available.
    pref : str, optional
        Prefix of peak parameters of interest.

    Returns
    -------
    float [u/z]
        Mean of Hyper-EMG fit of peak of interest.

    """
    if fit_model.startswith("emg"):
        N_left_tails = int(fit_model[3])
        N_right_tails = int(fit_model[4])
        li_eta_m, li_tau_m, li_eta_p, li_tau_p = [],[],[],[]
        for i in np.arange(1,N_left_tails+1):
            if N_left_tails == 1:
                li_eta_m = [1]
            else:
                li_eta_m.append(pars[pref+'eta_m'+str(i)].value)
            li_tau_m.append(pars[pref+'tau_m'+str(i)].value)
        for i in np.arange(1,N_right_tails+1):
            if N_right_tails == 1:
                li_eta_p = [1]
            else:
                li_eta_p.append(pars[pref+'eta_p'+str(i)].value)
            li_tau_p.append(pars[pref+'tau_p'+str(i)].value)
        if N_left_tails == 0:
            theta = 0
        elif N_right_tails == 0:
            theta = 1
        else:
            theta = pars[pref+'theta'].value
        mu_EMG = mu_emg(pars[pref+'mu'].value,
                        theta,
                        tuple(li_eta_m),tuple(li_tau_m),
                        tuple(li_eta_p),tuple(li_tau_p) )
    elif fit_model == "Gaussian":
        mu_EMG = pars[pref+'mu'].value
    else:
        raise Exception("`fit_result` is not from a Hyper-EMG fit.")

    return mu_EMG


def erfcxinv(y):
    """Approximate inverse of the scaled complementary error function

    This approximation is adapted from code by John D'Errico posted at
    https://www.mathworks.com/matlabcentral/answers/302915-inverse-of-erfcx-scaled-complementary-error-function

    """
    # erfcx inverse, for y no larger than 1.
    # for y <= 1, use the large x approximation for a starting value.
    mask = (y <= 1)
    ret = np.zeros_like(y)
    ret[mask] = 1./(y[mask]*sqrt(pi))
    # for y > 1, use exp(x^2) as a very rough approximation
    # to erfcx
    ret[~mask] = -sqrt(np.log(y[~mask]))
    for n in range(7):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            denom = 2*ret*erfcx(ret) - 2/sqrt(pi)
            ret += np.where(denom==0, 0.0, - (erfcx(ret) - y)/denom)
    return ret


def get_mu0(x_m, init_pars, fit_model):
    """Estimate initial value of Gaussian centroid `mu` from the peak's mode

    Parameters
    ----------
    x_m : float
        Mode (i.e. x-position of the maximum) of the distribution.
    init_pars : dict
        Dictionary with initial values of the shape parameters.
    fit_model : str
        Name of used fit model (e.g. 'emg01', 'emg10', 'emg12'...).

    Notes
    -----
    For a Gaussian, the mean `mu` is simply equal to the `x_m` argument.

    For highly asymmetric hyper-EMG distributions the centroid of the underlying
    Gaussian `mu` can strongly deviate from the mode :math:`x_{m}` (i.e. the
    x-position of the peak maximum). Hence, the initial Gaussian centroid `mu`
    (:math:`\\mu`) is calculated by rearranging the equation for the mode of the
    hyper-EMG distribution:

    .. math::

      \\mu = x_{m}
             - \\theta\\sum_{i=1}^{N_-}\\eta_{-i}\\left(\\sqrt{2}\\sigma
               \\cdot\\mathrm{erfcxinv}\\left( \\frac{\\tau_{-i}}{\\sigma}
               \\sqrt{\\frac{2}{\\pi}}\\right) - \\frac{\\sigma^2}{\\tau_{-i}}
               \\right) \\\\
             + (1-\\theta)\\sum_{i=1}^{N_-}\\eta_{+i}\\left(\\sqrt{2}\\sigma
               \\cdot\\mathrm{erfcxinv}\\left( \\frac{\\tau_{+i}}{\\sigma}
               \\sqrt{\\frac{2}{\\pi}}\\right) - \\frac{\\sigma^2}{\\tau_{-i}}
               \\right),

    where the mode :math:`x_{m}` can be estimated by the peak marker position
    `x_pos` and :func:`emgfit.fit_models.erfcxinv` is the inverse of the scaled
    complementary error function.

    """
    if fit_model == 'Gaussian':
        mu0 = x_m
    elif fit_model.startswith('emg') and len(fit_model) == 5:
        sigma = init_pars['sigma']
        t_order_m = int(fit_model[-2])
        sum_m = 0
        if t_order_m == 1:
            tau_m1 = init_pars['tau_m1']
            sum_m = (sqrt(2)*sigma*erfcxinv(tau_m1/sigma*sqrt(2/pi))
                     - sigma**2/tau_m1)
        else:
            for i in range(t_order_m):
                eta_mi = init_pars['eta_m{}'.format(i+1)]
                tau_mi = init_pars['tau_m{}'.format(i+1)]
                sum_m += eta_mi*(sqrt(2)*sigma*erfcxinv(tau_mi/sigma*sqrt(2/pi))
                                 - sigma**2/tau_mi)

        t_order_p = int(fit_model[-1])
        sum_p = 0
        if t_order_p == 1:
            tau_p1 = init_pars['tau_p1']
            sum_p = (sqrt(2)*sigma*erfcxinv(tau_p1/sigma*sqrt(2/pi))
                     - sigma**2/tau_p1)
        else:
            for i in range(t_order_p):
                eta_pi = init_pars['eta_p{}'.format(i+1)]
                tau_pi = init_pars['tau_p{}'.format(i+1)]
                sum_p += eta_pi*(sqrt(2)*sigma*erfcxinv(tau_pi/sigma*sqrt(2/pi))
                                 - sigma**2/tau_pi)

        if t_order_p == 1 and t_order_m == 0:
            theta = 0
        elif t_order_p == 0 and t_order_m == 1:
            theta = 1
        else:
            theta = init_pars['theta']

        mu0 = x_m - theta*sum_m + (1-theta)*sum_p
    else:
        msg = str("fit_model must be 'Gaussian' or 'emgXY' with X & Y denoting the tail orders!")
        raise Exception(msg)
    
    return mu0


###############################################################################
##### Define emgfit fit models
def Gaussian(peak_index, mu0, amp0, init_pars=pars_dict, vary_shape_pars=True, **kws):
    """
    Gaussian emgfit model (single-peak Gaussian fit model)

    Parameters
    ----------
    peak_index :  int
        Index of peak to fit.
    mu0 : float
        Initial guess for centroid of underlying Gaussian.
    amp0 : float
        Initial guess for peak amplitude.
    init_pars : dict
        Initial shape parameters ('amp' and 'mu' parameters in `init_pars`
        dictionary are updated with the given values for `amp0` and `mu0`).
    vary_shape_pars : bool
        Whether to vary or fix peak shape parameters (i.e. sigma, theta,
        eta's and tau's).
    kws : Keyword arguments to pass to EMGModel interface.

    Returns
    -------
    :class:`emgfit.model.EMGModel`
        `emgfit` model object

    """
    # Define model function
    def Gaussian(x, amp, mu, sigma):
        return  amp/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*sigma**2))
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak
    model = EMGModel(Gaussian, prefix=pref, nan_policy='propagate', 
                     vary_shape=vary_shape_pars, **kws)

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp0, min=0)
    model.set_param_hint(pref+'mu', value=mu0, min=mu0 - MU_VAR_NSIGMA*init_pars['sigma'], max=mu0 + MU_VAR_NSIGMA*init_pars['sigma'])
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=SIGMA_MIN, max=SIGMA_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)

    return model


def emg01(peak_index, mu0, amp0, init_pars=pars_dict, vary_shape_pars=True, **kws):
    """
    Hyper-EMG(0,1) emgfit model (single-peak fit model with one exponential 
    tail on the right)

    Parameters
    ----------
    peak_index :  int
        Index of peak to fit.
    mu0 : float
        Initial guess for centroid of underlying Gaussian.
    amp0 : float
        Initial guess for peak amplitude.
    init_pars : dict
        Initial shape parameters ('amp' and 'mu' parameters in `init_pars`
        dictionary are updated with the given values for `amp0` and `mu0`).
    vary_shape_pars : bool
        Whether to vary or fix peak shape parameters (i.e. sigma, theta,
        eta's and tau's).
    kws : Keyword arguments to pass to EMGModel interface.

    Returns
    -------
    :class:`emgfit.model.EMGModel`
        `emgfit` model object

    """
    # Define model function
    def emg01(x, amp, mu, sigma, tau_p1):
        return amp*h_emg(x, mu, sigma, 0, (0,),(0,),(1,),(tau_p1,))
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak
    model = EMGModel(emg01, prefix = pref, nan_policy='propagate', 
                     vary_shape=vary_shape_pars, **kws)

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp0, min=AMP_MIN)
    model.set_param_hint(pref+'mu', value=mu0, min=mu0 - MU_VAR_NSIGMA*init_pars['sigma'], max=mu0 + MU_VAR_NSIGMA*init_pars['sigma'])
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=SIGMA_MIN, max=SIGMA_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)

    return model


def emg10(peak_index, mu0, amp0, init_pars=pars_dict, vary_shape_pars=True, **kws):
    """
    Hyper-EMG(1,0) emgfit model (single-peak fit model with one exponential 
    tail on the left)

    Parameters
    ----------
    peak_index :  int
        Index of peak to fit.
    mu0 : float
        Initial guess for centroid of underlying Gaussian.
    amp0 : float
        Initial guess for peak amplitude.
    init_pars : dict
        Initial shape parameters ('amp' and 'mu' parameters in `init_pars`
        dictionary are updated with the given values for `amp0` and `mu0`).
    vary_shape_pars : bool
        Whether to vary or fix peak shape parameters (i.e. sigma, theta,
        eta's and tau's).
    kws : Keyword arguments to pass to EMGModel interface.

    Returns
    -------
    :class:`emgfit.model.EMGModel`
        `emgfit` model object

    """
    # Define model function
    def emg10(x, amp, mu, sigma, tau_m1):
        return amp*h_emg(x, mu, sigma, 1, (1,),(tau_m1,),(0,),(0,))
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak
    model = EMGModel(emg10, prefix = pref, nan_policy='propagate', 
                     vary_shape=vary_shape_pars, **kws)

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp0, min=AMP_MIN)
    model.set_param_hint(pref+'mu', value=mu0, min=mu0 - MU_VAR_NSIGMA*init_pars['sigma'], max=mu0 + MU_VAR_NSIGMA*init_pars['sigma'])
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=SIGMA_MIN, max=SIGMA_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)

    return model


def emg11(peak_index, mu0, amp0, init_pars=pars_dict, vary_shape_pars=True, **kws):
    """
    Hyper-EMG(1,1) emgfit model (single-peak fit model with one exponential 
    tail on the left and one exponential tail on the right)

    Parameters
    ----------
    peak_index :  int
        Index of peak to fit.
    mu0 : float
        Initial guess for centroid of underlying Gaussian.
    amp0 : float
        Initial guess for peak amplitude.
    init_pars : dict
        Initial shape parameters ('amp' and 'mu0' parameters in `init_pars`
        dictionary are updated with the given values for `amp0` and `mu0`).
    vary_shape_pars : bool
        Whether to vary or fix peak shape parameters (i.e. sigma, theta,
        eta's and tau's).
    kws : Keyword arguments to pass to EMGModel interface.

    Returns
    -------
    :class:`emgfit.model.EMGModel`
        `emgfit` model object

    """
    # Define model function
    def emg11(x, amp, mu, sigma, theta, tau_m1, tau_p1):
        return amp*h_emg(x, mu, sigma, theta, (1,),(tau_m1,),(1,),(tau_p1,)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak
    model = EMGModel(emg11, prefix = pref, nan_policy='propagate', 
                     vary_shape=vary_shape_pars, **kws)

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp0, min=AMP_MIN)
    model.set_param_hint(pref+'mu', value=mu0, min=mu0 - MU_VAR_NSIGMA*init_pars['sigma'], max=mu0 + MU_VAR_NSIGMA*init_pars['sigma'])
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=SIGMA_MIN, max=SIGMA_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)

    return model


def emg12(peak_index, mu0, amp0, init_pars=pars_dict, vary_shape_pars=True, **kws):
    """
    Hyper-EMG(1,2) emgfit model (single-peak fit model with one exponential 
    tail on the left and two exponential tails on the right)

    Parameters
    ----------
    peak_index :  int
        Index of peak to fit.
    mu0 : float
        Initial guess for centroid of underlying Gaussian.
    amp0 : float
        Initial guess for peak amplitude.
    init_pars : dict
        Initial shape parameters ('amp' and 'mu0' parameters in `init_pars`
        dictionary are updated with the given values for `amp0` and `mu0`).
    vary_shape_pars : bool
        Whether to vary or fix peak shape parameters (i.e. sigma, theta,
        eta's and tau's).
    kws : Keyword arguments to pass to EMGModel interface.

    Returns
    -------
    :class:`emgfit.model.EMGModel`
        `emgfit` model object

    """
    # Define model function
    def emg12(x, amp, mu, sigma, theta, tau_m1,eta_p1,eta_p2,tau_p1,tau_p2):
        return amp*h_emg(x, mu, sigma, theta, (1,),(tau_m1,),(eta_p1,eta_p2),(tau_p1,tau_p2)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = EMGModel(emg12, prefix = pref, nan_policy='propagate', 
                     vary_shape=vary_shape_pars, **kws)

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp0, min=AMP_MIN)
    model.set_param_hint(pref+'mu', value=mu0, min=mu0 - MU_VAR_NSIGMA*init_pars['sigma'], max=mu0 + MU_VAR_NSIGMA*init_pars['sigma'])
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=SIGMA_MIN, max=SIGMA_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, max=1, expr='1-'+pref+'eta_p1') # ensures normalization of eta_p's
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)

    return model


def emg21(peak_index, mu0, amp0, init_pars=pars_dict, vary_shape_pars=True, **kws):
    """
    Hyper-EMG(2,1) emgfit model (single-peak fit model with two exponential 
    tails on the left and one exponential tail on the right)

    Parameters
    ----------
    peak_index :  int
        Index of peak to fit.
    mu0 : float
        Initial guess for centroid of underlying Gaussian.
    amp0 : float
        Initial guess for peak amplitude.
    init_pars : dict
        Initial shape parameters ('amp' and 'mu' parameters in `init_pars`
        dictionary are updated with the given values for `amp0` and `mu0`).
    vary_shape_pars : bool
        Whether to vary or fix peak shape parameters (i.e. sigma, theta,
        eta's and tau's).
    kws : Keyword arguments to pass to EMGModel interface.

    Returns
    -------
    :class:`emgfit.model.EMGModel`
        `emgfit` model object

    """
    # Define model function
    def emg21(x, amp, mu, sigma, theta, eta_m1,eta_m2,tau_m1,tau_m2,tau_p1):
        return amp*h_emg(x, mu, sigma, theta, (eta_m1,eta_m2),(tau_m1,tau_m2),(1,),(tau_p1,)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = EMGModel(emg21, prefix = pref, nan_policy='propagate', 
                     vary_shape=vary_shape_pars, **kws)

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp0, min=AMP_MIN)
    model.set_param_hint(pref+'mu', value=mu0, min=mu0 - MU_VAR_NSIGMA*init_pars['sigma'], max=mu0 + MU_VAR_NSIGMA*init_pars['sigma'])
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=SIGMA_MIN, max=SIGMA_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1') # ensures normalization of eta_m's
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)

    return model


def emg22(peak_index, mu0, amp0, init_pars=pars_dict, vary_shape_pars=True, **kws):
    """
    Hyper-EMG(2,2) emgfit model (single-peak fit model with two exponential 
    tails on the left and two exponential tails on the right)

    Parameters
    ----------
    peak_index :  int
        Index of peak to fit.
    mu0 : float
        Initial guess for centroid of underlying Gaussian.
    amp0 : float
        Initial guess for peak amplitude.
    init_pars : dict
        Initial shape parameters of the reference peak ('amp' and 'mu'
        parameters in `init_pars` dictionary are updated with the given values
        for `amp0` and `mu0`).
    vary_shape_pars : bool
        Whether to vary or fix peak shape parameters (i.e. sigma, theta,
        eta's and tau's).
    kws : Keyword arguments to pass to EMGModel interface.

    Returns
    -------
    :class:`emgfit.model.EMGModel`
        `emgfit` model object

    """
    # Define model function
    def emg22(x, amp, mu, sigma, theta, eta_m1,eta_m2,tau_m1,tau_m2,eta_p1,eta_p2,tau_p1,tau_p2):
        return amp*h_emg(x, mu, sigma, theta, (eta_m1,eta_m2),(tau_m1,tau_m2),(eta_p1,eta_p2),(tau_p1,tau_p2)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = EMGModel(emg22, prefix = pref, nan_policy='propagate', 
                     vary_shape=vary_shape_pars, **kws)
    
    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp0, min=AMP_MIN)
    model.set_param_hint(pref+'mu', value=mu0, min=mu0 - MU_VAR_NSIGMA*init_pars['sigma'], max=mu0 + MU_VAR_NSIGMA*init_pars['sigma'])
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=SIGMA_MIN, max=SIGMA_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1') # ensures normalization of eta_m's
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, max=1, expr='1-'+pref+'eta_p1') # ensures normalization of eta_p's
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)

    return model


def emg23(peak_index, mu0, amp0, init_pars=pars_dict, vary_shape_pars=True, **kws):
    """
    Hyper-EMG(2,3) emgfit model (single-peak fit model with two exponential 
    tails on the left and three exponential tails on the right)

    Parameters
    ----------
    peak_index :  int
        Index of peak to fit.
    mu0 : float
        Initial guess for centroid of underlying Gaussian.
    amp0 : float
        Initial guess for peak amplitude.
    init_pars : dict
        Initial shape parameters ('amp' and 'mu' parameters in `init_pars`
        dictionary are updated with the given values for `amp0` and `mu0`).
    vary_shape_pars : bool
        Whether to vary or fix peak shape parameters (i.e. sigma, theta,
        eta's and tau's).
    kws : Keyword arguments to pass to EMGModel interface.

    Returns
    -------
    :class:`emgfit.model.EMGModel`
        `emgfit` model object

    """
    # Define model function
    def emg23(x, amp, mu, sigma, theta, eta_m1,eta_m2,tau_m1,tau_m2,eta_p1,eta_p2,eta_p3,delta_p,tau_p1,tau_p2,tau_p3):
        return amp*h_emg(x, mu, sigma, theta, (eta_m1,eta_m2),(tau_m1,tau_m2),(eta_p1,eta_p2,eta_p3),(tau_p1,tau_p2,tau_p3)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = EMGModel(emg23, prefix = pref, nan_policy='propagate', 
                     vary_shape=vary_shape_pars, **kws)

    if init_pars['eta_p2'] < 0: # avoids math errors in peak shape uncertainty estimation
        init_pars['eta_p2'] = 0
    if init_pars['eta_p2'] > 1:
        init_pars['eta_p2'] = 1

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp0, min=AMP_MIN)
    model.set_param_hint(pref+'mu', value=mu0, min=mu0 - MU_VAR_NSIGMA*init_pars['sigma'], max=mu0 + MU_VAR_NSIGMA*init_pars['sigma'])
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=SIGMA_MIN, max=SIGMA_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1') # ensures normalization of eta_m's
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'delta_p', value= init_pars['eta_p1']+init_pars['eta_p2'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, max=1, vary=vary_shape_pars, expr= pref+'delta_p-'+pref+'eta_p1')
    model.set_param_hint(pref+'eta_p3', value= 1-init_pars['eta_p1']-init_pars['eta_p2'], min=0, max=1, expr='1-'+pref+'eta_p1-'+pref+'eta_p2') # ensures normalization of eta_p's
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p3', value= init_pars['tau_p3'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)

    return model


def emg32(peak_index, mu0, amp0, init_pars=pars_dict, vary_shape_pars=True, **kws):
    """
    Hyper-EMG(3,2) emgfit model (single-peak fit model with three exponential
    tails on the left and two exponential tails on the right)

    Parameters
    ----------
    peak_index :  int
        Index of peak to fit.
    mu0 : float
        Initial guess for centroid of underlying Gaussian.
    amp0 : float
        Initial guess for peak amplitude.
    init_pars : dict
        Initial shape parameters ('amp' and 'mu' parameters in `init_pars`
        dictionary are updated with the given values for `amp0` and `mu0`).
    vary_shape_pars : bool
        Whether to vary or fix peak shape parameters (i.e. sigma, theta,
        eta's and tau's).
    kws : Keyword arguments to pass to EMGModel interface.

    Returns
    -------
    :class:`emgfit.model.EMGModel`
        `emgfit` model object

    """
    # Define model function
    def emg32(x, amp, mu, sigma, theta, eta_m1,eta_m2,eta_m3,delta_m,tau_m1,tau_m2,tau_m3,eta_p1,eta_p2,tau_p1,tau_p2):
        return amp*h_emg(x, mu, sigma, theta, (eta_m1,eta_m2,eta_m3),(tau_m1,tau_m2,tau_m3),(eta_p1,eta_p2),(tau_p1,tau_p2)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = EMGModel(emg32, prefix = pref, nan_policy='propagate', 
                     vary_shape=vary_shape_pars, **kws)

    if init_pars['eta_m2'] < 0: # avoids math errors in peak shape uncertainty estimation
        init_pars['eta_m2'] = 0
    if init_pars['eta_m2'] > 1:
        init_pars['eta_m2'] = 1

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp0, min=AMP_MIN)
    model.set_param_hint(pref+'mu', value=mu0, min=mu0 - MU_VAR_NSIGMA*init_pars['sigma'], max=mu0 + MU_VAR_NSIGMA*init_pars['sigma'])
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=SIGMA_MIN, max=SIGMA_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'delta_m', value= init_pars['eta_m1']+init_pars['eta_m2'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, vary=vary_shape_pars, expr= pref+'delta_m-'+pref+'eta_m1')
    model.set_param_hint(pref+'eta_m3', value= 1-init_pars['eta_m1']-init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1-'+pref+'eta_m2')
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m3', value= init_pars['tau_m3'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p2', value= 1-init_pars['eta_p1'], min=0, max=1, vary=vary_shape_pars, expr= '1-'+pref+'eta_p1') # ensures normalization of eta_p's
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)

    return model


def emg33(peak_index, mu0, amp0, init_pars=pars_dict, vary_shape_pars=True, **kws):
    """
    Hyper-EMG(3,3) emgfit model (single-peak fit model with three exponential
    tails on the left and three exponential tails on the right)

    Parameters
    ----------
    peak_index :  int
        Index of peak to fit.
    mu0 : float
        Initial guess for centroid of underlying Gaussian.
    amp0 : float
        Initial guess for peak amplitude.
    init_pars : dict
        Initial shape parameters ('amp' and 'mu' parameters in `init_pars`
        dictionary are updated with the given values for `amp0` and `mu0`).
    vary_shape_pars : bool
        Whether to vary or fix peak shape parameters (i.e. sigma, theta,
        eta's and tau's).
    kws : Keyword arguments to pass to EMGModel interface.

    Returns
    -------
    :class:`emgfit.model.EMGModel`
        `emgfit` model object

    """
    # Define model function
    def emg33(x, amp, mu, sigma, theta, eta_m1,eta_m2,eta_m3,delta_m,tau_m1,tau_m2,tau_m3,eta_p1,eta_p2,eta_p3,delta_p,tau_p1,tau_p2,tau_p3):
        return amp*h_emg(x, mu, sigma, theta, (eta_m1,eta_m2,eta_m3),(tau_m1,tau_m2,tau_m3),(eta_p1,eta_p2,eta_p3),(tau_p1,tau_p2,tau_p3)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = EMGModel(emg33, prefix = pref, nan_policy='propagate', 
                     vary_shape=vary_shape_pars, **kws)

    if init_pars['eta_m2'] < 0: # avoids math errors in peak shape uncertainty estimation
        init_pars['eta_m2'] = 0
    if init_pars['eta_m2'] > 1:
        init_pars['eta_m2'] = 1
    if init_pars['eta_p2'] < 0:
        init_pars['eta_p2'] = 0
    if init_pars['eta_p2'] > 1:
        init_pars['eta_p2'] = 1

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp0, min=AMP_MIN)
    model.set_param_hint(pref+'mu', value=mu0, min=mu0 - MU_VAR_NSIGMA*init_pars['sigma'], max=mu0 + MU_VAR_NSIGMA*init_pars['sigma'])
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=SIGMA_MIN, max=SIGMA_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'delta_m', value= init_pars['eta_m1']+init_pars['eta_m2'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, vary=vary_shape_pars, expr= pref+'delta_m-'+pref+'eta_m1')
    model.set_param_hint(pref+'eta_m3', value= 1-init_pars['eta_m1']-init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1-'+pref+'eta_m2') # ensures normalization of eta_m's
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m3', value= init_pars['tau_m3'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'delta_p', value= init_pars['eta_p1']+init_pars['eta_p2'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, max=1, vary=vary_shape_pars, expr= pref+'delta_p-'+pref+'eta_p1')
    model.set_param_hint(pref+'eta_p3', value= 1-init_pars['eta_p1']-init_pars['eta_p2'], min=0, max=1, expr='1-'+pref+'eta_p1-'+pref+'eta_p2') # ensures normalization of eta_p's
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p3', value= init_pars['tau_p3'], min=TAU_MIN, max=TAU_MAX_NSIGMA*init_pars['sigma'], vary=vary_shape_pars)

    return model


###############################################################################
###### Define emgfit ConstantModel class
class ConstantModel(EMGModel):
    """Constant background model""" 
    def __init__(self, cost_func='default', independent_vars=['x'], 
                 prefix='bkg', nan_policy='propagate',**kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        """Create constant background model 

        Parameters 
        ----------
        cost_func : str, optional
            Name of cost function to use for minimization - overrides the 
            model's :attr:`~lmfit.model.Model._residual` attribute. 

            - If ``'chi-square'``, the fit is performed by minimizing Pearson's
              chi-squared statistic:

              .. math::

                  \\chi^2_P = \\sum_i \\frac{(f(x_i) - y_i)^2}{f(x_i)}.

            - If ``'MLE'``, a binned maximum likelihood estimation is performed
              by minimizing the (doubled) negative log likelihood ratio:

              .. math::

                  L = 2\\sum_i \\left[ f(x_i) - y_i + y_i ln\\left(\\frac{y_i}{f(x_i)}\\right)\\right]

            - If ``'default'`` (default), use the standard residual from lmfit.

            See `Notes` of :meth:`~emgfit.spectrum.spectrum.peakfit` method for 
            more details.
        independent_vars : :obj:`list` of :obj:`str`, optional
            Arguments to `func` that are independent variables (default is
            None).
            parameters (default is None).
        prefix : str, optional
            Prefix used for the model.
        nan_policy : {'raise', 'propagate', 'omit'}, optional
            How to handle NaN and missing values in data. See Notes below.
        **kws : dict, optional
            Additional keyword arguments to pass to model function.

        """

        def constant(x, c=0.0):
            return c * np.ones(np.shape(x))
        super().__init__(constant, cost_func=cost_func, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = self.make_params()

        pars[f'{self.prefix}c'].set(value=data.mean())
        return update_param_vals(pars, self.prefix, **kwargs)
