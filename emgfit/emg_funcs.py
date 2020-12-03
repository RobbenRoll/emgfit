################################################################################
##### Module with numerically robust implementation of the hyper-exponentially-
##### modified Gaussian probability density function
##### Author: Stefan Paul

##### Import packages
import numpy as np
import lmfit as fit
from numpy import exp
from math import sqrt
import scipy.special.cython_special
from scipy.special import erfc, erfcx
from numba.extending import get_cython_function_address
from numba import vectorize, njit
import ctypes
import mpmath as mp
erfc_mp = np.frompyfunc(mp.erfc,1,1)
exp_mp = np.frompyfunc(mp.exp,1,1)

norm_precision = 1e-06 # level on which eta parameters must agree with unity

################################################################################
##### Define numba versions of scipy.special's erfc and erfcx functions using
##### the corresponding C functions from scipy.special.cython_special
erfc_addr = get_cython_function_address("scipy.special.cython_special",
                                        "__pyx_fuse_1erfc")
erfc_functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
c_erfc = erfc_functype(erfc_addr)

@vectorize('float64(float64)')
def _vec_erfc(x):
    return c_erfc(x)

@njit
def _erfc_jit(arg):
    return _vec_erfc(arg)

erfcx_addr = get_cython_function_address("scipy.special.cython_special",
                                         "__pyx_fuse_1erfcx")
erfcx_functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
c_erfcx = erfcx_functype(erfcx_addr)

@vectorize('float64(float64)')
def _vec_erfcx(x):
    return c_erfcx(x)

@njit
def _erfcx_jit(arg):
    return _vec_erfcx(arg)


################################################################################
##### Define general Hyper-EMG functions

@njit
def h_m_i(x,mu,sigma,eta_m,tau_m):
    """Internal helper function to calculate single negative EMG tail for
    h_m_emg."""
    erfcarg = np.atleast_1d(sigma/(sqrt(2)*tau_m) + (x-mu)/(sqrt(2)*sigma))
    mask = (erfcarg < 0)
    ret = np.empty_like(x)
    # Use Gauss*erfcx formulation to avoid overflow of exp and underflow of
    # erfc at larger pos. arguments:
    Gauss_erfcx = exp( -0.5*((x[~mask]-mu)/sigma)**2 )*_erfcx_jit(erfcarg[~mask])
    ret[~mask] = eta_m/(2*tau_m)*Gauss_erfcx
    # Use exp*erfc formulation to avoid overflow of erfcx at larger neg.
    # arguments:
    exp_erfc = exp(0.5*(sigma/tau_m)**2 + (x[mask]-mu)/tau_m)*_erfc_jit(erfcarg[mask])
    ret[mask] = 0.5*eta_m/tau_m*exp_erfc

    return ret


def h_m_i_prec(x,mu,sigma,eta_m,tau_m):
    """Arbitrary precision version of internal helper function for h_m_emg."""
    expval = exp_mp( 0.5*(sigma/tau_m)**2 + (x-mu)/tau_m )
    erfcval = erfc_mp( sigma/(sqrt(2)*tau_m) + (x-mu)/(sqrt(2)*sigma) )
    ret = 0.5*eta_m/tau_m*expval*erfcval
    return ret.astype(float)


@njit
def h_m_emg(x, mu, sigma, li_eta_m,li_tau_m):
    """Negative skewed exponentially-modified Gaussian (EMG) distribution.

    The lengths of `li_eta_m` & `li_tau_m` must match and define the order of
    negative tails.

    Parameters
    ----------
    x  : float >= 0
        Abscissa data (mass data).
    mu : float >= 0
        Mean value of underlying Gaussian distribution.
    sigma : float >= 0
        Standard deviation of underlying Gaussian distribution.
    li_eta_m : tuple
        Tuple containing the neg. tail weights with the signature:
        ``(eta_m1, eta_m2, ...)``.
    li_tau_m : tuple
        Tuple containing the neg. tail decay constants with the signature:
        ``(tau_m1, tau_m2, ...)``.

    Returns
    -------
    float
        Ordinate values of the negative skewed EMG distribution.

    Notes
    -----
    The Hyper-EMG probability distribution function was first introduced in
    `this publication`_ by Purushothaman et al. [#]_. The basic definitions and
    notations used here are adapted from this work.

    Each negative tail of a Hyper-EMG function can be expressed in two
    equivalent ways:

    .. math::

        h_\mathrm{emg,-i} = \\frac{\\eta_{-i}}{2\\tau_{-i}} \\exp{(-\\left(\\frac{x-\\mu}{\\sqrt{2}\\sigma}\\right)^2)} \mathrm{erfcx}(v)
        = \\frac{\\eta_{-i}}{2\\tau_{-i}} \\exp{(u)} \mathrm{erfc}(v),

    where :math:`u = \\frac{\\sigma}{\\sqrt{2}\\tau_{-i}} + \\frac{x-\mu}{\\sqrt{2}\\tau_{-i}}`
    and :math:`v = \\frac{\\sigma}{\\sqrt{2}\\tau_{-i}} + \\frac{x-\mu}{\\sqrt{2}\\sigma}`.
    In double float precision, the `exp(u)`_ routine overflows if u > 709.78. The
    complementary error function `erfc(v)`_ underflows to 0.0 if v > 26.54. The
    scaled complementary error function `erfcx(v)`_ overflows if v < -26.62. To
    circumvent those scenarios and always ensure an exact result, the underlying
    helper function for the calculation of a negative EMG tail :func:`h_m_i`
    uses the formulation in terms of `erfcx` whenever v >= 0 and switches to the
    `erfc`-formulation when v < 0.

    .. _`exp(u)`: https://numpy.org/doc/stable/reference/generated/numpy.exp.html#numpy.exp
    .. _`erfc(v)`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erfc.html
    .. _`erfcx(v)`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erfcx.html
    .. _`this publication`: https://www.sciencedirect.com/science/article/abs/pii/S1387380616302913

    References
    ----------
    .. [#] Purushothaman, S., et al. "Hyper-EMG: A new probability distribution
       function composed of Exponentially Modified Gaussian distributions to
       analyze asymmetric peak shapes in high-resolution time-of-flight mass
       spectrometry." International Journal of Mass Spectrometry 421 (2017):
       245-254.

    """
    li_eta_m = np.array(li_eta_m).astype(np.float_)
    li_tau_m = np.array(li_tau_m).astype(np.float_)
    t_order_m = len(li_eta_m) # order of negative tail exponentials
    sum_eta_m = 0.
    for i in range(t_order_m):
        sum_eta_m += li_eta_m[i]
    if abs(sum_eta_m - 1) > norm_precision:  # check normalization of eta_m's
        raise Exception("eta_m's don't add up to 1.")
    if len(li_tau_m) != t_order_m:  # check if all arguments match tail order
        raise Exception("orders of eta_m and tau_m do not match!")

    h_m = np.zeros_like(x)
    for i in range(t_order_m):
        eta_m = li_eta_m[i]
        tau_m = li_tau_m[i]
        h_m += h_m_i(x,mu,sigma,eta_m,tau_m)
    return h_m


@njit
def h_p_i(x,mu,sigma,eta_p,tau_p):
    """Internal helper function to calculate single positive EMG tail for
    h_p_emg."""
    erfcarg = np.atleast_1d(sigma/(sqrt(2)*tau_p) - (x-mu)/(sqrt(2)*sigma))
    mask = (erfcarg < 0)
    ret = np.empty_like(x)
    # Use Gauss*erfcx formulation to avoid overflow of exp and underflow of
    # erfc at larger pos. arguments:
    Gauss_erfcx = exp( -0.5*((x[~mask]-mu)/sigma)**2 )*_erfcx_jit(erfcarg[~mask])
    ret[~mask] = eta_p/(2*tau_p)*Gauss_erfcx
    # Use exp*erfc formulation to avoid overflow of erfcx at larger neg.
    # arguments:
    exp_erfc = exp(0.5*(sigma/tau_p)**2 - (x[mask]-mu)/tau_p)*_erfc_jit(erfcarg[mask])
    ret[mask] = 0.5*eta_p/tau_p*exp_erfc

    return ret


def h_p_i_prec(x,mu,sigma,eta_p,tau_p):
    """Arbitrary precision version of internal helper function for h_p_emg."""
    expval = exp_mp( 0.5*(sigma/tau_p)**2 - (x-mu)/tau_p )
    erfcval = erfc_mp( sigma/(sqrt(2)*tau_p) - (x-mu)/(sqrt(2)*sigma) )
    ret = 0.5*eta_p/tau_p*expval*erfcval
    return ret.astype(float)


@njit
def h_p_emg(x, mu, sigma, li_eta_p, li_tau_p):
    """Positive skewed exponentially-modified Gaussian (EMG) distribution.

    The lengths of `li_eta_p` & `li_tau_p` must match and define the order of
    positive tails.

    Parameters
    ----------
    x  : float >= 0
        Abscissa data (mass data).
    mu : float >= 0
        Mean value of underlying Gaussian distribution.
    sigma : float >= 0
        Standard deviation of underlying Gaussian distribution.
    li_eta_p : tuple
        Tuple containing the pos. tail weights with the signature:
        ``(eta_p1, eta_p2, ...)``.
    li_tau_p : tuple
        Tuple containing the pos. tail decay constants with the signature:
        ``(tau_p1, tau_p2, ...)``.

    Returns
    -------
    float
        Ordinate values of the positive skewed EMG distribution.

    Notes
    -----
    The Hyper-EMG probability distribution function was first introduced in
    `this publication`_ by Purushothaman et al. [#]_. The basic definitions and
    notations used here are adapted from this work.

    Each positive tail of a Hyper-EMG function can be expressed in two
    equivalent ways:

    .. math::

        h_\mathrm{emg,+i} = \\frac{\\eta_{+i}}{2\\tau_{+i}} \\exp{(-\\left(\\frac{x-\\mu}{\\sqrt{2}\\sigma}\\right)^2)} \mathrm{erfcx}(v)
        = \\frac{\\eta_{+i}}{2\\tau_{+i}} \\exp{(u)} \mathrm{erfc}(v),

    where :math:`u = \\frac{\\sigma}{\\sqrt{2}\\tau_{+i}} - \\frac{x-\mu}{\\sqrt{2}\\tau_{+i}}`
    and :math:`v = \\frac{\\sigma}{\\sqrt{2}\\tau_{+i}} - \\frac{x-\mu}{\\sqrt{2}\\sigma}`.
    In double precision, the `exp(u)`_ routine overflows if u > 709.78. The
    complementary error function `erfc(v)`_ underflows to 0.0 if v > 26.54. The
    scaled complementary error function `erfcx(v)`_ overflows if v < -26.62. To
    circumvent those scenarios and always ensure an exact result, the underlying
    helper function for the calculation of a negative EMG tail :func:`h_m_i`
    uses the formulation in terms of `erfcx` whenever v >= 0 and switches to the
    `erfc`-formulation when v < 0.

    .. _`exp(u)`: https://numpy.org/doc/stable/reference/generated/numpy.exp.html#numpy.exp
    .. _`erfc(v)`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erfc.html
    .. _`erfcx(v)`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erfcx.html
    .. _`this publication`: https://www.sciencedirect.com/science/article/abs/pii/S1387380616302913

    References
    ----------
    .. [#] Purushothaman, S., et al. "Hyper-EMG: A new probability distribution
       function composed of Exponentially Modified Gaussian distributions to
       analyze asymmetric peak shapes in high-resolution time-of-flight mass
       spectrometry." International Journal of Mass Spectrometry 421 (2017):
       245-254.

    """
    li_eta_p = np.array(li_eta_p).astype(np.float_)
    li_tau_p = np.array(li_tau_p).astype(np.float_)
    t_order_p = len(li_eta_p) # order of positive tails
    sum_eta_p = 0.
    for i in range(t_order_p):
        sum_eta_p += li_eta_p[i]
    if abs(sum_eta_p - 1) > norm_precision:  # check normalization of eta_p's
        raise Exception("eta_p's don't add up to 1.")
    if len(li_tau_p) != t_order_p:  # check if all arguments match tail order
        raise Exception("orders of eta_p and tau_p do not match!")

    h_p = np.zeros_like(x)
    for i in range(t_order_p):
        eta_p = li_eta_p[i]
        tau_p = li_tau_p[i]
        h_p += h_p_i(x,mu,sigma,eta_p,tau_p)
    return h_p


@njit
def h_emg(x, mu, sigma , theta, li_eta_m, li_tau_m, li_eta_p, li_tau_p):
    """Hyper-exponentially-modified Gaussian distribution (hyper-EMG).

    The lengths of `li_eta_m` & `li_tau_m` must match and define the order of
    negative tails. Likewise, the lengths of `li_eta_p` & `li_tau_p` must match
    and define the order of positive tails.

    Parameters
    ----------
    x  : float >= 0
        Abscissa data (mass data).
    mu : float >= 0
        Mean value of underlying Gaussian distribution.
    sigma : float >= 0
        Standard deviation of underlying Gaussian distribution.
    theta : float, 0 <= theta <= 1
        Left-right-weight factor (negative-skewed EMG weight: theta;
        positive-skewed EMG weight: 1 - theta).
    li_eta_m : tuple
        Tuple containing the neg. tail weights with the signature:
        ``(eta_m1, eta_m2, ...)``.
    li_tau_m : tuple
        Tuple containing the neg. tail decay constants with the signature:
        ``(tau_m1, tau_m2, ...)``.
    li_eta_p : tuple
        Tuple containing the pos. tail weights with the signature:
        ``(eta_p1, eta_p2, ...)``.
    li_tau_p : tuple
        Tuple containing the pos. tail decay constants with the signature:
        ``(tau_p1, tau_p2, ...)``.

    Returns
    -------
    float
        Ordinate of hyper-EMG distribution

    See also
    --------
    :func:`h_m_emg`
    :func:`h_p_emg`

    Notes
    -----
    The Hyper-EMG probability distribution function was first introduced in
    `this publication`_ by Purushothaman et al. [#]_. The basic definitions and
    notations used here are adapted from this work.

    The total hyper-EMG distribution `h_m_emg` is comprised of the negative- and
    positive-skewed EMG distributions `h_m_emg` and `h_p_emg` respectively and
    is calculated as:
    ``h_emg(x, mu, sigma, theta, li_eta_m, li_tau_m, li_eta_p, li_tau_p) =``
    ``theta*h_m_emg(x, mu, sigma, li_eta_m, li_tau_m) +
    (1-theta)*h_p_emg(x, mu, sigma, li_eta_p, li_tau_p)``.

    For algorithmic details, see `Notes` of :func:`h_m_emg` and :func:`h_p_emg`.

    .. _`this publication`: https://www.sciencedirect.com/science/article/abs/pii/S1387380616302913

    References
    ----------
    .. [#] Purushothaman, S., et al. "Hyper-EMG: A new probability distribution
       function composed of Exponentially Modified Gaussian distributions to
       analyze asymmetric peak shapes in high-resolution time-of-flight mass
       spectrometry." International Journal of Mass Spectrometry 421 (2017):
       245-254.

    """
    if theta == 1:
        h = h_m_emg(x, mu, sigma, li_eta_m, li_tau_m)
    elif theta == 0:
        h = h_p_emg(x, mu, sigma, li_eta_p, li_tau_p)
    else:
        neg_tail = h_m_emg(x, mu, sigma, li_eta_m, li_tau_m)
        pos_tail = h_p_emg(x, mu, sigma, li_eta_p, li_tau_p)
        h = theta*neg_tail + (1-theta)*pos_tail
    return h


def mu_emg(mu, theta, li_eta_m, li_tau_m, li_eta_p, li_tau_p):
    """Calculate mean of hyper-EMG distribution.

    The lengths of `li_eta_m` & `li_tau_m` must match and define the order of
    negative tails. Likewise, the lengths of `li_eta_p` & `li_tau_p` must match
    and define the order of positive tails.

    Parameters
    ----------
    mu : float >= 0
        Mean value of underlying Gaussian distribution.
    theta : float, 0 <= theta <= 1
        Left-right-weight factor (negative-skewed EMG weight: theta;
        positive-skewed EMG weight: 1 - theta).
    li_eta_m : tuple
        Tuple containing the neg. tail weights with the signature:
        ``(eta_m1, eta_m2, ...)``.
    li_tau_m : tuple
        Tuple containing the neg. tail decay constants with the signature:
        ``(tau_m1, tau_m2, ...)``.
    li_eta_p : tuple
        Tuple containing the pos. tail weights with the signature:
        ``(eta_p1, eta_p2, ...)``.
    li_tau_p : tuple
        Tuple containing the pos. tail decay constants with the signature:
        ``(tau_p1, tau_p2, ...)``.

    Returns
    -------
    float
        Mean of hyper-EMG distribution.

    Notes
    -----
    The Hyper-EMG probability distribution function was first introduced in
    `this publication`_ by Purushothaman et al. [#]_. The basic definitions and
    notations used here are adapted from this work.

    .. _`this publication`: https://www.sciencedirect.com/science/article/abs/pii/S1387380616302913

    References
    ----------
    .. [#] Purushothaman, S., et al. "Hyper-EMG: A new probability distribution
       function composed of Exponentially Modified Gaussian distributions to
       analyze asymmetric peak shapes in high-resolution time-of-flight mass
       spectrometry." International Journal of Mass Spectrometry 421 (2017):
       245-254.

    """
    if abs(sum(li_eta_m) - 1) > norm_precision:  # check normalization of eta_m's
        raise Exception("eta_m's don't add up to 1.")
    t_order_m = len(li_eta_m)
    sum_M_mh = 0
    for i in range(t_order_m):
        sum_M_mh += li_eta_m[i]*li_tau_m[i]

    if abs(sum(li_eta_p) - 1) > norm_precision:  # check normalization of eta_p's
        raise Exception("eta_p's don't add up to 1.")
    t_order_p = len(li_eta_p)
    sum_M_ph = 0
    for i in range(t_order_p):
        sum_M_ph += li_eta_p[i]*li_tau_p[i]

    return mu - theta*sum_M_mh + (1-theta)*sum_M_ph


def sigma_emg(sigma, theta, li_eta_m, li_tau_m, li_eta_p, li_tau_p):
    """Calculate standard deviation of hyper-EMG distribution.

    The lengths of `li_eta_m` & `li_tau_m` must match and define the order of
    negative tails. Likewise, the lengths of `li_eta_p` & `li_tau_p` must match
    and define the order of positive tails.

    Parameters
    ----------
    sigma : float >= 0
        Standard deviation of underlying Gaussian distribution.
    theta : float, 0 <= theta <= 1
        Left-right-weight factor (negative-skewed EMG weight: theta;
        positive-skewed EMG weight: 1 - theta).
    li_eta_m : tuple
        Tuple containing the neg. tail weights with the signature:
        ``(eta_m1, eta_m2, ...)``.
    li_tau_m : tuple
        Tuple containing the neg. tail decay constants with the signature:
        ``(tau_m1, tau_m2, ...)``.
    li_eta_p : tuple
        Tuple containing the pos. tail weights with the signature:
        ``(eta_p1, eta_p2, ...)``.
    li_tau_p : tuple
        Tuple containing the pos. tail decay constants with the signature:
        ``(tau_p1, tau_p2, ...)``.

    Returns
    -------
    float
        Standard deviation of hyper-EMG distribution.

    Notes
    -----
    The Hyper-EMG probability distribution function was first introduced in
    `this publication`_ by Purushothaman et al. [#]_. The basic definitions and
    notations used here are adapted from this work.

    .. _`this publication`: https://www.sciencedirect.com/science/article/abs/pii/S1387380616302913

    References
    ----------
    .. [#] Purushothaman, S., et al. "Hyper-EMG: A new probability distribution
       function composed of Exponentially Modified Gaussian distributions to
       analyze asymmetric peak shapes in high-resolution time-of-flight mass
       spectrometry." International Journal of Mass Spectrometry 421 (2017):
       245-254.

    """
    if abs(sum(li_eta_m) - 1) > norm_precision:  # check normalization of eta_m's
        raise Exception("eta_m's don't add up to 1.")
    t_order_m = len(li_eta_m)

    sum_M_mh = 0
    sum_S_mh = 0
    for i in range(t_order_m):
        sum_M_mh += li_eta_m[i]* li_tau_m[i]
        sum_S_mh += (li_eta_m[i] + li_eta_m[i]*(1.-li_eta_m[i])**2)*li_tau_m[i]**2

    if abs(sum(li_eta_p) - 1) > norm_precision:  # check normalization of eta_p's
        raise Exception("eta_p's don't add up to 1.")
    t_order_p = len(li_eta_p)
    sum_M_ph = 0
    sum_S_ph = 0
    for i in range(t_order_p):
        sum_M_ph += li_eta_p[i]* li_tau_p[i]
        sum_S_ph += (li_eta_p[i] + li_eta_p[i]*(1.-li_eta_p[i])**2)*li_tau_p[i]**2

    S_h = sigma**2 + theta*sum_S_mh + (1-theta)*sum_S_ph + theta*(1.-theta)*(sum_M_mh + sum_M_ph)**2
    return np.sqrt(S_h)


################################################################################
