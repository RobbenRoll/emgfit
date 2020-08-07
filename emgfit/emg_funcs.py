################################################################################
##### Module with general Hyper-EMG functions
##### Author: Stefan Paul

##### Import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lmfit as fit
import scipy.constants as con
#import scipy.special as spl
from numpy import exp, nan_to_num, clip, where, isfinite
from spycial import erfc
from math import sqrt
from numba import njit

################################################################################
##### Define general Hyper-EMG functions

norm_precision = 1e-06 # level on which eta parameters must agree with unity

# def bounded_exp(arg):
#     """ Numerically stable exponential function which avoids under- or overflow by setting bounds on argument
#     """
#     max_arg = 680 # max_arg = 600 results in maximal y-value of 3.7730203e+260
#     min_arg = -1000000000000000000
#     arg = np.where(arg > max_arg, max_arg, arg)
#     arg = np.where(arg < min_arg, min_arg, arg)
#     return np.exp(arg)

@njit #(parallel=True)
def h_m_i(x,mu,sigma,eta_m,tau_m):
    """Helper function for  h_m_emg """
    ret = eta_m/(2*tau_m)*exp( (sigma/(sqrt(2)*tau_m))**2 + (x-mu)/tau_m )*erfc( sigma/(sqrt(2)*tau_m) + (x-mu)/(sqrt(2)*sigma) )
    return ret

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

    Note
    ----
    Depending on the choice of parameters, numerical artifacts (multiplications
    by infinity due to overflow of :func:`numpy.exp` for arguments > 709.7) can
    result in non-finite values for ``h_m`` (see code below).
    These numerical artifacts are handled by setting non-finite ``h_m`` values
    to ``0``. A different option would be the usage of a package with arbitrary
    precision float numbers (e.g. mpmath). However, the latter would yield
    extremely long computation times.

    """
    t_order_m = len(li_eta_m) # order of negative tail exponentials
    if abs(sum(li_eta_m) - 1) > norm_precision:  # check normalization of eta_m's
        raise Exception("eta_m's don't add up to 1.")
    if len(li_tau_m) != t_order_m:  # check if all arguments match tail order
        raise Exception("orders of eta_m and tau_m do not match!")

    h_m = 0.
    for i in range(t_order_m):
        eta_m = li_eta_m[i]
        tau_m = li_tau_m[i]
        h_i = h_m_i(x,mu,sigma,eta_m,tau_m)
        h_m += where(isfinite(h_i),h_i,0.)  #np.where(np.isfinite(h_m_i),h_m_i,0)
    return h_m


@njit #(parallel=True)
def h_p_i(x,mu,sigma,eta_p,tau_p):
    """Helper function for  h_p_emg """
    ret = eta_p/(2*tau_p)*exp( (sigma/(sqrt(2)*tau_p))**2 - (x-mu)/tau_p )*erfc( sigma/(sqrt(2)*tau_p) - (x-mu)/(sqrt(2)*sigma) )
    return ret


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

    Note
    ----
    Depending on the choice of parameters, numerical artifacts (multiplications
    by infinity due to overflow of :func:`numpy.exp` for arguments > 709.7) can
    result in non-finite values for ``h_p`` (see code below).
    These numerical artifacts are handled by setting non-finite ``h_p`` values
    to ``0``. A different option would be the usage of a package with arbitrary
    precision float numbers (e.g. mpmath). However, the latter would yield
    extremely long computation times.

    """
    t_order_p = len(li_eta_p) # order of positive tails
    if abs(sum(li_eta_p) - 1) > norm_precision:  # check normalization of eta_p's
        raise Exception("eta_p's don't add up to 1.")
    if len(li_tau_p) != t_order_p:  # check if all arguments match tail order
        raise Exception("orders of eta_p and tau_p do not match!")

    h_p = 0.
    for i in range(t_order_p):
        eta_p = li_eta_p[i]
        tau_p = li_tau_p[i]
        h_i = h_p_i(x,mu,sigma,eta_p,tau_p)
        h_p += where(isfinite(h_i),h_i,0.) #np.where(np.isfinite(h_p_i),h_p_i,0)
    return h_p


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
        Ordinate of hyper-EMG distribution.

    Note
    ----
    Depending on the choice of parameters, numerical artifacts (due to overflow
    of :func:`numpy.exp` for arguments >709.7) could result in non-finite return
    values for :func:`h_m_emg` and :func:`h_p_emg` (and thus for :func:`h_emg`).
    See docs of :func:`h_m_emg` & :func:`h_p_emg` for how those artifacts are
    handled.

    See also
    --------
    :func:`h_m_emg`
    :func:`h_p_emg`

    """
    if theta == 1:
        h = h_m_emg(x, mu, sigma, li_eta_m, li_tau_m)
    elif theta == 0:
        h = h_p_emg(x, mu, sigma, li_eta_p, li_tau_p)
    else:
        h = theta*h_m_emg(x, mu, sigma, li_eta_m, li_tau_m) + (1-theta)*h_p_emg(x, mu, sigma, li_eta_p, li_tau_p)
    return h  #clip(h,a_min=0,a_max=1e295) # clipping to avoid overflow errors


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
