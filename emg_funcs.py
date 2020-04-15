###################################################################################################
##### Python module for Hyper-EMG fitting of TOF mass spectra
##### Fitting routines taken from lmfit package
##### Code by Stefan Paul, 2019-12-28

##### Import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lmfit as fit
import scipy.constants as con
import scipy.special as spl
#from numba import jit, prange

###################################################################################################
##### Define general Hyper-EMG functions (with high precision math package)
# from numba import vectorize, float64
# import math
# @vectorize([float64(float64)])
# def math_erfc(x):
#     return math.erfc(x)
# import mpmath as mp
# import gmpy2
# np_exp = np.frompyfunc(gmpy2.exp,1,1)   #np.frompyfunc(mp.fp.exp,1,1) # convert mp.exp function to numpy function, avoids error in lmfit.Model( ... )
# np_erfc = np.frompyfunc(gmpy2.erfc,1,1)   #np.frompyfunc(mp.fp.erfc,1,1) # convert mp.exp function to numpy function, avoids error in lmfit.Model( ... )
norm_precision = 6 # number of decimals on which eta parameters have to agree with unity (avoids numerical errors due to rounding)

def bounded_exp(arg):
    """ Numerically stable exponential function which avoids under- or overflow by setting bounds on argument
    """
    max_arg = 680 # max_arg = 600 results in maximal y-value of 3.7730203e+260
    min_arg = -1000000000000000000
    arg = np.where(arg > max_arg, max_arg, arg)
    arg = np.where(arg < min_arg, min_arg, arg)
    return np.exp(arg)

# def exp_erfc_m(x,mu,sigma,tau_m):
#     val = np_exp( (sigma/(np.sqrt(2)*tau_m))**2 + (x-mu)/tau_m )*np_erfc( sigma/(np.sqrt(2)*tau_m) + (x-mu)/(np.sqrt(2)*sigma) )
#     return np.float(val)
# vect_exp_erfc_m = np.vectorize(exp_erfc_m)
#
# def exp_erfc_p(x,mu,sigma,tau_p):
#     val = np_exp( (sigma/(np.sqrt(2)*tau_p))**2 - (x-mu)/tau_p )*np_erfc( sigma/(np.sqrt(2)*tau_p) - (x-mu)/(np.sqrt(2)*sigma) )
#     return np.float(val)
# vect_exp_erfc_p = np.vectorize(exp_erfc_p)

# Define negative skewed hyper-EMG particle distribution functiDon (NS-EMG PDF)
def h_m_emg(x, mu, sigma, *t_args):
    """ Negative skewed hyper-EMG particle distribution function (NS-EMG PDF)

    Parameters:
    x (float >= 0 ): abscissa values (mass or TOF data)
    mu (float >= 0): mean value of unmodified Gaussian distribution
    sigma (float >= 0): standard deviation of unmodified Gaussian distribution
    t_args = (eta_m1, eta_m2, ...), (tau_m1, tau_m2, ...):  variable-length list of tail shape arguments (length of tuples defines the tail order)

    Returns:
    float: ordinate values of NS-EMG PDF
    """
    li_eta_m = t_args[0]
    li_tau_m = t_args[1]
    t_order_m = len(li_eta_m) # order of negative tail exponentials
    if np.round(np.sum(li_eta_m), decimals=norm_precision) != 1.0:  # check normalization of eta_m's
        raise Exception("eta_m's don't add up to 1")
    if len(li_tau_m) != t_order_m:  # check if all arguments match tail order
        raise Exception("orders of eta_m and tau_m do not match!")

    # @jit(parallel=True, fastmath=True)
    # def calc_tails_m():
    #     for i in prange(t_order_m):
    #         h_m = np.array([0])
    #         eta_m = li_eta_m[i]
    #         tau_m = li_tau_m[i]
    #         val = eta_m/(2*tau_m)*np.exp( (sigma/(np.sqrt(2)*tau_m))**2 + (x-mu)/tau_m )*math_erfc( sigma/(np.sqrt(2)*tau_m) + (x-mu)/(np.sqrt(2)*sigma) )
    #         h_m += np.where(np.isfinite(val), val, np.zeros_like(val))  # eta_m/(2*tau_m)*vect_exp_erfc_m(x,mu,sigma,tau_m)
    #         return h_m
    # h_m = calc_tails_m()

    h_m = 0.
    for i in range(t_order_m):
        eta_m = li_eta_m[i]
        tau_m = li_tau_m[i]
        h_m += np.nan_to_num(eta_m/(2*tau_m)*np.exp( (sigma/(np.sqrt(2)*tau_m))**2 + (x-mu)/tau_m )*spl.erfc( sigma/(np.sqrt(2)*tau_m) + (x-mu)/(np.sqrt(2)*sigma) ))  # eta_m/(2*tau_m)*vect_exp_erfc_m(x,mu,sigma,tau_m)
    # print("h_m:"+str(h_m))
    return h_m

# Define positive skewed exponentially-modified Gaussian particle distribution function (PS-EMG PDF)
def h_p_emg(x, mu, sigma, *t_args):
    # variable-length list of tail arguments: t_args = (eta_p1, eta_p2), (tau_p1, tau_p2), ...
    """ Negative skewed hyper-EMG particle distribution function (NS-EMG PDF)

    Parameters:
    x (float >= 0 ): abscissa (mass or TOF data)
    mu (float >= 0): mean value of unmodified Gaussian distribution
    sigma (float > 0): standard deviation of unmodified Gaussian distribution
    t_args = (eta_p1, eta_p2, ...), (tau_p1, tau_p2, ...):  variable-length list of tail shape arguments (length of tuples defines the tail order)

    Returns:
    float: ordinate of NS-EMG PDF
    """
    li_eta_p = t_args[0]
    li_tau_p = t_args[1]
    t_order_p = len(li_eta_p) # order of positive tails
    if np.round(np.sum(li_eta_p), decimals=norm_precision) != 1.0:  # check normalization of eta_p's
        raise Exception("eta_p's don't add up to 1")
    if len(li_tau_p) != t_order_p:  # check if all arguments match tail order
        raise Exception("orders of eta_p and tau_p do not match!")

    # @jit(parallel=True,fastmath=True)
    # def calc_tails_p():
    #     for i in prange(t_order_p):
    #         h_p = np.array([0.])
    #         eta_p = li_eta_p[i]
    #         tau_p = li_tau_p[i]
    #         val = eta_p/(2*tau_p)*np.exp( (sigma/(np.sqrt(2)*tau_p))**2 - (x-mu)/tau_p )*math_erfc( sigma/(np.sqrt(2)*tau_p) - (x-mu)/(np.sqrt(2)*sigma) )
    #         h_p += np.where(np.isfinite(val), val, np.zeros_like(val))  # eta_m/(2*tau_m)*vect_exp_erfc_m(x,mu,sigma,tau_m)
    #         # eta_p/(2*tau_p)*vect_exp_erfc_p(x,mu,sigma,tau_p)
    #         return h_p
    # h_p = calc_tails_p()

    h_p = 0.
    for i in range(t_order_p):
        eta_p = li_eta_p[i]
        tau_p = li_tau_p[i]
        h_p += np.nan_to_num(eta_p/(2*tau_p)*np.exp( (sigma/(np.sqrt(2)*tau_p))**2 - (x-mu)/tau_p )*spl.erfc( sigma/(np.sqrt(2)*tau_p) - (x-mu)/(np.sqrt(2)*sigma) ))  # eta_p/(2*tau_p)*vect_exp_erfc_p(x,mu,sigma,tau_p)
    # print("h_p:"+str(h_p))
    return h_p

# Hyper-EMG PDF
def h_emg(x, mu, sigma , theta, *t_args):
    """ Hyper-EMG particle distribution function (hyper-EMG PDF)

    Parameters:
    x (float >= 0 ): abscissa (mass or TOF data)
    mu (float >= 0): mean value of unmodified Gaussian distribution
    sigma (float > 0): standard deviation of unmodified Gaussian distribution
    theta (float 0 <= theta <= 1): left-right-weight factor (NS EMG weight: theta; PS EMG weight: 1 - theta)
    t_args = (eta_m1, eta_m2, ...), (tau_m1, tau_m2, ...),  (eta_p1, eta_p2, ...), (tau_p1, tau_p2, ...):
            variable-length list of tail shape arguments (length of respective tuples defines the PS and NS tail orders)

    Returns:
    float: ordinate of hyper-EMG PDF
    """
    li_eta_m = t_args[0]
    li_tau_m = t_args[1]
    li_eta_p = t_args[2]
    li_tau_p = t_args[3]
    if theta == 1:
        h = h_m_emg(x, mu, sigma, li_eta_m, li_tau_m)
    elif theta == 0:
        h = h_p_emg(x, mu, sigma, li_eta_p, li_tau_p)
    else:
        h = theta*h_m_emg(x, mu, sigma, li_eta_m, li_tau_m) + (1-theta)*h_p_emg(x, mu, sigma, li_eta_p, li_tau_p)
    return  np.clip(h,a_min=None,a_max=1e295) # clipping to avoid overflow errors
    """if isinstance(x,np.ndarray):
        h = np.array([])
        for x_i in x:
            if (x_i - mu)/sigma < 5:
                li_eta_m = t_args[0]
                li_tau_m = t_args[1]
                li_eta_p = t_args[2]
                li_tau_p = t_args[3]
                h = np.append(h ,theta*h_m_emg(x_i, mu, sigma, li_eta_m, li_tau_m) + (1-theta)*h_p_emg(x_i, mu, sigma, li_eta_p, li_tau_p) )
            else:
                h = np.append(h, 0)
        return h"""

def mu_emg(mu,theta,*t_args):
    """ Calculates mean of hyper-EMG particle distribution function (hyper-EMG PDF)

    Parameters:
    mu (float >= 0): mean value of unmodified Gaussian distribution
    theta (float 0 <= theta <= 1): left-right-weight factor (NS EMG weight: theta; PS EMG weight: 1 - theta)
    t_args = (eta_m1, eta_m2, ...), (tau_m1, tau_m2, ...),  (eta_p1, eta_p2, ...), (tau_p1, tau_p2, ...):
            variable-length list of tail shape arguments (length of respective tuples defines the PS and NS tail orders)

    Returns:
    float: mean of hyper-EMG PDF

    """
    li_eta_m = t_args[0]
    if np.round(np.sum(li_eta_m), decimals=norm_precision) != 1.0:  # check normalization of eta_m's
        raise Exception("eta_m's don't add up to 1")
    li_tau_m = t_args[1]
    t_order_m = len(li_eta_m)
    sum_M_mh = 0
    for i in range(t_order_m):
        sum_M_mh += li_eta_m[i]*li_tau_m[i]

    li_eta_p = t_args[2]
    if np.round(np.sum(li_eta_p), decimals=norm_precision) != 1.0:  # check normalization of eta_p's
        raise Exception("eta_p's don't add up to 1")
    li_tau_p = t_args[3]
    t_order_p = len(li_eta_p)
    sum_M_ph = 0
    for i in range(t_order_p):
        sum_M_ph += li_eta_p[i]*li_tau_p[i]

    return mu - theta*sum_M_mh + (1-theta)*sum_M_ph


def sigma_emg(sigma,theta,*t_args):
    """ Calculates standard deviation of hyper-EMG particle distribution function (hyper-EMG PDF)

    Parameters:
    sigma (float > 0): standard deviation of unmodified Gaussian distribution
    theta (float 0 <= theta <= 1): left-right-weight factor (NS EMG weight: theta; PS EMG weight: 1 - theta)
    t_args = (eta_m1, eta_m2, ...), (tau_m1, tau_m2, ...),  (eta_p1, eta_p2, ...), (tau_p1, tau_p2, ...):
            variable-length list of tail shape arguments (length of respective tuples defines the PS and NS tail orders)

    Returns:
    float: standard deviation of hyper-EMG PDF

    """
    li_eta_m = t_args[0]
    if np.round(np.sum(li_eta_m), decimals=norm_precision) != 1.0:  # check normalization of eta_m's
        raise Exception("eta_m's don't add up to 1")
    li_tau_m = t_args[1]
    t_order_m = len(li_eta_m)

    sum_M_mh = 0
    sum_S_mh = 0
    for i in range(t_order_m):
        sum_M_mh += li_eta_m[i]* li_tau_m[i]
        sum_S_mh += (li_eta_m[i] + li_eta_m[i]*(1.-li_eta_m[i])**2)*li_tau_m[i]**2

    li_eta_p = t_args[2]
    if np.round(np.sum(li_eta_p), decimals=norm_precision) != 1.0:  # check normalization of eta_p's
        raise Exception("eta_p's don't add up to 1")
    li_tau_p = t_args[3]
    t_order_p = len(li_eta_p)
    sum_M_ph = 0
    sum_S_ph = 0
    for i in range(t_order_p):
        sum_M_ph += li_eta_p[i]* li_tau_p[i]
        sum_S_ph += (li_eta_p[i] + li_eta_p[i]*(1.-li_eta_p[i])**2)*li_tau_p[i]**2

    S_h = sigma**2 + theta*sum_S_mh + (1-theta)*sum_S_ph + theta*(1.-theta)*(sum_M_mh + sum_M_ph)**2
    return np.sqrt(S_h)


###################################################################################################
