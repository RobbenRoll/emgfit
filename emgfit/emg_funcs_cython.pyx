# cython: infer_types=True
################################################################################
##### Module with general Hyper-EMG functions
##### Author: Stefan Paul

##### Import packages
import numpy as np
cimport numpy as cnp
cimport cython
import scipy.special as spl
#import numexpr as ne
#ne.set_vml_accuracy_mode('high')
#from numba import jit, prange

################################################################################
##### Define general Hyper-EMG functions
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

def h_m_emg(x, mu, sigma, *t_args):
    """Negative skewed exponentially-modified Gaussian (EMG) distribution.

    Parameters
    ----------
    x  : float >= 0
        Abscissa data (mass data).
    mu : float >= 0
        Mean value of underlying Gaussian distribution.
    sigma : float >= 0
        Standard deviation of underlying Gaussian distribution.
    t_args : :class:`tuple`, :class:`tuple`
        Two variable-length tuples of neg. tail shape arguments with the
        signature: ``(eta_m1, eta_m2, ...), (tau_m1, tau_m2, ...)``.
        The length of the tuples must match and defines the order of neg. tails.

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
    li_eta_m = t_args[0]
    li_tau_m = t_args[1]
    t_order_m = len(li_eta_m) # order of negative tail exponentials
    if np.round(np.sum(li_eta_m), decimals=norm_precision) != 1.0:  # check normalization of eta_m's
        raise Exception("eta_m's don't add up to 1.")
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
        h_m_i = eta_m/(2*tau_m)*np.exp( (sigma/(np.sqrt(2)*tau_m))**2 + (x-mu)/tau_m )*spl.erfc( sigma/(np.sqrt(2)*tau_m) + (x-mu)/(np.sqrt(2)*sigma) )
        #erfc_i = spl.erfc(sigma/(np.sqrt(2)*tau_m))
        #h_m_i = ne.evaluate('eta_m/(2*tau_m)*exp( (sigma/(sqrt(2)*tau_m))**2 + (x-mu)/tau_m )*erfc_i + (x-mu)/(sqrt(2)*sigma)',optimization='moderate')
        h_m += np.where(np.isfinite(h_m_i),h_m_i,0) #np.nan_to_num(eta_m/(2*tau_m)*np.exp( (sigma/(np.sqrt(2)*tau_m))**2 + (x-mu)/tau_m )*spl.erfc( sigma/(np.sqrt(2)*tau_m) + (x-mu)/(np.sqrt(2)*sigma) ))  # eta_m/(2*tau_m)*vect_exp_erfc_m(x,mu,sigma,tau_m)
    # print("h_m:"+str(h_m))
    return h_m
#h_m_emg = np.vectorize(h_m_emg,excluded=['mu','sigma','*t_args'])


from libc.math cimport exp, erfc, sqrt, pow
# Define positive skewed exponentially-modified Gaussian particle distribution function (PS-EMG PDF)
@cython.boundscheck(False)
def h_p_emg(cnp.ndarray[cnp.float_t, ndim=1] x, cnp.float_t mu, cnp.float_t sigma, *t_args):
    """Positive skewed exponentially-modified Gaussian (EMG) distribution.

    Parameters
    ----------
    x  : float >= 0
        Abscissa data (mass data).
    mu : float >= 0
        Mean value of underlying Gaussian distribution.
    sigma : float >= 0
        Standard deviation of underlying Gaussian distribution.
    t_args : :class:`tuple`, :class:`tuple`
        Two variable-length tuples of pos. tail shape arguments with
        the signature:
        ``(eta_m1, eta_m2, ...), (tau_m1, tau_m2, ...)``.
        The length of the tuples must match and defines the order of pos. tails.

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
    cdef cnp.ndarray[cnp.float64_t, ndim=1] li_eta_p, li_tau_p
    cdef int t_order_p
    cdef cnp.float64_t h_p, eta_p, tau_p, h_p_i

    li_eta_p = t_args[0]
    li_tau_p = t_args[1]
    t_order_p = len(li_eta_p) # order of positive tails
    assert (np.sum(li_eta_p) - 1) < norm_precision, "eta_p's don't add up to 1."
    if len(li_tau_p) != t_order_p:  # check if all arguments match tail order
        raise Exception("orders of eta_p and tau_p do not match!")

    h_p = 0.
    for i in range(t_order_p):
        eta_p = li_eta_p[i]
        tau_p = li_tau_p[i]
        h_p_i = eta_p/(2*tau_p)*exp( pow(sigma/(sqrt(2)*tau_p),2.0) - (x-mu)/tau_p )*erfc( sigma/(sqrt(2)*tau_p) - (x-mu)/(sqrt(2)*sigma) )
        #erfc_i = spl.erfc(sigma/(np.sqrt(2)*tau_p))
        #h_p_i = ne.evaluate('eta_p/(2*tau_p)*exp( (sigma/(sqrt(2)*tau_p))**2 - (x-mu)/tau_p )*erfc_i - (x-mu)/(sqrt(2)*sigma)',optimization='moderate')
        h_p += h_p_i # np.where(np.isfinite(h_p_i),h_p_i,0) #np.nan_to_num(eta_p/(2*tau_p)*np.exp( (sigma/(np.sqrt(2)*tau_p))**2 - (x-mu)/tau_p )*spl.erfc( sigma/(np.sqrt(2)*tau_p) - (x-mu)/(np.sqrt(2)*sigma) ))  # eta_p/(2*tau_p)*vect_exp_erfc_p(x,mu,sigma,tau_p)
    # print("h_p:"+str(h_p))
    return h_p
#h_p_emg = np.vectorize(h_p_emg,excluded=['mu','sigma','t_args'])

cdef double sq(double x):
    return x * x

cdef extern from "fast_exp.c":
    double exp_fast_schraudolph( double )

cdef double fast_exp(double a):
    exp_fast_schraudolph(a)

@cython.cdivision(True)     # Deactivate Zero Division Exception checking
cdef double expa(double  x, int terms = 30):
    cdef double sum, power, fact
    cdef int i
    sum = 0.
    power = 1.
    fact = 1.
    for i in range(terms):
        sum += power/fact
        power *= x
        fact *= i+1
    return sum

from cython. parallel import prange

@cython.cdivision(True)     # Deactivate Zero Division Exception checking
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def h_p_emg2(double[:] x, double mu, double sigma, double eta_p1, double eta_p2, double eta_p3, double tau_p1, double tau_p2, double tau_p3): # cnp.ndarray[double, ndim=1]
    assert (eta_p1 + eta_p2 + eta_p3 - 1) < norm_precision, "eta_p's don't add up to 1."
    #if len(arr_eta_p) != len(arr_tau_p):  # check if all arguments match tail order
    #    raise Exception("orders of eta_p and tau_p do not match!")
    cdef Py_ssize_t n = x.size
    cdef double h_p1, h_p2, h_p3
    cdef cnp.ndarray[cnp.float64_t, ndim=1] h_p
    h_p = np.zeros(n, dtype=np.float64)
    cdef double[:] h_p_view = h_p
    cdef double sqrt2 = sqrt(2)
    cdef double prefac_p1 =  eta_p1/(2*tau_p1)*exp(0.5*sq(sigma/tau_p1))
    cdef double prefac_p2 = eta_p2/(2*tau_p2)*exp(0.5*sq(sigma/tau_p2))
    cdef double prefac_p3 = eta_p3/(2*tau_p3)*exp(0.5*sq(sigma/tau_p3))
    for i in range(n):
        h_p1 = prefac_p1*exp( - (x[i]-mu)/tau_p1 )*erfc( sigma/(sqrt2*tau_p1) - (x[i]-mu)/(sqrt2*sigma) )
        h_p2 = prefac_p2*exp( - (x[i]-mu)/tau_p2 )*erfc( sigma/(sqrt2*tau_p2) - (x[i]-mu)/(sqrt2*sigma) )
        h_p3 = prefac_p3*exp( - (x[i]-mu)/tau_p3 )*erfc( sigma/(sqrt2*tau_p3) - (x[i]-mu)/(sqrt2*sigma) )
        h_p_view[i] = h_p1 + h_p2 + h_p3
    return h_p

def h_emg(x, mu, sigma , theta, *t_args):
    """Hyper-exponentially-modified Gaussian distribution (hyper-EMG).

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
    t_args : :class:`tuple`, :class:`tuple`, :class:`tuple`, :class:`tuple`
        Four variable-length tuples of neg. and pos. tail shape arguments with
        the signature: ``(eta_m1, eta_m2, ...), (tau_m1, tau_m2, ...), (eta_p1, eta_p2, ...), (tau_p1, tau_p2, ...)``.
        The length of the first and last two tuples defines the order of neg.
        and pos. tails, respectively.

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
        #h_m = h_m_emg(x, mu, sigma, li_eta_m, li_tau_m)
        #h_p = h_p_emg(x, mu, sigma, li_eta_p, li_tau_p)
        #h = ne.evaluate('theta*h_m + (1-theta)*h_p',optimization='moderate')
    return  h #np.clip(h,a_min=None,a_max=1e295) # clipping to avoid overflow errors
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
    """Calculate mean of hyper-EMG distribution.

    Parameters
    ----------
    mu : float >= 0
        Mean value of underlying Gaussian distribution.
    theta : float, 0 <= theta <= 1
        Left-right-weight factor (negative-skewed EMG weight: theta;
        positive-skewed EMG weight: 1 - theta).
    t_args : :class:`tuple`, :class:`tuple`, :class:`tuple`, :class:`tuple`
        Four variable-length tuples of neg. and pos. tail shape arguments with
        the signature: ``(eta_m1, eta_m2, ...), (tau_m1, tau_m2, ...), (eta_p1, eta_p2, ...), (tau_p1, tau_p2, ...)``.

    Returns
    -------
    float
        Mean of hyper-EMG distribution.

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
    """Calculate standard deviation of hyper-EMG distribution.

    Parameters
    ----------
    sigma : float >= 0
        Standard deviation of underlying Gaussian distribution.
    theta : float, 0 <= theta <= 1
        Left-right-weight factor (negative-skewed EMG weight: theta;
        positive-skewed EMG weight: 1 - theta).
    t_args : :class:`tuple`, :class:`tuple`, :class:`tuple`, :class:`tuple`
        Four variable-length tuples of neg. and pos. tail shape arguments with
        the signature: ``(eta_m1, eta_m2, ...), (tau_m1, tau_m2, ...), (eta_p1, eta_p2, ...), (tau_p1, tau_p2, ...)``.

    Returns
    -------
    float
        Standard deviation of hyper-EMG distribution.

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


################################################################################
