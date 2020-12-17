import pytest
import emgfit as emg
import numpy as np


class Test_h_emg(object):

    def h_emg_mpmath(self, x, mu, sigma , theta, li_eta_m, li_tau_m, li_eta_p,
                     li_tau_p):
        norm_precision = 1e-06
        # Define precision Hyper-EMG components with mpmath functions
        def h_m(x, mu, sigma, li_eta_m,li_tau_m):
            t_order_m = len(li_eta_m) # order of negative tail exponentials
            if abs(sum(li_eta_m) - 1) > norm_precision:  # check normalization of eta_m's
                raise Exception("eta_m's don't add up to 1.")
            if len(li_tau_m) != t_order_m:  # check if all arguments match tail order
                raise Exception("orders of eta_m and tau_m do not match!")

            h_m = 0.
            for i in range(t_order_m):
                eta_m = li_eta_m[i]
                tau_m = li_tau_m[i]
                h_m += emg.emg_funcs.h_m_i_prec(x,mu,sigma,eta_m,tau_m)
            return h_m

        def h_p(x, mu, sigma, li_eta_p, li_tau_p):
            t_order_p = len(li_eta_p) # order of positive tails
            if abs(sum(li_eta_p) - 1) > norm_precision:  # check normalization of eta_p's
                raise Exception("eta_p's don't add up to 1.")
            if len(li_tau_p) != t_order_p:  # check if all arguments match tail order
                raise Exception("orders of eta_p and tau_p do not match!")

            h_p = 0.
            for i in range(t_order_p):
                eta_p = li_eta_p[i]
                tau_p = li_tau_p[i]
                h_p += emg.emg_funcs.h_p_i_prec(x,mu,sigma,eta_p,tau_p)
            return h_p

        # Calculate full Hyper-EMG function
        if theta == 1:
            h = h_m(x, mu, sigma, li_eta_m, li_tau_m)
        elif theta == 0:
            h = h_p(x, mu, sigma, li_eta_p, li_tau_p)
        else:
            h = theta*h_m(x, mu, sigma, li_eta_m, li_tau_m) + (1-theta)*h_p(x, mu, sigma, li_eta_p, li_tau_p)
        return h

    def test_accuracy(self):
        """Test accuracy of h_emg against its mpmath version """
        scl_factor = 1
        amp = 0.45*scl_factor
        mu = 99.912
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
        x = np.linspace(mu-10,mu+10,10000)

        ret = emg.emg_funcs.h_emg(x, mu, sigma , theta,
                                  (eta_m1,eta_m2,eta_m3),
                                  (tau_m1,tau_m2,tau_m3),
                                  (eta_p1,eta_p2,eta_p3),
                                  (tau_p1,tau_p2,tau_p3))
        ret_mpmath = self.h_emg_mpmath(x, mu, sigma , theta,
                                       (eta_m1,eta_m2,eta_m3),
                                       (tau_m1,tau_m2,tau_m3),
                                       (eta_p1,eta_p2,eta_p3),
                                       (tau_p1,tau_p2,tau_p3))

        assert np.allclose(ret,ret_mpmath,rtol=1e-12)

    def test_extreme_args(self):
        """Check for finiteness at extreme arguments """
        scl_factor = 1
        amp = 0.45*scl_factor
        mu = 99.912
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
        x0 = np.array([0.,50.,150.,200.,300.])

        ret = emg.emg_funcs.h_emg(x0, mu, sigma , theta, (eta_m1,eta_m2,eta_m3),
                                  (tau_m1,tau_m2,tau_m3),(eta_p1,eta_p2,eta_p3),
                                  (tau_p1,tau_p2,tau_p3))

        assert np.all(np.isfinite(ret))
