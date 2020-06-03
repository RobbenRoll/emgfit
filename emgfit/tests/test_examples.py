def test_one_plus_one_is_two():
    "Check that one and one are indeed two."
    assert 1 + 1 == 2

import emgfit as emg
import numpy as np

def test_h_emg_for_nan():
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
    x = np.linspace(mu-0.01,mu+0.01,10000)
    output = emg.emg_funcs.h_emg(x, mu, sigma , theta, (eta_m1,eta_m2,eta_m3),
                                 (tau_m1,tau_m2,tau_m3),(eta_p1,eta_p2,eta_p3),
                                 (tau_p1,tau_p2,tau_p3))
    assert np.all(np.isfinite(output))
