import pytest
import emgfit
import numpy as np

@pytest.fixture
def test_sampling():
    """Helper function to check result of simulate_events() against target PDF
    """
    def test_sampling_func(modname, mu, shape_pars, bkg_c=0, scl_fac=1.0,
                           N_events=10000000):
        x_min = mu - 10*shape_pars["sigma"]
        x_max = mu + 10*shape_pars["sigma"]

        # Simulate events
        from emgfit.sample import simulate_events
        N_bins = int((x_max-x_min)/(0.1*shape_pars["sigma"]))
        bin_width = (x_max - x_min)/N_bins
        amp = N_events*bin_width

        np.random.seed(0)
        data = simulate_events(shape_pars, mu, amp, bkg_c, N_events, x_min,
                               x_max, scl_facs=scl_fac, out='hist',
                               N_bins=N_bins)

        # Check simulated events against generating PDF
        x = data.index.values
        y_sim = data.values.flatten()

        import emgfit.fit_models as fit_models
        modfunc = getattr(fit_models, modname)
        scaled_pars = fit_models.scl_init_pars(shape_pars, scl_coeff=scl_fac)
        mod = modfunc(0, mu, amp, init_pars=scaled_pars, vary_shape_pars=True)
        pars = mod.make_params()
        y_PDF = mod.eval(pars,x=x) # target distribution

        allclose = np.allclose(y_sim, y_PDF, atol=15, rtol=15e-01)
        msg0 = "y_sim & y_PDF differ by >15% and >15counts in some bins."
        assert allclose, msg0

        mean_residual = np.mean(y_PDF - y_sim)
        msg1 = "The mean of y_sim - y_PDF differs from 0 by more than 1."
        assert abs(mean_residual) <= 1, msg1

    return test_sampling_func

class TestSampling():

    def test_Gaussian_sampling(self, test_sampling):
        """Test simulate_events() for sampling from Gaussian PDF"""
        modname = "Gaussian"
        mu = 1000.151
        sigma = 0.5
        shape_pars = {'sigma': sigma}
        test_sampling(modname, mu, shape_pars)

    def test_emg10_sampling(self, test_sampling):
        modname = "emg10"
        mu = 100.1
        sigma = 0.0001
        tau_m1 = 90e-06
        shape_pars = {'sigma': sigma,
                      'tau_m1': tau_m1}
        test_sampling(modname, mu, shape_pars)

    def test_emg21_sampling(self, test_sampling):
        """Test simulate_events() for sampling from emg21 PDF with scaling"""
        modname = "emg21"
        mu = 10.1
        sigma = 0.0005
        theta = 0.41
        eta_m1 = 0.9
        eta_m2 = 1 - eta_m1
        tau_m1 = 900e-06
        tau_m2 = 2000e-06
        tau_p1 = 320e-06
        shape_pars = {'sigma': sigma,
                      'theta': theta,
                      'eta_m1': eta_m1,
                      'eta_m2': eta_m2,
                      'tau_m1': tau_m1,
                      'tau_m2': tau_m2,
                      'tau_p1': tau_p1}
        scl_fac = 0.4
        test_sampling(modname, mu, shape_pars, scl_fac=0.4)


    def test_emg33_sampling(self, test_sampling):
        """Test simulate_events() for sampling from emg33 PDF"""
        modname = "emg33"
        mu = 100.17654
        sigma = 0.005
        theta = 0.81
        eta_m1 = 0.6
        eta_m2 = 0.2
        eta_m3 = 1 - eta_m1 - eta_m2
        tau_m1 = 30e-06
        tau_m2 = 870e-06
        tau_m3 = 2500e-06
        eta_p1 = 0.8
        eta_p2 = 0.15
        eta_p3 = 1 - eta_p1 - eta_p2
        tau_p1 = 320e-06
        tau_p2 = 691e-06
        tau_p3 = 2100e-06
        shape_pars = {'sigma': sigma,
                      'theta': theta,
                      'eta_m1': eta_m1,
                      'eta_m2': eta_m2,
                      'eta_m3': eta_m3,
                      'tau_m1': tau_m1,
                      'tau_m2': tau_m2,
                      'tau_m3': tau_m3,
                      'eta_p1': eta_p1,
                      'eta_p2': eta_p2,
                      'eta_p3': eta_p3,
                      'tau_p1': tau_p1,
                      'tau_p2': tau_p2,
                      'tau_p3': tau_p3}
        test_sampling(modname, mu, shape_pars)

    def test_simulate_spectrum(self):
        # Define spectrum parameters
        modname = "emg21"
        sigma = 0.000045
        theta = 0.41
        eta_m1 = 0.7
        eta_m2 = 1 - eta_m1
        tau_m1 = 30e-06
        tau_m2 = 300e-06
        tau_p1 = 120e-06
        shape_pars = {'sigma':sigma,
                      'theta': theta,
                      'eta_m1': eta_m1,
                      'eta_m2': eta_m2,
                      'tau_m1': tau_m1,
                      'tau_m2': tau_m2,
                      'tau_p1': tau_p1}

        mus = [57, 57.0012]
        x_min = mus[0] - 20*shape_pars["sigma"]
        x_max = mus[-1] + 20*shape_pars["sigma"]
        N_events = 10000000
        bkg_c = 1.5e03

        # Simulate events, create reference spectrum and fit
        from emgfit.sample import simulate_events, simulate_spectrum
        np.random.seed(49)
        N_bins = int((x_max-x_min)/(0.1*shape_pars["sigma"]))
        bin_width = (x_max - x_min)/N_bins
        N_bkg = 0.4*N_bins
        amps = np.array([0.3,0.7])*(N_events-N_bkg)*bin_width
        data = simulate_events(shape_pars, mus, amps, bkg_c, N_events, x_min,
                               x_max, out='hist', N_bins=N_bins)
        spec = emgfit.spectrum(df=data, show_plot=False)
        spec.add_peak(57, verbose= False)
        spec.add_peak(57.0012, verbose = False)
        m, m_err = 57, 1e-06
        spec.set_lit_values(0, m, m_err, verbose = False)
        spec.determine_peak_shape(index_shape_calib=1, x_fit_range=20*sigma,
                                  fit_model=modname, vary_tail_order=False,
                                  show_plots=False)
        spec.fit_peaks(index_mass_calib=0, show_plots=False)

        # Create new simulated spectrum from reference spectrum
        new_spec = simulate_spectrum(spec, copy_spec=False)

        data_ref = spec.data
        data_sim = new_spec.data
        msg0 = "Bin positions x_sim and x_ref disagree by >1e-12."
        assert np.allclose(data_ref.index,data_sim.index, rtol=1e-12), msg0
        msg1 = "Counts y_sim and y_ref deviate by >15% in some bins."
        assert np.allclose(data_ref.values, data_sim.values, rtol=0.15), msg1
