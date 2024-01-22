import pytest
import numpy as np
import emgfit as emg
import emgfit.hypothesis_tests as tests
from emgfit.sample import simulate_events

class TestHypothesisTests:
    def test_run_GV_likelihood_ratio_test(self):
        """Test GV likelihood ratio test with simulated data"""
        # Define parameters of reference spectrum
        shape_pars = emg.fit_models.create_default_init_pars(mass_number=100)
        mus = [100.00134, 100.00134 + 900e-06]
        amps = [1., 0.006]
        bkg_c = 1.
        x_range = 0.015
        x_min, x_max = mus[0] - x_range/2, mus[0] + x_range/2
        N_events = 10000
        bin_width = 20e-06
        N_bins = int(np.round((x_max - x_min)/bin_width))

        # Simulate spectrum
        np.random.seed(1543) 
        data = simulate_events(shape_pars, mus, amps, bkg_c, N_events, 
                               x_min, x_max, N_bins=N_bins)
        spec = emg.spectrum(df=data, show_plot=False)
        spec.add_peak(mus[0]+20e-06)

        # Set peak shape to the true peak shape used for the event sampling
        spec.index_shape_calib = 0
        spec.shape_cal_pars = shape_pars
        spec.fit_model = "emg33"

        # Fit data with null-model
        spec.set_lit_values(0, 100.00134, 0.0)
        spec.show_peak_properties()
        spec.fit_peaks(sigmas_of_conf_band=3, index_mass_calib=0, 
                       show_plots=False)

        # Define constants for LRTs
        alt_mu = mus[0] + 910e-06
        alt_x_min = alt_mu -  992e-06
        alt_x_max = alt_mu + 1008e-06 
        GV_seed = 42
        steps = 100
        N_GV_spectra = 5 #100

        # Run GV LRT
        LRT_results = tests.run_GV_likelihood_ratio_test(
                          spec, 0, alt_x_min, alt_x_max, alt_x_steps=steps,
                          min_significance=3, N_spectra=N_GV_spectra, c0=0.5, 
                          seed=GV_seed, show_fits=False, show_upcrossings=False)


        assert np.isclose(LRT_results["LLR"], 15.71, rtol=1e-03, atol=1e-02)
        assert np.isclose(LRT_results["p-value"], 1.23e-03, rtol=1e-03, 
                          atol=1e-05)
        assert np.isclose(LRT_results["p-value error"], 2.00e-04, rtol=1e-03, 
                          atol=1e-05)
        assert LRT_results["reject_null_model"] is True


    def test_MC_likelihood_ratio_test(self):
        """Test MC likelihood ratio test with simulated data"""
        # Define parameters of reference spectrum
        shape_pars = emg.fit_models.create_default_init_pars(mass_number=100)
        mus = [100.00134, 100.00134 + 900e-06]
        amps = [1., 0.006]
        bkg_c = 1.
        x_range = 0.015
        x_min, x_max = mus[0] - x_range/2, mus[0] + x_range/2
        N_events = 10000
        bin_width = 20e-06
        N_bins = int(np.round((x_max - x_min)/bin_width))

        # Simulate spectrum
        np.random.seed(1543) 
        data = simulate_events(shape_pars, mus, amps, bkg_c, N_events, 
                               x_min, x_max, N_bins=N_bins)
        spec = emg.spectrum(df=data, show_plot=False)
        spec.add_peak(mus[0]+20e-06)

        # Set peak shape to the true peak shape used for the event sampling
        spec.index_shape_calib = 0
        spec.shape_cal_pars = shape_pars
        spec.fit_model = "emg33"

        # Fit data with null-model
        spec.set_lit_values(0, 100.00134, 0.0)
        spec.show_peak_properties()
        spec.fit_peaks(sigmas_of_conf_band=3, index_mass_calib=0, 
                       show_plots=False)

        # Define constants for LRTs
        alt_mu = mus[0] + 910e-06
        alt_x_min = alt_mu -  992e-06
        alt_x_max = alt_mu + 1008e-06
        seed = 42
        N_MC_spectra = 5000 
        from emgfit.fit_models import get_mu0
        alt_mu_min = get_mu0(alt_x_min, spec.shape_cal_pars, spec.fit_model)
        alt_mu_max = get_mu0(alt_x_max, spec.shape_cal_pars, spec.fit_model)

        # Run MC LRT
        MC_LRT_results = tests.run_MC_likelihood_ratio_test(
                             spec, 0, alt_mu, alt_mu_min=alt_mu_min, 
                             alt_mu_max=alt_mu_max, N_spectra=N_MC_spectra, 
                             seed=seed, min_significance=3, show_plots=False, 
                             show_results=False, show_LLR_hist=False)

        assert np.isclose(MC_LRT_results["LLR"], 15.71, rtol=1e-03, atol=1e-02)
        assert np.isclose(MC_LRT_results["p-value"], 0.000600120024004,
                          rtol=1e-02, atol=1e-05)
        assert np.isclose(MC_LRT_results["p-value error"], 0.00034647945740525,
                          rtol=1e-02, atol=1e-05)
        assert MC_LRT_results["reject_null_model"] is True 
