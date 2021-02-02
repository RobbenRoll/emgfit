import pytest
import emgfit as emg
import numpy as np

class Test_spectrum:

    def test_fitting_accuracy(self):
        """Check accuracy of fitting using simulated spectrum and test
        calculation of literature values for doubly charged and isomeric species

        """
        # Create simulated spectrum data
        from emgfit.sample import simulate_events
        true_sigma = 7.77901056381226e-05
        true_theta = 0.6591808159640057
        true_eta_m1 = 0.7393102752716145
        true_eta_m2 = 0.2606897247283855
        true_tau_m1 = 4.4723478031626915e-05
        true_tau_m2 = 0.00011112601042960299
        true_eta_p1 = 0.7315780388972555
        true_eta_p2 = 0.2684219611027445
        true_tau_p1 = 7.130854298242941e-05
        true_tau_p2 = 0.0002741372066519157
        true_bkg_c = 1.036125336704966
        shape_pars = {'sigma' : true_sigma,
                      'theta' : true_theta,
                      'eta_m1': true_eta_m1,
                      'eta_m2': true_eta_m2,
                      'tau_m1': true_tau_m1,
                      'tau_m2': true_tau_m2,
                      'eta_p1': true_eta_p1,
                      'eta_p2': true_eta_p2,
                      'tau_p1': true_tau_p1,
                      'tau_p2': true_tau_p2,
                      'bkg_c' : true_bkg_c}

        true_mus = [57.93479320009094, 57.935203, 57.93959511435116,
                    115.90064566418187/2]
        true_amps = [0.38916170, 0.05940254, 0.94656384, 0.20934518]
        true_N_events = 67636
        x_min = true_mus[0] - 0.004
        x_max = true_mus[-1] + 0.005
        bin_width = 2.37221e-05
        N_bins = int((x_max - x_min)/bin_width)

        # Set random seed for reproducibility, other seeds can result in
        # assertion errors below
        np.random.seed(12)
        data = simulate_events(shape_pars, true_mus, true_amps, true_bkg_c,
                               true_N_events, x_min, x_max, out='hist',
                               N_bins=N_bins)

        # Instantiate spectrum object, calibrate peak shape and fit all peaks
        spec = emg.spectrum(df=data,show_plot=False)
        spec.detect_peaks(thres=0.0053, plot_smoothed_spec=False,
                          plot_2nd_deriv=False, plot_detection_result=False)
        msg0 = "Incorrect number of peaks detected."
        assert len(spec.peaks) == len(true_mus), msg0
        spec.assign_species(["Ni58:-1e","Co58:-1e","Mn58?:-1e","Sn116:-2e"])
        spec.assign_species("Mn58m?:-1e", peak_index=2, Ex=71.77, Ex_error=0.05)
        spec.determine_peak_shape(species_shape_calib="Mn58m?:-1e",
                                  show_plots=False)
        spec.fit_peaks(species_mass_calib="Ni58:-1e",show_plots=False)

        # Perform checks
        for p in spec.peaks:
            if p.species == "Ni58:-1e":
                continue # skip calibrant
            msg1 = "ME deviates from literature by more than 1 sigma."
            assert p.m_dev_keV <= p.mass_error_keV, msg1

            # Check calculation of (atomic) ME for doubly charged species
            if p.species == "Sn116:-2e":
                atomic_ME_lit = -91525.97
                ME_dev_keV = p.atomic_ME_keV - atomic_ME_lit
                msg2 = str("Respective deviation of ionic mass and atomic mass "
                           "excess from literature differ by > 1 sigma for "
                           "Sn116:-2e.")
                assert abs(ME_dev_keV - p.m_dev_keV) < p.mass_error_keV, msg2
