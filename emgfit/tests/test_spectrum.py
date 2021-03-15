import pytest
import emgfit as emg
import numpy as np

class Test_spectrum:
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

    # Get literature mass values from AME2020
    m_e = 0.000548579909065 # CODATA value from physics.nist.gov
    m_Ni58 = 57.935341650
    m_err_Ni58 = 0.374e-06
    m_Co58 = 57.935751292
    m_err_Co58 = 1.237e-06
    m_Mn58 = 57.940066643
    m_err_Mn58 = 2.900e-06
    m_Sn116 = 115.901742825
    m_err_Sn116 = 0.103
    ME_Sn116_keV = -91525.979

    true_mus = [m_Ni58 - m_e, m_Co58 - m_e, m_Mn58 - m_e, m_Sn116/2 - m_e] #[57.93479320009094, 57.935203, 57.93959511435116,
               #    115.90064566418187/2]
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


    def test_grabbing_of_AME_values(self):
        # Define reference literature values

        m_Ni58_AME16 = 57.935341780 - self.m_e
        m_err_Ni58_AME16 = 0.400e-06
        m_Co58_AME16 = 57.935751429 - self.m_e
        m_err_Co58_AME16 = 1.245e-06
        atol = 1e-09 # tolerance [u] up to which absolute agreement is demanded

        # Instantiate spectrum object
        spec = emg.spectrum(df=self.data, show_plot=False)
        spec.add_peak(57.9, species="Ni58:-1e")
        spec.add_peak(57.95, species="Co58:-1e", lit_src="AME2016")

        # Test defaulting to most recent AME database
        p0 = spec.peaks[0]
        msg0 = "default m_AME value of 'Ni58:-1e' deviates from AME2020 value"
        assert np.isclose(p0.m_AME, self.m_Ni58, atol=atol), msg0
        msg1 = "default m_AME_error of 'Ni58:-1e' deviates from AME2020 value"
        assert np.isclose(p0.m_AME_error, self.m_err_Ni58, atol=atol), msg1

        # Test switching to older AME database via add_peak()
        p1 = spec.peaks[1]
        msg2 = "AME2016 value invoked with add_peak() deviates from reference"
        assert np.isclose(p1.m_AME, m_Co58_AME16, atol), msg2
        msg3 = "AME2016 error invoked with add_peak() deviates from reference"
        assert np.isclose(p1.m_AME_error, m_err_Co58_AME16, atol), msg3
        msg4 = "Flagging for AME2016 values invoked with add_peak() failed"
        assert 'lit_src: AME2016' in p1.comment, msg4

        # Test switching to older AME database via assign_species()
        spec.assign_species("Ni58:-1e", peak_index=0, lit_src = 'AME2016')
        msg5 = "AME2016 value invoked with assign_species() deviates from ref."
        assert np.isclose(p0.m_AME, m_Ni58_AME16, atol), msg5
        msg6 = "AME2016 error invoked with assign_species() deviates from ref."
        assert np.isclose(p0.m_AME_error, m_err_Ni58_AME16, atol), msg6
        msg7 = "Flagging for AME16 values invoked with assign_species() failed"
        assert 'lit_src: AME2016' in p0.comment, msg7


    def test_fitting_accuracy(self):
        """Check accuracy of fitting using simulated spectrum and test
        calculation of literature values for doubly charged and isomeric species

        """
        # Instantiate spectrum object, calibrate peak shape and fit all peaks
        spec = emg.spectrum(df=self.data,show_plot=False)
        spec.detect_peaks(thres=0.0053, plot_smoothed_spec=False,
                          plot_2nd_deriv=False, plot_detection_result=False)
        msg0 = "Incorrect number of peaks detected."
        assert len(spec.peaks) == len(self.true_mus), msg0
        spec.assign_species(["Ni58:-1e","Co58:-1e","Mn58?:-1e","Sn116:-2e"])
        spec.assign_species("Mn58m?:-1e", peak_index=2, Ex=71.77, Ex_error=0.05)
        spec.determine_peak_shape(species_shape_calib="Mn58m?:-1e",
                                  show_plots=False)
        spec.fit_peaks(species_mass_calib="Ni58:-1e",show_plots=False)

        # Perform accuracy checks
        for p in spec.peaks:
            if p.species == "Ni58:-1e":
                continue # skip calibrant
            msg1 = "ME deviates from literature by more than 1 sigma."
            assert p.m_dev_keV <= p.mass_error_keV, msg1

            # Check calculation of (atomic) ME for doubly charged species
            if p.species == "Sn116:-2e":
                ME_dev_keV = p.atomic_ME_keV - self.ME_Sn116_keV
                msg2 = str("Respective deviation of ionic mass and atomic mass "
                           "excess from literature differ by > 1 sigma for "
                           "Sn116:-2e.")
                assert abs(ME_dev_keV - p.m_dev_keV) < p.mass_error_keV, msg2
