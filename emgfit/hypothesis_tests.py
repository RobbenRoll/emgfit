################################################################################
##### Python module for performing hypothesis tests on TOF mass spectra
##### Author: Stefan Paul

##### Import packages
import copy
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm.auto import tqdm
from emgfit.sample import simulate_spectrum

################################################################################
def _likelihood_ratio_test(spec, null_result_index, alt_x_pos, x_fit_cen=None,
                           x_fit_range=None, vary_alt_mu=True,
                           vary_baseline=True, verbose=False,
                           show_plots=False, show_results=False):
    """Perform a local likelihood ratio test on the specified spectrum

    Parameters
    ----------
    spec : :class:`emgfit.spectrum.spectrum`
        Spectrum object to perform likelihood ratio test on.
    null_result_index : int
        Index (of one) of the peak(s) present in the null-model fit.
    alt_x_pos : float [u]
        Position of the hypothesized alternative peak.
    x_fit_cen : float [u], optional
        Center of the x-range to fit. Defaults to the center of the fit result
        asociated with `null_result_index`.
    x_fit_range : float [u], optional
        Width of the x-range to fit.  Defaults to the range of the fit result
        asociated with `null_result_index`.
    vary_alt_mu : bool, optional
        Whether to vary the alternative-peak centroid in the fit.
    vary_baseline : bool, optional
        If `True`, the constant background will be fitted with a varying
        uniform baseline parameter `bkg_c`. If `False`, the baseline parameter
        `bkg_c` will be fixed to 0.
    verbose : bool, optional
        Whether to print status updates and results.
    show_plots : bool, optional
        Whether to show plots of the fit results.
    show_results : bool, optional
        Whether to display reports with the fit results.



    #TODO
    """
    try:
        fit_model = spec.fit_model
    except:
        raise
        print("Could not define the fit model to use. Ensure that a "
              "successful peak-shape calibration has been performed.")
    ref_result = spec.fit_results[null_result_index]
    if x_fit_cen is None:
        x_fit_cen = ref_result.x_fit_cen
    if x_fit_range is None:
        x_fit_range = ref_result.x_fit_range
    if ref_result.cost_func != "MLE":
        raise Exception("This method is only applicable to fits with the "
                        "'MLE' cost function." )
    if verbose:
        print("# Fit data with null model #")
    try:
        null_result = spec.peakfit(x_fit_cen=x_fit_cen,
                                   x_fit_range=x_fit_range,
                                   fit_model=fit_model,
                                   cost_func='MLE',
                                   vary_baseline=vary_baseline,
                                   show_plots=show_plots)

        if show_results:
            display(null_result)
        null_LLR = null_result.chisqr
    except ValueError:
        warnings.warn("Fit with null model failed failed with ValueError.")
        null_result = None
        null_LLR = np.nan

    alt_spec = copy.deepcopy(spec)
    if verbose:
        print("# Fit data with alternative model #")
    alt_spec.add_peak(x_pos=alt_x_pos, verbose=verbose)

    if not vary_alt_mu:
        # Fix centroid of alternative peak
        alt_peak = [p for p in alt_spec.peaks if p.x_pos==alt_x_pos][0]
        pref_alt_peak = "p{}_".format(alt_spec.peaks.index(alt_peak))
        par_hint_args = {pref_alt_peak+"mu" : {"vary" : False}}
    else:
        par_hint_args = {}

    try: #TODO: FIx fit range to same range as for null model fit?!
        alt_result = alt_spec.peakfit(x_fit_cen=null_result.x_fit_cen,
                                      x_fit_range=null_result.x_fit_range,
                                      fit_model=fit_model,
                                      cost_func='MLE',
                                      vary_baseline=vary_baseline,
                                      show_plots=show_plots,
                                      par_hint_args=par_hint_args)
        if show_results:
            display(alt_result)
        alt_LLR = alt_result.chisqr
    except ValueError:
        warnings.warn("Fit with alternative model failed with ValueError.")
        alt_result = None
        alt_LLR = np.nan

    # Calculate (doubled) log-likelihood ratio
    LLR = null_LLR - alt_LLR
    from scipy.stats import chi2
    dof = 1 # difference in number of free parameters
    p_val = chi2.sf(LLR, dof, loc=0, scale=1) # sf := 1 - cdf
    if verbose:
        print("Log-likelihood ratio test statistic:  LLR = {:.2f}".format(LLR))

    return LLR, null_result, alt_result


def run_serial_MC_likelihood_ratio_test(spec, null_result_index, alt_x_pos,
                                        min_significance=3, N_spectra=10000):
    """
    spec : :class:`emgfit.spectrum.spectrum`
        Sprectrum object to perform MC
    null_result_index : int
        Index (of one) of the peak(s) present in the null-model fit.

    """
    from scipy.stats import norm
    alpha = norm.sf(min_significance, loc=0, scale=1) # sf := 1 - cdf

    ref_null_result = spec.fit_results[null_result_index]
    if ref_null_result.cost_func != "MLE":
        raise Exception("The likelihood ratio test is only compatible with "
                        "the `MLE` cost function.")
    x_fit_cen = ref_null_result.x_fit_cen
    x_fit_range = ref_null_result.x_fit_range

    print("\n##### Performing Monte Carlo likelihood ratio test #####")
    print("N_spectra:",N_spectra)
    print(f"Test at {min_significance:.1f} sigma significance level, i.e. alpha = {alpha:.2e}")
    print()

    # Perform LRT on the observed data
    print("### Determine test statistic for observed data ###")
    LLR, null_res, alt_res = _likelihood_ratio_test(spec, null_result_index,
                                                    alt_x_pos,
                                                    verbose=True,
                                                    show_results=False,
                                                    show_plots=True,
                                                    vary_alt_mu=True)

    print("\n### Determine test statistic for simulated Monte Carlo spectra ###")
    # Run LRTs on spectra sampled from best null-model fit of the observed data
    MC_LLRs = []
    for i in tqdm(range(N_spectra)):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            sim_spec = simulate_spectrum(spec, copy_spec=True)

        LLR_i, _, _ = _likelihood_ratio_test(sim_spec, null_result_index,
                                             alt_x_pos,
                                             verbose=False,
                                             show_results=False,
                                             show_plots=False,
                                             vary_alt_mu=True)
        MC_LLRs.append(LLR_i)

    #plt.figure(figsize=(8,12))
    plt.hist(MC_LLRs, density=False, bins=20)
    plt.gca().axvline(LLR, color="black")
    plt.xlabel("Likelihood ratio test statistic")
    plt.ylabel("Occurences")
    plt.yscale("log")
    plt.show()

    # Determine p-value from the fraction of LLR samples above the obs. LLR
    N_above = np.sum(np.where(np.array(MC_LLRs) > LLR, 1, 0))
    N_tot = np.sum(np.isfinite(np.array(MC_LLRs)))
    p_val = N_above/N_tot
    if N_above > 0:
        p_val_err = np.sqrt(p_val/N_tot)
    elif N_above == 0:
        p_val_err = np.sqrt(1/N_tot)
    print("Monte Carlo p-value: p=", p_val,"+-", p_val_err)
    if p_val_err < alpha:
        success= True
        # Compare to defined alpha
        if p_val < alpha:
            print(f"p < alpha = {alpha:.2e} => Reject null model in favor of "
                   "alternative model with additional peak!")
            reject_null_model = True
        else:
            print(f"p > alpha = {alpha:.2e} => Reject alternative model in favor "
                   "of null model with fewer peaks!")
            reject_null_model = False
    else:
        warnings.warn("MC error of the p-value >= alpha. "
                      "Re-run with larger `N_spectra`!")
        success = False
        reject_null_model = None

    LRT_results = {"success" : success,
                   "LLR" : LLR,
                   "MC LLRs" : MC_LLRs,
                   "p-value" : p_val,
                   "p-value error" : p_val_err,
                   "reject_null_model" : reject_null_model}
    return LRT_results


def run_MC_likelihood_ratio_test(spec, null_result_index, alt_x_pos,
                                 min_significance=3, N_spectra=10000,
                                 seed=None, n_cores=-1):
    """Perform Monte Carlo likelihood ratio test by fitting simulated spectra

    The simulated spectra are sampled from the null model stored in the
    spectrum's fit results list and asociated with the peak index
    `null_result_index`.


    spec : :class:`emgfit.spectrum.spectrum`
        Spectrum object to perform likelihood ratio test on.
    null_result_index : int
        Index (of one) of the peak(s) present in the null-model fit.
    alt_x_pos : float, optional
        Whether to

    """
    from scipy.stats import norm
    alpha = norm.sf(min_significance, loc=0, scale=1) # sf := 1 - cdf

    ref_null_result = spec.fit_results[null_result_index]
    if ref_null_result.cost_func != "MLE":
        raise Exception("The likelihood ratio test is only compatible with "
                        "the `MLE` cost function.")
    x_fit_cen = ref_null_result.x_fit_cen
    x_fit_range = ref_null_result.x_fit_range

    print("\n##### Performing Monte Carlo likelihood ratio test #####")
    print("N_spectra:",N_spectra)
    print(f"Test at {min_significance:.1f} sigma significance level, i.e. alpha = {alpha:.2e}")
    print()

    # Perform LRT on the observed data
    print("### Determine test statistic for observed data ###")
    LLR, _, alt_res = _likelihood_ratio_test(spec, null_result_index,
                                             alt_x_pos,
                                             verbose=True,
                                             show_results=False,
                                             show_plots=True,
                                             vary_alt_mu=True)

    # Run LRTs on spectra sampled from best null-model fit of the observed data
    print("\n### Determine test statistic for simulated Monte Carlo spectra ###")
    if seed is None:
        seed = np.random.randint(2**31)
    from emgfit.sample import fit_simulated_spectra
    null_results = fit_simulated_spectra(spec, ref_null_result,
                                         N_spectra=N_spectra,
                                         seed=seed, n_cores=n_cores)
    alt_results = fit_simulated_spectra(spec, ref_null_result,
                                        alt_result=alt_res,
                                        N_spectra=N_spectra,
                                        seed=seed, n_cores=n_cores)
    # Calculate Monte Carlo LRT statistics
    MC_LLRs = []
    for null, alt in zip(null_results, alt_results):
        if null is None or alt is None:
            MC_LLRs.append(np.nan)
        elif null.success is False or alt.success is False:
            MC_LLRs.append(np.nan)
        else:
            MC_LLRs.append(null.chisqr - alt.chisqr)

    #plt.figure(figsize=(8,12))
    plt.hist(MC_LLRs, density=False, bins=20)
    plt.gca().axvline(LLR, color="black")
    plt.xlabel("Likelihood ratio test statistic")
    plt.ylabel("Occurences")
    plt.yscale("log")
    plt.show()

    # Determine p-value from the fraction of LLR samples above the obs. LLR
    N_above = np.sum(np.where(np.array(MC_LLRs) > LLR, 1, 0))
    N_tot = np.sum(np.isfinite(np.array(MC_LLRs)))
    p_val = N_above/N_tot
    if N_above > 0:
        p_val_err = np.sqrt(p_val/N_tot)
    elif N_above == 0:
        p_val_err = np.sqrt(1/N_tot)
    print("Monte Carlo p-value: p =", p_val,"+-", p_val_err)
    if p_val_err < alpha:
        success= True
        # Compare to defined alpha
        if p_val < alpha:
            print(f"p < alpha = {alpha:.2e} => Reject null model in favor of "
                   "alternative model with additional peak!")
            reject_null_model = True
        else:
            print(f"p > alpha = {alpha:.2e} => Reject alternative model in favor "
                   "of null model with fewer peaks!")
            reject_null_model = False
    else:
        warnings.warn("MC error of the p-value >= alpha. "
                      "Re-run with larger `N_spectra`!")
        success = False
        reject_null_model = None

    LRT_results = {"success" : success,
                   "LLR" : LLR,
                   "MC LLRs" : MC_LLRs,
                   "p-value" : p_val,
                   "p-value error" : p_val_err,
                   "reject_null_model" : reject_null_model}
    return LRT_results


def run_GV_LLR_test(spec, alt_x_min, alt_x_max, alt_x_steps=100,
                    min_significance=3, N_spectra=100, c0=0.5, seed=None,
                    show_upcrossings=True, show_fits=True):
    """Perform a likelihood ratio test following the method of Gross & Vitells

    **Decide on an appropriate significance level before executing this method
    and set the `min_significance` argument accordingly!**

    Parameters
    ----------
    spec : :class:`~emgfit.spectrum.spectrum`
        Spectrum object to perform test on.
    alt_x_min : float
        Minimal x-position to use in the alternative-peak position scan.
    alt_x_max : float
        Maximal x-position to use in the alternative-peak position scan.
    alt_x_steps : int, optional, default: 100
        Number of steps to take in the alternative-peak position scan.
    min_significance : float [sigma], default: 3
        Minimal significance level (in sigma) required to reject the null model
        in favour of the alternative model.
    N_spectra : int, optional, default: 100
        Number of simulated spectra to fit at each x-position.
    c0 : float, optional, default: 0.5
        Threshold to use in determining the expected number of upcrossings.
    seed : int, optional
        Random seed to use for reproducible event sampling.
    show_upcrossings : bool, optional, default: True
        Whether to show plots of the range of upcrossings.
    show_fits : bool, optional, default: True
        Whether to show plots of the null- and alternative-model fits to the
        observed data.

    Returns
    -------
    dict
        Dictionary with results of the likelihood ratio test.

    See also
    --------
    :func:`_likelihood_ratio_test`

    Notes
    -----
    When the exact location of a hypothesized alternative peak is unknown, one
    may test for its presence by performing multiple hypothesis tests with
    different fixed alternative-peak positions. However, performing multiple
    tests on the same dataset artificially increases the rate of false discovery
    due to the increased chance for random background fluctuations to mimick a
    signal. In the high-energy particle physics literature, this complication is
    referred to as the look-elsewhere effect. To obtain a global p-value that
    correctly quantifies the likelihood to observe the alternative peak anywhere
    in the tested region, a procedure is needed that accounts for correlations
    between the local p-values obtained for the various tested peak positions.
    To this end, this function adapts the method outlined by Gross and Vitells
    in [#Gross]_. Namely, a conservative upper limit on the global p-value
    :math:`p` is deduced from the relation:

    .. math::

       p = P(LRT > c) \leq P(\chi^2_1 > c)/2 + \langle N(c_0)\rangle e^{-\left(c-c_0\right)/2}

    where :math:`P(LRT > c)` is the probability for the likelihood ratio test
    statistic (LRT) to exceed the maximum of the observed local LRT statistic
    :math:`c`, :math:`P(\chi^2_1 > c)` is the probability that the :math:`chi^2`
    statistic with one degree of freedom exceeds the level :math:`c` and
    :math:`\langle N(c_0)\rangle ` is the expected number of times the local LRT
    statistics surpass the threshold level :math:`c_0 \ll c` under the null
    hypothesis. This number is estimated by simulating :math:`N_{spectra}`
    spectra from the null model and taken as the mean number of times the local
    LRT statistics cross up through the specified threshold level :math:`c_0`.
    In principle, :math:`c_0` should be chosen as small as possible but care
    should be taken that the mean spacing between detected upcrossings does
    not fall below the typical width of the observed peaks.

    References
    ----------
    .. [#Gross] Gross, Eilam, and Ofer Vitells. "Trial factors for the look
       elsewhere effect in high energy physics." The European Physical Journal
       C 70 (2010): 525-530.

    """
    from scipy.stats import norm
    alpha = norm.sf(min_significance, loc=0, scale=1) # sf := 1 - cdf
    alt_x_pos = np.linspace(alt_x_min, alt_x_max, alt_x_steps+1)
    scan_res = np.mean(alt_x_pos[1:] - alt_x_pos[:-1])
    avg_bin_width = np.mean(spec.data.index.values[1:] - spec.data.index.values[:-1])
    # Check for approriately fine scan resolution:
    if scan_res > 2*avg_bin_width:
        warnings.warn("The resolution of the peak-position scan is coarser "
                      "than the average bin width of the spectrum. Consider "
                      "raising the `alt_x_steps` argument.")

    print("\n##### Performing GV likelihood ratio test (LRT) #####")
    print(f"N_spectra: {N_spectra}")
    print(f"Range of peak position scan: [{alt_x_min}, {alt_x_max}]")
    print(f"Resolution of peak position scan: {scan_res:.3e}")
    print(f"Test at {min_significance:.1f} sigma significance level")
    print(f"alpha = {alpha:.2e}\n")

    obs_LLRs = []
    null_results = []
    alt_results = []
    li_n_upcross = []
    # Determine LRT statistic for all alternative peak positions from obs. data
    for x_alt in np.atleast_1d(alt_x_pos):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            obs_LLR, null_res, alt_res = _likelihood_ratio_test(spec,
                                                                null_result_index,
                                                                x_alt,
                                                                verbose=False,
                                                                vary_alt_mu=False,
                                                                show_plots=False,
                                                                show_results=False)
        obs_LLRs.append(obs_LLR)
        null_results.append(null_res)
        alt_results.append(alt_res)

    max_LLR = max(obs_LLRs)
    idx_max_LLR = obs_LLRs.index(max_LLR)
    x_max_LLR = alt_x_pos[np.argmax(obs_LLRs)]
    # Plot null and alternative model fits yielding maximal LRT statistic
    if show_fits:
        print("### Fit results yielding the maximal LRT statistic with the observed data ###")
        print("# Null-model fit # ")
        spec.plot_fit(fit_result=null_results[idx_max_LLR])
        print("# Alternative-model fit # ")
        spec.plot_fit(fit_result=alt_results[idx_max_LLR])

    if show_upcrossings:
        plt.title("LRT statistics for observed data")
        ax = plt.gca()
        plt.plot(alt_x_pos, obs_LLRs, ".-")
        ax.axhline(c0, color="black")
        ax.axvline(x_max_LLR, color="black")
        #ax.ticklabel_format(useOffset=False, style='plain')
        plt.xlabel("Alternative-peak centroid")
        plt.ylabel("Local LRT statistic")
        plt.show()
    print(f"Max. LRT statistic determined from exp. data: {max_LLR:.2f} at {x_max_LLR:.6f} u \n")

    print("### Determine mean number of upcrossings from simulated spectra ###")
    all_sim_LLRs = []
    if seed is not None:
        np.random.seed(seed)
    for i in tqdm(range(N_spectra)):
        sim_LLRs = []
        # Simulate spectrum from best-fit null model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            sim_spec = simulate_spectrum(spec, copy_spec=True)

        for x_alt in np.atleast_1d(alt_x_pos):
            # Fit data with null and alternative model to determine local LRT
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                LLR_i, _, _ = _likelihood_ratio_test(sim_spec,
                                                     null_result_index,
                                                     x_alt,
                                                     verbose=False,
                                                     vary_alt_mu=False,
                                                     show_plots=False,
                                                     show_results=False)
            sim_LLRs.append(LLR_i)

        # Detect upcrossings
        i_upcross = [i for i in np.arange(1, alt_x_steps+1) if sim_LLRs[i-1] < c0 and sim_LLRs[i] > c0]
        n_upcross = len(i_upcross)
        li_n_upcross.append(n_upcross)
        all_sim_LLRs.append(sim_LLRs)

        # Plot local LRTs over alternative-peak position
        if show_upcrossings:
            plt.title(f"LRT statistics for simulated spectrum #{i:.0f}")
            ax = plt.gca()
            plt.plot(alt_x_pos, sim_LLRs, ".-")
            ax.axhline(c0, color="black")
            for i in i_upcross:
                ax.axvline(alt_x_pos[i], color="black")
            #ax.ticklabel_format(useOffset=False, style='plain')
            plt.xlabel("Alternative-peak centroid")
            plt.ylabel("Local LRT statistic")
            plt.show()

    # Calculate global p-value
    from scipy.stats import chi2
    mean_n_upcross = np.mean(li_n_upcross)
    std_n_upcross = np.std(li_n_upcross, ddof=1)
    err_mean_n_upcross = std_n_upcross/np.sqrt(len(li_n_upcross))
    print(f"Mean number of upcrossings: {mean_n_upcross:.2f} +- {err_mean_n_upcross:.2f}")
    dof = 1 # difference in number of free parameters in null & alt. model
    p_val = 0.5*chi2.sf(max_LLR, dof, loc=0, scale=1) + mean_n_upcross*np.exp(-0.5*(max_LLR-c0))
    p_val_err = err_mean_n_upcross*np.exp(-0.5*(max_LLR-c0))
    signif = norm.isf(p_val)
    print(f"Global p-value: p = {p_val:.2e} +- {p_val_err:.2e} ({signif:.2f} sigma significance) \n")

    # Compare to defined alpha
    if p_val < alpha:
        print(f"p < alpha = {alpha:.2e} => Reject null model in favor of "
               "alternative model with additional peak!")
        reject_null_model = True
        if p_val + min_significance*p_val_err > alpha:
            warnings.warn("The p-value exceeds alpha within its "
                          f"{min_significance}-sigma confidence interval. "
                          "Considering re-running with larger `N_spectra`.")
    else:
        print(f"p > alpha = {alpha:.2e} => Reject alternative model in favor "
               "of null model with fewer peaks!")
        reject_null_model = False

    if show_upcrossings:
        plt.hist(np.array(all_sim_LLRs).flatten(), density=False, bins=20)
        plt.gca().axvline(max_LLR, color="black")
        plt.xlabel("Local LRT statistic")
        plt.ylabel("Occurences")
        plt.show()

    LRT_results = {"LLR" : max_LLR,
                   "MC LLRs" : all_sim_LLRs,
                   "p-value" : p_val,
                   "p-value error" : p_val_err,
                   "reject_null_model" : reject_null_model}

    return LRT_results
