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
from IPython.display import display

################################################################################
def _likelihood_ratio_test(spec, ref_result, alt_x_pos, x_fit_cen=None,
                           x_fit_range=None, vary_alt_mu=True,
                           alt_mu_min=None, alt_mu_max=None,
                           vary_baseline=True, par_hint_args=None,
                           verbose=False, show_plots=False, show_results=False):
    """Perform a local likelihood ratio test on the specified spectrum

    Parameters
    ----------
    spec : :class:`emgfit.spectrum.spectrum`
        Spectrum object to perform likelihood ratio test on.
    ref_result : :class:`emgfit.model.EMGModelResult`
        Fit result storing the null model.
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
    alt_mu_min : float [u], optional
        Lower boundary to use when varying the alternative-peak centroid.
        Defaults to the range defined by the `MU_VAR_NSIGMA` constant in the
        :mod:`emgfit.fit_models` module.
    alt_mu_max : float [u], optional
        Upper boundary to use when varying the alternative-peak centroid.
        Defaults to the range defined by the `MU_VAR_NSIGMA` constant in the
        :mod:`emgfit.fit_models` module.
    vary_baseline : bool, optional
        If `True`, the constant background will be fitted with a varying
        uniform baseline parameter `bkg_c`. If `False`, the baseline parameter
        `bkg_c` will be fixed to 0.
    par_hint_args : dict of dicts, optional
        Arguments to pass to :meth:`lmfit.model.Model.set_param_hint` to
        modify or add model parameters. See docs of
        :meth:`~emgfit.spectrum.spectrum.peakfit` method for details.
    verbose : bool, optional
        Whether to print status updates and results.
    show_plots : bool, optional
        Whether to show plots of the fit results.
    show_results : bool, optional
        Whether to display reports with the fit results.

    Returns 
    -------
    tuple of format (float, :class:`~emgfit.model.EMGModelResult`, :class:`~emgfit.model.EMGModelResult`)
        Tuple storing the Log-likelihood ratio of null and alternative model, 
        the null fit result and the alternative model result.

    """
    fit_model = ref_result.fit_model
    if x_fit_cen is None:
        x_fit_cen = ref_result.x_fit_cen
    if x_fit_range is None:
        x_fit_range = ref_result.x_fit_range
    if par_hint_args is None:
        par_hint_args = ref_result.par_hint_args
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
                                   par_hint_args=par_hint_args,
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

    # Update initial parameters of alternative peak
    alt_peak = [p for p in alt_spec.peaks if p.x_pos==alt_x_pos][0]
    pref_alt_peak = "p{}_".format(alt_spec.peaks.index(alt_peak))
    alt_mu_hints = {}
    if vary_alt_mu:
        if alt_mu_min is not None:
            alt_mu_hints.update({"min" : alt_mu_min})
        if alt_mu_max is not None:
            alt_mu_hints.update({"max" : alt_mu_max})
    else:
        alt_mu_hints.update({"vary" : False})
    alt_par_hint_args = copy.deepcopy(par_hint_args)
    alt_par_hint_args.update({pref_alt_peak+"mu" : alt_mu_hints})

    try:
        alt_result = alt_spec.peakfit(x_fit_cen=null_result.x_fit_cen,
                                      x_fit_range=null_result.x_fit_range,
                                      fit_model=fit_model,
                                      cost_func='MLE',
                                      vary_baseline=vary_baseline,
                                      show_plots=show_plots,
                                      par_hint_args=alt_par_hint_args)
        if show_results:
            display(alt_result)
        alt_LLR = alt_result.chisqr
    except ValueError:
        warnings.warn("Fit with alternative model failed with ValueError.")
        alt_result = None
        alt_LLR = np.nan

    # Calculate likelihood ratio test statistic
    LLR = null_LLR - alt_LLR
    if verbose:
        print("Log-likelihood ratio test statistic:  LLR = {:.2f}".format(LLR))

    return LLR, null_result, alt_result


def run_MC_likelihood_ratio_test(spec, null_result_index, alt_x_pos,
                                 alt_mu_min=None, alt_mu_max=None,
                                 min_significance=3, N_spectra=10000,
                                 vary_ref_mus_and_amps=False,
                                 vary_ref_peak_shape=False,
                                 seed=None, n_cores=-1, show_plots=True, 
                                 show_results=True, show_LLR_hist=True):
    """Perform Monte Carlo likelihood ratio test by fitting simulated spectra

    The simulated spectra are sampled from the null model stored in the
    spectrum's fit results list and asociated with the peak index
    `null_result_index`.

    Parameters
    ----------
    spec : :class:`~emgfit.spectrum.spectrum`
        Spectrum object to perform likelihood ratio test on.
    null_result_index : int
        Index (of one) of the peak(s) present in the null-model fit.
    alt_x_pos : float, optional
        Initial position to use for alternative peak
    alt_mu_min : float [u], optional
        Lower boundary to use when varying the alternative-peak centroid.
        Defaults to the range defined by the `MU_VAR_NSIGMA` constant in the
        :mod:`emgfit.fit_models` module.
    alt_mu_max : float [u], optional
        Upper boundary to use when varying the alternative-peak centroid.
        Defaults to the range defined by the `MU_VAR_NSIGMA` constant in the
        :mod:`emgfit.fit_models` module.
    vary_ref_mus_and_amps : bool, optional
        Whether to randomly vary the peak positions and the peak and background
        amplitudes of the reference spectrum within their parameter 
        uncertainties.
    vary_ref_peak_shape : bool, optional
        Whether to vary the reference peak shape used for the event sampling in
        the creation of simulated spectra. If `True`, `N_spectra` parameter
        samples are drawn randomly with replacement from the 
        :attr:`~emgfit.spectrum.spectrum.MCMC_par_samples` obtained in the MCMC 
        shape parameter sampling.
    min_significance : float, optional, default: 3
        Critical significance level for rejecting the null hypothesis (measured
        in sigma).
    N_spectra : int, optional, default: 10000
        Number of simulated spectra to sample from the null model.
    seed : int, optional
        Random seed to use for reproducible sampling.
    n_cores : int, optional, default: -1
        Number of CPU cores to use for parallelized sampling and fitting of
        simulated spectra. If ``-1``, all available cores are used.
    show_plots : bool, optional
        Whether to show plots of the fit results.
    show_results : bool, optional
        Whether to display reports with the fit results.
    show_LLR_hist : bool, optional 
        Whether to display histogram of log-likelihood ratio values collected
        for p-value determination.

    Returns
    -------
    dict
        Dictionary with results of the likelihood ratio test.

    See also
    --------
    :func:`run_GV_likelihood_ratio_test`
    :func:`_likelihood_ratio_test`

    Notes 
    -----
    Simulated spectra are created by randomly sampling events from the null 
    model best fitting the observed data. These simulated spectra are then 
    fitted with both the null and the alternative model and the respective 
    values for the likelihood ratio test statistic :math:`\Lambda` are 
    calculated using the relation 

    .. math::

      \\Lambda = \\log\\left(\\frac{\\mathcal{L}(H_1)}{\\mathcal{L}(H_0)}\\right) = L(H_1) - L(H_0),

    where :math:`L(H_0)` and :math:`L(H_1)` denote the MLE cost function values 
    (i.e. the negative doubled log-likelihood values) obtained from the 
    null-model and alternative-model fits, respectively, and 
    :math:`\mathcal{L}(H_0)` and :math:`\mathcal{L}(H_1)` mark the 
    corresponding likelihood functions. Finally, the p-value is calculated as 

    .. math:: 

      p = \\frac{N_>}{N_< + N_>}, 

    where :math:`N_<` and :math:`N_>` denote the number of likelihood ratio 
    values :math:`\Lambda` that fall below and above the observed value for the 
    likelihood ratio test statistic :math:`\Lambda_\mathrm{obs}`, respectively. 

    """
    from scipy.stats import norm
    alpha = norm.sf(min_significance, loc=0, scale=1) # sf := 1 - cdf
    ref_null_result = spec.fit_results[null_result_index]
    if vary_ref_peak_shape:
        MC_shape_par_samples = spec.MCMC_par_samples.sample(n=N_spectra,
                                                            replace=True)
    else:
        MC_shape_par_samples = None

    print("\n##### Performing Monte Carlo likelihood ratio test #####")
    print("N_spectra:",N_spectra)
    print(f"Test at {min_significance:.1f} sigma significance level, i.e. alpha = {alpha:.2e}")

    # Perform LRT on the observed data
    print("\n### Determine test statistic for observed data ###")
    LLR, _, alt_res = _likelihood_ratio_test(spec, ref_null_result,
                                             alt_x_pos, verbose=True,
                                             show_results=show_results,
                                             show_plots=show_plots,
                                             alt_mu_min=alt_mu_min,
                                             alt_mu_max=alt_mu_max,
                                             vary_alt_mu=True)

    # Run LRTs on spectra sampled from best null-model fit of the observed data
    print("\n### Determine test statistic for simulated Monte Carlo spectra ###")
    if seed is None:
        seed = np.random.randint(2**31)
    from emgfit.sample import fit_simulated_spectra
    null_results = fit_simulated_spectra(spec, ref_null_result,
                                        N_spectra=N_spectra,
                                        randomize_ref_mus_and_amps=vary_ref_mus_and_amps,
                                        MC_shape_par_samples=MC_shape_par_samples,
                                        seed=seed, n_cores=n_cores)
    alt_results = fit_simulated_spectra(spec, ref_null_result,
                                        alt_result=alt_res, N_spectra=N_spectra,
                                        randomize_ref_mus_and_amps=vary_ref_mus_and_amps,
                                        MC_shape_par_samples=MC_shape_par_samples,
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

    if show_LLR_hist:
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
    print(f"Monte Carlo p-value: p = {p_val:.2e} +- {p_val_err:.2e}")
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


def run_GV_likelihood_ratio_test(spec, null_result_index, alt_x_min, alt_x_max,
                                 alt_x_steps=100, min_significance=3,
                                 N_spectra=100, c0=0.5, seed=None,
                                 show_upcrossings=True, show_fits=True):
    """Perform a likelihood ratio test following the method of Gross & Vitells

    **Decide on an appropriate significance level before executing this method
    and set the `min_significance` argument accordingly!**

    Parameters
    ----------
    spec : :class:`~emgfit.spectrum.spectrum`
        Spectrum object to perform test on.
    null_result_index : int
        Index (of one) of the peak(s) present in the null-model fit.
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
    :func:`run_MC_likelihood_ratio_test`
    :func:`_likelihood_ratio_test`

    Notes
    -----
    When the exact location of a hypothesized alternative peak is unknown, one
    may test for its presence by performing multiple hypothesis tests with
    different fixed alternative-peak positions. However, performing multiple
    tests on the same dataset artificially increases the rate of false discovery
    due to the increased chance for random background fluctuations to mimick a
    signal. In the high-energy particle physics literature, this complication 
    is referred to as the look-elsewhere effect. To obtain a global p-value 
    that correctly quantifies the likelihood to observe the alternative peak 
    anywhere in the tested region, a procedure is needed that accounts for 
    correlations between the local p-values obtained for the various tested 
    peak positions. To this end, this function adapts the method outlined by 
    Gross and Vitells in [#Gross]_. Namely, an upper limit on the global 
    p-value :math:`p` is deduced from the relation:

    .. math::

       p = P(LLR > c) \\leq P(\\chi^2_1 > c)/2 + \\langle N(c_0)\\rangle e^{-\\left(c-c_0\\right)/2},

    where :math:`P(LLR > c)` is the probability for the log-likelihood ratio 
    statistic (LLR) to exceed the maximum of the observed local LLR statistic
    :math:`c`, :math:`P(\chi^2_1 > c)` is the probability that the :math:`\chi^2`
    statistic with one degree of freedom exceeds the level :math:`c` and
    :math:`\\langle N(c_0)\\rangle` is the expected number of times the local 
    LLR test statistics surpass the threshold level :math:`c_0 \ll c` under the 
    null hypothesis. This number is estimated by simulating :math:`N_{spectra}`
    spectra from the null model and taken as the mean number of times the local
    LRT statistics cross up through the specified threshold level :math:`c_0`.
    In principle, :math:`c_0` should be chosen as small as possible but care
    should be taken that the mean spacing between detected upcrossings does
    not fall below the typical width of the observed peaks.

    References
    ----------
    .. [#Gross] Gross, Eilam, and Vitells, Ofer. "Trial factors for the look
       elsewhere effect in high energy physics." The European Physical Journal
       C 70 (2010): 525-530.

    """
    from scipy.stats import norm
    alpha = norm.sf(min_significance, loc=0, scale=1) # sf := 1 - cdf
    alt_x_pos = np.linspace(alt_x_min, alt_x_max, alt_x_steps+1)
    scan_res = np.mean(alt_x_pos[1:] - alt_x_pos[:-1])
    avg_bin_width = np.mean(spec.data.index.values[1:] - spec.data.index.values[:-1])
    ref_null_result = spec.fit_results[null_result_index]
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
                                                                ref_null_result,
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
    x_max_LLR = alt_x_pos[idx_max_LLR]
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
        plt.xlabel("Alternative-peak centroid [u]")
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
                                                     ref_null_result,
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
            plt.xlabel("Alternative-peak centroid [u]")
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
                   "Mean number of upcrossings" : mean_n_upcross,
                   "Error mean number of upcrossings" : std_n_upcross,
                   "p-value" : p_val,
                   "p-value error" : p_val_err,
                   "reject_null_model" : reject_null_model}

    return LRT_results
