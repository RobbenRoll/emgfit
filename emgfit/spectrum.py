################################################################################
##### Python module for peak fitting in TOF mass spectra
##### Author: Stefan Paul

##### Import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sig
import scipy.special as spl
import time
import copy
from IPython.display import display
from .config import *
from .ame_funcs import get_AME_values
import emgfit.emg_funcs as emg_funcs
import emgfit.fit_models as fit_models
import lmfit as fit
import os
import dill
# Remove dill types from pickle registry to avoid pickle errors in parallelized
# fits:
dill.extend(False)


################################################################################
##### Define peak class
class peak:
    """Object storing all relevant information about a mass peak.

    Most peak attributes are intialized as ``None`` and are later automatically
    updated by methods of the spectrum class, e.g.
    :meth:`spectrum.determine_peak_shape` or :meth:`spectrum.fit_peaks`.

    Attributes
    ----------
    x_pos : float [u]
        Coarse position of peak centroid. In fits the Hyper-EMG parameter for
        the (Gaussian) peak centroid `mu` will be initialized at this value.
        Peak markers in plots are located at `x_pos`.
    species : str
        String with chemical formula of ion species asociated with peak.
        Species strings follow the :ref:`:-notation`.
        Examples: ``'1K39:-1e'``, ``'K39:-e'``, ``'3H1:1O16:-1e'``.
        **Do not forget to substract the electron**, otherwise the atomic not
        the ionic mass would be used as literature value!
        Alternatively, tentative assigments can be made by adding a ``'?'`` at
        the end of the species string (e.g.: ``'Sn100:-1e?'``, ``'?'``, ...).
    comment : str
        User comment for peak.
    m_AME : float [u], optional
        Ionic literature mass value, from AME2016 or user-defined.
    m_AME_error : float [u], optional
        Literature mass uncertainty, from AME2016 or user-defined.
    extrapolated : bool
        Boolean flag for extrapolated AME mass values (equivalent to '#' in AME).
    fit_model : str
        Name of model used to fit peak.
    cost_func : str
        Type of cost function used to fit peak (``'chi-square'`` or ``'MLE'``).
    red_chi : float
        Reduced chi-squared of peak fit. If the peak was fitted using ``'MLE'``,
        :attr:`red_chi` should be taken with caution.
    area, area_error : float [counts]
        Number of total counts in peak and corresponding uncertainty
        (calculated from amplitude parameter `amp` of peak fit).
    m_ion : float [u]
        Ionic mass value obtained in peak fit (after mass recalibration).
    rel_stat_error : float
        Relative statistical uncertainty of :attr:`m_ion`.
    rel_recal_error : float
        Relative uncertainty of :attr:`m_ion` due to mass recalibration.
    rel_peakshape_error : float
        Relative peak-shape uncertainty of :attr:`m_ion`.
    rel_mass_error : float
        Total relative mass uncertainty of :attr:`m_ion` (excluding systematics!).
        Includes statistical, peak-shape and recalibration uncertainty.
    A : int
        Atomic mass number of peak species.
    atomic_ME_keV : float [keV]
        (Atomic) mass excess corresponding to :attr:`m_ion`.
    mass_error_keV : float [keV]
        Total mass uncertainty of :attr:`m_ion` (excluding systematics!).
    m_dev_keV : float [keV]
        Deviation from literature value (:attr:`m_ion` - :attr:`m_AME`).

    """
    def __init__(self, x_pos, species, m_AME=None, m_AME_error=None, Ex=0.0,
                 Ex_error=0.0):
        """Instantiate a new :class:`peak` object.

        If a valid ion species is assigned with the `species` parameter the
        corresponding literature values will automatically be fetched from the
        AME2016 database. The syntax for species assignment follows that of MAc
        (for more details see documentation for `species` parameter below).

        If different literature values are to be used, the literature mass or
        mass uncertainty can be user-defined with :attr:`m_AME` and
        :attr:`m_AME_error`. This is useful for isomers and in cases where more
        recent measurements haven't been included in the AME yet.

        Parameters
        ----------

        x_pos : float [u]
            Coarse position of peak centroid. In fits the Hyper-EMG parameter
            for the (Gaussian) peak centroid `mu` will be initialized at this
            value. Peak markers in plots are located at `x_pos`.
        species : str
            String with chemical formula of ion species asociated with peak.
            Species strings follow the :ref:`:-notation`.
            Examples: ``'1K39:-1e'``, ``'K39:-e'``, ``'3H1:1O16:-1e'``.
            **Do not forget to substract the electron from singly-charged
            species**, otherwise the atomic not the ionic mass will be used as
            literature value! Alternatively, tentative assigments can be made by
            adding a ``'?'`` at the end of the species string
            (e.g.: ``'Sn100:-1e?'``, ``'?'``, ...).
        m_AME : float [u], optional
            User-defined literature mass value. Overwrites value fetched from
            AME2016. Useful for isomers or to use more up-to-date values.
        m_AME_error : float [u], optional
            User-defined literature mass uncertainty. Overwrites value fetched
            from AME2016.
        Ex : float [keV], optional, default : 0.0
            Isomer excitation energy (in keV) to add to ground-state literature
            mass. Irrelevant if the `m_AME` argument is used or if the peak is
            not labelled as isomer.
        Ex_error : float [keV], optional, default : 0.0
            Uncertainty of isomer excitation energy (in keV) to add in
            quadrature to ground-state literature mass uncertainty. Irrelevant
            if the `m_AME_error` argument is used or if the peak is not labelled
            as isomer.

        """
        self.x_pos = x_pos
        self.species = species # e.g. '1Cs133:-1e or 'Cs133:-e' or '4H1:1C12:-1e'
        self.comment = '-'
        self.m_AME = m_AME #
        self.m_AME_error = m_AME_error
        m, m_error, extrapol, A_tot = get_AME_values(species, Ex=Ex,
                                                     Ex_error=Ex_error)
        # If `m_AME` has not been user-defined, set it to AME value
        if self.m_AME is None:
             self.m_AME = m
         # If `m_AME_error` has not been user-defined, set it to AME value
        if self.m_AME_error is None:
            self.m_AME_error = m_error
        self.extrapolated = extrapol
        self.fit_model = None
        self.cost_func = None
        self.red_chi = None
        self.area = None
        self.area_error = None
        self.m_ion = None # ionic mass value from fit [u]
        self.rel_stat_error = None #
        self.rel_recal_error = None
        self.rel_peakshape_error = None
        self.rel_mass_error = None
        self.A = A_tot
        self.atomic_ME_keV = None # Mass excess = atomic mass[u] - A [keV]
        self.mass_error_keV = None
        self.m_dev_keV = None # TITAN - AME [keV]


    def update_lit_values(self, Ex=0.0, Ex_error=0.0):
        """Updates :attr:`m_AME`, :attr:`m_AME_error` and :attr:`extrapolated`
        peak attributes with AME2016 values for specified species.

        Parameters
        ----------
        Ex : float [keV], optional, default : 0.0
            Isomer excitation energy (in keV) to add to ground-state literature
            mass.
        Ex_error : float [keV], optional, default : 0.0
            Uncertainty of isomer excitation energy (in keV) to add in
            quadrature to ground-state literature mass uncertainty.

        """
        m, m_error, extrapol, A_tot = get_AME_values(self.species, Ex=Ex,
                                                     Ex_error=Ex_error)
        self.m_AME = m
        self.m_AME_error = m_error
        self.extrapolated = extrapol
        self.A = A_tot


    def print_properties(self):
        """Print the most relevant peak properties."""

        print("x_pos:",self.x_pos,"u")
        print("Species:",self.species)
        print("AME mass:",self.m_AME,"u     (",np.round(self.m_AME*u_to_keV,3),"keV )")
        print("AME mass uncertainty:",self.m_AME_error,"u         (",np.round(self.m_AME_error*u_to_keV,3),"keV )")
        print("AME mass extrapolated?",self.extrapolated)
        if self.fit_model is not None:
            print("Peak area: "+str(self.area)+" +- "+str(self.peak_area_error)+" counts")
            print("(Ionic) mass:",self.m_ion,"u     (",np.round(self.m_ion*u_to_keV,3),"keV )")
            print("Stat. mass uncertainty:",self.rel_stat_error*self.m_ion,"u     (",np.round(self.rel_stat_error*self.m_ion*u_to_keV,3),"keV )")
            print("Peak-shape mass uncertainty:",self.rel_peakshape_error*self.m_ion,"u     (",np.round(self.rel_peakshape_error*self.m_ion*u_to_keV,3),"keV )")
            print("Re-calibration mass uncertainty:",self.rel_recal_error*self.m_ion,"u     (",np.round(self.rel_recal_error*self.m_ion*u_to_keV,3),"keV )")
            print("Total mass uncertainty:",self.rel_mass_error*self.m_ion,"u     (",np.round(self.mass_error_keV,3),"keV )")
            print("Atomic mass excess:",np.round(self.atomic_ME_keV,3),"keV")
            print("m_ion - m_AME:",np.round(self.m_dev_keV,3),"keV")
            print("Reduced chi square:",np.round(self.red_chi))


################################################################################
###### Define spectrum class
class spectrum:
    r"""Object storing spectrum data, associated peaks and all relevant
    fit results - the workhorse of emgfit.

    Attributes
    ----------
    input_filename : str
        Name of input file containing the spectrum data.
    spectrum_comment : str, default: '-'
        User comment for entire spectrum (helpful for further processing after).
    fit_model : str
        Name of model used for peak-shape determination and further fitting.
    index_shape_calib : int
        Index of peak used for peak-shape calibration.
    red_chi_shape_cal : float
        Reduced chi-squared of peak-shape determination fit.
    fit_range_shape_cal : float [u]
        Fit range used for peak-shape calibration.
    shape_cal_result : :class:`lmfit.model.ModelResult`
        Fit result obtained in peak-shape calibration.
    shape_cal_pars : dict
        Model parameter values obtained in peak-shape calibration.
    shape_cal_errors : dict
        Model parameter uncertainties obtained in peak-shape calibration.
    index_mass_calib : int
        Peak index of mass calibrant peak.
    determined_A_stat_emg : bool
        Boolean flag for whether :attr:`A_stat_emg` was determined for this
        spectrum specifically using the :meth:`determine_A_stat_emg` method.
        If `True`, :attr:`A_stat_emg` was set using
        :meth:`determine_A_stat_emg`, otherwise the default value
        `emgfit.config.A_stat_emg_default` from the :mod:`~emgfit.config` module
        was used. For more details see docs of :meth:`determine_A_stat_emg`
        method.
    A_stat_emg : float
        Constant of proportionality for calculation of the statistical mass
        uncertainties. Defaults to `emgfit.config.A_stat_emg_default`
        as defined in the :mod:`~emgfit.config` module, unless the
        :meth:`determine_A_stat_emg` method is run.
    A_stat_emg_error : float
        Uncertainty of :attr:`A_stat_emg`.
    recal_fac : float, default: 1.0
        Scaling factor applied to :attr:`m_ion` in mass recalibration.
    rel_recal_error : float
        Relative uncertainty of recalibration factor :attr:`recal_fac`.
    recal_facs_pm : dict
        Modified recalibration factors obtained in peak-shape uncertainty
        evaluation by varying each shape parameter by plus and minus 1 standard
        deviation, respectively.
    eff_mass_shifts : :class:`numpy.ndarray` of dict
        Maximal effective mass shifts for each peak obtained in peak-shape
        uncertainty evaluation by varying each shape parameter by plus and minus
        1 standard deviation and only keeping the shift with the larger absolute
        magnitude. The `eff_mass_shifts` array contains a dictionary for each
        peak; the dictionaries have the following structure:
        {'<shape param. name> eff. mass shift' : [<maximal eff. mass shift>],...}
        For more details see docs of :meth:`_eval_peakshape_errors`.
    area_shifts : :class:`numpy.ndarray` of dict
        Maximal area change for each peak obtained in peak-shape uncertainty
        evaluation by varying each shape parameter by plus and minus 1 standard
        deviation and only keeping the shift with the larger absolute magnitude.
        The `eff_mass_shifts` array contains a dictionary for each peak; the
        dictionaries have the following structure:
        {'<shape param. name> eff. mass shift' : [<maximal eff. mass shift>],...}
        For the mass calibrant the dictionary holds the absolute shifts of the
        calibrant peak centroid (`calibrant centroid shift`). For more
        details see docs of :meth:`_eval_peakshape_errors`.
    peaks_with_errors_from_resampling : list of int
        List with indeces of peaks whose statistical mass and area uncertainties
        have been determined by fitting synthetic spectra resampled from the
        best-fit model (see :meth:`get_errors_from_resampling`).
    MCMC_par_samples : list of dict
        Shape parameter samples obtained via Markov chain Monte Carlo (MCMC)
        sampling with :meth:`_get_MCMC_par_samples`.
    MC_recal_facs : list of float
        Recalibration factors obtained in fits with Markov Chain Monte Carlo
        (MCMC) shape parameter samples in :meth:`get_MC_peakshape_errors`.
    peaks_with_MC_PS_errors : list of int
        List with indeces of peaks for which peak-shape errors have been
        determined by re-fitting with shape parameter sets from Markov Chain
        Monte Carlo sampling (see :meth:`get_MC_peakshape_errors`).
    peaks : list of :class:`peak`
        List containing all peaks associated with the spectrum sorted by
        ascending mass. The index of a peak within the `peaks` list is referred
        to as the ``peak_index``.
    fit_results : list of :class:`lmfit.model.ModelResult`
        List containing fit results (:class:`lmfit.model.ModelResult` objects)
        for peaks associated with spectrum.
    blinded_peaks : list of int
        List with indeces of peaks whose mass values and peak positions are to
        be hidden to enable blind analysis. The mass values will be unblinded
        upon export of the analysis results.
    data : :class:`pandas.DataFrame`
        Histogrammed spectrum data.
    mass_number : int
        Atomic mass number associated with central bin of spectrum.
    default_fit_range : float [u]
        Default mass range for fits, scaled to :attr:`mass_number` of spectrum.

    Notes
    -----
    The :attr:`fit_model` spectrum attribute seems somewhat redundant with
    the :attr:`peak.fit_model` peak attribute but ensures that no relevant
    information is lost.

    The :attr:`mass_number` is used for re-scaling of the default model
    parameters to the mass of interest. It is calculated upon data import by
    taking the median of all mass bin centers (after initial cutting of the
    spectrum) and rounding to the closest integer. This accounts for spectra
    potentially containing several mass units.

    The :attr:`default_fit_range` is scaled to the spectrum's :attr:`mass_number`
    using the relation:
    :math:`\text{default_fit_range} = 0.01\,u \cdot (\text{mass_number}/100)`

    """
    def __init__(self,filename=None,m_start=None,m_stop=None,skiprows = 18,
                 show_plot=True,df=None):
        """Create a :class:`spectrum` object by importing histogrammed mass data
        from .txt or .csv file.

        Input file format: two-column .csv- or .txt-file with tab-separated
                           values (column 1: mass bin, column 2: counts in bin).
                           This is the default format of MAc export files when
                           histogram mode is used.

        Optionally the spectrum can be cut to a specified fit range using the
        `m_start` and `m_stop` parameters. Mass data outside this range will be
        discarded and excluded from further analysis.

        If `show_plot` is True, a plot of the spectrum is shown including
        vertical markers for the `m_start` and `m_stop` mass cut-offs
        (if applicable).

        Parameters
        ----------
        filename : str, optional
            Filename of mass spectrum to analyze (as exported with MAc's
            histogram mode). If the input file is not located in the working
            directory the directory path has to be included in `filename`, too.
            If no `filename` is given, data must be provided via `df` argument.
	    m_start : float [u], optional
            Start of fit range, data at lower masses will be discarded.
	    m_stop : float [u], optional
            Stop of fit range, data at higher masses will be discarded.
        show_plot : bool, optional, default: True
            If `True`, shows a plot of full spectrum with vertical markers for
            `m_start` and `m_stop` cut-offs.
        df : :class:`pandas.DataFrame`, optional
            DataFrame with spectrum data to use, this enables the creation of a
            spectrum object from a DataFrame instead of from an external file.
            **Primarily intended for internal use.**

        Notes
        -----
        The option to import data via the `df` argument was added to enable the
        processing of bootstrapped spectra as regular :class:`spectrum` objects
        in the :meth:`determine_A_stat_emg` method. This feature is primarily
        intended for internal use. The parsed DataFrame must have an index
        column named 'Mass [u]' and a value column named 'Counts'.

	    """
        if filename is not None:
            data_uncut = pd.read_csv(filename, header=None,
                                     names=['Mass [u]', 'Counts'],
                                     skiprows=skiprows, delim_whitespace=True,
                                     index_col=False, dtype=float)
            data_uncut.set_index('Mass [u]',inplace =True)
            self.input_filename = filename
        elif df is not None:
            data_uncut = df
        else:
            raise Exception("Import failed, since input data was neither "
                            "specified with `filename` nor `df`.")
        self.spectrum_comment = '-'
        self.fit_model = None
        self.index_shape_calib = None
        self.red_chi_shape_cal = None
        self.fit_range_shape_cal = None
        self.shape_cal_result = None
        self.shape_cal_pars = None
        self.shape_cal_errors = None
        self.index_mass_calib = None
        self.determined_A_stat_emg = False
        self.A_stat_emg = A_stat_emg_default # initialize at default
        self.A_stat_emg_error = None
        self.recal_fac = 1.0
        self.rel_recal_error = None
        self.recal_facs_pm = None
        self.eff_mass_shifts = None
        self.eff_area_shifts = None
        self.peaks_with_errors_from_resampling = []
        self.MCMC_par_samples = None
        self.MC_recal_facs = None
        self.peaks_with_MC_PS_errors = []
        self.peaks = [] # list containing peaks associated with spectrum
        self.fit_results = [] # list containing fit results of all peaks
        self.blinded_peaks = []
        if m_start or m_stop: # cut input data to specified mass range
            self.data = data_uncut.loc[m_start:m_stop]
            plot_title = 'Spectrum with cut-off markers'
        else:
            self.data = data_uncut # dataframe containing mass spectrum data
            plot_title = 'Spectrum (fit full range)'
        # Set `mass_number` using median of mass bins after cutting spectrum and
        # round to closest integer:
        self.mass_number = int(np.round(self.data.index.values[
                                                        int(len(self.data)/2)]))
        self.default_fit_range = 0.01*(self.mass_number/100)
        if show_plot:
            fig  = plt.figure(figsize=(figwidth,figwidth*4.5/18),dpi=dpi)
            plt.title(plot_title)
            data_uncut.plot(ax=fig.gca(), legend=False)
            if m_start is not None:
                plt.vlines(m_start, 0, 1.2*np.max(self.data['Counts']),
                           color='black')
            if m_stop is not None:
                plt.vlines(m_stop, 0, 1.2*np.max(self.data['Counts']),
                           color='black')
            plt.yscale('log')
            plt.xlabel('m/z [u]')
            plt.ylabel('Counts per bin')
            plt.show()


    def add_spectrum_comment(self,comment,overwrite=False):
        """Add a general comment to the spectrum.

        By default the `comment` argument will be appended to the end of the
        current :attr:`spectrum_comment` attribute. If `overwrite` is set to
        `True` the current :attr:`spectrum_comment` is overwritten with
        `comment`.

        Parameters
        ----------
        comment : str
            Comment to add to spectrum.
        overwrite : bool
            If `True`, the current :attr:`spectrum_comment` attribute will be
            overwritten with `comment`, else `comment` is appended to the end of
            :attr:`spectrum_comment`.

        See also
        --------
        :meth:`add_peak_comment`

        Notes
        -----
        The :attr:`spectrum_comment` will be included in the output file storing
        all fit results and can hence be useful to pass on information for
        post-processing.

        If :attr:`spectrum_comment` is '-' (default value) it is always
        overwritten with `comment`.

        """
        try:
            if self.spectrum_comment in ('-', None) or overwrite:
                self.spectrum_comment = comment
            else:
                self.spectrum_comment = self.spectrum_comment+comment
            print("Spectrum comment was changed to: ",self.spectrum_comment)
        except TypeError:
            raise("Could not add comment 'comment' argument must of type "
                  "string.")


    @staticmethod
    def _smooth(x,window_len=11,window='hanning'):
        """Smooth the data for the peak detection.

        ** Intended for internal use only.**

    	This method is based on the convolution of a normalized window with the
        signal. The signal is prepared by introducing reflected copies of the
        signal (with the window size) in both ends so that transient parts are
        minimized in the begining and end part of the output signal.

    	Parameters
        ----------
        x : numpy.array
            The input data
        window_len : odd int, optional
            Length of the smoothing window; **must be an odd integer**!
    	window : str, optional
            Type of window from 'flat', 'hanning', 'hamming', 'bartlett',
            'blackman', flat window will produce a moving average smoothing.

    	Returns
        -------
        numpy.array
    	    The smoothed spectrum data.

    	See also
        --------
    	    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
            numpy.convolve, scipy.signal.lfilter

    	Notes
        -----
        length(output) != length(input), to correct this:
        return y[(window_len/2-1):-(window_len/2)] instead of just y.

        Method adapted from:
        https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

        Example
        -------
        >>> t=linspace(-2,2,0.1)
    	>>> x=sin(t)+randn(len(t))*0.1
    	>>> y=smooth(x)

    	"""
        if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")

        if window_len<3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window must be one of 'flat', 'hanning', "
                             "'hamming', 'bartlett', 'blackman'.")

        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='valid')
        return y[int(window_len/2+1):-int(window_len/2-1)]


    def plot(self, peaks=None, title="", fig=None, yscale='log', vmarkers=None,
             thres=None, ymin=None, xmin=None, xmax=None):
        """Plot mass spectrum (without fit curve).

        Vertical markers are added for all peaks specified with `peaks`.

        Parameters
        ----------
        peaks : list of :class:`peaks`, optional
            List of :class:`peaks` to show peak markers for. Defaults to
            :attr:`peaks`.
        title : str, optional
            Optional plot title.
        fig : :class:`matplotlib.pyplot.figure`, optional
            Figure object to plot onto.
        yscale : str, optional
            Scale of y-axis (`'linear'` or `'log'`), defaults to `'log'`.
        vmarkers : list of float [u], optional
            List with mass positions [u] to add vertical markers at.
        thres : float, optional
            y-level to add horizontal marker at (e.g. for indicating set
            threshold in peak detection).
        ymin : float, optional
            Lower bound of y-range to plot.
        xmin, xmax : float [u], optional
            Lower/upper bound of mass range to plot.

        See also
        --------
        :meth:`plot_fit`
        :meth:`plot_fit_zoom`

        """
        if peaks is None:
            peaks = self.peaks
        data = self.data # get spectrum data stored in dataframe 'self.data'
        ymax = data.max()[0]
        if fig is None:
            fig = plt.figure(figsize=(figwidth,figwidth*4.5/18),dpi=dpi)
        ax = fig.gca()
        data.plot(ax=ax, legend=False)
        plt.yscale(yscale)
        plt.xlabel('m/z [u]')
        plt.ylabel('Counts per bin')
        plt.title(title)
        try:
            plt.vlines(x=vmarkers, ymin=0, ymax=data.max(), color='black')
        except TypeError:
            pass
        if yscale == 'log':
            for p in peaks:
                plt.vlines(x=p.x_pos, ymin=0, ymax=1.05*ymax,
                           linestyles='dashed', color='black')
                plt.text(p.x_pos, 1.21*ymax, peaks.index(p),
                         horizontalalignment='center', fontsize=labelsize)
            if ymin:
                plt.ylim(ymin,2.45*ymax)
            else:
                plt.ylim(0.1,2.45*ymax)
        else:
            for p in peaks:
                plt.vlines(x=p.x_pos, ymin=0, ymax=1.03*ymax,
                           linestyles='dashed', color='black')
                plt.text(p.x_pos, 1.05*ymax, peaks.index(p),
                         horizontalalignment='center', fontsize=labelsize)
            if ymin:
                plt.ylim(ymin,1.12*ymax)
            else:
                plt.ylim(0,1.12*ymax)

        if thres:
            ax.axhline(y=thres, color='black')
        plt.xlim(xmin,xmax)
        ax.get_xaxis().get_major_formatter().set_useOffset(False) # no offset
        plt.show()


    @staticmethod
    def _plot_df(df, title='', fig=None, yscale='log', peaks=None,
                 vmarkers=None, thres=None, ymin=None, ylabel='Counts per bin',
                 xmin=None, xmax=None):
        """Plots spectrum data stored in :class:`pandas.DataFrame` `df`.

        **Intended for internal use.**

        Optionally with peak markers if:
        1. single or multiple x_pos are passed to `vmarkers`, OR
        2. list of peak objects is passed to `peaks`.

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            Spectrum data to plot.
        fig : :class:`matplotlib.pyplot.figure`, optional
            Figure object to plot onto.
        yscale : str, optional
            Scale of y-axis (`'linear'` or `'log'`), defaults to `'log'`.
        peaks : list of :class:`peaks`, optional
            List of :class:`peaks` to show peak markers for.
        vmarkers : list of float [u], optional
            List with mass positions [u] to add vertical markers at.
        thres : float, optional
            y-level to add horizontal marker at (e.g. for indicating set
            threshold in peak detection).
        ymin : float, optional
            Lower bound of y-range to plot.
        ylabel : str, optional
            Custom label for y-axis.
        xmin, xmax : float [u], optional
            Lower/upper bound of mass range to plot.

        See also
        --------
        :meth:`plot`
        :meth:`plot_fit`
        :meth:`plot_fit_zoom`

        """
        if fig is None:
            fig = plt.figure(figsize=(figwidth,figwidth*4.5/18), dpi=dpi)
        ax = fig.gca()
        df.plot(ax=ax, legend=False)
        plt.yscale(yscale)
        plt.xlabel('m/z [u]')
        plt.ylabel(ylabel)
        plt.title(title)
        try:
            plt.vlines(x=vmarkers, ymin=0, ymax=1.05*df.max(), color='black')
        except TypeError:
            pass
        try:
            li_x_pos = [p.x_pos for p in peaks]
            plt.vlines(x=li_x_pos, ymin=0, ymax=1.05*df.max(), color='black')
        except TypeError:
            pass
        if thres:
            ax.axhline(y=thres, color='black')
        if ymin:
            plt.ylim(ymin,)
        plt.xlim(xmin,xmax)
        ax.get_xaxis().get_major_formatter().set_useOffset(False) # no offset
        plt.show()


    def detect_peaks(self,thres=0.003,window_len=23,window='blackman',
                     width=2e-05, plot_smoothed_spec=True,
                     plot_2nd_deriv=True, plot_detection_result=True):
        """Perform automatic peak detection.

        The peak detection routine uses a scaled second derivative of the
        spectrum :attr:`data` after first applying some smoothing. This enables
        very sensitive yet robust peak detection. The parameters `thres`,
        `window_len` & `width` can be used to tune the smoothing and peak
        detection for maximal sensitivity.

        Parameters
        ----------
        thres : float, optional
            Threshold for peak detection in the inverted and scaled second
            derivative of the smoothed spectrum.
        window_len : odd int, optional
            Length of window used for smoothing the spectrum (in no. of bins).
            **Must be an ODD integer.**
        window : str, optional
            The window function used for smooting the spectrum. Defaults to
            ``'blackman'``. Other options: ``'flat'``, ``'hanning'``,
            ``'hamming'``, ``'bartlett'``. See also `NumPy window functions
            <https://docs.scipy.org/doc/numpy/reference/routines.window.html>`_.
        width : float [u], optional
            Minimal FWHM of peaks to be detected. Caution: To achieve maximal
            sensitivity for overlapping peaks this number might have to be set
            to less than the peak's FWHM! In challenging cases use the plot of
            the scaled inverted second derivative (by setting `plot_2nd_deriv`
            to `True`) to ensure that the detection threshold is set properly.
        plot_smoothed_spec : bool, optional
            If `True` a plot with the original and the smoothed spectrum is
            shown.
        plot_2nd_deriv : bool, optional
            If `True` a plot with the scaled, inverted second derivative of
            the smoothed spectrum is shown.
        plot_detection_result : bool, optional
            If `True` a plot of the spectrum with markers for the detected
            peaks is shown.

        See also
        --------
        :meth:`add_peak`
        :meth:`remove_peak`

        Note
        ----
        Running this method removes any pre-existing peaks.

        Notes
        -----
        For details on the smoothing, see docs of :meth:`_smooth` by calling:

        >>> help(emgfit.spectrum._smooth)

        """
        # Smooth spectrum (moving average with window function)
        data_smooth = self.data.copy()
        data_smooth['Counts'] = spectrum._smooth(self.data['Counts'].values,
                                                 window=window,
                                                 window_len=window_len)
        if plot_smoothed_spec:
            # Plot smoothed and original spectrum
            f = plt.figure(figsize=(figwidth,figwidth*4.5/18),dpi=dpi)
            ax = f.gca()
            self.data.plot(ax=ax)
            data_smooth.plot(ax=ax)
            plt.title("Smoothed spectrum")
            ax.legend(["Raw","Smoothed"])
            plt.ylim(0.1,)
            plt.yscale('log')
            plt.xlabel('m/z [u]')
            plt.ylabel('Counts per bin')
            plt.show()

        # Calculate second derivative using 2nd order central finite differences
        scale = 1/(data_smooth.values[1:-1] + 10) # scale to decrease y range
        # Use dm in denominator of sec_deriv if realistic units are desired:
        #dm = data_smooth.index[i+1]-data_smooth.index[i]
        sec_deriv = scale*(data_smooth.values[2:] - 2*data_smooth.values[1:-1] +
                           data_smooth.values[0:-2])/1**2
        data_sec_deriv = pd.DataFrame(data=sec_deriv, columns=['Counts'],
                                      index=data_smooth.iloc[1:-1].index)
        if plot_2nd_deriv:
            title = str("Scaled second derivative of spectrum - set threshold "
                        "indicated")
            self._plot_df(data_sec_deriv, title=title, yscale='linear',
                          ylabel='Amplitude [a.u.]', thres=-thres)

        # Take only negative part of re-scaled second derivative and invert
        sec_deriv_mod = np.where(sec_deriv < 0, -1.*sec_deriv, 0.0)
        data_sec_deriv_mod = pd.DataFrame(data=sec_deriv_mod,
                                          columns=['Counts'],
                                          index=data_smooth.iloc[1:-1].index)

        bin_width = self.data.index[1] - self.data.index[0] #assume uniform bins
        width_in_bins = int(width/bin_width) # width in units of bins
        peak_find = sig.find_peaks(data_sec_deriv_mod['Counts'].values,
                                   height=thres, width=width_in_bins)
        li_peak_pos = data_sec_deriv_mod.index.values[peak_find[0]]
        if plot_2nd_deriv:
            title = str("Negative part of scaled second derivative, inverted "
                        "- set threshold indicated")
            self._plot_df(data_sec_deriv_mod, title=title, thres=thres,
                          vmarkers=li_peak_pos, ymin=0.1*thres,
                          ylabel='Amplitude [a.u.]')

        # Reset peak list
        self.peaks = []

        # Create list of peak objects
        for x in li_peak_pos:
            p = peak(x,'?') # instantiate new peak
            self.peaks.append(p)
            self.fit_results.append(None)

        # Plot raw spectrum with detected peaks marked
        if plot_detection_result:
            self.plot(peaks=self.peaks,
                      title="Spectrum with detected peaks marked")
            print("Peak properties table after peak detection:")
            self.show_peak_properties()


    def add_peak(self, x_pos, species="?", m_AME=None, m_AME_error=None, Ex=0.0,
                 Ex_error=0.0, verbose=True):
        """Manually add a peak to the spectrum's :attr:`peaks` list.

        The position of the peak must be specified with the `x_pos` argument.
        If the peak's ionic species is provided with the `species` argument the
        corresponding AME literature values will be added to the :attr:`peak`.
        Alternatively, user-defined literature values can be provided with the
        `m_AME` and `m_AME_error` arguments. This option is helpful for isomers
        or in case of very recent measurements that haven't entered the AME yet.

        Parameters
        ----------
        x_pos : float [u]
            Position of peak to be added.
        species : str, optional
            :attr:`species` label for peak to be added following the
            :ref:`:-notation`. If assigned, :attr:`peak.m_AME`,
            :attr:`peak.m_AME_error` & :attr:`peak.extrapolated` are
            automatically updated with the corresponding AME literature values.
        m_AME : float [u], optional
            User-defined literature mass of peak to be added. Overwrites pre-
            existing :attr:`peak.m_AME` value.
        m_AME_error : float [u], optional
            User-defined literature mass uncertainty of peak to be added.
            Overwrites pre-existing :attr:`peak.m_AME_error`.
        Ex : float [keV], optional, default: 0.0
            Excitation energy of isomeric state in keV. When the peak is
            labelled as an isomer its literature mass :attr:`peak.m_AME`
            is calculated by adding `Ex` to the AME ground-state mass.
        Ex_error : float [keV], optional, default: 0.0
            Uncertainty of the excitation energy of the isomeric state in keV.
            When the peak is labelled as isomer its literature mass uncertainty
            :attr:`peak.m_AME_error` is calculated by adding `Ex_error` and the
            AME uncertainty of the ground-state mass in quadrature.
        verbose : bool, optional, default: `True`
            If `True`, a message is printed after successful peak addition.
            Intended for internal use only.

        See also
        --------
        :meth:`detect_peaks`
        :meth:`remove_peak`

        Note
        ----
        Adding a peak will shift the peak indeces of all peaks at higher masses
        by ``+1``.

        """
        # Check if there is already a peak with this marker position:
        if np.round(x_pos,6) in [np.round(p.x_pos,6) for p in self.peaks]:
            raise Exception("There is already a peak with the specified "
                            "`x_pos`.")
        # Instantiate new peak:
        p = peak(x_pos, species, m_AME=m_AME, m_AME_error=m_AME_error, Ex=Ex,
                 Ex_error=Ex_error)
        if m_AME is not None: # set mass number to closest int to m_AME value
            p.A = int(np.round(m_AME))
        self.peaks.append(p)
        def sort_x(peak):
            """Helper function for sorting peak list by marker pos. x_pos """
            return peak.x_pos
        self.peaks.sort(key=sort_x) # sort peak positions in ascending order
        peak_idx = self.peaks.index(p)
        self.fit_results.insert(peak_idx, None) # create empty fit result
        if verbose:
            print("Added peak at",x_pos,"u")


    def remove_peaks(self,peak_indeces=None,x_pos=None,species="?",verbose=True):
        """Remove specified peak(s) from the spectrum's :attr:`peaks` list.

        Select the peak(s) to be removed by specifying either the respective
        `peak_indeces`, `species` label(s) or peak marker position(s) `x_pos`.
        To remove multiple peaks at once, pass a list to one of the above
        arguments.

        Parameters
        ----------
        peak_indeces : int or list of int, optional
            Indeces of peak(s) to remove from the spectrum's :attr:`peaks` list
            (0-based!).
        x_pos : float or list of float [u]
            :attr:`x_pos` of peak(s) to remove from the spectrum's :attr:`peaks`
            list. Peak marker positions must be specified up to the 6th decimal.
        species : str or list of str
            :attr:`species` label(s) of peak(s) to remove from the spectrum's
            :attr:`peaks` list.

        Note
        ----
        Removing a peak will shift the peak indeces of all peaks at higher
        masses by ``-1``.

        Notes
        -----
        The current :attr:`peaks` list can be viewed by calling the
        :meth:`~spectrum.show_peak_properties` spectrum method.

        Added in version 0.2.0 (as successor method to `remove_peak`).

        """
        # Get indeces of peaks to remove
        err_msg1 = str("Use EITHER the `peak_indeces`, `x_pos` OR `species` "
                       "argument.")
        if peak_indeces is not None:
            assert x_pos is None and species == "?", err_msg1
            indeces = np.atleast_1d(peak_indeces)
        elif species != "?":
            assert x_pos is None, err_msg1
            species = np.atleast_1d(species)
            peaks = self.peaks
            indeces = [i for i in range(len(peaks))
                       if peaks[i].species in species]
            err_msg2 = str("Selection of one or multiple peaks from specified "
                           "`species` failed.")
            assert len(indeces) == len(species), err_msg2
        elif x_pos:
            x_pos = np.atleast_1d(x_pos)
            peaks = self.peaks
            indeces = [i for i in range(len(peaks))
                       if np.round(peaks[i].x_pos,6) in np.round(x_pos,6)]
            err_msg3 = str("Selection of one or multiple peaks from specified "
                           "`x_pos` failed.")
            assert len(indeces) == len(x_pos), err_msg3
        # Make safety copies for case of error in peak removals
        orig_peaks = copy.deepcopy(self.peaks)
        orig_results = copy.deepcopy(self.fit_results)
        rem_pos = []
        for i in sorted(indeces,reverse=True):
            try:
                rem_peak = self.peaks.pop(i)
                rem_pos.append(rem_peak.x_pos)
                self.fit_results.pop(i)
            except:
                # Restore original peaks and fit_results lists
                self.peaks = orig_peaks
                self.fit_results = orig_results
                msg = str("Removal of peak {0} failed! Restored the original "
                          " peaks and fit_results lists.").format(i)
                raise Exception(msg)
        if verbose:
            rem_pos.reverse() # switch to ascending order
            for x_pos in rem_pos:
                print("Removed peak at x_pos = {0:.6f} u".format(x_pos))


    def remove_peak(self,peak_index=None,x_pos=None,species="?"):
        """Remove specified peak from the spectrum's :attr:`peaks` list.

        Select the peak to be removed by specifying either the respective
        `peak_index`, `species` label or peak marker position `x_pos`.

        Parameters
        ----------
        peak_index : int or list of int, optional
            Indeces of peak(s) to remove from the spectrum's :attr:`peaks` list
            (0-based!).
        x_pos : float or list of float [u]
            :attr:`x_pos` of peak(s) to remove from the spectrum's :attr:`peaks`
            list. Peak marker positions must be specified up to the 6th decimal.
        species : str or list of str
            :attr:`species` label(s) of peak(s) to remove from the spectrum's
            :attr:`peaks` list.

        Note
        ----
        *This method is deprecated in v0.1.1 and will likely be removed in
        future versions, use :meth:`~spectrum.remove_peaks` instead!*

        """
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('once')
            msg = str("remove_peak is deprecated in v0.1.1 and will likely be "
                      "removed in future versions, use remove_peaks instead!")
            warnings.warn(msg, PendingDeprecationWarning)
        self.remove_peaks(peak_indeces=peak_index,x_pos=x_pos,species=species)


    def show_peak_properties(self):
        """Print properties of all peaks in :attr:`peaks` list.

        """
        #def None_to_nan(dict): # convert None values to NaN
        #    return { k: (np.nan if v is None else v) for k, v in dict.items() }
        #peak_dicts = [None_to_nan(p.__dict__) for p in self.peaks]
        peak_dicts = [p.__dict__ for p in self.peaks]
        u_format = "{:."+str(int(u_decimals))+"f}" # format of mass vals in [u]
        def fmt_m_u(value): # custom formatting to handle strings & None
            if type(value) not in (str,type(None)):
                value = u_format.format(value)
            return value
        rel_err_format = "{:.2e}" # format of relative uncertainty values
        def fmt_rel_err(value): # custom formatting to handle strings & None
            if type(value) not in (str,type(None)):
                value = rel_err_format.format(value)
            return value
        def fmt_A(value): # custom formatting to handle strings & None
            if type(value) not in (str,type(None)):
                value = "{:.0f}".format(value)
            return value
        format_dict = {"x_pos" : fmt_m_u,
                       "m_AME" : fmt_m_u,
                       "m_AME_error" : fmt_m_u,
                       "m_ion" : fmt_m_u,
                       "rel_stat_error" : fmt_rel_err,
                       "rel_recal_error" : fmt_rel_err,
                       "rel_peakshape_error" : fmt_rel_err,
                       "rel_mass_error" : fmt_rel_err,
                       "A" : fmt_A }
        df_prop = pd.DataFrame(peak_dicts)

        # Hide peaks of interest if blindfolded mode is on
        defined = [True if p.m_ion != None else False for p in self.peaks]
        pindeces = range(len(self.peaks))
        blinded = [True if i in self.blinded_peaks else False for i in pindeces]
        mask = np.logical_and(blinded, defined)
        df_prop.loc[mask, ['m_ion','atomic_ME_keV','m_dev_keV']] = 'blinded'

        # Apply formatting
        df_prop = df_prop.style.format(format_dict)

        # Mark peaks with MC PS errors with blue font
        df_prop = df_prop.apply(lambda col: ['color: royalblue'
                                      if i in self.peaks_with_MC_PS_errors
                                      else '' for i in range(col.size)],
                                      subset=['rel_peakshape_error'])
        # Mark peaks with statistical errors from resampling with green font
        df_prop = df_prop.apply(lambda col: ['color: forestgreen'
                                  if i in self.peaks_with_errors_from_resampling
                                  else '' for i in range(col.size)],
                                  subset=['area_error','rel_stat_error'])
        display(df_prop)
        any_MC_errs = any(self.peaks_with_MC_PS_errors)
        any_resampling_errs = any(self.peaks_with_errors_from_resampling)
        if any_MC_errs and any_resampling_errs: # add color legend
            from termcolor import colored
            print('        ',colored('stat. errors from resampling','green'),
                  colored('    Monte Carlo peakshape errors','blue'))
        elif any_MC_errs:
            from termcolor import colored
            print('                                    ',
                  colored('    Monte Carlo peakshape errors','blue'))
        elif any_resampling_errs:
            from termcolor import colored
            print('        ',colored('stat. errors from resampling','green'))


    def assign_species(self, species, peak_index=None, x_pos=None, Ex=0.0,
                       Ex_error=0.0):
        """Assign species label(s) to a single peak (or all peaks at once).

        If no single peak is selected with `peak_index` or `x_pos`, a list with
        species names for **all** peaks in the peak list must be passed to
        `species`. For already specified or unkown species insert ``None`` as a
        placeholder into the list to skip the species assignment for this peak.
        See `Notes` and `Examples` sections below for details on usage.

        Parameters
        ----------
        species : str or list of str
            The species name (or list of name strings) to be assigned to the
            selected peak (or to all peaks). For unkown or already assigned
            species, ``None`` should be inserted as placeholder at the
            corresponding position in the `species` list. :attr:`species` names
            must follow the :ref:`:-notation`.
        peak_index : int, optional
            Index of single peak to assign `species` name to.
        x_pos : float [u], optional
            :attr:`x_pos` of single peak to assign `species` name to. Must be
            specified up to 6th decimal.
        Ex : float [keV], optional, default: 0.0
            Excitation energy of isomeric state in keV. When the peak is
            labelled as isomer its literature mass :attr:`peak.m_AME` is
            calculated by adding `Ex` to the AME ground-state mass.
        Ex_error : float [keV], optional, default: 0.0
            Uncertainty of the excitation energy of the isomeric state in keV.
            When the peak is labelled as isomer its literature mass uncertainty
            :attr:`peak.m_AME_error` is calculated by adding `Ex_error` and the
            AME uncertainty of the ground-state mass in quadrature.

        Notes
        -----
        - Assignment of a single peak species:
          select peak by specifying peak position `x_pos` (up to 6th decimal) or
          `peak_index` argument (0-based! Check for peak index by calling
          :meth:`show_peak_properties` method on spectrum object).

        - Assignment of multiple peak species:
          Nothing should be passed to the 'peak_index' and 'x_pos' arguments.
          Instead the user specifies a list of the new species strings to the
          `species` argument (if there's N detected peaks, the list must have
          length N). Former species assignments can be kept by inserting blanks
          at the respective position in the `species` list, otherwise former
          species assignments are overwritten, also see examples below for
          usage.

         - Tentative assignments and isomers:
           Use ``'?'`` at the end of the `species` string or constituent element
           strings to indicate tentative assignments. Literature values are
           also fetched for peaks with tentative assignments.

         - Isomeric species:
           Isomers can be marked by appending a ``'m'`` or ``'m0'`` up to
           ``'m9'`` to the end of the respective element substring in `species`.
           For isomers no literature values are calculated unless the respective
           excitation energy is manually specified with the `Ex` argument.

        Examples
        --------
        Assign the peak with peak_index 2 (third-lowest-mass peak) as
        '1Cs133:-1e', leave all other peaks unchanged:

        >>> import emgfit as emg
        >>> spec = emg.spectrum(<input_file>) # mock code for foregoing data import
        >>> spec.detect_peaks() # mock code for foregoing peak detection
        >>> spec.assign_species('1Cs133:-1e', peak_index = 2)

        Assign multiple peaks:

        >>> import emgfit as emg
        >>> spec = emg.spectrum(<input_file>) # mock code for foregoing data import
        >>> spec.detect_peaks() # mock code for foregoing peak detection
        >>> spec.assign_species(['1Ru102:-1e', '1Pd102:-1e', 'Rh102:-1e?', None,'1Sr83:1F19:-1e', '?'])

        This assigns the species of the first, second, third and fourth peak
        with the respective labels in the specified list and fetches their AME
        values. The `'?'` ending of the ``'Rh102:-1e?'`` argument indicates a
        tentative species assignment, mind that literature values will still be
        calculated for this peak. Equivalently, ``'Rh102?:-1e'`` could have been
        used. The ``None`` argument leaves the species assignment of the 4th
        peak unchanged. The ``'?'`` argument overwrites any former species
        assignments to the last peak and marks the peak as unidentified.

        Mark peaks as isomers:

        >>> import emgfit as emg
        >>> spec = emg.spectrum(<input_file>) # mock code for foregoing data import
        >>> spec.detect_peaks() # mock code for foregoing peak detection
        >>> spec.assign_species('1In127:-1e', peak_index=0)    # ground state
        >>> spec.assign_species('1In127m:-1e', peak_index=1)   # first isomer
        >>> spec.assign_species('1In127m1:-1e', peak_index=2, Ex=1863,
        >>>                     Ex_error=58) # second isomer

        The above assigns peak 0 as ground state and fetches the corresponding
        literature values. Peak 1 is marked as the first isomeric state of
        In-127 but no literature values are calculated (since `Ex` is not
        specified). Peak 2 is marked as the second isomeric state of In-127 and
        the literature mass and its uncertainty are calculated from the
        respective ground-state AME values and the provided `Ex` and `Ex_error`.

        """
        try:
            if peak_index is not None:
                msg = "Use either the `peak_index` OR the `species` argument."
                assert x_pos is None, msg
                p = self.peaks[peak_index]
                p.species = species
                p.update_lit_values(Ex=Ex, Ex_error=Ex_error)
                print("Species of peak",peak_index,"assigned as",p.species)
            elif x_pos is not None:
                # Select peak at position 'x_pos'
                i = [i for i in range(len(self.peaks)) if abs(x_pos - self.peaks[i].x_pos) < 1e-06][0]
                p = self.peaks[i]
                p.species = species
                p.update_lit_values(Ex=Ex, Ex_error=Ex_error)
                print("Species of peak",i,"assigned as",p.species)
            elif len(species) == len(self.peaks):
                for i in range(len(species)): # loop over multiple species
                    species_i = species[i]
                    if species_i: # skip peak if 'None' given as argument
                        p = self.peaks[i]
                        p.species = species_i
                        p.update_lit_values(Ex=Ex, Ex_error=Ex_error)
                        print("Species of peak",i,"assigned as",p.species)
            else:
                msg = str("Peak selection in assign_species() failed. Check "
                          "method documentation for details on how to select "
                          "the peak of interest.")
                raise Exception(msg)
        except:
            print("\nSpecies assignment failed with the exception below. Check "
                  "method documentation for usage examples and syntax of "
                  "`species` strings.")
            raise


    def set_lit_values(self, peak_idx, m_AME, m_AME_error, extrapolated=False,
                       verbose=True):
        """Manually define the (ionic) literature mass and its error for a peak

        Existing values are overwritten.

        Parameters
        ----------
        peak_idx : int
            Index of peak to assign literature values for.
        m_AME : float [u]
            New literature mass.
        m_AME_error : float [u]
            New liteature mass uncertainty.
        extrapolated : bool, optional, default: False
            Flag indicating whether this literature value has been extrapolated.
        verbose : bool, optional, default: True
            Whether to print a status update after completion.

        Notes
        -----
        Added in version 0.3.6.

        """
        peak = self.peaks[peak_idx]
        peak.m_AME = m_AME
        peak.m_AME_error = m_AME_error
        peak.extrapolated = extrapolated
        peak.A = int(round(m_AME))
        if verbose is True:
            msg = str("Set literature mass of peak {} to m_AME = ({} +- {}) u"
                               ).format(peak_idx, peak.m_AME, peak.m_AME_error)
            print(msg)


    def add_peak_comment(self, comment, peak_index=None, x_pos=None,
                         species="?", overwrite=False):
        """Add a comment to a peak

        By default the `comment` argument will be appended to the end of the
        current :attr:`peak.comment` attribute (if the current comment is '-' it
        is overwritten by the `comment` argument). If `overwrite` is set `True`,
        the current :attr:`peak.comment` is overwritten with the 'comment'
        argument.

        Parameters
        ----------
        comment : str
            Comment to add to peak.
        peak_index : int, optional
            Index of :class:`peak` to add comment to.
        x_pos : float [u], optional
            :attr:`x_pos` of peak to add comment to (must be specified up to 6th
             decimal).
        species : str, optional
            :attr:`species` of peak to add comment to.
        overwrite : bool
            If `True` the current peak :attr:`comment` will be overwritten
            by `comment`, else `comment` is appended to the end of the current
            peak :attr:`comment`.

        Note
        ----
        The shape and mass calibrant peaks are automatically marked during the
        shape and mass calibration by inserting the protected flags
        ``'shape calibrant'``, ``'mass calibrant'`` or
        ``'shape and mass calibrant'`` into their peak comments. When
        user-defined comments are added to these peaks, it is ensured that the
        protected flags cannot be overwritten. **The above shape and mass
        calibrant flags should never be added to comments manually by the
        user!**

        """
        if peak_index is not None:
            pass
        elif species != "?": # select peak with label 'species'
            peak_index = [i for i in range(len(self.peaks))
                          if species == self.peaks[i].species][0]
        elif x_pos is not None: # select peak at position 'x_pos'
            peak_index = [i for i in range(len(self.peaks)) if np.round(x_pos,6)
                          == np.round(self.peaks[i].x_pos,6)][0]
        else:
            msg = str("Peak specification failed. Check method "
                      "documentation for details on peak selection.")
            raise Exception(msg)
        peak = self.peaks[peak_index]

        protected_flags = ('shape calibrant', 'shape & mass calibrant',
                           'mass calibrant') # order matters for overwriting!
        try:
            if any(s in comment for s in protected_flags):
                msg =  str("'shape calibrant','mass calibrant' and "
                           "'shape & mass calibrant' are protected flags. "
                           "User-defined comments must not contain these "
                           "flags. Re-phrase comment argument!")
                raise Exception(msg)
            if peak.comment == '-' or peak.comment is None:
                peak.comment = comment
            elif overwrite:
                if any(s in peak.comment for s in protected_flags):
                    import warnings
                    wmsg = str("The protected flags 'shape calibrant', "
                               "'mass calibrant' or 'shape & mass calibrant' "
                               "cannot be overwritten.")
                    warnings.warn(wmsg)
                    flag = [s for s in protected_flags if s in peak.comment][0]
                    peak.comment = peak.comment.replace(peak.comment,
                                                        flag+', '+comment)
                else:
                    peak.comment = comment
            else:
                peak.comment = peak.comment+comment
            print("Comment of peak",peak_index,"was changed to: ",peak.comment)
        except TypeError:
            raise TypeError("'comment' argument must be given as type string.")


    def set_blinded_peaks(self, indeces, overwrite=False):
        """Specify for which peaks mass values will be hidden for blind analysis

        This method adds peaks to the spectrum's list of
        :attr:`~emgfit.spectrum.spectrum.blinded_peaks`. For these peaks, the
        obtained mass values in the peak properties table and the peak position
        parameters `mu` in fit reports will be hidden. Literature values and
        mass uncertainties remain visible. All results are unblinded upon
        export with the :meth:`save_results` method.

        Parameters
        ----------
        indeces : int or list of int
            Indeces of peaks of interest whose obtained mass values are to be
            blinded.
        overwrite : bool, optional, default: False
            If `False` (default), the specified `indeces` are added to the
            :attr:`blinded_peaks` list. If `True`, the current
            :attr:`blinded_peaks` list is replaced by the specified `indeces`.

        Examples
        --------
        Activate blinding for peaks 0 & 3 of spectrum object `spec`:

        >>> spec.set_blinded_peaks([0,3])

        Add peak 3 to list of blinded peaks:

        >>> spec.set_blinded_peaks([3])

        Turn off blinding by resetting the blinded peaks attribute to an empty
        list:

        >>> spec.set_blinded_peaks([], overwrite=True)

        """
        indeces = np.atleast_1d(indeces).tolist()
        peak_indeces = range(len(self.peaks))
        if any(i not in peak_indeces for i in indeces):
            raise Exception("No peaks found for some of the specified indeces.")
        if overwrite is False:
            for idx in indeces:
                if idx not in self.blinded_peaks:
                    self.blinded_peaks.append(idx)
        else:
            self.blinded_peaks = indeces # overwrite list

        self.blinded_peaks.sort()
        s_list = ', '.join(map(str, self.blinded_peaks)) # convert list to str
        print("Blinding is activated for peaks:", s_list)


    def _show_blinded_report(self, result):
        """Display fit result with positions of blinded peaks replaced by NaN
        """
        orig_pars = copy.deepcopy(result.params)
        # Replace all `mu` parameter values of peaks to blind with NaN
        for idx in self.blinded_peaks:
            mu_key = 'p{0}_mu'.format(idx)
            try:
                result.params[mu_key].value = np.nan
            except KeyError:
                pass # skip blinded peaks that aren't in result
        display(result)
        # RESET parameter values
        result.params = orig_pars


    def _add_peak_markers(self,yscale='log',ymax=None,peaks=None):
        """Internal function for adding peak markers to current figure object.

        Place this function inside spectrum methods between ``plt.figure()`` and
        ``plt.show()``. Only for use on already fitted spectra!

        Parameters
        ----------
        yscale : str, optional
            Scale of y-axis, either 'linear' or 'log'.
        ymax : float
            Maximal y-value of spectrum data to plot. Used to set marker length.
        peaks : list of :class:`peak`
            List of peaks to add peak markers for.

        """
        if peaks is None:
            peaks = self.peaks
        data = self.data
        if yscale == 'log':
            for p in peaks:
                x_idx = np.argmin(np.abs(data.index.values - p.x_pos))
                ymin = data.iloc[x_idx]
                plt.vlines(x=p.x_pos, ymin=ymin, ymax=1.3*ymax,
                           linestyles='dashed', color='black')
                plt.text(p.x_pos, 1.44*ymax, self.peaks.index(p),
                         horizontalalignment='center', fontsize=labelsize)
        else:
            for p in peaks:
                x_idx = np.argmin(np.abs(data.index.values - p.x_pos))
                ymin = data.iloc[x_idx]
                plt.vlines(x=p.x_pos, ymin=ymin, ymax=1.11*ymax,
                           linestyles='dashed', color='black')
                plt.text(p.x_pos, 1.13*ymax, self.peaks.index(p),
                         horizontalalignment='center', fontsize=labelsize)


    def plot_fit(self, fit_result=None, plot_title=None,
                 show_peak_markers=True, sigmas_of_conf_band=0, error_every=1,
                 x_min=None, x_max=None, plot_filename=None):
        """Plot data and fit result in logarithmic and linear y-scale.

        Only a single fit result can be plotted with this method. If neither
        `fit_result` nor `x_min` and `x_max` are specified, the full mass range
        is plotted.

        Plots can be saved to a file using the `plot_filename` argument.

        Parameters
        ----------
        fit_result : :class:`lmfit.model.ModelResult`, optional, default: ``None``
            Fit result to plot. If ``None``, defaults to fit result of first
            peak in specified mass range (taken from :attr:`fit_results` list).
        plot_title : str or None, optional
            Title of plots. If ``None``, defaults to a string with the fit model
            name and cost function of the `fit_result` to ensure clear
            indication of how the fit was obtained.
        show_peak_markers : bool, optional, default: `True`
            If `True`, peak markers are added to the plots.
        sigmas_of_conf_band : int, optional, default: 0
            Coverage probability of confidence band in sigma (only shown in
            log-plot). If ``0``, no confidence band is shown (default).
        error_every : int, optional, default: 1
            Show error bars only for every `error_every`-th data point.
        x_min, x_max : float [u], optional
            Start and end of mass range to plot. If ``None``, defaults to the
            minimum and maximum of the spectrum's mass :attr:`data`.
        plot_filename : str, optional, default: None
            If not ``None``, the plots will be saved to two separate files named
            '<`plot_filename`>_log_plot.png' & '<`plot_filename`>_lin_plot.png'.
            **Caution: Existing files with identical name are overwritten.**

        """
        if fit_result is None and x_min is None and x_max is None: # full spec
            x_min = min(self.data.index)
            x_max = max(self.data.index)
        if fit_result is not None and x_min is None:
            x_min = min(fit_result.x)
        if fit_result is not None and x_max is None:
            x_max = max(fit_result.x)
        # Select peaks in mass range of interest:
        peaks_to_plot = [p for p in self.peaks if (x_min < p.x_pos < x_max)]
        idx_first_peak = self.peaks.index(peaks_to_plot[0])
        # If still not defined, get fit result from 1st peak in mass range
        if fit_result is None:
           fit_result = self.fit_results[idx_first_peak]
           idx_last_peak = self.peaks.index(peaks_to_plot[-1])
           if fit_result != self.fit_results[idx_last_peak]:
               raise Exception("Multiple fit results in specified mass range - "
                               "chose range to only include peaks contained in "
                               "a single fit result. ")

        if plot_title is None:
           plot_title = fit_result.fit_model+' '+fit_result.cost_func+' fit'
        i_min = np.argmin(np.abs(fit_result.x - x_min))
        i_max = np.argmin(np.abs(fit_result.x - x_max))
        y_max_log = max( max(self.data.values[i_min:i_max]),
                         max(fit_result.best_fit[i_min:i_max]) )
        y_max_lin = max( max(self.data.values[i_min:i_max]),
                         max(fit_result.init_fit[i_min:i_max]),
                         max(fit_result.best_fit[i_min:i_max]) )

        # Plot fit result with logarithmic y-scale
        f1 = plt.figure(figsize=(figwidth,figwidth*8.5/18), dpi=dpi)
        ax = f1.gca()
        plt.errorbar(fit_result.x, fit_result.y, yerr=fit_result.y_err, fmt='.',
                     color='royalblue', linewidth=0.5, markersize=msize,
                     errorevery=error_every, label='data', zorder=1)
        plt.plot(fit_result.x, fit_result.best_fit, '-', color='red',
                 linewidth=lwidth, label='best-fit', zorder=10)
        comps = fit_result.eval_components(x=fit_result.x)
        for peak in peaks_to_plot: # loop over peaks to plot
            peak_index = self.peaks.index(peak)
            pref = 'p{0}_'.format(peak_index)
            plt.plot(fit_result.x, comps[pref],'--', linewidth=lwidth, zorder=5)
        if show_peak_markers:
            self._add_peak_markers(yscale='log', ymax=y_max_log,
                                   peaks=peaks_to_plot)
        # add confidence band with specified number of sigmas
        if sigmas_of_conf_band!=0 and fit_result.errorbars == True:
            dely = fit_result.eval_uncertainty(sigma=sigmas_of_conf_band)
            label = str(sigmas_of_conf_band)+r'$\sigma$ confidence band'
            plt.fill_between(fit_result.x, fit_result.best_fit-dely,
                             fit_result.best_fit+dely, color='tomato',
                             alpha=0.5, label=label)
        plt.title(plot_title)
        plt.xlabel('m/z [u]')
        plt.ylabel('Counts per bin')
        plt.yscale('log')
        plt.ylim(0.1, 2.3*y_max_log)
        plt.xlim(x_min,x_max)
        ax.get_xaxis().get_major_formatter().set_useOffset(False) # no offset
        if plot_filename is not None:
            try:
                plt.savefig(plot_filename+'_log_plot.png', transparent=False,
                            dpi=600)
            except:
                raise
        plt.show()

        # Plot residuals and fit result with linear y-scale
        std_residual = (fit_result.best_fit - fit_result.y)/fit_result.y_err
        y_max_res = np.max(np.abs(std_residual))
        x_fine = np.arange(x_min,x_max,0.2*(fit_result.x[1]-fit_result.x[0]))
        y_fine = fit_result.eval(x=x_fine)
        f2, axs = plt.subplots(2,1,figsize=(figwidth,figwidth*8.5/18),dpi=dpi,
                               gridspec_kw={'height_ratios': [1, 2.5]})
        ax0 = axs[0]
        ax0.set_title(plot_title)
        ax0.plot(fit_result.x, std_residual,'.',color='royalblue',
                 markersize=msize)
        #ax0.hlines(1,x_min,x_max,linestyle='dashed', color='black')
        ax0.hlines(0,x_min,x_max, color='black', zorder=10)
        #ax0.hlines(-1,x_min,x_max,linestyle='dashed', color='black')
        ax0.set_ylim(-1.05*y_max_res, 1.05*y_max_res)
        ax0.set_ylabel(r'Residual / $\sigma$')
        #ax0.tick_params(axis='x', labelsize=0) # hide tick labels
        ax1 = axs[1]
        ax1.errorbar(fit_result.x, fit_result.y, yerr=fit_result.y_err, fmt='.',
                 color='royalblue', linewidth=1, markersize=msize, label='data',
                 errorevery=error_every, zorder=1)
        ax1.plot(x_fine, fit_result.eval(params=fit_result.init_params,x=x_fine),
                 linestyle='dashdot', color='green', label='init-fit', zorder=5)
        ax1.plot(x_fine, fit_result.eval(x=x_fine), '-', color='red',
                 linewidth=lwidth, label='best-fit', zorder=10)
        ax1.set_title('')
        ax1.set_ylim(-0.05*y_max_lin, 1.2*y_max_lin)
        ax1.set_ylabel('Counts per bin')
        ax1.legend()
        for ax in axs:
            ax.set_xlim(x_min,x_max)
            ax.get_xaxis().get_major_formatter().set_useOffset(False) # no offset
        if show_peak_markers:
            self._add_peak_markers(yscale='lin', ymax=y_max_lin,
                                   peaks=peaks_to_plot)
        plt.xlabel('m/z [u]')
        if plot_filename is not None:
            try:
                plt.savefig(plot_filename+'_lin_plot.png', transparent=False,
                            dpi=600)
            except:
                raise
        plt.show()


    def plot_fit_zoom(self, peak_indeces=None, x_center=None, x_range=0.01,
                      plot_title=None, show_peak_markers=True, error_every=1,
                      sigmas_of_conf_band=0, plot_filename=None):
        """Show logarithmic and linear plots of data and fit curve zoomed to
        peaks or mass range of interest.

        There is two alternatives to define the plots' mass ranges:

        1. Specifying peaks-of-interest with the `peak_indeces`
           argument. The mass range is then automatically chosen to include all
           peaks of interest. The minimal mass range to include around each peak
           of interest can be adjusted using `x_range`.
        2. Specifying a mass range of interest with the `x_center` and `x_range`
           arguments.

        Parameters
        ----------
        peak_indeces : int or list of ints, optional
            Index of single peak or indeces of multiple neighboring peaks to
            show (peaks must belong to the same :attr:`fit_result`).
        x_center : float [u], optional
            Center of manually specified mass range to plot.
        x_range : float [u], optional, default: 0.01
            Width of mass range to plot around 'x_center' or minimal mass range
            to include around each specified peak of interest.
        plot_title : str or None, optional
            Title of plots. If ``None``, defaults to a string with the fit model
            name and cost function of the `fit_result` to ensure clear
            indication of how the fit was obtained.
        show_peak_markers : bool, optional, default: `True`
            If `True`, peak markers are added to the plots.
        error_every : int, optional, default: 1
            Show error bars only for every `error_every`-th data point.
        sigmas_of_conf_band : int, optional, default: 0
            Coverage probability of confidence band in sigma (only shown in
            log-plot). If ``0``, no confidence band is shown (default).
        plot_filename : str or None, optional, default: None
            If not ``None``, the plots will be saved to two separate files named
            '<`plot_filename`>_log_plot.png' & '<`plot_filename`>_lin_plot.png'.
            **Caution: Existing files with identical name are overwritten.**

        """
        if isinstance(peak_indeces,list):
            x_min = self.peaks[peak_indeces[0]].x_pos - 0.5*x_range
            x_max = self.peaks[peak_indeces[-1]].x_pos + 0.5*x_range
        elif type(peak_indeces) == int:
            peak = self.peaks[peak_indeces]
            x_min = peak.x_pos - 0.5*x_range
            x_max = peak.x_pos + 0.5*x_range
        elif x_center is not None:
            x_min = x_center - 0.5*x_range
            x_max = x_center + 0.5*x_range
        else:
            raise Exception("\nMass range to plot could not be determined. "
                            "Check documentation on method parameters.\n")
        self.plot_fit(x_min=x_min, x_max=x_max, plot_title=plot_title,
                      show_peak_markers=show_peak_markers,
                      error_every=error_every,
                      sigmas_of_conf_band=sigmas_of_conf_band,
                      plot_filename=plot_filename)


    def comp_model(self,peaks_to_fit,model='emg22',init_pars=None,
                   vary_shape=False,vary_baseline=True,index_first_peak=None):
        """Create a multi-peak composite model with the specified peak shape.

        **Primarily intended for internal usage.**

        Parameters
        ----------
        peaks_to_fit : list of :class:`peak`
            :class:`peaks` to be fitted with composite model.
        model : str, optional
            Name of fit model to use for all peaks (e.g. ``'Gaussian'``,
            ``'emg12'``, ``'emg33'``, ... - for full list see
            :ref:`fit_model_list`).
        init_pars : dict, optional, default: ``None``
            Default initial shape parameters for fit model. If ``None`` the
            default parameters defined in the :mod:`~emgfit.fit_models` module
            will be used after scaling to the spectrum's :attr:`mass_number`.
            For more details and a list of all shape parameters see the
            :ref:`peak-shape calibration` article.
        vary_shape : bool, optional
            If `False` only the amplitude (`amp`) and Gaussian centroid (`mu`)
            model parameters will be varied in the fit. If `True`, the shape
            parameters (`sigma`, `theta`,`etas` and `taus`) will also be varied.
        vary_baseline : bool, optional
            If `True` a varying uniform baseline will be added to the fit
            model as varying model parameter `c_bkg`. If `False`, the baseline
            parameter `c_bkg` will be kept fixed at 0.
        index_first_peak : int, optional
            Index of first (lowest mass) peak in fit range, used for enforcing
            common shape for all peaks.

        Notes
        -----
        The initial amplitude for each peak is estimated by taking the counts in
        the bin closest to the peak's :attr:`x_pos` and scaling this number with
        an empirically determined constant and the spectrum's :attr:`mass_number`.

        """
        model = getattr(fit_models,model) # get single peak model from fit_models.py
        mod = fit.models.ConstantModel(prefix='bkg_') #(independent_vars='x',prefix='bkg_')
        if vary_baseline == True:
            mod.set_param_hint('bkg_c', value= 0.1, min=0,max=4, vary=True)
        else:
            mod.set_param_hint('bkg_c', value= 0.0, vary=False)
        df = self.data
        for peak in peaks_to_fit:
            peak_index = self.peaks.index(peak)
            # Get x_pos of closest bin
            x_pos = df.index[np.argmin(np.abs(df.index.values - peak.x_pos))]
            # Estimate initial amplitude from counts in closest bin, the
            # conversion factor 1/2500 is empirically determined and somewhat
            # shape-dependent:
            amp = max(df['Counts'].loc[x_pos]/2500*(self.mass_number/100),1e-04)
            if init_pars:
                this_mod = model(peak_index, peak.x_pos, amp,
                                 init_pars=init_pars,
                                 vary_shape_pars=vary_shape,
                                 index_first_peak=index_first_peak)
            else:
                this_mod = model(peak_index, peak.x_pos, amp,
                                 vary_shape_pars=vary_shape,
                                 index_first_peak=index_first_peak)
            mod = mod + this_mod
        return mod


    def _get_MCMC_par_samples(self, fit_result, steps=12000, burn=500, thin=250,
                              show_MCMC_fit_result=False, covar_map_fname=None,
                              n_cores=-1, MCMC_seed=1364):
        """Map out parameter covariances and posterior distributions using
        Markov-chain Monte Carlo (MCMC) sampling

        **This method is intended for internal usage and for single peaks
        only.**

        Parameters
        ----------
        fit_result : :class:`lmfit.model.ModelResult`
            Fit result of region explore with MCMC sampling. since `emcee` only
            efficiently samples unimodal distributions, `fit_result` should hold
            the result of a single-peak fit (typically of the peak-shape
            calibrant). The MCMC walkers are initialized with randomized
            parameter values drawn from normal distributions whose mean and
            standard deviation are given by the respective best-fit values and
            uncertainties stored in `fit_result`.
        steps : int, optional
            Number of MCMC sampling steps.
        burn : int, optional
            Number of initial sampling steps to discard ("burn-in" phase).
        thin : int, optional
            After sampling, only every `thin`-th sample is used for further
            treatment. It is recommended to set `thin`  to at least half the
            autocorrelation time.
        show_MCMC_fit_result : bool, optional, default: False
            If `True`, a maximum likelihood estimate is derived from the MCMC
            samples with best-fit values estimated by the median of the samples.
            The MCMC MLE result can be compared to the conventional `fit_result`
            as an additional crosscheck.
        covar_map_fname : str or None (default), optional
            If not `None`, the parameter covariance map will be saved as
            "<covar_map_fname>_covar_map.png".
        n_cores : int, optional, default: -1
            Number of CPU cores to use for parallelized sampling. If `-1`
            (default) all available cores will be used.
        MCMC_seed : int, optional
            Random state for reproducible sampling.

        Yields
        ------
        MCMC results saved in `result_emcee` attribute of fit_result.

        Notes
        -----
        Markov-Chain Monte Carlo (MCMC) algorithms are a powerful tool to
        efficiently sample the posterior probability density functions (PDFs) of
        model parameters. In simpler words: MCMC methods can be used to estimate
        the distributions of parameter values which are supported by the data.
        An MCMC algorithm sends out a number of so-called walkers on stochastic
        walks through the parameter space (in this method the number of MCMC
        walkers is fixed to 20 times the number of varied parameters). MCMC
        methods are particularly important in situations where conventional
        sampling techniques become intractable or inefficient. For MCMC sampling
        emgfit deploys lmfit's implementation of the
        :class:`emcee.EnsembleSampler` from the `emcee`_ package [1]_. Since
        emcee's :class:`~emcee.EnsembleSampler` is only optimized for uni-modal
        probability density functions this method should only be used to explore
        the parameter space of a single-peak fit.

        A complication with MCMC methods is that there is usually no rigorous
        way to prove that the sampling chain has converged to the true PDF.
        Instead it is at the user's disgression to decide after how many
        sampling steps a sufficient amount of convergence is achieved. Gladly,
        there is a number of heuristic tools that can help in judging
        convergence. The most common measure of the degree of convergence is the
        integrated autocorrelation time (`tau`). If the integrated
        autocorrelation time shows only small changes over time the MCMC chain
        can be assumed to be converged. To ensure a sufficient degree of
        convergence this method will issue a warning whenever the number of
        performed sampling steps is smaller than 50 times the integrated
        autocorrelation time of at least one parameter. If this rule of thumb is
        violated it is strongly advisable to run a longer chain. An additonal
        aid in judging the performance of the MCMC chain are the provided plots
        of the MCMC traces. These plots show the paths of all MCMC walkers
        through parameter space. Dramatic changes of the initial trace envelopes
        indicate that the chain has not reached a stationary state yet and is
        still in the so-called "burn-in" phase. Samples in this region are
        discarded by setting the `burn` argument to an appropriate number of
        steps (default burn-in: 500 steps).

        Another complication of MCMC algorithms is the fact that nearby samples
        in a MCMC chain are not indepedent. To reduce correlations between
        samples MCMC chains are usually "thinned out" by only storing the result
        of every m-th MCMC iteration. The number of steps after which two
        samples can be assumed to be uncorrelated/independent (so to say the
        memory of the chain) is given by the integrated autocorrelation time
        (tau). To be conservative, emgfit uses a thinning interval of ``m = 250``
        by default and issues a warning when ``m < tau`` for at least one of the
        parameters. Since more data is discarded, a larger thinning interval
        comes with a loss of precision of the posterior PDF estimates. However,
        a sufficient amount of thinning is still advisable since emgfit's MC
        peak-shape error determination (:meth:`get_MC_peakshape_errors`) relies
        on independent parameter samples.

        As a helpful measure for tuning MCMC chains, emgfit provides a plot of
        the "acceptance fraction" for each walker, i.e. the fraction of
        suggested walker steps which were accepted. The developers of emcee's
        EnsembleSampler suggest acceptance fractions between 0.2 and 0.5 as a
        rule of thumb for a well-behaved chain. Acceptance fractions falling
        short of this for many walkers can indicate poor initialization or a too
        small number of walkers.

        For more details on MCMC see the excellent introductory papers
        `"emcee: The MCMC hammer"`_ [1]_ and
        `"Data Analysis Recipes: Using Markov Chain Monte Carlo"`_ [2]_.

        .. _`emcee`: https://iopscience.iop.org/article/10.1086/670067
        .. _`"emcee: The MCMC hammer"`: https://iopscience.iop.org/article/10.1086/670067
        .. _`"Data Analysis Recipes: Using Markov Chain Monte Carlo"`:
           https://iopscience.iop.org/article/10.3847/1538-4365/aab76e

        See also
        --------
        :meth:`determine_peak_shape`
        :meth:`get_MC_peakshape_errors`

        References
        ----------
        .. [1] Foreman-Mackey, Daniel, et al. "emcee: the MCMC hammer."
           Publications of the Astronomical Society of the Pacific 125.925
           (2013): 306.
        .. [2] Hogg, David W., and Daniel Foreman-Mackey. "Data analysis
           recipes: Using markov chain monte carlo." The Astrophysical Journal
           Supplement Series 236.1 (2018): 11.

        """
        ## This feature is based on
        ## `<https://lmfit.github.io/lmfit-py/examples/example_emcee_Model_interface.html>`_.
        print("\n### Evaluating posterior PDFs using MCMC sampling ###\n")
        ndim = fit_result.nvarys # dimension of parameter space to explore
        print("Number of varied parameters:         ndim =",fit_result.nvarys)
        nwalkers = 20*fit_result.nvarys # total number of MCMC walkers
        emcee_params = fit_result.params.copy()
        print("Number of MCMC steps:               steps =",steps)
        print("Number of initial steps to discard:  burn =",burn)
        print("Length of thinning interval:         thin =",thin)

        # Get names, best-fit values and errors of parameters to vary
        varied_pars = emcee_params.copy()
        for key, p in emcee_params.items():
            if p.vary is False or p.expr is not None:
                del varied_pars[key]
        assert len(varied_pars.values()) == ndim,"Length of varied_pars != ndim"
        varied_par_names = [p.name for p in varied_pars.values()]
        varied_par_vals = [p.value for p in varied_pars.values()]
        varied_par_errs = [p.stderr for p in varied_pars.values()]


        # Initialize the walkers with normal dist. around best-fit values
        np.random.seed(MCMC_seed) # make MCMC chains reproducible
        r0 = np.array(varied_par_errs) # sigma of initial Gaussian PDFs
        p0 = [varied_par_vals + r0*np.random.randn(ndim) for i in range(nwalkers)]

        from multiprocessing import cpu_count, Pool
        if n_cores == -1:
            n_cores = int(cpu_count())
        pool = Pool(n_cores)
        print("Number of CPU cores to use:       n_cores =",n_cores)
        # Set emcee options
        # It is advisable to thin by about half the autocorrelation time
        emcee_kws = dict(steps=steps, burn=burn, thin=thin, nwalkers=nwalkers,
                         float_behavior='chi2', is_weighted=True, pos=p0,
                         progress='notebook', seed=MCMC_seed, workers=pool)

        mod = fit_result.model
        x = fit_result.x
        y = fit_result.y
        weights = fit_result.weights
        # Perform emcee MCMC sampling
        import warnings
        with warnings.catch_warnings(): # suppress DeprecationWarning from emcee
            warnings.filterwarnings("ignore", message=
                                 "This function will be removed in tqdm==5.0.0")
            result_emcee = mod.fit(y, x=x, params=emcee_params, weights=weights,
                                   method='emcee', nan_policy='propagate',
                                   fit_kws=emcee_kws)
        fit_result.flatchain = result_emcee.flatchain # store sampling chain

        # Save chain to HDF5 file #TODO
        #import h5py
        #import emcee
        #filename = self.input_filename+"_MCMC_sampling.h5"
        #hf = h5py.File(filename, 'w')
        #hf.create_dataset('dataset_1', data=d1)

        # Plot MCMC traces
        ##set_matplotlib_formats('png') # prevent excessive image file size
        fig, axes = plt.subplots(ndim,figsize=(figwidth, 2.7*ndim),sharex=True)
        samples = result_emcee.sampler.get_chain()[..., :, :] # thinned chain without burn-in
        for i in range(ndim):
            par_chains = samples[:, :, i] # chains for i-th varied parameter
            ax = axes[i]
            ax.plot(par_chains, 'k', alpha=0.3)
            ax.set_ylabel(varied_par_names[i],fontsize=16)
            ax.tick_params(axis='both',labelsize=16)
            ax.axvline(emcee_kws['burn'])
        axes[-1].set_xlabel('steps',fontsize=16)
        axes[0].set_title('MCMC traces before thinning with burn-in cut-off marker',
                          fontdict={'fontsize':17})
        plt.show()
        ##set_matplotlib_formats(plot_fmt)  # reset plot format

        # Plot acceptance fraction of emcee
        f = plt.figure(figsize=(figwidth,figwidth*0.5))
        ax = f.gca()
        ax.tick_params(axis='both',labelsize=16)
        plt.plot(result_emcee.acceptance_fraction)
        plt.xlabel('walker',fontsize=16)
        plt.ylabel('acceptance fraction',fontsize=16)
        plt.show()

        # Plot autocorrelation times of Parameters
        result_emcee.acor = result_emcee.sampler.get_autocorr_time(quiet=True)
        if any(thin < result_emcee.acor):
            import warnings
            warnings.warn("Thinning interval `thin` is less than the "
                          "integrated autocorrelation time for at least one "
                          "parameter. Consider increasing `thin` MCMC keyword "
                          "argument to ensure independent parameter samples.",
                          UserWarning)
        if hasattr(result_emcee, "acor"):
            print("Autocorrelation time for the parameters:")
            print("----------------------------------------")
            for i, p in enumerate(varied_pars):
                try:
                    print("{:>10}: {:.2f} steps".format(p,result_emcee.acor[i]))
                except IndexError:
                    print("\nEncountered index error in autocorrelation print.")

        print("\nTotal number of MCMC parameter sets after discarding burn-in "
              "and thinning:",len(result_emcee.flatchain),"\n")

        if show_MCMC_fit_result:
            plt.figure(figsize=(figwidth,figwidth*7/18),dpi=dpi)
            plt.plot(x, mod.eval(params=fit_result.params, x=x),
                     label=fit_result.cost_func, zorder=100)
            result_emcee.plot_fit(data_kws=dict(color='gray', markersize=msize))
            plt.title("MCMC result vs. {} result".format(fit_result.cost_func))
            plt.yscale("log")
            plt.xlabel("m/z [u]")
            plt.ylabel("Counts per bin")
            plt.show()

            fit.report_fit(result_emcee)

            # Find the maximum likelihood solution
            highest_prob = np.argmax(result_emcee.lnprob)
            hp_loc = np.unravel_index(highest_prob, result_emcee.lnprob.shape)
            mle_soln = result_emcee.chain[hp_loc]
            print("\nMaximum likelihood Estimation from MCMC")
            print(  "---------------------------------------")
            for ix, param in enumerate(varied_pars):
                try:
                    print(param + ': ' + str(mle_soln[ix]))
                except IndexError:
                    print("\nEncountered index error in MCMC MLE result print.")
                    pass

            # Use mu of first fitted peak
            first_mu = [s for s in varied_par_names if s.endswith('mu')][0]
            quantiles = np.percentile(result_emcee.flatchain[first_mu],
                                      [2.28, 15.9, 50, 84.2, 97.7])
            print("\n 1-sigma spread of mu:", 0.5 * (quantiles[3] - quantiles[1]))
            print(" 2-sigma spread of mu:",  0.5 * (quantiles[4] - quantiles[0]))

        # Plot parameter covariances returned by emcee

        #from copy import deepcopy
        #chain = deepcopy(result_emcee.flatchain)
        ### Format axes labels and add units
        # labels = []
        # for i, s in enumerate(varied_par_names):
        #     lab = s.lstrip("p0123456789_") # strip prefixes
        #     if (lab == 'sigma') or ('tau' in lab):
        #         chain[s] *= 1e06
        #         lab += r" [$\mu u$]"
        #     elif lab == 'mu':
        #         offset = np.round(varied_par_vals[i],3)
        #         chain[s] = (chain[s] - offset)*1e06
        #         lab += r" [$\mu u$] - {:.3f}$u$".format(offset)
        #     labels.append(lab)
        labels = [s.lstrip("p0123456789_") for s in result_emcee.var_names]
        peak_idx = varied_par_names[-1].split('p')[1].split('_')[0]
        print("\nCovariance map for peak {} with 0.16, 0.50 & 0.84 quantiles "
              "(dashed lines) and best-fit values (blue lines):".format(
              peak_idx))
        import corner
        ##set_matplotlib_formats('png') # prevent excessive image file size
        percentile_range = [0.99]*ndim  # percentile of samples to plot
        fig_cor, axes = plt.subplots(ndim,ndim,figsize=(16,16))
        corner.corner(result_emcee.flatchain, #chain
                      fig=fig_cor,
                      labels=labels,
                      bins=25,
                      max_n_ticks=3,
                      truths=list(varied_par_vals),
                      hist_bin_factor=2,
                      range=percentile_range,
                      levels=(1-np.exp(-0.5),),
                      quantiles=[0.1587, 0.5, 0.8413]) # 1-sigma contour assumes Gaussian PDFs
        for ax in fig_cor.get_axes():
            ax.tick_params(axis='both', labelsize=10)
            ax.xaxis.offsetText.set_fontsize(10)
            ax.yaxis.offsetText.set_fontsize(10)
            #ax.yaxis.set_offset_position('left')
            ax.xaxis.label.set_size(15)
            ax.yaxis.label.set_size(15)
            labelpad = 0.41
            ax.xaxis.set_label_coords(0.5, -0.3 - labelpad)
            ax.yaxis.set_label_coords(-0.3 - labelpad, 0.5)
        if covar_map_fname is not None:
            plt.savefig(covar_map_fname+"_covar_map.png", dpi=600,
                        pad_inches=0.3, bbox_inches='tight')
        plt.show()
        ##set_matplotlib_formats(plot_fmt) # reset image format


    def peakfit(self,fit_model='emg22', cost_func='chi-square', x_fit_cen=None,
                x_fit_range=None, init_pars=None, vary_shape=False,
                vary_baseline=True, method='least_squares', fit_kws=None,
                show_plots=True, show_peak_markers=True, sigmas_of_conf_band=0,
                error_every=1, plot_filename=None, map_par_covar=False,
                **MCMC_kwargs):
        """Internal routine for fitting peaks.

        Fits full spectrum or subrange (if `x_fit_cen` and `x_fit_range` are
        specified) and optionally shows results.

        **This method is for internal usage. Use :meth:`spectrum.fit_peaks`
        method to fit peaks and automatically update peak properties dataframe
        with obtained fit results!**

        Parameters
        ----------
        fit_model : str, optional, default: ``'emg22'``
            Name of fit model to use (e.g. ``'Gaussian'``, ``'emg12'``,
            ``'emg33'``, ... - for full list see :ref:`fit_model_list`).
        cost_func : str, optional, default: 'chi-square'
            Name of cost function to use for minimization.

            - If ``'chi-square'``, the fit is performed by minimizing Pearson's
              chi-squared statistic:

              .. math::

                  \\chi^2_P = \\sum_i \\frac{(f(x_i) - y_i)^2}{f(x_i)}.

            - If ``'MLE'``, a binned maximum likelihood estimation is performed
              by minimizing the (doubled) negative log-likelihood ratio:

              .. math::

                  L = 2\\sum_i \\left[ f(x_i) - y_i + y_i ln\\left(\\frac{y_i}{f(x_i)}\\right)\\right]

            See `Notes` below for details.
        x_fit_cen : float [u], optional
            Center of mass range to fit (only specify if subset of spectrum is
            to be fitted).
        x_fit_range : float [u], optional
            Width of mass range to fit (only specify if subset of spectrum is to
            be fitted, only relevant if `x_fit_cen` is likewise specified). If
            ``None``, defaults to :attr:`default_fit_range` spectrum attribute.
        init_pars : dict, optional
            Dictionary with initial shape parameter values for fit (optional).

            - If ``None`` (default) the parameters from the shape calibration
              are used (if no shape calibration has been performed yet the
              default parameters defined for mass 100 in the
              :mod:`emgfit.fit_models` module will be used after re-scaling to
              the spectrum's :attr:`mass_number`).
            - If ``'default'``, the default parameters defined for mass 100 in
              the :mod:`emgfit.fit_models` module will be used after re-scaling
              to the spectrum's :attr:`mass_number`.
            - To define custom initial values a parameter dictionary containing
              all model parameters and their values in the format
              ``{'<param name>':<param_value>,...}`` should be passed to
              `init_pars`. Mind that only the initial values to shape parameters
              (`sigma`, `theta`,`etas` and `taus`) can be user-defined. The
              `mu` parameter will be initialized at the peak's :attr:`x_cen`
              attribute and the initial peak amplitude `amp` is automatically
              estimated from the counts at the bin closest to `x_cen`. If a
              varying baseline is used in the fit, the baseline parameter
              `bgd_c` is always initialized at a value of 0.1.

        vary_shape : bool, optional, default: `False`
            If `False` peak-shape parameters (`sigma`, `theta`,`etas` and
            `taus`) are kept fixed at their initial values. If `True` the
            shared shape parameters are varied (ensuring identical shape
            parameters for all peaks).
        vary_baseline : bool, optional, default: `True`
            If `True`, the constant background will be fitted with a varying
            uniform baseline parameter `bkg_c` (initial value: 0.1).
            If `False`, the baseline parameter `bkg_c` will be fixed to 0.
        method : str, optional, default: `'least_squares'`
            Name of minimization algorithm to use. For full list of options
            check arguments of :func:`lmfit.minimizer.minimize`.
        fit_kws : dict, optional, default: None
            Options to pass to lmfit minimizer used in
            :meth:`lmfit.model.Model.fit` method.
        show_plots : bool, optional
            If `True` (default) linear and logarithmic plots of the spectrum
            with the best fit curve are displayed. For details see
            :meth:`spectrum.plot_fit`.
        show_peak_markers : bool, optional
            If `True` (default) peak markers are added to the plots.
        sigmas_of_conf_band : int, optional, default: 0
            Confidence level of confidence band around best fit curve in sigma.
            Note that the confidence band is only derived from the uncertainties
            of the parameters that are varied during the fit.
        error_every : int, optional, default: 1
            Show error bars only for every `error_every`-th data point.
        plot_filename : str, optional, default: None
            If not ``None``, the plots will be saved to two separate files named
            '<`plot_filename`>_log_plot.png' and '<`plot_filename`>_lin_plot.png'.
            **Caution: Existing files with identical name are overwritten.**
        map_par_covar : bool, optional
            If `True` the parameter covariances will be mapped using
            Markov-Chain Monte Carlo (MCMC) sampling and shown in a corner plot.
            This feature is only recommended for single-peak fits.
        **MCMC_kwargs : optional
            Options to send to :meth:`_get_MCMC_par_samples`. Only relevant when
            `map_par_covar` is True.

        Returns
        -------
        :class:`lmfit.model.ModelResult`
            Fit model result.

        See also
        --------
        :meth:`fit_peaks`
        :meth:`fit_calibrant`

        Notes
        -----
        In fits with the ``chi-square`` cost function the variance weights
        :math:`w_i` for the residuals are estimated using the latest model
        predictions: :math:`w_i = 1/(\\sigma_i^2 + \epsilon) = 1/(f(x_i)+ \epsilon)`,
        where :math:`\epsilon = 1e-10` is a small number added to increase
        numerical robustness when :math:`f(x_i)` approaches zero. On each
        iteration the weights are updated with the new values of the model
        function.

        When performing ``MLE`` fits including bins with low statistics the
        value for chi-squared as well as the parameter standard errors and
        correlations in the lmfit fit report should be taken with caution.
        This is because strictly speaking emgfit's ``MLE`` cost function only
        approximates a chi-squared distribution in the limit of a large number
        of counts in every bin ("Wick's theorem"). For a detailed derivation of
        this statement see pp. 94-95 of these `lecture slides by Mark Thompson`_.
        In practice and if needed, one can simply test the validity of the
        reported fit statistic as well as parameter standard errors &
        correlations by re-performing the same fit with `cost_func='chi-square'`
        and comparing the results. In all tested cases decent agreement was
        found even if the fit range contained low-statistics bins. Even if a
        deviation occurs this is irrelevant in most pratical cases since the
        mass errors reported in emgfit's peak properties table are independent
        of the lmfit parameter standard errors given as additional information
        below. Only the peak area errors are by default calculated using the
        standard errors of the `amp` parameters reported by lmfit.

        .. _`lecture slides by Mark Thompson`: https://www.hep.phy.cam.ac.uk/~thomson/lectures/statistics/Fitting_Handout.pdf

        Besides the asymptotic concergence to a chi-squared distribution
        emgfit's ``MLE`` cost function has a second handy property - all
        summands in the log-likelihood ratio are positive semi-definite:
        :math:`L_i = f(x_i) - y_i + y_i ln\\left(\\frac{y_i}{f(x_i)}\\right) \\geq 0`.
        Exploiting this property, the minimization of the log-likelihood ratio
        can be re-formulated into a least-squares problem (see also [#]_):

        .. math::

            L = 2\\sum_i L_i = 2\\sum_i \\left( \\sqrt{ L_i } \\right)^2.


        Instead of minimizing the scalar log-likelihood ratio, emgfit by default
        minimizes the sum-of-squares of the square-roots of the summands
        :math:`L_i` in the log-likelihood ratio. This is mathematically
        equivalent to minimizing :math:`L` and facilitates the
        usage of Scipy's highly efficient least-squares optimizers
        ('least_squares' & 'leastsq'). The latter yield significant speed-ups
        compared to scalar optimizers such as Scipy's 'Nelder-Mead' or 'Powell'
        methods. By default, emgfit's 'MLE' fits are performed with Scipy's
        'least_squares' optimizer, a variant of a Levenberg-Marquardt algorithm
        for bound-constrained problems. If a scalar optimizaton method is
        chosen emgfit uses the conventional approach of minimizing the scalar
        :math:`L`. For more details on these optimizers see the docs of
        :func:`lmfit.minimizer.minimize` and :class:`scipy.optimize`.

        References
        ----------
        .. [#] Ross, G. J. S. "Least squares optimisation of general
           log-likelihood functions and estimation of separable linear
           parameters." COMPSTAT 1982 5th Symposium held at Toulouse 1982.
           Physica, Heidelberg, 1982.

        """
        if x_fit_range is None:
            x_fit_range = self.default_fit_range
        if x_fit_cen:
            x_min = x_fit_cen - 0.5*x_fit_range
            x_max = x_fit_cen + 0.5*x_fit_range
            # Cut data to fit range
            df_fit = self.data[x_min:x_max]
            # Select peaks in fit range
            peaks_to_fit = [peak for peak in self.peaks if (x_min < peak.x_pos < x_max)]
        else:
            df_fit = self.data
            x_min = df_fit.index.values[0]
            x_max = df_fit.index.values[-1]
            x_fit_range = x_max - x_min
            x_fit_cen = 0.5*(x_max + x_min)
            peaks_to_fit = self.peaks
        if len(peaks_to_fit) == 0:
            raise Exception("Fit failed. No peaks in specified mass range.")
        x = df_fit.index.values
        y = df_fit['Counts'].values
        y_err = np.maximum(1,np.sqrt(y)) # assume Poisson (counting) statistics
        # Weights for residuals: residual = (fit_model - y) * weights
        weights = 1./y_err

        if init_pars == 'default':
            # Take default params defined in create_default_init_pars() in
            # fit_models.py and re-scale to spectrum's 'mass_number' attribute
            init_params = fit_models.create_default_init_pars(mass_number=self.mass_number)
        elif init_pars is not None:
            init_params = init_pars
        else:
            # Use shape parameters asociated with spectrum unless other
            # parameters have been specified
            if self.shape_cal_pars is None:
                raise Exception("No shape calibration parameters found. Either "
                                "perform a shape calibration upfront with "
                                "determine_peak_shape() or provide initial "
                                "shape parameter values with the `init_params` "
                                "argument.")
            init_params = self.shape_cal_pars


        if vary_shape == True:
            # Enforce shared shape parameters for all peaks
            index_first_peak = self.peaks.index(peaks_to_fit[0])
        else:
            index_first_peak = None

        model_name = str(fit_model)+' + const. background (bkg_c)'
        # Create multi-peak fit model
        mod = self.comp_model(peaks_to_fit=peaks_to_fit, model=fit_model,
                              init_pars=init_params, vary_shape=vary_shape,
                              vary_baseline=vary_baseline,
                              index_first_peak=index_first_peak)
        pars = mod.make_params() # create parameters object for model

        # Perform fit & store results
        if cost_func == 'chi-square':
            ## Pearson's chi-squared fit with iterative weights 1/Sqrt(f(x_i))
            mod_Pearson = mod
            eps = 1e-10 # small number to bound Pearson weights
            def resid_Pearson_chi_square(pars,y_data,weights,x=x):
                y_m = mod_Pearson.eval(pars,x=x)
                # Calculate weights for current iteration, add tiny number `eps`
                # in denominator for numerical stability
                weights = 1/np.sqrt(y_m + eps)
                return (y_m - y_data)*weights

            # Overwrite lmfit's standard least square residuals with iterative
            # residuals for Pearson chi-square fit
            mod_Pearson._residual = resid_Pearson_chi_square
            out = mod_Pearson.fit(y, params=pars, x=x, weights=weights,
                                  method=method, fit_kws=fit_kws,
                                  scale_covar=False, nan_policy='propagate')
            y_m = out.best_fit
            # Calculate final weights for plotting
            Pearson_weights = 1./np.sqrt(y_m + eps)
            out.y_err = 1./Pearson_weights
        elif cost_func == 'MLE':
            ## Binned max. likelihood fit using negative log-likelihood ratio
            mod_MLE = mod
            # Define sqrt of (doubled) negative log-likelihood ratio (NLLR)
            # summands:
            tiny = np.finfo(float).tiny # get smallest pos. float in numpy
            def sqrt_NLLR(pars,y_data,weights,x=x):
                y_m = mod_MLE.eval(pars,x=x)
                # Add tiniest pos. float representable by numpy to arguments of
                # np.log to smoothly handle divergences for log(arg -> 0)
                NLLR = 2*(y_m-y_data) + 2*y_data*(np.log(y_data+tiny)-np.log(y_m+tiny))
                ret = np.sqrt(NLLR)
                return ret

            # Overwrite lmfit's standard least square residuals with the
            # square-roots of the NLLR summands, this enables usage of scipy's
            # `least_squares` minimizer and yields much faster optimization
            # than with scalar minimizers
            mod_MLE._residual = sqrt_NLLR
            out = mod_MLE.fit(y, params=pars, x=x, weights=weights,
                              method=method, fit_kws=fit_kws, scale_covar=False,
                              calc_covar=False, nan_policy='propagate')
            out.y_err = 1./out.weights
        else:
            raise Exception("Error: Definition of `cost_func` failed!")
        out.x = x
        out.y = y
        out.fit_model = fit_model
        out.cost_func = cost_func
        out.method = method
        out.fit_kws = fit_kws
        out.x_fit_cen = x_fit_cen
        out.x_fit_range = x_fit_range
        out.vary_baseline = vary_baseline
        out.vary_shape = vary_shape

        if map_par_covar:
            self._get_MCMC_par_samples(out, **MCMC_kwargs)

        if show_plots:
            self.plot_fit(fit_result=out, show_peak_markers=show_peak_markers,
                          sigmas_of_conf_band=sigmas_of_conf_band, x_min=x_min,
                          x_max=x_max, error_every=error_every,
                          plot_filename=plot_filename)

        return out


    def calc_peak_area(self, peak_index, fit_result=None, decimals=2):
        """Calculate the peak area (counts in peak) and its stat. uncertainty.

        Area and area error are calculated using the peak's amplitude parameter
        `amp` and the width of the uniform binning of the spectrum. Therefore,
        the peak must have been fitted beforehand. In the case of overlapping
        peaks only the counts within the fit component of the specified peak are
        returned.

        Note
        ----
        This routine assumes the bin width to be uniform across the spectrum.
        The mass binning of most mass spectra is not perfectly uniform
        (usually time bins are uniform such that the width of mass bins has a
        quadratic scaling with mass). However, for isobaric species the
        quadratic term is usually so small that it can safely be neglected.


        Parameters
        ----------
        peak_index : str
            Index of peak of interest.
        fit_result : :class:`lmfit.model.ModelResult`, optional
            Fit result object to use for area calculation. If ``None`` (default)
            use corresponding fit result stored in
            :attr:`~emgfit.spectrum.spectrum.fit_results` list.
        decimals : int
            Number of decimals of returned output values.

        Returns
        -------
        list of float
            List with peak area and area error in format [area, area_error].

        """
        pref = 'p'+str(peak_index)+'_'
        area, area_err = np.nan, np.nan
        if fit_result is None:
            fit_result = self.fit_results[peak_index]
        # get width of mass bins, needed to convert peak amplitude (peak area in
        # units Counts/mass range) to Counts
        bin_width = self.data.index[1] - self.data.index[0]
        try:
            area = fit_result.best_values[pref+'amp']/bin_width
            area = np.round(area,decimals)
            try:
                area_err = fit_result.params[pref+'amp'].stderr/bin_width
                area_err = np.round(area_err,decimals)
            except TypeError as err:
                    import warnings
                    msg = 'Area error determination failed with Type error: '
                    msg += str(getattr(err, 'message', repr(err)))
                    warnings.warn(msg)
        except TypeError or AttributeError:
            msg = str('Area error determination failed. Could not get amplitude '
                      'parameter (`amp`) of peak. Likely the peak has not been '
                      'fitted successfully yet.')
            raise Exception(msg)
        return area, area_err


    def calc_FWHM_emg(self,peak_index,fit_result=None):
        """Calculate the full width at half maximum (FWHM) of a Hyper-EMG fit.

        The peak of interest must have previously been fitted with a Hyper-EMG
        model.

        Parameters
        ----------
        peak_index : int
            Index of peak of interest.
        fit_result : :class:`lmfit.model.ModelResult`, optional
            Fit result containing peak of interest. If ``None`` (default) the
            corresponding fit result from the spectrum's :attr:`fit_results`
            list will be fetched.

        Returns
        -------
        float
            Full width at half maximum of Hyper-EMG fit of peak of interest.

        """
        if fit_result is None and self.fit_results[peak_index] is not None:
            fit_result = self.fit_results[peak_index]
        elif fit_result is None:
            raise Exception("No matching fit result found in `fit_results` "
                            "list. Ensure the peak has been fitted.")

        pars = fit_result.params
        pref = 'p{0}_'.format(peak_index)
        mu = pars[pref+'mu'] # centroid of underlying Gaussian
        sigma = pars[pref+'sigma'] # sigma of underlying Gaussian
        x_range = sigma*30
        x = np.linspace(mu - 0.5*x_range, mu + 0.5*x_range,10000)
        comps = fit_result.eval_components(x=x)
        y = comps[pref] #fit_result.eval(pars,x=x)
        y_M = max(y)
        i_M = np.argmin(np.abs(y-y_M))
        y_HM = 0.5*y_M
        i_HM1 = np.argmin(np.abs(y[0:i_M]-y_HM))
        i_HM2 = i_M + np.argmin(np.abs(y[i_M:]-y_HM))
        if i_HM1 == 0 or i_HM2 == len(x):
            msg = str("FWHM points at boundary, likely a larger `x_range` "
                      "needs to be hardcoded into this method.")
            raise Exception(msg)
        FWHM = x[i_HM2] - x[i_HM1]

        return FWHM


    def calc_sigma_emg(self,peak_index,fit_result=None):
        """Calculate the standard deviation of a Hyper-EMG peak fit.

        The peak of interest must have previously been fitted with a Hyper-EMG
        model.

        Parameters
        ----------
        peak_index : int
            Index of peak of interest.
        fit_result : :class:`lmfit.model.ModelResult`, optional
            Fit result containing peak of interest. If ``None`` (default) the
            corresponding fit result from the spectrum's :attr:`fit_results`
            list will be fetched.

        Returns
        -------
        float
            Standard deviation of Hyper-EMG fit of peak of interest.

        """
        if fit_result is None and self.fit_results[peak_index] is not None:
            fit_result = self.fit_results[peak_index]
        elif fit_result is None:
            raise Exception("No matching fit result found in `fit_results` "
                            "list. Ensure the peak has been fitted.")

        pref = 'p{0}_'.format(peak_index)
        no_left_tails = int(fit_result.fit_model[3])
        no_right_tails = int(fit_result.fit_model[4])
        li_eta_m, li_tau_m, li_eta_p, li_tau_p = [],[],[],[]
        for i in np.arange(1,no_left_tails+1):
            if no_left_tails == 1:
                li_eta_m = [1]
            else:
                li_eta_m.append(fit_result.best_values[pref+'eta_m'+str(i)])
            li_tau_m.append(fit_result.best_values[pref+'tau_m'+str(i)])
        for i in np.arange(1,no_right_tails+1):
            if no_right_tails == 1:
                li_eta_p = [1]
            else:
                li_eta_p.append(fit_result.best_values[pref+'eta_p'+str(i)])
            li_tau_p.append(fit_result.best_values[pref+'tau_p'+str(i)])
        sigma_EMG = emg_funcs.sigma_emg(fit_result.best_values[pref+'sigma'],
                                        fit_result.best_values[pref+'theta'],
                                        tuple(li_eta_m),tuple(li_tau_m),
                                        tuple(li_eta_p),tuple(li_tau_p) )

        return sigma_EMG


    @staticmethod
    def bootstrap_spectrum(df,N_events=None,x_cen=None,x_range=0.02):
        """Create simulated spectrum via bootstrap resampling from dataset `df`.

        Parameters
        ----------
        df : class:`pandas.DataFrame`
            Original histogrammed spectrum data to re-sample from.
        N_events : int, optional
            Number of events to create via bootstrap re-sampling, defaults to
            number of events in original DataFrame `df`.
        x_cen : float [u], optional
            Center of mass range to re-sample from. If ``None``, re-sample from
            full mass range of input data `df`.
        x_range : float [u], optional
            Width of mass range to re-sample from. Defaults to 0.02 u. `x_range`
            is only relevant if a `x_cen` argument is specified.

        Returns
        -------
        :class:`pandas.DataFrame`
            Histogram with simulated spectrum data from non-parametric
            bootstrap.

        """
        if x_cen:
            x_min = x_cen - 0.5*x_range
            x_max = x_cen + 0.5*x_range
            df = df[x_min:x_max]
        mass_bins = df.index.values
        counts = df['Counts'].values.astype(int)

        # Convert histogrammed spectrum (equivalent to MAc HIST export mode) to
        # list of events (equivalent to MAc LIST export mode)
        orig_events =  np.repeat(mass_bins, counts, axis=0)

        # Create new DataFrame of events by bootstrapping from `orig_events`
        if N_events == None:
            N_events = len(orig_events)
        random_indeces = np.random.randint(0, len(orig_events), N_events)
        events = orig_events[random_indeces]
        df_events = pd.DataFrame(events)

        # Convert list of events back to a DataFrame with histogram data
        bin_centers = df.index.values
        bin_width = df.index.values[1] - df.index.values[0]
        bin_edges = np.append(bin_centers-0.5*bin_width,
                              bin_centers[-1]+0.5*bin_width)
        hist = np.histogram(df_events, bins=bin_edges)
        df_new = pd.DataFrame(data=hist[0], index=bin_centers, dtype=float,
                              columns=["Counts"])
        df_new.index.name = "Mass [u]"
        return df_new


    def determine_A_stat_emg(self, peak_index=None, species="?", x_pos=None,
                             x_range=None, N_spectra=1000, fit_model=None,
                             cost_func='MLE', method='least_squares',
                             fit_kws=None, vary_baseline=True,
                             plot_filename=None):
        """Determine the constant of proprotionality `A_stat_emg` for
        calculation of the statistical uncertainties of Hyper-EMG fits.

        This method updates the :attr:`A_stat_emg` & :attr:`A_stat_emg_error`
        spectrum attributes. The former will be used for all subsequent stat.
        error estimations.

        **This routine must be called AFTER a successful peak-shape calibration
        and should be called BEFORE the mass re-calibration.**

        `A_stat_emg` is determined by evaluating the statistical fluctuations of
        a representative peak's centroid as a function of the number of ions in
        the reference peak. The fluctuations are estimated by fitting
        a large number of synthetic spectra derived from the experimental
        data via bootstrap re-sampling. For details see `Notes` section below.

        Specify the peak to use for the bootstrap re-sampling by providing
        **either** of the `peak_index`, `species` and `x_pos` arguments. The
        peak should be well-separated and have decent statistics (typically the
        peak-shape calibrant is used).

        Parameters
        ----------
        peak_index : int, optional
            Index of representative peak to use for bootstrap re-sampling
            (typically, the peak-shape calibrant). The peak should have high
            statistics and must be well-separated from other peaks.
        species : str, optional
            String with species name of representative peak to use for bootstrap
            re-sampling (typically, the peak-shape calibrant). The peak should
            have high statistics and be well-separated from other peaks.
        x_pos : float [u], optional
            Marker position (:attr:`x_pos` spectrum attribute) of representative
            peak to use for bootstrap re-sampling (typically, the peak-shape
            calibrant). The peak should have high statistics and be well-
            separated from other peaks. `x_pos` must be specified up to the 6th
            decimal.
        x_range : float [u], optional
            Mass range around peak centroid over which events will be sampled
            and fitted. **Choose such that no secondary peaks are contained in
            the mass range!** If ``None`` defaults to :attr:`default_fit_range`
            spectrum attribute.
        N_spectra : int, optional, default: 1000
            Number of bootstrapped spectra to create at each number of ions.
        cost_func : str, optional, default: 'MLE'
            Name of cost function to use for minimization.

            - If ``'chi-square'``, the fit is performed by minimizing Pearson's
              chi-squared statistic:

              .. math::

                  \\chi^2_P = \\sum_i \\frac{(f(x_i) - y_i)^2}{f(x_i)}.

            - If ``'MLE'``, a binned maximum likelihood estimation is performed
              by minimizing the (doubled) negative log-likelihood ratio:

              .. math::

                  L = 2\\sum_i \\left[ f(x_i) - y_i + y_i ln\\left(\\frac{y_i}{f(x_i)}\\right)\\right]

            For details see `Notes` section of :meth:`peakfit` method documentation.
        method : str, optional, default: `'least_squares'`
            Name of minimization algorithm to use. For full list of options
            check arguments of :func:`lmfit.minimizer.minimize`.
        fit_kws : dict, optional, default: None
            Options to pass to lmfit minimizer used in
            :meth:`lmfit.model.Model.fit` method.
        vary_baseline : bool, optional, default: `True`
            If `True`, the constant background will be fitted with a varying
            uniform baseline parameter `bkg_c` (initial value: 0.1).
            If `False`, the baseline parameter `bkg_c` will be fixed to 0.
        plot_filename : str, optional, default: None
            If not ``None``, the plots will be saved to two separate files named
            '<`plot_filename`>_log_plot.png' and '<`plot_filename`>_lin_plot.png'.
            **Caution: Existing files with identical name are overwritten.**

        Notes
        -----
        As noted in [#]_, statistical errors of Hyper-EMG peak centroids obey
        the following scaling with the number of counts in the peak `N_counts`:

        .. math::  \\sigma_{stat} = A_{stat,emg} \\frac{FWHM}{\\sqrt{N_{counts}}},

        where the constant of proportionality `A_stat_emg` depends on the
        specific peak shape. This routine uses the following method to determine
        `A_stat_emg`:

        - `N_spectra` bootstrapped spectra are created for each of the following
          total numbers of events: [10,30,100,300,1000,3000,10000,30000].
        - Each bootstrapped spectrum is fitted and the best fit peak centroids
          are recorded.
        - The statistical uncertainties are estimated by taking the sample
          standard deviations of the recorded peak centroids at each value of
          `N_counts`. Since the best-fit peak area can deviate from the true
          number of re-sampled events in the spectrum, the mean best_fit area at
          each number of re-sampled events is used to determine `N_counts`.
        - `A_stat_emg` is finally determined by plotting the rel. statistical
          uncertainty as function of `N_counts` and fitting it with the above
          equation.

        The resulting value for `A_stat_emg` will be stored as spectrum
        attribute and will be used in all subsequent fits to calculate the stat.
        errors from the number of counts in the peak.

        References
        ----------
        .. [#] San Andrés, Samuel Ayet, et al. "High-resolution, accurate
           multiple-reflection time-of-flight mass spectrometry for short-lived,
           exotic nuclei of a few events in their ground and low-lying isomeric
           states." Physical Review C 99.6 (2019): 064313.

        """
        if peak_index is not None:
            pass
        elif species != "?":
            peak_index = [i for i in range(len(self.peaks)) if species == self.peaks[i].species][0] # select peak with species label 'species'
        elif x_pos is not None:
            peak_index = [i for i in range(len(self.peaks)) if np.round(x_pos,6) == np.round(self.peaks[i].x_pos,6)][0] # select peak at position 'x_pos'
        else:
            raise Exception("Peak specification failed. Check function"
                            "documentation for details on peak selection.\n")
        if fit_model is None:
            fit_model = self.fit_model
        if x_range is None:
            x_range = self.default_fit_range
        x_cen = self.peaks[peak_index].x_pos
        no_peaks_in_range = len([p for p in self.peaks if (x_cen - 0.5*x_range) <= p.x_pos <= (x_cen + 0.5*x_range)])
        if no_peaks_in_range > 1:
            raise Exception("More than one peak in current mass range. "
                            "This routine only works on well-separated, single "
                            "peaks. Please chose a smaller `x_range`!\n")
        li_N_counts = [10,30,100,300,1000,3000,10000,30000]
        print("Creating synthetic spectra via bootstrap re-sampling and "
              "fitting them for A_stat determination.")
        print("Depending on the choice of `N_spectra` this can take a few "
              "minutes. Interrupt kernel if this takes too long.")
        np.random.seed(seed=34) # to make bootstrapped spectra reproducible
        std_devs_of_mus = np.array([]) # standard deviation of sample means mu
        mean_areas = np.array([]) # array for numbers of detected counts
        from tqdm.auto import tqdm # add progress bar with tqdm
        t = tqdm(total=len(li_N_counts)*N_spectra)
        for N_counts in li_N_counts:
            mus = np.array([])
            areas = np.array([])

            for i in range(N_spectra):
                # create boostrapped spectrum data
                df_boot = spectrum.bootstrap_spectrum(self.data,
                                                      N_events=N_counts,
                                                      x_cen=x_cen,
                                                      x_range=x_range)
                # create boostrapped spectrum object
                spec_boot = spectrum(None,df=df_boot,show_plot=False)
                spec_boot.add_peak(x_cen,verbose=False)
                # fit boostrapped spectrum with model and (fixed) shape
                # parameters from peak-shape calibration
                try:
                    fit_result = spec_boot.peakfit(fit_model=self.fit_model,
                                                   x_fit_cen=x_cen,
                                                   x_fit_range=x_range,
                                                   cost_func=cost_func,
                                                   method=method,
                                                   fit_kws=fit_kws,
                                                   vary_baseline=vary_baseline,
                                                   init_pars=self.shape_cal_pars,
                                                   show_plots=False)
                    # Record centroid and area of peak 0
                    mus = np.append(mus,fit_result.params['p0_mu'].value)
                    area_i = spec_boot.calc_peak_area(0, fit_result=fit_result,
                                                      decimals=2)[0]
                    areas = np.append(areas,area_i)
                except ValueError:
                    print("Fit #{1} for N_counts = {0} failed with ValueError "
                          "(likely NaNs in y-model array).".format(N_counts,i))
                t.update()
            std_devs_of_mus = np.append(std_devs_of_mus,np.std(mus,ddof=1))
            mean_areas = np.append(mean_areas,np.mean(areas))

        t.close()
        mean_mu = np.mean(mus) # from last `N_counts` step only
        FWHM_gauss = 2*np.sqrt(2*np.log(2))*fit_result.params['p0_sigma'].value
        FWHM_emg = spec_boot.calc_FWHM_emg(peak_index=0,fit_result=fit_result)
        FWHM_emg_err = FWHM_gauss/FWHM_emg * self.shape_cal_errors['sigma']
        print("Done!\n")

        # Use no. of detected counts instead of true no. of re-sampling
        # events (i.e. li_N_counts) as x values
        x = mean_areas
        model = fit.models.PowerLawModel()
        pars = model.make_params()
        pars['exponent'].value = -0.5
        pars['exponent'].vary = False
        weights = np.sqrt(li_N_counts)
        out = model.fit(std_devs_of_mus,x=x,params=pars,weights=weights)
        print(out.fit_report())

        A_stat_gauss = 1/(2*np.sqrt(2*np.log(2)))
        A_stat_emg = out.best_values['amplitude']/FWHM_emg
        A_stat_emg_error = np.sqrt( (out.params['amplitude'].stderr/FWHM_emg)**2
                                   +(out.best_values['amplitude']*FWHM_emg_err/FWHM_emg**2)**2 )

        y = std_devs_of_mus/mean_mu
        f = plt.figure(figsize=(11,6), dpi=dpi)
        plt.title('A_stat_emg determination from bootstrapped spectra - '+
                  fit_model+' '+cost_func+' fits', fontdict = {'fontsize' : 14})
        plt.plot(x,y,'o',markersize=msize)
        plt.plot(x,out.best_fit/mean_mu)
        plt.plot(x,A_stat_gauss*FWHM_gauss/(np.sqrt(x)*mean_mu),'--')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Peak area [counts]",fontsize=14)
        plt.ylabel("Relative statistical uncertainty",fontsize=14)
        plt.legend(["Standard deviations of sample means",
                    "Stat. error of Hyper-EMG",
                    "Stat. error of underlying Gaussian"])
        plt.annotate('A_stat_emg: '+str(np.round(A_stat_emg,3))+' +- '+str(
                     np.round(A_stat_emg_error,3)), xy=(0.65, 0.75),
                     xycoords='axes fraction')
        if plot_filename is not None:
            try:
                plt.savefig(plot_filename+'_A_stat_emg_determination.png',
                            dpi=600)
            except:
                raise
        plt.show()

        self.determined_A_stat_emg = cost_func
        self.A_stat_emg = A_stat_emg
        self.A_stat_emg_error = A_stat_emg_error
        print("A_stat of a Gaussian model:",np.round(A_stat_gauss,3))
        print("Default A_stat_emg for Hyper-EMG models:",
              np.round(A_stat_emg_default,3))
        print("A_stat_emg for this spectrum's",self.fit_model,"fit model:",
             np.round(self.A_stat_emg,3),"+-",np.round(self.A_stat_emg_error,3))


    def determine_peak_shape(self, index_shape_calib=None,
                             species_shape_calib=None, fit_model='emg22',
                             cost_func='chi-square', init_pars = 'default',
                             x_fit_cen=None, x_fit_range=None,
                             vary_baseline=True, method='least_squares',
                             fit_kws=None, vary_tail_order=True,
                             show_fit_reports=False, show_plots=True,
                             show_peak_markers=True, sigmas_of_conf_band=0,
                             error_every=1, plot_filename=None,
                             map_par_covar=False, **MCMC_kwargs):
        """Determine optimal peak-shape parameters by fitting the specified
        peak-shape calibrant.

        If `vary_tail_order` is `True` (default) an automatic model selection
        is performed before the calibration of the peak-shape parameters.

        It is recommended to visually check whether the fit residuals
        are purely stochastic (as should be the case for a decent model). If
        this is not the case either the selected model does not describe the
        data well, the initial parameters lead to poor convergence or there are
        additional undetected peaks.

        Parameters
        ----------
        index_shape_calib : int, optional
            Index of shape-calibration peak. Preferrable alternative: Specify
            the shape-calibrant with the `species_shape_calib` argument.
        species_shape_calib : str, optional
            Species name of the shape-calibrant peak in :ref:`:-notation` (e.g.
            ``'K39:-1e'``). Alternatively, the peak to use can be specified with
            the `index_shape_calib` argument.
        fit_model : str, optional, default: 'emg22'
            Name of fit model to use for shape calibration (e.g. ``'Gaussian'``,
            ``'emg12'``, ``'emg33'``, ... - for full list see
            :ref:`fit_model_list`). If the automatic model selection
            (`vary_tail_order=True`) fails or is turned off, `fit_model` will be
            used for the shape calibration and set as the spectrum's
            :attr:`fit_model` attribute.
        cost_func : str, optional, default: 'chi-square'
            Name of cost function to use for minimization. **It is strongly
            recommended to use 'chi-square'-fitting for the peak-shape
            determination** since this yields more robust results for fits with
            many model parameters as well as more trustworthy parameter
            uncertainties (important for peak-shape error determinations).

            - If ``'chi-square'``, the fit is performed by minimizing Pearson's
              chi-squared statistic:

              .. math::

                  \\chi^2_P = \\sum_i \\frac{(f(x_i) - y_i)^2}{f(x_i)}.

            - If ``'MLE'``, a binned maximum likelihood estimation is performed
              by minimizing the (doubled) negative log likelihood ratio:

              .. math::

                  L = 2\\sum_i \\left[ f(x_i) - y_i + y_i ln\\left(\\frac{y_i}{f(x_i)}\\right)\\right]

            For details see `Notes` section of :meth:`peakfit` method documentation.
        init_pars : dict, optional
            Dictionary with initial shape parameter values for fit (optional).

            - If ``None`` or ``'default'`` (default), the default parameters
              defined for mass 100 in the :mod:`emgfit.fit_models` module will
              be used after re-scaling to the spectrum's :attr:`mass_number`.
            - To define custom initial values, a parameter dictionary containing
              all model parameters and their values in the format
              ``{'<param name>':<param_value>,...}`` should be passed to
              `init_pars`.

          Mind that only the initial values to shape parameters
          (`sigma`, `theta`,`etas` and `taus`) can be user-defined. The
          `mu` parameter will be initialized at the peak's :attr:`x_cen`
          attribute and the initial peak amplitude `amp` is automatically
          estimated from the counts at the bin closest to `x_cen`. If a
          varying baseline is used in the fit, the baseline parameter
          `bgd_c` is always initialized at a value of 0.1.

        x_fit_cen : float [u], optional
            Center of fit range. If ``None`` (default), the :attr:`x_pos`
            attribute of the shape-calibrant peak is used as `x_fit_cen`.
        x_fit_range : float [u], optional
            Mass range to fit. If ``None``, defaults to the
            :attr:`default_fit_range` spectrum attribute.
        vary_baseline : bool, optional, default: `True`
            If `True`, the background will be fitted with a varying uniform
            baseline parameter `bkg_c` (initial value: 0.1). If `False`, the
            baseline parameter `bkg_c` will be fixed to 0.
        method : str, optional, default: `'least_squares'`
            Name of minimization algorithm to use. For full list of options
            check arguments of :func:`lmfit.minimizer.minimize`.
        fit_kws : dict, optional, default: None
            Options to pass to lmfit minimizer used in
            :meth:`lmfit.model.Model.fit` method.
        vary_tail_order : bool, optional
            If `True` (default), before the calibration of the peak-shape
            parameters an automatized fit model selection is performed. For
            details on the automatic model selection, see `Notes` section below.
            If `False`, the specified `fit_model` argument is used as model
            for the peak-shape determination.
        show_fit_reports : bool, optional, default: True
            Whether to print fit reports for the fits in the automatic model
            selection.
        show_plots : bool, optional
            If `True` (default), linear and logarithmic plots of the spectrum
            and the best fit curve are displayed. For details see
            :meth:`spectrum.plot_fit`.
        show_peak_markers : bool, optional
            If `True` (default), peak markers are added to the plots.
        sigmas_of_conf_band : int, optional, default: 0
            Confidence level of confidence band around best fit curve in sigma.
        error_every : int, optional, default: 1
            Show error bars only for every `error_every`-th data point.
        plot_filename : str, optional, default: None
            If not ``None``, the plots of the shape-calibration will be saved to
            two separate files named '<`plot_filename`>_log_plot.png' and
            '<`plot_filename`>_lin_plot.png'. **Caution: Existing files with
            identical name are overwritten.**
        map_par_covar : bool, optional
            If `True` the parameter covariances will be mapped using
            Markov-Chain Monte Carlo (MCMC) sampling and shown in a corner plot.
            This feature is only recommended for single-peak fits.
        **MCMC_kwargs : optional
            Options to send to :meth:`_get_MCMC_par_samples`. Only relevant when
            `map_par_covar` is True.

        Notes
        -----
        Ideally the peak-shape calibration is performed on a well-separated peak
        with high statistics. If this is not possible, the peak-shape
        calibration can also be attempted using overlapping peaks since emgfit
        ensures shared and identical shape parameters for all peaks in a multi-
        peak fit.

        Automatic model selection:
        When the model selection is activated the routine tries to find the peak
        shape that minimizes the fit's chi-squared reduced by successively
        adding more tails on the right and left. Finally, that fit model is
        selected which yields the lowest chi-squared reduced without having any
        of the tail weight parameters `eta` compatible with zero within 1-sigma
        uncertainty. The latter models are excluded as is this an indication of
        overfitting. Models for which the calculation of any `eta` parameter
        uncertainty fails are likewise excluded from selection.

        """
        if index_shape_calib is not None and (species_shape_calib is None):
            peak = self.peaks[index_shape_calib]
        elif species_shape_calib:
            index_shape_calib = [i for i in range(len(self.peaks)) if
                                 species_shape_calib == self.peaks[i].species][0]
            peak = self.peaks[index_shape_calib]
        else:
            raise Exception("Definition of peak shape calibrant failed. Define "
                            "EITHER the index OR the species name of the peak "
                            "to use as shape calibrant! ")
        if init_pars == 'default' or init_pars is None:
            # Take default params defined in create_default_init_pars() in
            # fit_models.py and re-scale to spectrum's 'mass_number' attribute
            init_params = fit_models.create_default_init_pars(mass_number=self.mass_number)
        elif init_pars is not None: # take user-defined values
            init_params = init_pars
        else:
            raise Exception("Definition of initial parameters failed.")
        if x_fit_cen is None:
            x_fit_cen = peak.x_pos
        if x_fit_range is None:
            x_fit_range = self.default_fit_range

        if vary_tail_order == True and fit_model != 'Gaussian':
            print('\n##### Determine optimal tail order #####\n')
            # Fit peak with Hyper-EMG of increasingly higher tail orders and compile results
            # use fit model that produces the lowest chi-square without having eta's compatible with zero within errobar
            li_fit_models = ['Gaussian','emg01','emg10','emg11','emg12','emg21',
                             'emg22','emg23','emg32','emg33']
            li_red_chis = np.array([np.nan]*len(li_fit_models))
            li_red_chi_errs = np.array([np.nan]*len(li_fit_models))
            # Prepare list of flags for excluding models with tail parameters
            # compatible with zero within error or with failed error estimation:
            li_flags =np.array([False]*len(li_fit_models))
            for model in li_fit_models:
                try:
                    print("\n### Fitting data with",model,"###---------------------------------------------------------------------------------------------\n")
                    out = spectrum.peakfit(self, fit_model=model, cost_func=cost_func,
                                           x_fit_cen=x_fit_cen, x_fit_range=x_fit_range,
                                           init_pars=init_pars, vary_shape=True,
                                           vary_baseline=vary_baseline, method=method,
                                           fit_kws=fit_kws,
                                           show_plots=show_plots,
                                           show_peak_markers=show_peak_markers,
                                           sigmas_of_conf_band=sigmas_of_conf_band,
                                           error_every=error_every)
                    idx = li_fit_models.index(model)
                    li_red_chis[idx] = np.round(out.redchi,2)
                    li_red_chi_errs[idx] =  np.round(np.sqrt(2/out.nfree),2)

                    # Check emg models with tail orders >= 2 for overfitting
                    # (i.e. an eta or tau parameter agress with zero within 1-sigma)
                    # and check for existence of parameter uncertainties
                    if not out.errorbars:
                        print("WARNING: Could not get parameter uncertainties "
                              "from covariance matrix! This tail order will be "
                              "excluded from selection. ") # TODO: Consider adding conf_interval() option here.
                        # Mark model in order to exclude it below
                        li_flags[idx] = True
                    elif model.startswith('emg') and model not in ['emg01','emg10','emg11']:
                        no_left_tails = int(model[3])
                        no_right_tails = int(model[4])
                        # Must use first peak to be fit, since only its shape
                        # parameters are all unconstrained:
                        first_parname = list(out.params.keys())[2]
                        pref = first_parname.split('_')[0]+'_'
                        if no_left_tails > 1:
                            for i in np.arange(1,no_left_tails+1):
                                par_name = pref+"eta_m"+str(i)
                                val = out.params[par_name].value
                                err = out.params[par_name].stderr
                                if val < err:
                                    print("WARNING: {:10} = {:.3f} +- {:.3f} is compatible with zero within uncertainty.".format(par_name,val,err))
                                    li_flags[idx] = True # mark for exclusion
                                par_name = pref+"tau_m"+str(i)
                                val = out.params[par_name].value
                                err = out.params[par_name].stderr
                                if val < err:
                                    print("WARNING: {:10} = {:.1e} +- {:.1e} is compatible with zero within uncertainty.".format(par_name,val,err))
                                    li_flags[idx] = True # mark for exclusion
                        if no_right_tails > 1:
                            for i in np.arange(1,no_right_tails+1):
                                par_name = pref+"eta_p"+str(i)
                                val = out.params[par_name].value
                                err = out.params[par_name].stderr
                                if val < err:
                                    print("WARNING: {:10} = {:.3f} +- {:.3f} is compatible with zero within uncertainty.".format(par_name,val,err))
                                    li_flags[idx] = True  # mark for exclusion
                                par_name = pref+"tau_p"+str(i)
                                val = out.params[par_name].value
                                err = out.params[par_name].stderr
                                if val < err:
                                    print("WARNING: {:10} = {:.1e} +- {:.1e} is compatible with zero within uncertainty.".format(par_name,val,err))
                                    li_flags[idx] = True  # mark for exclusion
                        if li_flags[idx]:
                            print("             This tail order is likely overfitting the data and will be excluded from selection.")
                    print("\n"+str(model)+"-fit yields reduced chi-square of: "+str(li_red_chis[idx])+" +- "+str(li_red_chi_errs[idx]))
                    print()
                    if show_fit_reports:
                        self._show_blinded_report(out) # show fit report
                except ValueError:
                    print('\nWARNING:',model+'-fit failed due to NaN-values and was skipped! ----------------------------------------------\n')

            # Select best model, models with eta_flag == True are excluded
            idx_best_model = np.nanargmin(np.where(li_flags, np.inf, li_red_chis))
            best_model = li_fit_models[idx_best_model]
            self.fit_model = best_model
            print('\n##### RESULT OF AUTOMATIC MODEL SELECTION: #####\n')
            print('    Best fit model determined to be:  {}'.format(best_model))
            print('    Corresponding chi²-reduced:  {:1.2f} +- {:1.2f}\n'.format(
                   li_red_chis[idx_best_model], li_red_chi_errs[idx_best_model]))
        elif not vary_tail_order:
            self.fit_model = fit_model

        print('\n##### Peak-shape determination #####-------------------------------------------------------------------------------------------')
        # Perform fit
        out = spectrum.peakfit(self, fit_model=self.fit_model, cost_func=cost_func,
                               x_fit_cen=x_fit_cen, x_fit_range=x_fit_range,
                               init_pars=init_pars, vary_shape=True,
                               vary_baseline=vary_baseline, method=method,
                               fit_kws=fit_kws, show_plots=show_plots,
                               show_peak_markers=show_peak_markers,
                               sigmas_of_conf_band=sigmas_of_conf_band,
                               error_every=error_every,
                               plot_filename=plot_filename,
                               map_par_covar=map_par_covar, **MCMC_kwargs)

        # Set shape calibrant flag and store shape calibration results in
        # spectrum attributes
        self.index_mass_calib = None # reset mass calibrant flag
        for p in self.peaks: # reset 'shape calibrant' and 'mass calibrant' comment flags
            if 'shape & mass calibrant' in p.comment :
                p.comment = p.comment.replace('shape & mass calibrant','')
            elif p.comment == 'shape calibrant':
                p.comment = '-'
            elif 'shape calibrant' in p.comment:
                p.comment = p.comment.replace('shape calibrant','')
            elif p.comment == 'mass calibrant':
                p.comment = '-'
            elif 'mass calibrant' in p.comment:
                p.comment = p.comment.replace('mass calibrant','')
        if peak.comment == '-' or peak.comment == '' or peak.comment is None:
            peak.comment = 'shape calibrant'
        elif 'shape calibrant' not in peak.comment:
            peak.comment = 'shape calibrant, '+peak.comment
        self._show_blinded_report(out) # show fit report
        self.index_shape_calib = index_shape_calib
        self.red_chi_shape_cal = np.round(out.redchi,2)
        dict_pars = out.params.valuesdict()
        self.shape_cal_result = out # save fit result
        self.shape_cal_pars = {key.lstrip('p'+str(index_shape_calib)+'_'): val
                               for key, val in dict_pars.items()
                               if key.startswith('p'+str(index_shape_calib))}
        self.shape_cal_pars['bkg_c'] = dict_pars['bkg_c']
        self.shape_cal_errors = {} # dict for shape calibration parameter errors
        for par in out.params:
            if par.startswith('p'+str(index_shape_calib)):
                self.shape_cal_errors[par.lstrip('p'+str(index_shape_calib)+'_')] = out.params[par].stderr
        self.shape_cal_errors['bkg_c'] = out.params['bkg_c'].stderr
        self.fit_range_shape_cal = x_fit_range

        # Save thinned and flattened MCMC chain for MC peakshape evaluation
        if map_par_covar is True:
            self.MCMC_par_samples = out.flatchain


    def save_peak_shape_cal(self,filename):
        """Save peak shape parameters to a TXT-file.

        Parameters
        ----------
        filename : str
            Name of output file ('.txt' extension is automatically appended).

        """
        df1 = pd.DataFrame.from_dict(self.shape_cal_pars, orient='index',
                                     columns=['Value'])
        df1.index.rename('Model: '+str(self.fit_model),inplace=True)
        df2 = pd.DataFrame.from_dict(self.shape_cal_errors,orient='index',
                                     columns=['Error'])
        df = df1.join(df2)
        df.to_csv(str(filename)+'.txt', index=True,sep='\t')
        print('\nPeak-shape calibration saved to file: '+str(filename)+'.txt')


    def load_peak_shape_cal(self,filename):
        """Load peak shape from the TXT-file named 'filename.txt'.

        Successfully loaded shape calibration parameters and their uncertainties
        are used as the new :attr:`shape_cal_pars` and
        :attr:`shape_cal_errors` spectrum attributes respectively.


        Parameters
        ----------
        filename : str
            Name of input file ('.txt' extension is automatically appended).

        """
        df = pd.read_csv(str(filename)+'.txt',index_col=0,sep='\t')
        self.fit_model = df.index.name[7:]
        df_val = df['Value']
        df_err = df['Error']
        self.shape_cal_pars = df_val.to_dict()
        self.shape_cal_errors = df_err.to_dict()
        print('\nLoaded peak shape calibration from '+str(filename)+'.txt')


    def _eval_peakshape_errors(self, peak_indeces=[], fit_result=None,
                               verbose=False, show_shape_err_fits=False):
        """Calculate the relative peak-shape uncertainty of the specified peaks.

        **This internal method is automatically called by the :meth:`fit_peaks`
        and :meth:`fit_calibrant` methods and does not need to be run directly
        by the user.**

        The peak-shape uncertainties are obtained by re-fitting the specified
        peaks with each shape parameter individually varied by plus and minus 1
        sigma and recording the respective shift of the peak centroids w.r.t the
        original fit. From the shifted IOI centroids and the corresponding
        shifts of the calibrant centroid effective mass shifts are determined.
        For each varied parameter, the larger of the two eff. mass shifts are
        then added in quadrature to obtain the total peak-shape uncertainty.
        See `Notes` section below for a detailed explanation of the peak-shape
        error evaluation scheme.

        Note: All peaks in the specified `peak_indeces` list must
        have been fitted in the same multi-peak fit (and hence have the same
        lmfit modelresult `fit_result`)!

        This routine does not yield a peak-shape error for the mass calibrant,
        since this is zero by definition. Instead, for the mass calibrant the
        absolute shifts of the peak centroid are calculated and stored in the
        :attr:`eff_mass_shifts_pm` and :attr:`eff_mass_shifts` dictionaries.

        Parameters
        ----------
        peak_indeces : list
            List containing indeces of peaks to evaluate peak-shape uncertainty
            for, e.g. to evaluate peak-shape error of peaks 0 and 3 use
            ``peak_indeces=[0,3]``.
        fit_result : :class:`lmfit.model.ModelResult`, optional
            Fit result object to evaluate peak-shape error for.
        verbose : bool, optional, default: `False`
            If `True`, print all individual eff. mass shifts obtained by
            varying the shape parameters.
        show_shape_err_fits : bool, optional, default: `False`
            If `True`, show individual plots of re-fits for peak-shape error
            determination.

        See also
        --------
        :meth:`get_MC_peakshape_errors`

        Notes
        -----
        `sigma`,`theta`, all `eta` and all `tau` model parameters are considered
        "shape parameters" and varied by plus and minus one standard deviation
        in the peak-shape uncertainty evaluation. The peak amplitude, centroids
        and the baseline are always freely varying.

        The "peak-shape uncertainty" refers to the mass uncertainty due to
        uncertainties in the determination of the peak-shape parameters and due
        to deviations between the shape-calibrant and IOI peak shapes.
        Simply put, the peak-shape uncertainties are estimated by evaluating how
        much a given peak centroid is shifted when the shape parameters are
        varied by plus or minus their 1-sigma uncertainty. A peculiarity of
        emgfit's peak-shape error estimation routine is that only the centroid
        shifts **relative to the calibrant** are taken into account (hence
        '**effective** mass shifts').

        Inspired by the approach outlined in [#]_, the peak-shape uncertainties
        are obtained via the following procedure:

        - Since only effective mass shifts corrected for the corresponding
          shifts of the calibrant peak enter the peak-shape uncertainty,
          at first, the absolute centroid shifts of the mass calibrant must be
          evaluated. There are two options for this:

          - If the calibrant index is included in the `peak_indeces` argument,
            the original calibrant fit is re-performed with each shape parameter
            varied by plus and minus its 1-sigma confidence respectively while
            all other shape parameters are kept fixed at the original best-fit
            values. The resulting absolute "calibrant centroid shifts" are
            recorded and stored in the spectrum's :attr:`eff_mass_shifts_pm`
            dictionary. The shifted calibrant centroids are further used to
            calculate updated mass re-calibration factors. These are stored in
            the :attr:`recal_facs_pm` dictionary. Only the larger of the two
            centroid shifts due to the +/-1-sigma variation of each shape
            parameter are stored in the spectrum's :attr:`eff_mass_shifts`
            dictionary.
          - If the calibrant is not included in the `peak_indeces` list, the
            calibrant centroid shifts and the corresponding shifted
            recalibration factors must already have been obtained in a foregoing
            mass :ref:`recalibration`.

        - All non-calibrant peaks referenced in `peak_indeces` are treated in a
          similar way. The original fit that yielded the specified `fit_result`
          is re-performed with each shape parameter varied by plus and minus its
          1-sigma confidence respectively while all other shape parameters are
          kept fixed at the original best-fit values. However now, the effective
          mass shifts **after correction with the corresponding updated
          recalibration factor** are recorded and stored in the spectrum's
          :attr:`eff_mass_shifts_pm` dictionary. Only the larger of the two
          eff. mass shifts caused by the +/-1-sigma variation of each shape
          parameter are stored in the spectrum's :attr:`eff_mass_shifts`
          dictionary.
        - The estimates for the total peak-shape uncertainty of each peak are
          finally obtained by adding the eff. mass shifts stored in the
          :attr:`eff_mass_shifts` dictionary in quadrature.

        Mind that peak-shape area uncertainties are only calculated for ions-of-
        interest, not for the mass calibrant.

        References
        ----------
        .. [#] San Andrés, Samuel Ayet, et al. "High-resolution, accurate
           multiple-reflection time-of-flight mass spectrometry for short-lived,
           exotic nuclei of a few events in their ground and low-lying isomeric
           states." Physical Review C 99.6 (2019): 064313.

        """
        if self.shape_cal_pars is None:
            import warnings
            msg = str('Could not calculate peak-shape errors - no peak-shape '
                      'calibration yet!')
            warnings.warn(msg)
            return

        if verbose:
            print('\n##### Peak-shape uncertainty evaluation #####\n')
        if fit_result is None:
            fit_result = self.fit_results[peak_indeces[0]]
        pref = 'p{0}_'.format(peak_indeces[0])
        # grab shape parameters to be varied by +/- sigma:
        shape_pars = [key for key in self.shape_cal_pars
                      if (key.startswith(('sigma','theta','eta','tau','delta'))
                      and fit_result.params[pref+key].expr is None )]
        # Check whether `fit_result` contained the mass calibrant
        if self.index_mass_calib in peak_indeces:
            mass_calib_in_range = True
            # initialize empty dictionary
            self.recal_facs_pm = {}
            peak_indeces.remove(self.index_mass_calib)
            if verbose:
                print('Determining centroid shifts of mass calibrant.\n')
        else:
            mass_calib_in_range = False

        if self.eff_mass_shifts is None:
            # initialize arrays of empty dictionaries
            self.eff_mass_shifts = np.array([{} for i in range(len(self.peaks))])
            self.area_shifts = np.array([{} for i in range(len(self.peaks))])
        if verbose:
            print('All mass shifts below are corrected for the corresponding '
                  'shifts of the mass calibrant peak.\n')

        # Vary each shape parameter by plus and minus one standard deviation and
        # re-fit with all other shape parameters held fixed. Record the
        # corresponding fit results including the shifts of the (Gaussian) peak
        # centroids `mu`
        for par in shape_pars:
            pars = copy.deepcopy(self.shape_cal_pars) # avoid changes in original dict
            pars[par] = self.shape_cal_pars[par] + self.shape_cal_errors[par]
            if par == 'delta_m':
                pars['eta_m2'] = pars[par] - self.shape_cal_pars['eta_m1']
                pars['eta_m3'] = 1 - self.shape_cal_pars['eta_m1'] - pars['eta_m2']
            elif par == 'delta_p':
                pars['eta_p2'] = pars[par] - self.shape_cal_pars['eta_p1']
                pars['eta_p3'] = 1 - self.shape_cal_pars['eta_p1'] - pars['eta_p2']
            fit_result_p = self.peakfit(fit_model=fit_result.fit_model,
                                        cost_func=fit_result.cost_func,
                                        x_fit_cen=fit_result.x_fit_cen,
                                        x_fit_range=fit_result.x_fit_range,
                                        init_pars=pars, vary_shape=False,
                                        vary_baseline=fit_result.vary_baseline,
                                        method=fit_result.method,
                                        fit_kws=fit_result.fit_kws,
                                        show_plots=False)
            #display(fit_result_p) # show fit result

            pars[par] = self.shape_cal_pars[par] - self.shape_cal_errors[par]
            if par == 'delta_m':
                pars['eta_m2'] =  pars[par] - self.shape_cal_pars['eta_m1']
                pars['eta_m3'] = 1 - self.shape_cal_pars['eta_m1'] -  pars['eta_m2']
            elif par == 'delta_p':
                pars['eta_p2'] =  pars[par] - self.shape_cal_pars['eta_p1']
                pars['eta_m3'] = 1 - self.shape_cal_pars['eta_p1'] -  pars['eta_p2']
            fit_result_m = self.peakfit(fit_model=fit_result.fit_model,
                                        cost_func=fit_result.cost_func,
                                        x_fit_cen=fit_result.x_fit_cen,
                                        x_fit_range=fit_result.x_fit_range,
                                        init_pars=pars, vary_shape=False,
                                        vary_baseline=fit_result.vary_baseline,
                                        method=fit_result.method,
                                        fit_kws=fit_result.fit_kws,
                                        show_plots=False)
            #display(fit_result_m) # show fit result

            if show_shape_err_fits:
                fig, axs = plt.subplots(1,2,
                                        figsize=(figwidth*1.5, figwidth*6/18))
                ax0 = axs[0]
                ax0.set_title(r"Re-fit with ("+str(par)+" + 1$\sigma$) = {:.4E}".format(
                                self.shape_cal_pars[par]+self.shape_cal_errors[par]))
                ax0.errorbar(fit_result_p.x, fit_result_p.y, fmt='.',
                             yerr=fit_result_p.y_err, errorevery=10,
                             color="royalblue", linewidth=0.5, zorder=1)
                ax0.plot(fit_result.x, fit_result.best_fit, "--", color="red",
                         linewidth=lwidth, label="original fit",zorder=10)
                ax0.plot(fit_result_p.x, fit_result_p.best_fit, '-', zorder=5,
                         color="black", linewidth=lwidth, label="re-fit")
                ax1 = axs[1]
                ax1.set_title(r"Re-fit with ("+str(par)+" - 1$\sigma$) = {:.4E}".format(
                                self.shape_cal_pars[par]-self.shape_cal_errors[par]))
                ax1.errorbar(fit_result_m.x, fit_result_m.y, fmt='.',
                             yerr=fit_result_m.y_err, errorevery=10,
                             color="royalblue", linewidth=0.5, zorder=1)
                ax1.plot(fit_result.x, fit_result.best_fit, "--", color="red",
                         linewidth=lwidth, label="original fit", zorder=10)
                ax1.plot(fit_result_m.x, fit_result_m.best_fit, '-', zorder=5,
                         color="black", linewidth=lwidth, label="re-fit")
                for ax in axs:
                    ax.legend()
                    ax.set_yscale("log")
                    ax.set_ylim(0.1,)
                plt.show()

            # If mass calibrant is in fit range, determine its ABSOLUTE centroid
            # shifts first and use them to calculate 'shifted' mass
            # recalibration factors. The shifted recalibration factors are then
            # used to correct IOI centroid shifts for the corresponding shifts
            # of the mass calibrant
            # if calibrant is not in fit range, its centroid shifts must have
            # been determined in a foregoing mass re-calibration
            cal_idx = self.index_mass_calib
            if mass_calib_in_range:
                cal_peak = self.peaks[cal_idx]
                pref = 'p{0}_'.format(cal_idx)
                cen = fit_result.best_values[pref+'mu']
                new_cen_p =  fit_result_p.best_values[pref+'mu']
                new_cen_m = fit_result_m.best_values[pref+'mu']
                # recalibration factors obtained with shifted calib. centroids:
                recal_fac_p = cal_peak.m_AME/new_cen_p
                recal_fac_m = cal_peak.m_AME/new_cen_m
                self.recal_facs_pm[par+' recal facs pm'] = [recal_fac_p,recal_fac_m]
            else: # check if shifted recal. factors pre-exist, print error otherwise
                try:
                    isinstance(self.recal_facs_pm[par+' recal facs pm'],list)
                except:
                    raise Exception(
                    'No recalibration factors available for peak-shape '
                    'error evaluation.\n'
                    'Ensure that: \n'
                    '(a) Either the mass calibrant is in the fit range and specified\n'
                    '    with the `index_mass_calib` or `species_mass_calib` parameter, or\n'
                    '(b) if the mass calibrant is not in the fit range, a successful\n'
                    '    mass calibration has been performed upfront with fit_calibrant().')

            # Determine effective mass shifts
            # If calibrant is in fit range, the newly determined calibrant
            # centroid shifts are used to calculate the shifted recalibration
            # factors. Otherwise, the shifted re-calibration factors from a
            # foregoing mass calibration are used
            for peak_idx in peak_indeces: # IOIs only, mass calibrant excluded
                pref = 'p{0}_'.format(peak_idx)
                cen = fit_result.best_values[pref+'mu']
                bin_width = fit_result.x[1] - fit_result.x[0] # assume approx. uniform binning
                area = self.calc_peak_area(peak_idx,fit_result=fit_result)[0]

                new_area_p = self.calc_peak_area(peak_idx,fit_result=fit_result_p)[0]
                new_cen_p =  fit_result_p.best_values[pref+'mu']
                recal_fac_p = self.recal_facs_pm[par+' recal facs pm'][0]
                # effective mass & area shift for +1 sigma parameter variation:
                dm_p = recal_fac_p*new_cen_p - self.recal_fac*cen
                dA_p = new_area_p - area

                new_area_m = self.calc_peak_area(peak_idx,fit_result=fit_result_m)[0]
                new_cen_m = fit_result_m.best_values[pref+'mu']
                recal_fac_m = self.recal_facs_pm[par+' recal facs pm'][1]
                # effective mass & area shift for -1 sigma parameter variation:
                dm_m = recal_fac_m*new_cen_m - self.recal_fac*cen
                dA_m = new_area_m - area
                if verbose:
                    print(u'Re-fitting with {0:6} = {1: .2e} +/-{2: .2e} shifts peak {3:2d} by {4:6.2f}  / {5:6.2f} \u03BCu  & its area by {6: 5.1f} / {7: 5.1f} counts.'.format(
                          par, self.shape_cal_pars[par], self.shape_cal_errors[par], peak_idx, dm_p*1e06, dm_m*1e06, dA_p, dA_m))
                    if peak_idx == peak_indeces[-1]:
                        print()  # empty line between different parameter blocks
                # maximal shifts (mass shifts relative to calibrant centroid)
                self.eff_mass_shifts[peak_idx][par+' eff. mass shift'] = np.where(np.abs(dm_p) > np.abs(dm_m),dm_p,dm_m).item()
                self.area_shifts[peak_idx][par+' area shift'] = np.where(np.abs(dA_p) > np.abs(dA_m),dA_p,dA_m).item()

        # Calculate and update peak-shape mass and area errors by summing all
        # eff. mass shifts and all area shifts respectively in quadrature
        for peak_idx in peak_indeces:
            # Add eff. mass shifts in quadrature to get total PS mass error:
            mass_shift_vals = list(self.eff_mass_shifts[peak_idx].values())
            PS_mass_error = np.sqrt(np.sum(np.square(mass_shift_vals)))
            # Add area shifts in quadrature to get total PS area error:
            area_shift_vals = list(self.area_shifts[peak_idx].values())
            PS_area_error = np.sqrt(np.sum(np.square(area_shift_vals)))
            p = self.peaks[peak_idx]
            pref = 'p{0}_'.format(peak_idx)
            m_ion = fit_result.best_values[pref+'mu']*self.recal_fac
            p.rel_peakshape_error = PS_mass_error/m_ion
            p.area_error = np.round(np.sqrt(self.calc_peak_area(peak_idx,
                                            fit_result=fit_result)[1]**2
                                            + PS_area_error**2), 2)
            try: # remove MC PS error flag
                self.peaks_with_MC_PS_errors.remove(peak_idx)
            except ValueError: # index not in peaks_with_MC_PS_errors
                pass
            if verbose:
                pref = 'p{0}_'.format(peak_idx)
                print("Relative peak-shape error of peak {0:2d}: {1: 7.1e}".format(
                      peak_idx,p.rel_peakshape_error))


    def _eval_MC_peakshape_errors(self, peak_indeces=[], fit_result=None,
                                  verbose=True, show_hists=False,
                                  N_samples=1000,  n_cores=-1, seed=872,
                                  rerun_MCMC_sampling=False, **MCMC_kwargs):
        """Get peak-shape uncertainties for a fit result by re-fitting with many
        different MC-shape-parameter sets

        **This method is primarily intended for internal usage.**

        A representative subset of the shape parameter sets which are supported
        by the data is obtained by performing MCMC sampling on the peak-shape
        calibrant. If this has not already been done using the `map_par_covar`
        option in :meth:`determine_peak_shape`, the :meth:`_get_MCMC_par_samples`
        method will be automatically called here.

        The peaks specified by `peak_indeces` will be fitted with `N_samples`
        different shape parameter sets. The peak-shape uncertainties are then
        estimated as the RMS deviation of the obtained values from the best-fit
        values.

        The mass calibrant must either be included in `peak_indeces` or must
        have been processed with this method upfront (using the same `N_samples`
        and `seed` arguments to ensure identical sets of peak-shapes).

        Parameters
        ----------
        peak_indeces : int or list of int
            Indeces of peaks to evaluate MC peak-shape uncertainties for. The
            peaks of interest must belong to the same `fit_result`.
        fit_result : :class:`~lmfit.model.ModelResult`, optional
            Fit result for which MC peak-shape uncertainties are to be evaluated
            for. Defaults to the fit result stored for the peaks of interest in
            the :attr:`spectrum.fit_results` spectrum attribute.
        verbose : bool, optional
            Whether to print status updates.
        show_hists : bool, optional
            If `True` histograms of the effective mass shifts and peak areas
            obtained with the MC shape parameter sets are shown. Black vertical
            lines indicate the best-fit values stored in `fit_result`.
        N_samples : int, optional
            Number of different shape parameter sets to use. Defaults to 1000.
        n_cores : int, optional
            Number of CPU cores to use for parallelized fitting of simulated
            spectra. When set to `-1` (default) all available cores are used.
        seed : int, optional
            Random seed to use for reproducibility. Defaults to 872.
        rerun_MCMC_sampling : bool, optional
            When `False` (default) pre-existing MCMC parameter samples (e.g.
            obtained with :meth:`determine_peak_shape`) are used. If `True` or
            when there's no pre-existing MCMC samples, the MCMC sampling will be
            performed by this method.
        **MCMC_kwargs
            Keyword arguments to send to :meth:`_get_MCMC_par_samples` for
            control over the MCMC sampling.

        Returns
        -------
        array of floats, array of floats
            Peak-shape mass errors [u], peak-shape area errors [counts]
            Both arrays have the same length as `peak_indeces`.

        See also
        --------
        :meth:`get_MC_peakshape_errors`
        :meth:`_get_MCMC_par_samples`

        Notes
        -----
        For details on MCMC sampling see docs of :meth:`_get_MCMC_par_samples`.

        This method only supports peaks that belong to the same fit result. If
        peaks in multiple `fit_results` are to be treated or the peak properties
        are to be updated with the refined peak-shape errors use
        :meth:`get_MC_peakshape_errors` which wraps around this method.

        """

        peak_indeces = np.atleast_1d(peak_indeces)
        if self.shape_cal_result is None:
            raise Exception('Could not calculate peak-shape errors - '
                            'no peak-shape calibration yet!')
        # Check whether `fit_result` contained the mass calibrant
        if self.index_mass_calib in peak_indeces:
            mass_calib_in_range = True
        elif self.MC_recal_facs is not None:
            mass_calib_in_range = False
        else:
            raise Exception(
                  'No MC recalibration factors available for peak-shape '
                  'error evaluation.\n'
                  'Ensure that: \n'
                  '(a) Either the mass calibrant is in `peak_indeces`, or \n'
                  '(b) this method has been performed on the mass calibrant\n'
                  '    peak upfront.')

        for idx in peak_indeces:
            pname = "p{0}_mu".format(idx)
            if pname not in fit_result.model.param_names:
                raise Exception("Peak {0} not found in `fit_result`.".format(
                               idx))

        if fit_result is None:
            fit_result = self.fit_results[peak_indeces[0]]

        # If MCMC parameter samples have not already been obtained in PS
        # calibration, perform MCMC sampling on peak-shape calibrant here to get
        # shape parameter samples
        if (self.MCMC_par_samples is None) or (rerun_MCMC_sampling is True):
            try:
                MCMC_kwargs['n_cores'] = n_cores
                try: # check if `MCMC_seed` is specified, else set to `seed`
                    MCMC_kwargs['MCMC_seed']
                except KeyError:
                    MCMC_kwargs['MCMC_seed'] = seed
                self._get_MCMC_par_samples(self.shape_cal_result, **MCMC_kwargs)
                # Save thinned and flattened MCMC chain for peakshape evaluation
                flatchain = self.shape_cal_result.flatchain
                self.MCMC_par_samples = flatchain
            except Exception as err:
                print("Failed to obtain MCMC shape parameter samples with "
                      "exception:")
                raise Exception(err)

        if verbose:
            s_peaks = ",".join(map(str,peak_indeces))
            print("\n##### MC Peak-shape uncertainty evaluation for peaks "+s_peaks+" #####\n")
            if mass_calib_in_range:
                print("Determining MC recalibration factors from shifted "
                      "centroids of mass calibrant.\n")
            print("All mass uncertainties below take into account the "
                  "corresponding mass shifts of the calibrant peak.\n")

        # Pick random shape parameter sets from the parameter PDFs determined
        # via MCMC sampling in the peak-shape calibration
        if N_samples > len(self.MCMC_par_samples): # check for enough samples
            self.MCMC_par_samples = None # reset MCMC samples
            msg = str("Not enough MCMC samples available to draw `N_samples` "
                      "random parameter sets without replacement - re-run MCMC "
                      "sampling with more `steps` to obtain more samples or "
                      "use smaller `N_samples`.")
            raise ValueError(msg)
        par_samples = self.MCMC_par_samples.sample(n=N_samples,
                                                   replace=False,
                                                   random_state=seed)

        # Get index of first peak contained in shape calib. result:
        xmin_shape_cal = min(self.shape_cal_result.x)
        xmax_shape_cal = max(self.shape_cal_result.x)
        first_idx_shape_cal = min([idx for idx, p in enumerate(self.peaks)
                               if xmin_shape_cal < p.x_pos < xmax_shape_cal])
        par_samples.columns = par_samples.columns.str.replace('p'+str(
                                                    first_idx_shape_cal)+'_','')
        shape_par_samples = par_samples.to_dict('records') #(orient="row")

        # Determine tail order of fit model for normalization of initial etas
        if fit_result.fit_model.startswith('emg'):
            n_ltails = int(fit_result.fit_model.lstrip('emg')[0])
            n_rtails = int(fit_result.fit_model.lstrip('emg')[1])
        else:
            n_ltails = 0
            n_rtails = 0

        # Iterate over selected parameter sets and re-fit peaks with each set
        # recording the respective centroid shifts and areas
        bkg_c = fit_result.best_values['bkg_c']
        fit_model = fit_result.fit_model
        cost_func = fit_result.cost_func
        method = fit_result.method
        fit_kws = fit_result.fit_kws
        if fit_kws is None:
            fit_kws = {}
        x_cen = fit_result.x_fit_cen
        x_range = fit_result.x_fit_range
        x = fit_result.x
        y = fit_result.y
        weights = 1/np.maximum(1,np.sqrt(y))
        model = fit_result.model
        init_pars = fit_result.init_params

        from numpy import maximum, sqrt, array, log
        from joblib import Parallel, delayed
        from lmfit.model import save_model, load_model
        from lmfit.minimizer import minimize
        from copy import deepcopy
        import time
        datetime = time.localtime() # get current date and time
        datetime_str = time.strftime("%Y-%m-%d_%H-%M-%S", datetime)
        data_fname = self.input_filename.rsplit('.', 1)[0] # del. file extension
        modelfname =  data_fname+datetime_str+"_MC_PS_model.sav"
        save_model(model, modelfname)
        N_peaks = len(peak_indeces)
        N_events = int(np.sum(y))
        tiny = np.finfo(float).tiny # get smallest pos. float in numpy
        funcdefs = {'constant': fit.models.ConstantModel,
                    str(fit_model): getattr(fit_models,fit_model)}
        x_min = x_cen - 0.5*x_range
        x_max = x_cen + 0.5*x_range

        # Define function for parallelized fitting
        def refit(shape_pars):
            model = load_model(modelfname, funcdefs=funcdefs)
            pars = deepcopy(init_pars)
            # Calculate missing parameters from normalization
            if n_ltails == 2:
                shape_pars['eta_m2'] = 1 - shape_pars['eta_m1']
            elif n_ltails == 3:
                eta_m2 = shape_pars['delta_m'] - shape_pars['eta_m1']
                shape_pars['eta_m3'] = 1 - shape_pars['eta_m1'] - eta_m2
            if n_rtails == 2:
                shape_pars['eta_p2'] = 1 - shape_pars['eta_p1']
            elif n_rtails == 3:
                eta_p2 = shape_pars['delta_p'] - shape_pars['eta_p1']
                shape_pars['eta_p3'] = 1 - shape_pars['eta_p1'] - eta_p2
            # Update model parameters
            for par, val in shape_pars.items():
                if par == "bkg_c":
                    pars[par].value = val
                elif par.endswith(('amp','mu')):
                    pass
                else: # shape parameter
                    for idx in peak_indeces:
                        pref = "p{0}_".format(idx)
                        pars[pref+par].value = val

            if cost_func  == 'chi-square':
                ## Pearson's chi-squared fit with iterative weights 1/Sqrt(f(x_i))
                eps = 1e-10 # small number to bound Pearson weights
                def resid_Pearson_chi_square(pars,y_data,weights,x=x):
                    y_m = model.eval(pars,x=x)
                    # Calculate weights for current iteration, add tiny number `eps`
                    # in denominator for numerical stability
                    weights = 1/np.sqrt(y_m + eps)
                    return (y_m - y_data)*weights
                # Overwrite lmfit's standard least square residuals with iterative
                # residuals for Pearson chi-square fit
                model._residual = resid_Pearson_chi_square
            elif cost_func  == 'MLE':
                # Define sqrt of (doubled) negative log-likelihood ratio (NLLR)
                # summands:
                def sqrt_NLLR(pars,y_data,weights,x=x):
                    y_m = model.eval(pars,x=x) # model
                    # Add tiniest pos. float representable by numpy to arguments of
                    # np.log to smoothly handle divergences for log(arg -> 0)
                    NLLR = 2*(y_m-y_data) + 2*y_data*(log(y_data+tiny)-log(y_m+tiny))
                    ret = sqrt(NLLR)
                    return ret
                # Overwrite lmfit's standard least square residuals with the
                # square-roots of the NLLR summands, this enables usage of scipy's
                # `least_squares` minimizer and yields much faster optimization
                # than with scalar minimizers
                model._residual = sqrt_NLLR
            else:
                raise Exception("'cost_func' of given `fit_result` not supported.")

            # re-perform fit on simulated spectrum - for performance use only the
            # underlying Minimizer object instead of full lmfit model interface
            try:
                min_res = minimize(model._residual, pars, method=method,
                                   args=(y,weights), kws={'x':x},
                                   scale_covar=False, nan_policy='propagate',
                                   reduce_fcn=None, calc_covar=False, **fit_kws)

                # Record peak centroids and amplitudes
                new_mus = []
                new_amps = []
                for idx in peak_indeces:
                    pref = 'p{0}_'.format(idx)
                    mu = min_res.params[pref+'mu']
                    amp = min_res.params[pref+'amp']
                    new_mus.append(mu)
                    new_amps.append(amp)

                return np.array([new_mus, new_amps])

            except Exception as err:
                print("Skipped a parameter set due to error: ")
                print(err)
                return np.array([[np.nan]*N_peaks, [np.nan]*N_peaks])

        from tqdm.auto import tqdm
        print("Fitting peaks with "+str(N_samples)+" different MCMC-shape-"
              "parameter sets to determine refined peak-shape errors.")
        #res = np.array([refit(pars) for pars in tqdm(shape_par_samples)]) # serial version
        res = np.array(Parallel(n_jobs=n_cores)(delayed(refit)(pars)
                                for pars in tqdm(shape_par_samples)))
        # Force workers to shut down
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)
        os.remove(modelfname) # clean up

        # Format results
        trp_mus, trp_amps = res[:,0], res[:,1]
        mus = trp_mus.transpose()
        amps = trp_amps.transpose()

        # If mass calibrant is in fit range, use the shifted calibrant centroids
        # to get the respective recalibration factor for each MC parameter set
        # if calibrant is not in fit range, its centroid shifts must have
        # been determined beforehand with the same MC shape parameter sets
        cal_idx = self.index_mass_calib
        if mass_calib_in_range:
            cal_peak = self.peaks[cal_idx]
            i = np.where(peak_indeces == cal_idx)[0][0]
            MC_cal_cens = mus[i]
            self.MC_recal_facs = cal_peak.m_AME/MC_cal_cens

        # Determine effective mass and area errors
        MC_PS_mass_errs = []
        MC_PS_area_errs = []
        boxprops = dict(boxstyle='round', facecolor='grey', alpha=0.5)
        bin_width = x[1] - x[0] # assumes uniform binning
        for i, peak_idx in enumerate(peak_indeces):
            p = self.peaks[peak_idx]

            dm = self.MC_recal_facs*mus[i] - p.m_ion
            dm = dm[~np.isnan(dm)] # drop NaN values
            PS_mass_err = np.sqrt(np.mean(dm**2))
            MC_PS_mass_errs.append(PS_mass_err)

            darea = amps[i]/bin_width - p.area
            darea = darea[~np.isnan(darea)] # drop NaN values
            PS_area_err = np.sqrt(np.mean(darea**2))
            MC_PS_area_errs.append(PS_area_err)

            if show_hists:
                # Plot histograms
                f, ax = plt.subplots(nrows=1,ncols=2,
                                     figsize=(figwidth*1.5,figwidth*4/18*1.5))
                ax0, ax1 = ax.flatten()
                ax0.set_title("Centroids - peak {0}".format(peak_idx),
                              fontdict={'fontsize':17})
                ax0.hist(dm*1e06,bins=19)
                text0 = r"RMS dev. ={0: .1f} $\mu$u".format(PS_mass_err*1e06)
                ax0.text(0.65, 0.94, text0,transform=ax0.transAxes, fontsize=14,
                         verticalalignment='top', bbox=boxprops)
                ax0.axvline(0, color='black') # best-fit mu
                ax0.xaxis.get_offset_text().set_fontsize(15)
                ax0.tick_params(axis='both',labelsize=15)
                ax0.set_xlabel(r"Effective mass shift [$\mu$u]", fontsize=16)
                ax0.set_ylabel("Occurences", fontsize=16)
                ax1.set_title("Areas - peak {0}".format(peak_idx),
                              fontdict={'fontsize':17})
                ax1.hist(p.area + darea,bins=19)
                text1 = "RMS dev. ={0: .1f} counts".format(PS_area_err)
                ax1.text(0.6, 0.94, text1, transform=ax1.transAxes, fontsize=14,
                         verticalalignment='top', bbox=boxprops)
                ax1.axvline(p.area, color='black')
                ax1.xaxis.get_offset_text().set_fontsize(15)
                ax1.tick_params(axis='both',labelsize=15)
                ax1.set_xlabel("Peak area [counts]", fontsize=16)
                ax1.set_ylabel("Occurences", fontsize=16)
                plt.show()

        # Print results
        if verbose:
            print("### Results ###\n")
            print( "         Relative peak-shape (mass) uncertainty     Peak-shape uncertainty of ")
            print(u"         from +-1\u03C3 variation / from MC samples      peak areas from MC samples")
            for i, peak_idx in enumerate(peak_indeces):
                p = self.peaks[peak_idx]
                if peak_idx == cal_idx:
                    rel_PS_err = 0.0 # avoid NoneType Error in .format() below
                else:
                    rel_PS_err = p.rel_peakshape_error
                print("Peak {:2}:           {:6.2e}  /  {:6.2e}                   {:5.1f} counts".format(
                      peak_idx, rel_PS_err, MC_PS_mass_errs[i]/p.m_ion, MC_PS_area_errs[i]))

        return MC_PS_mass_errs, MC_PS_area_errs


    def get_MC_peakshape_errors(self, peak_indeces=[], verbose=True,
                                show_hists=False, show_peak_properties=True,
                                rerun_MCMC_sampling=False, N_samples=1000,
                                n_cores=-1, seed=872, **MCMC_kwargs):
        """Get peak-shape uncertainties by re-fitting peaks with many different
        MC-shape-parameter sets

        This method provides refined peak-shape uncertainties that account for
        non-normal distributions and correlations of shape parameters. To that
        end, the peaks of interest are re-fitted with `N_samples` different
        peak-shape parameter sets. For these parameter sets to be representative
        of all peak shapes supported by the data they are randomly drawn from a
        larger ensemble of parameter sets obtained from Markov-Chain Monte Carlo
        (MCMC) sampling on the peak-shape calibrant. The peak-shape uncertainty
        of the mass values and peak areas are estimated by the obtained RMS
        deviations from the best-fit values. Finally, the peak properties table
        is updated with the refined uncertainties.

        This method only takes effective mass shifts relative to the calibrant
        peak into account. For each peak shape the calibrant peak is re-fitted
        and the new recalibration factor is used to calculate the shifted
        ion-of-interest masses. Therefore, when the `peak_indeces` argument is
        used, it must include the mass calibrant index.

        Parameters
        ----------
        peak_indeces : int or list of int, optional
            Indeces of peaks to evaluate MC peak-shape uncertainties for.
        verbose : bool, optional
            Whether to print status updates and intermediate results.
        show_hists : bool, optional
            If `True` histograms of the effective mass shifts and peak areas
            obtained with the MC shape parameter sets are shown. Black vertical
            lines indicate the best-fit values obtained with :meth:`fit_peaks`.
        show_peak_properties : bool, optional
            If `True` the peak properties table including the updated peak-shape
            uncertainties is shown.
        rerun_MCMC_sampling : bool, optional
            When `False` (default) pre-existing MCMC parameter samples (e.g.
            obtained with :meth:`determine_peak_shape`) are used. If `True` or
            when there's no pre-existing MCMC samples, the MCMC sampling will be
            performed by this method.
        N_samples : int, optional
            Number of different shape parameter sets to use. Defaults to 1000.
        n_cores : int, optional
            Number of CPU cores to use for parallelized fitting of simulated
            spectra. When set to `-1` (default) all available cores are used.
        seed : int, optional
            Random seed to use for reproducibility. Defaults to 872.
        **MCMC_kwargs
            Keyword arguments to send to :meth:`_get_MCMC_par_samples` for
            control over the MCMC sampling.

        See also
        --------
        :meth:`_eval_MC_peakshape_errors`
        :meth:`_get_MCMC_par_samples`

        Notes
        -----
        This method relies on a representative sample of all the shape parameter
        sets which are supported by the data. These shape parameter sets are
        randomly drawn from a large sample of parameter sets obtained from
        Markov-Chain Monte Carlo (MCMC) sampling on the peak-shape calibrant.
        In MCMC sampling so-called walkers are sent on random walks to explore
        the parameter space. The latter is done with the
        :meth:`_get_MCMC_par_samples` method. If MCMC sampling has already
        been performed with the `map_par_covar` option in
        :meth:`determine_peak_shape`, these MCMC samples will be
        used for the MC peak-shape error evaluation. If there is no pre-existing
        MCMC parameter sets the :meth:`_get_MCMC_par_samples` method will be
        automatically evoked before the MC peak-shape error evaluation.

        Assuming that the samples obtained with the MCMC algorithm form a
        representative set of parameter samples and are sufficiently independent
        from each other, this method provides refined peak-shape uncertainties
        that account for correlations and non-normal posterior distributions
        of peak-shape parameters. In particular, this prevents overestimation of
        the uncertainties due to non-consideration of parameter correlations.

        For this method to be accurate a sufficiently large number of MCMC
        sampling steps should be performed and fits should be performed with a
        large number of parameter sets (``N_samples >= 1000``). For the MCMC
        parameter samples to be independent a sufficient amount of thinning has
        to be applied to remove autocorrelation between MCMC samples. Thinning
        refers to the common practice of only storing the results of every k-th
        MCMC iteration. The length and thinning of the MCMC chain is controlled
        with the `steps` and `thin` MCMC keyword arguments. For more details and
        references on MCMC sampling with emgfit see the docs of the underlying
        :meth:`_get_MCMC_par_samples` method.

        For the peak-shape mass uncertainties only effective mass shifts
        relative to the calibrant centroid are relevant. Therefore, the mass
        calibrant and the ions of interest (IOI) are fitted with the same
        peak-shape-parameter sets and the final mass values are calculated from
        the obtained IOI peak positions and the corresponding mass recalibration
        factors.

        The `delta_m` or `delta_p` parameters occuring in the case of hyper-EMG
        models with 3 pos. or 3 neg. tails are defined as
        ``delta_m = eta_p1 + eta_p2`` and ``delta_p = eta_p1 + eta_p2``,
        respectively.

        """
        if peak_indeces in ([], None):
            peak_indeces = np.arange(len(self.peaks)).tolist()
        peak_indeces = np.atleast_1d(peak_indeces).tolist()

        if self.index_mass_calib not in peak_indeces:
            raise Exception("Mass calibrant must be in `peak_indeces`.")

        # Collect fit_results for peaks in `peak_indeces`
        results = []
        POI = [] # 2D-list with indeces of interest for each fit_result
        peak_indeces.sort()
        for idx in peak_indeces:
            res = self.fit_results[idx]
            if res is None:
                raise Exception("No fit result found for peak {}.".format(idx))
            if res not in results:
                results.append(res)
                POI.append([idx])
            else:
                i_res = results.index(res)
                POI[i_res].append(idx)
            if idx == self.index_mass_calib:
                i_cal_res = i_res
        # Move calibrant result and corresponding peaks to the front of the
        # results and POI lists to ensure that the calibrant centroid shifts are
        # determined before other results are treated below
        results.insert(0, results.pop(i_cal_res))
        POI.insert(0, POI.pop(i_cal_res))

        # For each fit_result, perform many fits with the same MCMC parameter
        # sets (the latter is ensured by the identical `seed` arguments)
        for i_res, res in enumerate(results):
            PS_mass_errs, PS_area_errs = self._eval_MC_peakshape_errors(
                                                       peak_indeces=POI[i_res],
                                                       fit_result=res,
                                                       verbose=verbose,
                                                       show_hists=show_hists,
                                                       N_samples=N_samples,
                                                       n_cores=n_cores,
                                                       seed=seed,
                                                       rerun_MCMC_sampling=
                                                       rerun_MCMC_sampling,
                                                       **MCMC_kwargs)

            # Update peak properties with refined stat. and area uncertainties
            # and set `MC_PS_errs` flag
            for i_p, peak_idx in enumerate(POI[i_res]):
                if PS_area_errs[i_p]==np.nan  or PS_mass_errs[i_p]==np.nan:
                    import warnings
                    msg = str("Properties of peak {} not updated since "
                              "MC estimates of mass or area peak-shape error "
                              "are NaN.").format(peak_idx)
                    warings.warn(msg)
                    continue # skip updating properties of this peak
                p = self.peaks[peak_idx]
                pref = 'p{0}_'.format(peak_idx)
                m_ion = p.m_ion
                # Add best-fit area error and peakshape area error in quadrature
                try:
                    pm_area_shifts = list(self.area_shifts[peak_idx].values())
                except AttributeError: # area shifts not initialized
                    pm_area_shifts = []
                pm_PS_err = np.sqrt(np.sum(np.square(pm_area_shifts)))
                # Remove PS errors obtained via +- 1 sigma variation
                stat_area_err = np.sqrt(p.area_error**2 - pm_PS_err**2)
                p.area_error = np.round(np.sqrt(stat_area_err**2 +
                                                PS_area_errs[i_p]**2), 2)
                if peak_idx != self.index_mass_calib:
                    p.rel_peakshape_error = PS_mass_errs[i_p]/p.m_ion
                    self.peaks_with_MC_PS_errors.append(peak_idx)
                    self.peaks_with_MC_PS_errors.sort()
                    try:
                        p.rel_mass_error = np.sqrt(p.rel_stat_error**2 +
                                                   p.rel_peakshape_error**2 +
                                                   p.rel_recal_error**2)
                        p.mass_error_keV = np.round(
                                            p.rel_mass_error*p.m_ion*u_to_keV,3)
                    except TypeError:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("once")
                            msg = str("Could not update total mass error of "
                                      "peak {0} due to TypeError. Check if the "
                                      "`rel_stat_error` and `rel_recal_error` "
                                      "are defined. ").format(peak_idx)
                            warnings.warn(msg)
                try:
                    s_indeces = ",".join([s_indeces,str(peak_idx)])
                except UnboundLocalError: # handle first index in s_indeces
                    s_indeces = str(peak_idx)
            print("\nUpdated area error, peak-shape error and (total)"
                  " mass error of peak(s) "+s_indeces+".\n")

        if show_peak_properties:
            print() # insert blank line
            print("Peak properties table after MC peak-shape error evaluation:")
            self.show_peak_properties()


    def _update_calibrant_props(self, index_mass_calib, fit_result):
        """Determine recalibration factor and update mass calibrant peak
        properties.

        **Intended for internal use only.**

        Parameters
        ----------
        index_mass_calib : int
            Index of mass calibrant peak.
        fit_result : :class:`lmfit.model.ModelResult`
            Fit result to use for calculating the re-calibration factor and
            for updating the calibrant properties.

        """
        peak = self.peaks[index_mass_calib]
        if peak.m_AME is None or peak.m_AME_error is None:
            raise Exception("Mass calibration failed due to missing literature "
                            "values for calibrant. Ensure the species of the "
                            "calibrant peak has been assigned!")
        # Set 'mass calibrant' flag in peak comments
        for p in self.peaks: # reset 'mass calibrant' comment flag
            if 'shape & mass calibrant' in p.comment :
                p.comment = p.comment.replace('shape & mass calibrant',
                                              'shape calibrant')
            elif p.comment == 'mass calibrant':
                p.comment = '-'
            elif 'mass calibrant' in p.comment:
                p.comment = p.comment.replace('mass calibrant','')
        if 'shape calibrant' in peak.comment: # set flag
            peak.comment = peak.comment.replace('shape calibrant',
                                                'shape & mass calibrant')
        elif peak.comment == '-' or peak.comment == '' or peak.comment is None:
            peak.comment = 'mass calibrant'
        else:
            peak.comment = 'mass calibrant, '+peak.comment
        peak.fit_model = fit_result.fit_model
        peak.cost_func = fit_result.cost_func
        peak.area, peak.area_error = self.calc_peak_area(index_mass_calib,
                                                         fit_result=fit_result)
        pref = 'p{0}_'.format(index_mass_calib)
        peak.m_ion = fit_result.best_values[pref+'mu']
        # A_stat* FWHM/sqrt(area), w/ with A_stat_G = 0.42... and A_stat_emg
        # from `determine_A_stat_emg` method or default value from config.py
        if peak.fit_model == 'Gaussian':
            std_dev = fit_result.best_values[pref+'sigma']
        else:  # for emg models
            FWHM_emg = self.calc_FWHM_emg(index_mass_calib,fit_result=fit_result)
            std_dev = self.A_stat_emg*FWHM_emg
        stat_error = std_dev/np.sqrt(peak.area)
        peak.rel_stat_error = stat_error /peak.m_ion
        peak.rel_peakshape_error = None # reset to None
        peak.red_chi = np.round(fit_result.redchi, 2)
        try: # remove resampling error flag
            self.peaks_with_errors_from_resampling.remove(index_mass_calib)
        except ValueError: # index not in peaks_with_errors_from_resampling
            pass

        # Print error contributions of mass calibrant:
        print("\n##### Mass recalibration #####\n")
        print("Relative literature error of mass calibrant:     {:7.1e}".format(
              peak.m_AME_error/peak.m_ion))
        print("Relative statistical error of mass calibrant:    {:7.1e}".format(
              peak.rel_stat_error))

        # Determine recalibration factor
        self.recal_fac = peak.m_AME/peak.m_ion
        print("\nRecalibration factor:    {:1.9f} = 1 {:=+5.1e}".format(
              self.recal_fac,self.recal_fac-1))
        if np.abs(self.recal_fac - 1) > 1e-03:
            import warnings
            msg = str("Recalibration factor `recal_fac` deviates from unity by "
                      "more than a permille. Potentially, mass errors should "
                      "also be re-scaled with `recal_fac` (currently not "
                      "implemented in emgfit)!")
            warnings.warn(msg, UserWarning)
        # Set mass calibrant flag to prevent overwriting of calibration results
        self.index_mass_calib = index_mass_calib
        # Update peak properties with new calibrant centroid
        peak.m_ion = self.recal_fac*peak.m_ion # update calibrant centroid mass
        if peak.A:
            # atomic Mass excess (includes electron mass) [keV]
            peak.atomic_ME_keV = np.round((peak.m_ion+m_e-peak.A)*u_to_keV, 3)
        if peak.m_AME:
            peak.m_dev_keV = np.round( (peak.m_ion - peak.m_AME)*u_to_keV, 3)

        # Determine rel. recalibration error and update recalibration err. attribute
        peak.rel_recal_error = np.sqrt( (peak.m_AME_error/peak.m_AME)**2 +
                                         peak.rel_stat_error**2 )/self.recal_fac
        self.rel_recal_error = peak.rel_recal_error
        print("Relative recalibration error:    {:7.1e} \n".format(
              self.rel_recal_error))


    def fit_calibrant(self, index_mass_calib=None, species_mass_calib=None,
                      fit_model=None, cost_func='MLE', x_fit_cen=None,
                      x_fit_range=None, vary_baseline=True,
                      method='least_squares', fit_kws=None, show_plots=True,
                      show_peak_markers=True, sigmas_of_conf_band=0,
                      error_every=1, show_fit_report=True, plot_filename=None):
        """Determine mass re-calibration factor by fitting the selected
        calibrant peak.

        After the mass calibrant has been fitted the recalibration factor and
        its uncertainty are calculated and saved as the spectrum's
        :attr:`recal_fac` and :attr:`recal_fac_error` attributes.

        The calibrant peak can either be specified with the `index_mass_calib`
        or the `species_mass_calib` argument.

        Parameters
        ----------
        index_mass_calib : int, optional
            Index of mass calibrant peak.
        species_mass_calib : str, optional
            Species of peak to use as mass calibrant.
        fit_model : str, optional, default: ``'emg22'``
            Name of fit model to use (e.g. ``'Gaussian'``, ``'emg12'``,
            ``'emg33'``, ... - for full list see :ref:`fit_model_list`).
        cost_func : str, optional, default: 'chi-square'
            Name of cost function to use for minimization.

            - If ``'chi-square'``, the fit is performed by minimizing Pearson's
              chi-squared statistic:

              .. math::

                  \\chi^2_P = \\sum_i \\frac{(f(x_i) - y_i)^2}{f(x_i)}.

            - If ``'MLE'``, a binned maximum likelihood estimation is performed
              by minimizing the (doubled) negative log likelihood ratio:

              .. math::

                  L = 2\\sum_i \\left[ f(x_i) - y_i + y_i ln\\left(\\frac{y_i}{f(x_i)}\\right)\\right]

            See `Notes` of :meth:`spectrum.peakfit` for more details.
        x_fit_cen : float or None, [u], optional
            center of mass range to fit;
            if None, defaults to marker position (x_pos) of mass calibrant peak
        x_fit_range : float [u], optional
            width of mass range to fit; if None, defaults to 'default_fit_range' spectrum attribute
        vary_baseline : bool, optional, default: `True`
            If `True`, the constant background will be fitted with a varying
            uniform baseline parameter `bkg_c` (initial value: 0.1).
            If `False`, the baseline parameter `bkg_c` will be fixed to 0.
        method : str, optional, default: `'least_squares'`
            Name of minimization algorithm to use. For full list of options
            check arguments of :func:`lmfit.minimizer.minimize`.
        fit_kws : dict, optional, default: None
            Options to pass to lmfit minimizer used in
            :meth:`lmfit.model.Model.fit` method.
        show_plots : bool, optional
            If `True` (default) linear and logarithmic plots of the spectrum
            with the best fit curve are displayed. For details see
            :meth:`spectrum.plot_fit`.
        show_peak_markers : bool, optional
            If `True` (default) peak markers are added to the plots.
        sigmas_of_conf_band : int, optional, default: 0
            Confidence level of confidence band around best fit curve in sigma.
            Note that the confidence band is only derived from the uncertainties
            of the parameters that are varied during the fit.
        error_every : int, optional, default: 1
            Show error bars only for every `error_every`-th data point.
        show_fit_report : bool, optional
            If `True` (default) the fit results are reported.
        plot_filename : str, optional, default: None
            If not ``None``, the plots will be saved to two separate files named
            '<`plot_filename`>_log_plot.png' and '<`plot_filename`>_lin_plot.png'.
            **Caution: Existing files with identical name are overwritten.**

        See also
        --------
        :meth:`spectrum.fit_peaks`

        Notes
        -----
        The :meth:`spectrum.fit_peaks` method enables the simultaneous fitting
        of mass calibrant and ions of interest in a single multi-peak fit and
        can be used as an alternative to this method.

        After the calibrant fit the :meth:`spectrum._eval_peakshape_errors`
        method is automatically called to save the absolute calibrant centroid
        shifts as preparation for subsequent peak-shape error determinations.

        Since the spectrum has already been coarsely calibrated via the time-
        resolved calibration in the MR-TOF-MS's data acquisition software MAc,
        the recalibration (or precision calibration) factor is usually very
        close to unity. An error will be raised by the
        :meth:`spectrum._update_calibrant_props` method if
        :attr:`spectrum.recal_fac` deviates from unity by more than a permille
        since this causes some implicit approximations for the calculation of
        the final mass values and their uncertainties to break down.

        The statistical uncertainty of the peak is calculated via the following
        relation [#]_:

        .. math:

            \\sigma_{stat} = A_{stat} \\frac{FWHM}{\\sqrt(N_counts)}

        For Gaussians the constant of proportionality :math:`A_{stat}` is always
        given by :math:`A_{stat,G}` = 0.425. For Hyper-EMG models
        :math:`A_{stat}=A_{stat,emg}` is either set to the default value
        `A_stat_emg_default` defined in the :mod:`~emgfit.config` module or
        determined by running the :meth:`spectrum.determine_A_stat_emg` method.
        The latter is usually preferable since this accounts for the specifics
        of the given peak shape.

        References
        ----------
        .. [#] San Andrés, Samuel Ayet, et al. "High-resolution, accurate
           multiple-reflection time-of-flight mass spectrometry for short-lived,
           exotic nuclei of a few events in their ground and low-lying isomeric
           states." Physical Review C 99.6 (2019): 064313.

        """
        if index_mass_calib is not None and (species_mass_calib is None):
            peak = self.peaks[index_mass_calib]
        elif species_mass_calib:
            index_mass_calib = [i for i in range(len(self.peaks)) if species_mass_calib == self.peaks[i].species][0]
            peak = self.peaks[index_mass_calib]
        else:
            raise Exception("Definition of mass calibrant peak failed. Define "
                            "EITHER the index OR the species name of the peak "
                            "to use as mass calibrant! ")
        if x_fit_range is None:
            x_fit_range = self.default_fit_range

        print('##### Calibrant fit #####')
        if fit_model is None:
            fit_model = self.fit_model
        if x_fit_cen is None:
            x_fit_cen = peak.x_pos
        fit_result = spectrum.peakfit(self, fit_model=fit_model,
                                      cost_func=cost_func,
                                      x_fit_cen=x_fit_cen,
                                      x_fit_range=x_fit_range,
                                      vary_shape=False,
                                      vary_baseline=vary_baseline,
                                      method=method,
                                      fit_kws=fit_kws,
                                      show_plots=show_plots,
                                      show_peak_markers=show_peak_markers,
                                      sigmas_of_conf_band=sigmas_of_conf_band,
                                      error_every=error_every,
                                      plot_filename=plot_filename)
        if show_fit_report:
            self._show_blinded_report(fit_result)

        # Update recalibration factor and calibrant properties
        self._update_calibrant_props(index_mass_calib,fit_result)
        # Calculate updated recalibration factors from absolute centroid shifts
        # of calibrant and as prep for subsequent peak-shape error determination
        # for ions of interest
        self._eval_peakshape_errors(peak_indeces=[index_mass_calib],
                                    fit_result=fit_result, verbose=False)
        # Save fit result, in case calibrant is not fitted again
        self.fit_results[self.index_mass_calib] = fit_result


    def _update_peak_props(self, peaks, fit_result):
        """Update the peak properties using the given 'fit_result'.

        **Intended for internal use only.**

        The values of the mass calibrant will not be changed by
        this routine.

        Parameters
        ----------
        peaks : list
            List of indeces of peaks to update. (To get peak indeces, see plot
            markers or consult the peak properties table by calling the
            :meth:`spectrum.show_peak_properties` method)
        fit_result : :class:`lmfit.model.ModelResult`
            :class:`~lmfit.model.ModelResult` holding the fit results of all
            `peaks` to be updated.

        Note
        ----
        All peaks referenced by the 'peaks' argument must belong to the same
        `fit_result`. Not necessarily all peaks contained in `fit_result` will
        be updated, only the properties of peaks referenced with the `peaks`
        argument will be updated.

        """
        for p in peaks:
            if self.peaks.index(p) == self.index_mass_calib:
                pass  # prevent overwritting of mass recalibration results
            else:
                peak_idx = self.peaks.index(p)
                pref = 'p{0}_'.format(peak_idx)
                p.fit_model = fit_result.fit_model
                p.cost_func = fit_result.cost_func
                p.area = self.calc_peak_area(peak_idx,fit_result=fit_result)[0]
                if p.area_error is None:
                    # set in case area err has not already been defined in
                    # _eval_peakshape_errors()
                    p.area_error = self.calc_peak_area(peak_idx,fit_result=
                                                   fit_result)[1]
                p.m_ion = self.recal_fac*fit_result.best_values[pref+'mu']
                # stat_error = A_stat * FWHM / sqrt(peak_area), w/ with
                # A_stat_G = 0.42... and  A_stat_emg from `determine_A_stat_emg`
                # method or default value from config.py
                if p.fit_model == 'Gaussian':
                    std_dev = fit_result.best_values[pref+'sigma']
                else:  # for emg models
                    FWHM_emg = self.calc_FWHM_emg(peak_idx,fit_result=fit_result)
                    std_dev = self.A_stat_emg*FWHM_emg
                stat_error = std_dev/np.sqrt(p.area)
                p.rel_stat_error = stat_error/p.m_ion
                try: # remove resampling error flag
                    self.peaks_with_errors_from_resampling.remove(peak_idx)
                except ValueError: # index not in peaks_with_errors_from_resampling
                    pass
                if self.rel_recal_error:
                    p.rel_recal_error = self.rel_recal_error
                else:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter('once')
                        msg  = str('Could not set mass recalibration errors - '
                                   'no successful mass recalibration performed '
                                   'on spectrum yet.')
                        warnings.warn(msg)
                try:
                    # total relative uncertainty of mass value - includes:
                    # stat. mass uncertainty, peakshape uncertainty &
                    # recalibration uncertainty
                    p.rel_mass_error = np.sqrt(p.rel_stat_error**2 +
                                               p.rel_peakshape_error**2 +
                                               p.rel_recal_error**2)
                    p.mass_error_keV = np.round(
                                           p.rel_mass_error*p.m_ion*u_to_keV, 3)
                except TypeError:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter('once')
                        msg = str('Could not calculate total mass error due to '
                                  'TypeError.')
                        warnings.warn(msg)
                if p.A:
                    # atomic Mass excess (includes electron mass) [keV]
                    p.atomic_ME_keV = np.round((p.m_ion + m_e - p.A)*u_to_keV,3)
                if p.m_AME:
                    p.m_dev_keV = np.round( (p.m_ion - p.m_AME)*u_to_keV, 3)
                p.red_chi = np.round(fit_result.redchi, 2)


    def fit_peaks(self, peak_indeces=[], index_mass_calib=None,
                  species_mass_calib=None, x_fit_cen=None, x_fit_range=None,
                  fit_model=None, cost_func='MLE', method ='least_squares',
                  fit_kws=None, init_pars=None, vary_shape=False,
                  vary_baseline=True, show_plots=True, show_peak_markers=True,
                  sigmas_of_conf_band=0, error_every=1, plot_filename=None,
                  show_fit_report=True, show_shape_err_fits=False):
        """Fit peaks, update peaks properties and show results.

        By default, the full mass range and all peaks in the spectrum are
        fitted. Optionally, only peaks specified with `peak_indeces` or peaks in
        the mass range specified with `x_fit_cen` and `x_fit_range` are fitted.

        Optionally, the mass recalibration can be performed simultaneously with
        the IOI fit if the mass calibrant is in the fit range and specified with
        either the `index_mass_calib` or `species_mass_calib` arguments.
        Otherwise a mass recalibration must have been performed upfront.

        Before running this method a successful peak-shape calibration must have
        been performed with :meth:`determine_peak_shape`.

        Parameters
        ----------
        peak_indeces : list of int, optional
            List of neighbouring peaks to fit. The fit range will be chosen such
            that at least a mass range of `x_fit_range`/2 is included around
            each peak.
        x_fit_cen : float [u], optional
            Center of mass range to fit (only specify if a subset of the
            spectrum is to be fitted)
        x_fit_range : float [u], optional
            Width of mass range to fit. If ``None`` defaults to:
            :attr:`spectrum.default_fit_range` attribute, only specify if subset
            of spectrum is to be fitted. This argument is only relevant if
            `x_fit_cen` is also specified.
        fit_model : str, optional
            Name of fit model to use (e.g. ``'Gaussian'``, ``'emg12'``,
            ``'emg33'``, ... - for full list see :ref:`fit_model_list`). If
            ``None``, defaults to :attr:`~spectrum.fit_model` spectrum
            attribute.
        cost_func : str, optional, default: 'chi-square'
            Name of cost function to use for minimization.

            - If ``'chi-square'``, the fit is performed by minimizing Pearson's
              chi-squared statistic:

              .. math::

                  \\chi^2_P = \\sum_i \\frac{(f(x_i) - y_i)^2}{f(x_i)^2}.

            - If ``'MLE'``, a binned maximum likelihood estimation is performed
              by minimizing the (doubled) negative log likelihood ratio:

              .. math::

                  L = 2\\sum_i \\left[ f(x_i) - y_i + y_i ln\\left(\\frac{y_i}{f(x_i)}\\right)\\right]

            See `Notes` of :meth:`peakfit` method for details.
        method : str, optional, default: `'least_squares'`
            Name of minimization algorithm to use. For full list of options
            check arguments of :func:`lmfit.minimizer.minimize`.
        fit_kws : dict, optional, default: None
            Options to pass to lmfit minimizer used in
            :meth:`lmfit.model.Model.fit` method.
        init_pars : dict, optional
            Dictionary with initial shape parameter values for fit (optional).

            - If ``None`` (default) the parameters from the shape calibration
              (:attr:`peak_shape_pars` spectrum attribute) are used.
            - If ``'default'``, the default parameters defined for mass 100 in
              the :mod:`emgfit.fit_models` module will be used after re-scaling
              to the spectrum's :attr:`mass_number`.
            - To define custom initial values a parameter dictionary containing
              all model parameters and their values in the format
              ``{'<param name>':<param_value>,...}`` should be passed to
              `init_pars`.

          Mind that only the initial values to shape parameters
          (`sigma`, `theta`,`etas` and `taus`) can be user-defined. The
          `mu` parameter will be initialized at the peak's :attr:`x_cen`
          attribute and the initial peak amplitude `amp` is automatically
          estimated from the counts at the bin closest to `x_cen`. If a
          varying baseline is used in the fit, the baseline parameter
          `bgd_c` is always initialized at a value of 0.1.
        vary_shape : bool, optional, default: `False`
            If `False` peak-shape parameters (`sigma`, `theta`,`etas` and
            `taus`) are kept fixed at their initial values. If `True` the
            shared shape parameters are varied (ensuring identical shape
            parameters for all peaks).
        vary_baseline : bool, optional, default: `True`
            If `True`, the constant background will be fitted with a varying
            uniform baseline parameter `bkg_c` (initial value: 0.1).
            If `False`, the baseline parameter `bkg_c` will be fixed to 0.
        show_plots : bool, optional
            If `True` (default) linear and logarithmic plots of the spectrum
            with the best fit curve are displayed. For details see
            :meth:`spectrum.plot_fit`.
        show_peak_markers : bool, optional
            If `True` (default) peak markers are added to the plots.
        sigmas_of_conf_band : int, optional, default: 0
            Confidence level of confidence band around best-fit curve in sigma.
            Note that the confidence band is only derived from the uncertainties
            of the parameters that are varied during the fit.
        error_every : int, optional, default: 1
            Show errorbar only for every `error_every`-th data point.
        plot_filename : str, optional, default: None
            If not ``None``, the plots will be saved to two separate files named
            '<`plot_filename`>_log_plot.png' and '<`plot_filename`>_lin_plot.png'.
            **Caution: Existing files with identical name are overwritten.**
        show_fit_report : bool, optional
            If `True` (default) the detailed lmfit fit report is printed.
        show_shape_err_fits : bool, optional, default: True
            If `True`, plots of all fits performed for the peak-shape
            uncertainty evaluation are shown.

        Notes
        -------
        Updates peak properties dataframe with peak properties obtained in fit.

        """
        if fit_model is None:
            fit_model = self.fit_model
            if self.fit_model is None:
                raise Exception(
                        "No fit model found. Either perform a peak-shape "
                        "calibration upfront with determine_peak_shape() or "
                        "define a model with the `fit_model` argument.")
        if x_fit_range is None:
            x_fit_range = self.default_fit_range

        if peak_indeces != []: # get fit range from specified peak indeces
            peak_indeces.sort()
            if x_fit_cen is not None:
                raise Exception(
                        "Either select peaks to fit with `peak_indeces` OR by "
                        "manually setting the fit range with `x_fit_cen`. If "
                        "none of the above are specified all peaks are fitted.")
            if peak_indeces[-1] - peak_indeces[0] != len(peak_indeces) - 1:
                raise Exception(
                        "All peaks in `peak_indeces` must be direct neighbours "
                        "To process non-neighbouring peaks, run fit_peaks() "
                        "separately on each group of neighbouring peaks.")
            pos_first_peak = self.peaks[peak_indeces[0]].x_pos
            pos_last_peak = self.peaks[peak_indeces[-1]].x_pos
            x_fit_cen = (pos_last_peak + pos_first_peak)/2
            x_fit_range = x_fit_range + (pos_last_peak - pos_first_peak)
            x_min = x_fit_cen - x_fit_range/2
            x_max = x_fit_cen + x_fit_range/2
        elif x_fit_cen is not None: # fit user-defined mass range
            x_min = x_fit_cen - x_fit_range/2
            x_max = x_fit_cen + x_fit_range/2
        else: # fit full range
            x_min = self.data.index[0]
            x_max = self.data.index[-1]
        peaks_to_fit = [peak for peak in self.peaks if (x_min < peak.x_pos < x_max)]
        peak_indeces = [self.peaks.index(p) for p in peaks_to_fit]

        if index_mass_calib is None and species_mass_calib is not None:
            index_mass_calib = [i for i in range(len(self.peaks)) if
                                species_mass_calib == self.peaks[i].species][0]
            cal_pos = self.peaks[index_mass_calib].x_pos
        elif index_mass_calib is not None and species_mass_calib is not None:
            raise Exception("Definition of mass calibrant peak failed. Define "
                            "EITHER the index OR the species name of the peak "
                            "to use as mass calibrant! ")

        if index_mass_calib is not None and index_mass_calib not in peak_indeces:
            raise Exception("If a mass calibrant is specified its index "
                            "must be contained in `peak_indeces`.")

        # FIT ALL PEAKS
        fit_result = spectrum.peakfit(self, fit_model=fit_model,
                                      cost_func=cost_func,
                                      x_fit_cen=x_fit_cen,
                                      x_fit_range=x_fit_range,
                                      init_pars=init_pars,
                                      vary_shape=vary_shape,
                                      vary_baseline=vary_baseline,
                                      method=method,
                                      fit_kws=fit_kws,
                                      show_plots=show_plots,
                                      show_peak_markers=show_peak_markers,
                                      sigmas_of_conf_band=sigmas_of_conf_band,
                                      error_every=error_every,
                                      plot_filename=plot_filename)

        if index_mass_calib is not None:
            # Update recalibration factor and calibrant properties
            self._update_calibrant_props(index_mass_calib,fit_result)

        # Determine peak-shape errors
        try:
            self._eval_peakshape_errors(peak_indeces=peak_indeces,
                                        fit_result=fit_result, verbose=True,
                                        show_shape_err_fits=show_shape_err_fits)
        except KeyError:
            import warnings
            warnings.warn("Peak-shape error determination failed with KeyError. "
                          "Likely the used fit_model is inconsistent with the "
                          "shape calibration model.", UserWarning)
        self._update_peak_props(peaks_to_fit,fit_result)
        self.show_peak_properties()
        if show_fit_report:
            if cost_func == 'MLE':
                print("The values for chi-squared as well as the parameter "
                      "uncertainties and correlations reported by lmfit below "
                      "should be taken with caution when your MLE fit includes "
                      "bins with low statistics. For details see Notes section "
                      "in the spectrum.peakfit() method documentation.")
            self._show_blinded_report(fit_result)
        # Add results to fit_results list, only overwrite calibrant result if a
        # recalibration has been performed with this method
        for p in peaks_to_fit:
            if self.peaks.index(p) != self.index_mass_calib: # non-calib peak
                self.fit_results[self.peaks.index(p)] = fit_result
            elif index_mass_calib is not None: # new recalibration performed
                self.fit_results[self.peaks.index(p)] = fit_result


    def parametric_bootstrap(self, fit_result, peak_indeces=[],
                             N_spectra=1000, n_cores=-1, show_hists=False):
        """Get statistical and area uncertainties via resampling from best-fit
        PDF.

        **This method is primarily for internal usage.**

        This method provides bootstrap estimates of the statistical errors and
        peak area errors by evaluating the scatter of peak centroids and areas
        in fits of many simulated spectra. The simulated spectra are created by
        sampling events from the best-fit PDF asociated with `fit_result`
        (parametric bootstrap). Refined errors are calculated for each peak
        individually by taking the sample standard deviations of the obtained
        peak centroids and areas.

        *All peaks for which refined errors are to be evaluated must belong to
        the same lmfit ModelResult `fit_result`. Even if refined stat. errors
        are only to be extracted for a subset of the peaks contained in
        `fit_result` (as specified with `peak_indeces`), fits will be
        re-performed over the same x-range as `fit_result`.*

        Parameters
        ----------
        fit_result : :class:`lmfit.model.ModelResult`
            Fit result object to evaluate statistical errors for.
        peak_indeces : list, optional
            List containing indeces of peaks to determine refined stat. errors
            for, e.g. to evaluate peak-shape error of peaks 1 and 2 use
            ``peak_indeces=[1,2]``. Listed peaks must be included in
            `fit_result`. Defaults to all peaks contained in `fit_result`.
        N_spectra : int, optional
            Number of simulated spectra to fit. Defaults to 1000, which
            typically yields statistical uncertainty estimates with a relative
            precision of a few percent.
        n_cores : int, optional
            Number of CPU cores to use for parallelized fitting of simulated
            spectra. When set to `-1` (default) all available cores are used.
        show_hists : bool, optional, default: False
            If `True`, histograms of the obtained peak centroids and areas are
            shown. Black vertical lines indicate the best-fit values obtained
            from the measured data.

        Returns
        -------
        :class:`numpy.ndarray`, :class:`numpy.ndarray`
            Array with statistical errors [u], array with area errors [u]
            Array elements correspond to the results for the peaks selected in
            `peak_indeces` (in ascending order). If `peak_indeces` has not been
            specified it defaults to the indeces of all peaks contained in
            `fit_result`.

        See also
        --------
        :meth:`~spectrum.get_errors_from_resampling`

        """
        bkg_c = fit_result.best_values['bkg_c']
        fit_model = fit_result.fit_model
        cost_func = fit_result.cost_func
        method = fit_result.method
        shape_pars = self.shape_cal_pars
        x_cen = fit_result.x_fit_cen
        x_range = fit_result.x_fit_range
        x = fit_result.x
        y = fit_result.y
        model = fit_result.model
        init_pars = fit_result.init_params
        x_min = x_cen - 0.5*x_range
        x_max = x_cen + 0.5*x_range
        # Get indeces of ALL peaks contained in `fit_result`:
        fitted_peaks = [idx for idx, p in enumerate(self.peaks)
                        if x_min < p.x_pos < x_max]

        if peak_indeces is None:
            peak_indeces = fitted_peaks
        elif not all(ids in fitted_peaks for ids in peak_indeces):
            raise Exception("Not all peaks referenced in `peak_indeces` are "
                            "contained in `fit_result`.")

        # Collect ALL peaks, peak centroids and amplitudes of fit_result
        mus = []
        amps = []
        for idx in fitted_peaks:
            pref = 'p{0}_'.format(idx)
            mus.append(fit_result.best_values[pref+'mu'])
            amps.append(fit_result.best_values[pref+'amp'])

        from emgfit.sample import simulate_events
        from numpy import maximum, sqrt, array, log
        from joblib import Parallel, delayed
        from lmfit.model import save_model, load_model
        from lmfit.minimizer import minimize
        import time
        datetime = time.localtime() # get current date and time
        datetime_str = time.strftime("%Y-%m-%d_%H-%M-%S", datetime)
        data_fname = self.input_filename.rsplit('.', 1)[0] # del. file extension
        modelfname = data_fname+datetime_str+"_resampl_model.sav"
        save_model(model, modelfname)
        N_events = int(np.sum(y))
        tiny = np.finfo(float).tiny # get smallest pos. float in numpy
        funcdefs = {'constant': fit.models.ConstantModel,
                    str(fit_model): getattr(fit_models,fit_model)}
        print("Fitting {0} simulated spectra to ".format(N_spectra)+
              "determine statistical mass and peak area errors.")
        def refit():
            # create simulated spectrum data by sampling from fit-result PDF
            df =  simulate_events(shape_pars, mus, amps, bkg_c, N_events, x_min,
                                  x_max, out='hist', bin_cens=x)
            new_x = df.index.values
            new_y = df['Counts'].values
            new_y_err = np.maximum(1,np.sqrt(new_y)) # Poisson (counting) stats
            # Weights for residuals: residual = (fit_model - y) * weights
            new_weights = 1./new_y_err

            model = load_model(modelfname,funcdefs=funcdefs)
            if cost_func  == 'chi-square':
                ## Pearson's chi-squared fit with iterative weights 1/Sqrt(f(x))
                eps = 1e-10 # small number to bound Pearson weights
                def resid_Pearson_chi_square(pars,y_data,weights,x=x):
                    y_m = model.eval(pars,x=x)
                    # Calculate weights for current iteration, add tiny number
                    # `eps` in denominator for numerical stability
                    weights = 1/sqrt(y_m + eps)
                    return (y_m - y_data)*weights
                # Overwrite lmfit's standard least square residuals
                model._residual = resid_Pearson_chi_square
            elif cost_func  == 'MLE':
                # Define sqrt of (doubled) negative log-likelihood ratio (NLLR)
                # summands:
                def sqrt_NLLR(pars,y_data,weights,x=x):
                    y_m = model.eval(pars,x=x) # model
                    # Add tiniest pos. float representable by numpy to arguments
                    # of np.log to smoothly handle divergences for log(arg -> 0)
                    NLLR = 2*(y_m-y_data) + 2*y_data*(log(y_data+tiny)-log(y_m+tiny))
                    ret = sqrt(NLLR)
                    return ret
                # Overwrite lmfit's standard least square residuals
                model._residual = sqrt_NLLR
            else:
                raise Exception("'cost_func' of given `fit_result` not supported.")

            # re-perform fit on simulated spectrum - for performance use only the
            # underlying Minimizer object instead of full lmfit model interface
            try:
                min_res = minimize(model._residual, init_pars, method=method,
                                   args=(new_y,new_weights), kws={'x':x},
                                   scale_covar=False, nan_policy='propagate',
                                   reduce_fcn=None,calc_covar=False)

                # Record centroids and amplitudes pf peaks of interest
                new_mus = []
                new_amps = []
                for idx in peak_indeces:
                    pref = 'p{0}_'.format(idx)
                    mu = min_res.params[pref+'mu']
                    amp = min_res.params[pref+'amp']
                    new_mus.append(mu)
                    new_amps.append(amp)

                return np.array([new_mus, new_amps])

            except ValueError:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('always')
                    msg = str("Fit failed with ValueError (likely NaNs in "
                               "y-model array) and will be excluded.")
                    warnings.warn(msg, UserWarning)
                N_POI = len(peak_indeces)
                return np.array([[np.NaN]*N_POI, [np.NaN]*N_POI])

        from tqdm.auto import tqdm # add progress bar with tqdm
        #results = np.array([refit() for i in tqdm(range(N_spectra))]) # serial
        results = np.array(Parallel(n_jobs=n_cores)
                             (delayed(refit)() for i in tqdm(range(N_spectra))))
        # Force workers to shut down
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)
        os.remove(modelfname) # clean up

        # Format results
        arr_mus, arr_amps = results[:,0], results[:,1]
        transp_mus = arr_mus.transpose()
        transp_amps = arr_amps.transpose()
        stat_errs = np.nanstd(transp_mus,axis=1)
        bin_width = x[1] - x[0] # assume approximately uniform binning
        area_errs = np.nanstd(transp_amps,axis=1)/bin_width

        if show_hists: # plot histograms of centroids and areas
            boxprops = dict(boxstyle='round', facecolor='grey', alpha=0.5)
            for i, idx in enumerate(peak_indeces):
                pref = 'p{0}_'.format(idx)
                best_fit_mu = fit_result.best_values[pref+'mu']
                best_fit_area = fit_result.best_values[pref+'amp']/bin_width # assumes uniform binning
                f, ax = plt.subplots(nrows=1,ncols=2,
                                     figsize=(figwidth*1.5,figwidth*4/18*1.5))
                ax0, ax1 = ax.flatten()
                ax0.set_title("Centroid scatter - peak {0}".format(idx),
                               fontdict={'fontsize':17})
                ax0.hist( (transp_mus[i]-best_fit_mu)*1e06,bins=19)
                text0 = r"$\sigma = {0: .1f}$ $\mu$u".format(stat_errs[i]*1e06)
                ax0.text(0.78, 0.92, text0, transform=ax0.transAxes,
                         fontsize=14, verticalalignment='top', bbox=boxprops)
                ax0.axvline(0, color='black')
                ax0.tick_params(axis='both',labelsize=15)
                ax0.xaxis.get_offset_text().set_fontsize(15)
                ax0.set_xlabel(r"Peak position - best-fit value [$\mu$u]",
                               fontsize=16)
                ax0.set_ylabel("Occurences", fontsize=16)
                ax1.set_title("Area scatter - peak {0}".format(idx),
                              fontdict={'fontsize':17})
                ax1.hist( transp_amps[i]/bin_width,bins=19) # assumes uniform binning
                text1 = r"$\sigma = {0: .1f} $ counts".format(area_errs[i])
                ax1.text(0.7, 0.92, text1, transform=ax1.transAxes, fontsize=14,
                         verticalalignment='top', bbox=boxprops)
                ax1.axvline(best_fit_area, color='black')
                ax1.tick_params(axis='both',labelsize=15)
                ax1.xaxis.get_offset_text().set_fontsize(15)
                ax1.set_xlabel("Peak area [counts]", fontsize=16)
                ax1.set_ylabel("Occurences", fontsize=16)
                plt.show()

        return stat_errs, area_errs


    def get_errors_from_resampling(self, peak_indeces=[], N_spectra=1000,
                                   n_cores=-1, show_hists=False,
                                   show_peak_properties=True):
        """Get statistical and area uncertainties via resampling from best-fit
        PDF and update peak properties therewith.

        This method provides bootstrap estimates of the statistical errors and
        peak area errors by evaluating the scatter of peak centroids and areas
        in fits of many simulated spectra. The simulated spectra are created by
        sampling events from the best-fit PDF asociated with `fit_result`
        (parametric bootstrap). Refined errors are calculated for each peak
        individually by taking the sample standard deviations of the obtained
        peak centroids and areas.

        If the peaks in `peak_indeces` have been fitted separately a parametric
        bootstrap will be performed for each of the different fits.

        Parameters
        ----------
        peak_indeces : list, optional
            List containing indeces of peaks to determine refined stat. and area
            errors for, e.g. to evaluate peak-shape error of peaks 1 and 2 use
            ``peak_indeces=[1,2]``. Defaults to all peaks in the spectrum's
            :attr:`peaks` list.
        N_spectra : int, optional
            Number of simulated spectra to fit. Defaults to 1000, which
            typically yields statistical uncertainty estimates with a relative
            precision of a few percent.
        n_cores : int, optional
            Number of CPU cores to use for parallelized fitting of simulated
            spectra. When set to `-1` (default) all available cores are used.
        show_hists : bool, optional, default: False
            If `True`, histograms of the obtained peak centroids and areas are
            shown. Black vertical lines indicate the best-fit values obtained
            from the measured data.
        show_peak_properties : bool, optional, default: True
            If `True`, the peak properties table is shown after updating the
            statistical and area errors.

        See also
        --------
        :meth:`~spectrum.determine_A_stat_emg`
        :meth:`~spectrum.parametric_bootstrap`

        Notes
        -----
        Only the statistical mass and area uncertainties of ion-of-interest
        peaks are updated. The uncertainties of the mass calibrant and the
        recalibration uncertainty remain unaffected by this method.

        """
        if peak_indeces in ([], None):
            peak_indeces = np.arange(len(self.peaks)).tolist()
        peak_indeces.sort() # ensure ascending order
        # Collect fit_results for peaks in `peak_indeces`
        results = []
        POI = [] # 2D-list with indeces of interest for each fit_result
        for idx in peak_indeces:
            res = self.fit_results[idx]
            if res is None:
                raise Exception("No fit result found for peak {}".format(idx))
            if idx in self.peaks_with_errors_from_resampling:
                # Raise error to avoid incorrect error calculation.
                raise Exception("Peak has already been treated with this "
                                "method. Re-perform peak fit before running "
                                "this method again. ")
            if res not in results:
                results.append(res)
                POI.append([idx])
            else:
                i_res = results.index(res)
                POI[i_res].append(idx)

        # Perform bootstrap for each fit_result and update peak properties
        for res_i, res in enumerate(results):
            stat_errs, area_errs = self.parametric_bootstrap(
                                                        res,
                                                        peak_indeces=POI[res_i],
                                                        N_spectra=N_spectra,
                                                        n_cores=n_cores,
                                                        show_hists=show_hists)

            # Update peak properties with refined stat. and area uncertainties
            for p_i, peak_idx in enumerate(POI[res_i]):
                p = self.peaks[peak_idx]
                pref = 'p{0}_'.format(peak_idx)
                m_ion = p.m_ion
                p.rel_stat_error = stat_errs[p_i]/m_ion
                # Replace simple stat. area errors with resampling errors while
                # preserving the peakshape error contribution to area_error:
                old_stat_area_err = self.calc_peak_area(peak_idx)[1]
                PS_area_err = np.sqrt(p.area_error**2 - old_stat_area_err**2)
                p.area_error = np.round(np.sqrt(area_errs[p_i]**2 +
                                                PS_area_err**2), 2)
                self.peaks_with_errors_from_resampling.append(peak_idx)
            s_indeces = ", ".join(["{}".format(idx) for idx in POI[res_i]])
            print("Updated the statistical and peak area uncertainties of "
                  "peak(s) "+s_indeces+".\n")

        # If calibrant peak is in range, determine a new recal_fac_error from
        # the updated stat. calibrant error
        if self.index_mass_calib in peak_indeces: # Update recal_fac_error
            cal = self.peaks[self.index_mass_calib]
            cal.rel_recal_error = np.sqrt( (cal.m_AME_error/cal.m_AME)**2
                                    + cal.rel_stat_error**2 )/self.recal_fac
            self.rel_recal_error = cal.rel_recal_error
            print("Re-calculated mass recalibration error from updated "
                  "statistical uncertainty of mass calibrant.")
        # Update total mass errors of IOIs including new rel_recal_error
        updated_indeces = []
        for peak_idx in range(len(self.peaks)): # loop over ALL peaks
            if peak_idx == self.index_mass_calib: # skip calibrant
                continue
            try:
                p = self.peaks[peak_idx]
                p.rel_recal_error = self.rel_recal_error
                p.rel_mass_error = np.sqrt(p.rel_stat_error**2 +
                                           p.rel_peakshape_error**2 +
                                           p.rel_recal_error**2)
                p.mass_error_keV = np.round(p.rel_mass_error*p.m_ion*u_to_keV,3)
                if self.index_mass_calib in peak_indeces:
                    # update all peaks due to new recal_fac_error
                    updated_indeces.append(peak_idx)
                elif peak_idx in peak_indeces:
                    # update only peaks with new stat. error
                    updated_indeces.append(peak_idx)
            except TypeError:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("once")
                    msg = str("Could not update total mass error of peak "
                              "{0} due to TypeError.".format(peak_idx))
                    warnings.warn(msg)
        s_updated = ", ".join(str(idx) for idx in updated_indeces)
        print("Updated total mass errors of peaks {}.".format(s_updated))
        self.peaks_with_errors_from_resampling.sort()

        if show_peak_properties:
            print("\nUpdated peak properties table: ")
            self.show_peak_properties()


    def save_results(self, filename, save_plots=True):
        """Write the fit results to a XLSX file and the peak-shape calibration
        to a TXT file.

        Write results to an XLSX Excel file named `<filename>_results.xlsx`
        and save peak-shape calibration parameters to TXT file named
        `<filename>_peakshape_calib.txt`.

        The EXCEL file contains the following three worksheets:

        - general spectrum properties
        - peak properties and images of all obtained fit curves
        - results of the default peakshape-error evaluation in which shape
          parameters are varied by +-1 sigma

        By default, PNG images of all peak fits are saved to PNG-images in both
        linear and logarithmic scale.

        Parameters
        ----------
        filename : string
            Prefix of the files to be saved to (any provided file extensions are
            automatically removed and the necessary .xlsx & .txt extensions are
            appended).
        save_plots : bool, optional
            Whether to save images of all obtained fit curves to separate PNG
            files (default: `True`).


        """
        # Ensure no files are overwritten
        if os.path.isfile(str(filename)+"_results.xlsx"):
            raise Exception("File "+str(filename)+".xlsx already exists. No "
                            "files saved! Choose a different filename or "
                            "delete the original file and re-try.")
        if os.path.isfile(str(filename)+"_peakshape_calib.txt"):
            raise Exception("File "+str(filename)+"_peakshape_calib.txt already"
                            " exists. No files saved! Choose a different"
                            " filename or delete the original file and re-try.")

        # Make DataFrame with spectrum propeties
        spec_data = []
        datetime = time.localtime() # get current date and time
        datetime_string = time.strftime("%Y/%m/%d, %H:%M:%S", datetime)
        spec_data.append(["Saved on",datetime_string])
        import sys
        spec_data.append(["Python version",sys.version_info[0:3]])
        from . import __version__ # get emgfit version
        spec_data.append(["emgfit version",__version__])
        spec_data.append(["lmfit version",fit.__version__])
        from scipy import __version__ as scipy_version
        spec_data.append(["scipy version",scipy_version])
        spec_data.append(["numpy version",np.__version__])
        spec_data.append(["pandas version",pd.__version__])
        attributes = ['input_filename','mass_number','spectrum_comment',
                      'fit_model','red_chi_shape_cal','fit_range_shape_cal',
                      'determined_A_stat_emg','A_stat_emg','A_stat_emg_error',
                      'peaks_with_errors_from_resampling','recal_fac',
                      'rel_recal_error','peaks_with_MC_PS_errors',
                      'blinded_peaks']
        for attr in attributes:
            attr_val = getattr(self,attr)
            spec_data.append([attr,attr_val])
        df_spec = pd.DataFrame(data=spec_data, dtype=str)
        df_spec.set_index(df_spec.columns[0],inplace=True)

        # Make peak properties & eff. mass shifts DataFrames
        dict_peaks = [p.__dict__ for p in self.peaks]
        df_prop = pd.DataFrame(dict_peaks)
        df_prop.index.name = 'Peak index'
        frames = []
        keys = []
        for peak_idx in range(len(self.eff_mass_shifts)):
            if peak_idx == self.index_mass_calib:
                continue # skip mass calibrant
            df = pd.DataFrame.from_dict(self.eff_mass_shifts[peak_idx], orient='index')
            df.columns = ['Value [u]']
            frames.append(df)
            keys.append(str(peak_idx))
        df_eff_mass_shifts = pd.concat(frames, keys=keys)
        df_eff_mass_shifts.index.names = ['Peak index','Parameter']

        # Save lin. and log. plots of all fit results
        from IPython.utils import io
        with io.capture_output() as captured: # suppress output to notebook
            n_res = 0
            last_res = None
            for i, res in enumerate(self.fit_results):
                if res != last_res:
                    self.plot_fit(fit_result=res,
                                  plot_filename=filename+"_fit{}".format(n_res))
                    # Count the different fit results
                    n_res += 1
                last_res = res

        # Define functions to get column widths for auto-cell-width adjustment
        def lwidth(val):
            # get width, limited to <= 8.3 in case of numbers
            try: # string
                return len(val)
            except:  # number
                return min(len(str(val)), 9)
        def get_col_widths(dataframe):
            # Find the maximum length of the index column
            idx_max = max([len(str(s)) for s in dataframe.index.values] +
                                               [len(str(dataframe.index.name))])
            # Find max length of each column (add some width to colnames)
            cols_max = [max(max(lwidth(v) for v in dataframe[col].values),
                            len(str(col))+1)  for col in dataframe.columns]
            # return concatenated lengths of idx and cols
            return np.array([idx_max] + cols_max)

        # Write DataFrames to separate sheets of EXCEL file
        fname = filename+'_results.xlsx'
        with pd.ExcelWriter(fname, engine='xlsxwriter') as writer:
            df_spec.to_excel(writer, sheet_name='Spectrum properties',
                             header=False,)
            df_prop.to_excel(writer, sheet_name='Peak properties')
            df_eff_mass_shifts.to_excel(writer, sheet_name=
                                        'PS errors from +-1 sigma var.')
            workbook = writer.book
            spec_sheet = writer.sheets['Spectrum properties']
            prop_sheet = writer.sheets['Peak properties']
            mshift_sheet = writer.sheets['PS errors from +-1 sigma var.']
            # Adjust column widths
            for i, width in enumerate(get_col_widths(df_spec)):
                spec_sheet.set_column(i, i, width)
            for i, width in enumerate(get_col_widths(df_prop)):
                prop_sheet.set_column(i, i, width)
            for i, width in enumerate(get_col_widths(df_eff_mass_shifts)):
                mshift_sheet.set_column(i, i, width)
            # Mark peaks with stat errors from resampling with green font
            if self.peaks_with_errors_from_resampling not in ([],None):
                green_font = workbook.add_format({'font_color': 'green'})
                for idx in self.peaks_with_errors_from_resampling:
                    prop_sheet.conditional_format(idx+1, 10, idx+1, 10,
                                                  {'type':     'cell',
                                                   'criteria': '>=',
                                                   'value' : 0,
                                                   'format': green_font})
                    prop_sheet.conditional_format(idx+1, 13, idx+1, 13,
                                                  {'type':     'cell',
                                                   'criteria': '>=',
                                                   'value' : 0,
                                                   'format': green_font})
                prop_sheet.write_string(len(self.peaks)+1, 12,
                                        "Stat. errors from resampling",
                                        green_font) # add legend
            # Mark peaks with MC PS errors with blue font
            if self.peaks_with_MC_PS_errors not in ([],None):
                blue_font = workbook.add_format({'font_color': 'blue'})
                for idx in self.peaks_with_MC_PS_errors:
                    prop_sheet.conditional_format(idx+1, 15, idx+1, 15,
                                                  {'type':     'cell',
                                                   'criteria': '>=',
                                                   'value' : 0,
                                                   'format': blue_font})
                prop_sheet.write_string(len(self.peaks)+1, 15,
                                        "Monte Carlo peak-shape errors",
                                        blue_font) # add legend
            for i in range(n_res): # loop over fit results
                fname = filename+"_fit{}".format(i)
                prop_sheet.insert_image(len(self.peaks)+4+56*i,1,
                                        fname+'_log_plot.png',
                                        {'x_scale': 1.0,'y_scale':1.0})
                prop_sheet.insert_image(len(self.peaks)+30+56*i,1,
                                        fname+'_lin_plot.png',
                                        {'x_scale': 1.0,'y_scale':1.0})
        print("Fit results saved to file:",str(filename)+".xlsx")

        # Save peak-shape calibration to TXT-file
        try:
            self.save_peak_shape_cal(filename+"_peakshape_calib")
        except:
            raise

        if not save_plots: # Clean up temporary image files
            for i in range(n_res): # loop over fit results
                fname = filename+"_fit{}".format(i)
                os.remove(fname+'_log_plot.png')
                os.remove(fname+'_lin_plot.png')




################################################################################
