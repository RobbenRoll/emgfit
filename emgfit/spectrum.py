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
        Species strings follow the :-notation (likewise used in MAc).
        Examples: ``'1K39:-1e'``, ``'K39:-e'``, ``'2H1:1O16:-1e'``.
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
    def __init__(self,x_pos,species,m_AME=None,m_AME_error=None):
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
            Species strings follow the :-notation (likewise used in MAc).
            Examples: ``'1K39:-1e'``, ``'K39:-e'``, ``'2H1:1O16:-1e'``.
            **Do not forget to substract the electron from singly-charged
            species**, otherwise the atomic not the ionic mass will be used as
            literature value!
            Alternatively, tentative assigments can be made by adding a ``'?'``
            at the end of the species string (e.g.: ``'Sn100:-1e?'``, ``'?'``, ...).
        m_AME : float [u], optional
            User-defined literature mass value. Overwrites value fetched from
            AME2016. Useful for isomers or to use more up-to-date values.
        m_AME_error : float [u], optional
            User-defined literature mass uncertainty. Overwrites value fetched
            from AME2016.

        """
        self.x_pos = x_pos
        self.species = species # e.g. '1Cs133:-1e or 'Cs133:-e' or '4H1:1C12:-1e'
        self.comment = '-'
        self.m_AME = m_AME #
        self.m_AME_error = m_AME_error
        m, m_error, extrapol, A_tot = get_AME_values(species) # grab AME values
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
        self.atomic_ME_keV = None # (atomic) Mass excess = atomic mass[u] - A    [keV]
        self.mass_error_keV = None
        self.m_dev_keV = None # TITAN - AME [keV]

    def update_lit_values(self):
        """Updates :attr:`m_AME`, :attr:`m_AME_error` and :attr:`extrapolated`
        peak attributes with AME2016 values for specified species."""

        m, m_error, extrapol, A_tot = get_AME_values(self.species) # calculate values for species
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
            print("Peakshape uncertainty:",self.rel_peakshape_error*self.m_ion,"u     (",np.round(self.rel_peakshape_error*self.m_ion*u_to_keV,3),"keV )")
            print("Re-calibration uncertainty:",self.rel_recal_error*self.m_ion,"u     (",np.round(self.rel_recal_error*self.m_ion*u_to_keV,3),"keV )")
            print("Total mass uncertainty (before systematics):",self.rel_mass_error*self.m_ion,"u     (",np.round(self.mass_error_keV,3),"keV )")
            print("Atomic mass excess:",np.round(self.atomic_ME_keV,3),"keV")
            print("TITAN - AME:",np.round(self.m_dev_keV,3),"keV")
            print("χ_sq_red:",np.round(self.red_chi))


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
    red_chi_shape_calib : float
        Reduced chi-squared of peak-shape determination fit.
    fit_range_shape_calib : float [u]
        Fit range used for peak-shape calibration.
    shape_cal_pars : dict
        Model parameter values obtained in peak-shape calibration.
    shape_cal_errors : dict
        Model parameter uncertainties obtained in peak-shape calibration.
    index_mass_calib : int
        Peak index of mass calibrant peak.
    determined_A_stat_emg : bool
        Boolean flag for whether :attr:`A_stat_emg` was determined for this
        spectrum specifically using the :meth:`determine_A_stat_emg` method.
        If True, :attr:`A_stat_emg` was set using :meth:`determine_A_stat_emg`,
        otherwise the default value `emgfit.config.A_stat_emg_default`
        from the :mod:`~emgfit.config` module was used.
        For more details see docs of :meth:`determine_A_stat_emg` method.
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
    eff_mass_shifts_pm : :class:`numpy.ndarray` of dict
        Effective mass shift obtained in peak-shape uncertainty evaluation by
        varying each shape parameter by plus and minus 1 standard deviation,
        respectively. The mass shifts are effective in the sense that they are
        corrected for the corresponding shifts of the calibrant peak centroid.
        The `eff_mass_shifts_pm` array contains a dictionary for each peak;
        the dictionaries have the following structure:
        {'<shape param. name> eff. mass shift pm' :
        [<eff. mass shift for shape param. value +1 sigma>,
        <eff. mass shift for shape param. value -1 sigma>], ...}
        For the mass calibrant the dictionary holds the absolute shifts of the
        calibrant peak centroid (`calibrant centroid shift pm`). For more
        details see docs of :meth:`_eval_peakshape_errors`.
    eff_mass_shifts : :class:`numpy.ndarray` of dict
        Maximal effective mass shifts for each peak obtained in peak-shape
        uncertainty evaluation by varying each shape parameter by plus and minus
        1 standard deviation and only keeping the shift with the larger absolute
        magnitude. The `eff_mass_shifts` array contains a dictionary for each
        peak; the dictionaries have the following structure:
        {'<shape param. name> eff. mass shift' : [<maximal eff. mass shift>],...}
        For the mass calibrant the dictionary holds the absolute shifts of the
        calibrant peak centroid (`calibrant centroid shift`). For more
        details see docs of :meth:`_eval_peakshape_errors`.
    peaks : list of :class:`peak`
        List containing all peaks associated with the spectrum sorted by
        ascending mass. The index of a peak within the `peaks` list is referred
        to as the ``peak_index``.
    fit_results : list of :class:`lmfit.model.ModelResult`
        List containing fit results (:class:`lmfit.model.ModelResult` objects)
        for peaks associated with spectrum.
    data : :class:`pandas.DataFrame`
        Histogrammed spectrum data.
    mass_number : int
        Atomic mass number associated with central bin of spectrum.
    default_fit_range : float
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
            if True, shows a plot of full spectrum with vertical markers for
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
        intended for internal use.

	    """
        if filename is not None:
            data_uncut = pd.read_csv(filename,header=None,names=['Mass [u]', 'Counts'],
                                     skiprows=skiprows,delim_whitespace=True,index_col=False,dtype=float)
            data_uncut.set_index('Mass [u]',inplace =True)
            self.input_filename = filename
        elif df is not None:
            data_uncut = df
        else:
            raise Exception("ERROR: Import failed, since input data was neither specified with `filename` nor `df`.")
        self.spectrum_comment = '-'
        self.fit_model = None
        self.red_chi_shape_calib = None
        self.fit_range_shape_calib = None
        self.shape_cal_pars = None
        self.shape_cal_errors = []
        self.index_mass_calib = None
        self.determined_A_stat_emg = False
        self.A_stat_emg = A_stat_emg_default # initialize at default
        self.A_stat_emg_error = None
        self.recal_fac = 1.0
        self.rel_recal_error = None
        self.recal_facs_pm = None
        self.eff_mass_shifts_pm = None
        self.eff_mass_shifts = None
        self.peaks = [] # list containing peaks associated with spectrum
        self.fit_results = [] # list containing fit results of all peaks
        if m_start or m_stop: # cut input data to specified mass range
            self.data = data_uncut.loc[m_start:m_stop]
            plot_title = 'Spectrum with start and stop markers'
        else:
            self.data = data_uncut # dataframe containing mass spectrum data
            plot_title = 'Spectrum (fit full range)'
        # Set `mass_number` using median of mass bins after cutting spectrum and
        # round to closest integer:
        self.mass_number = int(np.round(self.data.index.values[int(len(self.data)/2)]))
        self.default_fit_range = 0.01*(self.mass_number/100)
        if show_plot:
            plt.rcParams.update({"font.size": 16})
            fig  = plt.figure(figsize=(20,8))
            plt.title(plot_title)
            data_uncut.plot(ax=fig.gca())
            plt.vlines(m_start,0,1.2*max(self.data['Counts']))
            plt.vlines(m_stop,0,1.2*max(self.data['Counts']))
            plt.yscale('log')
            plt.ylabel('Counts')
            plt.show()


    def add_spectrum_comment(self,comment,overwrite=False):
        """Add a general comment to the spectrum.

        By default the `comment` argument will be appended to the end of the current
        :attr:`spectrum_comment` attribute. If `overwrite` is set to ``True``
        the current :attr:`spectrum_comment` is overwritten with `comment`.

        Parameters
        ----------
        comment : str
            Comment to add to spectrum.
        overwrite : bool
            If True, the current :attr:`spectrum_comment` attribute will be
            overwritten with `comment`, else `comment` is appended to the end of
            :attr:`spectrum_comment`.

        Notes
        -----
        The :attr:`spectrum_comment` will be included in the output file storing all fit
        results and can hence be useful to pass on information for the
        post-processing.

        If :attr:`spectrum_comment` is '-' (default value) it is always
        overwritten with `comment`.

        See also
        --------
        :meth:`add_peak_comment`

        """
        try:
            if self.spectrum_comment == '-' or self.spectrum_comment is None or overwrite:
                self.spectrum_comment = comment
            else:
                self.spectrum_comment = self.spectrum_comment+comment
            print("Spectrum comment was changed to: ",self.spectrum_comment)
        except TypeError:
            print("ERROR: 'comment' argument must be given as type string.")
            return


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
            raise ValueError("Window must be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='valid')
        return y[int(window_len/2+1):-int(window_len/2-1)]


    def plot(self, peaks=None, title="", ax=None, yscale='log', vmarkers=None,
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
        ax : :class:`matplotlib.pyplot.axes`, optional
            Axes object to plot onto.
        yscale : str, optional
            Scale of y-axis (``'lin'`` for logarithmic, ``'log'`` for
            logarithmic), defaults to ``'log'``.
        vmarkers : list of float [u], optional
            List with mass positions [u] to add vertical markers at.
        thres : float, optional
            y-level to add horizontal marker at (e.g. for indicating set
            threshold in peak detection).
        ymin : float, optional
            Lower bound of y-range to plot.
        xmin, xmax : float [u], optional
            Lower/upper bound of mass range to plot.

        """
        if peaks is None:
            peaks = self.peaks
        data = self.data # get spectrum data stored in dataframe 'self.data'
        ymax = data.max()[0]
        data.plot(figsize=(20,6),ax=ax)
        plt.yscale(yscale)
        plt.ylabel('Counts')
        plt.title(title)
        try:
            plt.vlines(x=vmarkers,ymin=0,ymax=data.max())
        except TypeError:
            pass
        if yscale == 'log':
            #x_idx = np.argmin(np.abs(data.index.values - p.x_pos)) # set ymin = data.iloc[x_idx] to get peak markers starting at peak max.
            for p in peaks:
                plt.vlines(x=p.x_pos,ymin=0,ymax=1.05*ymax,linestyles='dashed')
                plt.text(p.x_pos, 1.21*ymax, peaks.index(p), horizontalalignment='center', fontsize=12)
            if ymin:
                plt.ylim(ymin,2*ymax)
            else:
                plt.ylim(0.1,2*ymax)
        else:
            #x_idx = np.argmin(np.abs(data.index.values - p.x_pos)) # set ymin = data.iloc[x_idx] to get peak markers starting at peak max.
            for p in peaks:
                plt.vlines(x=p.x_pos,ymin=0,ymax=1.03*ymax,linestyles='dashed')
                plt.text(p.x_pos, 1.05*ymax, peaks.index(p), horizontalalignment='center', fontsize=12)
            if ymin:
                plt.ylim(ymin,1.1*ymax)
            else:
                plt.ylim(0,1.1*ymax)

        if thres:
            plt.hlines(y=thres,xmin=data.index.min(),xmax=data.index.max())
        plt.xlim(xmin,xmax)
        plt.show()


    ##### Define static routine for plotting spectrum data stored in dataframe df (only for internal use within this class)
    @staticmethod
    def _plot_df(df,title="",ax=None,yscale='log',peaks=None,vmarkers=None,thres=None,ymin=None,xmin=None,xmax=None):
        """Plots spectrum data stored in :class:`pandas.DataFrame` `df`.

        **Intended for internal use.**

        Optionally with peak markers if:
        1. single or multiple x_pos are passed to `vmarkers`, OR
        2. list of peak objects is passed to `peaks`.

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            Spectrum data to plot.
        ax : :class:`matplotlib.pyplot.axes`, optional
            Axes object to plot onto.
        yscale : str, optional
            Scale of y-axis (``'lin'`` for logarithmic, ``'log'`` for
            logarithmic), defaults to ``'log'``.
        peaks : list of :class:`peaks`, optional
            List of :class:`peaks` to show peak markers for.
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
        :meth:`plot`
        :meth:`plot_fit`
        :meth:`plot_fit_zoom`

        """
        df.plot(figsize=(20,6),ax=ax)
        plt.yscale(yscale)
        plt.ylabel('Counts')
        plt.title(title)
        try:
            plt.vlines(x=vmarkers,ymin=0,ymax=df.max())
        except TypeError:
            pass
        try:
            li_x_pos = [p.x_pos for p in peaks]
            plt.vlines(x=li_x_pos,ymin=0,ymax=df.max())
        except TypeError:
            pass
        if thres:
            plt.hlines(y=thres,xmin=df.index.min(),xmax=df.index.max())
        if ymin:
            plt.ylim(ymin,)
        plt.xlim(xmin,xmax)
        plt.show()


    ##### Define peak detection routine
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
            to ``True``) to ensure that the detection threshold is set properly.
        plot_smoothed_spec : bool, optional
            If ``True`` a plot with the original and the smoothed spectrum is
            shown.
        plot_2nd_deriv : bool, optional
            If ``True`` a plot with the scaled, inverted second derivative of
            the smoothed spectrum is shown.
        plot_detection_result : bool, optional
            If ``True`` a plot of the spectrum with markers for the detected
            peaks is shown.

        Notes
        -----
        For details on the smoothing, see docs of :meth:`_smooth` by calling:

        >>> help(emgfit.spectrum._smooth)

        See also
        --------
        :meth:`add_peak`
        :meth:`remove_peak`

        """
        # Smooth spectrum (moving average with window function)
        data_smooth = self.data.copy()
        data_smooth['Counts'] = spectrum._smooth(self.data['Counts'].values,window_len=window_len,window=window)
        if plot_smoothed_spec:
            # Plot smoothed and original spectrum
            ax = self.data.plot(figsize=(20,6))
            data_smooth.plot(figsize=(20,6),ax=ax)
            plt.title("Smoothed spectrum")
            ax.legend(["Raw","Smoothed"])
            plt.ylim(0.1,)
            plt.yscale('log')
            plt.ylabel('Counts')
            plt.show()

        # Second derivative
        data_sec_deriv = data_smooth.iloc[1:-1].copy()
        for i in range(len(data_smooth.index) - 2):
            scale = 1/(data_smooth['Counts'].iloc[i+1]+10) # scale data to decrease y range
            #dm = data_smooth.index[i+1]-data_smooth.index[i] # use dm in denominator of deriv if realistic units are desired
            data_sec_deriv['Counts'].iloc[i] = scale*(data_smooth['Counts'].iloc[i+1] - 2*data_smooth['Counts'].iloc[i] + data_smooth['Counts'].iloc[i-1])/1**2 # Used second order central finite difference
            # data_sec_deriv['Counts'].iloc[i] = scale*(data_smooth['Counts'].iloc[i+2] - 2*data_smooth['Counts'].iloc[i+1] + data_smooth['Counts'].iloc[i])/1**2    # data_sec_deriv = data_smooth.iloc[0:-2].copy()
        if plot_2nd_deriv:
            self._plot_df(data_sec_deriv,title="Scaled second derivative of spectrum - set threshold indicated",yscale='linear',thres=-thres)

        # Take only negative part of re-scaled second derivative and invert
        data_sec_deriv_mod = data_smooth.iloc[1:-1].copy()
        for i in range(len(data_smooth.index) - 2):
            scale = -1/(data_smooth['Counts'].iloc[i]+10) # scale data to decrease y range
            # scale = -1/(data_smooth['Counts'].iloc[i+1]+10) # scale data to decrease y range
            value = scale*(data_smooth['Counts'].iloc[i+1] - 2*data_smooth['Counts'].iloc[i] + data_smooth['Counts'].iloc[i-1])/1**2 # Used (second order central finite difference)
            #value = scale*(data_smooth['Counts'].iloc[i+2] - 2*data_smooth['Counts'].iloc[i+1] + data_smooth['Counts'].iloc[i])/1**2 # Used (second order forward finite difference) # data_sec_deriv_mod = data_smooth.iloc[:-2].copy()
            if value > 0:
                data_sec_deriv_mod['Counts'].iloc[i] = value
            else:
                data_sec_deriv_mod['Counts'].iloc[i] = 0

        bin_width = self.data.index[1] - self.data.index[0]
        width_in_bins = int(width/bin_width) # width in units of bins, the prefactor is empirically determined and corrects for the width difference of the peak and its 2nd derivative
        peak_find = sig.find_peaks(data_sec_deriv_mod['Counts'].values,height=thres,width=width_in_bins)
        li_peak_pos = data_sec_deriv_mod.index.values[peak_find[0]]
        #peak_widths = sig.peak_widths(data_sec_deriv_mod['Counts'].values,peak_find[0])
        if plot_2nd_deriv:
            self._plot_df(data_sec_deriv_mod,title="Negative part of scaled second derivative, inverted - set threshold indicated",thres=thres,vmarkers=li_peak_pos,ymin=0.1*thres)

        # Create list of peak objects
        for x in li_peak_pos:
            p = peak(x,'?') # instantiate new peak
            self.peaks.append(p)
            self.fit_results.append(None)

        # Plot raw spectrum with detected peaks marked
        if plot_detection_result:
            self.plot(peaks=self.peaks,title="Spectrum with detected peaks marked")
            print("Peak properties table after peak detection:")
            self.show_peak_properties()


    def add_peak(self,x_pos,species="?",m_AME=None,m_AME_error=None,verbose=True):
        """Manually add a peak to the spectrum's :attr:`peaks` list.

        The position of the peak must be specified with the `x_pos` argument.
        If the peak's ionic species is provided with the `species` argument
        the corresponding AME literature values will be added to the :attr:`peak`.
        Alternatively, user-defined literature values can be provided with the
        `m_AME` and `m_AME_error` arguments. This option is helpful for isomers
        or in case of very recent measurements that haven't entered the AME yet.

        Parameters
        ----------
        x_pos : float [u]
            Position of peak to be added.
        species : str, optional
            :attr:`species` label for peak to be added following the :-notation
            (likewise used in MAc). If assigned, :attr:`peak.m_AME`,
            :attr:`peak.m_AME_error` & :attr:`peak.extrapolated` are
            automatically updated with the corresponding AME literature values.
        m_AME : float [u], optional
            User-defined literature mass for peak to be added. Overwrites pre-
            existing :attr:`peak.m_AME` value.
        m_AME_error : float [u], optional
            User-defined literature mass uncertainty for peak to be added.
            Overwrites pre-existing :attr:`peak.m_AME_error`.
        verbose : bool, optional, default: ``True``
            If ``True``, a message is printed after successful peak addition.
            Intended for internal use only.

        Note
        ----
        Adding a peak will shift the peak_indeces of all peaks at higher masses
        by ``+1``.

        See also
        --------
        :meth:`detect_peaks`
        :meth:`remove_peak`

        """
        p = peak(x_pos,species,m_AME=m_AME,m_AME_error=m_AME_error) # instantiate new peak
        if m_AME is not None: # set mass number to closest integer of m_AME value
            p.A = int(np.round(m_AME,0))
        self.peaks.append(p)
        self.fit_results.append(None)
        ##### Helper function for sorting list of peaks by marker positions 'x_pos'
        def sort_x(peak):
            return peak.x_pos
        self.peaks.sort(key=sort_x) # sort peak positions in ascending order
        if verbose:
            print("Added peak at ",x_pos," u")


    def remove_peaks(self,peak_indeces=None,x_pos=None,species="?"):
        """Remove specified peak(s) from the spectrum's :attr:`peaks` list.

        Select the peak to be removed by specifying either the respective
        `peak_index`, `species` label or peak marker position `x_pos`. To remove
        multiple peaks at once, pass a list to one of the above arguments.

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

        Notes
        -----
        The current :attr:`peaks` list can be viewed by calling the
        :meth:`~spectrum.show_peak_properties` spectrum method.

        Added in version 0.2.0 (as successor method to `remove_peak`)

        """
        # Get indeces of peaks to remove
        if peak_indeces is not None:
            indeces = np.atleast_1d(peak_indeces)
        elif species is not "?":
            peaks = self.peaks
            indeces = [i for i in range(len(peaks)) if species == peaks[i].species]
        elif x_pos:
            indeces = [i for i in range(len(self.peaks)) if np.round(x_pos,6) == np.round(self.peaks[i].x_pos,6)]
        for i in indeces:
            try:
                rem_peak = self.peaks.pop(i)
                self.fit_results.pop(i)
                print("Removed peak at ",rem_peak.x_pos," u")
            except:
                print("Removal of peak {0} failed!".format(i))
                raise
                # TODO: Revert previous peak removals if an error is occurs


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
        warnings.simplefilter('default')
        msg = str("remove_peak is deprecated in v0.1.1 and will likely be "
                  "removed in future versions, use remove_peaks instead!")
        warnings.warn(msg, PendingDeprecationWarning)
        self.remove_peaks(peak_indeces=peak_index,x_pos=x_pos,species=species)


    def show_peak_properties(self):
        """Print properties of all peaks in :attr:`peaks` list.

        """
        dict_peaks = [p.__dict__ for p in self.peaks]
        df_prop = pd.DataFrame(dict_peaks)
        display(df_prop)


    def assign_species(self,species,peak_index=None,x_pos=None):
        """Assign species label(s) to a single or all peaks.

        If no single peak is selected with `peak_index` or `x_pos`, a list with
        species names for all peaks in the peak list must be passed to
        `species`. For already specified or unkown species ``None`` must be
        inserted as a placeholder. See `Notes` and `Examples` sections below for
        details on usage.

        Parameters
        ----------
        species : str or list of str
            The species name (or list of name strings) to be assigned to the
            selected peak (or to all peaks). For unkown or already assigned
            species, ``None`` should be inserted as placeholder at the
            corresponding position in the `species` list. :attr:`species` names
            must follow the :-notation.                                         #TODO: Link to :-notation page
        peak_index : int, optional
            Index of single peak to assign `species` name to.
        x_pos : float [u], optional
            :attr:`x_pos` of single peak to assign `species` name to. Must be
            specified up to 6th decimal.

        Notes
        -----
        - Assignment of a single peak species:
          select peak by specifying peak position `x_pos` (up to 6th decimal) or
          `peak_index` argument (0-based! Check for peak index by calling
          :meth:show_peak_properties() method of spectrum object).

        - Assignment of multiple peak species:
          Nothing should be passed to the 'peak_index' and 'x_pos' arguments.
          Instead the user specifies a list of the new species strings to the
          `species` argument (if there's N detected peaks, the list must have
          length N). Former species assignments can be kept by inserting blanks
          at the respective position in the `species` list, otherwise former
          species assignments are overwritten, also see examples below for usage.

        Examples
        --------
        Assign the peak with peak_index 2 (third-lowest-mass peak) as '1Cs133:-1e',
        leave all other peaks unchanged:

        >>> import emgfit as emg
        >>> spec = emg.spectrum(<input_file>) # mock code for foregoing data import
        >>> spec.detect_peaks() # mock code for foregoing peak detection
        >>> spec.assign_species('1Cs133:-1e',peak_index = 2)

        Assign multiple peaks:

        >>> import emgfit as emg
        >>> spec = emg.spectrum(<input_file>) # mock code for foregoing data import
        >>> spec.detect_peaks() # mock code for foregoing peak detection
        >>> spec.assign_species(['1Ru102:-1e', '1Pd102:-1e', 'Rh102:-1e?', None,'1Sr83:1F19:-1e', '?'])

        This assigns the species of the first, second, third and fourth peak
        with the repsective labels in the specified list and fetches their AME
        values. The `'?'` in the ``'Rh102:-1e?'`` argument indicates a tentative
        species assignment, literature values will not be calculated for this
        peak. The ``None`` argument leaves the species assignment of the 4th
        peak unchanged. The ``'?'`` argument overwrites any former species
        assignments to the highest-mass peak and marks the peak as unidentified.

        """
        try:
            if peak_index is not None:
                p = self.peaks[peak_index]
                p.species = species
                p.update_lit_values() # overwrite m_AME, m_AME_error and extrapolated attributes with AME values for specified species
                print("Species of peak",peak_index,"assigned as",p.species)
            elif x_pos:
                i = [i for i in range(len(self.peaks)) if  np.round(x_pos,6) == np.round(self.peaks[i].x_pos,6)][0] # select peak at position 'x_pos'
                p = self.peaks[i]
                p.species = species
                p.update_lit_values() # overwrite m_AME, m_AME_error and extrapolated attributes with AME values for specified species
                print("Species of peak",i,"assigned as",p.species)
            elif len(species) == len(self.peaks) and peak_index is None and x_pos is None: # assignment of multiple species
                for i in range(len(species)):
                    species_i = species[i]
                    if species_i: # skip peak if 'None' given as argument
                        p = self.peaks[i]
                        p.species = species_i
                        p.update_lit_values() # overwrite m_AME, m_AME_error and extrapolated attributes with AME values for specified species
                        print("Species of peak",i,"assigned as",p.species)
            else:
                raise Exception('ERROR: Species assignment failed. Check method documentation for details on peak selection.\n')
        except:
            print('Errors occured in peak assignment!')
            raise


    def add_peak_comment(self,comment,peak_index=None,x_pos=None,species="?",
                         overwrite=False):
        """Add a comment to a peak.

        By default the `comment` argument will be appended to the end of the
        current :attr:`peak.comment` attribute (if the current comment is '-' it
        is overwritten by the `comment` argument). If `overwrite` is set ``True``,
        the current :attr:`peak.comment` is overwritten with the 'comment' argument.

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
            If ``True`` the current peak :attr:`comment` will be overwritten
            by `comment`, else `comment` is appended to the end of the current
            peak :attr:`comment`.

        Note
        ----
        The shape and mass calibrant peaks are automatically marked during the
        shape and mass calibration by inserting the protected flags ``'shape calibrant'``,
        ``'mass calibrant'`` or ``'shape and mass calibrant'`` into their peak
        comments. When user-defined comments are added to these peaks, it is
        ensured that the protected flags cannot be overwritten. **The above
        shape and mass calibrant flags should never be added to comments
        manually by the user!**

        """
        if peak_index is not None:
            pass
        elif species != "?":
            peak_index = [i for i in range(len(self.peaks)) if species == self.peaks[i].species][0] # select peak with species label 'species'
        elif x_pos is not None:
            peak_index = [i for i in range(len(self.peaks)) if np.round(x_pos,6) == np.round(self.peaks[i].x_pos,6)][0] # select peak at position 'x_pos'
        else:
            raise Exception("\nERROR: Peak specification failed. Check method documentation for details on peak selection.\n")
        peak = self.peaks[peak_index]

        protected_flags = ('shape calibrant','shape & mass calibrant','mass calibrant') # item order matters for comment overwriting!
        try:
            if any(s in comment for s in ('shape calibrant','mass calibrant','shape & mass calibrant')):
                print("ERROR: 'shape calibrant','mass calibrant' and 'shape & mass calibrant' are protected flags. User-defined comments must not contain these flags. Re-phrase comment argument!")
                return
            if peak.comment == '-' or peak.comment is None:
                peak.comment = comment
            elif overwrite:
                if any(s in peak.comment for s in protected_flags):
                    print("WARNING: The protected flags 'shape calibrant','mass calibrant' or 'shape & mass calibrant' cannot be overwritten.")
                    flag = [s for s in protected_flags if s in peak.comment][0]
                    peak.comment = peak.comment.replace(peak.comment,flag+', '+comment)
                else:
                    peak.comment = comment
            else:
                peak.comment = peak.comment+comment
            print("Comment of peak",peak_index,"was changed to: ",peak.comment)
        except TypeError:
            raise Exception("TYPE ERROR: 'comment' argument must be given as type string.")


    def _add_peak_markers(self,yscale='log',ymax=None,peaks=None):
        """Internal function for adding peak markers to current figure object.

        Place this function inside spectrum methods as ``self._add_peak_markers(...)``
        between ``plt.figure()`` and ``plt.show()``. Only for use on already
        fitted spectra!

        Parameters
        ----------
        yscale : str, optional
            Scale of y-axis, either 'lin' or 'log'.
        ymax : float
            Maximal y-value of spectrum data to plot. Used to set y-limits.
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
                plt.vlines(x=p.x_pos,ymin=ymin,ymax=1.38*ymax,linestyles='dashed')
                plt.text(p.x_pos, 1.5*ymax, self.peaks.index(p), horizontalalignment='center', fontsize=12)
        else:
            for p in peaks:
                x_idx = np.argmin(np.abs(data.index.values - p.x_pos))
                ymin = data.iloc[x_idx]
                plt.vlines(x=p.x_pos,ymin=ymin,ymax=1.14*ymax,linestyles='dashed')
                plt.text(p.x_pos, 1.16*ymax, self.peaks.index(p), horizontalalignment='center', fontsize=12)


    def plot_fit(self,fit_result=None,plot_title=None,show_peak_markers=True,
                 sigmas_of_conf_band=0,x_min=None,x_max=None,plot_filename=None):
        """Plot entire spectrum with fit curve in logarithmic and linear y-scale.

        Plots can be saved to a file using the `plot_filename` argument.

        Parameters
        ----------
        fit_result : :class:`lmfit.model.ModelResult`, optional, default: ``None``
            Fit result to plot. If ``None``, defaults to fit result of first
            peak in `peaks` (taken from :attr:`fit_results` list).
        plot_title : str or None, optional
            Title of plots. If ``None``, defaults to a string with the fit model
            name and cost function of the `fit_result` to ensure clear indication
            of how the fit was obtained.
        show_peak_markers : bool, optional, default: ``True``
            If ``True``, peak markers are added to the plots.
        sigmas_of_conf_band : int, optional, default: 0
            Coverage probability of confidence band in sigma (only shown in
            log-plot). If ``0``, no confidence band is shown (default).
        x_min, x_max : float [u], optional
            Start and end of mass range to plot. If ``None``, defaults to the
            minimum and maximum of the spectrum's mass :attr:`data` is used.
        plot_filename : str, optional, default: None
            If not ``None``, the plots will be saved to two separate files named
            '<`plot_filename`>_log_plot.png' and '<`plot_filename`>_lin_plot.png'.
            **Caution: Existing files with identical name are overwritten.**

        """
        if x_min is None:
            x_min = self.data.index.values[0]
        if x_max is None:
            x_max = self.data.index.values[-1]
        # Select peaks in mass range of interest:
        peaks_to_plot = [peak for peak in self.peaks if (x_min < peak.x_pos < x_max)]
        idx_first_peak = self.peaks.index(peaks_to_plot[0])
        if fit_result is None:
           fit_result = self.fit_results[idx_first_peak]
        if plot_title is None:
           plot_title = fit_result.fit_model+' '+fit_result.cost_func+' fit'
        i_min = np.argmin(np.abs(fit_result.x - x_min))
        i_max = np.argmin(np.abs(fit_result.x - x_max))
        y_max_log = max( max(self.data.values[i_min:i_max]), max(fit_result.best_fit[i_min:i_max]) )
        y_max_lin = max( max(self.data.values[i_min:i_max]), max(fit_result.init_fit[i_min:i_max]), max(fit_result.best_fit[i_min:i_max]) )
        weights = 1/fit_result.y_err[i_min:i_max]

        # Plot fit result with logarithmic y-scale
        f1 = plt.figure(figsize=(20,12))
        plt.errorbar(fit_result.x,fit_result.y,yerr=fit_result.y_err,fmt='.',color='royalblue',linewidth=0.5)
        plt.plot(fit_result.x, fit_result.best_fit,'-',color='red',linewidth=2)
        comps = fit_result.eval_components(x=fit_result.x)
        for peak in peaks_to_plot: # loop over peaks to plot
            peak_index = self.peaks.index(peak)
            pref = 'p{0}_'.format(peak_index)
            plt.plot(fit_result.x, comps[pref], '--',linewidth=2)
        if show_peak_markers:
            self._add_peak_markers(yscale='log',ymax=y_max_log,peaks=peaks_to_plot)
        if sigmas_of_conf_band!=0 and fit_result.errorbars == True: # add confidence band with specified number of sigmas
            dely = fit_result.eval_uncertainty(sigma=sigmas_of_conf_band)
            plt.fill_between(fit_result.x, fit_result.best_fit-dely, fit_result.best_fit+dely, color="#ABABAB", label=str(sigmas_of_conf_band)+'-$\sigma$ uncertainty band')
        plt.title(plot_title)
        plt.xlabel('m/z [u]')
        plt.ylabel('Counts per bin')
        plt.yscale('log')
        plt.ylim(0.1, 2*y_max_log)
        plt.xlim(x_min,x_max)
        if plot_filename is not None:
            try:
                plt.savefig(plot_filename+'_log_plot.png',dpi=500)
            except:
                raise
        plt.show()

        # Plot residuals and fit result with linear y-scale
        standardized_residual = (fit_result.best_fit - fit_result.y)/fit_result.y_err
        y_max_res = np.max(np.abs(standardized_residual))
        x_fine = np.arange(x_min,x_max,0.2*(fit_result.x[1]-fit_result.x[0]))
        y_fine = fit_result.eval(x=x_fine)
        f2, axs = plt.subplots(2,1,figsize=(20,12),gridspec_kw={'height_ratios': [1, 2.5]})
        ax0 = axs[0]
        ax0.set_title(plot_title)
        ax0.plot(fit_result.x, standardized_residual,'.',color='royalblue',markersize=8.5,label='residuals')
        #ax0.hlines(1,x_min,x_max,linestyle='dashed')
        ax0.hlines(0,x_min,x_max)
        #ax0.hlines(-1,x_min,x_max,linestyle='dashed')
        ax0.set_ylim(-1.05*y_max_res, 1.05*y_max_res)
        ax0.set_ylabel('Residual / $\sigma$')
        ax1 = axs[1]
        ax1.plot(x_fine, fit_result.eval(params=fit_result.init_params,x=x_fine),linestyle='dashdot',color='green',label='init-fit')
        ax1.plot(x_fine, fit_result.eval(x=x_fine),'-',color='red',linewidth=2,label='best-fit')
        ax1.errorbar(fit_result.x,fit_result.y,yerr=fit_result.y_err,fmt='.',color='royalblue',linewidth=1,markersize=8.5,label='data')
        ax1.set_title('')
        ax1.set_ylim(-0.05*y_max_lin, 1.2*y_max_lin)
        ax1.set_ylabel('Counts per bin')
        for ax in axs:
            ax.legend()
            ax.set_xlim(x_min,x_max)
        if show_peak_markers:
            self._add_peak_markers(yscale='lin',ymax=y_max_lin,peaks=peaks_to_plot)
        plt.xlabel('m/z [u]')
        if plot_filename is not None:
            try:
                plt.savefig(plot_filename+'_lin_plot.png',dpi=500)
            except:
                raise
        plt.show()


    def plot_fit_zoom(self,peak_indeces=None,x_center=None,x_range=0.01,
                      plot_title=None,show_peak_markers=True,
                      sigmas_of_conf_band=0,plot_filename=None):
        """Show logarithmic and linear plots of data and fit curve zoomed to peaks
        or mass range of interest.

        There is two alternatives to define the plots' mass ranges:

        1. Specifying peaks-of-interest with the `peak_indeces`
           argument. The mass range is then automatically chosen to include all
           peaks of interest. The minimal mass range to include around each peak of
           interest can be adjusted using `x_range`.
        2. Specifying a mass range of interest with the `x_center` and `x_range`
           arguments.

        Parameters
        ----------
        peak_indeces : int or list of ints, optional
            Index of single peak or indeces of multiple neighboring peaks to show
            (peaks must belong to the same :attr:`fit_result`).
        x_center : float [u], optional
            Center of manually specified mass range to plot.
        x_range : float [u], optional, default: 0.01
            Width of mass range to plot around 'x_center' or minimal mass range
            to include around each specified peak of interest.
        plot_title : str or None, optional
            Title of plots. If ``None``, defaults to a string with the fit model
            name and cost function of the `fit_result` to ensure clear indication
            of how the fit was obtained.
        show_peak_markers : bool, optional, default: ``True``
            If ``True``, peak markers are added to the plots.
        sigmas_of_conf_band : int, optional, default: 0
            Coverage probability of confidence band in sigma (only shown in
            log-plot). If ``0``, no confidence band is shown (default).
        x_min, x_max : float [u], optional
            Start and end of mass range to plot. If ``None``, defaults to the
            minimum and maximum of the spectrum's mass :attr:`data` is used.
        plot_filename : str or None, optional, default: None
            If not ``None``, the plots will be saved to two separate files named
            '<`plot_filename`>_log_plot.png' and '<`plot_filename`>_lin_plot.png'.
            **Caution: Existing files with identical name are overwritten.**

        """
        if isinstance(peak_indeces,list):
            x_min = self.peaks[peak_indeces[0]].x_pos - x_range/2
            x_max = self.peaks[peak_indeces[-1]].x_pos + x_range/2
        elif type(peak_indeces) == int:
            peak = self.peaks[peak_indeces]
            x_min = peak.x_pos - x_range/2
            x_max = peak.x_pos + x_range/2
        elif x_center is not None:
            x_min = x_center - x_range/2
            x_max = x_center + x_range/2
        else:
            raise Exception("\nMass range to plot could not be determined. Check documentation on method parameters.\n")
        self.plot_fit(x_min=x_min,x_max=x_max,plot_title=plot_title,show_peak_markers=show_peak_markers,sigmas_of_conf_band=sigmas_of_conf_band,plot_filename=plot_filename)


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
            ``'emg12'``, ``'emg33'``, ... - see :mod:`~emgfit.fit_models` module
            for all available fit models).
        init_pars : dict, optional, default: ``None``
            Default initial shape parameters for fit model. If ``None`` the
            default parameters defined in the :mod:`~emgfit.fit_models` module
            will be used after scaling to the spectrum's :attr:`mass_number`.           #TODO: add reference to definition of 'shape parameters'
        vary_shape : bool, optional
            If ``False`` only the amplitude (`amp`) and Gaussian centroid (`mu`)
            model parameters will be varied in the fit. If ``True``, the shape
            parameters (`sigma`, `theta`,`etas` and `taus`) will also be varied.        #TODO: add reference to definition of 'shape parameters'
        vary_baseline : bool, optional
            If ``True`` a varying uniform baseline will be added to the fit
            model as varying model parameter `c_bkg`. If ``False``, the baseline
            parameter `c_bkg` will be kept fixed at 0.

        Notes
        -----
        The initial amplitude for each peak is estimated by taking the counts in
        the bin closest to the peak's :attr:`x_pos` and scaling this number with
        an empirically determined constant and the spectrum's :attr:`mass_number`.

        """
        model = getattr(fit_models,model) # get single peak model from fit_models.py
        mod = fit.models.ConstantModel(independent_vars='x',prefix='bkg_')
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
                this_mod = model(peak_index, peak.x_pos, amp, init_pars=init_pars, vary_shape_pars=vary_shape, index_first_peak=index_first_peak)
            else:
                this_mod = model(peak_index, peak.x_pos, amp, vary_shape_pars=vary_shape, index_first_peak=index_first_peak)
            mod = mod + this_mod
        return mod


    def peakfit(self,fit_model='emg22', cost_func='chi-square', x_fit_cen=None,
                x_fit_range=None, init_pars=None, vary_shape=False,
                vary_baseline=True, method='least_squares', show_plots=True,
                show_peak_markers=True, sigmas_of_conf_band=0,
                plot_filename=None, eval_par_covar=False):
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
            ``'emg33'``, ... - see :mod:`emgfit.fit_models` module for all
            available fit models).
        cost_func : str, optional, default: 'chi-square'
            Name of cost function to use for minimization.

            - If ``'chi-square'``, the fit is performed by minimizing Pearson's
              chi-squared statistic:

              .. math::

                  \\chi^2_P = \\sum_i \\frac{(f(x_i) - y_i)^2}{f(x_i)^2}.

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

        vary_shape : bool, optional, default: ``False``
            If ``False`` peak-shape parameters (`sigma`, `theta`,`etas` and
            `taus`) are kept fixed at their initial values. If ``True`` the
            shared shape parameters are varied (ensuring identical shape
            parameters for all peaks).
        vary_baseline : bool, optional, default: ``True``
            If ``True``, the constant background will be fitted with a varying
            uniform baseline parameter `bkg_c` (initial value: 0.1).
            If ``False``, the baseline parameter `bkg_c` will be fixed to 0.
        method : str, optional, default: `'least_squares'`
            Name of minimization algorithm to use. For full list of options
            check arguments of :func:`lmfit.minimizer.minimize`.
        show_plots : bool, optional
            If ``True`` (default) linear and logarithmic plots of the spectrum
            with the best fit curve are displayed. For details see
            :meth:`spectrum.plot_fit`.
        show_peak_markers : bool, optional
            If ``True`` (default) peak markers are added to the plots.
        sigmas_of_conf_band : int, optional, default: 0
            Confidence level of confidence band around best fit curve in sigma.
            Note that the confidence band is only derived from the uncertainties
            of the parameters that are varied during the fit.
        plot_filename : str, optional, default: None
            If not ``None``, the plots will be saved to two separate files named
            '<`plot_filename`>_log_plot.png' and '<`plot_filename`>_lin_plot.png'.
            **Caution: Existing files with identical name are overwritten.**
        eval_par_covar : bool, optional
            If ``True`` the parameter covariances will be estimated using
            Markov-Chain Monte Carlo (MCMC) sampling. This feature is based on
            `<https://lmfit.github.io/lmfit-py/examples/example_emcee_Model_interface.html>`_.

        Returns
        -------
        :class:`lmfit.model.ModelResult`
            Fit model result.

        Notes
        -----

        In fits with the ``chi-square`` cost function the variance weights
        :math:`w_i` for the residuals are estimated as the square of the model
        predictions: :math:`w_i = 1/\sigma_i = 1/f(x_i)^2`. On each iteration
        the weights are updated with the new values of the model function.

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
        below. Only the peak area errors are calculated using the standard
        errors of the `amp` parameters reported by lmfit.

        .. _`lecture slides by Mark Thompson`: https://www.hep.phy.cam.ac.uk/~thomson/lectures/statistics/Fitting_Handout.pdf

        Besides the asymptotic concergence to a chi-squared distribution
        emgfit's ``MLE`` cost function has a second handy property - all
        summands in the log-likelihood ratio are positive semi-definite:
        :math:`L_i = f(x_i) - y_i + y_i ln\\left(\\frac{y_i}{f(x_i)}\\right) \\geq 0`.
        Exploiting this property, the minimization of the log-likelihood ratio
        can be re-formulated into a least-squares problem:

        .. math::

            L = 2\\sum_i L_i = 2\\sum_i \\left( \\sqrt{ L_i } \\right)^2.


        Instead of minimizing the scalar log-likelihood ratio, the sum-of-squares
        of the square-root of the summands :math:`L_i` in the log-likelihood
        ratio is minimized in emgfit. This facilitates the usage of Scipy's
        highly efficient least-squares optimizers ('least_squares' & 'leastsq')
        and leads to significant speed-ups compared to scalar optimizers such as
        Scipy's 'Nelder-Mead' or 'Powell' methods. By default, emgfit's 'MLE'
        fits are performed with Scipy's 'least_squares' optimizer, a variant of
        a Levenberg-Marquardt algorithm for bound-constrained problems. For more
        details on these optimizers see the docs of
        :func:`lmfit.minimizer.minimize` and :class:`scipy.optimize`.

        See also
        --------
        :meth:`fit_peaks`
        :meth:`fit_calibrant`

        """
        if x_fit_range is None:
            x_fit_range = self.default_fit_range
        if x_fit_cen:
            x_min = x_fit_cen - x_fit_range/2
            x_max = x_fit_cen + x_fit_range/2
            # Cut data to fit range
            df_fit = self.data[x_min:x_max]
            # Select peaks in fit range
            peaks_to_fit = [peak for peak in self.peaks if (x_min < peak.x_pos < x_max)]
        else:
            df_fit = self.data
            x_min = df_fit.index.values[0]
            x_max = df_fit.index.values[-1]
            peaks_to_fit = self.peaks
        if len(peaks_to_fit) == 0:
            raise Exception("Fit failed. No peaks in specified mass range.")
        x = df_fit.index.values
        y = df_fit['Counts'].values
        y_err = np.maximum(1,np.sqrt(y)) # assume Poisson (counting) statistics
        # Weights for residuals: residual = (fit_model - y) * weights
        weights = 1./y_err # np.nan_to_num(1./y_err, nan=0.0, posinf=0.0, neginf=None)

        if init_pars == 'default':
            # Take default params defined in create_default_init_pars() in
            # fit_models.py and re-scale to spectrum's 'mass_number' attribute
            init_params = fit_models.create_default_init_pars(mass_number=self.mass_number)
        elif init_pars is not None:
            init_params = init_pars
        else:
            # Use shape parameters asociated with spectrum unless other
            # parameters have been specified
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

        # Perform fit, print fit report
        if cost_func == 'chi-square':
            ## Pearson's chi-squared fit with iterative weights 1/Sqrt(f(x_i))
            ## Weights have a lower bound of 1
            mod_Pearson = mod
            def resid_Pearson_chi_square(pars,y_data,weights,x=x):
                y_m = mod_Pearson.eval(pars,x=x)
                # Calculate weights for current iteration, non-zero upper
                # bound of 1 implemented for numerical stability:
                weights = 1./np.maximum(1.,np.sqrt(y_m))

                return (y_m - y_data)*weights
            # Overwrite lmfit's standard least square residuals with iterative
            # residuals for Pearson chi-square fit
            mod_Pearson._residual = resid_Pearson_chi_square
            out = mod_Pearson.fit(y, params=pars, x=x, weights=weights,
                                  method=method, scale_covar=False,
                                  nan_policy='propagate')
            y_m = out.best_fit
            # Calculate final weights for plotting
            Pearson_weights = 1./np.maximum(1.,np.sqrt(y_m))
            out.y_err = 1./Pearson_weights
        elif cost_func == 'MLE':
            ## Binned max. likelihood fit using negative log-likelihood ratio
            mod_MLE = mod
            # Define sqrt of (doubled) negative log-likelihood ratio (NLLR)
            # summands:
            tiny = np.finfo(float).tiny # get smallest pos. float in numpy
            def sqrt_NLLR(pars,y_data,weights,x=x):
                y_m = mod_MLE.eval(pars,x=x) # model
                # Define NLLR using np.nan_to_num to prevent non-finite values
                # for (y_m,y_data) = (1,0), (0,0), (0,1)
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
                              method=method, scale_covar=False,
                              calc_covar=False, nan_policy='propagate')
            out.y_err = 1./out.weights
        else:
            raise Exception("Error: Definition of `cost_func` failed!")
        out.x = x
        out.y = y
        out.fit_model = fit_model
        out.cost_func = cost_func
        out.method = method
        out.x_fit_cen = x_fit_cen
        out.x_fit_range = x_fit_range
        out.vary_baseline = vary_baseline
        out.vary_shape = vary_shape
        #out.y_err = 1/out.weights #y_err

        if eval_par_covar:
            print("\n  ### Evaluating parameter covariances using MCMC method")
            ## Add emcee MCMC sampling
            emcee_kws = dict(steps=7000, burn=2500, thin=10, is_weighted=True, progress=True)
            emcee_params = out.params.copy()
            #emcee_params.add('__lnsigma', value=np.log(7.0), min=np.log(1.0), max=np.log(100.0))
            result_emcee = mod.fit(y, x=x, params=emcee_params, weights=weights, method='emcee', nan_policy='omit', fit_kws=emcee_kws)
            fit.report_fit(result_emcee)

            plt.figure(figsize=(12,8))
            plt.plot(x, mod.eval(params=out.params, x=x), label='least_squares', zorder=100)
            result_emcee.plot_fit(data_kws=dict(color='gray', markersize=2))
            plt.yscale("log")
            plt.show()

            ## Check acceptance fraction of emcee
            plt.plot(result_emcee.acceptance_fraction)
            plt.xlabel('walker')
            plt.ylabel('acceptance fraction')
            plt.show()

            ## Plot autocorrelation times of Parameters
            if hasattr(result_emcee, "acor"):
                print("Autocorrelation time for the parameters:")
                print("----------------------------------------")
                for i, p in enumerate(result_emcee.params):
                    print(p, result_emcee.acor[i])

            ## Plot parameter covariances returned by emcee
            import corner
            percentile_range = [0.8]*(out.nvarys)
            fig_corner = corner.corner(result_emcee.flatchain, labels=result_emcee.var_names, truths=list(result_emcee.params.valuesdict().values()),range=percentile_range)
            fig_corner.subplots_adjust(right=2,top=2)
            for ax in fig_corner.get_axes():
                ax.tick_params(axis='both', labelsize=17)
                ax.xaxis.label.set_size(27)
                ax.yaxis.label.set_size(27)

            print("\nmedian of posterior probability distribution")
            print('--------------------------------------------')
            fit.report_fit(result_emcee.params)

            ## Find the maximum likelihood solution
            highest_prob = np.argmax(result_emcee.lnprob)
            hp_loc = np.unravel_index(highest_prob, result_emcee.lnprob.shape)
            mle_soln = result_emcee.chain[hp_loc]
            print("\nMaximum likelihood Estimation")
            print('-----------------------------')
            for ix, param in enumerate(result_emcee.params):
                try:
                    print(param + ': ' + str(mle_soln[ix]))
                except IndexError:
                    pass

            pref = 'p'+str(self.peaks.index(peaks_to_fit[0]))+'_'
            quantiles = np.percentile(result_emcee.flatchain[pref+'mu'], [2.28, 15.9, 50, 84.2, 97.7])
            print("\n\n1 mu spread", 0.5 * (quantiles[3] - quantiles[1]))
            print("2 mu spread", 0.5 * (quantiles[4] - quantiles[0]))

        if show_plots:
            self.plot_fit(fit_result=out, show_peak_markers=show_peak_markers, sigmas_of_conf_band=sigmas_of_conf_band, x_min=x_min, x_max=x_max,plot_filename=plot_filename)

        return out


    def calc_peak_area(self, peak_index, fit_result=None, decimals=2):
        """Calculate peak area (counts in peak) and its error for specified peak.

        The peak area is calculated using the peak's amplitude parameter `amp`
        and the width of the uniform binning of the spectrum. Therefore, the
        peak must have been fitted beforehand. In the case of overlapping peaks
        only the counts within the fit component of the specified peak are
        returned.

        Note
        ----
        This routine assumes the bin width to be uniform across the spectrum.
        The mass binning of a MAc mass spectrum is not perfectly uniform
        (only time bins are uniform, mass bins have a marginal quadratic scaling
        with mass). However, for isobaric species the quadratic term should
        usually be so small that it can safely be neglected.


        Parameters
        ----------
        peak_index : str
            Index of peak of interest.
        fit_result : :class:`lmfit.model.ModelResult`, optional
            Fit result object to use for area calculation. If ``None`` (default)
            use corresponding fit result stored in :attr:`~emgfit.spectrum.spectrum.fit_results` list.
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
                    print('\nWARNING: Area error determination failed with Type error:',err,' \n')
                    pass
        except TypeError or AttributeError:
            print('WARNING: Area error determination failed. Could not get amplitude parameter (`amp`) of peak. Likely the peak has not been fitted successfully yet.')
            raise
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
            raise Exception("Error: No matching fit result found in `fit_results` list. Ensure the peak has been fitted.")

        pars = fit_result.params
        pref = 'p{0}_'.format(peak_index)
        mu = pars[pref+'mu'] # centroid of underlying Gaussian
        x_range = 0.05
        x = np.linspace(mu-x_range/2,mu+x_range/2,10000)
        comps = fit_result.eval_components(x=x)
        y = comps[pref] #fit_result.eval(pars,x=x)
        y_M = max(y)
        i_M = np.argmin(np.abs(y-y_M))
        y_HM = y_M /2
        i_HM1 = np.argmin(np.abs(y[0:i_M]-y_HM))
        i_HM2 = i_M + np.argmin(np.abs(y[i_M:]-y_HM))
        if i_HM1 == 0 or i_HM2 == len(x):
            print("ERROR: FWHM points at boundary, likely a larger x_range needs to be implemented for this function.")
            return
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
            raise Exception("Error: No matching fit result found in `fit_results` list. Ensure the peak has been fitted.")

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
        sigma_EMG = emg_funcs.sigma_emg(fit_result.best_values[pref+'sigma'],fit_result.best_values[pref+'theta'],tuple(li_eta_m),tuple(li_tau_m),tuple(li_eta_p),tuple(li_tau_p))

        return sigma_EMG


    @staticmethod
    def bootstrap_spectrum(df,N_events=None,x_cen=None,x_range=0.02):
        """Create simulated spectrum via bootstrap re-sampling from spectrum `df`.

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
            Histogram with simulated spectrum data from bootstrapping.

        """
        if x_cen:
            x_min = x_cen - x_range/2
            x_max = x_cen + x_range/2
            df = df[x_min:x_max]
        mass_bins = df.index.values
        counts = df['Counts'].values.astype(int)

        # Convert histogrammed spectrum (equivalent to MAc HIST export mode) to
        # list of events (equivalent to MAc LIST export mode)
        orig_events =  np.repeat(mass_bins,counts,axis=0)

        # Create new DataFrame of events by bootstrapping from `orig_events`
        if N_events == None:
            N_events = len(orig_events)
        random_indeces = np.random.randint(0,len(orig_events),N_events)
        events = orig_events[random_indeces]
        df_events = pd.DataFrame(events)

        # Convert list of events back to a DataFrame with histogram data
        bin_centers = df.index.values
        bin_width = df.index.values[1] - df.index.values[0]
        bin_edges = np.append(bin_centers-bin_width/2,bin_centers[-1]+bin_width/2)
        hist = np.histogram(df_events,bins=bin_edges)
        df_new = pd.DataFrame(data=hist[0],index=bin_centers,dtype=float,columns=["Counts"])
        df_new.index.name = "Mass [u]"
        return df_new


    def determine_A_stat_emg(self,peak_index=None,species="?",x_pos=None,
                             x_range=None,N_spectra=1000,fit_model=None,
                             cost_func='MLE',method='least_squares',
                             vary_baseline=True,plot_filename=None):
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

                  \\chi^2_P = \\sum_i \\frac{(f(x_i) - y_i)^2}{f(x_i)^2}.

            - If ``'MLE'``, a binned maximum likelihood estimation is performed
              by minimizing the (doubled) negative log-likelihood ratio:

              .. math::

                  L = 2\\sum_i \\left[ f(x_i) - y_i + y_i ln\\left(\\frac{y_i}{f(x_i)}\\right)\\right]

            For details see `Notes` section of :meth:`peakfit` method documentation.
        method : str, optional, default: `'least_squares'`
            Name of minimization algorithm to use. For full list of options
            check arguments of :func:`lmfit.minimizer.minimize`.
        vary_baseline : bool, optional, default: ``True``
            If ``True``, the constant background will be fitted with a varying
            uniform baseline parameter `bkg_c` (initial value: 0.1).
            If ``False``, the baseline parameter `bkg_c` will be fixed to 0.
        plot_filename : str, optional, default: None
            If not ``None``, the plots will be saved to two separate files named
            '<`plot_filename`>_log_plot.png' and '<`plot_filename`>_lin_plot.png'.
            **Caution: Existing files with identical name are overwritten.**

        Notes
        -----
        Statistical errors of Hyper-EMG peak centroids obey the following
        scaling with the number of counts in the peak `N_counts`:

        .. math::  \\sigma_{stat} = A_{stat,emg} \\frac{FWHM}{\\sqrt{N_{counts}}}

        This routine uses the following method to determine the constant of
        proportionality `A_stat_emg`:

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
        attribute and will be used for all subsequent stat. error determinations.

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
        no_peaks_in_range = len([p for p in self.peaks if (x_cen - x_range/2) <= p.x_pos <= (x_cen + x_range/2)])
        if no_peaks_in_range > 1:
            raise Exception("More than one peak in current mass range. "
                            "This routine only works on well-separated, single "
                            "peaks. Please chose a smaller `x_range`!\n")
        li_N_counts = [10,30,100,300,1000,3000,10000,30000]
        print("Creating synthetic spectra via bootstrap re-sampling and "
              "fitting  them for A_stat determination.")
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
                                                   vary_baseline=vary_baseline,
                                                   init_pars=self.shape_cal_pars,
                                                   show_plots=False)
                    # Record centroid and area of peak 0
                    mus = np.append(mus,fit_result.params['p0_mu'].value)
                    area_i = spec_boot.calc_peak_area(0, fit_result=fit_result, decimals=2)[0]
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
        FWHM_emg_err = FWHM_gauss/FWHM_emg * self.shape_cal_par_errors['sigma']
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
        A_stat_emg_error = np.sqrt( (out.params['amplitude'].stderr/FWHM_emg)**2 + (out.best_values['amplitude']*FWHM_emg_err/FWHM_emg**2)**2 )

        y = std_devs_of_mus/mean_mu
        f = plt.figure(figsize=(15,8))
        plt.title('A_stat_emg determination from bootstrapped spectra - '+fit_model+' '+cost_func+' fits')
        plt.plot(x,y,'o')
        plt.plot(x,out.best_fit/mean_mu)
        plt.plot(x,A_stat_gauss*FWHM_gauss/(np.sqrt(x)*mean_mu),'--')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Peak area [counts]")
        plt.ylabel("Relative statistical uncertainty")
        plt.legend(["Standard deviations of sample means","Stat. error of Hyper-EMG","Stat. error of underlying Gaussian"])
        plt.annotate('A_stat_emg: '+str(np.round(A_stat_emg,3))+' +- '+str(np.round(A_stat_emg_error,3)), xy=(0.7, 0.75), xycoords='axes fraction')
        if plot_filename is not None:
            try:
                plt.savefig(plot_filename+'_A_stat_emg_determination.png',dpi=500)
            except:
                raise
        plt.show()

        self.determined_A_stat_emg = cost_func
        self.A_stat_emg = A_stat_emg
        self.A_stat_emg_error = A_stat_emg_error
        print("A_stat of a Gaussian model:",np.round(A_stat_gauss,3))
        print("Default A_stat_emg for Hyper-EMG models:",np.round(A_stat_emg_default,3))
        print("A_stat_emg for this spectrum's",self.fit_model,"fit model:",np.round(self.A_stat_emg,3),"+-",np.round(self.A_stat_emg_error,3))


    def determine_peak_shape(self, index_shape_calib=None,
                             species_shape_calib=None, fit_model='emg22',
                             cost_func='chi-square', init_pars = 'default',
                             x_fit_cen=None, x_fit_range=None,
                             vary_baseline=True, method='least_squares',
                             vary_tail_order=True, show_fit_reports=False,
                             show_plots=True, show_peak_markers=True,
                             sigmas_of_conf_band=0, plot_filename=None,
                             eval_par_covar=False):
        """Determine optimal peak-shape parameters by fitting the specified
        peak-shape calibrant.

        If `vary_tail_order` is ``True`` (default) an automatic model selection
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
            Species name of the shape-calibrant peak (e.g. ``'K39:-1e'``,           #TODO add ref. to :-Notation
            alternatively, the peak to use can be specified with the
            `index_shape_calib` argument)
        fit_model : str, optional, default: 'emg22'
            Name of fit model to use for shape calibration (e.g. ``'Gaussian'``,
            ``'emg12'``, ``'emg33'``, ... - see :mod:`~emgfit.fit_models` module
            for all available fit models). If the automatic model selection
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

                  \\chi^2_P = \\sum_i \\frac{(f(x_i) - y_i)^2}{f(x_i)^2}.

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
        vary_baseline : bool, optional, default: ``True``
            If ``True``, the background will be fitted with a varying uniform
            baseline parameter `bkg_c` (initial value: 0.1). If ``False``, the
            baseline parameter `bkg_c` will be fixed to 0.
        method : str, optional, default: `'least_squares'`
            Name of minimization algorithm to use. For full list of options
            check arguments of :func:`lmfit.minimizer.minimize`.
        vary_tail_order : bool, optional
            If ``True`` (default), before the calibration of the peak-shape
            parameters an automatized fit model selection is performed. For
            details on the automatic model selection, see `Notes` section below.
            If ``False``, the specified `fit_model` argument is used as model
            for the peak-shape determination.
        show_fit_reports : bool, optional, default: True
            Whether to print fit reports for the fits in the automatic model
            selection.
        show_plots : bool, optional
            If ``True`` (default), linear and logarithmic plots of the spectrum
            and the best fit curve are displayed. For details see
            :meth:`spectrum.plot_fit`.
        show_peak_markers : bool, optional
            If ``True`` (default), peak markers are added to the plots.
        sigmas_of_conf_band : int, optional, default: 0
            Confidence level of confidence band around best fit curve in sigma.
        plot_filename : str, optional, default: None
            If not ``None``, the plots of the shape-calibration will be saved to
            two separate files named '<`plot_filename`>_log_plot.png' and
            '<`plot_filename`>_lin_plot.png'. **Caution: Existing files with
            identical name are overwritten.**
        eval_par_covar : bool, optional
            If ``True`` the parameter covariances will be estimated using
            Markov-Chain Monte Carlo (MCMC) sampling. This feature is based on
            `<https://lmfit.github.io/lmfit-py/examples/example_emcee_Model_interface.html>`_.

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
            index_shape_calib = [i for i in range(len(self.peaks)) if species_shape_calib == self.peaks[i].species][0]
            peak = self.peaks[index_shape_calib]
        else:
            print("\nERROR: Definition of peak shape calibrant failed. Define EITHER the index OR the species name of the peak to use as shape calibrant!\n")
            return
        if init_pars == 'default' or init_pars is None:
            # Take default params defined in create_default_init_pars() in
            # fit_models.py and re-scale to spectrum's 'mass_number' attribute
            init_params = fit_models.create_default_init_pars(mass_number=self.mass_number)
        elif init_pars is not None: # take user-defined values
            init_params = init_pars
        else:
            raise Exception("Error: Definition of initial parameters failed.")
        if x_fit_cen is None:
            x_fit_cen = peak.x_pos
        if x_fit_range is None:
            x_fit_range = self.default_fit_range

        if vary_tail_order == True and fit_model != 'Gaussian':
            print('\n##### Determine optimal tail order #####\n')
            # Fit peak with Hyper-EMG of increasingly higher tail orders and compile results
            # use fit model that produces the lowest chi-square without having eta's compatible with zero within errobar
            li_fit_models = ['Gaussian','emg01','emg10','emg11','emg12','emg21','emg22','emg23','emg32','emg33']
            li_red_chis = np.array([np.nan]*len(li_fit_models))
            li_red_chi_errs = np.array([np.nan]*len(li_fit_models))
            li_eta_flags =np.array([False]*len(li_fit_models)) # list of flags for models with eta's compatible with zero or eta's without error
            for model in li_fit_models:
                try:
                    print("\n### Fitting data with",model,"###---------------------------------------------------------------------------------------------\n")
                    out = spectrum.peakfit(self, fit_model=model, cost_func=cost_func,
                                           x_fit_cen=x_fit_cen, x_fit_range=x_fit_range,
                                           init_pars=init_pars, vary_shape=True,
                                           vary_baseline=vary_baseline, method=method,
                                           show_plots=show_plots,
                                           show_peak_markers=show_peak_markers,
                                           sigmas_of_conf_band=sigmas_of_conf_band)
                    idx = li_fit_models.index(model)
                    li_red_chis[idx] = np.round(out.redchi,2)
                    li_red_chi_errs[idx] =  np.round(np.sqrt(2/out.nfree),2)

                    # Check emg models with tail orders >= 2 for overfitting
                    # (i.e. a eta parameter agress with zero within its error)
                    # and check for existence of parameter uncertainties
                    if model.startswith('emg') and model not in ['emg01','emg10','emg11']:
                        no_left_tails = int(model[3])
                        no_right_tails = int(model[4])
                        # Must use first peak to be fit, since only its shape
                        # parameters are all unconstrained:
                        first_parname = list(out.params.keys())[2]
                        pref = first_parname.split('_')[0]+'_'
                        if no_left_tails > 1:
                            for i in np.arange(1,no_left_tails+1):
                                if not out.errorbars:
                                    print("WARNING: parameter uncertainty determination failed! This tail order will be excluded from selection.") # TO DO: Consider adding eval_uncertainty option here.
                                    # Mark model in order to exclude it below
                                    li_eta_flags[idx] = True
                                    break
                                par_name = pref+"eta_m"+str(i)
                                val = out.params[par_name].value
                                err = out.params[par_name].stderr
                                if val < err:
                                    print("WARNING:",par_name,"=",np.round(val,3),"+-",np.round(err,3)," is compatible with zero within uncertainty.")
                                    print("             This tail order is likely overfitting the data and will be excluded from selection.")
                                    # Mark model in order to exclude it below
                                    li_eta_flags[idx] = True
                        if no_right_tails > 1:
                            for i in np.arange(1,no_right_tails+1):
                                if not out.errorbars:
                                    print("WARNING: parameter uncertainty determination failed! This tail order will be excluded from selection.") # TO DO: Consider adding eval_uncertainty option here.
                                    # Mark model in order to exclude it below
                                    li_eta_flags[idx] = True
                                    break
                                par_name = pref+"eta_p"+str(i)
                                val = out.params[par_name].value
                                err = out.params[par_name].stderr
                                if val < err:
                                    print("WARNING:",par_name,"=",np.round(val,3),"+-",np.round(err,3)," is compatible with zero within uncertainty.")
                                    print("             This tail order is likely overfitting the data and will be excluded from selection.")
                                    li_eta_flags[idx] = True  # mark model in order to exclude it below

                    print("\n"+str(model)+"-fit yields reduced chi-square of: "+str(li_red_chis[idx])+" +- "+str(li_red_chi_errs[idx]))
                    print()
                    if show_fit_reports:
                        display(out) # show fit report
                except ValueError:
                    print('\nWARNING:',model+'-fit failed due to NaN-values and was skipped! ----------------------------------------------\n')

            # Select best model, models with eta_flag == True are excluded
            idx_best_model = np.nanargmin(np.where(li_eta_flags, np.inf, li_red_chis))
            best_model = li_fit_models[idx_best_model]
            self.fit_model = best_model
            print('\n##### RESULT OF AUTOMATIC MODEL SELECTION: #####\n')
            print('    Best fit model determined to be:',best_model)
            print('    Corresponding chi²-reduced:',li_red_chis[idx_best_model],'\n')
        elif not vary_tail_order:
            self.fit_model = fit_model

        print('\n##### Peak-shape determination #####-------------------------------------------------------------------------------------------')
        out = spectrum.peakfit(self, fit_model=self.fit_model, cost_func=cost_func, x_fit_cen=x_fit_cen, x_fit_range=x_fit_range, init_pars=init_pars ,vary_shape=True, vary_baseline=vary_baseline, method=method,show_plots=show_plots,show_peak_markers=show_peak_markers,sigmas_of_conf_band=sigmas_of_conf_band,plot_filename=plot_filename,eval_par_covar=eval_par_covar)

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
        if peak.comment == '-' or peak.comment == '' or peak.comment is None: # set 'shape calibrant' comment flag
            peak.comment = 'shape calibrant'
        elif 'shape calibrant' not in peak.comment:
            peak.comment = 'shape calibrant, '+peak.comment
        display(out)  # print(out.fit_report())
        self.red_chi_shape_calib = np.round(out.redchi,2)
        dict_pars = out.params.valuesdict()
        self.shape_cal_pars = {key.lstrip('p'+str(index_shape_calib)+'_'): val for key, val in dict_pars.items() if key.startswith('p'+str(index_shape_calib))}
        self.shape_cal_pars['bkg_c'] = dict_pars['bkg_c']
        self.shape_cal_par_errors = {} # dict to store shape calibration parameter errors
        for par in out.params:
            if par.startswith('p'+str(index_shape_calib)):
                self.shape_cal_par_errors[par.lstrip('p'+str(index_shape_calib)+'_')] = out.params[par].stderr
        self.shape_cal_par_errors['bkg_c'] = out.params['bkg_c'].stderr
        self.fit_range_shape_calib = x_fit_range


    def save_peak_shape_cal(self,filename):
        """Save peak shape parameters to a TXT-file.

        Parameters
        ----------
        filename : str
            Name of output file ('.txt' extension is automatically appended).

        """
        df1 = pd.DataFrame.from_dict(self.shape_cal_pars,orient='index',columns=['Value'])
        df1.index.rename('Model: '+str(self.fit_model),inplace=True)
        df2 = pd.DataFrame.from_dict(self.shape_cal_par_errors,orient='index',columns=['Error'])
        df = df1.join(df2)
        df.to_csv(str(filename)+'.txt', index=True,sep='\t')
        print('\nPeak-shape calibration saved to file: '+str(filename)+'.txt')


    def load_peak_shape_cal(self,filename):
        """Load peak shape from the TXT-file named 'filename.txt'.

        Successfully loaded shape calibration parameters and their uncertainties
        are used as the new :attr:`shape_cal_pars` and
        :attr:`shape_cal_par_errors` spectrum attributes respectively.


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
        self.shape_cal_par_errors = df_err.to_dict()
        print('\nLoaded peak shape calibration from '+str(filename)+'.txt')


    def _eval_peakshape_errors(self,peak_indeces=[],fit_result=None,
                               verbose=False,show_shape_err_fits=False):
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
        fit_result : lmfit modelresult, optional
            Fit result object to evaluate peak-shape error for.
        verbose : bool, optional, default: ``False``
            If ``True``, print all individual eff. mass shifts obtained by
            varying the shape parameters.
        show_shape_err_fits : bool, optional, default: ``False``
            If ``True``, show individual plots of re-fits for peak-shape error
            determination.

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

        The peak-shape uncertainties are obtained via the following procedure:

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
            mass recalibration.                                                 #TODO: add reference to mass re-calibration article!

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

        """
        if self.shape_cal_pars is None:
            print('\nWARNING: Could not calculate peak-shape errors - '
                  'no peak-shape calibration yet!\n')
            return

        if verbose:
            print('\n##### Peak-shape uncertainty evaluation #####\n')
            print('All mass shifts below are corrected for the corresponding '
                  'shifts of the calibrant peak.\n')
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
            print('Determining absolute centroid shifts of mass calibrant.\n')
        else:
            mass_calib_in_range = False
        if self.eff_mass_shifts is None:
            # initialize arrays of empty dictionaries
            self.eff_mass_shifts_pm = np.array([{} for i in range(len(self.peaks))])
            self.eff_mass_shifts = np.array([{} for i in range(len(self.peaks))])

        # Vary each shape parameter by plus and minus one standard deviation and
        # re-fit with all other shape parameters held fixed. Record the
        # corresponding fit results including the shifts of the (Gaussian) peak
        # centroids `mu`
        for par in shape_pars:
            pars = copy.deepcopy(self.shape_cal_pars) # deepcopy to avoid changes in original dictionary
            pars[par] = self.shape_cal_pars[par] + self.shape_cal_par_errors[par]
            if par == 'delta_m':
                pars['eta_m2'] = pars[par] - self.shape_cal_pars['eta_m1']
                pars['eta_m3'] = 1 - self.shape_cal_pars['eta_m1'] + pars['eta_m2']
            elif par == 'delta_p':
                pars['eta_p2'] = pars[par] - self.shape_cal_pars['eta_p1']
                pars['eta_p3'] = 1 - self.shape_cal_pars['eta_p1'] + pars['eta_p2']
            fit_result_p = self.peakfit(fit_model=fit_result.fit_model,
                                        cost_func=fit_result.cost_func,
                                        x_fit_cen=fit_result.x_fit_cen,
                                        x_fit_range=fit_result.x_fit_range,
                                        init_pars=pars, vary_shape=False,
                                        vary_baseline=fit_result.vary_baseline,
                                        method=fit_result.method,
                                        show_plots=False)
            #display(fit_result_p) # show fit result

            pars[par] = self.shape_cal_pars[par] - self.shape_cal_par_errors[par]
            if par == 'delta_m':
                pars['eta_m2'] =  pars[par] - self.shape_cal_pars['eta_m1']
                pars['eta_m3'] = 1 - self.shape_cal_pars['eta_m1'] +  pars['eta_m2']
            elif par == 'delta_p':
                pars['eta_p2'] =  pars[par] - self.shape_cal_pars['eta_p1']
                pars['eta_m3'] = 1 - self.shape_cal_pars['eta_p1'] +  pars['eta_p2']
            fit_result_m = self.peakfit(fit_model=fit_result.fit_model,
                                        cost_func=fit_result.cost_func,
                                        x_fit_cen=fit_result.x_fit_cen,
                                        x_fit_range=fit_result.x_fit_range,
                                        init_pars=pars, vary_shape=False,
                                        vary_baseline=fit_result.vary_baseline,
                                        method=fit_result.method,
                                        show_plots=False)
            #display(fit_result_m) # show fit result

            if show_shape_err_fits:
                fig, axs = plt.subplots(1,2,figsize=(20,6))
                ax0 = axs[0]
                ax0.set_title("Re-fit with ("+str(par)+" + 1 sigma) = {:.4E}".format(self.shape_cal_pars[par]+self.shape_cal_par_errors[par]))
                ax0.errorbar(fit_result_p.x,fit_result_p.y,yerr=fit_result_p.y_err,fmt='.',color='royalblue',linewidth=0.5)
                ax0.plot(fit_result.x, fit_result.best_fit,'--',color='black',linewidth=2,label="original fit")
                ax0.plot(fit_result_p.x, fit_result_p.best_fit,'-',color='red',linewidth=2,label="re-fit")
                ax1 = axs[1]
                ax1.set_title("Re-fit with ("+str(par)+" - 1 sigma) = {:.4E}".format(self.shape_cal_pars[par]-self.shape_cal_par_errors[par]))
                ax1.errorbar(fit_result_m.x,fit_result_m.y,yerr=fit_result_m.y_err,fmt='.',color='royalblue',linewidth=0.5)
                ax1.plot(fit_result.x, fit_result.best_fit,'--',color='black',linewidth=2,label="original fit")
                ax1.plot(fit_result_m.x, fit_result_m.best_fit,'-',color='red',linewidth=2,label="re-fit")
                for ax in axs:
                    ax.legend()
                    ax.set_yscale("log")
                    ax.set_ylim(0.7,)
                plt.show()

            # If mass calibrant is in fit range, determine its ABSOLUTE centroid
            # shifts first and use them to calculate 'shifted' mass
            # recalibration factors. The shifted recalibration factors are then
            # used to correct IOI centroid shifts for the corresponding shifts
            # of the mass calibrant
            # if calibrant is not in fit range, its centroid shifts must have
            # been determined in a foregoing mass re-calibration
            if mass_calib_in_range:
                cal_idx = self.index_mass_calib
                cal_peak = self.peaks[cal_idx]
                pref = 'p{0}_'.format(cal_idx)
                cen = fit_result.best_values[pref+'mu']
                new_cen_p =  fit_result_p.best_values[pref+'mu']
                delta_mu_p = new_cen_p - cen
                new_cen_m = fit_result_m.best_values[pref+'mu']
                delta_mu_m = new_cen_m - cen
                # recalibration factors obtained with shifted calib. centroids:
                recal_fac_p = cal_peak.m_AME/new_cen_p
                recal_fac_m = cal_peak.m_AME/new_cen_m
                self.recal_facs_pm[par+' recal facs pm'] = [recal_fac_p,recal_fac_m]
                 # plus and minus 1 sigma shifts of calibrant centroid [u]:
                self.eff_mass_shifts_pm[cal_idx][par+' calibrant centroid shift pm'] = [delta_mu_p,delta_mu_m]
                # maximal shifts of calibrant centroid [u]:
                max_eff_mass_shifts = np.where(np.abs(delta_mu_p) > np.abs(delta_mu_m),delta_mu_p,delta_mu_m).item()
                self.eff_mass_shifts[cal_idx][par+' calibrant centroid shift'] = max_eff_mass_shifts
            else: # check if shifted recal. factors pre-exist, print error otherwise
                try:
                    isinstance(self.eff_mass_shifts_pm[cal_idx][par+' calibrant centroid shift pm'],list)
                except:
                    raise Exception(
                    '\nERROR: No calibrant centroid shifts available for '
                    'peak-shape error evaluation. Ensure that: \n'
                    '(a) either the mass calibrant is in the fit range and specified\n'
                    '    with the `index_mass_calib` or `species_mass_calib` parameter, or\n'
                    '(b) if the mass calibrant is not in the fit range, a successful\n'
                    '    mass calibration has been performed upfront with fit_calibrant().')

            # Determine effective mass shifts
            # If calibrant is in fit range, the newly determined calibrant
            # centroid shifts will be used calculate the shifted recalibration
            # factors. Otherwise, the shifted re-calibration factors from a
            # foregoing mass calibration are used
            for peak_idx in peak_indeces: # IOIs only, mass calibrant excluded
                pref = 'p{0}_'.format(peak_idx)
                cen = fit_result.best_values[pref+'mu']

                new_cen_p =  fit_result_p.best_values[pref+'mu']
                recal_fac_p = self.recal_facs_pm[par+' recal facs pm'][0]
                # effective mass shift for +1 sigma parameter variation:
                delta_mu_p = recal_fac_p*new_cen_p - self.recal_fac*cen

                new_cen_m = fit_result_m.best_values[pref+'mu']
                recal_fac_m = self.recal_facs_pm[par+' recal facs pm'][1]
                # effective mass shift for -1 sigma parameter variation:
                delta_mu_m = recal_fac_m*new_cen_m - self.recal_fac*cen
                if verbose:
                    print(u'Re-fitting with {0} ={1: .2e} +/-{2: .2e} shifts peak {3} by {4: .3f} / {5: .3f} \u03BCu.'.format(par,self.shape_cal_pars[par],self.shape_cal_par_errors[par],peak_idx,delta_mu_p*1e06,delta_mu_m*1e06))
                    if peak_idx == peak_indeces[-1]:
                        print()  # empty line between different parameter blocks
                # shifts relative to calibrant centroid
                self.eff_mass_shifts_pm[peak_idx][par+' eff. mass shift pm'] = [delta_mu_p,delta_mu_m]
                # maximal shifts relative to calibrant centroid
                self.eff_mass_shifts[peak_idx][par+' eff. mass shift'] = np.where(np.abs(delta_mu_p) > np.abs(delta_mu_m),delta_mu_p,delta_mu_m).item()

        # Calculate and update relative peak-shape errors by summing effective
        # mass shifts in quadrature
        for peak_idx in peak_indeces:
            # Add eff. mass shifts in quadrature to get total peakshape error:
            shape_error = np.sqrt(np.sum(np.square( list(self.eff_mass_shifts[peak_idx].values()) )))
            p = self.peaks[peak_idx]
            m_ion = fit_result.best_values[pref+'mu']*self.recal_fac
            p.rel_peakshape_error = shape_error/m_ion
            if verbose:
                pref = 'p{0}_'.format(peak_idx)
                print("Relative peak-shape error of peak "+str(peak_idx)+":",np.round(p.rel_peakshape_error,9))


    def _update_calibrant_props(self,index_mass_calib,fit_result):
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
                p.comment = p.comment.replace('shape & mass calibrant','shape calibrant')
            elif p.comment == 'mass calibrant':
                p.comment = '-'
            elif 'mass calibrant' in p.comment:
                p.comment = p.comment.replace('mass calibrant','')
        if 'shape calibrant' in peak.comment: # set flag
            peak.comment = peak.comment.replace('shape calibrant','shape & mass calibrant')
        elif peak.comment == '-' or peak.comment == '' or peak.comment is None:
            peak.comment = 'mass calibrant'
        else:
            peak.comment = 'mass calibrant, '+peak.comment
        peak.fit_model = fit_result.fit_model
        peak.cost_func = fit_result.cost_func
        peak.area, peak.area_error = self.calc_peak_area(index_mass_calib,fit_result=fit_result)
        pref = 'p{0}_'.format(index_mass_calib)
        peak.m_ion = fit_result.best_values[pref+'mu']
        if peak.fit_model == 'Gaussian':
            std_dev = fit_result.best_values[pref+'sigma']
        else:  # for emg models
            FWHM_emg = self.calc_FWHM_emg(index_mass_calib,fit_result=fit_result)
            std_dev = self.A_stat_emg*FWHM_emg
        stat_error = std_dev/np.sqrt(peak.area) # A_stat* FWHM/sqrt(area), w/ with A_stat_G = 0.42... and A_stat_emg from `determine_A_stat_emg` method or default value from config.py
        peak.rel_stat_error = stat_error /peak.m_ion
        peak.rel_peakshape_error = None # reset to None
        peak.red_chi = np.round(fit_result.redchi, 2)

        # Print error contributions of mass calibrant:
        print("\n##### Mass recalibration #####\n")
        print("\nRelative literature error of mass calibrant:   ",np.round(peak.m_AME_error/peak.m_ion,9))
        print("Relative statistical error of mass calibrant:  ",np.round(peak.rel_stat_error,9))

        # Determine recalibration factor
        self.recal_fac = peak.m_AME/peak.m_ion
        print("\nRecalibration factor:      {:06.9f} = 1 {:=+1.2e}".format(self.recal_fac,self.recal_fac-1))
        if np.abs(self.recal_fac - 1) > 1e-02:
            print("\nWARNING: recalibration factor `recal_fac` deviates from unity by more than a permille.-----------------------------------------------")
            print(  "         Potentially, mass errors should also be re-scaled with `recal_fac` (currently not implemented)!-----------------------------")
        self.index_mass_calib = index_mass_calib # set mass calibrant flag to prevent overwriting of mass calibration results

        # Update peak properties with new calibrant centroid
        peak.m_ion = self.recal_fac*peak.m_ion # update centroid mass of calibrant peak
        if peak.A:
            peak.atomic_ME_keV = np.round((peak.m_ion + m_e - peak.A)*u_to_keV,3)   # atomic Mass excess (includes electron mass) [keV]
        if peak.m_AME:
            peak.m_dev_keV = np.round( (peak.m_ion - peak.m_AME)*u_to_keV, 3) # TITAN - AME [keV]

        # Determine rel. recalibration error and update recalibration error attribute
        peak.rel_recal_error = np.sqrt( (peak.m_AME_error/peak.m_AME)**2 + peak.rel_stat_error**2)/self.recal_fac
        self.rel_recal_error = peak.rel_recal_error
        print("Relative recalibration error:  "+str(np.round(self.rel_recal_error,9)),"\n")


    def fit_calibrant(self, index_mass_calib=None, species_mass_calib=None,
                      fit_model=None, cost_func='MLE', x_fit_cen=None,
                      x_fit_range=None, vary_baseline=True,
                      method='least_squares', show_plots=True,
                      show_peak_markers=True, sigmas_of_conf_band=0,
                      show_fit_report=True, plot_filename=None):
        """Determine mass re-calibration factor by fitting the selected
        calibrant peak.

        After the mass calibrant has been fitted the recalibration factor and
        its uncertainty are calculated saved as the spectrum's :attr:`recal_fac`
        and :attr:`recal_fac_error` attributes.

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
            ``'emg33'``, ... - see :mod:`emgfit.fit_models` module for all
            available fit models).
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

            See `Notes` of :meth:`spectrum.peakfit` for more details.
        x_fit_cen : float or None, [u], optional
            center of mass range to fit;
            if None, defaults to marker position (x_pos) of mass calibrant peak
        x_fit_range : float [u], optional
            width of mass range to fit; if None, defaults to 'default_fit_range' spectrum attribute
        vary_baseline : bool, optional, default: ``True``
            If ``True``, the constant background will be fitted with a varying
            uniform baseline parameter `bkg_c` (initial value: 0.1).
            If ``False``, the baseline parameter `bkg_c` will be fixed to 0.
        method : str, optional, default: `'least_squares'`
            Name of minimization algorithm to use. For full list of options
            check arguments of :func:`lmfit.minimizer.minimize`.
        show_plots : bool, optional
            If ``True`` (default) linear and logarithmic plots of the spectrum
            with the best fit curve are displayed. For details see
            :meth:`spectrum.plot_fit`.
        show_peak_markers : bool, optional
            If ``True`` (default) peak markers are added to the plots.
        sigmas_of_conf_band : int, optional, default: 0
            Confidence level of confidence band around best fit curve in sigma.
            Note that the confidence band is only derived from the uncertainties
            of the parameters that are varied during the fit.
        show_fit_report : bool, optional
            If ``True`` (default) the fit results are reported.
        plot_filename : str, optional, default: None
            If not ``None``, the plots will be saved to two separate files named
            '<`plot_filename`>_log_plot.png' and '<`plot_filename`>_lin_plot.png'.
            **Caution: Existing files with identical name are overwritten.**

        Notes
        -----
        The :meth:`spectrum.fit_peaks` method enables the simultaneous fitting
        of mass calibrant and ions of interest in a single multi-peak fit and
        can be used as an alternative to this method.

        After the calibrant fit the :meth:`spectrum._eval_peakshape_errors`
        method is automatically called to save the absolute calibrant centroid
        shifts as preparation for subsequent peak-shape error determinations.

        Since the spectrum has already been coarsely calibrated via the time-
        resolved calibration in the MR-TOF's data acquisition software MAc, the
        recalibration (or precision calibration) factor is usually very close to
        unity. An error will be raised by the :meth:`spectrum._update_calibrant_props`
        method if :attr:`spectrum.recal_fac` deviates from unity by more than a
        permille since this causes some implicit approximations for the
        calculation of the final mass values and their uncertainties to break
        down.

        The statistical uncertainty of the peak is calculated via the following
        relation:

        .. math:

            \\sigma_{stat} = A_{stat} \\frac{FWHM}{\\sqrt(N_counts)}

        For Gaussians the constant of proportionality :math:`A_{stat}` is always
        given by :math:`A_{stat,G}` = 0.425. For Hyper-EMG models
        :math:`A_{stat}=A_{stat,emg}` is either set to the default value
        `A_stat_emg_default` defined in the :mod:`~emgfit.config` module or
        determined by running the :meth:`spectrum.determine_A_stat_emg` method.
        The latter is usually preferable since this accounts for the specifics
        of the given peak shape.

        See also
        --------
        :meth:`spectrum.fit_peaks`

        """
        if index_mass_calib is not None and (species_mass_calib is None):
            peak = self.peaks[index_mass_calib]
        elif species_mass_calib:
            index_mass_calib = [i for i in range(len(self.peaks)) if species_mass_calib == self.peaks[i].species][0]
            peak = self.peaks[index_mass_calib]
        else:
            print("\nERROR: Definition of mass calibrant peak failed. Define EITHER the index OR the species name of the peak to use as mass calibrant!\n")
            return
        if x_fit_range is None:
            x_fit_range = self.default_fit_range

        print('##### Calibrant fit #####')
        if fit_model is None:
            fit_model = self.fit_model
        if x_fit_cen is None:
            x_fit_cen = peak.x_pos
        fit_result = spectrum.peakfit(self, fit_model=fit_model, cost_func=cost_func, x_fit_cen=x_fit_cen, x_fit_range=x_fit_range, vary_shape=False, vary_baseline=vary_baseline, method=method, show_plots=show_plots, show_peak_markers=show_peak_markers, sigmas_of_conf_band=sigmas_of_conf_band,plot_filename=plot_filename)
        if show_fit_report:
            display(fit_result)

        # Update recalibration factor and calibrant properties
        self._update_calibrant_props(index_mass_calib,fit_result)
        # Calculate updated recalibration factors from absolute centroid shifts
        # of calibrant and as prep for subsequent peak-shape error determination
        # for ions of interest
        self._eval_peakshape_errors(peak_indeces=[index_mass_calib],fit_result=fit_result,verbose=False)


    ##### Update peak list with fit values
    def _update_peak_props(self,peaks,fit_result):
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
                pass  # prevent overwritting of mass calibration results
            else:
                peak_idx = self.peaks.index(p)
                pref = 'p{0}_'.format(peak_idx)
                p.fit_model = fit_result.fit_model
                p.cost_func = fit_result.cost_func
                p.area = self.calc_peak_area(peak_idx,fit_result=fit_result)[0]
                p.area_error = self.calc_peak_area(peak_idx,fit_result=fit_result)[1]
                p.m_ion = self.recal_fac*fit_result.best_values[pref+'mu']
                if p.fit_model == 'Gaussian':
                    std_dev = fit_result.best_values[pref+'sigma']
                else:  # for emg models
                    FWHM_emg = self.calc_FWHM_emg(peak_idx,fit_result=fit_result)
                    std_dev = self.A_stat_emg*FWHM_emg
                stat_error = std_dev/np.sqrt(p.area)  # stat_error = A_stat * FWHM / sqrt(peak_area), w/ with A_stat_G = 0.42... and  A_stat_emg from `determine_A_stat_emg` method or default value from config.py
                p.rel_stat_error = stat_error/p.m_ion
                if self.rel_recal_error:
                    p.rel_recal_error = self.rel_recal_error
                elif p==peaks[0]: # only print once
                    print('WARNING: Could not set mass recalibration errors. No successful mass recalibration performed on spectrum yet.')
                try:
                    p.rel_mass_error = np.sqrt(p.rel_stat_error**2 + p.rel_peakshape_error**2 + p.rel_recal_error**2) # total relative uncertainty of mass value without systematics - includes: stat. mass uncertainty, peakshape uncertainty, recalibration uncertainty
                    p.mass_error_keV = p.rel_mass_error*p.m_ion*u_to_keV
                except TypeError:
                    if p==peaks[0]:
                        print('Could not calculate total mass error.')
                    pass
                if p.A:
                    p.atomic_ME_keV = np.round((p.m_ion + m_e - p.A)*u_to_keV,3)   # atomic Mass excess (includes electron mass) [keV]
                if p.m_AME:
                    p.m_dev_keV = np.round( (p.m_ion - p.m_AME)*u_to_keV, 3) # TITAN - AME [keV]
                p.red_chi = np.round(fit_result.redchi, 2)


    def fit_peaks(self, index_mass_calib=None, species_mass_calib=None,
                  x_fit_cen=None, x_fit_range=None, fit_model=None,
                  cost_func='MLE', method ='least_squares', init_pars=None,
                  vary_shape=False, vary_baseline=True, show_plots=True,
                  show_peak_markers=True,sigmas_of_conf_band=0,
                  plot_filename=None,show_fit_report=True,
                  show_shape_err_fits=False):
        """Fit peaks, update peaks properties and show results.

        Fits peaks in either the entire spectrum or optionally only the peaks
        in the mass range specified with `x_fit_cen` and `x_fit_range`.

        Optionally, the mass recalibration can be performed simultaneous with
        the IOI fit if the mass calibrant is in the fit range and specified with
        either the `index_mass_calib` or `species_mass_calib` arguments.

        Before running this method a successful peak-shape calibration must have
        been performed with :meth:`determine_peak_shape`.

        Parameters
        ----------
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
            ``'emg33'``, ... - see :mod:`emgfit.fit_models` module for all
            available fit models). If ``None``, defaults to
            :attr:`~spectrum.best_model` spectrum attribute.
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
        vary_shape (bool, optional): if False, peak-shape parameters (sigma, theta, eta's and tau's) are kept fixed at initial values; if True, they are varied (default: False)
        vary_baseline : bool, optional, default: True
            if True, the constant background will be fitted with a varying baseline paramter bkg_c (initial value: 0.1); otherwise the beseline paremter bkg_c will be fixed to 0.
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

        vary_shape : bool, optional, default: ``False``
            If ``False`` peak-shape parameters (`sigma`, `theta`,`etas` and
            `taus`) are kept fixed at their initial values. If ``True`` the
            shared shape parameters are varied (ensuring identical shape
            parameters for all peaks).
        vary_baseline : bool, optional, default: ``True``
            If ``True``, the constant background will be fitted with a varying
            uniform baseline parameter `bkg_c` (initial value: 0.1).
            If ``False``, the baseline parameter `bkg_c` will be fixed to 0.
        show_plots : bool, optional
            If ``True`` (default) linear and logarithmic plots of the spectrum
            with the best fit curve are displayed. For details see
            :meth:`spectrum.plot_fit`.
        show_peak_markers : bool, optional
            If ``True`` (default) peak markers are added to the plots.
        sigmas_of_conf_band : int, optional, default: 0
            Confidence level of confidence band around best-fit curve in sigma.
            Note that the confidence band is only derived from the uncertainties
            of the parameters that are varied during the fit.
        plot_filename : str, optional, default: None
            If not ``None``, the plots will be saved to two separate files named
            '<`plot_filename`>_log_plot.png' and '<`plot_filename`>_lin_plot.png'.
            **Caution: Existing files with identical name are overwritten.**
        show_fit_report : bool, optional
            If ``True`` (default) the detailed lmfit fit report is printed.
        show_shape_err_fits : bool, optional, default: True
            If ``True``, plots of all fits performed for the peak-shape
            uncertainty evaluation are shown.

        Notes
        -------
        Updates peak properties dataframe with peak properties obtained in fit.

        """
        if fit_model is None:
            fit_model = self.fit_model
        if x_fit_range is None:
            x_fit_range = self.default_fit_range

        if index_mass_calib is not None and (species_mass_calib is None):
            peak = self.peaks[index_mass_calib]
        elif species_mass_calib is not None:
            index_mass_calib = [i for i in range(len(self.peaks)) if species_mass_calib == self.peaks[i].species][0]
            peak = self.peaks[index_mass_calib]
        elif index_mass_calib is not None and species_mass_calib is not None:
            raise Exception("\nERROR: Definition of mass calibrant peak failed. Define EITHER the index OR the species name of the peak to use as mass calibrant!\n")

        # FIT ALL PEAKS
        fit_result = spectrum.peakfit(self, fit_model=fit_model, cost_func=cost_func,
                                      x_fit_cen=x_fit_cen, x_fit_range=x_fit_range,
                                      init_pars=init_pars, vary_shape=vary_shape,
                                      vary_baseline=vary_baseline, method=method,
                                      show_plots=show_plots,
                                      show_peak_markers=show_peak_markers,
                                      sigmas_of_conf_band=sigmas_of_conf_band,
                                      plot_filename=plot_filename)

        if index_mass_calib is not None:
            self._update_calibrant_props(index_mass_calib,fit_result) # Update recalibration factor and calibrant properties

        # Determine peak-shape errors
        if x_fit_cen:
            x_min = x_fit_cen - x_fit_range/2
            x_max = x_fit_cen + x_fit_range/2
            peaks_to_fit = [peak for peak in self.peaks if (x_min < peak.x_pos < x_max)] # get peaks in fit range
        else:
            peaks_to_fit = self.peaks

        peak_indeces = [self.peaks.index(p) for p in peaks_to_fit]
        try:
            self._eval_peakshape_errors(peak_indeces=peak_indeces,fit_result=fit_result,verbose=True,show_shape_err_fits=show_shape_err_fits)
        except KeyError:
            print("WARNING: Peak-shape error determination failed with KeyError. Likely the used fit_model is inconsistent with the shape calibration model.")
        self._update_peak_props(peaks_to_fit,fit_result)
        self.show_peak_properties()
        if show_fit_report:
            if cost_func is 'MLE':
                print("The values for chi-squared as well as the parameter "
                      "uncertainties and correlations reported by lmfit below "
                      "should be taken with caution when your MLE fit includes "
                      "bins with low statistics. For details see Notes section "
                      "in the spectrum.peakfit() method documentation.")
            display(fit_result)
        for p in peaks_to_fit:
            self.fit_results[self.peaks.index(p)] = fit_result


    ##### Save all relevant results to external files
    def save_results(self,filename):
        """Write the fit results to a XLSX file and the peak-shape calibration
        to a TXT file.

        Write results to an XLSX Excel file named `'filename'` and save
        peak-shape calibration parameters to TXT file named
        `'<filename>_peakshape_calib'`.

        The EXCEL file will contain critical spectrum properties and all peak
        properties (including the mass values) in two separate sheets.

        Parameters
        ----------
        filename : string
            Prefix of the files to be saved to (the .xlsx & .txt file endings
            are automatically appended).

        """
        # Ensure no files are overwritten
        if os.path.isfile(str(filename)+".xlsx"):
            print ("ERROR: File "+str(filename)+".xlsx already exists. No files saved! Choose a different filename or delete the original file and re-try.")
            return
        if os.path.isfile(str(filename)+"_peakshape_calib.txt"):
            print ("ERROR: File "+str(filename)+"_peakshape_calib.txt already exists. No files saved! Choose a different filename or delete the original file and re-try.")
            return

        # Make DataFrame with spectrum propeties
        datetime = time.localtime() # get current date and time
        datetime_string = time.strftime("%Y/%m/%d, %H:%M:%S", datetime)
        spec_data = np.array([["Saved on",datetime_string]]) # add datetime stamp
        import sys
        spec_data = np.append(spec_data, [["Python version",sys.version_info[0:3]]],axis=0)
        from . import __version__ # get emgfit version
        spec_data = np.append(spec_data, [["emgfit version",__version__]],axis=0)
        spec_data = np.append(spec_data, [["lmfit version",fit.__version__]],axis=0)
        from scipy import __version__ as scipy_version
        spec_data = np.append(spec_data, [["scipy version",scipy_version]],axis=0)
        spec_data = np.append(spec_data, [["numpy version",np.__version__]],axis=0)
        spec_data = np.append(spec_data, [["pandas version",pd.__version__]],axis=0)
        attributes = ['input_filename','mass_number','spectrum_comment','fit_model','red_chi_shape_calib','fit_range_shape_calib','determined_A_stat_emg','A_stat_emg','A_stat_emg_error','recal_fac','rel_recal_error']
        for attr in attributes:
            attr_val = getattr(self,attr)
            spec_data = np.append(spec_data, [[attr,attr_val]],axis=0)
        df_spec = pd.DataFrame(data=spec_data)
        df_spec.set_index(df_spec.columns[0],inplace=True)

        # Make peak properties & eff. mass shifts DataFrames
        dict_peaks = [p.__dict__ for p in self.peaks]
        df_prop = pd.DataFrame(dict_peaks)
        df_prop.index.name = "Peak index"
        frames = []
        keys = []
        for peak_idx in range(len(self.eff_mass_shifts)):
            df = pd.DataFrame.from_dict(self.eff_mass_shifts[peak_idx], orient='index')
            df.columns = ['Value [u]']
            frames.append(df)
            keys.append(str(peak_idx))
        df_eff_mass_shifts = pd.concat(frames, keys=keys)
        df_eff_mass_shifts.index.names = ['Peak index','Parameter']

        # Save lin. and log. plots of full fitted spectrum to temporary files
        # so they can be inserted into the XLSX file
        from IPython.utils import io
        with io.capture_output() as captured: # suppress function output to Jupyter notebook
            self.plot_fit(plot_filename=filename)

        # Write DataFrames to separate sheets of EXCEL file and save peak-shape calibration to TXT-file
        with pd.ExcelWriter(filename+'.xlsx',engine='xlsxwriter') as writer:
            df_spec.to_excel(writer,sheet_name='Spectrum properties')
            df_prop.to_excel(writer,sheet_name='Peak properties')
            prop_sheet = writer.sheets['Peak properties']
            prop_sheet.insert_image(len(df_prop)+2,1, filename+'_log_plot.png',{'x_scale': 0.45,'y_scale':0.45})
            prop_sheet.insert_image(len(df_prop)+26,1, filename+'_lin_plot.png',{'x_scale': 0.45,'y_scale':0.45})
            df_eff_mass_shifts.to_excel(writer,sheet_name='Mass shifts in PS error eval.')
        print("Fit results saved to file:",str(filename)+".xlsx")

        # Clean up temporary image files
        os.remove(filename+'_log_plot.png')
        os.remove(filename+'_lin_plot.png')

        try:
            self.save_peak_shape_cal(filename+"_peakshape_calib")
        except:
            raise




################################################################################
