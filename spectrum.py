###################################################################################################
##### Python module for peak detection in TOF mass spectra
##### Code by Stefan Paul, 2019-12-28

##### Import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sig
import scipy.special as spl
import time
import copy
from IPython.display import display
from emgfit.config import *
import emgfit.fit_models as fit_models
import emgfit.emg_funcs as emg_funcs
import lmfit as fit
import os
import warnings
# ignore irrelevant warnings by message
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
warnings.filterwarnings("ignore", message="overflow encountered in exp")
warnings.filterwarnings("ignore", category=RuntimeWarning)

#u_to_keV = config.u_to_keV
#u = config.u
#m_e = config.m_e

###################################################################################################
##### Define peak class

def splitspecies(s):
    """ Splits ion species string into list containing constituent atom strings (e.g. '4H1:1C12' returns ['4H1','1C12'] """
    return s.split(':')

def splitparticle(s):
    """ Extracts number, particle/element type and mass number of particle string (e.g. 1Cs133, Cs133, 1e) """
    if s[-1:] == '?': # handle unidentified species (indicated by '?' at end of string)
        return None, '?', None
    tail = s.lstrip('+-0123456789')
    head = s[:-len(tail)]
    if head == '+' or head == '': # handle missing number (if '+' given or 1 in front of single omitted)
        n = int(1)
    elif head == '-': # handle missing number
        n = int(-1)
    else:
        n = int(head) # leading number including sign (if present)
    El = tail.rstrip('0123456789') # central letters
    if El == 'e' and len(El) == len(tail): # handle electron strings, e.g. ':-1e'
        A = 0
    else:
        A = int(tail[len(El):]) # trailing number
    return n, El, A

def get_AME_values(species):
    """ Calculates the AME mass, AME mass error, the extrapolation flag and the mass number A of the given species string"""
    m = 0.0
    m_error_sq = 0.0
    A_tot = 0
    for ptype in splitspecies(species):
        n, El, A = splitparticle(ptype)
        extrapol = False # initialize boolean flag
        if ptype[-1:] == '?': # unidentified species
            m = None
            m_error = None
            A_tot = None
        elif El == 'e': # electron
            m += n*m_e
            # neglect uncertainty of m_e
        else: # regular atom
            m += n*mdata_AME(El,A)[2]
            m_error_sq += (n*mdata_AME(El,A)[3])**2
            m_error = np.sqrt(m_error_sq)
            A_tot += A
            if mdata_AME(El,A)[3] == 1: # extrapolated mass
                extrapol = True
    return m, m_error, extrapol, A_tot


class peak:
    def __init__(self,x_pos,species,m_AME=None,m_AME_error=None):
        """ Create new peak object """
        self.x_pos = x_pos
        self.species = species # e.g. '1Cs133:-1e or 'Cs133:-e' or '4H1:1C12:-1e'
        self.comment = '-'
        self.m_AME = m_AME #
        self.m_AME_error = m_AME_error
        m, m_error, extrapol, A_tot = get_AME_values(species) # get AME values for specified species
        if self.m_AME is None: # unless m_AME has been user-defined, the mass value of the specified 'species' is calculated from AME database
             self.m_AME = m
        if self.m_AME_error is None: # unless m_AME_error has been user-defined, the mass error of the specified 'species' is calculated from AME database
            self.m_AME_error = m_error
        self.extrapolated_yn = extrapol
        self.fit_model = None
        self.cost_func = None # cost function used to fit peak
        self.chi_sq_red = None # chi square reduced of peak fit
        self.area = None
        self.area_error = None
        self.m_fit = None # mass value from fit [u]
        self.rel_stat_error = None # A_stat * FWHM / sqrt(area) /m_fit, with A_stat_G = 0.42... and A_stat_emg from `determine_A_stat` method or default value from config.py
        self.rel_recal_error = None
        self.rel_peakshape_error = None
        self.rel_mass_error = None # total relative uncertainty of mass value - includes: stat. mass uncertainty, peakshape uncertainty, recalibration uncertainty
        self.A = A_tot
        self.atomic_ME_keV = None # atomic Mass excess = atomic mass[u] - A    [keV]
        self.mass_error_keV = None # total error of m_fit (before systematics) [keV]
        self.m_dev_keV = None # TITAN - AME [keV]

    def update_lit_values(self):
        """ Overwrite m_AME, m_AME_error and extrapolated_yn attributes of peak with AME values for specified species """
        m, m_error, extrapol, A_tot = get_AME_values(self.species) # calculate values for species
        self.m_AME = m
        self.m_AME_error = m_error
        self.extrapolated_yn = extrapol
        self.A = A_tot


    def properties(self):
        """ Print peak properties """
        print("x_pos:",self.x_pos,"u")
        print("Species:",self.species)
        print("AME mass:",self.m_AME,"u     (",np.round(self.m_AME*u_to_keV,3),"keV )")
        print("AME mass uncertainty:",self.m_AME_error,"u         (",np.round(self.m_AME_error*u_to_keV,3),"keV )")
        print("AME mass extrapolated?",self.extrapolated_yn)
        if self.fit_model is not None:
            print("Peak area: "+str(self.area)+" +- "+str(self.peak_area_error)+" counts")
            print("(Ionic) mass:",self.m_fit,"u     (",np.round(self.m_fit*u_to_keV,3),"keV )")
            print("Stat. mass uncertainty:",self.rel_stat_error*self.m_fit,"u     (",np.round(self.rel_stat_error*self.m_fit*u_to_keV,3),"keV )")
            print("Peakshape uncertainty:",self.rel_peakshape_error*self.m_fit,"u     (",np.round(self.rel_peakshape_error*self.m_fit*u_to_keV,3),"keV )")
            print("Re-calibration uncertainty:",self.rel_recal_error*self.m_fit,"u     (",np.round(self.rel_recal_error*self.m_fit*u_to_keV,3),"keV )")
            print("Total mass uncertainty (before systematics):",self.rel_mass_error*self.m_fit,"u     (",np.round(self.mass_error_keV,3),"keV )")
            print("Atomic mass excess:",np.round(self.atomic_ME_keV,3),"keV")
            print("TITAN - AME:",np.round(self.m_dev_keV,3),"keV")
            print("χ_sq_red:",np.round(self.chi_sq_red))


###################################################################################################
###### Define spectrum class
class spectrum:
    def __init__(self,filename,m_start=None,m_stop=None,skiprows = 18,show_plot=True,df=None):
        """
        Creates spectrum object by importing TOF data from .txt or .csv file, plotting full spectrum and then cutting spectrum to specified fit range {m_start;m_stop}
	    Input file format: two-column .csv- or .txt-file with tab separated values (column 1: mass bin, column 2: counts per bin)

        Parameters:
        -----------
	    filename : str or None
            string containing (path and) filename of mass spectrum to analyze (as exported from MAc in histogram mode)
            if None, data must be provided as DataFrame `df`
	    m_start : float, optional, [u]
            start of fit range, data at lower masses will be discarded
	    m_stop : float, optional, [u]
            stop of fit range, data at higher masses will be discarded
        show_plot : bool, optional, default: True
            if True, shows a plot of full spectrum with markers for `m_start` and `m_stop`
        df : pandas DataFrame, optional
            spectrum data, this enables the alternative creation of a spectrum object from a DataFrame instead of from an external file

	    Returns:
        --------
        Pandas dataframe 'data' containing mass data and plot of full spectrum with fit range markers
	    """
        if filename is not None:
            data_uncut = pd.read_csv(filename, header = None, names= ['Mass [u]', 'Counts'], skiprows = skiprows,delim_whitespace = True,index_col=False,dtype=float)
            data_uncut.set_index('Mass [u]',inplace =True)
        else:
            data_uncut = df
        self.fit_model = None
        self.best_redchi = None
        self.shape_cal_pars = None
        self.shape_cal_errors = []
        self.index_mass_calib = None
        self.determined_A_stat_emg = False
        self.A_stat_emg = A_stat_emg_default # initialize at default A_stat from config.py
        self.A_stat_emg_error = None
        self.recal_fac = 1.0
        self.rel_recal_error = None
        self.centroid_shifts_pm = None # array containing a dictionary for each peak with pos. and neg. centroid shifts for each parameter varied in peak-shape error evaluation
        self.centroid_shifts = None # array containing a dictionary for each peak with the maximal centroid shift for each parameter varied in peak-shape error evaluation
        self.peaks = [] # list containing peaks associated with spectrum (each peak is represented by an instance of the class 'peak')
        self.fit_results = [] # list containing fit results (lmfit modelresult objects) of peaks associated with spectrum
        plt.rcParams.update({"font.size": 15})
        if m_start or m_stop:
            self.data = data_uncut.loc[m_start:m_stop] # dataframe containing mass spectreum data (cut to fit range)
            plot_title = 'Spectrum with start and stop markers'
        else:
            self.data = data_uncut # dataframe containing mass spectrum data
            plot_title = 'Spectrum (fit full range)'
        if show_plot:
            fig  = plt.figure(figsize=(20,8))
            plt.title(plot_title)
            data_uncut.plot(ax=fig.gca())
            plt.vlines(m_start,0,1.2*max(self.data['Counts']))
            plt.vlines(m_stop,0,1.2*max(self.data['Counts']))
            plt.yscale('log')
            plt.ylabel('Counts')
            plt.show()


    ##### Define static method for smoothing spectrum before peak detection (taken from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html)
    @staticmethod
    def smooth(x,window_len=11,window='hanning'):
        """
        smooth the data using a window with requested size.

    	This method is based on the convolution of a scaled window with the signal.
    	The signal is prepared by introducing reflected copies of the signal
    	(with the window size) in both ends so that transient parts are minimized
    	in the begining and end part of the output signal.

    	input:
    	    x: the input signal
                window_len: the dimension of the smoothing window; should be an odd integer
    	    window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
    		    flat window will produce a moving average smoothin

    	output:
    	    the smoothed signal

    	example:
    	    t=linspace(-2,2,0.1)
    	    x=sin(t)+randn(len(t))*0.1
    	    y=smooth(x)

    	see also:
    	    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    	    scipy.signal.lfilter

    	TODO: the window parameter could be the window itself if an array instead of a string
    	NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
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
        #print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='valid')
        return y[int(window_len/2+1):-int(window_len/2-1)]


    ##### Routine for plotting spectrum
    def plot(self,peaks=None,title="",ax=None,yscale='log',vmarkers=None,thres=None,ymin=None,xmin=None,xmax=None):
        """
        Plots spectrum
        - with markers for all peaks stored in peak list 'self.peaks'
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


    ##### Define static routine for plotting spectrum data stored in dataframe df (only use for functions within this class)
    @staticmethod
    def plot_df(df,title="",ax=None,yscale='log',peaks=None,vmarkers=None,thres=None,ymin=None,xmin=None,xmax=None):
        """Plots spectrum data stored in dataframe 'df'

           - optionally with peak markers if
        	(a) single x_pos or array x_pos is passed to 'vmarkers', or
            (b) list of peak objects is passed to 'li_peaks'
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
    def detect_peaks(self,window='blackman',window_len=23,thres=0.003,width=2e-05,plot_smoothed_spec=True,plot_2nd_deriv=True,plot_detection_result=True):
        """
        Performs automatic peak detection on spectrum object using a scaled second derivative of the spectrum.

        width (float): minimal FWHM of peaks to be detected - in atomic mass units - Caution: To achieve maximal sensitivity for overlapping peaks this number might have to be set to less than the peak's FWHM!
        """
        # Smooth spectrum (moving average with window function)
        data_smooth = self.data.copy()
        data_smooth['Counts'] = spectrum.smooth(self.data['Counts'].values,window_len=window_len,window=window)
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
            self.plot_df(data_sec_deriv,title="Scaled second derivative of spectrum - set threshold indicated",yscale='linear',thres=-thres)

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
            self.plot_df(data_sec_deriv_mod,title="Negative part of scaled second derivative, inverted - set threshold indicated",thres=thres,vmarkers=li_peak_pos,ymin=0.1*thres)

        # Create list of peak objects
        for x in li_peak_pos:
            p = peak(x,'?') # instantiate new peak
            self.peaks.append(p)
            self.fit_results.append(None)

        # Plot raw spectrum with detected peaks marked
        if plot_detection_result:
            self.plot(peaks=self.peaks,title="Spectrum with detected peaks marked",ymin=0.6)


    ##### Add peak manually
    def add_peak(self,x_pos,species="?",m_AME=None,m_AME_error=None,verbose=True):
        """
        Manually add a peak at position 'x_pos' to peak list of spectrum
        - optionally assign 'species' (corresponding literature mass and mass error will then automatically be calculated from AME values)
        - optionally assign user-defined m_AME and m_AME_error (this overwrites the values calculated from AME database, use e.g. for isomers)
        """
        p = peak(x_pos,species,m_AME=m_AME,m_AME_error=m_AME_error) # instantiate new peak
        self.peaks.append(p)
        self.fit_results.append(None)
        ##### Helper function for sorting list of peaks by marker positions 'x_pos'
        def sort_x(peak):
            return peak.x_pos
        self.peaks.sort(key=sort_x) # sort peak positions in ascending order
        if verbose:
            print("Added peak at ",x_pos," u")


    ##### Remove peak manually
    def remove_peak(self,peak_index=None,x_pos=None,species="?"):
        """
        Remove a peak manually from peak list

        select peak by specifying species label, peak position 'x_pos' (up to 6th decimal) or peak index (0-based! Check for peak index by calling .show_peak_properties() method)
        """
        if peak_index:
            i = peak_index
        elif species != "?":
            i = [i for i in range(len(self.peaks)) if species == self.peaks[i].species][0] # select peak with species label 'species'
        elif x_pos:
            i = [i for i in range(len(self.peaks)) if np.round(x_pos,6) == np.round(self.peaks[i].x_pos,6)][0] # select peak at position 'x_pos'
        try:
            rem_peak = self.peaks.pop(i)
            self.fit_results.pop(i)
            print("Removed peak at ",rem_peak.x_pos," u")
        except:
            print("Peak removal failed!")
            raise


    ##### Print peak properties
    def show_peak_properties(self):
        """
        Print properties of all peaks in peak list
	    """
        dict_peaks = [p.__dict__ for p in self.peaks]
        df_prop = pd.DataFrame(dict_peaks)
        display(df_prop)
        #return df_prop


    ##### Specify identified species
    def assign_species(self,species,peak_index=None,x_pos=None):
        """
        Assign a species (label) either to single selected peak or to all peaks from the peak list
        - assignment of single peak species:
            select peak by specifying peak position 'x_pos' (up to 6th decimal) or peak index argument (0-based! Check for peak index by calling .show_peak_properties() method of spectrum object)     specify species name by assigning string to species object

        - assignment of multiple peak species:
            nothing should be assigned to the 'peak_index' and 'x_pos' arguments
            instead the user specficies a list of the new species strings to the species argument
            (if there's N detected peaks, the list must have length N!)
            Former species assignments can be kept by inserting blanks at the respective position in the new species list
	    Otherwise former species assignments are overwritten
            also see examples below for usage

        species (str or list):    The species name (or list of name strings) to be assigned to the selected peak (or to all peaks)


        Examples:
        ---------
        spec.assign_species('1Cs133:-1e',peak_index = 2)
            ->  assigns peak with peak_index 2 (third-lowest-mass peak) as '1Cs133:-1e', all other peaks remain unchanged

        spec.assign_species(['1Ru102:-1e', '1Pd102:-1e', 'Rh102:-1e', None,'1Sr83:1F19:-1e', '?'])
            -> assigns species of first, second, third and fourth peak with the species labels given in the above list
            -> the 'None' argument leaves the species assignment of the 4th peak unchanged (a former species assignment to this peak persists!)
            -> the '?' argument overwrites any former species assignment to the highest-mass-peak and marks the peak as unidentified
        """
        try:
            if peak_index is not None:
                p = self.peaks[peak_index]
                p.species = species
                p.update_lit_values() # overwrite m_AME, m_AME_error and extrapolated_yn attributes with AME values for specified species
                print("Species of peak",peak_index,"assigned as",p.species)
            elif x_pos:
                i = [i for i in range(len(self.peaks)) if  np.round(x_pos,6) == np.round(self.peaks[i].x_pos,6)][0] # select peak at position 'x_pos'
                p = self.peaks[i]
                p.species = species
                p.update_lit_values() # overwrite m_AME, m_AME_error and extrapolated_yn attributes with AME values for specified species
                print("Species of peak",i,"assigned as",p.species)
            elif len(species) == len(self.peaks) and peak_index is None and x_pos is None: # assignment of multiple species
                for i in range(len(species)):
                    species_i = species[i]
                    if species_i: # skip peak if 'None' given as argument
                        p = self.peaks[i]
                        p.species = species_i
                        p.update_lit_values() # overwrite m_AME, m_AME_error and extrapolated_yn attributes with AME values for specified species
                        print("Species of peak",i,"assigned as",p.species)
            else:
                print('WARNING: Species assignment failed.')
        except:
            print('Errors occured in peak assignment!')
            raise


    ##### Add peak comment manually
    def add_comment(self,peak_index,comment,overwrite=False):
        """
        Method for adding a comment to a peak.
        By default the string 'comment' will be added at the end of the current peak comment (if the current comment is '-' it is overwritten with 'comment' argument).
        If overwrite is set to 'True' the current peak comment is overwritten with the 'comment' argument.

        NOTE: It is possible to add further comments to the shape or mass calibrant peaks, however, the protected flags 'shape calibrant', 'mass calibrant' and 'shape & mass calibrant' will persist.
                 These flags are added to the peak comments automatically during the shape and mass calibration and should never be added to comments manually by the user!

        Parameters:
        -----------
        peak_index (int): index of peak to add comment to
        comment (str): comment to add to peak_detect.py
        overwrite (bool): boolean specifying whether to append to current comment or to overwrite it

        Returns:
        --------
        None
        """
        peak = self.peaks[peak_index]
        protected_flags = ('shape calibrant','shape & mass calibrant','mass calibrant') # item order matters for comment overwriting!
        try:
            if any(s in comment for s in ('shape calibrant','mass calibrant','shape & mass calibrant')):
                print("ERROR: 'shape calibrant','mass calibrant' and 'shape & mass calibrant' are protected flags. User-defined comments must not contain these flags. Re-phrase comment argument!")
                return
            if peak.comment == '-':
                peak.comment = comment
            elif overwrite:
                if any(s in peak.comment for s in protected_flags) and overwrite:
                    print("WARNING: The protected flags 'shape calibrant','mass calibrant' or 'shape & mass calibrant' cannot be overwritten.")
                    flag = [s for s in protected_flags if s in peak.comment][0]
                    peak.comment = peak.comment.replace(peak.comment,flag+comment)
                else:
                    peak.comment = comment
            else:
                peak.comment = peak.comment+comment
            print("Comment of peak",peak_index,"was changed to: ",peak.comment)
        except TypeError:
            print("ERROR: 'comment' argument must be given as type string.")
            pass


    #####  Add peak markers to plot of a fit
    def add_peak_markers(self,yscale='log',ymax=None,peaks=None):
        """
        (Internal) method for adding peak markers to current figure object, place this function as self.add_peak_markers between plt.figure() and plt.show(), only for use on already fitted spectra
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
                plt.vlines(x=p.x_pos,ymin=ymin,ymax=1.04*ymax,linestyles='dashed')
                plt.text(p.x_pos, 1.06*ymax, self.peaks.index(p), horizontalalignment='center', fontsize=12)


    ##### Plot fit of full spectrum
    def plot_fit(self,fit_result=None,plot_title=None,show_peak_markers=True,sigmas_of_conf_band=0,x_min=None,x_max=None,plot_filename=None):
        """
        Plots spectrum with fit in logarithmic and linear y-scale

        Parameters:
        -----------
        fit_result : lmfit modelresult, optional, default: None
            fit result to plot
            if None, defaults to fit result of first peak in plot range (from `fit_results` list of spectrum object)
        plot_title : str or None, optional, default :
            titles of plots, the default ensures clear indication of the used fit model
            if None, defaults to fit model name asociated with spectrum (`fit_model` attribute of spectrum object)
        show_peak_markers : bool, optional, default: True
            if True, peak markers are added to the plots
        sigmas_of_conf_band : int, optional, default: 0
            coverage probability of confidence band in sigma (only for log-plot);
            if 0, no confidence band is shown (default)
        x_min : float [u], optional, default: None
            start of mass range to plot
            if None, minimum of spectrum object's mass data is used
        x_max : float [u], optional, default: None
            end of mass range to plot
            if None, maximum of spectrum objects's mass data is used
        plot_filename : str or None, optional, default: None
            if not None, plot images will be saved to two separate files named '`plot_filename`_log_plot.png' and '`plot_filename`_lin_plot.png' respectively

        Returns:
        --------
            None
        """
        if x_min is None:
            x_min = self.data.index.values[0]
        if x_max is None:
            x_max = self.data.index.values[-1]
        peaks_to_plot = [peak for peak in self.peaks if (x_min < peak.x_pos < x_max)] # select peaks in mass range of interest
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
        y_max_res = max(np.abs(fit_result.residual[i_min:i_max]/weights)) + max(fit_result.y_err[i_min:i_max])

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
            self.add_peak_markers(yscale='log',ymax=y_max_log,peaks=peaks_to_plot)
        if sigmas_of_conf_band!=0 and fit_result.errorbars == True: # add confidence band with specified number of sigmas
            dely = fit_result.eval_uncertainty(sigma=sigmas_of_conf_band)
            plt.fill_between(fit_result.x, fit_result.best_fit-dely, fit_result.best_fit+dely, color="#ABABAB", label=str(sigmas_of_conf_band)+'-$\sigma$ uncertainty band')
        plt.title(plot_title)
        plt.rcParams.update({"font.size": 15})
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
        f2, axs = plt.subplots(2,1,figsize=(20,12),gridspec_kw={'height_ratios': [1, 2.5]})
        ax0 = axs[0]
        ax0.set_title(plot_title)
        ax0.plot(fit_result.x, standardized_residual,'.',color='royalblue',markersize=8.5,label='residuals')
        #ax0.hlines(1,x_min,x_max,linestyle='dashed')
        ax0.hlines(0,x_min,x_max)
        #ax0.hlines(-1,x_min,x_max,linestyle='dashed')
        ax0.set_ylim(-1.05*np.max(np.abs(standardized_residual)), 1.05*np.max(np.abs(standardized_residual)))
        ax0.set_ylabel('Residual / $\sigma$')
        ax1 = axs[1]
        fit_result.plot_fit(ax=ax1,show_init=True,yerr=fit_result.y_err,data_kws={'color':'royalblue','marker':'.','markersize':'8.5'}, fit_kws={'color':'red','linewidth':'2'},init_kws={'linestyle':'dashdot','color':'green'})
        ax1.set_title('')
        ax1.set_ylim(-0.05*y_max_lin, 1.1*y_max_lin)
        ax1.set_ylabel('Counts per bin')
        for ax in axs:
            ax.legend()
            ax.set_xlim(x_min,x_max)
        if show_peak_markers:
            self.add_peak_markers(yscale='lin',ymax=y_max_lin,peaks=peaks_to_plot)
        plt.xlabel('m/z [u]')
        if plot_filename is not None:
            try:
                plt.savefig(plot_filename+'_lin_plot.png',dpi=500)
            except:
                raise
        plt.show()

        """
        f2 = plt.figure(figsize=(20,12))
        fit_result.plot(fig=f2,show_init=True,yerr=fit_result.y_err)
        if show_peak_markers:
            self.add_peak_markers(yscale='lin',ymax=y_max_lin,peaks=peaks_to_plot)
        ax_res, ax_fit = f2.axes
        ax_res.set_title(plot_title)
        ax_res.set_ylim(-1.05*y_max_res, 1.05*y_max_res)
        plt.xlabel('m/z [u]')
        plt.ylabel('Counts per bin')
        plt.xlim(x_min,x_max)
        plt.ylim(-0.05*y_max_lin, 1.1*y_max_lin)
        if plot_filename is not None:
            try:
                plt.savefig(plot_filename+'_lin_plot.png',dpi=500)
            except:
                raise
        plt.show()
        """


    ##### Plot fit of spectrum zoomed to specified peak or specified mass range
    def plot_fit_zoom(self,peak_indeces=None,x_center=None,x_range=0.01,show_peak_markers=True,sigmas_of_conf_band=0,plot_filename=None):
        """
        Show logarithmic and linear plots of data and fit curve zoomed to mass range of interest
        - if peak(s) of interest are specified via `peak_indeces` the mass range to plot is automatically chosen to include all specified peaks
        - otherwise, the mass range must be specified manually with `x_center` and `x_range`

        Parameters:
        -----------
        peak_indeces : int or list of ints, optional, default: None
            index of single peak or of multiple neighboring peaks to show (peaks must belong to the same fit curve!)
        x_center : float [u], optional, default: None
            center of manually specified mass range to plot
        x_range : float [u], optional, default: 0.01
            width of mass range to plot around 'x_center' or minimal width to plot around specified peaks of interest
        show_peak_markers : bool, optional, default: True
            if True, peak markers are added to the plots
        sigmas_of_conf_band : int, optional, default: 0
            coverage probability of confidence band in sigma (only for log-plot);
            if 0, no confidence band is shown (default)
        plot_filename : str or None, optional, default: None
            if not None, plot images will be saved to two separate files named '`plot_filename`_log_plot.png' and '`plot_filename`_lin_plot.png' respectively

        Returns:
        --------
        None
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
            print("Mass range to plot could not be determined. Check documentation on function parameters.")
            return
        self.plot_fit(x_min=x_min,x_max=x_max,show_peak_markers=show_peak_markers,sigmas_of_conf_band=sigmas_of_conf_band,plot_filename=plot_filename)


    ##### Internal helper function for creating multi-peak fit models
    def comp_model(self,peaks_to_fit=None,model='emg22',init_pars=None,vary_shape=False,vary_baseline=True,index_first_peak=None):
        """ create multi-peak composite model from single-peak model """
        model = getattr(fit_models,model) # get single peak model from fit_models.py
        mod = fit.models.ConstantModel(independent_vars='x',prefix='bkg_')
        if vary_baseline == True:
            mod.set_param_hint('bkg_c', value= 0.1, min=0,max=4, vary=True) # 0.3, vary=True
        else:
            mod.set_param_hint('bkg_c', value= 0.0, vary=False)
        df = self.data
        for peak in peaks_to_fit:
            peak_index = self.peaks.index(peak)
            x_pos = df.index[np.argmin(np.abs(df.index.values - peak.x_pos))] # x_pos of closest bin
            amp = max(df['Counts'].loc[x_pos]/1200,1e-04) # estimate amplitude from peak maximum, the factor 1200 is empirically determined and shape-dependent
            if init_pars:
                this_mod = model(peak_index, peak.x_pos, amp, init_pars=init_pars, vary_shape_pars=vary_shape, index_first_peak=index_first_peak)
            else:
                this_mod = model(peak_index, peak.x_pos, amp, vary_shape_pars=vary_shape, index_first_peak=index_first_peak)
            mod = mod + this_mod
        return mod


    ##### Fit spectrum
    def peakfit(self,fit_model='emg22',cost_func='chi-square',x_fit_cen=None,x_fit_range=0.01,init_pars=None,vary_shape=False,vary_baseline=True,method='least_squares',show_plots=True,show_peak_markers=True,sigmas_of_conf_band=0,plot_filename=None,eval_par_covar=False,recal_fac=1.0):
        """
        Internal peak fitting routine, fits full spectrum or subrange (if x_fit_cen and x_fit_range are specified) and optionally shows results
        This method is for internal usage, use 'fit_peaks' method to fit spectrum and update peak properties dataframe with obtained fit results!

	    Parameters:
        -----------
        fit_model (str): name of fit model to use (e.g. 'Gaussian','emg12','emg33', ... - see fit_models.py for all available fit models)
        cost_func : str, optional, default: 'chi-square'
            name of cost function to use for minimization
            if 'chi-square', the fit is performed using Pearson's chi squared statistic: cost_func = sum( (f(x_i) - y_i)**2/f(x_i)**2 )
            if 'MLE', a binned maximum likelihood estimation is performed using the negative log likelihood ratio: cost_func = sum( f(x_i) - y_i + y_i*log(y_i/f(x_i)) )
	    x_fit_cen (float [u], optional): center of mass range to fit (only specify if subset of spectrum is to be fitted)
	    x_fit_range (float [u], optional): width of mass range to fit (default: 0.01, only specify if subset of spectrum is to be fitted)
	    init_pars (dict): dictionary with initial parameters for fit (optional), if set to 'default' the default parameters from 'fit_models.py'
                              are used, if set to 'None' the parameters from the shape calibration are used (if those do not exist yet
                              the default parameters are used)
        vary_shape (bool): if 'False' peak shape parameters (sigma, eta's, tau's and theta) are kept fixed at initial values,
                           if 'True' the shape parameters are varied but shared amongst all peaks (identical shape parameters for all peaks)
        vary_baseline : bool, optional, default: True
            if True, the constant background will be fitted with a varying baseline paramter bkg_c (initial value: 0.1); otherwise the beseline paremter bkg_c will be fixed to 0.
        method (str, optional): name of minimization algorithm to use (default: least_squares, for full list of options c.f. 'The minimize() function' at https://lmfit.github.io/lmfit-py/fitting.html)
        recal_fac (float): factor for correction of the final mass values (obtain recalibration factor from calibrant fit before fitting other peaks)

	    Returns:
        --------
        Fit model result object
        """
        if x_fit_cen:
            x_min = x_fit_cen - x_fit_range/2
            x_max = x_fit_cen + x_fit_range/2
            df_fit = self.data[x_min:x_max] # cut data to fit range
            peaks_to_fit = [peak for peak in self.peaks if (x_min < peak.x_pos < x_max)] # select peaks in fit range
        else:
            df_fit = self.data
            x_min = df_fit.index.values[0]
            x_max = df_fit.index.values[-1]
            peaks_to_fit = self.peaks
        x = df_fit.index.values

        #for i in range(len(df_fit['Counts'].values)): ## chi-squared gamma dist.
        #    if df_fit['Counts'].iloc[i] >= 1: ##
        #        df_fit['Counts'].iloc[i] += 1 ##
        y = df_fit['Counts'].values
        y_err = np.sqrt(y+1) # assume Poisson (counting) statistics -> standard deviation of each point approximated by sqrt(counts+1)
        weights = 1./y_err # np.nan_to_num(1./y_err, nan=0.0, posinf=0.0, neginf=None) # makes sure that residuals include division by statistical error (residual = (fit_model - y) * weights)


        if init_pars == 'default':
            init_params = None
        elif init_pars is not None:
            init_params = init_pars
        else:
            init_params = self.shape_cal_pars # use shape parameters asociated with spectrum, unless other parameters are specified

        if vary_shape == True:
            index_first_peak = self.peaks.index(peaks_to_fit[0]) # enforce shared shape parameters for all peaks
        else:
            index_first_peak = None

        model_name = str(fit_model)+' + const. background (bkg_c)'
        mod = self.comp_model(peaks_to_fit=peaks_to_fit,model=fit_model,init_pars=init_params,vary_shape=vary_shape,vary_baseline=vary_baseline,index_first_peak=index_first_peak) # create multi-peak fit model
        pars = mod.make_params()

        ### Minimizer fitting routine
        """from scipy.special import expi
        def resid(pars, x, y_data): ## modified chi-square-gamma
            y_m = mod.eval(pars,x=x)
            mean_chi_sq_i = 1 + np.exp(-y_m)*(y_m-1)
            var_chi_sq_i = y_m**3 *np.exp(-y_m)*( expi(y_m) - np.euler_gamma - np.log(y_m) + 4 ) - y_m**2 - y_m + np.exp(-y_m)*(-2*y_m**2 + 2*y_m + 1) + np.exp(-2*y_m)*( -y_m**2 + 2*y_m -1 )
            return (y_m - y_data - mean_chi_sq_i)*np.sqrt(2/var_chi_sq_i)  + 1    ## (y_m - y_data)  # * weights  # / y_m
        o1 = fit.minimize(resid, pars, args=(x, y), method=method)
        print("# Fit using sum of squares:\n")
        fit.report_fit(o1)"""

        # if not vary_shape:
        #     # ### Binned MLE fit
        #     def red_fcn_MLE(r):
        #         return float((r).sum())
        #
        #     mod_MLE = copy.deepcopy(mod)
        #     def neg_log_likelihood_MLE(pars,y_data,weights,x=x):
        #         """ Requires model object 'mod_MLE' to be defined as global variable above! """
        #         y_m = mod_MLE.eval(pars,x=x)
        #         return np.log(spl.factorial(y_data)) + y_m - y_data*np.log(y_m)
        #     mod_MLE._residual = neg_log_likelihood_MLE # overwrite lmfit's least square residuals with log likelihood
        #     out = mod_MLE.fit(y, pars, x=x, weights=weights, method='Nelder-Mead', calc_covar=False, nan_policy='omit',fit_kws={'reduce_fcn': red_fcn_MLE,'tol':1e-11})
        #     out.x = x
        #     out.y = y
        #     out.y_err = y_err
        #     out.residual = (out.eval()-y)*weights
        #     plt.errorbar(x,y,y_err)
        #     plt.plot(x,out.eval())
        #     plt.yscale('log')
        #     plt.show()
        #     print(out.fit_report())
        # else:


        # Perform fit, print fit report

        # Pearson's chi square with iterative weights
        if cost_func == 'chi-square': #if vary_shape:
            mod_Pearson = mod
            def resid_Pearson_chi_square(pars,y_data,weights,x=x):
                y_m = mod_Pearson.eval(pars,x=x)
                weights = np.where(y_m<=1e-15,1,1./np.sqrt(y_m)) ## chi-squared dist. with iteratively adapted weights 1/Sqrt(f(x_i)), non-zero minimal bounds implemented for numerical stability
                return (y_m - y_data)*weights
            mod_Pearson._residual = resid_Pearson_chi_square # overwrite lmfit's least square residuals with iterative residuals for Pearson chi-square
            out = mod_Pearson.fit(y, params=pars, x=x, weights=weights, method=method, scale_covar=False,nan_policy='propagate')
            y_m = out.best_fit
            Pearson_weights = np.where(y_m<=1e-15,1,1./np.sqrt(y_m)) # 1/Sqrt(f(x_i)) Pearson weights, non-zero minimal bounds implemented for numerical stability for
            out.y_err = 1/Pearson_weights
            #out = mod.fit(y, params=pars, x=x, weights=weights, method=method, scale_covar=False)
        elif cost_func == 'MLE':
            mod_MLE = mod
            def neg_log_likelihood_ratio_MLE_model_leastsq(pars,y_data,weights,x=x):
                y_m = mod_MLE.eval(pars,x=x)
                log_likelihood_ratio = np.abs( 2*(y_m - y_data + np.nan_to_num(y_data*np.log(y_data/y_m))))
                return np.sqrt(log_likelihood_ratio) #np.sqrt(np.log(spl.factorial(y_data)) + y_m - y_data*np.log(y_m))
            mod_MLE._residual = neg_log_likelihood_ratio_MLE_model_leastsq # overwrite lmfit's least square residuals with log likelihood
            out = mod_MLE.fit(y, params=pars, x=x, weights=weights, method=method, calc_covar=False, nan_policy='propagate') # 'user_fcn':neg_log_likelihood_MLE,
            out.y_err = 1/out.weights
        else:
            print("Definition of `cost_func` failed! Fit aborted.")
            return
        out.x = x
        out.y = y
        out.cost_func = cost_func
        out.fit_model = fit_model
        #out.y_err = 1/out.weights #y_err

        ### Binned MLE fit - initialized at chi-square fit best values
        # if not vary_shape:
        #     def red_fcn_MLE(r):
        #         return float((r).sum())
        #
        #     mod_MLE = copy.deepcopy(mod) # ensure original model object is not altered
        #     def neg_log_likelihood_MLE(pars,y_data,weights,x=x):
        #         """ Requires model object 'mod_MLE' to be defined as global variable above! """
        #         y_m = mod_MLE.eval(pars,x=x)
        #         return np.log(spl.factorial(y_data)) + y_m - y_data*np.log(y_m)
        #     mod_MLE._residual = neg_log_likelihood_MLE # overwrite lmfit's least square residuals with log likelihood
        #     out_MLE = mod_MLE.fit(y, out.params, x=x, weights=weights, method='Nelder-Mead', calc_covar=False, nan_policy='omit',fit_kws={'reduce_fcn': red_fcn_MLE,'tol':1e-9})
        #     f = plt.figure(figsize=(15,8))
        #     #plt.errorbar(x,y,y_err)
        #     #plt.plot(x,out_MLE.eval())
        #     out_MLE.plot(fig=f,show_init=True)
        #     plt.yscale('log')
        #     plt.show()
        #     print(out_MLE.fit_report())

        if eval_par_covar:
            ## Add emcee MCMC sampling
            emcee_kws = dict(steps=5000, burn=2500, thin=20, is_weighted=True, progress=True)
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


    ##### Internal helper function for calculating the peak area (number of counts in peak)
    def calc_peak_area(self, peak_index, fit_result=None, decimals=2):
        """ Calculate peak area (total counts in peak) and its error for a given peak using the peak amplitude and the bin_width

        Parameters:
        -----------
        peak_index (str): Index of peak of interest
        fit_result (modelresult): lmfit modelresult object to use for area calculation (optional), if 'None': use corresponding modelresult stored in list fit_results
        decimals (int): number of decimals to return outputs with


        Returns:
        --------
        list of two floats: [area,area_error]
        """
        pref = 'p'+str(peak_index)+'_'
        area, area_err = np.nan, np.nan
        #area = 0
        #for y_i in fit_result.eval_components(x=fit_result.x)[pref]: # Get counts in subpeaks from best fit to data
        #    area += y_i
        #    if np.isnan(y_i) == True:
        #    print("Warning: Encountered NaN values in "+str(self.peaks[peak_index].species)+"-subpeak! Those are omitted in area summation.")
        if fit_result is None:
            fit_result = self.fit_results[peak_index]
        bin_width = self.data.index[1] - self.data.index[0] # width of mass bins, needed to convert peak amplitude (peak area in units Counts/mass range) to Counts
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
            print('WARNING: Area error determination failed. Could not get amplitude parameter ("amp") of peak. Likely the peak has not been fitted successfully yet.')
            pass
        return [area, area_err]


    ##### Internal helper function for calculating FWHM of Hyper-EMG function
    @staticmethod
    def calc_FWHM_emg(peak_index,fit_result=None):
        """ Calculates FWHM of a EMG function"""
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

    ##### Internal helper function for calculating std. dev. of Hyper-EMG fit result
    @staticmethod
    def calc_sigma_emg(peak_index,fit_model=None,fit_result=None):
        pref = 'p{0}_'.format(peak_index)
        try:
            no_left_tails = int(fit_model[3])
            no_right_tails = int(fit_model[4])
        except TypeError:
            print('\nERROR: Calculation of sigma_emg failed. Fit model parameter must be string of form "emgXY" with X & Y int!\n')
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

    ##### Internal helper function for creating synthetic spectra via bootstrap re-sampling from experimental data
    @staticmethod
    def bootstrap_spectrum(df,N_events=None,x_cen=None,x_range=0.02):
        """ Create new histogram dataframe with `N_events` samples via bootstrap re-sampling from `df` """
        if x_cen:
            x_min = x_cen - x_range/2
            x_max = x_cen + x_range/2
            df = df[x_min:x_max]
        mass_bins = df.index.values
        counts = df['Counts'].values

        # Convert histogrammed spectrum (equivalent to MAc HIST export mode) to list of events (equivalent to MAc LIST export mode)
        orig_events = np.array([])
        for i in range(len(mass_bins)):
            orig_events = np.append(orig_events,[mass_bins[i]]*int(counts[i]))

        # Create new DataFrame of events by bootstrapping from `orig_events`
        if N_events == None:
            N_events = len(orig_events)
        random_indeces = np.random.randint(0,len(orig_events),N_events)
        events = orig_events[random_indeces]
        df_events = pd.DataFrame(events)

        # Convert list of events back to a DataFrame with histogrammed spectrum data
        bin_centers = df.index.values
        bin_width = df.index.values[1] - df.index.values[0]
        bin_edges = np.append(bin_centers-bin_width/2,bin_centers[-1]+bin_width/2)
        hist = np.histogram(df_events,bins=bin_edges)
        df_new = pd.DataFrame(data=hist[0],index=bin_centers,dtype=float,columns=["Counts"])
        df_new.index.name = "Mass [u]"
        return df_new


    def determine_A_stat_emg(self,peak_index,x_range=0.01,N_spectra=1000,fit_model=None,cost_func='MLE',method='least_squares',vary_baseline=True):
        """
        Determine the factor of proportionality for stat. error estimation A_stat_emg via bootstrap re-sampling from a peak in the spectrum. The relevant equation for this is: Stat. error = A_stat_emg * FWHM /np.sqrt(N_counts)
        The resulting value for A_stat_emg will be stored as spectrum attribute and will be used for all subsequent stat. error determinations.
        This routine must be called AFTER a successful peak-shape calibration (`determine_peak_shape` method) and should be called BEFORE the mass calibration.

        `N_spectra` bootstrapped spectra are created for each of the following total numbers of events: [10,30,100,300,1000,3000,10000,30000]
        Each bootstrapped spectrum is fitted to determine the peak centroid. A_stat_emg is then determined from a fit to the relative standard deviation of the peak centroid as function of determined peak area.

        Parameters:
        -----------
        peak_index : int
            index of peak to use for bootstrap re-sampling (typically, the peak-shape calibrant). The peak should have high statistics and be well-separated from other peaks to be representative for all peaks in the spectrum.
        x_range : float [u], optional, default: 0.01
            mass range around peak centroid over which events will be sampled and fitted. Choose such that no secondary peaks are contained in the mass range.
        N_spectra : int, optional, default: 1000
            number of bootstrapped spectra to create at each number of ions
        cost_func : str, optional, default: 'MLE'
            name of cost function to use for minimization
            if 'chi-square', the fit is performed using Pearson's chi squared statistic: cost_func = sum( (f(x_i) - y_i)**2/f(x_i)**2 )
            if 'MLE', a binned maximum likelihood estimation is performed using the negative log likelihood ratio: cost_func = sum( f(x_i) - y_i + y_i*log(y_i/f(x_i)) )

            #############
        """
        if fit_model is None:
            fit_model = self.fit_model
        x_cen = self.peaks[peak_index].x_pos
        no_peaks_in_range = len([p for p in self.peaks if (x_cen - x_range/2) <= p.x_pos <= (x_cen + x_range/2)])
        if no_peaks_in_range > 1:
            print("WARNING: More than one peak in current mass range. This routine can only be assumed to be accurate for well-separated, single peaks. It is strongly advisable to chose a smaller `x_range`!\n")
            #return
        li_N_counts = [10,30,100,300,1000,3000,10000,30000]
        print("Creating and fitting boostrapped spectra for A_stat determination, depending on the choice of `N_spectra` this can take a few minutes. Interrupt kernel if this takes too long.")
        np.random.seed(seed=0) # to make bootstrapped spectra reproducible
        std_devs_of_mus = np.array([]) # standard deviation of sample means mu
        mean_areas = np.array([]) # array for numbers of detected counts
        for N_counts in li_N_counts:
            mus = np.array([])
            areas = np.array([])
            for i in range(N_spectra):
                df_boot = spectrum.bootstrap_spectrum(self.data,N_events=N_counts,x_cen=x_cen,x_range=x_range) # create boostrapped spectrum data
                spec_boot = spectrum(None,df=df_boot,show_plot=False) # create boostrapped spectrum object
                spec_boot.add_peak(x_cen,verbose=False)
                fit_result = spec_boot.peakfit(fit_model=self.fit_model,x_fit_cen=x_cen,x_fit_range=x_range,cost_func=cost_func,method=method,vary_baseline=vary_baseline,init_pars=self.shape_cal_pars,show_plots=False) # fit boostrapped spectrum with model and (fixed) shape parameters from peak-shape calibration
                # Record peak centroid and area of hyper-EMG fit
                mus = np.append(mus,fit_result.params['p0_mu'].value)
                areas = np.append(areas,spec_boot.calc_peak_area(0, fit_result=fit_result, decimals=2)[0])
            std_devs_of_mus = np.append(std_devs_of_mus,np.std(mus,ddof=1))
            mean_areas = np.append(mean_areas,np.mean(areas))
        mean_mu = np.mean(mus) # from last `N_counts` step only
        FWHM_gauss = 2*np.sqrt(2*np.log(2))*fit_result.params['p0_sigma'].value
        FWHM_emg = spec_boot.calc_FWHM_emg(peak_index=0,fit_result=fit_result)
        FWHM_emg_err = FWHM_gauss/FWHM_emg * self.shape_cal_par_errors['sigma']
        print("Done!\n")

        x = mean_areas # use number of detected counts instead of true number of re-sampling events (li_N_counts)
        model = fit.models.PowerLawModel()
        pars = model.make_params()
        pars['exponent'].value = -0.5
        pars['exponent'].vary = False
        out = model.fit(std_devs_of_mus,x=x,params=pars)
        print(out.fit_report())

        A_stat_gauss = 1/(2*np.sqrt(2*np.log(2)))
        A_stat_emg = out.best_values['amplitude']/FWHM_emg
        A_stat_emg_error = np.sqrt( (out.params['amplitude'].stderr/FWHM_emg)**2 + (out.best_values['amplitude']*FWHM_emg_err/FWHM_emg**2)**2 )

        y = std_devs_of_mus/mean_mu
        f = plt.figure(figsize=(15,8))
        plt.plot(x,y,'o')
        plt.plot(x,out.best_fit/mean_mu)
        plt.plot(x,A_stat_gauss*FWHM_gauss/(np.sqrt(x)*mean_mu),'--')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Peak area [counts]")
        plt.ylabel("Relative statistical uncertainty")
        plt.legend(["Standard deviations of sample means","Stat. error of Hyper-EMG","Stat. error of underlying Gaussian"])
        plt.show()

        self.determined_A_stat_emg = cost_func
        self.A_stat_emg = A_stat_emg
        self.A_stat_emg_error = A_stat_emg_error
        print("A_stat of a Gaussian model:",np.round(A_stat_gauss,3))
        print("Default A_stat_emg for Hyper-EMG models:",np.round(A_stat_emg_default,3))
        print("A_stat_emg for this spectrum's",self.fit_model,"fit model:",np.round(self.A_stat_emg,3),"+-",np.round(self.A_stat_emg_error,3))


    def calc_peakshape_errors(self,peak_indeces=[],x_fit_cen=None,x_fit_range=None,fit_result=None,fit_model=None,cost_func='MLE',method='least_squares',vary_baseline=True,verbose=False,show_shape_err_fits=False):
        """
        Calculates the relative peak-shape uncertainty of the specified peaks. This is done by re-fitting the specified peaks with each shape parameter individually varied by plus and minus sigma and recording the respective shift of the peak centroid w.r.t the original fit.
        The maximal centroid shifts obtained for each varied parameter are then added in quadrature to obtain the total peak shape uncertainty.
        NOTE: All peaks in 'peak_indeces' list must have been fitted in the same multi-peak fit (and hence have the same lmfit modelresult 'fit_result')!

        Parameters:
        -----------
        peak_indeces (list): list containing indeces of peaks to evaluate peak-shape uncertainty for, e.g. to evaluate peak-shape error of peaks 0 and 3 use peak_indeces=[0,3]
        x_fit_cen (float): center of mass range of fit to evaluate peak-shape error for
        x_fit_range (float): width of mass range of fit to evaluate peak-shape error for
        fit_result (lmfit modelresult, optional): fit result object to evaluate peak-shape error for
        fit_model (str): name of fit model used to obtain fit result
        cost_func : str, optional, default: 'MLE'
            name of cost function used for minimization
	   method (str, optional): name of minimization algorithm to use (default: least_squares, for full list of options c.f. 'The minimize() function' at https://lmfit.github.io/lmfit-py/fitting.html)
        vary_baseline : bool, optional, default: True
            if True, the constant background will be fitted with a varying baseline paramter bkg_c (initial value: 0.1); otherwise the beseline paremter bkg_c will be fixed to 0.
        verbose (bool, optional): if True, print all individual centroid shifts caused by varying the shape parameters (default: False)
        show_shape_err_fits (bool, optional): if True, show individual plots of re-fits for peak-shape error determination (default: False)
        """
        if self.shape_cal_pars is None:
            print('\nWARNING: Could not calculate peak-shape errors - no peak-shape calibration yet!\n')
            return
        if self.index_mass_calib in peak_indeces:
            peak_indeces.remove(self.index_mass_calib)

        print('\n##### Peak-shape uncertainty evaluation #####\n')
        # Vary each shape parameter by plus and minus one sigma and sum resulting shifts of Gaussian centroid in quadrature to obtain peakshape error
        if fit_result is None:
            fit_result = self.fit_results[peak_indeces[0]]
        pref = 'p{0}_'.format(peak_indeces[0])
        shape_pars = [key for key in self.shape_cal_pars if (key.startswith(('sigma','theta','eta','tau','delta')) and fit_result.params[pref+key].expr is None )] # grab shape parameters to be varied by +/- sigma
        if self.centroid_shifts is None:
            self.centroid_shifts_pm = np.array([{} for i in range(len(self.peaks))]) # initialize array of empty dictionaries
            self.centroid_shifts = np.array([{} for i in range(len(self.peaks))]) # initialize array of empty dictionaries
        for par in shape_pars:
            pars = copy.deepcopy(self.shape_cal_pars) # deep copy to avoid changes in original dictionary

            pars[par] = self.shape_cal_pars[par] + self.shape_cal_par_errors[par]
            if par == 'delta_m':
                pars['eta_m2'] = pars[par] - self.shape_cal_pars['eta_m1']
                pars['eta_m3'] = 1 - self.shape_cal_pars['eta_m1'] + pars['eta_m2']
            elif par == 'delta_p':
                pars['eta_p2'] = pars[par] - self.shape_cal_pars['eta_p1']
                pars['eta_p3'] = 1 - self.shape_cal_pars['eta_p1'] + pars['eta_p2']
            fit_result_p = self.peakfit(fit_model=self.fit_model, cost_func=cost_func, x_fit_cen=x_fit_cen, x_fit_range=x_fit_range, init_pars=pars, vary_shape=False, vary_baseline=vary_baseline, method=method, show_plots=False)
            #display(fit_result_p) # show fit result

            pars[par] = self.shape_cal_pars[par] - self.shape_cal_par_errors[par]
            if par == 'delta_m':
                pars['eta_m2'] =  pars[par] - self.shape_cal_pars['eta_m1']
                pars['eta_m3'] = 1 - self.shape_cal_pars['eta_m1'] +  pars['eta_m2']
            elif par == 'delta_p':
                pars['eta_p2'] =  pars[par] - self.shape_cal_pars['eta_p1']
                pars['eta_m3'] = 1 - self.shape_cal_pars['eta_p1'] +  pars['eta_p2']
            fit_result_m = self.peakfit(fit_model=self.fit_model, cost_func=cost_func, x_fit_cen=x_fit_cen, x_fit_range=x_fit_range, init_pars=pars, vary_shape=False, vary_baseline=vary_baseline, method=method, show_plots=False)
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

            for peak_idx in peak_indeces:
                pref = 'p{0}_'.format(peak_idx)
                centroid = fit_result.best_values[pref+'mu']
                new_centroid_p =  fit_result_p.best_values[pref+'mu'] #[value for key, value in fit_result_p.best_values.items() if pref in key][1] # indexing makes sure that both Gaussian 'center' and Hyper-EMG 'mu' Parameters get fetched
                delta_mu_p = new_centroid_p - centroid
                new_centroid_m = fit_result_m.best_values[pref+'mu'] #[value for key, value in fit_result_m.best_values.items() if pref in key][1] # indexing makes sure that both Gaussian 'center' and Hyper-EMG 'mu' Parameters get fetched
                delta_mu_m = new_centroid_m - centroid
                if verbose:
                    print('Re-fitting with ',par,' = ',np.round(self.shape_cal_pars[par],6),'+/-',np.round(self.shape_cal_par_errors[par],6),' shifts centroid of peak',peak_idx,'by ',np.round(delta_mu_p*1e06,6),'/',np.round(delta_mu_m*1e06,3),'\u03BCu.')
                    if peak_idx == peak_indeces[-1]:
                        print()  # empty line between different parameter blocks
                self.centroid_shifts_pm[peak_idx][par+' centroid shift'] = [delta_mu_p,delta_mu_m]
                self.centroid_shifts[peak_idx][par+' centroid shift'] = np.where(np.abs(delta_mu_p) > np.abs(delta_mu_m),delta_mu_p,delta_mu_m).item()

        for peak_idx in peak_indeces:
            shape_error = np.sqrt(np.sum(np.square( list(self.centroid_shifts[peak_idx].values()) ))) # add centroid shifts in quadrature to obtain total peakshape error
            p = self.peaks[peak_idx]
            m_fit = fit_result.best_values[pref+'mu']*self.recal_fac
            p.rel_peakshape_error = shape_error/m_fit
            if verbose:
                pref = 'p{0}_'.format(peak_idx)
                print("Relative peak-shape error of peak "+str(peak_idx)+":",np.round(p.rel_peakshape_error,9))


    ##### Determine peak shape
    def determine_peak_shape(self, index_shape_calib=None, species_shape_calib=None, fit_model='emg22', cost_func='chi-square', init_pars = 'default', x_fit_cen=None, x_fit_range=0.01, vary_baseline=True, method='least_squares', vary_tail_order=True, show_fit_reports=True, show_plots=True, show_peak_markers=True, sigmas_of_conf_band=0):
        """
        Determine optimal tail order and peak shape parameters by fitting the selected peak-shape calibrant.

        If vary_tail_order is False, the automatic tail order determination is skipped. Otherwise the routine tries to find the peak shape that minimizes chi squared reduced by succesively adding more tails on the right and left.
        Finally, the fit model is selected which gives the lowest chi-square without having any of the tail weight parameters `eta` compatible with zero within errorbars. The latter models are excluded as is this a sign of overfitting.
        Likewise, models for which the calculation of eta parameter uncertainties fails are excluded from selection.

        It is further recommended to visually check whether the residuals are purely stochastic (as should be the case for a decent model).



        Parameters:
        -----------

        index_shape_calib : int or None, optional
            index of shape-calibration peak (the peak to use can also be specified with `species_shape_calib`)
        species_shape_calib : str or None, optional
            species name of shape calibrant (e.g. 'K39:-1e', the peak to use can also be specified with `index_shape_calib`)
        fit_model : str, optional, default: 'emg22'
            name of fit model to use for shape calibration (e.g. 'Gaussian','emg12','emg33', ... - see fit_models.py for all available fit models))
            If the automatic model selection (`vary_tail_order=True`) fails or is turned off, `fit_model` will be used for the shape calibration and set as the spectrum's fit model (fit_model spectrum attribute)
        cost_func : str, optional, default: 'chi-square'
            name of cost function to use for minimization
            if 'chi-square', the fit is performed using Pearson's chi squared statistic: cost_func = sum( (f(x_i) - y_i)**2/f(x_i)**2 )
            if 'MLE', a binned maximum likelihood estimation is performed using the negative log likelihood ratio: cost_func = sum( f(x_i) - y_i + y_i*log(y_i/f(x_i)) )
        init_pars : dict or 'default' or None, optional, default: 'default' (default parameters defined in fit_models.py script are used)
            initial model parameters for fit,
            must be supplied as a dictionary with parameter names as keys and parameter values as values
        x_fit_cen : float [u] or None, optional, default: None
            center of fit range; if None, the x_pos of the shape-calibration peak is used as `x_fit_cen`
        x_fit_range : float [u], optional, default: 0.01
            mass range to fit
        vary_baseline : bool, optional, default: True
            if True, the constant background will be fitted with a varying baseline paramter bkg_c (initial value: 0.1); otherwise the beseline paremter bkg_c will be fixed to 0.
        method : str, optional, default: 'least_squares'
            name of minimizer to use
        vary_tail_order : bool, optional, default: True
            if True, an automatic selection of the best fit model is performed
            if False, the `fit_model` parameter is used as fit model
        show_fit_reports : bool, optional, default: True
            whether to show fit reports for the fits in the automatic model selectio
        show_plots : bool, optional, default: 'True'
            whether to show plots with data and fit curves
        show_peak_markers : bool, optional, default: True
            if True, peak markers are included in the plots
        sigmas_of_conf_band : int, optional, default: 0
            coverage probability of confidence band in sigma (only for log-plot);
             if 0, no confidence band is shown (default)
        """
        if index_shape_calib is not None and (species_shape_calib is None):
            peak = self.peaks[index_shape_calib]
        elif species_shape_calib:
            index_shape_calib = [i for i in range(len(self.peaks)) if species_shape_calib == self.peaks[i].species][0]
            peak = self.peaks[index_shape_calib]
        else:
            print("\nERROR: Definition of peak shape calibrant failed. Define EITHER the index OR the species name of the peak to use as shape calibrant!\n")
            return

        if x_fit_cen is None:
            x_fit_cen = peak.x_pos

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
                    print("\n##### Fitting data with",model,"#####-----------------------------------------------------------------------------------------\n")
                    out = spectrum.peakfit(self, fit_model=model, cost_func=cost_func, x_fit_cen=x_fit_cen, x_fit_range=x_fit_range, init_pars=init_pars ,vary_shape=True, vary_baseline=vary_baseline, method=method,show_plots=show_plots,show_peak_markers=show_peak_markers,sigmas_of_conf_band=sigmas_of_conf_band)
                    idx = li_fit_models.index(model)
                    li_red_chis[idx] = np.round(out.redchi,2)
                    li_red_chi_errs[idx] =  np.round(np.sqrt(2/out.nfree),2)
                    if model.startswith('emg') and model not in ['emg01','emg10','emg11']: # check emg models with tail orders >= 2 for overfitting (i.e. a eta parameter agress with zero within its error)
                        no_left_tails = int(model[3])
                        no_right_tails = int(model[4])
                        first_parname = list(out.params.keys())[2] # must use first peak to be fit, since only its shape parameters are all varying
                        pref = first_parname.split('_')[0]+'_'
                        if no_left_tails > 1:
                            for i in np.arange(1,no_left_tails+1):
                                par_name = pref+"eta_m"+str(i)
                                val = out.params[par_name].value
                                err = out.params[par_name].stderr
                                try:
                                    if val < err:
                                        print("WARNING:",par_name,"=",np.round(val,3),"+-",np.round(err,3)," is compatible with zero within uncertainty.")
                                        print("             This tail order is likely overfitting the data and will be excluded from selection.")
                                        li_eta_flags[idx] = True # mark model in order to exclude it below
                                except TypeError:
                                    print("WARNING: parameter uncertainty of",par_name,"could not be calculated! This tail order will be excluded from selection.")
                                    li_eta_flags[idx] = True # mark model in order to exclude it below
                        if no_right_tails > 1:
                            for i in np.arange(1,no_right_tails+1):
                                par_name = pref+"eta_p"+str(i)
                                val = out.params[par_name].value
                                err = out.params[par_name].stderr
                                try:
                                    if val < err:
                                        print("WARNING:",par_name,"=",np.round(val,3),"+-",np.round(err,3)," is compatible with zero within uncertainty.")
                                        print("             This tail order is likely overfitting the data and will be excluded from selection.")
                                        li_eta_flags[idx] = True  # mark model in order to exclude it below
                                except TypeError:
                                    print("WARNING: parameter uncertainty of",par_name,"could not be calculated! This tail order will be excluded from selection.")
                                    li_eta_flags[idx] = True # mark model in order to exclude it below
                    print("\n"+str(model)+"-fit yields reduced chi-square of: "+str(li_red_chis[idx])+" +- "+str(li_red_chi_errs[idx]))
                    print()
                    if show_fit_reports:
                        display(out) # show fit report
                except ValueError:
                    print('\nWARNING:',model+'-fit failed due to NaN-values and was skipped! ----------------------------------------------\n')
            idx_best_model = np.nanargmin(np.where(li_eta_flags, np.inf, li_red_chis)) # exclude models with eta_flag == True
            best_model = li_fit_models[idx_best_model]
            self.fit_model = best_model
            self.best_redchi = li_red_chis[idx_best_model]
            print('##### Result of automatic model selection: #####')
            print('\nBest fit model determined to be:',best_model)
            print('Corresponding chi²-reduced:',self.best_redchi)
        elif vary_tail_order == False:
            self.fit_model = fit_model

        print('\n##### Peak-shape determination #####-------------------------------------------------------------------------------------------\n')
        out = spectrum.peakfit(self, fit_model=self.fit_model, cost_func=cost_func, x_fit_cen=x_fit_cen, x_fit_range=x_fit_range, init_pars=init_pars ,vary_shape=True, vary_baseline=vary_baseline, method=method,show_plots=show_plots,show_peak_markers=show_peak_markers,sigmas_of_conf_band=sigmas_of_conf_band,eval_par_covar=False)

        peak.comment = 'shape calibrant'
        display(out)  # print(out.fit_report())
        dict_pars = out.params.valuesdict()
        self.shape_cal_pars = {key.lstrip('p'+str(index_shape_calib)+'_'): val for key, val in dict_pars.items() if key.startswith('p'+str(index_shape_calib))}
        self.shape_cal_pars['bkg_c'] = dict_pars['bkg_c']
        self.shape_cal_par_errors = {} # dict to store shape calibration parameter errors
        for par in out.params:
            if par.startswith('p'+str(index_shape_calib)):
                self.shape_cal_par_errors[par.lstrip('p'+str(index_shape_calib)+'_')] = out.params[par].stderr
        self.shape_cal_par_errors['bkg_c'] = out.params['bkg_c'].stderr

        #return out


   ##### Save shape calibration parameters to TXT file
    def save_peak_shape_cal(self,filename):
        """ Save peak shape parameters (and their errors, if existent) to the TXT file 'filename.txt' """
        df1 = pd.DataFrame.from_dict(self.shape_cal_pars,orient='index',columns=['Value'])
        df1.index.rename('Model: '+str(self.fit_model),inplace=True)
        df2 = pd.DataFrame.from_dict(self.shape_cal_par_errors,orient='index',columns=['Error'])
        df = df1.join(df2)
        df.to_csv(str(filename)+'.txt', index=True,sep='\t')
        print('\nPeak-shape calibration saved to file: '+str(filename)+'.txt')


    ###### Load shape calibration parameters from TXT file
    def load_peak_shape_cal(self,filename):
        """ Load peak shape from the TXT file 'filename.txt' """
        df = pd.read_csv(str(filename)+'.txt',index_col=0,sep='\t')
        self.fit_model = df.index.name[7:]
        df_val = df['Value']
        df_err = df['Error']
        self.shape_cal_pars = df_val.to_dict()
        self.shape_cal_par_errors = df_err.to_dict()
        print('\nLoaded peak shape calibration from '+str(filename)+'.txt')


    ##### Fit mass calibrant
    def fit_calibrant(self, index_mass_calib=None, species_mass_calib=None, fit_model=None, cost_func='MLE', x_fit_cen=None, x_fit_range=0.01, vary_baseline=True, method='least_squares',show_plots=True,show_peak_markers=True,sigmas_of_conf_band=0,show_fit_report=True):
        """
        Determine mass recalibration factor for spectrum by fitting the selected mass calibrant

        Parameters:
        -----------
        fit_model : str, optional, default: None
            name of fit model to use, (names of usable models: 'Gaussian','emg01', ..., 'emg33' - see fit_models.py for all available fit models)
            if None, defaults to 'fit_model' attribute of spectrum object (as determined in peak-shape calibration)
        cost_func : str, optional, default: 'MLE'
            name of cost function to use for minimization
            if 'chi-square', the fit is performed using Pearson's chi squared statistic: cost_func = sum( (f(x_i) - y_i)**2/f(x_i)**2 )
            if 'MLE', a binned maximum likelihood estimation is performed using the negative log likelihood ratio: cost_func = sum( f(x_i) - y_i + y_i*log(y_i/f(x_i)) )
        x_fit_cen : float or None, [u], optional
            center of mass range to fit;
            if None, defaults to marker position (x_pos) of mass calibrant peak
        x_fit_range : float [u], optional, default: 0.01
            width of mass range to fit

        vary_baseline : bool, optional, default: True
            if True, the constant background will be fitted with a varying baseline paramter bkg_c (initial value: 0.1); otherwise the beseline paremter bkg_c will be fixed to 0.
        show_fit_report : bool, optional, default: True
            show detailed report with fit statistics and fit parameter results

        ###########

        """
        if index_mass_calib is not None and (species_mass_calib is None):
            peak = self.peaks[index_mass_calib]
        elif species_mass_calib:
            index_mass_calib = [i for i in range(len(self.peaks)) if species_mass_calib == self.peaks[i].species][0]
            peak = self.peaks[index_mass_calib]
        else:
            print("\nERROR: Definition of peak shape calibrant failed. Define EITHER the index OR the species name of the peak to use as shape calibrant!\n")
            return
        for p in self.peaks: # reset 'mass calibrant' flag
            if 'shape & mass calibrant' in p.comment :
                p.comment = p.comment.replace('shape & mass calibrant','shape calibrant')
            elif p.comment == 'mass calibrant':
                p.comment = '-'
            elif 'mass calibrant' in p.comment:
                p.comment = p.comment.replace('mass calibrant','')
        if 'shape calibrant' in peak.comment:
            peak.comment = peak.comment.replace('shape calibrant','shape & mass calibrant')
        elif peak.comment == '-' or peak.comment == '' or peak.comment is None:
            peak.comment = 'mass calibrant'
        else:
            peak.comment = 'mass calibrant, '+peak.comment

        print('##### Calibrant fit #####')
        if fit_model is None:
            fit_model = self.fit_model
        if x_fit_cen is None:
            x_fit_cen = peak.x_pos
        fit_result = spectrum.peakfit(self, fit_model=fit_model, cost_func=cost_func, x_fit_cen=x_fit_cen, x_fit_range=x_fit_range, vary_shape=False, vary_baseline=vary_baseline, method=method, show_plots=show_plots, show_peak_markers=show_peak_markers, sigmas_of_conf_band=sigmas_of_conf_band)
        if show_fit_report:
            display(fit_result)

        # Update peak properties
        pref = 'p{0}_'.format(index_mass_calib)
        peak.fit_model = fit_model
        peak.cost_func = cost_func
        peak.area, peak.area_error = self.calc_peak_area(index_mass_calib,fit_result=fit_result)
        peak.m_fit = fit_result.best_values[pref+'mu']
        if peak.fit_model == 'Gaussian':
            std_dev = fit_result.best_values[pref+'sigma']
        else:  # for emg models
            FWHM_emg = spectrum.calc_FWHM_emg(index_mass_calib,fit_result=fit_result)
            std_dev = self.A_stat_emg*FWHM_emg  #spectrum.calc_sigma_emg(peak_index=index_mass_calib,fit_model=fit_model,fit_result=fit_result)
        stat_error = std_dev/np.sqrt(peak.area) # A_stat* FWHM/sqrt(area), w/ with A_stat_G = 0.42... and A_stat_emg from `determine_A_stat` method or default value from config.py
        peak.rel_stat_error = stat_error /peak.m_fit
        peak.rel_peakshape_error = 0 # initialize at 0 to allow for calculation of rel_recal_error below in case calc_peakshpe_errors fails
        self.index_mass_calib = None # Reset index of mass calibrant to enable re-calculation of calibrant peak-shape error in case of foregoing calibration attempt
        self.calc_peakshape_errors(peak_indeces=[index_mass_calib],x_fit_cen=x_fit_cen,x_fit_range=x_fit_range,fit_result=fit_result,vary_baseline=vary_baseline,fit_model=fit_model,cost_func=cost_func,method=method,verbose=True)

        peak.chi_sq_red = np.round(fit_result.redchi, 2)

        # Print error contributions of mass calibrant:
        print("\nRelative literature error of mass calibrant:   ",np.round(peak.m_AME_error/peak.m_fit,9))
        print("Relative statistical error of mass calibrant:  ",np.round(peak.rel_stat_error,9))
        if peak.rel_peakshape_error != 0:
            print("Relative peak-shape error of mass calibrant:   ",np.round(peak.rel_peakshape_error,9))
        else:
            print("\nWARNING: Peak-shape error of mass calibrant could not be evaluated. The calculation of the recalibration error will ensue without peak-shape error contribution.")

        # Determine recalibration factor
        self.recal_fac = peak.m_AME/peak.m_fit
        print("\nRecalibration factor:  "+str(self.recal_fac))
        if np.abs(self.recal_fac - 1) > 1e-02:
            print("\nWARNING: recalibration factor `recal_fac` deviates from unity by more than a permille.-----------------------------------------------")
            print(  "         Potentially, mass errors should also be re-scaled with `recal_fac` (currently not implemented)!-----------------------------")

        # Update peak properties with new calibrant centroid
        peak.m_fit = self.recal_fac*peak.m_fit # update centroid mass of calibrant peak
        if peak.A:
            peak.atomic_ME_keV = np.round((peak.m_fit + m_e - peak.A)*u_to_keV,3)   # atomic Mass excess (includes electron mass) [keV]
        if peak.m_AME:
            peak.m_dev_keV = np.round( (peak.m_fit - peak.m_AME)*u_to_keV, 3) # TITAN - AME [keV]

        # Determine rel. recalibration error and update recalibration error attribute
        peak.rel_recal_error = np.sqrt( (peak.m_AME_error/peak.m_AME)**2 + peak.rel_stat_error**2 + peak.rel_peakshape_error**2 )
        self.rel_recal_error = peak.rel_recal_error
        print("Relative recalibration error:  "+str(np.round(self.rel_recal_error,9)))
        #try:
        #    peak.rel_mass_error = np.sqrt( (peak.rel_stat_error)**2 + (peak.rel_peakshape_error)**2 + (self.rel_recal_error)**2) # total relative uncertainty of mass value without systematics - includes: stat. mass uncertainty, peakshape uncertainty, calibration uncertainty
        #except TypeError:
        #    print('Could not calculate total fit error.')
        #    pass

        # Set mass_calibrant flag attribute
        self.index_mass_calib = index_mass_calib # mark this peak as mass calibrant to avoid calibrant fit results from being overwritten during regular fits later


    ##### Update peak list with fit values
    def update_peak_props(self,peaks=[],fit_result=None):
        """
        Internal routine to update peak properties DataFrame with fit results in 'fit_result'. All peaks referenced by 'peaks' parameter must belong to the same 'fit_result'. The values of the mass calibrant will not be changed by this routine.

        peaks (list): list of indeces of peaks to update (for peak indeces, see markers in plots or consult peak proeprties list by running 'self.show_peak_properties', where self is your spectrum object)
        fit_model (str): name of fit model used to obtain fit_result (default: fit_model attribute of respective peak)
        cost_func : str, optional, default: 'MLE'
            name of cost function used for minimization
        fit_result (lmfit modelresult): modelresult object holding fit_results of all peaks properties to be updated
                                        Note: Not necessarily all peaks contained in fit_result will be updated, only the properties of peaks referenced by 'peaks' parameter will be updated.
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
                p.m_fit = self.recal_fac*fit_result.best_values[pref+'mu']
                if p.fit_model == 'Gaussian':
                    std_dev = fit_result.best_values[pref+'sigma']
                else:  # for emg models
                    FWHM_emg = spectrum.calc_FWHM_emg(peak_idx,fit_result=fit_result)
                    std_dev = self.A_stat_emg*FWHM_emg  #spectrum.calc_sigma_emg(peak_index=index_mass_calib,fit_model=fit_model,fit_result=fit_result)
                stat_error = std_dev/np.sqrt(p.area)  # stat_error = A_stat * FWHM / sqrt(peak_area), w/ with A_stat_G = 0.42... and  A_stat_emg from `determine_A_stat` method or default value from config.py
                p.rel_stat_error = stat_error/p.m_fit
                if self.rel_recal_error:
                    p.rel_recal_error = self.rel_recal_error
                elif p==peaks[0]: # only print once
                    print('WARNING: Could not set mass recalibration errors. No successful mass recalibration performed on spectrum yet.')
                try:
                    p.rel_mass_error = np.sqrt(p.rel_stat_error**2 + p.rel_peakshape_error**2 + p.rel_recal_error**2) # total relative uncertainty of mass value without systematics - includes: stat. mass uncertainty, peakshape uncertainty, recalibration uncertainty
                    p.mass_error_keV = p.rel_mass_error*p.m_fit*u_to_keV
                except TypeError:
                    if p==peaks[0]:
                        print('Could not calculate total mass error.')
                    pass
                if p.A:
                    p.atomic_ME_keV = np.round((p.m_fit + m_e - p.A)*u_to_keV,3)   # atomic Mass excess (includes electron mass) [keV]
                if p.m_AME:
                    p.m_dev_keV = np.round( (p.m_fit - p.m_AME)*u_to_keV, 3) # TITAN - AME [keV]
                p.chi_sq_red = np.round(fit_result.redchi, 2)


    #### Fit spectrum
    def fit_peaks(self, x_fit_cen=None, x_fit_range=0.01, fit_model=None, cost_func='MLE', init_pars=None, vary_shape=False, vary_baseline=True, method ='least_squares',show_plots=True,show_peak_markers=True,sigmas_of_conf_band=0,plot_filename=None,show_fit_report=True,show_shape_err_fits=False):
        """
        Fit entire spectrum or part of spectrum (if x_fit_cen and x_fit_range are specified), show fit results and show updated peak properties

	    Parameters:
        -----------
        x_fit_cen (float [u], optional): center of mass range to fit (only specify if subset of spectrum is to be fitted)
	    x_fit_range (float [u], optional): width of mass range to fit (default: 0.01, only specify if subset of spectrum is to be fitted)
        fit_model (str, optional): name of fit model to use; if None, defaults to 'best_model' obtained with determine_peak_shape method (default: 'best_model' spectrum attribute, only specify if no best_model has been obtained or to test a specific fit model,
                                   e.g. 'Gaussian','emg12','emg33', ... - see fit_models.py for all available fit models)
        cost_func : str, optional, default: 'MLE'
            name of cost function to use for minimization
            if 'chi-square', the fit is performed using Pearson's chi squared statistic: cost_func = sum( (f(x_i) - y_i)**2/f(x_i)**2 )
            if 'MLE', a binned maximum likelihood estimation is performed using the negative log likelihood ratio: cost_func = sum( f(x_i) - y_i + y_i*log(y_i/f(x_i)) )
	    init_pars (dict, optional): dictionary with initial parameters for fit (default: 'x_pos' from marked peaks in fit_range, shape parameters from 'self.peak_shape_pars' from foregoing peak-shape determination)
        vary_shape (bool, optional): if False, peak-shape parameters (sigma, theta, eta's and tau's) are kept fixed at initial values; if True, they are varied (default: False)
        vary_baseline : bool, optional, default: True
            if True, the constant background will be fitted with a varying baseline paramter bkg_c (initial value: 0.1); otherwise the beseline paremter bkg_c will be fixed to 0.
        method (str, optional): name of minimization algorithm to use (default: least_squares, for full list of options c.f. 'The minimize() function' at https://lmfit.github.io/lmfit-py/fitting.html)

	    Returns:
        --------
        None (updates peak properties dataframe with peak properties obtained in fit)
        """
        if fit_model is None:
            fit_model = self.fit_model
        fit_result = spectrum.peakfit(self, fit_model=fit_model, cost_func=cost_func, x_fit_cen=x_fit_cen, x_fit_range=x_fit_range, init_pars=init_pars, vary_shape=vary_shape, vary_baseline=vary_baseline, method=method,show_plots=show_plots,show_peak_markers=show_peak_markers,sigmas_of_conf_band=sigmas_of_conf_band,plot_filename=plot_filename)
        if x_fit_cen:
            x_min = x_fit_cen - x_fit_range/2
            x_max = x_fit_cen + x_fit_range/2
            peaks_to_fit = [peak for peak in self.peaks if (x_min < peak.x_pos < x_max)] # get peaks in fit range
        else:
            peaks_to_fit = self.peaks

        peak_indeces = [self.peaks.index(p) for p in peaks_to_fit]
        try:
            self.calc_peakshape_errors(peak_indeces=peak_indeces,x_fit_cen=x_fit_cen,x_fit_range=x_fit_range,fit_result=fit_result,vary_baseline=vary_baseline,fit_model=fit_model,cost_func=cost_func,method=method,verbose=True,show_shape_err_fits=show_shape_err_fits)
        except KeyError:
            print("WARNING: Peak-shape error determination failed with KeyError. Likely the used fit_model collides with shape calibration model.")
        self.update_peak_props(peaks=peaks_to_fit,fit_result=fit_result)
        self.show_peak_properties()
        if show_fit_report:
            display(fit_result)
        for p in peaks_to_fit:
            self.fit_results[self.peaks.index(p)] = fit_result


    ##### Save all relevant results to external files
    def save_results(self,filename):
        """
        Write results to an XLSX Excel file with name `filename` and save peak-shape calibration parameters to TXT file with name `filename`+"_peakshape_calib".
        The EXCEL file will contain critical spectrum properties and all peak properties (including mass values) in two separate sheets.

        Parameters:
        -----------
        filename : string
            name of the XLSX-file to be saved to (.xlsx ending does not have to be included by user)

        Returns:
        --------
            None
        """
        # Ensure no files are overwritten
        if os.path.isfile(str(filename)+".xlsx"):
            print ("ERROR: File "+str(filename)+".xlsx already exists. No files saved! Chose a different filename or delete the original file and re-try.")
            return
        if os.path.isfile(str(filename)+"_peakshape_calib.txt"):
            print ("ERROR: File "+str(filename)+"_peakshape_calib.txt already exists. No files saved! Chose a different filename or delete the original file and re-try.")
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
        attributes = ['fit_model','best_redchi','determined_A_stat_emg','A_stat_emg','A_stat_emg_error','recal_fac','rel_recal_error']
        for attr in attributes:
            attr_val = getattr(self,attr)
            spec_data = np.append(spec_data, [[attr,attr_val]],axis=0)
        df_spec = pd.DataFrame(data=spec_data)
        df_spec.set_index(df_spec.columns[0],inplace=True)

        # Make peak properties DataFrame
        dict_peaks = [p.__dict__ for p in self.peaks]
        df_prop = pd.DataFrame(dict_peaks)
        df_prop.index.name = "Peak index"
        frames = []
        keys = []
        for peak_idx in range(len(self.centroid_shifts)):
            df = pd.DataFrame.from_dict(self.centroid_shifts[peak_idx], orient='index')
            df.columns = ['Value']
            frames.append(df)
            keys.append(str(peak_idx))
        df_centroid_shifts = pd.concat(frames, keys=keys)
        df_centroid_shifts.index.names = ['Peak index','Parameter']

        # Save lin. and log. plots of full fitted spectrum to file
        from IPython.utils import io
        with io.capture_output() as captured: # suppress function output to Jupyter notebook
            self.plot_fit(plot_filename=filename)

        # Write DataFrames to separate sheets of EXCEL file and save peak-shape calibration to TXT-file
        with pd.ExcelWriter(filename+'.xlsx') as writer:
            df_spec.to_excel(writer,sheet_name='Spectrum properties')
            df_prop.to_excel(writer,sheet_name='Peak properties')
            prop_sheet = writer.sheets['Peak properties']
            prop_sheet.insert_image(len(df_prop)+2,1, filename+'_log_plot.png',{'x_scale': 0.45,'y_scale':0.45})
            prop_sheet.insert_image(len(df_prop)+26,1, filename+'_lin_plot.png',{'x_scale': 0.45,'y_scale':0.45})
            df_centroid_shifts.to_excel(writer,sheet_name='Centroid shifts')
        print("Fit results saved to file:",str(filename)+".xlsx")

        # Clean up images
        os.remove(filename+'_log_plot.png')
        os.remove(filename+'_lin_plot.png')

        try:
            self.save_peak_shape_cal(filename+"_peakshape_calib")
        except:
            raise



####

###################################################################################################
