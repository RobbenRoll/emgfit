###################################################################################################
##### Python module for peak detection in TOF mass spectra
##### Code by Stefan Paul, 2019-12-28

##### Import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sig
import time
import copy
from IPython.display import display
from emgfit.config import *
import emgfit.fit_models as fit_models
import lmfit as fit
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
        # Calculate AME values for specified species
        """m = 0.0
        m_error_sq = 0.0
        A_tot = 0
        for ptype in splitspecies(species):
            n, El, A = splitparticle(ptype)
            extrapol = False # initialize boolean flag
            if ptype == '?': # unidentified species
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
                    extrapol = True     """
        m, m_error, extrapol, A_tot = get_AME_values(species)
        if self.m_AME == None: # unless m_AME has been user-defined, the mass value of the specified 'species' is calculated from AME database
             self.m_AME = m
        if self.m_AME_error == None: # unless m_AME_error has been user-defined, the mass error of the specified 'species' is calculated from AME database
            self.m_AME_error = m_error
        self.extrapolated_yn = extrapol
        self.fitted = False
        self.area = None
        self.m_fit = None
        self.stat_error = None # A_stat * Std. Dev. / sqrt(area), with A_stat as defined in config file
        self.peakshape_error = None
        self.cal_error = None
        self.m_fit_error = None # total uncertainty of mass value - includes: stat. mass uncertainty, peakshape uncertainty, calibration uncertainty
        self.A = A_tot
        self.ME_keV = None # Mass excess [keV]
        self.m_dev_keV = None # TITAN -AME [keV]
        self.chi_sq_red = None # chi square reduced of peak fit

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
        print("Extrapolated mass?",self.extrapolated_yn)
        if self.fitted == True:
            print("Peak area:",np.round(self.area,1),"counts")
            print("(Ionic) mass:",self.m_fit,"u     (",np.round(self.m_fit*u_to_keV,3),"keV )")
            print("Stat. mass uncertainty:",self.stat_error,"u     (",np.round(self.stat_error*u_to_keV,3),"keV )")
            print("Peakshape uncertainty:",self.peakshape_error,"u     (",np.round(self.peakshape_error*u_to_keV,3),"keV )")
            print("Calibration uncertainty:",self.cal_error,"u     (",np.round(self.cal_error*u_to_keV,3),"keV )")
            print("Total mass uncertainty (before systematics):",self.m_fit_error,"u     (",np.round(self.m_fit_error*u_to_keV,3),"keV )")
            print("Mass excess:",np.round(self.ME_keV,3),"keV")
            print("TITAN - AME:",np.round(self.m_dev_keV,3),"keV")
            print("χ_sq_red:",np.round(self.chi_sq_red))


###################################################################################################
##### Define spectrum class
class spectrum:
    def __init__(self,filename,m_start=None,m_stop=None,skiprows = 18):
        """
        Creates spectrum object by importing TOF data from .txt or .csv file, plotting full spectrum and then cutting spectrum to specified fit range {m_start;m_stop}
	Input file format: two-column .csv- or .txt-file with comma separated values 	column 1: mass bin,  column 2: counts per bin

        Parameters:
        -----------
	   filename (str): string containing (path and) filename of mass data
	    m_start (float): start of fit range
	    m_stop (float): stop of fit range

	Returns:
        --------
        Pandas dataframe 'data' containing mass data and plot of full spectrum with fit range markers
	"""
        data_uncut = pd.read_csv(filename, header = None, names= ['Mass [u]', 'Counts'], skiprows = skiprows,delim_whitespace = True,index_col=False,dtype=float)
        data_uncut.set_index('Mass [u]',inplace =True)
        self.fit_model = None
        self.shape_cal_pars = None
        self.mass_calibrant = None
        self.recal_fac = 1.0
        self.rel_calib_error = None
        self.rel_peakshape_error = None
        plt.rcParams.update({"font.size": 15})
        fig  = plt.figure(figsize=(20,8))
        self.peaks = [] # list containing peaks associated with spectrum (each peak is represented by an instance of the class 'peak')
        self.fit_results = [] # list containing fit results of peaks associated with spectrum
        if m_start or m_stop:
            self.data = data_uncut.loc[m_start:m_stop] # dataframe containing mass spectreum data (cut to fit range)
            plt.title('Spectrum with fit range markers')
        else:
            self.data = data_uncut # dataframe containing mass spectrum data
            plt.title('Spectrum (fit full range)')
        data_uncut.plot(ax=fig.gca())
        plt.vlines(m_start,0,1.2*max(self.data['Counts']))
        plt.vlines(m_stop,0,1.2*max(self.data['Counts']))
        plt.yscale('log')
        plt.ylabel('Counts')
        plt.show()


    ##### Define static method for smoothing spectrum before peak detection(taken from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html)
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
        if peaks == None:
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
    def detect_peaks(self,window='blackman',window_len=23,thres=0.003,width=0.01,plot_smoothed_spec=True,plot_2nd_deriv=True):
        """
        Performs automatic peak detection on spectrum object using a scaled second derivative of the spectrum.
        """
        # Smooth spectrum (moving average with window function)
        data_smooth = self.data.copy()
        data_smooth['Counts'] = spectrum.smooth(self.data['Counts'].values,window_len=window_len,window=window)
        # Plot smoothed ad original spectrum
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
            data_sec_deriv['Counts'].iloc[i] = scale*(data_smooth['Counts'].iloc[i+1] - 2*data_smooth['Counts'].iloc[i] + data_smooth['Counts'].iloc[i-1])/1**2 # Used (second order central finite difference)
            # data_sec_deriv['Counts'].iloc[i] = scale*(data_smooth['Counts'].iloc[i+2] - 2*data_smooth['Counts'].iloc[i+1] + data_smooth['Counts'].iloc[i])/1**2    # data_sec_deriv = data_smooth.iloc[0:-2].copy()
        spectrum.plot_df(data_sec_deriv,title="Scaled second derivative of spectrum - set threshold indicated",yscale='linear',thres=-thres)

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

        peak_find = sig.find_peaks(data_sec_deriv_mod['Counts'].values,height=thres,width=width)
        li_peak_pos = data_sec_deriv_mod.index.values[peak_find[0]]
        #peak_widths = sig.peak_widths(data_sec_deriv_mod['Counts'].values,peak_find[0])
        spectrum.plot_df(data_sec_deriv_mod,title="Negative part of scaled second derivative, inverted - set threshold indicated",thres=thres,vmarkers=li_peak_pos,ymin=0.1*thres)

        # Plot raw spectrum with detected peaks marked
        spectrum.plot_df(self.data,title="Spectrum with detected peaks marked",vmarkers=li_peak_pos)

        # Create list of peak objects
        for x in li_peak_pos:
            p = peak(x,'?') # instantiate new peak
            self.peaks.append(p)
            self.fit_results.append(None)

        #return self.peaks


    ##### Add peak manually
    def add_peak(self,x_pos,species="?",m_AME=None,m_AME_error=None):
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
        print("Added peak at ",x_pos," u")


    ##### Remove peak manually
    def remove_peak(self,peak_index=None,x_pos=None,species="?"):
        """
        Remove a peak manually from peak list

        select peak by specifying species label, peak position 'x_pos' or peak index (0-based! Check for peak index by calling .peak_properties() method)
        """
        if peak_index:
            i = peak_index
        elif species != "?":
            p = [peak for peak in self.peaks if species == peak.species] # select peak with species label 'species'
            i = peaks.index(p)
        elif x_pos:
            p = [peak for peak in self.peaks if x_pos == peak.x_pos] # select peak at position 'x_pos'
            i = self.peaks.index(p)
        try:
            rem_peak = self.peaks.pop(i)
            self.fit_results.pop(i)
            print("Removed peak at ",rem_peak.x_pos," u")
        except:
            print("Peak removal failed!")
            raise


    ##### Print peak properties
    def peak_properties(self):
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
            select peak by specifying peak position 'x_pos' or peak index argument (0-based! Check for peak index by calling .peak_properties() method of spectrum object)     specify species name by assigning string to species object

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
            if peak_index != None:
                p = self.peaks[peak_index]
                p.species = species
                p.update_lit_values() # overwrite m_AME, m_AME_error and extrapolated_yn attributes with AME values for specified species
                print("Species of peak",peak_index,"assigned as",p.species)
            elif x_pos:
                p = [peak for peak in self.peaks if x_pos == peak.x_pos] # select peak at position 'x_pos'
                i = self.peaks.index(p)
                p.species = species
                p.update_lit_values() # overwrite m_AME, m_AME_error and extrapolated_yn attributes with AME values for specified species
                print("Species of peak",i,"assigned as",p.species)
            elif len(species) == len(self.peaks) and peak_index == None and x_pos == None: # assignment of multiple species
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

    ##### Internal helper function for creating multi-peak fit models
    def comp_model(self,peaks_to_fit=None,model='emg22',init_pars=None,vary_shape=False,index_first_peak=None):
        """ create multi-peak composite model from single-peak model """
        model = getattr(fit_models,model) # get single peak model from fit_models.py
        mod = fit.models.ConstantModel(independent_vars='x',prefix='bkg_')
        mod.set_param_hint('bkg_c', value= 0.3, min=0,max=4)
        df = self.data
        for peak in peaks_to_fit:
            peak_index = self.peaks.index(peak)
            x_pos = df.index[np.argmin(np.abs(df.index.values - peak.x_pos))] # x_pos of closest bin
            amp = df['Counts'].loc[x_pos]/1200 # estimate amplitude from peak maximum, the factor 1200 is empirically determined and shape-dependent
            if init_pars:
                this_mod = model(peak_index, peak.x_pos, amp, init_pars=init_pars, vary_shape_pars=vary_shape, index_first_peak=index_first_peak)
            else:
                this_mod = model(peak_index, peak.x_pos, amp, vary_shape_pars=vary_shape, index_first_peak=index_first_peak)
            mod = mod + this_mod
        return mod


    # Add peak markers to plot of a fit
    def add_peak_markers(self,yscale='log',ymax=None,peaks=None):
        """
        (Internal) method for adding peak markers to current figure object, place this function as self.add_peak_markers between plt.figure() and plt.show(), only for use on already fitted spectra
        """
        if peaks == None:
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


    # Plot fit of full spectrum
    def plot_fit(self,fit_result=None,fit_model=None,ax=None,show_peak_markers=True,sigmas_of_uncer_band=0,thres=None,x_min=None,x_max=None):
        """
        Plots spectrum with fit
        - with markers for all peaks stored in peak list 'self.peaks'
        """
        if self.fit_model == None:
           fit_model = self.fit_model
        peaks_to_plot = [peak for peak in self.peaks if (x_min < peak.x_pos < x_max)] # select peaks in mass range of interest
        idx_first_peak = self.peaks.index(peaks_to_plot[0])
        if fit_result == None:
           fit_result = self.fit_results[idx_first_peak]
        i_min = np.argmin(np.abs(fit_result.x - x_min))
        i_max = np.argmin(np.abs(fit_result.x - x_max))
        y_max_log = max( max(self.data.values[i_min:i_max]), max(fit_result.best_fit[i_min:i_max]) )
        y_max_lin = max( max(self.data.values[i_min:i_max]), max(fit_result.init_fit[i_min:i_max]), max(fit_result.best_fit[i_min:i_max]) )

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
        if sigmas_of_uncer_band!=0 and fit_result.errorbars == True:
            dely = fit_result.eval_uncertainty(sigma=sigmas_of_uncer_band)
            plt.fill_between(fit_result.x, fit_result.best_fit-dely, fit_result.best_fit+dely, color="#ABABAB", label=str(sigmas_of_uncer_band)+'-$\sigma$ uncertainty band')
        plt.title(fit_model)
        plt.rcParams.update({"font.size": 15})
        plt.xlabel('Mass [u]')
        plt.ylabel('Counts per bin')
        plt.yscale('log')
        plt.ylim(0.1, 2*y_max_log)
        plt.xlim(x_min,x_max)
        plt.show()

        # Plot residuals and fit result with linear y-scale
        f2 = plt.figure(figsize=(20,12))
        fit_result.plot(fig=f2,show_init=True)
        if show_peak_markers:
            self.add_peak_markers(yscale='lin',ymax=y_max_lin,peaks=peaks_to_plot)
        ax_res, ax_fit = f2.axes
        ax_res.set_title(fit_model)
        plt.xlabel('Mass [u]')
        plt.ylabel('Counts per bin')
        plt.xlim(x_min,x_max)
        plt.ylim(-0.05*y_max_lin, 1.1*y_max_lin)
        plt.show()


    ##### Fit spectrum
    def peakfit(self,fit_model='emg22',x_fit_cen=None,x_fit_range=None,init_pars=None,vary_shape=False,method='least_squares',show_plots=True,show_peak_markers=True,sigmas_of_uncer_band=0,recal_fac=1.0):
        """
        Internal peak fitting routine, fits full spectrum or subrange (if x_fit_cen and x_fit_range are specified) and optionally shows results
        This method is for internal usage, use 'fit_peaks' method to fit spectrum and update peak properties dataframe with obtained fit results

	    Parameters:
        -----------
        fit_model (str): name of fit model to use (e.g. 'Gaussian','emg12','emg33', ... - see fit_models.py for all available fit models)
	    x_fit_cen (float): center of mass range to fit (optional, only specify if subset of spectrum is to be fitted)
	    x_fit_range (float): width of mass range to fit (optional, only specify if subset of spectrum is to be fitted)
	    init_pars (dict): dictionary with initial parameters for fit (optional), if set to 'default' the default parameters from 'fit_models.py'
                              are used, if set to 'None' the parameters from the shape calibration are used (if those do not exist yet
                              the default parameters are used)
        vary_shape (bool): if 'False' peak shape parameters (sigma, eta's, tau's and theta) are kept fixed at initial values,
                           if 'True' the shape parameters are varied but shared amongst all peaks (identical shape parameters for all peaks)
        method (str): fitting method to be used (default: 'least_squares')
        recal_fac (float): factor for correction of the final mass values (obtain recalibration factor from calibrant fit before fitting other peaks)

	    Returns:
        --------
        Fit model result object
        """
        if x_fit_cen and x_fit_range:
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
        y = df_fit['Counts'].values
        y_err = np.sqrt(y+1) # assume Poisson (counting) statistics -> standard deviation of each point approximated by sqrt(counts+1)
        weight_facs = 1./y_err # makes sure that residuals include division by statistical error (residual = (fit_model - y) * weights)

        if init_pars == 'default':
            init_params = None
        elif init_pars != None:
            init_params = init_pars
        else:
            init_params = self.shape_cal_pars # use shape parameters asociated with spectrum, unless other parameters are specified

        if vary_shape == True:
            index_first_peak = self.peaks.index(peaks_to_fit[0]) # enforce shared shape parameters for all peaks
        else:
            index_first_peak = None

        #if self.recal_fac != 1.0 and recal_fac == 1.0:
        #    recal_fac = self.recal_fac # use calib. factor attribute of spectrum unless different cal. factor is specified in peakfit function

        model_name = str(fit_model)+' + const. background (c_bgd)'
        mod = self.comp_model(peaks_to_fit=peaks_to_fit,model=fit_model,init_pars=init_params,vary_shape=vary_shape,index_first_peak=index_first_peak) # create multi-peak fit model
        pars = mod.make_params()

        # Perform fit, print fit report
        out = mod.fit(y, x=x, params=pars, weights=weight_facs, method=method, scale_covar=False)
        out.x = x
        out.y = y
        out.y_err = y_err
        #print(out.fit_report())

        if show_plots:
            self.plot_fit(fit_result=out, fit_model=fit_model, show_peak_markers=show_peak_markers, sigmas_of_uncer_band=sigmas_of_uncer_band, x_min=x_min, x_max=x_max)

        return out


    ##### Internal helper function to calculate peak area (counts in peak)
    def get_peak_area(self, peak_index, fit_result=None, decimals=2):
        pref = 'p'+str(peak_index)+'_'
        area = 0
        for y_i in fit_result.eval_components(x=fit_result.x)[pref]: # Get counts in subpeaks from best fit to data
            area += y_i
            if np.isnan(y_i) == True:
                print("Warning: Encountered NaN values in "+str(self.peaks[peak_index].species)+"-subpeak! Those are omitted in area summation.")
        return np.round(area,decimals)


    ##### Determine peak shape
    def determine_peak_shape(self, index_shape_calib=None, species_shape_calib=None, fit_model='emg22', init_pars = 'default', fit_range=0.01, method='least_squares',vary_tail_order=True,show_plots=True,show_peak_markers=True,sigmas_of_uncer_band=0):
        """
        Determine optimal tail order and peak shape parameters by fitting the selected peak-shape calibrant

        If a left and right tail order is specified by the user, the tail order determination is skipped.
        The routine tries to find the peak shape that minimizes chi squared reduced.


            fit_model (str): name of fit model to use (e.g. 'Gaussian','emg12','emg33', ... - see fit_models.py for all available fit models)
        """
        if index_shape_calib != None and (species_shape_calib == None):
            peak = self.peaks[index_shape_calib]
        elif species_shape_calib:
            peak = [p for p in self.peaks if species_shape_calib == p.species]
            index_shape_calib = self.peaks.index(peak)
        else:
            print("Definition of peak shape calibrant failed. Define EITHER the index OR the species name of the peak to use as shape calibrant!")
            return

        if vary_tail_order == True and fit_model != 'Gaussian':
            print('\n##### Determine optimal tail order #####\n')
            # Fit peak with Hyper-EMG of increasingly higher tail orders and compile results
            # fix fit_model to Hyper-EMG with lowest tail order that yields chi² reduced <= 1
            best_model = None
            best_redchi = 2
            li_fit_models = ['Gaussian','emg01','emg10','emg11','emg12','emg21','emg22','emg23','emg32','emg33']
            for model in li_fit_models:
                try:
                    out = spectrum.peakfit(self, fit_model=model, x_fit_cen=peak.x_pos, x_fit_range=fit_range, init_pars=init_pars ,vary_shape=True, method=method,show_plots=show_plots,show_peak_markers=show_peak_markers,sigmas_of_uncer_band=sigmas_of_uncer_band)
                    print(out.fit_report())
                    if out.redchi <= 1:
                       best_model = model
                       best_redchi = out.redchi
                       break
                    elif out.redchi < best_redchi:
                       best_redchi = out.redchi
                       best_model = model
                except ValueError:
                    print('\nWARNING:',model+'-fit failed due to NaN-values and was skipped! -----------------------------------------------------------\n')
            if best_model:
                self.fit_model = best_model
                print('\nBest fit model determined to be:',best_model)
                print('Corresponding chi²-reduced:',best_redchi)
            else:
                self.fit_model = fit_model
                print('No fit model found that produces chi²-reduced < 2. Continuing with specified or default fit_model.')
        elif vary_tail_order == False:
            self.fit_model = fit_model

        print('\n##### Peak shape determination #####\n')
        out = spectrum.peakfit(self, fit_model=self.fit_model, x_fit_cen=peak.x_pos, x_fit_range=fit_range, init_pars=init_pars ,vary_shape=True, method=method,show_plots=show_plots,show_peak_markers=show_peak_markers,sigmas_of_uncer_band=sigmas_of_uncer_band)

        peak.comment = 'shape calibrant'
        print(out.fit_report())
        dict_pars = out.params.valuesdict()
        self.shape_cal_pars = {key.lstrip('p'+str(index_shape_calib)+'_'): val for key, val in dict_pars.items() if key.startswith('p'+str(index_shape_calib))}
        self.shape_cal_par_errors = {} # dict to store shape calibration parameter errors
        for par in out.params:
            if par.startswith('p'+str(index_shape_calib)):
                self.shape_cal_par_errors[par.lstrip('p'+str(index_shape_calib)+'_')+' error'] = out.params[par].stderr

        print('\n##### Evaluate peak shape uncertainty #####\n')
        # Vary each shape parameter by plus and minus one sigma and sum resulting shifts of Gaussian centroid in quadrature to obtain rel. peakshape error
        shape_pars = [key for key in self.shape_cal_pars if key.startswith( ('sigma','theta','eta','tau') )]
        self.centroid_shifts = {}
        for par in shape_pars:
            pars = copy.deepcopy(self.shape_cal_pars) # deep copy to avoid changes in original dictionary
            centroid = list(out.best_values.values())[1] # indexing makes sure that both Gaussian 'center' and Hyper-EMG 'mu' Parameters get fetched
            pars[par] = self.shape_cal_pars[par] + self.shape_cal_par_errors[par+' error']
            out_p = spectrum.peakfit(self, fit_model=self.fit_model, x_fit_cen=peak.x_pos, x_fit_range=fit_range, init_pars=pars, vary_shape=False, method=method, show_plots=False)
            new_centroid = list(out_p.best_values.values())[1] # indexing makes sure that both Gaussian 'center' and Hyper-EMG 'mu' Parameters get fetched
            delta_mu_p = new_centroid - centroid
            print('Re-fitting with ',par,' = ',np.round(self.shape_cal_pars[par],6),'+',np.round(self.shape_cal_par_errors[par+' error'],6),' shifts centroid by ',np.round(delta_mu_p*1e06,6),'\u03BCu.')
            pars[par] = self.shape_cal_pars[par] - self.shape_cal_par_errors[par+' error']
            out_m = spectrum.peakfit(self, fit_model=self.fit_model, x_fit_cen=peak.x_pos, x_fit_range=fit_range, init_pars=pars, vary_shape=False, method=method, show_plots=False)
            new_centroid = list(out_m.best_values.values())[1] # indexing makes sure that both Gaussian 'center' and Hyper-EMG 'mu' Parameters get fetched
            delta_mu_m = new_centroid - centroid
            print('Re-fitting with ',par,' = ',np.round(self.shape_cal_pars[par],6),'-',np.round(self.shape_cal_par_errors[par+' error'],6),' shifts centroid by ',np.round(delta_mu_m*1e06,3),'\u03BCu.')
            self.centroid_shifts[par+' centroid shift'] = max([delta_mu_p,delta_mu_m])
        shape_error = np.sqrt(np.sum(np.square(list(self.centroid_shifts.values())))) # add centroid shifts in quadrature to obtain total peakshape error
        self.rel_peakshape_error = shape_error/centroid
        rel_error_rounded = np.round(self.rel_peakshape_error,9)
        if rel_error_rounded == 0:
            print('\nRelative peakshape error: < 1e-09')
        else:
            print('\nRelative peakshape error: ',rel_error_rounded)

        return out


    ##### Fit mass calibrant
    def fit_calibrant(self, index_mass_calib=None, species_mass_calib=None, fit_model=None, fit_range=0.01, method='least_squares',show_plots=True,show_peak_markers=True,sigmas_of_uncer_band=0):
        """
        Determine scale factor for spectrum by fitting the selected mass calibrant

            fit_model (str): name of fit model to use (e.g. 'Gaussian','emg12','emg33', ... - see fit_models.py for all available fit models)
        """
        peak = self.peaks[index_mass_calib]
        self.mass_calibrant = peak # mark this peak as mass calibrant to avoid calibrant fit results from being overwritten after fit of full spectrum
        if peak.comment == 'shape calibrant':
            peak.comment = 'shape & mass calibrant'
        else:
            peak.comment = 'mass calibrant'
        if index_mass_calib != None and (species_mass_calib == None):
            peak = self.peaks[index_mass_calib]
        elif species_mass_calib:
            peak = [p for p in self.peaks if species_mass_calib == p.species]
            index_mass_calib = self.peaks.index(peak)
        else:
            print("Definition of peak shape calibrant failed. Define EITHER the index OR the species name of the peak to use as shape calibrant!")
            return
        print('##### Calibrant fit #####')
        if fit_model == None:
            fit_model = self.fit_model
        out = spectrum.peakfit(self, fit_model=fit_model, x_fit_cen=peak.x_pos, x_fit_range=fit_range, vary_shape=False, method=method, show_plots=show_plots, show_peak_markers=show_peak_markers, sigmas_of_uncer_band=sigmas_of_uncer_band)

        # Update peak properties
        peak.fitted = out.success
        peak.area = self.get_peak_area(index_mass_calib,fit_result=out)
        peak.m_fit = out.best_values['p'+str(index_mass_calib)+'_mu']
        peak.stat_error = A_stat*out.best_values['p'+str(index_mass_calib)+'_sigma']/np.sqrt(peak.area)  # A_stat * Std. Dev. / sqrt(area)
        peak.peakshape_error = 0 ################################# FIX
        peak.chi_sq_red = np.round(out.redchi, 2)

        # Determine calibration factor
        self.recal_fac = peak.m_AME/peak.m_fit
        print("Recalibration factor: "+str(self.recal_fac))

        # Update peak properties with new calibrant centroid
        peak.m_fit = self.recal_fac*out.best_values['p'+str(index_mass_calib)+'_mu'] # update centroid mass of calibrant peak
        if peak.A:
            peak.ME_keV = (peak.A*u - peak.m_fit)*u_to_keV   # Mass excess [keV]
        if peak.m_AME:
            peak.m_dev_keV = np.round( (peak.m_fit - peak.m_AME)*u_to_keV, 3) # TITAN - AME [keV]

        # Determine rel. calibrant error and update calibration error
        self.rel_calib_error = np.sqrt( (peak.m_AME_error/peak.m_AME)**2 + (peak.stat_error/peak.m_fit)**2 + (peak.peakshape_error/peak.m_fit)**2 )
        print("Relative calibration error: "+str(np.round(self.rel_calib_error,9)))
        peak.cal_error = peak.m_fit * self.rel_calib_error


    ##### Update peak list with fit values
    def update_peak_props(self,peaks=[],fit_result=None):
        """
        Save fit results in list with peak properties.
        """
        for p in peaks:
            if p == self.mass_calibrant:
                pass
            else:
                p.fitted = fit_result.success
                peak_idx = self.peaks.index(p)
                p.area = self.get_peak_area(peak_idx,fit_result=fit_result)  #np.round(area,2)
                p.m_fit = self.recal_fac*fit_result.best_values['p'+str(peak_idx)+'_mu']
                p.stat_error = A_stat*fit_result.best_values['p'+str(peak_idx)+'_sigma']/np.sqrt(p.area)  # A_stat*Std. Dev./sqrt(area), w/ A_stat from config
                if self.rel_peakshape_error:
                    p.peakshape_error = p.m_fit * self.rel_peakshape_error
                elif p==peaks[0]:
                    print('WARNING: Could not calculate peak shape errors. No successful peak shape calibration performed on spectrum yet.')
                if self.rel_calib_error:
                    p.cal_error = p.m_fit * self.rel_calib_error
                elif p==peaks[0]: # only print once
                    print('WARNING: Could not calculate mass calibration errors. No successful mass calibration performed on spectrum yet.')
                try:
                    p.m_fit_error = np.sqrt(p.stat_error**2 + p.peakshape_error**2 + p.cal_error**2) # total uncertainty of mass value - includes: stat. mass uncertainty, peakshape uncertainty, calibration uncertainty
                except TypeError:
                    if p==peaks[0]:
                        print('Could not calculate total fit error.')
                    pass
                if p.A:
                    p.ME_keV = (p.A*u - p.m_fit)*u_to_keV   # Mass excess [keV]
                if p.m_AME:
                    p.m_dev_keV = np.round( (p.m_fit - p.m_AME)*u_to_keV, 3) # TITAN - AME [keV]
                p.chi_sq_red = np.round(fit_result.redchi, 2)


    #### Fit spectrum
    def fit_peaks(self, fit_model=None, x_fit_cen=None, x_fit_range=None, init_pars=None, vary_shape=False, method ='least_squares',show_plots=True,show_peak_markers=True,sigmas_of_uncer_band=0):
        """
        Fit entire spectrum or part of spectrum (if x_fit_cen and x_fit_range are specified), show results and show updated peak properties

	    Parameters:
        -----------
        fit_model (str): name of fit model to use (e.g. 'Gaussian','emg12','emg33', ... - see fit_models.py for all available fit models)
	    x_fit_cen (float): center of mass range to fit (optional, only specify if subset of spectrum is to be fitted)
	    x_fit_range (float): width of mass range to fit (optional, only specify if subset of spectrum is to be fitted)
	    init_pars (dict): dictionary with initial parameters for fit (optional)
        vary_shape (bool): if 'False' peak shape paramters (sigma, eta's, tau's and theta) are kept fixed at initial values, if 'True' they are varied
        method (str): fitting algorithm to use (full list under 'The minimize() function' at https://lmfit.github.io/lmfit-py/fitting.html)

	    Returns:
        --------
        Fit model result object (further shows updated dataframe with peak properties)
        """
        if fit_model == None:
            fit_model = self.fit_model
        out = spectrum.peakfit(self, fit_model=fit_model, x_fit_cen=x_fit_cen, x_fit_range=x_fit_range, init_pars=init_pars, vary_shape=vary_shape, method=method,show_plots=show_plots,show_peak_markers=show_peak_markers,sigmas_of_uncer_band=sigmas_of_uncer_band)
        if x_fit_cen and x_fit_range:
            x_min = x_fit_cen - x_fit_range/2
            x_max = x_fit_cen + x_fit_range/2
            peaks_to_fit = [peak for peak in self.peaks if (x_min < peak.x_pos < x_max)] # get peaks in fit range
        else:
            peaks_to_fit = self.peaks
        spectrum.update_peak_props(self,peaks=peaks_to_fit,fit_result=out)
        spectrum.peak_properties(self)
        display(out)
        for p in peaks_to_fit:
            self.fit_results[self.peaks.index(p)] = out


    # Plot fit of spectrum zoomed to specified peak or specified mass range
    def plot_fit_zoom(self,peak_indeces=None,x_center=None,x_range=0.01,show_peak_markers=True,sigmas_of_uncer_band=0,ax=None):
        """
        Plot fit result zoomed to a region of interest
        - if peak(s) of interest are specified via 'peak_indeces' the mass range to plot is chosen automatically
        - otherwise the mass range must be specified manually with x_center and x_range

        Parameters: ----------- peak_indeces (int or list of ints): index of
        single peak or of multiple neighboring peaks to show (peaks must belong
        to the same fit curve!) x_center (float): center of manually specified
        mass range to plot x_range (float): width of mass range to plot around
        'x_center' or minimal width to plot around specified peaks of interest
        """
        if isinstance(peak_indeces,list):
            x_min = self.peaks[peak_indeces[0]].x_pos - x_range/2
            x_max = self.peaks[peak_indeces[-1]].x_pos + x_range/2
        elif type(peak_indeces) == int:
            peak = self.peaks[peak_indeces]
            x_min = peak.x_pos - x_range/2
            x_max = peak.x_pos + x_range/2
        elif x_center != None:
            x_min = x_center - x_range/2
            x_max = x_center + x_range/2
        else:
            print("Mass range to plot could not be determined. Check documentation on function parameters")
            return
        self.plot_fit(ax=ax,x_min=x_min,x_max=x_max,show_peak_markers=show_peak_markers,sigmas_of_uncer_band=sigmas_of_uncer_band)


####

###################################################################################################
