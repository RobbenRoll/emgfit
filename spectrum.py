###################################################################################################
##### Python module for peak detection in TOF mass spectra
##### Code by Stefan Paul, 2019-12-28

##### Import packages 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sig
import time
from IPython.display import display
from emgfit.config import *
#u_to_keV = config.u_to_keV
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
        self.stat_error = None # A * Std. Dev. / sqrt(N) with A = 
        self.cal_error = None
        self.peakshape_error = None
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
            print("Calibration uncertainty:",self.cal_error,"u     (",np.round(self.cal_error*u_to_keV,3),"keV )")
            print("Peakshape uncertainty:",self.peakshape_error,"u     (",np.round(self.peakshape_error*u_to_keV,3),"keV )")
            print("Total mass uncertainty:",self.m_fit_error,"u     (",np.round(self.m_fit_error*u_to_keV,3),"keV )")
            print("Mass excess:",np.round(self.ME_keV,3),"keV")
            print("TITAN - AME:",np.round(self.m_dev_keV,3),"keV")
            print("χ_sq_red:",np.round(self.chi_sq_red))


###################################################################################################
##### Define spectrum class 

class spectrum:
    def __init__(self,filename,m_start=None,m_stop=None):
        """ 
        Creates spectrum object by importing TOF data from .txt file, plotting full spectrum and then cutting spectrum to specified fit range {m_start;m_stop}
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
        data_uncut = pd.read_csv(filename, header = None, names= ['Mass [u]', 'Counts'], skiprows = 18,delim_whitespace = True,index_col=False,dtype=float)
        data_uncut.set_index('Mass [u]',inplace =True)
        plt.rcParams.update({"font.size": 15})
        fig  = plt.figure(figsize=(20,8))
        self.peaks = [] # list containing peaks asociated with spectrum (each peak is represented by an instance of the class 'peak')
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
    def plot(self,title="",ax=None,yscale='log',vmarkers=None,thres=None,ymin=None,xmin=None,xmax=None):
        """
        Plots spectrum 
        - with markers for all peaks stored in peak list 'self.peaks'
        """
        peaks = self.peaks
        data = self.data # get spectrum data stored in dataframe 'self.data'
        data.plot(figsize=(20,6),ax=ax)
        plt.yscale(yscale)
        plt.ylabel('Counts')
        plt.title(title)
        try:
            plt.vlines(x=vmarkers,ymin=0,ymax=data.max())
        except TypeError:
            pass
        try:
            li_x_pos = [p.x_pos for p in peaks]
            plt.vlines(x=li_x_pos,ymin=0,ymax=data.max())
        except TypeError:
            pass
        if thres:
            plt.hlines(y=thres,xmin=data.index.min(),xmax=data.index.max())
        if ymin:
            plt.ylim(ymin,)
        plt.xlim(xmin,xmax)
        plt.show()
            

    ##### Define static routine for plotting spectrum data stored in dataframe df (only for use for functions within this class)
    def plot_df(df,title="",ax=None,yscale='log',peaks=None,vmarkers=None,thres=None,ymin=None,xmin=None,xmax=None):
        """Plots spectrum data stored in dataframe 'df'

           - optionally with peak markers if 
        	(a) single x_pos or array x_pos is passed to 'vmarkers'
	    or  (b) list of peak objects is passed to 'li_peaks'
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
        data_sec_deriv = data_smooth.iloc[:-2].copy()
        for i in range(len(data_smooth.index) - 2):
            scale = 1/(data_smooth['Counts'].iloc[i+1]+10) # scale data to decrease y range
            #dm = data_smooth.index[i+1]-data_smooth.index[i] # use dm in denominator of deriv if realistic units are desired
            data_sec_deriv['Counts'].iloc[i] = scale*(data_smooth['Counts'].iloc[i+2] - 2*data_smooth['Counts'].iloc[i+1] + data_smooth['Counts'].iloc[i])/1**2
        spectrum.plot_df(data_sec_deriv,title="Scaled second derivative of spectrum - set threshold indicated",yscale='linear',thres=-thres)

        # Take only negative part of re-scaled second derivative and invert
        data_sec_deriv_mod = data_smooth.iloc[:-2].copy()
        for i in range(len(data_smooth.index) - 2):
            scale = -1/(data_smooth['Counts'].iloc[i+1]+10) # scale data to decrease y range
            value = scale*(data_smooth['Counts'].iloc[i+2] - 2*data_smooth['Counts'].iloc[i+1] + data_smooth['Counts'].iloc[i])/1**2 # Used (second order forward finite difference)
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

        #return self.peaks 


    ##### Add peak manually
    def add_peak(self,x_pos,species="?",m_AME=None,m_AME_error=None):
        """
        Manually add a peak at position 'x_pos' to peak list of spectrum
        - optionally assign 'species' (corresponding literature mass and mass error will then automatically be calculated from corresponding AME values) 
        - optionally assign user-defined m_AME and m_AME_error (this overwrites the values calculated from AME database, use e.g. for isomers) 
        """
        p = peak(x_pos,species,m_AME=m_AME,m_AME_error=m_AME_error) # instantiate new peak
        self.peaks.append(p)
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

    # Specify identified species
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


    ##### Determine peak shape
    def fit_peak_shape(self,index_shape_det_peak,):
        """
        Determine optimal tail order and peak shape parameters by fitting the selected peak-shape calibrant 

        If a left and right tail order is specified by the user, the tail order determination is skipped. 
        The routine tries to find the peak shape that minimizes chi squared reduced.
        """
        return
        
        


#### 

###################################################################################################
