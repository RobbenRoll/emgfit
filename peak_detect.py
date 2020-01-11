###################################################################################################
##### Python module for peak detection in TOF mass spectra
##### Code by Stefan Paul, 2019-12-28

##### Import packages 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sig

###################################################################################################
##### Define routine for smoothing spectrum (taken from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html)

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

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


def plot_spec(df,title="",ax=None,yscale='log',vmarkers=None,thres=None,ymin=None):
    """Plots spectrum stored in dataframe 'df'

       - optionally with peak markers if single x_pos or array x_pos is passed to 'vmarkers'"""
    df.plot(figsize=(20,6),ax=ax)
    plt.yscale(yscale)
    plt.title(title)
    try:
        plt.vlines(x=vmarkers,ymin=0,ymax=df.max())
    except TypeError:
        pass
    if thres:
        plt.hlines(y=thres,xmin=df.index.min(),xmax=df.index.max())
    if ymin:
        plt.ylim(ymin,)
        


def peak_detect(df,window_len=23,thres=0.003,width=0.01,plot_smoothed_spec=True,plot_2nd_deriv=True):
    """ Performs automatic peak detection on spectrum 'df' via scaled second derivative of spectrum.
    """
    # Smooth spectrum (moving average with window function)
    df_smooth = df.copy()
    df_smooth['Counts'] = smooth(df['Counts'].values,window_len=window_len,window='blackman')
    # Plot smoothed ad original spectrum 
    ax = df.plot(figsize=(20,5)) 
    df_smooth.plot(figsize=(20,5),ax=ax)
    plt.title("Smoothed spectrum")
    plt.legend("Raw","Smoothed")
    plt.ylim(0.1,)
    plt.yscale('log')
    plt.show()

    # Second derivative 
    df_sec_deriv = df_smooth.iloc[:-2].copy()
    for i in range(len(df_smooth.index) - 2):
        scale = 1/(df_smooth['Counts'].iloc[i+1]+10) # scale data to decrease y range
        #dm = df_smooth.index[i+1]-df_smooth.index[i] # use dm in denominator of deriv if real units are desired
        df_sec_deriv['Counts'].iloc[i] = scale*(df_smooth['Counts'].iloc[i+2] - 2*df_smooth['Counts'].iloc[i+1] + df_smooth['Counts'].iloc[i])/1**2
    # Plot second derivative
    #df_sec_deriv.plot(figsize=(20,5))
    #plt.title("Second derivative of spectrum")
    #plt.hlines(y=-thres,xmin=df_sec_deriv.index.min(),xmax=df_sec_deriv.index.max())
    #plt.show()
    plot_spec(df_sec_deriv,title="Second derivative of spectrum",yscale='linear',thres=-thres)

    # Take only negative part of re-scaled second derivative and invert
    df_sec_deriv_mod = df_smooth.iloc[:-2].copy()
    for i in range(len(df_smooth.index) - 2):
        scale = -1/(df_smooth['Counts'].iloc[i+1]+10) # scale data to decrease y range
        value = scale*(df_smooth['Counts'].iloc[i+2] - 2*df_smooth['Counts'].iloc[i+1] + df_smooth['Counts'].iloc[i])/1**2 # Used (second order forward finite difference)
        if value > 0:
            df_sec_deriv_mod['Counts'].iloc[i] = value 
        else:
            df_sec_deriv_mod['Counts'].iloc[i] = 0

    peak_find = sig.find_peaks(df_sec_deriv_mod['Counts'].values,height=thres,width=width)
    li_peak_pos = df_sec_deriv_mod.index.values[peak_find[0]]
    #peak_widths = sig.peak_widths(df_sec_deriv_mod['Counts'].values,peak_find[0])
    
    #df_sec_deriv_mod.plot(figsize=(20,5))
    #plt.title("Negative part of second derivative, scaled and inverted")
    #plt.hlines(y=thres,xmin=df_sec_deriv_mod.index.min(),xmax=df_sec_deriv_mod.index.max())
    #plt.vlines(x=li_peak_pos,ymin=0,ymax=df_sec_deriv_mod.max())
    #plt.ylim(-0.02,0)
    #plt.yscale('log')
    #plt.show()
    plot_spec(df_sec_deriv_mod,title="Negative part of second derivative, scaled and inverted",thres=thres,vmarkers=li_peak_pos,ymin=1e-04)
    
    # Plot raw spectrum with detected peaks marked
    plot_spec(df,title="Spectrum with detected peaks marked",vmarkers=li_peak_pos)
    return li_peak_pos


def add_peak(li_peak_pos,x_pos,label="?",m_AME=None,m_AME_error=None):
    "Add a peak to be fitted manually"
    li_peak_pos = np.append(li_peak_pos,x_pos)
    li_peak_pos = np.sort(li_peak_pos) # sort peak positions in ascending order
    print("Peak added at ",x_pos," u")
    return li_peak_pos


# Specify identified species
def assign_species():
    return


#### 

###################################################################################################
