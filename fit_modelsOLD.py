###################################################################################################
##### Python module for Hyper-EMG fitting of TOF mass spectra
##### Fitting routines taken from lmfit package
##### Code by Stefan Paul, 2019-12-28


###################################################################################################
##### Define Hyper-EMG(2,2) single peak fitting routine
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lmfit as fit
import emgfit
from emgfit.emg_funcs import * 
u_to_keV = emgfit.u_to_keV
m_e = emgfit.m_e


# Define fit function
def h_emg_m2_p2(x, mu, sigma, theta, eta_m1,eta_m2,tau_m1,tau_m2,eta_p1,eta_p2,tau_p1,tau_p2,amp): 
    return amp*h_emg(x, mu, sigma, theta, (eta_m1,eta_m2),(tau_m1,tau_m2),(eta_p1,eta_p2),(tau_p1,tau_p2))

# Define function to perform hyper-EMG(2,2) fit
def peak_fit_emg_m2_p2(df_to_fit=None,x_cen=None,x_fit_range=None,amp=0.01,init_pars=None,vary_shape_pars=False,scl_fac=1,overlapping_peaks=False,m_AME= None, m_AME_error= None):
    """
    df_to_fit: dataframe with data to fit (index column: mass bin [u], data column: counts in bin)  
    x_cen: centre of mass window to fit [u]
    m_AME: literature value for IONIC mass of species to fit (m_e already substracted) [u]
    m_AME_error: literature error on IONIC mass of species to fit [u]
    init_pars: dictionary containing initial fit parameters
    vary_shape_pars: if False, fix peak shape parameters to init_pars (x_cen always remains free!)
    """
    x_min = x_cen - x_fit_range/2
    x_max = x_cen + x_fit_range/2
    df_fit = df_to_fit[x_min:x_max] # fix mass range to fit
    x = df_fit.index.values
    y = df_fit['Counts'].values
    y_err = np.sqrt(y+1) # assume Poisson (counting) statistics -> standard deviation of each point approximated by sqrt(counts+1)
    weight_facs = 1./y_err # makes sure that residuals include division by statistical error (residual = (fit_model - y) * weights)
    #print(weight_facs)
    
    # Exponentially modified Gaussian (EMG) fit
    emg_mod = fit.Model(h_emg_m2_p2,nan_policy='propagate') # define fit model
    pars = fit.Parameters() # create dictionary with fit parameters 
    # Add parameters to dictionary and initialize them 
    pars.add('amp', value=amp, min=0) 
    pars.add('mu', value=x_cen, min=x_cen-1, max=x_cen+1)
    pars.add('sigma', value= init_pars['sigma'], min=init_pars['sigma']-0.01, max=init_pars['sigma']+0.01, vary=vary_shape_pars)
    pars.add('theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    pars.add('eta_m1', value= init_pars['eta_m1'], min=0, max=1, vary=vary_shape_pars) 
    pars.add('eta_m2', value= init_pars['eta_m2'], min=0, max=1, expr='1-eta_m1') # ensures normalization of eta_m's
    pars.add('tau_m1', value= init_pars['tau_m1'], min=0, vary=vary_shape_pars)
    pars.add('tau_m2', value= init_pars['tau_m2'], min=0, vary=vary_shape_pars)
    pars.add('eta_p1', value= init_pars['eta_p1'], min=0, max=1, vary=vary_shape_pars)
    pars.add('eta_p2', value= init_pars['eta_p2'], min=0, expr='1-eta_p1') # ensures normalization of eta_p's
    pars.add('tau_p1', value= init_pars['tau_p1'], min=0, vary=vary_shape_pars)
    pars.add('tau_p2', value= init_pars['tau_p2'], min=0, vary=vary_shape_pars)
    out = emg_mod.fit(y, x=x, params=pars, weights= weight_facs)
    
    print(out.fit_report(min_correl=0.25))
    #print(out.best_fit)
    f = plt.figure(figsize=(14,8))
    out.plot(fig=f,show_init=True)
    plt.yscale('log')
    plt.show()

    # Sum up data to get counts in peak
    area = 0 
    for y_i in y:
        area += y_i
    # Get counts in peak from best fit to data, use this value in case of overlapping peaks
    area2 = 0
    for y_i in out.best_fit:
        area2 += y_i
    # Store and print resulting mass value after scaling
    out.m_fit_u = scl_fac*mu_emg(out.best_values['mu'],out.best_values['theta'],(out.best_values['eta_m1'],out.best_values['eta_m2']),(out.best_values['tau_m1'],out.best_values['tau_m2']),(out.best_values['eta_p1'],out.best_values['eta_p2']),(out.best_values['tau_p1'],out.best_values['tau_p2'])) # Gaussian use: out.best_values['mu']
    out.sigma_fit_u = scl_fac*sigma_emg(out.best_values['sigma'],out.best_values['theta'],(out.best_values['eta_m1'],out.best_values['eta_m2']),(out.best_values['tau_m1'],out.best_values['tau_m2']),(out.best_values['eta_p1'],out.best_values['eta_p2']),(out.best_values['tau_p1'],out.best_values['tau_p2'])) # for Gaussian use: out.best_values['sigma']    
    if overlapping_peaks == False:
        out.m_error_fit_u = out.sigma_fit_u/np.sqrt(area) # statistical mass error 
    elif overlapping_peaks == True:
        out.m_error_fit_u = out.sigma_fit_u/np.sqrt(area2) # statistical mass error  
    print("Measured counts in peak:  "+str(area))
    print("Counts in peak of fitted curve:  "+str(area2))
    print("Mass in u:  "+str(out.m_fit_u)+" +/- "+str( out.m_error_fit_u)+" u")
    print("Mass:  "+str(out.m_fit_u*u_to_keV)+" +/- "+str( out.m_error_fit_u*u_to_keV)+" keV")
    try: # handle missing uncertainty estimates
        print('Fit error on centroid [keV]: '+str(out.params['mu'].stderr*u_to_keV)+" keV")
    except TypeError:
        print('Fit error on centroid [keV]: Uncertainty estimation failed.')
    # Print literature mass (AME2016) after correction with electron mass
    if m_AME != None:
        m_lit = m_AME - m_e
        print("AME2016 mass - m_e:  "+str(m_lit*u_to_keV)+" +/- "+str(m_AME_error*u_to_keV)+" keV")
        print("TITAN - AME2016 mass:  "+str((out.m_fit_u-m_lit)*u_to_keV)+" keV")
    peak_fit_emg_m2_p2.out = out # store fit results in global variable


###################################################################################################
##### Define Hyper-EMG(2,2) multi-peak fitting routine
from IPython.display import display

def multi_peak_fit_emg_m2_p2(df_to_fit=None,x_fit_cen=None,x_fit_range=None,peak_labels=[],peak_pos=[],peak_amps=[], init_pars=None,vary_shape_pars=False,scl_fac=1,li_m_AME=[],li_m_AME_error=[]):
    x_min = x_fit_cen - x_fit_range/2
    x_max = x_fit_cen + x_fit_range/2
    df_fit = df_to_fit[x_min:x_max] # fix mass range to fit
    x = df_fit.index.values
    y = df_fit['Counts'].values
    y_err = np.sqrt(y+1) # assume Poisson (counting) statistics -> standard deviation of each point approximated by sqrt(counts+1)
    weight_facs = 1./y_err # makes sure that residuals include division by statistical error (residual = (fit_model - y) * weights)

    """# Plot data over fit window with markers for initial peak centroids 
    plt.figure(figsize=(20,5))
    plt.plot(x,y,'o-')
    plt.xlabel('Mass [u]')
    plt.ylabel('Counts per bin')
    plt.yscale('log')
    plt.xlim(x_min,x_max)
    for i in range(len(peak_pos)):
        x_pos = peak_pos[i]
        y_text = 1.4*max(df_fit.loc[x_pos-7e-04:x_pos+7e-04].values) # 1.2*max(df_fit.values)
        plt.vlines(x_pos,label='x',ymin=0,ymax=y_text)
        plt.text(s="%f" %x_pos,x=x_pos, y=y_text,fontsize=10) # vline label
    plt.show()
    """

    def make_model(peak_i):
        pref = 'emg{0}_'.format(peak_i) # determine prefix for respective peak
        mu_i = peak_pos[peak_i] # read in x position of respective peak
        amp_i = peak_amps[peak_i] # read in amplitude of respective peak
        model = fit.Model(h_emg_m2_p2, prefix = pref, nan_policy='propagate')
        
        # Add respective parameters for peak to dictionary and define starting values 
        model.set_param_hint(pref+'amp', value=amp_i, min=0) 
        model.set_param_hint(pref+'mu', value=mu_i, min=mu_i-1, max=mu_i+1)
        model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=init_pars['sigma']-0.005, max=init_pars['sigma']+0.005, vary=vary_shape_pars)
        model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
        model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, vary=vary_shape_pars) 
        model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1',vary=vary_shape_pars) # ensures normalization of eta_m's
        model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=0, vary=vary_shape_pars)
        model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=0, vary=vary_shape_pars)
        model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, vary=vary_shape_pars)
        model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, expr='1-'+pref+'eta_p1',vary=vary_shape_pars) # ensures normalization of eta_p's
        model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=0, vary=vary_shape_pars)
        model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=0, vary=vary_shape_pars)
        pars = model.make_params()
        return model
    
    # create multi-peak composite model
    mod = None
    for i in range(len(peak_pos)):
        this_mod = make_model(i)
        if mod is None:
            mod = this_mod
        else:
            mod = mod + this_mod

    # Perform fit, print fit report and plot resulting fit 
    out = mod.fit(y, x=x, weights = weight_facs,method='leastsq')
    comps = out.eval_components(x=x)
    print(out.fit_report())
    # Plot fit result with logarithmic and linear y-scale
    f1 = plt.figure(figsize=(14,12))
    plt.errorbar(x,y,yerr=y_err,fmt='g.',linewidth=0.5)
    plt.plot(x, out.best_fit, 'r-',linewidth=2)
    for peak_i in range(len(peak_pos)):
        pref = 'emg{0}_'.format(peak_i)
        plt.plot(x, comps[pref], '--')
    plt.xlabel('Mass [u]')
    plt.ylabel('Counts per bin')
    plt.yscale('log')
    plt.ylim(0.1,max(out.best_fit)*1.1)
    plt.show()

    f2 = plt.figure(figsize=(16,12))
    out.plot(fig=f2,show_init=True)
    plt.title("")
    plt.xlabel('Mass [u]')
    plt.ylabel('Counts per bin')
    plt.show()

    # Sum up data to get total number of counts in peak
    area = 0 
    for y_i in y:
        area += y_i
    print("Total measured counts in all peaks in fit window:  "+str(area))

    # create dictionaries to store various properties of the different subpeaks
    area2 = {} 
    out.m_fit_u = {} 
    out.sigma_fit_u = {}
    out.m_error_fit_u = {}
    out.fit_error_mu_u = {}
    pd.set_option('float_format', '{:f}'.format)
    df_fit_results = pd.DataFrame(data=None,columns=['Species','Counts in peak','Mass  [keV]','Stat. mass error [keV]','Fit error [keV]','TITAN - AME [keV]','χ_sqr_red','m_AME - m_e [keV]','m_AME error [keV]'])
        
    for i in range(len(peak_pos)): # print additional results for each subpeak
        print()
        pref = 'emg{0}_'.format(i) # determine prefix for respective peak
        area2[pref] = 0 
        for y_i in comps[pref]: # Get counts in subpeaks from best fit to data, use this value in case of overlapping peaks!
            if np.isnan(y_i) == True:
                print("Warning: Encountered NaN values in subpeak "+str(i)+"! Those are omitted in area summation. Check y-data array of this subpeak:")
                #print(comps[pref]) # print y data so user can check for NaN values
            else:
                area2[pref] += y_i
        # Store and print resulting mass value after scaling
        out.m_fit_u[pref] = scl_fac*mu_emg(out.best_values[pref+'mu'],out.best_values[pref+'theta'],(out.best_values[pref+'eta_m1'],out.best_values[pref+'eta_m2']),(out.best_values[pref+'tau_m1'],out.best_values[pref+'tau_m2']),(out.best_values[pref+'eta_p1'],out.best_values[pref+'eta_p2']),(out.best_values[pref+'tau_p1'],out.best_values[pref+'tau_p2'])) # Gaussian use: out.best_values['mu']
        out.sigma_fit_u[pref] = scl_fac*sigma_emg(out.best_values[pref+'sigma'],out.best_values[pref+'theta'],(out.best_values[pref+'eta_m1'],out.best_values[pref+'eta_m2']),(out.best_values[pref+'tau_m1'],out.best_values[pref+'tau_m2']),(out.best_values[pref+'eta_p1'],out.best_values[pref+'eta_p2']),(out.best_values[pref+'tau_p1'],out.best_values[pref+'tau_p2'])) # for Gaussian use: out.best_values['sigma']    
        out.m_error_fit_u[pref] = out.sigma_fit_u[pref]/np.sqrt(area2[pref]) # statistical mass error  
        try: # handle missing uncertainty estimates
            out.fit_error_mu_u[pref] = out.params[pref+'mu'].stderr  # estimate of error on centroid from fit algorithm  
        except TypeError:
            out.fit_error_mu_u[pref] = np.nan
        if peak_labels != []: 
            print("Peak "+str(peak_labels[i]))
        print("Counts in peak of fitted curve "+str(pref)+": "+str(area2[pref]))
        print("Mass in u:  "+str(out.m_fit_u[pref])+" +/- "+str( out.m_error_fit_u[pref])+" u")
        print("Mass:  "+str(out.m_fit_u[pref]*u_to_keV)+" +/- "+str( out.m_error_fit_u[pref]*u_to_keV)+" keV")
        # Print literature mass (AME2016) after correction with electron mass
        if len(li_m_AME) > i and (li_m_AME[i] != None):
            m_lit = li_m_AME[i] - m_e
            print("AME2016 mass - m_e:  "+str(m_lit*u_to_keV)+" +/- "+str(li_m_AME_error[i]*u_to_keV)+" keV")
            print("TITAN - AME2016 mass:  "+str((out.m_fit_u[pref]-m_lit)*u_to_keV)+" keV")
        if peak_labels != []: # identified species
            if len(li_m_AME) > i and (li_m_AME[i] != None):  # AME value available
                df_fit_results = df_fit_results.append({'Species' : str(peak_labels[i]),'Counts in peak' : np.round(area2[pref],decimals=1), 'Mass  [keV]' : float(out.m_fit_u[pref]*u_to_keV),'Stat. mass error [keV]' : float(out.m_error_fit_u[pref]*u_to_keV),'Fit error [keV]' : float(out.fit_error_mu_u[pref]*u_to_keV),'TITAN - AME [keV]' : float((out.m_fit_u[pref]-m_lit)*u_to_keV),'χ_sqr_red' : float(out.redchi),'m_AME - m_e [keV]' : float(m_lit*u_to_keV),'m_AME error [keV]' : float(li_m_AME_error[i]*u_to_keV)},ignore_index=True) 
            else:  # no AME value available
                df_fit_results = df_fit_results.append({'Species' : str(peak_labels[i]),'Counts in peak' : np.round(area2[pref],decimals=1), 'Mass  [keV]' : float(out.m_fit_u[pref]*u_to_keV),'Stat. mass error [keV]' : float(out.m_error_fit_u[pref]*u_to_keV),'Fit error [keV]' : float(out.fit_error_mu_u[pref]*u_to_keV),'TITAN - AME [keV]' : ' ','χ_sqr_red' : float(out.redchi),'m_AME - m_e [keV]' : ' ','m_AME error [keV]' : ' '},ignore_index=True)
        else: # unidentified species
            if len(li_m_AME) > i and (li_m_AME[i] != None):  # AME value available
                df_fit_results = df_fit_results.append({'Species' : '?','Counts in peak' : np.round(area2[pref],decimals=1), 'Mass  [keV]' : float(out.m_fit_u[pref]*u_to_keV),'Stat. mass error [keV]' : float(out.m_error_fit_u[pref]*u_to_keV),'Fit error [keV]' : float(out.fit_error_mu_u[pref]*u_to_keV),'TITAN - AME [keV]' : float((out.m_fit_u[pref]-m_lit)*u_to_keV),'χ_sqr_red' : float(out.redchi),'m_AME - m_e [keV]' : float(m_lit*u_to_keV),'m_AME error [keV]' : float(li_m_AME_error[i]*u_to_keV)},ignore_index=True)
            else:  # no AME value available
                df_fit_results = df_fit_results.append({'Species' : '?','Counts in peak' : np.round(area2[pref],decimals=1), 'Mass  [keV]' : float(out.m_fit_u[pref]*u_to_keV),'Stat. mass error [keV]' : float(out.m_error_fit_u[pref]*u_to_keV),'Fit error [keV]' : float(out.fit_error_mu_u[pref]*u_to_keV),'TITAN - AME [keV]' : ' ','χ_sqr_red' : float(out.redchi),'m_AME - m_e [keV]' : ' ','m_AME error [keV]' : ' '},ignore_index=True) 
    multi_peak_fit_emg_m2_p2.out = out # store fit results in global variable
    display(df_fit_results)
    


###################################################################################################
###################################################################################################

##### Define initial parameters and store them in dictionary
amp = 0.45
mu = None
sigma = 0.00018 #0.00017
theta = 0.35 # 0.6
eta_m1 = 0.9
eta_m2 = 0.1
eta_m3 = 0
tau_m1 = 25e-06 #[u]
tau_m2 = 400e-06
tau_m3 = 0
eta_p1 = 0.9
eta_p2 = 0.1
eta_p3 = 0
tau_p1 = 35e-06
tau_p2 = 400e-06
tau_p3 = 0

pars_dict = {'amp': amp, 'mu': mu, 'sigma': sigma, 'theta': theta, 'eta_m1': eta_m1, 'eta_m2': eta_m2, 'eta_m3': eta_m3, 'tau_m1': tau_m1, 'tau_m2': tau_m2, 'tau_m3': tau_m3, 'eta_p1': eta_p1, 'eta_p2': eta_p2, 'eta_p3': eta_p3, 'tau_p1': tau_p1, 'tau_p2': tau_p2, 'tau_p3': tau_p3}


##### Define emg22 fit model 
def emg22(peak_index, x_pos, amp, init_pars=pars_dict, vary_shape_pars=True):
    
    # Define model function
    def emg22(x, amp, mu, sigma, theta, eta_m1,eta_m2,tau_m1,tau_m2,eta_p1,eta_p2,tau_p1,tau_p2): 
        return amp*h_emg(x, mu, sigma, theta, (eta_m1,eta_m2),(tau_m1,tau_m2),(eta_p1,eta_p2),(tau_p1,tau_p2)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = fit.Model(emg22, prefix = pref, nan_policy='propagate')
 
    if shape_calibrant:
        constraint = shape_calibrant.eak 
    expr = constraint
    # Add parameters bounds or restrictions and define starting values 
    model.set_param_hint(pref+'amp', value=amp, min=0) 
    model.set_param_hint(pref+'mu', value=x_pos, min=x_pos-0.01, max=x_pos+0.01)
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=init_pars['sigma']-0.005, max=init_pars['sigma']+0.005, vary=vary_shape_pars)
    model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, vary=vary_shape_pars) 
    model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1',vary=vary_shape_pars) # ensures normalization of eta_m's
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=0, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=0, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, expr='1-'+pref+'eta_p1',vary=vary_shape_pars) # ensures normalization of eta_p's
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=0, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=0, vary=vary_shape_pars)
    pars = model.make_params() # create parameters object
    return model


"""
#### Define emg22 model class
# Define model function
def h_emg_m2_p2(x, amp, mu, sigma, theta, eta_m1,eta_m2,tau_m1,tau_m2,eta_p1,eta_p2,tau_p1,tau_p2): 
    return amp*h_emg(x, mu, sigma, theta, (eta_m1,eta_m2),(tau_m1,tau_m2),(eta_p1,eta_p2),(tau_p1,tau_p2)) # from emg_funcs.py

#####
def peak_fit_emg22(spectrum, x_fit_cen=None, x_fit_range=None, init_pars=None, vary_shape_pars=False, scl_fac=1):
    x_min = x_fit_cen - x_fit_range/2
    x_max = x_fit_cen + x_fit_range/2
    df_fit = spectrum.data[x_min:x_max] # cut data to fit range
    peaks_to_fit = [peak for peak in spectrum.peaks if (x_min < peak.x_pos < x_max)] # select peak at position 'x_pos'
    print(peaks_to_fit)
    x = df_fit.index.values
    y = df_fit['Counts'].values
    y_err = np.sqrt(y+1) # assume Poisson (counting) statistics -> standard deviation of each point approximated by sqrt(counts+1)
    weight_facs = 1./y_err # makes sure that residuals include division by statistical error (residual = (fit_model - y) * weights)

    make_model = emg22_fit.make_model_emg22 # define single-peak fit model
   
    # create multi-peak composite model from single-peak model
    mod = fit.models.ConstantModel(independent_vars='x',prefix='bkg_') # Background
    mod.set_param_hint('bkg_c', value= 0.3, min=0)
    for peak in peaks_to_fit: # loop over peaks to fit
        peak_index = spectrum.peaks.index(peak)
        x_pos = peak.x_pos
        amp = df_fit['Counts'].loc[x_pos]/2200 # estimate amplitude from peak maximum, the factor 2235 is empirically determined and shape-dependent
        this_mod = make_model(peak_index, x_pos, amp, vary_shape_pars=vary_shape)
        if mod is None: 
            mod = this_mod
        else:
            mod = mod + this_mod
    pars = mod.make_params()    

    # Perform fit, print fit report and plot resulting fit 
    out = mod.fit(y, x=x, params=pars, weights = weight_facs,method='leastsq',scale_covar=False)
    return out
"""




