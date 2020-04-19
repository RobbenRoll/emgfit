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
upper_bound_taus = 5e-02 # prevents minimizer from running towards virtually flat tails

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

def create_default_init_pars(mass_number=100):
    """ Re-scale default parameters for mass 100 to mass number of interest and store them in dictionary """
    # Default initial parameters for peaks around mass 100 (with re-scaling factor):
    scl_factor = mass_number/100
    amp = 0.45*scl_factor
    mu = None
    sigma = 0.00014*scl_factor # [u]
    theta = 0.5 # 0.35
    eta_m1 = 0.85
    eta_m2 = 0.10
    eta_m3 = 0.05
    tau_m1 = 50e-06*scl_factor # [u]
    tau_m2 = 500e-06*scl_factor # [u]
    tau_m3 = 1000e-06*scl_factor # [u]
    eta_p1 = 0.85
    eta_p2 = 0.10
    eta_p3 = 0.05
    tau_p1 = 50e-06*scl_factor # [u]
    tau_p2 = 600e-06*scl_factor # [u]
    tau_p3 = 1000e-06*scl_factor # [u]
    pars_dict = {'amp': amp, 'mu': mu, 'sigma': sigma, 'theta': theta, 'eta_m1': eta_m1, 'eta_m2': eta_m2, 'eta_m3': eta_m3, 'tau_m1': tau_m1, 'tau_m2': tau_m2, 'tau_m3': tau_m3, 'eta_p1': eta_p1, 'eta_p2': eta_p2, 'eta_p3': eta_p3, 'tau_p1': tau_p1, 'tau_p2': tau_p2, 'tau_p3': tau_p3}
    return pars_dict

##### Define default initial parameters and store them in dictionary
amp = 0.45
mu = None
sigma = 0.00018 #0.00017
theta = 0.5 # 0.35
eta_m1 = 0.85
eta_m2 = 0.10
eta_m3 = 0.05
tau_m1 = 50e-06 #[u]
tau_m2 = 500e-06
tau_m3 = 1000e-06
eta_p1 = 0.85
eta_p2 = 0.10
eta_p3 = 0.05
tau_p1 = 50e-06
tau_p2 = 600e-06
tau_p3 = 1000e-06

pars_dict = {'amp': amp, 'mu': mu, 'sigma': sigma, 'theta': theta, 'eta_m1': eta_m1, 'eta_m2': eta_m2, 'eta_m3': eta_m3, 'tau_m1': tau_m1, 'tau_m2': tau_m2, 'tau_m3': tau_m3, 'eta_p1': eta_p1, 'eta_p2': eta_p2, 'eta_p3': eta_p3, 'tau_p1': tau_p1, 'tau_p2': tau_p2, 'tau_p3': tau_p3}

###################################################################################################
##### Define Gaussian fit model ######################################################################
def Gaussian(peak_index, x_pos, amp, init_pars=pars_dict, vary_shape_pars=True, index_first_peak=None):
    """
    Gaussian lmfit model (single-peak Gaussian fit model)

    Parameters:
    -----------
        peak_index (int): index of peak to fit
        x_pos (float): initial guess of peak centroid
        amp (float): initial guess of peak amplitude
        init_pars (dict): initial parameters for fit ('amp' and 'mu' parameters in 'init_pars' dictionary are overwritten by 'amp' and 'x_pos' arguments)
        vary_shape_pars (bool): Boolean flag whether to vary or fix peak shape parameters (i.e. sigma, theta, eta's and tau's)
        index_first_peak (int): index of the first peak to be fit in a multi-peak-fit. Only use this during peak shape determination to enforce
                                common shape parameters for all peaks to be fitted. (For a regular fit this is done by setting 'vary_shape_pars = False'.)

    Returns:
    --------
    lmfit model object
    """
    # Define model function
    def Gaussian(x, amp, mu, sigma):
        return  amp/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*sigma**2))
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = fit.Model(Gaussian, prefix = pref, nan_policy='propagate')

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp, min=0)
    model.set_param_hint(pref+'mu', value=x_pos, min=x_pos-0.01, max=x_pos+0.01)
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, vary=vary_shape_pars)

    if index_first_peak != None and (peak_index != index_first_peak): # enfore common shape parameters for all peaks (only needed during peak shape calibration)
        first_pref = 'p{0}_'.format(index_first_peak)
        model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, expr=first_pref+'sigma')

    return model


###################################################################################################
##### Define emg01 fit model ######################################################################
def emg01(peak_index, x_pos, amp, init_pars=pars_dict, vary_shape_pars=True, index_first_peak=None):
    """
    Hyper-EMG(0,1) lmfit model (single-peak fit model with one exponential tail on the right)

    Parameters:
    -----------
        peak_index (int): index of peak to fit
        x_pos (float): initial guess of peak centroid
        amp (float): initial guess of peak amplitude
        init_pars (dict): initial parameters for fit ('amp' and 'mu' parameters in 'init_pars' dictionary are overwritten by 'amp' and 'x_pos' arguments)
        vary_shape_pars (bool): Boolean flag whether to vary or fix peak shape parameters (i.e. sigma, theta, eta's and tau's)
        index_first_peak (int): index of the first peak to be fit in a multi-peak-fit. Only use this during peak shape determination to enforce
                                common shape parameters for all peaks to be fitted. (For a regular fit this is done by setting 'vary_shape_pars = False'.)

    Returns:
    --------
    lmfit model object
    """
    # Define model function
    def emg01(x, amp, mu, sigma, tau_p1):
        return amp*h_emg(x, mu, sigma, 0, (0,),(0,),(1,),(tau_p1,)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = fit.Model(emg01, prefix = pref, nan_policy='propagate')

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp, min=1e-20)
    model.set_param_hint(pref+'mu', value=x_pos, min=x_pos-0.01, max=x_pos+0.01)
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)

    if index_first_peak != None and (peak_index != index_first_peak): # enfore common shape parameters for all peaks (only needed during peak shape calibration)
        first_pref = 'p{0}_'.format(index_first_peak)
        model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, expr=first_pref+'sigma')
        model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p1')

    return model


###################################################################################################
##### Define emg10 fit model ######################################################################
def emg10(peak_index, x_pos, amp, init_pars=pars_dict, vary_shape_pars=True, index_first_peak=None):
    """
    Hyper-EMG(1,0) lmfit model (single-peak fit model with one exponential tail on the left)

    Parameters:
    -----------
        peak_index (int): index of peak to fit
        x_pos (float): initial guess of peak centroid
        amp (float): initial guess of peak amplitude
        init_pars (dict): initial parameters for fit ('amp' and 'mu' parameters in 'init_pars' dictionary are overwritten by 'amp' and 'x_pos' arguments)
        vary_shape_pars (bool): Boolean flag whether to vary or fix peak shape parameters (i.e. sigma, theta, eta's and tau's)
        index_first_peak (int): index of the first peak to be fit in a multi-peak-fit. Only use this during peak shape determination to enforce
                                common shape parameters for all peaks to be fitted. (For a regular fit this is done by setting 'vary_shape_pars = False'.)

    Returns:
    --------
    lmfit model object
    """
    # Define model function
    def emg10(x, amp, mu, sigma, tau_m1):
        return amp*h_emg(x, mu, sigma, 1, (1,),(tau_m1,),(0,),(0,)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = fit.Model(emg10, prefix = pref, nan_policy='propagate')

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp, min=1e-20)
    model.set_param_hint(pref+'mu', value=x_pos, min=x_pos-0.01, max=x_pos+0.01)
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)

    if index_first_peak != None and (peak_index != index_first_peak): # enfore common shape parameters for all peaks (only needed during peak shape calibration)
        first_pref = 'p{0}_'.format(index_first_peak)
        model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, expr=first_pref+'sigma')
        model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m1')

    return model


###################################################################################################
##### Define emg11 fit model ######################################################################
def emg11(peak_index, x_pos, amp, init_pars=pars_dict, vary_shape_pars=True, index_first_peak=None):
    """
    Hyper-EMG(1,1) lmfit model (single-peak fit model with one exponential tail on the left and one exponential tail on the right)

    Parameters:
    -----------
        peak_index (int): index of peak to fit
        x_pos (float): initial guess of peak centroid
        amp (float): initial guess of peak amplitude
        init_pars (dict): initial parameters for fit ('amp' and 'mu' parameters in 'init_pars' dictionary are overwritten by 'amp' and 'x_pos' arguments)
        vary_shape_pars (bool): Boolean flag whether to vary or fix peak shape parameters (i.e. sigma, theta, eta's and tau's)
        index_first_peak (int): index of the first peak to be fit in a multi-peak-fit. Only use this during peak shape determination to enforce
                                common shape parameters for all peaks to be fitted. (For a regular fit this is done by setting 'vary_shape_pars = False'.)

    Returns:
    --------
    lmfit model object
    """
    # Define model function
    def emg11(x, amp, mu, sigma, theta, tau_m1, tau_p1):
        return amp*h_emg(x, mu, sigma, theta, (1,),(tau_m1,),(1,),(tau_p1,)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = fit.Model(emg11, prefix = pref, nan_policy='propagate')

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp, min=1e-20)
    model.set_param_hint(pref+'mu', value=x_pos, min=x_pos-0.01, max=x_pos+0.01)
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, vary=vary_shape_pars)
    model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)

    if index_first_peak != None and (peak_index != index_first_peak): # enfore common shape parameters for all peaks (only needed during peak shape calibration)
        first_pref = 'p{0}_'.format(index_first_peak)
        model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, expr=first_pref+'sigma')
        model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, expr=first_pref+'theta')
        model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m1')
        model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p1')

    return model


###################################################################################################
##### Define emg12 fit model
def emg12(peak_index, x_pos, amp, init_pars=pars_dict, vary_shape_pars=True, index_first_peak=None):
    """
    Hyper-EMG(1,2) lmfit model (single-peak fit model with one exponential tail on the left and two exponential tails on the right)

    Parameters:
    -----------
        peak_index (int): index of peak to fit
        x_pos (float): initial guess of peak centroid
        amp (float): initial guess of peak amplitude
        init_pars (dict): initial parameters for fit ('amp' and 'mu' parameters in 'init_pars' dictionary are overwritten by 'amp' and 'x_pos' arguments)
        vary_shape_pars (bool): Boolean flag whether to vary or fix peak shape parameters (i.e. sigma, theta, eta's and tau's)
        index_first_peak (int): index of the first peak to be fit in a multi-peak-fit. Only use this during peak shape determination to enforce
                                common shape parameters for all peaks to be fitted. (For a regular fit this is done by setting 'vary_shape_pars = False'.)

    Returns:
    --------
    lmfit model object
    """
    # Define model function
    def emg12(x, amp, mu, sigma, theta, tau_m1,eta_p1,eta_p2,tau_p1,tau_p2):
        return amp*h_emg(x, mu, sigma, theta, (1,),(tau_m1,),(eta_p1,eta_p2),(tau_p1,tau_p2)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = fit.Model(emg12, prefix = pref, nan_policy='propagate')

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp, min=1e-20)
    model.set_param_hint(pref+'mu', value=x_pos, min=x_pos-0.01, max=x_pos+0.01)
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, vary=vary_shape_pars)
    model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, max=1, expr='1-'+pref+'eta_p1') # ensures normalization of eta_p's
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)

    if index_first_peak != None and (peak_index != index_first_peak): # enfore common shape parameters for all peaks (only needed during peak shape calibration)
        first_pref = 'p{0}_'.format(index_first_peak)
        model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, expr=first_pref+'sigma')
        model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, expr=first_pref+'theta')
        model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m1')
        model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, expr=first_pref+'eta_p1')
        model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, max=1, expr='1-'+pref+'eta_p1') # ensures normalization of eta_p's
        model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p1')
        model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p2')

    return model


###################################################################################################
##### Define emg21 fit model
def emg21(peak_index, x_pos, amp, init_pars=pars_dict, vary_shape_pars=True, index_first_peak=None):
    """
    Hyper-EMG(2,1) lmfit model (single-peak fit model with two exponential tails on the left and one exponential tail on the right)

    Parameters:
    -----------
        peak_index (int): index of peak to fit
        x_pos (float): initial guess of peak centroid
        amp (float): initial guess of peak amplitude
        init_pars (dict): initial parameters for fit ('amp' and 'mu' parameters in 'init_pars' dictionary are overwritten by 'amp' and 'x_pos' arguments)
        vary_shape_pars (bool): Boolean flag whether to vary or fix peak shape parameters (i.e. sigma, theta, eta's and tau's)
        index_first_peak (int): index of the first peak to be fit in a multi-peak-fit. Only use this during peak shape determination to enforce
                                common shape parameters for all peaks to be fitted. (For a regular fit this is done by setting 'vary_shape_pars = False'.)

    Returns:
    --------
    lmfit model object
    """
    # Define model function
    def emg21(x, amp, mu, sigma, theta, eta_m1,eta_m2,tau_m1,tau_m2,tau_p1):
        return amp*h_emg(x, mu, sigma, theta, (eta_m1,eta_m2),(tau_m1,tau_m2),(1,),(tau_p1,)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = fit.Model(emg21, prefix = pref, nan_policy='propagate')

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp, min=1e-20)
    model.set_param_hint(pref+'mu', value=x_pos, min=x_pos-0.01, max=x_pos+0.01)
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, vary=vary_shape_pars)
    model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1') # ensures normalization of eta_m's
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)

    if index_first_peak != None and (peak_index != index_first_peak): # enfore common shape parameters for all peaks (only needed during peak shape calibration)
        first_pref = 'p{0}_'.format(index_first_peak)
        model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, expr=first_pref+'sigma')
        model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, expr=first_pref+'theta')
        model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, expr=first_pref+'eta_m1' )
        model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1') # ensures normalization of eta_m's
        model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m1')
        model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m2')
        model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p1')

    return model


###################################################################################################
##### Define emg22 fit model
def emg22(peak_index, x_pos, amp, init_pars=pars_dict, vary_shape_pars=True, index_first_peak=None):
    """
    Hyper-EMG(2,2) lmfit model (single-peak fit model with two exponential tails on the left and two exponential tails on the right)

    Parameters:
    -----------
        peak_index (int): index of peak to fit
        x_pos (float): initial guess of peak centroid
        amp (float): initial guess of peak amplitude
        init_pars (dict): initial parameters for fit ('amp' and 'mu' parameters in 'init_pars' dictionary are overwritten by 'amp' and 'x_pos' arguments)
        vary_shape_pars (bool): Boolean flag whether to vary or fix peak shape parameters (i.e. sigma, theta, eta's and tau's)
        index_first_peak (int): index of the first peak to be fit in a multi-peak-fit. Only use this during peak shape determination to enforce
                                common shape parameters for all peaks to be fitted. (For a regular fit this is done by setting 'vary_shape_pars = False'.)

    Returns:
    --------
    lmfit model object
    """
    # Define model function
    def emg22(x, amp, mu, sigma, theta, eta_m1,eta_m2,tau_m1,tau_m2,eta_p1,eta_p2,tau_p1,tau_p2):
        return amp*h_emg(x, mu, sigma, theta, (eta_m1,eta_m2),(tau_m1,tau_m2),(eta_p1,eta_p2),(tau_p1,tau_p2)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = fit.Model(emg22, prefix = pref, nan_policy='propagate')

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp, min=1e-20)
    model.set_param_hint(pref+'mu', value=x_pos, min=x_pos-0.01, max=x_pos+0.01)
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, vary=vary_shape_pars)
    model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1') # ensures normalization of eta_m's
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, max=1, expr='1-'+pref+'eta_p1') # ensures normalization of eta_p's
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)

    if index_first_peak != None and (peak_index != index_first_peak): # enfore common shape parameters for all peaks (only needed during peak shape calibration)
        first_pref = 'p{0}_'.format(index_first_peak)
        model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, expr=first_pref+'sigma')
        model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, expr=first_pref+'theta')
        model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, expr=first_pref+'eta_m1' )
        model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1') # ensures normalization of eta_m's
        model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m1')
        model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m2')
        model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, expr=first_pref+'eta_p1')
        model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, max=1, expr='1-'+pref+'eta_p1') # ensures normalization of eta_p's
        model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p1')
        model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p2')

    return model


###################################################################################################
##### Define emg23 fit model
def emg23(peak_index, x_pos, amp, init_pars=pars_dict, vary_shape_pars=True, index_first_peak=None):
    """
    Hyper-EMG(2,3) lmfit model (single-peak fit model with two exponential tails on the left and three exponential tails on the right)

    Parameters:
    -----------
        peak_index (int): index of peak to fit
        x_pos (float): initial guess of peak centroid
        amp (float): initial guess of peak amplitude
        init_pars (dict): initial parameters for fit ('amp' and 'mu' parameters in 'init_pars' dictionary are overwritten by 'amp' and 'x_pos' arguments)
        vary_shape_pars (bool): Boolean flag whether to vary or fix peak shape parameters (i.e. sigma, theta, eta's and tau's)
        index_first_peak (int): index of the first peak to be fit in a multi-peak-fit. Only use this during peak shape determination to enforce
                                common shape parameters for all peaks to be fitted. (For a regular fit this is done by setting 'vary_shape_pars = False'.)

    Returns:
    --------
    lmfit model object
    """
    # Define model function
    def emg23(x, amp, mu, sigma, theta, eta_m1,eta_m2,tau_m1,tau_m2,eta_p1,eta_p2,eta_p3,delta_p,tau_p1,tau_p2,tau_p3):
        return amp*h_emg(x, mu, sigma, theta, (eta_m1,eta_m2),(tau_m1,tau_m2),(eta_p1,eta_p2,eta_p3),(tau_p1,tau_p2,tau_p3)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = fit.Model(emg23, prefix = pref, nan_policy='propagate')

    if init_pars['eta_p2'] < 0: # avoids math errors in peak shape uncertainty estimation
        init_pars['eta_p2'] = 0
    if init_pars['eta_p2'] > 1:
        init_pars['eta_p2'] = 1

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp, min=1e-20)
    model.set_param_hint(pref+'mu', value=x_pos, min=x_pos-0.01, max=x_pos+0.01)
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, vary=vary_shape_pars)
    model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1') # ensures normalization of eta_m's
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'delta_p', value= init_pars['eta_p1']+init_pars['eta_p2'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, max=1, vary=vary_shape_pars, expr= pref+'delta_p-'+pref+'eta_p1')
    model.set_param_hint(pref+'eta_p3', value= 1-init_pars['eta_p1']-init_pars['eta_p2'], min=0, max=1, expr='1-'+pref+'eta_p1-'+pref+'eta_p2') # ensures normalization of eta_p's
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p3', value= init_pars['tau_p3'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)

    if index_first_peak != None and (peak_index != index_first_peak): # enfore common shape parameters for all peaks (only needed during peak shape calibration)
        first_pref = 'p{0}_'.format(index_first_peak)
        model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, expr=first_pref+'sigma')
        model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, expr=first_pref+'theta')
        model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, expr=first_pref+'eta_m1' )
        model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1') # ensures normalization of eta_m's
        model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m1')
        model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m2')
        model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, expr=first_pref+'eta_p1')
        model.set_param_hint(pref+'delta_p', value= init_pars['eta_p1']+init_pars['eta_p2'], min=0, max=1, expr=first_pref+'delta_p')
        model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, max=1, expr= pref+'delta_p-'+pref+'eta_p1')
        model.set_param_hint(pref+'eta_p3', value= 1-init_pars['eta_p1']-init_pars['eta_p2'], min=0, max=1, expr='1-'+pref+'eta_p1-'+pref+'eta_p2') # ensures norm. of eta_p's
        model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p1')
        model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p2')
        model.set_param_hint(pref+'tau_p3', value= init_pars['tau_p3'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p3')

    return model


###################################################################################################
##### Define emg32 fit model
def emg32(peak_index, x_pos, amp, init_pars=pars_dict, vary_shape_pars=True, index_first_peak=None):
    """
    Hyper-EMG(3,2) lmfit model (single-peak fit model with three exponential tails on the left and two exponential tails on the right)

    Parameters:
    -----------
        peak_index (int): index of peak to fit
        x_pos (float): initial guess of peak centroid
        amp (float): initial guess of peak amplitude
        init_pars (dict): initial parameters for fit ('amp' and 'mu' parameters in 'init_pars' dictionary are overwritten by 'amp' and 'x_pos' arguments)
        vary_shape_pars (bool): Boolean flag whether to vary or fix peak shape parameters (i.e. sigma, theta, eta's and tau's)
        index_first_peak (int): index of the first peak to be fit in a multi-peak-fit. Only use this during peak shape determination to enforce
                                common shape parameters for all peaks to be fitted. (For a regular fit this is done by setting 'vary_shape_pars = False'.)

    Returns:
    --------
    lmfit model object
    """
    # Define model function
    def emg32(x, amp, mu, sigma, theta, eta_m1,eta_m2,eta_m3,delta_m,tau_m1,tau_m2,tau_m3,eta_p1,eta_p2,tau_p1,tau_p2):
        return amp*h_emg(x, mu, sigma, theta, (eta_m1,eta_m2,eta_m3),(tau_m1,tau_m2,tau_m3),(eta_p1,eta_p2),(tau_p1,tau_p2)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = fit.Model(emg32, prefix = pref, nan_policy='propagate')

    if init_pars['eta_m2'] < 0: # avoids math errors in peak shape uncertainty estimation
        init_pars['eta_m2'] = 0
    if init_pars['eta_m2'] > 1:
        init_pars['eta_m2'] = 1

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp, min=1e-20)
    model.set_param_hint(pref+'mu', value=x_pos, min=x_pos-0.01, max=x_pos+0.01)
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, vary=vary_shape_pars)
    model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'delta_m', value= init_pars['eta_m1']+init_pars['eta_m2'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, vary=vary_shape_pars, expr= pref+'delta_m-'+pref+'eta_m1')
    model.set_param_hint(pref+'eta_m3', value= 1-init_pars['eta_m1']-init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1-'+pref+'eta_m2')
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m3', value= init_pars['tau_m3'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p2', value= 1-init_pars['eta_p1'], min=0, max=1, vary=vary_shape_pars, expr= '1-'+pref+'eta_p1') # ensures normalization of eta_p's
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)

    if index_first_peak != None and (peak_index != index_first_peak): # enfore common shape parameters for all peaks (only needed during peak shape calibration)
        first_pref = 'p{0}_'.format(index_first_peak)
        model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, expr=first_pref+'sigma')
        model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, expr=first_pref+'theta')
        model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, expr=first_pref+'eta_m1' )
        model.set_param_hint(pref+'delta_m', value= init_pars['eta_m1']+init_pars['eta_m2'], min=0, max=1, expr= first_pref+'delta_m')
        model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, vary=vary_shape_pars, expr= pref+'delta_m-'+pref+'eta_m1')
        model.set_param_hint(pref+'eta_m3', value= 1-init_pars['eta_m1']-init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1-'+pref+'eta_m2') # ensures normalization of eta_m's
        model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m1')
        model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m2')
        model.set_param_hint(pref+'tau_m3', value= init_pars['tau_m3'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m3')
        model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, expr=first_pref+'eta_p1')
        model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, max=1, expr= '1-'+pref+'eta_p1') # ensures norm. of eta_p's
        model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p1')
        model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p2')

    return model


###################################################################################################
##### Define emg33 fit model
def emg33(peak_index, x_pos, amp, init_pars=pars_dict, vary_shape_pars=True, index_first_peak=None):
    """
    Hyper-EMG(3,3) lmfit model (single-peak fit model with three exponential tails on the left and three exponential tails on the right)

    Parameters:
    -----------
        peak_index (int): index of peak to fit
        x_pos (float): initial guess of peak centroid
        amp (float): initial guess of peak amplitude
        init_pars (dict): initial parameters for fit ('amp' and 'mu' parameters in 'init_pars' dictionary are overwritten by 'amp' and 'x_pos' arguments)
        vary_shape_pars (bool): Boolean flag whether to vary or fix peak shape parameters (i.e. sigma, theta, eta's and tau's)
        index_first_peak (int): index of the first peak to be fit in a multi-peak-fit. Only use this during peak shape determination to enforce
                                common shape parameters for all peaks to be fitted. (For a regular fit this is done by setting 'vary_shape_pars = False'.)

    Returns:
    --------
    lmfit model object
    """
    # Define model function
    def emg33(x, amp, mu, sigma, theta, eta_m1,eta_m2,eta_m3,delta_m,tau_m1,tau_m2,tau_m3,eta_p1,eta_p2,eta_p3,delta_p,tau_p1,tau_p2,tau_p3):
        return amp*h_emg(x, mu, sigma, theta, (eta_m1,eta_m2,eta_m3),(tau_m1,tau_m2,tau_m3),(eta_p1,eta_p2,eta_p3),(tau_p1,tau_p2,tau_p3)) # from emg_funcs.py
    pref = 'p{0}_'.format(peak_index) # set prefix for respective peak (e.g. 'p0' for peak with index 0)
    model = fit.Model(emg33, prefix = pref, nan_policy='propagate')

    if init_pars['eta_m2'] < 0: # avoids math errors in peak shape uncertainty estimation
        init_pars['eta_m2'] = 0
    if init_pars['eta_m2'] > 1:
        init_pars['eta_m2'] = 1
    if init_pars['eta_p2'] < 0:
        init_pars['eta_p2'] = 0
    if init_pars['eta_p2'] > 1:
        init_pars['eta_p2'] = 1

    # Add parameters bounds or restrictions and define starting values
    model.set_param_hint(pref+'amp', value=amp, min=1e-20)
    model.set_param_hint(pref+'mu', value=x_pos, min=x_pos-0.01, max=x_pos+0.01)
    model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, vary=vary_shape_pars)
    model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'delta_m', value= init_pars['eta_m1']+init_pars['eta_m2'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, vary=vary_shape_pars, expr= pref+'delta_m-'+pref+'eta_m1')
    model.set_param_hint(pref+'eta_m3', value= 1-init_pars['eta_m1']-init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1-'+pref+'eta_m2') # ensures normalization of eta_m's
    model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_m3', value= init_pars['tau_m3'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'delta_p', value= init_pars['eta_p1']+init_pars['eta_p2'], min=0, max=1, vary=vary_shape_pars)
    model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, max=1, vary=vary_shape_pars, expr= pref+'delta_p-'+pref+'eta_p1')
    model.set_param_hint(pref+'eta_p3', value= 1-init_pars['eta_p1']-init_pars['eta_p2'], min=0, max=1, expr='1-'+pref+'eta_p1-'+pref+'eta_p2') # ensures normalization of eta_p's
    model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)
    model.set_param_hint(pref+'tau_p3', value= init_pars['tau_p3'], min=1e-12, max=upper_bound_taus, vary=vary_shape_pars)

    if index_first_peak != None and (peak_index != index_first_peak): # enfore common shape parameters for all peaks (only needed during peak shape calibration)
        first_pref = 'p{0}_'.format(index_first_peak)
        model.set_param_hint(pref+'sigma', value= init_pars['sigma'], min=0, max=init_pars['sigma']+0.005, expr=first_pref+'sigma')
        model.set_param_hint(pref+'theta', value= init_pars['theta'], min=0, max=1, expr=first_pref+'theta')
        model.set_param_hint(pref+'eta_m1', value= init_pars['eta_m1'], min=0, max=1, expr=first_pref+'eta_m1' )
        model.set_param_hint(pref+'delta_m', value= init_pars['eta_m1']+init_pars['eta_m2'], min=0, max=1, expr= first_pref+'delta_m')
        model.set_param_hint(pref+'eta_m2', value= init_pars['eta_m2'], min=0, max=1, expr= pref+'delta_m-'+pref+'eta_m1')
        model.set_param_hint(pref+'eta_m3', value= 1-init_pars['eta_m1']-init_pars['eta_m2'], min=0, max=1, expr='1-'+pref+'eta_m1-'+pref+'eta_m2') # ensures normalization of eta_m's
        model.set_param_hint(pref+'tau_m1', value= init_pars['tau_m1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m1')
        model.set_param_hint(pref+'tau_m2', value= init_pars['tau_m2'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m2')
        model.set_param_hint(pref+'tau_m3', value= init_pars['tau_m3'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_m3')
        model.set_param_hint(pref+'eta_p1', value= init_pars['eta_p1'], min=0, max=1, expr=first_pref+'eta_p1')
        model.set_param_hint(pref+'delta_p', value= init_pars['eta_p1']+init_pars['eta_p2'], min=0, max=1, expr= first_pref+'delta_p')
        model.set_param_hint(pref+'eta_p2', value= init_pars['eta_p2'], min=0, max=1, expr= pref+'delta_p-'+pref+'eta_p1')
        model.set_param_hint(pref+'eta_p3', value= 1-init_pars['eta_p1']-init_pars['eta_p2'], min=0, max=1, expr='1-'+pref+'eta_p1-'+pref+'eta_p2') # ensures normalization of eta_p's
        model.set_param_hint(pref+'tau_p1', value= init_pars['tau_p1'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p1')
        model.set_param_hint(pref+'tau_p2', value= init_pars['tau_p2'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p2')
        model.set_param_hint(pref+'tau_p3', value= init_pars['tau_p3'], min=1e-12, max=upper_bound_taus, expr=first_pref+'tau_p3')

    return model

# expr= pref+'delta-'+pref+'eta_p1 if ('+pref+'eta_p1 <='+pref+'delta) else '+pref+'eta_p1-'+pref+'delta'
#1 - init_pars['eta_p3']
# init_pars['eta_p3']



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
