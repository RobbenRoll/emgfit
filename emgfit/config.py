################################################################################
##### Configuration file with general settings for emgfit package
##### Author: Stefan Paul

# Load packages
import scipy.constants as con
import matplotlib.pyplot as plt
import pandas as pd

##### Define fundamental constants
global m_e, u, u_to_keV
m_e = con.physical_constants["electron mass in u"][0]  # or 548.5799.0907e-06 # electron mass [u] (from AME2016, Huang2017)
m_p = con.physical_constants["proton mass in u"][0]
m_n = con.physical_constants["neutron mass in u"][0]
u = con.u
# Conversion factor from u to keV:
u_to_keV = con.physical_constants["atomic mass constant energy equivalent in MeV"][0]*1000

##### Define constants for fit routines
# Constant of proportionality for calculating statistical mass error of
# HyperEMG-fits:  stat_error = A_stat_emg * FWHM / sqrt(peak_area[counts])
A_stat_emg_default = 0.52

##### Plot appearance
plt.rcParams.update({"errorbar.capsize": 2})
plt.rcParams.update({"font.size": 12.5})
plt.rcParams.update({"errorbar.capsize": 1.0})
#from IPython.display import set_matplotlib_formats
#plot_fmt = 'retina'
#set_matplotlib_formats(plot_fmt) # Defines default image format in notebooks
dpi = 600 # manually set image resolution - not necessary with SVGs
figwidth = 11.8 # width of spectrum figures
msize = 6 # size of data point markers
lwidth = 1.5 # linewidth of fit curves
labelsize = 10 # size of peak labels

##### Appearance of DataFrames
pd.set_option("precision",2) # global displayed float precision
u_decimals = 6 # displayed decimals for mass values in atomic mass units u

################################################################################
