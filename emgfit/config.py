################################################################################
##### Configuration file for emgfit package
##### Author: Stefan Paul

# Load packages
import scipy.constants as con
import matplotlib.pyplot as plt
import pandas as pd

##### Set image resolution
#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('retina')
# For even higher res, use the following instead:
plt.rcParams['figure.dpi'] = 500

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
plt.rcParams.update({'errorbar.capsize': 2})
plt.rcParams.update({"font.size": 15})

##### Appearance of DataFrames
pd.set_option('precision',2) # global displayed float precision
u_digits = 6 # displayed precision of mass values in atomic mass units u

################################################################################
