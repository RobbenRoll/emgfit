################################################################################
##### Configuration file for emgfit package
##### Author: Stefan Paul

# Load packages
import scipy.constants as con
import matplotlib.pyplot as plt


##### Define fundamental constants
global m_e, u, u_to_keV
m_e = con.physical_constants["electron mass in u"][0]  # or 548.5799.0907e-06 # electron mass [u] (from AME2016, Huang2017)
u = con.u
u_to_keV = con.physical_constants["atomic mass constant energy equivalent in MeV"][0]*1000 # conversion factor from u to keV

##### Define constants for fit routines
A_stat_emg_default = 0.52  # constant of proportionality for calculating statistical mass error of HyperEmg-fits:  stat_error = A_stat * FWHM / sqrt(peak_area)

##### Plot appearance
plt.rcParams.update({'errorbar.capsize': 2})


################################################################################
