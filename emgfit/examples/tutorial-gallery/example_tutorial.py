"""
Tutorial
========

This tutorial notebook gives a basic example of the workflow with emgfit. This
notebook can serve as a template for fitting your own spectra. Feel free to
remove any of the comments and adapt this to your own needs.

"""
import emgfit as emg

### Import mass data, plot full spectrum and choose fit range
filename = "2019-09-13_004-_006 SUMMED High stats 62Ga" # input file (as exported with MAc's hist-mode)
skiprows = 38 # number of header rows to skip upon data import
m_start = 61.9243 # low-mass cut off
m_stop = 61.962 # high-mass cut off
spec = emg.spectrum(filename+'.txt',m_start,m_stop,skiprows=skiprows)

###############################################################################
# Adding peaks to the spectrum
# ----------------------------
# This can be done with the automatic peak detection (:meth:`detect_peaks`
# method) and/or by manually adding peaks (:meth:add_peak method).
#
# All info about the peaks is compiled in the peak properties table. The table's
# left-most column shows the respective peak indeces. The peaks' `x_pos` will be
# used as initial values for the (Gaussian) peak centroids in fits.

### Detect peaks and add them to spectrum object 'spec'
spec.detect_peaks() # automatic peak detection
#spec.add_peak(61.925,species='?') # manually add a peak at x_pos = 61.925u
#spec.remove_peak(peak_index=0) # manually remove the peak with index 0

###############################################################################
# Assign the identified species to the peaks and add comments (OPTIONAL)
# ----------------------------------------------------------------------

spec.assign_species(['Ni62:-1e','Cu62:-1e',None,'Ga62:-1e','Ti46:O16:-1e',
                     'Sc46:O16:-1e','Ca43:F19:-1e',None])
spec.add_peak_comment('Non-isobaric',peak_index=2)
spec.show_peak_properties() # check the changes by printing the peak properties table

###############################################################################
# Select the optimal Hyper-EMG tail order and perform the peak-shape calibration
# ------------------------------------------------------------------------------
#It is recommended that the peak-shape calibration is done with a chi-squared
# fit (default) since this yields more robust results and more trusworthy
# parameter uncertainty estimates.

#spec.determine_peak_shape(species_shape_calib='Ca43:F19:-1e') # default settings and automatic model selection
spec.determine_peak_shape(species_shape_calib='Ca43:F19:-1e',x_fit_range=0.0045) # user-defined fit range
#spec.determine_peak_shape(species_shape_calib='Ca43:F19:-1e',fit_model='emg12',vary_tail_order=False) # user-defined model

###############################################################################
# Determine constant of proportionality A_stat_emg for subsequent stat. error estimations (OPTIONAL, feel free to skip this)
# --------------------------------------------------------------------------------------------------------------------------
# The statistical uncertainties of Hyper-EMG fits are estimated using the
# equation:
# $\sigma_{stat} = A_{stat,emg} \cdot \frac{\mathrm{FWHM}}{\sqrt{N_{counts}}}$
# where $\mathrm{FWHM}$ and $N_{counts}$ refer to the full width at half
# maximum and the number of counts in the respective peak. This method will
# typically run for ~10 minutes if N_spetra=1000 (default) is used.
# If this step is skipped, the default value $A_{stat,emg} = 0.52$ will be used.
# Optionally, the plot can be saved by using the `plot_filename` argument.

# Determine A_stat_emg and save the resulting plot
spec.determine_A_stat_emg(species='Ca43:F19:-1e',x_range=0.004,N_spectra=10)

###############################################################################
# Fit all peaks in spectrum, perform mass (re-)calibration, determine
# peak-shape uncertainties and update peak properties table with the results
# The simultaneous mass recalibration is invoked by specifying the
# `species_mass_calib` (or alternatively the `index_mass_calib`) argument

# Maximum likelihood fit of all peaks in the spectrum
spec.fit_peaks(species_mass_calib='Ti46:O16:-1e')
# Alternative: Fit restricted to a user-defined mass range
#spec.fit_peaks(species_mass_calib='Ti46:O16:-1e',x_fit_cen=61.9455,x_fit_range=0.01)
