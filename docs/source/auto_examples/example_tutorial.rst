.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_example_tutorial.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_example_tutorial.py:


Tutorial
========

This tutorial notebook gives a basic example of the workflow with emgfit. This
notebook can serve as a template for fitting your own spectra. Feel free to
remove any of the comments and adapt this to your own needs.


.. code-block:: default

    import emgfit as emg

    ### Import mass data, plot full spectrum and choose fit range
    filename = "2019-09-13_004-_006 SUMMED High stats 62Ga" # input file (as exported with MAc's hist-mode)
    skiprows = 38 # number of header rows to skip upon data import
    m_start = 61.9243 # low-mass cut off
    m_stop = 61.962 # high-mass cut off
    spec = emg.spectrum(filename+'.txt',m_start,m_stop,skiprows=skiprows)




.. image:: /auto_examples/images/sphx_glr_example_tutorial_001.png
    :alt: Spectrum with start and stop markers
    :class: sphx-glr-single-img





Adding peaks to the spectrum
----------------------------
This can be done with the automatic peak detection (:meth:`detect_peaks`
method) and/or by manually adding peaks (:meth:add_peak method).

All info about the peaks is compiled in the peak properties table. The table's
left-most column shows the respective peak indeces. The peaks' `x_pos` will be
used as initial values for the (Gaussian) peak centroids in fits.


.. code-block:: default


    ### Detect peaks and add them to spectrum object 'spec'
    spec.detect_peaks() # automatic peak detection
    #spec.add_peak(61.925,species='?') # manually add a peak at x_pos = 61.925u
    #spec.remove_peak(peak_index=0) # manually remove the peak with index 0




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_002.png
          :alt: Smoothed spectrum
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_003.png
          :alt: Scaled second derivative of spectrum - set threshold indicated
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_004.png
          :alt: Negative part of scaled second derivative, inverted - set threshold indicated
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_005.png
          :alt: Spectrum with detected peaks marked
          :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Peak properties table after peak detection:
           x_pos species comment m_AME m_AME_error  extrapolated fit_model cost_func red_chi  area area_error m_fit rel_stat_error rel_recal_error rel_peakshape_error rel_mass_error     A atomic_ME_keV mass_error_keV m_dev_keV
    0  61.927800       ?       -  None        None         False      None      None    None  None       None  None           None            None                None           None  None          None           None      None
    1  61.932021       ?       -  None        None         False      None      None    None  None       None  None           None            None                None           None  None          None           None      None
    2  61.934369       ?       -  None        None         False      None      None    None  None       None  None           None            None                None           None  None          None           None      None
    3  61.943618       ?       -  None        None         False      None      None    None  None       None  None           None            None                None           None  None          None           None      None
    4  61.946994       ?       -  None        None         False      None      None    None  None       None  None           None            None                None           None  None          None           None      None
    5  61.949527       ?       -  None        None         False      None      None    None  None       None  None           None            None                None           None  None          None           None      None
    6  61.956611       ?       -  None        None         False      None      None    None  None       None  None           None            None                None           None  None          None           None      None
    7  61.958997       ?       -  None        None         False      None      None    None  None       None  None           None            None                None           None  None          None           None      None




Assign the identified species to the peaks and add comments (OPTIONAL)
----------------------------------------------------------------------


.. code-block:: default


    spec.assign_species(['Ni62:-1e','Cu62:-1e',None,'Ga62:-1e','Ti46:O16:-1e',
                         'Sc46:O16:-1e','Ca43:F19:-1e',None])
    spec.add_peak_comment('Non-isobaric',peak_index=2)
    spec.show_peak_properties() # check the changes by printing the peak properties table





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Species of peak 0 assigned as Ni62:-1e
    Species of peak 1 assigned as Cu62:-1e
    Species of peak 3 assigned as Ga62:-1e
    Species of peak 4 assigned as Ti46:O16:-1e
    Species of peak 5 assigned as Sc46:O16:-1e
    Species of peak 6 assigned as Ca43:F19:-1e
    Comment of peak 2 was changed to:  Non-isobaric
           x_pos       species       comment      m_AME   m_AME_error  extrapolated fit_model cost_func red_chi  ... m_fit rel_stat_error rel_recal_error rel_peakshape_error rel_mass_error     A atomic_ME_keV  mass_error_keV m_dev_keV
    0  61.927800      Ni62:-1e             -  61.927796  4.700000e-07         False      None      None    None  ...  None           None            None                None           None  62.0          None            None      None
    1  61.932021      Cu62:-1e             -  61.932046  6.940000e-07         False      None      None    None  ...  None           None            None                None           None  62.0          None            None      None
    2  61.934369             ?  Non-isobaric        NaN           NaN         False      None      None    None  ...  None           None            None                None           None   NaN          None            None      None
    3  61.943618      Ga62:-1e             -  61.943641  6.940000e-07         False      None      None    None  ...  None           None            None                None           None  62.0          None            None      None
    4  61.946994  Ti46:O16:-1e             -  61.946993  1.760001e-07         False      None      None    None  ...  None           None            None                None           None  62.0          None            None      None
    5  61.949527  Sc46:O16:-1e             -  61.949534  7.320000e-07         False      None      None    None  ...  None           None            None                None           None  62.0          None            None      None
    6  61.956611  Ca43:F19:-1e             -  61.956621  2.440018e-07         False      None      None    None  ...  None           None            None                None           None  62.0          None            None      None
    7  61.958997             ?             -        NaN           NaN         False      None      None    None  ...  None           None            None                None           None   NaN          None            None      None

    [8 rows x 20 columns]




Select the optimal Hyper-EMG tail order and perform the peak-shape calibration
 ------------------------------------------------------------------------------
It is recommended that the peak-shape calibration is done with a chi-squared
 fit (default) since this yields more robust results and more trusworthy
 parameter uncertainty estimates.


.. code-block:: default


    #spec.determine_peak_shape(species_shape_calib='Ca43:F19:-1e') # default settings and automatic model selection
    spec.determine_peak_shape(species_shape_calib='Ca43:F19:-1e',x_fit_range=0.0045) # user-defined fit range
    #spec.determine_peak_shape(species_shape_calib='Ca43:F19:-1e',fit_model='emg12',vary_tail_order=False) # user-defined model




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_006.png
          :alt: Gaussian chi-square fit
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_007.png
          :alt: Gaussian chi-square fit
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_008.png
          :alt: emg01 chi-square fit
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_009.png
          :alt: emg01 chi-square fit
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_010.png
          :alt: emg10 chi-square fit
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_011.png
          :alt: emg10 chi-square fit
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_012.png
          :alt: emg11 chi-square fit
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_013.png
          :alt: emg11 chi-square fit
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_014.png
          :alt: emg12 chi-square fit
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_015.png
          :alt: emg12 chi-square fit
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_016.png
          :alt: emg21 chi-square fit
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_017.png
          :alt: emg21 chi-square fit
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_018.png
          :alt: emg22 chi-square fit
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_019.png
          :alt: emg22 chi-square fit
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_020.png
          :alt: emg23 chi-square fit
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_021.png
          :alt: emg23 chi-square fit
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_022.png
          :alt: emg32 chi-square fit
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_023.png
          :alt: emg32 chi-square fit
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_024.png
          :alt: emg33 chi-square fit
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_025.png
          :alt: emg33 chi-square fit
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_026.png
          :alt: emg22 chi-square fit
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_027.png
          :alt: emg22 chi-square fit
          :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    ##### Determine optimal tail order #####


    ##### Fitting data with Gaussian #####-----------------------------------------------------------------------------------------


    Gaussian-fit yields reduced chi-square of: 45.57 +- 0.13


    ##### Fitting data with emg01 #####-----------------------------------------------------------------------------------------


    emg01-fit yields reduced chi-square of: 13.79 +- 0.13


    ##### Fitting data with emg10 #####-----------------------------------------------------------------------------------------


    emg10-fit yields reduced chi-square of: 45.96 +- 0.13


    ##### Fitting data with emg11 #####-----------------------------------------------------------------------------------------


    emg11-fit yields reduced chi-square of: 2.6 +- 0.13


    ##### Fitting data with emg12 #####-----------------------------------------------------------------------------------------


    emg12-fit yields reduced chi-square of: 1.22 +- 0.13


    ##### Fitting data with emg21 #####-----------------------------------------------------------------------------------------


    emg21-fit yields reduced chi-square of: 1.47 +- 0.13


    ##### Fitting data with emg22 #####-----------------------------------------------------------------------------------------


    emg22-fit yields reduced chi-square of: 0.96 +- 0.13


    ##### Fitting data with emg23 #####-----------------------------------------------------------------------------------------

    WARNING: p6_eta_m1 = 0.891 +- 2.086  is compatible with zero within uncertainty.
                 This tail order is likely overfitting the data and will be excluded from selection.
    WARNING: p6_eta_m2 = 0.109 +- 2.086  is compatible with zero within uncertainty.
                 This tail order is likely overfitting the data and will be excluded from selection.
    WARNING: p6_eta_p1 = 0.443 +- 6.29  is compatible with zero within uncertainty.
                 This tail order is likely overfitting the data and will be excluded from selection.
    WARNING: p6_eta_p2 = 0.458 +- 4.679  is compatible with zero within uncertainty.
                 This tail order is likely overfitting the data and will be excluded from selection.
    WARNING: p6_eta_p3 = 0.099 +- 6.29  is compatible with zero within uncertainty.
                 This tail order is likely overfitting the data and will be excluded from selection.

    emg23-fit yields reduced chi-square of: 0.98 +- 0.13


    ##### Fitting data with emg32 #####-----------------------------------------------------------------------------------------

    WARNING: p6_eta_m3 = 0.007 +- 0.1  is compatible with zero within uncertainty.
                 This tail order is likely overfitting the data and will be excluded from selection.

    emg32-fit yields reduced chi-square of: 0.98 +- 0.13


    ##### Fitting data with emg33 #####-----------------------------------------------------------------------------------------

    WARNING: p6_eta_m1 = 0.872 +- 3.638  is compatible with zero within uncertainty.
                 This tail order is likely overfitting the data and will be excluded from selection.
    WARNING: p6_eta_m2 = 0.12 +- 3.387  is compatible with zero within uncertainty.
                 This tail order is likely overfitting the data and will be excluded from selection.
    WARNING: p6_eta_m3 = 0.009 +- 3.638  is compatible with zero within uncertainty.
                 This tail order is likely overfitting the data and will be excluded from selection.
    WARNING: p6_eta_p1 = 0.445 +- 10.079  is compatible with zero within uncertainty.
                 This tail order is likely overfitting the data and will be excluded from selection.
    WARNING: p6_eta_p2 = 0.459 +- 7.601  is compatible with zero within uncertainty.
                 This tail order is likely overfitting the data and will be excluded from selection.
    WARNING: p6_eta_p3 = 0.096 +- 10.079  is compatible with zero within uncertainty.
                 This tail order is likely overfitting the data and will be excluded from selection.

    emg33-fit yields reduced chi-square of: 0.99 +- 0.14


    ##### RESULT OF AUTOMATIC MODEL SELECTION: #####

        Best fit model determined to be: emg22
        Corresponding chi²-reduced: 0.96 


    ##### Peak-shape determination #####-------------------------------------------------------------------------------------------
    <lmfit.model.ModelResult object at 0x000001E73F3D03A0>




Determine constant of proportionality A_stat_emg for subsequent stat. error estimations (OPTIONAL, feel free to skip this)
--------------------------------------------------------------------------------------------------------------------------
The statistical uncertainties of Hyper-EMG fits are estimated using the
equation:
$\sigma_{stat} = A_{stat,emg} \cdot \frac{\mathrm{FWHM}}{\sqrt{N_{counts}}}$
where $\mathrm{FWHM}$ and $N_{counts}$ refer to the full width at half
maximum and the number of counts in the respective peak. This method will
typically run for ~10 minutes if N_spetra=1000 (default) is used.
If this step is skipped, the default value $A_{stat,emg} = 0.52$ will be used.
Optionally, the plot can be saved by using the `plot_filename` argument.


.. code-block:: default


    # Determine A_stat_emg and save the resulting plot
    spec.determine_A_stat_emg(species='Ca43:F19:-1e',x_range=0.004,N_spectra=10)




.. image:: /auto_examples/images/sphx_glr_example_tutorial_028.png
    :alt: A_stat_emg determination from bootstrapped spectra - emg22 MLE fits
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Creating synthetic spectra by bootstrapped re-sampling and fitting them for A_stat determination.
    Depending on the choice of `N_spectra` this can take a few minutes. Interrupt kernel if this takes too long.

    Done!

    [[Model]]
        Model(powerlaw)
    [[Fit Statistics]]
        # fitting method   = leastsq
        # function evals   = 5
        # data points      = 8
        # variables        = 1
        chi-square         = 1.5599e-08
        reduced chi-square = 2.2284e-09
        Akaike info crit   = -158.444191
        Bayesian info crit = -158.364750
    [[Variables]]
        amplitude:  1.5233e-04 +/- 1.6364e-05 (10.74%) (init = 1)
        exponent:  -0.5 (fixed)
    A_stat of a Gaussian model: 0.425
    Default A_stat_emg for Hyper-EMG models: 0.52
    A_stat_emg for this spectrum's emg22 fit model: 0.635 +- 0.068




Fit all peaks in spectrum, perform mass (re-)calibration, determine
peak-shape uncertainties and update peak properties table with the results
The simultaneous mass recalibration is invoked by specifying the
`species_mass_calib` (or alternatively the `index_mass_calib`) argument


.. code-block:: default


    # Maximum likelihood fit of all peaks in the spectrum
    spec.fit_peaks(species_mass_calib='Ti46:O16:-1e')
    # Alternative: Fit restricted to a user-defined mass range
    #spec.fit_peaks(species_mass_calib='Ti46:O16:-1e',x_fit_cen=61.9455,x_fit_range=0.01)



.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_029.png
          :alt: emg22 MLE fit
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_example_tutorial_030.png
          :alt: emg22 MLE fit
          :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    ##### Mass recalibration #####


    Relative literature error of mass calibrant:    3e-09
    Relative statistical error of mass calibrant:   1.3e-08

    Recalibration factor:      0.999999711 = 1 -2.89e-07
    Relative recalibration error:  1.4e-08 


    ##### Peak-shape uncertainty evaluation #####

    Determining absolute centroid shifts of mass calibrant.

    Re-fitting with  sigma  =  8.3e-05 +/- 3e-06  shifts Δm of peak 0 and mass calibrant by 0.497351 / -0.268 μu. 
    Re-fitting with  sigma  =  8.3e-05 +/- 3e-06  shifts Δm of peak 1 and mass calibrant by 1.419093 / -0.367 μu. 
    Re-fitting with  sigma  =  8.3e-05 +/- 3e-06  shifts Δm of peak 2 and mass calibrant by -0.124142 / 0.337 μu. 
    Re-fitting with  sigma  =  8.3e-05 +/- 3e-06  shifts Δm of peak 3 and mass calibrant by 0.009671 / 0.025 μu. 
    Re-fitting with  sigma  =  8.3e-05 +/- 3e-06  shifts Δm of peak 5 and mass calibrant by -0.013996 / 0.002 μu. 
    Re-fitting with  sigma  =  8.3e-05 +/- 3e-06  shifts Δm of peak 6 and mass calibrant by 0.011674 / -0.032 μu. 
    Re-fitting with  sigma  =  8.3e-05 +/- 3e-06  shifts Δm of peak 7 and mass calibrant by -1.469082 / 0.908 μu. 

    Re-fitting with  theta  =  0.724714 +/- 0.023491  shifts Δm of peak 0 and mass calibrant by -1.447898 / 1.312 μu. 
    Re-fitting with  theta  =  0.724714 +/- 0.023491  shifts Δm of peak 1 and mass calibrant by -1.406531 / 1.334 μu. 
    Re-fitting with  theta  =  0.724714 +/- 0.023491  shifts Δm of peak 2 and mass calibrant by -0.434005 / 0.331 μu. 
    Re-fitting with  theta  =  0.724714 +/- 0.023491  shifts Δm of peak 3 and mass calibrant by -0.254026 / 0.14 μu. 
    Re-fitting with  theta  =  0.724714 +/- 0.023491  shifts Δm of peak 5 and mass calibrant by -0.269747 / 0.237 μu. 
    Re-fitting with  theta  =  0.724714 +/- 0.023491  shifts Δm of peak 6 and mass calibrant by -0.090195 / 0.031 μu. 
    Re-fitting with  theta  =  0.724714 +/- 0.023491  shifts Δm of peak 7 and mass calibrant by -1.874811 / 1.815 μu. 

    Re-fitting with  eta_m1  =  0.920191 +/- 0.023131  shifts Δm of peak 0 and mass calibrant by 0.789385 / -0.741 μu. 
    Re-fitting with  eta_m1  =  0.920191 +/- 0.023131  shifts Δm of peak 1 and mass calibrant by 0.461702 / -0.356 μu. 
    Re-fitting with  eta_m1  =  0.920191 +/- 0.023131  shifts Δm of peak 2 and mass calibrant by 0.451256 / -0.44 μu. 
    Re-fitting with  eta_m1  =  0.920191 +/- 0.023131  shifts Δm of peak 3 and mass calibrant by 0.404974 / -0.411 μu. 
    Re-fitting with  eta_m1  =  0.920191 +/- 0.023131  shifts Δm of peak 5 and mass calibrant by 0.307791 / -0.254 μu. 
    Re-fitting with  eta_m1  =  0.920191 +/- 0.023131  shifts Δm of peak 6 and mass calibrant by 0.014109 / -0.104 μu. 
    Re-fitting with  eta_m1  =  0.920191 +/- 0.023131  shifts Δm of peak 7 and mass calibrant by 0.687912 / -0.701 μu. 

    Re-fitting with  tau_m1  =  4.5e-05 +/- 6e-06  shifts Δm of peak 0 and mass calibrant by -0.57012 / 0.613 μu. 
    Re-fitting with  tau_m1  =  4.5e-05 +/- 6e-06  shifts Δm of peak 1 and mass calibrant by 0.842105 / 0.017 μu. 
    Re-fitting with  tau_m1  =  4.5e-05 +/- 6e-06  shifts Δm of peak 2 and mass calibrant by -0.310446 / 0.48 μu. 
    Re-fitting with  tau_m1  =  4.5e-05 +/- 6e-06  shifts Δm of peak 3 and mass calibrant by -0.122479 / 0.103 μu. 
    Re-fitting with  tau_m1  =  4.5e-05 +/- 6e-06  shifts Δm of peak 5 and mass calibrant by -0.211683 / 0.112 μu. 
    Re-fitting with  tau_m1  =  4.5e-05 +/- 6e-06  shifts Δm of peak 6 and mass calibrant by -0.031522 / -0.011 μu. 
    Re-fitting with  tau_m1  =  4.5e-05 +/- 6e-06  shifts Δm of peak 7 and mass calibrant by -2.051381 / 1.601 μu. 

    Re-fitting with  tau_m2  =  0.000177 +/- 2.5e-05  shifts Δm of peak 0 and mass calibrant by -0.29217 / 0.398 μu. 
    Re-fitting with  tau_m2  =  0.000177 +/- 2.5e-05  shifts Δm of peak 1 and mass calibrant by -0.191015 / 0.302 μu. 
    Re-fitting with  tau_m2  =  0.000177 +/- 2.5e-05  shifts Δm of peak 2 and mass calibrant by -0.207078 / 0.247 μu. 
    Re-fitting with  tau_m2  =  0.000177 +/- 2.5e-05  shifts Δm of peak 3 and mass calibrant by -0.198827 / 0.265 μu. 
    Re-fitting with  tau_m2  =  0.000177 +/- 2.5e-05  shifts Δm of peak 5 and mass calibrant by -0.079731 / 0.189 μu. 
    Re-fitting with  tau_m2  =  0.000177 +/- 2.5e-05  shifts Δm of peak 6 and mass calibrant by -0.040278 / 0.025 μu. 
    Re-fitting with  tau_m2  =  0.000177 +/- 2.5e-05  shifts Δm of peak 7 and mass calibrant by 0.104563 / 0.023 μu. 

    Re-fitting with  eta_p1  =  0.790092 +/- 0.044934  shifts Δm of peak 0 and mass calibrant by -0.257875 / 0.127 μu. 
    Re-fitting with  eta_p1  =  0.790092 +/- 0.044934  shifts Δm of peak 1 and mass calibrant by -0.308044 / 0.236 μu. 
    Re-fitting with  eta_p1  =  0.790092 +/- 0.044934  shifts Δm of peak 2 and mass calibrant by -0.266946 / 0.17 μu. 
    Re-fitting with  eta_p1  =  0.790092 +/- 0.044934  shifts Δm of peak 3 and mass calibrant by -0.253966 / 0.15 μu. 
    Re-fitting with  eta_p1  =  0.790092 +/- 0.044934  shifts Δm of peak 5 and mass calibrant by -0.234102 / 0.188 μu. 
    Re-fitting with  eta_p1  =  0.790092 +/- 0.044934  shifts Δm of peak 6 and mass calibrant by 0.005052 / -0.022 μu. 
    Re-fitting with  eta_p1  =  0.790092 +/- 0.044934  shifts Δm of peak 7 and mass calibrant by -3.950556 / 3.447 μu. 

    Re-fitting with  tau_p1  =  0.000124 +/- 1.3e-05  shifts Δm of peak 0 and mass calibrant by 0.59994 / -0.879 μu. 
    Re-fitting with  tau_p1  =  0.000124 +/- 1.3e-05  shifts Δm of peak 1 and mass calibrant by 0.62328 / -0.843 μu. 
    Re-fitting with  tau_p1  =  0.000124 +/- 1.3e-05  shifts Δm of peak 2 and mass calibrant by 0.381974 / -0.49 μu. 
    Re-fitting with  tau_p1  =  0.000124 +/- 1.3e-05  shifts Δm of peak 3 and mass calibrant by 0.214752 / -0.268 μu. 
    Re-fitting with  tau_p1  =  0.000124 +/- 1.3e-05  shifts Δm of peak 5 and mass calibrant by 0.04681 / -0.092 μu. 
    Re-fitting with  tau_p1  =  0.000124 +/- 1.3e-05  shifts Δm of peak 6 and mass calibrant by -0.039736 / -0.043 μu. 
    Re-fitting with  tau_p1  =  0.000124 +/- 1.3e-05  shifts Δm of peak 7 and mass calibrant by 0.724095 / -0.84 μu. 

    Re-fitting with  tau_p2  =  0.000408 +/- 5.5e-05  shifts Δm of peak 0 and mass calibrant by -0.047891 / -0.028 μu. 
    Re-fitting with  tau_p2  =  0.000408 +/- 5.5e-05  shifts Δm of peak 1 and mass calibrant by 0.075922 / -0.105 μu. 
    Re-fitting with  tau_p2  =  0.000408 +/- 5.5e-05  shifts Δm of peak 2 and mass calibrant by 0.019801 / -0.089 μu. 
    Re-fitting with  tau_p2  =  0.000408 +/- 5.5e-05  shifts Δm of peak 3 and mass calibrant by 0.053686 / -0.141 μu. 
    Re-fitting with  tau_p2  =  0.000408 +/- 5.5e-05  shifts Δm of peak 5 and mass calibrant by 0.39755 / -0.396 μu. 
    Re-fitting with  tau_p2  =  0.000408 +/- 5.5e-05  shifts Δm of peak 6 and mass calibrant by -0.049007 / 0.027 μu. 
    Re-fitting with  tau_p2  =  0.000408 +/- 5.5e-05  shifts Δm of peak 7 and mass calibrant by 5.770581 / -7.819 μu. 

    Relative peak-shape error of peak 0: 3.4e-08
    Relative peak-shape error of peak 1: 3.9e-08
    Relative peak-shape error of peak 2: 1.7e-08
    Relative peak-shape error of peak 3: 1.1e-08
    Relative peak-shape error of peak 5: 1.1e-08
    Relative peak-shape error of peak 6: 3e-09
    Relative peak-shape error of peak 7: 1.51e-07
           x_pos       species          comment      m_AME   m_AME_error  extrapolated fit_model cost_func  ...  rel_stat_error  rel_recal_error  rel_peakshape_error  rel_mass_error     A  atomic_ME_keV  mass_error_keV  m_dev_keV
    0  61.927800      Ni62:-1e                -  61.927796  4.700000e-07         False     emg22       MLE  ...    4.201252e-07     1.364463e-08         3.363408e-08    4.216902e-07  62.0     -66740.227       24.325359      6.103
    1  61.932021      Cu62:-1e                -  61.932046  6.940000e-07         False     emg22       MLE  ...    4.903586e-07     1.364463e-08         3.896354e-08    4.920934e-07  62.0     -62753.555       28.388556     33.878
    2  61.934369             ?     Non-isobaric        NaN           NaN         False     emg22       MLE  ...    3.948739e-08     1.364463e-08         1.705503e-08    4.512542e-08   NaN            NaN        2.603353        NaN
    3  61.943618      Ga62:-1e                -  61.943641  6.940000e-07         False     emg22       MLE  ...    8.024891e-08     1.364463e-08         1.113415e-08    8.215858e-08  62.0     -51992.089        4.740561     -5.177
    4  61.946994  Ti46:O16:-1e   mass calibrant  61.946993  1.760001e-07         False     emg22       MLE  ...    1.334555e-08     1.364463e-08                  NaN             NaN  62.0     -48864.806             NaN      0.000
    5  61.949527  Sc46:O16:-1e                -  61.949534  7.320000e-07         False     emg22       MLE  ...    6.260639e-08     1.364463e-08         1.106218e-08    6.502390e-08  62.0     -46492.523        3.752245      5.702
    6  61.956611  Ca43:F19:-1e  shape calibrant  61.956621  2.440018e-07         False     emg22       MLE  ...    1.527343e-08     1.364463e-08         2.667518e-09    2.065355e-08  62.0     -39895.649        1.191962      0.623
    7  61.958997             ?                -        NaN           NaN         False     emg22       MLE  ...    5.232160e-07     1.364463e-08         1.512689e-07    5.448150e-07   NaN            NaN       31.443685        NaN

    [8 rows x 20 columns]
    <lmfit.model.ModelResult object at 0x000001E74F70BAC0>





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 1 minutes  5.372 seconds)


.. _sphx_glr_download_auto_examples_example_tutorial.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: example_tutorial.py <example_tutorial.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: example_tutorial.ipynb <example_tutorial.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
