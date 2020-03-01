* Uncertainty of each bin: 1/sqrt(N+1)
* No-rescaling of covariance matrix
* uncertainty of emg peak centroid obtained from sigma via standard error of mean plus add. factor: 1.25?? * sigma/sqrt(N) with N counts in peak 
* m_AME is the AME mass of the entire species (to get the ionic instead of the atomic mass ':-1e' has to be added to the species name!)
* re-scaling of peak-shape parameters (sigma & tau's) can be neglected for isobaric species since the rel. mass differences (< 0.1%) are much smaller than the rel. shape parameter errors (~ 1-10 %)
* final mass values of IOIs recalibrated via multiplication with calibration factor 'cal_fac' obtained from fit of mass calibrant
* 'centroid' refers to centroid of underlying Gaussian (instead of true centroid of Hyper-EMG) 



Things to control by user:
* Input file name
* Fit range
* Fit function
* Peak detection parameters (smoothing_window_len, threshold, peak width)
* Peak for peak shape determination (fit range for shape determination)
* Peak shape parameters?
* Tail orders pre-defined or optimized?
* Species in spectrum
* Calibrant peak

Options: 
* Save and load peak shape
* Save fit results (save fit result dataframe as excel file)
* Save output plots (Individually or all?)

Flow chart:

* Plot full spectrum
* Cut data to specified fit range
* Detect all peaks and plot result	(-> create peak objects)
* Query identification of peaks  
* Query peak numbers for peak shape calibrant and mass calibrant
* Determine optimal tail order (minimize chi_sq_red)
* Determine peak shape and peak shape parameters
* Determine peak shape uncertainty (vary all parameters within 1-sigma and determine shift of mass centroid)  
* Show peak shape fit results
* Fit calibrant peak to obtain calibration factor
* Scale peak shape paramters (not necessary for isobaric species)
* Fit full spectrum
* Show fit output and save results: Plot full fit curve and zooms of regions of interest, compile fit results  




class peak
attributes: peak parameters

class spectrum
attributes: data - mass spectrum data stored in pandas dataframe
            peaks - list of all peak objects asociated with spectrum
            shape_calib_peak - index of shape calibration peak
            init_peak_shape_pars - initial values of peak shape parameters
            shape_calib_fit_results - 
            peak_shape_pars - peak shape parameters obtained from peak calibration
            mass_calib_peak - index of mass calibration peak
            mass_calib_fit_results - 
            scl_factor - obtained from mass calibration

General fitting routine:
Parameters: df_to_fit= None, x_fit_cen = None, x_fit_range = 0.005, model = emg22, init_pars = None, vary_shape_pars = False, scl_fac = 1, x_pos = [] 


* set min bound of all tau parameters to 1e-12 (instead of 0) to avoid division by zero errors, especially during peak shape uncertainty estimation

TO DO:
* add peak marker labels/indeces to plots!
* add warning for eta's not summing to 1 to make_model_ ?            
* add more fit functions 
* implement tail order determination
* ADD peakshape uncertainty to calibrant fit and other fits, add uncertainty of scaling factor
* implement save model result feature
* enable peak shape determination with multiple peaks? (need to constrain parameters of neighboring peaks to shape calibrant parameters)
* fix 'uncertainties could not be estimated, PAR at initial value' error (usually circumvented by chosing other initial parameters)
* test peak shape uncertainty estimation (do eta's get changed as expected?)

* increase ftol and xtol of fit algorithm?? - seems irrelevant 
* Evaluate recalibration error??

* Obtain peakshape errors by simply re-scaling peakshape error of shape calibrant peak to other peak masses?


* x + y + z = 1
0 <= y <= 1
y = delta - x
delta <= 1

p2 = delta - p1
p1 + p2 + p3 = 1
-> delta = 1 - p3


p2 = p1 - delta
p1 + p2 + p3 = 1
-> 2*p1 -delta + p3 = 1
-> delta = 2*p1 + p3 - 1
