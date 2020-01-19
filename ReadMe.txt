
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
* Fit calibrant peak to obtain scaling factor
* Scale parameters
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


TO DO:
* add peak marker labels to plots?
            




