
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
* Save fit results 
* Save output plots

Flow chart:

* Plot full spectrum
* Cut data to specified fit range
* Detect all peaks and plot result	(-> create peak objects)
* Query identification of peaks and peak numbers for peak shape calibrant and mass calibrant
* Determine peak shape and peak shape parameters
* Determine optimal tail order
* Determine peak shape uncertainty  
* Show peak shape fit results
* Fit calibrant peak to obtain scaling factor
* Scale parameters
* Fit full spectrum
* Show fit output and save results: Plot full fit curve and zooms of regions of interest, compile fit results  
