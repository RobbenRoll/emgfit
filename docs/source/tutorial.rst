Tutorial
========

The following tutorial file will give you an introduction to fitting a mass
spectrum with `emgfit`:


At first emgfit has to be imported

.. .. jupyter-execute:: ../../examples/data_import/data_import.py

.. jupyter-execute::

  import emgfit as emg
  ### Import mass data, plot full spectrum and choose fit range
  filename = "2019-09-13_004-_006 SUMMED High stats 62Ga" # input file (as exported with MAc's hist-mode)
  skiprows = 38 # number of header rows to skip upon data import
  m_start = 61.9243 # low-mass cut off
  m_stop = 61.962 # high-mass cut off
  spec = emg.spectrum(filename+'.txt',m_start,m_stop,skiprows=skiprows)

heading

.. jupyter-execute::

  ### Detect peaks and add them to spectrum object 'spec'
  spec.detect_peaks() # automatic peak detection
  #spec.add_peak(61.925,species='?') # manually add a peak at x_pos = 61.925u
  #spec.remove_peak(peak_index=0) # manually remove the peak with index 0

heading

.. jupyter-execute::

  spec.assign_species(['Ni62:-1e','Cu62:-1e',None,'Ga62:-1e','Ti46:O16:-1e','Sc46:O16:-1e','Ca43:F19:-1e',None])
  spec.add_peak_comment('Non-isobaric',peak_index=2)
  spec.show_peak_properties() # check the changes by printing the peak properties table

..

  heading

  .. jupyter-execute::

    #spec.determine_peak_shape(species_shape_calib='Ca43:F19:-1e') # default settings and automatic model selection
    spec.determine_peak_shape(species_shape_calib='Ca43:F19:-1e',x_fit_range=0.0045) # user-defined fit range
    #spec.determine_peak_shape(species_shape_calib='Ca43:F19:-1e',fit_model='emg12',vary_tail_order=False) # user-defined model

  heading

  .. jupyter-execute::

    # Maximum likelihood fit of all peaks in the spectrum
    spec.fit_peaks(species_mass_calib='Ti46:O16:-1e')
    # Alternative: Fit restricted to a user-defined mass range
    #spec.fit_peaks(species_mass_calib='Ti46:O16:-1e',x_fit_cen=61.9455,x_fit_range=0.01)

Get method docs by:

Get attribute docs by:

Central concepts:
peaks
spectrum class
peaks list
peak identification

Include example spectrum in package (also for doctests).
