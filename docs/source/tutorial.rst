Tutorial
========

The following tutorial file will give you an introduction to fitting a mass
spectrum with `emgfit`:
:ref:`Jupyter notebook with tutorial <C:/Users/Stefan/Dropbox/Beam time analysis/Hyper EMG fitting/emgfit/examples/tutorial/emgfit_tutorial.ipynb>`

At first emgfit has to be imported
.. jupyter-execute::
  
    import emgfit as emg
    ### Import mass data, plot full spectrum and choose fit range
    filename = "2019-09-13_004-_006 SUMMED High stats 62Ga" # input file (as exported with MAc's hist-mode)
    skiprows = 38 # number of header rows to skip upon data import
    m_start = 61.9243 # low-mass cut off
    m_stop = 61.962 # high-mass cut off
    spec = emg.spectrum(filename+'.txt',m_start,m_stop,skiprows=skiprows)


Get method docs by:

Get attribute docs by:

Central concepts:
peaks
spectrum class
peaks list
peak identification

Include example spectrum in package (also for doctests).
