"""
Tutorial
========

"""
import emgfit as emg
### Import mass data, plot full spectrum and choose fit range
filename = "2019-09-13_004-_006 SUMMED High stats 62Ga" # input file (as exported with MAc's hist-mode)
skiprows = 38 # number of header rows to skip upon data import
m_start = 61.9243 # low-mass cut off
m_stop = 61.962 # high-mass cut off
spec = emg.spectrum(filename+'.txt',m_start,m_stop,skiprows=skiprows)
