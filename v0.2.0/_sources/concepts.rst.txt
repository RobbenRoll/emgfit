Central concepts
================

Only singly-charged isobaric ions are currently supported by `emgfit`!

The peak and spectrum classes
-----------------------------

Importing data and creating spectrum objects
--------------------------------------------


Adding peak objects to a spectrum
---------------------------------

Assigning species to peaks and fetching AME values
--------------------------------------------------

Available fit models
--------------------

Peak fitting
------------


Peak-shape calibration
----------------------

Mass re-calibration
-------------------

Fitting peaks of interest
-------------------------

Multi-peak fits

Fits of regions of interest




Peak-shape uncertainty evaluation
---------------------------------

#One peculiarity of this method is that only the centroid shifts of the     #TODO: move this section to the peak-shape error evaluation article
peaks-of-interest relative to the (shifted) centroid of the mass
calibrant are taken into account for the peak-shape error evaluation.
This is because the mass re-calibration ensures that only relative
centroid shifts with respect to the calibrant enter the final mass
values. If varying the shape parameters shifts the peaks-of-interest and
the calibrant peak by the same amount, the final mass value is not
altered. Despite the uncertainty of the peak-shape parameters the peak
shape of isobaric peaks can be assumed to be identical. The mass
dependence of shape parameters is negligible for isobaric species. The
above argument relies on the condition that a decent time-resolved
calibration (TRC) with sufficient calibrant statistics per block has
been performed (otherwise, the IOI peaks can be broadened w.r.t. the
calibrant). Hence, the peaks-of-interest and the calibrant peak should
both be re-fitted.

:-Notation of chemical substances
---------------------------------

Just like the MR-TOF-MS data acquisition software MAc, `emgfit` follows the
:-notation of chemical compounds. The chemical composition of an ion is denoted
as a single string in which the constituting isotopes are separated by a colon
(``:``). Each isotope is denoted as ``'n(El)A'`` where `El` is the corresponding
element symbol, `n` denotes the number of atoms of the isotope `El` and A is the
respective atomic mass number. The charge state of the ion is indicated by
substracting the desired number of electrons from the atomic species (i.e.
singly-charged = ``'':-1e'``, doubly-charged = ``'':-2e'`` etc.). The
substraction of the electron is important since otherwise the atomic instead
of the ionic mass is used for subsequent calculations. Mind that `emgfit`
currently only supports singly-charged ions.
