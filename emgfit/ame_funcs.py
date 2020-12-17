###################################################################################################
##### Module for importing and handling of data from the Atomic Mass Evaluation
##### (AME) for emgfit package
##### Author: Stefan Paul

from .config import *
from pathlib import Path
import pandas as pd
import numpy as np

##### Import AME data into pandas dataframe
directory = Path(__file__).parent  # get directory containing this file
filename = str(directory)+"/AME2016/AME2016_formatted.csv"
df_AME = pd.read_csv(filename, encoding = 'unicode_escape')
df_AME.set_index(['A','Element'],inplace=True)

##### Define functions
def mdata_AME(El,A):
    """Grabs atomic mass data from AME2016 [u].

    Parameters
    ----------
    El : str
        String with element symbol.
    A : int
        Mass number of isotope of interest.

    Returns
    -------
    list (str,int,float,float,bool)
        [Element, Z, A, atomic AME mass, atomic AME mass error, boolean flag for
        extrapolated AME mass]

    """
    try:
        Z = df_AME['Z'].loc[(A,El)]
        m_AME = df_AME['ATOMIC MASS [µu]'].loc[(A,El)]*1e-06
        m_AME_error = df_AME['Error ATOMIC MASS [µu]'].loc[(A,El)]*1e-06
        extrapolated_yn = df_AME['Extrapolated?'].loc[(A,El)]
    except KeyError:
        msg = "No AME2016 values found for {0}-{1}.".format(El,A)
        raise Exception(msg) from None

    return [El, Z, A, m_AME, m_AME_error, extrapolated_yn]


def get_El_from_Z(Z):
    """Convenience function to grab element symbol from given proton number.

    Parameters
    ----------
    Z : int or array-like of int
        Proton number.

    Returns
    -------
    str
        Symbol of element with specified proton number.
    """
    if isinstance(Z, (list, tuple, np.ndarray)):
        li = [df_AME.index[df_AME['Z'] == Z_i].get_level_values('Element')[0] for Z_i in Z]
        El = np.array(li)
    else:
        El = df_AME.index[df_AME['Z'] == Z].get_level_values('Element')[0]
    return El


def splitspecies(s):
    """Splits ion species string into list of constituent atom strings.

    Parameters
    ----------
    s : str
        Input species string.

    Returns
    -------
    list of str
        List of constituent atom strings contained in `s`.

    Examples
    -------
    >>> splitspecies('4H1:1C12')  # returns ``['4H1','1C12']``
    >>> splitspecies('H1:O16')    # returns ``['H1','O16']``

    """
    return s.split(':')


def splitparticle(s):
    """Extracts number, particle/element type and mass number of particle string
    (e.g. 1Cs133, Cs133, 1e).

    Parameters
    ----------
    s : str
        Input particle string.

    Returns
    -------
    tuple of (int, str, int)
        Tuple of signature (number of particles, element symbol, atomic mass number)

    Examples
    --------
    >>> splitspecies('1Cs133') # returns (1,'Cs',133)
    >>> splitspecies('-e')     # returns (-1,'e',0)


    """
    tail = s.lstrip('+-0123456789')
    head = s[:-len(tail)]
    if head == '+' or head == '': # handle omitted 1 or plus sign
        n = int(1)
    elif head == '-': # handle omitted 1
        n = int(-1)
    else:
        n = int(head) # leading number including sign (if present)
    El = tail.rstrip('0123456789') # get central letters
    if El == 'e' and len(El) == len(tail): # handle electron strings, e.g. ':-1e'
        A = 0
    else: # handle hadrons
        A = int(tail[len(El):]) # trailing number
    return n, El, A


def get_AME_values(species, Ex=0.0, Ex_error=0.0):
    """Calculates the AME mass, AME mass error, the extrapolation flag and the
    mass number A of the given atomic or molecular species.

    Parameters
    ----------
    species : str
        String with name of species to grab AME values for.
    Ex : float [keV], optional, default: 0.0
        Isomer excitation energy in keV to add to ground-state literature
        mass.
    Ex_error : float [keV], optional, default: 0.0
        Uncertainty of isomer excitation energy in keV to add in quadrature
        to ground-state literature mass uncertainty.

    Notes
    -----
    `species` strings follow the :ref:`:-notation`.

    Atoms with the nucleus in an isomeric state are flagged by appending an 'm'
    or any of 'm0', 'm1', ..., 'm9' to the atom's substring. For species with
    isomers the literature mass values are only returned if the excitation
    energy is user-specified with the `Ex` argument. In this case, `Ex` is
    added to the AME value for the ground state mass and `Ex_error` is added in
    quadrature to the respective AME uncertainty.

    Returns
    -------
    tuple of (float,float,bool,int)
        List containing relevant AME data for `species`:
        (AME mass, AME mass uncertainty, ``True`` if extrapolated species, atomic mass number)

    """
    m = 0.0
    m_error_sq = 0.0
    A_tot = 0
    extrapol = False # initialize boolean flag as False
    isomer_flags = ('m','m0','m1','m2','m3','m4','m5','m6','m7','m8','m9')
    isomer_count = 0 # number of isomeric particle types found
    for ptype in splitspecies(species): # loop over particle/atom types

        # Abort when encountering unidentified species
        if ptype == '?':
            m, m_error, extrapol, A_tot = None, None, False, None
            break # still check warnings below

        # Remove trailing '?' (flag for tentative species IDs)
        ptype = ptype.rstrip('? ')

        # Handle isomers (flagged by 'm' or 'm0' to 'm9' at the end)
        if ptype.endswith(isomer_flags):
            if Ex == 0.0: # abort if no `Ex` is given
                from warnings import warn
                msg = str("{} is labelled as isomer but no excitation energy "
                          "was specified with the `Ex` argument."
                          ).format(species)
                warn(msg)
                return None, None, False, None
            if Ex_error == 0.0:
                from warnings import warn
                msg = str("Uncertainty of isomer excitation energy `Ex_error` "
                          "unspecified for {}.").format(ptype)
                warn(msg)
            isomer_count += 1
            ptype = ptype.rstrip('0123456789')
            ptype = ptype.rstrip('m')

        # Update lit. values
        n, El, A = splitparticle(ptype)
        if El == 'e' and m is not None: # electron
            m += n*m_e
            # neglect uncertainty of m_e
        elif isomer_count == 1:
            m += n*(mdata_AME(El,A)[3] + Ex/u_to_keV)
            m_error_sq += (n*mdata_AME(El,A)[4])**2 + (n*Ex_error/u_to_keV)**2
            m_error = np.sqrt(m_error_sq)
            A_tot += n*A
            if mdata_AME(El,A)[5]:
                extrapol = True # boolean flag for any extrapolated masses contained in species
        else: # regular atom
            m += n*(mdata_AME(El,A)[3])
            m_error_sq += (n*mdata_AME(El,A)[4])**2
            m_error = np.sqrt(m_error_sq)
            A_tot += n*A
            if mdata_AME(El,A)[5]:
                extrapol = True # boolean flag for any extrapolated masses contained in species

    # Issue warnings for isomers if needed
    if Ex != 0.0 and isomer_count == 0: # no isomer flag but Ex given
        from warnings import warn
        msg = str("The specified isomer excitation energy `Ex` and its "
                  "error `Ex_error` are ignored since the species label "
                  "'{}' does not contain any isomer markers.").format(species)
        warn(msg)
    elif isomer_count > 1:
        from warnings import warn
        msg = str("{} contains multiple isomeric constituents. This method "
                  "only supports ions with a single isotopic species in an "
                  "isomeric state.").format(species)
        warn(msg)

    return m, m_error, extrapol, A_tot
