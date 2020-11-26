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
        [Element, Z, A, atomic AME mass, atomic AME mass error, boolean flag for extrapolated AME mass]

    """
    Z = df_AME['Z'].loc[(A,El)]
    m_AME = df_AME['ATOMIC MASS [µu]'].loc[(A,El)]*1e-06
    m_AME_error = df_AME['Error ATOMIC MASS [µu]'].loc[(A,El)]*1e-06
    extrapolated_yn = df_AME['Extrapolated?'].loc[(A,El)]
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
    ``splitspecies('4H1:1C12')`` returns ``['4H1','1C12']``
    ``splitspecies('H1:O16')`` returns ``['H1','O16']``

    """
    return s.split(':')


def splitparticle(s):
    """Extracts number, particle/element type and mass number of particle string
    (e.g. 1Cs133, Cs133, 1e).

    """
    if s[-1:] == '?': # handle unidentified species (indicated by '?' at end of string)
        return None, '?', None
    tail = s.lstrip('+-0123456789')
    head = s[:-len(tail)]
    if head == '+' or head == '': # handle missing number (if '+' given or 1 in front of symbol omitted)
        n = int(1)
    elif head == '-': # handle missing number
        n = int(-1)
    else:
        n = int(head) # leading number including sign (if present)
    El = tail.rstrip('0123456789') # get central letters
    if El == 'e' and len(El) == len(tail): # handle electron strings, e.g. ':-1e'
        A = 0
    else: # handle hadrons
        A = int(tail[len(El):]) # trailing number
    return n, El, A


def get_AME_values(species):
    """Calculates the AME mass, AME mass error, the extrapolation flag and the
    mass number A of the given species.

    Parameters
    ----------
    species : str
        string with name of species to grab AME for

    Returns
    -------
    list of [float,float,bool,int]
        List containing relevant AME data for `species`:
        [AME mass, AME mass uncertainty, ``True`` if extrapolated species, atomic mass number]

    """
    m = 0.0
    m_error_sq = 0.0
    A_tot = 0
    extrapol = False # initialize boolean flag as False
    for ptype in splitspecies(species):
        n, El, A = splitparticle(ptype)
        if ptype[-1:] == '?': # unidentified species
            m = None
            m_error = None
            A_tot = None
        elif El == 'e': # electron
            m += n*m_e
            # neglect uncertainty of m_e
        else: # regular atom
            m += n*mdata_AME(El,A)[3]
            m_error_sq += (n*mdata_AME(El,A)[4])**2
            m_error = np.sqrt(m_error_sq)
            A_tot += A
            if  mdata_AME(El,A)[5]:
                extrapol = True # boolean flag for any extrapolated masses contained in species
    return m, m_error, extrapol, A_tot
