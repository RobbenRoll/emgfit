import pytest
import emgfit as emg
import numpy as np

class Test_get_AME_values(object):

    def test_isomer_AME_values(self):
        prec = 8 # number of decimals up to which agreement is demanded
        # Check calculation of ionic mass of second isomer of In-127
        species = '1In127m1:-1e'
        # g.s. mass and m_e from AME2016, Ex from ENSDF
        m_ref = 126.917448546 - 0.00054857990907 + 0.00200001266
        m_ref_error = np.sqrt((22.713e-06)**2 + (62.2655579e-06)**2)
        extrapol_ref = False
        A_ref = 127
        from emgfit.ame_funcs import get_AME_values
        m_AME, m_AME_error, extrapol, A  = get_AME_values(species,
                                                          Ex=1863,
                                                          Ex_error=58)
        msg0 = "`m_AME` of In127m1:-1e deviates from reference"
        assert np.round(m_AME, prec) == np.round(m_ref,prec), msg0
        msg1 = "`m_AME_error` of In127m1:-1e deviates from reference"
        assert np.round(m_AME_error, prec) == np.round(m_ref_error,prec), msg1
        msg2 = "wrong extrapol. flag for In127m1:-1e"
        assert extrapol == extrapol_ref, msg2
        msg3 = "atomic mass number of In127m1:-1e deviates from reference "
        assert A == A_ref, msg3
