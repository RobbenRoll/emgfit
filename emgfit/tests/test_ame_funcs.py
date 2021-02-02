import pytest
import emgfit as emg
import numpy as np

class Test_get_AME_values:
    decimals = 8 # number of decimals up to which absolute agreement is demanded
    atol = 10**(-decimals)
    rtol = 0 # relative tolerance up to which agreement is demanded
    m_e_ref = 0.00054857990907

    def _isclose(self,a,b, atol=None, rtol=None):
        """Custom version of np.isclose. """
        abs_tol = atol or self.atol
        rel_tol = rtol or self.rtol
        return np.isclose(a, b, atol=abs_tol, rtol=rel_tol)


    def test_molecular_AME_values(self):
        """Check calculation of ionic mass of H2O (incl. tentative ID flag)"""
        species = '2H1:1O16?:-1e'
        # atomic masses and m_e from AME2016 [u], neglect electron binding
        # energy:
        m_H1_ref = 1.00782503224
        m_O16_ref = 15.99491461960
        m_ref = 2*m_H1_ref + m_O16_ref - self.m_e_ref
        m_ref_error = np.sqrt((2*0.00009e-06)**2 + (0.00017e-06)**2)
        extrapol_ref = False
        A_ref = 18
        from emgfit.ame_funcs import get_AME_values
        m_AME, m_AME_error, extrapol, A  = get_AME_values(species)
        msg0 = "`m_AME` of {} deviates from reference".format(species)
        assert self._isclose(m_AME, m_ref), msg0
        msg1 = "`m_AME_error` of {} deviates from reference".format(species)
        assert self._isclose(m_AME_error, m_ref_error), msg1
        msg2 = "wrong extrapol. flag for {}".format(species)
        assert extrapol == extrapol_ref, msg2
        msg3 = "atomic mass number of {} deviates from reference".format(species)
        assert A == A_ref, msg3


    def test_isomer_AME_values(self):
        """Check calculation of ionic mass of second isomer of In-127 """
        species = '1In127m1:-1e'
        # g.s. mass and m_e from AME2016 [u], Ex from ENSDF [u]:
        m_ref = 126.917448546 - self.m_e_ref  + 0.00200001266
        m_ref_error = np.sqrt((22.713e-06)**2 + (62.2655579e-06)**2)
        extrapol_ref = False
        A_ref = 127
        from emgfit.ame_funcs import get_AME_values
        m_AME, m_AME_error, extrapol, A  = get_AME_values(species,
                                                          Ex=1863,
                                                          Ex_error=58)
        msg0 = "`m_AME` of {} deviates from reference".format(species)
        assert self._isclose(m_AME, m_ref), msg0
        msg1 = "`m_AME_error` of {} deviates from reference".format(species)
        assert self._isclose(m_AME_error, m_ref_error), msg1
        msg2 = "wrong extrapol. flag for {}".format(species)
        assert extrapol == extrapol_ref, msg2
        msg3 = "atomic mass number of {} deviates from reference".format(species)
        assert A == A_ref, msg3


    def test_doubly_charged_AME_values(self):
        """Check calculation of ionic mass of doubly charged Sn-116 """
        species = 'Sn116:-2e'
        # atomic mass and m_e from AME2016 [u]:
        m_ref = 115.901742824 - 2*self.m_e_ref
        m_ref_error = 1.03e-07
        extrapol_ref = False
        A_ref = 116
        z_ref = 2

        from emgfit.ame_funcs import get_AME_values, get_charge_state
        m_AME, m_AME_error, extrapol, A  = get_AME_values(species)
        z = get_charge_state(species)

        msg0 = "`m_AME` of {} deviates from reference".format(species)
        assert self._isclose(m_AME, m_ref), msg0
        msg1 = "`m_AME_error` of {} deviates from reference".format(species)
        assert self._isclose(m_AME_error, m_ref_error), msg1
        msg2 = "wrong extrapol. flag for {}".format(species)
        assert extrapol == extrapol_ref, msg2
        msg3 = "atomic mass number of {} deviates from reference".format(species)
        assert A == A_ref, msg3
        msg4 = "result of get_charge_state() deviates from reference"
        assert z == z_ref, msg4


    def test_get_El_from_Z(self):
        """Check grabbing of element symbol from proton number """
        proton_numbers = [0,1,6,67,118]
        El_ref = ['n','H','C','Ho','Ei']
        from emgfit.ame_funcs import get_El_from_Z
        for i, Z in enumerate(proton_numbers):
            El = get_El_from_Z(Z)
            msg = str("result of get_El_from_Z({}) deviates from "
                      "reference").format(Z)
            assert El==El_ref[i], msg
