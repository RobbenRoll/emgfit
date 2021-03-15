import pytest
import emgfit as emg
import numpy as np

class Test_get_AME_values:
    """Tests of emgfit literature values against reference values

    Electron binding energies and the uncertainty of m_e are neglected!

    """
    decimals = 8 # number of decimals up to which absolute agreement is demanded
    atol = 10**(-decimals)
    rtol = 0 # relative tolerance up to which agreement is demanded
    m_e_ref = 0.000548579909065 # CODATA value from physics.nist.gov
    u_to_keV_ref = 931494.10242  # CODATA value from physics.nist.gov
    lit_src = 'AME2016'

    def _isclose(self, a ,b, atol=None, rtol=None):
        """Custom version of np.isclose. """
        abs_tol = atol or self.atol
        rel_tol = rtol or self.rtol
        return np.isclose(a, b, atol=abs_tol, rtol=rel_tol)


    def _check_lit_values(self, species, m_ref, m_ref_error, extrapol_ref, A_ref,
                          z_ref=1, Ex=0.0, Ex_error=0.0):
        """Helper method for checking agreement with reference values """
        from emgfit.ame_funcs import get_AME_values, get_charge_state
        m_AME, m_AME_error, extrapol, A  = get_AME_values(species,
                                                          Ex=Ex,
                                                          Ex_error=Ex_error,
                                                          src=self.lit_src)
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


    def check_u_to_keV_conversion(self):
        """Check the u_to_keV factor from Scipy against value from
        """
        msg = "u_to_keV factor deviates from CODATA reference value"
        assert self._isclose(emg.u_to_keV, self.u_to_keV_ref,atol=5,rtol=0), msg


    def test_extrapol_AME_value(self):
        """Check calculation of ionic mass of extrapolated superheavy species
        """
        species = '1Bh264:-1e'
        # atomic masses and m_e from AME2016 [u]:
        m_ref = 264.124593 - self.m_e_ref
        m_ref_error =  190e-06
        extrapol_ref = True
        A_ref = 264
        z_ref = 1

        self._check_lit_values(species, m_ref, m_ref_error, extrapol_ref, A_ref,
                               z_ref=z_ref, Ex=0.0, Ex_error=0.0)


    def test_molecular_AME_values(self):
        """Check calculation of ionic mass of H2O (incl. tentative ID flag)"""
        species = '2H1:1O16?:-1e'
        # atomic masses and m_e from AME2016 [u]:
        m_H1_ref = 1.00782503224
        m_H1_ref_error = 0.00009e-06
        m_O16_ref = 15.99491461960
        m_O16_ref_error = 0.00017e-06
        m_ref = 2*m_H1_ref + m_O16_ref - self.m_e_ref
        m_ref_error = np.sqrt((2*m_H1_ref_error)**2 + m_O16_ref_error**2)
        extrapol_ref = False
        A_ref = 18
        z_ref = 1

        self._check_lit_values(species, m_ref, m_ref_error, extrapol_ref, A_ref,
                               z_ref=z_ref, Ex=0.0, Ex_error=0.0)


    def test_isomer_AME_values(self):
        """Check calculation of ionic mass of second isomer of In-127 """
        species = '1In127m1:-1e'
        # g.s. mass and m_e from AME2016 [u], Ex from ENSDF [keV]:
        Ex = 1863
        Ex_error = 58
        m_In127_ref = 126.917448546
        m_In127_ref_error = 22.713e-06
        m_ref =  m_In127_ref + Ex/self.u_to_keV_ref - self.m_e_ref
        m_ref_error = np.sqrt(m_In127_ref_error**2 +
                              (Ex_error/self.u_to_keV_ref)**2)
        extrapol_ref = False
        A_ref = 127
        z_ref = 1
        print(m_In127_ref)
        from emgfit.ame_funcs import get_AME_values
        print(get_AME_values('1In127', src='AME2016'))
        self._check_lit_values(species, m_ref, m_ref_error, extrapol_ref, A_ref,
                               z_ref=z_ref, Ex=Ex, Ex_error=Ex_error)


    def test_molecular_isomer_AME_values(self):
        """Check calculation of ionic mass of second isomer of Sr85m:F19:-1e """
        species = 'Sr85m:F19:-1e'
        # g.s. mass and m_e from AME2016 [u], Ex from ENSDF [keV]:
        Ex = 238.79
        Ex_error = 0.05
        m_Sr85_ref = 84.912932043
        m_Sr85_ref_error = 3.020e-06
        m_F19_ref = 18.99840316288
        m_F19_ref_error = 0.00093e-06
        m_ref = m_Sr85_ref + Ex/self.u_to_keV_ref + m_F19_ref - self.m_e_ref
        m_ref_error = np.sqrt(m_Sr85_ref_error**2 +
                              (Ex_error/self.u_to_keV_ref)**2 +
                              m_F19_ref_error**2)
        extrapol_ref = False
        A_ref = 104
        z_ref = 1

        self._check_lit_values(species, m_ref, m_ref_error, extrapol_ref, A_ref,
                               z_ref=z_ref, Ex=Ex, Ex_error=Ex_error)


    def test_doubly_charged_AME_values(self):
        """Check calculation of ionic mass of doubly charged Sn-116 """
        species = 'Sn116:-2e'
        # atomic mass and m_e from AME2016 [u]:
        m_ref = 115.901742824 - 2*self.m_e_ref
        m_ref_error = 0.103e-06
        extrapol_ref = False
        A_ref = 116
        z_ref = 2

        self._check_lit_values(species, m_ref, m_ref_error, extrapol_ref, A_ref,
                               z_ref=z_ref)


    def test_doubly_charged_molecular_isomer_AME_values(self):
        """Check calculation of ionic mass of first isomer of 2Y89m:O16:-2e """
        species = '2Y89m:1O16:-2e'
        # g.s. mass and m_e from AME2016 [u], Ex from ENSDF [keV]:
        Ex = 908.97
        Ex_error = 0.03
        m_Y89_ref = 88.905841205
        m_Y89_ref_error = 1.730e-06
        m_O16_ref = 15.99491461960
        m_O16_ref_error = 0.00017e-6
        m_ref = 2*(m_Y89_ref + Ex/self.u_to_keV_ref) + m_O16_ref -2*self.m_e_ref
        m_ref_error = np.sqrt((2*m_Y89_ref_error)**2 +
                              (2*Ex_error/self.u_to_keV_ref)**2 +
                              (m_O16_ref_error)**2)
        extrapol_ref = False
        A_ref = 194
        z_ref = 2

        self._check_lit_values(species, m_ref, m_ref_error, extrapol_ref, A_ref,
                               z_ref=z_ref, Ex=Ex, Ex_error=Ex_error)


    def test_get_El_from_Z(self):
        """Check grabbing of element symbol from proton number """
        proton_numbers = [0,1,6,67,118]
        El_ref = ['n','H','C','Ho','Og']
        from emgfit.ame_funcs import get_El_from_Z
        for i, Z in enumerate(proton_numbers):
            El = get_El_from_Z(Z)
            msg = str("result of get_El_from_Z({}) deviates from "
                      "reference").format(Z)
            assert El==El_ref[i], msg
