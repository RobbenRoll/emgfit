import pytest
import emgfit as emg
import numpy as np
from emgfit.model import EMGModel, EMGModelResult

class TestModel:
    def test_EMGModel(self):
        peak_index = 0
        mu0 = 100.
        amp0 = 1.
        pars_dict = emg.fit_models.pars_dict
        from emgfit.fit_models import emg01
        mod = emg01(peak_index, mu0, amp0, init_pars=pars_dict, 
                    cost_func="chi-square", vary_shape_pars=True, 
                    vary_baseline=True)
        
        from emgfit.model import save_model, load_model
        save_model(mod, "test_model.sav")
        mod = load_model("test_model.sav")

        assert mod.vary_baseline is True
        assert mod.vary_shape is True
        assert mod.func.__name__ == "emg01"
        assert mod.cost_func == "chi-square"
        def resid_Pearson_chi_square(pars, y_data, weights, model, **kwargs):
            y_m = model.eval(pars, **kwargs)
            weights = 1./np.sqrt(y_m + emg.model.EPS)
            return (y_m - y_data)*weights
        y_data = np.random.randint(0, 100, size=100)
        x = np.arange(1,100,len(y_data))
        pars = mod.make_params()
        ref_resid = resid_Pearson_chi_square(pars, y_data, None, mod, x=x)
        assert (mod._residual(pars, y_data, None, x=x) == ref_resid).all()
        assert pars["p0_mu"] == mu0


    def test_EMGModelResult(self):
        peak_index = 0
        mu = 100.
        pars_dict = emg.fit_models.pars_dict
        scl_fac = 1.0
        method = "least_squares"
        cost_func = "chi-square"
        N_events = 10000      

        x_min = mu - 10*pars_dict["sigma"]
        x_max = mu + 10*pars_dict["sigma"]

        # Simulate events
        from emgfit.sample import simulate_events
        N_bins = int((x_max - x_min)/(0.1*pars_dict["sigma"]))
        bin_width = (x_max - x_min)/N_bins
        amp = N_events*bin_width
        bkg_c = 0.

        np.random.seed(0)
        data = simulate_events(pars_dict, mu, amp, bkg_c, N_events, x_min,
                               x_max, scl_facs=scl_fac, out='hist', 
                               N_bins=N_bins)
        
        from emgfit.fit_models import emg33
        mod = emg33(peak_index, mu, amp, init_pars=pars_dict, 
                    cost_func=cost_func, vary_shape_pars=False, 
                    vary_baseline=True)

        pars = mod.make_params() # create parameters object for model

        result = mod.fit(data.values, params=pars, x=data.index.values, 
                         method=method, fitted_peaks=[0])
        
        from emgfit.model import save_modelresult, load_modelresult
        save_modelresult(result, "test_modelresult.sav")
        result = load_modelresult("test_modelresult.sav")
        
        assert (result.y == data.values).all()
        y_m = result.best_fit
        Pearson_weights = 1./np.sqrt(y_m + emg.model.EPS)
        ref_y_err = 1./Pearson_weights
        assert (result.y_err == ref_y_err).all()
        ref_fit_range = max(data.index) - min(data.index)
        assert np.isclose(result.x_fit_range, ref_fit_range)
        assert np.isclose(result.x_fit_cen, min(data.index) + ref_fit_range/2)
        assert result.fit_model == "emg33"
        assert result.vary_baseline == True 
        assert result.vary_shape == False
        assert result.par_hint_args == {}
        assert result.cost_func == cost_func
        assert result.method == "least_squares"
    
    