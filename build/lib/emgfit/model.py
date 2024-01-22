##################################################################################
##### Module defining hyper-EMG model interface based on parent classes from lmfit 
##### Authors: lmfit authors and Stefan Paul
import lmfit
import operator
import json
import warnings
import numpy as np
import lmfit.lineshapes as lineshapes
from lmfit.jsonutils import HAS_DILL
from copy import deepcopy
from lmfit.model import _align, tiny
from lmfit import Parameter, Parameters
from lmfit.jsonutils import HAS_DILL, decode4js, encode4js
from numpy import sqrt, log

EPS = 1e-10 # small number to bound Pearson cost function weights

# Use pandas.isnull for aligning missing data if pandas is available.
# otherwise use numpy.isnan
try:
    from pandas import Series, isnull
except ImportError:
    isnull = np.isnan
    Series = type(NotImplemented)

def coerce_arraylike(x):
    """
    coerce lists, tuples, and pandas Series, hdf5 Groups, etc to an
    ndarray float64 or complex128, but leave other data structures
    and objects unchanged

    Function adopted from :mod:`lmfit.model` module.
    
    """
    if isinstance(x, (list, tuple, Series)) or hasattr(x, '__array__'):
        if np.isrealobj(x):
            return np.asfarray(x)
        if np.iscomplexobj(x):
            return np.asfarray(x, dtype=np.complex128)
    return x


class EMGModel(lmfit.model.Model):
    """Create hyper-EMG fit model  
    
    This class inherits from the :class`lmfit.model.Model` class and creates a 
    single-peak model. This class enables overriding of lmfit's default 
    residuals with emgfit's custom cost functions, thereby enabling fits beyond 
    standard least squares minimization. 
        
    """
    def __init__(self, func, cost_func="default", 
                 par_hint_args={}, vary_baseline=True, vary_shape=True, 
                 independent_vars=['x'], param_names=None, 
                 nan_policy='propagate', prefix='', name=None, **kws):
        """
        The model function will normally take an independent variable
        (generally, the first argument) and a series of arguments that are
        meant to be parameters for the model. It will return an array of
        data to model some data as for a curve-fitting problem.

        Method adopted from :mod:`lmfit.model` module.

        Parameters
        ----------
        func : callable
            Function to be wrapped.
        cost_func : str, optional
            Name of cost function to use for minimization - overrides the 
            model's :attr:`~lmfit.model.Model._residual` attribute. 

            - If ``'chi-square'``, the fit is performed by minimizing Pearson's
              chi-squared statistic:

              .. math::

                  \\chi^2_P = \\sum_i \\frac{(f(x_i) - y_i)^2}{f(x_i)}.

            - If ``'MLE'``, a binned maximum likelihood estimation is performed
              by minimizing the (doubled) negative log likelihood ratio:

              .. math::

                  L = 2\\sum_i \\left[ f(x_i) - y_i + y_i ln\\left(\\frac{y_i}{f(x_i)}\\right)\\right]

            - If ``'default'`` (default), use the standard residual from lmfit.

            See `Notes` of :meth:`~emgfit.spectrum.spectrum.peakfit` method for 
            more details.
        par_hint_args : dict of dicts, optional
            Arguments to pass to :meth:`lmfit.model.Model.set_param_hint` to
            modify or add model parameters. The keys of the `par_hint_args`
            dictionary specify parameter names; the values must likewise be
            dictionaries that hold the respective keyword arguments to pass to
            :meth:`~lmfit.model.Model.set_param_hint`.
        vary_baseline : bool, optional, default: `True`
            If `True`, the constant background will be fitted with a varying
            uniform baseline parameter `bkg_c`.
            If `False`, the baseline parameter `bkg_c` will be fixed to 0.
        vary_shape : bool, optional, default: `False`
            If `False` peak-shape parameters of hyper-EMG models (`sigma`, 
            `theta`,`etas` and `taus`) are kept fixed at their initial values. 
            If `True` the shared shape parameters are varied (ensuring 
            identical shape parameters for all peaks).
        independent_vars : :obj:`list` of :obj:`str`, optional
            Arguments to `func` that are independent variables (default is
            None).
        param_names : :obj:`list` of :obj:`str`, optional
            Names of arguments to `func` that are to be made into
            parameters (default is None).
        nan_policy : {'raise', 'propagate', 'omit'}, optional
            How to handle NaN and missing values in data. See Notes below.
        prefix : str, optional
            Prefix used for the model.
        name : str, optional
            Name for the model. When None (default) the name is the same
            as the model function (`func`).
        **kws : dict, optional
            Additional keyword arguments to pass to model function.

        Notes
        -----
        1. Parameter names are inferred from the function arguments, and a
        residual function is automatically constructed.

        2. The model function must return an array that will be the same
        size as the data being modeled.

        3. `nan_policy` sets what to do when a NaN or missing value is
        seen in the data. Should be one of:

           - `'raise'` : raise a `ValueError` (default)
           - `'propagate'` : do nothing
           - `'omit'` : drop missing data

        Examples
        --------
        The model function will normally take an independent variable
        (generally, the first argument) and a series of arguments that are
        meant to be parameters for the model. Thus, a simple peak using a
        Gaussian defined as:

        >>> import numpy as np
        >>> def gaussian(x, amp, cen, wid):
        ...     return amp * np.exp(-(x-cen)**2 / wid)

        can be turned into a Model with:

        >>> gmodel = Model(gaussian)

        this will automatically discover the names of the independent
        variables and parameters:

        >>> print(gmodel.param_names, gmodel.independent_vars)
        ['amp', 'cen', 'wid'], ['x']

        """        
        super().__init__(func, independent_vars=independent_vars, 
                         param_names=param_names, nan_policy=nan_policy, 
                         prefix=prefix, name=name, **kws)
        self._override_residual(cost_func)
        self.par_hint_args = par_hint_args
        self.vary_baseline = vary_baseline
        self.vary_shape = vary_shape


    def _override_residual(self, cost_func):
        """Override residual method 
        
        Parameters
        ----------
        cost_func : str, optional 
            Name of cost function to use for minimization - overrides the 
            model's :attr:`~lmfit.model.Model._residual` attribute. 

            - If ``'chi-square'``, the fit is performed by minimizing Pearson's
              chi-squared statistic:

              .. math::

                  \\chi^2_P = \\sum_i \\frac{(f(x_i) - y_i)^2}{f(x_i)}.

            - If ``'MLE'``, a binned maximum likelihood estimation is performed
              by minimizing the (doubled) negative log likelihood ratio:

              .. math::

                  L = 2\\sum_i \\left[ f(x_i) - y_i + y_i ln\\left(\\frac{y_i}{f(x_i)}\\right)\\right]

            - If ``'default'`` (default), use the standrad residual from lmfit.

            See `Notes` of :meth:`~emgfit.spectrum.spectrum.peakfit` method for 
            more details.

        """
        if cost_func == "default":
            pass
        elif cost_func == "chi-square":
            # Pearson's chi-squared cost function with iterative weights 1/Sqrt(f(x_i))
            def resid_Pearson_chi_square(pars, y_data, weights, **kwargs):
                y_m = self.eval(pars, **kwargs)
                # Calculate weights for current iteration, add tiny number `EPS`
                # in denominator for numerical stability
                weights = 1./sqrt(y_m + EPS)
                return (y_m - y_data)*weights
            # Override lmfit's standard least square residuals with iterative
            # residuals for Pearson chi-square fit
            self._residual = resid_Pearson_chi_square
        elif cost_func == "MLE":
            # Define sqrt of (doubled) negative log-likelihood ratio (NLLR) summands
            def sqrt_NLLR(pars, y_data, weights, **kwargs):
                y_m = self.eval(pars, **kwargs) 
                # Add tiniest pos. float representable by numpy to arguments of
                # np.log to smoothly handle divergences for log(arg -> 0)
                NLLR = 2*(y_m-y_data) + 2*y_data*(log(y_data+tiny)-log(y_m+tiny))
                ret = sqrt(NLLR)
                return ret
            # Override lmfit's standard least square residual with the
            # square-roots of the NLLR summands, this enables usage of scipy's
            # `least_squares` minimizer and yields much faster optimization
            # than with scalar minimizers
            self._residual = sqrt_NLLR    
        else:
            raise Exception(" Definition of `cost_func` failed!")
        self.cost_func = cost_func


    def _reprstring(self, long=False):
        """Print representation string

        Method adopted from :mod:`lmfit.model` module.
        """
        out = self._name
        opts = []
        if len(self._prefix) > 0:
            opts.append(f"prefix='{self._prefix}'")
        if long:
            for k, v in self.opts.items():
                opts.append(f"{k}='{v}'")
        if len(opts) > 0:
            out = f"{out}, {', '.join(opts)}"
        return f"EMGModel({out})"
        

    def _get_state(self):
        """Save a Model for serialization.

        Note: like the standard-ish '__getstate__' method but not really
        useful with Pickle.

        Method modified from :mod:`lmfit.model` module.

        """
        funcdef = None
        if HAS_DILL:
            funcdef = self.func
        if self.func.__name__ == '_eval':
            funcdef = self.expr
        state = (self.func.__name__, funcdef, self.cost_func, 
                 self.par_hint_args, self.vary_baseline, 
                 self.vary_shape, self._name, self._prefix,
                 self.independent_vars, self._param_root_names,
                 self.param_hints, self.nan_policy, self.opts)
        return (state, None, None)
    

    def _set_state(self, state, funcdefs=None):
        """Restore Model from serialization.

        Note: like the standard-ish '__setstate__' method but not really
        useful with Pickle.

        Method adopted from :mod:`lmfit.model` module.

        Parameters
        ----------
        state
            Serialized state from `_get_state`.
        funcdefs : dict, optional
            Dictionary of function definitions to use to construct Model.

        """
        return _buildmodel(state, funcdefs=funcdefs)
    

    def __repr__(self):
        """Return representation of EMGModel."""
        return f"<emgfit.EMGModel: {self.name}>"
    

    def fit(self, data, params=None, weights=None, fitted_peaks=None, 
            method='least_squares', iter_cb=None, scale_covar=True, verbose=False, 
            fit_kws=None, nan_policy=None, calc_covar=True, max_nfev=None,
            coerce_farray=True, **kwargs):
        """Fit the model to the data using the supplied Parameters.

        Method modified from :mod:`lmfit.model` module.

        Parameters
        ----------
        data : array_like
            Array of data to be fit.
        params : Parameters, optional
            Parameters to use in fit (default is None).
        weights : array_like, optional
            Weights to use for the calculation of the fit residual [i.e.,
            `weights*(data-fit)`]. Default is None; must have the same size as
            `data`.
        fitted_peaks : list of :class:`emgfit.spectrum.peak`
            List of peaks to be fitted.
        method : str, optional
            Name of fitting method to use (default is `'least_squares'`).
        iter_cb : callable, optional
            Callback function to call at each iteration (default is None).
        scale_covar : bool, optional
            Whether to automatically scale the covariance matrix when
            calculating uncertainties (default is True).
        verbose : bool, optional
            Whether to print a message when a new parameter is added
            because of a hint (default is True).
        fit_kws : dict, optional
            Options to pass to the minimizer being used.
        nan_policy : {'raise', 'propagate', 'omit'}, optional
            What to do when encountering NaNs when fitting Model.
        calc_covar : bool, optional
            Whether to calculate the covariance matrix (default is True)
            for solvers other than `'leastsq'` and `'least_squares'`.
            Requires the ``numdifftools`` package to be installed.
        max_nfev : int or None, optional
            Maximum number of function evaluations (default is None). The
            default value depends on the fitting method.
        coerce_farray : bool, optional
            Whether to coerce data and independent data to be ndarrays
            with dtype of float64 (or complex128).  If set to False, data
            and independent data are not coerced at all, but the output of
            the model function will be. (default is True)
        **kwargs : optional
            Arguments to pass to the model function, possibly overriding
            parameters.

        Returns
        -------
        :class:`emgfit.model.EMGModelResult`
            EMGModelResult

        Notes
        -----
        1. if `params` is None, the values for all parameters are expected
        to be provided as keyword arguments. Mixing `params` and
        keyword arguments is deprecated (see `Model.eval`).

        2. all non-parameter arguments for the model function, **including
        all the independent variables** will need to be passed in using
        keyword arguments.

        3. Parameters are copied on input, so that the original Parameter objects
        are unchanged, and the updated values are in the returned `EMGModelResult`.

        Examples
        --------
        Take ``t`` to be the independent variable and data to be the curve
        we will fit. Use keyword arguments to set initial guesses:

        >>> result = my_model.fit(data, tau=5, N=3, t=t)

        Or, for more control, pass a Parameters object.

        >>> result = my_model.fit(data, params, t=t)

        """
        if params is None:
            params = self.make_params(verbose=verbose)
        else:
            params = deepcopy(params)

        # If any kwargs match parameter names, override params.
        param_kwargs = set(kwargs.keys()) & set(self.param_names)
        for name in param_kwargs:
            p = kwargs[name]
            if isinstance(p, Parameter):
                p.name = name  # allows N=Parameter(value=5) with implicit name
                params[name] = deepcopy(p)
            else:
                params[name].set(value=p)
            del kwargs[name]

        # All remaining kwargs should correspond to independent variables.
        for name in kwargs:
            if name not in self.independent_vars:
                warnings.warn(f"The keyword argument {name} does not " +
                              "match any arguments of the model function. " +
                              "It will be ignored.", UserWarning)

        # If any parameter is not initialized raise a more helpful error.
        missing_param = any(p not in params.keys() for p in self.param_names)
        blank_param = any((p.value is None and p.expr is None)
                          for p in params.values())
        if missing_param or blank_param:
            msg = ('Assign each parameter an initial value by passing '
                   'Parameters or keyword arguments to fit.\n')
            missing = [p for p in self.param_names if p not in params.keys()]
            blank = [name for name, p in params.items()
                     if p.value is None and p.expr is None]
            msg += f'Missing parameters: {str(missing)}\n'
            msg += f'Non initialized parameters: {str(blank)}'
            raise ValueError(msg)

        # Handle null/missing values.
        if nan_policy is not None:
            self.nan_policy = nan_policy

        mask = None
        if self.nan_policy == 'omit':
            mask = ~isnull(data)
            if mask is not None:
                data = data[mask]
            if weights is not None:
                weights = _align(weights, mask, data)

        # If independent_vars and data are alignable (pandas), align them,
        # and apply the mask from above if there is one.
        for var in self.independent_vars:
            if not np.isscalar(kwargs[var]):
                kwargs[var] = _align(kwargs[var], mask, data)

        if coerce_farray:
            # coerce data and independent variable(s) that are 'array-like' (list,
            # tuples, pandas Series) to float64/complex128.
            data = coerce_arraylike(data)
            for var in self.independent_vars:
                kwargs[var] = coerce_arraylike(kwargs[var])

        if fit_kws is None:
            fit_kws = {}

        output = EMGModelResult(self, params, data=data, weights=weights, 
                                fitted_peaks=fitted_peaks, method=method, 
                                fcn_kws=kwargs, iter_cb=iter_cb, 
                                scale_covar=scale_covar, 
                                nan_policy=self.nan_policy, 
                                calc_covar=calc_covar, 
                                max_nfev=max_nfev, **fit_kws)
        output.fit(data=data, weights=weights) 
        output.components = self.components
        return output
    

    def __add__(self, other):
        """+"""
        return CompositeEMGModel(self, other, operator.add)

    def __sub__(self, other):
        """-"""
        return CompositeEMGModel(self, other, operator.sub)

    def __mul__(self, other):
        """*"""
        return CompositeEMGModel(self, other, operator.mul)

    def __truediv__(self, other):
        """/"""
        return CompositeEMGModel(self, other, operator.truediv)
    


class CompositeEMGModel(EMGModel):
    """Combine two EMGModels (`left` and `right`) with binary operator (`op`).

    Normally, one does not have to explicitly create a `CompositeModel`,
    but can use normal Python operators ``+``, ``-``, ``*``, and ``/`` to
    combine components as in::

    >>> mod = EMGModel(fcn1) + EMGModel(fcn2) * EMGModel(fcn3)

    Class modified from :mod:`lmfit.model.CompositeModel` class.

    """

    _known_ops = {operator.add: '+', operator.sub: '-',
                  operator.mul: '*', operator.truediv: '/'}

    def __init__(self, left, right, op, **kws):
        """Create composite EMG model

        Method modified from :mod:`lmfit.model` module.

        Parameters
        ----------
        left : Model
            Left-hand model.
        right : Model
            Right-hand model.
        op : callable binary operator
            Operator to combine `left` and `right` models.
        **kws : optional
            Additional keywords are passed to `Model` when creating this
            new model.

        Notes
        -----
        The two models can use different independent variables.

        """
        if not isinstance(left, EMGModel):
            raise ValueError(f'CompositeModel: argument {left} is not a Model')
        if not isinstance(right, EMGModel):
            raise ValueError(f'CompositeModel: argument {right} is not a Model')
        if not callable(op):
            raise ValueError(f'CompositeModel: operator {op} is not callable')

        self.left = left
        self.right = right
        self.op = op

        name_collisions = set(left.param_names) & set(right.param_names)
        if len(name_collisions) > 0:
            msg = ''
            for collision in name_collisions:
                msg += (f"\nTwo models have parameters named '{collision}'; "
                        "use distinct names.")
            raise NameError(msg)

        # the unique ``independent_vars`` of the left and right model are
        # combined to ``independent_vars`` of the ``CompositeModel``
        if 'independent_vars' not in kws:
            ivars = self.left.independent_vars + self.right.independent_vars
            kws['independent_vars'] = list(np.unique(ivars))
        if 'nan_policy' not in kws:
            kws['nan_policy'] = self.left.nan_policy
        if 'cost_func' not in kws:
            kws['cost_func'] = self.left.cost_func

        def _tmp(self, *args, **kws):
            pass
        EMGModel.__init__(self, _tmp, **kws)

        for side in (left, right):
            prefix = side.prefix
            for basename, hint in side.param_hints.items():
                self.param_hints[f"{prefix}{basename}"] = hint
    
    def _parse_params(self):
        """Method adopted from :mod:`lmfit.model` module."""
        self._func_haskeywords = (self.left._func_haskeywords or
                                  self.right._func_haskeywords)
        self._func_allargs = (self.left._func_allargs +
                              self.right._func_allargs)
        self.def_vals = deepcopy(self.right.def_vals)
        self.def_vals.update(self.left.def_vals)
        self.opts = deepcopy(self.right.opts)
        self.opts.update(self.left.opts)

    def _reprstring(self, long=False):
        """Method adopted from :mod:`lmfit.model` module."""
        return (f"({self.left._reprstring(long=long)} "
                f"{self._known_ops.get(self.op, self.op)} "
                f"{self.right._reprstring(long=long)})")

    def eval(self, params=None, **kwargs):
        """Evaluate model function for composite model.
        
        Method adopted from :mod:`lmfit.model` module.

        """
        return self.op(self.left.eval(params=params, **kwargs),
                       self.right.eval(params=params, **kwargs))

    def eval_components(self, **kwargs):
        """Return dictionary of name, results for each component.
        
        Method adopted from :mod:`lmfit.model` module.

        """
        out = dict(self.left.eval_components(**kwargs))
        out.update(self.right.eval_components(**kwargs))
        return out

    def post_fit(self, fitresult):
        """function that is called just after fit, can be overloaded by
        subclasses to add non-fitting 'calculated parameters'

        Method adopted from :mod:`lmfit.model` module.

        """
        self.left.post_fit(fitresult)
        self.right.post_fit(fitresult)

    @property
    def param_names(self):
        """Return parameter names for composite model.
        
        Method adopted from :mod:`lmfit.model` module.

        """
        return self.left.param_names + self.right.param_names

    @property
    def components(self):
        """Return components for composite model.
        
        Method adopted from :mod:`lmfit.model` module.

        """
        return self.left.components + self.right.components

    def _get_state(self):
        """Method adopted from :mod:`lmfit.model` module."""
        return (self.left._get_state(),
                self.right._get_state(), self.op.__name__)

    def _set_state(self, state, funcdefs=None):
        """Method adopted from :mod:`lmfit.model` module."""
        return _buildmodel(state, funcdefs=funcdefs)

    def _make_all_args(self, params=None, **kwargs):
        """Generate **all** function arguments for all functions.

        Method adopted from :mod:`lmfit.model` module.

        """
        out = self.right._make_all_args(params=params, **kwargs)
        out.update(self.left._make_all_args(params=params, **kwargs))
        return out



def save_model(model, fname):
    """Save an EMGModel to a file.

    Function adopted from :mod:`lmfit.model` module.

    Parameters
    ----------
    model : :class:`~emgfit.model.EMGModel`
        Model to be saved.
    fname : str
        Name of file for saved Model.

    """
    with open(fname, 'w') as fout:
        model.dump(fout)


def load_model(fname, funcdefs=None): 
    """Load a saved EMGModel from a file.

    Function modified from :mod:`lmfit.model` module.

    Parameters
    ----------
    fname : str
        Name of file containing saved EMGModel.
    funcdefs : dict, optional
        Dictionary of custom function names and definitions.

    Returns
    -------
    Model
        Model object loaded from file.

    """
    m = EMGModel(lambda x: x)
    with open(fname) as fh:
        model = m.load(fh, funcdefs=funcdefs)
    return model


def _buildmodel(state, funcdefs=None):
    """Build EMGModel from saved state.

    Intended for internal use only.

    Function modified from :mod:`lmfit.model` module.

    """
    if len(state) != 3:
        raise ValueError("Cannot restore Model")
    known_funcs = {}
    for fname in lineshapes.functions:
        fcn = getattr(lineshapes, fname, None)
        if callable(fcn):
            known_funcs[fname] = fcn
    if funcdefs is not None:
        known_funcs.update(funcdefs)

    left, right, op = state
    if op is None and right is None:
        (fname, fcndef, cost_func, par_hint_args, vary_baseline, vary_shape, 
         name, prefix, ivars, pnames, 
         phints, nan_policy, opts) = left
        if not callable(fcndef) and fname in known_funcs:
            fcndef = known_funcs[fname]

        if fcndef is None:
            raise ValueError("Cannot restore Model: model function not found")

        if fname == '_eval' and isinstance(fcndef, str):
            raise NotImplementedError
        else:
            model = EMGModel(fcndef, cost_func=cost_func, 
                             par_hint_args=par_hint_args, 
                             vary_baseline=bool(vary_baseline), 
                             vary_shape=bool(vary_shape), 
                             name=name, prefix=prefix,
                             independent_vars=ivars, param_names=pnames,
                             nan_policy=nan_policy, **opts)

        for name, hint in phints.items():
            model.set_param_hint(name, **hint)
        return model
    else:
        lmodel = _buildmodel(left, funcdefs=funcdefs)
        rmodel = _buildmodel(right, funcdefs=funcdefs)
        return CompositeEMGModel(lmodel, rmodel, getattr(operator, op))
    

def save_modelresult(modelresult, fname):
    """Save an EMGModelResult to a file.

    Function adopted from :mod:`lmfit.model` module.

    Parameters
    ----------
    modelresult : :class:`~emgfit.model.EMGModelResult`
        EMGModelResult to be saved.
    fname : str
        Name of file for saved EMGModelResult.

    """
    with open(fname, 'w') as fout:
        modelresult.dump(fout)


def load_modelresult(fname, funcdefs=None):
    """Load a saved EMGModelResult from a file.

    Function modified from :mod:`lmfit.model` module.

    Parameters
    ----------
    fname : str
        Name of file containing saved EMGModelResult.
    funcdefs : dict, optional
        Dictionary of custom function names and definitions.

    Returns
    -------
    :class:`~emgfit.model.EMGModelResult`
        EMGModelResult object loaded from file.

    """
    params = Parameters()
    modres = EMGModelResult(EMGModel(lambda x: x), params)
    with open(fname) as fh:
        mresult = modres.load(fh, funcdefs=funcdefs)
    return mresult



################################################################################
###### Define emgfit EMGModelResult class
class EMGModelResult(lmfit.model.ModelResult):
    """Result from an EMGModel fit."""

    def __init__(self, model, params, data=None, weights=None, 
                 fitted_peaks=None, method='least_squares', fcn_args=None, 
                 fcn_kws=None, iter_cb=None, scale_covar=False, 
                 nan_policy='propagate', calc_covar=False, max_nfev=None, 
                 **fit_kws):
        """
        Parameters
        ----------
        model : Model
            Model to use.
        params : Parameters
            Parameters with initial values for model.
        data : array_like
            Ordinate values of data to be modeled.
        weights : array_like, optional
            Weights to multiply ``(data-model)`` for default fit residual.
        fitted_peaks : list of :class:`emgfit.spectrum.spectrum.peak`
            List of fitted peak objects.
        method : str, optional
            Name of minimization method to use (default is `'least_squares'`).
        fcn_args : sequence, optional
            Positional arguments to send to model function.
        fcn_kws : dict, optional
            Keyword arguments to send to model function.
        iter_cb : callable, optional
            Function to call on each iteration of fit.
        scale_covar : bool, optional
            Whether to scale covariance matrix for uncertainty evaluation.
        nan_policy : {'raise', 'propagate', 'omit'}, optional
            What to do when encountering NaNs when fitting Model.
        calc_covar : bool, optional
            Whether to calculate the covariance matrix (default is True)
            for solvers other than `'leastsq'` and `'least_squares'`.
            Requires the ``numdifftools`` package to be installed.
        max_nfev : int or None, optional
            Maximum number of function evaluations (default is None). The
            default value depends on the fitting method.
        **fit_kws : optional
            Keyword arguments to send to minimization routine.

        """
        if fcn_kws is None:
            fcn_kws = {}
        self.fitted_peaks = fitted_peaks
        self.fit_kws = fit_kws
        super().__init__(model, params, data=data, weights=weights, method=method, 
                         fcn_args=fcn_args, fcn_kws=fcn_kws, iter_cb=iter_cb, 
                         scale_covar=scale_covar, nan_policy=nan_policy, 
                         calc_covar=calc_covar, max_nfev=max_nfev, **fit_kws)
    
    @property 
    def x(self):
        """Ordinate values of fitted data"""
        if self.userkws in [None, {}]:
            x = None
        else:
            x = self.userkws[self.model.independent_vars[0]]
        return x

    @property
    def y(self):
        """Abscissa values of fitted data"""   
        return self.data 
    
    @property 
    def y_err(self):
        """Uncertainties of abscissa values of fitted data"""   
        if self.cost_func == "chi-square":
            # Calculate final weights for plotting
            y_m = self.best_fit
            Pearson_weights = 1./sqrt(y_m + EPS)
            y_err = 1./Pearson_weights
        else:
            y_err = 1./self.weights
        return y_err
    
    @property
    def x_fit_range(self):
        """Range of fitted ordinate values"""   
        return  max(self.x) - min(self.x)
    
    @property
    def x_fit_cen(self):
        """Centre of fitted ordinate range""" 
        return min(self.x) + 0.5*self.x_fit_range
    
    @property 
    def fit_model(self):
        """Name of hyper-EMG model used in fit""" 
        if isinstance(self.model, CompositeEMGModel): 
            fit_model = self.model.components[-1].func.__name__
        else:
            fit_model = self.model.func.__name__
        return fit_model
    
    @property 
    def vary_baseline(self):
        """Whether the amplitude of the uniform background was varied in the fit"""
        return self.model.vary_baseline

    @property 
    def vary_shape(self):
        """Whether the peak shape (and scale) parameters were varied in the fit"""
        return self.model.vary_shape
    
    @property 
    def par_hint_args(self):
        """User-specified parameter hints used in the fit"""
        return self.model.par_hint_args
    
    @property 
    def cost_func(self):
        """Name of the cost function used in the fit"""
        return self.model.cost_func
   

    def fit(self, data=None, fitted_peaks=None, params=None, weights=None, 
            method=None, nan_policy=None, **kwargs):
        """Re-perform fit for a Model, given data and params.  

        Method modified from :mod:`lmfit.model` module.

        Parameters
        ----------
        data : array_like, optional 
            Data to be modeled.
        fitted_peaks : list of :class:`emgfit.spectrum.spectrum.peak`
            List of peak objects to be fitted.
        params : Parameters, optional
            Parameters with initial values for model.
        weights : array_like, optional
            Weights to multiply ``(data-model)`` for fit residual - only 
            relevant when using the default lmfit cost function. 
        method : str, optional
            Name of minimization method to use (default is `'least_squares'`).
        nan_policy : {'raise', 'propagate', 'omit'}, optional
            What to do when encountering NaNs when fitting Model.
        **kwargs : optional
            Keyword arguments to send to minimization routine.

        """
        if data is not None:
            self.data = data
        if fitted_peaks is not None:
            self.fitted_peaks = fitted_peaks
        if params is not None:
            self.init_params = params
        if weights is not None:
            self.weights = weights
        if method is not None:
            self.method = method
        if nan_policy is not None:
            self.nan_policy = nan_policy

        self.ci_out = None
        self.userargs = (self.data, self.weights)
        self.userkws.update(kwargs)
        self.init_fit = self.model.eval(params=self.params, **self.userkws)
        _ret = self.minimize(method=self.method)
        self.model.post_fit(_ret)
        _ret.params.create_uvars(covar=_ret.covar)

        for attr in dir(_ret):
            if not attr.startswith('_') and attr != "x":
                try:
                    setattr(self, attr, getattr(_ret, attr))
                except AttributeError:
                    pass

        self.init_values = self.model._make_all_args(self.init_params)
        self.best_values = self.model._make_all_args(_ret.params)
        self.best_fit = self.model.eval(params=_ret.params, **self.userkws)
        if (self.data is not None and len(self.data) > 1
           and isinstance(self.best_fit, np.ndarray)
           and len(self.best_fit) > 1):
            dat = coerce_arraylike(self.data)
            resid = ((dat - self.best_fit)**2).sum()
            sstot = ((dat - dat.mean())**2).sum()
            self.rsquared = 1.0 - resid/max(tiny, sstot)
    
    def __repr__(self):
        """Return representation of EMGModelResult.
        
        Method modified from :mod:`lmfit.model` module.

        """
        return f"<emgfit.EMGModelResult: {self.model.name}>"
    
    def loads(self, s, funcdefs=None, **kws):
        """Load ModelResult from a JSON string.

        Parameters
        ----------
        s : str
            String representation of ModelResult, as from `dumps`.
        funcdefs : dict, optional
            Dictionary of custom function names and definitions.
        **kws : optional
            Keyword arguments that are passed to `json.loads`.

        Returns
        -------
        ModelResult
            ModelResult instance from JSON string representation.

        See Also
        --------
        load, dumps, json.dumps

        """
        modres = json.loads(s, **kws)
        if 'modelresult' not in modres['__class__'].lower():
            raise AttributeError('ModelResult.loads() needs saved ModelResult')

        modres = decode4js(modres)
        if 'model' not in modres or 'params' not in modres:
            raise AttributeError('ModelResult.loads() needs valid ModelResult')

        # model
        self.model = _buildmodel(decode4js(modres['model']), funcdefs=funcdefs)

        if funcdefs:
            # Remove model function so as not pass it into the _asteval.symtable
            funcdefs.pop(self.model.func.__name__, None)

        # how params are saved was changed with version 2:
        modres_vers = modres.get('__version__', '1')
        if modres_vers == '1':
            for target in ('params', 'init_params'):
                state = {'unique_symbols': modres['unique_symbols'], 'params': []}
                for parstate in modres['params']:
                    _par = Parameter(name='')
                    _par.__setstate__(parstate)
                    state['params'].append(_par)
                _params = Parameters(usersyms=funcdefs)
                _params.__setstate__(state)
                setattr(self, target, _params)

        elif modres_vers == '2':
            for target in ('params', 'init_params'):
                _pars = Parameters()
                _pars.loads(modres[target])
                if funcdefs:
                    for key, val in funcdefs.items():
                        _pars._asteval.symtable[key] = val
                setattr(self, target, _pars)

        for attr in ('aborted', 'aic', 'best_fit', 'best_values', 'bic',
                     'chisqr', 'ci_out', 'col_deriv', 'covar', 'data',
                     'errorbars', 'fjac', 'flatchain', 'ier', 'init_fit',
                     'init_values', 'kws', 'lmdif_message', 'message',
                     'method', 'nan_policy', 'ndata', 'nfev', 'nfree',
                     'nvarys', 'redchi', 'residual', 'rsquared', 'scale_covar',
                     'calc_covar', 'success', 'userargs', 'userkws',
                     'var_names', 'weights', 'user_options', 'fitted_peaks', 
                     'fit_kws'):
            setattr(self, attr, decode4js(modres.get(attr, None)))

        self.best_fit = self.model.eval(self.params, **self.userkws)
        if len(self.userargs) == 2:
            self.data = self.userargs[0]
            self.weights = self.userargs[1]

        for parname, val in self.init_values.items():
            par = self.init_params.get(parname, None)
            if par is not None:
                par.correl = par.stderr = None
                par.value = par.init_value = self.init_values[parname]

        self.init_fit = self.model.eval(self.init_params, **self.userkws)
        self.result = lmfit.minimizer.MinimizerResult()
        self.result.params = self.params

        if self.errorbars and self.covar is not None:
            self.uvars = self.result.params.create_uvars(covar=self.covar)

        self.init_vals = list(self.init_values.items())
        return self

    def load(self, fp, funcdefs=None, **kws):
        """Load JSON representation of ModelResult from a file-like object.

        Parameters
        ----------
        fp : file-like object
            An open and `.read()`-supporting file-like object.
        funcdefs : dict, optional
            Dictionary of function definitions to use to construct Model.
        **kws : optional
            Keyword arguments that are passed to `loads`.

        Returns
        -------
        ModelResult
            ModelResult created from `fp`.

        See Also
        --------
        dump, loads, json.load

        """
        return self.loads(fp.read(), funcdefs=funcdefs, **kws)

    