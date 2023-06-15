import casadi as ad
import ipyopt
from ipyopt.optimize import IPOPT_RETURN_CODES
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.special

import statsmodels.base.model as smm
import statsmodels.discrete.discrete_model as smd
from statsmodels.tools.decorators import cached_value, cache_readonly


from .util import random_multinomial
from .nestspec import NestSpec


class ModelBase:
    def __init__(self, data, param_names, casadi_function_opts):
        self.param_names = list(param_names)
        self.num_params = len(self.param_names)
        self.casadi_function_opts = casadi_function_opts
        self.data = data
        self.exog_names_ = data.xnames()
        self.endog_names = data.ynames()
        self.exog_shape = data.exog_shape
        self.endog_shape = data.endog_shape
        self.nobs = data.nobs
        self.exog_name_to_i = \
            {self.exog_names[i]: i for i in range(len(self.exog_names))}
        self.endog_name_to_i = \
            {self.endog_names[i]: i for i in range(len(self.endog_names))}
        # TODO: FIGURE OUT WHAT THE HELL TO DO WITH df_model and k_extra
        self.k_extra = 0
        self.df_model = self.num_params
        self.df_resid = self.nobs - self.df_model

        # --- statsmodels results compatibility:
        self.exog = np.broadcast_to(0., self.exog_shape)
        self.endog = np.broadcast_to(0., self.endog_shape)
        self.param_names_instead_of_exog_names = False

    @property
    def exog_names(self):
        """
        self.param_names_instead_of_exog_names exists because statsmodels
        summary uses exog_names for names of params
        """
        if getattr(self, 'param_names_instead_of_exog_names', False):
            return self.param_names
        return self.exog_names_

    def _loglike_casadi_sym(self):
        """
        Returns (params,endog,exog,f,g,H). params, f, g, and H are SX.
        Note that this should be calculated for a single row only, that is,
        exog and endog should only have a single row.
        """
        raise NotImplementedError

    @cached_value
    def _loglike_casadi_funcs(self):
        p, y, x, f, g, H = self._loglike_casadi_sym()
        args = [p, y, x]
        return (ad.Function('f', args, [f], self.casadi_function_opts),
                ad.Function('g', args, [g], self.casadi_function_opts),
                ad.Function('Hnz', args, [H.get_nz(False, slice(None))],
                            self.casadi_function_opts),
                H.sparsity())

    def _eval_casadi_function_on_data(self, f, params, out=None):
        """
        Assumption: f output is column vector or scalar
        """
        if out is None:
            out = np.zeros(f.size_out(0))
        else:
            out[...] = 0.
        outshape = out.shape
        if out.ndim < 2:
            out = out[:, None]
        for i in range(0, self.nobs, self.data.max_rows):
            endog, exog = self.data.get_endog_exog(
                i, min(i + self.data.max_rows, self.nobs))
            res = f(params, endog.T, exog.T)
            out[:, :] += ad.sum2(res)
        return out.reshape(outshape)

    def loglike(self, params):
        f, g, Hnz, Hsp = self._loglike_casadi_funcs
        return self._eval_casadi_function_on_data(f, params).item()

    def score(self, params, out=None):
        f, g, Hnz, Hsp = self._loglike_casadi_funcs
        return self._eval_casadi_function_on_data(g, params, out)

    def information(self, params, sparse_out=False):
        return -self.hessian(params, sparse_out)

    def hessian(self, params, sparse_out=False):
        f, g, Hnz, Hsp = self._loglike_casadi_funcs
        Hdata = self._eval_casadi_function_on_data(Hnz, params)
        indptr, indices = Hsp.get_ccs()
        res = scipy.sparse.csc_matrix((Hdata.flatten(), indices, indptr))
        return res if sparse_out else res.toarray()

    def _fit_result(self, params, convergence_msg):
        Hval = self.hessian(params)
        try:
            Hinv = scipy.linalg.inv(-Hval)
        except scipy.linalg.LinAlgError:
            print("Information matrix:", -Hval)
            print("Eigenvalues:", scipy.linalg.eigh(-Hval)[0])
            raise
        # TODO: OTHER OPTIONS
        mlefit = smm.LikelihoodModelResults(self, params, Hinv, scale=1.)
        mlefit.mle_retvals = {'converged': convergence_msg}

        results = CustomModelBaseResults(self, mlefit)
        # any modification to results necessary?
        return results

    def fit(self, start_params, lbx, ubx, constraints=None, ipopt_options={}):
        """
        lbx is lower bounds on parameters, ubx is upper bounds.
        """
        def f(params):
            return -self.loglike(params)

        def grad_f(params, out):
            self.score(params, out)
            out[:] = -out
            return out

        # TODO: CONSTRAINTS
        def g(params, out):
            return out

        def jac_g(params, out):
            return out

        def h(params, lagrange, obj_factor, out):
            H = -obj_factor * self.hessian(params, sparse_out=True)
            # + lagrange...
            out[:] = H.data
            return out

        Hsp = self._loglike_casadi_funcs[3]
        Hccs = Hsp.get_ccs()
        Hindptr = np.array(Hccs[0])
        Hcolindices = np.repeat(np.arange(len(Hindptr) - 1),
                                Hindptr[1:] - Hindptr[:-1])
        Hrowindices = np.array(Hccs[1])

        nlp = ipyopt.Problem(
            n=self.num_params,
            x_l=np.array(lbx),
            x_u=np.array(ubx),
            m=0,
            g_l=np.empty(0),
            g_u=np.empty(0),
            sparsity_indices_jac_g=(np.empty(0), np.empty(0)),
            sparsity_indices_h=(Hrowindices, Hcolindices),
            eval_f=f,
            eval_grad_f=grad_f,
            eval_g=g,
            eval_jac_g=jac_g,
            eval_h=h,
            ipopt_options=dict(ipopt_options))
        p_opt, llf, status = nlp.solve(x0=start_params.copy())
        return self._fit_result(np.array(p_opt).flatten(),
                                IPOPT_RETURN_CODES[status])

    def fit_null(self):
        """
        Fit with constant exog. For CustomModelBaseResults.llnull
        """
        raise NotImplementedError


class NestedLogitModel(ModelBase):
    """
    Nested Logit model.
    """

    def __init__(self, data, classes, nests, availability_vars, params,
                 casadi_function_opts):
        """
        classes: dict, each entry is (class_name: endog_name),
            where the null class has a class_name of 0.
        nests: List of lists, to be passed to NestSpec.
        availability_vars: dict, each entry is (class_name: exog_name).
            If exog_name is None, assume said class is always available
            If exog_name is missing, assume said class is never available
        params: dict, each entry is (name: spec), where spec is a dict
            where each entry is (exog_name: class_names).
            exog_name is None means intercept

        Note that as of Python 3.7, dicts are ordered
        """
        # TODO: ERROR CHECKING
        self.classes = dict(classes)
        self.nestspec = NestSpec(nests)
        self.availability_vars = dict(availability_vars)
        self.params = dict(params)

        self.classes_r = {self.classes[class_name]: class_name
                          for class_name in self.classes}
        self.params_l = list(self.params.items())
        param_names = [param_name for param_name, _ in self.params_l] + \
            ['nest ' + str(i) for i in range(self.nestspec.num_nests - 1)]

        super().__init__(data, param_names, casadi_function_opts)

    def load_nestspec(self, params, endog_row, exog_row):
        """
        Writes the utilities and nestmods into self.nestspec.
        """
        self.nestspec.set_nest_mods(params, len(self.params))

        nestsets = [set() for _ in self.nestspec.nodes]
        for i in range(self.nestspec.num_classes + 1):
            nestsets[i].add(self.nestspec.nodes[i].name)
        for i in range(self.nestspec.num_classes + 1,
                       len(self.nestspec.nodes)):
            nestsets[i] = \
                set().union(*[nestsets[child.i]
                              for child in self.nestspec.nodes[i].children])

        utilities = [0. for i in range(len(self.nestspec.nodes))]
        for i in range(len(self.params)):
            for exog_name, pclass_names in self.params_l[i][1].items():
                if exog_name is None:
                    term = params[i]
                else:
                    ind = self.exog_name_to_i[exog_name]
                    term = params[i] * exog_row[ind]
                class_names = set(pclass_names)
                nodes = []
                for j in range(len(nestsets) - 1, 0, -1):
                    if nestsets[j].issubset(class_names):
                        nodes.append(j)
                        class_names -= nestsets[j]
                        if not class_names:
                            break
                for nodei in nodes:
                    utilities[nodei] += term

        self.nestspec.clear_data()
        for j in range(self.nestspec.num_classes + 1):
            class_name = self.nestspec.nodes[j].name
            av_var = False
            if class_name in self.availability_vars:
                av_exog_name = self.availability_vars[class_name]
                if av_exog_name is None:
                    av_var = True
                else:
                    av_var = exog_row[self.exog_name_to_i[av_exog_name]]
            count = 0.
            if class_name in self.classes:
                endog_name = self.classes[class_name]
                count = endog_row[self.endog_name_to_i[endog_name]]
            self.nestspec.set_data_on_classi(j, utilities[j], count, av_var)
        for j in range(self.nestspec.num_classes + 1, len(nestsets)):
            self.nestspec.set_utility_extra_on_nesti(j, utilities[j])
        self.nestspec.set_nest_data()

    def _loglike_casadi_sym(self):
        params = ad.SX.sym('p', self.num_params)
        endog_row = ad.SX.sym('y', self.endog_shape[1])
        exog_row = ad.SX.sym('x', self.exog_shape[1])
        self.load_nestspec(params, endog_row, exog_row)

        f = self.nestspec.loglike()
        H, g = ad.hessian(f, params)
        return params, endog_row, exog_row, f, g, H

    @cached_value
    def _probs_casadi(self):
        params = ad.SX.sym('p', self.num_params)
        exog_row = ad.SX.sym('x', self.exog_shape[1])
        self.load_nestspec(params, np.zeros(self.endog_shape[1]), exog_row)

        dout = {}
        # so the order is same as in endog
        for endog_name in self.endog_names:
            if endog_name in self.classes_r:
                class_name = self.classes_r[endog_name]
                if class_name in self.nestspec.class_name_to_i:
                    dout[endog_name] = \
                        self.nestspec.class_name_to_i[class_name]
        dout_l = list(dout.items())
        probs_sym = ad.SX.zeros(len(dout_l))
        for i in range(len(dout_l)):
            probs_sym[i] = self.nestspec.get_prob(dout_l[i][1])

        return (ad.Function('q', [params, exog_row], [probs_sym],
                            self.casadi_function_opts),
                [endog_name for endog_name, _ in dout_l])

    @property
    def endog_out_colnames(self):
        return self._probs_casadi[1]

    def predict(self, params, exog, which='mean', total_counts=None):
        """
        Note: column names of output are given by endog_out_colnames.
        """
        exog = np.asarray(exog)
        if total_counts is None:
            total_counts = np.broadcast_to(1., exog.shape[0])
        total_counts = total_counts.squeeze()
        assert total_counts.shape == (exog.shape[0],)
        params = np.array(params)
        total_counts = total_counts[:, None]
        probs = np.array(self._probs_casadi[0](params, exog.T)).T
        if which == 'mean':
            return total_counts * probs
        elif which == 'var':
            return total_counts * probs * (1 - probs)
        raise ValueError("which must be 'mean' or 'var'")

    def generate_endog(self, bitgen, params, exog, total_counts=None):
        """
        bitgen is an np.Generator.

        Note: column names of output are given by endog_out_colnames.
        """
        exog = np.asarray(exog)
        if total_counts is None:
            total_counts = np.broadcast_to(1., exog.shape[0])
        total_counts = total_counts.squeeze()
        probs = self.predict(params, exog)
        probs /= np.sum(probs, axis=1)[:, None] + 1e-14
        # TODO: numerical stability
        return random_multinomial(bitgen, total_counts, probs, exog.shape[0])

    @cached_value
    def default_start_params(self):
        return np.concatenate([np.zeros(len(self.params_l)),
                               np.ones(self.nestspec.num_nests - 1)])

    @cached_value
    def default_lbx(self):
        return np.concatenate([np.full(len(self.params_l), -np.inf),
                               np.full(self.nestspec.num_nests - 1, 0.1)])

    @cached_value
    def default_ubx(self):
        return np.concatenate([np.full(len(self.params_l), np.inf),
                               np.ones(self.nestspec.num_nests - 1)])

    def fit(self, start_params=None, lbx=None, ubx=None, constraints=None,
            ipopt_options={}):
        if start_params is None:
            start_params = self.default_start_params
        if lbx is None:
            lbx = self.default_lbx
        if ubx is None:
            ubx = self.default_ubx
        return super().fit(start_params, lbx, ubx, constraints, ipopt_options)

    def fit_null(self):
        # TODO: ACTUALLY DO PROPERLY
        return self._fit_result(self.default_start_params, 'lol')


class CustomModelBaseResults(smd.DiscreteResults):
    @cache_readonly
    def llnull(self):
        """
        For McFadden's pseudo-R^2
        """
        return self.model.fit_null().llf

    def summary(self):
        self.model.param_names_instead_of_exog_names = True
        result = super().summary()
        self.model.param_names_instead_of_exog_names = False
        return result

