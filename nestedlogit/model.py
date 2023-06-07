import casadi as ad
import numpy as np
import scipy.linalg
import scipy.special

import statsmodels.base.model as smm
import statsmodels.discrete.discrete_model as smd
from statsmodels.tools.decorators import cached_value, cache_readonly


from .util import sx_eval, random_multinomial
from .nestspec import NestSpec


class CustomModelBase(smd.DiscreteModel):
    """
    This base class uses a Casadi SXFunction for the log-likelihood.
    """

    def __init__(self, endog, exog, **kwargs):
        super().__init__(endog, exog, **kwargs)
        self.param_names_instead_of_exog_names = False
        # TODO: FIGURE OUT WHAT THE HELL TO DO WITH df_model and k_extra

    @property
    def exog_names(self):
        """
        self.param_names_instead_of_exog_names exists because statsmodels
        summary uses exog_names for names of params
        """
        if self.param_names_instead_of_exog_names:
            return self.param_names
        return super().exog_names

    def loglike_casadi_sx(self):
        """
        Returns [p,f,g,H] where p,f,g,H are all SX
        """
        # TODO: individual functions, use ipyopt or something
        raise NotImplementedError

    @cached_value
    def loglike_casadi_sx_(self):
        return self.loglike_casadi_sx()

    def loglike(self, params):
        p, f, g, H = self.loglike_casadi_sx_
        return sx_eval(f, p, params).flatten()

    def score(self, params):
        p, f, g, H = self.loglike_casadi_sx_
        return sx_eval(g, p, params).flatten()

    def information(self, params):
        return -self.hessian(params)

    def hessian(self, params):
        p, f, g, H = self.loglike_casadi_sx_
        return sx_eval(H, p, params)

    def fit_res(self, params):
        p, f, g, H = self.loglike_casadi_sx_
        Hval = sx_eval(H, p, params)
        try:
            Hinv = np.linalg.inv(-Hval)
        except np.linalg.LinAlgError:
            print(-Hval)
            print(scipy.linalg.eigh(-Hval)[0])
            np.linalg.inv(-Hval)
        # TODO: OTHER OPTIONS
        mlefit = smm.LikelihoodModelResults(self, params, Hinv, scale=1.)
        mlefit.mle_retvals = {'converged': True}

        # TODO: THIS SHOULD ACTUALLY BE IN THE CHILD CLASS
        results = CustomModelBaseResults(self, mlefit)
        # any modification to results necessary?
        return results

    def fit(self, start_params, lbx=-10000., ubx=10000., constraints=None,
            options={}):
        """
        lbx is lower bounds on parameters, ubx is upper bounds.
        """
        # TODO: CONSTRAINTS
        p, f, g, H = self.loglike_casadi_sx_
        S = ad.nlpsol('S', 'ipopt', {'x': p, 'f': -f}, options)
        p_opt = S(x0=start_params, lbx=lbx, ubx=ubx)['x']
        # TODO: FIND HOW THE HELL TO FIND EXIT STATUS

        return self.fit_res(np.array(p_opt).flatten())

    def fit_null(self):
        """
        Fit with constant exog. For CustomModelBaseResults.llnull
        """
        raise NotImplementedError


class NestedLogitModel(CustomModelBase):
    """
    Nested Logit model.
    """

    def __init__(self, endog, exog, classes, nests, availability_vars, varz,
                 params, **kwargs):
        """
        classes: dict, each entry is (class_name: endog_name),
            where the null class has a class_name of 0.
        nests: List of lists, to be passed to NestSpec.
        availability_vars: dict, each entry is (class_name: exog_name).
            If exog_name is None, assume said class is always available
            If exog_name is missing, assume said class is never available
        varz: dict, each entry is (exog_name: class_names),
            where the given variable applies only to classes in class_names.
        params: dict, each entry is (name: spec), where spec is a dict
            where each entry is (exog_name: class_names).
            exog_name is None means intercept

        Note that as of Python 3.7, dicts are ordered
        """
        super().__init__(endog, exog, **kwargs)
        self.exog_name_to_i = \
            {self.exog_names[i]: i for i in range(len(self.exog_names))}
        self.endog_name_to_i = \
            {self.endog_names[i]: i for i in range(len(self.endog_names))}

        # TODO: ERROR CHECKING
        self.classes = dict(classes)
        self.nestspec = NestSpec(nests)
        self.availability_vars = dict(availability_vars)
        self.varz = dict(varz)
        self.params = dict(params)

        self.classes_r = {self.classes[class_name]: class_name
                          for class_name in self.classes}
        self.availability_vars_l = list(self.availability_vars.items())
        self.availability_var_to_i = \
            {self.availability_vars_l[i][0]: i
             for i in range(len(self.availability_vars_l))}
        self.varz_l = list(self.varz.items())
        self.params_l = list(self.params.items())
        self.param_names = [param_name for param_name, _ in self.params_l] + \
            ['nest ' + str(i) for i in range(self.nestspec.num_nests - 1)]

    @cached_value
    def default_start_params(self):
        return np.concatenate([np.zeros(len(self.params_l)),
                               np.ones(self.nestspec.num_nests - 1)])

    # writes the utilities and everything into self.nestspec
    def load_nestspec(self, params, varz, av_vars, counts):
        self.nestspec.set_nest_mods(params, len(self.params))

        nestsets = [set() for _ in self.nestspec.nodes]
        for i in range(self.nestspec.num_classes + 1):
            nestsets[i].add(self.nestspec.nodes[i].name)
        for i in range(self.nestspec.num_classes + 1,
                       len(self.nestspec.nodes)):
            nestsets[i] = \
                set().union(*[nestsets[child.i]
                              for child in self.nestspec.nodes[i].children])

        utilities = ad.SX.zeros(len(self.nestspec.nodes))
        exog_name_to_vi = {self.varz_l[i][0]: i for i in range(len(self.varz))}
        for i in range(len(self.params)):
            for exog_name, pclass_names in self.params_l[i][1].items():
                if exog_name is None:
                    term = params[i]
                    class_names = set(pclass_names)
                else:
                    vind = exog_name_to_vi[exog_name]
                    term = params[i] * varz[vind]
                    vclass_names = self.varz_l[vind][1]
                    class_names = set(pclass_names) & set(vclass_names)
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
                if self.availability_vars[class_name] is None:
                    av_var = True
                else:
                    av_var = av_vars[self.availability_var_to_i[class_name]]
            self.nestspec.set_data_on_classi(
                j, utilities[j], counts[j], av_var)
        for j in range(self.nestspec.num_classes + 1, len(nestsets)):
            self.nestspec.set_utility_extra_on_nesti(j, utilities[j])
        self.nestspec.set_nest_data()

    def _stack_all_sx(self, stack_counts):
        params = ad.SX.sym('p', len(self.params) + self.nestspec.num_nests - 1)
        varz = ad.SX.sym('x', len(self.varz))
        av_vars = ad.SX.sym('z', len(self.availability_vars))
        counts = ad.SX.sym('n', self.nestspec.num_classes + 1)
        return (params, varz, av_vars, counts,
                [params] +
                [varz[i] for i in range(varz.shape[0])] +
                [av_vars[i] for i in range(av_vars.shape[0])] +
                ([counts[i] for i in range(counts.shape[0])] if stack_counts
                 else []))

    def _stack_all_vals(self, params, exog, endog=None):
        """
        Counterpart to _stack_all_sx providing values corresponding to the list
        of symbolic variables returned
        """
        return ([params] +
                [exog[:, self.exog_name_to_i[exog_name]][None, :]
                 for exog_name, _ in self.varz_l] +
                [True if exog_name is None else
                 exog[:, self.exog_name_to_i[exog_name]][None, :]
                 for _, exog_name in self.availability_vars_l] +
                ([] if endog is None else
                 [endog[:, self.endog_name_to_i[
                     self.classes[self.nestspec.nodes[i].name]]][None, :]
                  if self.nestspec.nodes[i].name in self.classes else 0.
                  for i in range(self.nestspec.num_classes + 1)]))

    def loglike_casadi_sx(self):
        params, varz, av_vars, counts, stacked_args = self._stack_all_sx(True)
        self.load_nestspec(params, varz, av_vars, counts)

        loglike_sx = self.nestspec.loglike()
        loglike_f = ad.Function('f', stacked_args, [loglike_sx])

        stacked_args_vals = self._stack_all_vals(params, self.exog, self.endog)
        ll_elem = loglike_f(*stacked_args_vals)
        f = ad.sum2(ll_elem)
        H, g = ad.hessian(f, params)
        return params, f, g, H

    @cached_value
    def _probs_casadi_func(self):
        params, varz, av_vars, counts, stacked_args = self._stack_all_sx(False)
        self.load_nestspec(params, varz, av_vars, counts)

        probs_sx = ad.SX.sym('q', self.nestspec.num_classes + 1)
        for i in range(self.nestspec.num_classes + 1):
            probs_sx[i] = self.nestspec.get_prob(i)

        return ad.Function('q', stacked_args, [probs_sx])

    def predict(self, params, exog=None, which='mean', total_counts=None):
        if exog is None:
            exog = self.exog
        if total_counts is None:
            total_counts = np.full(exog.shape[0], 1)
        assert total_counts.shape == (exog.shape[0],)
        params = np.array(params)
        total_counts = total_counts[:, None]
        stacked_args_vals = self._stack_all_vals(params, self.exog)
        probs = np.array(self._probs_casadi_func(*stacked_args_vals)).T
        if which == 'mean':
            return total_counts * probs
        elif which == 'var':
            return total_counts * probs * (1 - probs)
        raise ValueError("which must be 'mean' or 'var'")

    # bitgen is a np.Generator, returns dict as passed to pandas DataFrame
    def generate_endog(self, bitgen, params, exog=None, total_counts=None):
        if exog is None:
            exog = self.exog
        exog = np.asarray(exog)
        if total_counts is None:
            total_counts = np.full(exog.shape[0], 1)
        probs = self.predict(params, exog, total_counts=total_counts)
        probs /= np.sum(probs, axis=1)[:, None] + 1e-14
        # TODO: numerical stability
        out = random_multinomial(bitgen, total_counts, probs, exog.shape[0])
        dout = {}
        # so the order is same as in endog
        for endog_name in self.endog_names:
            if endog_name in self.classes_r:
                class_name = self.classes_r[endog_name]
                if class_name in self.nestspec.class_name_to_i:
                    i = self.nestspec.class_name_to_i[class_name]
                    dout[endog_name] = out[:, i]
        return dout

    def fit_null(self):
        # TODO: ACTUALLY DO PROPERLY
        return self.fit_res(self.default_start_params)


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

