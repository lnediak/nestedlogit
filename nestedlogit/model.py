import casadi as ad
import numpy as np
import scipy.linalg
import scipy.special

import statsmodels.base.model as smm
import statsmodels.discrete.discrete_model as smd
from statsmodels.tools.decorators import cached_value, cache_readonly


from .util import sx_eval
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
        if self.param_names_instead_of_exog_names:
            return self.param_names
        return super().exog_names

    def loglike_casadi_sx(self):
        """
        Returns [p,f,g,H] where p,f,g,H are all SX
        """
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
            print(scipy.linalg.eigh(-Hval))
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
        self.varz_l = list(self.varz.items())
        self.params = dict(params)
        self.params_l = list(self.params.items())
        self.param_names = [param_name for param_name, _ in self.params_l] + \
            ['nest ' + str(i) for i in range(self.nestspec.num_nests - 1)]

    def loglike_casadi_sx(self):
        p = ad.SX.sym('p', len(self.params) + self.nestspec.num_nests - 1)
        self.nestspec.set_nest_mods(p, len(self.params))

        nestsets = [set() for _ in self.nestspec.nodes]
        for i in range(self.nestspec.num_classes + 1):
            nestsets[i].add(self.nestspec.nodes[i].name)
        for i in range(self.nestspec.num_classes + 1,
                       len(self.nestspec.nodes)):
            nestsets[i] = \
                set().union(*[nestsets[child.i]
                              for child in self.nestspec.nodes[i].children])

        sx_u = ad.SX.zeros(len(self.nestspec.nodes))
        sx_vars = ad.SX.sym('x', len(self.varz))
        exog_name_to_vi = {self.varz_l[i][0]: i for i in range(len(self.varz))}
        for i in range(len(self.params)):
            for exog_name, pclass_names in self.params_l[i][1].items():
                if exog_name is None:
                    term = p[i]
                    class_names = set(pclass_names)
                else:
                    vind = exog_name_to_vi[exog_name]
                    term = p[i] * sx_vars[vind]
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
                    sx_u[nodei] += term
        sxfunc_u = ad.Function('u', [sx_vars, p], [sx_u])

        f = 0.
        # we only need to do this once, so who cares if it is slow
        for i in range(self.exog.shape[0]):
            vars_vals = np.array([self.exog[i][self.exog_name_to_i[exog_name]]
                                  for exog_name, _ in self.varz_l])
            utilities = sxfunc_u(vars_vals, p)
            self.nestspec.clear_data()
            some_set = False
            for j in range(self.nestspec.num_classes + 1):
                class_name = self.nestspec.nodes[j].name
                if class_name in self.availability_vars:
                    av_var = self.availability_vars[class_name]
                    if av_var is None or \
                            self.exog[i][self.exog_name_to_i[av_var]]:
                        some_set = True
                        self.nestspec.set_data_on_classi(
                            j, utilities[j],
                            self.endog[i][self.endog_name_to_i[
                                self.classes[class_name]]])
            if not some_set:
                continue
            for j in range(self.nestspec.num_classes + 1, len(nestsets)):
                self.nestspec.set_utility_extra_on_nesti(j, utilities[j])
            self.nestspec.set_nest_data()
            f += self.nestspec.loglike()

        H, g = ad.hessian(f, p)
        return p, f, g, H

    def fit_null(self):
        # TODO: ACTUALLY DO PROPERLY
        return self.fit_res(
            np.zeros(len(self.params) + self.nestspec.num_nests - 1))


class CustomModelBaseResults(smd.DiscreteResults):
    @ cache_readonly
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
