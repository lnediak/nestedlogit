import casadi as ad
import cyipopt
from functools import cached_property
import numpy as np
import pandas as pd
import scipy

from .util import random_multinomial
from .nestspec import NestSpec
from .results import ModelResults


class ModelBase:
    def __init__(self, data, default_params_dict, casadi_function_opts):
        self.param_names = list(default_params_dict)
        self.num_params = len(self.param_names)
        self.default_start_params = np.array(
            list(default_params_dict.values()), dtype=np.float64
        )
        self.default_lbx = np.full(self.num_params, -np.inf)
        self.default_ubx = np.full(self.num_params, np.inf)
        self.casadi_function_opts = casadi_function_opts
        self.data = data
        self.data_names = data.columns
        self.nobs = data.nobs
        self.data_name_to_i = {
            self.data_names[i]: i for i in range(len(self.data_names))
        }
        self.df_resid = self.nobs - self.num_params

    def params_arr(self, params, default_params=None):
        """
        If params is ndarray, returns flattened.
        Otherwise, treats params as dict-like, and fills missing with defaults.
        """
        if default_params is None:
            default_params = self.default_start_params
        if default_params.shape != (self.num_params,):
            raise ValueError(
                "Invalid shape for default_params: default_params.shape "
                f"({default_params.size}) != ({self.num_params},)"
            )
        if isinstance(params, np.ndarray):
            params = params.flatten()
            if len(params) != self.num_params:
                raise ValueError(
                    "Invalid size for params: params.size "
                    f"({params.size}) != {self.num_params}"
                )
            return params

        for name in params:
            if name not in self.param_names:
                warnings.warn(f"Invalid param name {name}")
        params_arr = default_params.copy()
        for idx, name in zip(range(self.num_params), self.param_names):
            try:
                params_arr[idx] = params[name]
            except KeyError:
                pass
        return params_arr

    def params_dict(self, params):
        return {self.param_names[i]: params[i] for i in range(self.num_params)}

    def _loglike_casadi_sym(self):
        """
        Returns (params,data_row,f,g,H), all these being SX.
        Note that this should be calculated for a single row only, that is,
        data_row should only have a single row, f should be scalar.
        """
        raise NotImplementedError

    @cached_property
    def _loglike_casadi_funcs(self):
        p, d, f, g, H = self._loglike_casadi_sym()
        H = ad.tril(H)
        args = [p, d]
        return (
            ad.Function("f", args, [f], self.casadi_function_opts),
            ad.Function("g", args, [g], self.casadi_function_opts),
            ad.Function(
                "Hnz",
                args,
                [H.get_nz(False, slice(None))],
                self.casadi_function_opts,
            ),
            H.sparsity(),
        )

    def _eval_casadi_function_on_data(self, f, params, out=None):
        """
        Assumption: f output is column vector or scalar
        """
        params = self.params_arr(params)
        if out is None:
            out = np.zeros(f.size_out(0)).squeeze()
        else:
            out[...] = 0.0
        outshape = out.shape
        out = np.atleast_1d(out.squeeze())
        for i in range(0, self.nobs, self.data.max_rows):
            nrows = min(self.data.max_rows, self.nobs - i)
            data_rows = self.data.get_data(i, i + nrows)
            # weird casadi calling convention
            out[...] += np.sum(f(params, data_rows.T).toarray(), axis=1)
        return out.reshape(outshape)

    def loglike(self, params):
        f, g, Hnz, Hsp = self._loglike_casadi_funcs
        return self._eval_casadi_function_on_data(f, params).item()

    def score(self, params, out=None):
        f, g, Hnz, Hsp = self._loglike_casadi_funcs
        return self._eval_casadi_function_on_data(g, params, out)

    def information(self, params, sparse_out=False, tril=False):
        return -self.hessian(params, sparse_out, tril)

    def hessian(self, params, sparse_out=False, tril=False):
        f, g, Hnz, Hsp = self._loglike_casadi_funcs
        Hdata = self._eval_casadi_function_on_data(Hnz, params)
        indptr, indices = Hsp.get_ccs()
        res = scipy.sparse.csc_matrix((Hdata.flatten(), indices, indptr))
        if not tril:
            res += res.T - scipy.sparse.diags(res.diagonal(), format="csc")
        return res if sparse_out else res.toarray()

    def _fit_result(self, params, convergence_msg):
        # TODO: USE convergence_msg
        Hval = self.hessian(params)
        try:
            Hinv = scipy.linalg.inv(-Hval)
        except scipy.linalg.LinAlgError:
            print("Information matrix:", -Hval)
            print("Eigenvalues:", scipy.linalg.eigh(-Hval)[0])
            raise
        return ModelResults(self, params, Hinv)

    def fit(self, start_params, lbx, ubx, ipopt_options={}):
        """
        lbx is lower bounds on parameters, ubx is upper bounds.
        """

        class LoglikeSpec:
            def __init__(self, model):
                self.model = model

            def objective(self, params):
                return -self.model.loglike(params)

            def gradient(self, params):
                return -self.model.score(params)

            def constraints(self, params):
                return np.empty(0)

            def jacobian(params):
                return np.empty((0, len(params)))

            def hessian(self, params, lagrange, obj_factor):
                H = -obj_factor * self.model.hessian(
                    params, sparse_out=True, tril=True
                )
                # + lagrange...
                return H.data

            def hessianstructure(self):
                Hsp = self.model._loglike_casadi_funcs[3]
                Hccs = Hsp.get_ccs()
                Hindptr = np.array(Hccs[0])
                Hcolindices = np.repeat(
                    np.arange(len(Hindptr) - 1), Hindptr[1:] - Hindptr[:-1]
                )
                Hrowindices = np.array(Hccs[1])
                return (Hrowindices, Hcolindices)

        nlp = cyipopt.Problem(
            n=self.num_params,
            m=0,
            problem_obj=LoglikeSpec(self),
            lb=self.params_arr(lbx, self.default_lbx),
            ub=self.params_arr(ubx, self.default_ubx),
            cl=np.empty(0),
            cu=np.empty(0),
        )
        for key, value in ipopt_options.items():
            nlp.add_option(key, value)
        p_opt, info = nlp.solve(self.params_arr(start_params).copy())
        return self._fit_result(np.array(p_opt).flatten(), info["status_msg"])

    def fit_null(self):
        """
        Fit with constant exog. For CustomModelBaseResults.llnull
        """
        raise NotImplementedError


class NestedLogitModel(ModelBase):
    """Nested Logit model."""

    def __init__(
        self,
        data,
        classes,
        nests={},
        availability_vars={},
        coefficients={},
        nest_params={},
        casadi_function_opts={},  # TODO: SET DECENT DEFAULT
    ):
        """
        - classes: dict, each entry is (class_name: endog_name).
          - If this is a list or tuple, uses {n: n for n in classes}.
        - nests: dict, each entry is (nest_name: class_or_nest_names).
        - availability_vars: dict, each entry is (class_name: exog_name).
            If exog_name is missing, assume said class is always available
        - coefficients: dict, each entry is (coef_name: spec), where spec is a
            dict where each entry is (exog_name: class_or_nest_names).
          - The classes the (coef_name*exog_name) term applies to will then be
            all the classes in the given class_or_nest_names plus all the
            classes in the nests in class_or_nest_names.
          - exog_name is None means intercept.
          - Note that if some class does not have any terms specified to apply
            to it, it will have a utility of 0 by default.
        - nest_params: dict, each entry is (nest_param_name: nest_names). If a
            nest is not found among nest_params.values(), it will
            automatically get a nest_param with the same name as itself. Thus,
            using nest_params={} generates an independent param for each nest.

        Note that dicts are ordered (since we're in Python 3.7+)
        """
        # TODO: FORMULAS INSTEAD OF JUST NAMES
        if isinstance(classes, (list, tuple)):
            classes = {n: n for n in classes}
        # TODO: ERROR CHECKING
        self.classes = dict(classes)
        self.nests = dict(nests)
        self.nestspec = NestSpec(self.nests, list(self.classes))
        self.availability_vars = dict(availability_vars)
        self.coefs = dict(coefficients)
        self.nest_params = dict(nest_params)
        self.nest_params = {
            **{
                n: [n]
                for n in set(self.nests)
                - set().union(*self.nest_params.values())
            },
            **self.nest_params,
        }

        self.classes_r = {
            self.classes[class_name]: class_name for class_name in self.classes
        }
        self.coefs_l = list(self.coefs.items())
        self.exog_names = [
            n for _, v in self.coefs_l for n in v if n is not None
        ]
        self.exog_names.extend(self.availability_vars.values())
        self.endog_names = list(self.classes_r)

        default_params_dict = {
            **{coef_name: 0 for coef_name, _ in self.coefs_l},
            **{nparam_name: 1 for nparam_name in self.nest_params},
        }
        super().__init__(data, default_params_dict, casadi_function_opts)

        self.default_lbx = self._concat_vals(-np.inf, 0.1)
        self.default_ubx = self._concat_vals(np.inf, 1.0)

    def _concat_vals(self, a, b):
        av = np.full(len(self.coefs_l), a)
        bv = np.full(len(self.nest_params), b)
        return np.concatenate([av, bv])

    def load_nestspec(self, params, data_row):
        """
        Writes the utilities and nestmods into self.nestspec.
        """
        self.nest_to_param_i = {
            nest: i
            for i, nest_names in zip(
                range(len(self.coefs_l), self.num_params),
                self.nest_params.values(),
            )
            for nest in nest_names
        }
        self.nestspec.set_nest_mods(
            [params[self.nest_to_param_i[nest]] for nest in self.nests]
        )

        def write_nestsets(node, nestsets):
            if node.children:
                for child in node.children:
                    nestsets[node.i] |= write_nestsets(child, nestsets)
                return nestsets[node.i]
            nestsets[node.i] = {node.name}
            return nestsets[node.i]

        nestsets = [set() for _ in self.nestspec.nodes]
        write_nestsets(self.nestspec.root, nestsets)

        def write_nodes(nodei, class_names, nodes):
            if nestsets[nodei].issubset(class_names):
                nodes.append(nodei)
                class_names -= nestsets[nodei]
                if not class_names:
                    return True
            for child in self.nestspec.nodes[nodei].children:
                if write_nodes(child.i, class_names, nodes):
                    return True
            return False

        utilities = [0.0 for _ in self.nestspec.nodes]
        for i in range(len(self.coefs_l)):
            for data_name, names_raw in self.coefs_l[i][1].items():
                if data_name is None:
                    term = params[i]
                else:
                    ind = self.data_name_to_i[data_name]
                    term = params[i] * data_row[ind]
                class_names = set().union(
                    *[
                        nestsets[self.nestspec.name_to_node[n].i]
                        for n in names_raw
                    ]
                )
                nodes = []
                write_nodes(self.nestspec.root.i, class_names, nodes)
                for nodei in nodes:
                    utilities[nodei] += term

        self.nestspec.clear_data()
        for j in range(self.nestspec.num_classes):
            class_name = self.nestspec.nodes[j].name
            av_var = True
            if class_name in self.availability_vars:
                av_data_name = self.availability_vars[class_name]
                av_var = data_row[self.data_name_to_i[av_data_name]]
            endog_name = self.classes[class_name]
            count = data_row[self.data_name_to_i[endog_name]]
            self.nestspec.set_data_on_classi(j, utilities[j], count, av_var)
        for j in range(self.nestspec.num_classes, len(nestsets)):
            self.nestspec.set_utility_extra_on_nesti(j, utilities[j])
        self.nestspec.eval_nest_data()

    def _loglike_casadi_sym(self):
        params = ad.SX.sym("p", self.num_params)
        data_row = ad.SX.sym("d", self.data.shape[1])
        self.load_nestspec(params, data_row)

        f = self.nestspec.loglike()
        H, g = ad.hessian(f, params)
        return params, data_row, f, g, H

    @cached_property
    def _probs_casadi(self):
        params = ad.SX.sym("p", self.num_params)
        exog_row = ad.SX.sym("x", len(self.exog_names))
        tmp = {self.exog_names[i]: i for i in range(len(self.exog_names))}
        data_row = [
            exog_row[tmp[n]] if n in tmp else 0.0 for n in self.data_names
        ]
        self.load_nestspec(params, data_row)

        dout = []
        # so the order is same as in data
        for endog_name in self.data_names:
            if endog_name in self.classes_r:
                class_name = self.classes_r[endog_name]
                node = self.nestspec.name_to_node[class_name]
                dout.append((endog_name, node.i))
        probs_sym = ad.SX.zeros(len(dout))
        for i in range(len(dout)):
            probs_sym[i] = self.nestspec.get_prob(dout[i][1])

        return (
            ad.Function(
                "q", [params, exog_row], [probs_sym], self.casadi_function_opts
            ),
            [endog_name for endog_name, _ in dout],
        )

    def predict(self, params, exog, which="mean", total_counts=None):
        """
        Note: column names of output are given by endog_out_colnames.
        """
        exog = np.asarray(exog)
        if total_counts is None:
            total_counts = np.broadcast_to(1.0, exog.shape[0])
        total_counts = total_counts.squeeze()
        assert total_counts.shape == (exog.shape[0],)
        params = self.params_arr(params)
        total_counts = total_counts[:, None]
        probs_func, columns = self._probs_casadi
        probs = np.array(probs_func(params, exog.T)).T
        if which == "mean":
            return pd.DataFrame(total_counts * probs, columns=columns)
        elif which == "var":
            return pd.DataFrame(
                total_counts * probs * (1 - probs), columns=columns
            )
        raise ValueError("which must be 'mean' or 'var'")

    def generate_endog(self, bitgen, params, exog, total_counts=None):
        """
        bitgen is an np.Generator.

        Note: column names of output are given by endog_out_colnames.
        """
        exog = np.asarray(exog)
        if total_counts is None:
            total_counts = np.broadcast_to(1.0, exog.shape[0])
        total_counts = total_counts.squeeze()
        probs = np.asarray(self.predict(params, exog))
        probs /= np.sum(probs, axis=1)[:, None] + 1e-14
        # TODO: numerical stability
        res = random_multinomial(bitgen, total_counts, probs, exog.shape[0])
        return pd.DataFrame(res, columns=self._probs_casadi[1])

    def fit(self, start_params={}, lbx={}, ubx={}, ipopt_options={}):
        return super().fit(start_params, lbx, ubx, ipopt_options)

    def fit_null(self, *args, **kwargs):
        cols = list(self.availability_vars.values()) + self.endog_names
        model = NestedLogitModel(
            self.data.subdata(cols),
            classes=self.classes,
            availability_vars=self.availability_vars,
            coefficients={
                cls: {None: [cls]} for cls in list(self.classes)[1:]
            },
        )
        return model.fit(*args, **kwargs)
