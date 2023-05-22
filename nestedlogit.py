import numpy as np
import scipy.special
import statsmodels.base.model as smm
import statsmodels.discrete.discrete_model as smd
from statsmodels.tools.decorators import cached_value, cache_readonly

import casadi as ad


def ad_logsumexp(a):
    if isinstance(a, (ad.DM, ad.MX, ad.SX)):
        return ad.logsumexp(a)
    return scipy.special.logsumexp(a)


class NestSpec:
    """
    Describes a specification of nests.
    """

    class NestNode:
        parent = None  # means root
        children = []  # empty means leaf

        # for a nest, this is the "utility" before exponentiating nest mod
        utility = 0.

        i = 0  # class or nest index
        is_valid = False  # from exog if leaf

        # nest_mod is n / p where p is exponent (from params) of parent nest,
        # n is exponent (from params) of current nest. These values are 1 if
        # they do not exist, it is just p for leafs, and is 0 for root node.
        nest_mod = 0.

        # leaf-only:
        name = 0  # class name
        count = 0  # from endog
        # nest-only:
        utility_extra = None  # utility from exog vals for the nest, if any

        def __init__(self, a=[]):
            try:
                self.children = list(a)
            except TypeError:
                self.name = a

    root = NestNode([NestNode(0)])  # root node
    nodes = [root[0]]
    class_name_to_i = {0: 0}
    num_classes = 0  # null class not included
    num_nests = 1  # just the root

    def _finish_init(self):
        def act_on_node(node, parent, lst):
            node.parent = parent
            if node.children:
                toret = 1
                for child in node.children:
                    toret += act_on_node(child, node, lst)
                return toret
            lst.append(node)
            return 0
        self.nodes = []
        self.num_nests = act_on_node(self.root, None, self.nodes)
        # null class node is first because of how nodes are added to root
        class_names = [leaf.name for leaf in self.nodes[1:]]
        self.num_classes = len(class_names)
        if len(set(class_names)) != len(class_names):
            raise ValueError("Same class in multiple nests")
        for i in range(1, len(self.nodes)):
            self.nodes[i].i = i
            class_name_to_i[self.nodes[i].name] = i
        i = 0
        seen = set()
        # this way, nodes on the same level end up together
        nodesr = [root]
        while i < len(nodesr):
            for child in reversed(nodesr[i].children):
                if child.children and id(child) not in seen:
                    nodesr.append(child)
                    seen.add(id(child))
            i += 1
        self.nodes += reversed(nodesr)
        for i in range(self.num_classes + 1, len(self.nodes)):
            self.nodes[i].i = i

    def __init__(self):
        self._finish_init()

    def __init__(self, lst, null_class_name=0):
        """
        Expects nests to be represented as iterables in an iterable.
        Example: lst = [[2, 4], [[3, 6], 1], 5]
        """
        def set_node_to_list(node, lst):
            try:
                for ele in lst:
                    node.children.append(NestNode())
                    set_node_to_list(node.children[-1], ele)
            except TypeError:
                node.name = lst

        for lst_needs_to_be_an_iterable in lst:
            pass
        self.root.children[0].name = null_class_name
        set_node_to_list(self.root, lst)
        self._finish_init()

    def clear_data(self):
        for node in self.nodes:
            node.is_valid = False

    def set_data_on_classi(self, i, utility, count):
        self.nodes[i].utility = utility
        self.nodes[i].count = count
        self.nodes[i].is_valid = True

    def set_utility_extra_on_nesti(self, i, utility_extra):
        self.nodes[i].utility_extra = utility_extra

    def set_nest_mods(self, arr, voff):
        """
        voff is offset in arr (arr[voff] is mod for first nest).
        """
        # set so that arr[num_classes + 1 - voff] is mod for first nest.
        voff = self.num_classes + 1 - voff

        self.nodes[0].nest_mod = 1.
        self.root.nest_mod = 0.
        for i in range(1, len(self.nodes) - 1):
            if self.nodes[i].parent != self.root:
                if i > self.num_classes:
                    nestMods[i] = arr[i - voff] / \
                        arr[self.nodes[i].parent.i - voff]
                else:
                    nestMods[i] = arr[self.nodes[i].parent.i - voff]
            else:
                if i > self.num_classes:
                    nestMods[i] = arr[i - voff]
                else:
                    nestMods[i] = 1.

    def set_nest_data(self):
        def handle_node(node):
            if node.children:
                node.count = 0
                ul = []
                for child in node.children:
                    is_valid, u = handle_node(node.children[i])
                    if is_valid:
                        node.count += node.children[i].count
                        ul.append(u)
                if not ul:
                    node.is_valid = False
                    return False, 0.
                u = ad.SX.zeros(len(ul))
                for i in range(len(ul)):
                    u[i] = ul[i]
                node.utility = ad_logsumexp(u)
                node.is_valid = True
                toret = node.utility * node.nest_mod
                return True, \
                    node.utility_extra ? toret + node.utility_extra: toret
            if node.is_valid:
                return True, node.utility / node.nest_mod
            return False, 0.
        handle_node(root)

    def loglike(self):
        toret = 0
        for node in self.nodes:
            if node.is_valid:
                if node.children:
                    # recall self.root.nest_mod == 0
                    toret += node.count * (node.nest_mod - 1) * node.utility
                else:
                    toret += node.count * node.utility / node.nest_mod
        return toret

    def log_prob(self, i):
        node = self.nodes[i]
        if not node.is_valid:
            return 0.
        toret = node.utility / node.nest_mod
        while node.parent:
            node = node.parent
            toret += (node.nest_mod - 1) * node.utility
        return toret


def _sx_eval(f, x, xval):
    return ad.Function('f', [x], [f])(xval)


class CustomModelBase(smd.DiscreteModel):
    """
    This base class uses a Casadi SXFunction for the log-likelihood.
    """

    def __init__(self, endog, exog, **kwargs):
        super().__init__(endog, exog, **kwargs)
        # TODO: FIGURE OUT WHAT THE HELL TO DO WITH df_model and k_extra

    @cached_value
    def loglike_casadi_sx_(self):
        return self.loglike_casadi_sx()

    def loglike_casadi_sx(self):
        """
        Returns [p,f,g,H] where p,f,g,H are all SX
        """
        raise NotImplementedError

    def loglike(self, params):
        p, f, g, H = self.loglike_casadi_sx_()
        return _sx_eval(f, p, params)

    def score(self, params):
        p, f, g, H = self.loglike_casadi_sx_()
        return _sx_eval(g, p, params)

    def information(self, params):
        return -self.hessian(params)

    def hessian(self, params):
        p, f, g, H = self.loglike_casadi_sx_()
        return _sx_eval(H, p, params)

    def fit(self, start_params, lbx=-np.inf, ubx=np.inf, constraints=None):
        """
        lbx is lower bounds on parameters, ubx is upper bounds.
        """
        # TODO: CONSTRAINTS
        p, f, g, H = self.loglike_casadi_sx_()
        S = ad.nlpsol('S', 'ipopt', {'x': p, 'f': -f})
        p_opt = S(x0=start_params, 'lbx'=lbx, 'ubx'=ubx)['x']

        Hinv = np.linalg.inv(-_sx_eval(H, p, p_opt))
        # TODO: OTHER OPTIONS
        mlefit = smm.LikelihoodModelResults(self, p_opt, Hinv, scale=1.)

        # any modification to mlefit necessary?
        return mlefit

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

    def loglike_casadi_sx():
        p = ad.SX('p', len(self.params) + self.nestspec.num_nests)
        self.nestspec.set_nest_mods(p, len(self.params))

        nestsets = [set()] * len(self.nestspec.nodes)
        for i in range(self.nestspec.num_classes + 1):
            nestsets[i].add(i)
        for i in range(self.nestspec.num_classes + 1,
                       len(self.nestspec.nodes)):
            nestsets[i] = \
                set().union(*[nestsets[child.i]
                              for child in self.nestspec.nodes[i].children])

        sx_u = ad.SX.zeros(len(self.nestspec.nodes))
        sx_vars = ad.SX('x', len(self.varz))
        exog_name_to_vi = {self.varz_l[i][0]: i for i in range(len(self.varz))}
        for i in range(len(self.params)):
            for exog_name, pclass_names in self.params_l[i][1]:
                if exog_name is None:
                    term = p[i]
                    class_names = pclass_names
                else:
                    vind = exog_name_to_vi[exog_name]
                    term = p[i] * sx_vars[vind]
                    vclass_names = self.varz_l[vind][1]
                    class_names = pclass_names & vclass_names
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
            for i in range(self.nestspec.num_classes + 1):
                class_name = self.nestspec.nodes[i].name
                if self.exog[i][self.exog_name_to_i[
                        self.availability_vars[class_name]]]:
                    some_set = True
                    self.nestspec.set_data_on_classi(
                        i, utilities[i], self.endog[i][self.endog_name_to_i[
                            self.classes[class_name]]])
            if not some_set:
                continue
            for i in range(self.nestspec.num_classes + 1, len(nestsets)):
                self.nestspec.set_utility_extra_on_nesti(i, utilities[i])
            self.nestspec.set_nest_data()
            f += self.nestspec.loglike()

        return p, f, ad.gradient(f, p), ad.hessian(f, p)


class CustomModelBaseResults(smd.DiscreteResults):
    @cache_readonly
    def llnull(self):
        """
        For McFadden's pseudo-R^2
        """
        return self.model.fit_null().llf

