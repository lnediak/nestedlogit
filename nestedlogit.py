import numpy as np
import scipy.special
import statsmodels.base.model as smm
import statsmodels.discrete.discrete_model as smd
from statsmodels.tools.decorators import cached_value

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
                self.i = a

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
        while self.nodes[i].parent:
            p = self.nodes[i].parent
            if id(p) not in seen:
                self.nodes.append(p)
                seen.add(id(p))
            i += 1
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
        voff is set so that arr[num_classes + 1 - voff] is mod for first nest.
        """
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
                u = ad.SX.zeros(len(node.children))
                for i in range(len(node.children)):
                    u[i] = handle_node(node.children[i])
                    node.count += node.children[i].count
                node.utility = ad_logsumexp(u)
                node.is_valid = True
                toret = node.utility * node.nest_mod
                return node.utility_extra ? toret + node.utility_extra: toret
            return node.utility / node.nest_mod
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


class CasadiLoglikeModel(smd.DiscreteModel):
    def pdf(self, X):
        raise NotImplementedError

    def cdf(self, X):
        raise NotImplementedError

    @cached_value
    def loglike_casadi_sx(self):
        """
        Returns [p,f,g,H] where f,g,H are SXFunctions and p is SX
        """
        raise NotImplementedError

    def loglike(self, params):
        return self.loglike_casadi_sx()[1](params)

    def score(self, params):
        return self.loglike_casadi_sx()[2](params)

    def information(self, params):
        return -self.hessian(params)

    def hessian(self, params):
        return self.loglike_casadi_sx()[3](params)

    def fit(self, start_params, constraints=None):
        # TODO: CONSTRAINTS DO
        p, f, g, H = self.loglike_casadi_sx()
        S = ad.nlpsol('S', 'ipopt', {'x0': p, 'f': -f})
        p_opt = S(x0=start_params)['x']

        Hinv = np.linalg.inv(-H(p))
        # TODO: OTHER OPTIONS
        mlefit = smm.LikelihoodModelResults(self, p_opt, Hinv, scale=1.)

        # any modification to mlefit necessary?
        return mlefit


class CasadiLoglikeModelResults(smd.DiscreteResults):
    @cache_readonly
    def llnull(self):
        # TODO: statsmodels implementation here calls fit, which we've changed
        raise NotImplementedError

