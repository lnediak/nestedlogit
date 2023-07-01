import casadi as ad
import numpy as np
import warnings

from .util import ad_logsumexp


class NestNode:
    parent = None  # means root
    children = None  # empty means leaf

    # for a nest, this is the "utility" before exponentiating nest mod
    utility = 0.0

    i = 0  # class or nest index
    is_valid = False  # from exog if leaf

    # nest_mod is n / p where p is exponent (from params) of parent nest,
    # n is exponent (from params) of current nest. These values are 1 if
    # they do not exist, it is just p for leafs, and is 0 for root node.
    nest_mod = 1.0

    name = None  # class name
    count = 0.0  # from endog (for leaves; for nests it's sum of children)

    # nest-only:
    utility_extra = 0.0  # utility from exog vals for the nest, if any

    def __init__(self, name=None, children=[]):
        self.name = name
        self.children = list(children)

    def __repr__(self):
        return "NestNode(" + ", ".join(
            [
                f"{f}={getattr(self, f)}"
                for f in NestNode.__dict__
                if not f.startswith("__")
            ]
        )


class NestSpec:
    """
    Describes a specification of nests.
    """

    def _basic_init(self):
        self.root = NestNode()
        self.nodes = [self.root]
        self.name_to_node = {None: self.root}
        self.num_classes = 0
        self.num_nests = 1  # just the root

    def _finish_init(self):
        def act_on_node(node, parent):
            node.parent = parent
            if node.count:
                raise ValueError(
                    f"{node.name} found in multiple nests or part of a cycle"
                )
            node.count = 1  # temp mark
            if len(node.children) == 1:
                warnings.warn(f"{node.children[0].name} has no siblings")
            for child in node.children:
                act_on_node(child, node)

        act_on_node(self.root, None)
        for i in range(len(self.nodes)):
            self.nodes[i].count = 0.0
            self.nodes[i].i = i

    def __init__(self):
        self._basic_init()
        self._finish_init()

    def _add_node(self, node):
        """Literally only for __init__"""
        self.nodes.append(node)
        self.name_to_node[node.name] = node

    def __init__(self, nests_dict, class_names):
        """
        nests_dict: same format as nests as passed to NestedLogitModel
        class_names: self-explanatory
        """
        self._basic_init()
        self.num_classes = len(class_names)
        self.num_nests = len(nests_dict) + 1  # for the root

        self.nodes = []  # so we can put root at the end lol
        for name in class_names:
            if not isinstance(name, str) and not isinstance(name, int):
                raise ValueError(
                    f"{name} has invalid type for class name "
                    "(is not str or int)"
                )
            self._add_node(NestNode(name=name))
        s_class_names = set(class_names)
        if len(s_class_names) != len(class_names):
            raise ValueError("Duplicates in class_names")
        free_classes = {n: None for n in class_names}  # for ordering
        for name, child_names in nests_dict.items():
            if not isinstance(name, str):
                raise ValueError(
                    f"{name} has invalid type for nest name (is not str)"
                )
            if name in s_class_names:
                raise ValueError(
                    f"{name} was listed as a nest name and a class name"
                )
            child_names = list(child_names)
            if not child_names:
                raise ValueError(f"{name} has no children")
            self._add_node(NestNode(name=name, children=child_names))
            for child in child_names:
                if child in s_class_names:
                    del free_classes[child]
        self.root.children = list(free_classes) + list(nests_dict)
        self._add_node(self.root)

        # now replace our names with actual nodes in our children lists
        for i in range(self.num_classes, len(self.nodes)):
            node = self.nodes[i]
            for j in range(len(node.children)):
                name = node.children[j]
                if not isinstance(name, str) and not isinstance(name, int):
                    raise ValueError(
                        f"{name} has invalid type for class/nest name "
                        "(is not str or int)"
                    )
                if name not in self.name_to_node:
                    raise ValueError(f"{name} not found in nests or classes")
                node.children[j] = self.name_to_node[name]
        self._finish_init()

    def clear_data(self):
        for node in self.nodes:
            node.is_valid = False
            node.utility = 0.0
            node.count = 0.0
            node.utility_extra = 0.0

    def set_data_on_classi(self, i, utility, count, is_valid=True):
        self.nodes[i].utility = utility
        self.nodes[i].count = count
        self.nodes[i].is_valid = is_valid

    def set_utility_extra_on_nesti(self, i, utility_extra):
        self.nodes[i].utility_extra = utility_extra

    def set_nest_mods(self, arr):
        """
        arr is aligned against self.nodes so arr[0] is mod for first nest
        """
        off = self.num_classes

        self.root.nest_mod = 0.0
        for i in range(len(self.nodes) - 1):
            node = self.nodes[i]
            if node.parent != self.root:
                if node.children:
                    node.nest_mod = arr[i - off] / arr[node.parent.i - off]
                else:
                    node.nest_mod = arr[node.parent.i - off]
            else:
                if node.children:
                    node.nest_mod = arr[i - off]
                else:
                    node.nest_mod = 1.0

    def eval_nest_data(self):
        """Evaluates .utility, .count, and .is_valid of all nest nodes."""

        def handle_node(node):
            if node.children:
                node.count = 0
                m_is_valid = 1.0
                ul = []
                for child in node.children:
                    is_valid, u = handle_node(child)
                    node.count += is_valid * child.count
                    ul.append((is_valid, u))
                    m_is_valid *= 1 - is_valid
                u = ad.SX.zeros(len(ul))
                z = ad.SX.zeros(len(ul))
                for i in range(len(ul)):
                    u[i] = ul[i][1]
                    z[i] = ul[i][0]
                node.utility = ad_logsumexp(ad.log(z) + u)
                node.is_valid = 1.0 - m_is_valid
                toret = node.utility * node.nest_mod
                return node.is_valid, toret + node.utility_extra
            return node.is_valid, node.is_valid * node.utility / node.nest_mod

        handle_node(self.root)

    def loglike(self):
        """
        Before using loglike or get_prob:
        - call clear_data (if necessary)
        - call set_data_on_classi for each
        - call set_utility_extra_on_nesti for each
        - call set_nest_mods
        - then call eval_nest_data (which uses all the above)
        """
        toret = 0
        for node in self.nodes:
            if node.children:
                # recall self.root.nest_mod == 0
                tmp = node.count * (node.nest_mod - 1) * node.utility
                toret += node.is_valid * tmp
            else:
                tmp = node.count * node.utility / node.nest_mod
                toret += node.is_valid * tmp
        return toret

    def get_prob(self, i):
        node = self.nodes[i]
        toret = node.utility / node.nest_mod
        while node.parent:
            node = node.parent
            toret += (node.nest_mod - 1) * node.utility
        return self.nodes[i].is_valid * ad.exp(toret)
