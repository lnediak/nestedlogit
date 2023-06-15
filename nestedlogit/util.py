import casadi as ad
import numpy as np


def ad_logsumexp(a):
    if isinstance(a, ad.SX):
        # this function actually doesn't work as of CasADi 3.6.3
        # I'mma just use my own implementation
        if not a.is_column():
            raise ValueError("a is not a column vector")
        mx = ad.mmax(a)
        return mx + ad.log(ad.sum1(ad.exp(a - mx)))
    return ad.logsumexp(a)


class CasadiFunctionWrapper:
    def __init__(self, name, inputs, out, opts):
        """
        Note: inputs are all assumed to be column vectors, also len(out) == 1
        """
        assert len(out) == 1
        self.name = name
        self.inputs = inputs
        self.out = out[0]
        self.is_mx = isinstance(self.out, ad.MX)
        self.opts = opts
        self.f = ad.Function(name, inputs, out)
        self.f_dict = {}

    def __call__(self, *args):
        assert len(args) == len(self.inputs)
        args = list(args)
        for i in range(len(args)):
            if args[i].ndim < 2:
                args[i] = args[i][:, None]
            if args[i].shape[0] == 1:
                args[i] = args[i].T
            assert args[i].ndim == 2
        shape_all = tuple([arg.shape for arg in args])
        if shape_all in self.f_dict:
            return self.f_dict[shape_all](*args)
        inputs = [
            ad.MX.sym('x', arg.shape[0], arg.shape[1]) if self.is_mx else
            ad.MX.sym('x', arg.shape[0], arg.shape[1]) for arg in args]
        res = ad.sum2(self.f(*inputs))
        f = ad.Function(self.name, inputs, [res], self.opts)
        if self.opts.get('jit', False):
            if 'jit_options' in self.opts:
                if self.opts['jit_options'].get('verbose', False):
                    print('Finished compilation.')
        self.f_dict[shape_all] = f
        return f(*args)

    def size_out(self, i):
        assert i == 0
        return self.out.shape


def random_multinomial(bitgen, n, p, size):
    """
    Extends np.random.multinomial to be vectorized for different probabilities.
    Based on numpy random_multinomial.
    Arguably cursed.
    Only yields performance increase if there are many samples to be taken but
    not many values per sample.
    """
    # cursed, but I don't know a better way to parse size into a tuple
    size = np.broadcast_to(np.empty(1), size).shape

    out = np.full(size + (p.shape[-1],), 0)
    remaining_p = np.ones(p.shape[:-1])
    dn = n
    for j in range(p.shape[-1] - 1):
        out[..., j] = bitgen.binomial(dn, p[..., j] / remaining_p, size)
        dn -= out[..., j]
        remaining_p -= p[..., j]
    out[..., -1] = dn
    return out

