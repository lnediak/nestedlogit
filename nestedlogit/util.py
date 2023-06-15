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

