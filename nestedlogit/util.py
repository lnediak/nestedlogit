import casadi as ad
import numpy as np


def sx_eval(f, x, xval):
    return np.array(ad.Function('f', [x], [f])(xval))


def ad_logsumexp(a, b=None):
    if isinstance(a, (ad.DM, ad.MX, ad.SX)):
        # this function actually doesn't work as of CasADi 3.6.3
        # I'mma just use my own implementation
        # return ad.logsumexp(a)
        if not a.is_column():
            raise ValueError("a is not a column vector")
        mx = ad.mmax(a)
        if b is None:
            return mx + ad.log(ad.sum1(ad.exp(a - mx)))
        return mx + ad.log(ad.sum1(b * ad.exp(a - mx)))
    return scipy.special.logsumexp(a, b=b)

