import numpy as np

import _C_logitthing

def solve(options, t, i, x, n):
    assert isinstance(options, str)
    assert isinstance(t, np.ndarray);
    assert isinstance(i, np.ndarray);
    assert isinstance(x, np.ndarray);
    assert isinstance(n, np.ndarray);
    # XXX: generalize these?
    assert t.dtype == np.int64
    assert i.dtype == np.int64
    assert n.dtype == np.int64
    assert x.dtype == np.double
    assert len(t.shape) == 1
    assert len(i.shape) == 1
    assert len(x.shape) == 2
    assert len(n.shape) == 1
    sz = t.shape[0]
    assert sz == i.shape[0]
    assert sz == x.shape[0]
    assert sz == n.shape[0]

    # XXX: "factorize" i
    assert len(i[i < 0]) == 0
    isz = np.amax(i);
    return _C_logitthing.solve(options, isz, t, i, x, n)

