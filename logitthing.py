import numpy as np

import _C_logitthing

def _parseNS0(ns, out, sz, lvl, pari):
    if (len(out) <= lvl):
        out.append([0])
        sz[0] += 1
    if not isinstance(ns, list):
        out[lvl].append(pari)
        out[lvl][0] += 1
        sz[0] += 1
        return lvl
    out[lvl].append(pari)
    out[lvl][0] += 1
    sz[0] += 1
    res = -1
    atleast2nests = False
    for sub in ns:
        tmp = _parseNS0(sub, out, sz, lvl + 1, len(out[lvl]) - 1)
        if res < 0:
            res = tmp
        else:
            atleast2nests = True
            assert res == tmp
    assert atleast2nests
    assert res >= 0
    return res

def _parseNS(ns):
    out = [[0]]
    sz = [1]
    for sub in ns:
        _parseNS0(sub, out, sz, 0, -1)
    toret = np.zeros(sz[0] + 1, dtype=np.int64)
    ind = 0
    begi = 0
    for level in reversed(out):
        toret[ind] = level[0]
        ind += 1
        begi = ind + level[0] + 1
        for val in level[1:]:
            if val >= 0:
                toret[ind] = begi + val
            else:
                toret[ind] = ind
            ind += 1
    return toret

def solve(options, ns, t, i, x, n):
    assert isinstance(options, str)
    assert isinstance(t, np.ndarray)
    assert isinstance(i, np.ndarray)
    assert isinstance(x, np.ndarray)
    assert isinstance(n, np.ndarray)
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

    nestSpec = _parseNS(ns)
    # XXX: "factorize" i
    assert len(i[i < 0]) == 0
    assert len(i[i > nestSpec[0]]) == 0
    return _C_logitthing.solve(options, nestSpec, t, i, x, n)

