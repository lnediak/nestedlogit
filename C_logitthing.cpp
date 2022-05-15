#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <numpy/arrayobject.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include <cppad/ipopt/solve.hpp>

namespace {

typedef std::int64_t int64;

typedef CppAD::AD<double> ADd;

struct DataEntry {
  int64 t;
  std::unique_ptr<double[]> x; /// x[(i-1)*xsz+k]=(x_it)_k, 1<=i<=isz, 0<=k<xsz
  std::unique_ptr<int64[]> n;  /// n[i]=n_it where 0<=i<=isz
  std::unique_ptr<bool[]> z;   /// z[i]=z_i where 0<=i<=isz
  DataEntry(int64 isz, int64 xsz)
      : t(0), x(new double[isz * xsz]), n(new int64[isz + 1]),
        z(new bool[isz + 1]) {}
};
template <class Tt, class Ti, class Tn> struct DataReader {
  npy_intp ti, tsz;
  char *t, *i, *x, *n;
  npy_intp ts, is, xs0, xs1, ns;
  int64 isz, xsz;
  DataEntry out;
  DataReader(PyArrayObject *ta, PyArrayObject *ia, PyArrayObject *xa,
             PyArrayObject *na, int64 isz, int64 xsz)
      : ti(0), isz(isz), xsz(xsz), out(isz, xsz) {
    tsz = PyArray_DIM(ta, 0);
    t = static_cast<char *>(PyArray_DATA(ta));
    ts = PyArray_STRIDE(ta, 0);
    i = static_cast<char *>(PyArray_DATA(ia));
    is = PyArray_STRIDE(ia, 0);
    x = static_cast<char *>(PyArray_DATA(xa));
    xs0 = PyArray_STRIDE(xa, 0);
    xs1 = PyArray_STRIDE(xa, 1);
    n = static_cast<char *>(PyArray_DATA(na));
    ns = PyArray_STRIDE(na, 0);
  }
  bool next() {
    if (ti >= tsz) {
      return false;
    }
    std::memset(out.z.get(), 0, sizeof(bool) * (isz + 1));
    std::memset(out.n.get(), 0, sizeof(int64) * (isz + 1));
    out.t = *reinterpret_cast<Tt *>(t);
    do {
      int64 ci = *reinterpret_cast<Ti *>(i);
      assert(0 <= ci && ci <= isz);
      if (ci) {
        for (int64 in = 0, ix = 0; in < xsz; in++, ix += xs1) {
          out.x[(ci - 1) * xsz + in] = *reinterpret_cast<double *>(x + ix);
        }
      }
      out.z[ci] = true;
      out.n[ci] = *reinterpret_cast<Tn *>(n);
      t += ts;
      ti++;
      i += is;
      x += xs0;
      n += ns;
    } while (ti < tsz && *reinterpret_cast<Tt *>(t) == out.t);
    return true;
  }
  void reset() {
    t -= ts * ti;
    i -= is * ti;
    x -= xs0 * ti;
    n -= ns * ti;
    ti = 0;
  }
};

/**
  nestSpec[0] is isz
  nestSpec[1..isz] will give an index, in nestSpec of the nest class i is in
  nestSpec[isz+1] is the number of nests
  nestSpec[j] will give an index, in nestSpec of the nest that nest j is in
  etc...
  nestSpec[j] == j says that j is on the highest level of nesting,
  nestSpec[nsz - 1] = 0
*/
template <class T> struct UtilAdder {
  struct PT {
    T v;
    int64 n;
    bool z = false;
  };
  int64 nsz;
  int64 *nestSpec;
  std::unique_ptr<PT[]> vals;    // vals[0..isz] is log, vals[nsz-1] is special
  std::unique_ptr<T[]> nestMods; // not all of these are used, but whatever
  UtilAdder(int64 nsz, int64 *nestSpec)
      : nsz(nsz), nestSpec(nestSpec), vals(new PT[nsz]), nestMods(new T[nsz]) {}
  void setNestMods(T *vars, const int64 from = 0) {
    if (nestSpec[from + 1 + nestSpec[from]]) {
      if (from) {
        for (int64 i = from + 1; i <= nestSpec[from]; i++) {
          nestMods[i] = vars[i] / vars[nestSpec[i]];
        }
      } else {
        for (int64 i = 1; i <= nestSpec[0]; i++) {
          nestMods[i] = vars[i];
        }
      }
      setNestMods(vars, from + 1 + nestSpec[from]);
    } else if (from) {
      for (int64 i = from + 1; i <= nestSpec[from]; i++) {
        nestMods[i] = vars[i];
      }
    }
  }
  void clearVals() {
    for (int64 i = nsz; i--;) {
      vals[i].v = vals[i].z = vals[i].n = 0;
    }
  }
  /// i is class, u is utility, n is number
  void set(int64 i, T u, int64 n) {
    vals[i].v = u;
    vals[i].n = n;
    vals[i].z = true;
  }
  void setLayers(int64 from = 0) {
    int64 ii = from + 1 + nestSpec[from];
    for (int64 i = from + 1, end = from + nestSpec[from]; i <= end; i++) {
      if (!vals[i].z) {
        continue;
      }
      int64 ii = nextSpec[ii] ? nextSpec[i] : ii;
      vals[ii].z = true;
      vals[ii].n += vals[i].n;
      if (from) {
        vals[ii].v += CppAD::pow(vals[i].v, nestMods[i]);
      } else if (nextSpec[ii]) {
        vals[ii].v += CppAD::exp(vals[i].v / nestMods[i]);
      } else {
        vals[ii].v += CppAD::exp(vals[i].v);
      }
    }
    if (nextSpec[ii]) {
      setLayers(from + 1 + nestSpec[from]);
    }
  }
  void addTo(T &l) const {
    setLayers();
    // TODO: CONTINUE HERE
  }
};

/// beta[(i-1)*xsz + k] = (beta_i)_k
template <class Reader> struct FG_eval {
  Reader &r;
  typedef CPPAD_TESTVECTOR(ADd) ADvector;
  void operator()(ADvector &fg, const ADvector &beta) const {
    assert(fg.size() == 1);
    assert(beta.size() == r.isz * r.xsz);
    ADd t = 0;
    while (r.next()) {
      ADd tmp1 = 0;
      ADd tmp2 = 0;
      double nt = r.out.n[0];
      double lnt = std::lgamma(r.out.n[0] + 1);
      for (long i1 = r.isz; i1--;) {
        if (!r.out.z[i1 + 1]) {
          continue;
        }
        ADd tmp = 0;
        for (long k = r.xsz; k--;) {
          tmp += r.out.x[i1 * r.xsz + k] * beta[i1 * r.xsz + k];
        }
        tmp1 += r.out.n[i1 + 1] * tmp;
        tmp2 += CppAD::exp(tmp);
        nt += r.out.n[i1 + 1];
        lnt += std::lgamma(r.out.n[i1 + 1] + 1);
      }
      t +=
          std::lgamma(nt + 1) - lnt + tmp1 - nt * CppAD::log(r.out.z[0] + tmp2);
    }
    r.reset(); // meh kek
    fg[0] = -t;
  }
};

PyObject *solve(PyObject *, PyObject *args) {
  const char *options;
  long isz;
  PyArrayObject *t, *i, *x, *n;
  if (!PyArg_ParseTuple(args, "slOOOO", &options, &isz, &t, &i, &x, &n)) {
    return NULL;
  }
  if (!t || !i || !x || !n) {
    return NULL;
  }
  long xsz = PyArray_DIM(x, 1);
  // XXX: generalize t, i, n type here
  DataReader<int64, int64, int64> dataReader(t, i, x, n, isz, xsz);
  typedef CPPAD_TESTVECTOR(double) Dvector;
  std::size_t betasz = isz * xsz;
  Dvector bi(betasz), bl(betasz), bu(betasz);
  for (long ii = betasz; ii--;) {
    bi[ii] = 0;
    bl[ii] = -1e19;
    bu[ii] = 1e19;
  }
  CppAD::ipopt::solve_result<Dvector> solution;
  FG_eval<decltype(dataReader)> fg_eval{dataReader};
  CppAD::ipopt::solve<Dvector, decltype(fg_eval)>(std::string(options), bi, bl,
                                                  bu, Dvector(0), Dvector(0),
                                                  fg_eval, solution);
  assert(solution.status == decltype(solution)::success);
  npy_intp tmp[2] = {isz, xsz};
  PyObject *ret = PyArray_ZEROS(2, tmp, NPY_DOUBLE, 0);
  std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject *>(ret)),
              &solution.x[0], sizeof(double) * betasz);
  return ret;
}

PyMethodDef C_logitthingMethods[] = {
    {"solve", solve, METH_VARARGS, "who cares about docstring"}, {0, 0, 0, 0}};

PyModuleDef C_logitthingModule = {PyModuleDef_HEAD_INIT, "_C_logitthing", NULL,
                                  -1, C_logitthingMethods};

} // namespace

PyMODINIT_FUNC PyInit__C_logitthing() {
  import_array();
  return PyModule_Create(&C_logitthingModule);
}

