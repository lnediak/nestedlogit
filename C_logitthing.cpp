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

#include <cppad/ipopt/solve.hpp>

namespace {

typedef std::int64_t int64;

typedef CppAD::AD<double> ADd;

struct DataEntry {
  int64 t;
  std::unique_ptr<double[]> x; /// x[(i-1)*xsz+k]=(x_it)_k, 1<=i<=isz, 0<=k<xsz
  std::unique_ptr<int64[]> n;  /// n[i]=n_it where 0<=i<=isz
  std::unique_ptr<bool[]> z;   /// z[i-1]=z_i where 1<=i<=isz
  DataEntry(int64 isz, int64 xsz)
      : t(0), x(new double[isz * xsz]), n(new int64[isz + 1]),
        z(new bool[isz]) {}
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
    std::memset(out.z.get(), 0, sizeof(bool) * isz);
    out.t = *reinterpret_cast<Tt *>(t);
    do {
      int64 ci = *reinterpret_cast<Ti *>(i);
      assert(0 <= ci && ci <= isz);
      if (ci) {
        for (int64 in = 0, ix = 0; in < xsz; in++, ix += xs1) {
          out.x[(ci - 1) * xsz + in] = *reinterpret_cast<double *>(x + ix);
        }
        out.z[ci - 1] = true;
      }
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
        if (!r.out.z[i1]) {
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
      t += std::lgamma(nt + 1) - lnt + tmp1 - nt * CppAD::log1p(tmp2);
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

