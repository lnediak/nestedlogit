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

struct Dataset {
  long isz, xsz;
  struct Entry {
    long t;
    double *x; /// x[(i-1)*xsz+k] = (x_it)_k where 1<=i<=isz and 0<=k<xsz
    long *n;   /// n[i] = n_it where 0<=i<=isz
    bool *z;   /// z[i-1] = z_i where 1<=i<=isz
  };
  std::unique_ptr<char[]> pool;
  std::vector<Entry> entries;

  Dataset(long isz, long xsz) : isz(isz), xsz(xsz) {}

  /// long format, i->t,i,x[0..xsz-1],n, and sz is number of such
  template <class Iter> bool initData(Iter iter, npy_intp sz) {
    entries.clear();
    if (!sz) {
      return true;
    }
    std::size_t numE = 1;
    Iter it = iter;
    long ct = it->t;
    for (npy_intp i = sz; --i;) {
      ++it;
      if (ct != it->t) {
        ct = it->t;
        numE++;
      }
    }
    entries.resize(numE);
    std::size_t zstride = isz;
    std::size_t nstride = isz + 1;
    std::size_t xstride = isz * xsz;
    std::size_t noff = sizeof(bool) * zstride * numE;
    std::size_t xoff = noff + sizeof(long) * nstride * numE;
    pool.reset(new char[xoff + sizeof(double) * xstride * numE]{0});
    bool *zp = reinterpret_cast<bool *>(pool.get());
    long *np = reinterpret_cast<long *>(&pool[noff]);
    double *xp = reinterpret_cast<double *>(&pool[xoff]);
    auto eiter = entries.begin();
    eiter->t = iter->t;
    eiter->x = xp;
    eiter->n = np;
    eiter->z = zp;
    for (npy_intp i = sz - 1;;) {
      if (eiter->t != iter->t) {
        ++eiter;
        eiter->t = iter->t;
        eiter->x = xp += xstride;
        eiter->n = np += nstride;
        eiter->z = zp += zstride;
      }
      if (iter->i) {
        std::memcpy(xp + (iter->i - 1) * xsz, iter->x, sizeof(double) * xsz);
        zp[iter->i - 1] = true;
      }
      np[iter->i] = iter->n;
      // to avoid the one-past-end iter
      if (i--) {
        ++iter;
      } else {
        break;
      }
    }
    return true;
  }
};

typedef CppAD::AD<double> ADd;

/// beta[(i-1)*xsz + k] = (beta_i)_k
struct FG_eval {
  Dataset &d;
  typedef CPPAD_TESTVECTOR(ADd) ADvector;
  void operator()(ADvector &fg, const ADvector &beta) const {
    assert(fg.size() == 1);
    assert(beta.size() == d.isz * d.xsz);
    ADd t = 0;
    for (const Dataset::Entry &e : d.entries) {
      ADd tmp1 = 0;
      ADd tmp2 = 0;
      double nt = e.n[0];
      double lnt = std::lgamma(e.n[0] + 1);
      for (long i1 = d.isz; i1--;) {
        if (!e.z[i1]) {
          continue;
        }
        ADd tmp = 0;
        for (long k = d.xsz; k--;) {
          tmp += e.x[i1 * d.xsz + k] * beta[i1 * d.xsz + k];
        }
        tmp1 += e.n[i1 + 1] * tmp;
        tmp2 += CppAD::exp(tmp);
        nt += e.n[i1 + 1];
        lnt += std::lgamma(e.n[i1 + 1] + 1);
      }
      t += std::lgamma(nt + 1) - lnt + tmp1 - nt * CppAD::log1p(tmp2);
    }
    fg[0] = -t;
  }
};

/// Warning: no one-past-end iterator rip
template <class Tt, class Ti, class Tn> struct SolveIter {
  char *t, *i, *x, *n;
  npy_intp ts, is, xs0, xs1, ns;
  std::vector<double> tmp;
  struct Res {
    long t, i, n;
    double *x;
  } tmpr;
  SolveIter(PyArrayObject *ta, PyArrayObject *ia, PyArrayObject *xa,
            PyArrayObject *na, long xsz)
      : tmp(xsz) {
    t = static_cast<char *>(PyArray_DATA(ta));
    ts = PyArray_STRIDE(ta, 0);
    i = static_cast<char *>(PyArray_DATA(ia));
    is = PyArray_STRIDE(ia, 0);
    x = static_cast<char *>(PyArray_DATA(xa));
    xs0 = PyArray_STRIDE(xa, 0);
    xs1 = PyArray_STRIDE(xa, 1);
    n = static_cast<char *>(PyArray_DATA(na));
    ns = PyArray_STRIDE(na, 0);
    copyIntoTmp();
  }
  void copyIntoTmp() {
    for (std::size_t ind = tmp.size(), indx = (ind - 1) * xs1; ind--;
         indx -= xs1) {
      tmp[ind] = *reinterpret_cast<double *>(x + indx);
    }
    tmpr = {*reinterpret_cast<Tt *>(t), *reinterpret_cast<Ti *>(i),
            *reinterpret_cast<Tn *>(n), &tmp[0]};
  }
  const Res &operator*() { return tmpr; }
  const Res *operator->() { return &tmpr; }
  SolveIter<Tt, Ti, Tn> &operator++() {
    t += ts;
    i += is;
    x += xs0;
    n += ns;
    copyIntoTmp();
    return *this;
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
  Py_INCREF(t);
  Py_INCREF(i);
  Py_INCREF(x);
  Py_INCREF(n);
  long xsz = PyArray_DIM(x, 1);
  npy_intp sz = PyArray_DIM(t, 0);
  Dataset dataset(isz, xsz);
  // XXX: generalize t, i, n type here
  dataset.initData(
      SolveIter<std::int64_t, std::int64_t, std::int64_t>(t, i, x, n, xsz), sz);
  /*
  for (const Dataset::Entry &e : dataset.entries) {
    std::cout << "t: " << e.t << std::endl;
    std::cout << "x: ";
    for (long i = 0; i < isz * xsz; i++) {
      std::cout << e.x[i] << " ";
    }
    std::cout << std::endl << "n: ";
    for (long i = 0; i <= isz; i++) {
      std::cout << e.n[i] << " ";
    }
    std::cout << std::endl << "z: ";
    for (long i = 0; i < isz; i++) {
      std::cout << e.z[i] << " ";
    }
    std::cout << std::endl << std::endl;
  }
  */

  typedef CPPAD_TESTVECTOR(double) Dvector;
  std::size_t betasz = isz * xsz;
  Dvector bi(betasz), bl(betasz), bu(betasz);
  for (long ii = betasz; ii--;) {
    bi[ii] = 0;
    bl[ii] = -1e19;
    bu[ii] = 1e19;
  }
  CppAD::ipopt::solve_result<Dvector> solution;
  FG_eval fg_eval{dataset};
  CppAD::ipopt::solve<Dvector, FG_eval>(std::string(options), bi, bl, bu,
                                        Dvector(0), Dvector(0), fg_eval,
                                        solution);
  assert(solution.status == decltype(solution)::success);
  npy_intp tmp[2] = {isz, xsz};
  PyObject *ret = PyArray_ZEROS(2, tmp, NPY_DOUBLE, 0);
  std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject *>(ret)),
              &solution.x[0], sizeof(double) * betasz);
  Py_DECREF(t);
  Py_DECREF(i);
  Py_DECREF(x);
  Py_DECREF(n);
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

