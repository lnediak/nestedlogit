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

template <class T, class T1> struct UtilAdderB {
  T lnum, den; /// log numerator, denominator
  int64 nt;    /// total number to take as exponent of denominator
  int64 nest, *nestSpec;
  T1 expo;
  /// u is utility, n is number, i1 is i-1
  void add(T u, int64 n, int64 i1) {
    if (nestSpec[i1] != nest) {
      return;
    }
    lnum += n * u / expo;
    den += CppAD::exp(u / expo);
    nt += n;
  }
};

template <class T, class S> struct NestData {
  int64 ssz;
  std::unique_ptr<S[]> sub;
  int64 *nestSpec; /// nestSpec[i-1] is ind of nest containing class i
  T expo;
  NestData(int64 ssz, int64 *nestSpec, T expo)
      : ssz(ssz), sub(new S[ssz]), nestSpec(nestSpec), expo(expo) {}
  template <class S> T lmodResult(S d) { return expo * CppAD::log(d.den()); }
  template <class S> T modResult(S d) { return CppAD::pow(d.den(), expo); }
};
template <class T, class S> struct NestDataLast {
  int64 ssz;
  std::unique_ptr<S[]> sub;
  int64 *nestSpec;
  T expoi;
  NestDataLast(int64 ssz, int64 *nestSpec, T expoi)
      : ssz(ssz), sub(new S[ssz]), nestSpec(nestSpec), expoi(expoi) {}
  template <class S> T lmodResult(S d) { return d.lden() / expoi; }
  template <class S> T modResult(S d) { return CppAD::exp(lmodResult(d)); }
};
template <class T, class S, class NS> struct UtilAdder {
  NS *ns;
  int64 nt; /// total number to take as exponent of denominator
  UtilAdder() : ns(nullptr), nt(0) {}
  template <class... Args> void init(NS *arg, Args args...) {
    ns = arg;
    for (int64 ni = ns.ssz; ni--;) {
      ns->sub[ni].init(args...);
    }
  }
  void reset() {
    nt = 0;
    for (int64 ni = ns.ssz; ni--;) {
      ns->sub[ni].reset();
    }
  }
  /// u is utility, n is number, i1 is i-1
  void add(T u, int64 n, int64 i1) {
    nt += n;
    ns->sub[ns.nestSpec[i1]].add(u, n, i1);
  }
  T den() const {
    T res = 0;
    for (int64 ni = ns.ssz; ni--;) {
      res += ns.modResult(ns->sub[ni]);
    }
    return res;
  }
  void apply(T &l) const {
    for (int64 ni = ns.ssz; ni--;) {
      ns->sub[ni].apply(l);
      l += ns->sub[ni].nt * ns->lmodResult(sub[ni]);
    }
    l -= nt * CppAD::log(den());
  }
};
template <class T> TWrap {
  T t;
  int64 nt;
  void init() {
    t = 0;
    nt = 0;
  }
  void reset() {
    t = 0;
    nt = 0;
  }
  void add(T u, int64 n, int64) {
    t = u;
    nt = n;
  }
  T lden() const { return t; }
  void apply(T & l) const {}
};
template <class T>
using AdderLv1 = UtilAdder<T, TWrap<T>, NestDataLast<T, TWrap<T>>>;

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

