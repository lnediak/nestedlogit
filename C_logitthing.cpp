#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <numpy/arrayobject.h>
#include <numpy/random/distributions.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
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
/// readN means just reading, !readN means using for generateData
template <class Tt, class Ti, class Tn, bool readN> struct DataReader {
  npy_intp ti, tsz;
  char *t, *i, *x, *n, *norig;
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
    norig = n = static_cast<char *>(PyArray_DATA(na));
    ns = PyArray_STRIDE(na, 0);
  }
  bool next() {
    if (ti >= tsz) {
      return false;
    }
    std::memset(out.z.get(), 0, sizeof(bool) * (isz + 1));
    std::memset(out.n.get(), 0, sizeof(int64) * (isz + 1));
    out.t = *reinterpret_cast<Tt *>(t);
    Tt nt = 0;
    do {
      int64 ci = *reinterpret_cast<Ti *>(i);
      assert(0 <= ci && ci <= isz);
      if (ci) {
        for (int64 in = 0, ix = 0; in < xsz; in++, ix += xs1) {
          out.x[(ci - 1) * xsz + in] = *reinterpret_cast<double *>(x + ix);
        }
      }
      out.z[ci] = true;
      if (readN) {
        out.n[ci] = *reinterpret_cast<Tn *>(n);
      } else {
        out.n[ci] = n - norig;
        nt += *reinterpret_cast<Tn *>(n);
      }
      t += ts;
      ti++;
      i += is;
      x += xs0;
      n += ns;
    } while (ti < tsz && *reinterpret_cast<Tt *>(t) == out.t);
    if (!readN) {
      out.t = nt; // stealing the otherwise useless field...
    }
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
  typedef T value_type;
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
  template <class TI>
  void setNestMods(const TI &vars, int64 voff, const int64 from = 0) {
    int64 ii = from + 1 + nestSpec[from];
    if (nestSpec[ii] && from) {
      for (int64 i = from + 1; i <= nestSpec[from]; i++) {
        nestMods[i] = vars[i - voff] / vars[nestSpec[i] - voff];
      }
    } else if (nestSpec[ii] || from) {
      for (int64 i = from + 1; i <= nestSpec[from]; i++) {
        nestMods[i] = vars[nestSpec[i] - voff];
      }
    }
    if (nestSpec[ii]) {
      setNestMods(vars, voff + 1, from + 1 + nestSpec[from]);
    }
  }
  int64 varsDepth() const {
    int64 toret = 0;
    int64 i = 1;
    while (i != nestSpec[i]) {
      i = nestSpec[i];
      toret++;
    }
    return toret;
  }
  int64 varsLen() const { return nsz - nestSpec[0] - 2 - varsDepth(); }

  void clearVals() {
    for (int64 i = nsz; i--;) {
      vals[i].v = vals[i].n = vals[i].z = 0;
    }
  }
  /// i is class, u is utility, n is number
  void set(int64 i, T u, int64 n) {
    vals[i].v = u;
    vals[i].n = n;
    vals[i].z = true;
  }
  void setLayers(int64 from = 0) {
    int64 ni = from + 1 + nestSpec[from];
    for (int64 i = from + 1, end = from + nestSpec[from]; i <= end; i++) {
      if (!vals[i].z) {
        continue;
      }
      int64 ii = nestSpec[ni] ? nestSpec[i] : ni;
      T tmp;
      if (from) {
        tmp = CppAD::pow(vals[i].v, nestMods[i]);
      } else if (nestSpec[ii]) {
        tmp = CppAD::exp(vals[i].v / nestMods[i]);
      } else {
        tmp = CppAD::exp(vals[i].v);
      }
      vals[ii].v = vals[ii].z ? vals[ii].v + tmp : tmp;
      vals[ii].z = true;
      vals[ii].n += vals[i].n;
    }
    if (nestSpec[ni]) {
      setLayers(from + 1 + nestSpec[from]);
    } else {
      vals[ni].n += vals[0].n;
    }
  }
  /// call setLayers first, does not include the multinomial constant factor
  void addTo(T &l) const {
    int64 i = 0;
    for (int64 end = nestSpec[i]; ++i <= end;) {
      l += nestSpec[nestSpec[0] + 1] ? vals[i].n * vals[i].v / nestMods[i]
                                     : vals[i].n * vals[i].v;
    }
    while (nestSpec[i]) {
      for (int64 end = i + nestSpec[i]; ++i <= end;) {
        l += vals[i].n * (nestMods[i] - 1) * CppAD::log(vals[i].v);
      }
    }
    l -= vals[i].n * CppAD::log(vals[0].z + vals[i].v);
  }
  /// call setLayers first, given values passed to set, get prob of class i
  T getProb(int64 i) const {
    T res = CppAD::exp(i == nestSpec[i] ? vals[i].v : vals[i].n / nestMods[i]);
    while (i != nestSpec[i]) {
      i = nestSpec[i];
      res *= CppAD::pow(vals[i].v, nestMods[i] - 1);
    }
    return res / (vals[0].z + vals[nsz - 1].v);
  }
};

/// beta[(i-1)*xsz + k] = (beta_i)_k
template <class Reader, class Adder> struct FG_eval {
  Reader &r;
  Adder &a;
  typedef typename std::remove_cv<
      typename std::remove_reference<Adder>::type>::type::value_type value_type;
  typedef CPPAD_TESTVECTOR(ADd) ADvector;
  void operator()(ADvector &fg, const ADvector &beta) const {
    assert(fg.size() == 1);
    assert(beta.size() == r.isz * r.xsz + a.varsLen());
    a.setNestMods(beta, r.isz + 2 - r.isz * r.xsz);
    ADd t = 0;
    // expecting readN on r
    while (r.next()) {
      a.clearVals();
      if (r.out.z[0]) {
        a.set(0, 0, r.out.n[0]);
      }
      for (int64 i1 = r.isz; i1--;) {
        if (!r.out.z[i1 + 1]) {
          continue;
        }
        value_type tmp = r.out.x[i1 * r.xsz] * beta[i1 * r.xsz];
        for (int64 k = r.xsz; --k;) {
          tmp += r.out.x[i1 * r.xsz + k] * beta[i1 * r.xsz + k];
        }
        a.set(i1 + 1, tmp, r.out.n[i1 + 1]);
      }
      a.setLayers();
      a.addTo(t);
    }
    r.reset(); // meh kek
    fg[0] = -t;
  }
  // I am using r.out.t for the total `n` for that observation
  // while using r.out.n[i] as pointer offset for r.norig
  template <class Dvec>
  void generateData(const Dvec &beta, bitgen_t *bitgen_state) const {
    assert(beta.size() == r.isz * r.xsz + a.varsLen());
    a.setNestMods(beta, r.isz + 2 - r.isz * r.xsz);
    // as passed to random_multinomial:
    npy_intp d = r.isz + 1;
    std::unique_ptr<npy_int64[]> mnix(new npy_int64[d]);
    std::unique_ptr<double[]> pix(new double[d]);
    binomial_t binomial;
    // expecting !readN on r
    while (r.next()) {
      a.clearVals();
      if (r.out.z[0]) {
        a.set(0, 0, 0);
      }
      for (int64 i1 = r.isz; i1--;) {
        if (!r.out.z[i1 + 1]) {
          continue;
        }
        value_type tmp = r.out.x[i1 * r.xsz] * beta[i1 * r.xsz];
        for (int64 k = r.xsz; --k;) {
          tmp += r.out.x[i1 * r.xsz + k] * beta[i1 * r.xsz + k];
        }
        a.set(i1 + 1, tmp, 0);
      }
      a.setLayers();
      // lazy me
      std::memset(pix.get(), 0, sizeof(double) * d);
      for (int64 i = r.isz + 1; i--;) {
        if (r.out.z[i]) {
          pix[i] = a.getProb(i);
        }
      }
      random_multinomial(bitgen_state, r.out.t, mnix.get(), pix.get(), d,
                         &binomial);
      for (int64 i = r.isz + 1; i--;) {
        if (r.out.z[i]) {
          *reinterpret_cast<npy_int64 *>(r.norig + r.out.n[i]) = mnix[i];
        }
      }
    }
  }
};

std::unique_ptr<int64[]> copyNpArr(PyArrayObject *a, npy_intp &len) {
  len = PyArray_DIM(a, 0);
  std::unique_ptr<int64[]> toret(new int64[len]);
  npy_intp sd = PyArray_STRIDE(a, 0);
  char *ptr = static_cast<char *>(PyArray_DATA(a));
  for (npy_intp i = 0; i < len; i++, ptr += sd) {
    // XXX: generalize type here
    toret[i] = *reinterpret_cast<npy_int64 *>(ptr);
  }
  return toret;
}
std::vector<double> copyNpDArr(PyArrayObject *a) {
  std::size_t len = PyArray_DIM(a, 0);
  std::vector<double> toret(len);
  npy_intp sd = PyArray_STRIDE(a, 0);
  char *ptr = static_cast<char *>(PyArray_DATA(a));
  for (std::size_t i = 0; i < len; i++, ptr += sd) {
    toret[i] = *reinterpret_cast<double *>(ptr);
  }
  return toret;
}

PyObject *genData(PyObject *, PyObject *args) {
  PyObject *btgen;
  PyArrayObject *betaarr, *ns, *t, *i, *x, *n;
  if (!PyArg_ParseTuple(args, "OOOOOOOO", &btgen, &betaarr, &ns, &t, &i, &x,
                        &n)) {
    PyErr_SetString(PyExc_ValueError, "parsing arguments went wrong?");
    return NULL;
  }
  if (!btgen || !ns || !t || !i || !x || !n) {
    PyErr_SetString(PyExc_ValueError, "something happened to the parsed vals");
    return NULL;
  }
  bitgen_t *bitgen_state =
      static_cast<bitgen_t *>(PyCapsule_GetPointer(btgen, "BitGenerator"));
  std::vector<double> beta = copyNpDArr(betaarr);
  npy_intp nsz;
  std::unique_ptr<int64[]> nestSpec = copyNpArr(ns, nsz);
  int64 isz = nestSpec[0];
  int64 xsz = PyArray_DIM(x, 1);
  // XXX: generalize t, i, n type here
  DataReader<npy_int64, npy_int64, npy_int64, false> dataReader(t, i, x, n, isz,
                                                                xsz);
  UtilAdder<double> utilAdder(nsz, nestSpec.get());
  FG_eval<decltype(dataReader), decltype(utilAdder)> fg_eval{dataReader,
                                                             utilAdder};
  fg_eval.generateData(beta, bitgen_state);
  return Py_BuildValue(""); // return None
}

PyObject *solve(PyObject *, PyObject *args) {
  const char *options;
  PyArrayObject *ns, *t, *i, *x, *n;
  if (!PyArg_ParseTuple(args, "sOOOOO", &options, &ns, &t, &i, &x, &n)) {
    PyErr_SetString(PyExc_ValueError, "parsing arguments went wrong?");
    return NULL;
  }
  if (!ns || !t || !i || !x || !n) {
    PyErr_SetString(PyExc_ValueError, "something happened to the np arrays");
    return NULL;
  }
  npy_intp nsz;
  std::unique_ptr<int64[]> nestSpec = copyNpArr(ns, nsz);
  int64 isz = nestSpec[0];
  int64 xsz = PyArray_DIM(x, 1);
  // XXX: generalize t, i, n type here
  DataReader<npy_int64, npy_int64, npy_int64, true> dataReader(t, i, x, n, isz,
                                                               xsz);
  UtilAdder<ADd> utilAdder(nsz, nestSpec.get());
  typedef CPPAD_TESTVECTOR(double) Dvector;
  std::size_t betasz = isz * xsz + utilAdder.varsLen();
  Dvector bi(betasz), bl(betasz), bu(betasz);
  for (std::size_t ii = isz * xsz; ii--;) {
    bi[ii] = 0;
    bl[ii] = -1e19;
    bu[ii] = 1e19;
  }
  // XXX: add option for these
  for (std::size_t ii = isz * xsz; ii < betasz; ii++) {
    bi[ii] = bl[ii] = 0;
    bu[ii] = 1;
  }
  CppAD::ipopt::solve_result<Dvector> solution;
  FG_eval<decltype(dataReader), decltype(utilAdder)> fg_eval{dataReader,
                                                             utilAdder};
  CppAD::ipopt::solve<Dvector, decltype(fg_eval)>(std::string(options), bi, bl,
                                                  bu, Dvector(0), Dvector(0),
                                                  fg_eval, solution);
  assert(solution.status == decltype(solution)::success);
  npy_intp tmp[2] = {isz, xsz};
  PyObject *par = PyArray_ZEROS(2, tmp, NPY_DOUBLE, 0);
  if (!par) {
    PyErr_SetString(PyExc_RuntimeError, "failed to create a new np array");
    return NULL;
  }
  std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject *>(par)),
              &solution.x[0], sizeof(double) * isz * xsz);
  PyObject *ret = PyTuple_New(1 + utilAdder.varsDepth());
  if (!ret) {
    PyErr_SetString(PyExc_RuntimeError, "failed to create a new tuple");
    return NULL;
  }
  PyTuple_SetItem(ret, 0, par);
  int64 soli = isz * xsz;
  for (int64 i = isz + 1, ri = 1; nestSpec[i];
       i += 1 + nestSpec[i], soli += nestSpec[i], ri++) {
    PyObject *var = PyArray_ZEROS(1, &nestSpec[i], NPY_DOUBLE, 0);
    if (!var) {
      PyErr_SetString(PyExc_RuntimeError, "failed to create a new np array");
      return NULL;
    }
    std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject *>(par)),
                &solution.x[soli], sizeof(double) * nestSpec[i]);
    PyTuple_SetItem(ret, ri, var);
  }
  return ret;
}

// -- tests?

void slowWriteProbs3L(const int64 *nestSpec, const bool *z, const double *u,
                      const double *in, const double *ou, double *p) {
  int64 isz = nestSpec[0];
  int64 insz = nestSpec[isz + 1];
  int64 ousz = nestSpec[isz + insz + 2];
  for (int64 i = 0; i <= isz; i++) {
    if (!z[i]) {
      continue;
    }
    int64 innest = nestSpec[i] - isz - 2;
    int64 ounest = nestSpec[nestSpec[i]] - isz - insz - 3;
    if (i) {
      p[i] = std::exp(u[i] / in[innest]);
    } else {
      p[i] = 1;
    }
    double tmp = 0;
    for (int64 j = 1; j <= isz; j++) {
      if (nestSpec[j] == nestSpec[i]) {
        tmp += std::exp(u[j] / in[innest]);
      }
    }
    p[i] *= std::pow(tmp, in[innest] / ou[ounest] - 1);
    tmp = 0;
    for (int64 k = isz + 2; k < isz + insz + 2; k++) {
      if (nestSpec[k] == nestSpec[nestSpec[i]]) {
        double itmp = 0;
        for (int64 j = 1; j <= isz; j++) {
          if (nestSpec[j] == k) {
            itmp += std::exp(u[j] / in[k - isz - 2]);
          }
        }
        tmp += std::pow(itmp, in[k - isz - 2] / ou[ounest]);
      }
    }
    p[i] *= std::pow(tmp, ou[ounest] - 1);
    tmp = z[0];
    for (int64 l = isz + insz + 3; l < isz + insz + ousz + 3; l++) {
      double otmp = 0;
      for (int64 k = isz + 2; k < isz + insz + 2; k++) {
        if (nestSpec[k] == l) {
          double itmp = 0;
          for (int64 j = 1; j <= isz; j++) {
            if (nestSpec[j] == k) {
              itmp += std::exp(u[j] / in[k - isz - 2]);
            }
          }
          otmp += std::pow(itmp, in[k - isz - 2] / ou[ounest]);
        }
      }
      tmp += std::pow(otmp, ou[l - isz - insz - 3]);
    }
    p[i] /= tmp;
  }
}

PyObject *runTests(PyObject *, PyObject *) {
  int64 nestSpec[] = {10, 14, 14, 14, 15, 16, 16, 16, 12, 13, 13,
                      5,  19, 19, 18, 18, 18, 2,  18, 19, 0};
  UtilAdder<double> utilAdder(21, nestSpec);
  std::uniform_int_distribution<bool> disti(0, 1);
  std::uniform_int_distribution<double> distr(-10, 10);
  for (int spam = 0; spam < 1000; spam++) {
    std::mt19937 mtrand(spam);
    bool z[] = {disti(mtrand), disti(mtrand), disti(mtrand), disti(mtrand),
                disti(mtrand), disti(mtrand), disti(mtrand), disti(mtrand),
                disti(mtrand), disti(mtrand)};
    bool zpass = false;
    for (int i = 0; i <= 10; i++) {
      if (z[i]) {
        zpass = true;
      }
    }
    if (!zpass) {
      z[4] = true; // no reason
    }
    double u[] = {0,
                  distr(mtrand),
                  distr(mtrand),
                  distr(mtrand),
                  distr(mtrand),
                  distr(mtrand),
                  distr(mtrand),
                  distr(mtrand),
                  distr(mtrand),
                  distr(mtrand),
                  distr(mtrand)};
    double vars[] = {distr(mtrand), distr(mtrand), distr(mtrand), distr(mtrand),
                     distr(mtrand), distr(mtrand), distr(mtrand)};
    double *in = vars;
    double *ou = in + 5;
    double p[10];
    slowWriteProbs3L(nestSpec, z, u, in, ou, p);
    utilAdder.setNestMods(vars, 0);
    utilAdder.clearVals();
    for (int i = 0; i <= 10; i++) {
      if (z[i]) {
        utilAdder.set(i, u[i], 0);
      }
    }
    for (int i = 0; i <= 10; i++) {
      if (std::abs(utilAdder.getProb(i) - p[i]) >= 1e-5) {
        return NULL;
      }
    }
  }
  return Py_BuildValue("");
}

// -- actual wrapping stuff

PyMethodDef C_logitthingMethods[] = {
    {"solve", solve, METH_VARARGS, "who cares about docstring"},
    {"genData", genData, METH_VARARGS, "docstings are a pain"},
    {"runTests", runTests, METH_VARARGS, "runs tests?"},
    {0, 0, 0, 0}};

PyModuleDef C_logitthingModule = {PyModuleDef_HEAD_INIT, "_C_logitthing", NULL,
                                  -1, C_logitthingMethods};

} // namespace

PyMODINIT_FUNC PyInit__C_logitthing() {
  import_array();
  return PyModule_Create(&C_logitthingModule);
}
