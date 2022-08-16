#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/random/distributions.h>

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

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)
#define ASSERT_T(expr)                                                         \
  if (!(expr)) {                                                               \
    throw std::runtime_error(__FILE__ ":" STRINGIFY(__LINE__) ": `" #expr      \
                                                              "` not true.");  \
  }

#define PRINT_V(expr) std::cout << #expr ": " << (expr) << std::endl

namespace {

typedef std::int64_t int64;

typedef CppAD::AD<double> ADd;

/// for generateData, t is used as total n, and n gets ptr offs from norig
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
    out.t = *reinterpret_cast<Tt *>(t);
    Tt nt = 0;
    do {
      int64 ci = *reinterpret_cast<Ti *>(i);
      ASSERT_T(0 <= ci && ci <= isz);
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
      i += is;
      x += xs0;
      n += ns;
      ti++;
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
  const int64 *nestSpec;
  std::unique_ptr<PT[]> vals;    // vals[0..isz] is log, vals[nsz-1] is special
  std::unique_ptr<T[]> nestMods; // read setNestMods for spec
  UtilAdder(int64 nsz, const int64 *nestSpec)
      : nsz(nsz), nestSpec(nestSpec), vals(new PT[nsz]), nestMods(new T[nsz]) {}
  /// set voff so that vars[isz + 1 - voff] is mod for first nest
  template <class TI> void setNestMods(const TI &vars, int64 voff) {
    for (int64 i = 1; i < nsz - 1; i++) {
      if (i == nestSpec[i]) {
        if (i > nestSpec[0]) {
          nestMods[i] = vars[i - voff];
        } else {
          nestMods[i] = 1;
        }
      } else {
        if (i > nestSpec[0]) {
          nestMods[i] = vars[i - voff] / vars[nestSpec[i] - voff];
        } else {
          nestMods[i] = vars[nestSpec[i] - voff];
        }
      }
    }
  }

  int64 varsLen() const { return nsz - 2 - nestSpec[0]; }

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
  void setLayers() {
    for (int64 i = 1; i < nsz - 1; i++) {
      if (!vals[i].z) {
        continue;
      }
      T tmp;
      if (i > nestSpec[0]) {
        tmp = CppAD::pow(vals[i].v, nestMods[i]);
      } else if (i == nestSpec[i]) {
        tmp = CppAD::exp(vals[i].v);
      } else {
        tmp = CppAD::exp(vals[i].v / nestMods[i]);
      }
      int64 ii = i == nestSpec[i] ? nsz - 1 : nestSpec[i];
      vals[ii].v = vals[ii].z ? vals[ii].v + tmp : tmp;
      vals[ii].z = true;
      vals[ii].n += vals[i].n;
    }
    if (vals[0].z) {
      int64 ii = nsz - 1;
      vals[ii].v = vals[ii].z ? vals[ii].v + 1 : 1;
      vals[ii].z = true;
      vals[ii].n += vals[0].n;
    }
  }
  /// call setLayers first, does not include the multinomial constant factor
  void addTo(T &l) const {
    int64 i = 0;
    while (++i <= nestSpec[0]) {
      if (vals[i].z) {
        l += vals[i].n *
             (i == nestSpec[i] ? vals[i].v : vals[i].v / nestMods[i]);
      }
    }
    i--;
    while (++i < nsz - 1) {
      // there might be empty nests due to the z values being false
      if (vals[i].z) {
        l += vals[i].n * (nestMods[i] - 1) * CppAD::log(vals[i].v);
      }
    }
    if (vals[i].z) {
      l -= vals[i].n * CppAD::log(vals[i].v);
    }
  }
  /// call setLayers first, given values passed to set, get prob of class i
  T getProb(int64 i) const {
    if (!vals[i].z) {
      return 0;
    }
    T res =
        i ? CppAD::exp(i == nestSpec[i] ? vals[i].v : vals[i].v / nestMods[i])
          : 1;
    while (i && i != nestSpec[i]) {
      i = nestSpec[i];
      res *= CppAD::pow(vals[i].v, nestMods[i] - 1);
    }
    return res / vals[nsz - 1].v;
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
    ASSERT_T(fg.size() == 1);
    ASSERT_T(static_cast<int64>(beta.size()) == r.isz * r.xsz + a.varsLen());
    a.setNestMods(beta, r.isz + 1 - r.isz * r.xsz);
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
    ASSERT_T(static_cast<int64>(beta.size()) == r.isz * r.xsz + a.varsLen());
    a.setNestMods(beta, r.isz + 1 - r.isz * r.xsz);
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
      std::memset(pix.get(), 0, sizeof(double) * d);
      double totalp = 0;
      for (int64 i = r.isz + 1; i--;) {
        if (r.out.z[i]) {
          pix[i] = a.getProb(i);
          totalp += pix[i];
        }
      }
      ASSERT_T(std::abs(totalp - 1) < 1e-3);
      // so numpy doesn't complain
      if (totalp > 1) {
        for (int64 i = r.isz + 1; i--;) {
          pix[i] /= totalp;
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
  if (!PyArg_ParseTuple(args, "sO!O!O!O!O!", &options, &PyArray_Type, &ns,
                        &PyArray_Type, &t, &PyArray_Type, &i, &PyArray_Type, &x,
                        &PyArray_Type, &n)) {
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
  // XXX: add option for these
  for (std::size_t ii = isz * xsz; ii--;) {
    bi[ii] = 0;
    bl[ii] = -10;
    bu[ii] = 10;
  }
  for (std::size_t ii = isz * xsz; ii < betasz; ii++) {
    bi[ii] = bl[ii] = 0.1;
    bu[ii] = 1;
  }
  CppAD::ipopt::solve_result<Dvector> solution;
  FG_eval<decltype(dataReader), decltype(utilAdder)> fg_eval{dataReader,
                                                             utilAdder};
  CppAD::ipopt::solve<Dvector, decltype(fg_eval)>(std::string(options), bi, bl,
                                                  bu, Dvector(0), Dvector(0),
                                                  fg_eval, solution);
  ASSERT_T(solution.status == decltype(solution)::success);
  npy_intp tmp[2] = {isz, xsz};
  PyObject *par = PyArray_ZEROS(2, tmp, NPY_DOUBLE, 0);
  if (!par) {
    PyErr_SetString(PyExc_RuntimeError, "failed to create a new np array");
    return NULL;
  }
  std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject *>(par)),
              &solution.x[0], sizeof(double) * isz * xsz);
  npy_intp nmods = utilAdder.varsLen();
  PyObject *var = PyArray_ZEROS(1, &nmods, NPY_DOUBLE, 0);
  if (!var) {
    PyErr_SetString(PyExc_RuntimeError, "failed to create a new np array");
    return NULL;
  }
  std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject *>(par)),
              &solution.x[isz * xsz], sizeof(double) * nmods);
  PyObject *ret = PyTuple_New(2);
  if (!ret) {
    PyErr_SetString(PyExc_RuntimeError, "failed to create a new tuple");
    return NULL;
  }
  PyTuple_SetItem(ret, 0, par);
  PyTuple_SetItem(ret, 1, var);
  return ret;
}

// -- tests?

/*
  just following the formula directly, ugly lel... fyi, this assumes my old
  format for nestSpec, which includes a counts for each "layer" (just like isz);
  example in runTests
*/
void slowWriteProbs3L(const int64 *nestSpec, const bool *z, const double *u,
                      const double *in, const double *ou, double *p) {
  int64 isz = nestSpec[0];
  int64 insz = nestSpec[isz + 1];
  int64 ousz = nestSpec[isz + insz + 2];
  int64 inoff = isz + 2;
  int64 ouoff = isz + insz + 3;
  for (int64 i = 0; i <= isz; i++) {
    if (!z[i]) {
      p[i] = 0;
      continue;
    }
    p[i] = 1;
    if (i) {
      int64 innest;
      int64 ounest;
      if (i == nestSpec[i]) {
        innest = ounest = -1;
      } else if (nestSpec[i] == nestSpec[nestSpec[i]]) {
        if (nestSpec[i] < isz + insz + 2) {
          innest = nestSpec[i];
          ounest = -1;
        } else {
          innest = -1;
          ounest = nestSpec[nestSpec[i]];
        }
      } else {
        innest = nestSpec[i];
        ounest = nestSpec[nestSpec[i]];
      }
      double innmod = innest >= 0 ? in[innest - inoff] : 1;
      double ounmod = ounest >= 0 ? ou[ounest - ouoff] : 1;
      p[i] = std::exp(u[i] / (innest >= 0 ? innmod : ounmod));
      double tmp = 0;
      for (int64 j = 1; j <= isz; j++) {
        if (z[j] && nestSpec[j] == innest) {
          tmp += std::exp(u[j] / innmod);
        }
      }
      p[i] *= tmp ? std::pow(tmp, innmod / ounmod - 1) : 1;
      tmp = 0;
      for (int64 j = 1; j <= isz; j++) {
        if (z[j] && nestSpec[j] == ounest) {
          tmp += std::exp(u[j] / ounmod);
        }
      }
      for (int64 k = inoff; k < inoff + insz; k++) {
        if (nestSpec[k] == ounest) {
          double itmp = 0;
          for (int64 j = 1; j <= isz; j++) {
            if (z[j] && nestSpec[j] == k) {
              itmp += std::exp(u[j] / in[k - inoff]);
            }
          }
          tmp += itmp ? std::pow(itmp, in[k - inoff] / ounmod) : 0;
        }
      }
      p[i] *= tmp ? std::pow(tmp, ounmod - 1) : 1;
    }
    double tmp = z[0];
    for (int64 j = 1; j <= isz; j++) {
      if (z[j] && j == nestSpec[j]) {
        tmp += std::exp(u[j]);
      }
    }
    for (int64 k = inoff; k < inoff + insz; k++) {
      if (k == nestSpec[k]) {
        double itmp = 0;
        for (int64 j = 1; j <= isz; j++) {
          if (z[j] && nestSpec[j] == k) {
            itmp += std::exp(u[j] / in[k - inoff]);
          }
        }
        tmp += itmp ? std::pow(itmp, in[k - inoff]) : 0;
      }
    }
    for (int64 l = ouoff; l < ouoff + ousz; l++) {
      double otmp = 0;
      for (int64 j = 1; j <= isz; j++) {
        if (z[j] && nestSpec[j] == l) {
          otmp += std::exp(u[j] / ou[l - ouoff]);
        }
      }
      for (int64 k = inoff; k < inoff + insz; k++) {
        if (nestSpec[k] == l) {
          double itmp = 0;
          for (int64 j = 1; j <= isz; j++) {
            if (z[j] && nestSpec[j] == k) {
              itmp += std::exp(u[j] / in[k - inoff]);
            }
          }
          otmp += itmp ? std::pow(itmp, in[k - inoff] / ou[l - ouoff]) : 0;
        }
      }
      tmp += otmp ? std::pow(otmp, ou[l - ouoff]) : 0;
    }
    p[i] /= tmp;
  }
}

PyObject *runTests(PyObject *, PyObject *) {
  // outer(29:3): 31 31 30 32 30 32 -
  // clean inner: 22 23 24 25 26 27 28
  // -----------
  // inner(21:7): -  26 25 27 30 -  22 28 22 26 -  31 24 30 23 -  24 25 27 32 23
  // class(0:20): 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20
  const int64 nestSpecOld[] = {
      20, 26, 25, 27, 30, 5,  22, 28, 22, 26, 10,
      31, 24, 30, 23, 15, 24, 25, 27, 32, 23, // class

      7,  31, 31, 30, 32, 30, 32, 28, // inner

      3,  30, 31, 32, // outer

      0,
  };
  const int64 isz = nestSpecOld[0];
  const int64 insz = nestSpecOld[isz + 1];
  const int64 ousz = nestSpecOld[isz + insz + 2];

  int64 nestSpec[sizeof(nestSpecOld) / sizeof(int64) - 2];
  nestSpec[0] = nestSpecOld[0];
  for (int64 ii = 0, count = 0; nestSpecOld[ii];
       ii += nestSpecOld[ii] + 1, count++) {
    for (int64 i = ii + 1; i <= ii + nestSpecOld[ii]; i++) {
      int64 val = nestSpecOld[i];
      if (val > isz) {
        val--;
      }
      if (val > isz + insz) {
        val--;
      }
      nestSpec[i - count] = val;
    }
  }
  nestSpec[sizeof(nestSpec) / sizeof(int64) - 1] = 0;

  UtilAdder<double> utilAdder(sizeof(nestSpec) / sizeof(int64), nestSpec);
  std::uniform_int_distribution<int> disti(0, 1);
  std::uniform_real_distribution<double> distr(-10, 10);
  std::uniform_real_distribution<double> dists(0.1, 1);
  for (int spam = 0; spam < 1000; spam++) {
    std::mt19937 mtrand(spam);
    bool z[isz + 1];
    bool zpass = false;
    for (int i = 0; i <= isz; i++) {
      z[i] = disti(mtrand);
      if (z[i]) {
        zpass = true;
      }
    }
    if (!zpass) {
      // no reason for the 4
      z[4] = true;
    }
    double u[isz + 1];
    for (int i = 0; i <= isz; i++) {
      u[i] = distr(mtrand);
    }
    double vars[insz + ousz];
    for (int i = 0; i < insz + ousz; i++) {
      vars[i] = dists(mtrand);
    }
    double *in = vars;
    double *ou = in + insz;
    double p[isz + 1];
    slowWriteProbs3L(nestSpecOld, z, u, in, ou, p);
    utilAdder.setNestMods(&vars[0], isz + 1);
    utilAdder.clearVals();
    for (int i = 0; i <= isz; i++) {
      if (z[i]) {
        utilAdder.set(i, u[i], 0);
      }
    }
    utilAdder.setLayers();
    for (int i = 0; i <= isz; i++) {
      double utilAdderP = utilAdder.getProb(i);
      if (std::abs(utilAdderP - p[i]) >= 1e-5) {
        std::cerr << "on iteration " << spam << ", class " << i
                  << ", utilAdder.getProb(i) == " << utilAdderP
                  << " while p[i] == " << p[i] << std::endl;
        PyErr_SetString(PyExc_RuntimeError, "read above");
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
