#include <cassert>
#include <cstring>
#include <memory>
#include <string>

#include <cppad/ipopt/solve.hpp>

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

  /// long format, i->t,i,x[0..xsz-1],n,z, and sz is number of such
  template <class Iter> bool initData(Iter iter, std::size_t sz) {
    entries.clear();
    if (!sz) {
      return true;
    }
    std::size_t numE = 1;
    Iter it = iter;
    long ct = it->t;
    for (std::size_t i = sz; --i;) {
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
    pool.reset(new char[xoff + sizeof(double) * xstride * numE](0));
    bool *zp = reinterpret_cast<bool *>(pool.get());
    long *np = reinterpret_cast<long *>(&pool[noff]);
    double *xp = reinterpret_cast<double *>(&pool[xoff]);
    auto eiter = entries.begin();
    eiter->t = iter->t - 1; // so it initializes properly
    for (std::size_t i = sz; i--; ++iter) {
      if (eiter->t != iter->t) {
        ++eiter;
        eiter->t = iter->t;
        eiter->x = xp += xstride;
        eiter->n = np += nstride;
        eiter->z = zp += zstride;
      }
      if (iter->i && iter->z) {
        std::memcpy(xp + (iter->i - 1), iter->x, sizeof(double) * xsz);
        zp[iter->i - 1] = true;
      }
      np[iter->i] = iter->n;
    }
  }
};

typedef CppAD::AD<double> ADd;
typedef CPPAD_TESTVECTOR(ADd) ADvector;

/// beta[(i-1)*xsz + k] = (beta_i)_k
struct FG_eval {
  Dataset &d;
  void operator()(ADvector &fg, const ADvector &beta) const {
    assert(fg.size() == 1);
    assert(beta.size() == d.isz * d.xsz);
    ADd t = 0;
    for (const Entry &e : d.entries) {
      ADd tmp1 = 0;
      ADd tmp2 = 0;
      ADd nt = e.n[0];
      ADd lnt = CppAD::lgamma(e.n[0] + 1);
      for (long i1 = d.isz; i--;) {
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
        lnt += CppAD::lgamma(e.n[i1 + 1] + 1);
      }
      t += CppAD::lgamma(nt + 1) - lnt + tmp1 - nt * CppAD::log1p(tmp2);
    }
    fg[0] = t;
  }
};

int main() {
  //
}

