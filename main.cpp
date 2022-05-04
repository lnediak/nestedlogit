#include <cassert>

#include <cppad/ipopt/solve.hpp>

struct Dataset {
  long isz, xsz;
  struct Entry {
    double t, *x; /// x[i * xsz + k] = (x_it)_k where 0<=i<isz and 0<=k<xsz
    long *n1, *n0; /// n[i] = n_it where 0<=i<isz
  };
  std::vector<Entry> entries;

  Dataset(std::istream &is) {
    //
  }
};

typedef CppAD::AD<double> ADd;
typedef CPPAD_TESTVECTOR(ADd) ADvector;

/// beta[i * xsz + k] = (beta_i)_k
struct FG_eval {
  Dataset &d;
  void operator()(ADvector &fg, const ADvector &beta) const {
    assert(fg.size() == 1);
    assert(beta.size() == d.isz * d.xsz);
    ADd t = 0;
    for (const Entry &e : d.entries) {
      ADd tmp1 = 0;
      ADd tmp2 = 0;
      ADd nt = 0;
      for (long i = d.isz; i--;) {
        ADd tmp = 0;
        for (long k = d.xsz; k--;) {
          tmp += e.x[i * d.xsz + k] * beta[i * d.xsz + k];
        }
        tmp1 += e.n1[i] * tmp;
        tmp2 += CppAD::exp(tmp);
        nt += e.n1[i] + e.n0[i];
      }
      t += tmp1 - nt * CppAD::log1p(tmp2);
    }
    fg[0] = t;
  }
};

int main() {
  //
}

