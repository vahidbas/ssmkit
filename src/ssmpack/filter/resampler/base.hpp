/**
 * @file base.hpp
 * @author Vahid Bastani
 *
 * Base class for resampling method
 *
 * J. D. Hol, T. B. Schon and F. Gustafsson, "On Resampling Algorithms for
 * Particle Filters," Nonlinear Statistical Signal Processing Workshop, 2006
 * IEEE, Cambridge, UK, 2006, pp. 79-82
 */
#ifndef SSMPACK_FILTER_RESAMPLER_BASE
#define SSMPACK_FILTER_RESAMPLER_BASE

#include <armadillo>

namespace ssmpack {
namespace filter {
namespace resampler {

/** Base resampling class
 */
template<class T>
class BaseResampler;

template <template <class> class Method, class Criterion>
class BaseResampler<Method<Criterion>> {
  protected:
  Criterion criterion_;
  
  public:
   BaseResampler(Criterion criterion) : criterion_{criterion} {}

   template <class Particles, class Weights>
   void operator()(Particles &pars, Weights &w) {
     // return if resampling criterion is false
     if (!criterion_(w))
       return;

     // generate ordered numbers
     auto u = static_cast<Method<Criterion> *>(this)
                  ->generateOrderedNumbers(w.n_rows);

     auto ws = arma::cumsum(w);
     auto ind = arma::uvec(w.n_rows);
     int cnt = 0;
     ind.imbue([&cnt, &ws, &u]() {
       return static_cast<arma::uvec>(arma::find(ws > u[cnt++], 1, "first"))(0);
     });

     auto old_pars = pars;

     cnt = 0;
     pars.each_col([&ind, &old_pars, &cnt](arma::vec &col) {
       col = old_pars.col(ind[cnt++]);
     });

     w.fill(1.0 / w.n_rows);
   }
};

} // namespace resampler
} // namespace filter
} // namespace ssmpack
#endif //SSMPACK_FILTER_RESAMPLER_IDENTITY
