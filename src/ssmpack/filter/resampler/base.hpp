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
     auto u_it = u.begin();

     Particles old_pars = pars;

     pars.each_col([&u_it, &old_pars,
                    &ws](arma::Col<typename Particles::elem_type> &col) {
       col = old_pars.col(
           static_cast<arma::uvec>(arma::find(ws > *u_it++, 1, "first"))(0));
     });

     w.fill(1.0 / w.n_rows);
   }
};

} // namespace resampler
} // namespace filter
} // namespace ssmpack
#endif //SSMPACK_FILTER_RESAMPLER_IDENTITY
