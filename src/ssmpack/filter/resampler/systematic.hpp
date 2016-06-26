/**
 * @file systematic.hpp
 * @author Vahid Bastani
 *
 * Systematic resampling method
 */
#ifndef SSMPACK_FILTER_RESAMPLER_SYSTEMATIC
#define SSMPACK_FILTER_RESAMPLER_SYSTEMATIC

#include "ssmpack/filter/resampler/base.hpp"
#include "ssmpack/random/generator.hpp"

#include <armadillo>

#include <random>

namespace ssmpack {
namespace filter {
namespace resampler {

/** Implements systematic resampling method
 */
template <class Criterion>
class Systematic : public BaseResampler<Systematic<Criterion>> {
  friend class BaseResampler<Systematic<Criterion>>;

 private:
  std::uniform_real_distribution<double> uniform_;

 protected:
  arma::vec generateOrderedNumbers(const int &num_par) {
    double u0 = uniform_(random::Generator::get().getGenerator());
    arma::vec u(num_par);
    int k = 0;
    u.imbue([&u0, &num_par, &k]() { return (k++ + u0) / num_par; });
    return u;
  }

 public:
  Systematic(Criterion criterion)
      : BaseResampler<Systematic<Criterion>>(criterion) {}
};

template<class Criterion>
Systematic<Criterion> makeSystematic(Criterion criterion){
  return Systematic<Criterion>(criterion);
}

} // namespace resampler
} // namespace filter
} // namespace ssmpack
#endif // SSMPACK_FILTER_RESAMPLER_SYSTEMATIC
