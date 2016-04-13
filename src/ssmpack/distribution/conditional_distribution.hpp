/**
 * @file parametric_conditional.hpp
 * @author Vahid Bastani
 *
 * Generic class for parametric conditional density/distribution function.
 */
#ifndef SSMPACK_DISTRIBUTION_PARAMETRIC_CONDITIONAL_HPP
#define SSMPACK_DISTRIBUTION_PARAMETRIC_CONDITIONAL_HPP

#include "ssmpack/distribution/particle.hpp"

#include <boost/function_types/result_type.hpp>
#include <boost/function_types/parameter_types.hpp>
#include <boost/function_types/function_arity.hpp>

#include <vector>
#include <type_traits>
#include <algorithm>

namespace ssmpack {
namespace distribution {

template <typename TPDF, typename TParamMap>
class ParametericConditional {
  // note: it is not thraed-safe
 private:
  using TPDF_ = typename std::remove_reference<TPDF>::type;
  using TParamMap_ = typename std::remove_reference<TParamMap>::type;

 public:
  //! Type of the condition variable.
  using TConditionVAR = typename TParamMap_::TConditionVAR;
  //! Type of the random variable.
  using TRandomVAR = typename TPDF_::TRandomVAR;
  //! Type of the distribution parameter.
  using TParameter = typename TParamMap_::TParameter;
  //! Type of the particle pair.
  using TParticle = Particle<TRandomVAR>;

 private:
  TPDF_ pdf_;
  TParamMap_ param_map_;

 public:
  ParametericConditional(TPDF d, TParamMap f)
      : pdf_(d), param_map_(f){};

  TRandomVAR random(TConditionVAR cv) {
    return pdf_.parameterize(param_map_(cv)).random();
  }

  const TPDF &getDistribution(TConditionVAR cv) {
    return pdf_.parameterize(param_map_(cv));
  }

  double likelihood(TRandomVAR rv, TConditionVAR cv) {
    return pdf_.parameterize(param_map_(cv)).likelihood(rv);
  }

  // sample one particle from conditioned distribution
  TParticle particle(TConditionVAR cv) {
    Particle<TRandomVAR> p;
    pdf_.parameterize(param_map_(cv));
    p.point = pdf_.random();
    p.weight = pdf_.likelihood(p.point);
    return p;
  }

  // sample N particle from conditioned distribution
  void particle(std::vector<TParticle> &pars, size_t N, TConditionVAR cv) {
    pars.clear();
    pars.resize(N);
    std::generate_n(pars, N, [cv]() { return particle(cv); });
  }

  template <typename TConditionPDF>
  double approxiamteMarginalLikelihood_MC(TRandomVAR rv, size_t N,
                                          TConditionPDF cv_dist) {
    std::vector<typename TConditionPDF::TParticle> pars;
    cv_dist.particle(pars, N);
    normalize_particles(pars);
    double lik = 0;
    std::for_each(pars.begin(), pars.end(),
                  [&lik](typename TConditionPDF::TParticle &par) {
                    lik += par.weight * likelihood(par.point);
                  });
  }
};

template <typename TPDF, typename TParamMap>
ParametericConditional<TPDF, TParamMap>
makeParametericConditional(TPDF &&pdf, TParamMap &&map) {
  return ParametericConditional<TPDF, TParamMap>(std::forward<TPDF>(pdf),
                                                 std::forward<TParamMap>(map));
}
} // namespace distribution
} // namespace ssmpack

#endif // SSMPACK_DISTRIBUTION_PARAMETRIC_CONDITIONAL_HPP
