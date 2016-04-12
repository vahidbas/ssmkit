#pragma once
#include <type_traits>
#include <boost/function_types/result_type.hpp>
#include <boost/function_types/parameter_types.hpp>
#include <boost/function_types/function_arity.hpp>

#include <vector>
#include <algorithm>

#include "ssmpack/distribution/particle.hpp"

namespace PROJECT_NAME {
namespace distribution {

template <typename DIST_TYPE, typename PARAM_FUNC>
class ParametericConditionalDistribution {
  // note: it is not thraed-safe
public:
  using URDIST = typename std::remove_reference<DIST_TYPE>::type;
  using URFUNC = typename std::remove_reference<PARAM_FUNC>::type;

  using CV_TYPE = typename URFUNC::CV_TYPE;
  using RV_TYPE = typename URDIST::RV_TYPE;
  using PARAM_TYPE = typename URFUNC::PARAM_TYPE;
  using PARTICLE_TYPE = Particle<RV_TYPE>;

public:
  ParametericConditionalDistribution(DIST_TYPE d, PARAM_FUNC f)
      : dist(d), param_func(f){};

  RV_TYPE Random(CV_TYPE cv) {
    return dist.Parameterize(param_func(cv)).Random();
  }

  const DIST_TYPE &getDistribution(CV_TYPE cv) {
    return dist.Parameterize(param_func(cv));
  }

  double Likelihood(RV_TYPE rv, CV_TYPE cv) {
    return dist.Parameterize(param_func(cv)).Likelihood(rv);
  }

  // sample one particle from conditioned distribution
  PARTICLE_TYPE ParticleSample(CV_TYPE cv) {
    Particle<RV_TYPE> p;
    dist.Parameterize(param_func(cv));
    p.point = dist.Random();
    p.weight = dist.Likelihood(p.point);
    return p;
  }

  // sample N particle from conditioned distribution
  void ParticleSample_N(std::vector<PARTICLE_TYPE> &pars, size_t N,
                        CV_TYPE cv) {
    pars.clear();
    pars.resize(N);
    std::generate_n(pars, N, [cv]() { return ParticleSample(cv); });
  }

  template <typename CV_DIST>
  double ApproxiamteMarginalLikelihood_MC(RV_TYPE rv, size_t N,
                                          CV_DIST cv_dist) {
    std::vector<typename CV_DIST::PARTICLE_TYPE> pars;
    cv_dist.ParticleSample_N(pars, N);
    normalize_particles(pars);
    double lik = 0;
    std::for_each(pars.begin(), pars.end(),
                  [&lik](typename CV_DIST::PARTICLE_TYPE &par) {
                    lik += par.weight * Likelihood(par.point);
                  });
  }

private:
  DIST_TYPE dist;
  PARAM_FUNC param_func;
};

template <typename T1, typename T2>
ParametericConditionalDistribution<T1, T2>
makeParametericConditionalDistribution(T1 &&t1, T2 &&t2) {
  return ParametericConditionalDistribution<T1, T2>(std::forward<T1>(t1),
                                                    std::forward<T2>(t2));
}
} // namespace distribution
} // namespace PROJECT_NAME

