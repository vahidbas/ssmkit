#pragma once

#include <tuple>
#include <algorithm>
#include <vector>
#include <mlpack/core.hpp>

#include "ssmpack/distribution/particle.hpp"

namespace ssmpack {
namespace distribution {
template <size_t D>
class Gaussian {

 public:
  using TRandomVAR = arma::vec::fixed<D>;
  using TParticle = Particle<TRandomVAR>;
  using TParameterVAR = std::tuple<arma::vec::fixed<D>, arma::mat::fixed<D, D>>;

  Gaussian() : dist(D){};
  //       Gaussian(const size_t dimention): dist(dimention) {}
  Gaussian(const arma::vec &mean, const arma::mat &covariance)
      : dist(mean, covariance) {}

  arma::vec random() { return dist.Random(); }
  double likelihood(const arma::vec &observation) {
    return dist.Probability(observation);
  }

  // sample one particle
  TParticle particle() {
    TParticle p;
    p.point = random();
    p.weight = likelihood(p.point);
    return p;
  }

  // sample N particles
  void particle(std::vector<TParticle> &pars, size_t N) {
    pars.clear();
    pars.resize(N);
    std::generate_n(pars.begin(), N, [this]() { return particle(); });
  }

  Gaussian &parameterize(const TParameterVAR &parameters) {
    dist.Mean() = std::get<0>(parameters);
    dist.Covariance(std::get<1>(parameters));
    return (*this);
  }

 private:
  mlpack::distribution::GaussianDistribution dist;
};

} // ssmpack
} // distribution
