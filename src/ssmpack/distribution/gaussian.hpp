#pragma once

#include "ssmpack/random/generator.hpp"

#include <armadillo>

#include <cmath>

namespace ssmpack {
namespace distribution {

template <size_t D>
class Gaussian {

 public:
  //  using TRandomVAR = arma::vec::fixed<D>;
  //  using TParticle = Particle<TRandomVAR>;
  using TParameterVAR = std::tuple<arma::vec::fixed<D>, arma::mat::fixed<D, D>>;

 private:
  arma::vec::fixed<D> mean_;
  arma::mat::fixed<D, D> covariance_;
  std::normal_distribution<double> normal_;
  static constexpr double pi = 3.1415926535897;

  arma::mat::fixed<D, D> inv_cov_;
  double part_;
  arma::mat::fixed<D, D> chol_dec_;

 public:
  Gaussian() : Gaussian(arma::zeros(D), arma::eye(D,D)) {}
  //       Gaussian(const size_t dimention): dist(dimention) {}
  Gaussian(arma::vec::fixed<D> mean, arma::mat::fixed<D, D> covariance)
      : mean_(std::move(mean)), covariance_(std::move(covariance)) {
    calcDistConstants();
  }

  arma::vec::fixed<D> random() {
    arma::vec::fixed<D> rnd; // emmm any better way?
    rnd.imbue(
        [&]() { return normal_(random::Generator::get().getGenerator()); });
    return mean_ + chol_dec_ * rnd;
  }

  double likelihood(const arma::vec &rv) {
    const auto diff = rv - mean_;
    const arma::vec tmp = diff.t() * inv_cov_ * diff;
    const double expt = -tmp(0) / 2;
    return part_ * std::exp(expt);
  }

  Gaussian &parameterize(const TParameterVAR &parameters) {
    mean_ = std::get<0>(parameters);
    covariance_ = std::get<1>(parameters);
    calcDistConstants();
    return (*this);
  }

 private:
  void calcDistConstants() {
    inv_cov_ = arma::inv(covariance_);

    double det_cov = arma::det(covariance_);
    double den_pi = 1 / std::pow(2 * pi, D / 2);
    part_ = den_pi * (1 / std::sqrt(det_cov));

    chol_dec_ = arma::chol(covariance_);
  }
};

} // ssmpack
} // distribution
