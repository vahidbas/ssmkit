/**
 * @file gaussian.hpp
 * @author Vahid Bastani
 *
 * Multivariate Gaussian distribution
 */
#ifndef SSMPACK_DISTRIBUTION_GAUSSIAN_HPP
#define SSMPACK_DISTRIBUTION_GAUSSIAN_HPP

#include "ssmpack/random/generator.hpp"

#include <armadillo>

#include <cmath>

namespace ssmpack {
namespace distribution {

/** A D-dimensional multivariate Gaussian distribution.
 * \f{equation}{ \mathcal{N}(\mathbf{x}| \mu, \Sigma) = 
 * \frac{1}{(2\pi)^{D/2}\sqrt{|\Sigma|}}
 * \exp(-\frac{1}{2}\mathbf{x}^T\Sigma^{-1}\mathbf{x})\f}
 */
template <size_t D>
class Gaussian {

 public:
  //! Data type of the parameter variable \f$ \theta = \{\mu, \Sigma\} \f$.
  using TParameterVAR = std::tuple<arma::vec::fixed<D>, arma::mat::fixed<D, D>>;

 private:
  //! mean vector \f$\mu\f$.
  arma::vec::fixed<D> mean_;
  //! covariance matrix \f$\Sigma\f$.
  arma::mat::fixed<D, D> covariance_;
  /** a normal distribution random generator \f$\mathcal{N}(0,1)\f$ */
  std::normal_distribution<double> normal_;
  //! \f$\pi\f$
  static constexpr double pi = 3.1415926535897;
  //! inverse of covariance matrix \f$\Sigma^{-1}\f$.
  arma::mat::fixed<D, D> inv_cov_;
  //! partitioning function \f$\frac{1}{(2\pi)^{D/2}\sqrt{|\Sigma|}}\f$.
  double part_;
  //! Cholesky decomposition of covariance matrix \f$L^TL=\Sigma\f$.
  arma::mat::fixed<D, D> chol_dec_;

 private:
  /**
   * calculates useful constant values for avoiding recalculation for every 
   * call to likelihood().
   */
  void calcDistConstants() {
    inv_cov_ = arma::inv(covariance_);
    // calculate partition function
    double det_cov = arma::det(covariance_);
    double den_pi = 1 / std::sqrt(std::pow(2 * pi, D));
    part_ = den_pi * (1 / std::sqrt(det_cov));
    // Cholesky decomposition
    chol_dec_ = arma::chol(covariance_, "lower");
  }

 public:
  /** Default constructor.
   * Returns D dimensional Gaussian distribution with zero mean and identity
   * covariance matrix.
   */
  Gaussian() : Gaussian(arma::zeros(D), arma::eye(D, D)) {}
  
  /**
   * Returns D dimensional Gaussian with given mean and covariance. The
   * covariance should be positive definite.
   * 
   * @param mean The mean vector.
   * @param covariance The covariance matrix.
   */
  Gaussian(arma::vec::fixed<D> mean, arma::mat::fixed<D, D> covariance)
      : mean_(std::move(mean)), covariance_(std::move(covariance)) {
    calcDistConstants();
  }

  /** Returns a random variable from the distribution.
   */
  arma::vec::fixed<D> random() {
    arma::vec::fixed<D> rnd; // emmm any better way?
    rnd.imbue(
        [&]() { return normal_(random::Generator::get().getGenerator()); });
    return mean_ + chol_dec_ * rnd;
  }

  /** Returns the likelihood of a given random variable.
   * @param rv The random variable for which likelihood is calculated.
   */
  double likelihood(const arma::vec &rv) const {
    const auto diff = rv - mean_;
    const arma::vec tmp = diff.t() * inv_cov_ * diff;
    const double expt = -tmp(0) / 2;
    return part_ * std::exp(expt);
  }

  /** Changes the mean and covariance of the distribution with the given
   * parameters.
   * @param parameters A tuple containing mean and covariance.
   * @return Reference to the current instance.
   */
  Gaussian &parameterize(const TParameterVAR &parameters) {
    return parameterize(std::get<0>(parameters), std::get<1>(parameters));
  }

  /** Changes the mean and covariance of the distribution with the given values.
   * @param mean The mean vector.
   * @param covariance The covariance matrix.
   * @return Reference to the current instance.
   */
  Gaussian &parameterize(const arma::vec::fixed<D> &mean,
                         const arma::mat::fixed<D, D> &covariance) {
    mean_ = mean;
    covariance_ = covariance;
    calcDistConstants();
    return (*this);
  }
  //! Returns the mean vector 
  const arma::vec::fixed<D>& getMean() const { return mean_; }
  //! Returns the covariance matrix
  const arma::mat::fixed<D, D>& getCovariance() const { return covariance_; }
};

} // ssmpack
} // distribution

#endif // SSMPACK_DISTRIBUTION_GAUSSIAN_HPP
