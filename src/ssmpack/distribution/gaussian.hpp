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
 * \f[ \mathcal{N}(\mathbf{x}| \mu, \Sigma) = 
 * \frac{1}{(2\pi)^{D/2}\sqrt{|\Sigma|}}
 * \exp(-\frac{1}{2}\mathbf{x}^T\Sigma^{-1}\mathbf{x})\f]
 */
class Gaussian {

 public:
  //! Data type of the parameter variable \f$ \theta = \{\mu, \Sigma\} \f$.
  using TParameterVAR = std::tuple<arma::vec, arma::mat>;

 private:
  //! mean vector \f$\mu\f$.
  arma::vec mean_;
  //! covariance matrix \f$\Sigma\f$.
  arma::mat covariance_;
  /** a normal distribution random generator \f$\mathcal{N}(0,1)\f$ */
  std::normal_distribution<double> normal_;
  //! \f$\pi\f$
  static constexpr double pi = 3.1415926535897;
  //! inverse of covariance matrix \f$\Sigma^{-1}\f$.
  arma::mat inv_cov_;
  //! partitioning function \f$\frac{1}{(2\pi)^{D/2}\sqrt{|\Sigma|}}\f$.
  double part_;
  //! Cholesky decomposition of covariance matrix, i.e. \f$L\f$ such that  \f$LL^T=\Sigma\f$.
  arma::mat chol_dec_;
  //! Dimension \f$D\f$
  int dim_;

 private:
  /**
   * Calculates useful constant values for avoiding recalculation for every 
   * call to likelihood().
   */
  void calcDistConstants() {
    dim_ = mean_.n_rows;
    inv_cov_ = arma::inv(covariance_);
    // calculate partition function
    double det_cov = arma::det(covariance_);
    double den_pi = 1 / std::sqrt(std::pow(2 * pi, dim_));
    part_ = den_pi * (1 / std::sqrt(det_cov));
    // Cholesky decomposition
    chol_dec_ = arma::chol(covariance_, "lower");
  }

 public:
  Gaussian() = delete;
  /** Default constructor.
   * Returns D dimensional Gaussian distribution with zero mean and identity
   * covariance matrix.
   */
  Gaussian(int dim) : Gaussian(arma::zeros(dim), arma::eye(dim, dim)) {}
  
  /**
   * Returns D dimensional Gaussian with given mean and covariance. The
   * covariance should be positive definite.
   * 
   * @param mean The mean vector.
   * @param covariance The covariance matrix.
   */
  Gaussian(arma::vec mean, arma::mat covariance)
      : mean_(std::move(mean)), covariance_(std::move(covariance)) {
    calcDistConstants();
  }

  /** Returns a random variable from the distribution.
   * \f[ \mathbf{x} \sim \mathcal{N}(\mu, \Sigma) \f]
   * @return The random vector \f$\mathbf{x}\f$
   */
  arma::vec random() {
    arma::vec rnd(dim_); // how much overload the vector construction has?
    rnd.imbue(
        [&]() { return normal_(random::Generator::get().getGenerator()); });
    return mean_ + chol_dec_ * rnd;
  }

  /** Returns the likelihood of a given random variable.
   * \f[p(\mathbf{x}) = \frac{1}{(2\pi)^{D/2}\sqrt{|\Sigma|}}
   * \exp(-\frac{1}{2}\mathbf{x}^T\Sigma^{-1}\mathbf{x})\f]
   * @param rv The random variable \f$\mathbf{x}\f$ for which likelihood is calculated.
   * @return \f$p(\mathbf{x})\f$.
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
  Gaussian &parameterize(const arma::vec &mean,
                         const arma::mat &covariance) {
    mean_ = mean;
    covariance_ = covariance;
    calcDistConstants();
    return (*this);
  }
  //! Returns the mean vector 
  const arma::vec& getMean() const { return mean_; }
  //! Returns the covariance matrix
  const arma::mat& getCovariance() const { return covariance_; }
};

} // namespace distribution
} // namespace ssmpack

#endif // SSMPACK_DISTRIBUTION_GAUSSIAN_HPP
