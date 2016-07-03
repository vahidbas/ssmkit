#ifndef SSMPACK_MODEL_SWITCHING_ADDITIVE_LINEAR_GAUSSIAN_HPP
#define SSMPACK_MODEL_SWITCHING_ADDITIVE_LINEAR_GAUSSIAN_HPP

#include <armadillo>

namespace ssmkit {
namespace map {

struct SwitchingAdditiveLinearGaussian {
  using TParameter = std::tuple<arma::vec, arma::mat>;
  using TConditionVAR = arma::vec;

  SwitchingAdditiveLinearGaussian(arma::mat trans, arma::mat cov, arma::mat b)
      : biases{b}, transfer{trans}, covariance{cov} {}
  // should not be overloaded, should not be template
  TParameter operator()(const TConditionVAR &x, const int &k) const {
    return std::make_tuple(transfer * x + biases.col(k), covariance);
  }

  arma::mat biases;
  arma::mat transfer;
  arma::mat covariance;
};


} // namespace map
} // namespace ssmkit

#endif // SSMPACK_MODEL_SWITCHING_ADDITIVE_LINEAR_GAUSSIAN_HPP
