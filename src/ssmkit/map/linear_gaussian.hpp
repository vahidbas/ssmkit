#ifndef SSMPACK_MODEL_LINEAR_GAUSSIAN_HPP
#define SSMPACK_MODEL_LINEAR_GAUSSIAN_HPP

#include <armadillo>

namespace ssmkit {
namespace map {

struct LinearGaussian {
  using TParameter = std::tuple<arma::vec, arma::mat>;
  using TConditionVAR = arma::vec;
  
  LinearGaussian(arma::mat trans, arma::mat cov) : transfer{trans},
  covariance{cov} {}
// should not be overloaded, should not be template
  TParameter operator()(const TConditionVAR &x) const {
    return std::make_tuple(transfer * x, covariance);
  }

  arma::mat transfer;
  arma::mat covariance;
};

} // namespace map
} // namespace ssmkit

#endif //SSMPACK_MODEL_LINEAR_GAUSSIAN_HPP
