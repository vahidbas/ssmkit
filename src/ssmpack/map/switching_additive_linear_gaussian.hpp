#ifndef SSMPACK_MODEL_SWITCHING_ADDITIVE_LINEAR_GAUSSIAN_HPP
#define SSMPACK_MODEL_SWITCHING_ADDITIVE_LINEAR_GAUSSIAN_HPP

#include <armadillo>

namespace ssmpack {
namespace map {

template <arma::uword VN, arma::uword VM, arma::uword VK>
struct SwitchingAdditiveLinearGaussian {
  using TParameter = std::tuple<arma::vec::fixed<VN>, arma::mat::fixed<VN, VN>>;
  using TConditionVAR = arma::vec::fixed<VM>;

// should not be overloaded, should not be template
  TParameter operator()(const TConditionVAR &x, const int &k) const {
    return std::make_tuple(transfer * x + biases.col(k), covariance);
  }

  arma::mat::fixed<VM, VK> biases;
  arma::mat::fixed<VN, VM> transfer;
  arma::mat::fixed<VN, VN> covariance;
};

template <arma::uword VN, arma::uword VM, arma::uword VK>
SwitchingAdditiveLinearGaussian<VN, VM, VK>
makeSwitchingAdditiveLinearGaussian(arma::mat::fixed<VN, VM> transfer,
arma::mat::fixed<VN, VN> covariance, arma::mat::fixed<VM, VK> biases) {
  return {biases, transfer, covariance};
}

} // namespace map
} // namespace ssmpack

#endif // SSMPACK_MODEL_SWITCHING_ADDITIVE_LINEAR_GAUSSIAN_HPP
