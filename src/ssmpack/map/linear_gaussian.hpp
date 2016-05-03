#ifndef SSMPACK_MODEL_LINEAR_GAUSSIAN_HPP
#define SSMPACK_MODEL_LINEAR_GAUSSIAN_HPP

#include <armadillo>

namespace ssmpack {
namespace map {

template <arma::uword VN, arma::uword VM>
struct LinearGaussian {
  using TParameter = std::tuple<arma::vec::fixed<VN>, arma::mat::fixed<VN, VN>>;
  using TConditionVAR = arma::vec::fixed<VM>;

// should not be overloaded, should not be template
  TParameter operator()(const TConditionVAR &x) const {
    return std::make_tuple(transfer * x, covariance);
  }

  arma::mat::fixed<VN, VM> transfer;
  arma::mat::fixed<VN, VN> covariance;
};

template <arma::uword VN, arma::uword VM>
LinearGaussian<VN,VM> makeLinearGaussian(arma::mat::fixed<VN,VM> transfer,
arma::mat::fixed<VN,VN> covariance)
{
  return {transfer, covariance};
}

} // namespace map
} // namespace ssmpack

#endif //SSMPACK_MODEL_LINEAR_GAUSSIAN_HPP
