#ifndef SSMPACK_MODEL_LINEAR_GAUSSIAN_HPP
#define SSMPACK_MODEL_LINEAR_GAUSSIAN_HPP

#include <armadillo>

namespace ssmpack {
namespace model {

template <size_t VN, size_t VM>
struct LinearGaussian {
  using TParameter = std::tuple<arma::vec::fixed<VN>, arma::mat::fixed<VN, VM>>;
  using TConditionVAR = arma::vec::fixed<VM>;

  TParameter operator()(TConditionVAR &x) {
    return std::make_tuple(dynamic * x, covariance);
  }

  arma::mat::fixed<VN, VM> dynamic;
  arma::mat::fixed<VN, VN> covariance;
};

} // namespace model
} // namespace ssmpack

#endif //SSMPACK_MODEL_LINEAR_GAUSSIAN_HPP
