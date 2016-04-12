#pragma once
#include <armadillo>

namespace PROJECT_NAME {
namespace model {
template <size_t N, size_t M> struct LinearGaussian {
  using PARAM_TYPE = std::tuple<arma::vec::fixed<N>, arma::mat::fixed<N, M>>;
  using CV_TYPE = arma::vec::fixed<M>;

  PARAM_TYPE operator()(CV_TYPE &x) {
    return std::make_tuple(dynamic * x, covariance);
  }

  arma::mat::fixed<N, M> dynamic;
  arma::mat::fixed<N, N> covariance;
};
} // namespace model
} // namespace PROJECT_NAME
