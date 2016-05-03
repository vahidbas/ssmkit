#ifndef SSMPACK_MODEL_TRANSITION_MATRIX_HPP
#define SSMPACK_MODEL_TRANSITION_MATRIX_HPP

#include <armadillo>

namespace ssmpack {
namespace map {

template <arma::uword VN>
struct TransitionMatrix {
  using TParameter = arma::vec::fixed<VN>;
  using TConditionVAR = int;

  // should not be overloaded, should not be template
  TParameter operator()(const TConditionVAR &x) const {
    return transfer.col(x);
  }

  arma::mat::fixed<VN, VN> transfer;
};

template <arma::uword VN>
TransitionMatrix<VN> makeTransitionMatrix(arma::mat::fixed<VN, VN> matrix) {
  return {matrix};
}

} // namespace map
} // namespace ssmpack

#endif // SSMPACK_MODEL_TRANSITION_MATRIX_HPP
