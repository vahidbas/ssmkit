#ifndef SSMPACK_MODEL_TRANSITION_MATRIX_HPP
#define SSMPACK_MODEL_TRANSITION_MATRIX_HPP

#include <armadillo>

namespace ssmkit {
namespace map {

struct TransitionMatrix {
  using TParameter = arma::vec;
  using TConditionVAR = int;
  
  TransitionMatrix(arma::mat t) : transfer{t} {}
  // should not be overloaded, should not be template
  TParameter operator()(const TConditionVAR &x) const {
    return transfer.col(x);
  }

  arma::mat transfer;
};


} // namespace map
} // namespace ssmkit

#endif // SSMPACK_MODEL_TRANSITION_MATRIX_HPP
