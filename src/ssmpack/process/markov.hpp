/**
 * @file markov.hpp
 * @author Vahid Bastani
 *
 * Generic class for first-order Markovian stochastic process.
 */
#ifndef SSMPACK_PROCESS_MARKOV_HPP
#define SSMPACK_PROCESS_MARKOV_HPP

#include "ssmpack/distribution/conditional_distribution.hpp"

#include <type_traits>
#include <vector>
#include <algorithm>
#include <cmath>

namespace ssmpack {
namespace process {

/**
 * A first-order Markov process
 */
template <typename TTransitionCPDF>
class Markov {};

template <typename TPDF, typename TParamMap>
class Markov<distribution::Conditional<TPDF, TParamMap>> {
 public:
  Markov(distribution::Conditional<TPDF, TParamMap> cpdf) : cpdf_(cpdf) {}
  template <typename... Args>
  auto random(Args... args) -> decltype(std::declval<TPDF>().random()) {
    state_ = cpdf_.random(state_, args...);
    return state_;
  }

  template <typename... Args>
  double likelihood(decltype(std::declval<TPDF>().random()) rv, Args... args) {
    return cpdf_.likelihood(rv, state_, args...);
  }
 private:
  distribution::Conditional<TPDF, TParamMap> cpdf_;
  decltype(std::declval<TPDF>().random()) state_;
};

} // namespace process
} // namespace ssmpack

#endif // SSMPACK_PROCESS_MARKOV_HPP
