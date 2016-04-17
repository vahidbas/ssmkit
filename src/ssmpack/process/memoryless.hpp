/**
 * @file memoryless.hpp
 * @author Vahid Bastani
 *
 * Generic class for first-order Markovian stochastic process.
 */
#ifndef SSMPACK_PROCESS_MEMORYLESS_HPP
#define SSMPACK_PROCESS_MEMORYLESS_HPP

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
template <typename TStateCPDF>
class Memoryless {};

template <typename TPDF, typename TParamMap>
class Memoryless<distribution::Conditional<TPDF, TParamMap>> {

public:
Memoryless(distribution::Conditional<TPDF, TParamMap> cpdf) : cpdf_(cpdf) {}

  template<typename... Args>
  auto random(Args... args) -> decltype(std::declval<TPDF>().random())
  {
    return cpdf_.random(args...);
  }
  private:
distribution::Conditional<TPDF, TParamMap> cpdf_;
};

} // namespace process
} // namespace ssmpack

#endif // SSMPACK_PROCESS_MEMORYLESS_HPP
