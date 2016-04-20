/**
 * @file memoryless.hpp
 * @author Vahid Bastani
 *
 * Generic class for first-order Markovian stochastic process.
 */
#ifndef SSMPACK_PROCESS_MEMORYLESS_HPP
#define SSMPACK_PROCESS_MEMORYLESS_HPP

#include "ssmpack/distribution/conditional.hpp"
#include "ssmpack/process/base_process.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

namespace ssmpack {
namespace process {

/**
 * A first-order Markov process
 */
template <typename TStateCPDF>
class Memoryless {};

template <typename TPDF, typename TParamMap>
class Memoryless<distribution::Conditional<TPDF, TParamMap>>
    : public BaseProcess<
          Memoryless<distribution::Conditional<TPDF, TParamMap>>> {
 private:
  distribution::Conditional<TPDF, TParamMap> cpdf_;

 public:
  Memoryless(distribution::Conditional<TPDF, TParamMap> cpdf)
      : cpdf_(std::move(cpdf)) {}

  auto initialize() -> decltype(std::declval<TPDF>().random()) {
    /* memoryless doesn't have sensible initialization so initialize simply
     * returns the default value of the random variable type.
     */
    decltype(std::declval<TPDF>().random()) rv;
    return rv;
  }

  template <typename... Args>
  auto random(const Args &... args) -> decltype(std::declval<TPDF>().random()) {
    return cpdf_.random(args...);
  }

  template <typename... Args>
  double likelihood(const decltype(std::declval<TPDF>().random()) &rv,
                    const Args &... args) {
    return cpdf_.likelihood(rv, args...);
  }
};

} // namespace process
} // namespace ssmpack

#endif // SSMPACK_PROCESS_MEMORYLESS_HPP
