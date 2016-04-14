/**
 * @file parametric_conditional.hpp
 * @author Vahid Bastani
 *
 * Generic class for parametric conditional density/distribution function.
 */
#ifndef SSMPACK_DISTRIBUTION_PARAMETRIC_CONDITIONAL_HPP
#define SSMPACK_DISTRIBUTION_PARAMETRIC_CONDITIONAL_HPP

#include <vector>
#include <type_traits>
#include <algorithm>
#include <utility>

namespace ssmpack {
namespace distribution {

template <typename TPDF, typename TParamMap>
class Conditional {
 public:
  Conditional(TPDF pdf, TParamMap map) : pdf_(pdf), map_(map) {}
  template <typename... Args>
  auto random(Args... args) -> decltype(std::declval<TPDF>().random()) {
    pdf_.parameterize(map_(args...));
    return pdf_.random();
  }

  template <typename... Args>
  double likelihood(decltype(std::declval<TPDF>().random()) rv, Args... args) {
    pdf_.parameterize(map_(args...));
    return pdf_.likelihood(rv);
  }

 private:
  TPDF pdf_;
  TParamMap map_;
};

} // namespace distribution
} // namespace ssmpack

#endif // SSMPACK_DISTRIBUTION_PARAMETRIC_CONDITIONAL_HPP
