/**
 * @file parametric_conditional.hpp
 * @author Vahid Bastani
 *
 * Generic class for parametric conditional density/distribution function.
 */
#ifndef SSMPACK_DISTRIBUTION_PARAMETRIC_CONDITIONAL_HPP
#define SSMPACK_DISTRIBUTION_PARAMETRIC_CONDITIONAL_HPP

#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>

namespace ssmpack {
namespace distribution {

template <typename TPDF, typename TParamMap>
class Conditional {
 public:
  Conditional(TPDF pdf, TParamMap map)
      : pdf_(std::move(pdf)), map_(std::move(map)) {}

  template <typename... Args>
  auto random(const Args &... args) -> decltype(std::declval<TPDF>().random()) {
    pdf_.parameterize(map_(args...));
    return pdf_.random();
  }

  template <typename... Args>
  double likelihood(const decltype(std::declval<TPDF>().random()) &rv,
                    const Args &... args) {
    pdf_.parameterize(map_(args...));
    return pdf_.likelihood(rv);
  }

  const TPDF & getPDF() const {return pdf_;}
  const TParamMap & getParamMap() const {return map_;}
 private:
  TPDF pdf_;
  TParamMap map_;
};

template <class TPDF, class TParamMap>
Conditional<TPDF, TParamMap> makeConditional(TPDF pdf, TParamMap map) {
  return Conditional<TPDF, TParamMap>(std::move(pdf),
                                      std::move(map));
}

} // namespace distribution
} // namespace ssmpack

#endif // SSMPACK_DISTRIBUTION_PARAMETRIC_CONDITIONAL_HPP
