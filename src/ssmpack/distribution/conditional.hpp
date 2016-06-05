/**
 * @file conditional.hpp
 * @author Vahid Bastani
 *
 * Implementation of Generic class for parametric conditional density/distribution function.
 */
#ifndef SSMPACK_DISTRIBUTION_PARAMETRIC_CONDITIONAL_HPP
#define SSMPACK_DISTRIBUTION_PARAMETRIC_CONDITIONAL_HPP

#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>

namespace ssmpack {
namespace distribution {

/** Conditional distribution function
 *
 * Let \f$x \in X\f$ and \f$(y_0, \cdots, y_N) \in \{Y_0, \cdots, Y_n\}  \f$, the class defines a conditional distribution of
 * \f$x\f$ given \f$y_0, \cdots, y_N\f$:
 * \f{equation}{p(x|y_0, \cdots, y_N) = \mathcal{F}(g(y_0, \cdots, y_N))\f}
 * where \f$\mathcal{F}(\theta)\f$ is a probability distribution with
 * parameter \f$\theta \in \Theta \f$ and \f$g:\{Y_0, \cdots, Y_n\} \rightarrow \Theta\f$ 
 * is a function that maps a variables \f$(y_0^*, \cdots, y_N^*)\f$ to a parameter
 * \f$\theta^*\f$, i.e. \f$\theta^* = g(y_0^*, \cdots, y_N^*)\f$.
 *
 * @tparam TPDF Type of the distribution \f$\mathcal{F}(\theta)\f$
 * @tparam TParamMap Type of the parameter map \f$g(.)\f$
 */
template <typename TPDF, typename TParamMap>
class Conditional {
 public:
  /** Constructors returns a conditional distribution object.
   * @note use ::makeConditional builder for convenient template argument
   * deduction.
   * @param pdf A probability distribution object that implements \f$\mathcal{F}(\theta)\f$.
   * @param map A callable object that implements \f$g(.)\f$.
   * @pre \p pdf should provide \a random, \a likelihood and \a parameterize
   * methods, e.g. Gaussian. \p map should be callable with return type equivalent
   * to parameter type of \p pdf, e.g. map::LinearGaussian.
   */
  Conditional(TPDF pdf, TParamMap map)
      : pdf_(std::move(pdf)), map_(std::move(map)) {}
  /** Sample from distribution
   *
   * @param args... Condition variables \f$y_0, \cdots, y_N\f$.
   * @return random variable \f$x\f$.
   */
  template <typename... Args>
  auto random(const Args &... args) -> decltype(std::declval<TPDF>().random()) {
    pdf_.parameterize(map_(args...));
    return pdf_.random();
  }

  /** Calculate the likelihood of a random variable
   *
   * @param args... Condition variables \f$y_0, \cdots, y_N\f$.
   * @param rv random variable \f$x\f$.
   * @return likelihood \f$p(x|y_0, \cdots, y_N)\f$.
   */
  template <typename... Args>
  double likelihood(const decltype(std::declval<TPDF>().random()) &rv,
                    const Args &... args) {
    pdf_.parameterize(map_(args...));
    return pdf_.likelihood(rv);
  }

  //! Returns a reference to \f$\mathcal{F}(\theta)\f$.
  const TPDF & getPDF() const {return pdf_;}
  //! Returns a reference to \f$g(.)\f$.
  const TParamMap & getParamMap() const {return map_;}
 private:
  //! \f$\mathcal{F}(\theta)\f$.
  TPDF pdf_;
  //! \f$g(.)\f$.
  TParamMap map_;
};

/** Convenient builder that returns a conditional distribution object.
 * Use this when you want to use type deduction for \p TPDF and \p TParamMap
 * @return Conditional object
 * @param pdf A probability distribution object that implements \f$\mathcal{F}(\theta)\f$.
 * @param map A callable object that implements \f$g(.)\f$.
 * @pre \p pdf should provide \a random, \a likelihood and \a parameterize
 * methods, e.g. Gaussian. \p map should be callable with return type equivalent
 * to parameter type of \p pdf, e.g. map::LinearGaussian.
 */
template <class TPDF, class TParamMap>
Conditional<TPDF, TParamMap> makeConditional(TPDF pdf, TParamMap map) {
  return Conditional<TPDF, TParamMap>(std::move(pdf),
                                      std::move(map));
}

} // namespace distribution
} // namespace ssmpack

#endif // SSMPACK_DISTRIBUTION_PARAMETRIC_CONDITIONAL_HPP
