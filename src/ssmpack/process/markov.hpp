/**
 * @file markov.hpp
 * @author Vahid Bastani
 *
 * Generic class for first-order Markovian stochastic process.
 */
#ifndef SSMPACK_PROCESS_MARKOV_HPP
#define SSMPACK_PROCESS_MARKOV_HPP

#include "ssmpack/distribution/conditional.hpp"
#include "ssmpack/process/base_process.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

namespace ssmpack {
namespace process {

/** A first-order Markov process.
 * Implementation of markov process defined using initial PDF
 * \f$p(\mathbf{x}_0)\f$ and state transition PDF
 * \f$p(\mathbf{x}_k|\mathbf{x}_{k-1}, y^1_k, \cdots, y^N_k)\f$
 * \image html markov.png "Dynamic Bayesian Network model of Markov process"
 */
template <typename TPDF, typename TParamMap, typename TInitialPDF>
class Markov
    : public BaseProcess<Markov<TPDF, TParamMap, TInitialPDF>> {
 public:

  /** Constructor
   *
   * The process is constructed from an initial PDF (\p init_pdf) and a
   * distribution::Conditional PDF (\p cpdf). The first condition variable of \p cpdf 
   * is used to connect the time slices,
   * i.e. If \p cpdf is \f$p(\mathbf{x_k}|y_0, \cdots, y_N)\f$ then \f$\mathbf{x_{k-1}}\f$ 
   * is passed to \f$y_0\f$ for random() sampling likelihood() calculation. The
   * rest of condition variables (\f$y_1, \cdots, y_N\f$) are disposed to the
   * interface of random() and likelihood() methods. 
   *
   * @note use ::makeMarkov builder for convenient template argument deduction
   * @param init_pdf Initial probability distribution \f$p(\mathbf{x}_0)\f$
   * @param cpdf distribution::Conditional PDF characterizing inter time-slice dependency \f$p(\mathbf{x}_k|\mathbf{x}_{k-1}, y^1_k, \cdots, y^N_k)\f$
   *
   * @pre \p init_pdf should provide \p random and \p likelihood methods.
   * @pre The type of the random variable and the first condition variable of \p cpdf should be the same. The \p cpdf should have at least one condition variable. 
   */
  Markov(distribution::Conditional<TPDF, TParamMap> cpdf, TInitialPDF init_pdf)
      : cpdf_(std::move(cpdf)), init_pdf_(std::move(init_pdf)) {}

  /** initialize process
   *
   * Samples initial random variable \f$\mathbf{x}_0\f$ and stores it
   * internally.
   *
   * @return Initial random variable \f$\mathbf{x}_0\f$.
   * @par Side Effects
   * Sets the internal state to \f$\mathbf{x}_0\f$.
   */
  auto initialize() -> decltype(std::declval<TPDF>().random()) {
    state_ = init_pdf_.random();
    return state_;
  }

  /** Sample from process
   *
   * Samples one random variable \f$\mathbf{x}_k\f$ from the process and stores it internally.
   *
   * @param args ... Process condition (control) variables (\f$y^1_k, \cdots, y^N_k\f$) if any.
   * @return Random variable \f$\mathbf{x}_k\f$.
   * @par Side Effects
   * Sets the internal state to \f$\mathbf{x}_k\f$.
   */
  template <typename... Args>
  auto random(const Args &... args) -> decltype(std::declval<TPDF>().random()) {
    state_ = cpdf_.random(state_, args...);
    return state_; // why not returning reference?
  }

  /** Calculate likelihood
   *
   * Calculate the likelihood of one random variable \f$p(\mathbf{x}_k|\mathbf{x}_{k-1}, y^1_k, \cdots, y^N_k)\f$.
   *
   * @param rv The random variable \f$\mathbf{x}_k\f$.
   * @param args ... Process condition (control) variables (\f$y^1_k, \cdots, y^N_k\f$) if any.
   * @return The likelihood of random variable \f$\mathbf{x}_k\f$.
   *
   */
  template <typename... Args>
  double likelihood(const decltype(std::declval<TPDF>().random()) &rv,
                    const Args &... args) {
    return cpdf_.likelihood(rv, state_, args...);
  }

 //! Returns a reference to internal CPDF 
 distribution::Conditional<TPDF, TParamMap> & getCPDF(){return
 cpdf_;}

 //! Returns a reference to initial PDF
 TInitialPDF & getInitialPDF() {return init_pdf_;}

 private:
  //! CPDF defines inter time-slice dependency
  distribution::Conditional<TPDF, TParamMap> cpdf_;
  //! Initial pdf
  TInitialPDF init_pdf_;
  //! The process state
  decltype(std::declval<TPDF>().random()) state_;
};

/** Convenient builder for Markov process.
 * Use this when you want to use type deduction
 * @return Markov process object
 * @param init_pdf Initial probability distribution \f$p(\mathbf{x}_0)\f$
 * @param cpdf distribution::Conditional PDF characterizing inter time-slice dependency \f$p(\mathbf{x}_k|\mathbf{x}_{k-1}, y^1_k, \cdots, y^N_k)\f$
 *
 * @pre \p init_pdf should provide \p random and \p likelihood methods.
 * @pre The type of the random variable and the first condition variable of \p cpdf should be the same. The \p cpdf should have at least one condition variable. 
 */
template <typename TPDF, typename TParamMap, typename TInitialPDF>
Markov<TPDF, TParamMap, TInitialPDF>
makeMarkov(distribution::Conditional<TPDF, TParamMap> cpdf,
           TInitialPDF init_pdf) {
  return Markov<TPDF, TParamMap, TInitialPDF>(
      cpdf, init_pdf);
}

} // namespace process
} // namespace ssmpack

#endif // SSMPACK_PROCESS_MARKOV_HPP
