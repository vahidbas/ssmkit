/**
 * @file memoryless.hpp
 * @author Vahid Bastani
 *
 * Generic class for first-order Memorylessian stochastic process.
 */
#ifndef SSMPACK_PROCESS_MEMORYLESS_HPP
#define SSMPACK_PROCESS_MEMORYLESS_HPP

#include "ssmkit/distribution/conditional.hpp"
#include "ssmkit/process/base_process.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

namespace ssmkit {
namespace process {

/** A Memoryless (independent/white) random process.
 * Implementation of memoryless process defined using CPDF
 * \f$p(\mathbf{x}_k| y^0_k, \cdots, y^N_k)\f$
 * \image html memoryless.png "Dynamic Bayesian Network model of Memoryless process"
 * @note this process has no state.
 */
template <typename TPDF, typename TParamMap>
class Memoryless : public BaseProcess<Memoryless<TPDF, TParamMap>> {
 private:
  //! Core conditional distribution object
  distribution::Conditional<TPDF, TParamMap> cpdf_;

 public:
  
  /** Constructor
   *
   * The process is constructed from a distribution::Conditional PDF (\p cpdf). The
   * condition variables (\f$y_0, \cdots, y_N\f$) are disposed to the
   * interface of random() and likelihood() methods. 
   *
   * @note use ::makeMemoryless builder for convenient template argument deduction
   * @param cpdf distribution::Conditional PDF \f$p(\mathbf{x}_k| y^0_k, \cdots, y^N_k)\f$
   *
   */
  Memoryless(distribution::Conditional<TPDF, TParamMap> cpdf)
      : cpdf_(std::move(cpdf)) {}
  
  
  /** initialize process
   *
   * The memoryless process has no state and initialization. This method is
   * provided for interface consistency of process class.
   *
   * @return Default value of the random variable type.
   */
  auto initialize() -> decltype(std::declval<TPDF>().random()) {
    /* memoryless doesn't have sensible initialization so initialize simply
     * returns the default value of the random variable type.
     */
    decltype(std::declval<TPDF>().random()) rv;
    return rv;
  }

  /** Sample from process
   *
   * Samples one random variable \f$\mathbf{x}_k\f$ from the process.
   *
   * @param args ... Process condition (control) variables (\f$y^0_k, \cdots, y^N_k\f$) if any.
   * @return Random variable \f$\mathbf{x}_k\f$.
   */
  template <typename... Args>
  auto random(const Args &... args) -> decltype(std::declval<TPDF>().random()) {
    return cpdf_.random(args...);
  }

  /** Calculate likelihood
   *
   * Calculate the likelihood of one random variable \f$p(\mathbf{x}_k| y^0_k, \cdots, y^N_k)\f$.
   *
   * @param rv The random variable \f$\mathbf{x}_k\f$.
   * @param args ... Process condition (control) variables (\f$y^0_k, \cdots, y^N_k\f$) if any.
   * @return The likelihood of random variable \f$\mathbf{x}_k\f$.
   *
   */
  template <typename... Args>
  double likelihood(const decltype(std::declval<TPDF>().random()) &rv,
                    const Args &... args) {
    return cpdf_.likelihood(rv, args...);
  }

  //! Returns a reference to internal CPDF 
  distribution::Conditional<TPDF, TParamMap> & 
  getCPDF() {return cpdf_;}
};

/** Convenient builder for Memoryless process.
 * Use this when you want to use type deduction
 * @return Memoryless process object
 * @param cpdf distribution::Conditional PDF defining the process \f$p(\mathbf{x}_k| y^0_k, \cdots, y^N_k)\f$
 *
 */
template <typename TPDF, typename TParamMap>
Memoryless<TPDF, TParamMap>
makeMemoryless(distribution::Conditional<TPDF, TParamMap> cpdf) {
  return Memoryless<TPDF, TParamMap>(cpdf);
}

} // namespace process
} // namespace ssmkit

#endif // SSMPACK_PROCESS_MEMORYLESS_HPP
