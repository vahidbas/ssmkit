/**
 * @file hierarchical.hpp
 * @author Vahid Bastani
 *
 * Implements class representing Hierarchical dynamic Bayesian network.
 */
#ifndef SSMPACK_PROCESS_HIERARCHICAL_HPP
#define SSMPACK_PROCESS_HIERARCHICAL_HPP

#include "ssmpack/process/base_process.hpp"
#include "ssmpack/process/traits.hpp"
#include "ssmpack/process/hierarchical_detail.hpp"

#include <utility>
#include <tuple>

namespace ssmpack {
namespace process {

/** A stochastic process constructed as hierarchy of stochastic processes
 *
 * A hierarchical dynamic Bayesian network is constructed by vertically
 * stacking layers of stochastic processes such that in every time-slice a layer is
 * only dependent on its immediate upper layer.
 *
 * \image html hierarchical.png "Dynamic Bayesian Network model of Hierarchical process"
 *
 * @tparam Args... Type of the process levels
 */
template <class... Args>
class Hierarchical : public BaseProcess<Hierarchical<Args...>> {
  // check if all argument are process types! (can C++17 concepts used instead) 
  static_assert(AreProcesses<Args...>::value, "");

 private:
  //! Number of condition variables of each process \f$(N^0, N^1, \cdots, N^L)\f$
  using TArities = std::tuple<typename ProcessTraits<Args>::TArity...>;
  //! Number of levels \f$L\f$
  static constexpr size_t depth = sizeof...(Args)-1;

 public:
  /** Type of the random variable. 
   * This is a tuple of the random variables of every layer in top-down order
   * \f$(\mathbf{x}_k^0, \cdots, \mathbf{x}_k^L)\f$.
   */
  using TRandomVAR = std::tuple<typename ProcessTraits<Args>::TRandomVAR...>;

 private:
  //! Process object of every level
  std::tuple<Args...> processes_;

 public:
  /** Constructor
   *
   * Construct a Hierarchical process object from a sequence of \p processes.
   * The first control variable of every process layer other than first one 
   * \f$(y^{1,0}, \cdots, y^{L,0})\f$ is used to connect upper layer to lower layer.
   *
   * @note use ::makeHierarchical for convenient template argument deduction
   * @param[in] processes ... Process objects of each level in top-down order
   * @pre Except the first process layer every other layer depends on its upper
   * layer. The first control variable of all the process object other than first one
   * is used to connect layers. Thus, it should have the same type as the upper
   * layer random variable. Otherwise compilation will fail.
   */
  Hierarchical(Args... processes) : processes_{std::move(processes)...} {}

  /** Initialize process
   *
   * Initialize and return initial random variable of all layers.
   */
  TRandomVAR initialize() {
    TRandomVAR rvs;
    detail::GetInitialized<depth>::apply(rvs, processes_);
    return rvs;
  }
  /** Sample random variable
   *
   * Sample one random variable from all layers \f$(\mathbf{x}_k^0, \cdots, \mathbf{x}_k^L)\f$.
   *
   * @param[in] args ... Control variables in top-down order \f$(y^{0,0}_k, \cdots,
   * y^{0,N^0}_k, y^{1,0}_k, \cdots, y^{1,N^1}_k, \cdots, y^{L,0}_k, \cdots, y^{L,N^L}_k)\f$
   *
   * @return \f$(\mathbf{x}_k^0, \cdots, \mathbf{x}_k^L)\f$
   * @par Side Effects
   * the method calls \p random method of every process layer. Depending on type
   * of the process layer its internal state may changes.
   */
  template <class... TVARs>
  TRandomVAR random(const TVARs &... args) {
    // what if args are more than what actually required?

    /* we make a variable and pass its reference to the GetRandom class member
     * apply which takes random variable of level 0 and recursively passes the
     * same reference to the GetRandom of other levels until reaches depth.
     */
    TRandomVAR rvs;
    constexpr size_t arity = std::tuple_element<0, TArities>::type::value;
    detail::GetRandom<0, arity, depth, TArities>::apply(
        std::make_index_sequence<arity>(), rvs, processes_,
        std::make_tuple(args...));
    return rvs;
  }

  /** Calculate likelihood
   *
   * calculate likelihood of one random variable \f$(\mathbf{x}_k^0, \cdots, \mathbf{x}_k^L)\f$
   * given internal states \f$(\mathbf{x}_{k-1}^0, \cdots, \mathbf{x}_{k-1}^L)\f$ and
   * control variables \f$(y^{0,0}_k, \cdots,
   * y^{0,N^0}_k, y^{1,0}_k, \cdots, y^{1,N^1}_k, \cdots, y^{L,0}_k, \cdots, y^{L,N^L}_k)\f$
   *
   * @param[in] rvs random variable whose likelihood is calculated 
   * \f$(\mathbf{x}_k^0, \cdots, \mathbf{x}_k^L)\f$
   * @param[in] args ... Control variables in top-down order \f$(y^{0,0}_k, \cdots,
   * y^{0,N^0}_k, y^{1,0}_k, \cdots, y^{1,N^1}_k, \cdots, y^{L,0}_k, \cdots, y^{L,N^L}_k)\f$
   *
   * @return Likelihood of \p rvs
   */
  template <class... TArgs>
  double likelihood(const TRandomVAR &rvs, const TArgs &... args) {
    double lik;
    constexpr size_t arity = std::tuple_element<0, TArities>::type::value;
    detail::GetLikelihood<0, arity, depth, TArities>::apply(
        tao::seq::make_index_range<0, arity>(), lik, rvs, processes_,
        std::make_tuple(args...));
    return lik;
  }

 /**
 * get a reference to the process at level L
 *
 * @par Example
 * @code{.cpp}
 * // get the second level process of hierarchical_process
 * second_level = hierarchical_process.getProcess<1>()
 * @endcode
 */
  template<size_t L>
  typename std::tuple_element<L, std::tuple<Args...>>::type &
  getProcess() {return std::get<L>(processes_); }
};

/** A convenient builder for Hierarchical process.
 * use this for template argument deduction.
 * @param args ... Process layers, see Hierarchical::Hierarchical 
 * @tparam TArgs ... Type of the process layers
 * @return Hierarchical process object build from \p args...
 * @pre See Hierarchical::Hierarchical 
 */
template <class... TArgs>
Hierarchical<TArgs...> makeHierarchical(TArgs... args) {
  return Hierarchical<TArgs...>(args...);
}

} // namespace process
} // namespace ssmpack

#endif // SSMPACK_PROCESS_HIERARCHICAL_HPP
