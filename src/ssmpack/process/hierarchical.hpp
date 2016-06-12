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

/** A dynamic Bayesian network constructed as hierarchy of processes
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
  // check if all argument are process types! (is id possible using C++17 concepts) 
  static_assert(AreProcesses<Args...>::value, "");

 private:
  using TPDFs = std::tuple<typename ProcessTraits<Args>::TPDF...>;
  using TParamMaps = std::tuple<typename ProcessTraits<Args>::TParamMap...>;
  using TArities = std::tuple<typename ProcessTraits<Args>::TArity...>;

  static constexpr size_t depth = sizeof...(Args)-1;

 public:
  /** Type of the random variable. 
   * This is a tuple of the random variables of every layer in top-down order
   */
  using TRandomVAR = std::tuple<typename ProcessTraits<Args>::TRandomVAR...>;

 private:
  std::tuple<Args...> processes_;

 public:
  Hierarchical(Args... processes) : processes_{std::move(processes)...} {}

  TRandomVAR initialize() {
    TRandomVAR rvs;
    detail::GetInitialized<depth>::apply(rvs, processes_);
    return rvs;
  }

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
 * get a reference to the process at level I
 */
  template<size_t II>
  typename std::tuple_element<II, std::tuple<Args...>>::type &
  getProcess() {return std::get<II>(processes_); }
};

/** A convenient builder for Hierarchical process.
 * use this for template argument deduction.
 * @param args... Process layers
 * @tparam TArgs... Type of the process layers
 * @return Hierarchical process object build from \p args...
 */
template <class... TArgs>
Hierarchical<TArgs...> makeHierarchical(TArgs... args) {
  return Hierarchical<TArgs...>(args...);
}

} // namespace process
} // namespace ssmpack

#endif // SSMPACK_PROCESS_HIERARCHICAL_HPP
