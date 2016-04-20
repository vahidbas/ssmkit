#pragma once

#include "ssmpack/process/base_process.hpp"
#include "ssmpack/process/traits.hpp"

#include <tao/seq/make_integer_range.hpp> //https://github.com/taocpp/sequences

#include <algorithm>
#include <utility>
#include <vector>

namespace ssmpack {
namespace process {

namespace detail {
//===============================
template <size_t N, size_t NA, size_t D, class TArities>
struct GetRandom {
  template <class TRV, class TP, class TA, size_t... Is>
  static void apply(std::index_sequence<Is...>, TRV &rvs, TP &prc,
                    const TA &args) {
    std::get<N>(rvs) =
        std::get<N>(prc).random(std::get<N - 1>(rvs), std::get<Is>(args)...);
    constexpr size_t arity =
        std::tuple_element<N + 1, TArities>::type::value - 1;
    GetRandom<N + 1, NA + arity, D, TArities>::apply(
        tao::seq::make_index_range<NA, NA + arity>(), rvs, prc, args);
  }
};

template <size_t NA, size_t D, class TArities>
struct GetRandom<0, NA, D, TArities> {
  template <class TRV, class TP, class TA, size_t... Is>
  static void apply(std::index_sequence<Is...>, TRV &rvs, TP &prc,
                    const TA &args) {
    std::get<0>(rvs) = std::get<0>(prc).random(std::get<Is>(args)...);
    constexpr size_t arity = std::tuple_element<1, TArities>::type::value - 1;
    GetRandom<1, NA + arity, D, TArities>::apply(
        tao::seq::make_index_range<NA, NA + arity>(), rvs, prc, args);
  }
};

template <size_t NA, size_t D, class TArities>
struct GetRandom<D, NA, D, TArities> {
  template <class TRV, class TP, class TA, size_t... Is>
  static void apply(std::index_sequence<Is...>, TRV &rvs, TP &prc,
                    const TA &args) {
    std::get<D>(rvs) =
        std::get<D>(prc).random(std::get<D - 1>(rvs), std::get<Is>(args)...);
  }
};
//========================================
template <size_t N>
struct GetInitialized {
  template <class TRV, class TP>
  static void apply(TRV &rvs, TP &prc) {
    std::get<N>(rvs) = std::get<N>(prc).initialize();

    GetInitialized<N - 1>::apply(rvs, prc);
  }
};

template <>
struct GetInitialized<0> {
  template <class TRV, class TP>
  static void apply(TRV &rvs, TP &prc) {
    std::get<0>(rvs) = std::get<0>(prc).initialize();
  }
};

//===============================================================
template <size_t N, size_t NA, size_t D, class TArities>
struct GetLikelihood {
  template <class TRV, class TP, class TA, size_t... Is>
  static void apply(std::index_sequence<Is...>, double &lik, const TRV &rvs,
                    TP &prc, const TA &args) {

    lik *= std::get<N>(prc).likelihood(std::get<N>(rvs), std::get<N - 1>(rvs),
                                       std::get<Is>(args)...);

    constexpr size_t arity =
        std::tuple_element<N + 1, TArities>::type::value - 1;
    GetLikelihood<N + 1, NA + arity, D, TArities>::apply(
        tao::seq::make_index_range<NA, NA + arity>(), lik, rvs, prc, args);
  }
};

template <size_t NA, size_t D, class TArities>
struct GetLikelihood<0, NA, D, TArities> {
  template <class TRV, class TP, class TA, size_t... Is>
  static void apply(std::index_sequence<Is...>, double &lik, const TRV &rvs,
                    TP &prc, const TA &args) {

    lik *= std::get<0>(prc).likelihood(std::get<0>(rvs), std::get<Is>(args)...);

    constexpr size_t arity = std::tuple_element<1, TArities>::type::value - 1;
    GetLikelihood<1, NA + arity, D, TArities>::apply(
        tao::seq::make_index_range<NA, NA + arity>(), lik, rvs, prc, args);
  }
};

template <size_t NA, size_t D, class TArities>
struct GetLikelihood<D, NA, D, TArities> {
  template <class TRV, class TP, class TA, size_t... Is>
  static void apply(std::index_sequence<Is...>, double &lik, const TRV &rvs,
                    TP &prc, const TA &args) {

    lik *= std::get<D>(prc).likelihood(std::get<D>(rvs), std::get<D - 1>(rvs),
                                       std::get<Is>(args)...);
  }
};

} // namespace detail

template <class... Args>
class Hierarchical : public BaseProcess<Hierarchical<Args...>> {
  static_assert(AreProcesses<Args...>::value, "");
  using TPDFs = std::tuple<typename ProcessTraits<Args>::TPDF...>;
  using TParamMaps = std::tuple<typename ProcessTraits<Args>::TParamMap...>;
  using TRandomVARs = std::tuple<typename ProcessTraits<Args>::TRandomVAR...>;
  using TArities = std::tuple<typename ProcessTraits<Args>::TArity...>;

  static constexpr size_t depth = sizeof...(Args)-1;

 public:
  Hierarchical(Args... processes) : processes_{std::move(processes)...} {}

  TRandomVARs initialize() {
    TRandomVARs rvs;
    detail::GetInitialized<depth>::apply(rvs, processes_);
    return rvs;
  }

  template <class... TVARs>
  TRandomVARs random(const TVARs &... args) {
    // what if args are more than what actually required?

    /* we make a variable and pass its reference to the GetRandom class member
     * apply which takes random variable of level 0 and recursively passes the
     * same reference to the GetRandom of other levels until reaches depth.
     */
    TRandomVARs rvs;
    constexpr size_t arity = std::tuple_element<0, TArities>::type::value;
    detail::GetRandom<0, arity, depth, TArities>::apply(
        tao::seq::make_index_range<0, arity>(), rvs, processes_,
        std::make_tuple(args...));
    return rvs;
  }

  template <class... TArgs>
  double likelihood(const TRandomVARs &rvs, const TArgs &... args) {
    double lik;
    constexpr size_t arity = std::tuple_element<0, TArities>::type::value;
    detail::GetLikelihood<0, arity, depth, TArities>::apply(
        tao::seq::make_index_range<0, arity>(), lik, rvs, processes_,
        std::make_tuple(args...));
    return lik;

  }

 private:
  std::tuple<Args...> processes_;
};
} // namespace process
} // namespace ssmpack
