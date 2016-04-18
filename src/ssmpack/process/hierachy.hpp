#pragma once

#include "ssmpack/process/traits.hpp"

#include <tao/seq/make_integer_range.hpp> //https://github.com/taocpp/sequences

#include <vector>
#include <algorithm>

namespace ssmpack {
namespace process {

namespace detail {
template <size_t N, size_t NA, size_t D, typename TArities>
struct GetRandom {
  template <typename TRV, typename TP, typename TA, size_t... Is>
  static void apply(std::index_sequence<Is...>, TRV &rvs, TP &prc, TA args) {
    std::get<N>(rvs) =
        std::get<N>(prc).random(std::get<N - 1>(rvs), std::get<Is>(args)...);
    constexpr size_t arity =
        std::tuple_element<N + 1, TArities>::type::value - 1;
    GetRandom<N + 1, NA + arity, D, TArities>::apply(
        tao::seq::make_index_range<NA, NA + arity>(), rvs, prc, args);
  }
};

template <size_t NA, size_t D, typename TArities>
struct GetRandom<0, NA, D, TArities> {
  template <typename TRV, typename TP, typename TA, size_t... Is>
  static void apply(std::index_sequence<Is...>, TRV &rvs, TP &prc, TA args) {
    std::get<0>(rvs) = std::get<0>(prc).random(std::get<Is>(args)...);
    constexpr size_t arity = std::tuple_element<1, TArities>::type::value - 1;
    GetRandom<1, NA + arity, D, TArities>::apply(
        tao::seq::make_index_range<NA, NA + arity>(), rvs, prc, args);
  }
};

template <size_t NA, size_t D, typename TArities>
struct GetRandom<D, NA, D, TArities> {
  template <typename TRV, typename TP, typename TA, size_t... Is>
  static void apply(std::index_sequence<Is...>, TRV &rvs, TP &prc, TA args) {
    std::get<D>(rvs) =
        std::get<D>(prc).random(std::get<D - 1>(rvs), std::get<Is>(args)...);
  }
};
} // namespace detail

template <typename... Args>
struct Hierachy {
  static_assert(AreProcesses<Args...>::value, "");
  using TPDFs = std::tuple<typename ProcessTraits<Args>::TPDF...>;
  using TParamMaps = std::tuple<typename ProcessTraits<Args>::TParamMap...>;
  using TRandomVARs = std::tuple<typename ProcessTraits<Args>::TRandomVAR...>;
  using TArities = std::tuple<typename ProcessTraits<Args>::TArity...>;

  static constexpr size_t depth = sizeof...(Args)-1;

  template <typename... TVARs>
  TRandomVARs random(TVARs... args) {
    TRandomVARs a;
    constexpr size_t arity = std::tuple_element<0, TArities>::type::value;
    detail::GetRandom<0, arity, depth, TArities>::apply(
        tao::seq::make_index_range<0, arity>(), a, processes_,
        std::make_tuple(args...));
    return a;
  }

  template<typename... TVARs>
  void random_n(std::vector<TRandomVARs> samples, size_t n, TVARs... args)
  {
    samples.clear();
    samples.resize(n);

  }

  Hierachy(Args... processes) : processes_{processes...} {}

  std::tuple<Args...> processes_;
};
} // namespace process
} // namespace ssmpack
