#pragma once

/// @cond DEV
//exclude all from doxygen documantation

#include <tao/seq/make_integer_range.hpp> //https://github.com/taocpp/sequences

namespace ssmkit {
namespace process {

namespace detail {
//==========GetRandom==========
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
//===========GetInitialized===================
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

//===================GetLikelihood=============================
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
} // namespace process
} // namespace ssmkit

/// @endcond
