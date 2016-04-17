#pragma once

#include "ssmpack/process/traits.hpp"

#include <tao/seq/make_integer_range.hpp>  //https://github.com/taocpp/sequences

namespace ssmpack {
namespace process {

template <typename... Args>
struct Hierachy {
  static_assert(AreProcesses<Args...>::value, "");
  using TPDFs = std::tuple<typename ProcessTraits<Args>::TPDF...>;
  using TParamMaps = std::tuple<typename ProcessTraits<Args>::TParamMap...>;
  using TRandomVARs = std::tuple<typename ProcessTraits<Args>::TRandomVAR...>;
  using TArities = std::tuple<typename ProcessTraits<Args>::TArities...>;

  static constexpr size_t depth = sizeof...(Args);

  template <typename... TVARs>
  TRandomVARs random(TVARs... args) {
    TRandomVARs a;
    constexpr size_t arity = std::tuple_element<0,TArities>::type::value>; 
    function<0,
    std::tuple<TVARs...>, arity>::apply(tao::seq::make_index_range<0,arity>(), a, processes_, std::make_tuple<args...>);
    return a;
  }

  //  template <typename F, typename... A, size_t... Is>
  //  TRandomVARs for_each(std::index_sequence<Is...>, F &&f, A... args) {
  //    return {f(std::get<Is>(processes_), args...)...};
  //  }

  Hierachy(Args... processes) : processes_{processes...} {}

  template <size_t N, typename A, size_t NA>
  struct function {
    template<size_t... Is>
    static void apply(std::index_sequence<Is...>, TRandomVARs &rvs, std::tuple<Args...> &prc ,A args) {
      std::get<N>(rvs) =
          std::get<N>(prc).random(std::get<N - 1>(rvs), std::get<Is>(args)...);
      function<N + 1, A>::apply(rvs, prc, args);
    constexpr size_t arity = std::tuple_element<N+1,TArities>::type::value>;  
      function<N+1, A,
      NA+arity>::apply(tao::seq::make_index_range<NA,NA+arity>(),rvs, prc, args);
    }
  };

  template <typename A, size_t NA>
  struct function<0, A, NA> {
    template<size_t... Is>
    static void apply(std::index_sequence<Is...>, TRandomVARs &rvs, std::tuple<Args...> &prc ,A args) {
      std::get<0>(rvs) = std::get<0>(prc).random(std::get<Is>(args)...);
    constexpr size_t arity = std::tuple_element<1,TArities>::type::value>; 
      function<1, A, NA+arity>::apply(tao::seq::make_index_range<NA,NA+arity>,rvs, prc, args);
    }
  };

  template <typename A>
  struct function<depth, A, NA> {
    template<size_t... Is>
    static void apply(std::index_sequence<Is...>, TRandomVARs &rvs, std::tuple<Args...> &prc ,A args) {
      std::get<depth>(rvs) =
          std::get<depth>(prc).random(std::get<depth - 1>(rvs),
          std::get<Is>(args)...);
    }
  };

    std::tuple<Args...> processes_;
  };
}
}
