#pragma once

#include "ssmpack/process/traits.hpp"

namespace ssmpack {
namespace process {

template <typename... Args>
struct Hierachy {
  static_assert(AreProcesses<Args...>::value, "");
  using TPDFs = std::tuple<typename ProcessTraits<Args>::TPDF...>;
  using TParamMaps = std::tuple<typename ProcessTraits<Args>::TParamMap...>;
  using TRandomVARs = std::tuple<typename ProcessTraits<Args>::TRandomVAR...>;

  static constexpr size_t depth = sizeof...(Args);

  template <typename... TVARs>
  TRandomVARs random(TVARs... args) {
    TRandomVARs a;
    function<0, TVARs...>::apply(a, processes_, args...);
    return a;
  }

  //  template <typename F, typename... A, size_t... Is>
  //  TRandomVARs for_each(std::index_sequence<Is...>, F &&f, A... args) {
  //    return {f(std::get<Is>(processes_), args...)...};
  //  }

  Hierachy(Args... processes) : processes_{processes...} {}

  template <size_t N, typename... A>
  struct function {
    static void apply(TRandomVARs &rvs, std::tuple<Args...> &prc ,A... args) {
      std::get<N>(rvs) =
          std::get<N>(prc).random(std::get<N - 1>(rvs), args...);
      function<N + 1, A...>::apply(rvs, prc, args...);
    }
  };

  template <typename... A>
  struct function<0, A...> {
    static void apply(TRandomVARs &rvs,  std::tuple<Args...> &prc, A... args) {
      std::get<0>(rvs) = std::get<0>(prc).random(args...);
      function<1, A...>::apply(rvs, prc, args...);
    }
  };

  template <typename... A>
  struct function<depth, A...> {
    static void apply(TRandomVARs &rvs, std::tuple<Args...> &prc, A... args) {
      std::get<depth>(rvs) =
          std::get<depth>(prc).random(std::get<depth - 1>(rvs), args...);
    }
  };

    std::tuple<Args...> processes_;
  };
}
}
