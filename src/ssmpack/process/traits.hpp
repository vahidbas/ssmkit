/**
 * @file traits.hpp
 * @author Vahid Bastani
 *
 * Generic class for first-order Markovian stochastic process.
 */
#ifndef SSMPACK_PROCESS_TRAITS_HPP
#define SSMPACK_PROCESS_TRAITS_HPP

#include "ssmpack/process/markov.hpp"

#include <type_traits>

namespace ssmpack {
namespace process {

  template<typename T>
  struct IsProcess : std::false_type;

  template<typename A>
  struct IsProcess<Markov<A>> : std::true_type;

  //==========================================
  template<typename... Args>
  struct AreProcesses;

  template<typename... Args>
  struct AreProcesses<T, Args...>{
    static constexpr bool value = AreAllProcess<Args...>::value &&
    IsProcess<T>::value;
  };

  template<>
  struct AreProcesses<> {
    static constexpr bool value = true;
  };
}
}

#endif

