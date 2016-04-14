#pragma once 

#include "ssmpack/process/traits.hpp"

namespace ssmpack {
  namespace process {

template<typename... Args>
struct Hierachy {
  static_assert(AreProcesses<Args...>::value, "");
  using TPDFs = std::tuple< typename ProcessTraits<Args>::TPDF ...>;
  using TParamMaps = std::tuple< typename ProcessTraits<Args>::TParamMap ...>;
  using TRandomVARs = std::tuple< typename ProcessTraits<Args>::TRandomVAR ...>;
  static constexpr size_t depth = sizeof...(Args);
  
};

}
}
