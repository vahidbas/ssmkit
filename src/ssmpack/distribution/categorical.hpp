#pragma once

#include "ssmpack/random/generator.hpp"

#include <armadillo>

namespace ssmpack {
namespace distribution {
template <size_t VNum, class TValueType = int>
class Categorical {
  using TParameterVar = arma::vec::fixed<VNum>;

 private:
 TParameterVar param_;
 TParameterVar cdf_;

 std::uniform_real_distribution uniform_;
 TValueType max_;

 public:
  Categorical(arma::vec::fixed<VNum> parameters)
      : param_(std::move(parameters)) {}

  TValueType random() {
    double rv = uniform_(randm::Generator::get().getGenerator());
    for (TValueType i = 0; i < max_; ++i)
      if (rv < cdf_(i))
        return i;
  }
};

} // namespace ssmpack
} // namespace distribution
