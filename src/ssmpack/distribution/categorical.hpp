#pragma once

#include "ssmpack/random/generator.hpp"

#include <armadillo>

namespace ssmpack {
namespace distribution {
template <class TValueType = int>
class Categorical {
  using TParameterVar = arma::vec;

 private:
 TParameterVar param_;
 TParameterVar cdf_;

 std::uniform_real_distribution<double> uniform_;
 TValueType max_;

 public:
  Categorical() : Categorical(arma::ones<arma::vec>(1)) {}
  Categorical(TParameterVar parameters)
      : param_(std::move(parameters)) {calcCDF(); calcMax();}

  TValueType random() {
    double rv = uniform_(random::Generator::get().getGenerator());
    for (TValueType i = 0; i < max_; ++i)
      if (rv < cdf_(i))
        return i;

    return 0; // this line never get reached
  }

  double likelihood(const TValueType &rv) {
    return param_(rv);
  }

  Categorical &parameterize(const TParameterVar & param){
    param_ = param;
    calcCDF();
    calcMax();
  }

  private:
   void calcCDF() { cdf_ = arma::cumsum(param_); }
   void calcMax() { max_ = param_.n_rows; }
};

} // namespace ssmpack
} // namespace distribution
