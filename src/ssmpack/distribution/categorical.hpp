/**
 * @file categorical.hpp
 * @author Vahid Bastani
 *
 * implementation of categorical (multinomial) distribution
 */
#ifndef SSMPACK_DISTRIBUTION_CATEGORICAL_HPP
#define SSMPACK_DISTRIBUTION_CATEGORICAL_HPP

#include "ssmpack/random/generator.hpp"

#include <armadillo>

namespace ssmpack {
namespace distribution {

/** Categorical (multinomial) distribution
 * 
 * \f{equation}{p(x|\mathbf{p}) = \mathcal{Cat}(\mathbf{p})\f}
 * where \f$\mathbf{p} = [p_0, \cdots, p_N]^T\f$ and \f$p(x=i|\mathbf{p}) = p_i\f$
 */
class Categorical {
  //! Type of the parameter vector \f$\mathbf{p}\f$.
  using TParameterVar = arma::vec;
  //! Type of the random variable \f$x\f$.
  using TValueType = unsigned int;

 private:
 //! The parameter vector \f$\mathbf{p}\f$.
 TParameterVar param_;
 //! Cumulative distribution function. 
 TParameterVar cdf_;
 //! Core random number distribution.
 std::uniform_real_distribution<double> uniform_;
 //! Length of the parameter vector \f$N+1\f$.
 TValueType max_;

 public:
  //! Default constructor \f$\mathbf{p}=[1]\f$.
  Categorical() : Categorical(arma::ones<arma::vec>(1)) {}
  /** Constructor
   * @param parameter The parameter vector \f$\mathbf{p}\f$ whose elements
   * should sum to 1.0.
   */
  Categorical(TParameterVar parameters)
      : param_(std::move(parameters)) {calcCDF(); calcMax();}
  //! Return a random variable from the distribution.
  TValueType random() {
    double rv = uniform_(random::Generator::get().getGenerator());
    for (TValueType i = 0; i < max_; ++i)
      if (rv < cdf_(i))
        return i;

    return 0; // this line never get reached
  }
  //! Return likelihood of the given random variable
  double likelihood(const TValueType &rv) {
    return param_(rv);
  }
  //! Change parameters of the distribution
  Categorical &parameterize(const TParameterVar & param){
    param_ = param;
    calcCDF();
    calcMax();
    return *this;
  }

  private:
   void calcCDF() { cdf_ = arma::cumsum(param_); }
   void calcMax() { max_ = param_.n_rows; }
};

} // namespace ssmpack
} // namespace distribution

#endif //SSMPACK_DISTRIBUTION_CATEGORICAL_HPP
