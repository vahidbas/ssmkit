/**
 * @file kalmen.hpp
 * @author Vahid Bastani
 *
 */
#ifndef SSMPACK_FILTER_KALMEN_HPP
#define SSMPACK_FILTER_KALMEN_HPP

#include "ssmpack/distribution/gaussian.hpp"
#include "ssmpack/distribution/conditional.hpp"
#include "ssmpack/process/markov.hpp"
#include "ssmpack/process/memoryless.hpp"
#include "ssmpack/process/hierarchical.hpp"

#include <armadillo>

namespace ssmpack {
namespace filter {

template<class TProcess>
class Kalman;

template <class T1, class T2, size_t D1, size_t D2>
class Kalman <
  process::Hierarchical < 
    process::Markov<
      distribution::Conditional<distribution::Gaussian<D1>, T1>,distribution::Gaussian<D1>>,
    process::Memoryless <
      distribution::Conditional<distribution::Gaussian<D2>, T2 > > > >{

using TProcess = 
  process::Hierarchical < 
    process::Markov<
      distribution::Conditional<distribution::Gaussian<D1>, T1>,distribution::Gaussian<D1>>,
    process::Memoryless <
      distribution::Conditional<distribution::Gaussian<D2>, T2 > > > ;


 TProcess process_;
 const arma::mat::fixed<D1,D1> & dyn_mat_;
 const arma::mat::fixed<D2,D1> & mes_mat_;
 const arma::mat::fixed<D1,D1> & dyn_cov_;
 const arma::mat::fixed<D2,D2> & mes_cov_;
 arma::vec::fixed<D1> state_vec_;
 arma::mat::fixed<D1,D1> state_cov_;
 arma::vec::fixed<D1> p_state_vec_;
 arma::mat::fixed<D1,D1> p_state_cov_;

 public:
  Kalman(const TProcess &process)
      : process_(process),
        dyn_mat_(
            process_.template getProcess<0>().getCPDF().getParamMap().transfer),
        mes_mat_(
            process_.template getProcess<1>().getCPDF().getParamMap().transfer),
        dyn_cov_(process_.template getProcess<0>()
                     .getCPDF()
                     .getParamMap()
                     .covariance),
        mes_cov_(process_.template getProcess<1>()
                     .getCPDF()
                     .getParamMap()
                     .covariance) {}

  template <class... TArgs>
  void predict(const TArgs &... args)
  {
    // use the map function to pass controls, avoiding control definition
    // is move used here??
    p_state_vec_ = std::get<0>(
        process_.template getProcess<0>().getCPDF().getParamMap()(state_vec_, args...));
    p_state_cov_ = dyn_mat_ * state_cov_ * dyn_mat_.t() + dyn_cov_;
  }

  template <class... TArgs>
  void correct(const arma::vec::fixed<D2> &measurement, const TArgs &... args)
  {
    arma::vec inovation =
        measurement -
        std::get<1>(process_.template getProcess<0>().getCPDF().getParamMap()(
            p_state_vec_, args...));

    arma::mat inovation_cov =
        mes_mat_ * p_state_cov_ * mes_mat_.t() + mes_cov_;
    arma::mat kalman_gain =
        p_state_cov_ * mes_mat_.t() * arma::inv_sympd(inovation_cov);

    state_vec_ = p_state_vec_ + kalman_gain * inovation;
    state_cov_ = p_state_cov_ - kalman_gain * mes_mat_ * p_state_cov_;
  }

  void initialize()
  {
    state_vec_ = 
            process_.template getProcess<0>().getInitialPDF().getMean();
    state_cov_ = 
            process_.template getProcess<0>().getInitialPDF().getCovariance();
  }
};

template<class TProcess>
Kalman<TProcess> makeKalman(TProcess process)
{
  return Kalman<TProcess>(process);
}

} // namespace filter
} // namespace ssmpack

#endif // SSMPACK_FILTER_KALMEN_HPP
