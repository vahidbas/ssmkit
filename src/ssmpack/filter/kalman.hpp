/**
 * @file kalmen.hpp
 * @author Vahid Bastani
 *
 */
#ifndef SSMPACK_FILTER_KALMAN_HPP
#define SSMPACK_FILTER_KALMAN_HPP

#include "ssmpack/distribution/gaussian.hpp"
#include "ssmpack/distribution/conditional.hpp"
#include "ssmpack/process/markov.hpp"
#include "ssmpack/process/memoryless.hpp"
#include "ssmpack/process/hierarchical.hpp"
#include "ssmpack/filter/recursive_bayesian_base.hpp"
#include <armadillo>

namespace ssmpack {
namespace filter {

using process::Hierarchical;
using process::Markov;
using process::Memoryless;
using distribution::Conditional;
using distribution::Gaussian;

template <class STA_MAP, class OBS_MAP, size_t STA_D, size_t OBS_D>
class Kalman
    : public RecursiveBayesianBase<Kalman<STA_MAP, OBS_MAP, STA_D, OBS_D>> {

 public:
  using TProcess =
      Hierarchical<Markov<Conditional<Gaussian<STA_D>, STA_MAP>, Gaussian<STA_D>>,
                   Memoryless<Conditional<Gaussian<OBS_D>, OBS_MAP>>>;

  using TCompeleteState =
      std::tuple<arma::vec::fixed<STA_D>, arma::mat::fixed<STA_D, STA_D>>;
 private:
  TProcess process_;
  const arma::mat::fixed<STA_D, STA_D> &dyn_mat_;
  const arma::mat::fixed<OBS_D, STA_D> &mes_mat_;
  const arma::mat::fixed<STA_D, STA_D> &dyn_cov_;
  const arma::mat::fixed<OBS_D, OBS_D> &mes_cov_;
  arma::vec::fixed<STA_D> state_vec_;
  arma::mat::fixed<STA_D, STA_D> state_cov_;
  arma::vec::fixed<STA_D> p_state_vec_;
  arma::mat::fixed<STA_D, STA_D> p_state_cov_;

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
  void predict(const TArgs &... args) {
    // use the map function to pass controls, avoiding control definition
    // is move used here??
    p_state_vec_ =
        std::get<0>(process_.template getProcess<0>().getCPDF().getParamMap()(
            state_vec_, args...));
    p_state_cov_ = dyn_mat_ * state_cov_ * dyn_mat_.t() + dyn_cov_;
  }

  template <class... TArgs>
  TCompeleteState correct(const arma::vec::fixed<OBS_D> &measurement,
                          const TArgs &... args) {
    arma::vec inovation =
        measurement -
        std::get<0>(process_.template getProcess<1>().getCPDF().getParamMap()(
            p_state_vec_, args...));

    arma::mat inovation_cov = mes_mat_ * p_state_cov_ * mes_mat_.t() + mes_cov_;
    arma::mat kalman_gain =
        p_state_cov_ * mes_mat_.t() * arma::inv_sympd(inovation_cov);

    state_vec_ = p_state_vec_ + kalman_gain * inovation;
    state_cov_ = p_state_cov_ - kalman_gain * mes_mat_ * p_state_cov_;
    return std::make_tuple(state_vec_, state_cov_);
  }

  TCompeleteState initialize() {
    state_vec_ = process_.template getProcess<0>().getInitialPDF().getMean();
    state_cov_ =
        process_.template getProcess<0>().getInitialPDF().getCovariance();
    return std::make_tuple(state_vec_, state_cov_);
  }
};

template <class STA_MAP, class OBS_MAP, size_t STA_D, size_t OBS_D>
Kalman<STA_MAP, OBS_MAP, STA_D, OBS_D> makeKalman(
    Hierarchical<Markov<Conditional<Gaussian<STA_D>, STA_MAP>, Gaussian<STA_D>>,
                 Memoryless<Conditional<Gaussian<OBS_D>, OBS_MAP>>> process) {
  return Kalman<STA_MAP, OBS_MAP, STA_D, OBS_D>(process);
}

} // namespace filter
} // namespace ssmpack

#endif // SSMPACK_FILTER_KALMAN_HPP
