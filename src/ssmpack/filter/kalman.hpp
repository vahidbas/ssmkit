/**
 * @file kalmen.hpp
 * @author Vahid Bastani
 *
 * Implementation of Kalman filter
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

/** Kalman filter
 */
template <class STA_MAP, class OBS_MAP>
class Kalman
    : public RecursiveBayesianBase<Kalman<STA_MAP, OBS_MAP>> {

 public:
  //! Type of process object
  using TProcess =
      Hierarchical<Markov<Gaussian, STA_MAP, Gaussian>,
                   Memoryless<Gaussian, OBS_MAP>>;
  //! Type of the posterior state \f$(\hat{\mathbf{x}}, \hat{\mathbf{P}})\f$ 
  using TCompeleteState =
      std::tuple<arma::vec, arma::mat>;

 private:
  //! The process object
  TProcess process_;
  //! The state transition matrix \f$\mathbf{F}\f$
  const arma::mat &dyn_mat_;
  //! The measurement matrix \f$\mathbf{H}\f$
  const arma::mat &mes_mat_;
  //! The covariance of dynamic noise \f$\mathbf{Q}\f$
  const arma::mat &dyn_cov_;
  //! The covariance of measurement noise \f$\mathbf{R}\f$
  const arma::mat &mes_cov_;
  //! The corrected state vector \f$\mathbf{x}_{t|t}\f$
  arma::vec state_vec_;
  //! The corrected state covariance \f$\mathbf{P}_{t|t}\f$
  arma::mat state_cov_;
  //! The predicted state vector \f$\mathbf{x}_{t|t-1}\f$
  arma::vec p_state_vec_;
  //! The predicted state covariance \f$\mathbf{P}_{t|t-1}\f$
  arma::mat p_state_cov_;

 public:
  /** Construct a Kalman filter
   *
   * Construct a Kalman filter with parameters taken from \p process argument.
   */
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

  
  /** Prediction
   *
   * Performs the prediction step.
   *
   * \f{equation}{\hat{\mathbf{x}}_{t|t-1} = \mathbf{F}\hat{\mathbf{x}}_{t-1|t-1}+u_d(y^d_1,
   * \cdots, y^d_{N_d}) \f}
   * \f{equation}{\mathbf{P}_{t|t-1} = \mathbf{F}\mathbf{P}_{t-1|t-1}\mathbf{F}^T+\mathbf{Q}\f}
   *
   * @param args... Control variables \f$y^d_1, \cdots, y^d_{N_d}\f$ of the dynamic process, if any.
   */
  template <class... TArgs>
  void predict(const TArgs &... args) {
    // use the map function to pass controls, avoiding control definition
    // is move used here??
    p_state_vec_ =
        std::get<0>(process_.template getProcess<0>().getCPDF().getParamMap()(
            state_vec_, args...));
    p_state_cov_ = dyn_mat_ * state_cov_ * dyn_mat_.t() + dyn_cov_;
  }
  
  /** Correction
   *
   * Performs correction step.
   *
   * \f{equation}{\tilde{\mathbf{z}}_t=\mathbf{z}_t-\mathbf{H}\hat{\mathbf{x}}_{t|t-1}-u_m(y^m_1, \cdots, y^m_{N_m})\f}
   * \f{equation}{\mathbf {S}_t=\mathbf{H}\mathbf{P}_{t|t-1}\mathbf{H}^T+\mathbf{R} \f}
   * \f{equation}{\mathbf{K}_t=\mathbf{P}_{t|t-1}\mathbf{H}^T\mathbf{S}_t^{-1} \f}
   * \f{equation}{\hat{\mathbf{x}}_{t|t}=\hat{\mathbf{x}}_{t|t-1}+\mathbf{K}_{t}\tilde{\mathbf{z}}_t \f}
   * \f{equation}{\mathbf{P}_{t|t}=(I-\mathbf{K}_t\mathbf{H})\mathbf{P}_{t|t-1}\f}
   *
   * @param measurement Measurement vector \f$\mathbf{z}_t\f$.
   * @param args... Control variables \f$y^m_1, \cdots, y^m_{N_m}\f$ of the measurement process, if any.
   * @return Estimated state \f$(\hat{\mathbf{x}}_{t|t}, \mathbf{P}_{t|t})\f$
   */
  template <class... TArgs>
  TCompeleteState correct(const arma::vec &measurement,
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
  /** Initialization
   *
   * @return Initial state \f$(\hat{\mathbf{x}}_{0|0}, \mathbf{P}_{0|0})\f$
   */
  TCompeleteState initialize() {
    state_vec_ = process_.template getProcess<0>().getInitialPDF().getMean();
    state_cov_ =
        process_.template getProcess<0>().getInitialPDF().getCovariance();
    return std::make_tuple(state_vec_, state_cov_);
  }
};

template <class STA_MAP, class OBS_MAP>
Kalman<STA_MAP, OBS_MAP> makeKalman(
    Hierarchical<Markov<Gaussian, STA_MAP, Gaussian>,
                 Memoryless<Gaussian, OBS_MAP>> process) {
  return Kalman<STA_MAP, OBS_MAP>(process);
}

} // namespace filter
} // namespace ssmpack

#endif // SSMPACK_FILTER_KALMAN_HPP
