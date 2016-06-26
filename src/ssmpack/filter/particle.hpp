/**
 * @file particle.hpp
 * @author Vahid Bastani
 *
 * Generic particle filter.
 */
#ifndef SSMPACK_FILTER_PARTICLE_HPP
#define SSMPACK_FILTER_PARTICLE_HPP

#include "ssmpack/distribution/conditional.hpp"
#include "ssmpack/filter/recursive_bayesian_base.hpp"
#include "ssmpack/process/hierarchical.hpp"
#include "ssmpack/process/markov.hpp"
#include "ssmpack/process/memoryless.hpp"
#include <armadillo>

#include <tuple>

namespace ssmpack {
namespace filter {

using process::Hierarchical;
using process::Markov;
using process::Memoryless;
using distribution::Conditional;

/** Particle Filter.
 */
template <class Process, class Resampler>
class Particle : public RecursiveBayesianBase<Particle<Process, Resampler>> {
 public:
  /** Type of the state posterior
   *
   * \f$ \{\mathbf{x}^{(i)}_t,\omega^{(i)}\}_{i=1}^{M}\f$
   */
  using CompeleteState = std::tuple<arma::mat, arma::vec>;

 private:
  //! Particle weights \f$ \{\omega^{(i)}\}_{i=1}^{M}\f$.
  arma::vec w_;
  //! State particles \f$ \{\mathbf{x}^{(i)}_t\}_{i=1}^{M}\f$.
  arma::mat state_par_;
  //! The process model
  Process process_;
  //! Resampling algorithm
  Resampler resampler_;
  //! Number of particles \f$M\f$.
  unsigned long num_;

 private:
  //! Normalizes the sum of weights to one
  void normalizeWeights(void) { w_ = w_ / arma::sum(w_); }

 public:
  /** Constructor
   *
   * returns a Particle filter object.
   *
   * @param process The process model object that the PF is defined for
   * @param resampler The resampling algorithm
   * @param particles_num Number of particles \f$M\f$
   */
  Particle(Process process, Resampler resampler, unsigned long particles_num)
      : process_{process}, resampler_{resampler}, num_{particles_num} {
    // initialized w_ and state_par_
    w_.resize(num_);
    // take one sample to find out dimension
    auto tmp = process_.template getProcess<0>().getInitialPDF().random();
    state_par_.resize(tmp.size(), num_);
  }
  /** Prediction
   *
   * Performs prediction step.
   * \f{equation}{\mathbf{x}^{(i)}_t \sim p(\mathbf{x}_t| \tilde{\mathbf{x}}^{(i)}_{t-1}, y^d_1, \cdots, y^d_{N_d})
   * \quad \text{for} \quad i=1,\cdots,M\f}
   *
   * @param args... Control variables \f$y^d_1, \cdots, y^d_{N_d}\f$ of the dynamic process, if any.
   */
  template <class... Args>
  void predict(const Args &... args) {
    state_par_.each_col([this, &args...](arma::vec &v, const Args &... args) {
      v = process_.template getProcess<0>().getCPDF().random(v, args...);
    });
  }
  /** Correction
   *
   * Performs correction step.
   *
   * \f{equation}{ \omega^{(i)} = \tilde{\omega}^{(i)} p(\mathbf{z}_t| \mathbf{x}^{(i)}_t, y^m_1, \cdots, y^m_{N_m}) \f}
   * \f{equation}{\{\mathbf{x}^{(i)}_t,\omega^{(i)}\}_{i=1}^{M}
   * \overset{\mbox{resample}}{\longrightarrow}
   * \{\tilde{\mathbf{x}}^{(i)}_t,\tilde{\omega}^{(i)}\}_{i=1}^{M}\f}
   *
   * @param measurement Measurement vector \f$\mathbf{z}_t\f$.
   * @param args... Control variables \f$y^m_1, \cdots, y^m_{N_m}\f$ of the measurement process, if any.
   * @return Estimated state \f$\{\tilde{\mathbf{x}}^{(i)}_t,\tilde{\omega}^{(i)}\}_{i=1}^{M}\f$
   */
  template <class Measurement, class... TArgs>
  CompeleteState correct(const Measurement &measurement,
                         const TArgs &... args) {
    unsigned long cnt = 0;
    w_.for_each([this, &cnt, &measurement, &args...](double &e) {
      e *= process_.template getProcess<1>().getCPDF().likelihood(
          measurement, state_par_.col(cnt++), args...);
    });

    normalizeWeights();

    resampler_(state_par_, w_);

    return std::make_tuple(state_par_, w_);
  }
  /** Initialization
   *
   * @return Estimated state \f$\{\tilde{\mathbf{x}}^{(i)}_0,\tilde{\omega}^{(i)}\}_{i=1}^{M}\f$
   */
  CompeleteState initialize() {
    state_par_.each_col([this](arma::vec &v) {
      v = process_.template getProcess<0>().getInitialPDF().random();
    });

    unsigned long cnt = 0;
    w_.for_each([this, &cnt](double &e) {
      e = process_.template getProcess<0>().getInitialPDF().likelihood(
          state_par_.col(cnt++));
    });

    normalizeWeights();

    return std::make_tuple(state_par_, w_);
  }
  //! @return Estimated state \f$\{\tilde{\omega}^{(i)}\}_{i=1}^{M}\f$
  const arma::vec &getWeights(void) const { return w_; }
  //! @return Estimated state \f$\{\tilde{\mathbf{x}}^{(i)}_t\}_{i=1}^{M}\f$
  const arma::mat &getStateParticles(void) const { return state_par_; }
};

/**
 */
template <class StatePDF, class StateParamMap, class InitialPDF,
          class MeasurementPDF, class MeasurementParamMap, class Resampler>
auto makeParticle(
    Hierarchical<Markov<StatePDF, StateParamMap, InitialPDF>,
                 Memoryless<MeasurementPDF, MeasurementParamMap>> process,
    Resampler resampler, unsigned long particle_num) {
  return Particle<Hierarchical<Markov<StatePDF, StateParamMap, InitialPDF>,
                               Memoryless<MeasurementPDF, MeasurementParamMap>>,
                  Resampler>(process, resampler, particle_num);
}

} // namespace filter
} // namespace ssmpack

#endif // SSMPACK_FILTER_PARTICLE_HPP
