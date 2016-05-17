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

/** Base template.
 * this class only defines the template interface and
 * does not implement anything. Refer to specialization
 * %Particle\<Hierarchical\<Markov\<StateMap, StateCPDF, InitialPDF\>,
 *                           Memoryless\<MeasurementMap, MeasurementCPDF\>\>,
 *              ImportanceCPDF, Resampler\>
 * for actual implementation.
 */
template <class TProcess, class Resampler>
class Particle;

template <class MeasurementMap, class StateMap, class MeasurementCPDF,
          class StateCPDF, class InitialPDF, class Resampler>
class Particle<
    Hierarchical<Markov<Conditional<StateMap, StateCPDF>, InitialPDF>,
                 Memoryless<Conditional<MeasurementMap, MeasurementCPDF>>>,
    Resampler>
    : public RecursiveBayesianBase<Particle<
          Hierarchical<
              Markov<Conditional<StateMap, StateCPDF>, InitialPDF>,
              Memoryless<Conditional<MeasurementMap, MeasurementCPDF>>>,
          Resampler>> {
 public:
  using Process =
      Hierarchical<Markov<Conditional<StateMap, StateCPDF>, InitialPDF>,
                   Memoryless<Conditional<MeasurementMap, MeasurementCPDF>>>;

  /** Type of the state posterior \f$ \{\mathbf{x}^{(i)}_t,
   * \omega^{(i)}\}_{i=0}^{N-1}\f$
   */
  using CompeleteState = std::tuple<arma::mat, arma::vec>;

 private:
  //! Particle weights \f$ \{\omega^{(i)}\}_{i=0}^{N-1}\f$.
  arma::vec w_;
  //! State particles \f$ \{\mathbf{x}^{(i)}_t\}_{i=0}^{N-1}\f$.
  arma::mat state_par_;
  //! Previous time state particles \f$
  // \{\mathbf{x}^{(i)}_{t-1}\}_{i=0}^{N-1}\f$.
  // arma::mat state_par_old_;
  //! The process model
  Process process_;
  //! Resampling algorithm
  Resampler resampler_;
  //! Number of particles \f$N\f$.
  unsigned long num_;

 private:
  void normalizeWeights(void) { w_ = w_ / arma::sum(w_); }

 public:
  Particle(Process process, Resampler resampler, unsigned long particles_num)
      : process_{process}, resampler_{resampler}, num_{particles_num} {
    // initialized w_ and state_par_
    w_.resize(num_);
    // take one sample to find out dimension
    auto tmp = process_.template getProcess<0>().getInitialPDF().random();
    state_par_.resize(tmp.size(), num_);
  }

  template <class... Args>
  void predict(const Args &... args) {
    state_par_.each_col([this, &args...](arma::vec &v, const Args &... args) {
      v = process_.template getProcess<0>().getCPDF().random(v, args...);
    });
  }

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

  const arma::vec &getWeights(void) const { return w_; }
  const arma::mat &getStateParticles(void) const { return state_par_; }
};

template <class Process, class Resampler>
Particle<Process, Resampler> makeParticle(Process process, Resampler resampler,
                                          unsigned long particle_num) {
  return Particle<Process, Resampler>(process, resampler, particle_num);
}

} // namespace filter
} // namespace ssmpack

#endif // SSMPACK_FILTER_PARTICLE_HPP
