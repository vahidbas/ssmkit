/**
 * @file particle.hpp
 * @author Vahid Bastani
 *
 * Generic particle filter.
 */
#ifndef SSMPACK_FILTER_PARTICLE_HPP
#define SSMPACK_FILTER_PARTICLE_HPP

#include "ssmpack/distribution/conditional.hpp"
#include "ssmpack/process/markov.hpp"
#include "ssmpack/process/memoryless.hpp"
#include "ssmpack/process/hierarchical.hpp"
#include "ssmpack/filter/recursive_bayesian_base.hpp"
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
template <class TProcess, class TImportanceCPDF, class Resampler>
class Particle;

template <class MeasurementMap, class StateMap, class MeasuremenCtPDF,
          class StateCPDF, class InitialPDF, class ImportanceCPDF,
          class Resampler>
class Particle<Hierarchical<Markov<StateMap, StateCPDF, InitialPDF>,
                            Memoryless<MeasurementMap, MeasurementCPDF>>,
               ImportanceCPDF, Resampler> {
 public:
  using Process = Hierarchical<Markov<StateMap, StateCPDF, InitialPDF>,
                               Memoryless<MeasurementMap, MeasurementCPDF>>;
  /** Type of the state posterior \f$ \{\mathbf{x}^{(i)},
   * \omega^{(i)}\}_{i=0}^{N-1}\f$
   */
  using CompeleteState = std::tuple<arma::mat, arma::vec>;

 private:
  //! Particle weights \f$ \{\omega^{(i)}\}_{i=0}^{N-1}\f$.
  arma::vec w_;
  //! State particles \f$ \{\mathbf{x}^{(i)}\}_{i=0}^{N-1}\f$.
  arma::mat state_par_;
  //! The process model
  Process process_;
  //! Number of particles \f$N\f$.
  unsigned long num_;

  template <class... Args>
  void predict(const Args &... args) {
  }

  template <class... TArgs>
  CompeleteState correct(const arma::vec::fixed<D2> &measurement,
                          const TArgs &... args) {
    return //??;
  }

  CompeleteState initialize() {
    state_par_.each_col([&this](arma::vec & v) {
      v = process_.template getProcess<0>().getInitialPDF().random();
    });

    w_.for_each([&this](double &e) {
      e = process_.template getProcess<0>().getInitialPDF().likelihood(
          state_par_.col(?));
    });
    return;
  }
}

} // namespace filter
} // namespace ssmpack

#endif // SSMPACK_FILTER_PARTICLE_HPP
